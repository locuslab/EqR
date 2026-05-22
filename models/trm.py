from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import torch
from pydantic import BaseModel
from torch import nn

from models.common import trunc_normal_init_
from models.layers import Attention, CastedEmbedding, CastedLinear, CosSin, RotaryEmbedding, RotaryEmbedding2D, SwiGLU, rms_norm


@dataclass
class LatentCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class ModelCarry:
    inner_carry: LatentCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]
    global_step: Optional[int] = None
    halt_counter: Optional[torch.Tensor] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class TRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    board_height: Optional[int] = None
    board_width: Optional[int] = None
    halt_max_steps: int
    halt_exploration_prob: float
    disable_act_halt: bool = False
    act_inference: bool = False
    halt_threshold: float = 0.0
    halt_confirm_steps: int = 0
    halt_confirm_mode: str = "consecutive"
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    no_ACT_continue: bool = True


def _sinusoidal(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim, dtype=torch.float32)
    if dim == 0:
        return pe
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    angles = pos * div.unsqueeze(0)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
    return pe


class Block(nn.Module):
    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config
        if config.mlp_t:
            self.mlp_t = SwiGLU(config.seq_len, config.expansion)
        else:
            self.self_attn = Attention(config.hidden_size, config.hidden_size // config.num_heads, config.num_heads, config.num_heads, causal=False)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin, hidden_states), self.norm_eps)
        return rms_norm(hidden_states + self.mlp(hidden_states), self.norm_eps)


class Blocks(nn.Module):
    def __init__(self, layers: list[Block]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class InnerNetwork(nn.Module):
    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, embed_std, self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        self._init_pos(embed_std)
        self.L_level = Blocks([Block(config) for _ in range(config.L_layers)])
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _board_dims(self) -> Tuple[int, int]:
        if self.config.board_height is not None and self.config.board_width is not None:
            return self.config.board_height, self.config.board_width
        n = int(self.config.seq_len**0.5)
        if n * n != self.config.seq_len:
            raise ValueError(f"seq_len {self.config.seq_len} is not a perfect square. Please specify board_height and board_width explicitly.")
        return n, n

    def _init_pos(self, embed_std: float) -> None:
        head_dim = self.config.hidden_size // self.config.num_heads
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(head_dim, self.config.seq_len, self.config.rope_theta)
        elif self.config.pos_encodings == "rope2d":
            h, w = self._board_dims()
            self.rotary_emb = RotaryEmbedding2D(head_dim, h, w, self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, embed_std, self.forward_dtype)
        elif self.config.pos_encodings == "absolute":
            self.register_buffer("absolute_pos_embedding", _sinusoidal(self.config.seq_len, self.config.hidden_size).to(self.forward_dtype), persistent=True)

    def _cos_sin(self) -> CosSin:
        return self.rotary_emb() if hasattr(self, "rotary_emb") else None

    def _pos(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.pos_encodings == "learned":
            return 0.707106781 * (x + self.embed_pos.embedding_weight.to(self.forward_dtype))
        if self.config.pos_encodings == "absolute":
            return 0.707106781 * (x + self.absolute_pos_embedding.to(device=x.device, dtype=self.forward_dtype))
        return x

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_scale * self._pos(self.embed_tokens(input_ids.to(torch.int32)))

    def _distribution_embeddings(self, probs: torch.Tensor) -> torch.Tensor:
        x = (probs.to(torch.float32) @ self.embed_tokens.embedding_weight.to(torch.float32)).to(self.forward_dtype)
        return self.embed_scale * self._pos(x)

    def empty_carry(self, batch_size: int) -> LatentCarry:
        shape = (batch_size, self.config.seq_len, self.config.hidden_size)
        return LatentCarry(torch.empty(shape, dtype=self.forward_dtype), torch.empty(shape, dtype=self.forward_dtype))

    def reset_carry(self, reset_flag: torch.Tensor, carry: LatentCarry) -> LatentCarry:
        h = self.H_init.to(device=carry.z_H.device, dtype=carry.z_H.dtype).view(1, 1, -1)
        l = self.L_init.to(device=carry.z_L.device, dtype=carry.z_L.dtype).view(1, 1, -1)
        mask = reset_flag.to(device=carry.z_H.device).view(-1, 1, 1)
        return LatentCarry(torch.where(mask, h, carry.z_H), torch.where(mask, l, carry.z_L))

    def latent_recursion(self, z_H: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, seq: Dict[str, CosSin]) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x, **seq)
        z_H = self.L_level(z_H, z_L, **seq)
        return z_H, z_L

    def deep_recursion(self, z_H: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, seq: Dict[str, CosSin]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                z_H, z_L = self.latent_recursion(z_H, z_L, x, seq)
        return self.latent_recursion(z_H, z_L, x, seq)

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z_H, z_L = self.deep_recursion(
            carry.z_H,
            carry.z_L,
            self._input_embeddings(batch["inputs"]),
            {"cos_sin": self._cos_sin()},
        )
        logits = self.lm_head(z_H)
        q = self.q_head(z_H[:, 0]).to(torch.float32)
        return LatentCarry(z_H.detach(), z_L.detach()), logits, (q[..., 0], q[..., 1])

    def forward_no_carry(self, z_H: torch.Tensor, z_L: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z_H, z_L = self.deep_recursion(z_H, z_L, self._input_embeddings(batch["inputs"]), {"cos_sin": self._cos_sin()})
        q = self.q_head(z_H[:, 0]).to(torch.float32)
        return (z_H, z_L), self.lm_head(z_H), (q[..., 0], q[..., 1])

    def concat_states(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        return torch.cat((z_H, z_L), dim=1)

    def split_states(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.config.seq_len
        return state[:, :n, :], state[:, n:, :]

    def _logits_embeddings(self, logits: torch.Tensor, puzzle_ids: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
        del puzzle_ids
        if logits.ndim != 3 or logits.shape[-1] != self.config.vocab_size:
            raise ValueError(f"Expected logits shape (B, S, {self.config.vocab_size}), got {tuple(logits.shape)}")
        temp = float(temperature)
        if abs(temp) < 1e-6:
            temp = 1e-6
        return self._distribution_embeddings(torch.softmax(logits.to(torch.float32) / temp, dim=-1))

    def _probs_embeddings(self, probs: torch.Tensor, puzzle_ids: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
        del puzzle_ids
        if probs.ndim != 3 or probs.shape[-1] != self.config.vocab_size:
            raise ValueError(f"Expected probs shape (B, S, {self.config.vocab_size}), got {tuple(probs.shape)}")
        probs = probs.to(torch.float32).clamp_min(0.0)
        return self._distribution_embeddings(probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps))


class TRMModel(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = TRMConfig(**config_dict)
        self.inner = InnerNetwork(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ModelCarry:
        b, device = batch["inputs"].shape[0], batch["inputs"].device
        c = self.inner.empty_carry(b)
        return ModelCarry(
            inner_carry=LatentCarry(c.z_H.to(device), c.z_L.to(device)),
            steps=torch.zeros((b,), dtype=torch.int32, device=device),
            halted=torch.ones((b,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
            halt_counter=torch.zeros((b,), dtype=torch.int32, device=device),
        )

    def forward(self, carry: ModelCarry, batch: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[ModelCarry, Dict[str, torch.Tensor]]:
        del kwargs
        device = batch["inputs"].device
        halted_prev, steps_prev = carry.halted.to(device), carry.steps.to(device)
        counter_prev = carry.halt_counter.to(device) if carry.halt_counter is not None else None
        inner = self.inner.reset_carry(halted_prev, LatentCarry(carry.inner_carry.z_H.to(device), carry.inner_carry.z_L.to(device)))
        steps = torch.where(halted_prev, torch.zeros_like(steps_prev), steps_prev)
        counter = torch.zeros_like(steps) if counter_prev is None else torch.where(halted_prev, torch.zeros_like(counter_prev), counter_prev)
        data = {k: torch.where(halted_prev.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v.to(device)) for k, v in carry.current_data.items()}

        inner, logits, (q_halt, q_continue) = self.inner(inner, data)
        out = {"logits": logits, "q_halt_logits": q_halt, "q_continue_logits": q_continue}
        with torch.no_grad():
            steps = steps + 1
            is_last = steps >= self.config.halt_max_steps
            halted = is_last
            if self.config.halt_max_steps > 1 and (self.training or self.config.act_inference) and not self.config.disable_act_halt:
                signal = q_halt > float(self.config.halt_threshold) if self.config.no_ACT_continue else q_halt > (q_continue + float(self.config.halt_threshold))
                if self.config.halt_confirm_steps > 0:
                    mode = str(self.config.halt_confirm_mode or "consecutive").lower()
                    counter = torch.where(signal, counter + 1, counter if mode in {"cumulative", "accumulate", "total"} else torch.zeros_like(counter))
                    halted = halted | (counter >= 1 + int(self.config.halt_confirm_steps))
                else:
                    counter = torch.where(signal, counter + 1, torch.zeros_like(counter))
                    halted = halted | signal
                min_steps = (torch.rand_like(q_halt) < self.config.halt_exploration_prob) * torch.randint_like(steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (steps >= min_steps)
                if self.training and not self.config.no_ACT_continue:
                    _, _, (nqh, nqc) = self.inner(inner, data)
                    out["target_q_continue"] = torch.sigmoid(torch.where(is_last, nqh, torch.maximum(nqh, nqc)))
        return ModelCarry(inner, steps, halted, data, halt_counter=counter), out

    def pack_solver_state(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        return self.inner.concat_states(z_H, z_L)

    def unpack_solver_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inner.split_states(state)

    def initial_solver_state(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, device = batch["inputs"].shape[0], batch["inputs"].device
        shape = (b, self.config.seq_len, self.config.hidden_size)
        z_H, z_L = torch.empty(shape, dtype=self.inner.forward_dtype, device=device), torch.empty(shape, dtype=self.inner.forward_dtype, device=device)
        z_H.copy_(self.inner.H_init.to(device=device).expand_as(z_H))
        z_L.copy_(self.inner.L_init.to(device=device).expand_as(z_L))
        return self.pack_solver_state(z_H, z_L)

    def solver_step(self, state: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z_H, z_L = self.unpack_solver_state(state)
        (z_H, z_L), logits, q = self.inner.forward_no_carry(z_H, z_L, batch)
        return self.pack_solver_state(z_H.detach(), z_L.detach()), logits, q


TinyRecursiveReasoningModel_ACTV1InnerCarry = LatentCarry
TinyRecursiveReasoningModel_ACTV1Carry = ModelCarry
TinyRecursiveReasoningModel_ACTV1Config = TRMConfig
TinyRecursiveReasoningModel_ACTV1 = TRMModel

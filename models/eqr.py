from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class EqRConfig(BaseModel):
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
    pos_encodings: Optional[str]
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    board_height: Optional[int] = None
    board_width: Optional[int] = None
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    lambda_: float = 0.95
    noise_scale: float = 0.01
    H_init_std: float = 1.0
    L_init_std: float = 1.0


class ReasoningBlock(nn.Module):
    def __init__(self, config: EqRConfig) -> None:
        super().__init__()
        self.config = config
        if config.mlp_t:
            self.mlp_t = SwiGLU(hidden_size=config.seq_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        return rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)


class NoisyReasoningModule(nn.Module):
    def __init__(self, config: EqRConfig, layers: List[ReasoningBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.lambda_ = config.lambda_
        self.noise_scale = config.noise_scale

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        updated = hidden_states + input_injection
        for layer in self.layers:
            updated = layer(hidden_states=updated, **kwargs)
        noise = torch.randn_like(hidden_states) * self.noise_scale
        return (1 - self.lambda_) * hidden_states + self.lambda_ * updated + noise


class InnerNetwork(nn.Module):
    def __init__(self, config: EqRConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self._random_reset_field_stds = {"z_H": float(config.H_init_std), "z_L": float(config.L_init_std)}
        self._random_reset_z_H_buffer: Optional[torch.Tensor] = None
        self._random_reset_z_L_buffer: Optional[torch.Tensor] = None

        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        self._init_pos()
        self.L_level = NoisyReasoningModule(config, [ReasoningBlock(config) for _ in range(config.L_layers)])
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=config.H_init_std), persistent=False)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=config.L_init_std), persistent=False)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _board_dims(self) -> Tuple[int, int]:
        if self.config.board_height is not None and self.config.board_width is not None:
            return self.config.board_height, self.config.board_width
        board_size = int(self.config.seq_len**0.5)
        if board_size * board_size != self.config.seq_len:
            raise ValueError(f"seq_len {self.config.seq_len} is not a perfect square. Specify board_height and board_width explicitly.")
        return board_size, board_size

    def _init_pos(self) -> None:
        pos = self.config.pos_encodings
        head_dim = self.config.hidden_size // self.config.num_heads
        if pos == "rope":
            self.rotary_emb = RotaryEmbedding(head_dim, self.config.seq_len, self.config.rope_theta)
        elif pos == "rope2d":
            h, w = self._board_dims()
            self.rotary_emb = RotaryEmbedding2D(head_dim, h, w, self.config.rope_theta)
        elif pos not in {"none", None}:
            raise ValueError(f"Unknown pos_encodings '{pos}'")

    def _cos_sin(self) -> CosSin:
        return self.rotary_emb() if hasattr(self, "rotary_emb") else None

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_scale * self.embed_tokens(input_ids.to(torch.int32))

    def empty_carry(self, batch_size: int, device: Optional[torch.device] = None) -> LatentCarry:
        shape = (batch_size, self.config.seq_len, self.config.hidden_size)
        return LatentCarry(
            z_H=torch.empty(shape, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(shape, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: LatentCarry, *, reset_indices: Optional[torch.Tensor] = None) -> LatentCarry:
        reset_indices = reset_flag.nonzero(as_tuple=False).flatten() if reset_indices is None else reset_indices
        if reset_indices.numel() == 0:
            return carry
        n = int(reset_indices.numel())
        for name in ("z_H", "z_L"):
            state = getattr(carry, name)
            shape = state.shape[1:]
            buf_name = f"_random_reset_{name}_buffer"
            buf = getattr(self, buf_name)
            if buf is None or buf.device != state.device or buf.dtype != state.dtype or buf.shape[1:] != shape or buf.shape[0] < n:
                buf = torch.empty((n if buf is None else max(n, buf.shape[0] * 2),) + shape, dtype=state.dtype, device=state.device)
                setattr(self, buf_name, buf)
            init = trunc_normal_init_(buf[:n], std=float(self._random_reset_field_stds[name]))
            state[reset_indices.to(device=state.device)] = init
        return carry

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

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, torch.Tensor]:
        z_H, z_L = self.deep_recursion(
            carry.z_H,
            carry.z_L,
            self._input_embeddings(batch["inputs"]),
            {"cos_sin": self._cos_sin()},
        )
        logits = self.lm_head(z_H).contiguous()
        q = self.q_head(z_H[:, 0]).to(torch.float32)
        return LatentCarry(z_H=z_H.detach(), z_L=z_L.detach()), logits, q[..., 0]


def _disable_torch_compile(fn: Any) -> Any:
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    if disable is None:
        disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return disable(fn) if disable is not None else fn


@_disable_torch_compile
def _reset_rows(
    inner: Any,
    reset_flag: torch.Tensor,
    inner_carry: LatentCarry,
    current_data: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[LatentCarry, Dict[str, torch.Tensor]]:
    idx = reset_flag.nonzero(as_tuple=False).flatten()
    inner_carry = inner.reset_carry(reset_flag, inner_carry, reset_indices=idx)
    current_data = {k: v.to(device=batch[k].device) for k, v in current_data.items()}
    if idx.numel() > 0:
        idx = idx.to(device=device)
        for k, v in current_data.items():
            v[idx] = batch[k][idx]
    return inner_carry, current_data


class EqRModel(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = EqRConfig(**config_dict)
        self.inner = InnerNetwork(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ModelCarry:
        b = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return ModelCarry(
            inner_carry=self.inner.empty_carry(b, device=device),
            steps=torch.zeros((b,), dtype=torch.int32, device=device),
            halted=torch.ones((b,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: ModelCarry, batch: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[ModelCarry, Dict[str, torch.Tensor]]:
        del kwargs
        device = batch["inputs"].device
        reset = carry.halted.to(device=device)
        inner_carry, current_data = _reset_rows(
            self.inner,
            reset,
            LatentCarry(carry.inner_carry.z_H.to(device=device), carry.inner_carry.z_L.to(device=device)),
            carry.current_data,
            batch,
            device,
        )
        steps = torch.where(reset, torch.zeros_like(carry.steps, device=device), carry.steps.to(device=device))
        inner_carry, logits, q_halt = self.inner(inner_carry, current_data)
        with torch.no_grad():
            steps = steps + 1
            halted = steps >= self.config.halt_max_steps
            if self.training and self.config.halt_max_steps > 1:
                halted = (halted | (q_halt > 0)) & (
                    steps
                    >= (
                        (torch.rand_like(q_halt) < self.config.halt_exploration_prob)
                        * torch.randint_like(steps, low=2, high=self.config.halt_max_steps + 1)
                    )
                )
        return ModelCarry(inner_carry, steps, halted, current_data), {"logits": logits, "q_halt_logits": q_halt}

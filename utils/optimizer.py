"""Optimizer scheduling and checkpoint compatibility helpers for EqR training."""

from typing import Any, Callable, Dict, Iterable, List, Sequence
import itertools
import torch
from torch import nn
import math
from config.schema import PretrainConfig
from utils.training import TrainState


__all__ = [
    "compute_lr",
    "build_param_id_to_name_map",
    "summarize_optimizer_state",
    "maybe_reorder_optimizer_states",
]


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (
        min_ratio
        + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    )

def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )
    
def build_param_id_to_name_map(model: nn.Module) -> Dict[int, str]:
    """
    Create an id->name lookup that covers both parameters and buffers.
    Buffers are prefixed with '[buffer]' to distinguish them in logs.
    """
    mapping: Dict[int, str] = {}
    for name, param in model.named_parameters():
        mapping[id(param)] = name
    for name, buffer in model.named_buffers():
        mapping[id(buffer)] = f"[buffer]{name}"
    return mapping


def summarize_optimizer_state(
    optimizer: torch.optim.Optimizer,
    opt_state: Dict[str, Any],
    param_id_to_name: Dict[int, str],
) -> str:
    """Build a short, human-readable summary for optimizer state mismatches."""
    current_groups = optimizer.param_groups
    ckpt_groups = opt_state.get("param_groups", [])
    ckpt_state = opt_state.get("state", {})

    lines = [
        f"Current optimizer has {len(current_groups)} param_groups;",
        f"Checkpoint has {len(ckpt_groups)} param_groups and {len(ckpt_state)} state entries.",
    ]

    def _format_param_list(params: Iterable[torch.Tensor]) -> str:
        names: List[str] = []
        for p in params:
            name = param_id_to_name.get(id(p), "<unknown>")
            shape = tuple(p.shape) if hasattr(p, "shape") else "?"
            names.append(f"{name}{shape}")
        if len(names) > 8:
            names = names[:8] + ["..."]
        return ", ".join(names)

    lines.append("Current groups:")
    for idx, group in enumerate(current_groups):
        params = group.get("params", [])
        lines.append(
            f"  [{idx}] params={len(params)}, lr={group.get('lr')}, "
            f"wd={group.get('weight_decay')}, names=[{_format_param_list(params)}]"
        )

    lines.append("Checkpoint groups:")
    for idx, group in enumerate(ckpt_groups):
        params = group.get("params", [])
        preview = params[:8]
        if len(params) > 8:
            preview = preview + ["..."]
        lines.append(
            f"  [{idx}] params={len(params)}, lr={group.get('lr')}, "
            f"wd={group.get('weight_decay')}, param_ids={preview}"
        )

    return "\n".join(lines)


def _count_group_params(groups: Sequence[Dict[str, Any]]) -> List[int]:
    return [len(g.get("params", [])) for g in groups]


def maybe_reorder_optimizer_states(
    optimizers: Sequence[torch.optim.Optimizer],
    optimizer_states: Sequence[Dict[str, Any]],
    log_fn: Callable[[str], None] | None = None,
) -> List[Dict[str, Any]]:
    """
    If checkpoint states appear to be saved in a different optimizer order (e.g., after
    refactoring), reorder them to minimize param count mismatches.
    Only applies when the reordering strictly reduces the mismatch score.
    """
    if len(optimizers) != len(optimizer_states) or len(optimizers) <= 1:
        return list(optimizer_states)

    current_counts = [_count_group_params(opt.param_groups) for opt in optimizers]
    ckpt_counts = [_count_group_params(state.get("param_groups", [])) for state in optimizer_states]

    # Flatten for a simple distance metric: sum of absolute differences in group sizes.
    def _score(order: Sequence[int]) -> int:
        score = 0
        for opt_idx, state_idx in enumerate(order):
            curr = current_counts[opt_idx]
            ckpt = ckpt_counts[state_idx]
            # pad to same length
            max_len = max(len(curr), len(ckpt))
            curr = curr + [0] * (max_len - len(curr))
            ckpt = ckpt + [0] * (max_len - len(ckpt))
            score += sum(abs(c - s) for c, s in zip(curr, ckpt))
        return score

    identity = tuple(range(len(optimizer_states)))
    best_perm = identity
    best_score = _score(identity)

    for perm in itertools.permutations(range(len(optimizer_states))):
        score = _score(perm)
        if score < best_score:
            best_score = score
            best_perm = perm

    identity_score = _score(identity)
    if best_perm != identity and best_score < identity_score:
        reordered = [optimizer_states[i] for i in best_perm]
        if log_fn is not None:
            log_fn(
                f"Reordered optimizer states {list(identity)} -> {list(best_perm)} "
                f"using param count matching. Current groups: {current_counts}, "
                f"checkpoint groups: {ckpt_counts}"
            )
        return reordered

    return list(optimizer_states)

"""Metric partitioning and reduction helpers for distributed training/eval."""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist


SPECIAL_METRIC_KEYS: Set[str] = frozenset({"max_steps", "min_steps", "max_q_logits", "min_q_logits"})


def partition_metrics(
    metrics: Dict[str, torch.Tensor], special_keys: Optional[Set[str]] = None
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Split metric tensors into regular vs. special keys needing custom reduction."""
    keys = sorted(metrics.keys())
    special_keys = special_keys or SPECIAL_METRIC_KEYS
    regular_keys: List[str] = []
    special_metrics: Dict[str, torch.Tensor] = {}

    for key in keys:
        tensor = metrics[key]
        if key in special_keys:
            special_metrics[key] = tensor.detach().to(torch.float32)
        else:
            regular_keys.append(key)

    return regular_keys, special_metrics


def reduce_tensor(
    tensor: torch.Tensor,
    world_size: int,
    reduce_group: Optional[dist.ProcessGroup],
    op: Any,
) -> torch.Tensor:
    """All-reduce a tensor if world_size > 1 and return the reduced tensor."""
    if world_size > 1:
        if reduce_group is not None:
            tensor_cpu = tensor.cpu()
            if dist.is_initialized():
                dist.reduce(tensor_cpu, dst=0, group=reduce_group, op=op)
            return tensor_cpu
        
        if dist.is_initialized():
            dist.reduce(tensor, dst=0, op=op)
    return tensor


def reduce_special_metrics(
    special_metrics: Dict[str, torch.Tensor],
    world_size: int,
    reduce_group: Optional[dist.ProcessGroup],
) -> Dict[str, torch.Tensor]:
    """Reduce special metrics (min/max) across ranks."""
    reduced: Dict[str, torch.Tensor] = {}
    for name, tensor in special_metrics.items():
        op = dist.ReduceOp.MIN if "min" in name else dist.ReduceOp.MAX
        reduced_tensor = reduce_tensor(tensor, world_size, reduce_group, op)
        reduced[name] = reduced_tensor
    return reduced


def init_special_buffer(
    name: str,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a vector buffer for tracking per-set extrema."""
    sentinel = float("-inf") if "max" in name else float("inf")
    return torch.full((size,), sentinel, dtype=torch.float32, device=device)


def update_special_buffer(buffer: torch.Tensor, index: int, value: torch.Tensor, name: str) -> None:
    """Update an extrema buffer in-place with the provided value."""
    value_float = value.detach().to(buffer.device, torch.float32)
    if "min" in name:
        buffer[index] = torch.minimum(buffer[index], value_float)
    else:
        buffer[index] = torch.maximum(buffer[index], value_float)


__all__ = [
    "SPECIAL_METRIC_KEYS",
    "partition_metrics",
    "reduce_tensor",
    "reduce_special_metrics",
    "init_special_buffer",
    "update_special_buffer",
]

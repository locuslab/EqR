import math
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist

from utils.printing import rank_zero_print_warning


def compute_pass_at_k(n: int, k: int, c: int) -> float:
    n, k, c = int(n), int(k), int(c)
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c == n:
        return 1.0
    if n - c < k:
        return 1.0
    try:
        result = 1.0 - math.comb(n - c, k) / math.comb(n, k)
        return max(0.0, min(1.0, result))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _should_log_grad_norms(config, step: int) -> bool:
    interval = getattr(config, "grad_norm_log_interval_steps", None)
    return interval is not None and interval > 0 and (step == 1 or step % interval == 0)


def _should_log_steps_hist(config, step: int) -> bool:
    interval = getattr(config, "steps_hist_log_interval_steps", None)
    return interval is not None and interval > 0 and (step == 1 or step % interval == 0)


def _collect_raw_steps_for_logging(raw_steps: torch.Tensor, world_size: int) -> List[float]:
    filtered = raw_steps.detach().flatten()
    filtered = filtered[filtered > 0]
    local_values = filtered.to(torch.float32).cpu().tolist() if filtered.numel() > 0 else []

    if world_size > 1 and dist.is_initialized():
        gathered: List[Optional[List[float]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_values)
        merged: List[float] = []
        for part in gathered:
            if part:
                merged.extend(part)
        return merged

    return local_values


def _build_steps_hist_plotly(values: Sequence[float], num_bins: int = 50):
    try:
        import plotly.graph_objects as go
    except Exception:
        rank_zero_print_warning("Plotly is not available; skipping steps histogram figure.")
        return None

    if not values:
        return None

    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=num_bins)])
    fig.update_layout(
        title="Inference Steps Histogram",
        xaxis_title="Steps",
        yaxis_title="Count",
        bargap=0.05,
    )
    return fig


__all__ = [
    "compute_pass_at_k",
    "_should_log_grad_norms",
    "_should_log_steps_hist",
    "_collect_raw_steps_for_logging",
    "_build_steps_hist_plotly",
]

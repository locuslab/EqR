import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch
from models.ema import EMAHelper
from models.losses.base import BaseLossHead
from utils.printing import colored_exception

__all__ = [
    "TrainState",
    "build_wandb_run_dir",
    "apply_to_tensors",
    "tree_to_device",
    "tree_to_cpu_detached",
    "ensure_rng_byte_tensor",
    "set_deterministic_mode",
]


@dataclass
class TrainState:

    model: BaseLossHead
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    steps_per_epoch: int
    ema_helper: Optional[EMAHelper] = None
    steps_hist_buffer: List[float] = field(default_factory=list)


def build_wandb_run_dir(run_id: str, timestamp: Optional[datetime] = None) -> str:
    ts = timestamp or datetime.now()
    date_dir = f"{ts.year}-{ts.month}-{ts.day}"
    time_dir = f"{ts.hour}-{ts.minute}-{ts.second}"
    return os.path.join("outputs", run_id, date_dir, time_dir)


def apply_to_tensors(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return fn(obj)
    if is_dataclass(obj):
        updates = {}
        for f in fields(obj):
            updates[f.name] = apply_to_tensors(getattr(obj, f.name), fn)
        return replace(obj, **updates)
    if isinstance(obj, dict):
        return {k: apply_to_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        mapped = [apply_to_tensors(v, fn) for v in obj]
        return type(obj)(mapped)
    return obj


def tree_to_device(obj: Any, device: torch.device | str):
    return apply_to_tensors(obj, lambda t: t.to(device))


def tree_to_cpu_detached(obj: Any):

    def _convert(t: torch.Tensor):
        if t.device.type != "cpu":
            return t.detach().cpu()
        return t.detach().clone()

    return apply_to_tensors(obj, _convert)


def ensure_rng_byte_tensor(state: Any) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        tensor = state.detach()
    elif isinstance(state, np.ndarray):
        tensor = torch.from_numpy(state)
    elif isinstance(state, (list, tuple)):
        tensor = torch.tensor(state, dtype=torch.uint8)
    elif isinstance(state, (bytes, bytearray)):
        tensor = torch.tensor(list(state), dtype=torch.uint8)
    else:
        colored_exception(TypeError, f"Unsupported RNG state type: {type(state)}")

    tensor = tensor.to(device="cpu", dtype=torch.uint8, copy=True)
    return tensor.contiguous()


def set_deterministic_mode(config, rank=0):
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(os.getcwd(), '.torchinductor_cache')
    os.environ['TORCHDYNAMO_CACHE_SIZE_LIMIT'] = '1'
    
    if rank == 0:
        from utils.printing import rank_zero_print_info
        rank_zero_print_info("Deterministic mode enabled:")
        rank_zero_print_info(f"  - CUDA deterministic: {torch.backends.cudnn.deterministic if torch.cuda.is_available() else 'N/A'}")
        rank_zero_print_info(f"  - CUDA benchmark: {torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'N/A'}")
        rank_zero_print_info(f"  - PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
    
    import random
    import numpy as np
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    torch.random.manual_seed(config.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + rank)
        torch.cuda.manual_seed_all(config.seed + rank)

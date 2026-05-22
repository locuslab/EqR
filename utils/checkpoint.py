"""Checkpoint loading, saving, and code snapshot helpers for EqR runs."""

from typing import Dict, List, Any
import torch
import torch.nn as nn
from dataclasses import replace
from utils.printing import colored_exception, rank_zero_print_info, rank_zero_print_warning
from config.schema import PretrainConfig
from models.ema import EMAHelper
from torch.utils.data import DataLoader
import os
import shutil
import random
import numpy as np
import torch.distributed as dist
from contextlib import suppress
import tarfile
from omegaconf import DictConfig, OmegaConf
from utils.optimizer import (
    maybe_reorder_optimizer_states,
    build_param_id_to_name_map,
    summarize_optimizer_state,
)
from utils.training import (
    TrainState,
    ensure_rng_byte_tensor,
    tree_to_cpu_detached,
    tree_to_device,
)
from utils.wandb import get_run_id
import wandb
from typing import Optional
from utils.env import get_env

__all__ = [
    "canonicalize_orig_mod_wrappers",
    "load_model_state_dict",
    "load_ema",
    "save_checkpoint",
    "load_training_state",
    "save_code_and_config",
]


def canonicalize_orig_mod_wrappers(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Insert missing _orig_mod segments so older checkpoints load with compiled models."""

    def _strip_orig_mod_segments(key: str) -> str:
        return ".".join(part for part in key.split(".") if part != "_orig_mod")

    model_state_keys = list(model.state_dict().keys())
    expected_keys = set(model_state_keys)
    canonical_to_expected: Dict[str, List[str]] = {}
    for expected_key in model_state_keys:
        canonical = _strip_orig_mod_segments(expected_key)
        canonical_to_expected.setdefault(canonical, []).append(expected_key)

    keys_to_remove: List[str] = []
    remapped_entries: Dict[str, torch.Tensor] = {}
    for key, tensor in list(state_dict.items()):
        if key in expected_keys:
            continue
        canonical = _strip_orig_mod_segments(key)
        targets = canonical_to_expected.get(canonical)
        if not targets:
            continue

        inserted = False
        for target_key in targets:
            if target_key not in state_dict and target_key not in remapped_entries:
                remapped_entries[target_key] = tensor
                inserted = True

        if inserted:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        state_dict.pop(key, None)

    if remapped_entries:
        state_dict.update(remapped_entries)
        print(f"Adjusted {len(remapped_entries)} checkpoint key(s) to include '_orig_mod' segments.")


def load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool,
    assign: bool = False,
) -> None:
    """Load model weights with consistent key normalization.

    - Inserts missing `_orig_mod` segments when loading into compiled models.
    - Optionally relaxes matching via `strict=False` (default across the repo).
    - Keeps `assign=False` by default to avoid breaking optimizers (training).
    """
    canonicalize_orig_mod_wrappers(model, state_dict)

    # `CastedSparseEmbedding.local_weights/local_ids` are process-local, per-batch cache
    # tensors. Their leading dimension depends
    # on the (per-process) batch size, so checkpoints from different batch sizes will
    # naturally mismatch. They are safe to drop during load: they will be refreshed
    # from `weights[...]` on the first forward.
    dropped_sparse_cache: List[str] = []
    for k in list(state_dict.keys()):
        if k.endswith("puzzle_emb.local_weights") or k.endswith("puzzle_emb.local_ids"):
            state_dict.pop(k, None)
            dropped_sparse_cache.append(k)
    if dropped_sparse_cache:
        rank_zero_print_warning(
            f"[Checkpoint] Dropped {len(dropped_sparse_cache)} sparse cache key(s) "
            f"(batch-size dependent): e.g. {dropped_sparse_cache[:3]}"
        )

    # Best-effort: resize puzzle embedding if the checkpoint and current config disagree.
    try:
        expected_shape = model.model.inner.puzzle_emb.weights.shape  # type: ignore[attr-defined]
    except Exception:
        expected_shape = None

    if expected_shape is not None:
        for key, tensor in list(state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                continue
            if not key.endswith("puzzle_emb.weights"):
                continue
            if tensor.shape == expected_shape:
                continue
            print(f"Resetting puzzle embedding as shape is different. Found {tensor.shape}, Expected {expected_shape}")
            state_dict[key] = torch.mean(tensor, dim=0, keepdim=True).expand(expected_shape).contiguous()

    try:
        ret = model.load_state_dict(state_dict, assign=assign, strict=strict)
        if not strict and (ret.missing_keys or ret.unexpected_keys):
            rank_zero_print_warning(f"[Checkpoint] Missing keys: {len(ret.missing_keys)}")
            if len(ret.missing_keys) < 20: 
                rank_zero_print_warning(f"  Missing: {ret.missing_keys}")
            rank_zero_print_warning(f"[Checkpoint] Unexpected keys: {len(ret.unexpected_keys)}")
            if len(ret.unexpected_keys) < 20:
                rank_zero_print_warning(f"  Unexpected: {ret.unexpected_keys}")
        return
    except Exception as e:
        rank_zero_print_warning(f"[Checkpoint] Initial load_state_dict failed: {e}, falling back to '_orig_mod' stripping.")
        # Fallback: try stripping `_orig_mod.` prefixes from checkpoint keys.
        stripped = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        canonicalize_orig_mod_wrappers(model, stripped)

        # Apply the same sparse-cache drop on the fallback dict.
        for k in list(stripped.keys()):
            if k.endswith("puzzle_emb.local_weights") or k.endswith("puzzle_emb.local_ids"):
                stripped.pop(k, None)

        ret = model.load_state_dict(stripped, assign=assign, strict=strict)
        if not strict and (ret.missing_keys or ret.unexpected_keys):
            print(f"[Checkpoint-Fallback] Missing keys: {len(ret.missing_keys)}")
            if len(ret.missing_keys) < 20:
                print(f"  Missing: {ret.missing_keys}")
            print(f"[Checkpoint-Fallback] Unexpected keys: {len(ret.unexpected_keys)}")
            if len(ret.unexpected_keys) < 20:
                print(f"  Unexpected: {ret.unexpected_keys}")


def load_ema(checkpoint_obj: Dict[str, Any], train_state, config: PretrainConfig, rank: int):
    
    ema_state = checkpoint_obj.get("ema")
    if ema_state is None:
        colored_exception(KeyError, "Checkpoint missing 'ema' state_dict for EMA model.")
    mu_default = getattr(config, "ema_rate", 0.999)
    mu = ema_state.get("mu", mu_default)
    shadow = ema_state.get("shadow")
    if shadow is None:
        tensor_entries = {k: v for k, v in ema_state.items() if isinstance(v, torch.Tensor)}
        shadow = tensor_entries if tensor_entries else {}
    if train_state.ema_helper is None:
        train_state.ema_helper = EMAHelper(mu=mu)
    train_state.ema_helper.mu = mu
    shadow = shadow or {}

    try:
        first_param = next(train_state.model.parameters())
        target_device = first_param.device
    except StopIteration:
        target_device = torch.device("cpu")
        
    shadow = {k.replace("_orig_mod.", ""): v.to(target_device) for k, v in shadow.items()}

    # Drop batch-size/process-local sparse embedding cache params from EMA.
    # These can legitimately change shape with per-process batch size (e.g. 12 -> 48)
    # and are not meaningful to EMA anyway.
    dropped_ema_keys: List[str] = []
    for k in list(shadow.keys()):
        if k.endswith("puzzle_emb.local_weights") or k.endswith("puzzle_emb.local_ids"):
            shadow.pop(k, None)
            dropped_ema_keys.append(k)
    if dropped_ema_keys and rank == 0:
        rank_zero_print_warning(
            f"[Checkpoint][EMA] Dropped {len(dropped_ema_keys)} sparse cache key(s) from EMA shadow "
            f"(batch-size dependent): e.g. {dropped_ema_keys[:3]}",
            color="yellow",
        )

    if shadow:
        train_state.ema_helper.load_state_dict(shadow)
            
    else:
        train_state.ema_helper.register(train_state.model)
    eval_model = train_state.ema_helper.ema_copy(train_state.model)
    train_state_eval = replace(train_state, model=eval_model)
    return train_state_eval



def save_checkpoint(
    config: PretrainConfig,
    train_state: TrainState,
    train_loader: DataLoader,
    *,
    next_iteration: int,
    rank: int,
):
    if rank != 0 or config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    run_id = get_run_id() or os.environ.get("WANDB_RUN_ID") or (wandb.run.id if wandb.run is not None else None)
    filename = f"step_{train_state.step}"
    if run_id:
        filename = f"{filename}_{run_id}"

    path = os.path.join(config.checkpoint_path, f"{filename}.pth")

    model_sd = train_state.model.state_dict()
    # Do not save batch-size/process-local sparse cache tensors.
    # These are refreshed from `weights[...]` during forward and may change shape
    # when per-process batch size changes (e.g. 12 -> 48), causing load failures.
    dropped_model_keys: List[str] = []
    for k in list(model_sd.keys()):
        if k.endswith("puzzle_emb.local_weights") or k.endswith("puzzle_emb.local_ids"):
            model_sd.pop(k, None)
            dropped_model_keys.append(k)

    checkpoint: Dict[str, Any] = {
        "step": train_state.step,
        "total_steps": train_state.total_steps,
        "model": model_sd,
        "optimizers": [opt.state_dict() for opt in train_state.optimizers],
        "optimizer_lrs": list(train_state.optimizer_lrs),
        "carry": tree_to_cpu_detached(train_state.carry),
        "rng": {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "train_dataset": train_loader.dataset.state_dict() if hasattr(train_loader, "dataset") else None,
        "iteration": next_iteration,
        "config": config.model_dump(),
    }

    if dropped_model_keys:
        rank_zero_print_info(
            f"[Checkpoint] Not saving {len(dropped_model_keys)} sparse cache key(s) "
            f"(batch-size dependent): e.g. {dropped_model_keys[:3]}"
        )

    if train_state.ema_helper is not None:
        ema_shadow = tree_to_cpu_detached(train_state.ema_helper.state_dict())
        dropped_ema_keys: List[str] = []
        if isinstance(ema_shadow, dict):
            for k in list(ema_shadow.keys()):
                if str(k).endswith("puzzle_emb.local_weights") or str(k).endswith("puzzle_emb.local_ids"):
                    ema_shadow.pop(k, None)
                    dropped_ema_keys.append(str(k))
        if dropped_ema_keys:
            rank_zero_print_info(
                f"[Checkpoint][EMA] Not saving {len(dropped_ema_keys)} sparse cache key(s) "
                f"(batch-size dependent): e.g. {dropped_ema_keys[:3]}"
            )
        checkpoint["ema"] = {
            "mu": train_state.ema_helper.mu,
            "shadow": ema_shadow,
        }

    checkpoint["wandb_run_id"] = run_id

    if wandb.run is not None:
        checkpoint["wandb"] = {
            "id": wandb.run.id,
            "name": wandb.run.name,
            "project": wandb.run.project,
            "entity": wandb.run.entity,
            "dir": wandb.run.dir,
            "url": getattr(wandb.run, "url", None),
        }

    torch.save(checkpoint, path)
    
    rank_zero_print_info(f"Saved checkpoint to {path}")


def load_training_state(
    config: PretrainConfig,
    train_state: TrainState,
    train_loader: DataLoader,
    *,
    rank: int,
    world_size: int,
    load_weights_only: bool = False,
) -> Optional[Dict[str, Any]]:
    if config.load_checkpoint is None:
        return None

    if rank == 0:
        print(f"Loading checkpoint {config.load_checkpoint}")

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_obj = torch.load(config.load_checkpoint, map_location=map_location, weights_only=False)

    def _can_torch_distributed_broadcast() -> bool:
        try:
            return bool(world_size > 1 and dist.is_available() and dist.is_initialized())
        except Exception:
            return False

    # Legacy checkpoints only contain model weights
    if not isinstance(checkpoint_obj, dict) or "model" not in checkpoint_obj:
        load_model_state_dict(
            train_state.model,
            checkpoint_obj,  # type: ignore[arg-type]
            strict=getattr(config, "load_strict", False),
            assign=False,
        )
        if _can_torch_distributed_broadcast():
            with torch.no_grad():
                for param in list(train_state.model.parameters()) + list(train_state.model.buffers()):
                    dist.broadcast(param, src=0)
        return None

    checkpoint: Dict[str, Any] = checkpoint_obj

    if load_weights_only:
        if rank == 0:
            rank_zero_print_info("load_weights_only=True: loading model parameters only (skipping optimizer/rng/EMA/state).")
        load_model_state_dict(
            train_state.model,
            checkpoint.get("model", {}),
            strict=getattr(config, "load_strict", False),
            assign=False,
        )
        if _can_torch_distributed_broadcast():
            with torch.no_grad():
                for param in list(train_state.model.parameters()) + list(train_state.model.buffers()):
                    dist.broadcast(param, src=0)
        return None

    load_model_state_dict(
        train_state.model,
        checkpoint["model"],
        strict=getattr(config, "load_strict", False),
        assign=False,
    )

    optimizer_states = checkpoint.get("optimizers", [])
    optimizer_states = maybe_reorder_optimizer_states(
        train_state.optimizers,
        optimizer_states,
        log_fn=(lambda msg: rank_zero_print_warning(msg)) if rank == 0 else None,
    )
    param_id_to_name = build_param_id_to_name_map(train_state.model)
    for optimizer, opt_state in zip(train_state.optimizers, optimizer_states):
        if rank == 0:
            current_group_cnt = len(optimizer.param_groups)
            ckpt_group_cnt = len(opt_state.get("param_groups", []))
            if current_group_cnt != ckpt_group_cnt:
                rank_zero_print_warning(
                    f"Optimizer param_group count mismatch (current {current_group_cnt} vs checkpoint {ckpt_group_cnt})."
                )
                debug_summary = summarize_optimizer_state(optimizer, opt_state, param_id_to_name)
                rank_zero_print_warning(debug_summary)
        try:
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if rank == 0:
                debug_summary = summarize_optimizer_state(optimizer, opt_state, param_id_to_name)
                rank_zero_print_warning("Optimizer load_state_dict failed; state summary:\n" + debug_summary)
            rank_zero_print_warning(f"Warning: Failed to load optimizer state: {e}")

    train_state.optimizer_lrs = tuple(checkpoint.get("optimizer_lrs", train_state.optimizer_lrs))
    train_state.step = int(checkpoint.get("step", train_state.step))
    if train_state.total_steps != int(checkpoint.get("total_steps", train_state.total_steps)):
        rank_zero_print_warning(
            f"Checkpoint total_steps {checkpoint.get('total_steps')} differs from current total_steps {train_state.total_steps}. Using config value.",
            color="yellow",
        )
    train_state.total_steps = max(train_state.total_steps, 1)
    if config.max_steps is not None:
        train_state.total_steps = min(train_state.total_steps, config.max_steps)
        train_state.step = min(train_state.step, train_state.total_steps)

    if checkpoint.get("carry") is not None:
        try:
            target_device = next(train_state.model.parameters()).device
        except StopIteration:
            target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train_state.carry = tree_to_device(checkpoint["carry"], target_device)

    rng_state = checkpoint.get("rng")
    if rng_state is not None:
        torch_state = rng_state.get("torch")
        if torch_state is not None:
            torch.random.set_rng_state(ensure_rng_byte_tensor(torch_state))
        cuda_state = rng_state.get("cuda")
        if torch.cuda.is_available() and cuda_state is not None:
            if isinstance(cuda_state, (list, tuple)):
                cuda_states = [ensure_rng_byte_tensor(state) for state in cuda_state]
            else:
                cuda_states = [ensure_rng_byte_tensor(cuda_state)]
            torch.cuda.set_rng_state_all(cuda_states)

    numpy_state = checkpoint.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    python_state = checkpoint.get("python")
    if python_state is not None:
        random.setstate(python_state)

    dataset_state = checkpoint.get("train_dataset")
    if dataset_state is not None:
        train_loader.dataset.load_state_dict(dataset_state)  # type: ignore[attr-defined]

    ema_state = checkpoint.get("ema")
    if ema_state is not None:
        if not getattr(config, "ema", False):
            config.ema = True
        mu_default = getattr(config, "ema_rate", 0.999)
        if isinstance(ema_state, dict):
            mu = ema_state.get("mu", mu_default)
            shadow = ema_state.get("shadow")
            if shadow is None:
                tensor_entries = {k: v for k, v in ema_state.items() if isinstance(v, torch.Tensor)}
                shadow = tensor_entries if tensor_entries else {}
        else:
            mu = mu_default
            shadow = ema_state
        if train_state.ema_helper is None:
            train_state.ema_helper = EMAHelper(mu=mu)
        train_state.ema_helper.mu = mu
        shadow = shadow or {}

        try:
            first_param = next(train_state.model.parameters())
            target_device = first_param.device
        except StopIteration:
            target_device = torch.device("cpu")

        if shadow:
            # Drop batch-size/process-local sparse embedding cache params from EMA shadow.
            # They are safe to reinit and can mismatch shapes across runs.
            for k in list(shadow.keys()):
                if str(k).endswith("puzzle_emb.local_weights") or str(k).endswith("puzzle_emb.local_ids"):
                    shadow.pop(k, None)
            train_state.ema_helper.load_state_dict(tree_to_device(shadow, target_device))
            rank_zero_print_info("Loaded EMA Shadow from checkpoint.")
        else:
            train_state.ema_helper.register(train_state.model)
    elif getattr(config, "ema", False) and train_state.ema_helper is None:
        train_state.ema_helper = EMAHelper(mu=getattr(config, "ema_rate", 0.999))
        train_state.ema_helper.register(train_state.model)

    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        current_config = config.model_dump()
        checkpoint_keys = set(checkpoint_config.keys())
        current_keys = set(current_config.keys())

        if checkpoint_keys != current_keys:
            missing = current_keys - checkpoint_keys
            extra = checkpoint_keys - current_keys
            parts = []
            if missing:
                parts.append(
                    "missing keys in checkpoint: " + ", ".join(sorted(missing))
                )
            if extra:
                parts.append(
                    "unexpected keys in checkpoint: " + ", ".join(sorted(extra))
                )
            detail = "; ".join(parts)
            if rank == 0:
                warning_lines = [
                    "Checkpoint configuration does not match the current run configuration. ",
                    f"{detail}. Ensure the training script is launched with the same configuration as when the checkpoint was created."
                ]
                warning_message = "\n".join(warning_lines)
                rank_zero_print_warning(warning_message, color="yellow")

        def _find_nested_diffs(ckpt_val, curr_val, path=""):
            """Recursively find and format differences in nested configs."""
            diffs = []
            if isinstance(ckpt_val, dict) and isinstance(curr_val, dict):
                ckpt_keys = set(ckpt_val.keys())
                curr_keys = set(curr_val.keys())
                all_keys = ckpt_keys | curr_keys
                
                for key in sorted(all_keys):
                    new_path = f"{path}.{key}" if path else key
                    cv = ckpt_val.get(key)
                    nv = curr_val.get(key)
                    
                    if key not in ckpt_keys:
                        diffs.append((new_path, "MISSING in checkpoint", None, nv))
                    elif key not in curr_keys:
                        diffs.append((new_path, "EXTRA in checkpoint", cv, None))
                    elif cv != nv:
                        if isinstance(cv, dict) and isinstance(nv, dict):
                            diffs.extend(_find_nested_diffs(cv, nv, new_path))
                        else:
                            diffs.append((new_path, "VALUE DIFFERS", cv, nv))
            elif ckpt_val != curr_val:
                diffs.append((path, "VALUE DIFFERS", ckpt_val, curr_val))
            return diffs
        
        mismatched_diffs = []
        for key in sorted(current_keys):
            checkpoint_value = checkpoint_config.get(key)
            current_value = current_config.get(key)
            if checkpoint_value != current_value:
                diffs = _find_nested_diffs(checkpoint_value, current_value, key)
                mismatched_diffs.extend(diffs)

        if mismatched_diffs and rank == 0:
            import json
            
            def _format_value(val, indent_level=0):
                """Format a value with proper indentation for readability."""
                indent = "    " * indent_level
                if isinstance(val, dict):
                    if not val:
                        return "{}"
                    try:
                        formatted = json.dumps(val, indent=2, default=str)
                        lines = formatted.split('\n')
                        if len(lines) > 1:
                            lines = [lines[0]] + [indent + line for line in lines[1:]]
                            return '\n'.join(lines)
                        return formatted
                    except Exception:
                        return repr(val)
                elif isinstance(val, list):
                    try:
                        formatted = json.dumps(val, indent=2, default=str)
                        lines = formatted.split('\n')
                        if len(lines) > 1:
                            lines = [lines[0]] + [indent + line for line in lines[1:]]
                            return '\n'.join(lines)
                        return formatted
                    except Exception:
                        return repr(val)
                else:
                    return repr(val)
            
            warning_lines = [
                "Configuration values differ between checkpoint and current run:",
            ]
            for path, status, ckpt_v, curr_v in mismatched_diffs:
                warning_lines.append(f"  {path}:")
                if status == "VALUE DIFFERS":
                    warning_lines.append(f"    checkpoint: {_format_value(ckpt_v, indent_level=1)}")
                    warning_lines.append(f"    current:    {_format_value(curr_v, indent_level=1)}")
                elif status == "MISSING in checkpoint":
                    warning_lines.append(f"    [MISSING] current: {_format_value(curr_v, indent_level=1)}")
                elif status == "EXTRA in checkpoint":
                    warning_lines.append(f"    [EXTRA] checkpoint: {_format_value(ckpt_v, indent_level=1)}")
            
            warning_lines.append("Using checkpoint values; verify that this is intentional.")
            warning_message = "\n".join(warning_lines)
            rank_zero_print_warning(warning_message, color="yellow")

        config.project_name = checkpoint_config.get("project_name", config.project_name)
        config.run_name = checkpoint_config.get("run_name", config.run_name)
        config.checkpoint_path = checkpoint_config.get("checkpoint_path", config.checkpoint_path)

    resume_info = {
        "next_iteration": int(checkpoint.get("iteration", 0)),
        "wandb": checkpoint.get("wandb"),
        "wandb_dir": checkpoint.get("wandb_dir"),
        "wandb_run_id": checkpoint.get("wandb_run_id"),
    }

    if _can_torch_distributed_broadcast():
        with torch.no_grad():
            for param in list(train_state.model.parameters()) + list(train_state.model.buffers()):
                dist.broadcast(param, src=0)

    return resume_info

def save_code_and_config(
    config: PretrainConfig,
    hydra_config: DictConfig,
    wandb_meta: Dict[str, Any],
    *,
    run_dir: Optional[str] = None,
    rank: Optional[int] = None,
):
    target_dir = run_dir or config.checkpoint_path
    if target_dir is None:
        return

    os.makedirs(target_dir, exist_ok=True)

    source_items = [
        "config", "dataclass", "dataset", "evaluators", 
        "models", "utils", "scripts",
        "pretrain.py", "evaluate.py",
    ]

    code_dir = os.path.join(target_dir, "code") 
    os.makedirs(code_dir, exist_ok=True)
    
    total_size = 0
    file_count = 0
    
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for item in source_items:
        src_path = os.path.join(workspace_root, item)
        if not os.path.exists(src_path):
            continue
            
        dst_path = os.path.join(code_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)
            total_size += os.path.getsize(src_path)
            file_count += 1
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            
            def ignore_patterns(path, names):
                return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]
                
            shutil.copytree(src_path, dst_path, ignore=ignore_patterns)
            
            for root, _, files in os.walk(src_path):
                for f in files:
                    if "__pycache__" not in root and not f.endswith(".pyc"):
                        fp = os.path.join(root, f)
                        if not os.path.islink(fp):
                             total_size += os.path.getsize(fp)
                             file_count += 1

    hydra_container = OmegaConf.to_container(hydra_config, resolve=True)
    if isinstance(hydra_container, dict):
        hydra_container["wandb_meta"] = wandb_meta
    else:
        hydra_container = {
            "hydra_config": hydra_container,
            "wandb_meta": wandb_meta,
        }
    full_config = OmegaConf.create(hydra_container)
    config_file = os.path.join(target_dir, "all_config.yaml")
    OmegaConf.save(config=full_config, f=config_file)
    
    rank_zero_print_info(f"Saved {file_count} code files to {code_dir}, total size: {total_size / 1024 / 1024:.2f} MB")

    # Log code (only when wandb active)
    # NOTE: `run.log_code(...)` is convenient but can be hard to find in the UI.
    # We also log an explicit Artifact so it reliably appears under the run's
    # "Artifacts" tab (and uploads even when git metadata is absent).
    if wandb.run is not None:
        # W&B has two relevant upload paths:
        #  - wandb.run.log_code(...): uses Artifacts under the hood.
        #  - wandb.save(...): uploads files (Files tab), often more robust.
        #
        # In some environments, artifact creation can fail with 400 errors such as
        # "Invalid Client ID digest". We therefore support a strict file-based
        # fallback for code snapshots.
        #
        # Control behavior with:
        #   WANDB_CODE_UPLOAD_MODE=file_strict|auto|artifact_strict|off
        #     file_strict (default): upload code_snapshot.tar.gz via wandb.save.
        #     auto: try log_code; if it fails, fall back to tarball upload.
        #     artifact_strict: require log_code to succeed; fail otherwise.
        #     off: do not upload code.
        #
        # file_strict keeps training stable and ensures code is visible in the run Files tab.
        code_upload_mode = (get_env("WANDB_CODE_UPLOAD_MODE") or "file_strict").strip().lower()

        def _wandb_save_now(path: str, *, base_path: Optional[str] = None) -> None:
            """Compatibility helper across wandb versions."""
            try:
                wandb.save(path, base_path=base_path, policy="now")  # type: ignore[call-arg]
            except TypeError:
                # Older clients may not support `policy`.
                wandb.save(path, base_path=base_path)

        def _maybe_make_code_tarball() -> Optional[str]:
            """Create a tar.gz snapshot of code_dir and return its path."""
            try:
                archive_path = os.path.join(target_dir, "code_snapshot.tar.gz")
                # Recreate to avoid uploading stale contents.
                with suppress(Exception):
                    if os.path.exists(archive_path):
                        os.remove(archive_path)
                with tarfile.open(archive_path, "w:gz") as tar:
                    # Store under a stable top-level folder name.
                    tar.add(code_dir, arcname="code")
                return archive_path
            except Exception:
                return None

        def _upload_code_tarball_strict() -> str:
            archive_path = _maybe_make_code_tarball()
            if archive_path is None:
                raise RuntimeError("failed to create code_snapshot.tar.gz")
            _wandb_save_now(archive_path, base_path=target_dir)
            # Also save the resolved config for convenience.
            with suppress(Exception):
                _wandb_save_now(config_file, base_path=target_dir)
            return archive_path

        if code_upload_mode in {"0", "false", "no", "off"}:
            return

        # Legacy knob: WANDB_LOG_CODE_ARTIFACT=0 disables tarball upload.
        legacy_disable_tarball = get_env("WANDB_LOG_CODE_ARTIFACT")
        tarball_enabled = not (
            legacy_disable_tarball and legacy_disable_tarball.strip().lower() in {"0", "false", "no", "off"}
        )

        # file_strict: skip artifact-based log_code; require tarball upload.
        if code_upload_mode == "file_strict":
            if not tarball_enabled:
                raise RuntimeError(
                    "WANDB_CODE_UPLOAD_MODE=file_strict requires tarball upload, but WANDB_LOG_CODE_ARTIFACT=0 disabled it."
                )
            archive_path = _upload_code_tarball_strict()
            rank_zero_print_info(f"Uploaded W&B file: {os.path.basename(archive_path)}")
            return

        # auto / artifact_strict: attempt log_code first.
        log_code_exc: Optional[BaseException] = None
        try:
            code_art = wandb.run.log_code(code_dir)
            # `log_code` logs a code artifact under the hood; wait so failures
            # (auth/backend/artifact issues) are surfaced immediately.
            if code_art is not None and hasattr(code_art, "wait"):
                code_art.wait()
            rank_zero_print_info(
                "================================================\n"
                "Logged code to W&B via wandb.run.log_code(code_dir). "
                "Note: this typically appears in the run UI under the Code section, not necessarily under Files."
            )
        except Exception as exc:
            log_code_exc = exc

        if log_code_exc is None:
            # Optional: also upload a tarball so code is easy to find in Files.
            if tarball_enabled:
                with suppress(Exception):
                    archive_path = _upload_code_tarball_strict()
                    rank_zero_print_info(f"Uploaded W&B file: {os.path.basename(archive_path)}")
            return

        # log_code failed.
        if code_upload_mode == "artifact_strict":
            # Best-effort: upload tarball for debugging, then fail hard.
            if tarball_enabled:
                with suppress(Exception):
                    _upload_code_tarball_strict()
            raise RuntimeError(
                "wandb.run.log_code(code_dir) failed (or failed to upload); cannot continue because artifact-based code logging is required. "
                f"error={log_code_exc}"
            ) from log_code_exc

        # auto mode: fall back to tarball upload if enabled.
        if tarball_enabled:
            try:
                archive_path = _upload_code_tarball_strict()
                rank_zero_print_warning(
                    "wandb.run.log_code(code_dir) failed; using code_snapshot.tar.gz via wandb.save (Files tab) instead.",
                    color="yellow",
                )
                rank_zero_print_info(f"Uploaded W&B file: {os.path.basename(archive_path)}")
                return
            except Exception as tar_exc:
                raise RuntimeError(
                    "wandb.run.log_code(code_dir) failed and fallback file upload also failed; cannot continue because code logging is required. "
                    f"log_code_error={log_code_exc}; file_upload_error={tar_exc}"
                ) from tar_exc

        # No tarball fallback available; fail.
        raise RuntimeError(
            "wandb.run.log_code(code_dir) failed, and tarball upload was disabled; cannot continue because code logging is required. "
            f"error={log_code_exc}"
        ) from log_code_exc

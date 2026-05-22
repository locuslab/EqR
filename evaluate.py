from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf

from config.schema import PretrainConfig
from evaluators.eval_config import CheckpointEvalConfig, EvalConfig
from evaluators.eval_fn import evaluate
from pretrain import create_dataloader, create_evaluators, init_train_state
from utils.checkpoint import load_ema, load_model_state_dict
from utils.env import get_env
from utils.printing import colored_exception, print_info, rank_zero_print_info, rank_zero_print_warning


torch.set_float32_matmul_precision("high")


def set_nested_attr(obj: Any, key: str, value: Any) -> None:
    cur = obj
    parts = key.replace("/", ".").split(".")
    for part in parts[:-1]:
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    if isinstance(cur, dict):
        cur[parts[-1]] = value
    else:
        setattr(cur, parts[-1], value)


def _list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)) else [value]


def _param_combos(*items: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, List[Any]] = {}
    for item in items:
        for key, value in (item or {}).items():
            merged[key.replace("/", ".")] = _list(value)
    if not merged:
        return [{}]
    keys = list(merged)
    return [dict(zip(keys, values)) for values in product(*(merged[k] for k in keys))]


def _apply_overrides(target: Any, values: Dict[str, Any], *, rank: int = 0) -> None:
    for key, value in values.items():
        try:
            set_nested_attr(target, key, value)
            if rank == 0:
                rank_zero_print_info(f"override {key}={value}")
        except Exception as exc:
            if rank == 0:
                rank_zero_print_warning(f"skipped override {key}: {exc}")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (DictConfig, ListConfig)):
        return _jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _load_eval_config(raw: EvalConfig) -> EvalConfig:
    if raw.eval_yaml is None:
        return raw
    path = Path(raw.eval_yaml)
    if not path.exists():
        colored_exception(FileNotFoundError, f"evaluation yaml not found: {path}")
    base = OmegaConf.load(path)
    cli = OmegaConf.create(raw.model_dump(mode="python", exclude_unset=True))
    data = OmegaConf.to_container(OmegaConf.merge(base, cli), resolve=True)
    if not isinstance(data, dict):
        colored_exception(ValueError, "evaluation yaml must contain a dict")
    return EvalConfig(**data)


def _load_checkpoint(path: str) -> Tuple[Dict[str, Any], str]:
    obj = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    if not isinstance(obj, dict):
        colored_exception(ValueError, "checkpoint must be a dict with model/config entries")
    return obj, path


def _dataset_path(path: str) -> str:
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return path
    prefix = get_env("DATASET_PATH_PREFIX") or get_env("DATA_PATH_PREFIX") or get_env("DATA_ROOT")
    return str(Path(os.path.expanduser(prefix)) / path) if prefix else path


def _base_config(eval_cfg: EvalConfig, ckpt: Dict[str, Any], ckpt_path: str, rank: int) -> PretrainConfig:
    cfg = ckpt.get("config")
    if cfg is None:
        cfg_path = Path(ckpt_path).parent.parent / "all_config.yaml"
        if not cfg_path.exists():
            colored_exception(FileNotFoundError, f"missing checkpoint config and {cfg_path}")
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    config = PretrainConfig(**cfg)
    config.checkpoint_path = str(Path(ckpt_path).parent)
    config.eval_save_outputs = eval_cfg.save_outputs
    if eval_cfg.loss_head_name:
        config.arch.loss.name = eval_cfg.loss_head_name
    if eval_cfg.global_batch_size is not None:
        config.global_batch_size = eval_cfg.global_batch_size
    if eval_cfg.dataset_data_path:
        config.dataset.data_path = str(eval_cfg.dataset_data_path)
    if eval_cfg.load_strict is not None:
        config.load_strict = bool(eval_cfg.load_strict)
    config.dataset.data_path = _dataset_path(config.dataset.data_path)
    _seed(config.seed, rank)
    return config


def _seed(seed: Optional[int], rank: int) -> int:
    value = int(seed or 0) + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return value - rank


def _unwrap_inner(model: Any) -> Optional[Any]:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "model"):
        model = model.model
    return getattr(model, "inner", None)


def _setup_random_reset_carry(model: Any, reset_std: float, rank: int) -> None:
    inner = _unwrap_inner(model)
    if inner is None or not hasattr(inner, "reset_carry"):
        if rank == 0:
            rank_zero_print_warning("different_init requested, but model has no inner.reset_carry")
        return
    from models.common import trunc_normal_init_

    std = max(abs(float(reset_std)), 1e-6)
    stds = getattr(inner, "_random_reset_field_stds", {}) or {}

    def random_reset(_self: Any, reset_flag: torch.Tensor, carry: Any, **_: Any) -> Any:
        mask = reset_flag.to(carry.z_H.device).view(-1, 1, 1)
        h = trunc_normal_init_(torch.empty_like(carry.z_H), std=float(stds.get("z_H", std)))
        l = trunc_normal_init_(torch.empty_like(carry.z_L), std=float(stds.get("z_L", std)))
        z_h = torch.where(mask, h, carry.z_H)
        z_l = torch.where(mask.to(carry.z_L.device), l, carry.z_L)
        return replace(carry, z_H=z_h, z_L=z_l)

    inner.reset_carry = random_reset.__get__(inner, type(inner))


def _summarize(metrics: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    flat: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                scalar = float(metric_value)
                rows.append({"split": key, "metric": metric, "value": scalar})
                flat[f"{key}/{metric}"] = scalar
        else:
            scalar = float(value)
            rows.append({"split": "global", "metric": key, "value": scalar})
            flat[key] = scalar
    return rows, flat


def _write_metrics(path: Path, name: str, rows: List[Dict[str, Any]], flat: Dict[str, float], elapsed: float, config: PretrainConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    with (path / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump({"metrics": flat, "eval_time_sec": elapsed, "config": _jsonable(config.model_dump())}, handle, indent=2, sort_keys=True)


def _init_dist() -> Tuple[int, int, Optional[dist.ProcessGroup]]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, None
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world, dist.new_group(backend="gloo")


def _run_checkpoint(eval_cfg: EvalConfig, ckpt_cfg: CheckpointEvalConfig, rank: int, world: int, cpu_group: Optional[dist.ProcessGroup]) -> None:
    ckpt, ckpt_path = _load_checkpoint(ckpt_cfg.checkpoint)
    config = _base_config(eval_cfg, ckpt, ckpt_path, rank)
    _apply_overrides(config, ckpt_cfg.init_override or {}, rank=rank)

    train_state, eval_loader, eval_metadata = _init_state(config, rank, world)
    load_model_state_dict(
        train_state.model,
        ckpt.get("model", ckpt),
        strict=bool(getattr(config, "load_strict", False)),
        assign=True,
    )
    train_state.step = int(ckpt.get("step", 0))
    train_state_eval = load_ema(ckpt, train_state, config, rank) if config.ema and "ema" in ckpt else train_state
    train_state_eval.model.eval()

    if "DISABLE_COMPILE" not in os.environ:
        train_state_eval.model = torch.compile(train_state_eval.model, dynamic=False)  # type: ignore[assignment]

    combos = _param_combos(eval_cfg.shared_param_iter, eval_cfg.param_iter, ckpt_cfg.param_iter)
    for idx, combo in enumerate(combos):
        run_config = config.model_copy(deep=True)
        _apply_overrides(run_config, combo, rank=rank)
        loss_head = getattr(train_state_eval.model, "_orig_mod", train_state_eval.model)
        _apply_overrides(loss_head.model.config, {k.removeprefix("arch."): v for k, v in combo.items() if k.startswith("arch.")}, rank=rank)
        _seed(run_config.seed, rank)
        if eval_cfg.different_init and eval_cfg.different_init > 1:
            _setup_random_reset_carry(train_state_eval.model, eval_cfg.different_init_reset_std, rank)

        suffix = ckpt_cfg.suffix or eval_cfg.suffix or "eval"
        label = "_".join(f"{k.replace('.', '_')}-{v}" for k, v in combo.items()) or f"iter_{idx:02d}"
        eval_dir = Path(run_config.checkpoint_path).parent / "eval_preds" / f"step_{train_state_eval.step}_{suffix}_{label}"
        run_config.eval_dir = str(eval_dir)
        run_config.convergence_top_k = eval_cfg.convergence_top_k
        run_config.convergence_window = eval_cfg.convergence_window
        run_config.convergence_vis_plots = eval_cfg.convergence_vis_plots

        metrics, elapsed = evaluate(
            run_config,
            train_state_eval,
            eval_loader,
            eval_metadata,
            create_evaluators(run_config, eval_metadata),
            rank=rank,
            world_size=world,
            cpu_group=cpu_group,
            max_eval_steps=eval_cfg.max_eval_steps,
            different_init=eval_cfg.different_init,
        )
        if rank == 0 and metrics:
            rows, flat = _summarize(metrics)
            name = f"eval_metrics_step_{train_state_eval.step}"
            _write_metrics(eval_dir, name, rows, flat, elapsed, run_config)
            print_info(f"saved {eval_dir / (name + '.json')}")


def _init_state(config: PretrainConfig, rank: int, world: int):
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=rank,
        world_size=world,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=rank,
        world_size=world,
    )
    return init_train_state(config, train_metadata, rank=rank, world_size=world), eval_loader, eval_metadata


def launch() -> None:
    cli = OmegaConf.to_container(OmegaConf.from_cli(), resolve=True)
    if not isinstance(cli, dict):
        colored_exception(ValueError, "CLI arguments must parse to a dict")
    cli = {str(k).lstrip("-"): v for k, v in cli.items()}
    eval_cfg = _load_eval_config(EvalConfig(**cli))
    rank, world, cpu_group = _init_dist()

    checkpoints: List[CheckpointEvalConfig] = list(eval_cfg.checkpoint_list)
    if eval_cfg.checkpoint:
        checkpoints.insert(
            0,
            CheckpointEvalConfig(
                checkpoint=eval_cfg.checkpoint,
                param_iter=eval_cfg.param_iter,
                suffix=eval_cfg.suffix,
                description=eval_cfg.description,
                wandb_tags=eval_cfg.wandb_tags,
                wandb_run_name=eval_cfg.wandb_run_name,
            ),
        )
    if not checkpoints:
        colored_exception(ValueError, "no checkpoint provided")
    if rank == 0:
        print_info(f"evaluating {len(checkpoints)} checkpoint(s)")

    try:
        for ckpt_cfg in checkpoints:
            _run_checkpoint(eval_cfg, ckpt_cfg, rank, world, cpu_group)
    finally:
        if dist.is_initialized():
            if cpu_group is not None:
                dist.destroy_process_group(cpu_group)
            dist.destroy_process_group()


if __name__ == "__main__":
    launch()

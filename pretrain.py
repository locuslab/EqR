from typing import Optional, Any, List, Dict
from dataclasses import replace
import os
import math

import multiprocessing as mp
from multiprocessing.managers import SyncManager

from utils.env import get_env
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

import tqdm
import random
import numpy as np
from evaluators.eval_fn import evaluate
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
torch.set_float32_matmul_precision('high')
try:
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
except Exception:
    pass  # Older PyTorch versions do not have this setting

from dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from utils.training import (
    build_wandb_run_dir,
    set_deterministic_mode,
    TrainState,
    tree_to_device,
)
from utils.printing import rank_zero_print_warning, rank_zero_print_info, _log_line
from utils.metric_reduce import (
    SPECIAL_METRIC_KEYS,
    partition_metrics,
    reduce_special_metrics,
    reduce_tensor,
)

from utils.checkpoint import (
    load_training_state,
    save_checkpoint,
    save_code_and_config,
)

from datetime import datetime, timedelta
from models.ema import EMAHelper
from utils.metrics import (
    _should_log_grad_norms,
    _should_log_steps_hist, _collect_raw_steps_for_logging, _build_steps_hist_plotly,
)
from utils.wandb import set_run as wandb_set_run
from utils.wandb import _config_to_wandb_dict
from utils.optimizer import compute_lr
from utils.secrets import apply_wandb_secrets_to_training_config
from config.schema import PretrainConfig
from contextlib import suppress

WANDB_RUN_ID: Optional[str] = None
WANDB_RUN_DIR: Optional[str] = None

_SHARED_STATE_MANAGERS: List[SyncManager] = []


def _resolve_run_root(hydra_config: Optional[DictConfig] = None) -> str:
    """Resolve a stable root directory for outputs.

    Hydra typically changes the process CWD to a per-job directory. If we base
    output paths on os.getcwd(), results end up scattered under the Hydra run
    directory and users won't see `./outputs/<WANDB_RUN_ID>/...` where they
    expect.

    Priority:
      1) OUTPUT_ROOT / RUN_ROOT (explicit override)
      2) hydra.runtime.cwd (original working directory)
      3) directory containing this script (repo root in this project)
      4) current working directory
    """

    env_root = get_env("OUTPUT_ROOT") or get_env("RUN_ROOT")
    if isinstance(env_root, str) and env_root.strip():
        return os.path.abspath(os.path.expanduser(os.path.expandvars(env_root.strip())))

    if hydra_config is not None:
        with suppress(Exception):
            original_cwd = OmegaConf.select(hydra_config, "hydra.runtime.cwd")
            if isinstance(original_cwd, str) and original_cwd.strip():
                return os.path.abspath(os.path.expanduser(os.path.expandvars(original_cwd.strip())))

    with suppress(Exception):
        return os.path.dirname(os.path.abspath(__file__))

    return os.getcwd()


def apply_gradient_checkpointing(model: nn.Module, config: PretrainConfig, rank: int):
    """
    Apply gradient checkpointing to model layers without modifying the original model class.
    This wraps the forward methods of reasoning blocks with torch.utils.checkpoint.
    """
    if not config.gradient_checkpoint:
        return model
    
    if rank == 0:
        rank_zero_print_info("Applying gradient checkpointing to model layers...")
    
    inner_model = model.model if hasattr(model, 'model') else model
    
    if hasattr(inner_model, 'inner'):
        inner_model = inner_model.inner

    if hasattr(inner_model, 'L_level') and hasattr(inner_model.L_level, 'layers'):
        original_layers = inner_model.L_level.layers

        for i, layer in enumerate(original_layers):
            original_forward = layer.forward

            def create_checkpointed_forward(orig_forward):
                def checkpointed_forward(*args, **kwargs):
                    if layer.training:
                        def forward_wrapper(*tensor_args):
                            return orig_forward(*tensor_args, **kwargs)

                        return checkpoint(forward_wrapper, *args, use_reentrant=False)
                    else:
                        return orig_forward(*args, **kwargs)
                return checkpointed_forward
            
            layer.forward = create_checkpointed_forward(original_forward)
        
        if rank == 0:
            rank_zero_print_info(f"Applied gradient checkpointing to {len(original_layers)} layers in L_level")
    
    return model


def create_dataloader(
    config: PretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    **kwargs,
):
    enable_shared_sampler_state = kwargs.pop("enable_shared_sampler_state", True)

    shared_sampler_state = None
    if split == "train" and enable_shared_sampler_state:
        try:
            manager = mp.Manager()
        except Exception as exc:  # pragma: no cover - fallback for sandboxed environments
            rank_zero_print_warning(
                f"Shared sampler state disabled (multiprocessing.Manager unavailable): {exc}"
            )
            enable_shared_sampler_state = False
        else:
            _SHARED_STATE_MANAGERS.append(manager)
            shared_sampler_state = manager.dict()

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed, dataset_path=config.dataset.data_path, rank=rank, num_replicas=world_size, **kwargs
        ),
        split=split,
        shared_sampler_state=shared_sampler_state,
    )
    
    def worker_init_fn(worker_id: int):
        """Set random seeds for each worker to ensure reproducibility."""
        worker_seed = config.seed + rank + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    loader_kwargs = {
        "batch_size": None,
        "num_workers": 1,
        "persistent_workers": True,
        "pin_memory": torch.cuda.is_available(),
        "prefetch_factor": 8,
    }

    dataloader = DataLoader(
        dataset,
        **loader_kwargs,
        worker_init_fn=worker_init_fn,
    )
    return dataloader, dataset.metadata


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    init_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device(init_device):
        model: nn.Module = model_cls(model_cfg)
        loss_kwargs = dict(config.arch.loss.__pydantic_extra__)
        model = loss_head_cls(model, **loss_kwargs)  # type: ignore
        
        model = apply_gradient_checkpointing(model, config, rank)

        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    optimizer_kwargs = dict(config.optimizer_kwargs)
    rank_zero_print_info(f"Using optimizer: adam_atan2 with args: {optimizer_kwargs}")
    from adam_atan2 import AdamATan2

    betas = optimizer_kwargs.pop("betas", None)
    if betas is None:
        beta1 = optimizer_kwargs.pop("beta1", config.beta1)
        beta2 = optimizer_kwargs.pop("beta2", config.beta2)
        betas = (beta1, beta2)

    weight_decay = config.weight_decay
    optimizers = [
        AdamATan2(
            model.parameters(),
            lr=0,  # Needs to be set by scheduler
            weight_decay=weight_decay,
            betas=betas,
            **optimizer_kwargs,
        ),
    ]

    return model, optimizers, [config.lr]



def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    """
    Here, we are actually not calculating the total_samples in the dataset, but the total number of examples to be seen during training.
    say we have 1000 original items, if we generate with 1000 aug per item, then we have 1001 x 1000 examples saved to local
    However, when training, we only sample one item from one group of puzzles. That sampled item contributes however many examples it contains toward filling the batch.
    """

    total_samples = int(
        train_metadata.total_groups * train_metadata.mean_puzzle_examples
    )

    steps_per_epoch = max(
        math.floor(total_samples / max(int(config.global_batch_size), 1)),
        1,
    )
    estimated_steps = int(config.epochs * steps_per_epoch)
    rank_zero_print_info(f"""
epochs: {config.epochs}
total_groups: {train_metadata.total_groups}
mean_puzzle_examples: {train_metadata.mean_puzzle_examples}
puzzle_ids: {train_metadata.num_puzzle_identifiers}
vocab_size: {train_metadata.vocab_size}
seq_len: {train_metadata.seq_len}

total_samples: {total_samples}
global_batch_size: {config.global_batch_size}                         
Estimated total training steps: {estimated_steps}""")
    rank_zero_print_info(f"Steps per epoch: {steps_per_epoch}")
    if config.max_steps is not None:
        total_steps = min(estimated_steps, config.max_steps)
        rank_zero_print_info(f"Using max_steps limit: {config.max_steps}, total_steps set to: {total_steps}")
    else:
        total_steps = estimated_steps

    model, optimizers, optimizer_lrs = create_model(
        config,
        train_metadata,
        rank=rank,
        world_size=world_size,
    )

    ema_helper = None
    if getattr(config, "ema", False):
        ema_helper = EMAHelper(mu=getattr(config, "ema_rate", 0.999))
        ema_helper.register(model)

    return TrainState(
        step=0,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        ema_helper=ema_helper,
    )


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    evaluators = []
    for cfg in config.dataset.evaluators:
        cls = load_model_class(cfg.name, "evaluators.")(
            data_path=config.dataset.data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
        )  # type: ignore
        evaluators.append(cls)

    return evaluators


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        train_state.step = train_state.total_steps
        return

    try:
        device = next(train_state.model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = tree_to_device(batch, device)

    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    try:
        train_state.carry.global_step = int(train_state.step)
    except Exception:
        pass

    steps_per_epoch = max(train_state.steps_per_epoch, 1)
    current_epoch = (train_state.step - 1) // steps_per_epoch


    model_forward_signatures = list(train_state.model.forward.__code__.co_varnames)
    
    if "compute_target_q" in model_forward_signatures:
        compute_target_q = train_state.step % config.target_q_update_every == 0
        train_state.carry, loss, metrics, _, all_finish = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[], compute_target_q=compute_target_q
        )
    else:
        train_state.carry, loss, metrics, _, all_finish = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[]
        )
    try:
        train_state.carry.global_step = int(train_state.step)
    except Exception:
        pass

    loss_to_backprop = (1 / global_batch_size) * loss

    retain_graph = bool(getattr(config, "retain_graph", False))
    loss_to_backprop.backward(retain_graph=retain_graph)

    if world_size > 1:
        for param in train_state.model.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
            dist.all_reduce(param.grad)

    norms = {}
    if _should_log_grad_norms(config, train_state.step):
        total_sq: float = 0.0
        max_norm: float = 0.0
        num_tensors: int = 0
        for param in train_state.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            param_norm = float(torch.norm(grad.to(torch.float32)))
            total_sq += param_norm * param_norm
            if param_norm > max_norm:
                max_norm = param_norm
            num_tensors += 1

        norms["train/grad_norm/total"] = float(math.sqrt(total_sq)) if num_tensors > 0 else 0.0
        norms["train/grad_norm/max"] = float(max_norm)
        norms["train/grad_norm/num_tensors"] = float(num_tensors)

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step

        optim.step()
        optim.zero_grad()

    raw_steps = metrics.pop("raw_steps", None)
    if raw_steps is not None and getattr(config, "steps_hist_log_interval_steps", None):
        steps_for_hist = _collect_raw_steps_for_logging(raw_steps, world_size)
        if steps_for_hist and rank == 0:
            train_state.steps_hist_buffer.extend(steps_for_hist)

    if len(metrics):

        metric_keys, special_metric_tensors = partition_metrics(metrics, SPECIAL_METRIC_KEYS)
        if not metric_keys:
            raise RuntimeError("No aggregatable metrics were produced during training.")

        metric_values = torch.stack([metrics[k] for k in metric_keys])
        metric_values = reduce_tensor(metric_values, world_size, None, dist.ReduceOp.SUM)

        special_reduce_results: Dict[str, float] = {}
        if special_metric_tensors:
            reduced_special = reduce_special_metrics(special_metric_tensors, world_size, None)
            if rank == 0:
                for sk, tensor in reduced_special.items():
                    special_reduce_results[sk] = float(tensor.detach().cpu().item())

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(float(reduced_metrics.get("count", 1.0)), 1.0)
            processed_metrics: Dict[str, float] = {}
            for k, v in reduced_metrics.items():
                if "count" in k:
                    continue
                if k.startswith("fprl/"):
                    processed_metrics[f"train/{k}"] = v / float(world_size)
                else:
                    if k in {"zH_zL_cos_sim", "zH_zL_pearson", "zH_mean", "zH_std", "zL_mean", "zL_std"}:
                        denom = float(global_batch_size)
                    else:
                        denom = global_batch_size if k.endswith("loss") else count
                    processed_metrics[f"train/{k}"] = v / denom


            for sk, global_value in special_reduce_results.items():
                processed_metrics[f"train/{sk}"] = global_value

            processed_metrics["train/lr"] = lr_this_step
            processed_metrics["train/epoch"] = current_epoch
            if _should_log_steps_hist(config, train_state.step) and train_state.steps_hist_buffer:
                processed_metrics["train/steps_hist"] = wandb.Histogram(train_state.steps_hist_buffer, num_bins=config.arch.halt_max_steps)
                processed_metrics["train/steps_hist_count"] = len(train_state.steps_hist_buffer)
                if getattr(config, "steps_hist_plotly", False):
                    fig = _build_steps_hist_plotly(train_state.steps_hist_buffer, num_bins = config.arch.halt_max_steps)
                    if fig is not None:
                        processed_metrics["train/steps_hist_plotly"] = fig
                train_state.steps_hist_buffer.clear()

            processed_metrics.update(norms)
            metrics_valid = reduced_metrics.get("count", None)
            if not metrics_valid:
                drop_when_empty = {
                "train/accuracy",
                "train/exact_accuracy",
                "train/q_halt_accuracy",
                "train/q_halt_precision",
                "train/q_halt_recall",
                "train/halted_q_logits",
                "train/max_q_logits",
                "train/min_q_logits",
                "train/steps",
                "train/step_utility",
                "train/max_steps",
                "train/min_steps",
            }
                for key in drop_when_empty:
                    processed_metrics.pop(key, None)
            return processed_metrics 


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore
        apply_wandb_secrets_to_training_config(config)

        if config.run_name is None:
            model_id = config.arch.short_name if config.arch.short_name else config.arch.name.split('@')[-1]
            config.run_name = f"{model_id}-{config.dataset.name}"
            
        objects = [config]
        
        rank_zero_print_info(f"Run Name: {config.run_name}, Project Name: {config.project_name}")

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl",timeout=timedelta(hours=24))

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo",timeout=timedelta(hours=24))
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    set_deterministic_mode(config, rank=RANK)

    if not config.train_epochs_per_iter:
        config.train_epochs_per_iter = config.epochs
    train_epochs_per_iter = config.train_epochs_per_iter
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    if RANK == 0:
        print(f"running on dataset {config.dataset.name} from {config.dataset.data_path}")

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
        evaluators = create_evaluators(config, eval_metadata)
    except FileNotFoundError:
        eval_loader = eval_metadata = None
        evaluators = []

    # Reset after data loading so all ranks initialize identical models.
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    torch.manual_seed(config.seed)
    train_state = init_train_state(
        config,
        train_metadata,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    resume_info = load_training_state(
        config,
        train_state,
        train_loader,
        rank=RANK,
        world_size=WORLD_SIZE,
        load_weights_only=config.load_weights_only,
    )

    start_iteration = 0
    if resume_info is not None:
        start_iteration = min(int(resume_info.get("next_iteration", 0)), total_iters)

    progress_bar = None
    wandb_id_short = None
    if RANK == 0:

        run_start_time = datetime.now()
        wandb_kwargs: Dict[str, Any] = {
            "project": config.project_name,
            "config": _config_to_wandb_dict(config),
            "mode": (
                (config.wandb_mode.lower() if isinstance(config.wandb_mode, str) else config.wandb_mode)
                or "online"
            ),
            "entity": config.entity
        }
        if wandb_kwargs["mode"] == "disable":
            wandb_kwargs["mode"] = "disabled"
        
        if resume_info is None:
            wandb_kwargs.update({"name": config.run_name, "tags": [f"{config.dataset.name}"]})

        if resume_info is not None and resume_info.get("wandb") is not None:
            wandb_meta = resume_info["wandb"]
            if wandb_meta.get("name"):
                config.run_name = wandb_meta["name"]
                wandb_kwargs["name"] = config.run_name

            if (
                wandb_meta.get("id")
                and not config.load_weights_only
            ):
                wandb_kwargs["id"] = wandb_meta["id"]
                wandb_kwargs["resume"] = "must"

        # Honor a user-provided W&B run id when resuming a tracked run.
        env_run_id = os.environ.get("WANDB_RUN_ID")
        if resume_info is None and env_run_id:
            wandb_kwargs.setdefault("id", env_run_id)

        wandb_run_id = wandb_kwargs.get("id") or wandb.util.generate_id()
        wandb_kwargs.setdefault("id", wandb_run_id)

        run_root = _resolve_run_root(hydra_config)
        computed_run_dir = os.path.join(run_root, build_wandb_run_dir(wandb_run_id, timestamp=run_start_time))
        computed_run_dir = os.path.abspath(computed_run_dir)

        os.makedirs(computed_run_dir, exist_ok=True)

        global WANDB_RUN_ID, WANDB_RUN_DIR
        WANDB_RUN_ID = wandb_run_id
        rank_zero_print_info(f"WANDB_RUN_ID: {WANDB_RUN_ID}, {wandb_kwargs.get('id')}") if RANK == 0 else None
        WANDB_RUN_DIR = computed_run_dir
        wandb_set_run(WANDB_RUN_ID, WANDB_RUN_DIR)

        wandb_kwargs["dir"] = computed_run_dir
        rank_zero_print_info(f"Results will be saved in: {WANDB_RUN_DIR}") if RANK == 0 else None

        run = wandb.init(**wandb_kwargs)  # type: ignore[arg-type]
        wandb.define_metric("*", step_metric="step", step_sync=True)

        active_run_dir = WANDB_RUN_DIR
        config.checkpoint_path = os.path.join(active_run_dir, "checkpoints")
        os.makedirs(config.checkpoint_path, exist_ok=True)

        if progress_bar is None:
            wandb_id_short = WANDB_RUN_ID[-3:] if WANDB_RUN_ID else "unknown"
            progress_bar = tqdm.tqdm(
                total=train_state.total_steps,
                initial=train_state.step,
                desc=f"Training [{wandb_id_short}]",
                dynamic_ncols=True,
                leave=True,
                position=0,
                miniters=1,
            )
            progress_bar.set_postfix({"loss": 0.0})

        if run is not None:
            run.config.update({"checkpoint_path": config.checkpoint_path}, allow_val_change=True)

        if wandb.run is not None and train_state.step > 0:
            wandb.run._step = train_state.step

        wandb_meta_for_config = {
            "id": WANDB_RUN_ID,
            "name": config.run_name,
            "project": config.project_name,
            "entity": getattr(run, "entity", None) if run is not None else wandb_kwargs.get("entity"),
            "dir": active_run_dir,
            "url": getattr(run, "url", None) if run is not None else None,
            "mode": wandb_kwargs["mode"],
            "resume": wandb_kwargs.get("resume"),
            "tags": wandb_kwargs.get("tags"),
        }
        should_save_code = (RANK == 0)

        if should_save_code:
            save_code_and_config(
                config,
                hydra_config,
                wandb_meta_for_config,
                run_dir=active_run_dir,
                rank=RANK,
            )
            num_params = sum(x.numel() for x in train_state.model.parameters())
            rank_zero_print_info(f"Model parameters: {num_params}")
            wandb.log({
                "num_params": num_params,
                "step": train_state.step
            })

    if "DISABLE_COMPILE" not in os.environ:
        # Keep compilation deterministic across restarts and ranks.
        compile_kwargs = {
            "dynamic": False,
            "mode": "default",
        }

        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.cache_size_limit = 32
        
        train_state.model = torch.compile(train_state.model, **compile_kwargs)  # type: ignore
        
        if config.ema:
            train_state.ema_helper.register(train_state.model)
            rank_zero_print_info("Model compiled with torch.compile() in deterministic mode. EMA helper re-registered.")
    else:
        rank_zero_print_warning("Model compile is disabled.")


    if config.ema and RANK == 0:
        _log_line("Setup EMA", progress_bar=progress_bar)
        
    eval_interval_steps = config.eval_interval_steps
    checkpoint_interval_steps = config.checkpoint_interval_steps
    has_eval_loader = eval_loader is not None and eval_metadata is not None

    last_ckpt_step: Optional[int] = None
    training_done = False

    for _iter_id in range(start_iteration, total_iters):
        if training_done:
            if RANK == 0:
                _log_line("Training completed.", progress_bar=progress_bar)
            break
        if RANK == 0 and progress_bar is not None:
            wandb_id_display = wandb_id_short if wandb_id_short else "unknown"
            progress_bar.set_description(
                f"Training [{wandb_id_display}] E{_iter_id * train_epochs_per_iter}-{_iter_id * train_epochs_per_iter + train_epochs_per_iter - 1}"
            )
            progress_bar.refresh()

        train_state.model.train()
        for i, (set_name, batch, global_batch_size) in enumerate(train_loader):
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
            )

            if metrics is not None:
                metrics["step"] = train_state.step
                if RANK == 0:
                    wandb.log(metrics)
                    if progress_bar is not None:
                        progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                        train_loss = metrics.get("train/total_loss", metrics.get("train/lm_loss", 0.0))
                        progress_bar.set_postfix({"loss": f"{train_loss:.6f}"})

            if train_state.ema_helper is not None:
                train_state.ema_helper.update(train_state.model)
                
            if (
                checkpoint_interval_steps is not None
                and train_state.step > 0
                and train_state.step % checkpoint_interval_steps == 0
            ):
                if train_state.step % (train_state.steps_per_epoch * config.train_epochs_per_iter) == 0:
                    next_iteration = (_iter_id + 1)
                else:
                    next_iteration = _iter_id
                save_checkpoint(
                    config,
                    train_state,
                    train_loader,
                    next_iteration=next_iteration,
                    rank=RANK,
                )
                last_ckpt_step = train_state.step

            if (
                has_eval_loader
                and eval_interval_steps is not None
                and train_state.step > 0
                and train_state.step % eval_interval_steps == 0
            ):
                if train_state.ema_helper is not None:
                    if RANK == 0:
                        _log_line("Switching to EMA weights for evaluation", progress_bar=progress_bar)
                    eval_model = train_state.ema_helper.ema_copy(train_state.model)
                    train_state_eval = replace(train_state, model=eval_model)
                else:
                    train_state_eval = train_state
                train_state_eval.model.eval()

                eval_metrics, eval_time_passed = evaluate(
                    config,
                    train_state_eval,
                    eval_loader,
                    eval_metadata,
                    evaluators,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP,
                    progress_bar=progress_bar,
                )

                if RANK == 0 and eval_metrics is not None:
                    eval_metrics["step"] = train_state.step
                    print(eval_metrics)
                    wandb.log(eval_metrics)

                train_state.model.train()
                if config.gradient_checkpoint:
                    train_state.model = apply_gradient_checkpointing(train_state.model, config, RANK)
                if RANK == 0 and progress_bar is not None:
                    progress_bar.refresh()
                    

            if train_state.step >= train_state.total_steps:
                training_done = True
                break

        if RANK == 0 and last_ckpt_step != train_state.step:
            if training_done or (_iter_id == total_iters - 1):
                save_checkpoint(
                    config,
                    train_state,
                    train_loader,
                    next_iteration=_iter_id + 1,
                    rank=RANK,
                )
                last_ckpt_step = train_state.step
                
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
        
    launch()

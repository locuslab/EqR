from __future__ import annotations

from typing import Any, Dict, List, Optional

import pydantic

from utils.printing import colored_exception


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    short_name: Optional[str] = None
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


class DatasetConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    data_path: str
    evaluators: List[EvaluatorConfig] = []


class PretrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    arch: ArchConfig
    dataset: DatasetConfig
    optimizer_kwargs: Dict[str, Any] = pydantic.Field(default_factory=dict)

    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    gradient_accumulation_steps: int = 1

    weight_decay: float
    beta1: float
    beta2: float

    grad_clip_norm: Optional[float] = None

    target_q_update_every: int

    project_name: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    resume: str | bool | None = False
    load_checkpoint: Optional[str] = None
    load_weights_only: bool = False
    load_strict: bool = False

    seed: int = 0
    train_epochs_per_iter: Optional[int] = None
    eval_interval_steps: Optional[int] = None
    eval_save_outputs: List[str] = []
    heavy_metrics_log_interval: Optional[int] = 100
    steps_hist_log_interval_steps: Optional[int] = 100
    steps_hist_plotly: bool = True
    checkpoint_interval_steps: Optional[int] = None
    max_steps: Optional[int] = None

    wandb_mode: Optional[str] = None

    ema: bool = False
    ema_rate: float = 0.999

    debug: bool = False
    debug_trace_steps: int = 0
    grad_norm_log_interval_steps: Optional[int] = None
    hidden_state_log_interval_steps: Optional[int] = None

    gradient_checkpoint: bool = False
    retain_graph: bool = False

    eval_sync_max_wait_time: float = 300.0
    eval_sync_poll_interval: float = 2.0
    eval_sync_status_log_interval: float = 10.0

    eval_description: Optional[str] = None
    eval_dir: Optional[str] = None

    convergence_window: int = 10
    convergence_top_k: Optional[int] = None
    convergence_vis_plots: List[str] = []

    @pydantic.model_validator(mode="after")
    def _validate_intervals(self) -> PretrainConfig:
        if self.train_epochs_per_iter is not None and self.train_epochs_per_iter <= 0:
            colored_exception(ValueError, "train_epochs_per_iter must be a positive integer when provided.")
        if self.eval_interval_steps is not None and self.eval_interval_steps <= 0:
            colored_exception(ValueError, "eval_interval_steps must be a positive integer when provided.")
        if self.steps_hist_log_interval_steps is not None and self.steps_hist_log_interval_steps <= 0:
            colored_exception(ValueError, "steps_hist_log_interval_steps must be a positive integer when provided.")
        if self.checkpoint_interval_steps is not None and self.checkpoint_interval_steps <= 0:
            colored_exception(ValueError, "checkpoint_interval_steps must be a positive integer when provided.")
        if self.max_steps is not None and self.max_steps <= 0:
            colored_exception(ValueError, "max_steps must be a positive integer when provided.")
        if self.ema_rate <= 0.0 or self.ema_rate > 1.0:
            colored_exception(ValueError, "ema_rate must be in the interval (0, 1].")
        return self


__all__ = [
    "LossConfig",
    "ArchConfig",
    "EvaluatorConfig",
    "DatasetConfig",
    "PretrainConfig",
]

from typing import Any, Dict, List, Optional, Union

import pydantic
from pydantic import Field


class CheckpointEvalConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    checkpoint: str
    param_iter: Dict[str, Any] = Field(default_factory=dict)
    suffix: Optional[str] = None
    description: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_run_name: Optional[str] = None
    init_override: Optional[Dict[str, Any]] = None


class EvalConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow", populate_by_name=True)

    checkpoint: Optional[str] = None
    checkpoint_list: List[CheckpointEvalConfig] = Field(default_factory=list)
    global_batch_size: Optional[int] = None
    dataset_data_path: Optional[str] = None
    load_strict: Optional[bool] = None
    top_k: Optional[Union[int, List[int]]] = None
    eval_yaml: Optional[str] = Field(default=None, alias="eval_config")
    suffix: Optional[str] = None
    save_outputs: List[str] = Field(default_factory=lambda: ["inputs", "labels", "puzzle_identifiers", "logits"])
    param_iter: Dict[str, Any] = Field(default_factory=dict)
    shared_param_iter: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    max_eval_steps: Optional[int] = None
    force_rerun: bool = False
    wandb_run_name: Optional[str] = Field(default=None, alias="name")
    wandb_tags: Optional[List[str]] = None
    loss_head_name: Optional[str] = "losses@InferenceLossHead"
    different_init: Optional[int] = None
    different_init_reset_std: float = 1.0
    convergence_top_k: Optional[int] = None
    convergence_window: int = 3
    convergence_vis_plots: List[str] = Field(default_factory=list)

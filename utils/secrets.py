from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _wandb_secrets() -> dict[str, Any]:
    path = Path(os.environ.get("EQR_SECRETS_FILE", "config/secrets.yaml")).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text()) or {}).get("wandb") or {}


def _apply(config: Any, project_attr: str, entity_attr: str) -> None:
    wandb = _wandb_secrets()
    project = wandb.get("project")
    entity = wandb.get("entity")
    if project and getattr(config, project_attr, None) is None:
        setattr(config, project_attr, str(project))
    if entity and getattr(config, entity_attr, None) is None:
        setattr(config, entity_attr, str(entity))


def apply_wandb_secrets_to_training_config(config: Any) -> None:
    _apply(config, "project_name", "entity")


def apply_wandb_secrets_to_eval_config(config: Any) -> None:
    _apply(config, "wandb_project", "wandb_entity")

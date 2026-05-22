import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from exceptiongroup import suppress
from omegaconf import OmegaConf

from utils.printing import rank_zero_print_info, rank_zero_print_warning
import wandb

if TYPE_CHECKING:
    from config.schema import PretrainConfig

_RUN_ID: Optional[str] = None
_RUN_DIR: Optional[str] = None


__all__ = [
    "set_run",
    "get_run_id",
    "_WandbTableManager",
    "_config_to_wandb_dict",
]


def set_run(run_id: Optional[str], run_dir: Optional[str] = None) -> None:
    global _RUN_ID, _RUN_DIR
    _RUN_ID = run_id
    if run_dir is not None:
        _RUN_DIR = run_dir
    if run_id:
        os.environ["WANDB_RUN_ID"] = run_id
    if run_dir:
        os.environ["WANDB_RUN_DIR"] = run_dir


def get_run_id() -> Optional[str]:
    return _RUN_ID


class _WandbTableManager:
    def __init__(
        self,
        run: Optional[wandb.sdk.wandb_run.Run],
        table_name: str,
        *,
        base_columns: Iterable[str],
        artifact_alias: Optional[str] = "latest",
    ) -> None:
        self.run = run
        self.table_name = table_name
        self.artifact_alias = artifact_alias
        self.base_columns = list(base_columns)
        self.columns: List[str] = list(self.base_columns)
        self.rows: List[Dict[str, Any]] = []
        self.artifact_type: str = "dataset"
        if self.run is not None:
            self._load_existing_table()

    def _load_existing_table(self) -> None:
        if not self.artifact_alias:
            return
        try:
            artifact = self.run.use_artifact(self.artifact_alias)  # type: ignore[union-attr]
            rank_zero_print_info(f"Loaded artifact '{self.artifact_alias}' for wandb table '{self.table_name}'.")
            existing_type = getattr(artifact, "type", None)
            if isinstance(existing_type, str):
                if existing_type == "run_table":
                    rank_zero_print_warning(
                        "Artifact type 'run_table' is reserved; overriding to 'dataset' for future logging."
                    )
                    self.artifact_type = "dataset"
                else:
                    self.artifact_type = existing_type
        except Exception as exc:  # pragma: no cover - network interactions
            rank_zero_print_warning(f"Warning: failed to load artifact '{self.artifact_alias}': {exc}")
            return
        try:
            table = artifact.get(self.table_name)
            rank_zero_print_info(f"Loaded existing wandb table '{self.table_name}' from artifact '{self.artifact_alias}'.")
        except Exception as exc:  # pragma: no cover - artifact schema errors
            rank_zero_print_warning(
                f"Warning: artifact '{self.artifact_alias}' missing table '{self.table_name}': {exc}"
            )
            return

        existing_columns = list(getattr(table, "columns", []))
        existing_rows = getattr(table, "data", [])

        if existing_columns:
            self.columns = list(existing_columns)
        for row in existing_rows:
            row_dict = {
                column: row[idx] if idx < len(row) else None
                for idx, column in enumerate(existing_columns)
            }
            self.rows.append(self._cast_row_values(row_dict))

        self._normalize_column_order()

    def append_row(self, row: Dict[str, Any]) -> None:
        normalized_row = self._cast_row_values(row)
        self.rows.append(normalized_row)
        for key in row.keys():
            if key not in self.columns:
                self.columns.append(key)
        self._normalize_column_order()

    def _normalize_column_order(self) -> None:
        base_present = [col for col in self.base_columns if col in self.columns]
        remainder = [col for col in self.columns if col not in base_present]
        remainder_sorted = sorted(remainder)
        self.columns = base_present + remainder_sorted

    def log(self) -> None:
        if self.run is None:
            return
        table = wandb.Table(columns=self.columns)
        for row in self.rows:
            table.add_data(*(row.get(column) for column in self.columns))
        self.run.log({self.table_name: table})
        self._log_artifact(table)

    def _log_artifact(self, table: wandb.Table) -> None:
        if not self.artifact_alias:
            return
        artifact_name, alias = self._split_artifact_alias()
        try:
            artifact_type = self.artifact_type or "dataset"
            if artifact_type == "run_table":
                artifact_type = "dataset"
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add(table, self.table_name)
            self.run.log_artifact(artifact, aliases=[alias])
        except Exception as exc:  # pragma: no cover - network interactions
            print(f"Warning: failed to log artifact '{self.artifact_alias}': {exc}")

    def _split_artifact_alias(self) -> Tuple[str, str]:
        if not self.artifact_alias:
            return ("", "latest")
        if ":" in self.artifact_alias:
            name, alias = self.artifact_alias.split(":", 1)
            alias = alias or "latest"
        else:
            name, alias = self.artifact_alias, "latest"
        return name, alias

    def _cast_row_values(self, row: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(row)
        seed_value = normalized.get("seed")
        if seed_value is not None:
            normalized["seed"] = str(seed_value)
        return normalized

def _config_to_wandb_dict(config: "PretrainConfig") -> Dict[str, Any]:
    """Convert the training config into plain Python types for W&B logging."""
    raw_config = config.model_dump()

    with suppress(Exception):
        converted = OmegaConf.to_container(OmegaConf.create(raw_config), resolve=True)
        if isinstance(converted, dict):
            return converted

    return raw_config

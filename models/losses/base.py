"""Base loss-head interface and token losses for EqR."""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class BaseLossHead(nn.Module):
    """Interface implemented by EqR training and inference loss heads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the wrapper."""
        super().__init__(*args, **kwargs)
        self.model: nn.Module

    def initial_carry(self, *args: Any, **kwargs: Any) -> Any:
        """Create an initial model carry."""
        raise NotImplementedError()

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs: Any,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Run a model step and return carry, loss, metrics, optional outputs, and finish flag."""
        raise NotImplementedError()


def _stablemax_transform(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """Apply the stablemax positive transform."""
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return log probabilities under stablemax."""
    transformed = _stablemax_transform(x)
    return torch.log(transformed / torch.sum(transformed, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute token-wise stablemax cross entropy."""
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs,
        index=transformed_labels.to(torch.long).unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute token-wise softmax cross entropy with an optional valid mask."""
    if valid_mask is not None:
        labels = torch.where(
            valid_mask,
            labels,
            torch.tensor(ignore_index, device=labels.device, dtype=labels.dtype),
        )

    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)

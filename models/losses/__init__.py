"""Public loss-head exports for the minimal EqR release."""

IGNORE_LABEL_ID = -100

from .base import BaseLossHead
from .loss_heads import ACTLossHead, InferenceLossHead

__all__ = [
    "IGNORE_LABEL_ID",
    "BaseLossHead",
    "ACTLossHead",
    "InferenceLossHead",
]

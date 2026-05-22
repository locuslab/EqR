"""Minimal initialization helpers for EqR model weights and latent states."""

import math

import torch


def trunc_normal_init_(
    tensor: torch.Tensor,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    """Fill ``tensor`` with JAX-style compensated truncated-normal samples."""
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()

        sqrt2 = math.sqrt(2.0)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2.0

        c = (2.0 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower ** 2)
        pdf_l = c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(1.0 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

        tensor.uniform_(a, b)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clip_(lower * comp_std, upper * comp_std)
        return tensor


__all__ = ["trunc_normal_init_"]

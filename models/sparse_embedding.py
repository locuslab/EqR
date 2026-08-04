from typing import Iterable, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.optim.optimizer import Optimizer

from models.common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype) -> None:
        super().__init__()
        self.cast_to = cast_to
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std),
            persistent=True,
        )
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def _apply(self, fn):
        super()._apply(fn)
        self.local_weights = nn.Buffer(self.local_weights.detach().requires_grad_(True), persistent=False)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real_bs = inputs.shape[0]
        indices = inputs.to(torch.long)
        if not self.training or real_bs != self.local_weights.shape[0]:
            return self.weights[indices].to(self.cast_to)

        with torch.no_grad():
            self.local_weights.copy_(self.weights[indices])
            self.local_ids.copy_(inputs.to(torch.int32))
        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            local_weights_grad = local_ids = weights = None
            if len(group["params"]) != 3:
                raise ValueError("Sparse embedding optimizer expects weights, local_weights, local_ids.")
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
            if local_weights_grad is None:
                continue
            if local_ids is None or weights is None:
                raise ValueError("Malformed sparse embedding buffers.")
            _sparse_emb_signsgd_dist(
                local_weights_grad,
                local_ids,
                weights,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                world_size=group["world_size"],
            )
        return loss


def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    lr: float,
    weight_decay: float,
    world_size: int,
) -> None:
    n, dim = local_weights_grad.shape
    all_weights_grad = local_weights_grad
    all_ids = local_ids
    if world_size > 1:
        all_weights_grad = torch.empty((world_size * n, dim), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * n, dtype=local_ids.dtype, device=local_ids.device)
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)

    grad_ids, inv = all_ids.unique(return_inverse=True)
    grad = torch.zeros((grad_ids.shape[0], dim), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, dim), all_weights_grad)

    updated = weights[grad_ids]
    updated.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    weights[grad_ids] = updated

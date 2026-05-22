from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID
from models.losses.base import BaseLossHead, softmax_cross_entropy, stablemax_cross_entropy
from utils.training import tree_to_device


def _carry_get(carry: Any, key: str, default: Any = None) -> Any:
    if isinstance(carry, dict):
        return carry.get(key, default)
    return getattr(carry, key, default)


def _compute_z_alignment_per_sample(
    z_h: torch.Tensor,
    z_l: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    min_seq = min(z_h.shape[1], z_l.shape[1])
    z_h_f = z_h[:, :min_seq].to(torch.float32)
    z_l_f = z_l[:, :min_seq].to(torch.float32)

    cos_tok = F.cosine_similarity(z_h_f, z_l_f, dim=-1)
    cos_per_sample = cos_tok.mean(dim=-1)

    z_h_mean_d = z_h_f.mean(dim=-1, keepdim=True)
    z_l_mean_d = z_l_f.mean(dim=-1, keepdim=True)
    z_h_centered = z_h_f - z_h_mean_d
    z_l_centered = z_l_f - z_l_mean_d
    pearson_num = (z_h_centered * z_l_centered).sum(dim=-1)
    pearson_den = z_h_centered.norm(dim=-1) * z_l_centered.norm(dim=-1)
    pearson_per_sample = (pearson_num / pearson_den.clamp_min(eps)).mean(dim=-1)

    z_h_stats = (z_h_f.mean(dim=-1).mean(dim=-1), z_h_f.std(dim=-1).mean(dim=-1))
    z_l_stats = (z_l_f.mean(dim=-1).mean(dim=-1), z_l_f.std(dim=-1).mean(dim=-1))
    return cos_per_sample, pearson_per_sample, z_h_stats, z_l_stats


class ACTLossHead(BaseLossHead):

    def __init__(self, model: nn.Module, loss_type: str, **kwargs: Any) -> None:

        del kwargs
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.initial_carry(*args, **kwargs)  # type: ignore[attr-defined]

    def compute_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        loss_divisor: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:

        del temperature
        lm_loss = (
            self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=valid_mask) / loss_divisor
        ).sum()
        return {"lm_loss": lm_loss}

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs: Any,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        new_carry, outputs = self.model(**model_kwargs)
        logits = outputs.get("logits", None)
        if isinstance(logits, torch.Tensor):
            outputs["logits"] = logits.contiguous()
        labels = new_carry.current_data["labels"]
        new_carry = tree_to_device(new_carry, outputs["logits"].device)
        preds = torch.argmax(outputs["logits"], dim=-1)

        prev_halted = outputs.pop("prev_halted", None)
        prev_preds = _carry_get(new_carry, "prev_preds", None)
        if prev_preds is not None:
            mask_source = prev_halted if prev_halted is not None else new_carry.halted
            preds = torch.where(mask_source.unsqueeze(-1), prev_preds.to(preds.dtype), preds)
            new_carry.prev_preds = preds.detach()

        outputs["preds"] = preds
        metrics: Dict[str, torch.Tensor]
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)
            count = valid_metrics.sum()

            metrics = self._build_step_metrics(
                outputs=outputs,
                carry=new_carry,
                loss_divisor=loss_divisor,
                loss_counts=loss_counts,
                is_correct=is_correct,
                seq_is_correct=seq_is_correct,
                valid_metrics=valid_metrics,
                count=count,
            )

        losses = self.compute_lm_loss(outputs["logits"], labels, valid_mask=mask, loss_divisor=loss_divisor)
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics["q_halt_loss"] = q_halt_loss.detach()
        metrics.update({key: value.detach() for key, value in losses.items()})

        detached_outputs = {key: outputs[key].detach() for key in return_keys if key in outputs}
        total_loss = losses["lm_loss"] + 0.5 * q_halt_loss
        metrics["total_loss"] = total_loss.detach()
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

    def _build_step_metrics(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        carry: Any,
        loss_divisor: torch.Tensor,
        loss_counts: torch.Tensor,
        is_correct: torch.Tensor,
        seq_is_correct: torch.Tensor,
        valid_metrics: torch.Tensor,
        count: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        metrics: Dict[str, torch.Tensor] = {
            "count": count,
            "accuracy": torch.zeros_like(count, dtype=torch.float32),
            "exact_accuracy": torch.zeros_like(count),
            **self._zh_zl_metrics(carry, count),
            "halted_q_logits": torch.zeros_like(count),
            "max_q_logits": torch.zeros_like(count, dtype=torch.float32),
            "min_q_logits": torch.full_like(count, float("inf"), dtype=torch.float32),
            "q_halt_accuracy": torch.zeros_like(count),
            "q_halt_precision": torch.zeros_like(count, dtype=torch.float32),
            "q_halt_recall": torch.zeros_like(count, dtype=torch.float32),
            "steps": torch.zeros_like(count, dtype=torch.float32),
            "max_steps": torch.zeros_like(count, dtype=torch.float32),
            "min_steps": torch.full_like(count, float("inf"), dtype=torch.float32),
            "raw_steps": torch.zeros_like(carry.steps, dtype=torch.float32),
        }
        if (count > 0).any():
            metrics.update(
                {
                    "accuracy": torch.where(
                        valid_metrics,
                        (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                        0,
                    ).sum(),
                    "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                    "halted_q_logits": torch.where(
                        valid_metrics,
                        outputs["q_halt_logits"],
                        torch.zeros_like(loss_counts, dtype=torch.float32),
                    ).sum(),
                    "max_q_logits": torch.where(
                        valid_metrics,
                        outputs["q_halt_logits"].to(torch.float32),
                        torch.full_like(loss_counts, float("-inf"), dtype=torch.float32),
                    ).max(),
                    "min_q_logits": torch.where(
                        valid_metrics,
                        outputs["q_halt_logits"].to(torch.float32),
                        torch.full_like(loss_counts, float("inf"), dtype=torch.float32),
                    ).min(),
                    "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] > 0) == seq_is_correct)).sum(),
                    "q_halt_precision": torch.where(
                        valid_metrics,
                        ((outputs["q_halt_logits"] > 0) & seq_is_correct).to(torch.float32).sum()
                        / ((outputs["q_halt_logits"] >= 0).to(torch.float32).sum().clamp_min(1)),
                        0,
                    ).sum(),
                    "q_halt_recall": torch.where(
                        valid_metrics,
                        ((outputs["q_halt_logits"] > 0) & seq_is_correct).to(torch.float32).sum()
                        / (seq_is_correct.to(torch.float32).sum().clamp_min(1)),
                        0,
                    ).sum(),
                    "steps": torch.where(valid_metrics, carry.steps.to(torch.float32), 0).sum(),
                    "max_steps": torch.where(
                        valid_metrics,
                        carry.steps.to(torch.float32),
                        torch.zeros_like(carry.steps, dtype=torch.float32),
                    ).max(),
                    "min_steps": torch.where(
                        valid_metrics,
                        carry.steps.to(torch.float32),
                        torch.full_like(carry.steps, self.model.config.halt_max_steps, dtype=torch.float32),
                    ).min(),
                    "raw_steps": torch.where(
                        valid_metrics,
                        carry.steps.to(torch.float32),
                        torch.zeros_like(carry.steps, dtype=torch.float32),
                    ),
                }
            )
        return metrics

    def _zh_zl_metrics(self, carry: Any, count: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = torch.zeros_like(count, dtype=torch.float32)
        metrics = {
            "zH_zL_cos_sim": zero,
            "zH_zL_pearson": zero,
            "zH_mean": zero,
            "zH_std": zero,
            "zL_mean": zero,
            "zL_std": zero,
        }
        inner_carry = getattr(carry, "inner_carry", None)
        if inner_carry is None or not hasattr(inner_carry, "z_H") or not hasattr(inner_carry, "z_L"):
            return metrics
        z_h = getattr(inner_carry, "z_H")
        z_l = getattr(inner_carry, "z_L")
        if not isinstance(z_h, torch.Tensor) or not isinstance(z_l, torch.Tensor):
            return metrics
        if z_h.ndim != 3 or z_l.ndim != 3:
            return metrics

        cos_per_sample, pearson_per_sample, z_h_stats, z_l_stats = _compute_z_alignment_per_sample(z_h, z_l)
        metrics["zH_zL_cos_sim"] = cos_per_sample.sum()
        metrics["zH_zL_pearson"] = pearson_per_sample.sum()
        metrics["zH_mean"] = z_h_stats[0].sum()
        metrics["zH_std"] = z_h_stats[1].sum()
        metrics["zL_mean"] = z_l_stats[0].sum()
        metrics["zL_std"] = z_l_stats[1].sum()
        return metrics


class InferenceLossHead(ACTLossHead):
    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs: Any,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        logits = outputs.get("logits", None)
        if isinstance(logits, torch.Tensor):
            outputs["logits"] = logits.contiguous()
        total_loss, metrics, detached_outputs, all_finish = self.compute_metrics_from_outputs(
            outputs=outputs,
            carry=new_carry,
            return_keys=return_keys,
        )
        return new_carry, total_loss, metrics, detached_outputs, all_finish

    def compute_metrics_from_outputs(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        carry: Any,
        return_keys: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:

        labels = carry.current_data["labels"]
        device = labels.device
        preds = torch.argmax(outputs["logits"], dim=-1)

        prev_halted = outputs.pop("prev_halted", None)
        prev_preds = _carry_get(carry, "prev_preds", None)
        if prev_preds is not None:
            mask_source = prev_halted if prev_halted is not None else carry.halted
            preds = torch.where(mask_source.to(device).unsqueeze(-1), prev_preds.to(device=device, dtype=preds.dtype), preds)
            carry.prev_preds = preds.detach().to(carry.steps.device)

        outputs["preds"] = preds
        mask = (labels != IGNORE_LABEL_ID).to(device)
        with torch.no_grad():
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            steps = carry.steps.to(device)
            max_steps = int(getattr(self.model.config, "halt_max_steps", 0) or 0)
            forced_halt = (steps >= max_steps) if max_steps > 0 else carry.halted.to(device)
            valid_metrics = forced_halt & (loss_counts > 0)
            count = valid_metrics.sum()

            metrics = {
                "count": count,
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                **self._zh_zl_metrics(carry, count),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] > 0) == seq_is_correct)).sum(),
                "q_halt_precision": torch.where(
                    valid_metrics,
                    ((outputs["q_halt_logits"] > 0) & seq_is_correct).to(torch.float32).sum()
                    / ((outputs["q_halt_logits"] >= 0).to(torch.float32).sum().clamp_min(1)),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "q_halt_recall": torch.where(
                    valid_metrics,
                    ((outputs["q_halt_logits"] > 0) & seq_is_correct).to(torch.float32).sum()
                    / (seq_is_correct.to(torch.float32).sum().clamp_min(1)),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "steps": torch.where(valid_metrics, steps.to(torch.float32), torch.zeros_like(loss_counts, dtype=torch.float32)).sum(),
            }

        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor
        ).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()

        detached_outputs = {key: outputs[key].detach() for key in return_keys if key in outputs}
        total_loss = lm_loss + 0.5 * q_halt_loss
        metrics["total_loss"] = total_loss.detach()
        finished_mask = carry.halted.to(device)
        if max_steps > 0:
            finished_mask = finished_mask | (steps >= max_steps)
        return total_loss, metrics, detached_outputs, finished_mask.all()

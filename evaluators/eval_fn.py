from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import os
import time

import torch
import torch.distributed as dist
import tqdm
from omegaconf import OmegaConf

from config.schema import PretrainConfig
from dataset.puzzle_dataset import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID
from utils.metric_reduce import (
    SPECIAL_METRIC_KEYS,
    init_special_buffer,
    partition_metrics,
    reduce_special_metrics,
    update_special_buffer,
)
from utils.metrics import compute_pass_at_k
from utils.training import TrainState, tree_to_device


def _device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _hidden(carry: Any) -> Optional[torch.Tensor]:
    inner = getattr(carry, "inner_carry", None)
    if inner is None:
        return None
    if hasattr(inner, "z_H") and hasattr(inner, "z_L"):
        return torch.cat((inner.z_H, inner.z_L), dim=1)
    if hasattr(inner, "z_H"):
        return inner.z_H
    return None


def _residual_metrics(values: List[torch.Tensor], steps: int, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        f"residual_of_{i + 1}_steps": values[i].to(ref) if i < len(values) else torch.zeros_like(ref)
        for i in range(max(0, int(steps)))
    }


def _gather_object(obj: Any, world_size: int, group: Optional[dist.ProcessGroup]) -> List[Any]:
    if world_size <= 1:
        return [obj]
    out = [None for _ in range(world_size)]
    dist.all_gather_object(out, obj, group=group)
    return out


def _reduce_sum(tensor: torch.Tensor, world_size: int, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    if world_size <= 1:
        return tensor
    if group is not None:
        tensor = tensor.cpu()
        dist.reduce(tensor, dst=0, group=group)
    else:
        dist.reduce(tensor, dst=0)
    return tensor


def _metric_state(device: torch.device, sets: int) -> Dict[str, Any]:
    return {"keys": [], "values": None, "special": {}, "seen_special": set(), "device": device, "sets": sets}


def _accumulate(state: Dict[str, Any], metrics: Dict[str, torch.Tensor], set_id: int) -> None:
    metrics = dict(metrics)
    metrics.pop("raw_steps", None)
    regular, special = partition_metrics(metrics, SPECIAL_METRIC_KEYS)
    values = state["values"]
    if values is None:
        state["keys"] = regular
        values = torch.zeros((state["sets"], len(regular)), dtype=torch.float32, device=state["device"])
        state["values"] = values
    if regular != state["keys"]:
        raise RuntimeError(f"evaluation metric keys changed: {regular} != {state['keys']}")
    if regular:
        values[set_id] += torch.stack([metrics[k].detach().to(state["device"], torch.float32) for k in regular])
    for key, value in special.items():
        buf = state["special"].setdefault(key, init_special_buffer(key, state["sets"], state["device"]))
        update_special_buffer(buf, set_id, value, key)
    state["seen_special"].update(special.keys())


def _finalize_state(
    state: Dict[str, Any],
    set_ids: Dict[str, int],
    rank: int,
    world_size: int,
    group: Optional[dist.ProcessGroup],
) -> Optional[Dict[str, Dict[str, float]]]:
    keys = state["keys"]
    gathered = _gather_object(keys if keys else None, world_size, group)
    for item in gathered:
        if item:
            keys = list(item)
            break
    if state["values"] is None:
        state["values"] = torch.zeros((len(set_ids), len(keys)), dtype=torch.float32, device=state["device"])
    values = _reduce_sum(state["values"], world_size, group)

    seen: Set[str] = set(state["seen_special"])
    for item in _gather_object(sorted(seen) if seen else None, world_size, group):
        if item:
            seen.update(item)
    for key in seen:
        state["special"].setdefault(key, init_special_buffer(key, len(set_ids), state["device"]))
    special = reduce_special_metrics(state["special"], world_size, group) if state["special"] else {}

    if rank != 0:
        return None

    arr = values.detach().float().cpu()
    out: Dict[str, Dict[str, float]] = {}
    for name, set_id in set_ids.items():
        row = {metric: float(arr[set_id, i].item()) for i, metric in enumerate(keys)}
        count = row.pop("count", 0.0)
        denom = count if count > 0 else 1.0
        out[name] = {metric: value / denom for metric, value in row.items()}
        for metric, tensor in special.items():
            out[name][metric] = float(tensor.detach().cpu()[set_id].item())
    return out


def _di_state(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "samples": torch.zeros((), dtype=torch.float64, device=device),
        "any": torch.zeros((), dtype=torch.float64, device=device),
        "pass_sum": torch.zeros((), dtype=torch.float64, device=device),
        "majority_exact": torch.zeros((), dtype=torch.float64, device=device),
        "majority_token_ok": torch.zeros((), dtype=torch.float64, device=device),
        "majority_token_total": torch.zeros((), dtype=torch.float64, device=device),
        **{f"pass@{k}": torch.zeros((), dtype=torch.float64, device=device) for k in (1, 2, 4, 8, 16, 32, 64, 128)},
    }


def _update_di(state: Dict[str, torch.Tensor], preds: torch.Tensor, labels: torch.Tensor, n: int) -> None:
    bsz = labels.shape[0]
    pred = preds.view(bsz, n, -1)
    flat_labels = labels.view(bsz, -1)
    mask = flat_labels.ne(IGNORE_LABEL_ID)
    exact = (pred.eq(flat_labels[:, None]) | ~mask[:, None]).all(dim=-1)
    correct_counts = exact.sum(dim=1)
    state["samples"] += bsz
    state["any"] += correct_counts.gt(0).sum()
    state["pass_sum"] += correct_counts.to(torch.float64).sum() / float(n)
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        if k < n:
            state[f"pass@{k}"] += sum(compute_pass_at_k(n, k, int(c.item())) for c in correct_counts)

    for sample_pred, label, sample_mask in zip(pred, flat_labels, mask):
        votes: Dict[Tuple[int, ...], int] = {}
        for row in sample_pred.detach().cpu():
            key = tuple(int(x) for x in row.tolist())
            votes[key] = votes.get(key, 0) + 1
        winner = max(votes.items(), key=lambda item: item[1])[0]
        winner_t = torch.tensor(winner, dtype=label.dtype, device=label.device)
        state["majority_exact"] += float(((winner_t == label) | ~sample_mask).all().item())
        state["majority_token_ok"] += ((winner_t == label) & sample_mask).sum().to(state["majority_token_ok"])
        state["majority_token_total"] += sample_mask.sum().to(state["majority_token_total"])


def _finish_di(
    state: Optional[Dict[str, torch.Tensor]],
    n: Optional[int],
    rank: int,
    world_size: int,
    group: Optional[dist.ProcessGroup],
) -> Dict[str, float]:
    if not state or not n:
        return {}
    packed = torch.stack(list(state.values()))
    names = list(state.keys())
    packed = _reduce_sum(packed, world_size, group)
    if rank != 0:
        return {}
    vals = {key: float(packed[i].item()) for i, key in enumerate(names)}
    total = vals["samples"]
    token_total = vals["majority_token_total"]
    metrics = {
        "different_init/avg_pass_rate": vals["pass_sum"] / total if total else 0.0,
        "different_init/any_correct": vals["any"] / total if total else 0.0,
        "different_init/majority_vote_acc": vals["majority_exact"] / total if total else 0.0,
        "different_init/total_samples": total,
        "different_init/n": float(n),
        "majority_vote/exact_accuracy": vals["majority_exact"] / total if total else 0.0,
        "majority_vote/accuracy": vals["majority_token_ok"] / token_total if token_total else 0.0,
        "majority_vote/num_samples": total,
    }
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        if k < n:
            metrics[f"different_init/pass@{k}"] = vals[f"pass@{k}"] / total if total else 0.0
    return metrics


def _conv_state(device: torch.device, top_k: int) -> Dict[str, torch.Tensor]:
    out = {
        "samples": torch.zeros((), dtype=torch.float64, device=device),
        "topk_correct": torch.zeros((), dtype=torch.float64, device=device),
        "any": torch.zeros((), dtype=torch.float64, device=device),
        "score": torch.zeros((), dtype=torch.float64, device=device),
    }
    out.update({f"hist_{i}": torch.zeros((), dtype=torch.float64, device=device) for i in range(top_k + 1)})
    out.update({f"prefix_{i}": torch.zeros((), dtype=torch.float64, device=device) for i in range(top_k)})
    return out


def _update_conv(state: Dict[str, torch.Tensor], preds: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, top_k: int) -> None:
    bsz, n = labels.shape[0], scores.shape[1]
    pred = preds.view(bsz, n, -1)
    flat_labels = labels.view(bsz, -1)
    mask = flat_labels.ne(IGNORE_LABEL_ID)
    exact = (pred.eq(flat_labels[:, None]) | ~mask[:, None]).all(dim=-1)
    for flags, row_scores in zip(exact, scores):
        order = torch.argsort(row_scores)[:top_k]
        chosen = flags[order].to(torch.float64)
        correct = int(chosen.sum().item())
        state["samples"] += 1.0
        state["topk_correct"] += float(correct)
        state["any"] += float(correct > 0)
        state["score"] += row_scores[order].to(torch.float64).mean()
        state[f"hist_{correct}"] += 1.0
        for i in range(top_k):
            state[f"prefix_{i}"] += chosen[: i + 1].sum()


def _stream_core_state(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "samples": torch.zeros((), dtype=torch.float64, device=device),
        "token_ok": torch.zeros((), dtype=torch.float64, device=device),
        "token_total": torch.zeros((), dtype=torch.float64, device=device),
        "exact": torch.zeros((), dtype=torch.float64, device=device),
        "steps": torch.zeros((), dtype=torch.float64, device=device),
        "forced_max_steps": torch.zeros((), dtype=torch.float64, device=device),
        "q_halt_accuracy": torch.zeros((), dtype=torch.float64, device=device),
        "q_halt_precision_num": torch.zeros((), dtype=torch.float64, device=device),
        "q_halt_precision_den": torch.zeros((), dtype=torch.float64, device=device),
        "q_halt_recall_num": torch.zeros((), dtype=torch.float64, device=device),
        "q_halt_recall_den": torch.zeros((), dtype=torch.float64, device=device),
    }


def _update_stream_core(
    state: Dict[str, torch.Tensor],
    preds: torch.Tensor,
    labels: torch.Tensor,
    q_halt: torch.Tensor,
    steps: torch.Tensor,
    *,
    halt_threshold: float,
    halt_max_steps: int,
) -> None:
    mask = labels.ne(IGNORE_LABEL_ID)
    token_total = mask.sum()
    token_ok = (preds.eq(labels) & mask).sum()
    loss_counts = mask.sum(dim=-1)
    exact = ((preds.eq(labels) | ~mask).all(dim=-1) & loss_counts.gt(0))
    q_positive = q_halt > float(halt_threshold)

    state["samples"] += labels.shape[0]
    state["token_ok"] += token_ok.to(state["token_ok"])
    state["token_total"] += token_total.to(state["token_total"])
    state["exact"] += exact.sum().to(state["exact"])
    state["steps"] += steps.to(torch.float64).sum()
    state["forced_max_steps"] += steps.ge(int(halt_max_steps)).sum().to(state["forced_max_steps"])
    state["q_halt_accuracy"] += q_positive.eq(exact).sum().to(state["q_halt_accuracy"])
    state["q_halt_precision_num"] += (q_positive & exact).sum().to(state["q_halt_precision_num"])
    state["q_halt_precision_den"] += q_positive.sum().to(state["q_halt_precision_den"])
    state["q_halt_recall_num"] += (q_positive & exact).sum().to(state["q_halt_recall_num"])
    state["q_halt_recall_den"] += exact.sum().to(state["q_halt_recall_den"])


def _finish_stream_core(
    states: Dict[str, Dict[str, torch.Tensor]],
    rank: int,
    world_size: int,
    group: Optional[dist.ProcessGroup],
) -> Optional[Dict[str, Dict[str, float]]]:
    names = list(next(iter(states.values())).keys()) if states else []
    if not names:
        return {} if rank == 0 else None
    packed = torch.stack([torch.stack([state[name] for name in names]) for state in states.values()])
    packed = _reduce_sum(packed, world_size, group)
    if rank != 0:
        return None
    out: Dict[str, Dict[str, float]] = {}
    for row_idx, set_name in enumerate(states):
        vals = {name: float(packed[row_idx, col_idx].item()) for col_idx, name in enumerate(names)}
        samples = vals["samples"]
        token_total = vals["token_total"]
        precision_den = vals["q_halt_precision_den"]
        recall_den = vals["q_halt_recall_den"]
        out[set_name] = {
            "accuracy": vals["token_ok"] / token_total if token_total else 0.0,
            "exact_accuracy": vals["exact"] / samples if samples else 0.0,
            "steps": vals["steps"] / samples if samples else 0.0,
            "forced_max_steps": vals["forced_max_steps"] / samples if samples else 0.0,
            "q_halt_accuracy": vals["q_halt_accuracy"] / samples if samples else 0.0,
            "q_halt_precision": vals["q_halt_precision_num"] / precision_den if precision_den else 0.0,
            "q_halt_recall": vals["q_halt_recall_num"] / recall_den if recall_den else 0.0,
        }
    return out


def _finish_conv(
    state: Optional[Dict[str, torch.Tensor]],
    top_k: int,
    rank: int,
    world_size: int,
    group: Optional[dist.ProcessGroup],
) -> Dict[str, float]:
    if not state or top_k <= 0:
        return {}
    packed = _reduce_sum(torch.stack(list(state.values())), world_size, group)
    names = list(state.keys())
    if rank != 0:
        return {}
    vals = {key: float(packed[i].item()) for i, key in enumerate(names)}
    total = vals["samples"]
    if total <= 0:
        return {}
    metrics = {
        "convergence_top_k/exact_accuracy": vals["topk_correct"] / (total * top_k),
        "convergence_top_k/any_correct": vals["any"] / total,
        "convergence_top_k/avg_convergence_score": vals["score"] / total,
        "convergence_top_k/total_samples": total,
        "convergence_top_k/k": float(top_k),
    }
    for k in (1, 2, 4, 8, 16, 32, 64):
        if k < top_k:
            metrics[f"convergence_top_k/pass@{k}"] = sum(
                vals[f"hist_{c}"] * compute_pass_at_k(top_k, k, c) for c in range(top_k + 1)
            ) / total
    for i in range(top_k):
        metrics[f"convergence_top_k/cumulative_exact_acc_top{i + 1}"] = vals[f"prefix_{i}"] / (total * (i + 1))
    return metrics


def _unwrap_eval_model(model: torch.nn.Module) -> torch.nn.Module:
    model = getattr(model, "_orig_mod", model)
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError("ACT streaming eval requires a loss head with a .model attribute.")
    return getattr(inner, "_orig_mod", inner)


def _select_rows(obj: Any, idx: torch.Tensor) -> Any:
    if torch.is_tensor(obj):
        return obj.index_select(0, idx) if obj.ndim > 0 else obj
    if is_dataclass(obj):
        return type(obj)(**{field.name: _select_rows(getattr(obj, field.name), idx) for field in fields(obj)})
    if isinstance(obj, dict):
        return {key: _select_rows(value, idx) for key, value in obj.items()}
    return obj


def _evaluate_act_streaming(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    progress_bar: Optional[Any],
    max_eval_steps: Optional[int],
    different_init: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], float]:
    if different_init is None or different_init <= 1:
        raise ValueError("ACT streaming eval requires different_init > 1.")
    if config.eval_save_outputs:
        raise ValueError("ACT streaming eval does not support eval_save_outputs yet.")

    start = time.time()
    device = _device(train_state.model)
    raw_model = _unwrap_eval_model(train_state.model)
    n_init = int(different_init)
    conv_k = min(int(getattr(config, "convergence_top_k", 0) or 0), n_init)
    window = int(getattr(config, "convergence_window", 0) or 0)
    halt_max_steps = int(getattr(raw_model.config, "halt_max_steps", getattr(config.arch, "halt_max_steps", 0)) or 0)
    if halt_max_steps <= 0:
        raise ValueError("ACT streaming eval requires a positive halt_max_steps.")
    halt_threshold = float(getattr(config, "eval_act_halt_threshold", 0.0) or 0.0)
    halt_min_steps = getattr(config, "eval_act_halt_min_steps", None)
    halt_min_steps = max(1, int(halt_min_steps)) if halt_min_steps is not None else 1
    slot_override = getattr(config, "eval_act_streaming_slots", None)
    slot_override = int(slot_override) if slot_override is not None else None
    if slot_override is not None and slot_override < 1:
        raise ValueError("eval_act_streaming_slots must be positive when set.")
    score_mode = str(
        getattr(
            config,
            "eval_act_streaming_selection_score",
            getattr(config, "eval_act_streaming_score", "convergence"),
        )
        or "convergence"
    ).lower()
    if score_mode not in {"q_halt", "convergence", "steps"}:
        raise ValueError("eval_act_streaming_selection_score must be one of: q_halt, convergence, steps.")

    core = {name: _stream_core_state(device) for name in eval_metadata.sets}
    di = _di_state(device)
    conv = _conv_state(device, conv_k) if conv_k > 0 else None

    total = getattr(getattr(eval_loader, "dataset", None), "num_batches", lambda: None)()
    if max_eval_steps is not None and total is not None:
        total = min(total, max_eval_steps)
    bar = tqdm.tqdm(
        eval_loader,
        total=total,
        desc=f"ACT Stream ({world_size} ranks)",
        disable=rank != 0,
        leave=False,
        dynamic_ncols=True,
        position=1 if progress_bar is not None else 0,
    )

    with torch.inference_mode():
        for step, (set_name, batch, _) in enumerate(bar, start=1):
            if max_eval_steps is not None and step > max_eval_steps:
                break
            original_batch = tree_to_device(batch, device)
            batch_size = int(original_batch["inputs"].shape[0])
            if batch_size == 0:
                continue

            total_jobs = batch_size * n_init
            slots = min(slot_override or batch_size, total_jobs)
            sample_order = torch.arange(batch_size, device=device).repeat(n_init)
            init_order = torch.arange(n_init, device=device).repeat_interleave(batch_size)
            next_job = slots
            active_sample = sample_order[:slots].clone()
            active_init = init_order[:slots].clone()
            active_batch = {key: value.index_select(0, active_sample) for key, value in original_batch.items()}
            carry = raw_model.initial_carry(active_batch)  # type: ignore[attr-defined]

            completed_preds = torch.empty((batch_size, n_init, *original_batch["labels"].shape[1:]), dtype=torch.long, device=device)
            scores = torch.full((batch_size, n_init), float("inf"), dtype=torch.float32, device=device)
            prev_logits: Optional[torch.Tensor] = None
            has_prev = torch.zeros((slots,), dtype=torch.bool, device=device)
            score_sum = torch.zeros((slots,), dtype=torch.float32, device=device)
            score_count = torch.zeros((slots,), dtype=torch.int32, device=device)
            score_window = torch.zeros((slots, max(window, 1)), dtype=torch.float32, device=device)
            score_cursor = 0
            completed = 0

            while active_sample.numel() > 0:
                carry, outputs = raw_model(carry=carry, batch=active_batch)
                logits = outputs["logits"].detach()
                q_halt = outputs["q_halt_logits"].detach()
                steps = carry.steps.to(device=device)

                if prev_logits is not None:
                    delta = (logits - prev_logits).norm(dim=-1).mean(dim=-1).to(torch.float32)
                    delta = torch.where(has_prev, delta, torch.zeros_like(delta))
                    if window > 0:
                        old = score_window[:, score_cursor % window]
                        score_window[:, score_cursor % window] = delta
                        score_sum += delta - torch.where(score_count.ge(window), old, torch.zeros_like(old))
                        score_count = torch.where(has_prev, torch.minimum(score_count + 1, torch.full_like(score_count, window)), score_count)
                        score_cursor += 1
                    else:
                        score_sum += delta
                        score_count = torch.where(has_prev, score_count + 1, score_count)
                prev_logits = logits
                has_prev.fill_(True)

                should_halt = (q_halt > halt_threshold) & steps.ge(halt_min_steps)
                should_halt = should_halt | steps.ge(halt_max_steps)
                if not bool(should_halt.any().item()):
                    continue

                halted_idx = should_halt.nonzero(as_tuple=False).flatten()
                sample_idx = active_sample.index_select(0, halted_idx)
                init_idx = active_init.index_select(0, halted_idx)
                final_preds = logits.index_select(0, halted_idx).argmax(dim=-1)
                final_labels = original_batch["labels"].index_select(0, sample_idx)
                final_q = q_halt.index_select(0, halted_idx)
                final_steps = steps.index_select(0, halted_idx)
                denom = score_count.index_select(0, halted_idx).clamp_min(1).to(torch.float32)
                if score_mode == "q_halt":
                    final_scores = -final_q.to(torch.float32)
                elif score_mode == "steps":
                    final_scores = final_steps.to(torch.float32)
                else:
                    final_scores = score_sum.index_select(0, halted_idx) / denom

                completed_preds[sample_idx, init_idx] = final_preds
                scores[sample_idx, init_idx] = final_scores
                _update_stream_core(
                    core[set_name],
                    final_preds,
                    final_labels,
                    final_q,
                    final_steps,
                    halt_threshold=halt_threshold,
                    halt_max_steps=halt_max_steps,
                )
                completed += int(halted_idx.numel())

                refilled = torch.zeros_like(should_halt)
                for slot in halted_idx.tolist():
                    if next_job >= total_jobs:
                        continue
                    sample = sample_order[next_job]
                    init = init_order[next_job]
                    next_job += 1
                    active_sample[slot] = sample
                    active_init[slot] = init
                    for key, value in active_batch.items():
                        value[slot] = original_batch[key][sample]
                    carry.halted[slot] = True
                    carry.steps[slot] = 0
                    has_prev[slot] = False
                    score_sum[slot] = 0
                    score_count[slot] = 0
                    score_window[slot].zero_()
                    refilled[slot] = True

                if completed >= total_jobs:
                    break
                keep = (~should_halt) | refilled
                if not bool(keep.all().item()):
                    keep_idx = keep.nonzero(as_tuple=False).flatten()
                    active_sample = active_sample.index_select(0, keep_idx)
                    active_init = active_init.index_select(0, keep_idx)
                    active_batch = {key: value.index_select(0, keep_idx) for key, value in active_batch.items()}
                    carry = _select_rows(carry, keep_idx)
                    prev_logits = prev_logits.index_select(0, keep_idx) if prev_logits is not None else None
                    has_prev = has_prev.index_select(0, keep_idx)
                    score_sum = score_sum.index_select(0, keep_idx)
                    score_count = score_count.index_select(0, keep_idx)
                    score_window = score_window.index_select(0, keep_idx)

            flat_preds = completed_preds.reshape(batch_size * n_init, *completed_preds.shape[2:])
            _update_di(di, flat_preds, original_batch["labels"], n_init)
            if conv is not None:
                _update_conv(conv, flat_preds, original_batch["labels"], scores, conv_k)

    bar.close()
    reduced = _finish_stream_core(core, rank, world_size, cpu_group)
    di_metrics = _finish_di(di, n_init, rank, world_size, cpu_group)
    conv_metrics = _finish_conv(conv, conv_k, rank, world_size, cpu_group)
    if rank == 0 and reduced is not None:
        reduced.update(di_metrics)
        reduced.update(conv_metrics)
        if config.eval_dir is not None:
            os.makedirs(config.eval_dir, exist_ok=True)
            OmegaConf.save(config=OmegaConf.create(config.model_dump()), f=os.path.join(config.eval_dir, "eval_config.yaml"))
    return reduced, time.time() - start


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    progress_bar: Optional[Any] = None,
    max_eval_steps: Optional[int] = None,
    carry: Optional[Any] = None,
    inference_runner: Optional[Any] = None,
    different_init: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], float]:
    start = time.time()
    device = _device(train_state.model)
    if bool(getattr(config, "eval_act_streaming", False)):
        if inference_runner is not None:
            raise ValueError("ACT streaming eval does not support custom inference runners.")
        if carry is not None:
            raise ValueError("ACT streaming eval does not support a preloaded carry.")
        if evaluators:
            raise ValueError("ACT streaming eval does not support dataset evaluators yet.")
        return _evaluate_act_streaming(
            config=config,
            train_state=train_state,
            eval_loader=eval_loader,
            eval_metadata=eval_metadata,
            rank=rank,
            world_size=world_size,
            cpu_group=cpu_group,
            progress_bar=progress_bar,
            max_eval_steps=max_eval_steps,
            different_init=different_init,
        )

    set_ids = {name: i for i, name in enumerate(eval_metadata.sets)}
    state = _metric_state(device, len(set_ids))
    return_keys = set(config.eval_save_outputs)
    save_preds: Dict[str, List[torch.Tensor]] = {}
    n_init = int(different_init or 1)
    di = _di_state(device) if n_init > 1 else None
    conv_k = min(int(getattr(config, "convergence_top_k", 0) or 0), n_init)
    conv = _conv_state(device, conv_k) if n_init > 1 and conv_k > 0 else None

    for evaluator in evaluators:
        evaluator.begin_eval()
        return_keys.update(evaluator.required_outputs)
    if n_init > 1:
        return_keys.add("preds")
    if conv is not None:
        return_keys.add("logits")

    total = getattr(getattr(eval_loader, "dataset", None), "num_batches", lambda: None)()
    if max_eval_steps is not None and total is not None:
        total = min(total, max_eval_steps)
    bar = tqdm.tqdm(
        eval_loader,
        total=total,
        desc=f"Eval ({world_size} ranks)",
        disable=rank != 0,
        leave=False,
        dynamic_ncols=True,
        position=1 if progress_bar is not None else 0,
    )

    with torch.inference_mode():
        for step, (set_name, batch, _) in enumerate(bar, start=1):
            if max_eval_steps is not None and step > max_eval_steps:
                break
            original_batch = tree_to_device(batch, device)
            if original_batch["inputs"].shape[0] == 0:
                continue
            batch = original_batch
            if n_init > 1:
                batch = {key: value.repeat_interleave(n_init, dim=0) for key, value in batch.items()}
            if carry is None:
                carry = train_state.model.initial_carry(batch)  # type: ignore[attr-defined]
            else:
                carry = tree_to_device(carry, device)

            residuals: List[torch.Tensor] = []
            trajectory: List[torch.Tensor] = []
            prev = _hidden(carry)
            while True:
                if inference_runner is None:
                    carry, loss, metrics, preds, all_finish = train_state.model(
                        carry=carry,
                        batch=batch,
                        return_keys=return_keys,
                    )
                else:
                    carry, loss, metrics, preds, all_finish, _ = inference_runner.run(
                        train_state=train_state,
                        batch=batch,
                        return_keys=return_keys,
                        initial_carry=carry,
                    )
                if conv is not None and preds is not None and "logits" in preds:
                    trajectory.append(preds["logits"].detach())
                cur = _hidden(carry)
                if cur is not None and prev is not None:
                    residuals.append((cur - prev).flatten(start_dim=1).to(torch.float32).norm(dim=1).sum())
                prev = None if cur is None else cur.detach()
                if bool(all_finish.detach().all().item() if torch.is_tensor(all_finish) else all_finish):
                    break

            max_steps = getattr(getattr(getattr(train_state.model, "model", None), "config", None), "halt_max_steps", len(residuals))
            if "count" in metrics:
                metrics.update(_residual_metrics(residuals, int(max_steps), metrics["count"]))
            set_id = set_ids[set_name]
            _accumulate(state, metrics, set_id)

            preds = preds or {}
            if n_init > 1 and "preds" in preds:
                _update_di(di, preds["preds"], original_batch["labels"], n_init)  # type: ignore[arg-type]
                if conv is not None and len(trajectory) > 1:
                    deltas = torch.stack(trajectory)[1:] - torch.stack(trajectory)[:-1]
                    window = int(getattr(config, "convergence_window", 0) or 0)
                    if window > 0 and window < deltas.shape[0]:
                        deltas = deltas[-window:]
                    scores = deltas.norm(dim=-1).mean(dim=(0, -1)).view(original_batch["inputs"].shape[0], n_init)
                    _update_conv(conv, preds["preds"], original_batch["labels"], scores, conv_k)

            eval_batch = original_batch if n_init > 1 else batch
            eval_preds = {
                key: value.view(original_batch["inputs"].shape[0], n_init, *value.shape[1:])[:, 0]
                if n_init > 1 and value.shape[0] == original_batch["inputs"].shape[0] * n_init
                else value
                for key, value in preds.items()
            }
            for evaluator in evaluators:
                evaluator.update_batch(eval_batch, eval_preds)
            for collection in (eval_batch, eval_preds):
                for key, value in collection.items():
                    if key in config.eval_save_outputs:
                        save_preds.setdefault(key, []).append(value.detach().cpu())
            del loss

    bar.close()
    reduced = _finalize_state(state, set_ids, rank, world_size, cpu_group)

    if reduced is not None:
        for evaluator in evaluators:
            save_path = None
            if config.eval_dir is not None:
                save_path = os.path.join(config.eval_dir, f"evaluator_{evaluator.__class__.__name__}")
                os.makedirs(save_path, exist_ok=True)
            metrics = evaluator.result(save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics:
                reduced.update(metrics)
    di_metrics = _finish_di(di, n_init if n_init > 1 else None, rank, world_size, cpu_group)
    conv_metrics = _finish_conv(conv, conv_k, rank, world_size, cpu_group)
    if rank == 0 and reduced is not None:
        reduced.update(di_metrics)
        reduced.update(conv_metrics)

    if config.eval_dir is not None:
        os.makedirs(config.eval_dir, exist_ok=True)
        if save_preds:
            torch.save({k: torch.cat(v, dim=0) for k, v in save_preds.items()}, os.path.join(config.eval_dir, f"all_preds.{rank}"))
        if config.eval_save_carry:
            torch.save(carry, os.path.join(config.eval_dir, f"final_carry.{rank}"))
        if rank == 0:
            OmegaConf.save(config=OmegaConf.create(config.model_dump()), f=os.path.join(config.eval_dir, "eval_config.yaml"))

    return reduced, time.time() - start

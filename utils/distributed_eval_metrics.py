"""Distributed majority-vote and convergence-selection eval metrics."""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
import numpy as np


__all__ = [
    "compute_majority_vote_distributed",
    "compute_convergence_topk_distributed",
]


def _gather_tensor_across_ranks(
    tensor: torch.Tensor,
    world_size: int,
    *,
    cpu_group: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """All-gather a tensor across ranks and return CPU tensors."""
    if world_size <= 1:
        return [tensor.detach().cpu()]

    if cpu_group is not None and dist.is_initialized():
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=cpu_group)
        return [t.detach().cpu() for t in gathered]

    if not dist.is_initialized():
        raise RuntimeError("Distributed backend is not initialized")

    # Default group path: if NCCL, tensors must be on CUDA.
    backend = None
    try:
        backend = dist.get_backend()
    except Exception:
        backend = None

    gather_tensor = tensor
    if backend == "nccl" and gather_tensor.device.type != "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but CUDA is not available")
        gather_tensor = gather_tensor.cuda()

    gathered = [torch.zeros_like(gather_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, gather_tensor)
    return [t.detach().cpu() for t in gathered]


def _gather_object_across_ranks(
    obj: object,
    world_size: int,
    *,
    cpu_group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> List[object]:
    """All-gather a Python object across ranks."""
    del device
    if world_size <= 1:
        return [obj]

    if cpu_group is not None and dist.is_initialized():
        gathered: List[object] = [None] * world_size
        dist.all_gather_object(gathered, obj, group=cpu_group)
        return gathered

    if not dist.is_initialized():
        raise RuntimeError("Distributed backend is not initialized")

    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def compute_majority_vote_distributed(
    different_init_results: List[Dict],
    different_init: int,
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[Dict[str, float], Dict[int, int], List[Tuple[int, float, int]]]:
    """Aggregate per-sample majority-vote metrics across ranks."""
    from utils.metrics import compute_pass_at_k

    local_total_samples = len(different_init_results)
    local_correct_any = sum(1 for r in different_init_results if r['n_correct'] > 0)
    local_pass_rate_sum = sum(r['n_correct'] / r['n_samples'] for r in different_init_results)

    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    k_values = [k for k in k_values if k < different_init]

    local_pass_at_k_sums = {k: 0.0 for k in k_values}
    for r in different_init_results:
        n = r['n_samples']
        c = r['n_correct']
        for k in k_values:
            pass_k = compute_pass_at_k(n, k, c)
            local_pass_at_k_sums[k] += pass_k

    local_majority_vote_correct = 0
    local_majority_vote_token_correct = 0
    local_majority_vote_token_total = 0
    local_majority_vote_distribution = {i: 0 for i in range(1, different_init + 1)}
    local_vote_convergence_pairs = []

    for r in different_init_results:
        if 'predictions' not in r or 'label' not in r:
            continue

        predictions = r['predictions']
        label = r['label']

        seq_dict = {}
        for i in range(predictions.shape[0]):
            seq_tuple = tuple(predictions[i].tolist())
            seq_dict[seq_tuple] = seq_dict.get(seq_tuple, 0) + 1

        max_votes = max(seq_dict.values())
        majority_seqs = [seq for seq, count in seq_dict.items() if count == max_votes]
        majority_seq = torch.tensor(majority_seqs[0], dtype=torch.long)

        # Token-level accuracy (ignore PAD=0 by convention across puzzle datasets).
        # This mirrors the common "accuracy" metric (avg token accuracy) while the
        # majority-vote correctness below is an "exact match" metric.
        valid_mask = label.ne(0)
        local_majority_vote_token_total += int(valid_mask.sum().item())
        local_majority_vote_token_correct += int(((majority_seq == label) & valid_mask).sum().item())

        is_correct = (majority_seq == label).all().item()
        if is_correct:
            local_majority_vote_correct += 1

        local_majority_vote_distribution[max_votes] += 1

        if 'convergence_score' in r and r['convergence_score'] > 0:
            local_vote_convergence_pairs.append((
                max_votes,
                float(r['convergence_score']),
                int(is_correct),
            ))

    if world_size > 1:
        stats_list = [
            local_total_samples,
            local_correct_any,
            local_pass_rate_sum,
            local_majority_vote_correct,
            local_majority_vote_token_correct,
            local_majority_vote_token_total,
        ] + [local_pass_at_k_sums[k] for k in k_values] + \
            [local_majority_vote_distribution[i] for i in range(1, different_init + 1)]

        device = torch.device("cuda") if torch.cuda.is_available() and dist.is_initialized() and dist.get_backend() == "nccl" else torch.device("cpu")
        stats_tensor = torch.tensor(stats_list, dtype=torch.float32, device=device)
        
        gathered_stats = _gather_tensor_across_ranks(stats_tensor, world_size, cpu_group=cpu_group)

        all_vote_convergence_pairs = _gather_object_across_ranks(
            local_vote_convergence_pairs,
            world_size,
            cpu_group=cpu_group,
            device=None,
        )
    else:
        gathered_stats = [torch.tensor([
            local_total_samples,
            local_correct_any,
            local_pass_rate_sum,
            local_majority_vote_correct,
            local_majority_vote_token_correct,
            local_majority_vote_token_total,
        ] + [local_pass_at_k_sums[k] for k in k_values] + \
            [local_majority_vote_distribution[i] for i in range(1, different_init + 1)],
        dtype=torch.float64, device=torch.device("cpu"))]
        all_vote_convergence_pairs = [local_vote_convergence_pairs]

    if rank == 0:
        total_samples = int(sum(s[0].item() for s in gathered_stats))
        total_correct_any = int(sum(s[1].item() for s in gathered_stats))
        total_pass_rate_sum = sum(s[2].item() for s in gathered_stats)
        total_majority_vote_correct = int(sum(s[3].item() for s in gathered_stats))
        total_majority_vote_token_correct = int(sum(s[4].item() for s in gathered_stats))
        total_majority_vote_token_total = int(sum(s[5].item() for s in gathered_stats))
        
        pass_at_k_results = {}
        for idx, k in enumerate(k_values):
            total_pass_k_sum = sum(s[6 + idx].item() for s in gathered_stats)
            pass_at_k_results[f"pass@{k}"] = total_pass_k_sum / total_samples if total_samples > 0 else 0.0
        
        majority_vote_distribution = {}
        for i in range(1, different_init + 1):
            idx = 6 + len(k_values) + (i - 1)
            majority_vote_distribution[i] = int(sum(s[idx].item() for s in gathered_stats))
        
        avg_pass_rate = total_pass_rate_sum / total_samples if total_samples > 0 else 0.0
        majority_vote_accuracy = total_majority_vote_correct / total_samples if total_samples > 0 else 0.0
        majority_vote_token_accuracy = (
            total_majority_vote_token_correct / total_majority_vote_token_total
            if total_majority_vote_token_total > 0
            else 0.0
        )
        
        vote_convergence_pairs = []
        for pairs in all_vote_convergence_pairs:
            if pairs:
                vote_convergence_pairs.extend(pairs)
        
        metrics = {
            "different_init/avg_pass_rate": avg_pass_rate,
            "different_init/any_correct": total_correct_any / total_samples if total_samples > 0 else 0.0,
            "different_init/majority_vote_acc": majority_vote_accuracy,
            "different_init/total_samples": total_samples,
            "different_init/n": different_init,

            "majority_vote/exact_accuracy": majority_vote_accuracy,
            "majority_vote/accuracy": majority_vote_token_accuracy,
            "majority_vote/num_samples": total_samples,
        }
        metrics.update({f"different_init/{k}": v for k, v in pass_at_k_results.items()})
        
        return metrics, majority_vote_distribution, vote_convergence_pairs
    else:
        return {}, {}, []


def compute_convergence_topk_distributed(
    convergence_data: List[Dict],
    different_init: int,
    top_k: int,
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup] = None,
    gather_scatter: bool = False,
) -> Tuple[Dict[str, float], List[float], List[Tuple[int, float, int]]]:
    """Aggregate top-k most-converged selection metrics across ranks."""
    from utils.metrics import compute_pass_at_k

    local_total_samples = 0
    local_topk_correct_sum = 0.0
    local_any_correct_count = 0.0
    local_topk_convergence_sum = 0.0
    local_n_correct_hist = np.zeros(top_k + 1, dtype=np.float64)
    local_prefix_correct = np.zeros(top_k, dtype=np.float64)
    local_vote_convergence_pairs = []

    for sample_data in convergence_data:
        convergence_scores = np.asarray(sample_data['convergence_scores']).flatten()
        correct_flags = np.asarray(sample_data['correct_flags']).flatten()
        
        if convergence_scores.shape[0] != different_init or correct_flags.shape[0] != different_init:
            continue

        sorted_indices = np.argsort(convergence_scores)
        sorted_correct = correct_flags[sorted_indices]

        topk_indices = sorted_indices[:top_k]
        topk_correct = int(correct_flags[topk_indices].sum())
        local_total_samples += 1
        local_topk_correct_sum += float(topk_correct)
        local_any_correct_count += 1.0 if topk_correct > 0 else 0.0
        local_n_correct_hist[topk_correct] += 1.0
        
        topk_conv_score = float(convergence_scores[topk_indices].mean())
        local_topk_convergence_sum += topk_conv_score

        for i in range(top_k):
            local_prefix_correct[i] += float(sorted_correct[: i + 1].sum())

        if 'vote_count' in sample_data:
            is_top1_correct = int(sorted_correct[0] == 1)
            local_vote_convergence_pairs.append((
                sample_data['vote_count'],
                float(topk_conv_score),
                is_top1_correct,
            ))

    k_values = [1, 2, 4, 8, 16, 32, 64]
    k_values = [k for k in k_values if k < top_k]

    stats_list = [
        float(local_total_samples),
        float(local_topk_correct_sum),
        float(local_any_correct_count),
        float(local_topk_convergence_sum),
    ]
    stats_list.extend(local_n_correct_hist.tolist())
    stats_list.extend(local_prefix_correct.tolist())

    if torch.cuda.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        stats_device = torch.device("cuda")
    else:
        stats_device = torch.device("cpu")
    stats_tensor = torch.tensor(stats_list, dtype=torch.float64, device=stats_device)

    if world_size > 1:
        gathered_stats = _gather_tensor_across_ranks(stats_tensor, world_size, cpu_group=cpu_group)
    else:
        gathered_stats = [stats_tensor.detach().cpu()]

    all_vote_convergence_pairs: List[object] = []
    if gather_scatter:
        if world_size > 1:
            all_vote_convergence_pairs = _gather_object_across_ranks(
                local_vote_convergence_pairs,
                world_size,
                cpu_group=cpu_group,
            )
        else:
            all_vote_convergence_pairs = [local_vote_convergence_pairs]

    if rank == 0:
        total_samples = int(sum(s[0].item() for s in gathered_stats))
        if total_samples == 0:
            return {}, [], []

        total_topk_correct_sum = sum(s[1].item() for s in gathered_stats)
        total_any_correct = sum(s[2].item() for s in gathered_stats)
        total_topk_convergence_sum = sum(s[3].item() for s in gathered_stats)

        base = 4
        total_hist = np.zeros(top_k + 1, dtype=np.float64)
        for i in range(top_k + 1):
            total_hist[i] = sum(s[base + i].item() for s in gathered_stats)

        prefix_base = base + (top_k + 1)
        total_prefix_correct = np.zeros(top_k, dtype=np.float64)
        for i in range(top_k):
            total_prefix_correct[i] = sum(s[prefix_base + i].item() for s in gathered_stats)

        topk_exact_accuracy = float(total_topk_correct_sum / (total_samples * top_k))
        topk_any_correct = float(total_any_correct / total_samples)
        avg_topk_convergence = float(total_topk_convergence_sum / total_samples)

        topk_pass_at_k_results = {}
        for k in k_values:
            pass_k_sum = 0.0
            for n_correct in range(top_k + 1):
                cnt = total_hist[n_correct]
                if cnt <= 0:
                    continue
                pass_k_sum += cnt * float(compute_pass_at_k(top_k, k, int(n_correct)))
            topk_pass_at_k_results[f"pass@{k}"] = float(pass_k_sum / total_samples)

        cumulative_exact_accs = []
        for k_curr in range(1, top_k + 1):
            exact_correct_count = total_prefix_correct[k_curr - 1]
            cumulative_exact_acc = exact_correct_count / (total_samples * k_curr)
            cumulative_exact_accs.append(float(cumulative_exact_acc))

        vote_convergence_pairs: List[Tuple[int, float, int]] = []
        for pairs in all_vote_convergence_pairs:
            if pairs:
                vote_convergence_pairs.extend(pairs)
        
        metrics = {
            "convergence_top_k/exact_accuracy": topk_exact_accuracy,
            "convergence_top_k/any_correct": topk_any_correct,
            "convergence_top_k/avg_convergence_score": avg_topk_convergence,
            "convergence_top_k/total_samples": total_samples,
            "convergence_top_k/k": top_k,
        }
        metrics.update({f"convergence_top_k/{k}": v for k, v in topk_pass_at_k_results.items()})
        
        for k_curr, cum_acc in enumerate(cumulative_exact_accs, start=1):
            metrics[f"convergence_top_k/cumulative_exact_acc_top{k_curr}"] = cum_acc
        
        return metrics, cumulative_exact_accs, vote_convergence_pairs
    else:
        return {}, [], []

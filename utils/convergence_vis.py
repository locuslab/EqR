"""
Convergence visualization utilities for analyzing model prediction stability.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import torch
import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    "calculate_convergence",
    "calculate_convergence_statistics",
    "calculate_pca_convergence",
    "compute_sample_confidence",
    "plot_rank_convergence_correct_confidence",
    "plot_pca_convergence_comparison",
]


def calculate_convergence(logits_traces: List[torch.Tensor], only_last_k: Optional[int] = 3) -> torch.Tensor:
    """Return per-sample mean logit-delta norm; lower means more converged."""
    ref_device = logits_traces[0].device
    logits_traces = [t.float().to(ref_device) for t in logits_traces]

    stacked_logits = torch.stack(logits_traces, dim=0)
    deltas = stacked_logits[1:, :, :, :] - stacked_logits[:-1, :, :, :]
    if only_last_k is not None and only_last_k < deltas.shape[0]:
        deltas = deltas[-only_last_k:, :, :, :]
    delta_norms = deltas.norm(dim=-1)
    convergence = delta_norms.mean(dim=(0, 2))
    return convergence


def calculate_convergence_statistics(
    logits_traces: List[torch.Tensor], 
    only_last_k: Optional[int] = 3
) -> Dict[str, torch.Tensor]:
    """
    Calculate comprehensive convergence statistics for each sample.
    
    This function computes multiple metrics to capture different aspects of convergence:
    
    Basic Distributional Statistics:
    - Mean: average convergence (lower = better)
    - Std: standard deviation across positions and time steps
    - Max: worst-case convergence (captures the most unconverged channel)
    - P95: 95th percentile (robust upper bound, filters extreme outliers)
    - MAD: Median Absolute Deviation (robust measure of spread, less sensitive to outliers)
    - Skewness: distribution asymmetry (high positive skew = few large outliers)
    
    Trajectory-Based Metrics (inspired by fixed-point iteration theory):
    - Final residual: d_K = ||z_K - z_{K-1}|| (how close to fixed point at last step)
    - Integrated residual: sum of γ^k * d_k (weighted sum favoring early convergence)
    - Convergence rate: ρ_k = d_k / d_{k-1} (geometric convergence indicator)
    - Rate stability: std of convergence rates (stable convergence vs oscillating)
    
    Args:
        logits_traces: list of length T, each element is (N, L, V)
        only_last_k: if not None, only use the last K steps for convergence calculation
        
    Returns:
        dict with keys: 'mean', 'std', 'max', 'p95', 'mad', 'skewness',
        'final_residual', 'integrated_residual', 'avg_conv_rate', 'rate_stability',
        each shape (N,)
    """
    ref_device = logits_traces[0].device
    logits_traces = [t.float().to(ref_device) for t in logits_traces]

    stacked_logits = torch.stack(logits_traces, dim=0)
    deltas = stacked_logits[1:, :, :, :] - stacked_logits[:-1, :, :, :]
    if only_last_k is not None and only_last_k < deltas.shape[0]:
        deltas = deltas[-only_last_k:, :, :, :]
    delta_norms = deltas.norm(dim=-1)

    N = delta_norms.shape[1]
    T_steps = delta_norms.shape[0]

    stats = {}

    for sample_idx in range(N):
        sample_norms = delta_norms[:, sample_idx, :].flatten()

        mean_val = sample_norms.mean()
        std_val = sample_norms.std()
        max_val = sample_norms.max()
        p95_val = torch.quantile(sample_norms, 0.95)

        median_val = torch.median(sample_norms)
        mad_val = torch.median(torch.abs(sample_norms - median_val))

        if std_val > 1e-8:
            centered = sample_norms - mean_val
            skew_val = (centered ** 3).mean() / (std_val ** 3)
        else:
            skew_val = torch.tensor(0.0, device=ref_device)

        residuals_per_step = delta_norms[:, sample_idx, :].mean(dim=1)

        final_residual = residuals_per_step[-1]
        gamma = 0.95
        weights = torch.tensor([gamma ** k for k in range(T_steps)], device=ref_device)
        integrated_residual = (weights * residuals_per_step).sum()

        if T_steps > 1:
            conv_rates = residuals_per_step[1:] / (residuals_per_step[:-1] + 1e-9)
            avg_conv_rate = conv_rates.mean()
            rate_stability = conv_rates.std()
        else:
            avg_conv_rate = torch.tensor(1.0, device=ref_device)
            rate_stability = torch.tensor(0.0, device=ref_device)

        if sample_idx == 0:
            for key in ['mean', 'std', 'max', 'p95', 'mad', 'skewness',
                       'final_residual', 'integrated_residual', 'avg_conv_rate', 'rate_stability']:
                stats[key] = []
        
        stats['mean'].append(mean_val)
        stats['std'].append(std_val)
        stats['max'].append(max_val)
        stats['p95'].append(p95_val)
        stats['mad'].append(mad_val)
        stats['skewness'].append(skew_val)
        stats['final_residual'].append(final_residual)
        stats['integrated_residual'].append(integrated_residual)
        stats['avg_conv_rate'].append(avg_conv_rate)
        stats['rate_stability'].append(rate_stability)

    for key in stats:
        stats[key] = torch.stack(stats[key])

    return stats


def calculate_pca_convergence(
    logits_traces: List[torch.Tensor],
    top_k: int = 5,
    bottom_k: int = 5,
    only_last_k: Optional[int] = 3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Calculate convergence in PCA space to see if principal components or tail components
    have better discriminative power.
    
    Args:
        logits_traces: list of length T, each element is (N, L, V)
        top_k: number of top principal components to analyze
        bottom_k: number of bottom principal components to analyze
        only_last_k: if not None, only use the last K steps for convergence calculation
        
    Returns:
        top_k_convergence: (N,) mean convergence of top k principal components
        bottom_k_convergence: (N,) mean convergence of bottom k principal components
        pca_info: dict with PCA analysis info (explained_variance_ratio, etc.)
    """
    ref_device = logits_traces[0].device
    logits_traces = [t.float().to(ref_device) for t in logits_traces]

    stacked_logits = torch.stack(logits_traces, dim=0)
    T, N, L, vocab_size = stacked_logits.shape

    logits_flat = stacked_logits.reshape(-1, vocab_size)

    mean = logits_flat.mean(dim=0, keepdim=True)
    centered = logits_flat - mean

    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    variance = S ** 2 / (logits_flat.shape[0] - 1)
    explained_variance_ratio = variance / variance.sum()

    pca_components = Vt.T

    logits_reshaped = stacked_logits.reshape(T, N*L, vocab_size)
    centered_logits = logits_reshaped - mean.reshape(1, 1, vocab_size)
    logits_pca = centered_logits @ pca_components

    logits_pca = logits_pca.reshape(T, N, L, vocab_size)

    logits_pca_top = logits_pca[:, :, :, :top_k]
    deltas_top = logits_pca_top[1:] - logits_pca_top[:-1]

    if only_last_k is not None and only_last_k < deltas_top.shape[0]:
        deltas_top = deltas_top[-only_last_k:]

    delta_norms_top = deltas_top.abs()
    top_k_convergence = delta_norms_top.mean(dim=[0, 2, 3])

    logits_pca_bottom = logits_pca[:, :, :, -bottom_k:]
    deltas_bottom = logits_pca_bottom[1:] - logits_pca_bottom[:-1]

    if only_last_k is not None and only_last_k < deltas_bottom.shape[0]:
        deltas_bottom = deltas_bottom[-only_last_k:]

    delta_norms_bottom = deltas_bottom.abs()
    bottom_k_convergence = delta_norms_bottom.mean(dim=[0, 2, 3])

    pca_info = {
        'explained_variance_ratio': explained_variance_ratio.cpu(),
        'cumsum_variance': explained_variance_ratio.cumsum(dim=0).cpu(),
        'top_k_variance': explained_variance_ratio[:top_k].sum().item(),
        'bottom_k_variance': explained_variance_ratio[-bottom_k:].sum().item(),
        'n_components': pca_components.shape[1],
    }

    return top_k_convergence, bottom_k_convergence, pca_info


def compute_sample_confidence(
    logits_traces: List[torch.Tensor],
    preds: torch.Tensor,
    reduce_over_positions: str = "mean",
) -> torch.Tensor:
    """
    Compute per-sample confidence using the final-step logits.

    Args:
        logits_traces: list of length T with tensors of shape (N, L, V)
        preds: (N, L) tensor of predicted token indices (any device)
        reduce_over_positions: "mean" or "min"

    Returns:
        confidences: (N,) tensor with one confidence per sample (same device as logits)
    """
    last_logits = logits_traces[-1]
    device = last_logits.device

    preds = preds.to(device)

    probs = torch.softmax(last_logits, dim=-1)
    pred_probs = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)

    if reduce_over_positions == "mean":
        confidences = pred_probs.mean(dim=1)
    elif reduce_over_positions == "min":
        confidences = pred_probs.min(dim=1).values
    else:
        raise ValueError(f"Unknown reduce_over_positions: {reduce_over_positions}")

    return confidences


def plot_rank_convergence_correct_confidence(
    logits_traces: List[torch.Tensor],
    preds: torch.Tensor,
    labels: torch.Tensor,
    reduce_over_positions: str = "mean",
    title: str = "Comprehensive Convergence Analysis",
    save_path: Optional[str] = None,
    different_init: Optional[int] = None,
):
    """
    Comprehensive visualization with multiple subplots showing both distributional
    and trajectory-based convergence metrics.
    
    This visualization is prediction-level: with different_init > 1, each sample
    contributes one prediction per initialization.
    
    Layout (4x2 grid):
    1. Top Row (span both cols): Mean convergence, confidence, and correctness
    2. 2nd Row Left: Std and Max (worst-case distributional indicators)
    3. 2nd Row Right: P95 and MAD (robust spread measures)
    4. 3rd Row Left: Skewness (outlier detection)
    5. 3rd Row Right: Final vs Integrated Residual (trajectory quality)
    6. 4th Row (span both cols): Convergence Rate Analysis
    
    These metrics are inspired by fixed-point iteration theory where convergence
    quality can be assessed both from distributional properties and trajectory behavior.

    Args:
        logits_traces: list of length T, each (N, L, V) where N = n_samples * different_init
        preds: (N, L) predictions
        labels: (N, L) ground truth labels
        reduce_over_positions: passed to compute_sample_confidence
        title: plot title
        save_path: if not None, save the figure to this path
        different_init: number of random initializations per sample (for annotation)
    """
    # Use logits device as reference
    ref_device = logits_traces[-1].device
    preds_dev = preds.to(ref_device)
    labels_dev = labels.to(ref_device)

    stats = calculate_convergence_statistics(logits_traces)
    confidence = compute_sample_confidence(logits_traces, preds_dev, reduce_over_positions)
    correctness = (preds_dev == labels_dev).all(dim=1)

    sorted_idx = torch.argsort(stats['mean'])

    mean_sorted = stats['mean'][sorted_idx]
    std_sorted = stats['std'][sorted_idx]
    max_sorted = stats['max'][sorted_idx]
    p95_sorted = stats['p95'][sorted_idx]
    mad_sorted = stats['mad'][sorted_idx]
    skew_sorted = stats['skewness'][sorted_idx]
    final_res_sorted = stats['final_residual'][sorted_idx]
    integrated_res_sorted = stats['integrated_residual'][sorted_idx]
    avg_rate_sorted = stats['avg_conv_rate'][sorted_idx]
    rate_stab_sorted = stats['rate_stability'][sorted_idx]
    conf_sorted = confidence[sorted_idx]
    corr_sorted = correctness[sorted_idx].to(torch.float32)

    def normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    mean_norm = normalize(mean_sorted)
    std_norm = normalize(std_sorted)
    max_norm = normalize(max_sorted)
    p95_norm = normalize(p95_sorted)
    mad_norm = normalize(mad_sorted)
    skew_min = skew_sorted.min()
    skew_max = skew_sorted.max()
    skew_norm = (skew_sorted - skew_min) / (skew_max - skew_min + 1e-8)
    final_res_norm = normalize(final_res_sorted)
    integrated_res_norm = normalize(integrated_res_sorted)
    avg_rate_norm = normalize(avg_rate_sorted)
    rate_stab_norm = normalize(rate_stab_sorted)

    x = torch.arange(mean_sorted.shape[0])
    x_np = x.float().cpu().numpy()
    mean_np = mean_norm.detach().float().cpu().numpy()
    std_np = std_norm.detach().float().cpu().numpy()
    max_np = max_norm.detach().float().cpu().numpy()
    p95_np = p95_norm.detach().float().cpu().numpy()
    mad_np = mad_norm.detach().float().cpu().numpy()
    skew_np = skew_norm.detach().float().cpu().numpy()
    final_res_np = final_res_norm.detach().float().cpu().numpy()
    integrated_res_np = integrated_res_norm.detach().float().cpu().numpy()
    avg_rate_np = avg_rate_norm.detach().float().cpu().numpy()
    rate_stab_np = rate_stab_norm.detach().float().cpu().numpy()
    conf_np = conf_sorted.detach().float().cpu().numpy()
    corr_np = corr_sorted.detach().float().cpu().numpy()

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

    n_predictions = len(x_np)
    if different_init is not None and different_init > 1:
        n_samples = n_predictions // different_init
        data_info = f"(First Batch: {n_predictions} predictions = {n_samples} samples × {different_init} inits)"
    else:
        data_info = f"({n_predictions} predictions from first batch)"

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x_np, mean_np, label="mean convergence", linewidth=2.5, color='#2E86AB', alpha=0.9)
    ax1.plot(x_np, conf_np, label="confidence", linewidth=2, color='#A23B72', alpha=0.8)
    ax1.scatter(x_np, corr_np, label="correctness", marker='x', alpha=0.7, s=40, color='#F18F01')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("Prediction Index (sorted by mean convergence, left = more converged)", fontsize=11)
    ax1.set_ylabel("Normalized Value", fontsize=11)
    ax1.set_title(f"Mean Convergence, Confidence & Correctness\n{data_info}", fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_np, std_np, label="std (variability)", linewidth=2, color='#C73E1D', alpha=0.8)
    ax2.plot(x_np, max_np, label="max (worst channel)", linewidth=2, color='#6A0572', alpha=0.8, linestyle='--')
    ax2.scatter(x_np, corr_np, label="correctness", marker='x', alpha=0.5, s=20, color='#F18F01')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Prediction Index (sorted by mean convergence)", fontsize=10)
    ax2.set_ylabel("Normalized Value", fontsize=10)
    ax2.set_title("Std & Max: Detecting Unconverged Channels", fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax2.text(0.02, 0.98, "High std/max → some channels\nare not converging well", 
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x_np, p95_np, label="P95 (95th percentile)", linewidth=2, color='#118AB2', alpha=0.8)
    ax3.plot(x_np, mad_np, label="MAD (robust spread)", linewidth=2, color='#06D6A0', alpha=0.8, linestyle='--')
    ax3.scatter(x_np, corr_np, label="correctness", marker='x', alpha=0.5, s=20, color='#F18F01')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_xlabel("Prediction Index (sorted by mean convergence)", fontsize=10)
    ax3.set_ylabel("Normalized Value", fontsize=10)
    ax3.set_title("P95 & MAD: Robust Upper Bounds", fontsize=11, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax3.text(0.02, 0.98, "MAD is robust to extreme\noutliers vs std", 
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(x_np, skew_np, label="skewness (asymmetry)", linewidth=2.5, color='#EF476F', alpha=0.8)
    ax4.scatter(x_np, corr_np * 0.5, label="correctness (×0.5)", marker='x', alpha=0.6, s=40, color='#F18F01')
    ax4.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='normalized 0')
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_xlabel("Prediction Index (sorted by mean convergence)", fontsize=10)
    ax4.set_ylabel("Normalized Skewness", fontsize=10)
    ax4.set_title("Skewness: Distribution Asymmetry", fontsize=11, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    ax4.text(0.02, 0.98, "High skewness → outlier\nchannels with large deltas", 
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(x_np, final_res_np, label="final residual (d_K)", linewidth=2, color='#9B59B6', alpha=0.8)
    ax5.plot(x_np, integrated_res_np, label="integrated residual (γ=0.95)", linewidth=2, color='#E74C3C', alpha=0.8, linestyle='--')
    ax5.scatter(x_np, corr_np * 0.5, label="correctness (×0.5)", marker='x', alpha=0.5, s=20, color='#F18F01')
    ax5.set_ylim(-0.1, 1.1)
    ax5.set_xlabel("Prediction Index (sorted by mean convergence)", fontsize=10)
    ax5.set_ylabel("Normalized Residual", fontsize=10)
    ax5.set_title("Trajectory Residuals", fontsize=11, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)

    ax5.text(0.02, 0.98, "Integrated residual considers\nfull trajectory history", 
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

    ax6 = fig.add_subplot(gs[3, :])
    ax6.plot(x_np, avg_rate_np, label="avg conv rate (ρ̄)", linewidth=2.5, color='#3498DB', alpha=0.8)
    ax6.plot(x_np, rate_stab_np, label="rate stability (σ_ρ)", linewidth=2, color='#F39C12', alpha=0.7, linestyle='-.')
    ax6.scatter(x_np, corr_np * 0.5, label="correctness (×0.5)", marker='x', alpha=0.6, s=40, color='#F18F01')
    ax6.set_ylim(-0.1, 1.1)
    ax6.set_xlabel("Prediction Index (sorted by mean convergence)", fontsize=11)
    ax6.set_ylabel("Normalized Value", fontsize=11)
    ax6.set_title("Convergence Rate: ρ_k = d_k / d_{k-1}", fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.3)

    ax6.text(0.02, 0.98, "Low avg rate & stable → fast convergence\nHigh rate or unstable → slow/oscillating", 
             transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive convergence analysis plot to: {save_path}")
    else:
        plt.show()


def plot_pca_convergence_comparison(
    logits_traces: List[torch.Tensor],
    preds: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 5,
    bottom_k: int = 5,
    only_last_k: Optional[int] = 3,
    title: str = "PCA Space Convergence Analysis",
    save_path: Optional[str] = None,
    different_init: Optional[int] = None,
):
    """
    Compare convergence in different spaces:
    1. Full space (all components) - mean convergence
    2. Top-k principal components
    3. Bottom-k principal components
    
    This helps identify whether convergence issues are in the main signal directions
    (top PCs) or in the noise/tail directions (bottom PCs).
    
    This visualization is prediction-level: with different_init > 1, each sample
    contributes one prediction per initialization.
    
    Args:
        logits_traces: list of length T, each (N, L, V) where N = n_samples * different_init
        preds: (N, L) predictions
        labels: (N, L) ground truth
        top_k: number of top principal components
        bottom_k: number of bottom principal components
        only_last_k: only use last k steps for convergence
        title: plot title
        save_path: if not None, save figure to this path
        different_init: number of random initializations per sample (for annotation)
    """
    # Calculate convergences
    ref_device = logits_traces[-1].device
    preds_dev = preds.to(ref_device)
    labels_dev = labels.to(ref_device)

    full_conv = calculate_convergence(logits_traces, only_last_k=only_last_k)

    top_k_conv, bottom_k_conv, pca_info = calculate_pca_convergence(
        logits_traces, top_k=top_k, bottom_k=bottom_k, only_last_k=only_last_k
    )

    correctness = (preds_dev == labels_dev).all(dim=1).to(torch.float32)

    sorted_idx = torch.argsort(full_conv)

    full_sorted = full_conv[sorted_idx]
    top_k_sorted = top_k_conv[sorted_idx]
    bottom_k_sorted = bottom_k_conv[sorted_idx]
    corr_sorted = correctness[sorted_idx]

    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-9:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    full_norm = normalize(full_sorted).float().cpu().numpy()
    top_k_norm = normalize(top_k_sorted).float().cpu().numpy()
    bottom_k_norm = normalize(bottom_k_sorted).float().cpu().numpy()
    corr_np = corr_sorted.float().cpu().numpy()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    x = np.arange(len(full_sorted))
    colors = ['#FF6B6B' if c == 0 else '#4ECDC4' for c in corr_np]

    n_predictions = len(x)
    if different_init is not None and different_init > 1:
        n_samples = n_predictions // different_init
        data_info = f"(First Batch: {n_predictions} predictions = {n_samples} samples × {different_init} inits)"
    else:
        data_info = f"({n_predictions} predictions from first batch)"

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, full_norm, label='Full Space Mean Conv', linewidth=2.5, color='#2E86AB', alpha=0.9)
    ax1.scatter(x, corr_np, c=colors, label='Correctness', alpha=0.6, s=30, marker='x')
    ax1.set_xlabel('Prediction Index (sorted by convergence)', fontsize=10)
    ax1.set_ylabel('Normalized Convergence', fontsize=11)
    ax1.set_title(f'Full Space (All Components)\n{data_info}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, top_k_norm, label=f'Top-{top_k} PCs Conv', linewidth=2.5, color='#A23B72', alpha=0.9)
    ax2.scatter(x, corr_np, c=colors, label='Correctness', alpha=0.6, s=30, marker='x')
    ax2.set_xlabel('Prediction Index (sorted by convergence)', fontsize=10)
    ax2.set_ylabel('Normalized Convergence', fontsize=11)
    ax2.set_title(f'Top {top_k} Principal Components', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    top_k_var_pct = pca_info['top_k_variance'] * 100
    ax2.text(0.02, 0.98, f"Explains {top_k_var_pct:.1f}% variance", 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, bottom_k_norm, label=f'Bottom-{bottom_k} PCs Conv', linewidth=2.5, color='#F18F01', alpha=0.9)
    ax3.scatter(x, corr_np, c=colors, label='Correctness', alpha=0.6, s=30, marker='x')
    ax3.set_xlabel('Prediction Index (sorted by convergence)', fontsize=10)
    ax3.set_ylabel('Normalized Convergence', fontsize=11)
    ax3.set_title(f'Bottom {bottom_k} Principal Components', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    bottom_k_var_pct = pca_info['bottom_k_variance'] * 100
    ax3.text(0.02, 0.98, f"Explains {bottom_k_var_pct:.2f}% variance", 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, full_norm, label='Full Space', linewidth=2, color='#2E86AB', alpha=0.8)
    ax4.plot(x, top_k_norm, label=f'Top-{top_k} PCs', linewidth=2, color='#A23B72', alpha=0.8, linestyle='--')
    ax4.plot(x, bottom_k_norm, label=f'Bottom-{bottom_k} PCs', linewidth=2, color='#F18F01', alpha=0.8, linestyle='-.')
    ax4.set_xlabel('Prediction Index (sorted by convergence)', fontsize=10)
    ax4.set_ylabel('Normalized Convergence', fontsize=11)
    ax4.set_title('Convergence Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, :])
    variance_ratio = pca_info['explained_variance_ratio'].numpy()
    cumsum_var = pca_info['cumsum_variance'].numpy()
    n_show = min(50, len(variance_ratio))

    ax5_twin = ax5.twinx()
    ax5.bar(range(n_show), variance_ratio[:n_show], alpha=0.6, color='#3498DB', label='Individual Variance')
    ax5_twin.plot(range(n_show), cumsum_var[:n_show], color='#E74C3C', linewidth=2.5, label='Cumulative Variance')

    ax5.axvspan(-0.5, top_k-0.5, alpha=0.2, color='#A23B72', label=f'Top-{top_k} region')
    ax5.axvspan(len(variance_ratio)-bottom_k-0.5, len(variance_ratio)-0.5, alpha=0.2, color='#F18F01', label=f'Bottom-{bottom_k} region')

    ax5.set_xlabel('Principal Component Index', fontsize=11)
    ax5.set_ylabel('Explained Variance Ratio', fontsize=11)
    ax5_twin.set_ylabel('Cumulative Variance', fontsize=11)
    ax5.set_title(f'PCA Explained Variance (Total {pca_info["n_components"]} components)', fontsize=12, fontweight='bold')

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PCA convergence comparison plot to: {save_path}")
    else:
        plt.show()

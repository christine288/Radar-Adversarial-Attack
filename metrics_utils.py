from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


def compute_multiclass_map(
    y_true: Sequence[int],
    y_score: Sequence[Sequence[float]] | np.ndarray,
    num_classes: int,
) -> float:
    """Compute one-vs-rest mAP for multiclass classification.

    Only classes that appear at least once in ``y_true`` are included in the
    macro average to avoid undefined AP on empty-positive classes.
    """
    if num_classes <= 0:
        return 0.0

    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    if y_true_arr.size == 0 or y_score_arr.size == 0:
        return 0.0
    if y_score_arr.ndim != 2 or y_score_arr.shape[0] != y_true_arr.shape[0]:
        raise ValueError(
            "y_score must have shape [n_samples, n_classes] and align with y_true"
        )

    classes = np.arange(num_classes, dtype=np.int64)
    y_true_bin = label_binarize(y_true_arr, classes=classes)
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)
    if y_true_bin.shape[1] != num_classes:
        y_true_bin = np.pad(
            y_true_bin,
            ((0, 0), (0, max(0, num_classes - y_true_bin.shape[1]))),
            mode="constant",
            constant_values=0,
        )

    ap_values: list[float] = []
    for class_idx in range(num_classes):
        positives = int(y_true_bin[:, class_idx].sum())
        if positives <= 0:
            continue
        ap = average_precision_score(y_true_bin[:, class_idx], y_score_arr[:, class_idx])
        ap_values.append(float(ap))

    return float(np.mean(ap_values)) if ap_values else 0.0


def format_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def compute_signal_quality_metrics(
    x_ref: torch.Tensor,
    x_adv: torch.Tensor,
) -> dict[str, float]:
    delta = (x_adv - x_ref).reshape(x_ref.size(0), -1)
    perturb_l2 = delta.norm(p=2, dim=1)
    distortion_mae = delta.abs().mean(dim=1)

    x_ref_flat = x_ref.reshape(x_ref.size(0), -1)
    x_adv_flat = x_adv.reshape(x_adv.size(0), -1)
    mu_x = x_ref_flat.mean(dim=1)
    mu_y = x_adv_flat.mean(dim=1)
    var_x = ((x_ref_flat - mu_x.unsqueeze(1)) ** 2).mean(dim=1)
    var_y = ((x_adv_flat - mu_y.unsqueeze(1)) ** 2).mean(dim=1)
    cov_xy = ((x_ref_flat - mu_x.unsqueeze(1)) * (x_adv_flat - mu_y.unsqueeze(1))).mean(dim=1)

    data_range = (x_ref_flat.max(dim=1).values - x_ref_flat.min(dim=1).values).clamp_min(1e-6)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    )
    ssim = ssim.clamp(min=-1.0, max=1.0)

    mse = delta.pow(2).mean(dim=1).clamp_min(1e-12)
    psnr = 10.0 * torch.log10((data_range.pow(2)) / mse)

    return {
        "avg_perturbation_distance": float(perturb_l2.mean().item()),
        "avg_distortion": float(distortion_mae.mean().item()),
        "avg_ssim": float(ssim.mean().item()),
        "avg_psnr": float(psnr.mean().item()),
    }

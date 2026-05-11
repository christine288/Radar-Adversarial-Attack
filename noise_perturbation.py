"""Noise perturbation utilities for robustness experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def clamp_if_needed(x: torch.Tensor, clamp: Optional[Tuple[float, float]]) -> torch.Tensor:
    if clamp is None:
        return x
    lo, hi = clamp
    return torch.clamp(x, lo, hi)


def project_relative_change(
    x_ref: torch.Tensor,
    x_adv: torch.Tensor,
    max_rel_change: Optional[float] = 0.05,
    eps: float = 1e-6,
    min_scale: float = 0.0,
) -> torch.Tensor:
    """
    Project perturbation to relative L-infinity box:
    |x_adv - x_ref| <= max_rel_change * max(|x_ref|, eps)
    """
    if max_rel_change is None:
        return x_adv
    if max_rel_change <= 0:
        return x_ref
    scale = torch.maximum(x_ref.abs(), torch.full_like(x_ref, float(max(eps, min_scale))))
    delta_limit = float(max_rel_change) * scale
    delta = torch.clamp(x_adv - x_ref, min=-delta_limit, max=delta_limit)
    return x_ref + delta


def relative_change_stats(
    x_ref: torch.Tensor,
    x_adv: torch.Tensor,
    eps: float = 1e-6,
    min_scale: float = 0.0,
) -> dict[str, float]:
    """
    Return mean/max relative absolute change for reporting.

    ``min_scale`` 与 ``project_relative_change`` 中一致：分母为 ``max(|x_ref|, max(eps, min_scale))``，
    避免 ``|x_ref|`` 接近 0 时相对变化被夸大。
    """
    floor = float(max(eps, min_scale))
    scale = torch.maximum(x_ref.abs(), torch.full_like(x_ref, floor))
    rel = (x_adv - x_ref).abs() / scale
    return {
        "mean_rel_change": float(rel.mean().item()),
        "max_rel_change": float(rel.max().item()),
    }


def add_gaussian_noise(
    x: torch.Tensor,
    sigma: float = 0.5,
    max_rel_change: Optional[float] = 0.05,
    min_scale: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Add Gaussian noise: x_noisy = x + N(0, sigma^2)."""
    noise = torch.randn_like(x) * float(sigma)
    x_noisy = x + noise
    x_noisy = project_relative_change(x, x_noisy, max_rel_change=max_rel_change, min_scale=min_scale)
    return clamp_if_needed(x_noisy, clamp)


def add_speckle_noise(
    x: torch.Tensor,
    sigma: float = 0.5,
    max_rel_change: Optional[float] = 0.05,
    min_scale: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Add multiplicative speckle-like noise: x_noisy = x * (1 + n)."""
    noise = torch.randn_like(x) * float(sigma)
    x_noisy = x * (1.0 + noise)
    x_noisy = project_relative_change(x, x_noisy, max_rel_change=max_rel_change, min_scale=min_scale)
    return clamp_if_needed(x_noisy, clamp)


def add_salt_pepper_noise(
    x: torch.Tensor,
    prob: float = 0.2,
    salt_value: float = 1.0,
    pepper_value: float = 0.0,
    max_rel_change: Optional[float] = 0.05,
    min_scale: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Add salt-and-pepper noise by random pixel replacement."""
    if prob <= 0:
        return clamp_if_needed(x, clamp)
    prob = float(prob)
    salt_mask = torch.rand_like(x) < (prob / 2.0)
    pepper_mask = torch.rand_like(x) < (prob / 2.0)
    x_noisy = x.clone()
    x_noisy[salt_mask] = float(salt_value)
    x_noisy[pepper_mask] = float(pepper_value)
    x_noisy = project_relative_change(x, x_noisy, max_rel_change=max_rel_change, min_scale=min_scale)
    return clamp_if_needed(x_noisy, clamp)

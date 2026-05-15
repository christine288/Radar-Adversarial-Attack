"""White-box adversarial attacks and evaluation CLI for radar track classification."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, TensorDataset

from data_utils import TrackSample, batch_tracks_to_sequences
from device_utils import print_device_banner, resolve_device_str
from metrics_utils import compute_multiclass_map
from metrics_utils import compute_signal_quality_metrics
from metrics_utils import format_pct
from model import RadarTrackTransformer
from noise_perturbation import clamp_if_needed
from noise_perturbation import project_relative_change
from noise_perturbation import relative_change_stats


@dataclass
class AttackResult:
    """Container for generated adversarial examples and simple diagnostics."""

    x_adv: torch.Tensor
    logits: torch.Tensor
    y_pred: torch.Tensor
    mean_rel_change: float
    max_rel_change: float


def _batch_l2_distance(x_ref: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    return (x_adv - x_ref).reshape(x_ref.size(0), -1).norm(p=2, dim=1)


def _batch_index_select(matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    row_idx = torch.arange(matrix.size(0), device=matrix.device)
    return matrix[row_idx, indices]


def _resolve_targets(
    y_true: torch.Tensor,
    y_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if y_target is None:
        return y_true
    if y_target.shape != y_true.shape:
        raise ValueError(f"y_target shape {tuple(y_target.shape)} must match y_true shape {tuple(y_true.shape)}")
    return y_target


def _attack_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    targeted: bool,
    y_target: Optional[torch.Tensor] = None,
    confidence: float = 0.0,
    loss_mode: str = "margin",
) -> torch.Tensor:
    """Return scalar objective to maximize for gradient-based attack updates."""
    targets = _resolve_targets(y_true, y_target if targeted else None)
    sel = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets.view(-1, 1), float("-inf"))
    best_other = masked.max(dim=1).values

    if targeted:
        margin_objective = sel - best_other
        ce_objective = -F.cross_entropy(logits, targets, reduction="mean")
    else:
        margin_objective = best_other - sel
        ce_objective = F.cross_entropy(logits, y_true, reduction="mean")

    if loss_mode == "ce":
        loss = ce_objective
    elif loss_mode == "margin":
        loss = margin_objective.mean() + 0.2 * ce_objective
    elif loss_mode == "margin_only":
        loss = margin_objective.mean()
    else:
        raise ValueError(f"unknown loss_mode: {loss_mode}")

    if confidence > 0:
        if targeted:
            margin_bonus = torch.clamp(margin_objective - confidence, min=0.0).mean()
        else:
            margin_bonus = torch.clamp(margin_objective - confidence, min=0.0).mean()
        loss = loss + margin_bonus
    return loss


def _finalize_attack(
    model: torch.nn.Module,
    x_ref: torch.Tensor,
    x_adv: torch.Tensor,
) -> AttackResult:
    with torch.no_grad():
        logits = model(x_adv)
        y_pred = logits.argmax(dim=1)
        stats = relative_change_stats(x_ref, x_adv)
    return AttackResult(
        x_adv=x_adv.detach(),
        logits=logits.detach(),
        y_pred=y_pred.detach(),
        mean_rel_change=stats["mean_rel_change"],
        max_rel_change=stats["max_rel_change"],
    )


def _relative_step_tensor(
    x_ref: torch.Tensor,
    *,
    step_size: float,
    max_rel_change: Optional[float],
    budget_floor: float,
) -> torch.Tensor | float:
    if max_rel_change is None:
        return float(step_size)
    scale = torch.maximum(x_ref.abs(), torch.full_like(x_ref, float(max(budget_floor, 1e-6))))
    delta_limit = float(max_rel_change) * scale
    return float(step_size) * delta_limit


def _untargeted_margin(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, y_true.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y_true.view(-1, 1), float("-inf"))
    other_best = masked.max(dim=1).values
    return true_logits - other_best


def _attack_success_and_score(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    targeted: bool,
    y_target: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = _resolve_targets(y_true, y_target if targeted else None)
    sel = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets.view(-1, 1), float("-inf"))
    best_other = masked.max(dim=1).values
    pred = logits.argmax(dim=1)
    if targeted:
        success = pred.eq(targets)
        score = best_other - sel
    else:
        success = pred.ne(y_true)
        score = sel - best_other
    return success, score


def _candidate_target_classes(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    num_classes = logits.size(1)
    k = max(1, min(int(top_k), max(1, num_classes - 1)))
    masked = logits.clone()
    masked.scatter_(1, y_true.view(-1, 1), float("-inf"))
    return masked.topk(k=k, dim=1).indices


def _select_stronger_attack_result(
    current: Optional[AttackResult],
    candidate: AttackResult,
    x_ref: torch.Tensor,
    y_true: torch.Tensor,
    *,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
) -> AttackResult:
    if current is None:
        return candidate

    current_success, current_score = _attack_success_and_score(
        current.logits,
        y_true,
        targeted=targeted,
        y_target=y_target,
    )
    candidate_success, candidate_score = _attack_success_and_score(
        candidate.logits,
        y_true,
        targeted=targeted,
        y_target=y_target,
    )
    pick_candidate = (~current_success & candidate_success) | (
        current_success.eq(candidate_success) & (candidate_score < current_score)
    )
    if not pick_candidate.any():
        return current

    x_adv = current.x_adv.clone()
    logits = current.logits.clone()
    y_pred = current.y_pred.clone()
    x_adv[pick_candidate] = candidate.x_adv[pick_candidate]
    logits[pick_candidate] = candidate.logits[pick_candidate]
    y_pred[pick_candidate] = candidate.y_pred[pick_candidate]
    merged_stats = relative_change_stats(x_ref, x_adv)
    return AttackResult(
        x_adv=x_adv,
        logits=logits,
        y_pred=y_pred,
        mean_rel_change=merged_stats["mean_rel_change"],
        max_rel_change=merged_stats["max_rel_change"],
    )


def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    step_size: float = 1.0,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
    max_rel_change: Optional[float] = 0.05,
    budget_floor: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
    loss_mode: str = "ce",
) -> AttackResult:
    """Single-step FGSM under the project's relative change constraint."""
    model.eval()
    x_ref = x.detach()
    x_adv = x_ref.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = _attack_loss(logits, y_true, targeted=targeted, y_target=y_target, loss_mode=loss_mode)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
    direction = grad.sign()
    if targeted:
        direction = -direction

    step = _relative_step_tensor(
        x_ref,
        step_size=step_size,
        max_rel_change=max_rel_change,
        budget_floor=budget_floor,
    )
    x_adv = x_adv.detach() + step * direction
    x_adv = project_relative_change(
        x_ref,
        x_adv,
        max_rel_change=max_rel_change,
        min_scale=budget_floor,
    )
    x_adv = clamp_if_needed(x_adv, clamp)
    return _finalize_attack(model, x_ref, x_adv)


def pgd_linf_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    step_size: float = 0.01,
    num_steps: int = 20,
    random_start: bool = True,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
    max_rel_change: Optional[float] = 0.05,
    budget_floor: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
    loss_mode: str = "margin",
    momentum: float = 0.75,
) -> AttackResult:
    """Projected Gradient Descent under relative Linf-style perturbation budget."""
    model.eval()
    x_ref = x.detach()

    if random_start and max_rel_change is not None and max_rel_change > 0:
        scale = torch.maximum(x_ref.abs(), torch.full_like(x_ref, float(max(budget_floor, 1e-6))))
        delta = torch.empty_like(x_ref).uniform_(-1.0, 1.0) * (float(max_rel_change) * scale)
        x_adv = x_ref + delta
        x_adv = clamp_if_needed(x_adv, clamp)
    else:
        x_adv = x_ref.clone()

    best_result: Optional[AttackResult] = None
    grad_momentum = torch.zeros_like(x_ref)

    step = _relative_step_tensor(
        x_ref,
        step_size=step_size,
        max_rel_change=max_rel_change,
        budget_floor=budget_floor,
    )

    for _ in range(int(num_steps)):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = _attack_loss(
            logits,
            y_true,
            targeted=targeted,
            y_target=y_target,
            loss_mode=loss_mode,
        )
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        grad_norm = grad.abs().mean(dim=tuple(range(1, grad.ndim)), keepdim=True).clamp_min(1e-12)
        grad_momentum = float(momentum) * grad_momentum + grad / grad_norm
        direction = grad_momentum.sign()
        if targeted:
            direction = -direction
        x_adv = x_adv.detach() + step * direction
        x_adv = project_relative_change(
            x_ref,
            x_adv,
            max_rel_change=max_rel_change,
            min_scale=budget_floor,
        )
        x_adv = clamp_if_needed(x_adv, clamp)
        candidate = _finalize_attack(model, x_ref, x_adv)
        best_result = _select_stronger_attack_result(
            best_result,
            candidate,
            x_ref,
            y_true,
            targeted=targeted,
            y_target=y_target,
        )
    if best_result is None:
        return _finalize_attack(model, x_ref, x_adv)
    return best_result


def cw_l2_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    c: float = 1e-2,
    lr: float = 1e-2,
    num_steps: int = 200,
    confidence: float = 0.0,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
    max_rel_change: Optional[float] = 0.05,
    budget_floor: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
    random_start: bool = True,
) -> AttackResult:
    """Lightweight CW-L2 style attack with optional relative-change projection."""
    model.eval()
    x_ref = x.detach()
    if random_start and max_rel_change is not None and max_rel_change > 0:
        scale = torch.maximum(x_ref.abs(), torch.full_like(x_ref, float(max(budget_floor, 1e-6))))
        delta_init = torch.empty_like(x_ref).uniform_(-1.0, 1.0) * (float(max_rel_change) * scale)
    else:
        delta_init = torch.zeros_like(x_ref)
    delta = delta_init.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))

    best_x_adv = x_ref.clone()
    best_score = torch.full((x_ref.size(0),), float("inf"), device=x_ref.device)

    for _ in range(int(num_steps)):
        x_adv = x_ref + delta
        x_adv = project_relative_change(
            x_ref,
            x_adv,
            max_rel_change=max_rel_change,
            min_scale=budget_floor,
        )
        x_adv = clamp_if_needed(x_adv, clamp)

        logits = model(x_adv)
        targets = _resolve_targets(y_true, y_target if targeted else None)

        sel = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, targets.view(-1, 1), float("-inf"))
        best_other = masked.max(dim=1).values

        if targeted:
            f_term = torch.clamp(best_other - sel + float(confidence), min=0.0)
            success = logits.argmax(dim=1).eq(targets)
        else:
            f_term = torch.clamp(sel - best_other + float(confidence), min=0.0)
            success = logits.argmax(dim=1).ne(y_true)

        l2_term = (x_adv - x_ref).reshape(x_ref.size(0), -1).pow(2).sum(dim=1)
        total_loss = l2_term.mean() + float(c) * f_term.mean()

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            improved = success & (l2_term < best_score)
            if improved.any():
                best_x_adv[improved] = x_adv.detach()[improved]
                best_score[improved] = l2_term[improved]

    fallback_mask = torch.isinf(best_score)
    if fallback_mask.any():
        with torch.no_grad():
            x_adv = x_ref + delta
            x_adv = project_relative_change(
                x_ref,
                x_adv,
                max_rel_change=max_rel_change,
                min_scale=budget_floor,
            )
            x_adv = clamp_if_needed(x_adv, clamp)
            best_x_adv[fallback_mask] = x_adv.detach()[fallback_mask]

    return _finalize_attack(model, x_ref, best_x_adv)


def deepfool_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    num_steps: int = 50,
    overshoot: float = 0.02,
    max_rel_change: Optional[float] = 0.05,
    budget_floor: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
) -> AttackResult:
    """
    多类 DeepFool（迭代线性化）：对每个样本在「真类 vs 其它类」的决策边界上取最小 L2 方向步长，
    再乘以 ``(1 + overshoot)``，最后 ``project_relative_change``（与 FGSM/PGD/CW 一致）。

    默认 **untargeted**：在所有 ``k != y_true`` 中选使一步扰动范数最小的类边界；
    **targeted** 时只对 ``y_target`` 对应类求边界步长。
    实现上按 batch 维逐样本前向/反传，避免错误地向量化（见 Moosavi-Dezfooli et al., DeepFool）。
    """
    model.eval()
    x_ref = x.detach()
    x_adv = x_ref.clone()
    num_classes = int(model.head[-1].out_features)
    if num_classes < 2:
        return _finalize_attack(model, x_ref, x_adv)

    for _ in range(int(num_steps)):
        with torch.no_grad():
            logits_chk = model(x_adv)
            pred = logits_chk.argmax(dim=1)
            if (pred != y_true).all():
                break

        delta_accum = torch.zeros_like(x_adv)
        B = x_adv.size(0)
        for b in range(B):
            if not targeted and pred[b] != y_true[b]:
                continue
            if targeted:
                if y_target is None:
                    raise ValueError("targeted DeepFool 需要 y_target")
                if pred[b] == y_target[b]:
                    continue

            yt = int(y_true[b].item())
            if targeted:
                assert y_target is not None
                k_list = [int(y_target[b].item())]
                if k_list[0] == yt:
                    continue
            else:
                k_list = [k for k in range(num_classes) if k != yt]

            x_b = x_adv[b : b + 1].clone().detach().requires_grad_(True)
            lb = model(x_b)[0]
            grad_y = torch.autograd.grad(lb[yt], x_b, retain_graph=True, create_graph=False)[0]

            best_r: Optional[torch.Tensor] = None
            best_l2_sq = float("inf")

            for idx, k in enumerate(k_list):
                last_k = idx == len(k_list) - 1
                grad_k = torch.autograd.grad(lb[k], x_b, retain_graph=not last_k, create_graph=False)[0]
                w = grad_k - grad_y
                fk = (lb[k] - lb[yt]).detach()
                denom = w.reshape(-1).dot(w.reshape(-1)).clamp_min(1e-12)
                r_b = (fk / denom) * w
                nsq = float(r_b.reshape(-1).dot(r_b.reshape(-1)).item())
                if nsq < best_l2_sq:
                    best_l2_sq = nsq
                    best_r = r_b

            if best_r is not None:
                delta_accum[b : b + 1] = (1.0 + float(overshoot)) * best_r

        x_adv = x_adv + delta_accum
        x_adv = project_relative_change(
            x_ref,
            x_adv,
            max_rel_change=max_rel_change,
            min_scale=budget_floor,
        )
        x_adv = clamp_if_needed(x_adv, clamp)

    return _finalize_attack(model, x_ref, x_adv)


def build_loader(samples: List[TrackSample], batch_size: int) -> DataLoader:
    x, y = batch_tracks_to_sequences(samples)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_model(model_path: str, device: str) -> RadarTrackTransformer:
    try:
        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(model_path, map_location=device)
    except PermissionError as e:
        raise PermissionError(
            f"无法读取模型文件（权限拒绝）: {model_path}\n"
            "请确认路径是权重 .pth 文件本身（不要填目录）、未被其它程序占用，且不要使用文档占位符「...」。"
        ) from e
    model = RadarTrackTransformer(input_size=ckpt["input_size"], num_classes=ckpt.get("num_classes", 6))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def _compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: List[List[float]],
    num_classes: int,
) -> Dict[str, float]:
    target_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    sample_accuracy = float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)) if y_true else 0.0
    return {
        "target_recall": target_recall,
        "sample_accuracy": sample_accuracy,
        "mAP": compute_multiclass_map(y_true, y_score, num_classes=num_classes),
    }


def _compute_interference_success_rate(
    y_true: List[int],
    y_pred_clean: List[int],
    y_pred_attack: List[int],
) -> float:
    clean_correct_idx = [i for i, (yt, pc) in enumerate(zip(y_true, y_pred_clean)) if yt == pc]
    if not clean_correct_idx:
        return 0.0
    success = sum(1 for i in clean_correct_idx if y_pred_attack[i] != y_true[i])
    return float(success / len(clean_correct_idx))


def evaluate_whitebox_attack(
    *,
    model: RadarTrackTransformer,
    loader: DataLoader,
    device: str,
    attack: str,
    fgsm_step_size: float,
    pgd_step_size: float,
    pgd_steps: int,
    attack_restarts: int,
    cw_c: float,
    cw_lr: float,
    cw_steps: int,
    cw_confidence: float,
    max_rel_change: float | None,
    budget_floor: float,
    clamp: tuple[float, float] | None,
    attack_clean_only: bool,
    pgd_momentum: float,
    attack_loss: str,
    targeted_topk: int,
    deepfool_steps: int = 50,
    deepfool_overshoot: float = 0.02,
) -> Dict[str, float]:
    y_true_all: List[int] = []
    y_pred_clean_all: List[int] = []
    y_pred_attack_all: List[int] = []
    y_score_clean_all: List[List[float]] = []
    y_score_attack_all: List[List[float]] = []
    clean_correct_total = 0
    attack_success_total = 0
    rel_mean_weighted_sum = 0.0
    rel_max_overall = 0.0
    attacked_sample_total = 0
    restart_success_rates: List[float] = []
    perturb_distance_sum = 0.0
    distortion_sum = 0.0
    ssim_sum = 0.0
    psnr_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            clean_logits = model(x)
            pred_clean = clean_logits.argmax(dim=1)
            clean_probs = torch.softmax(clean_logits, dim=1)

        attack_mask = pred_clean.eq(y) if attack_clean_only else torch.ones_like(y, dtype=torch.bool)
        y_pred_attack_batch = pred_clean.clone()

        if attack_mask.any():
            x_attack = x[attack_mask]
            y_attack = y[attack_mask]
            clean_logits_attack = clean_logits[attack_mask]
            best_res: Optional[AttackResult] = None
            restart_count = 1 if attack == "fgsm" else max(1, int(attack_restarts))
            per_restart_success_counts: List[int] = []
            target_candidates = _candidate_target_classes(clean_logits_attack, y_attack, targeted_topk)
            target_count = int(target_candidates.size(1))
            for _ in range(restart_count):
                untargeted_candidates: List[AttackResult] = []
                if attack == "fgsm":
                    untargeted_candidates.append(
                        fgsm_attack(
                            model,
                            x_attack,
                            y_attack,
                            step_size=fgsm_step_size,
                            targeted=False,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            loss_mode=attack_loss,
                        )
                    )
                elif attack == "pgd":
                    untargeted_candidates.append(
                        pgd_linf_attack(
                            model,
                            x_attack,
                            y_attack,
                            step_size=pgd_step_size,
                            num_steps=pgd_steps,
                            random_start=True,
                            targeted=False,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            loss_mode=attack_loss,
                            momentum=pgd_momentum,
                        )
                    )
                elif attack == "cw":
                    untargeted_candidates.append(
                        cw_l2_attack(
                            model,
                            x_attack,
                            y_attack,
                            c=cw_c,
                            lr=cw_lr,
                            num_steps=cw_steps,
                            confidence=cw_confidence,
                            targeted=False,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            random_start=True,
                        )
                    )
                elif attack == "deepfool":
                    untargeted_candidates.append(
                        deepfool_attack(
                            model,
                            x_attack,
                            y_attack,
                            num_steps=deepfool_steps,
                            overshoot=deepfool_overshoot,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            targeted=False,
                        )
                    )
                else:
                    raise ValueError(f"unknown attack: {attack}")

                restart_best = untargeted_candidates[0]
                for target_idx in range(target_count):
                    y_target = target_candidates[:, target_idx]
                    if attack == "fgsm":
                        targeted_candidate = fgsm_attack(
                            model,
                            x_attack,
                            y_attack,
                            step_size=fgsm_step_size,
                            targeted=True,
                            y_target=y_target,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            loss_mode=attack_loss,
                        )
                    elif attack == "pgd":
                        targeted_candidate = pgd_linf_attack(
                            model,
                            x_attack,
                            y_attack,
                            step_size=pgd_step_size,
                            num_steps=pgd_steps,
                            random_start=True,
                            targeted=True,
                            y_target=y_target,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            loss_mode=attack_loss,
                            momentum=pgd_momentum,
                        )
                    elif attack == "cw":
                        targeted_candidate = cw_l2_attack(
                            model,
                            x_attack,
                            y_attack,
                            c=cw_c,
                            lr=cw_lr,
                            num_steps=cw_steps,
                            confidence=cw_confidence,
                            targeted=True,
                            y_target=y_target,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            random_start=True,
                        )
                    elif attack == "deepfool":
                        targeted_candidate = deepfool_attack(
                            model,
                            x_attack,
                            y_attack,
                            num_steps=deepfool_steps,
                            overshoot=deepfool_overshoot,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            targeted=True,
                            y_target=y_target,
                        )
                    else:
                        raise ValueError(f"unknown attack: {attack}")
                    restart_best = _select_stronger_attack_result(
                        restart_best,
                        targeted_candidate,
                        x_attack,
                        y_attack,
                    )

                per_restart_success_counts.append(int(restart_best.y_pred.ne(y_attack).sum().item()))
                best_res = _select_stronger_attack_result(best_res, restart_best, x_attack, y_attack)
            assert best_res is not None
            y_pred_attack_batch[attack_mask] = best_res.y_pred
            attack_probs_batch = clean_probs.clone()
            attack_probs_batch[attack_mask] = torch.softmax(best_res.logits, dim=1)
            attacked_sample_total += int(x_attack.size(0))
            quality = compute_signal_quality_metrics(x_attack, best_res.x_adv)
            batch_attacked = int(x_attack.size(0))
            perturb_distance_sum += quality["avg_perturbation_distance"] * batch_attacked
            distortion_sum += quality["avg_distortion"] * batch_attacked
            ssim_sum += quality["avg_ssim"] * batch_attacked
            psnr_sum += quality["avg_psnr"] * batch_attacked
            rel_mean_weighted_sum += float(best_res.mean_rel_change) * int(x_attack.size(0))
            rel_max_overall = max(rel_max_overall, float(best_res.max_rel_change))
            if per_restart_success_counts:
                restart_success_rates.extend(
                    [float(v / max(1, int(x_attack.size(0)))) for v in per_restart_success_counts]
                )

        else:
            attack_probs_batch = clean_probs

        y_true_all.extend(y.cpu().numpy().tolist())
        y_pred_clean_all.extend(pred_clean.cpu().numpy().tolist())
        y_pred_attack_all.extend(y_pred_attack_batch.cpu().numpy().tolist())
        y_score_clean_all.extend(clean_probs.cpu().numpy().tolist())
        y_score_attack_all.extend(attack_probs_batch.cpu().numpy().tolist())
        clean_correct_mask = pred_clean.eq(y)
        attack_success_mask = clean_correct_mask & y_pred_attack_batch.ne(y)
        clean_correct_total += int(clean_correct_mask.sum().item())
        attack_success_total += int(attack_success_mask.sum().item())

    clean_metrics = _compute_metrics(
        y_true_all,
        y_pred_clean_all,
        y_score_clean_all,
        num_classes=model.head[-1].out_features,
    )
    attack_metrics = _compute_metrics(
        y_true_all,
        y_pred_attack_all,
        y_score_attack_all,
        num_classes=model.head[-1].out_features,
    )
    isr = _compute_interference_success_rate(y_true_all, y_pred_clean_all, y_pred_attack_all)
    total_samples = len(y_true_all)
    restart_mean = float(sum(restart_success_rates) / len(restart_success_rates)) if restart_success_rates else 0.0
    restart_std = (
        float(torch.tensor(restart_success_rates, dtype=torch.float32).std(unbiased=False).item())
        if restart_success_rates
        else 0.0
    )

    return {
        "clean_target_recall": clean_metrics["target_recall"],
        "clean_sample_accuracy": clean_metrics["sample_accuracy"],
        "clean_mAP": clean_metrics["mAP"],
        "attack_target_recall": attack_metrics["target_recall"],
        "attack_sample_accuracy": attack_metrics["sample_accuracy"],
        "attack_mAP": attack_metrics["mAP"],
        "attack_success_rate": isr,
        "attack_success_count": attack_success_total,
        "clean_correct_count": clean_correct_total,
        "attacked_sample_count": attacked_sample_total,
        "attacked_sample_rate": float(attacked_sample_total / total_samples) if total_samples else 0.0,
        "total_samples": total_samples,
        "avg_perturbation_distance": float(perturb_distance_sum / attacked_sample_total)
        if attacked_sample_total
        else 0.0,
        "avg_distortion": float(distortion_sum / attacked_sample_total) if attacked_sample_total else 0.0,
        "avg_ssim": float(ssim_sum / attacked_sample_total) if attacked_sample_total else 0.0,
        "avg_psnr": float(psnr_sum / attacked_sample_total) if attacked_sample_total else 0.0,
        "mean_rel_change": float(rel_mean_weighted_sum / total_samples) if total_samples else 0.0,
        "max_rel_change": rel_max_overall,
        "restart_success_rate_mean": restart_mean,
        "restart_success_rate_std": restart_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="White-box adversarial evaluation (FGSM/PGD/CW/DeepFool)")
    parser.add_argument("--model_path", type=str, default="radar_transformer.pth")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda", help="默认 cuda；无 CUDA 回退 CPU")
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--mat_test_dir", type=str, default=None, help="测试用 .mat 根目录")
    parser.add_argument("--mat_dir", type=str, default=None, help="同 --mat_test_dir")
    parser.add_argument("--attack", type=str, default="all", choices=["all", "fgsm", "pgd", "cw", "deepfool"])
    parser.add_argument(
        "--max_rel_change",
        type=float,
        default=0.05,
        help="相对 L∞ 预算（如 0.05=5%%）；可改为 0.1 做更强约束下的对比。与 --no_rel_budget 二选一",
    )
    parser.add_argument(
        "--no_rel_budget",
        action="store_true",
        help="不做相对 L∞ 投影（等价 max_rel_change=None），用于对比是否因约束过强导致成功率偏低",
    )
    parser.add_argument("--budget_floor", type=float, default=0.0, help="最小尺度地板，避免接近 0 的维度预算过小")
    parser.add_argument("--clamp_min", type=float, default=None)
    parser.add_argument("--clamp_max", type=float, default=None)
    parser.add_argument("--fgsm_step_size", type=float, default=1.0)
    parser.add_argument("--pgd_step_size", type=float, default=1.0, help="PGD 步长系数（乘以 max_rel_change*scale）")
    parser.add_argument("--pgd_steps", type=int, default=300)
    parser.add_argument("--pgd_momentum", type=float, default=0.9)
    parser.add_argument("--attack_restarts", type=int, default=10)
    parser.add_argument(
        "--attack_loss",
        type=str,
        default="ce",
        choices=["margin", "margin_only", "ce"],
        help="梯度损失；ce 梯度通常更利于强攻击（默认可改回 margin）",
    )
    parser.add_argument(
        "--attack_clean_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only attack clean-correct samples. Recommended for attack success rate evaluation.",
    )
    parser.add_argument("--cw_c", type=float, default=50.0)
    parser.add_argument("--cw_lr", type=float, default=0.005)
    parser.add_argument("--cw_steps", type=int, default=1000)
    parser.add_argument("--cw_confidence", type=float, default=1.0)
    parser.add_argument("--deepfool_steps", type=int, default=50, help="DeepFool 迭代步数（逐样本梯度，较慢）")
    parser.add_argument("--deepfool_overshoot", type=float, default=0.02, help="DeepFool 步长乘以 (1+overshoot)")
    parser.add_argument(
        "--debug_grad",
        action="store_true",
        help="打印首个 batch 的输入梯度统计（用于确认是否获取到梯度）",
    )
    parser.add_argument(
        "--targeted_topk",
        type=int,
        default=5,
        help="Try targeted attacks toward top-k non-true classes and keep the strongest result.",
    )
    args = parser.parse_args()

    def _reject_path_placeholder(name: str, value: str | None) -> None:
        if value is None:
            return
        s = str(value).strip()
        if s in ("...", "..", ".") or (len(s) >= 3 and all(c == "." for c in s)):
            parser.error(
                f"{name} 不能为占位符（如 ...），请填写本机真实路径。"
                f" 模型示例: dataoutput/model_aug_transformer.pth"
            )

    _reject_path_placeholder("--model_path", args.model_path)
    _reject_path_placeholder("--mat_test_dir", args.mat_test_dir)
    _reject_path_placeholder("--mat_dir", args.mat_dir)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    mat_test_root = args.mat_test_dir or args.mat_dir
    if not mat_test_root:
        parser.error("必须指定 --mat_test_dir 或 --mat_dir（测试用 MAT 目录）")

    device = resolve_device_str(args.device, require_gpu=args.require_gpu)
    print_device_banner(device)
    model = load_model(args.model_path, device=device)

    from mat_loader import load_mat_directory

    meta_path = str(args.model_path) + ".meta.json"
    tgt = 32
    if Path(meta_path).is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tgt = int(meta.get("mat_target_points", 32))
    test_samples, _ = load_mat_directory(mat_test_root, target_points=tgt)
    loader = build_loader(test_samples, batch_size=args.batch_size)

    clamp = None
    if args.clamp_min is not None or args.clamp_max is not None:
        if args.clamp_min is None or args.clamp_max is None:
            raise ValueError("clamp_min 和 clamp_max 需要同时提供，或都不提供")
        clamp = (float(args.clamp_min), float(args.clamp_max))

    attacks = ["fgsm", "pgd", "cw", "deepfool"] if args.attack == "all" else [args.attack]
    print("白盒对抗攻击评估（PyTorch）:")
    for name in attacks:
        if args.debug_grad:
            x0, y0 = next(iter(loader))
            x0 = x0.to(device)
            y0 = y0.to(device)
            x0 = x0.detach().requires_grad_(True)
            logits0 = model(x0)
            loss0 = _attack_loss(
                logits0,
                y0,
                targeted=False,
                y_target=None,
                confidence=0.0,
                loss_mode=str(args.attack_loss),
            )
            g0 = torch.autograd.grad(loss0, x0, only_inputs=True)[0]
            with torch.no_grad():
                g_abs = g0.abs()
                zero_frac = float((g_abs <= 0).float().mean().item())
                nan_frac = float(torch.isnan(g0).float().mean().item())
                inf_frac = float(torch.isinf(g0).float().mean().item())
                print(f"[debug_grad] attack={name}")
                print(f"[debug_grad] grad_abs_mean={float(g_abs.mean().item()):.6e}")
                print(f"[debug_grad] grad_abs_max ={float(g_abs.max().item()):.6e}")
                print(f"[debug_grad] zero_frac   ={zero_frac * 100:.2f}%")
                print(f"[debug_grad] nan_frac    ={nan_frac * 100:.2f}%")
                print(f"[debug_grad] inf_frac    ={inf_frac * 100:.2f}%")

        r = evaluate_whitebox_attack(
            model=model,
            loader=loader,
            device=device,
            attack=name,
            fgsm_step_size=float(args.fgsm_step_size),
            pgd_step_size=float(args.pgd_step_size),
            pgd_steps=int(args.pgd_steps),
            attack_restarts=int(args.attack_restarts),
            cw_c=float(args.cw_c),
            cw_lr=float(args.cw_lr),
            cw_steps=int(args.cw_steps),
            cw_confidence=float(args.cw_confidence),
            max_rel_change=None if bool(args.no_rel_budget) else args.max_rel_change,
            budget_floor=float(args.budget_floor),
            clamp=clamp,
            attack_clean_only=bool(args.attack_clean_only),
            pgd_momentum=float(args.pgd_momentum),
            attack_loss=str(args.attack_loss),
            targeted_topk=int(args.targeted_topk),
            deepfool_steps=int(args.deepfool_steps),
            deepfool_overshoot=float(args.deepfool_overshoot),
        )
        print(f"- {name}:")
        print(f"  Clean Sample Accuracy: {r['clean_sample_accuracy']:.4f}")
        print(f"  Attack Sample Accuracy: {r['attack_sample_accuracy']:.4f}")
        print(f"  Clean Target Recall: {r['clean_target_recall']:.4f}")
        print(f"  Attack Target Recall: {r['attack_target_recall']:.4f}")
        print(f"  Clean mAP: {r['clean_mAP']:.4f}")
        print(f"  Attack mAP: {r['attack_mAP']:.4f}")
        print(
            f"  Percent Metrics: clean_acc={format_pct(r['clean_sample_accuracy'])}, "
            f"attack_acc={format_pct(r['attack_sample_accuracy'])}, "
            f"clean_mAP={format_pct(r['clean_mAP'])}, "
            f"attack_mAP={format_pct(r['attack_mAP'])}"
        )
        print(
            f"  Attack Success Rate: {100.0 * r['attack_success_rate']:.2f}% "
            f"({r['attack_success_count']}/{r['clean_correct_count']} clean-correct flipped)"
        )
        print(f"  Average Perturbation Distance: {r['avg_perturbation_distance']:.6f}")
        print(f"  Average Distortion: {r['avg_distortion']:.6f}")
        print(f"  Structural Similarity (SSIM): {r['avg_ssim']:.6f}")
        print(f"  Peak Signal-to-Noise Ratio (PSNR): {r['avg_psnr']:.6f} dB")
        print(
            f"  Attacked Samples: {r['attacked_sample_count']}/{r['total_samples']} "
            f"({100.0 * r['attacked_sample_rate']:.2f}%)"
        )
        print(f"  Restart Success Mean: {100.0 * r['restart_success_rate_mean']:.2f}%")
        print(f"  Restart Success Std: {100.0 * r['restart_success_rate_std']:.2f}%")
        print(f"  Mean Rel Change: {r['mean_rel_change']:.4f}")
        print(f"  Max Rel Change: {r['max_rel_change']:.4f}")


__all__ = [
    "AttackResult",
    "cw_l2_attack",
    "deepfool_attack",
    "fgsm_attack",
    "pgd_linf_attack",
]


if __name__ == "__main__":
    main()

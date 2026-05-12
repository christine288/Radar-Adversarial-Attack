"""Black-box adversarial attacks and evaluation CLI for radar track classification.

Implements three black-box attack strategies:
  - Square Attack  : score-based, L-inf, highly query-efficient (推荐首选)
  - NES-PGD        : gradient-estimation via Natural Evolution Strategies
  - Transfer Attack : surrogate-model white-box → target-model transfer

评估指标 (≥6 类):
  1. 攻击成功率 (Attack Success Rate / ISR)
  2. 对抗估计平均回合步数 (Avg Query Rounds per Sample)
  3. 攻击后样本准确率 (Attack Sample Accuracy)
  4. 攻击后 mAP
  5. 攻击后 Target Recall (macro)
  6. 平均相对扰动幅度 (Mean Relative Change)
  7. 平均扰动距离 (Avg Perturbation Distance / L2)
  8. 结构相似度 SSIM
  9. 峰值信噪比 PSNR
  10. Restart 成功率均值 / 标准差
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, TensorDataset

from data_utils import TrackSample, batch_tracks_to_sequences
from device_utils import print_device_banner, resolve_device_str
from metrics_utils import (
    compute_multiclass_map,
    compute_signal_quality_metrics,
    format_pct,
)
from model import RadarTrackTransformer
from noise_perturbation import (
    clamp_if_needed,
    project_relative_change,
    relative_change_stats,
)

# ---------------------------------------------------------------------------
# 通用数据结构
# ---------------------------------------------------------------------------


@dataclass
class BlackboxResult:
    """黑盒攻击产出的对抗样本及诊断信息。"""

    x_adv: torch.Tensor
    y_pred: torch.Tensor
    logits_or_probs: torch.Tensor          # 视攻击方法可能是 logits 或 softmax 概率
    mean_rel_change: float
    max_rel_change: float
    queries_per_sample: torch.Tensor       # shape (B,), 每个样本实际用掉的查询轮次


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _relative_budget_tensor(
    x_ref: torch.Tensor,
    max_rel_change: float,
    budget_floor: float,
) -> torch.Tensor:
    """返回与 x_ref 同形状的逐元素 L-inf 预算张量。"""
    scale = torch.maximum(x_ref.abs(), torch.full_like(x_ref, max(budget_floor, 1e-9)))
    return max_rel_change * scale


def _finalize(
    model: torch.nn.Module,
    x_ref: torch.Tensor,
    x_adv: torch.Tensor,
    queries: torch.Tensor,
    needs_forward: bool = True,
) -> BlackboxResult:
    with torch.no_grad():
        if needs_forward:
            out = model(x_adv)
        else:
            out = model(x_adv)
        probs = torch.softmax(out, dim=1)
        y_pred = out.argmax(dim=1)
        stats = relative_change_stats(x_ref, x_adv)
    return BlackboxResult(
        x_adv=x_adv.detach(),
        y_pred=y_pred.detach(),
        logits_or_probs=probs.detach(),
        mean_rel_change=stats["mean_rel_change"],
        max_rel_change=stats["max_rel_change"],
        queries_per_sample=queries.detach().cpu(),
    )


def _score_fn(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    targeted: bool,
    y_target: Optional[torch.Tensor],
) -> torch.Tensor:
    """返回 margin score（越大越好，攻击目标是让其 < 0）。"""
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    B = x.size(0)
    if targeted:
        assert y_target is not None
        sel = probs[torch.arange(B), y_target]
        masked = probs.clone()
        masked[torch.arange(B), y_target] = -1.0
        best_other = masked.max(dim=1).values
        return sel - best_other          # targeted: 希望 < 0 → 攻击让 sel 最大
    else:
        sel = probs[torch.arange(B), y]
        masked = probs.clone()
        masked[torch.arange(B), y] = -1.0
        best_other = masked.max(dim=1).values
        return sel - best_other          # untargeted: 希望 < 0 → 攻击让 sel 最小


# ---------------------------------------------------------------------------
# 1. Square Attack (L-inf, score-based)
#    参考: Andriushchenko et al., "Square Attack", ECCV 2020
# ---------------------------------------------------------------------------


def _square_attack_init(
    x_ref: torch.Tensor,
    budget: torch.Tensor,
    window_h: int,
) -> torch.Tensor:
    """初始化：在随机水平条纹上施加 ±budget 扰动。"""
    B, T, C = x_ref.shape
    delta = torch.zeros_like(x_ref)
    for b in range(B):
        t0 = np.random.randint(0, max(1, T - window_h + 1))
        t1 = min(t0 + window_h, T)
        sign = torch.randint(0, 2, (1, C), device=x_ref.device).float() * 2 - 1
        delta[b, t0:t1, :] = sign * budget[b, t0:t1, :]
    return delta


def square_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    max_queries: int = 5000,
    max_rel_change: float = 0.05,
    budget_floor: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
    p_init: float = 0.8,
) -> BlackboxResult:
    """
    Square Attack — L-inf 黑盒攻击。

    每步在随机时序窗口内以 ±budget 翻转扰动符号，
    只接受使 margin score 下降（攻击成功方向）的更新。

    Args:
        p_init: 初始窗口比例（相对序列长度），随迭代衰减。
    """
    model.eval()
    x_ref = x.detach()
    B, T, C = x_ref.shape
    budget = _relative_budget_tensor(x_ref, max_rel_change, budget_floor)

    # 初始化
    window_h = max(1, int(round(p_init * T)))
    delta = _square_attack_init(x_ref, budget, window_h)
    x_adv = clamp_if_needed(x_ref + delta, clamp)
    x_adv = project_relative_change(x_ref, x_adv, max_rel_change=max_rel_change, min_scale=budget_floor)

    score = _score_fn(model, x_adv, y_true, targeted=targeted, y_target=y_target)
    queries = torch.ones(B, dtype=torch.float32)          # init = 1 次查询
    done = torch.zeros(B, dtype=torch.bool)

    # p 衰减表（模拟论文中的 p schedule）
    def _p_schedule(q: int) -> float:
        """p 随查询次数衰减，参考 Square Attack 原论文 Table 1."""
        ratio = q / max_queries
        if ratio < 0.001:
            return p_init
        elif ratio < 0.01:
            return p_init * 0.5
        elif ratio < 0.1:
            return p_init * 0.2
        elif ratio < 0.5:
            return p_init * 0.1
        else:
            return max(p_init * 0.05, 1.0 / T)

    global_q = 1
    _print_interval = max(1, max_queries // 20)   # 每 5% 打印一次
    while global_q < max_queries:
        p = _p_schedule(global_q)
        window_h = max(1, int(round(p * T)))

        for b in range(B):
            if done[b]:
                continue
            # 在随机窗口内提出新候选扰动
            t0 = np.random.randint(0, max(1, T - window_h + 1))
            t1 = min(t0 + window_h, T)

            delta_new = delta[b].clone()
            # 翻转窗口内符号，并从 budget 边界重新采样
            sign = torch.randint(0, 2, (1, C), device=x_ref.device).float() * 2 - 1
            delta_new[t0:t1, :] = sign * budget[b, t0:t1, :]

            x_new = clamp_if_needed(x_ref[b:b+1] + delta_new.unsqueeze(0), clamp)
            x_new = project_relative_change(
                x_ref[b:b+1], x_new,
                max_rel_change=max_rel_change, min_scale=budget_floor,
            )
            score_new = _score_fn(model, x_new, y_true[b:b+1],
                                  targeted=targeted,
                                  y_target=y_target[b:b+1] if y_target is not None else None)
            queries[b] += 1

            # 接受改进（margin 下降）
            if score_new.item() < score[b].item():
                delta[b] = delta_new
                x_adv[b] = x_new[0]
                score[b] = score_new[0]

            # 检查是否已攻击成功
            with torch.no_grad():
                pred = model(x_adv[b:b+1]).argmax(dim=1)
            if targeted:
                done[b] = pred.eq(y_target[b:b+1]).item()
            else:
                done[b] = pred.ne(y_true[b:b+1]).item()

        global_q += 1
        if global_q % _print_interval == 0 or done.all():
            n_done = int(done.sum().item())
            print(
                f"  [Square] query={global_q}/{max_queries}  "
                f"done={n_done}/{B}  "
                f"success_rate={100.0 * n_done / B:.1f}%",
                flush=True,
            )
        if done.all():
            break

    return _finalize(model, x_ref, x_adv, queries)


# ---------------------------------------------------------------------------
# 2. NES-PGD (Natural Evolution Strategies gradient estimation)
#    参考: Ilyas et al., "Black-box Adversarial Attacks with Limited Queries", ICML 2018
# ---------------------------------------------------------------------------


def nes_pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    nes_samples: int = 20,
    nes_sigma: float = 0.01,
    num_steps: int = 100,
    step_size: float = 0.3,
    max_rel_change: float = 0.05,
    budget_floor: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
    momentum: float = 0.9,
) -> BlackboxResult:
    """
    NES 梯度估计 + PGD 迭代。

    用正负对称采样（antithetic sampling）估计得分函数对输入的梯度：
        ĝ = (1 / (2 * n * σ)) * Σ_i [ score(x + σu_i) - score(x - σu_i) ] * u_i

    再按 sign(ĝ) 做 PGD 步更新，与白盒 PGD 保持同等投影逻辑。
    """
    model.eval()
    x_ref = x.detach()
    B, T, C = x_ref.shape
    budget = _relative_budget_tensor(x_ref, max_rel_change, budget_floor)

    x_adv = x_ref.clone()
    queries = torch.zeros(B, dtype=torch.float32)
    grad_momentum = torch.zeros_like(x_ref)
    done = torch.zeros(B, dtype=torch.bool)

    for step in range(num_steps):
        # NES 梯度估计
        grad_est = torch.zeros_like(x_adv)
        n = int(nes_samples)
        for _ in range(n):
            u = torch.randn_like(x_adv)
            u_norm = u / (u.reshape(B, -1).norm(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-12))
            x_pos = x_adv + float(nes_sigma) * u_norm
            x_neg = x_adv - float(nes_sigma) * u_norm
            x_pos = project_relative_change(x_ref, x_pos, max_rel_change=max_rel_change, min_scale=budget_floor)
            x_neg = project_relative_change(x_ref, x_neg, max_rel_change=max_rel_change, min_scale=budget_floor)
            x_pos = clamp_if_needed(x_pos, clamp)
            x_neg = clamp_if_needed(x_neg, clamp)

            s_pos = _score_fn(model, x_pos, y_true, targeted=targeted, y_target=y_target)
            s_neg = _score_fn(model, x_neg, y_true, targeted=targeted, y_target=y_target)
            queries += 2.0

            diff = (s_pos - s_neg).reshape(B, 1, 1)    # (B,1,1)
            grad_est += diff * u_norm

        grad_est /= max(2 * n * float(nes_sigma), 1e-12)

        # 有动量的 sign 更新（untargeted: 沿梯度正方向增大 margin，即翻转标签）
        g_norm = grad_est.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        grad_momentum = float(momentum) * grad_momentum + grad_est / g_norm
        direction = grad_momentum.sign()
        if targeted:
            direction = -direction

        actual_step = float(step_size) * budget
        x_adv = x_adv + actual_step * direction
        x_adv = project_relative_change(x_ref, x_adv, max_rel_change=max_rel_change, min_scale=budget_floor)
        x_adv = clamp_if_needed(x_adv, clamp)

        # 早停
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        if targeted:
            done = preds.eq(y_target) if y_target is not None else done
        else:
            done = preds.ne(y_true)
        if done.all():
            break

    return _finalize(model, x_ref, x_adv, queries)


# ---------------------------------------------------------------------------
# 3. Transfer Attack (surrogate model → target model)
#    用与目标模型相同架构但独立初始化的替代模型，在替代模型上做 PGD，
#    再直接迁移到目标模型评估。
# ---------------------------------------------------------------------------


def _build_surrogate(
    input_size: Tuple[int, int],
    num_classes: int,
    device: str,
    *,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
) -> RadarTrackTransformer:
    """构造一个更小的替代 Transformer，与目标模型架构相同但参数独立。"""
    surrogate = RadarTrackTransformer(
        input_size=input_size,
        num_classes=num_classes,
        dropout=dropout,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )
    surrogate.to(device).eval()
    return surrogate


def _pgd_on_surrogate(
    surrogate: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    step_size: float,
    num_steps: int,
    max_rel_change: float,
    budget_floor: float,
    clamp: Optional[Tuple[float, float]],
    momentum: float,
    targeted: bool,
    y_target: Optional[torch.Tensor],
    num_restarts: int,
) -> torch.Tensor:
    """在替代模型上多重启 PGD，返回最强对抗样本。"""
    x_ref = x.detach()
    B, T, C = x_ref.shape
    budget = _relative_budget_tensor(x_ref, max_rel_change, budget_floor)

    best_x_adv = x_ref.clone()
    best_score = torch.full((B,), float("inf"), device=x.device)

    for _ in range(num_restarts):
        # 随机初始化
        scale = torch.maximum(x_ref.abs(), torch.full_like(x_ref, max(budget_floor, 1e-9)))
        delta = torch.empty_like(x_ref).uniform_(-1.0, 1.0) * (max_rel_change * scale)
        x_adv = clamp_if_needed(x_ref + delta, clamp)
        x_adv = project_relative_change(x_ref, x_adv, max_rel_change=max_rel_change, min_scale=budget_floor)
        grad_mom = torch.zeros_like(x_ref)

        for _ in range(num_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = surrogate(x_adv)
            targets = y_target if (targeted and y_target is not None) else y_true
            if targeted:
                loss = F.cross_entropy(logits, targets)
            else:
                loss = -F.cross_entropy(logits, targets)   # maximize loss → untargeted
            grad = torch.autograd.grad(loss, x_adv)[0]
            g_norm = grad.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            grad_mom = float(momentum) * grad_mom + grad / g_norm
            direction = grad_mom.sign()
            if targeted:
                direction = -direction
            actual_step = float(step_size) * budget
            x_adv = x_adv.detach() + actual_step * direction
            x_adv = project_relative_change(x_ref, x_adv, max_rel_change=max_rel_change, min_scale=budget_floor)
            x_adv = clamp_if_needed(x_adv, clamp)

        with torch.no_grad():
            surrogate_logits = surrogate(x_adv)
            l2_sq = (x_adv - x_ref).reshape(B, -1).pow(2).sum(dim=1)
            improved = l2_sq < best_score
            if improved.any():
                best_x_adv[improved] = x_adv.detach()[improved]
                best_score[improved] = l2_sq[improved]

    return best_x_adv


def transfer_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    surrogate: Optional[torch.nn.Module] = None,
    pgd_steps: int = 200,
    pgd_step_size: float = 0.3,
    num_restarts: int = 5,
    max_rel_change: float = 0.05,
    budget_floor: float = 0.0,
    clamp: Optional[Tuple[float, float]] = None,
    momentum: float = 0.9,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> BlackboxResult:
    """
    迁移攻击：在替代模型（surrogate）上生成对抗样本，直接迁移到目标 model 评估。
    若未提供 surrogate，自动构造一个随机初始化的小型 Transformer。
    查询数 = 0（目标模型完全不查询）。
    """
    model.eval()
    x_ref = x.detach()
    B, T, C = x_ref.shape

    if surrogate is None:
        num_classes = int(model.head[-1].out_features)
        surrogate = _build_surrogate(
            input_size=(T, C),
            num_classes=num_classes,
            device=device,
        )

    x_adv = _pgd_on_surrogate(
        surrogate, x_ref, y_true,
        step_size=pgd_step_size,
        num_steps=pgd_steps,
        max_rel_change=max_rel_change,
        budget_floor=budget_floor,
        clamp=clamp,
        momentum=momentum,
        targeted=targeted,
        y_target=y_target,
        num_restarts=num_restarts,
    )
    queries = torch.zeros(B, dtype=torch.float32)   # 目标模型查询次数 = 0
    return _finalize(model, x_ref, x_adv, queries)


# ---------------------------------------------------------------------------
# 评估主循环
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: List[List[float]],
    num_classes: int,
) -> Dict[str, float]:
    acc = float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)) if y_true else 0.0
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    mAP = compute_multiclass_map(y_true, y_score, num_classes=num_classes)
    return {"accuracy": acc, "recall": recall, "mAP": mAP}


def _compute_isr(
    y_true: List[int],
    y_pred_clean: List[int],
    y_pred_attack: List[int],
) -> float:
    """Interference Success Rate：在干净预测正确的样本中，攻击翻转的比例。"""
    correct_idx = [i for i, (yt, pc) in enumerate(zip(y_true, y_pred_clean)) if yt == pc]
    if not correct_idx:
        return 0.0
    flipped = sum(1 for i in correct_idx if y_pred_attack[i] != y_true[i])
    return flipped / len(correct_idx)


def evaluate_blackbox_attack(
    *,
    model: RadarTrackTransformer,
    loader: DataLoader,
    device: str,
    attack: str,
    # Square Attack 参数
    sq_max_queries: int,
    sq_p_init: float,
    # NES 参数
    nes_samples: int,
    nes_sigma: float,
    nes_steps: int,
    nes_step_size: float,
    nes_momentum: float,
    # Transfer 参数
    tr_pgd_steps: int,
    tr_pgd_step_size: float,
    tr_restarts: int,
    tr_momentum: float,
    # 通用
    max_rel_change: float,
    budget_floor: float,
    clamp: Optional[Tuple[float, float]],
    attack_clean_only: bool,
    targeted_topk: int,
    attack_restarts: int,          # Square/NES 重启次数
    surrogate: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:

    y_true_all: List[int] = []
    y_pred_clean_all: List[int] = []
    y_pred_attack_all: List[int] = []
    y_score_clean_all: List[List[float]] = []
    y_score_attack_all: List[List[float]] = []

    clean_correct_total = 0
    attack_success_total = 0
    attacked_sample_total = 0
    total_samples = 0

    queries_sum = 0.0
    queries_attacked_count = 0
    queries_list: List[float] = []    # 用于计算标准差

    perturb_dist_sum = 0.0
    distortion_sum = 0.0
    ssim_sum = 0.0
    psnr_sum = 0.0
    rel_mean_wsum = 0.0
    rel_max_overall = 0.0

    restart_success_rates: List[float] = []

    num_classes = int(model.head[-1].out_features)
    n_batches = len(loader)
    print(f"\n[{attack.upper()}] 开始评估，共 {n_batches} 个 batch", flush=True)

    for batch_idx, (x, y) in enumerate(loader, 1):
        print(
            f"\n[{attack.upper()}] Batch {batch_idx}/{n_batches}  "
            f"(累计攻击样本={attacked_sample_total}, "
            f"累计成功={attack_success_total})",
            flush=True,
        )
        x = x.to(device)
        y = y.to(device)
        B = x.size(0)
        total_samples += B

        with torch.no_grad():
            clean_logits = model(x)
            pred_clean = clean_logits.argmax(dim=1)
            clean_probs = torch.softmax(clean_logits, dim=1)

        attack_mask = pred_clean.eq(y) if attack_clean_only else torch.ones(B, dtype=torch.bool, device=device)
        y_pred_attack_batch = pred_clean.clone()
        attack_probs_batch = clean_probs.clone()

        if attack_mask.any():
            x_atk = x[attack_mask]
            y_atk = y[attack_mask]
            n_atk = x_atk.size(0)
            attacked_sample_total += n_atk

            # 候选目标类（用于 targeted 模式）
            clean_logits_atk = clean_logits[attack_mask]
            k = max(1, min(int(targeted_topk), max(1, num_classes - 1)))
            masked_for_topk = clean_logits_atk.clone()
            masked_for_topk.scatter_(1, y_atk.view(-1, 1), float("-inf"))
            target_candidates = masked_for_topk.topk(k=k, dim=1).indices   # (n_atk, k)

            best_res: Optional[BlackboxResult] = None

            n_restarts = 1 if attack == "transfer" else max(1, int(attack_restarts))

            for restart_idx in range(n_restarts):
                # --- untargeted ---
                if attack == "square":
                    res_u = square_attack(
                        model, x_atk, y_atk,
                        max_queries=sq_max_queries,
                        max_rel_change=max_rel_change,
                        budget_floor=budget_floor,
                        clamp=clamp,
                        targeted=False,
                        p_init=sq_p_init,
                    )
                elif attack == "nes":
                    res_u = nes_pgd_attack(
                        model, x_atk, y_atk,
                        nes_samples=nes_samples,
                        nes_sigma=nes_sigma,
                        num_steps=nes_steps,
                        step_size=nes_step_size,
                        max_rel_change=max_rel_change,
                        budget_floor=budget_floor,
                        clamp=clamp,
                        targeted=False,
                        momentum=nes_momentum,
                    )
                elif attack == "transfer":
                    res_u = transfer_attack(
                        model, x_atk, y_atk,
                        surrogate=surrogate,
                        pgd_steps=tr_pgd_steps,
                        pgd_step_size=tr_pgd_step_size,
                        num_restarts=tr_restarts,
                        max_rel_change=max_rel_change,
                        budget_floor=budget_floor,
                        clamp=clamp,
                        momentum=tr_momentum,
                        targeted=False,
                        device=device,
                    )
                else:
                    raise ValueError(f"unknown attack: {attack}")

                restart_batch_best = res_u
                # --- targeted（多目标类）---
                for ti in range(k):
                    y_tgt = target_candidates[:, ti]
                    if attack == "square":
                        res_t = square_attack(
                            model, x_atk, y_atk,
                            max_queries=sq_max_queries,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            targeted=True,
                            y_target=y_tgt,
                            p_init=sq_p_init,
                        )
                    elif attack == "nes":
                        res_t = nes_pgd_attack(
                            model, x_atk, y_atk,
                            nes_samples=nes_samples,
                            nes_sigma=nes_sigma,
                            num_steps=nes_steps,
                            step_size=nes_step_size,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            targeted=True,
                            y_target=y_tgt,
                            momentum=nes_momentum,
                        )
                    elif attack == "transfer":
                        res_t = transfer_attack(
                            model, x_atk, y_atk,
                            surrogate=surrogate,
                            pgd_steps=tr_pgd_steps,
                            pgd_step_size=tr_pgd_step_size,
                            num_restarts=tr_restarts,
                            max_rel_change=max_rel_change,
                            budget_floor=budget_floor,
                            clamp=clamp,
                            momentum=tr_momentum,
                            targeted=True,
                            y_target=y_tgt,
                            device=device,
                        )
                    else:
                        raise ValueError(f"unknown attack: {attack}")

                    # 合并：翻转优先，再取总查询数较少者
                    cur_flip = restart_batch_best.y_pred.to(device).ne(y_atk)
                    new_flip = res_t.y_pred.to(device).ne(y_atk)
                    take = (~cur_flip) & new_flip
                    if take.any():
                        take_dev = take.to(device)
                        x_merged = restart_batch_best.x_adv.clone()
                        x_merged[take_dev] = res_t.x_adv[take_dev]
                        lp_merged = restart_batch_best.logits_or_probs.clone()
                        lp_merged[take_dev] = res_t.logits_or_probs[take_dev]
                        yp_merged = restart_batch_best.y_pred.clone().cpu()
                        yp_merged[take.cpu()] = res_t.y_pred.cpu()[take.cpu()]
                        q_merged = restart_batch_best.queries_per_sample + res_t.queries_per_sample
                        stats = relative_change_stats(x_atk, x_merged)
                        restart_batch_best = BlackboxResult(
                            x_adv=x_merged,
                            y_pred=yp_merged,
                            logits_or_probs=lp_merged,
                            mean_rel_change=stats["mean_rel_change"],
                            max_rel_change=stats["max_rel_change"],
                            queries_per_sample=q_merged,
                        )

                restart_success_rates.append(
                    float(restart_batch_best.y_pred.to(device).ne(y_atk).float().mean().item())
                )
                # 跨 restart 合并
                if best_res is None:
                    best_res = restart_batch_best
                else:
                    cur_flip = best_res.y_pred.to(device).ne(y_atk)
                    new_flip = restart_batch_best.y_pred.to(device).ne(y_atk)
                    take = (~cur_flip) & new_flip
                    if take.any():
                        take_dev = take.to(device)
                        x_m = best_res.x_adv.clone()
                        x_m[take_dev] = restart_batch_best.x_adv[take_dev]
                        lp_m = best_res.logits_or_probs.clone()
                        lp_m[take_dev] = restart_batch_best.logits_or_probs[take_dev]
                        yp_m = best_res.y_pred.clone().cpu()
                        yp_m[take.cpu()] = restart_batch_best.y_pred.cpu()[take.cpu()]
                        q_m = best_res.queries_per_sample + restart_batch_best.queries_per_sample
                        stats = relative_change_stats(x_atk, x_m)
                        best_res = BlackboxResult(
                            x_adv=x_m,
                            y_pred=yp_m,
                            logits_or_probs=lp_m,
                            mean_rel_change=stats["mean_rel_change"],
                            max_rel_change=stats["max_rel_change"],
                            queries_per_sample=q_m,
                        )

            assert best_res is not None

            y_pred_attack_batch[attack_mask] = best_res.y_pred.to(device)
            attack_probs_batch[attack_mask] = best_res.logits_or_probs.to(device)

            # 统计查询轮次
            q = best_res.queries_per_sample.float()
            queries_sum += q.sum().item()
            queries_attacked_count += n_atk
            queries_list.extend(q.tolist())

            # 信号质量
            quality = compute_signal_quality_metrics(x_atk, best_res.x_adv)
            perturb_dist_sum += quality["avg_perturbation_distance"] * n_atk
            distortion_sum += quality["avg_distortion"] * n_atk
            ssim_sum += quality["avg_ssim"] * n_atk
            psnr_sum += quality["avg_psnr"] * n_atk
            rel_mean_wsum += float(best_res.mean_rel_change) * n_atk
            rel_max_overall = max(rel_max_overall, float(best_res.max_rel_change))

        # 累积全局统计
        clean_correct_mask = pred_clean.eq(y)
        attack_success_mask = clean_correct_mask & y_pred_attack_batch.ne(y)
        clean_correct_total += int(clean_correct_mask.sum().item())
        attack_success_total += int(attack_success_mask.sum().item())

        y_true_all.extend(y.cpu().tolist())
        y_pred_clean_all.extend(pred_clean.cpu().tolist())
        y_pred_attack_all.extend(y_pred_attack_batch.cpu().tolist())
        y_score_clean_all.extend(clean_probs.cpu().tolist())
        y_score_attack_all.extend(attack_probs_batch.cpu().tolist())

    clean_m = _compute_metrics(y_true_all, y_pred_clean_all, y_score_clean_all, num_classes)
    attack_m = _compute_metrics(y_true_all, y_pred_attack_all, y_score_attack_all, num_classes)
    isr = _compute_isr(y_true_all, y_pred_clean_all, y_pred_attack_all)

    avg_q = queries_sum / max(1, queries_attacked_count)
    std_q = float(np.std(queries_list)) if queries_list else 0.0

    rst_mean = float(np.mean(restart_success_rates)) if restart_success_rates else 0.0
    rst_std = float(np.std(restart_success_rates)) if restart_success_rates else 0.0

    return {
        # ① 攻击成功率
        "attack_success_rate": isr,
        "attack_success_count": attack_success_total,
        "clean_correct_count": clean_correct_total,
        # ② 平均查询轮次
        "avg_query_rounds": avg_q,
        "std_query_rounds": std_q,
        # ③ 攻击后准确率
        "attack_sample_accuracy": attack_m["accuracy"],
        "clean_sample_accuracy": clean_m["accuracy"],
        # ④ 攻击后 mAP
        "attack_mAP": attack_m["mAP"],
        "clean_mAP": clean_m["mAP"],
        # ⑤ 攻击后 Target Recall
        "attack_target_recall": attack_m["recall"],
        "clean_target_recall": clean_m["recall"],
        # ⑥ 平均相对扰动幅度
        "mean_rel_change": float(rel_mean_wsum / max(1, attacked_sample_total)),
        "max_rel_change": rel_max_overall,
        # ⑦ 平均扰动距离 L2
        "avg_perturbation_distance": float(perturb_dist_sum / max(1, attacked_sample_total)),
        # ⑧ SSIM
        "avg_ssim": float(ssim_sum / max(1, attacked_sample_total)),
        # ⑨ PSNR
        "avg_psnr": float(psnr_sum / max(1, attacked_sample_total)),
        # ⑩ Restart 成功率
        "restart_success_rate_mean": rst_mean,
        "restart_success_rate_std": rst_std,
        # 其他
        "avg_distortion": float(distortion_sum / max(1, attacked_sample_total)),
        "attacked_sample_count": attacked_sample_total,
        "attacked_sample_rate": float(attacked_sample_total / max(1, total_samples)),
        "total_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# 模型加载 / DataLoader
# ---------------------------------------------------------------------------


def load_model(model_path: str, device: str) -> RadarTrackTransformer:
    try:
        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(model_path, map_location=device)
    except PermissionError as e:
        raise PermissionError(
            f"无法读取模型文件（权限拒绝）: {model_path}\n"
            "请确认路径是权重 .pth 文件本身（不要填目录）。"
        ) from e
    model = RadarTrackTransformer(
        input_size=ckpt["input_size"],
        num_classes=ckpt.get("num_classes", 6),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def build_loader(samples: List[TrackSample], batch_size: int) -> DataLoader:
    x, y = batch_tracks_to_sequences(samples)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_result(name: str, r: Dict[str, float]) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Attack: {name.upper()}")
    print(sep)
    print(f"  ① 攻击成功率 (ISR)          : {100.0 * r['attack_success_rate']:.2f}%"
          f"  ({r['attack_success_count']}/{r['clean_correct_count']} clean-correct flipped)")
    print(f"  ② 平均查询轮次              : {r['avg_query_rounds']:.1f}  "
          f"(std={r['std_query_rounds']:.1f})")
    print(f"  ③ 干净准确率 / 攻击后准确率 : {format_pct(r['clean_sample_accuracy'])} / "
          f"{format_pct(r['attack_sample_accuracy'])}")
    print(f"  ④ 干净 mAP   / 攻击后 mAP  : {format_pct(r['clean_mAP'])} / "
          f"{format_pct(r['attack_mAP'])}")
    print(f"  ⑤ 干净 Recall/ 攻击后Recall: {format_pct(r['clean_target_recall'])} / "
          f"{format_pct(r['attack_target_recall'])}")
    print(f"  ⑥ 平均相对扰动幅度          : {r['mean_rel_change']:.4f}  "
          f"(max={r['max_rel_change']:.4f})")
    print(f"  ⑦ 平均扰动距离 (L2)         : {r['avg_perturbation_distance']:.6f}")
    print(f"  ⑧ 结构相似度 (SSIM)         : {r['avg_ssim']:.6f}")
    print(f"  ⑨ 峰值信噪比 (PSNR)         : {r['avg_psnr']:.4f} dB")
    print(f"  ⑩ Restart成功率 mean/std    : {100.0 * r['restart_success_rate_mean']:.2f}% / "
          f"{100.0 * r['restart_success_rate_std']:.2f}%")
    print(f"     平均失真度 (Distortion)   : {r['avg_distortion']:.6f}")
    print(f"     已攻击样本数              : {r['attacked_sample_count']}/{r['total_samples']}"
          f"  ({100.0 * r['attacked_sample_rate']:.1f}%)")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Black-box adversarial evaluation (Square / NES-PGD / Transfer)"
    )
    # 通用
    parser.add_argument("--model_path", type=str, default="radar_transformer.pth")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="黑盒攻击计算量大，建议 32~128")
    parser.add_argument("--device", type=str, default="cuda",
                        help="默认 cuda；无 CUDA 时回退 CPU")
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--mat_test_dir", type=str, default=None)
    parser.add_argument("--mat_dir", type=str, default=None,
                        help="同 --mat_test_dir（兼容旧参数）")
    parser.add_argument("--attack", type=str, default="all",
                        choices=["all", "square", "nes", "transfer"])
    parser.add_argument("--max_rel_change", type=float, default=0.20,
                        help="相对 L∞ 预算（建议黑盒使用 0.20）")
    parser.add_argument("--no_rel_budget", action="store_true",
                        help="不做相对 L∞ 投影（无约束上界测试）")
    parser.add_argument("--budget_floor", type=float, default=0.01)
    parser.add_argument("--clamp_min", type=float, default=None)
    parser.add_argument("--clamp_max", type=float, default=None)
    parser.add_argument("--attack_clean_only", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--targeted_topk", type=int, default=5,
                        help="尝试 top-k 目标类并保留最强结果")
    parser.add_argument("--attack_restarts", type=int, default=3,
                        help="Square/NES 重启次数（Transfer 固定=1）")
    # Square Attack
    parser.add_argument("--sq_max_queries", type=int, default=5000)
    parser.add_argument("--sq_p_init", type=float, default=0.8,
                        help="Square Attack 初始窗口比例（0~1 相对序列长度）")
    # NES
    parser.add_argument("--nes_samples", type=int, default=20,
                        help="每步 NES 采样对数（查询数 = 2 * nes_samples * nes_steps）")
    parser.add_argument("--nes_sigma", type=float, default=0.01)
    parser.add_argument("--nes_steps", type=int, default=100)
    parser.add_argument("--nes_step_size", type=float, default=0.3)
    parser.add_argument("--nes_momentum", type=float, default=0.9)
    # Transfer
    parser.add_argument("--tr_pgd_steps", type=int, default=200)
    parser.add_argument("--tr_pgd_step_size", type=float, default=0.3)
    parser.add_argument("--tr_restarts", type=int, default=5,
                        help="替代模型 PGD 重启次数")
    parser.add_argument("--tr_momentum", type=float, default=0.9)

    args = parser.parse_args()

    # 路径校验
    for name, val in [("--model_path", args.model_path),
                      ("--mat_test_dir", args.mat_test_dir),
                      ("--mat_dir", args.mat_dir)]:
        if val is None:
            continue
        s = str(val).strip()
        if s in ("...", "..", ".") or (len(s) >= 3 and all(c == "." for c in s)):
            parser.error(f"{name} 不能为占位符，请填写真实路径。")

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    mat_root = args.mat_test_dir or args.mat_dir
    if not mat_root:
        parser.error("必须指定 --mat_test_dir 或 --mat_dir")

    clamp = None
    if args.clamp_min is not None or args.clamp_max is not None:
        if args.clamp_min is None or args.clamp_max is None:
            raise ValueError("clamp_min 和 clamp_max 必须同时提供")
        clamp = (float(args.clamp_min), float(args.clamp_max))

    device = resolve_device_str(args.device, require_gpu=args.require_gpu)
    print_device_banner(device)

    model = load_model(args.model_path, device=device)

    meta_path = str(args.model_path) + ".meta.json"
    tgt_pts = 32
    if Path(meta_path).is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tgt_pts = int(meta.get("mat_target_points", 32))

    from mat_loader import load_mat_directory
    test_samples, _ = load_mat_directory(mat_root, target_points=tgt_pts)
    loader = build_loader(test_samples, batch_size=args.batch_size)

    max_rel = None if args.no_rel_budget else args.max_rel_change
    attacks = ["square", "nes", "transfer"] if args.attack == "all" else [args.attack]

    # 预先构造 surrogate（transfer 专用，避免每 batch 重建）
    surrogate = None
    if "transfer" in attacks:
        x0, _ = next(iter(loader))
        _, T, C = x0.shape
        num_classes = int(model.head[-1].out_features)
        surrogate = _build_surrogate(
            input_size=(T, C),
            num_classes=num_classes,
            device=device,
        )
        print(f"[transfer] 随机替代模型已构造 (T={T}, C={C}, classes={num_classes})")

    n_total = len(test_samples)
    n_batches = (n_total + args.batch_size - 1) // args.batch_size
    print("\n" + "=" * 60, flush=True)
    print("  黑盒对抗攻击评估（PyTorch）", flush=True)
    print("=" * 60, flush=True)
    print(f"  模型        : {args.model_path}", flush=True)
    print(f"  测试目录    : {mat_root}", flush=True)
    print(f"  测试样本数  : {n_total}  batch_size={args.batch_size}  共 {n_batches} 个 batch", flush=True)
    print(f"  攻击方法    : {attacks}", flush=True)
    print(f"  max_rel_change={max_rel}  budget_floor={args.budget_floor}", flush=True)
    print(f"  sq_max_queries={args.sq_max_queries}  restarts={args.attack_restarts}  targeted_topk={args.targeted_topk}", flush=True)
    print("=" * 60 + "\n", flush=True)

    for name in attacks:
        r = evaluate_blackbox_attack(
            model=model,
            loader=loader,
            device=device,
            attack=name,
            sq_max_queries=args.sq_max_queries,
            sq_p_init=args.sq_p_init,
            nes_samples=args.nes_samples,
            nes_sigma=args.nes_sigma,
            nes_steps=args.nes_steps,
            nes_step_size=args.nes_step_size,
            nes_momentum=args.nes_momentum,
            tr_pgd_steps=args.tr_pgd_steps,
            tr_pgd_step_size=args.tr_pgd_step_size,
            tr_restarts=args.tr_restarts,
            tr_momentum=args.tr_momentum,
            max_rel_change=max_rel if max_rel is not None else 0.20,
            budget_floor=args.budget_floor,
            clamp=clamp,
            attack_clean_only=args.attack_clean_only,
            targeted_topk=args.targeted_topk,
            attack_restarts=args.attack_restarts,
            surrogate=surrogate if name == "transfer" else None,
        )
        _print_result(name, r)


if __name__ == "__main__":
    main()
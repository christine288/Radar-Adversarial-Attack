from __future__ import annotations

import contextlib
import io
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import torch

from data_utils import TRACK_CHANNEL_KEYS, TrackSample
from dataset_service import prepare_dataset
from mat_loader import load_mat_track
from model_service import _resolve_model_file, validate_model
from robust_evaluate import build_loader as robust_build_loader
from noise_perturbation import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise, relative_change_stats
from whitebox_attacks import (
    _candidate_target_classes,
    _select_stronger_attack_result,
    cw_l2_attack,
    fgsm_attack,
    load_model as load_whitebox_model,
    pgd_linf_attack,
)


WHITEBOX_ATTACKS = {"fgsm", "pgd", "cw"}
BLACKBOX_ATTACKS = {"square", "nes", "transfer"}
NOISE_ATTACKS = {"gaussian", "salt_pepper", "speckle"}
SUPPORTED_ATTACKS = WHITEBOX_ATTACKS | BLACKBOX_ATTACKS | NOISE_ATTACKS


def _load_blackbox_attacks() -> tuple[Any, Any, Any, Any]:
    from blackbox_attacks import BlackboxResult, nes_pgd_attack, square_attack, transfer_attack

    return BlackboxResult, nes_pgd_attack, square_attack, transfer_attack

WHITEBOX_DEFAULTS = {
    "fgsm_step_size": 1.0,
    "pgd_step_size": 1.0,
    "pgd_steps": 300,
    "pgd_momentum": 0.9,
    "attack_restarts": 10,
    "attack_loss": "ce",
    "attack_clean_only": True,
    "targeted_topk": 5,
    "max_rel_change": 0.05,
    "budget_floor": 0.0,
    "cw_c": 50.0,
    "cw_lr": 0.005,
    "cw_steps": 1000,
    "cw_confidence": 1.0,
}

NOISE_DEFAULTS = {
    "max_rel_change": 0.05,
    "worst_of_k": 16,
    "sigma": 0.8,
    "sp_prob": 0.3,
    "speckle_sigma": 0.8,
}

BLACKBOX_DEFAULTS = {
    "max_rel_change": 0.05,
    "budget_floor": 0.01,
    "attack_clean_only": True,
    "square_max_queries": 2000,
    "square_p_init": 0.8,
    "square_attack_restarts": 1,
    "square_targeted_topk": 1,
    "nes_samples": 20,
    "nes_sigma": 0.01,
    "nes_steps": 100,
    "nes_step_size": 0.3,
    "nes_momentum": 0.9,
    "nes_attack_restarts": 1,
    "nes_targeted_topk": 1,
    "tr_pgd_steps": 200,
    "tr_pgd_step_size": 0.3,
    "tr_restarts": 5,
    "tr_momentum": 0.9,
    "transfer_targeted_topk": 5,
}

INTERFACE_EVAL_RESULT_FILE = "_interface_eval_result.json"


def _normalise_attack_method(attack_method: str) -> str:
    method = str(attack_method).strip().lower()
    if method not in SUPPORTED_ATTACKS:
        raise RuntimeError(
            "Unsupported attack_method. Supported methods: "
            + ", ".join(sorted(SUPPORTED_ATTACKS))
        )
    return method


def _run_silently(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def _is_mat_dataset(dataset_path: Path) -> bool:
    return prepare_dataset(str(dataset_path))


def _compute_track_perturbation_metrics(clean_tracks: List[np.ndarray], adv_tracks: List[np.ndarray]) -> Dict:
    if not clean_tracks or not adv_tracks:
        return {
            "mean_L2": 0.0,
            "max_L2": 0.0,
            "mad": 0.0,
            "mean_abs_delta": 0.0,
            "mean_rel_change": 0.0,
            "max_rel_change": 0.0,
        }

    clean = np.concatenate([np.asarray(x, dtype=np.float64) for x in clean_tracks], axis=0)
    adv = np.concatenate([np.asarray(x, dtype=np.float64) for x in adv_tracks], axis=0)
    if clean.shape != adv.shape or clean.ndim != 2 or clean.shape[1] != len(TRACK_CHANNEL_KEYS):
        raise RuntimeError("Clean and adversarial tracks must align as N x 12 arrays")

    delta = adv - clean
    abs_delta = np.abs(delta)
    l2_per_step = np.linalg.norm(delta, axis=1)
    mean_l2 = float(np.mean(l2_per_step)) if l2_per_step.size else 0.0

    rel_scale = np.maximum(np.abs(clean), 0.01)
    rel_change = abs_delta / rel_scale
    return {
        "mean_L2": mean_l2,
        "max_L2": float(np.max(l2_per_step)) if l2_per_step.size else 0.0,
        "mad": float(np.mean(np.abs(l2_per_step - mean_l2))) if l2_per_step.size else 0.0,
        "mean_abs_delta": float(np.mean(abs_delta)) if abs_delta.size else 0.0,
        "mean_rel_change": float(np.mean(rel_change)) if rel_change.size else 0.0,
        "max_rel_change": float(np.max(rel_change)) if rel_change.size else 0.0,
    }


def _compute_tensor_perturbation_metrics(x_clean: np.ndarray, x_adv: np.ndarray) -> Dict:
    clean = np.asarray(x_clean, dtype=np.float64)
    adv = np.asarray(x_adv, dtype=np.float64)
    if clean.shape != adv.shape:
        raise RuntimeError("Clean and adversarial tensors must align")

    delta = adv - clean
    flat_delta = delta.reshape(delta.shape[0], -1)
    l2_per_sample = np.linalg.norm(flat_delta, axis=1)
    mean_l2 = float(np.mean(l2_per_sample)) if l2_per_sample.size else 0.0
    scale = np.maximum(np.abs(clean), 0.01)
    rel_change = np.abs(delta) / scale
    return {
        "mean_L2": mean_l2,
        "max_L2": float(np.max(l2_per_sample)) if l2_per_sample.size else 0.0,
        "mad": float(np.mean(np.abs(l2_per_sample - mean_l2))) if l2_per_sample.size else 0.0,
        "mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "mean_rel_change": float(np.mean(rel_change)) if rel_change.size else 0.0,
        "max_rel_change": float(np.max(rel_change)) if rel_change.size else 0.0,
    }


def _attack_success_rate_from_preds(y_true: List[int], y_pred_clean: List[int], y_pred_adv: List[int]) -> float:
    clean_correct_idx = [i for i, (yt, pc) in enumerate(zip(y_true, y_pred_clean)) if yt == pc]
    if not clean_correct_idx:
        return 0.0
    success = sum(1 for i in clean_correct_idx if y_pred_adv[i] != y_true[i])
    return float(success / len(clean_correct_idx))


def _format_interface_eval_result(clean_metrics: Dict, adv_metrics: Dict, attack_success: float) -> Dict:
    clean_map = float(clean_metrics.get("mAP", 0.0))
    adv_map = float(adv_metrics.get("mAP", 0.0))
    clean_acc = float(clean_metrics.get("accuracy", 0.0))
    adv_acc = float(adv_metrics.get("accuracy", 0.0))
    clean_f1 = float(clean_metrics.get("f1_macro", 0.0))
    adv_f1 = float(adv_metrics.get("f1_macro", 0.0))
    retention = adv_map / clean_map if clean_map > 0 else 0.0
    return {
        "clean": {
            "accuracy": clean_acc,
            "macro_f1": clean_f1,
            "classification_mAP": clean_map,
        },
        "Adversarial": {
            "accuracy": adv_acc,
            "macro_f1": adv_f1,
            "classification_mAP": adv_map,
            "mAP_retention_rate": retention,
            "attack_success_rate": float(attack_success),
            "robust_accuracy": adv_acc,
        },
    }


def _evaluate_processed_attack_result(
    model: torch.nn.Module,
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    y_true: np.ndarray,
    *,
    device: str,
) -> Dict:
    from metrics_utils import compute_multiclass_map
    from sklearn.metrics import precision_recall_fscore_support

    def _predict(x_np: np.ndarray) -> tuple[List[int], List[List[float]]]:
        preds: List[int] = []
        scores: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, x_np.shape[0], 64):
                x_batch = torch.from_numpy(x_np[start:start + 64].astype(np.float32, copy=False)).to(device)
                logits = model(x_batch)
                probs = torch.softmax(logits, dim=1)
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                scores.extend(probs.cpu().tolist())
        return preds, scores

    y_list = np.asarray(y_true, dtype=np.int64).tolist()
    num_classes = int(model.head[-1].out_features)
    clean_pred, clean_score = _predict(x_clean)
    adv_pred, adv_score = _predict(x_adv)
    clean_metrics = {
        "accuracy": float(np.mean(np.asarray(clean_pred, dtype=np.int64) == np.asarray(y_list, dtype=np.int64))),
        "mAP": compute_multiclass_map(y_list, clean_score, num_classes=num_classes),
    }
    _, _, clean_f1, _ = precision_recall_fscore_support(
        y_list,
        clean_pred,
        average="macro",
        zero_division=0,
    )
    clean_metrics["f1_macro"] = float(clean_f1)
    adv_metrics = {
        "accuracy": float(np.mean(np.asarray(adv_pred, dtype=np.int64) == np.asarray(y_list, dtype=np.int64))),
        "mAP": compute_multiclass_map(y_list, adv_score, num_classes=num_classes),
    }
    _, _, adv_f1, _ = precision_recall_fscore_support(
        y_list,
        adv_pred,
        average="macro",
        zero_division=0,
    )
    adv_metrics["f1_macro"] = float(adv_f1)
    return _format_interface_eval_result(
        clean_metrics,
        adv_metrics,
        _attack_success_rate_from_preds(y_list, clean_pred, adv_pred),
    )


def _resolve_model_file_or_raise(model_path: str) -> Path:
    model_file = _resolve_model_file(model_path)
    if model_file is None:
        raise RuntimeError("model_path must be a directory with exactly one .pth/.pt file")
    return model_file


def _load_model_meta(model_file: Path) -> Dict:
    meta_path = Path(str(model_file) + ".meta.json")
    if not meta_path.is_file():
        return {}
    with meta_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _attack_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _print_attack_runtime_banner(attack_method: str, *, uses_transformer: bool) -> None:
    """保留运行时日志入口，接口默认不打印攻击方法与计算设备。"""
    return None


def _iter_mat_paths(dataset_path: Path) -> List[Path]:
    mat_files: List[Path] = []
    subdirs = [p for p in sorted(dataset_path.iterdir()) if p.is_dir()]
    if subdirs:
        for sd in subdirs:
            mat_files.extend(sorted(sd.glob("*.mat")))
    else:
        mat_files.extend(sorted(dataset_path.glob("*.mat")))
    return mat_files


def _load_mat_samples(dataset_path: Path, target_points: int = 32) -> tuple[list[TrackSample], list[Path]]:
    mat_paths = _iter_mat_paths(dataset_path)
    if not mat_paths:
        raise RuntimeError(f"No .mat files found under {dataset_path}")

    class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    samples: List[TrackSample] = []
    for mat_path in mat_paths:
        if mat_path.parent == dataset_path:
            label = 0
        else:
            label = class_names.index(mat_path.parent.name)
        samples.append(load_mat_track(mat_path, target_points=target_points, label=label))
    return samples, mat_paths


def _inverse_log_preprocess(value: np.ndarray, n: float, signed: bool) -> np.ndarray:
    if signed:
        sign = np.sign(value)
        return sign * float(n) * (np.power(10.0, np.abs(value) / 10.0) - 1.0)
    return float(n) * np.power(10.0, value / 10.0)


def _invert_preprocessed_track(x_adv: np.ndarray, raw_track: np.ndarray) -> np.ndarray:
    if x_adv.shape != raw_track.shape:
        raise ValueError("Attack output shape does not match raw track shape")

    out = np.empty_like(raw_track, dtype=np.float32)
    out[:, 0] = _inverse_log_preprocess(x_adv[:, 0], n=50.0, signed=True)
    out[:, 1] = _inverse_log_preprocess(x_adv[:, 1], n=1000.0, signed=False)
    out[:, 2] = _inverse_log_preprocess(x_adv[:, 2], n=100.0, signed=False)
    out[:, 3] = _inverse_log_preprocess(x_adv[:, 3], n=50.0, signed=True)
    out[:, 4] = _inverse_log_preprocess(x_adv[:, 4], n=1000.0, signed=False)
    out[:, 5] = _inverse_log_preprocess(x_adv[:, 5], n=100.0, signed=False)
    out[:, 6] = 100.0 * np.power(10.0, x_adv[:, 6] / 10.0)
    out[:, 7] = 100.0 * np.power(10.0, x_adv[:, 7] / 10.0)

    for col_idx in (8, 9):
        raw_col = raw_track[:, col_idx].astype(np.float64)
        col0 = float(raw_col[0])
        scale = float(np.max(np.abs(raw_col - col0)) + 1e-6)
        out[:, col_idx] = col0 + x_adv[:, col_idx] * scale

    iframe = raw_track[:, 10].astype(np.float64)
    min_iframe = float(np.min(iframe))
    range_iframe = float(np.max(iframe - min_iframe) + 1e-6)
    out[:, 10] = x_adv[:, 10] * range_iframe + min_iframe

    snr = raw_track[:, 11].astype(np.float64)
    mean_snr = float(np.mean(snr))
    std_snr = float(np.std(snr) + 1e-6)
    out[:, 11] = x_adv[:, 11] * std_snr + mean_snr
    return np.asarray(out, dtype=np.float32)


def _save_mat_track(track: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for idx, key in enumerate(TRACK_CHANNEL_KEYS):
            f.create_dataset(key, data=track[:, idx].astype(np.float32))


def _generate_whitebox_batch(model: torch.nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor, attack_method: str) -> np.ndarray:
    d = WHITEBOX_DEFAULTS
    x_ref = x_batch.detach()
    x_out = x_ref.clone()

    with torch.no_grad():
        clean_logits = model(x_ref)
        pred_clean = clean_logits.argmax(dim=1)

    attack_mask = pred_clean.eq(y_batch) if d["attack_clean_only"] else torch.ones_like(y_batch, dtype=torch.bool)
    if not attack_mask.any():
        return x_out.cpu().numpy()

    x_attack = x_ref[attack_mask]
    y_attack = y_batch[attack_mask]
    clean_logits_attack = clean_logits[attack_mask]
    target_candidates = _candidate_target_classes(clean_logits_attack, y_attack, int(d["targeted_topk"]))
    restart_count = 1 if attack_method == "fgsm" else int(d["attack_restarts"])
    best_res = None

    for _ in range(restart_count):
        if attack_method == "fgsm":
            restart_best = fgsm_attack(
                model,
                x_attack,
                y_attack,
                step_size=float(d["fgsm_step_size"]),
                targeted=False,
                max_rel_change=float(d["max_rel_change"]),
                budget_floor=float(d["budget_floor"]),
                clamp=None,
                loss_mode=str(d["attack_loss"]),
            )
        elif attack_method == "pgd":
            restart_best = pgd_linf_attack(
                model,
                x_attack,
                y_attack,
                step_size=float(d["pgd_step_size"]),
                num_steps=int(d["pgd_steps"]),
                random_start=True,
                targeted=False,
                max_rel_change=float(d["max_rel_change"]),
                budget_floor=float(d["budget_floor"]),
                clamp=None,
                loss_mode=str(d["attack_loss"]),
                momentum=float(d["pgd_momentum"]),
            )
        elif attack_method == "cw":
            restart_best = cw_l2_attack(
                model,
                x_attack,
                y_attack,
                c=float(d["cw_c"]),
                lr=float(d["cw_lr"]),
                num_steps=int(d["cw_steps"]),
                confidence=float(d["cw_confidence"]),
                targeted=False,
                max_rel_change=float(d["max_rel_change"]),
                budget_floor=float(d["budget_floor"]),
                clamp=None,
                random_start=True,
            )
        else:
            raise RuntimeError(f"Unsupported white-box attack method: {attack_method}")

        for target_idx in range(int(target_candidates.size(1))):
            y_target = target_candidates[:, target_idx]
            if attack_method == "fgsm":
                targeted_candidate = fgsm_attack(
                    model,
                    x_attack,
                    y_attack,
                    step_size=float(d["fgsm_step_size"]),
                    targeted=True,
                    y_target=y_target,
                    max_rel_change=float(d["max_rel_change"]),
                    budget_floor=float(d["budget_floor"]),
                    clamp=None,
                    loss_mode=str(d["attack_loss"]),
                )
            elif attack_method == "pgd":
                targeted_candidate = pgd_linf_attack(
                    model,
                    x_attack,
                    y_attack,
                    step_size=float(d["pgd_step_size"]),
                    num_steps=int(d["pgd_steps"]),
                    random_start=True,
                    targeted=True,
                    y_target=y_target,
                    max_rel_change=float(d["max_rel_change"]),
                    budget_floor=float(d["budget_floor"]),
                    clamp=None,
                    loss_mode=str(d["attack_loss"]),
                    momentum=float(d["pgd_momentum"]),
                )
            elif attack_method == "cw":
                targeted_candidate = cw_l2_attack(
                    model,
                    x_attack,
                    y_attack,
                    c=float(d["cw_c"]),
                    lr=float(d["cw_lr"]),
                    num_steps=int(d["cw_steps"]),
                    confidence=float(d["cw_confidence"]),
                    targeted=True,
                    y_target=y_target,
                    max_rel_change=float(d["max_rel_change"]),
                    budget_floor=float(d["budget_floor"]),
                    clamp=None,
                    random_start=True,
                )
            else:
                raise RuntimeError(f"Unsupported white-box attack method: {attack_method}")
            restart_best = _select_stronger_attack_result(restart_best, targeted_candidate, x_attack, y_attack)

        best_res = _select_stronger_attack_result(best_res, restart_best, x_attack, y_attack)

    assert best_res is not None
    x_out[attack_mask] = best_res.x_adv.detach()
    return x_out.cpu().numpy()


def _generate_noise_batch(model: torch.nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor, attack_method: str) -> np.ndarray:
    d = NOISE_DEFAULTS

    def _make_noisy(x_in: torch.Tensor) -> torch.Tensor:
        if attack_method == "gaussian":
            return add_gaussian_noise(
                x_in,
                sigma=float(d["sigma"]),
                max_rel_change=float(d["max_rel_change"]),
                min_scale=0.0,
                clamp=None,
            )
        if attack_method == "salt_pepper":
            return add_salt_pepper_noise(
                x_in,
                prob=float(d["sp_prob"]),
                max_rel_change=float(d["max_rel_change"]),
                min_scale=0.0,
                clamp=None,
            )
        if attack_method == "speckle":
            return add_speckle_noise(
                x_in,
                sigma=float(d["speckle_sigma"]),
                max_rel_change=float(d["max_rel_change"]),
                min_scale=0.0,
                clamp=None,
            )
        raise RuntimeError(f"Unsupported noise attack method: {attack_method}")

    worst_of_k = int(d["worst_of_k"])
    if worst_of_k <= 1:
        return _make_noisy(x_batch).detach().cpu().numpy()

    logits_cands: List[torch.Tensor] = []
    x_eval_cands: List[torch.Tensor] = []
    true_logits_cands: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(worst_of_k):
            x_eval = _make_noisy(x_batch)
            logits = model(x_eval)
            x_eval_cands.append(x_eval)
            logits_cands.append(logits)
            true_logits_cands.append(logits.gather(1, y_batch.view(-1, 1)).squeeze(1))
        cand_scores = torch.stack(true_logits_cands, dim=1)
        worst_idx = cand_scores.argmin(dim=1)
        x_eval_stack = torch.stack(x_eval_cands, dim=1)
        x_worst = x_eval_stack[torch.arange(x_eval_stack.size(0), device=x_batch.device), worst_idx]
    return x_worst.detach().cpu().numpy()


def _merge_blackbox_results(current: BlackboxResult, candidate: BlackboxResult, x_ref: torch.Tensor, y_true: torch.Tensor) -> BlackboxResult:
    BlackboxResult, _, _, _ = _load_blackbox_attacks()
    cur_flip = current.y_pred.to(y_true.device).ne(y_true)
    new_flip = candidate.y_pred.to(y_true.device).ne(y_true)
    take = (~cur_flip) & new_flip
    if not take.any():
        return current

    x_merged = current.x_adv.clone()
    x_merged[take] = candidate.x_adv[take]
    logits_merged = current.logits_or_probs.clone()
    logits_merged[take] = candidate.logits_or_probs[take]
    y_merged = current.y_pred.clone().cpu()
    y_merged[take.cpu()] = candidate.y_pred.cpu()[take.cpu()]
    q_merged = current.queries_per_sample + candidate.queries_per_sample
    stats = relative_change_stats(x_ref, x_merged)
    return BlackboxResult(
        x_adv=x_merged,
        y_pred=y_merged,
        logits_or_probs=logits_merged,
        mean_rel_change=stats["mean_rel_change"],
        max_rel_change=stats["max_rel_change"],
        queries_per_sample=q_merged,
    )


def _generate_blackbox_batch(model: torch.nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor, attack_method: str) -> np.ndarray:
    _, nes_pgd_attack, square_attack, transfer_attack = _load_blackbox_attacks()
    d = BLACKBOX_DEFAULTS
    x_ref = x_batch.detach()
    x_out = x_ref.clone()

    with torch.no_grad():
        clean_logits = model(x_ref)
        pred_clean = clean_logits.argmax(dim=1)

    attack_mask = pred_clean.eq(y_batch) if d["attack_clean_only"] else torch.ones_like(y_batch, dtype=torch.bool)
    if not attack_mask.any():
        return x_out.cpu().numpy()

    x_attack = x_ref[attack_mask]
    y_attack = y_batch[attack_mask]
    clean_logits_attack = clean_logits[attack_mask]
    num_classes = int(model.head[-1].out_features)

    if attack_method == "square":
        targeted_topk = int(d["square_targeted_topk"])
        attack_restarts = int(d["square_attack_restarts"])
    elif attack_method == "nes":
        targeted_topk = int(d["nes_targeted_topk"])
        attack_restarts = int(d["nes_attack_restarts"])
    else:
        targeted_topk = int(d["transfer_targeted_topk"])
        attack_restarts = 1

    k = max(1, min(targeted_topk, max(1, num_classes - 1)))
    masked = clean_logits_attack.clone()
    masked.scatter_(1, y_attack.view(-1, 1), float("-inf"))
    target_candidates = masked.topk(k=k, dim=1).indices
    best_res = None

    for _ in range(attack_restarts):
        if attack_method == "square":
            restart_best = square_attack(
                model,
                x_attack,
                y_attack,
                max_queries=int(d["square_max_queries"]),
                max_rel_change=float(d["max_rel_change"]),
                budget_floor=float(d["budget_floor"]),
                clamp=None,
                targeted=False,
                p_init=float(d["square_p_init"]),
            )
        elif attack_method == "nes":
            restart_best = nes_pgd_attack(
                model,
                x_attack,
                y_attack,
                nes_samples=int(d["nes_samples"]),
                nes_sigma=float(d["nes_sigma"]),
                num_steps=int(d["nes_steps"]),
                step_size=float(d["nes_step_size"]),
                max_rel_change=float(d["max_rel_change"]),
                budget_floor=float(d["budget_floor"]),
                clamp=None,
                targeted=False,
                momentum=float(d["nes_momentum"]),
            )
        elif attack_method == "transfer":
            restart_best = transfer_attack(
                model,
                x_attack,
                y_attack,
                pgd_steps=int(d["tr_pgd_steps"]),
                pgd_step_size=float(d["tr_pgd_step_size"]),
                num_restarts=int(d["tr_restarts"]),
                max_rel_change=float(d["max_rel_change"]),
                budget_floor=float(d["budget_floor"]),
                clamp=None,
                momentum=float(d["tr_momentum"]),
                targeted=False,
                device=x_attack.device.type,
            )
        else:
            raise RuntimeError(f"Unsupported black-box attack method: {attack_method}")

        for target_idx in range(k):
            y_target = target_candidates[:, target_idx]
            if attack_method == "square":
                targeted_candidate = square_attack(
                    model,
                    x_attack,
                    y_attack,
                    max_queries=int(d["square_max_queries"]),
                    max_rel_change=float(d["max_rel_change"]),
                    budget_floor=float(d["budget_floor"]),
                    clamp=None,
                    targeted=True,
                    y_target=y_target,
                    p_init=float(d["square_p_init"]),
                )
            elif attack_method == "nes":
                targeted_candidate = nes_pgd_attack(
                    model,
                    x_attack,
                    y_attack,
                    nes_samples=int(d["nes_samples"]),
                    nes_sigma=float(d["nes_sigma"]),
                    num_steps=int(d["nes_steps"]),
                    step_size=float(d["nes_step_size"]),
                    max_rel_change=float(d["max_rel_change"]),
                    budget_floor=float(d["budget_floor"]),
                    clamp=None,
                    targeted=True,
                    y_target=y_target,
                    momentum=float(d["nes_momentum"]),
                )
            else:
                targeted_candidate = transfer_attack(
                    model,
                    x_attack,
                    y_attack,
                    pgd_steps=int(d["tr_pgd_steps"]),
                    pgd_step_size=float(d["tr_pgd_step_size"]),
                    num_restarts=int(d["tr_restarts"]),
                    max_rel_change=float(d["max_rel_change"]),
                    budget_floor=float(d["budget_floor"]),
                    clamp=None,
                    momentum=float(d["tr_momentum"]),
                    targeted=True,
                    y_target=y_target,
                    device=x_attack.device.type,
                )
            restart_best = _merge_blackbox_results(restart_best, targeted_candidate, x_attack, y_attack)

        best_res = restart_best if best_res is None else _merge_blackbox_results(best_res, restart_best, x_attack, y_attack)

    assert best_res is not None
    x_out[attack_mask] = best_res.x_adv.detach()
    return x_out.cpu().numpy()


def _generate_mat_adversarial(
    model_file: Path,
    dataset_path: Path,
    adv_path: Path,
    attack_method: str,
) -> Dict[str, float]:
    meta = _load_model_meta(model_file)
    target_points = int(meta.get("mat_target_points", 32))

    samples, mat_paths = _load_mat_samples(dataset_path, target_points=target_points)
    loader = robust_build_loader(samples, batch_size=64)
    device = _attack_device()
    model = load_whitebox_model(str(model_file), device=device)

    x_clean_batches: List[np.ndarray] = []
    x_adv_batches: List[np.ndarray] = []
    y_batches: List[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        x_clean_batches.append(x_batch.detach().cpu().numpy())
        y_batches.append(y_batch.detach().cpu().numpy())
        if attack_method in WHITEBOX_ATTACKS:
            x_adv_batches.append(_run_silently(_generate_whitebox_batch, model, x_batch, y_batch, attack_method))
        elif attack_method in BLACKBOX_ATTACKS:
            x_adv_batches.append(_run_silently(_generate_blackbox_batch, model, x_batch, y_batch, attack_method))
        elif attack_method in NOISE_ATTACKS:
            x_adv_batches.append(_run_silently(_generate_noise_batch, model, x_batch, y_batch, attack_method))
        else:
            raise RuntimeError(f"Unsupported attack method for MAT dataset: {attack_method}")

    x_clean = np.concatenate(x_clean_batches, axis=0)
    x_adv = np.concatenate(x_adv_batches, axis=0)
    y_all = np.concatenate(y_batches, axis=0)
    if x_adv.shape[0] != len(samples):
        raise RuntimeError("Generated adversarial batch count does not match sample count")

    adv_path.mkdir(parents=True, exist_ok=True)
    attack_metrics = _compute_tensor_perturbation_metrics(x_clean, x_adv)
    interface_eval = _evaluate_processed_attack_result(
        model,
        x_clean,
        x_adv,
        y_all,
        device=device,
    )
    with (adv_path / INTERFACE_EVAL_RESULT_FILE).open("w", encoding="utf-8") as fh:
        json.dump(interface_eval, fh, ensure_ascii=False, indent=2)

    for sample, mat_path, x_adv_sample in zip(samples, mat_paths, x_adv):
        raw_adv = _invert_preprocessed_track(x_adv_sample, sample.track)
        _save_mat_track(raw_adv, adv_path / mat_path.relative_to(dataset_path))
    return attack_metrics


def generate_adversarial(model_path: str, dataset_path: str, adv_dataset_path: str, attack_method: str) -> Dict:
    """Generate adversarial samples and return similarity metrics."""
    ds = Path(dataset_path)
    adv = Path(adv_dataset_path)
    method = _normalise_attack_method(attack_method)

    if not validate_model(model_path, dataset_path):
        raise RuntimeError("Model or dataset validation failed")
    model_file = _resolve_model_file_or_raise(model_path)

    if adv.exists():
        shutil.rmtree(adv)

    if _is_mat_dataset(ds):
        _print_attack_runtime_banner(method, uses_transformer=True)
        return _generate_mat_adversarial(model_file, ds, adv, method)

    raise RuntimeError("Unsupported dataset format. Expected <dataset_path>/<class_name>/*.mat")

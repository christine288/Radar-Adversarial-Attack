from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch

from data_utils import TRACK_CHANNEL_KEYS, TrackSample
from mat_loader import load_mat_track
from model_service import validate_model
from robust_evaluate import build_loader as robust_build_loader
from whitebox_attacks import (
    cw_l2_attack,
    deepfool_attack,
    fgsm_attack,
    load_model as load_whitebox_model,
    pgd_linf_attack,
)
from blackbox_attacks import (
    nes_pgd_attack,
    square_attack,
    transfer_attack,
)
from dataset_service import prepare_dataset


def _is_custom_dataset(dataset_path: Path) -> bool:
    return (dataset_path / "dataset.yaml").is_file()


def _is_kitti_dataset(dataset_path: Path) -> bool:
    training = dataset_path / "training"
    if not training.is_dir():
        return False
    return all((training / sub).is_dir() for sub in ("velodyne", "calib", "label_2"))


def _ensure_imagesets(dataset_path: Path) -> None:
    imagesets_dir = dataset_path / "ImageSets"
    imagesets_dir.mkdir(exist_ok=True)
    val_file = imagesets_dir / "val.txt"
    if not val_file.exists():
        if _is_custom_dataset(dataset_path):
            source_dir = dataset_path / "points"
        else:
            source_dir = dataset_path / "training" / "velodyne"
        frames = [f.stem for f in source_dir.iterdir() if f.is_file()]
        frames.sort()
        with val_file.open("w", encoding="utf-8") as fh:
            for fid in frames:
                fh.write(fid + "\n")


def _read_bin_points(bin_path: Path) -> np.ndarray:
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Unexpected KITTI .bin shape: {bin_path}")
    return raw.reshape(-1, 4)


def _save_bin_points(bin_path: Path, points: np.ndarray) -> None:
    points.astype(np.float32).tofile(bin_path)


def _perturb_points(points: np.ndarray, scale: float) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    noise = np.random.normal(loc=0.0, scale=scale, size=points.shape)
    return points + noise


def _compute_metrics_from_noise(noise: np.ndarray) -> Dict[str, float]:
    deltas = np.linalg.norm(noise.reshape(-1, noise.shape[-1]), axis=1)
    if deltas.size == 0:
        return {"mean_L2": 0.0, "max_L2": 0.0, "mad": 0.0, "chamfer": 0.0, "pts_ratio": 0.0}
    mean_L2 = float(np.mean(deltas))
    max_L2 = float(np.max(deltas))
    mad = float(np.mean(np.abs(deltas - mean_L2)))
    chamfer = float(np.mean(deltas))
    return {"mean_L2": mean_L2, "max_L2": max_L2, "mad": mad, "chamfer": chamfer, "pts_ratio": 1.0}


def _append_dataset_yaml(src: Path, dst: Path) -> None:
    yaml_file = src / "dataset.yaml"
    if yaml_file.is_file():
        shutil.copy2(yaml_file, dst / "dataset.yaml")


def _copy_labels(src: Path, dst: Path) -> None:
    if src.is_dir():
        dst.mkdir(exist_ok=True, parents=True)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, dst / item.name)


def _is_mat_dataset(dataset_path: Path) -> bool:
    if not dataset_path.exists() or not dataset_path.is_dir():
        return False
    return any(dataset_path.rglob("*.mat"))


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

    samples: List[TrackSample] = []
    for mat_path in mat_paths:
        parent = mat_path.parent
        if parent == dataset_path:
            label = 0
        else:
            label = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()]).index(parent.name)
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


def _apply_noise_attack(x: torch.Tensor, attack_method: str) -> np.ndarray:
    from noise_perturbation import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise

    if attack_method == "gaussian":
        x_adv = add_gaussian_noise(x, sigma=0.5, max_rel_change=0.05, min_scale=0.0, clamp=None)
    elif attack_method == "salt_pepper":
        x_adv = add_salt_pepper_noise(x, prob=0.2, max_rel_change=0.05, min_scale=0.0, clamp=None)
    elif attack_method == "speckle":
        x_adv = add_speckle_noise(x, sigma=0.5, max_rel_change=0.05, min_scale=0.0, clamp=None)
    else:
        raise RuntimeError(f"Unsupported noise attack method: {attack_method}")
    return x_adv.cpu().numpy()


def _generate_mat_adversarial(
    model_path: str,
    dataset_path: Path,
    adv_path: Path,
    attack_method: str,
) -> Dict[str, float]:
    meta_path = f"{model_path}.meta.json"
    target_points = 32
    if Path(meta_path).is_file():
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        target_points = int(meta.get("mat_target_points", target_points))

    samples, mat_paths = _load_mat_samples(dataset_path, target_points=target_points)
    loader = robust_build_loader(samples, batch_size=64)
    model = load_whitebox_model(model_path, device="cpu")

    x_adv_batches: List[np.ndarray] = []
    for x_batch, y_batch in loader:
        if attack_method in {"fgsm", "pgd", "cw", "deepfool"}:
            x_batch = x_batch.to("cpu")
            y_batch = y_batch.to("cpu")
            if attack_method == "fgsm":
                res = fgsm_attack(model, x_batch, y_batch, step_size=1.0, max_rel_change=0.05, budget_floor=0.0, clamp=None)
            elif attack_method == "pgd":
                res = pgd_linf_attack(model, x_batch, y_batch, step_size=0.01, num_steps=20, max_rel_change=0.05, budget_floor=0.0, clamp=None)
            elif attack_method == "cw":
                res = cw_l2_attack(model, x_batch, y_batch, c=1e-2, lr=1e-2, num_steps=100, confidence=0.0, max_rel_change=0.05, budget_floor=0.0, clamp=None)
            else:
                res = deepfool_attack(model, x_batch, y_batch, num_steps=50, overshoot=0.02, max_rel_change=0.05, budget_floor=0.0, clamp=None)
            x_adv_batches.append(res.x_adv.cpu().numpy())
        elif attack_method in {"square", "nes", "transfer"}:
            x_batch = x_batch.to("cpu")
            y_batch = y_batch.to("cpu")
            if attack_method == "square":
                res = square_attack(model, x_batch, y_batch, max_queries=200, max_rel_change=0.05, budget_floor=0.0, clamp=None, targeted=False, p_init=0.8)
            elif attack_method == "nes":
                res = nes_pgd_attack(model, x_batch, y_batch, nes_samples=10, nes_sigma=0.01, num_steps=50, step_size=0.3, max_rel_change=0.05, budget_floor=0.0, clamp=None, targeted=False, momentum=0.9)
            else:
                res = transfer_attack(model, x_batch, y_batch, pgd_steps=100, pgd_step_size=0.3, num_restarts=3, max_rel_change=0.05, budget_floor=0.0, clamp=None, momentum=0.9, targeted=False, device="cpu")
            x_adv_batches.append(res.x_adv.cpu().numpy())
        elif attack_method in {"gaussian", "salt_pepper", "speckle"}:
            x_adv_batches.append(_apply_noise_attack(x_batch, attack_method))
        else:
            raise RuntimeError(f"Unsupported attack method for MAT dataset: {attack_method}")

    x_adv = np.concatenate(x_adv_batches, axis=0)
    if x_adv.shape[0] != len(samples):
        raise RuntimeError("Generated adversarial batch count does not match sample count")

    adv_path.mkdir(parents=True, exist_ok=True)
    all_deltas: List[float] = []
    for sample, mat_path, x_adv_sample in zip(samples, mat_paths, x_adv):
        raw_adv = _invert_preprocessed_track(x_adv_sample, sample.track)
        _save_mat_track(raw_adv, adv_path / mat_path.relative_to(dataset_path))
        diff = np.linalg.norm((raw_adv - sample.track).reshape(-1, sample.track.shape[-1]), axis=1)
        all_deltas.extend(diff.tolist())

    return _compute_metrics_from_noise(np.asarray(all_deltas, dtype=np.float64).reshape(-1, 1))


def generate_adversarial(model_path: str, dataset_path: str, adv_dataset_path: str, attack_method: str) -> Dict:
    """生成对抗数据集并返回相似度指标。"""
    ds = Path(dataset_path)
    adv = Path(adv_dataset_path)

    if not validate_model(model_path, dataset_path):
        raise RuntimeError("Model or dataset validation failed")

    if _is_mat_dataset(ds):
        if adv.exists():
            shutil.rmtree(adv)
        return _generate_mat_adversarial(model_path, ds, adv, attack_method)

    if not prepare_dataset(dataset_path):
        raise RuntimeError("Dataset preparation failed")

    adv.mkdir(parents=True, exist_ok=True)

    if _is_custom_dataset(ds):
        (adv / "points").mkdir(exist_ok=True, parents=True)
        _copy_labels(ds / "labels", adv / "labels")
        _append_dataset_yaml(ds, adv)
        _ensure_imagesets(adv)
        all_deltas: List[float] = []
        for npy_path in sorted((ds / "points").glob("*.npy")):
            points = np.load(npy_path)
            scale = 0.01 * max(1.0, np.max(np.abs(points)))
            if attack_method == "fgsm":
                scale *= 0.6
            elif attack_method == "pgd":
                scale *= 1.0
            elif attack_method == "cw":
                scale *= 1.5
            elif attack_method == "deepfool":
                scale *= 1.2
            adv_points = _perturb_points(points, scale)
            np.save(adv / "points" / npy_path.name, adv_points)
            deltas = np.linalg.norm((adv_points - points).reshape(-1, points.shape[-1]), axis=1)
            all_deltas.extend(deltas.tolist())
        if not all_deltas:
            raise RuntimeError("Custom dataset points directory is empty")
        return _compute_metrics_from_noise(np.asarray(all_deltas, dtype=np.float64).reshape(-1, 1))
    elif _is_kitti_dataset(ds):
        src_training = ds / "training"
        dst_training = adv / "training"
        dst_training.mkdir(parents=True, exist_ok=True)
        (dst_training / "velodyne").mkdir(exist_ok=True, parents=True)
        (dst_training / "calib").mkdir(exist_ok=True, parents=True)
        (dst_training / "label_2").mkdir(exist_ok=True, parents=True)
        _ensure_imagesets(adv)
        all_deltas: List[float] = []
        for src_bin in sorted((src_training / "velodyne").glob("*.bin")):
            points = _read_bin_points(src_bin)
            scale = 0.01 * max(1.0, np.max(np.abs(points)))
            if attack_method == "fgsm":
                scale *= 0.6
            elif attack_method == "pgd":
                scale *= 1.0
            elif attack_method == "cw":
                scale *= 1.5
            elif attack_method == "deepfool":
                scale *= 1.2
            adv_points = _perturb_points(points, scale)
            _save_bin_points(dst_training / "velodyne" / src_bin.name, adv_points)
            deltas = np.linalg.norm((adv_points - points).reshape(-1, points.shape[-1]), axis=1)
            all_deltas.extend(deltas.tolist())
        for src_calib in (src_training / "calib").glob("*"):
            if src_calib.is_file():
                shutil.copy2(src_calib, dst_training / "calib" / src_calib.name)
        for src_label in (src_training / "label_2").glob("*"):
            if src_label.is_file():
                shutil.copy2(src_label, dst_training / "label_2" / src_label.name)
        if not all_deltas:
            raise RuntimeError("KITTI velodyne directory is empty")
        return _compute_metrics_from_noise(np.asarray(all_deltas, dtype=np.float64).reshape(-1, 1))
    else:
        raise RuntimeError("Unsupported dataset format for adversarial generation")

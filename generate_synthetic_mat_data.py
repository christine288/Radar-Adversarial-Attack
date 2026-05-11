"""按现有 MAT 航迹分布生成合成样本（用于训练集扩充）。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

TRACK_KEYS = [
    "V_m",
    "R_m",
    "A_m",
    "V",
    "R",
    "A",
    "E_m",
    "E",
    "DATA_time",
    "GPS_time_in_data",
    "Iframecnt",
    "SNR",
]


def _safe_std(x: np.ndarray) -> np.ndarray:
    return np.std(x, axis=0) + 1e-6


def _signed_log_forward(x: np.ndarray, n: float) -> np.ndarray:
    """signed-log: y=sign(x)*10*log10(1+|x|/n)."""
    return np.sign(x) * 10.0 * np.log10(1.0 + np.abs(x) / float(n))


def _signed_log_inverse(y: np.ndarray, n: float) -> np.ndarray:
    """inverse of signed-log."""
    return np.sign(y) * float(n) * (np.power(10.0, np.abs(y) / 10.0) - 1.0)


def _paper_log_forward_positive(x: np.ndarray, n: float) -> np.ndarray:
    """y=10*log10(x/n)，要求 x>0。"""
    return 10.0 * np.log10(np.clip(x, 1e-6, None) / float(n))


def _paper_log_inverse_positive(y: np.ndarray, n: float) -> np.ndarray:
    """对数逆变换：x=n*10^(y/10)。"""
    return float(n) * np.power(10.0, y / 10.0)


def _read_track(mat_path: Path) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        cols = [np.asarray(f[k], dtype=np.float64).reshape(-1) for k in TRACK_KEYS]
    n = min(len(c) for c in cols)
    return np.stack([c[:n] for c in cols], axis=1)


def _resample(track: np.ndarray, target_len: int) -> np.ndarray:
    if track.shape[0] == target_len:
        return track.copy()
    old_t = np.linspace(0.0, 1.0, track.shape[0])
    new_t = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, track.shape[1]), dtype=np.float64)
    for c in range(track.shape[1]):
        out[:, c] = np.interp(new_t, old_t, track[:, c])
    return out


def _smooth_1d(x: np.ndarray, win: int = 7) -> np.ndarray:
    """简单滑动平均，保留时序连续性。"""
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, k, mode="same")


def _synthesize_from_class_tracks(
    class_tracks: List[np.ndarray],
    rng: np.random.Generator,
    noise_scale: float,
    paper_log_enhance: bool = True,
    anomaly_prob: float = 0.0,
    anomaly_scale: float = 3.0,
) -> np.ndarray:
    # 1) 用同类两条航迹混合作为原型，避免“纯加噪”导致假模式
    i = int(rng.integers(0, len(class_tracks)))
    j = int(rng.integers(0, len(class_tracks)))
    base_a = class_tracks[i]
    base_b = class_tracks[j]
    target_len = int(np.clip(round(base_a.shape[0] * rng.uniform(0.8, 1.25)), 24, 420))
    xa = _resample(base_a, target_len)
    xb = _resample(base_b, target_len)
    lam = float(rng.uniform(0.35, 0.75))
    x = lam * xa + (1.0 - lam) * xb #俩条真实的航迹混合

    cat = np.concatenate(class_tracks, axis=0)
    col_std = _safe_std(cat)
    col_mean = np.mean(cat, axis=0)

    # 2) 分列加入平滑噪声（时间连续），再加微弱随机游走漂移
    for c in range(x.shape[1]):
        n = rng.normal(0.0, col_std[c] * noise_scale, size=target_len)
        n = _smooth_1d(n, win=7)
        x[:, c] += n

    # 2.1) 在对数域扰动后再逆变换，减少量纲差异影响
    if paper_log_enhance:
        # 正值通道按对数：distance/time
        # R_m(1), R(4), DATA_time(8), GPS_time_in_data(9)
        pos_log_cols: List[Tuple[int, float]] = [
            (1, 1000.0),   # distance
            (4, 1000.0),   # distance
            (8, 1_000_000.0),  # time
            (9, 1_000_000.0),  # time
        ]
        for c, n_base in pos_log_cols:
            y = _paper_log_forward_positive(x[:, c], n=n_base)
            y += _smooth_1d(rng.normal(0.0, noise_scale * 0.55, size=target_len), win=5)
            x[:, c] = _paper_log_inverse_positive(y, n=n_base)

        # 带正负的通道采用 signed-log，避免角度/速度被硬裁成正数
        # V_m(0), A_m(2), V(3), A(5), E_m(6), E(7)
        signed_log_cols: List[Tuple[int, float]] = [
            (0, 50.0),    # speed
            (2, 100.0),   # orientation
            (3, 50.0),    # speed
            (5, 100.0),   # orientation
            (6, 100.0),   # pitch
            (7, 100.0),   # pitch
        ]
        for c, n_base in signed_log_cols:
            y = _signed_log_forward(x[:, c], n=n_base)
            y += _smooth_1d(rng.normal(0.0, noise_scale * 0.45, size=target_len), win=5)
            x[:, c] = _signed_log_inverse(y, n=n_base)
    drift = np.cumsum(
        rng.normal(0.0, col_std * (noise_scale * 0.05), size=x.shape),
        axis=0,
    )
    x += drift #模拟真实数据的误差 漂移

    # 约束：距离、时间、帧计数必须有物理可读性
    x[:, 1] = np.clip(x[:, 1], 0.05, None)  # R_m
    x[:, 4] = np.clip(x[:, 4], 0.05, None)  # R

    # 角度与俯仰限制到常见范围（避免极端噪声破坏）
    x[:, 2] = np.clip(x[:, 2], -180.0, 180.0)  # A_m
    x[:, 5] = np.clip(x[:, 5], -180.0, 180.0)  # A
    x[:, 6] = np.clip(x[:, 6], -90.0, 90.0)    # E_m
    x[:, 7] = np.clip(x[:, 7], -90.0, 90.0)    # E

    # 3) 关联量 (V,R,A,E) 与测量量 (V_m,R_m,A_m,E_m) 保持同向关系
    x[:, 3] = 0.7 * x[:, 0] + 0.3 * x[:, 3] + rng.normal(0.0, col_std[3] * 0.05, size=target_len)
    x[:, 4] = 0.7 * x[:, 1] + 0.3 * x[:, 4] + rng.normal(0.0, col_std[4] * 0.05, size=target_len)
    x[:, 5] = 0.7 * x[:, 2] + 0.3 * x[:, 5] + rng.normal(0.0, col_std[5] * 0.05, size=target_len)
    x[:, 7] = 0.7 * x[:, 6] + 0.3 * x[:, 7] + rng.normal(0.0, col_std[7] * 0.05, size=target_len)

    # 时间列强制单调递增，步长参考原数据中位数
    def _median_dt(col: np.ndarray) -> float:
        dt = np.diff(col)
        dt = dt[dt > 0]
        return float(np.median(dt)) if dt.size else 1.0

    dt_data = _median_dt(cat[:, 8])
    dt_gps = _median_dt(cat[:, 9])
    t0 = float(x[0, 8])
    g0 = float(x[0, 9])
    x[:, 8] = t0 + np.arange(target_len, dtype=np.float64) * max(dt_data, 1e-3)
    x[:, 9] = g0 + np.arange(target_len, dtype=np.float64) * max(dt_gps, 1e-3)

    # 帧号递增
    start_frame = int(max(0.0, np.min(cat[:, 10])))
    x[:, 10] = np.arange(start_frame, start_frame + target_len, dtype=np.float64)

    # SNR 收敛到该类分布附近，防止极端异常值
    snr_mu, snr_std = float(col_mean[11]), float(col_std[11])
    x[:, 11] = np.clip(x[:, 11], snr_mu - 3.0 * snr_std, snr_mu + 3.0 * snr_std)
    x[:, 11] = np.clip(x[:, 11], 0.0, 120.0)

    # 4) 异常点注入：模拟论文中“带波动干扰的测试集”
    if anomaly_prob > 0.0:
        anom_prob = float(np.clip(anomaly_prob, 0.0, 0.5))
        mask = rng.random(target_len) < anom_prob
        if np.any(mask):
            # 重点扰动距离/方位相关列，且用平滑项避免过于尖锐
            for c in (1, 2, 4, 5):
                a = rng.normal(0.0, col_std[c] * float(anomaly_scale), size=target_len)
                a = _smooth_1d(a, win=3)
                x[mask, c] += a[mask]

    return x.astype(np.float32)


def _write_track(track: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for i, k in enumerate(TRACK_KEYS):
            f.create_dataset(k, data=track[:, i], dtype="float32")


def generate_dataset(
    source_root: Path,
    output_root: Path,
    synthetic_per_class: int,
    noise_scale: float,
    seed: int,
    copy_original: bool,
    target_count_per_class: int,
    paper_log_enhance: bool,
    anomaly_prob: float,
    anomaly_scale: float,
) -> None:
    rng = np.random.default_rng(seed)
    classes = [d for d in sorted(source_root.iterdir()) if d.is_dir()]
    if not classes:
        raise ValueError(f"未找到类别子目录: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    for cls_dir in classes:
        mats = sorted(cls_dir.glob("*.mat"))
        if not mats:
            print(f"[skip] {cls_dir.name}: 无 .mat 文件")
            continue

        out_cls = output_root / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)
        tracks = [_read_track(p) for p in mats]

        if copy_original:
            for p in mats:
                shutil.copy2(p, out_cls / p.name)

        synth_n = int(synthetic_per_class)
        if target_count_per_class > 0:
            if copy_original:
                synth_n = max(0, int(target_count_per_class) - len(mats))
            else:
                synth_n = max(0, int(target_count_per_class))

        for i in range(synth_n):
            syn = _synthesize_from_class_tracks(
                tracks,
                rng=rng,
                noise_scale=noise_scale,
                paper_log_enhance=paper_log_enhance,
                anomaly_prob=anomaly_prob,
                anomaly_scale=anomaly_scale,
            )
            out_name = f"SYNTH_{cls_dir.name.replace(' ', '_')}_{i+1:04d}.mat"
            _write_track(syn, out_cls / out_name)

        print(
            f"[ok] {cls_dir.name}: 原始 {len(mats)} + 合成 {synth_n}"
            f" -> 输出目录 {out_cls}"
        )

    print(f"\n完成。增强训练集目录: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成与现有 MAT 结构一致的合成航迹样本")
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("project_data/mat_train"),
        help="原始训练集目录（按类别子目录组织）",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("project_data/mat_train_augmented"),
        help="输出目录（不会覆盖 source_root）",
    )
    parser.add_argument("--synthetic_per_class", type=int, default=30, help="每类生成数量")
    parser.add_argument("--noise_scale", type=float, default=0.18, help="扰动强度（0.05~0.35 之间常用）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--target_count_per_class",
        type=int,
        default=0,
        help="每类目标总数（>0 时自动按类补齐；copy_original 时为 原始+合成 的总数）",
    )
    parser.add_argument(
        "--disable_paper_log_enhance",
        action="store_true",
        help="关闭论文风格对数域扰动增强（默认开启）",
    )
    parser.add_argument(
        "--anomaly_prob",
        type=float,
        default=0.02,
        help="异常点注入概率（模拟波动干扰，建议 0.00~0.08）",
    )
    parser.add_argument(
        "--anomaly_scale",
        type=float,
        default=3.0,
        help="异常点扰动倍率（相对该类通道标准差）",
    )
    parser.add_argument(
        "--no_copy_original",
        action="store_true",
        help="不复制原始样本到输出目录，仅保留合成样本",
    )
    args = parser.parse_args()

    generate_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        synthetic_per_class=args.synthetic_per_class,
        noise_scale=args.noise_scale,
        seed=args.seed,
        copy_original=not args.no_copy_original,
        target_count_per_class=args.target_count_per_class,
        paper_log_enhance=not args.disable_paper_log_enhance,
        anomaly_prob=args.anomaly_prob,
        anomaly_scale=args.anomaly_scale,
    )


if __name__ == "__main__":
    main()

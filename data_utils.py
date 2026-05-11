"""数据工具：定义航迹样本结构，并将航迹序列转换为 Transformer 输入。"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# MAT 根节点字段顺序；与 ``track`` 第二维列顺序一致，全部参与 ``track_to_matrix`` → 训练/测试
TRACK_CHANNEL_KEYS: Tuple[str, ...] = (
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
)
NUM_TRACK_CHANNELS: int = len(TRACK_CHANNEL_KEYS)


@dataclass
class TrackSample:
    """
    单条航迹样本（时序），包含多个航迹点以及类别标签。

    ``track`` 形状为 ``(points, NUM_TRACK_CHANNELS)``，列顺序见 ``TRACK_CHANNEL_KEYS``。
    距离列 ``R_m``、``R`` 在 ``mat_loader`` 中已按 km→m 规则统一为米后再入模。
    """

    track: np.ndarray  # (points, NUM_TRACK_CHANNELS)
    label: int         # 航迹类别编号，从 0 开始


def log_preprocess(
    x: np.ndarray,
    n_doppler: float = 50.0,
    n_distance: float = 1000.0,
    n_orientation: float = 100.0,
) -> np.ndarray:
    r"""
    对数预处理（论文思路扩展到时序三通道）。

    - 距离、方位角：X' = 10*log10(X/n)（要求为正，内部 clip）
    - 多普勒速度：可正可负，使用 signed log：sign(v)*10*log10(1+|v|/n)

    参数
    ----
    x: shape (..., 3)，最后一维依次为 [doppler, distance, orientation]
    """
    if x.shape[-1] != 3:
        raise ValueError(f"last dim must be 3, got {x.shape}")

    d = x[..., 0]
    r = x[..., 1]
    az = x[..., 2]

    d_out = np.sign(d) * 10.0 * np.log10(1.0 + np.abs(d) / float(n_doppler))

    r_safe = np.clip(r, 1e-6, None)
    az_safe = np.clip(az, 1e-6, None)
    r_out = 10.0 * np.log10(r_safe / float(n_distance))
    az_out = 10.0 * np.log10(az_safe / float(n_orientation))

    return np.stack([d_out, r_out, az_out], axis=-1)


def preprocess_track_channels(track: np.ndarray) -> np.ndarray:
    """
    对 ``(T, NUM_TRACK_CHANNELS)`` 主序列逐列做可喂给 Transformer 的变换。

    - 与 ``V_m,R_m,A_m`` 相同物理含义的 ``V,R,A``：共用 ``log_preprocess``（三列一组）
    - ``E_m,E``：俯仰（度），按方位类 ``10*log10(clip/100)``
    - ``DATA_time`` / ``GPS_time_in_data``：去起点后按幅值归一
    - ``Iframecnt``： min-max 到 [0,1]（按该条轨迹）
    - ``SNR``：该条轨迹内 z-score
    """
    if track.ndim != 2 or track.shape[1] != NUM_TRACK_CHANNELS:
        raise ValueError(f"track 应为 (points, {NUM_TRACK_CHANNELS}), got {track.shape}")

    x = track.astype(np.float64)
    t = x.shape[0]
    out = np.zeros((t, NUM_TRACK_CHANNELS), dtype=np.float64)

    out[:, :3] = log_preprocess(x[:, :3])
    out[:, 3:6] = log_preprocess(x[:, 3:6])

    em = np.clip(x[:, 6], 1e-6, None)
    ee = np.clip(x[:, 7], 1e-6, None)
    out[:, 6] = 10.0 * np.log10(em / 100.0)
    out[:, 7] = 10.0 * np.log10(ee / 100.0)

    for j in (8, 9):
        col = x[:, j] - x[0, j]
        m = float(np.max(np.abs(col)) + 1e-6)
        out[:, j] = col / m

    fr = x[:, 10] - np.min(x[:, 10])
    m_fr = float(np.max(fr) + 1e-6)
    out[:, 10] = fr / m_fr

    snr = x[:, 11]
    out[:, 11] = (snr - float(np.mean(snr))) / (float(np.std(snr)) + 1e-6)

    return np.asarray(out, dtype=np.float32)


def resample_track(track: np.ndarray, target_points: int) -> np.ndarray:
    """将 ``(T, C)`` 时序对各列线性插值到 ``target_points`` 个时刻。"""
    if track.ndim != 2:
        raise ValueError(f"track must be 2-D, got {track.shape}")
    t, c = track.shape[0], track.shape[1]
    if t == target_points:
        return track.astype(np.float32)
    if t < 2:
        raise ValueError(f"resample_track 需要至少 2 个点，got {t}")
    x_old = np.linspace(0.0, 1.0, t)
    x_new = np.linspace(0.0, 1.0, target_points)
    out = np.zeros((target_points, c), dtype=np.float32)
    for j in range(c):
        out[:, j] = np.interp(x_new, x_old, track[:, j].astype(np.float64)).astype(np.float32)
    return out


def track_to_matrix(track: np.ndarray) -> np.ndarray:
    """
    将多通道时序航迹转换为二维矩阵（与原先 ``points*3`` 思路相同，现为 ``points*NUM_TRACK_CHANNELS``）。

    1. ``preprocess_track_channels``
    2. 按时间展开为一维向量 v，长度 N = points * NUM_TRACK_CHANNELS
    3. 复制 v 得到 (N, N) 矩阵
    """
    if track.ndim != 2 or track.shape[1] != NUM_TRACK_CHANNELS:
        raise ValueError(f"track shape must be (points, {NUM_TRACK_CHANNELS}), got {track.shape}")

    processed = preprocess_track_channels(track)
    v = processed.reshape(-1).astype(np.float32)
    n = v.shape[0]
    matrix = np.tile(v[None, :], (n, 1))
    return matrix


def batch_tracks_to_matrices(tracks: List[TrackSample]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 TrackSample 列表转换为 (X, y)。

    X: (batch, 1, H, W)  兼容旧版二维矩阵输入
    y: (batch,)
    """
    matrices: List[np.ndarray] = []
    labels: List[int] = []

    for sample in tracks:
        mat = track_to_matrix(sample.track)
        matrices.append(mat[None, :, :])
        labels.append(sample.label)

    x = np.stack(matrices, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y


def batch_tracks_to_sequences(tracks: List[TrackSample]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 TrackSample 列表转换为 Transformer 输入序列。

    X: (batch, T, C)，其中 C = NUM_TRACK_CHANNELS
    y: (batch,)
    """
    seqs: List[np.ndarray] = []
    labels: List[int] = []

    for sample in tracks:
        seq = preprocess_track_channels(sample.track)
        seqs.append(seq)
        labels.append(sample.label)

    x = np.stack(seqs, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y

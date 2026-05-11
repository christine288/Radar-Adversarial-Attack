"""
从 MATLAB v7.3（HDF5）.mat 文件加载航迹，转为 TrackSample。

主序列``track`` 形状为 ``(T, NUM_TRACK_CHANNELS)``，列顺序与 ``data_utils.TRACK_CHANNEL_KEYS`` 一致，
**全部字段**经预处理后进入 Transformer，用于训练与测试：

V_m, R_m, A_m（测量）；V, R, A（关联）；E_m, E；DATA_time；GPS_time_in_data；Iframecnt；SNR
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from data_utils import TRACK_CHANNEL_KEYS, TrackSample, resample_track

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("读取 v7.3 .mat 需要安装 h5py: pip install h5py") from e

# 与 data_utils.TRACK_CHANNEL_KEYS 相同顺序，便于外部 `from mat_loader import ...` 引用
TR_FIELD_KEYS: Tuple[str, ...] = TRACK_CHANNEL_KEYS


def _read_dataset(f: "h5py.File", key: str) -> np.ndarray:
    if key not in f:
        raise KeyError(f"文件中不存在数据集 '{key}'，可用键: {list(f.keys())}")
    arr = np.array(f[key])
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def inspect_track_sample(
    sample: TrackSample,
    *,
    max_rows: int = 8,
    mat_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    将 ``track`` 全部主序列列拼成表格打印前几行（需安装 pandas）。

    需要自行核对加载结果时调用；也可通过 ``load_mat_track(..., show_preview=True)`` 自动打印。
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("inspect_track_sample 需要 pandas: pip install pandas") from e

    n, c = sample.track.shape[0], sample.track.shape[1]
    if c != len(TRACK_CHANNEL_KEYS):
        raise ValueError(f"track 列数 {c} 与 TRACK_CHANNEL_KEYS 长度不一致")
    cols = {TRACK_CHANNEL_KEYS[i]: sample.track[:, i] for i in range(c)}

    df = pd.DataFrame(cols)
    banner = "========== MAT track 预览 (inspect_track_sample) =========="
    path_line = f"文件: {mat_path}" if mat_path else "(未指定路径)"
    body = df.head(int(max_rows)).to_string()
    print(banner, path_line, body, sep="\n", flush=True)


def _h5_key_per_column(keys: Tuple[str, str, str]) -> Tuple[str, ...]:
    """前三列可用 ``keys`` 覆盖（兼容非标命名），其余列固定为 ``TRACK_CHANNEL_KEYS[3:]``。"""
    return (keys[0], keys[1], keys[2]) + tuple(TRACK_CHANNEL_KEYS[3:])


def load_mat_track(
    mat_path: Union[str, Path],
    keys: Tuple[str, str, str] = ("V_m", "R_m", "A_m"),
    range_scale: float = 1000.0,
    range_is_km: Optional[bool] = None,
    target_points: Optional[int] = None,
    label: int = 0,
    show_preview: bool = False,
    preview_rows: int = 8,
) -> TrackSample:
    """
    读取单个 .mat 文件为一条航迹；``track`` 含 ``TRACK_CHANNEL_KEYS`` 全部列，供训练/测试统一使用。

    Parameters
    ----------
    keys
        前三列 (速度测、距离测、方位测) 的 HDF5 数据集名；后 9 列名固定。
    range_scale / range_is_km
        ``R_m`` 与 ``R``（距离关联）在视为 km 时同乘 ``range_scale`` 转为米。
    target_points
        若非 None，对 ``track`` 整表插值到该长度（与 Transformer 时序长度一致）。
    show_preview
        为 True 时在控制台打印表格前 ``preview_rows`` 行。
    """
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(str(mat_path))

    h5_names = _h5_key_per_column(keys)
    columns: List[np.ndarray] = []
    with h5py.File(mat_path, "r") as f:
        for name in h5_names:
            columns.append(_read_dataset(f, name))

    n = min(len(col) for col in columns)
    if n < 2:
        raise ValueError(f"有效点数过少: {n}")

    track = np.stack([col[:n].astype(np.float64, copy=False) for col in columns], axis=1)

    if range_is_km is None:
        range_is_km = max(float(np.max(track[:, 1])), float(np.max(track[:, 4]))) < 100.0
    if range_is_km:
        track[:, 1] *= float(range_scale)
        track[:, 4] *= float(range_scale)

    track = track.astype(np.float32)

    if target_points is not None:
        track = resample_track(track, target_points)

    sample = TrackSample(track=track, label=label)
    if show_preview:
        inspect_track_sample(sample, max_rows=preview_rows, mat_path=mat_path)
    return sample


def load_mat_directory(
    root: Union[str, Path],
    target_points: int = 32,
    keys: Tuple[str, str, str] = ("V_m", "R_m", "A_m"),
    range_scale: float = 1000.0,
    range_is_km: Optional[bool] = None,
    extensions: Sequence[str] = (".mat",),
    show_preview: bool = False,
    preview_rows: int = 8,
) -> Tuple[List[TrackSample], Dict[str, int]]:
    """
    从目录加载多条航迹，用于训练/测试。

    目录结构（推荐，多类别）::

        root/
          class_a/
            a1.mat
          class_b/
            b1.mat

    若 root 下直接是 .mat 文件（无子目录），则全部标签为 0，并打印警告。

    show_preview
        为 True 时，仅在加载结束后对**第一条成功加载**的样本调用 ``inspect_track_sample``，
        避免打印成千上万行。

    Returns
    -------
    samples, class_to_idx
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    samples: List[TrackSample] = []
    skipped: List[str] = []
    class_to_idx: Dict[str, int] = {}
    first_loaded_path: Optional[Path] = None

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    mat_files_flat = [p for p in root.iterdir() if p.suffix.lower() in extensions]

    if subdirs:
        subdirs = sorted(subdirs, key=lambda p: p.name)
        class_to_idx = {p.name: i for i, p in enumerate(subdirs)}
        for sd in subdirs:
            label = class_to_idx[sd.name]
            for fp in sorted(sd.glob("*.mat")):
                if fp.suffix.lower() not in extensions:
                    continue
                try:
                    samples.append(
                        load_mat_track(
                            fp,
                            keys=keys,
                            range_scale=range_scale,
                            range_is_km=range_is_km,
                            target_points=target_points,
                            label=label,
                            show_preview=False,
                        )
                    )
                    if first_loaded_path is None:
                        first_loaded_path = Path(fp)
                except OSError as e:
                    # h5py 只能读 MATLAB v7.3(HDF5)；若不是该格式或文件损坏，会在这里报错
                    skipped.append(str(fp))
                    import warnings

                    warnings.warn(
                        f"跳过无法用 h5py 打开的 .mat（非 v7.3 或损坏）: {fp}. 错误: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
                except Exception as e:  # pragma: no cover
                    skipped.append(str(fp))
                    import warnings

                    warnings.warn(
                        f"跳过读取失败的 .mat: {fp}. 错误: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
    elif mat_files_flat:
        import warnings

        warnings.warn(
            f"目录 {root} 下没有子文件夹，所有 .mat 将使用标签 0。"
            "多类别请使用 root/类别名/*.mat 结构。",
            UserWarning,
            stacklevel=2,
        )
        for fp in sorted(mat_files_flat):
            samples.append(
                load_mat_track(
                    fp,
                    keys=keys,
                    range_scale=range_scale,
                    range_is_km=range_is_km,
                    target_points=target_points,
                    label=0,
                    show_preview=False,
                )
            )
            if first_loaded_path is None:
                first_loaded_path = Path(fp)
        class_to_idx = {"default": 0}
    else:
        raise FileNotFoundError(f"在 {root} 下未找到子目录或 .mat 文件")

    if not samples:
        raise RuntimeError(f"未从 {root} 加载到任何 .mat 航迹")

    if skipped:
        import warnings

        warnings.warn(
            f"从 {root} 跳过 {len(skipped)} 个无法读取的 .mat 文件（通常不是 v7.3 或已损坏）。",
            UserWarning,
            stacklevel=2,
        )

    if show_preview and samples and first_loaded_path is not None:
        print(
            "\n========== MAT track 预览 (load_mat_directory / show_preview) ==========\n"
            "下列为目录中**第一条成功加载**的样本（非排序后的第一条文件名）。\n",
            flush=True,
        )
        inspect_track_sample(samples[0], max_rows=preview_rows, mat_path=first_loaded_path)

    return samples, class_to_idx


def train_val_split(
    samples: List[TrackSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[TrackSample], List[TrackSample]]:
    """
    训练 / 验证划分。

    在样本量足够时，**按类别分层**，并保证：
    1) 训练集里每个类别至少 1 条；
    2) 验证集里每个类别也尽量至少 1 条（当 val 样本量与各类样本数允许时）。

    若总样本过少，无法同时满足「验证至少 1 条」与「训练覆盖全部类」，则回退为随机划分并告警。
    """
    import warnings
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    n = len(samples)
    if n == 1:
        return samples, samples.copy()

    n_val_target = max(1, int(n * val_ratio))
    n_train_target = n - n_val_target

    by_label: Dict[int, List[TrackSample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)
    for lb in by_label:
        rng.shuffle(by_label[lb])

    labels_sorted = sorted(by_label.keys())
    n_classes = len(labels_sorted)

    def _random_split_fallback() -> Tuple[List[TrackSample], List[TrackSample]]:
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val = [samples[int(i)] for i in val_idx]
        train = [samples[int(i)] for i in train_idx]
        if not train:
            train = [val[0]]
            val = val[1:] if len(val) > 1 else train
        return train, val

    # 训练集至少要能覆盖每个类别 1 条，否则模型根本学不到该类
    if n_train_target < n_classes:
        warnings.warn(
            f"样本过少 (n={n}，约 {n_val_target} 条验证): 无法在训练集中同时覆盖全部 {n_classes} 个类别。"
            "已回退为随机划分；请增加每类 .mat 数量。",
            UserWarning,
            stacklevel=2,
        )
        return _random_split_fallback()

    train: List[TrackSample] = []
    val: List[TrackSample] = []

    # 第 1 步：训练集每类至少 1 条
    for lb in labels_sorted:
        lst = by_label[lb]
        train.append(lst.pop(0))

    # 第 2 步：如果可行，验证集每类至少 1 条
    # 可行条件：
    # - val 目标样本数至少能容纳所有类别
    # - 每个类别在拿走 1 条训练样本后仍有剩余样本
    can_cover_all_classes_in_val = (
        n_val_target >= n_classes and all(len(by_label[lb]) >= 1 for lb in labels_sorted)
    )
    if can_cover_all_classes_in_val:
        for lb in labels_sorted:
            lst = by_label[lb]
            val.append(lst.pop(0))

    pool: List[TrackSample] = []
    for lb in labels_sorted:
        pool.extend(by_label[lb])

    rng.shuffle(pool)
    need_train = n_train_target - len(train)
    need_val = n_val_target - len(val)
    assert need_train >= 0 and need_val >= 0

    train.extend(pool[:need_train])
    val.extend(pool[need_train : need_train + need_val])

    assert len(train) == n_train_target and len(val) == n_val_target
    return train, val

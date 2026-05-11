"""
将 DAUR 轨迹类 v7.3（HDF5）.mat 中的时序字段读入内存，整理为表格并写入文件。

默认导出：
- 轨迹表：每行一个采样点（点序号与各字段一列），UTF-8 CSV（带 BOM，便于 Excel 打开）
- 可选：File_head 标量元数据，单独一个单行 CSV

用法示例：
python analyze_mat_to_table.py "project_data\mat_train\bird\20230619105927_DAUR_TR_Bird_04_25713.mat" -o "output\1_track_table.csv"
python analyze_mat_to_table.py path/to/track.mat
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

# 与 listH5mat / 数据集说明中 TR 字段一致（根节点下的 Dataset 名）
DEFAULT_TR_FIELDS: List[str] = [
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


def _as_1d_double(dset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dset[()], dtype=np.float64).reshape(-1)


def read_root_field(f: h5py.File, name: str) -> Optional[np.ndarray]:
    if name not in f:
        return None
    obj = f[name]
    if not isinstance(obj, h5py.Dataset):
        return None
    return _as_1d_double(obj)


def read_tr_table(
    mat_path: Path,
    field_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """读取多条根级时序字段，对齐到最短公共长度（避免个别字段长度不一致）。"""
    names = field_names or DEFAULT_TR_FIELDS
    columns: Dict[str, np.ndarray] = {}

    with h5py.File(mat_path, "r") as f:
        lengths: List[int] = []
        for name in names:
            arr = read_root_field(f, name)
            if arr is None or arr.size == 0:
                continue
            columns[name] = arr
            lengths.append(int(arr.size))

        if not columns:
            raise ValueError(f"未在 {mat_path} 根节点找到任何已知轨迹字段: {names}")

        n = min(lengths)
        if not lengths or max(lengths) != min(lengths):
            import warnings

            warnings.warn(
                f"{mat_path.name}: 各字段长度不一致 {lengths}，已截断到最短长度 {n} 行对齐。",
                UserWarning,
                stacklevel=2,
            )

        data = {k: v[:n].astype(float) for k, v in columns.items()}

    df = pd.DataFrame(data)
    df.insert(0, "point_index", np.arange(1, n + 1, dtype=np.int64))
    return df


def read_file_head_row(mat_path: Path) -> Optional[pd.DataFrame]:
    """读取 /File_head 下标量数据集，组成单行宽表。"""
    row: Dict[str, float] = {}

    with h5py.File(mat_path, "r") as f:
        if "File_head" not in f:
            return None
        g = f["File_head"]
        for k in sorted(g.keys()):
            obj = g[k]
            if not isinstance(obj, h5py.Dataset):
                continue
            arr = np.asarray(obj[()], dtype=np.float64).reshape(-1)
            if arr.size == 0:
                continue
            if arr.size > 1:
                import warnings

                warnings.warn(
                    f"File_head/{k} 长度 {arr.size}>1，只取首元素写入元数据表。",
                    UserWarning,
                    stacklevel=2,
                )
            row[k] = float(arr.flat[0])

    if not row:
        return None
    return pd.DataFrame([row])


def export_mat_to_csv(
    mat_path: Path,
    output_csv: Path,
    *,
    include_file_head: bool = True,
    field_names: Optional[List[str]] = None,
) -> None:
    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = read_tr_table(mat_path, field_names=field_names)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if include_file_head:
        meta = read_file_head_row(mat_path)
        if meta is not None:
            meta_path = output_csv.with_name(output_csv.stem + "_file_head.csv")
            meta.to_csv(meta_path, index=False, encoding="utf-8-sig")


def _batch_process(input_dir: Path, output_dir: Path, **kwargs) -> None:
    mats = sorted(input_dir.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"{input_dir} 下未找到 .mat")
    output_dir.mkdir(parents=True, exist_ok=True)
    for mp in mats:
        out = output_dir / (mp.stem + "_track_table.csv")
        export_mat_to_csv(mp, out, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="MAT 轨迹字段 → CSV 表格")
    parser.add_argument(
        "path",
        type=str,
        help="单个 .mat 文件路径，或（配合 --batch）包含多个 .mat 的目录",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 CSV 路径；未指定时与 .mat 同目录，文件名为 <stem>_track_table.csv",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="path 为目录时，批量处理其中所有 .mat，输出到 --output 目录",
    )
    parser.add_argument(
        "--no-file-head",
        action="store_true",
        help="不写出 File_head 元数据 CSV（*_file_head.csv）",
    )
    args = parser.parse_args()

    p = Path(args.path).resolve()
    include_meta = not args.no_file_head

    if args.batch:
        if not p.is_dir():
            raise NotADirectoryError(f"--batch 需要目录: {p}")
        out_dir = Path(args.output).resolve() if args.output else p.parent / "mat_tables"
        _batch_process(p, out_dir, include_file_head=include_meta)
        print(f"已批量写出到: {out_dir}")
        return

    if not p.is_file():
        raise FileNotFoundError(f"不是文件: {p}")

    if args.output:
        out = Path(args.output)
    else:
        out = p.parent / f"{p.stem}_track_table.csv"

    export_mat_to_csv(p, out, include_file_head=include_meta)
    print(f"轨迹表: {out.resolve()}")
    if include_meta and read_file_head_row(p) is not None:
        print(f"元数据表: {out.with_name(out.stem + '_file_head.csv').resolve()}")


if __name__ == "__main__":
    main()

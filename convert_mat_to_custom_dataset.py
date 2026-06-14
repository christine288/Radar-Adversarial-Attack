from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from data_utils import TRACK_CHANNEL_KEYS


DATASETS = (
    ("project_data/mat_train_augmented", "project_data/custom_train"),
    ("project_data/mat_test_augmented", "project_data/custom_test"),
)


def _safe_id(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_")


def _read_vector(handle: h5py.File, key: str) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"missing key {key}")
    return np.asarray(handle[key], dtype=np.float64).reshape(-1)


def _mat_to_points(mat_path: Path) -> np.ndarray:
    return _track_to_points(_mat_to_track(mat_path))


def _mat_to_track(mat_path: Path) -> np.ndarray:
    with h5py.File(mat_path, "r") as handle:
        columns = [_read_vector(handle, key) for key in TRACK_CHANNEL_KEYS]
    n = min(col.size for col in columns)
    if n < 1:
        raise ValueError(f"empty track: {mat_path}")
    track = np.stack([col[:n].astype(np.float64, copy=False) for col in columns], axis=1)
    if max(float(np.nanmax(track[:, 1])), float(np.nanmax(track[:, 4]))) < 100.0:
        track[:, 1] *= 1000.0
        track[:, 4] *= 1000.0
    return track.astype(np.float32)


def _track_to_points(track: np.ndarray) -> np.ndarray:
    radius = track[:, 1].astype(np.float64, copy=False)
    azimuth = np.deg2rad(track[:, 2].astype(np.float64, copy=False))
    elevation = np.deg2rad(track[:, 6].astype(np.float64, copy=False))
    intensity = track[:, 11].astype(np.float64, copy=False)

    cos_el = np.cos(elevation)
    x = radius * cos_el * np.cos(azimuth)
    y = radius * cos_el * np.sin(azimuth)
    z = radius * np.sin(elevation)

    points = np.stack([x, y, z, intensity], axis=1)
    return points.astype(np.float32)


def _write_dataset_yaml(output_root: Path, classes: list[str]) -> None:
    lines = [
        "DATASET: CustomDataset",
        f"POINT_CLOUD_RANGE: [{-2000}, {-2000}, {-2000}, {2000}, {2000}, {2000}]",
        "POINT_FEATURE_ENCODING:",
        "  used_feature_list: [x, y, z, intensity]",
        "CLASS_NAMES:",
    ]
    lines.extend(f"  - {name}" for name in classes)
    lines.append("")
    (output_root / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")


def convert_dataset(source_root: Path, output_root: Path) -> int:
    points_dir = output_root / "points"
    tracks_dir = output_root / "tracks"
    labels_dir = output_root / "labels"
    imagesets_dir = output_root / "ImageSets"

    points_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    classes = [p.name for p in class_dirs]
    frame_ids: list[str] = []

    for class_dir in class_dirs:
        class_id = _safe_id(class_dir.name)
        for mat_path in sorted(class_dir.glob("*.mat")):
            frame_id = f"{class_id}__{mat_path.stem}"
            track = _mat_to_track(mat_path)
            points = _track_to_points(track)
            np.save(tracks_dir / f"{frame_id}.npy", track)
            np.save(points_dir / f"{frame_id}.npy", points)
            (labels_dir / f"{frame_id}.txt").write_text(
                f"0 0 0 0 0 0 0 {class_dir.name}\n",
                encoding="utf-8",
            )
            frame_ids.append(frame_id)

    _write_dataset_yaml(output_root, classes)
    (imagesets_dir / "val.txt").write_text(
        "".join(f"{frame_id}\n" for frame_id in sorted(frame_ids)),
        encoding="utf-8",
    )
    return len(frame_ids)


def main() -> None:
    root = Path(__file__).resolve().parent
    for source, output in DATASETS:
        source_root = root / source
        output_root = root / output
        count = convert_dataset(source_root, output_root)
        print(f"{output}: {count} samples")


if __name__ == "__main__":
    main()

from pathlib import Path
import os


def prepare_dataset(dataset_path: str) -> bool:
    """检查数据集目录结构是否符合规范，并在缺失时生成 `ImageSets/val.txt`。

    支持两种格式：
    - KITTI 格式：`training/velodyne`, `training/calib`, `training/label_2`
    - Custom 格式：存在 `dataset.yaml`，并包含 `points/` 和 `labels/`

    返回:
        bool: 验证并准备成功返回 True，失败返回 False。
    """
    p = Path(dataset_path)
    if not p.exists():
        return False

    # 判断 Custom 格式（通过 dataset.yaml）
    if (p / "dataset.yaml").exists():
        points_dir = p / "points"
        labels_dir = p / "labels"
        if not points_dir.is_dir() or not labels_dir.is_dir():
            return False

        imagesets_dir = p / "ImageSets"
        imagesets_dir.mkdir(exist_ok=True)
        val_file = imagesets_dir / "val.txt"
        if not val_file.exists():
            frames = [f.stem for f in points_dir.iterdir() if f.is_file()]
            frames.sort()
            with val_file.open("w", encoding="utf-8") as fh:
                for fid in frames:
                    fh.write(fid + "\n")
        return True

    # 否则尝试 KITTI 格式检测
    training = p / "training"
    if not training.is_dir():
        return False
    velodyne = training / "velodyne"
    calib = training / "calib"
    label_2 = training / "label_2"
    if not (velodyne.is_dir() and calib.is_dir() and label_2.is_dir()):
        return False

    imagesets_dir = p / "ImageSets"
    imagesets_dir.mkdir(exist_ok=True)
    val_file = imagesets_dir / "val.txt"
    if not val_file.exists():
        frames = [f.stem for f in velodyne.iterdir() if f.is_file()]
        frames.sort()
        with val_file.open("w", encoding="utf-8") as fh:
            for fid in frames:
                fh.write(fid + "\n")
    return True

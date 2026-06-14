from pathlib import Path


def _is_mat_class_dataset(dataset_path: Path) -> bool:
    """MAT 航迹分类：根目录下按类别子目录存放 ``*.mat``。"""
    if not dataset_path.is_dir():
        return False
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not class_dirs:
        return False
    return all(any(cls_dir.glob("*.mat")) for cls_dir in class_dirs)


def prepare_dataset(dataset_path: str) -> bool:
    """检查数据集目录结构是否符合平台约定。

    仅支持 MAT 航迹分类格式：

        <dataset_path>/
          <class_a>/
            *.mat
          <class_b>/
            *.mat

    返回:
        bool: 验证并准备成功返回 True，失败返回 False。
    """
    p = Path(dataset_path)
    if not p.exists():
        return False

    return _is_mat_class_dataset(p)

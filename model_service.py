from pathlib import Path
from typing import Optional

from dataset_service import prepare_dataset


def _resolve_model_path(model_path: str) -> Optional[Path]:
    p = Path(model_path)
    if p.is_file():
        if p.suffix in (".pth", ".pt"):
            return p
        return None
    if p.is_dir():
        candidates = list(p.glob("*.pth")) + list(p.glob("*.pt"))
        if len(candidates) == 1:
            return candidates[0]
        return None
    return None


def _find_model_file_in_dir(d: Path) -> Optional[Path]:
    pths = list(d.glob("*.pth")) + list(d.glob("*.pt"))
    if len(pths) == 1:
        return pths[0]
    return None


def validate_model(model_path: str, dataset_path: str) -> bool:
    """校验模型目录/文件并同时校验数据集。

    规则：
    - 如果传入的是文件，必须以 .pth 或 .pt 结尾
    - 如果传入的是目录，目录下必须有且仅有一个 .pth/.pt 文件
    - data 集必须通过 `prepare_dataset` 校验

    返回:
        bool: 验证成功返回 True，否则 False
    """
    model_file = _resolve_model_path(model_path)
    if model_file is None:
        return False

    try:
        import torch
        from model import RadarTrackTransformer

        try:
            ckpt = torch.load(model_file, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(model_file, map_location="cpu")

        if not isinstance(ckpt, dict):
            return False
        if "input_size" not in ckpt or "model_state_dict" not in ckpt:
            return False

        model = RadarTrackTransformer(input_size=ckpt["input_size"], num_classes=ckpt.get("num_classes", 6))
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        return False

    return prepare_dataset(dataset_path)

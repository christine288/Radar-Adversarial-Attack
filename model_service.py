from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from dataset_service import prepare_dataset


DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "model_type": "transformer",
    "input_size": (32, 12),
    "num_classes": 6,
    "dropout": 0.2,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 256,
}


def _resolve_model_file(model_path: str) -> Optional[Path]:
    """Resolve the single model file from a platform model directory."""
    p = Path(model_path)
    if not p.is_dir():
        return None

    candidates = sorted(
        [item for item in p.iterdir() if item.is_file() and item.suffix.lower() in {".pth", ".pt"}]
    )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return [_parse_scalar(item) for item in items]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Read the flat model.yaml keys used by this service without requiring PyYAML."""
    config: Dict[str, Any] = {}
    if not yaml_path.is_file():
        return config

    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line or line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            config[key] = _parse_scalar(value)
    return config


def _normalise_input_size(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = _parse_scalar(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _load_checkpoint(model_file: Path, map_location: str = "cpu") -> Optional[Mapping[str, Any]]:
    try:
        import torch

        try:
            checkpoint = torch.load(model_file, map_location=map_location, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_file, map_location=map_location)
    except Exception:
        return None

    if not isinstance(checkpoint, Mapping):
        return None
    return checkpoint


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, Mapping):
        return state_dict

    # Pure state_dict files are also valid when model.yaml or defaults provide the architecture.
    if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
        return checkpoint
    return None


def _infer_transformer_config(state_dict: Mapping[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    inferred = dict(config)

    input_weight = state_dict.get("input_proj.weight")
    if input_weight is not None and hasattr(input_weight, "shape") and len(input_weight.shape) == 2:
        inferred["d_model"] = int(input_weight.shape[0])
        inferred.setdefault("input_size", (32, int(input_weight.shape[1])))

    head_weight = state_dict.get("head.3.weight")
    if head_weight is not None and hasattr(head_weight, "shape") and len(head_weight.shape) == 2:
        inferred["num_classes"] = int(head_weight.shape[0])

    layer_ids = []
    prefix = "encoder.layers."
    for key in state_dict.keys():
        if isinstance(key, str) and key.startswith(prefix):
            rest = key[len(prefix) :]
            layer = rest.split(".", 1)[0]
            if layer.isdigit():
                layer_ids.append(int(layer))
    if layer_ids:
        inferred["num_layers"] = max(layer_ids) + 1

    ff_weight = state_dict.get("encoder.layers.0.linear1.weight")
    if ff_weight is not None and hasattr(ff_weight, "shape") and len(ff_weight.shape) == 2:
        inferred["dim_feedforward"] = int(ff_weight.shape[0])

    return inferred


def _normalise_transformer_checkpoint(
    model_file: Path,
    config: Optional[Dict[str, Any]] = None,
    map_location: str = "cpu",
) -> Optional[Tuple[Mapping[str, Any], Dict[str, Any]]]:
    checkpoint = _load_checkpoint(model_file, map_location=map_location)
    if checkpoint is None:
        return None

    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        return None

    merged = dict(DEFAULT_MODEL_CONFIG)
    if config:
        merged.update(config)
    merged.update({key: checkpoint[key] for key in ("input_size", "num_classes", "model_type") if key in checkpoint})
    merged = _infer_transformer_config(state_dict, merged)

    if str(merged.get("model_type", "transformer")).lower() not in {"transformer", "radartracktransformer"}:
        return None

    input_size = _normalise_input_size(merged.get("input_size"))
    if input_size is None:
        return None
    merged["input_size"] = input_size
    return state_dict, merged


def load_radar_transformer_model(
    model_file: Union[str, Path],
    device: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
):
    """Load a checkpoint whose state_dict is compatible with RadarTrackTransformer."""
    model_file = Path(model_file)
    normalised = _normalise_transformer_checkpoint(model_file, config=config, map_location=device)
    if normalised is None:
        raise RuntimeError("Model checkpoint is not compatible with RadarTrackTransformer")

    state_dict, merged = normalised
    from model import RadarTrackTransformer

    model = RadarTrackTransformer(
        input_size=merged["input_size"],
        num_classes=int(merged.get("num_classes", DEFAULT_MODEL_CONFIG["num_classes"])),
        dropout=float(merged.get("dropout", DEFAULT_MODEL_CONFIG["dropout"])),
        d_model=int(merged.get("d_model", DEFAULT_MODEL_CONFIG["d_model"])),
        nhead=int(merged.get("nhead", DEFAULT_MODEL_CONFIG["nhead"])),
        num_layers=int(merged.get("num_layers", DEFAULT_MODEL_CONFIG["num_layers"])),
        dim_feedforward=int(merged.get("dim_feedforward", DEFAULT_MODEL_CONFIG["dim_feedforward"])),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _validate_optional_interface(model_dir: Path) -> bool:
    interface_path = model_dir / "my_interface.py"
    if not interface_path.is_file():
        return True

    try:
        spec = importlib.util.spec_from_file_location("_user_model_interface", interface_path)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return False
    return hasattr(module, "ModelInterface")


def _validate_dataset(dataset_path: str) -> bool:
    return prepare_dataset(dataset_path)


def _validate_transformer_checkpoint(model_file: Path, config: Dict[str, Any]) -> bool:
    try:
        load_radar_transformer_model(model_file, device="cpu", config=config)
    except Exception:
        return False
    return True


def validate_model(model_path: str, dataset_path: str) -> bool:
    """Validate the model package/file and the dataset.

    Platform model directory format:
        model_dir/
          model.pth       required; exactly one .pth or .pt file
          model.yaml      optional; flat keys such as input_size and num_classes
          my_interface.py optional; if present, it must define ModelInterface
    """
    model_file = _resolve_model_file(model_path)
    if model_file is None:
        return False

    model_root = Path(model_path)
    if not _validate_optional_interface(model_root):
        return False

    config = _load_simple_yaml(model_root / "model.yaml")
    if not _validate_transformer_checkpoint(model_file, config):
        return False

    return _validate_dataset(dataset_path)

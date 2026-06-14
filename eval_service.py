from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from dataset_service import prepare_dataset
from model_service import _resolve_model_file, validate_model

INTERFACE_EVAL_RESULT_FILE = "_interface_eval_result.json"


def _attack_success_rate(y_true: List[int], y_pred_clean: List[int], y_pred_adv: List[int]) -> float:
    if not y_true or len(y_true) != len(y_pred_clean) or len(y_true) != len(y_pred_adv):
        return 0.0
    clean_correct_idx = [i for i, (yt, pc) in enumerate(zip(y_true, y_pred_clean)) if yt == pc]
    if not clean_correct_idx:
        return 0.0
    success = sum(1 for i in clean_correct_idx if y_pred_adv[i] != y_true[i])
    return float(success / len(clean_correct_idx))


def _format_result(clean_metrics: Dict, adv_metrics: Dict, attack_success: float) -> Dict:
    clean_map = float(clean_metrics.get("mAP", 0.0))
    adv_map = float(adv_metrics.get("mAP", 0.0))
    clean_acc = float(clean_metrics.get("accuracy", 0.0))
    adv_acc = float(adv_metrics.get("accuracy", 0.0))
    clean_f1 = float(clean_metrics.get("f1_macro", 0.0))
    adv_f1 = float(adv_metrics.get("f1_macro", 0.0))
    retention = adv_map / clean_map if clean_map > 0 else 0.0
    return {
        "clean": {
            "accuracy": clean_acc,
            "macro_f1": clean_f1,
            "classification_mAP": clean_map,
        },
        "Adversarial": {
            "accuracy": adv_acc,
            "macro_f1": adv_f1,
            "classification_mAP": adv_map,
            "mAP_retention_rate": retention,
            "attack_success_rate": float(attack_success),
            "robust_accuracy": adv_acc,
        },
    }


def _evaluate_mat_robustness(model_file: Path, dataset_path: str, adv_dataset_path: str) -> Dict:
    from evaluate import evaluate_model

    clean_result = evaluate_model(
        model_path=str(model_file),
        device="cpu",
        mat_test_dir=dataset_path,
        show_confusion=False,
        return_predictions=True,
        verbose=False,
    )
    adv_result = evaluate_model(
        model_path=str(model_file),
        device="cpu",
        mat_test_dir=adv_dataset_path,
        show_confusion=False,
        return_predictions=True,
        verbose=False,
    )
    clean_metrics = clean_result["metrics"]
    adv_metrics = adv_result["metrics"]
    attack_success = _attack_success_rate(
        clean_result["y_true"],
        clean_result["y_pred"],
        adv_result["y_pred"],
    )
    return _format_result(clean_metrics, adv_metrics, attack_success)


def evaluate_robustness(model_path: str, dataset_path: str, adv_dataset_path: str) -> Dict:
    """Evaluate clean and adversarial robustness for ``<dataset_path>/<class_name>/*.mat`` datasets."""
    clean_ds = Path(dataset_path)
    adv_ds = Path(adv_dataset_path)
    if not clean_ds.exists() or not adv_ds.exists():
        raise RuntimeError("dataset_path and adv_dataset_path must both exist")
    if not prepare_dataset(str(clean_ds)):
        raise RuntimeError("Clean dataset format is invalid. Expected <dataset_path>/<class_name>/*.mat")
    if not prepare_dataset(str(adv_ds)):
        raise RuntimeError("Adversarial dataset format is invalid. Expected <adv_dataset_path>/<class_name>/*.mat")
    if not validate_model(model_path, dataset_path):
        raise RuntimeError("Model or clean dataset validation failed")

    interface_eval_path = adv_ds / INTERFACE_EVAL_RESULT_FILE
    if interface_eval_path.is_file():
        with interface_eval_path.open(encoding="utf-8") as fh:
            return json.load(fh)

    model_file = _resolve_model_file(model_path)
    if model_file is None:
        raise RuntimeError("model_path must be a directory with exactly one .pth/.pt file")

    return _evaluate_mat_robustness(model_file, dataset_path, adv_dataset_path)

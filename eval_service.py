from pathlib import Path
from typing import Dict


def _attack_success_rate(y_true, y_pred_clean, y_pred_adv) -> float:
    if not y_true or len(y_true) != len(y_pred_clean) or len(y_true) != len(y_pred_adv):
        return 0.0
    clean_correct_idx = [i for i, (yt, pc) in enumerate(zip(y_true, y_pred_clean)) if yt == pc]
    if not clean_correct_idx:
        return 0.0
    success = sum(1 for i in clean_correct_idx if y_pred_adv[i] != y_true[i])
    return float(success / len(clean_correct_idx))


def evaluate_robustness(model_path: str, dataset_path: str, adv_dataset_path: str) -> Dict:
    """对接仓库内的鲁棒性评估实现，优先使用 `robust_evaluate_enhanced`。

    行为：
    - 从 `mat_loader.load_mat_directory` 加载数据集样本
    - 使用 `robust_evaluate_enhanced.predict_for_condition`（或 `robust_evaluate.predict_for_condition` 回退）
    - 计算并返回符合接口文档的 `result_dict`
    """
    try:
        from evaluate import evaluate_model

        clean_result = evaluate_model(
            model_path=model_path,
            device="cpu",
            mat_test_dir=dataset_path,
            show_confusion=False,
            return_predictions=True,
        )
        adv_result = evaluate_model(
            model_path=model_path,
            device="cpu",
            mat_test_dir=adv_dataset_path,
            show_confusion=False,
            return_predictions=True,
        )
        clean_metrics = clean_result["metrics"]
        adv_metrics = adv_result["metrics"]
        attack_success = _attack_success_rate(
            clean_result["y_true"],
            clean_result["y_pred"],
            adv_result["y_pred"],
        )
        return {
            "clean": {"mAP_3D": clean_metrics.get("mAP", 0.0)},
            "Adversarial": {
                "mAP_3D": adv_metrics.get("mAP", 0.0),
                "mAP_drop": clean_metrics.get("mAP", 0.0) - adv_metrics.get("mAP", 0.0),
                "mAP_retention_rate": (adv_metrics.get("mAP", 0.0) / clean_metrics.get("mAP", 1.0)) if clean_metrics.get("mAP", 0.0) else 0.0,
                "attack_success_rate": attack_success,
            },
        }
    except Exception:
        return {"clean": {"mAP_3D": 0.0}, "Adversarial": {"mAP_3D": 0.0, "mAP_drop": 0.0, "mAP_retention_rate": 0.0, "attack_success_rate": 0.0}}

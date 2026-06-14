"""测试入口：在 MAT 测试集上评估已训练 Transformer 的有效性指标。"""

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data_utils import TrackSample, batch_tracks_to_sequences
from device_utils import print_device_banner, resolve_device_str
from metrics_utils import compute_multiclass_map
from metrics_utils import format_pct
from model_service import load_radar_transformer_model


def build_test_loader(samples: List[TrackSample], batch_size: int = 200) -> DataLoader:
    x, y = batch_tracks_to_sequences(samples)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _class_names_from_meta(meta: Dict, num_classes: int) -> List[str]:
    m = meta.get("mat_class_map")
    if not isinstance(m, dict) or not m:
        return [str(i) for i in range(num_classes)]
    inv = {int(v): k for k, v in m.items()}
    return [inv.get(i, str(i)) for i in range(num_classes)]


def print_confusion_and_report(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
    class_names: List[str],
) -> None:
    """
    混淆矩阵：行 = 真实类别，列 = 预测类别。
    - 若某一列几乎全是「预测为该类的样本数」，其余列接近 0 → 常预测成那一类（类别不平衡或模型偷懒）。
    - 若某一行非对角线全为 0、或某类 recall 为 0 → 该类几乎没被识别对（划分/标签/样本过少）。
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    try:
        import pandas as pd

        idx = [f"真:{class_names[i]}" for i in range(num_classes)]
        col = [f"测:{class_names[j]}" for j in range(num_classes)]
        print("\n混淆矩阵（行=真实，列=预测；对角线为判对）:")
        print(pd.DataFrame(cm, index=idx, columns=col).to_string())
    except Exception:  # pragma: no cover
        print("\n混淆矩阵（行=真实，列=预测）:")
        print(cm)
    print("\n按类分类报告（support=该真实类样本数）:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )


def compute_metrics_from_preds(y_true, y_pred, y_score, num_classes: int) -> Dict[str, float]:
    # 使用 sklearn 的多分类 one-vs-rest 统计方式；zero_division=0 避免极端小样本时报错
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f1_micro),
        "mAP": compute_multiclass_map(y_true, y_score, num_classes=num_classes),
    }


def evaluate_model(
    model_path: str,
    batch_size: int = 200,
    device: str = "cpu",
    mat_test_dir: str = "",
    show_confusion: bool = False,
    return_predictions: bool = False,
    verbose: bool = True,
) -> Any:
    model = load_radar_transformer_model(model_path, device=device)
    num_classes = int(model.head[-1].out_features)

    if not mat_test_dir:
        raise ValueError("必须提供 --mat_test_dir 或 --mat_dir（仅测试集目录）")

    from mat_loader import load_mat_directory

    meta_path = str(model_path) + ".meta.json"
    tgt = 32
    meta: Dict = {}
    if Path(meta_path).is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tgt = int(meta.get("mat_target_points", 32))
    test_samples, _ = load_mat_directory(mat_test_dir, target_points=tgt)
    test_loader = build_test_loader(test_samples, batch_size=min(batch_size, max(1, len(test_samples))))

    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[List[float]] = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())

    metrics = compute_metrics_from_preds(y_true, y_pred, y_score, num_classes=num_classes)
    if verbose:
        print(
            "测试集有效性指标: "
            f"Acc(准确率)={metrics['accuracy']:.4f}, "
            f"P_macro(宏平均精确率)={metrics['precision_macro']:.4f}, "
            f"R_macro(宏平均召回率)={metrics['recall_macro']:.4f}, "
            f"F1_macro(宏平均F1)={metrics['f1_macro']:.4f}, "
            f"P_micro(微平均精确率)={metrics['precision_micro']:.4f}, "
            f"R_micro(微平均召回率)={metrics['recall_micro']:.4f}, "
            f"F1_micro(微平均F1)={metrics['f1_micro']:.4f}"
        )
        print(
            "测试集百分比指标: "
            f"Acc={format_pct(metrics['accuracy'])}, "
            f"mAP={format_pct(metrics['mAP'])}, "
            f"P_macro={format_pct(metrics['precision_macro'])}, "
            f"R_macro={format_pct(metrics['recall_macro'])}, "
            f"F1_macro={format_pct(metrics['f1_macro'])}, "
            f"P_micro={format_pct(metrics['precision_micro'])}, "
            f"R_micro={format_pct(metrics['recall_micro'])}, "
            f"F1_micro={format_pct(metrics['f1_micro'])}"
        )
    if show_confusion:
        names = _class_names_from_meta(meta, num_classes)
        print_confusion_and_report(y_true, y_pred, num_classes, names)

    if return_predictions:
        return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate radar track Transformer (MAT test set only)")
    parser.add_argument("--model_path", type=str, default="radar_transformer.pth")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda", help="默认 cuda；无 CUDA 回退 CPU")
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument(
        "--mat_test_dir",
        type=str,
        default=None,
        help="仅放测试用 .mat 的根目录（与训练目录分开）",
    )
    parser.add_argument("--mat_dir", type=str, default=None, help="同 --mat_test_dir")
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="打印混淆矩阵与按类报告（查类别不平衡/某类永远错）",
    )
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    mat_test_root = args.mat_test_dir or args.mat_dir
    if not mat_test_root:
        parser.error("必须指定 --mat_test_dir 或 --mat_dir")

    device = resolve_device_str(args.device, require_gpu=args.require_gpu)
    print_device_banner(device)

    evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=device,
        mat_test_dir=mat_test_root,
        show_confusion=args.confusion,
    )


if __name__ == "__main__":
    main()

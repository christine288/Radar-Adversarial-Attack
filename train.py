"""训练入口：加载 MAT 数据，切分 train/val，训练并保存最佳 Transformer 权重。"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_utils import TrackSample, batch_tracks_to_sequences
from device_utils import print_device_banner, resolve_device_str
from metrics_utils import compute_multiclass_map
from metrics_utils import format_pct
from model import RadarTrackTransformer


def _balanced_class_weights(
    train_labels: List[int],
    num_classes: int,
    mode: str = "balanced",
) -> torch.Tensor:
    """
    - ``balanced``：与 sklearn 一致，w_c ∝ n / (n_classes * n_c)。
    - ``sqrt``：w_c ∝ sqrt(n / (n_classes * n_c))，对少数类的放大更温和。
    再对「有样本的类」归一化使均值为 1。
    """
    y = np.asarray(train_labels, dtype=np.int64)
    counts = np.bincount(y, minlength=num_classes)
    n = len(y)
    w = np.ones(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            base = n / (num_classes * float(counts[c]))
            if mode == "sqrt":
                w[c] = float(np.sqrt(base))
            else:
                w[c] = base
    active = counts > 0
    if active.any():
        w[active] = w[active] / w[active].mean()
    return torch.tensor(w, dtype=torch.float32)


def _sample_weights_for_sampler(
    train_labels: List[int],
    num_classes: int,
    mode: str = "balanced",
) -> torch.Tensor:
    """采样权重：balanced 用 1/n_c；sqrt 用 sqrt(1/n_c)（更温和）。"""
    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=num_classes)
    if mode == "sqrt":
        sw = np.array([np.sqrt(1.0 / counts[lb]) for lb in train_labels], dtype=np.float64)
    else:
        sw = np.array([1.0 / counts[lb] for lb in train_labels], dtype=np.float64)
    return torch.from_numpy(sw)


def _metric_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "range": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "range": float(arr.max() - arr.min()),
    }


def train_model(
    epochs: int = 100,
    batch_size: int = 200,
    lr: float = 1e-5,
    dropout: float = 0.5,
    device: str = "cpu",
    save_path: str = "radar_transformer.pth",
    mat_dir: str = "",
    mat_target_points: int = 32,
    num_classes: Optional[int] = None,
    mat_val_ratio: float = 0.2,
    split_seed: int = 42,
    mat_show_preview: bool = False,
    max_train_samples: Optional[int] = None,
    class_balance: bool = True,
    class_weight_mode: str = "balanced",
    val_confusion: bool = False,
    round_size: int = 50,
) -> None:
    """仅支持从 MAT 目录加载航迹数据训练（见 README 目录结构）。"""
    from mat_loader import load_mat_directory, train_val_split

    if not mat_dir:
        raise ValueError("必须提供 --mat_train_dir 或 --mat_dir（训练用 .mat 根目录）")

    if mat_show_preview:
        print(
            "[train] 已开启 --mat_show_preview：加载完成后将在下方打印首条样本的 track 表（见 MAT track 预览 横幅）。\n",
            flush=True,
        )

    samples, class_map = load_mat_directory(
        mat_dir,
        target_points=mat_target_points,
        show_preview=mat_show_preview,
    )
    if max_train_samples is not None and max_train_samples > 0:
        if max_train_samples < len(samples):
            samples = samples[:max_train_samples]
            print(
                f"[debug] max_train_samples={max_train_samples}：仅使用前 {len(samples)} 条样本做划分与训练。\n",
                flush=True,
            )
        else:
            print(
                f"[debug] max_train_samples={max_train_samples} >= 已加载 {len(samples)} 条，未截断。\n",
                flush=True,
            )
    train_s, val_s = train_val_split(samples, val_ratio=mat_val_ratio, seed=split_seed)
    # ---- 调试信息：确认标签分布、训练/验证样本数 ----
    train_labels = [s.label for s in train_s]
    val_labels = [s.label for s in val_s]
    n_total = len(samples)
    n_train = len(train_s)
    n_val = len(val_s)
    n_cls_from_data = max(s.label for s in samples) + 1 if samples else 0
    print(
        f"[data] total={n_total}, train={n_train}, val={n_val}, "
        f"class_map={class_map}, max_label+1={n_cls_from_data}"
    )
    # 统计每类样本数（避免 numpy import 额外开销，用 Python 计数即可）
    def _count(labels: List[int]) -> dict:
        d: dict = {}
        for lb in labels:
            d[lb] = d.get(lb, 0) + 1
        return d

    print(f"[data] train label counts: {_count(train_labels)}")
    print(f"[data] val label counts: {_count(val_labels)}")
    inferred = max(s.label for s in samples) + 1
    n_cls = num_classes if num_classes is not None else max(inferred, len(class_map))
    if n_cls < 2:
        raise ValueError(
            "分类任务至少需要 2 个类别。请按「类别文件夹/*.mat」组织数据。"
        )
    eff_batch = min(batch_size, max(1, len(train_s)))
    x_tr, y_tr = batch_tracks_to_sequences(train_s)
    x_va, y_va = batch_tracks_to_sequences(val_s)

    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    cw_used: Optional[torch.Tensor] = None
    if class_balance and len(train_s) >= 2:
        cw = _balanced_class_weights(train_labels, n_cls, mode=class_weight_mode)
        cw_used = cw
        sw = _sample_weights_for_sampler(train_labels, n_cls, mode=class_weight_mode)
        sampler = WeightedRandomSampler(
            weights=sw,
            num_samples=len(train_s),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=eff_batch,
            sampler=sampler,
        )
        print(
            f"[train] 已启用类别不均衡处理（mode={class_weight_mode}）："
            f"加权交叉熵 weight={cw.numpy().round(4).tolist()}；"
            f"WeightedRandomSampler。\n",
            flush=True,
        )
        loss_weight = cw.to(device)
    else:
        train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True)
        loss_weight = None
        if not class_balance:
            print("[train] 已关闭 class_balance（普通 shuffle + 无类别权重）。\n", flush=True)
        elif len(train_s) < 2:
            print("[train] 训练集仅 1 条，跳过类别均衡（与关闭 class_balance 相同）。\n", flush=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
        batch_size=min(eff_batch, max(1, len(val_s))),
        shuffle=False,
    )
    mat_meta = {"mat_class_map": class_map, "mat_target_points": mat_target_points}

    sample_x, _ = next(iter(train_loader))
    _, t, c = sample_x.shape
    model = RadarTrackTransformer(input_size=(t, c), num_classes=n_cls, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight) if loss_weight is not None else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path("dataoutput")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / Path(save_path).name

    best_val_acc = 0.0
    epoch_metrics: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score: List[List[float]] = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                probs = torch.softmax(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(predicted.cpu().numpy().tolist())
                y_score.extend(probs.cpu().numpy().tolist())

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_map = compute_multiclass_map(y_true, y_score, num_classes=n_cls)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        epoch_metrics.append(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "mAP": float(val_map),
                "precision": float(p_macro),
                "recall": float(r_macro),
                "f1": float(f1_macro),
            }
        )

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Acc: {val_acc:.6f} ({val_correct}/{val_total})"
        )
        print(
            f"Val 百分比指标: Acc={format_pct(val_acc)}, "
            f"mAP={format_pct(val_map)}, "
            f"Precision={format_pct(p_macro)}, "
            f"Recall={format_pct(r_macro)}, "
            f"F1={format_pct(f1_macro)}"
        )

        if val_confusion and epoch == epochs:
            from evaluate import _class_names_from_meta, print_confusion_and_report

            vnames = _class_names_from_meta(mat_meta, n_cls)
            print(f"\n[验证集] 最后一轮 Epoch {epoch} 混淆矩阵（可与测试集 evaluate --confusion 对照）:\n", flush=True)
            print_confusion_and_report(y_true, y_pred, n_cls, vnames)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": (t, c),
                    "num_classes": n_cls,
                    "model_type": "transformer",
                },
                checkpoint_path,
            )
            meta_path = str(checkpoint_path) + ".meta.json"
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "mat_class_map": mat_meta["mat_class_map"],
                        "mat_target_points": mat_meta["mat_target_points"],
                        "num_classes": n_cls,
                        "input_size": [t, c],
                        "model_type": "transformer",
                        "class_balance": cw_used is not None,
                        "class_weights_used": cw_used.numpy().tolist() if cw_used is not None else None,
                        "class_weight_mode": class_weight_mode,
                    },
                    mf,
                    ensure_ascii=False,
                    indent=2,
                )

    print(f"训练结束，最佳验证准确率: {best_val_acc:.4f}，模型已保存到 {checkpoint_path}")

    # 按 round_size 将 epoch 分轮统计（例如 100 epoch -> 两轮各 50）
    if round_size > 0 and len(epoch_metrics) > 0:
        print("\n按轮次统计验证准确率：")
        round_acc_means: List[float] = []
        total_epochs = len(epoch_metrics)
        n_rounds = (total_epochs + round_size - 1) // round_size
        for ridx in range(n_rounds):
            s = ridx * round_size
            e = min((ridx + 1) * round_size, total_epochs)
            seg = [m["accuracy"] for m in epoch_metrics[s:e]]
            seg_mean = float(np.mean(seg))
            round_acc_means.append(seg_mean)
            print(f"  第 {ridx + 1} 轮 (Epoch {s + 1}-{e}) 平均准确率: {seg_mean * 100:.2f}%")
        final_round_avg = float(np.mean(round_acc_means))
        print(f"  各轮平均准确率再平均（最终）: {final_round_avg * 100:.2f}%")

    print("\n基于每个 epoch 的验证集结果统计：")
    stats_map = {
        "accuracy": "准确率（Accuracy）",
        "recall": "召回率（Recall）",
        "f1": "F-score",
        "precision": "精确率（Precision）",
    }
    stats_map["mAP"] = "mAP"
    for key in ["precision", "recall", "f1", "accuracy", "mAP"]:
        s = _metric_stats([m[key] for m in epoch_metrics])
        print(f"{stats_map[key]}：")
        print(
            f"  平均值：{s['mean'] * 100:.2f}%  "
            f"标准差：{s['std'] * 100:.2f}%  "
            f"最大值：{s['max'] * 100:.2f}%  "
            f"最小值：{s['min'] * 100:.2f}%  "
            f"范围：{s['range'] * 100:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Radar track Transformer training (MAT only)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="默认 cuda；无 CUDA 时回退 CPU。可设 cpu。",
    )
    parser.add_argument("--require_gpu", action="store_true", help="无 CUDA 则报错退出")
    parser.add_argument("--save_path", type=str, default="radar_transformer.pth")
    parser.add_argument(
        "--mat_train_dir",
        type=str,
        default=None,
        help="训练+验证用 .mat 根目录（与测试目录分开）",
    )
    parser.add_argument(
        "--mat_dir",
        type=str,
        default=None,
        help="同 --mat_train_dir（兼容旧参数）",
    )
    parser.add_argument("--mat_target_points", type=int, default=32)
    parser.add_argument("--mat_val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42, help="训练/验证划分随机种子")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument(
        "--mat_show_preview",
        action="store_true",
        help="加载数据后打印第一条样本的 track 全列表格（调试用）",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="调试：加载后只保留前 N 条样本再划分 train/val（用于 1～2 条过拟合自检）",
    )
    parser.add_argument(
        "--no_class_balance",
        action="store_true",
        help="关闭类别不均衡处理（不加权交叉熵、不用 WeightedRandomSampler）",
    )
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        choices=("balanced", "sqrt"),
        default="balanced",
        help="类别权重与采样强度：balanced=标准逆频率；sqrt=更温和（缓解总猜某一类）",
    )
    parser.add_argument(
        "--val_confusion",
        action="store_true",
        help="训练结束后打印最后一轮验证集混淆矩阵与按类报告（与 evaluate --confusion 格式一致）",
    )
    parser.add_argument(
        "--round_size",
        type=int,
        default=50,
        help="按多少个 epoch 为一轮统计平均验证准确率（默认 50）",
    )
    args = parser.parse_args()

    mat_root = args.mat_train_dir or args.mat_dir
    if not mat_root:
        parser.error("必须指定 --mat_train_dir 或 --mat_dir")

    Path("dataoutput").mkdir(parents=True, exist_ok=True)
    device = resolve_device_str(args.device, require_gpu=args.require_gpu)
    print_device_banner(device)
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        device=device,
        save_path=args.save_path,
        mat_dir=mat_root,
        mat_target_points=args.mat_target_points,
        num_classes=args.num_classes,
        mat_val_ratio=args.mat_val_ratio,
        split_seed=args.split_seed,
        mat_show_preview=args.mat_show_preview,
        max_train_samples=args.max_train_samples,
        class_balance=not args.no_class_balance,
        class_weight_mode=args.class_weight_mode,
        val_confusion=args.val_confusion,
        round_size=args.round_size,
    )


if __name__ == "__main__":
    main()

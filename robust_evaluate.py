"""噪声鲁棒性评估：对比无噪声/高斯噪声/椒盐噪声/斑点噪声下的检测性能。"""

import argparse
import json
from pathlib import Path
from typing import List
from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import recall_score

from device_utils import print_device_banner, resolve_device_str
from noise_perturbation import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise
from data_utils import TrackSample, batch_tracks_to_sequences
from metrics_utils import compute_multiclass_map
from metrics_utils import compute_signal_quality_metrics
from metrics_utils import format_pct
from model import RadarTrackTransformer


def build_loader(samples: List[TrackSample], batch_size: int) -> DataLoader:
    x, y = batch_tracks_to_sequences(samples)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def compute_metrics_from_preds(y_true, y_pred, y_score, num_classes: int) -> Dict[str, float]:
    # 简单目标召回率：用 macro recall 表示对各类别目标的平均召回能力
    target_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    # 样本级准确率：样本（航迹序列）级别预测正确比例
    sample_accuracy = float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)) if y_true else 0.0
    return {
        "target_recall": target_recall,
        "sample_accuracy": sample_accuracy,
        "mAP": compute_multiclass_map(y_true, y_score, num_classes=num_classes),
    }


def load_model(model_path: str, device: str) -> RadarTrackTransformer:
    # PyTorch 新版本建议使用 weights_only=True 以避免反序列化任意对象的安全风险
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    model = RadarTrackTransformer(input_size=ckpt["input_size"], num_classes=ckpt.get("num_classes", 6))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def predict_for_condition(
    condition: str,
    model: RadarTrackTransformer,
    loader: DataLoader,
    device: str,
    sigma: float,
    sp_prob: float,
    speckle_sigma: float,
    max_rel_change: float | None,
    worst_of_k: int,
    clamp: tuple[float, float] | None,
) -> tuple[List[int], List[int], List[List[float]], Dict[str, float]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[List[float]] = []
    attacked_sample_total = 0
    perturb_distance_sum = 0.0
    distortion_sum = 0.0
    ssim_sum = 0.0
    psnr_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y_dev = y.to(device)

        def _make_noisy(x_in: torch.Tensor) -> torch.Tensor:
            if condition == "clean":
                return x_in
            if condition == "gaussian":
                return add_gaussian_noise(
                    x_in,
                    sigma=sigma,
                    max_rel_change=max_rel_change,
                    clamp=clamp,
                )
            if condition == "salt_pepper":
                return add_salt_pepper_noise(
                    x_in,
                    prob=sp_prob,
                    max_rel_change=max_rel_change,
                    clamp=clamp,
                )
            if condition == "speckle":
                return add_speckle_noise(
                    x_in,
                    sigma=speckle_sigma,
                    max_rel_change=max_rel_change,
                    clamp=clamp,
                )
            raise ValueError(f"unknown condition: {condition}")

        with torch.no_grad():
            if condition == "clean" or worst_of_k <= 1:
                x_eval = _make_noisy(x)
                logits = model(x_eval)
            else:
                logits_cands: List[torch.Tensor] = []
                x_eval_cands: List[torch.Tensor] = []
                true_logits_cands: List[torch.Tensor] = []
                for _ in range(worst_of_k):
                    x_eval = _make_noisy(x)
                    x_eval_cands.append(x_eval)
                    lg = model(x_eval)
                    logits_cands.append(lg)
                    true_lg = lg.gather(1, y_dev.view(-1, 1)).squeeze(1)
                    true_logits_cands.append(true_lg)
                cand_scores = torch.stack(true_logits_cands, dim=1)  # [B, K], 越小越“坏”
                worst_idx = cand_scores.argmin(dim=1)
                logits_stack = torch.stack(logits_cands, dim=1)  # [B, K, C]
                logits = logits_stack[torch.arange(logits_stack.size(0), device=device), worst_idx]
                x_eval_stack = torch.stack(x_eval_cands, dim=1)
                x_eval = x_eval_stack[torch.arange(x_eval_stack.size(0), device=device), worst_idx]
            pred = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
        if condition != "clean":
            quality = compute_signal_quality_metrics(x, x_eval)
            batch_size_now = int(x.size(0))
            attacked_sample_total += batch_size_now
            perturb_distance_sum += quality["avg_perturbation_distance"] * batch_size_now
            distortion_sum += quality["avg_distortion"] * batch_size_now
            ssim_sum += quality["avg_ssim"] * batch_size_now
            psnr_sum += quality["avg_psnr"] * batch_size_now
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_score.extend(probs.cpu().numpy().tolist())

    quality_metrics = {
        "avg_perturbation_distance": float(perturb_distance_sum / attacked_sample_total)
        if attacked_sample_total
        else 0.0,
        "avg_distortion": float(distortion_sum / attacked_sample_total) if attacked_sample_total else 0.0,
        "avg_ssim": float(ssim_sum / attacked_sample_total) if attacked_sample_total else 1.0,
        "avg_psnr": float(psnr_sum / attacked_sample_total) if attacked_sample_total else float("inf"),
    }
    return y_true, y_pred, y_score, quality_metrics


def compute_interference_success_rate(
    y_true: List[int],
    y_pred_clean: List[int],
    y_pred_noisy: List[int],
) -> float:
    """
    干扰成功率（非定向）:
    在 clean 条件下原本预测正确的样本中，被干扰后预测错误的比例。
    """
    clean_correct_idx = [i for i, (yt, pc) in enumerate(zip(y_true, y_pred_clean)) if yt == pc]
    if not clean_correct_idx:
        return 0.0
    success = sum(1 for i in clean_correct_idx if y_pred_noisy[i] != y_true[i])
    return float(success / len(clean_correct_idx))


def main():
    parser = argparse.ArgumentParser(
        description="Noise robustness evaluation (clean/gaussian/salt_pepper/speckle)"
    )
    parser.add_argument("--model_path", type=str, default="radar_transformer.pth")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备：默认 cuda（优先 GPU）；无 CUDA 时自动回退 CPU。可设 cpu。",
    )
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="若无 CUDA 则报错退出，不静默回退 CPU",
    )

    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["all", "clean", "gaussian", "salt_pepper", "speckle"],
    )
    parser.add_argument("--sigma", type=float, default=0.5, help="高斯噪声标准差")
    parser.add_argument("--speckle_sigma", type=float, default=0.5, help="斑点噪声强度（乘性噪声标准差）")
    parser.add_argument("--sp_prob", type=float, default=0.2, help="椒盐噪声概率")
    parser.add_argument(
        "--max_rel_change",
        type=float,
        default=0.05,
        help="输入特征最大相对变化比例（如 0.05 表示不超过 5%%）",
    )
    parser.add_argument(
        "--worst_of_k",
        type=int,
        default=8,
        help="每个 batch 随机采样 K 次自然噪声并选最差结果（提升攻击成功率）",
    )
    parser.add_argument("--clamp_min", type=float, default=None)
    parser.add_argument("--clamp_max", type=float, default=None)
    parser.add_argument(
        "--mat_test_dir",
        type=str,
        default=None,
        help="测试用 .mat 根目录（与训练目录分开；鲁棒性评估在此数据上运行）",
    )
    parser.add_argument("--mat_dir", type=str, default=None, help="同 --mat_test_dir")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    mat_test = args.mat_test_dir or args.mat_dir
    if not mat_test:
        parser.error("必须指定 --mat_test_dir 或 --mat_dir（测试用 MAT 目录）")

    device = resolve_device_str(args.device, require_gpu=args.require_gpu)
    print_device_banner(device)
    model = load_model(args.model_path, device=device)

    from mat_loader import load_mat_directory

    meta_path = str(args.model_path) + ".meta.json"
    tgt = 32
    if Path(meta_path).is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tgt = int(meta.get("mat_target_points", 32))
    test_samples, _ = load_mat_directory(mat_test, target_points=tgt)
    loader = build_loader(test_samples, batch_size=args.batch_size)

    clamp = None
    if args.clamp_min is not None or args.clamp_max is not None:
        if args.clamp_min is None or args.clamp_max is None:
            raise ValueError("clamp_min 和 clamp_max 需要同时提供，或都不提供")
        clamp = (float(args.clamp_min), float(args.clamp_max))

    conditions = ["clean", "gaussian", "salt_pepper", "speckle"] if args.condition == "all" else [args.condition]
    print("噪声鲁棒性实验（PyTorch）:")

    # 先计算 clean 预测，作为干扰成功率的基线
    y_true_clean, y_pred_clean, _, _ = predict_for_condition(
        condition="clean",
        model=model,
        loader=loader,
        device=device,
        sigma=float(args.sigma),
        sp_prob=float(args.sp_prob),
        speckle_sigma=float(args.speckle_sigma),
        max_rel_change=args.max_rel_change,
        worst_of_k=max(1, int(args.worst_of_k)),
        clamp=clamp,
    )

    for cond in conditions:
        y_true, y_pred, y_score, quality = predict_for_condition(
            condition=cond,
            model=model,
            loader=loader,
            device=device,
            sigma=float(args.sigma),
            sp_prob=float(args.sp_prob),
            speckle_sigma=float(args.speckle_sigma),
            max_rel_change=args.max_rel_change,
            worst_of_k=max(1, int(args.worst_of_k)),
            clamp=clamp,
        )
        m = compute_metrics_from_preds(y_true, y_pred, y_score, num_classes=model.head[-1].out_features)
        if cond == "clean":
            isr = 0.0
        else:
            isr = compute_interference_success_rate(y_true_clean, y_pred_clean, y_pred)
        print(
            f"- {cond}: "
            f"Target Recall(目标召回率)={m['target_recall']:.4f}, "
            f"Sample Accuracy(样本级准确率)={m['sample_accuracy']:.4f}, "
            f"Interference Success Rate(干扰成功率)={isr:.4f}"
        )

        print(
            f"  百分比指标: Recall={format_pct(m['target_recall'])}, "
            f"Accuracy={format_pct(m['sample_accuracy'])}, "
            f"mAP={format_pct(m['mAP'])}, "
            f"ISR={format_pct(isr)}"
        )

        print(f"  Average Perturbation Distance: {quality['avg_perturbation_distance']:.6f}")
        print(f"  Average Distortion: {quality['avg_distortion']:.6f}")
        print(f"  Structural Similarity (SSIM): {quality['avg_ssim']:.6f}")
        print(f"  Peak Signal-to-Noise Ratio (PSNR): {quality['avg_psnr']:.6f} dB")

if __name__ == "__main__":
    main()

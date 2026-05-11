"""Enhanced noise robustness evaluation for radar track classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
from typing import List

import torch
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from data_utils import TrackSample
from data_utils import batch_tracks_to_sequences
from device_utils import print_device_banner
from device_utils import resolve_device_str
from metrics_utils import compute_multiclass_map
from metrics_utils import compute_signal_quality_metrics
from metrics_utils import format_pct
from model import RadarTrackTransformer
from noise_perturbation import add_channel_dropout_noise
from noise_perturbation import add_correlated_drift_noise
from noise_perturbation import add_gaussian_noise
from noise_perturbation import add_impulse_burst_noise
from noise_perturbation import add_salt_pepper_noise
from noise_perturbation import add_speckle_noise
from noise_perturbation import relative_change_stats


def build_loader(samples: List[TrackSample], batch_size: int) -> DataLoader:
    x, y = batch_tracks_to_sequences(samples)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def compute_metrics_from_preds(
    y_true: List[int],
    y_pred: List[int],
    y_score: List[List[float]],
    num_classes: int,
) -> Dict[str, float]:
    target_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    image_detection_rate = float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)) if y_true else 0.0
    return {
        "target_recall": target_recall,
        "image_detection_rate": image_detection_rate,
        "mAP": compute_multiclass_map(y_true, y_score, num_classes=num_classes),
    }


def summarize_change_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not stats_list:
        return {"mean_rel_change": 0.0, "max_rel_change": 0.0}
    return {
        "mean_rel_change": float(sum(s["mean_rel_change"] for s in stats_list) / len(stats_list)),
        "max_rel_change": float(max(s["max_rel_change"] for s in stats_list)),
    }


def load_model(model_path: str, device: str) -> RadarTrackTransformer:
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    model = RadarTrackTransformer(input_size=ckpt["input_size"], num_classes=ckpt.get("num_classes", 6))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def select_worst_candidate(logits_cands: List[torch.Tensor], y_dev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the candidate with the smallest clean-label margin for each sample."""
    margins: List[torch.Tensor] = []
    for lg in logits_cands:
        true_lg = lg.gather(1, y_dev.view(-1, 1)).squeeze(1)
        masked = lg.clone()
        masked.scatter_(1, y_dev.view(-1, 1), float("-inf"))
        other_best = masked.max(dim=1).values
        margins.append(true_lg - other_best)
    margin_stack = torch.stack(margins, dim=1)
    worst_idx = margin_stack.argmin(dim=1)
    logits_stack = torch.stack(logits_cands, dim=1)
    row_idx = torch.arange(logits_stack.size(0), device=logits_stack.device)
    return logits_stack[row_idx, worst_idx], worst_idx


def predict_for_condition(
    condition: str,
    model: RadarTrackTransformer,
    loader: DataLoader,
    device: str,
    sigma: float,
    speckle_sigma: float,
    sp_prob: float,
    burst_prob: float,
    burst_len: int,
    burst_amp: float,
    drift_scale: float,
    drift_kernel: int,
    channel_drop_prob: float,
    channel_attenuation: float,
    max_rel_change: float | None,
    budget_floor: float,
    worst_of_k: int,
    clamp: tuple[float, float] | None,
) -> tuple[List[int], List[int], List[List[float]], Dict[str, float]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[List[float]] = []
    change_stats: List[Dict[str, float]] = []
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
                    min_scale=budget_floor,
                    clamp=clamp,
                )
            if condition == "speckle":
                return add_speckle_noise(
                    x_in,
                    sigma=speckle_sigma,
                    max_rel_change=max_rel_change,
                    min_scale=budget_floor,
                    clamp=clamp,
                )
            if condition == "salt_pepper":
                return add_salt_pepper_noise(
                    x_in,
                    prob=sp_prob,
                    max_rel_change=max_rel_change,
                    min_scale=budget_floor,
                    clamp=clamp,
                )
            if condition == "impulse_burst":
                return add_impulse_burst_noise(
                    x_in,
                    burst_prob=burst_prob,
                    burst_len=burst_len,
                    burst_amp=burst_amp,
                    max_rel_change=max_rel_change,
                    min_scale=budget_floor,
                    clamp=clamp,
                )
            if condition == "correlated_drift":
                return add_correlated_drift_noise(
                    x_in,
                    drift_scale=drift_scale,
                    smooth_kernel=drift_kernel,
                    max_rel_change=max_rel_change,
                    min_scale=budget_floor,
                    clamp=clamp,
                )
            if condition == "channel_dropout":
                return add_channel_dropout_noise(
                    x_in,
                    drop_prob=channel_drop_prob,
                    attenuation=channel_attenuation,
                    max_rel_change=max_rel_change,
                    min_scale=budget_floor,
                    clamp=clamp,
                )
            raise ValueError(f"unknown condition: {condition}")

        with torch.no_grad():
            if condition == "clean" or worst_of_k <= 1:
                x_eval = _make_noisy(x)
                logits = model(x_eval)
                change_stats.append(relative_change_stats(x, x_eval))
            else:
                logits_cands: List[torch.Tensor] = []
                x_eval_cands: List[torch.Tensor] = []
                for _ in range(worst_of_k):
                    x_eval = _make_noisy(x)
                    x_eval_cands.append(x_eval)
                    logits_cands.append(model(x_eval))
                logits, worst_idx = select_worst_candidate(logits_cands, y_dev)
                per_batch_stats = [relative_change_stats(x, x_eval) for x_eval in x_eval_cands]
                worst_idx_cpu = worst_idx.detach().cpu().tolist()
                change_stats.extend(per_batch_stats[idx] for idx in worst_idx_cpu)
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

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_score.extend(probs.cpu().numpy().tolist())

    summary = summarize_change_stats(change_stats)
    summary.update(
        {
            "avg_perturbation_distance": float(perturb_distance_sum / attacked_sample_total)
            if attacked_sample_total
            else 0.0,
            "avg_distortion": float(distortion_sum / attacked_sample_total) if attacked_sample_total else 0.0,
            "avg_ssim": float(ssim_sum / attacked_sample_total) if attacked_sample_total else 1.0,
            "avg_psnr": float(psnr_sum / attacked_sample_total) if attacked_sample_total else float("inf"),
        }
    )
    return y_true, y_pred, y_score, summary


def compute_interference_success_rate(
    y_true: List[int],
    y_pred_clean: List[int],
    y_pred_noisy: List[int],
) -> float:
    clean_correct_idx = [i for i, (yt, pc) in enumerate(zip(y_true, y_pred_clean)) if yt == pc]
    if not clean_correct_idx:
        return 0.0
    success = sum(1 for i in clean_correct_idx if y_pred_noisy[i] != y_true[i])
    return float(success / len(clean_correct_idx))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced noise robustness evaluation with structured natural noise variants"
    )
    parser.add_argument("--model_path", type=str, default="radar_transformer.pth")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=[
            "all",
            "clean",
            "gaussian",
            "salt_pepper",
            "impulse_burst",
            "speckle",
            "correlated_drift",
            "channel_dropout",
        ],
    )
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--speckle_sigma", type=float, default=0.5)
    parser.add_argument("--sp_prob", type=float, default=0.2)
    parser.add_argument("--burst_prob", type=float, default=1.0)
    parser.add_argument("--burst_len", type=int, default=16)
    parser.add_argument("--burst_amp", type=float, default=1.0)
    parser.add_argument("--drift_scale", type=float, default=0.5)
    parser.add_argument("--drift_kernel", type=int, default=9)
    parser.add_argument("--channel_drop_prob", type=float, default=0.3)
    parser.add_argument("--channel_attenuation", type=float, default=0.8)
    parser.add_argument("--max_rel_change", type=float, default=0.05)
    parser.add_argument("--budget_floor", type=float, default=0.05)
    parser.add_argument("--worst_of_k", type=int, default=8)
    parser.add_argument("--clamp_min", type=float, default=None)
    parser.add_argument("--clamp_max", type=float, default=None)
    parser.add_argument("--mat_test_dir", type=str, default=None)
    parser.add_argument("--mat_dir", type=str, default=None)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")

    mat_test = args.mat_test_dir or args.mat_dir
    if not mat_test:
        parser.error("Please provide --mat_test_dir or --mat_dir.")

    device = resolve_device_str(args.device, require_gpu=args.require_gpu)
    print_device_banner(device)
    model = load_model(args.model_path, device=device)

    from mat_loader import load_mat_directory

    meta_path = str(args.model_path) + ".meta.json"
    target_points = 32
    if Path(meta_path).is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        target_points = int(meta.get("mat_target_points", 32))

    test_samples, _ = load_mat_directory(mat_test, target_points=target_points)
    loader = build_loader(test_samples, batch_size=args.batch_size)

    clamp = None
    if args.clamp_min is not None or args.clamp_max is not None:
        if args.clamp_min is None or args.clamp_max is None:
            raise ValueError("clamp_min and clamp_max must be provided together.")
        clamp = (float(args.clamp_min), float(args.clamp_max))

    conditions = (
        [
            "clean",
            "gaussian",
            "salt_pepper",
            "impulse_burst",
            "speckle",
            "correlated_drift",
            "channel_dropout",
        ]
        if args.condition == "all"
        else [args.condition]
    )

    print("Enhanced noise robustness experiment (PyTorch):")

    y_true_clean, y_pred_clean, _, clean_change_stats = predict_for_condition(
        condition="clean",
        model=model,
        loader=loader,
        device=device,
        sigma=float(args.sigma),
        speckle_sigma=float(args.speckle_sigma),
        sp_prob=float(args.sp_prob),
        burst_prob=float(args.burst_prob),
        burst_len=int(args.burst_len),
        burst_amp=float(args.burst_amp),
        drift_scale=float(args.drift_scale),
        drift_kernel=int(args.drift_kernel),
        channel_drop_prob=float(args.channel_drop_prob),
        channel_attenuation=float(args.channel_attenuation),
        max_rel_change=args.max_rel_change,
        budget_floor=float(args.budget_floor),
        worst_of_k=max(1, int(args.worst_of_k)),
        clamp=clamp,
    )

    for cond in conditions:
        y_true, y_pred, y_score, change_stats = predict_for_condition(
            condition=cond,
            model=model,
            loader=loader,
            device=device,
            sigma=float(args.sigma),
            speckle_sigma=float(args.speckle_sigma),
            sp_prob=float(args.sp_prob),
            burst_prob=float(args.burst_prob),
            burst_len=int(args.burst_len),
            burst_amp=float(args.burst_amp),
            drift_scale=float(args.drift_scale),
            drift_kernel=int(args.drift_kernel),
            channel_drop_prob=float(args.channel_drop_prob),
            channel_attenuation=float(args.channel_attenuation),
            max_rel_change=args.max_rel_change,
            budget_floor=float(args.budget_floor),
            worst_of_k=max(1, int(args.worst_of_k)),
            clamp=clamp,
        )
        metrics = compute_metrics_from_preds(y_true, y_pred, y_score, num_classes=model.head[-1].out_features)
        if cond == "clean":
            isr = 0.0
            change_stats = clean_change_stats
        else:
            isr = compute_interference_success_rate(y_true_clean, y_pred_clean, y_pred)
        print(
            f"- {cond}: "
            f"Target Recall={metrics['target_recall']:.4f}, "
            f"Image Detection Rate={metrics['image_detection_rate']:.4f}, "
            f"Interference Success Rate={isr:.4f}, "
            f"Mean Relative Change={change_stats['mean_rel_change']:.4f}, "
            f"Max Relative Change={change_stats['max_rel_change']:.4f}"
        )
        print(
            f"  Percent Metrics: Recall={format_pct(metrics['target_recall'])}, "
            f"Accuracy={format_pct(metrics['image_detection_rate'])}, "
            f"mAP={format_pct(metrics['mAP'])}, "
            f"ISR={format_pct(isr)}"
        )
        print(f"  Average Perturbation Distance: {change_stats['avg_perturbation_distance']:.6f}")
        print(f"  Average Distortion: {change_stats['avg_distortion']:.6f}")
        print(f"  Structural Similarity (SSIM): {change_stats['avg_ssim']:.6f}")
        print(f"  Peak Signal-to-Noise Ratio (PSNR): {change_stats['avg_psnr']:.6f} dB")


if __name__ == "__main__":
    main()

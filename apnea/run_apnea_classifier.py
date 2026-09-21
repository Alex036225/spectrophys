\
"""Export frozen SpectroPhys-MT rPPG and evaluate apnea classification by subject folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import signal, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neural_methods.model.SpectroPhys import SpectroPhys


TASKS = ("pr", "hr", "rr", "spo2", "sbp", "dbp", "map")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-fs", type=float, default=10.0)
    parser.add_argument("--model-fs", type=float, default=30.0)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=72)
    parser.add_argument("--window-frames", type=int, default=640)
    parser.add_argument("--stride-frames", type=int, default=320)
    parser.add_argument("--feature-window-sec", type=float, default=6.0)
    parser.add_argument("--feature-hop-sec", type=float, default=3.0)
    parser.add_argument("--select-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(checkpoint: Path, device: torch.device) -> SpectroPhys:
    model = SpectroPhys(
        feature_dim=128,
        latent_dim=32,
        hidden_dim=128,
        tcn_layers=4,
        temporal_module="prototype_preserving_slow_rr",
        phase_fs=30.0,
        physio_hr_low_bpm=45.0,
        physio_hr_high_bpm=180.0,
        physio_rr_low_bpm=6.0,
        physio_rr_high_bpm=45.0,
        physio_num_freq_bins=48,
        physio_long_context=True,
        encoder_input_normalization="channel_mean_center",
        waveform_head_type="gru",
        vital_representation_adapter=True,
        scalar_tasks=TASKS,
        rate_bin_aux=True,
        rate_bin_num_bins=96,
        rate_bin_scalar_mode="blend",
        scalar_hr_low_bpm=45.0,
        scalar_hr_high_bpm=180.0,
        scalar_rr_low_bpm=6.0,
        scalar_rr_high_bpm=45.0,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    core_missing = [
        key
        for key in incompatible.missing_keys
        if key.startswith(("stem.", "base_encoder.", "encoder_head.", "temporal_model.", "regressor_head."))
    ]
    if core_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch: core_missing={core_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(
        f"checkpoint_loaded={checkpoint} missing_noncore={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    return model.to(device).eval()


def window_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def prepare_video(frames: np.ndarray, target_frames: int, height: int, width: int) -> torch.Tensor:
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    video = video.permute(1, 0, 2, 3).unsqueeze(0)
    return F.interpolate(
        video,
        size=(target_frames, height, width),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


@torch.inference_mode()
def infer_rppg(
    model: SpectroPhys,
    frames: np.ndarray,
    source_fs: float,
    model_fs: float,
    height: int,
    width: int,
    window_frames: int,
    stride_frames: int,
    device: torch.device,
) -> np.ndarray:
    target_frames = int(round(len(frames) * model_fs / source_fs))
    video = prepare_video(frames, target_frames, height, width)
    if target_frames < window_frames:
        pad = window_frames - target_frames
        video = F.pad(video, (0, 0, 0, 0, 0, pad), mode="replicate")
    total = video.shape[1]
    output = np.zeros(total, dtype=np.float64)
    weight_sum = np.zeros(total, dtype=np.float64)
    blend = np.maximum(np.hanning(window_frames), 0.05)
    starts = window_starts(total, window_frames, stride_frames)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for start in starts:
        clip = video[:, start : start + window_frames].unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            pred, _ = model(clip)
        pred = pred.reshape(-1).float().cpu().numpy()
        if len(pred) != window_frames:
            pred = signal.resample(pred, window_frames)
        pred = pred - np.mean(pred)
        output[start : start + window_frames] += pred * blend
        weight_sum[start : start + window_frames] += blend
    output = output / np.maximum(weight_sum, 1e-8)
    output = output[:target_frames]
    return ((output - output.mean()) / (output.std() + 1e-8)).astype(np.float32)


def spectral_features(values: np.ndarray, fs: float) -> dict[str, float]:
    nperseg = min(512, len(values))
    freqs, power = signal.welch(values, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    total_mask = (freqs >= 0.05) & (freqs <= 4.0)
    total = float(np.trapz(power[total_mask], freqs[total_mask]) + 1e-12)
    features: dict[str, float] = {}
    bands = {
        "very_low": (0.05, 0.15),
        "resp_low": (0.15, 0.4),
        "resp_high": (0.4, 0.8),
        "transition": (0.8, 1.3),
        "pulse": (1.3, 3.5),
    }
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        features[f"rel_power_{name}"] = float(np.trapz(power[mask], freqs[mask]) / total) if mask.any() else 0.0
    normalized = power[total_mask] / (power[total_mask].sum() + 1e-12)
    features["spectral_entropy"] = float(-np.sum(normalized * np.log(normalized + 1e-12)) / np.log(max(2, len(normalized))))
    for name, low, high in (("resp", 0.08, 0.8), ("pulse", 1.3, 3.5)):
        mask = (freqs >= low) & (freqs <= high)
        band_power = power[mask]
        if len(band_power):
            peak = int(np.argmax(band_power))
            features[f"peak_{name}_hz"] = float(freqs[mask][peak])
            features[f"peak_{name}_concentration"] = float(band_power[peak] / (band_power.sum() + 1e-12))
        else:
            features[f"peak_{name}_hz"] = 0.0
            features[f"peak_{name}_concentration"] = 0.0
    features["pulse_resp_power_ratio"] = features["rel_power_pulse"] / (
        features["rel_power_resp_low"] + features["rel_power_resp_high"] + 1e-8
    )
    return features


def segment_features(values: np.ndarray, fs: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean()
    first = np.diff(centered)
    second = np.diff(centered, n=2)
    features = {
        "std": float(centered.std()),
        "mad": float(np.median(np.abs(centered - np.median(centered)))),
        "iqr": float(np.percentile(centered, 75) - np.percentile(centered, 25)),
        "range": float(np.ptp(centered)),
        "skew": float(stats.skew(centered, bias=False)),
        "kurtosis": float(stats.kurtosis(centered, bias=False)),
        "diff_rms": float(np.sqrt(np.mean(first**2))) if len(first) else 0.0,
        "diff_abs": float(np.mean(np.abs(first))) if len(first) else 0.0,
        "diff2_rms": float(np.sqrt(np.mean(second**2))) if len(second) else 0.0,
        "zero_cross_rate": float(np.mean(centered[:-1] * centered[1:] < 0)) if len(centered) > 1 else 0.0,
    }
    features.update(spectral_features(centered, fs))
    return {key: float(np.nan_to_num(value)) for key, value in features.items()}


def clip_features(values: np.ndarray, fs: float, window_sec: float, hop_sec: float) -> dict[str, float]:
    window = max(32, int(round(window_sec * fs)))
    hop = max(1, int(round(hop_sec * fs)))
    starts = window_starts(len(values), window, hop)
    rows = [segment_features(values[start : start + window], fs) for start in starts]
    keys = sorted(rows[0])
    result = {f"whole_{key}": value for key, value in segment_features(values, fs).items()}
    for key in keys:
        column = np.asarray([row[key] for row in rows], dtype=np.float64)
        for name, value in (
            ("mean", column.mean()),
            ("std", column.std()),
            ("p10", np.percentile(column, 10)),
            ("p90", np.percentile(column, 90)),
        ):
            result[f"window_{key}_{name}"] = float(np.nan_to_num(value))
    return result


def make_classifier(select_k: int, feature_count: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(select_k, feature_count))),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def choose_threshold(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    select_k: int,
) -> float:
    unique_groups = np.unique(groups)
    splits = min(4, len(unique_groups))
    if splits < 2:
        return 0.5
    inner = GroupKFold(n_splits=splits)
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    for train_idx, valid_idx in inner.split(features, labels, groups):
        if len(np.unique(labels[train_idx])) < 2:
            continue
        model = make_classifier(select_k, features.shape[1])
        model.fit(features[train_idx], labels[train_idx])
        probabilities[valid_idx] = model.predict_proba(features[valid_idx])[:, 1]
    valid = np.isfinite(probabilities)
    if valid.sum() < 4 or len(np.unique(labels[valid])) < 2:
        return 0.5
    candidates = np.linspace(0.1, 0.9, 161)
    scores = np.asarray(
        [balanced_accuracy_score(labels[valid], probabilities[valid] >= threshold) for threshold in candidates]
    )
    best = np.flatnonzero(scores == scores.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def metric_dict(labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cross_validated_classification(table: pd.DataFrame, select_k: int) -> tuple[pd.DataFrame, dict]:
    meta = {"clip_id", "subject", "fold", "label", "roi_effective", "signal_path"}
    feature_columns = [column for column in table.columns if column not in meta]
    features = np.nan_to_num(table[feature_columns].to_numpy(dtype=np.float64))
    labels = table["label"].to_numpy(dtype=int)
    groups = table["subject"].astype(str).to_numpy()
    folds = table["fold"].to_numpy(dtype=int)
    probabilities = np.full(len(table), np.nan, dtype=np.float64)
    thresholds = np.full(len(table), np.nan, dtype=np.float64)
    selected_by_fold: dict[str, list[str]] = {}
    for fold in sorted(np.unique(folds)):
        test_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise RuntimeError(f"Subject leakage in fold {fold}: {sorted(overlap)}")
        threshold = choose_threshold(features[train_idx], labels[train_idx], groups[train_idx], select_k)
        classifier = make_classifier(select_k, features.shape[1])
        classifier.fit(features[train_idx], labels[train_idx])
        probabilities[test_idx] = classifier.predict_proba(features[test_idx])[:, 1]
        mask = classifier.named_steps["select"].get_support()
        selected_by_fold[str(fold)] = list(np.asarray(feature_columns)[mask])
        thresholds[test_idx] = threshold
        print(
            f"fold={fold} train_clips={len(train_idx)} test_clips={len(test_idx)} "
            f"train_subjects={len(set(groups[train_idx]))} test_subjects={len(set(groups[test_idx]))} "
            f"inner_threshold={threshold:.3f}"
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Missing out-of-fold probabilities")
    output = table[["clip_id", "subject", "fold", "label", "roi_effective", "signal_path"]].copy()
    output["probability_apnea"] = probabilities
    output["inner_threshold"] = thresholds
    output["prediction_nested_threshold"] = (probabilities >= thresholds).astype(int)
    output["prediction_threshold_0.5"] = (probabilities >= 0.5).astype(int)
    metrics = {
        "protocol": "manifest subject-disjoint 5-fold OOF; all fitting and threshold selection are train-fold only",
        "n_clips": int(len(table)),
        "n_subjects": int(len(np.unique(groups))),
        "positive_clips": int(labels.sum()),
        "negative_clips": int((labels == 0).sum()),
        "select_k": int(min(select_k, len(feature_columns))),
        "nested_train_threshold": metric_dict(labels, probabilities, output["prediction_nested_threshold"].to_numpy()),
        "fixed_threshold_0.5": metric_dict(labels, probabilities, output["prediction_threshold_0.5"].to_numpy()),
        "selected_features_by_fold": selected_by_fold,
    }
    return output, metrics


def save_plots(output_dir: Path, signal_rows: list[dict], predictions: pd.DataFrame, metrics: dict) -> None:
    examples = []
    for label in (0, 1):
        examples.extend([row for row in signal_rows if row["label"] == label][:2])
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 7), sharex=False, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, examples):
        data = np.load(row["signal_path"])
        time = data["time_sec"]
        axis.plot(time, data["rppg"], color="#246B8E", linewidth=0.8)
        axis.set_title(f"{row['clip_id']} | {'apnea' if row['label'] else 'normal'} | {row['roi_effective']}", loc="left", fontsize=9)
        axis.set_ylabel("rPPG (z)")
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(output_dir / "rppg_examples.png", dpi=220)
    plt.close(fig)

    labels = predictions["label"].to_numpy()
    probabilities = predictions["probability_apnea"].to_numpy()
    predicted = predictions["prediction_nested_threshold"].to_numpy()
    fpr, tpr, _ = roc_curve(labels, probabilities)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    axes[0].plot(fpr, tpr, color="#B23A48", linewidth=2)
    axes[0].plot([0, 1], [0, 1], color="0.65", linestyle="--")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"Subject-OOF ROC (AUC {metrics['nested_train_threshold']['roc_auc']:.3f})")
    image = axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Apnea"], yticklabels=["Normal", "Apnea"], xlabel="Predicted", ylabel="True", title="OOF confusion matrix")
    fig.colorbar(image, ax=axes[1], fraction=0.046)
    fig.savefig(output_dir / "classification_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal_dir = args.output_dir / "rppg_signals"
    signal_dir.mkdir(exist_ok=True)
    manifest_path = args.data_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"clip_id", "subject", "clip_label", "fold", "clip_path", "roi_effective"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} clips={len(manifest)} subjects={manifest['subject'].nunique()}")
    model = build_model(args.checkpoint, device)
    signal_rows: list[dict] = []
    feature_rows: list[dict] = []
    for index, row in manifest.iterrows():
        clip_id = str(row["clip_id"])
        input_path = args.data_root / str(row["clip_path"])
        output_path = signal_dir / f"{clip_id}.npz"
        if args.force_export or not output_path.exists():
            with np.load(input_path, allow_pickle=False) as data:
                frames = data["frames"]
            rppg = infer_rppg(
                model,
                frames,
                args.source_fs,
                args.model_fs,
                args.height,
                args.width,
                args.window_frames,
                args.stride_frames,
                device,
            )
            time_sec = np.arange(len(rppg), dtype=np.float32) / args.model_fs
            np.savez_compressed(
                output_path,
                rppg=rppg,
                time_sec=time_sec,
                fs=np.float32(args.model_fs),
                clip_id=np.asarray(clip_id),
                subject=np.asarray(str(row["subject"])),
                label=np.int64(row["clip_label"]),
            )
        with np.load(output_path, allow_pickle=False) as saved:
            rppg = saved["rppg"].astype(np.float64)
        if len(rppg) < int(args.model_fs * 10) or not np.isfinite(rppg).all():
            raise RuntimeError(f"Invalid rPPG output for {clip_id}: shape={rppg.shape}")
        base = {
            "clip_id": clip_id,
            "subject": str(row["subject"]),
            "fold": int(row["fold"]),
            "label": int(row["clip_label"]),
            "roi_effective": str(row["roi_effective"]),
            "signal_path": str(output_path),
        }
        signal_rows.append(base)
        feature_rows.append(
            {
                **base,
                **clip_features(rppg, args.model_fs, args.feature_window_sec, args.feature_hop_sec),
            }
        )
        print(f"exported={index + 1}/{len(manifest)} clip={clip_id} samples={len(rppg)}")
    feature_table = pd.DataFrame(feature_rows)
    feature_table.to_csv(args.output_dir / "clip_features.csv", index=False)
    pd.DataFrame(signal_rows).to_csv(args.output_dir / "rppg_manifest.csv", index=False)
    predictions, metrics = cross_validated_classification(feature_table, args.select_k)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "data_manifest": str(manifest_path),
            "data_manifest_sha256": sha256(manifest_path),
            "source_fps": args.source_fs,
            "model_fps": args.model_fs,
            "model_window_frames": args.window_frames,
            "model_stride_frames": args.stride_frames,
            "classifier": "StandardScaler + train-fold SelectKBest(f_classif) + shrinkage LDA",
        }
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
    save_plots(args.output_dir, signal_rows, predictions, metrics)
    print("FINAL_METRICS " + json.dumps(metrics["nested_train_threshold"], sort_keys=True))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

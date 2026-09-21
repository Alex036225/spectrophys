\
"""Benchmark adapter for McJackTang/FusionVitals on single-RGB cached clips.

The public FusionVitals repo is centered on LADH RGB+IR data, but the released
code also includes a single-RGB "face" path. This adapter keeps that
single-stream model structure for our existing 72x72x160 RGB benchmark:

* 3D SE-augmented encoder from FusionPhysNet
* rPPG decoder branch
* scalar SpO2 head built from the predicted rPPG
* RR branch kept in the model but not supervised because the current manifest
  does not provide RR labels

Training uses the same source+ZPU-train / ZPU-valid / ZPU-test manifests used
by the other subject19 benchmark runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from evaluation.metrics import calculate_metrics


def parse_clip(path: str) -> tuple[str, int]:
    match = re.match(r"(.+)_input(\d+)\.npy$", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse cached clip name: {path}")
    return match.group(1), int(match.group(2))


def finite_float(value: str | float | int | None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if "input_files" not in (reader.fieldnames or []):
        raise ValueError(f"{path} missing input_files column")
    for row in rows:
        subject, clip_index = parse_clip(row["input_files"])
        row.setdefault("subject", subject)
        row.setdefault("clip_index", str(clip_index))
        if not row.get("ppg_label_file"):
            row["ppg_label_file"] = row["input_files"].replace("input", "label")
        if not row.get("label_polarity"):
            row["label_polarity"] = "1.0"
        if not row.get("ppg_mask"):
            row["ppg_mask"] = "1"
        if not row.get("spo2_mask"):
            row["spo2_mask"] = "0"
    return rows


class CachedFusionVitalsDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], frames: int = 160):
        self.rows = rows
        self.frames = int(frames)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        video = np.load(row["input_files"]).astype(np.float32)[: self.frames]
        if video.max(initial=0.0) > 2.0:
            video = video / np.float32(255.0)
        video = np.transpose(video, (3, 0, 1, 2))

        ppg_mask = 1.0 if str(row.get("ppg_mask", "1")) == "1" else 0.0
        label_path = row["ppg_label_file"]
        if ppg_mask > 0 and Path(label_path).exists():
            ppg = np.load(label_path).astype(np.float32)[: self.frames]
            ppg = ppg * np.float32(float(row.get("label_polarity", "1.0") or 1.0))
        else:
            ppg = np.zeros((self.frames,), dtype=np.float32)
            ppg_mask = 0.0

        spo2_value = finite_float(row.get("spo2_pct"))
        spo2_mask = 1.0 if str(row.get("spo2_mask", "0")) == "1" and spo2_value is not None else 0.0
        if spo2_value is None:
            spo2_value = 97.0

        return (
            torch.from_numpy(video),
            torch.from_numpy(ppg),
            torch.tensor(ppg_mask, dtype=torch.float32),
            torch.tensor(float(spo2_value), dtype=torch.float32),
            torch.tensor(spo2_mask, dtype=torch.float32),
            row["subject"],
            int(row["clip_index"]),
        )


def build_sampler(rows: list[dict[str, str]], spo2_positive_weight: float) -> WeightedRandomSampler:
    weights = np.ones((len(rows),), dtype=np.float64)
    for idx, row in enumerate(rows):
        if str(row.get("spo2_mask", "0")) == "1":
            weights[idx] *= float(spo2_positive_weight)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(rows),
        replacement=True,
    )


def normalize_signal_batch(signal_batch: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = torch.mean(signal_batch, dim=1, keepdim=True)
    std = torch.std(signal_batch, dim=1, keepdim=True, unbiased=False)
    return (signal_batch - mean) / (std + eps)


def neg_pearson_loss(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    preds = normalize_signal_batch(preds)
    labels = normalize_signal_batch(labels)
    per_sample = 1.0 - torch.mean(preds * labels, dim=1)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (per_sample * mask).sum() / denom


def masked_weighted_spo2_mse(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return preds.new_zeros(())
    diff = preds[valid] - labels[valid]
    mean_target = torch.mean(labels[valid])
    weight = torch.square(100.0 - mean_target)
    return torch.mean(torch.square(diff)) * weight


def metric_summary(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    diff = preds - labels
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mape = float(np.mean(np.abs(diff / np.clip(labels, 1e-6, None))) * 100.0)
    pearson = (
        float(np.corrcoef(preds, labels)[0, 1])
        if len(preds) > 1 and np.std(preds) > 0 and np.std(labels) > 0
        else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "mape": mape, "pearson": pearson}


def make_eval_config(model_name: str):
    return SimpleNamespace(
        TOOLBOX_MODE="train_and_test",
        TRAIN=SimpleNamespace(MODEL_FILE_NAME=model_name),
        INFERENCE=SimpleNamespace(
            EVALUATION_METHOD="FFT",
            EVALUATION_WINDOW=SimpleNamespace(USE_SMALLER_WINDOW=True, WINDOW_SIZE=30),
        ),
        TEST=SimpleNamespace(
            METRICS=["MAE", "RMSE", "MAPE", "Pearson"],
            DATA=SimpleNamespace(
                FS=30,
                PREPROCESS=SimpleNamespace(LABEL_TYPE="Standardized"),
                DATASET="JointCached",
            ),
        ),
    )


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _, _ = x.shape
        pooled = self.avg_pool(x).view(batch, channels)
        weights = self.fc(pooled).view(batch, channels, 1, 1, 1)
        return x * weights


class FusionVitalsFaceNet(nn.Module):
    """Single-RGB path extracted from FusionVitals FusionPhysNet."""

    def __init__(self, frames: int = 160):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 16, (1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            SEBlock(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )
        self.conv8 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )
        self.pool_spatial = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.pool_spatiotemporal = nn.MaxPool3d((2, 2, 2), stride=2)

        self.rppg_branch = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, kernel_size=1),
        )
        self.rr_branch = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, kernel_size=1),
        )
        self.spo2_head = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool_spatial(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool_spatiotemporal(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool_spatiotemporal(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool_spatial(x)
        x = self.conv8(x)
        return x

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, frames, _, _ = video.shape
        encoded = self.encode_video(video)
        rppg = self.rppg_branch(encoded).view(batch, frames)
        rr = self.rr_branch(encoded).view(batch, frames)
        spo2 = self.spo2_head(rppg.unsqueeze(1)).view(batch)
        spo2 = spo2 * 15.0 + 85.0
        return rppg, spo2, rr


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    loss_sum = 0.0
    ppg_loss_sum = 0.0
    spo2_loss_sum = 0.0
    count = 0
    for video, ppg, ppg_mask, spo2_target, spo2_mask, _, _ in loader:
        video = video.to(device=device, dtype=torch.float32, non_blocking=True)
        ppg = ppg.to(device=device, dtype=torch.float32, non_blocking=True)
        ppg_mask = ppg_mask.to(device=device, dtype=torch.float32, non_blocking=True)
        spo2_target = spo2_target.to(device=device, dtype=torch.float32, non_blocking=True)
        spo2_mask = spo2_mask.to(device=device, dtype=torch.float32, non_blocking=True)

        pred_ppg, pred_spo2, _ = model(video)
        loss_ppg = neg_pearson_loss(pred_ppg, ppg, ppg_mask)
        loss_spo2 = masked_weighted_spo2_mse(pred_spo2, spo2_target, spo2_mask)
        loss = args.ppg_weight * loss_ppg + args.spo2_weight * loss_spo2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch = video.shape[0]
        count += batch
        loss_sum += float(loss.detach().cpu()) * batch
        ppg_loss_sum += float(loss_ppg.detach().cpu()) * batch
        spo2_loss_sum += float(loss_spo2.detach().cpu()) * batch

    denom = max(count, 1)
    return {
        "loss": loss_sum / denom,
        "ppg_loss": ppg_loss_sum / denom,
        "spo2_loss": spo2_loss_sum / denom,
    }


@torch.no_grad()
def evaluate(model, loader, device, args, save_outputs: bool = False):
    model.eval()
    loss_sum = 0.0
    ppg_loss_sum = 0.0
    spo2_loss_sum = 0.0
    count = 0
    spo2_preds = []
    spo2_labels = []
    waveform_predictions = {}
    waveform_labels = {}
    outputs = {}

    for video, ppg, ppg_mask, spo2_target, spo2_mask, subjects, clip_indices in loader:
        video = video.to(device=device, dtype=torch.float32, non_blocking=True)
        ppg = ppg.to(device=device, dtype=torch.float32, non_blocking=True)
        ppg_mask = ppg_mask.to(device=device, dtype=torch.float32, non_blocking=True)
        spo2_target = spo2_target.to(device=device, dtype=torch.float32, non_blocking=True)
        spo2_mask = spo2_mask.to(device=device, dtype=torch.float32, non_blocking=True)

        pred_ppg, pred_spo2, pred_rr = model(video)
        loss_ppg = neg_pearson_loss(pred_ppg, ppg, ppg_mask)
        loss_spo2 = masked_weighted_spo2_mse(pred_spo2, spo2_target, spo2_mask)
        loss = args.ppg_weight * loss_ppg + args.spo2_weight * loss_spo2

        batch = video.shape[0]
        count += batch
        loss_sum += float(loss.detach().cpu()) * batch
        ppg_loss_sum += float(loss_ppg.detach().cpu()) * batch
        spo2_loss_sum += float(loss_spo2.detach().cpu()) * batch

        pred_ppg_cpu = pred_ppg.detach().cpu()
        ppg_cpu = ppg.detach().cpu()
        pred_spo2_cpu = pred_spo2.detach().cpu()
        spo2_target_cpu = spo2_target.detach().cpu()
        spo2_mask_cpu = spo2_mask.detach().cpu()
        pred_rr_cpu = pred_rr.detach().cpu()

        for item_idx, subject in enumerate(subjects):
            sort_index = int(clip_indices[item_idx])
            waveform_predictions.setdefault(subject, {})[sort_index] = pred_ppg_cpu[item_idx]
            waveform_labels.setdefault(subject, {})[sort_index] = ppg_cpu[item_idx]
            if float(spo2_mask_cpu[item_idx]) > 0.5:
                spo2_preds.append(float(pred_spo2_cpu[item_idx]))
                spo2_labels.append(float(spo2_target_cpu[item_idx]))
            if save_outputs:
                outputs.setdefault(subject, {})[sort_index] = {
                    "spo2_pred": float(pred_spo2_cpu[item_idx]),
                    "spo2_label": float(spo2_target_cpu[item_idx]),
                    "spo2_mask": float(spo2_mask_cpu[item_idx]),
                    "waveform_pred": pred_ppg_cpu[item_idx].numpy().astype(np.float32, copy=False),
                    "waveform_label": ppg_cpu[item_idx].numpy().astype(np.float32, copy=False),
                    "rr_pred": pred_rr_cpu[item_idx].numpy().astype(np.float32, copy=False),
                }

    metrics = {
        "loss": loss_sum / max(count, 1),
        "ppg_loss": ppg_loss_sum / max(count, 1),
        "spo2_loss": spo2_loss_sum / max(count, 1),
    }
    if spo2_preds:
        metrics.update(
            {f"spo2_{key}": value for key, value in metric_summary(np.asarray(spo2_preds), np.asarray(spo2_labels)).items()}
        )
    else:
        for key in ("mae", "rmse", "mape", "pearson"):
            metrics[f"spo2_{key}"] = float("nan")
    return metrics, waveform_predictions, waveform_labels, outputs


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(
        f"{key}={value:.5f}" if math.isfinite(value) else f"{key}=nan"
        for key, value in sorted(metrics.items())
    )


def print_spo2_summary(metrics: dict[str, float], num_samples: int) -> None:
    if not math.isfinite(metrics.get("spo2_mae", float("nan"))):
        print("SpO2 MAE (Scalar): nan +/- nan")
        return
    std_err = 0.0 if num_samples <= 1 else metrics["spo2_rmse"] / math.sqrt(num_samples)
    print(f"SpO2 MAE (Scalar): {metrics['spo2_mae']} +/- {std_err}")
    print(f"SpO2 RMSE (Scalar): {metrics['spo2_rmse']}")
    print(f"SpO2 MAPE (Scalar): {metrics['spo2_mape']}")
    print(f"SpO2 Pearson (Scalar): {metrics['spo2_pearson']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", default="runs/exp/fusionvitals_face")
    parser.add_argument("--model-name", default="fusionvitals_face_subject19")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ppg-weight", type=float, default=1.0)
    parser.add_argument("--spo2-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--spo2-positive-weight", type=float, default=12.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest(args.train_csv)
    valid_rows = read_manifest(args.valid_csv)
    test_rows = read_manifest(args.test_csv)
    train_spo2 = sum(str(row.get("spo2_mask", "0")) == "1" for row in train_rows)
    valid_spo2 = sum(str(row.get("spo2_mask", "0")) == "1" for row in valid_rows)
    test_spo2 = sum(str(row.get("spo2_mask", "0")) == "1" for row in test_rows)
    print(
        f"train_rows={len(train_rows)} spo2_rows={train_spo2} "
        f"valid_rows={len(valid_rows)} spo2_rows={valid_spo2} "
        f"test_rows={len(test_rows)} spo2_rows={test_spo2}"
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "drop_last": False,
    }
    train_loader = DataLoader(
        CachedFusionVitalsDataset(train_rows, frames=args.frames),
        batch_size=args.batch_size,
        sampler=build_sampler(train_rows, args.spo2_positive_weight),
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        CachedFusionVitalsDataset(valid_rows, frames=args.frames),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        CachedFusionVitalsDataset(test_rows, frames=args.frames),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = FusionVitalsFaceNet(frames=args.frames).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = output_dir / f"{args.model_name}_best.pth"
    best_valid_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        valid_metrics, _, _, _ = evaluate(model, valid_loader, device, args, save_outputs=False)
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch} train {format_metrics(train_metrics)}")
        print(f"epoch={epoch} valid {format_metrics(valid_metrics)} best_valid_loss={best_valid_loss:.5f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_valid, _, _, valid_outputs = evaluate(model, valid_loader, device, args, save_outputs=True)
    final_test, test_predictions, test_labels, test_outputs = evaluate(model, test_loader, device, args, save_outputs=True)

    valid_pickle = output_dir / f"{args.model_name}_valid_outputs.pickle"
    test_pickle = output_dir / f"{args.model_name}_test_outputs.pickle"
    with valid_pickle.open("wb") as handle:
        pickle.dump(valid_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with test_pickle.open("wb") as handle:
        pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"best_epoch={best_epoch}")
    print(f"best_checkpoint={best_path}")
    print(f"valid_outputs={valid_pickle}")
    print(f"test_outputs={test_pickle}")
    print(f"Saving outputs to: {test_pickle}")
    print(f"final_valid {format_metrics(final_valid)}")
    print(f"final_test {format_metrics(final_test)}")
    calculate_metrics(test_predictions, test_labels, make_eval_config(args.model_name))
    print_spo2_summary(final_test, test_spo2)


if __name__ == "__main__":
    main()

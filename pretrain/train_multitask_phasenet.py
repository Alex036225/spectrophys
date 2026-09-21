\
"""Train PhaseNet with masked PPG/PR/HR/RR/SpO2 supervision."""

from __future__ import annotations

import argparse
import csv
import io
import math
import pickle
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from neural_methods.model.PhaseNet import PhaseNet


TASKS = ("pr", "hr", "rr", "spo2", "sbp", "dbp", "map")
TASK_STATS = {
    "pr": {"target": "pr_bpm", "mask": "pr_mask", "center": 80.0, "scale": 30.0, "unit": "bpm"},
    "hr": {"target": "hr_bpm", "mask": "hr_mask", "center": 80.0, "scale": 30.0, "unit": "bpm"},
    "rr": {"target": "rr_bpm", "mask": "rr_mask", "center": 16.0, "scale": 8.0, "unit": "resp/min"},
    "spo2": {"target": "spo2_pct", "mask": "spo2_mask", "center": 97.0, "scale": 3.0, "unit": "%"},
    "sbp": {"target": "sbp_mmhg", "mask": "sbp_mask", "center": 120.0, "scale": 25.0, "unit": "mmHg"},
    "dbp": {"target": "dbp_mmhg", "mask": "dbp_mask", "center": 70.0, "scale": 15.0, "unit": "mmHg"},
    "map": {"target": "map_mmhg", "mask": "map_mask", "center": 90.0, "scale": 20.0, "unit": "mmHg"},
}


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


class DomainAdversary(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_domains: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, z_seq: torch.Tensor, reverse_scale: float = 1.0) -> torch.Tensor:
        pooled = torch.cat(
            [
                torch.mean(z_seq, dim=1),
                torch.std(z_seq, dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return self.net(GradientReverse.apply(pooled, reverse_scale))


class TaskStreamDomainAdversary(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_domains: int, dropout: float = 0.1):
        super().__init__()
        self.hr_adversary = DomainAdversary(feature_dim, hidden_dim, num_domains, dropout=dropout)
        self.rr_adversary = DomainAdversary(feature_dim, hidden_dim, num_domains, dropout=dropout)

    def forward(
        self,
        hr_seq: torch.Tensor | None,
        rr_seq: torch.Tensor | None,
        reverse_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        logits = {}
        if hr_seq is not None:
            logits["hr"] = self.hr_adversary(hr_seq, reverse_scale=reverse_scale)
        if rr_seq is not None:
            logits["rr"] = self.rr_adversary(rr_seq, reverse_scale=reverse_scale)
        return logits


def parse_clip(path: str) -> tuple[str, int]:
    match = re.match(r"(.+)_input(\d+)\.npy$", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse cached clip name: {path}")
    return match.group(1), int(match.group(2))


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if "input_files" not in (reader.fieldnames or []):
        raise ValueError(f"{path} missing input_files column")
    for row in rows:
        subject, clip_index = parse_clip(row["input_files"])
        if not row.get("subject"):
            row["subject"] = subject
        if not row.get("clip_index"):
            row["clip_index"] = str(clip_index)
        if not row.get("ppg_label_file"):
            row["ppg_label_file"] = row["input_files"].replace("input", "label")
        if not row.get("label_polarity"):
            row["label_polarity"] = "1.0"
        if not row.get("ppg_mask"):
            row["ppg_mask"] = "1"
    return rows


def limit_round_robin(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["subject"], []).append(row)
    for subject in grouped:
        grouped[subject] = sorted(grouped[subject], key=lambda item: int(item["clip_index"]))
    subjects = sorted(grouped)
    cursors = {subject: 0 for subject in subjects}
    selected = []
    while len(selected) < limit:
        made_progress = False
        for subject in subjects:
            cursor = cursors[subject]
            if cursor < len(grouped[subject]):
                selected.append(grouped[subject][cursor])
                cursors[subject] += 1
                made_progress = True
                if len(selected) >= limit:
                    break
        if not made_progress:
            break
    return selected


def filter_excluded_datasets(rows: list[dict[str, str]], excluded: str) -> list[dict[str, str]]:
    excluded_names = {item.strip() for item in str(excluded or "").split(",") if item.strip()}
    if not excluded_names:
        return rows
    return [row for row in rows if row.get("dataset", "") not in excluded_names]


def build_context_samples(rows: list[dict[str, str]], context_clips: int, stride: int) -> list[dict[str, str]]:
    context_clips = int(context_clips)
    if context_clips <= 1:
        return rows
    stride = max(int(stride), 1)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row.get("dataset", ""), row["subject"]), []).append(row)

    samples: list[dict[str, str]] = []
    for _, subject_rows in sorted(grouped.items()):
        ordered = sorted(subject_rows, key=lambda item: int(item["clip_index"]))
        for start in range(0, len(ordered) - context_clips + 1, stride):
            window = ordered[start:start + context_clips]
            clip_indices = [int(item["clip_index"]) for item in window]
            sample = dict(window[0])
            sample["_context_rows"] = window
            sample["context_start_clip"] = str(clip_indices[0])
            sample["context_end_clip"] = str(clip_indices[-1])
            sample["clip_index"] = str(clip_indices[0])
            for task in TASKS:
                spec = TASK_STATS[task]
                values = [
                    finite_float(item.get(spec["target"]))
                    for item in window
                    if item.get(spec["mask"], "0") == "1" and finite_float(item.get(spec["target"])) is not None
                ]
                if values:
                    sample[spec["target"]] = str(float(np.mean(values)))
                    sample[spec["mask"]] = "1"
                else:
                    sample[spec["target"]] = ""
                    sample[spec["mask"]] = "0"
            sample["ppg_mask"] = "1" if all(str(item.get("ppg_mask", "1")) == "1" for item in window) else "0"
            samples.append(sample)
    return samples


def make_weighted_sampler(rows: list[dict[str, str]], args: argparse.Namespace) -> WeightedRandomSampler | None:
    if not args.weighted_sampler:
        return None
    weights = np.ones((len(rows),), dtype=np.float64)
    if args.sampler_dataset_balance:
        counts = {}
        for row in rows:
            dataset = row.get("dataset", "")
            counts[dataset] = counts.get(dataset, 0) + 1
        for idx, row in enumerate(rows):
            weights[idx] *= len(rows) / max(counts.get(row.get("dataset", ""), 1), 1)
    if args.sampler_hr_balance:
        bins = {}
        row_bins = []
        width = max(float(args.sampler_hr_bin_width), 1.0)
        for row in rows:
            value = finite_float(row.get(TASK_STATS["hr"]["target"]))
            if row.get(TASK_STATS["hr"]["mask"], "0") == "1" and value is not None:
                bin_key = int(math.floor(value / width))
                bins[bin_key] = bins.get(bin_key, 0) + 1
                row_bins.append(bin_key)
            else:
                row_bins.append(None)
        valid_count = sum(1 for item in row_bins if item is not None)
        for idx, bin_key in enumerate(row_bins):
            if bin_key is not None:
                weights[idx] *= max(valid_count, 1) / max(bins.get(bin_key, 1), 1)
    if args.sampler_rr_balance:
        bins = {}
        row_bins = []
        width = max(float(args.sampler_rr_bin_width), 0.5)
        for row in rows:
            value = finite_float(row.get(TASK_STATS["rr"]["target"]))
            if row.get(TASK_STATS["rr"]["mask"], "0") == "1" and value is not None:
                bin_key = int(math.floor(value / width))
                bins[bin_key] = bins.get(bin_key, 0) + 1
                row_bins.append(bin_key)
            else:
                row_bins.append(None)
        valid_count = sum(1 for item in row_bins if item is not None)
        for idx, bin_key in enumerate(row_bins):
            if bin_key is not None:
                weights[idx] *= max(valid_count, 1) / max(bins.get(bin_key, 1), 1)
    if args.sampler_zpu_weight != 1.0:
        for idx, row in enumerate(rows):
            if row.get("dataset") == "ZPU":
                weights[idx] *= float(args.sampler_zpu_weight)
    if args.sampler_high_hr_weight != 1.0:
        threshold = float(args.sampler_high_hr_threshold)
        for idx, row in enumerate(rows):
            value = finite_float(row.get(TASK_STATS["hr"]["target"]))
            if value is not None and value >= threshold:
                weights[idx] *= float(args.sampler_high_hr_weight)
    weights = np.clip(weights, 1e-8, np.percentile(weights, 99.5) if len(weights) else 1.0)
    num_samples = max(1, int(round(len(rows) * float(args.sampler_num_samples_multiplier))))
    print(
        "weighted_sampler "
        f"num_samples={num_samples} weight_min={weights.min():.6g} "
        f"weight_mean={weights.mean():.6g} weight_max={weights.max():.6g}"
    )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def finite_float(value: str | float | int | None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def build_support_context(rows: list[dict[str, str]], tasks: str) -> torch.Tensor:
    features: list[float] = []
    selected_tasks = [item.strip() for item in str(tasks or "").split(",") if item.strip()]
    for task in selected_tasks:
        if task not in TASK_STATS:
            raise ValueError(f"Unsupported support conditioning task: {task}")
        spec = TASK_STATS[task]
        values = []
        for row in rows:
            value = finite_float(row.get(spec["target"]))
            mask = row.get(spec["mask"], "0") == "1"
            if value is not None and mask:
                values.append((value - spec["center"]) / spec["scale"])
        coverage = float(len(values)) / max(len(rows), 1)
        if values:
            arr = np.asarray(values, dtype=np.float32)
            stats = [
                float(arr.mean()),
                float(arr.std()),
                float(np.quantile(arr, 0.10)),
                float(np.quantile(arr, 0.50)),
                float(np.quantile(arr, 0.90)),
                coverage,
            ]
        else:
            stats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        features.extend(stats)
    return torch.tensor(features, dtype=torch.float32)


def corrupt_video(video: np.ndarray, corruption: str, severity: int, index: int) -> np.ndarray:
    if corruption == "none" or severity <= 0:
        return video
    severity = int(np.clip(severity, 1, 5))
    result = video.copy()
    if corruption == "brightness":
        factor = 1.0 + 0.12 * severity
        return np.clip(result * factor, 0.0, 255.0).astype(np.float32)
    if corruption == "noise":
        rng = np.random.default_rng(20260731 + int(index))
        sigma = 2.5 * severity
        noise = rng.normal(0.0, sigma, size=result.shape).astype(np.float32)
        return np.clip(result + noise, 0.0, 255.0).astype(np.float32)
    if corruption == "frame_drop":
        stride = max(2, 7 - severity)
        for frame_index in range(stride, len(result), stride):
            result[frame_index] = result[frame_index - 1]
        return result
    if corruption == "roi_shift":
        shift = 2 * severity
        shifted = np.roll(result, shift=(shift, -shift), axis=(1, 2))
        shifted[:, :shift, :, :] = 0.0
        shifted[:, :, -shift:, :] = 0.0
        return shifted
    if corruption == "spatial_downsample":
        size = [64, 56, 48, 36, 24][severity - 1]
        converted = []
        for frame in result:
            image = Image.fromarray(np.clip(frame, 0.0, 255.0).astype(np.uint8), mode="RGB")
            image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
            image = image.resize((result.shape[2], result.shape[1]), resample=Image.Resampling.BILINEAR)
            converted.append(np.asarray(image, dtype=np.float32))
        return np.stack(converted, axis=0)
    if corruption == "static_frame":
        return np.repeat(result.mean(axis=0, keepdims=True), len(result), axis=0).astype(np.float32)
    if corruption == "temporal_shuffle":
        rng = np.random.default_rng(20260801 + int(index))
        return result[rng.permutation(len(result))].copy()
    if corruption == "temporal_reverse":
        return result[::-1].copy()
    if corruption == "channel_permute":
        return result[..., [2, 1, 0]].copy()
    if corruption == "grayscale":
        gray = 0.299 * result[..., 0] + 0.587 * result[..., 1] + 0.114 * result[..., 2]
        return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    if corruption == "spatial_shuffle":
        rng = np.random.default_rng(20260802 + int(index))
        grid = 4
        tile_h, tile_w = result.shape[1] // grid, result.shape[2] // grid
        tiles = [result[:, row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w]
                 for row in range(grid) for col in range(grid)]
        order = rng.permutation(grid * grid)
        rows = [np.concatenate([tiles[order[row * grid + col]] for col in range(grid)], axis=2)
                for row in range(grid)]
        return np.concatenate(rows, axis=1).astype(np.float32)

    converted = []
    for frame in result:
        image = Image.fromarray(np.clip(frame, 0.0, 255.0).astype(np.uint8), mode="RGB")
        if corruption == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=0.55 * severity))
        elif corruption == "jpeg":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=max(10, 92 - 15 * severity))
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        else:
            raise ValueError(f"Unsupported corruption: {corruption}")
        converted.append(np.asarray(image, dtype=np.float32))
    return np.stack(converted, axis=0)


class MultiTaskClipDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        frames: int = 160,
        dataset_to_idx: dict[str, int] | None = None,
        corruption: str = "none",
        severity: int = 0,
    ):
        self.rows = rows
        self.frames = int(frames)
        self.dataset_to_idx = dataset_to_idx or {}
        self.corruption = corruption
        self.severity = int(severity)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        context_rows = row.get("_context_rows") or [row]
        video_chunks = [np.load(item["input_files"]).astype(np.float32)[: self.frames] for item in context_rows]
        video = np.concatenate(video_chunks, axis=0)
        video = corrupt_video(video, self.corruption, self.severity, index)
        video = np.transpose(video, (3, 0, 1, 2))

        ppg_mask = 1.0 if str(row.get("ppg_mask", "1")) == "1" else 0.0
        ppg_chunks = []
        for item in context_rows:
            label_path = item.get("ppg_label_file") or item["input_files"].replace("input", "label")
            if ppg_mask > 0 and Path(label_path).exists():
                ppg_item = np.load(label_path).astype(np.float32)[: self.frames]
                ppg_item = ppg_item * np.float32(float(item.get("label_polarity", "1.0") or 1.0))
            else:
                ppg_item = np.zeros((self.frames,), dtype=np.float32)
                ppg_mask = 0.0
            ppg_chunks.append(ppg_item)
        if ppg_chunks:
            ppg = np.concatenate(ppg_chunks, axis=0)
        else:
            ppg = np.zeros((self.frames,), dtype=np.float32)
            ppg_mask = 0.0

        scalars = []
        scalar_masks = []
        for task in TASKS:
            spec = TASK_STATS[task]
            value = finite_float(row.get(spec["target"]))
            mask = 1.0 if row.get(spec["mask"], "0") == "1" and value is not None else 0.0
            if value is None:
                value = spec["center"]
            scalars.append((float(value) - spec["center"]) / spec["scale"])
            scalar_masks.append(mask)
        dataset_idx = int(self.dataset_to_idx.get(row.get("dataset", ""), 0))

        return (
            torch.from_numpy(video),
            torch.from_numpy(ppg),
            torch.tensor(ppg_mask, dtype=torch.float32),
            torch.tensor(scalars, dtype=torch.float32),
            torch.tensor(scalar_masks, dtype=torch.float32),
            torch.tensor(dataset_idx, dtype=torch.long),
            row["subject"],
            int(row["clip_index"]),
        )


def normalize_signal_batch(signal_batch: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = torch.mean(signal_batch, dim=1, keepdim=True)
    std = torch.std(signal_batch, dim=1, keepdim=True, unbiased=False)
    return (signal_batch - mean) / (std + eps)


def neg_pearson_per_sample(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    preds = normalize_signal_batch(preds, eps=eps)
    labels = normalize_signal_batch(labels, eps=eps)
    return 1.0 - torch.mean(preds * labels, dim=1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum(), min=1.0)
    return (values * mask).sum() / denom


def scalar_predictions_to_tensor(outputs: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if outputs is None:
        raise ValueError("PhaseNet did not return scalar outputs. Instantiate with scalar_tasks.")
    return torch.stack([outputs[task] for task in TASKS], dim=1).to(device)


def soft_fft_bpm(
    signal_batch: torch.Tensor,
    fs: float,
    low_bpm: float,
    high_bpm: float,
    nfft: int,
    temperature: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    signal_batch = normalize_signal_batch(signal_batch)
    n_fft = max(int(nfft), int(signal_batch.shape[1]))
    window = torch.hann_window(signal_batch.shape[1], device=signal_batch.device, dtype=signal_batch.dtype)
    spectrum = torch.fft.rfft(signal_batch * window.unsqueeze(0), n=n_fft, dim=1)
    power = spectrum.real.square() + spectrum.imag.square() + eps
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs).to(signal_batch.device)
    bpms = freqs * 60.0
    mask = (bpms >= low_bpm) & (bpms <= high_bpm)
    if not torch.any(mask):
        raise ValueError(f"No FFT bins in HR band {low_bpm}..{high_bpm} bpm")
    band_power = power[:, mask]
    band_bpms = bpms[mask]
    weights = torch.softmax(torch.log(band_power) * temperature, dim=1)
    return torch.sum(weights * band_bpms.unsqueeze(0), dim=1)


def rate_bin_auxiliary_loss(
    scalar_outputs: dict[str, torch.Tensor],
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    args: argparse.Namespace,
    pseudo_rr_bpm: torch.Tensor | None = None,
    pseudo_rr_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    total = scalar_targets.new_tensor(0.0)
    used = 0
    sigma_by_task = {
        "hr": max(float(args.rate_bin_hr_sigma_bpm), 1e-6),
        "rr": max(float(args.rate_bin_rr_sigma_bpm), 1e-6),
    }
    for task in ("hr", "rr"):
        logits = scalar_outputs.get(f"_{task}_rate_logits")
        centers = scalar_outputs.get(f"_{task}_rate_bins")
        if logits is None or centers is None:
            continue
        task_idx = TASKS.index(task)
        mask = scalar_masks[:, task_idx]
        target_bpm = denormalize_task(scalar_targets[:, task_idx], task)
        if task == "rr" and pseudo_rr_bpm is not None and pseudo_rr_mask is not None:
            use_pseudo = (mask <= 0) & (pseudo_rr_mask > 0)
            if torch.any(use_pseudo):
                target_bpm = torch.where(use_pseudo, pseudo_rr_bpm.to(target_bpm.device, target_bpm.dtype), target_bpm)
                mask = torch.maximum(mask, pseudo_rr_mask.to(mask.device, mask.dtype))
        if float(mask.sum().detach().cpu()) <= 0.0:
            continue
        centers = centers.to(device=logits.device, dtype=logits.dtype)
        distances = (centers.unsqueeze(0) - target_bpm.unsqueeze(1)).abs()
        soft_target = torch.softmax(-distances / sigma_by_task[task], dim=1)
        log_prob = F.log_softmax(logits, dim=1)
        per_sample = -(soft_target * log_prob).sum(dim=1)
        total = total + masked_mean(per_sample, mask)
        used += 1
    if used == 0:
        return total
    return total / used


def rate_query_auxiliary_loss(
    scalar_outputs: dict[str, torch.Tensor],
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    args: argparse.Namespace,
    pseudo_rr_bpm: torch.Tensor | None = None,
    pseudo_rr_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    total = scalar_targets.new_tensor(0.0)
    used = 0
    sigma_by_task = {
        "hr": max(float(args.rate_query_hr_sigma_bpm), 1e-6),
        "rr": max(float(args.rate_query_rr_sigma_bpm), 1e-6),
    }
    for task in ("hr", "rr"):
        logits = scalar_outputs.get(f"_{task}_rate_query_logits")
        centers = scalar_outputs.get(f"_{task}_rate_query_bins")
        if logits is None or centers is None:
            continue
        task_idx = TASKS.index(task)
        mask = scalar_masks[:, task_idx]
        target_bpm = denormalize_task(scalar_targets[:, task_idx], task)
        if task == "rr" and pseudo_rr_bpm is not None and pseudo_rr_mask is not None:
            use_pseudo = (mask <= 0) & (pseudo_rr_mask > 0)
            if torch.any(use_pseudo):
                target_bpm = torch.where(use_pseudo, pseudo_rr_bpm.to(target_bpm.device, target_bpm.dtype), target_bpm)
                mask = torch.maximum(mask, pseudo_rr_mask.to(mask.device, mask.dtype))
        if float(mask.sum().detach().cpu()) <= 0.0:
            continue
        centers = centers.to(device=logits.device, dtype=logits.dtype)
        distances = (centers.unsqueeze(0) - target_bpm.unsqueeze(1)).abs()
        soft_target = torch.softmax(-distances / sigma_by_task[task], dim=1)
        log_prob = F.log_softmax(logits, dim=1)
        per_sample = -(soft_target * log_prob).sum(dim=1)
        total = total + masked_mean(per_sample, mask)
        used += 1
    if used == 0:
        return total
    return total / used


def rate_bin_entropy_loss(scalar_outputs: dict[str, torch.Tensor], scalar_masks: torch.Tensor) -> torch.Tensor:
    total = scalar_masks.new_tensor(0.0)
    used = 0
    for task in ("hr", "rr"):
        logits = scalar_outputs.get(f"_{task}_rate_logits")
        if logits is None:
            continue
        task_mask = scalar_masks[:, TASKS.index(task)]
        if float(task_mask.sum().detach().cpu()) <= 0.0:
            continue
        prob = torch.softmax(logits, dim=1)
        entropy = -(prob * torch.log(torch.clamp(prob, min=1e-8))).sum(dim=1)
        total = total + masked_mean(entropy, task_mask)
        used += 1
    if used == 0:
        return total
    return total / used


def rate_bin_consistency_loss(
    scalar_outputs: dict[str, torch.Tensor],
    pred_scalars: torch.Tensor,
    scalar_masks: torch.Tensor,
) -> torch.Tensor:
    total = pred_scalars.new_tensor(0.0)
    used = 0
    for task in ("hr", "rr"):
        logits = scalar_outputs.get(f"_{task}_rate_logits")
        centers = scalar_outputs.get(f"_{task}_rate_bins")
        if logits is None or centers is None:
            continue
        task_idx = TASKS.index(task)
        task_mask = scalar_masks[:, task_idx]
        if float(task_mask.sum().detach().cpu()) <= 0.0:
            continue
        centers = centers.to(device=logits.device, dtype=logits.dtype)
        expected_bpm = torch.sum(torch.softmax(logits, dim=1) * centers.unsqueeze(0), dim=1)
        if task == "hr":
            expected_scalar = (expected_bpm - TASK_STATS["hr"]["center"]) / TASK_STATS["hr"]["scale"]
        else:
            expected_scalar = (expected_bpm - TASK_STATS["rr"]["center"]) / TASK_STATS["rr"]["scale"]
        per_sample = F.smooth_l1_loss(pred_scalars[:, task_idx], expected_scalar, reduction="none")
        total = total + masked_mean(per_sample, task_mask)
        used += 1
    if used == 0:
        return total
    return total / used


def state_rate_auxiliary_loss(
    scalar_outputs: dict[str, torch.Tensor],
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    pseudo_rr_bpm: torch.Tensor | None = None,
    pseudo_rr_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    total = scalar_targets.new_tensor(0.0)
    used = 0
    for task in ("hr", "rr"):
        pred_bpm = scalar_outputs.get(f"_{task}_state_rate_bpm")
        if pred_bpm is None:
            continue
        task_idx = TASKS.index(task)
        target_bpm = denormalize_task(scalar_targets[:, task_idx], task)
        mask = scalar_masks[:, task_idx]
        if task == "rr" and pseudo_rr_bpm is not None and pseudo_rr_mask is not None:
            use_pseudo = (mask <= 0) & (pseudo_rr_mask > 0)
            if torch.any(use_pseudo):
                target_bpm = torch.where(use_pseudo, pseudo_rr_bpm.to(target_bpm.device, target_bpm.dtype), target_bpm)
                mask = torch.maximum(mask, pseudo_rr_mask.to(mask.device, mask.dtype))
        if float(mask.sum().detach().cpu()) <= 0.0:
            continue
        pred_norm = (pred_bpm.to(target_bpm.device, target_bpm.dtype) - TASK_STATS[task]["center"]) / TASK_STATS[task]["scale"]
        target_norm = (target_bpm - TASK_STATS[task]["center"]) / TASK_STATS[task]["scale"]
        total = total + masked_mean(F.smooth_l1_loss(pred_norm, target_norm, reduction="none"), mask)
        used += 1
    if used == 0:
        return total
    return total / used


def state_rate_ppg_consistency_loss(
    scalar_outputs: dict[str, torch.Tensor],
    pred_ppg: torch.Tensor,
    ppg_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    total = pred_ppg.new_tensor(0.0)
    used = 0
    for task, weight, low_bpm, high_bpm, temperature in (
        ("hr", 1.0, args.hr_fft_low_bpm, args.hr_fft_high_bpm, args.hr_fft_temperature),
        ("rr", args.state_rate_ppg_rr_weight, args.rr_pseudo_fft_low_bpm, args.rr_pseudo_fft_high_bpm, args.rr_pseudo_fft_temperature),
    ):
        state_bpm = scalar_outputs.get(f"_{task}_state_rate_bpm")
        if state_bpm is None or float(weight) <= 0.0:
            continue
        ppg_bpm = soft_fft_bpm(
            pred_ppg,
            fs=args.fs,
            low_bpm=low_bpm,
            high_bpm=high_bpm,
            nfft=args.hr_fft_nfft if task == "hr" else args.rr_pseudo_fft_nfft,
            temperature=temperature,
        )
        state_norm = (state_bpm.to(ppg_bpm.device, ppg_bpm.dtype) - TASK_STATS[task]["center"]) / TASK_STATS[task]["scale"]
        ppg_norm = (ppg_bpm - TASK_STATS[task]["center"]) / TASK_STATS[task]["scale"]
        total = total + float(weight) * masked_mean(F.smooth_l1_loss(state_norm, ppg_norm, reduction="none"), ppg_mask)
        used += float(weight)
    if used <= 0.0:
        return total
    return total / used


def context_clip_smoothness_loss(
    z_seq: torch.Tensor | None,
    pred_ppg: torch.Tensor,
    ppg_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if z_seq is None or int(args.context_clips) <= 1:
        zero = pred_ppg.new_tensor(0.0)
        return zero, {
            "context_feature_smooth_loss": 0.0,
            "context_hr_smooth_loss": 0.0,
            "context_rr_smooth_loss": 0.0,
        }
    context_clips = int(args.context_clips)
    frames = int(args.frames)
    total_frames = context_clips * frames
    if pred_ppg.shape[1] < total_frames or z_seq.shape[1] < total_frames:
        zero = pred_ppg.new_tensor(0.0)
        return zero, {
            "context_feature_smooth_loss": 0.0,
            "context_hr_smooth_loss": 0.0,
            "context_rr_smooth_loss": 0.0,
        }

    z_context = z_seq[:, :total_frames].reshape(z_seq.shape[0], context_clips, frames, z_seq.shape[-1])
    z_summary = torch.cat(
        [
            z_context.mean(dim=2),
            z_context.std(dim=2, unbiased=False),
        ],
        dim=-1,
    )
    feature_delta = 1.0 - F.cosine_similarity(z_summary[:, 1:], z_summary[:, :-1], dim=-1)
    feature_loss = feature_delta.mean()

    ppg_context = pred_ppg[:, :total_frames].reshape(pred_ppg.shape[0] * context_clips, frames)
    hr_bpm = soft_fft_bpm(
        ppg_context,
        fs=args.fs,
        low_bpm=args.hr_fft_low_bpm,
        high_bpm=args.hr_fft_high_bpm,
        nfft=args.hr_fft_nfft,
        temperature=args.hr_fft_temperature,
    ).reshape(pred_ppg.shape[0], context_clips)
    rr_bpm = soft_fft_bpm(
        ppg_context,
        fs=args.fs,
        low_bpm=args.rr_pseudo_fft_low_bpm,
        high_bpm=args.rr_pseudo_fft_high_bpm,
        nfft=args.rr_pseudo_fft_nfft,
        temperature=args.rr_pseudo_fft_temperature,
    ).reshape(pred_ppg.shape[0], context_clips)
    pair_mask = ppg_mask[:, None].expand(-1, context_clips - 1).reshape(-1)
    hr_delta = (hr_bpm[:, 1:] - hr_bpm[:, :-1]).reshape(-1) / TASK_STATS["hr"]["scale"]
    rr_delta = (rr_bpm[:, 1:] - rr_bpm[:, :-1]).reshape(-1) / TASK_STATS["rr"]["scale"]
    hr_loss = masked_mean(F.smooth_l1_loss(hr_delta, torch.zeros_like(hr_delta), reduction="none"), pair_mask)
    rr_loss = masked_mean(F.smooth_l1_loss(rr_delta, torch.zeros_like(rr_delta), reduction="none"), pair_mask)
    total = (
        args.context_feature_smooth_weight * feature_loss
        + args.context_hr_smooth_weight * hr_loss
        + args.context_rr_smooth_weight * rr_loss
    )
    return total, {
        "context_feature_smooth_loss": float(feature_loss.detach().cpu()),
        "context_hr_smooth_loss": float(hr_loss.detach().cpu()),
        "context_rr_smooth_loss": float(rr_loss.detach().cpu()),
    }


def rate_similarity_contrastive_loss(
    z_seq: torch.Tensor,
    bpm: torch.Tensor,
    mask: torch.Tensor,
    sigma_bpm: float,
    temperature: float,
) -> torch.Tensor:
    valid = mask > 0
    if int(valid.sum().detach().cpu()) < 2:
        return z_seq.new_tensor(0.0)
    pooled = torch.cat(
        [
            torch.mean(z_seq, dim=1),
            torch.std(z_seq, dim=1, unbiased=False),
        ],
        dim=-1,
    )
    pooled = F.normalize(pooled, dim=-1)
    sim = torch.matmul(pooled, pooled.transpose(0, 1)) / max(float(temperature), 1e-6)
    pair_mask = valid[:, None] & valid[None, :]
    eye = torch.eye(z_seq.shape[0], device=z_seq.device, dtype=torch.bool)
    pair_mask = pair_mask & ~eye
    if int(pair_mask.sum().detach().cpu()) == 0:
        return z_seq.new_tensor(0.0)
    rate_dist = torch.abs(bpm[:, None] - bpm[None, :])
    target = torch.exp(-rate_dist / max(float(sigma_bpm), 1e-6)) * pair_mask.to(z_seq.dtype)
    target_sum = target.sum(dim=1, keepdim=True)
    anchor_mask = target_sum.squeeze(1) > 0
    if int(anchor_mask.sum().detach().cpu()) == 0:
        return z_seq.new_tensor(0.0)
    target = target / torch.clamp(target_sum, min=1e-8)
    sim = sim.masked_fill(~pair_mask, -1e4)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(target * log_prob).sum(dim=1)
    return loss[anchor_mask].mean()


def task_stream_rate_contrastive_loss(
    hr_seq: torch.Tensor | None,
    rr_seq: torch.Tensor | None,
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    ppg: torch.Tensor,
    ppg_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = ppg.new_tensor(0.0)
    hr_loss = base
    rr_loss = base
    if hr_seq is None and rr_seq is None:
        raise RuntimeError("task_stream_spectral_alignment_weight is enabled but no HR/RR streams were exposed")
    if hr_seq is not None:
        hr_bpm = denormalize_task(scalar_targets[:, TASKS.index("hr")], "hr")
        hr_mask = scalar_masks[:, TASKS.index("hr")]
        hr_loss = rate_similarity_contrastive_loss(
            hr_seq,
            hr_bpm,
            hr_mask,
            sigma_bpm=args.task_stream_hr_sigma,
            temperature=args.task_stream_temperature,
        )
    if rr_seq is not None:
        rr_target_mask = scalar_masks[:, TASKS.index("rr")]
        if float(rr_target_mask.detach().sum().cpu()) > 1.0:
            rr_bpm = denormalize_task(scalar_targets[:, TASKS.index("rr")], "rr")
            rr_mask = rr_target_mask
        else:
            rr_bpm = soft_fft_bpm(
                ppg,
                fs=args.fs,
                low_bpm=args.task_stream_rr_low_bpm,
                high_bpm=args.task_stream_rr_high_bpm,
                nfft=args.task_stream_fft_nfft,
                temperature=args.task_stream_fft_temperature,
            )
            rr_mask = ppg_mask
        rr_loss = rate_similarity_contrastive_loss(
            rr_seq,
            rr_bpm,
            rr_mask,
            sigma_bpm=args.task_stream_rr_sigma,
            temperature=args.task_stream_temperature,
        )
    total = hr_loss + args.task_stream_rr_weight * rr_loss
    return total, {
        "task_stream_contrastive_loss": float(total.detach().cpu()),
        "task_stream_hr_contrastive_loss": float(hr_loss.detach().cpu()),
        "task_stream_rr_contrastive_loss": float(rr_loss.detach().cpu()),
    }


def rate_conditioned_cross_domain_loss(
    z_seq: torch.Tensor | None,
    bpm: torch.Tensor,
    mask: torch.Tensor,
    dataset_ids: torch.Tensor,
    sigma_bpm: float,
    temperature: float,
) -> torch.Tensor:
    if z_seq is None:
        return bpm.new_tensor(0.0)
    valid = mask > 0
    if int(valid.sum().detach().cpu()) < 2:
        return z_seq.new_tensor(0.0)
    pooled = torch.cat(
        [
            torch.mean(z_seq, dim=1),
            torch.std(z_seq, dim=1, unbiased=False),
        ],
        dim=-1,
    )
    pooled = F.normalize(pooled, dim=-1)
    sim = torch.matmul(pooled, pooled.transpose(0, 1)) / max(float(temperature), 1e-6)
    eye = torch.eye(z_seq.shape[0], device=z_seq.device, dtype=torch.bool)
    cross_domain = dataset_ids[:, None] != dataset_ids[None, :]
    pair_mask = valid[:, None] & valid[None, :] & cross_domain & ~eye
    if int(pair_mask.sum().detach().cpu()) == 0:
        return z_seq.new_tensor(0.0)
    rate_dist = torch.abs(bpm[:, None] - bpm[None, :])
    target = torch.exp(-rate_dist.square() / (2.0 * max(float(sigma_bpm), 1e-6) ** 2))
    target = target * pair_mask.to(z_seq.dtype)
    target_sum = target.sum(dim=1, keepdim=True)
    anchor_mask = target_sum.squeeze(1) > 0
    if int(anchor_mask.sum().detach().cpu()) == 0:
        return z_seq.new_tensor(0.0)
    target = target / torch.clamp(target_sum, min=1e-8)
    sim = sim.masked_fill(~pair_mask, -1e4)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(target * log_prob).sum(dim=1)
    return loss[anchor_mask].mean()


def task_stream_cross_domain_alignment_loss(
    hr_seq: torch.Tensor | None,
    rr_seq: torch.Tensor | None,
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    ppg: torch.Tensor,
    ppg_mask: torch.Tensor,
    dataset_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = ppg.new_tensor(0.0)
    if int(torch.unique(dataset_ids).numel()) < 2:
        return base, {
            "task_stream_cross_domain_alignment_loss": 0.0,
            "task_stream_cross_domain_hr_loss": 0.0,
            "task_stream_cross_domain_rr_loss": 0.0,
        }
    hr_loss = base
    rr_loss = base
    if hr_seq is not None:
        hr_bpm = denormalize_task(scalar_targets[:, TASKS.index("hr")], "hr")
        hr_mask = scalar_masks[:, TASKS.index("hr")]
        hr_loss = rate_conditioned_cross_domain_loss(
            hr_seq,
            hr_bpm,
            hr_mask,
            dataset_ids,
            sigma_bpm=args.cross_domain_hr_sigma,
            temperature=args.cross_domain_temperature,
        )
    if rr_seq is not None:
        rr_target_mask = scalar_masks[:, TASKS.index("rr")]
        if float(rr_target_mask.detach().sum().cpu()) > 1.0:
            rr_bpm = denormalize_task(scalar_targets[:, TASKS.index("rr")], "rr")
            rr_mask = rr_target_mask
        else:
            rr_bpm = soft_fft_bpm(
                ppg,
                fs=args.fs,
                low_bpm=args.cross_domain_rr_low_bpm,
                high_bpm=args.cross_domain_rr_high_bpm,
                nfft=args.cross_domain_rr_nfft,
                temperature=args.cross_domain_rr_fft_temperature,
            )
            rr_mask = ppg_mask
        rr_loss = rate_conditioned_cross_domain_loss(
            rr_seq,
            rr_bpm,
            rr_mask,
            dataset_ids,
            sigma_bpm=args.cross_domain_rr_sigma,
            temperature=args.cross_domain_temperature,
        )
    total = hr_loss + args.cross_domain_rr_weight * rr_loss
    return total, {
        "task_stream_cross_domain_alignment_loss": float(total.detach().cpu()),
        "task_stream_cross_domain_hr_loss": float(hr_loss.detach().cpu()),
        "task_stream_cross_domain_rr_loss": float(rr_loss.detach().cpu()),
    }


def source_rate_ordering_loss(
    z_seq: torch.Tensor | None,
    bpm: torch.Tensor,
    mask: torch.Tensor,
    dataset_ids: torch.Tensor,
    scale_bpm: float,
    cross_domain_only: bool,
) -> torch.Tensor:
    if z_seq is None:
        return bpm.new_tensor(0.0)
    valid = mask > 0
    if int(valid.sum().detach().cpu()) < 2:
        return z_seq.new_tensor(0.0)
    pooled = torch.cat([torch.mean(z_seq, dim=1), torch.std(z_seq, dim=1, unbiased=False)], dim=-1)
    pooled = F.normalize(pooled, dim=-1)
    feature_dist = (1.0 - torch.matmul(pooled, pooled.transpose(0, 1))).clamp(min=0.0, max=2.0) * 0.5
    rate_dist = (torch.abs(bpm[:, None] - bpm[None, :]) / max(float(scale_bpm), 1e-6)).clamp(max=1.0)
    eye = torch.eye(z_seq.shape[0], device=z_seq.device, dtype=torch.bool)
    pair_mask = valid[:, None] & valid[None, :] & ~eye
    if cross_domain_only:
        pair_mask = pair_mask & (dataset_ids[:, None] != dataset_ids[None, :])
    if int(pair_mask.sum().detach().cpu()) == 0:
        return z_seq.new_tensor(0.0)
    return F.smooth_l1_loss(feature_dist[pair_mask], rate_dist.to(feature_dist.dtype)[pair_mask])


def source_task_stream_rate_ordering_loss(
    hr_seq: torch.Tensor | None,
    rr_seq: torch.Tensor | None,
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    ppg: torch.Tensor,
    ppg_mask: torch.Tensor,
    dataset_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = ppg.new_tensor(0.0)
    hr_loss = base
    rr_loss = base
    if hr_seq is not None:
        hr_bpm = denormalize_task(scalar_targets[:, TASKS.index("hr")], "hr")
        hr_mask = scalar_masks[:, TASKS.index("hr")]
        hr_loss = source_rate_ordering_loss(
            hr_seq,
            hr_bpm,
            hr_mask,
            dataset_ids,
            scale_bpm=args.source_rate_ordering_hr_scale,
            cross_domain_only=args.source_rate_ordering_cross_domain_only,
        )
    if rr_seq is not None:
        rr_target_mask = scalar_masks[:, TASKS.index("rr")]
        if float(rr_target_mask.detach().sum().cpu()) > 1.0:
            rr_bpm = denormalize_task(scalar_targets[:, TASKS.index("rr")], "rr")
            rr_mask = rr_target_mask
        else:
            rr_bpm = soft_fft_bpm(
                ppg,
                fs=args.fs,
                low_bpm=args.source_rate_ordering_rr_low_bpm,
                high_bpm=args.source_rate_ordering_rr_high_bpm,
                nfft=args.source_rate_ordering_rr_nfft,
                temperature=args.source_rate_ordering_rr_fft_temperature,
            )
            rr_mask = ppg_mask
        rr_loss = source_rate_ordering_loss(
            rr_seq,
            rr_bpm,
            rr_mask,
            dataset_ids,
            scale_bpm=args.source_rate_ordering_rr_scale,
            cross_domain_only=args.source_rate_ordering_cross_domain_only,
        )
    total = hr_loss + args.source_rate_ordering_rr_weight * rr_loss
    return total, {
        "source_rate_ordering_loss": float(total.detach().cpu()),
        "source_rate_ordering_hr_loss": float(hr_loss.detach().cpu()),
        "source_rate_ordering_rr_loss": float(rr_loss.detach().cpu()),
    }


def source_episodic_residual_adaptation_loss(
    scalar_outputs: dict[str, torch.Tensor],
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    dataset_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_scalars = scalar_predictions_to_tensor(scalar_outputs, scalar_targets.device)
    tasks = [item.strip() for item in str(args.source_episode_tasks or "").split(",") if item.strip()]
    if not tasks:
        tasks = ["hr", "rr", "spo2"]
    task_indices = [TASKS.index(task) for task in tasks if task in TASKS]
    if not task_indices:
        zero = scalar_targets.new_tensor(0.0)
        return zero, {"source_episode_residual_loss": 0.0, "source_episode_domains": 0.0}

    task_weight_values = {
        "pr": float(args.pr_weight),
        "hr": float(args.hr_weight),
        "rr": float(args.rr_weight),
        "spo2": float(args.spo2_weight),
        "sbp": float(args.sbp_weight),
        "dbp": float(args.dbp_weight),
        "map": float(args.map_weight),
    }
    total = scalar_targets.new_tensor(0.0)
    used_terms = 0
    used_domains = 0
    min_support = max(int(args.source_episode_min_support), 1)
    min_query = max(int(args.source_episode_min_query), 1)
    for dataset_id in torch.unique(dataset_ids):
        domain_idx = torch.nonzero(dataset_ids == dataset_id, as_tuple=False).flatten()
        if int(domain_idx.numel()) < min_support + min_query:
            continue
        support_idx = domain_idx[::2]
        query_idx = domain_idx[1::2]
        if int(support_idx.numel()) < min_support or int(query_idx.numel()) < min_query:
            support_idx = domain_idx[:min_support]
            query_idx = domain_idx[min_support:]
        if int(query_idx.numel()) < min_query:
            continue
        domain_used = False
        for task_idx in task_indices:
            task = TASKS[task_idx]
            support_mask = scalar_masks[support_idx, task_idx]
            query_mask = scalar_masks[query_idx, task_idx]
            if float(support_mask.detach().sum().cpu()) < min_support:
                continue
            if float(query_mask.detach().sum().cpu()) <= 0.0:
                continue
            residual = masked_mean(
                scalar_targets[support_idx, task_idx] - pred_scalars[support_idx, task_idx],
                support_mask,
            )
            adapted_query = pred_scalars[query_idx, task_idx] + float(args.source_episode_residual_scale) * residual
            query_loss = masked_mean(
                F.smooth_l1_loss(adapted_query, scalar_targets[query_idx, task_idx], reduction="none"),
                query_mask,
            )
            task_weight = max(task_weight_values.get(task, 1.0), 0.0)
            if task_weight <= 0.0:
                task_weight = 1.0
            total = total + task_weight * query_loss
            used_terms += 1
            domain_used = True
        if domain_used:
            used_domains += 1
    if used_terms == 0:
        zero = scalar_targets.new_tensor(0.0)
        return zero, {"source_episode_residual_loss": 0.0, "source_episode_domains": 0.0}
    total = total / used_terms
    return total, {
        "source_episode_residual_loss": float(total.detach().cpu()),
        "source_episode_domains": float(used_domains),
    }


def stream_soft_fft_bpm(
    seq: torch.Tensor,
    fs: float,
    low_bpm: float,
    high_bpm: float,
    nfft: int,
    temperature: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if seq.dim() != 3:
        raise ValueError(f"Expected stream sequence [B,T,C], got {tuple(seq.shape)}")
    x = seq - seq.mean(dim=1, keepdim=True)
    x = x / (x.std(dim=1, keepdim=True, unbiased=False) + eps)
    n_fft = max(int(nfft), int(seq.shape[1]))
    window = torch.hann_window(seq.shape[1], device=seq.device, dtype=seq.dtype).view(1, -1, 1)
    spectrum = torch.fft.rfft(x * window, n=n_fft, dim=1)
    power = spectrum.real.square() + spectrum.imag.square() + eps
    power = power.mean(dim=2)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs).to(seq.device)
    bpms = freqs * 60.0
    mask = (bpms >= low_bpm) & (bpms <= high_bpm)
    if not torch.any(mask):
        raise ValueError(f"No FFT bins in stream band {low_bpm}..{high_bpm} bpm")
    band_power = power[:, mask]
    band_bpms = bpms[mask]
    weights = torch.softmax(torch.log(band_power) * temperature, dim=1)
    return torch.sum(weights * band_bpms.unsqueeze(0), dim=1)


def task_stream_spectral_alignment_loss(
    hr_seq: torch.Tensor | None,
    rr_seq: torch.Tensor | None,
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    ppg: torch.Tensor,
    ppg_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = ppg.new_tensor(0.0)
    hr_loss = base
    rr_loss = base
    if hr_seq is not None:
        pred_hr_bpm = stream_soft_fft_bpm(
            hr_seq,
            fs=args.fs,
            low_bpm=args.stream_spectral_hr_low_bpm,
            high_bpm=args.stream_spectral_hr_high_bpm,
            nfft=args.stream_spectral_hr_nfft,
            temperature=args.stream_spectral_temperature,
        )
        hr_target_norm = scalar_targets[:, TASKS.index("hr")]
        pred_hr_norm = (pred_hr_bpm - TASK_STATS["hr"]["center"]) / TASK_STATS["hr"]["scale"]
        hr_mask = scalar_masks[:, TASKS.index("hr")]
        hr_loss = masked_mean(F.smooth_l1_loss(pred_hr_norm, hr_target_norm, reduction="none"), hr_mask)
    if rr_seq is not None:
        rr_target_mask = scalar_masks[:, TASKS.index("rr")]
        if float(rr_target_mask.detach().sum().cpu()) > 0.0:
            rr_target_norm = scalar_targets[:, TASKS.index("rr")]
            rr_mask = rr_target_mask
        else:
            pseudo_rr_bpm = soft_fft_bpm(
                ppg,
                fs=args.fs,
                low_bpm=args.stream_spectral_rr_low_bpm,
                high_bpm=args.stream_spectral_rr_high_bpm,
                nfft=args.stream_spectral_rr_nfft,
                temperature=args.stream_spectral_temperature,
            )
            rr_target_norm = (pseudo_rr_bpm - TASK_STATS["rr"]["center"]) / TASK_STATS["rr"]["scale"]
            rr_mask = ppg_mask
        pred_rr_bpm = stream_soft_fft_bpm(
            rr_seq,
            fs=args.fs,
            low_bpm=args.stream_spectral_rr_low_bpm,
            high_bpm=args.stream_spectral_rr_high_bpm,
            nfft=args.stream_spectral_rr_nfft,
            temperature=args.stream_spectral_temperature,
        )
        pred_rr_norm = (pred_rr_bpm - TASK_STATS["rr"]["center"]) / TASK_STATS["rr"]["scale"]
        rr_loss = masked_mean(F.smooth_l1_loss(pred_rr_norm, rr_target_norm, reduction="none"), rr_mask)
    total = hr_loss + args.stream_spectral_rr_weight * rr_loss
    return total, {
        "task_stream_spectral_alignment_loss": float(total.detach().cpu()),
        "task_stream_spectral_hr_loss": float(hr_loss.detach().cpu()),
        "task_stream_spectral_rr_loss": float(rr_loss.detach().cpu()),
    }


def temporal_model_from(model):
    return getattr(model.module, "temporal_model", None) if hasattr(model, "module") else getattr(model, "temporal_model", None)


def add_adaptive_prior_logs(logs: dict[str, float], model) -> None:
    temporal_model = temporal_model_from(model)
    if temporal_model is None:
        return
    weights = getattr(temporal_model, "last_adaptive_prior_weights", None)
    if weights is not None:
        weights_cpu = weights.detach().cpu().flatten()
        for idx, name in enumerate(("slow", "state", "env")):
            if idx < weights_cpu.numel():
                logs[f"adaptive_prior_{name}_weight"] = float(weights_cpu[idx])
    gates = getattr(temporal_model, "last_adaptive_prior_gate_values", None)
    if gates is not None:
        gates_cpu = gates.detach().cpu().flatten()
        for idx, name in enumerate(("rr", "hr", "output")):
            if idx < gates_cpu.numel():
                logs[f"adaptive_prior_{name}_gate"] = float(gates_cpu[idx])


def band_stream_disentangle_loss(hr_seq: torch.Tensor | None, rr_seq: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor | None:
    if hr_seq is None or rr_seq is None:
        return None
    hr = torch.cat([hr_seq.mean(dim=1), hr_seq.std(dim=1, unbiased=False)], dim=-1)
    rr = torch.cat([rr_seq.mean(dim=1), rr_seq.std(dim=1, unbiased=False)], dim=-1)
    hr = (hr - hr.mean(dim=0, keepdim=True)) / (hr.std(dim=0, keepdim=True, unbiased=False) + eps)
    rr = (rr - rr.mean(dim=0, keepdim=True)) / (rr.std(dim=0, keepdim=True, unbiased=False) + eps)
    corr = torch.mean(hr * rr, dim=0).square().mean()
    hr_var = torch.sqrt(hr_seq.var(dim=(0, 1), unbiased=False) + eps)
    rr_var = torch.sqrt(rr_seq.var(dim=(0, 1), unbiased=False) + eps)
    var_floor = F.relu(1.0 - hr_var).mean() + F.relu(1.0 - rr_var).mean()
    return corr + 0.1 * var_floor


def strict_train_sanity_check(
    loss: torch.Tensor,
    logs: dict[str, float],
    temporal_model,
    args: argparse.Namespace,
) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu())}")
    for key, value in logs.items():
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Non-finite metric {key}: {value}")
    if args.band_disentangle_weight <= 0.0 and not str(args.temporal_module).startswith("complex_dual_band"):
        return
    hr_seq = getattr(temporal_model, "last_hr_seq", None)
    rr_seq = getattr(temporal_model, "last_rr_seq", None)
    if hr_seq is None or rr_seq is None:
        raise RuntimeError("Temporal model did not expose last_hr_seq/last_rr_seq")
    if hr_seq.shape != rr_seq.shape:
        raise RuntimeError(f"HR/RR stream shape mismatch: {tuple(hr_seq.shape)} vs {tuple(rr_seq.shape)}")
    if hr_seq.dim() != 3:
        raise RuntimeError(f"Expected HR/RR streams to be [B,T,C], got {tuple(hr_seq.shape)}")
    if not torch.isfinite(hr_seq).all() or not torch.isfinite(rr_seq).all():
        raise FloatingPointError("Non-finite HR/RR stream values")
    if args.band_disentangle_weight > 0.0 and "band_disentangle_loss" not in logs:
        raise RuntimeError("band_disentangle_weight is enabled but band_disentangle_loss was not logged")


def multitask_loss(
    pred_ppg: torch.Tensor,
    ppg: torch.Tensor,
    ppg_mask: torch.Tensor,
    scalar_outputs: dict[str, torch.Tensor],
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    recon_loss: torch.Tensor,
    args: argparse.Namespace,
    task_log_vars: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_ppg = normalize_signal_batch(pred_ppg)
    ppg = normalize_signal_batch(ppg)
    ppg_losses = neg_pearson_per_sample(pred_ppg, ppg)
    ppg_loss = masked_mean(ppg_losses, ppg_mask)

    pred_scalars = scalar_predictions_to_tensor(scalar_outputs, scalar_targets.device)
    scalar_losses = F.smooth_l1_loss(pred_scalars, scalar_targets, reduction="none")
    task_losses = {}
    scalar_loss = scalar_targets.new_tensor(0.0)
    task_weights = torch.tensor(
        [args.pr_weight, args.hr_weight, args.rr_weight, args.spo2_weight, args.sbp_weight, args.dbp_weight, args.map_weight],
        device=scalar_targets.device,
    )
    if task_log_vars is not None:
        task_log_vars = task_log_vars.to(device=scalar_targets.device, dtype=scalar_targets.dtype)
        if task_log_vars.numel() != len(TASKS):
            raise ValueError(f"Expected {len(TASKS)} task log vars, got {task_log_vars.numel()}")
    for task_idx, task in enumerate(TASKS):
        task_mask = scalar_masks[:, task_idx]
        task_loss = masked_mean(scalar_losses[:, task_idx], task_mask)
        task_losses[f"{task}_loss"] = float(task_loss.detach().cpu())
        if task_log_vars is None:
            scalar_loss = scalar_loss + task_weights[task_idx] * task_loss
        else:
            precision = torch.exp(-task_log_vars[task_idx])
            scalar_loss = scalar_loss + precision * task_weights[task_idx] * task_loss + task_log_vars[task_idx]

    hr_freq_loss = scalar_targets.new_tensor(0.0)
    if args.hr_from_ppg_weight > 0.0:
        pred_hr_bpm = soft_fft_bpm(
            pred_ppg,
            fs=args.fs,
            low_bpm=args.hr_fft_low_bpm,
            high_bpm=args.hr_fft_high_bpm,
            nfft=args.hr_fft_nfft,
            temperature=args.hr_fft_temperature,
        )
        hr_target_norm = scalar_targets[:, TASKS.index("hr")]
        pred_hr_norm = (pred_hr_bpm - TASK_STATS["hr"]["center"]) / TASK_STATS["hr"]["scale"]
        hr_mask = scalar_masks[:, TASKS.index("hr")] * ppg_mask
        hr_freq_loss = masked_mean(F.smooth_l1_loss(pred_hr_norm, hr_target_norm, reduction="none"), hr_mask)

    rr_freq_loss = scalar_targets.new_tensor(0.0)
    if args.rr_from_ppg_weight > 0.0:
        pred_rr_bpm = soft_fft_bpm(
            pred_ppg,
            fs=args.fs,
            low_bpm=args.rr_fft_low_bpm,
            high_bpm=args.rr_fft_high_bpm,
            nfft=args.rr_fft_nfft,
            temperature=args.rr_fft_temperature,
        )
        rr_target_norm = scalar_targets[:, TASKS.index("rr")]
        pred_rr_norm = (pred_rr_bpm - TASK_STATS["rr"]["center"]) / TASK_STATS["rr"]["scale"]
        rr_mask = scalar_masks[:, TASKS.index("rr")] * ppg_mask
        rr_freq_loss = masked_mean(F.smooth_l1_loss(pred_rr_norm, rr_target_norm, reduction="none"), rr_mask)

    rr_pseudo_loss = scalar_targets.new_tensor(0.0)
    if args.rr_pseudo_from_ppg_weight > 0.0:
        pseudo_rr_bpm = soft_fft_bpm(
            ppg,
            fs=args.fs,
            low_bpm=args.rr_pseudo_fft_low_bpm,
            high_bpm=args.rr_pseudo_fft_high_bpm,
            nfft=args.rr_pseudo_fft_nfft,
            temperature=args.rr_pseudo_fft_temperature,
        )
        pseudo_rr_norm = (pseudo_rr_bpm - TASK_STATS["rr"]["center"]) / TASK_STATS["rr"]["scale"]
        pred_rr_norm = pred_scalars[:, TASKS.index("rr")]
        rr_pseudo_loss = masked_mean(F.smooth_l1_loss(pred_rr_norm, pseudo_rr_norm, reduction="none"), ppg_mask)

    pseudo_rr_bpm_for_rate_bins = None
    pseudo_rr_mask_for_rate_bins = None
    if (
        args.rr_pseudo_rate_bin_weight > 0.0
        and (scalar_outputs.get("_rr_rate_logits") is not None or scalar_outputs.get("_rr_rate_query_logits") is not None)
    ):
        pseudo_rr_bpm_for_rate_bins = soft_fft_bpm(
            ppg,
            fs=args.fs,
            low_bpm=args.rr_pseudo_fft_low_bpm,
            high_bpm=args.rr_pseudo_fft_high_bpm,
            nfft=args.rr_pseudo_fft_nfft,
            temperature=args.rr_pseudo_fft_temperature,
        )
        pseudo_rr_mask_for_rate_bins = ppg_mask * float(args.rr_pseudo_rate_bin_weight)

    rate_bin_loss = scalar_targets.new_tensor(0.0)
    if args.rate_bin_aux_weight > 0.0:
        rate_bin_loss = rate_bin_auxiliary_loss(
            scalar_outputs,
            scalar_targets,
            scalar_masks,
            args,
            pseudo_rr_bpm=pseudo_rr_bpm_for_rate_bins,
            pseudo_rr_mask=pseudo_rr_mask_for_rate_bins,
        )
    rate_query_loss = scalar_targets.new_tensor(0.0)
    if args.rate_query_aux_weight > 0.0:
        rate_query_loss = rate_query_auxiliary_loss(
            scalar_outputs,
            scalar_targets,
            scalar_masks,
            args,
            pseudo_rr_bpm=pseudo_rr_bpm_for_rate_bins,
            pseudo_rr_mask=pseudo_rr_mask_for_rate_bins,
        )
    rate_bin_entropy = scalar_targets.new_tensor(0.0)
    if args.rate_bin_entropy_weight > 0.0:
        rate_bin_entropy = rate_bin_entropy_loss(scalar_outputs, scalar_masks)
    rate_bin_consistency = scalar_targets.new_tensor(0.0)
    if args.rate_bin_consistency_weight > 0.0:
        rate_bin_consistency = rate_bin_consistency_loss(scalar_outputs, pred_scalars, scalar_masks)
    state_rate_loss = scalar_targets.new_tensor(0.0)
    if args.state_rate_aux_weight > 0.0:
        state_rate_loss = state_rate_auxiliary_loss(
            scalar_outputs,
            scalar_targets,
            scalar_masks,
            pseudo_rr_bpm=pseudo_rr_bpm_for_rate_bins,
            pseudo_rr_mask=pseudo_rr_mask_for_rate_bins,
        )
    state_rate_ppg_loss = scalar_targets.new_tensor(0.0)
    if args.state_rate_ppg_consistency_weight > 0.0:
        state_rate_ppg_loss = state_rate_ppg_consistency_loss(scalar_outputs, pred_ppg, ppg_mask, args)

    total = (
        args.ppg_weight * ppg_loss
        + scalar_loss
        + args.hr_from_ppg_weight * hr_freq_loss
        + args.rr_from_ppg_weight * rr_freq_loss
        + args.rr_pseudo_from_ppg_weight * rr_pseudo_loss
        + args.rate_bin_aux_weight * rate_bin_loss
        + args.rate_query_aux_weight * rate_query_loss
        + args.rate_bin_entropy_weight * rate_bin_entropy
        + args.rate_bin_consistency_weight * rate_bin_consistency
        + args.state_rate_aux_weight * state_rate_loss
        + args.state_rate_ppg_consistency_weight * state_rate_ppg_loss
        + args.recon_weight * recon_loss
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "ppg_loss": float(ppg_loss.detach().cpu()),
        "hr_from_ppg_loss": float(hr_freq_loss.detach().cpu()),
        "rr_from_ppg_loss": float(rr_freq_loss.detach().cpu()),
        "rr_pseudo_from_ppg_loss": float(rr_pseudo_loss.detach().cpu()),
        "rate_bin_aux_loss": float(rate_bin_loss.detach().cpu()),
        "rate_query_aux_loss": float(rate_query_loss.detach().cpu()),
        "rate_bin_entropy_loss": float(rate_bin_entropy.detach().cpu()),
        "rate_bin_consistency_loss": float(rate_bin_consistency.detach().cpu()),
        "state_rate_aux_loss": float(state_rate_loss.detach().cpu()),
        "state_rate_ppg_consistency_loss": float(state_rate_ppg_loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        **task_losses,
    }
    return total, logs


def source_rate_group_dro_loss(
    scalar_outputs: dict[str, torch.Tensor],
    scalar_targets: torch.Tensor,
    scalar_masks: torch.Tensor,
    dataset_ids: torch.Tensor,
    ppg: torch.Tensor,
    ppg_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_scalars = scalar_predictions_to_tensor(scalar_outputs, scalar_targets.device)
    task_names = [item.strip() for item in str(args.source_rate_group_dro_tasks or "").split(",") if item.strip()]
    if not task_names:
        task_names = ["hr", "rr"]

    losses = []
    group_count = 0
    min_count = max(int(args.source_rate_group_dro_min_count), 1)
    eta = max(float(args.source_rate_group_dro_eta), 1e-6)
    widths = {
        "hr": max(float(args.source_rate_group_hr_bin_width), 1.0),
        "rr": max(float(args.source_rate_group_rr_bin_width), 0.5),
        "spo2": max(float(args.source_rate_group_spo2_bin_width), 0.5),
    }
    task_weights = {
        "hr": max(float(args.hr_weight), 0.0),
        "rr": max(float(args.rr_weight), 0.0),
        "spo2": max(float(args.spo2_weight), 0.0),
    }
    pseudo_rr_bpm = None
    if "rr" in task_names and float(args.source_rate_group_dro_pseudo_rr_weight) > 0.0:
        pseudo_rr_bpm = soft_fft_bpm(
            ppg,
            fs=args.fs,
            low_bpm=args.rr_pseudo_fft_low_bpm,
            high_bpm=args.rr_pseudo_fft_high_bpm,
            nfft=args.rr_pseudo_fft_nfft,
            temperature=args.rr_pseudo_fft_temperature,
        )
    for task in task_names:
        if task not in TASKS or task not in widths:
            continue
        task_idx = TASKS.index(task)
        mask = scalar_masks[:, task_idx] > 0
        target = scalar_targets[:, task_idx]
        if task == "rr" and pseudo_rr_bpm is not None:
            use_pseudo = (scalar_masks[:, task_idx] <= 0) & (ppg_mask > 0)
            pseudo_target = (pseudo_rr_bpm - TASK_STATS["rr"]["center"]) / TASK_STATS["rr"]["scale"]
            target = torch.where(use_pseudo, pseudo_target.to(target.device, target.dtype), target)
            mask = mask | use_pseudo
        if int(mask.sum().detach().cpu()) < min_count:
            continue
        per_sample = F.smooth_l1_loss(pred_scalars[:, task_idx], target, reduction="none")
        bpm = denormalize_task(target, task)
        bins = torch.floor(bpm / widths[task]).to(torch.long)
        group_keys = dataset_ids.to(torch.long) * 10000 + bins
        for key in torch.unique(group_keys[mask]):
            group_mask = mask & (group_keys == key)
            if int(group_mask.sum().detach().cpu()) < min_count:
                continue
            group_loss = per_sample[group_mask].mean()
            weight = task_weights.get(task, 1.0)
            if task == "rr" and pseudo_rr_bpm is not None:
                weight *= float(args.source_rate_group_dro_pseudo_rr_weight)
            losses.append(group_loss * (weight if weight > 0.0 else 1.0))
            group_count += 1
    if not losses:
        zero = scalar_targets.new_tensor(0.0)
        return zero, {
            "source_rate_group_dro_loss": 0.0,
            "source_rate_group_dro_groups": 0.0,
            "source_rate_group_dro_worst": 0.0,
        }
    stacked = torch.stack(losses)
    robust = eta * torch.logsumexp(stacked / eta, dim=0) - eta * math.log(float(stacked.numel()))
    return robust, {
        "source_rate_group_dro_loss": float(robust.detach().cpu()),
        "source_rate_group_dro_groups": float(group_count),
        "source_rate_group_dro_worst": float(stacked.detach().max().cpu()),
    }


def train_epoch(
    model,
    loader,
    optimizer,
    args,
    device,
    task_log_vars: torch.Tensor | None = None,
    domain_adversary: DomainAdversary | None = None,
    task_stream_domain_adversary: TaskStreamDomainAdversary | None = None,
    task_stream_subject_adversary: TaskStreamDomainAdversary | None = None,
    subject_to_idx: dict[str, int] | None = None,
    l2_sp_anchors: dict[str, torch.Tensor] | None = None,
    ema: "ModelEMA | None" = None,
):
    model.train()
    if domain_adversary is not None:
        domain_adversary.train()
    if task_stream_domain_adversary is not None:
        task_stream_domain_adversary.train()
    if task_stream_subject_adversary is not None:
        task_stream_subject_adversary.train()
    sums = {}
    count = 0
    for video, ppg, ppg_mask, scalar_targets, scalar_masks, dataset_ids, subjects, sort_indices in loader:
        video = prepare_video(video, args, device)
        ppg = ppg.to(device, dtype=torch.float32)
        ppg_mask = ppg_mask.to(device, dtype=torch.float32)
        scalar_targets = scalar_targets.to(device, dtype=torch.float32)
        scalar_masks = scalar_masks.to(device, dtype=torch.float32)
        dataset_ids = dataset_ids.to(device, dtype=torch.long)
        frame_offsets = sort_indices.to(device) * args.frames if args.absolute_time_decoder else None
        needs_features = (
            args.rate_contrastive_weight > 0.0
            or args.task_stream_contrastive_weight > 0.0
            or args.band_disentangle_weight > 0.0
            or args.context_smooth_weight > 0.0
            or args.domain_adversarial_weight > 0.0
            or args.task_stream_domain_adversarial_weight > 0.0
            or args.task_stream_subject_adversarial_weight > 0.0
            or args.source_view_consistency_weight > 0.0
            or args.source_view_prediction_consistency_weight > 0.0
            or args.target_view_consistency_weight > 0.0
            or args.target_view_prediction_consistency_weight > 0.0
            or args.target_view_token_vicreg_weight > 0.0
            or args.source_masked_prediction_weight > 0.0
            or args.factor_regularization_weight > 0.0
            or args.task_stream_spectral_alignment_weight > 0.0
            or args.task_stream_cross_domain_alignment_weight > 0.0
            or args.source_rate_ordering_weight > 0.0
        )
        if needs_features:
            pred_ppg, recon_loss, _, scalar_outputs, z_clean_seq, _ = model.forward_with_multitask_features(
                video,
                frame_offsets=frame_offsets,
                domain_ids=dataset_ids,
                support_context=getattr(args, "support_context_tensor", None),
            )
        else:
            pred_ppg, recon_loss, _, scalar_outputs = model.forward_with_multitask(
                video,
                frame_offsets=frame_offsets,
                domain_ids=dataset_ids,
                support_context=getattr(args, "support_context_tensor", None),
            )
            z_clean_seq = None
        loss, logs = multitask_loss(
            pred_ppg,
            ppg,
            ppg_mask,
            scalar_outputs,
            scalar_targets,
            scalar_masks,
            recon_loss,
            args,
            task_log_vars=task_log_vars,
        )
        add_adaptive_prior_logs(logs, model)
        if args.source_rate_group_dro_weight > 0.0:
            group_loss, group_logs = source_rate_group_dro_loss(
                scalar_outputs,
                scalar_targets,
                scalar_masks,
                dataset_ids,
                ppg,
                ppg_mask,
                args,
            )
            loss = loss + args.source_rate_group_dro_weight * group_loss
            logs.update(group_logs)
        if (
            (args.source_view_consistency_weight > 0.0 or args.source_view_prediction_consistency_weight > 0.0)
            and z_clean_seq is not None
        ):
            temporal_model = temporal_model_from(model)
            hr_ref = getattr(temporal_model, "last_hr_seq", None)
            rr_ref = getattr(temporal_model, "last_rr_seq", None)
            aug_video = augment_source_view(
                video,
                brightness_jitter=args.source_view_brightness_jitter,
                noise_std=args.source_view_noise_std,
            )
            pred_aug, _, _, scalar_outputs_aug, z_aug, _ = model.forward_with_multitask_features(
                aug_video,
                frame_offsets=frame_offsets,
                domain_ids=dataset_ids,
                support_context=getattr(args, "support_context_tensor", None),
            )
            temporal_model = temporal_model_from(model)
            if args.source_view_consistency_weight > 0.0:
                view_loss, view_logs = source_view_consistency_loss(
                    z_clean_seq,
                    hr_ref,
                    rr_ref,
                    z_aug,
                    getattr(temporal_model, "last_hr_seq", None),
                    getattr(temporal_model, "last_rr_seq", None),
                    prefix="source_view",
                )
                if view_loss is not None:
                    loss = loss + args.source_view_consistency_weight * view_loss
                    logs.update(view_logs)
            if args.source_view_prediction_consistency_weight > 0.0:
                view_loss, view_logs = source_view_prediction_consistency_loss(
                    pred_ppg,
                    pred_aug,
                    scalar_outputs,
                    scalar_outputs_aug,
                    prefix="source_view",
                )
                loss = loss + args.source_view_prediction_consistency_weight * view_loss
                logs.update(view_logs)
        if (
            (
                args.target_view_consistency_weight > 0.0
                or args.target_view_prediction_consistency_weight > 0.0
                or args.target_view_token_vicreg_weight > 0.0
            )
            and z_clean_seq is not None
        ):
            temporal_model = temporal_model_from(model)
            hr_ref = getattr(temporal_model, "last_hr_seq", None)
            rr_ref = getattr(temporal_model, "last_rr_seq", None)
            aug_video = augment_source_view(
                video,
                brightness_jitter=args.target_view_brightness_jitter,
                noise_std=args.target_view_noise_std,
            )
            pred_aug, _, _, scalar_outputs_aug, z_aug, _ = model.forward_with_multitask_features(
                aug_video,
                frame_offsets=frame_offsets,
                domain_ids=dataset_ids,
                support_context=getattr(args, "support_context_tensor", None),
            )
            temporal_model = temporal_model_from(model)
            if args.target_view_consistency_weight > 0.0:
                view_loss, view_logs = source_view_consistency_loss(
                    z_clean_seq,
                    hr_ref,
                    rr_ref,
                    z_aug,
                    getattr(temporal_model, "last_hr_seq", None),
                    getattr(temporal_model, "last_rr_seq", None),
                    prefix="target_view",
                )
                if view_loss is not None:
                    loss = loss + args.target_view_consistency_weight * view_loss
                    logs.update(view_logs)
            if args.target_view_prediction_consistency_weight > 0.0:
                view_loss, view_logs = source_view_prediction_consistency_loss(
                    pred_ppg,
                    pred_aug,
                    scalar_outputs,
                    scalar_outputs_aug,
                    prefix="target_view",
                )
                loss = loss + args.target_view_prediction_consistency_weight * view_loss
                logs.update(view_logs)
            if args.target_view_token_vicreg_weight > 0.0:
                vicreg_loss, vicreg_logs = token_view_vicreg_loss(
                    z_clean_seq,
                    hr_ref,
                    rr_ref,
                    z_aug,
                    getattr(temporal_model, "last_hr_seq", None),
                    getattr(temporal_model, "last_rr_seq", None),
                    variance_weight=args.target_view_token_vicreg_variance_weight,
                    covariance_weight=args.target_view_token_vicreg_covariance_weight,
                    prefix="target_view",
                )
                if vicreg_loss is not None:
                    loss = loss + args.target_view_token_vicreg_weight * vicreg_loss
                    logs.update(vicreg_logs)
        if args.source_masked_prediction_weight > 0.0:
            masked_video = mask_source_temporal_view(
                video,
                mask_fraction=args.source_masked_temporal_fraction,
                min_span=args.source_masked_min_span,
            )
            pred_masked, recon_masked, _, scalar_outputs_masked = model.forward_with_multitask(
                masked_video,
                frame_offsets=frame_offsets,
                domain_ids=dataset_ids,
                support_context=getattr(args, "support_context_tensor", None),
            )
            masked_loss, masked_logs = multitask_loss(
                pred_masked,
                ppg,
                ppg_mask,
                scalar_outputs_masked,
                scalar_targets,
                scalar_masks,
                recon_masked,
                args,
                task_log_vars=task_log_vars,
            )
            loss = loss + args.source_masked_prediction_weight * masked_loss
            logs["source_masked_prediction_loss"] = float(masked_loss.detach().cpu())
            for key in ("ppg_loss", "hr_loss", "rr_loss", "spo2_loss", "hr_from_ppg_loss", "rr_pseudo_from_ppg_loss"):
                if key in masked_logs:
                    logs[f"source_masked_{key}"] = masked_logs[key]
        if args.source_temporal_reverse_weight > 0.0:
            reversed_video = reverse_temporal_view(video)
            reversed_ppg = torch.flip(ppg, dims=[1])
            pred_reversed, recon_reversed, _, scalar_outputs_reversed = model.forward_with_multitask(
                reversed_video,
                frame_offsets=frame_offsets,
                domain_ids=dataset_ids,
                support_context=getattr(args, "support_context_tensor", None),
            )
            reverse_loss, reverse_logs = multitask_loss(
                pred_reversed,
                reversed_ppg,
                ppg_mask,
                scalar_outputs_reversed,
                scalar_targets,
                scalar_masks,
                recon_reversed,
                args,
                task_log_vars=task_log_vars,
            )
            loss = loss + args.source_temporal_reverse_weight * reverse_loss
            logs["source_temporal_reverse_loss"] = float(reverse_loss.detach().cpu())
            for key in ("ppg_loss", "hr_loss", "rr_loss", "spo2_loss", "hr_from_ppg_loss", "rr_pseudo_from_ppg_loss"):
                if key in reverse_logs:
                    logs[f"source_reverse_{key}"] = reverse_logs[key]
        if args.rate_contrastive_weight > 0.0 and z_clean_seq is not None:
            hr_bpm = denormalize_task(scalar_targets[:, TASKS.index("hr")], "hr")
            hr_mask = scalar_masks[:, TASKS.index("hr")]
            rr_bpm = soft_fft_bpm(
                ppg,
                fs=args.fs,
                low_bpm=args.rate_contrastive_rr_low_bpm,
                high_bpm=args.rate_contrastive_rr_high_bpm,
                nfft=args.rate_contrastive_fft_nfft,
                temperature=args.rate_contrastive_fft_temperature,
            )
            hr_contrast = rate_similarity_contrastive_loss(
                z_clean_seq,
                hr_bpm,
                hr_mask,
                sigma_bpm=args.rate_contrastive_hr_sigma,
                temperature=args.rate_contrastive_temperature,
            )
            rr_contrast = rate_similarity_contrastive_loss(
                z_clean_seq,
                rr_bpm,
                ppg_mask,
                sigma_bpm=args.rate_contrastive_rr_sigma,
                temperature=args.rate_contrastive_temperature,
            )
            contrast_loss = hr_contrast + args.rate_contrastive_rr_weight * rr_contrast
            loss = loss + args.rate_contrastive_weight * contrast_loss
            logs["rate_contrastive_loss"] = float(contrast_loss.detach().cpu())
            logs["hr_contrastive_loss"] = float(hr_contrast.detach().cpu())
            logs["rr_contrastive_loss"] = float(rr_contrast.detach().cpu())
        if args.task_stream_contrastive_weight > 0.0:
            temporal_model = temporal_model_from(model)
            stream_loss, stream_logs = task_stream_rate_contrastive_loss(
                getattr(temporal_model, "last_hr_seq", None),
                getattr(temporal_model, "last_rr_seq", None),
                scalar_targets,
                scalar_masks,
                ppg,
                ppg_mask,
                args,
            )
            loss = loss + args.task_stream_contrastive_weight * stream_loss
            logs.update(stream_logs)
        if args.task_stream_spectral_alignment_weight > 0.0:
            temporal_model = temporal_model_from(model)
            stream_spectral_loss, stream_spectral_logs = task_stream_spectral_alignment_loss(
                getattr(temporal_model, "last_hr_seq", None),
                getattr(temporal_model, "last_rr_seq", None),
                scalar_targets,
                scalar_masks,
                ppg,
                ppg_mask,
                args,
            )
            loss = loss + args.task_stream_spectral_alignment_weight * stream_spectral_loss
            logs.update(stream_spectral_logs)
        if args.task_stream_cross_domain_alignment_weight > 0.0:
            temporal_model = temporal_model_from(model)
            cross_domain_loss, cross_domain_logs = task_stream_cross_domain_alignment_loss(
                getattr(temporal_model, "last_hr_seq", None),
                getattr(temporal_model, "last_rr_seq", None),
                scalar_targets,
                scalar_masks,
                ppg,
                ppg_mask,
                dataset_ids,
                args,
            )
            loss = loss + args.task_stream_cross_domain_alignment_weight * cross_domain_loss
            logs.update(cross_domain_logs)
        if args.source_rate_ordering_weight > 0.0:
            temporal_model = temporal_model_from(model)
            ordering_loss, ordering_logs = source_task_stream_rate_ordering_loss(
                getattr(temporal_model, "last_hr_seq", None),
                getattr(temporal_model, "last_rr_seq", None),
                scalar_targets,
                scalar_masks,
                ppg,
                ppg_mask,
                dataset_ids,
                args,
            )
            loss = loss + args.source_rate_ordering_weight * ordering_loss
            logs.update(ordering_logs)
        if args.source_episode_residual_weight > 0.0:
            episode_loss, episode_logs = source_episodic_residual_adaptation_loss(
                scalar_outputs,
                scalar_targets,
                scalar_masks,
                dataset_ids,
                args,
            )
            loss = loss + args.source_episode_residual_weight * episode_loss
            logs.update(episode_logs)
        if args.band_disentangle_weight > 0.0:
            temporal_model = temporal_model_from(model)
            disentangle_loss = band_stream_disentangle_loss(
                getattr(temporal_model, "last_hr_seq", None),
                getattr(temporal_model, "last_rr_seq", None),
            )
            if disentangle_loss is not None:
                loss = loss + args.band_disentangle_weight * disentangle_loss
                logs["band_disentangle_loss"] = float(disentangle_loss.detach().cpu())
        if args.factor_regularization_weight > 0.0:
            temporal_model = temporal_model_from(model)
            factor_loss_fn = getattr(temporal_model, "factor_regularization_loss", None)
            factor_loss = factor_loss_fn() if callable(factor_loss_fn) else None
            if factor_loss is not None:
                loss = loss + args.factor_regularization_weight * factor_loss
                logs["factor_regularization_loss"] = float(factor_loss.detach().cpu())
                factor_orth = getattr(temporal_model, "factor_orth_loss", None)
                factor_recon = getattr(temporal_model, "factor_recon_loss", None)
                if factor_orth is not None:
                    logs["factor_orth_loss"] = float(factor_orth.detach().cpu())
                if factor_recon is not None:
                    logs["factor_recon_loss"] = float(factor_recon.detach().cpu())
        if args.context_smooth_weight > 0.0:
            context_loss, context_logs = context_clip_smoothness_loss(z_clean_seq, pred_ppg, ppg_mask, args)
            loss = loss + args.context_smooth_weight * context_loss
            logs["context_smooth_loss"] = float(context_loss.detach().cpu())
            logs.update(context_logs)
        if args.domain_adversarial_weight > 0.0 and domain_adversary is not None and z_clean_seq is not None:
            domain_logits = domain_adversary(z_clean_seq, reverse_scale=args.domain_adversarial_lambda)
            domain_loss = F.cross_entropy(domain_logits, dataset_ids)
            domain_acc = (domain_logits.argmax(dim=1) == dataset_ids).float().mean()
            loss = loss + args.domain_adversarial_weight * domain_loss
            logs["domain_adversarial_loss"] = float(domain_loss.detach().cpu())
            logs["domain_adversarial_acc"] = float(domain_acc.detach().cpu())
        if args.task_stream_domain_adversarial_weight > 0.0 and task_stream_domain_adversary is not None:
            temporal_model = temporal_model_from(model)
            stream_logits = task_stream_domain_adversary(
                getattr(temporal_model, "last_hr_seq", None),
                getattr(temporal_model, "last_rr_seq", None),
                reverse_scale=args.task_stream_domain_adversarial_lambda,
            )
            stream_domain_loss = dataset_ids.new_tensor(0.0, dtype=torch.float32)
            used_streams = 0
            for stream_name, logits in stream_logits.items():
                single_loss = F.cross_entropy(logits, dataset_ids)
                single_acc = (logits.argmax(dim=1) == dataset_ids).float().mean()
                stream_domain_loss = stream_domain_loss + single_loss
                used_streams += 1
                logs[f"task_stream_{stream_name}_domain_loss"] = float(single_loss.detach().cpu())
                logs[f"task_stream_{stream_name}_domain_acc"] = float(single_acc.detach().cpu())
            if used_streams > 0:
                stream_domain_loss = stream_domain_loss / used_streams
                loss = loss + args.task_stream_domain_adversarial_weight * stream_domain_loss
                logs["task_stream_domain_adversarial_loss"] = float(stream_domain_loss.detach().cpu())
        if (
            args.task_stream_subject_adversarial_weight > 0.0
            and task_stream_subject_adversary is not None
            and subject_to_idx is not None
        ):
            temporal_model = temporal_model_from(model)
            subject_ids = torch.as_tensor(
                [int(subject_to_idx.get(str(subject), 0)) for subject in subjects],
                device=device,
                dtype=torch.long,
            )
            stream_logits = task_stream_subject_adversary(
                getattr(temporal_model, "last_hr_seq", None),
                getattr(temporal_model, "last_rr_seq", None),
                reverse_scale=args.task_stream_subject_adversarial_lambda,
            )
            stream_subject_loss = dataset_ids.new_tensor(0.0, dtype=torch.float32)
            used_streams = 0
            for stream_name, logits in stream_logits.items():
                single_loss = F.cross_entropy(logits, subject_ids)
                single_acc = (logits.argmax(dim=1) == subject_ids).float().mean()
                stream_subject_loss = stream_subject_loss + single_loss
                used_streams += 1
                logs[f"task_stream_{stream_name}_subject_loss"] = float(single_loss.detach().cpu())
                logs[f"task_stream_{stream_name}_subject_acc"] = float(single_acc.detach().cpu())
            if used_streams > 0:
                stream_subject_loss = stream_subject_loss / used_streams
                loss = loss + args.task_stream_subject_adversarial_weight * stream_subject_loss
                logs["task_stream_subject_adversarial_loss"] = float(stream_subject_loss.detach().cpu())
        if args.l2_sp_weight > 0.0 and l2_sp_anchors is not None:
            anchor_loss = l2_sp_loss(model, l2_sp_anchors)
            if anchor_loss is not None:
                loss = loss + args.l2_sp_weight * anchor_loss
                logs["l2_sp_loss"] = float(anchor_loss.detach().cpu())
        if args.strict_train_sanity:
            strict_train_sanity_check(loss, logs, temporal_model_from(model), args)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        batch_size = video.shape[0]
        count += batch_size
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + value * batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def denormalize_task(values: torch.Tensor, task: str) -> torch.Tensor:
    spec = TASK_STATS[task]
    return values * spec["scale"] + spec["center"]


@torch.no_grad()
def evaluate(model, loader, args, device, save_outputs=False, task_log_vars: torch.Tensor | None = None):
    model.eval()
    sums = {}
    scalar_abs_errors = {task: 0.0 for task in TASKS}
    scalar_sq_errors = {task: 0.0 for task in TASKS}
    scalar_ape_errors = {task: 0.0 for task in TASKS}
    scalar_counts = {task: 0.0 for task in TASKS}
    scalar_preds = {task: [] for task in TASKS}
    scalar_targets_all = {task: [] for task in TASKS}
    scalar_bin_errors = {task: {} for task in ("hr", "rr", "sbp", "dbp", "map")}
    outputs = {}
    count = 0
    for video, ppg, ppg_mask, scalar_targets, scalar_masks, dataset_ids, subjects, sort_indices in loader:
        video = prepare_video(video, args, device)
        ppg = ppg.to(device, dtype=torch.float32)
        ppg_mask = ppg_mask.to(device, dtype=torch.float32)
        scalar_targets = scalar_targets.to(device, dtype=torch.float32)
        scalar_masks = scalar_masks.to(device, dtype=torch.float32)
        dataset_ids = dataset_ids.to(device, dtype=torch.long)
        frame_offsets = sort_indices.to(device) * args.frames if args.absolute_time_decoder else None
        pred_ppg, recon_loss, _, scalar_outputs = model.forward_with_multitask(
            video,
            frame_offsets=frame_offsets,
            domain_ids=dataset_ids,
            support_context=getattr(args, "support_context_tensor", None),
        )
        _, logs = multitask_loss(
            pred_ppg,
            ppg,
            ppg_mask,
            scalar_outputs,
            scalar_targets,
            scalar_masks,
            recon_loss,
            args,
            task_log_vars=task_log_vars,
        )
        add_adaptive_prior_logs(logs, model)
        pred_scalars = scalar_predictions_to_tensor(scalar_outputs, device)
        batch_size = video.shape[0]
        count += batch_size
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + value * batch_size
        for task_idx, task in enumerate(TASKS):
            pred_actual = denormalize_task(pred_scalars[:, task_idx], task)
            target_actual = denormalize_task(scalar_targets[:, task_idx], task)
            mask = scalar_masks[:, task_idx]
            abs_error = torch.abs(pred_actual - target_actual)
            scalar_abs_errors[task] += float((abs_error * mask).sum().cpu())
            scalar_sq_errors[task] += float((((pred_actual - target_actual) ** 2) * mask).sum().cpu())
            scalar_ape_errors[task] += float(((abs_error / torch.clamp(torch.abs(target_actual), min=1e-6)) * mask).sum().cpu())
            scalar_counts[task] += float(mask.sum().cpu())
            valid = mask > 0
            if torch.any(valid):
                scalar_preds[task].extend(pred_actual[valid].detach().cpu().tolist())
                scalar_targets_all[task].extend(target_actual[valid].detach().cpu().tolist())
            if task in scalar_bin_errors:
                if task == "hr":
                    width = args.eval_hr_bin_width
                elif task == "rr":
                    width = args.eval_rr_bin_width
                else:
                    width = args.eval_bp_bin_width
                width = max(float(width), 1e-6)
                for error_value, target_value, mask_value in zip(
                    abs_error.detach().cpu().tolist(),
                    target_actual.detach().cpu().tolist(),
                    mask.detach().cpu().tolist(),
                ):
                    if mask_value <= 0:
                        continue
                    bin_key = int(math.floor(float(target_value) / width))
                    error_sum, error_count = scalar_bin_errors[task].get(bin_key, (0.0, 0.0))
                    scalar_bin_errors[task][bin_key] = (
                        error_sum + float(error_value),
                        error_count + 1.0,
                    )
        if save_outputs:
            pred_ppg_cpu = pred_ppg.detach().cpu().numpy()
            ppg_cpu = ppg.detach().cpu().numpy()
            pred_scalars_cpu = pred_scalars.detach().cpu()
            target_scalars_cpu = scalar_targets.detach().cpu()
            masks_cpu = scalar_masks.detach().cpu().numpy()
            for item_idx, subject in enumerate(subjects):
                outputs.setdefault(subject, {})[int(sort_indices[item_idx])] = {
                    "prediction": pred_ppg_cpu[item_idx],
                    "label": ppg_cpu[item_idx],
                    "scalars_pred": {
                        task: float(denormalize_task(pred_scalars_cpu[item_idx, task_idx], task))
                        for task_idx, task in enumerate(TASKS)
                    },
                    "scalars_label": {
                        task: float(denormalize_task(target_scalars_cpu[item_idx, task_idx], task))
                        for task_idx, task in enumerate(TASKS)
                    },
                    "scalar_masks": {task: float(masks_cpu[item_idx, task_idx]) for task_idx, task in enumerate(TASKS)},
                }
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    for task in TASKS:
        if scalar_counts[task] > 0:
            metrics[f"{task}_mae"] = scalar_abs_errors[task] / scalar_counts[task]
            metrics[f"{task}_rmse"] = math.sqrt(scalar_sq_errors[task] / scalar_counts[task])
            metrics[f"{task}_mape"] = 100.0 * scalar_ape_errors[task] / scalar_counts[task]
            pred_values = np.asarray(scalar_preds[task], dtype=np.float64)
            target_values = np.asarray(scalar_targets_all[task], dtype=np.float64)
            if pred_values.size > 1 and np.std(pred_values) > 0 and np.std(target_values) > 0:
                metrics[f"{task}_pearson"] = float(np.corrcoef(pred_values, target_values)[0, 1])
            else:
                metrics[f"{task}_pearson"] = float("nan")
        else:
            metrics[f"{task}_mae"] = float("nan")
            metrics[f"{task}_rmse"] = float("nan")
            metrics[f"{task}_mape"] = float("nan")
            metrics[f"{task}_pearson"] = float("nan")
    for task, bins in scalar_bin_errors.items():
        bin_maes = [error_sum / error_count for error_sum, error_count in bins.values() if error_count > 0]
        metrics[f"{task}_balanced_mae"] = float(np.mean(bin_maes)) if bin_maes else float("nan")
    return metrics, outputs


def prepare_video(video, args, device):
    video = video.to(device, dtype=torch.float32)
    if args.raw_div255:
        video = video / 255.0
    return video


def augment_source_view(video: torch.Tensor, brightness_jitter: float, noise_std: float) -> torch.Tensor:
    augmented = video
    if brightness_jitter > 0.0:
        shape = [video.shape[0]] + [1] * (video.dim() - 1)
        scale = 1.0 + (torch.rand(shape, device=video.device, dtype=video.dtype) * 2.0 - 1.0) * float(brightness_jitter)
        augmented = augmented * scale
    if noise_std > 0.0:
        reduce_dims = tuple(range(1, video.dim()))
        sample_std = video.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1.0)
        augmented = augmented + torch.randn_like(video) * sample_std * float(noise_std)
    return augmented


def mask_source_temporal_view(video: torch.Tensor, mask_fraction: float, min_span: int) -> torch.Tensor:
    if mask_fraction <= 0.0 or video.dim() < 5:
        return video
    temporal_dim = 2
    frames = int(video.shape[temporal_dim])
    if frames <= 1:
        return video
    span = max(int(min_span), int(round(frames * float(mask_fraction))))
    span = min(max(span, 1), frames)
    masked = video.clone()
    fill = video.mean(dim=temporal_dim, keepdim=True)
    for batch_idx in range(video.shape[0]):
        start = int(torch.randint(0, frames - span + 1, (1,), device=video.device).item())
        index = [slice(None)] * video.dim()
        index[0] = batch_idx
        index[temporal_dim] = slice(start, start + span)
        fill_index = [slice(None)] * video.dim()
        fill_index[0] = batch_idx
        masked[tuple(index)] = fill[tuple(fill_index)]
    return masked


def reverse_temporal_view(video: torch.Tensor) -> torch.Tensor:
    if video.dim() < 5:
        return video
    return torch.flip(video, dims=[2])


def pooled_stream_features(seq: torch.Tensor | None) -> torch.Tensor | None:
    if seq is None:
        return None
    pooled = torch.cat([seq.mean(dim=1), seq.std(dim=1, unbiased=False)], dim=-1)
    return F.normalize(pooled, dim=-1)


def source_view_consistency_loss(
    z_ref: torch.Tensor | None,
    hr_ref: torch.Tensor | None,
    rr_ref: torch.Tensor | None,
    z_aug: torch.Tensor | None,
    hr_aug: torch.Tensor | None,
    rr_aug: torch.Tensor | None,
    prefix: str = "source_view",
) -> tuple[torch.Tensor | None, dict[str, float]]:
    losses = []
    logs = {}
    for name, ref, aug in (("z", z_ref, z_aug), ("hr", hr_ref, hr_aug), ("rr", rr_ref, rr_aug)):
        ref_pooled = pooled_stream_features(ref)
        aug_pooled = pooled_stream_features(aug)
        if ref_pooled is None or aug_pooled is None:
            continue
        item = F.smooth_l1_loss(aug_pooled, ref_pooled.detach())
        losses.append(item)
        logs[f"{prefix}_{name}_consistency_loss"] = float(item.detach().cpu())
    if not losses:
        return None, logs
    total = torch.stack(losses).mean()
    logs[f"{prefix}_consistency_loss"] = float(total.detach().cpu())
    return total, logs


def source_view_prediction_consistency_loss(
    pred_ref: torch.Tensor,
    pred_aug: torch.Tensor,
    scalars_ref: dict[str, torch.Tensor],
    scalars_aug: dict[str, torch.Tensor],
    prefix: str = "source_view",
) -> tuple[torch.Tensor, dict[str, float]]:
    waveform_loss = F.smooth_l1_loss(normalize_signal_batch(pred_aug), normalize_signal_batch(pred_ref.detach()))
    scalar_losses = []
    for task in ("hr", "rr", "spo2"):
        if task in scalars_ref and task in scalars_aug:
            scalar_losses.append(F.smooth_l1_loss(scalars_aug[task], scalars_ref[task].detach()))
    if scalar_losses:
        scalar_loss = torch.stack(scalar_losses).mean()
    else:
        scalar_loss = waveform_loss.new_tensor(0.0)
    total = waveform_loss + scalar_loss
    return total, {
        f"{prefix}_prediction_consistency_loss": float(total.detach().cpu()),
        f"{prefix}_waveform_consistency_loss": float(waveform_loss.detach().cpu()),
        f"{prefix}_scalar_consistency_loss": float(scalar_loss.detach().cpu()),
    }


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Expected square matrix, got {tuple(x.shape)}")
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def token_view_vicreg_item(
    ref: torch.Tensor,
    aug: torch.Tensor,
    variance_weight: float,
    covariance_weight: float,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_tokens = ref.reshape(-1, ref.shape[-1])
    aug_tokens = aug.reshape(-1, aug.shape[-1])
    invariance = F.smooth_l1_loss(aug_tokens, ref_tokens.detach())

    def var_cov_loss(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens - tokens.mean(dim=0, keepdim=True)
        std = torch.sqrt(tokens.var(dim=0, unbiased=False) + eps)
        variance = F.relu(1.0 - std).mean()
        if tokens.shape[0] <= 1:
            covariance = tokens.new_tensor(0.0)
        else:
            cov = (tokens.T @ tokens) / max(tokens.shape[0] - 1, 1)
            covariance = off_diagonal(cov).square().sum() / max(tokens.shape[-1], 1)
        return variance, covariance

    ref_var, ref_cov = var_cov_loss(ref_tokens)
    aug_var, aug_cov = var_cov_loss(aug_tokens)
    variance = 0.5 * (ref_var + aug_var)
    covariance = 0.5 * (ref_cov + aug_cov)
    total = invariance + variance_weight * variance + covariance_weight * covariance
    return total, invariance, variance, covariance


def token_view_vicreg_loss(
    z_ref: torch.Tensor | None,
    hr_ref: torch.Tensor | None,
    rr_ref: torch.Tensor | None,
    z_aug: torch.Tensor | None,
    hr_aug: torch.Tensor | None,
    rr_aug: torch.Tensor | None,
    variance_weight: float,
    covariance_weight: float,
    prefix: str = "target_view",
) -> tuple[torch.Tensor | None, dict[str, float]]:
    losses = []
    logs = {}
    for name, ref, aug in (("z", z_ref, z_aug), ("hr", hr_ref, hr_aug), ("rr", rr_ref, rr_aug)):
        if ref is None or aug is None:
            continue
        item, inv, var, cov = token_view_vicreg_item(
            ref,
            aug,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
        )
        losses.append(item)
        logs[f"{prefix}_{name}_token_vicreg_loss"] = float(item.detach().cpu())
        logs[f"{prefix}_{name}_token_invariance_loss"] = float(inv.detach().cpu())
        logs[f"{prefix}_{name}_token_variance_loss"] = float(var.detach().cpu())
        logs[f"{prefix}_{name}_token_covariance_loss"] = float(cov.detach().cpu())
    if not losses:
        return None, logs
    total = torch.stack(losses).mean()
    logs[f"{prefix}_token_vicreg_loss"] = float(total.detach().cpu())
    return total, logs


def l2_sp_loss(model: nn.Module, anchors: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
    if not anchors:
        return None
    total = None
    used = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in anchors:
            continue
        item = (param - anchors[name].to(device=param.device, dtype=param.dtype)).square().mean()
        total = item if total is None else total + item
        used += 1
    if total is None or used == 0:
        return None
    return total / used


def unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def strip_module_prefix(state):
    return {key.replace("module.", "", 1) if key.startswith("module.") else key: value for key, value in state.items()}


def remap_legacy_encoder_keys(state_dict):
    if any(key.startswith("stem.") for key in state_dict):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("base_encoder.0."):
            new_key = key.replace("base_encoder.0.", "stem.0.", 1)
        elif key.startswith("base_encoder.1."):
            new_key = key.replace("base_encoder.1.", "stem.1.", 1)
        elif key.startswith("base_encoder.2."):
            new_key = key.replace("base_encoder.2.", "stem.2.", 1)
        elif key.startswith("base_encoder.3."):
            new_key = key.replace("base_encoder.3.", "base_encoder.0.", 1)
        elif key.startswith("base_encoder.4."):
            new_key = key.replace("base_encoder.4.", "base_encoder.1.", 1)
        elif key.startswith("base_encoder.5."):
            new_key = key.replace("base_encoder.5.", "base_encoder.2.", 1)
        remapped[new_key] = value
    return remapped


def remap_scalar_heads_for_residual_temporal(state_dict):
    remapped = {}
    pattern = re.compile(r"^scalar_heads\.([^.]+)\.(.+)$")
    for key, value in state_dict.items():
        match = pattern.match(key)
        if match and not match.group(2).startswith(("pooled_head.", "temporal_residual.", "residual_gate")):
            remapped[f"scalar_heads.{match.group(1)}.pooled_head.{match.group(2)}"] = value
        else:
            remapped[key] = value
    return remapped


def make_model(args, device):
    model = PhaseNet(
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        tcn_layers=args.tcn_layers,
        temporal_module=args.temporal_module,
        phase_fs=args.fs,
        physio_hr_low_bpm=args.physio_hr_low_bpm,
        physio_hr_high_bpm=args.physio_hr_high_bpm,
        physio_rr_low_bpm=args.physio_rr_low_bpm,
        physio_rr_high_bpm=args.physio_rr_high_bpm,
        physio_num_freq_bins=args.physio_num_freq_bins,
        physio_long_context=args.physio_long_context,
        hr_num_bins=0,
        rgb_pos_fusion=False,
        encoder_input_normalization=args.input_normalization,
        waveform_head_type=args.waveform_head_type,
        encoder_motion_fusion=args.encoder_motion_fusion,
        encoder_temporal_pyramid=args.encoder_temporal_pyramid,
        encoder_color_motion_branch=args.encoder_color_motion_branch,
        mixstyle_stem=args.mixstyle_stem,
        mixstyle_encoder=args.mixstyle_encoder,
        mixstyle_prob=args.mixstyle_prob,
        mixstyle_alpha=args.mixstyle_alpha,
        dataset_conditioning=args.dataset_conditioning,
        task_dataset_conditioning=args.task_dataset_conditioning,
        dataset_num_domains=args.dataset_num_domains,
        long_range_mixer=args.long_range_mixer,
        representation_refiner=args.representation_refiner,
        representation_bottleneck_dim=args.representation_bottleneck_dim,
        vital_representation_adapter=args.vital_representation_adapter,
        cross_task_representation_adapter=args.cross_task_representation_adapter,
        rr_only_slow_adapter=args.rr_only_slow_adapter,
        prototype_scalar_readout=args.prototype_scalar_readout,
        prototype_scalar_tokens=args.prototype_scalar_tokens,
        pre_temporal_physio_adapter=args.pre_temporal_physio_adapter,
        pre_temporal_physio_tokens=args.pre_temporal_physio_tokens,
        raw_motion_temporal_adapter=args.raw_motion_temporal_adapter,
        post_temporal_lowrank_adapter=args.post_temporal_lowrank_adapter,
        post_temporal_lowrank_rank=args.post_temporal_lowrank_rank,
        hr_anchored_rr_adapter=args.hr_anchored_rr_adapter,
        hr_anchored_rr_rank=args.hr_anchored_rr_rank,
        episodic_physio_whitening_adapter=args.episodic_physio_whitening_adapter,
        episode_context_physio_mixer=args.episode_context_physio_mixer,
        episode_context_physio_tokens=args.episode_context_physio_tokens,
        temporal_position_physio_adapter=args.temporal_position_physio_adapter,
        rate_query_physio_adapter=args.rate_query_physio_adapter,
        rate_query_physio_bins=args.rate_query_physio_bins,
        task_conditioned_physio_norm=args.task_conditioned_physio_norm,
        task_conditioned_physio_tokens=args.task_conditioned_physio_tokens,
        support_conditioning=args.support_conditioning,
        support_context_dim=args.support_context_dim,
        scalar_tasks=TASKS,
        scalar_head_hidden_dim=args.scalar_head_hidden_dim,
        scalar_head_type=args.scalar_head_type,
        rr_lowfreq_branch=args.rr_lowfreq_branch,
        rr_lowfreq_mode=args.rr_lowfreq_mode,
        video_rate_branch=args.video_rate_branch,
        video_rate_mode=args.video_rate_mode,
        color_scalar_branch=args.color_scalar_branch,
        color_scalar_mode=args.color_scalar_mode,
        hr_band_residual_branch=args.hr_band_residual_branch,
        rate_bin_aux=args.rate_bin_aux_weight > 0.0,
        rate_bin_num_bins=args.rate_bin_num_bins,
        rate_bin_scalar_mode=args.rate_bin_scalar_mode,
        scalar_hr_low_bpm=args.scalar_hr_low_bpm,
        scalar_hr_high_bpm=args.scalar_hr_high_bpm,
        scalar_rr_low_bpm=args.scalar_rr_low_bpm,
        scalar_rr_high_bpm=args.scalar_rr_high_bpm,
    ).to(device)
    if args.pretrained_checkpoint:
        state = strip_module_prefix(remap_legacy_encoder_keys(unwrap_state_dict(torch.load(args.pretrained_checkpoint, map_location=device))))
        if args.scalar_head_type in ("residual_temporal", "temporal_residual"):
            state = remap_scalar_heads_for_residual_temporal(state)
        incompatible = model.load_state_dict(state, strict=False)
        print(
            f"loaded_pretrained={args.pretrained_checkpoint} "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    return model


def _set_requires_grad(module, requires_grad: bool) -> int:
    if module is None:
        return 0
    changed = 0
    for param in module.parameters():
        if param.requires_grad != requires_grad:
            param.requires_grad = requires_grad
            changed += param.numel()
    return changed


def apply_freeze_policy(model, args) -> tuple[int, int]:
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        param.requires_grad = False

    if args.freeze_non_scalar:
        for name, param in model.named_parameters():
            if name.startswith((
                "scalar_heads.",
                "rate_residual_heads.",
                "rate_residual_gates.",
                "video_rate_heads.",
                "video_rate_gates.",
                "color_scalar_heads.",
                "color_scalar_gates.",
                "hr_band_residual_heads.",
                "hr_band_residual_gates.",
                "prototype_scalar_heads.",
                "prototype_scalar_gates.",
                "rate_bin_heads.",
                "rate_bin_scalar_gates.",
            )):
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    if args.unfreeze_backbone_stem:
        _set_requires_grad(getattr(model, "stem", None), True)
        if getattr(model, "motion_stem", None) is not None:
            _set_requires_grad(getattr(model, "motion_stem", None), True)
    if args.unfreeze_temporal_pyramid:
        _set_requires_grad(getattr(model, "temporal_pyramid", None), True)
        gate = getattr(model, "temporal_pyramid_gate", None)
        if gate is not None:
            gate.requires_grad = True
    if args.unfreeze_backbone_last_blocks > 0:
        blocks = list(getattr(model, "base_encoder", []))
        num_blocks = min(int(args.unfreeze_backbone_last_blocks), len(blocks))
        for block in blocks[-num_blocks:]:
            _set_requires_grad(block, True)
    if args.unfreeze_backbone_encoder_head:
        _set_requires_grad(getattr(model, "encoder_head", None), True)
    if args.unfreeze_temporal_model:
        _set_requires_grad(getattr(model, "temporal_model", None), True)
    unfreeze_substrings = [
        item.strip()
        for item in str(getattr(args, "unfreeze_parameter_substrings", "") or "").split(",")
        if item.strip()
    ]
    if unfreeze_substrings:
        for name, param in model.named_parameters():
            if any(token in name for token in unfreeze_substrings):
                param.requires_grad = True
    if args.unfreeze_pre_temporal_physio_adapter or args.pre_temporal_physio_adapter:
        _set_requires_grad(getattr(model, "pre_temporal_physio_adapter", None), True)
    if args.unfreeze_raw_motion_temporal_adapter or args.raw_motion_temporal_adapter:
        _set_requires_grad(getattr(model, "raw_motion_temporal_adapter", None), True)
    if args.unfreeze_post_temporal_lowrank_adapter or args.post_temporal_lowrank_adapter:
        _set_requires_grad(getattr(model, "post_temporal_lowrank_adapter", None), True)
    if args.unfreeze_hr_anchored_rr_adapter or args.hr_anchored_rr_adapter:
        _set_requires_grad(getattr(model, "hr_anchored_rr_adapter", None), True)
    if args.unfreeze_episodic_physio_whitening_adapter or args.episodic_physio_whitening_adapter:
        _set_requires_grad(getattr(model, "episodic_physio_whitening_adapter", None), True)
    if args.unfreeze_episode_context_physio_mixer or args.episode_context_physio_mixer:
        _set_requires_grad(getattr(model, "episode_context_physio_mixer", None), True)
    if args.unfreeze_temporal_position_physio_adapter or args.temporal_position_physio_adapter:
        _set_requires_grad(getattr(model, "temporal_position_physio_adapter", None), True)
    if args.unfreeze_rate_query_physio_adapter or args.rate_query_physio_adapter:
        _set_requires_grad(getattr(model, "rate_query_physio_adapter", None), True)
    if args.unfreeze_task_conditioned_physio_norm or args.task_conditioned_physio_norm:
        _set_requires_grad(getattr(model, "task_conditioned_physio_norm", None), True)
    if args.unfreeze_support_conditioning or args.support_conditioning:
        _set_requires_grad(getattr(model, "support_conditioning", None), True)
    if args.unfreeze_waveform_head:
        _set_requires_grad(getattr(model, "regressor_head", None), True)
        _set_requires_grad(getattr(model, "frequency_waveform_decoder", None), True)
    if args.unfreeze_temporal_refiner:
        _set_requires_grad(getattr(model, "temporal_refiner", None), True)
    if args.unfreeze_representation_refiner:
        _set_requires_grad(getattr(model, "representation_refiner", None), True)
    if args.unfreeze_vital_representation_adapter or args.vital_representation_adapter:
        _set_requires_grad(getattr(model, "vital_representation_adapter", None), True)
    if args.unfreeze_cross_task_representation_adapter or args.cross_task_representation_adapter:
        _set_requires_grad(getattr(model, "cross_task_representation_adapter", None), True)
    if args.unfreeze_rr_only_slow_adapter or args.rr_only_slow_adapter:
        _set_requires_grad(getattr(model, "rr_only_slow_adapter", None), True)
    if args.unfreeze_prototype_scalar_readout or args.prototype_scalar_readout:
        _set_requires_grad(getattr(model, "prototype_scalar_heads", None), True)
        for param in getattr(model, "prototype_scalar_gates", {}).values():
            param.requires_grad = True
    if args.unfreeze_rr_lowfreq_branch or args.rr_lowfreq_branch:
        _set_requires_grad(getattr(model, "rr_lowfreq_head", None), True)
    if args.video_rate_branch:
        _set_requires_grad(getattr(model, "video_rate_heads", None), True)
        for param in getattr(model, "video_rate_gates", {}).values():
            param.requires_grad = True
    if args.color_scalar_branch:
        _set_requires_grad(getattr(model, "color_scalar_heads", None), True)
        for param in getattr(model, "color_scalar_gates", {}).values():
            param.requires_grad = True
    if args.hr_band_residual_branch:
        _set_requires_grad(getattr(model, "hr_band_residual_heads", None), True)
        for param in getattr(model, "hr_band_residual_gates", {}).values():
            param.requires_grad = True

    allowed_scalar_tasks = {
        item.strip()
        for item in str(getattr(args, "train_scalar_tasks", "") or "").split(",")
        if item.strip()
    }
    if allowed_scalar_tasks:
        task_module_prefixes = (
            "scalar_heads",
            "rate_residual_heads",
            "video_rate_heads",
            "color_scalar_heads",
            "hr_band_residual_heads",
            "prototype_scalar_heads",
            "rate_bin_heads",
        )
        task_gate_prefixes = (
            "rate_residual_gates",
            "video_rate_gates",
            "color_scalar_gates",
            "hr_band_residual_gates",
            "prototype_scalar_gates",
            "rate_bin_scalar_gates",
        )
        for name, param in model.named_parameters():
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] in task_module_prefixes and parts[1] not in allowed_scalar_tasks:
                param.requires_grad = False
            elif len(parts) >= 2 and parts[0] in task_gate_prefixes and parts[1] not in allowed_scalar_tasks:
                param.requires_grad = False
            elif name.startswith("rr_lowfreq_head.") and "rr" not in allowed_scalar_tasks:
                param.requires_grad = False

    for param in model.parameters():
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(
        f"{key}={value:.5f}" if math.isfinite(value) else f"{key}=nan"
        for key, value in sorted(metrics.items())
    )


def write_metric_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selection_value(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "loss":
        return metrics.get("loss", float("inf"))
    if metric_name == "composite_mae":
        values = [
            metrics.get("pr_mae", float("nan")),
            metrics.get("hr_mae", float("nan")),
            metrics.get("rr_mae", float("nan")),
            metrics.get("spo2_mae", float("nan")),
            metrics.get("sbp_mae", float("nan")),
            metrics.get("dbp_mae", float("nan")),
            metrics.get("map_mae", float("nan")),
        ]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("inf")
    if metric_name in ("hrpr_mae", "hr_pr_mae"):
        values = [
            metrics.get("pr_mae", float("nan")),
            metrics.get("hr_mae", float("nan")),
        ]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("inf")
    if metric_name in ("hrrr_mae", "hr_rr_mae", "rrhr_mae", "rr_hr_mae"):
        values = [
            metrics.get("hr_mae", float("nan")),
            metrics.get("rr_mae", float("nan")),
        ]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("inf")
    if metric_name in ("balanced_hrrr_mae", "balanced_hr_rr_mae", "hr_rr_balanced_mae"):
        values = [
            metrics.get("hr_balanced_mae", float("nan")),
            metrics.get("rr_balanced_mae", float("nan")),
        ]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("inf")
    value = metrics.get(metric_name, float("inf"))
    return value if math.isfinite(value) else float("inf")


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        for name, value in self.shadow.items():
            state[name] = value.detach().cpu().clone()
        return state

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.copy_(backup[name].to(device=param.device, dtype=param.dtype))


def evaluate_with_optional_ema(
    model,
    loader,
    args,
    device,
    ema: ModelEMA | None = None,
    save_outputs=False,
    task_log_vars: torch.Tensor | None = None,
):
    if ema is None:
        return evaluate(model, loader, args, device, save_outputs=save_outputs, task_log_vars=task_log_vars)
    backup = ema.apply_to(model)
    try:
        return evaluate(model, loader, args, device, save_outputs=save_outputs, task_log_vars=task_log_vars)
    finally:
        ema.restore(model, backup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--output-dir", default="runs/multitask_phasenet")
    parser.add_argument("--model-name", default="multitask_phasenet")
    parser.add_argument("--pretrained-checkpoint", default="")
    parser.add_argument("--eval-only-checkpoint", default="")
    parser.add_argument(
        "--test-corruption",
        choices=("none", "brightness", "noise", "blur", "jpeg", "frame_drop", "roi_shift", "spatial_downsample",
                 "static_frame", "temporal_shuffle", "temporal_reverse", "channel_permute", "grayscale", "spatial_shuffle"),
        default="none",
    )
    parser.add_argument("--corruption-severity", type=int, choices=range(0, 6), default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--context-clips", type=int, default=1)
    parser.add_argument("--context-stride", type=int, default=1)
    parser.add_argument("--eval-context-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-div255", action="store_true")
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--input-normalization", default="none")
    parser.add_argument("--waveform-head-type", default="multiscale")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tcn-layers", type=int, default=4)
    parser.add_argument("--temporal-module", default="gated_tcn", choices=["gated_tcn", "tcn", "default", "phase_aware", "rppg_phase", "phase_temporal", "physio_mixer", "physio", "physio_representation", "dual_rate_physio", "dual_physio", "dual_rate", "task_token_physio", "task_token", "task_token_mixer", "physio_oscillator", "oscillator", "phase_oscillator", "resp_envelope_physio", "resp_envelope", "resp_env", "spectral_token_physio", "spectral_token", "freq_token", "long_context_spectral_memory", "context_memory", "spectral_memory", "dual_band_cross_attention", "dual_band_cross", "band_cross", "complex_dual_band_cross_attention", "complex_dual_band", "complex_band_cross", "complex_physio_prototype_memory", "prototype_memory", "proto_memory", "decoupled_physio_prototype_memory", "decoupled_proto_memory", "dppm", "orthogonalized_physio_prototype_memory", "orthogonalized_proto_memory", "oppm", "cross_scale_cardioresp_decomposition", "cross_scale_decomp", "cscd", "subharmonic_cardioresp_memory", "subharmonic_memory", "scrm", "lowfreq_motion_subharmonic", "lowfreq_motion_state", "lfms", "subharmonic_rate_anchor_memory", "rate_anchor_subharmonic", "sram", "subharmonic_cross_scale_memory", "subharmonic_cross_scale", "scsm", "subharmonic_cross_scale_readout", "subharmonic_rr_readout", "scsr", "subharmonic_envelope_residual", "subharmonic_envelope", "serm", "subharmonic_envelope_rr_readout", "subharmonic_envelope_rr", "serr", "subharmonic_instance_stable_residual", "subharmonic_instance_stable", "sisr", "subharmonic_stream_norm_residual", "subharmonic_stream_norm", "ssnr", "subharmonic_task_basis_residual", "subharmonic_task_basis", "stbr", "subharmonic_channel_calibration", "subharmonic_channel_calib", "sccm", "subharmonic_state_residual", "subharmonic_state", "ssrm", "subharmonic_temporal_filter_residual", "subharmonic_temporal_filter", "stfr", "subharmonic_dual_phase_residual", "subharmonic_dual_phase", "sdpr", "subharmonic_parallel_state_fusion", "subharmonic_state_fusion", "spsf", "subharmonic_rr_phase_residual", "subharmonic_rr_phase", "srpr", "subharmonic_slow_rr_readout", "subharmonic_slow_readout", "ssro", "tri_prior_cardioresp_memory", "tri_prior_memory", "tpcm", "adaptive_prior_cardioresp_memory", "adaptive_prior_memory", "apcm", "harmonic_frequency_operator", "frequency_operator", "freq_operator", "hfom", "factorized_physio_subspace", "physio_subspace", "fpsm", "style_factorized_physio_subspace", "style_physio_subspace", "sfpsm", "state_space_physio_memory", "physio_state_space", "sspm", "oscillator_subharmonic_hr_residual", "osc_subharmonic_hr", "oshr", "subharmonic_slow_rr", "subharmonic_proto_slow_rr", "ssrr", "lowrank_cardioresp_residual", "lowrank_residual_memory", "lrrm", "prototype_preserving_slow_rr", "proto_slow_rr", "ppsr", "cardioresp_envelope_memory", "cardio_resp_envelope", "crem", "cardioresp_state_memory", "cardio_resp_state", "crsm", "continuous_cardioresp_state", "continuous_state", "ccrsm", "adaptive_cardioresp_state", "adaptive_state", "acrsm", "latent_cardioresp_dynamics", "latent_dynamics", "lcdm", "hierarchical_cardioresp_dynamics", "hier_cardioresp_dynamics", "hcdm"])
    parser.add_argument("--physio-hr-low-bpm", type=float, default=45.0)
    parser.add_argument("--physio-hr-high-bpm", type=float, default=150.0)
    parser.add_argument("--physio-rr-low-bpm", type=float, default=6.0)
    parser.add_argument("--physio-rr-high-bpm", type=float, default=30.0)
    parser.add_argument("--physio-num-freq-bins", type=int, default=96)
    parser.add_argument("--physio-long-context", action="store_true")
    parser.add_argument("--scalar-head-hidden-dim", type=int, default=128)
    parser.add_argument("--scalar-head-type", default="pooled", choices=["pooled", "summary", "mlp", "temporal", "attention", "temporal_attention", "residual_temporal", "temporal_residual", "hybrid_spectral", "spectral_rate", "bandlimited_rate", "pooled_plus_spectral_rate", "pooled_spectral_rate", "residual_spectral_rate"])
    parser.add_argument("--rr-lowfreq-branch", action="store_true")
    parser.add_argument("--rr-lowfreq-mode", default="replace", choices=["replace", "only", "residual", "add"])
    parser.add_argument("--video-rate-branch", action="store_true")
    parser.add_argument("--video-rate-mode", default="replace", choices=["replace", "only", "blend", "gated", "residual", "add"])
    parser.add_argument("--color-scalar-branch", action="store_true")
    parser.add_argument("--color-scalar-mode", default="residual", choices=["replace", "only", "blend", "gated", "residual", "add"])
    parser.add_argument("--hr-band-residual-branch", action="store_true")
    parser.add_argument("--encoder-motion-fusion", action="store_true")
    parser.add_argument("--encoder-temporal-pyramid", action="store_true")
    parser.add_argument("--encoder-color-motion-branch", action="store_true")
    parser.add_argument("--mixstyle-stem", action="store_true")
    parser.add_argument("--mixstyle-encoder", action="store_true")
    parser.add_argument("--mixstyle-prob", type=float, default=0.5)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument("--dataset-conditioning", action="store_true")
    parser.add_argument("--task-dataset-conditioning", action="store_true")
    parser.add_argument("--dataset-num-domains", type=int, default=0)
    parser.add_argument("--representation-refiner", action="store_true")
    parser.add_argument("--long-range-mixer", action="store_true")
    parser.add_argument("--representation-bottleneck-dim", type=int, default=32)
    parser.add_argument("--vital-representation-adapter", action="store_true")
    parser.add_argument("--cross-task-representation-adapter", action="store_true")
    parser.add_argument("--rr-only-slow-adapter", action="store_true")
    parser.add_argument("--prototype-scalar-readout", action="store_true")
    parser.add_argument("--prototype-scalar-tokens", type=int, default=6)
    parser.add_argument("--pre-temporal-physio-adapter", action="store_true")
    parser.add_argument("--pre-temporal-physio-tokens", type=int, default=6)
    parser.add_argument("--raw-motion-temporal-adapter", action="store_true")
    parser.add_argument("--post-temporal-lowrank-adapter", action="store_true")
    parser.add_argument("--post-temporal-lowrank-rank", type=int, default=16)
    parser.add_argument("--hr-anchored-rr-adapter", action="store_true")
    parser.add_argument("--hr-anchored-rr-rank", type=int, default=16)
    parser.add_argument("--episodic-physio-whitening-adapter", action="store_true")
    parser.add_argument("--episode-context-physio-mixer", action="store_true")
    parser.add_argument("--episode-context-physio-tokens", type=int, default=6)
    parser.add_argument("--temporal-position-physio-adapter", action="store_true")
    parser.add_argument("--rate-query-physio-adapter", action="store_true")
    parser.add_argument("--rate-query-physio-bins", type=int, default=32)
    parser.add_argument("--task-conditioned-physio-norm", action="store_true")
    parser.add_argument("--task-conditioned-physio-tokens", type=int, default=4)
    parser.add_argument("--support-conditioning", action="store_true")
    parser.add_argument("--support-condition-tasks", default="hr,rr,spo2")
    parser.add_argument("--absolute-time-decoder", action="store_true")
    parser.add_argument("--ppg-weight", type=float, default=1.0)
    parser.add_argument("--pr-weight", type=float, default=0.2)
    parser.add_argument("--hr-weight", type=float, default=0.2)
    parser.add_argument("--rr-weight", type=float, default=0.2)
    parser.add_argument("--spo2-weight", type=float, default=0.2)
    parser.add_argument("--sbp-weight", type=float, default=0.2)
    parser.add_argument("--dbp-weight", type=float, default=0.2)
    parser.add_argument("--map-weight", type=float, default=0.2)
    parser.add_argument("--adaptive-task-weights", action="store_true")
    parser.add_argument("--task-weight-logvar-init", type=float, default=0.0)
    parser.add_argument("--hr-from-ppg-weight", type=float, default=0.0)
    parser.add_argument("--hr-fft-low-bpm", type=float, default=45.0)
    parser.add_argument("--hr-fft-high-bpm", type=float, default=180.0)
    parser.add_argument("--hr-fft-nfft", type=int, default=512)
    parser.add_argument("--hr-fft-temperature", type=float, default=10.0)
    parser.add_argument("--rr-from-ppg-weight", type=float, default=0.0)
    parser.add_argument("--rr-fft-low-bpm", type=float, default=6.0)
    parser.add_argument("--rr-fft-high-bpm", type=float, default=45.0)
    parser.add_argument("--rr-fft-nfft", type=int, default=2048)
    parser.add_argument("--rr-fft-temperature", type=float, default=8.0)
    parser.add_argument("--rr-pseudo-from-ppg-weight", type=float, default=0.0)
    parser.add_argument("--rr-pseudo-fft-low-bpm", type=float, default=6.0)
    parser.add_argument("--rr-pseudo-fft-high-bpm", type=float, default=45.0)
    parser.add_argument("--rr-pseudo-fft-nfft", type=int, default=2048)
    parser.add_argument("--rr-pseudo-fft-temperature", type=float, default=8.0)
    parser.add_argument("--rate-bin-aux-weight", type=float, default=0.0)
    parser.add_argument("--rate-bin-num-bins", type=int, default=96)
    parser.add_argument("--rate-bin-hr-sigma-bpm", type=float, default=6.0)
    parser.add_argument("--rate-bin-rr-sigma-bpm", type=float, default=1.5)
    parser.add_argument("--rate-bin-scalar-mode", default="none", choices=["none", "aux", "blend", "gated", "replace", "only", "residual", "add"])
    parser.add_argument("--rate-query-aux-weight", type=float, default=0.0)
    parser.add_argument("--rate-query-hr-sigma-bpm", type=float, default=6.0)
    parser.add_argument("--rate-query-rr-sigma-bpm", type=float, default=1.5)
    parser.add_argument("--rr-pseudo-rate-bin-weight", type=float, default=0.0)
    parser.add_argument("--rate-bin-entropy-weight", type=float, default=0.0)
    parser.add_argument("--rate-bin-consistency-weight", type=float, default=0.0)
    parser.add_argument("--state-rate-aux-weight", type=float, default=0.0)
    parser.add_argument("--state-rate-ppg-consistency-weight", type=float, default=0.0)
    parser.add_argument("--state-rate-ppg-rr-weight", type=float, default=1.0)
    parser.add_argument("--context-smooth-weight", type=float, default=0.0)
    parser.add_argument("--context-feature-smooth-weight", type=float, default=0.5)
    parser.add_argument("--context-hr-smooth-weight", type=float, default=1.0)
    parser.add_argument("--context-rr-smooth-weight", type=float, default=1.0)
    parser.add_argument("--domain-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--domain-adversarial-lambda", type=float, default=1.0)
    parser.add_argument("--domain-adversarial-hidden-dim", type=int, default=128)
    parser.add_argument("--rate-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--rate-contrastive-rr-weight", type=float, default=1.0)
    parser.add_argument("--rate-contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--rate-contrastive-hr-sigma", type=float, default=8.0)
    parser.add_argument("--rate-contrastive-rr-sigma", type=float, default=2.0)
    parser.add_argument("--rate-contrastive-rr-low-bpm", type=float, default=6.0)
    parser.add_argument("--rate-contrastive-rr-high-bpm", type=float, default=45.0)
    parser.add_argument("--rate-contrastive-fft-nfft", type=int, default=2048)
    parser.add_argument("--rate-contrastive-fft-temperature", type=float, default=8.0)
    parser.add_argument("--task-stream-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--task-stream-rr-weight", type=float, default=1.0)
    parser.add_argument("--task-stream-temperature", type=float, default=0.2)
    parser.add_argument("--task-stream-hr-sigma", type=float, default=8.0)
    parser.add_argument("--task-stream-rr-sigma", type=float, default=2.0)
    parser.add_argument("--task-stream-rr-low-bpm", type=float, default=6.0)
    parser.add_argument("--task-stream-rr-high-bpm", type=float, default=45.0)
    parser.add_argument("--task-stream-fft-nfft", type=int, default=2048)
    parser.add_argument("--task-stream-fft-temperature", type=float, default=8.0)
    parser.add_argument("--task-stream-subject-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--task-stream-subject-adversarial-lambda", type=float, default=1.0)
    parser.add_argument("--task-stream-subject-adversarial-hidden-dim", type=int, default=128)
    parser.add_argument("--task-stream-spectral-alignment-weight", type=float, default=0.0)
    parser.add_argument("--stream-spectral-rr-weight", type=float, default=1.0)
    parser.add_argument("--stream-spectral-hr-low-bpm", type=float, default=45.0)
    parser.add_argument("--stream-spectral-hr-high-bpm", type=float, default=180.0)
    parser.add_argument("--stream-spectral-rr-low-bpm", type=float, default=6.0)
    parser.add_argument("--stream-spectral-rr-high-bpm", type=float, default=45.0)
    parser.add_argument("--stream-spectral-hr-nfft", type=int, default=512)
    parser.add_argument("--stream-spectral-rr-nfft", type=int, default=2048)
    parser.add_argument("--stream-spectral-temperature", type=float, default=8.0)
    parser.add_argument("--task-stream-cross-domain-alignment-weight", type=float, default=0.0)
    parser.add_argument("--cross-domain-rr-weight", type=float, default=1.0)
    parser.add_argument("--cross-domain-temperature", type=float, default=0.2)
    parser.add_argument("--cross-domain-hr-sigma", type=float, default=8.0)
    parser.add_argument("--cross-domain-rr-sigma", type=float, default=2.0)
    parser.add_argument("--cross-domain-rr-low-bpm", type=float, default=6.0)
    parser.add_argument("--cross-domain-rr-high-bpm", type=float, default=45.0)
    parser.add_argument("--cross-domain-rr-nfft", type=int, default=2048)
    parser.add_argument("--cross-domain-rr-fft-temperature", type=float, default=8.0)
    parser.add_argument("--source-rate-ordering-weight", type=float, default=0.0)
    parser.add_argument("--source-rate-ordering-rr-weight", type=float, default=1.0)
    parser.add_argument("--source-rate-ordering-hr-scale", type=float, default=80.0)
    parser.add_argument("--source-rate-ordering-rr-scale", type=float, default=30.0)
    parser.add_argument("--source-rate-ordering-cross-domain-only", action="store_true")
    parser.add_argument("--source-rate-ordering-rr-low-bpm", type=float, default=4.0)
    parser.add_argument("--source-rate-ordering-rr-high-bpm", type=float, default=70.0)
    parser.add_argument("--source-rate-ordering-rr-nfft", type=int, default=2048)
    parser.add_argument("--source-rate-ordering-rr-fft-temperature", type=float, default=8.0)
    parser.add_argument("--source-episode-residual-weight", type=float, default=0.0)
    parser.add_argument("--source-episode-tasks", default="hr,rr,spo2")
    parser.add_argument("--source-episode-min-support", type=int, default=1)
    parser.add_argument("--source-episode-min-query", type=int, default=1)
    parser.add_argument("--source-episode-residual-scale", type=float, default=1.0)
    parser.add_argument("--task-stream-domain-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--task-stream-domain-adversarial-lambda", type=float, default=1.0)
    parser.add_argument("--task-stream-domain-adversarial-hidden-dim", type=int, default=128)
    parser.add_argument("--source-rate-group-dro-weight", type=float, default=0.0)
    parser.add_argument("--source-rate-group-dro-eta", type=float, default=0.2)
    parser.add_argument("--source-rate-group-dro-min-count", type=int, default=1)
    parser.add_argument("--source-rate-group-dro-tasks", default="hr,rr")
    parser.add_argument("--source-rate-group-dro-pseudo-rr-weight", type=float, default=1.0)
    parser.add_argument("--source-rate-group-hr-bin-width", type=float, default=8.0)
    parser.add_argument("--source-rate-group-rr-bin-width", type=float, default=2.0)
    parser.add_argument("--source-rate-group-spo2-bin-width", type=float, default=1.0)
    parser.add_argument("--source-view-consistency-weight", type=float, default=0.0)
    parser.add_argument("--source-view-prediction-consistency-weight", type=float, default=0.0)
    parser.add_argument("--source-view-brightness-jitter", type=float, default=0.05)
    parser.add_argument("--source-view-noise-std", type=float, default=0.01)
    parser.add_argument("--target-view-consistency-weight", type=float, default=0.0)
    parser.add_argument("--target-view-prediction-consistency-weight", type=float, default=0.0)
    parser.add_argument("--target-view-token-vicreg-weight", type=float, default=0.0)
    parser.add_argument("--target-view-token-vicreg-variance-weight", type=float, default=0.02)
    parser.add_argument("--target-view-token-vicreg-covariance-weight", type=float, default=0.002)
    parser.add_argument("--target-view-brightness-jitter", type=float, default=0.03)
    parser.add_argument("--target-view-noise-std", type=float, default=0.005)
    parser.add_argument("--source-masked-prediction-weight", type=float, default=0.0)
    parser.add_argument("--source-masked-temporal-fraction", type=float, default=0.25)
    parser.add_argument("--source-masked-min-span", type=int, default=16)
    parser.add_argument("--source-temporal-reverse-weight", type=float, default=0.0)
    parser.add_argument("--l2-sp-weight", type=float, default=0.0)
    parser.add_argument("--band-disentangle-weight", type=float, default=0.0)
    parser.add_argument("--factor-regularization-weight", type=float, default=0.0)
    parser.add_argument("--strict-train-sanity", action="store_true")
    parser.add_argument("--scalar-hr-low-bpm", type=float, default=45.0)
    parser.add_argument("--scalar-hr-high-bpm", type=float, default=150.0)
    parser.add_argument("--scalar-rr-low-bpm", type=float, default=6.0)
    parser.add_argument("--scalar-rr-high-bpm", type=float, default=30.0)
    parser.add_argument("--recon-weight", type=float, default=0.02)
    parser.add_argument("--selection-metric", default="loss")
    parser.add_argument("--eval-test-every-epoch", action="store_true")
    parser.add_argument("--select-best-on-test", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--eval-hr-bin-width", type=float, default=8.0)
    parser.add_argument("--eval-rr-bin-width", type=float, default=1.0)
    parser.add_argument("--eval-bp-bin-width", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-valid-clips", type=int, default=0)
    parser.add_argument("--exclude-datasets", default="")
    parser.add_argument("--freeze-non-scalar", action="store_true")
    parser.add_argument("--freeze-scalar-heads", action="store_true")
    parser.add_argument("--freeze-scalar-pooled", action="store_true")
    parser.add_argument("--train-scalar-tasks", default="")
    parser.add_argument("--unfreeze-backbone-stem", action="store_true")
    parser.add_argument("--unfreeze-backbone-last-blocks", type=int, default=0)
    parser.add_argument("--unfreeze-backbone-encoder-head", action="store_true")
    parser.add_argument("--unfreeze-temporal-model", action="store_true")
    parser.add_argument("--unfreeze-parameter-substrings", default="")
    parser.add_argument("--unfreeze-pre-temporal-physio-adapter", action="store_true")
    parser.add_argument("--unfreeze-raw-motion-temporal-adapter", action="store_true")
    parser.add_argument("--unfreeze-post-temporal-lowrank-adapter", action="store_true")
    parser.add_argument("--unfreeze-hr-anchored-rr-adapter", action="store_true")
    parser.add_argument("--unfreeze-episodic-physio-whitening-adapter", action="store_true")
    parser.add_argument("--unfreeze-episode-context-physio-mixer", action="store_true")
    parser.add_argument("--unfreeze-temporal-position-physio-adapter", action="store_true")
    parser.add_argument("--unfreeze-rate-query-physio-adapter", action="store_true")
    parser.add_argument("--unfreeze-task-conditioned-physio-norm", action="store_true")
    parser.add_argument("--unfreeze-support-conditioning", action="store_true")
    parser.add_argument("--unfreeze-waveform-head", action="store_true")
    parser.add_argument("--unfreeze-temporal-pyramid", action="store_true")
    parser.add_argument("--unfreeze-temporal-refiner", action="store_true")
    parser.add_argument("--unfreeze-representation-refiner", action="store_true")
    parser.add_argument("--unfreeze-vital-representation-adapter", action="store_true")
    parser.add_argument("--unfreeze-cross-task-representation-adapter", action="store_true")
    parser.add_argument("--unfreeze-rr-only-slow-adapter", action="store_true")
    parser.add_argument("--unfreeze-prototype-scalar-readout", action="store_true")
    parser.add_argument("--unfreeze-rr-lowfreq-branch", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--sampler-dataset-balance", action="store_true")
    parser.add_argument("--sampler-hr-balance", action="store_true")
    parser.add_argument("--sampler-hr-bin-width", type=float, default=10.0)
    parser.add_argument("--sampler-rr-balance", action="store_true")
    parser.add_argument("--sampler-rr-bin-width", type=float, default=2.0)
    parser.add_argument("--sampler-zpu-weight", type=float, default=1.0)
    parser.add_argument("--sampler-high-hr-threshold", type=float, default=95.0)
    parser.add_argument("--sampler-high-hr-weight", type=float, default=1.0)
    parser.add_argument("--sampler-num-samples-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    if args.select_best_on_test and (not args.eval_test_every_epoch or not args.test_csv):
        raise ValueError("--select-best-on-test requires --eval-test-every-epoch and --test-csv")

    data_generator = seed_everything(args.seed)
    print(f"seed={args.seed}")
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = limit_round_robin(filter_excluded_datasets(read_manifest(args.train_csv), args.exclude_datasets), args.max_train_clips)
    valid_rows = limit_round_robin(filter_excluded_datasets(read_manifest(args.valid_csv), args.exclude_datasets), args.max_valid_clips)
    test_rows = read_manifest(args.test_csv) if args.test_csv else []
    if args.support_conditioning:
        support_context = build_support_context(train_rows, args.support_condition_tasks)
        args.support_context_tensor = support_context
        args.support_context_dim = int(support_context.numel())
        print(
            f"support_conditioning tasks={args.support_condition_tasks} "
            f"context_dim={args.support_context_dim} "
            f"context={[round(float(item), 5) for item in support_context.tolist()]}"
        )
    else:
        args.support_context_tensor = None
        args.support_context_dim = 0
    dataset_names = sorted({row.get("dataset", "") for row in (train_rows + valid_rows + test_rows)})
    dataset_to_idx = {name: idx for idx, name in enumerate(dataset_names)}
    subject_names = sorted({str(row.get("subject", "")) for row in train_rows})
    subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
    if args.dataset_conditioning or args.task_dataset_conditioning:
        args.dataset_num_domains = max(args.dataset_num_domains, len(dataset_names))
        print(f"dataset_conditioning datasets={dataset_names}")
    if args.context_clips > 1:
        train_rows = build_context_samples(train_rows, args.context_clips, args.context_stride)
        valid_rows = build_context_samples(valid_rows, args.context_clips, args.eval_context_stride)
        test_rows = build_context_samples(test_rows, args.context_clips, args.eval_context_stride) if test_rows else []
    print(f"train_rows={len(train_rows)} valid_rows={len(valid_rows)} test_rows={len(test_rows)}")
    train_sampler = make_weighted_sampler(train_rows, args)

    train_loader = DataLoader(
        MultiTaskClipDataset(train_rows, frames=args.frames, dataset_to_idx=dataset_to_idx),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )
    valid_loader = DataLoader(
        MultiTaskClipDataset(valid_rows, frames=args.frames, dataset_to_idx=dataset_to_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    test_loader = (
        DataLoader(
            MultiTaskClipDataset(
                test_rows,
                frames=args.frames,
                dataset_to_idx=dataset_to_idx,
                corruption=args.test_corruption,
                severity=args.corruption_severity,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            drop_last=False,
            worker_init_fn=seed_worker,
        )
        if test_rows
        else None
    )

    if args.eval_only_checkpoint:
        args.pretrained_checkpoint = args.eval_only_checkpoint
    model = make_model(args, device)
    if args.eval_only_checkpoint:
        final_rows = []
        valid_metrics, valid_outputs = evaluate(model, valid_loader, args, device, save_outputs=True)
        valid_pickle = output_dir / f"{args.model_name}_valid_outputs.pickle"
        with valid_pickle.open("wb") as handle:
            pickle.dump(valid_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        final_rows.append({"split": "valid", "selection_source": "eval_only", **valid_metrics})
        print(f"eval_only_checkpoint={args.eval_only_checkpoint}")
        print(f"final_valid {format_metrics(valid_metrics)}")
        if test_loader is not None:
            test_metrics, test_outputs = evaluate(model, test_loader, args, device, save_outputs=True)
            test_pickle = output_dir / f"{args.model_name}_test_outputs.pickle"
            with test_pickle.open("wb") as handle:
                pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            final_rows.append({"split": "test", "selection_source": "eval_only", **test_metrics})
            print(f"test_corruption={args.test_corruption} severity={args.corruption_severity}")
            print(f"final_test {format_metrics(test_metrics)}")
        write_metric_rows(output_dir / f"{args.model_name}_final_metrics.csv", final_rows)
        return
    if (
        args.freeze_non_scalar
        or args.unfreeze_backbone_stem
        or args.unfreeze_backbone_last_blocks > 0
        or args.unfreeze_backbone_encoder_head
        or args.unfreeze_temporal_model
        or args.unfreeze_pre_temporal_physio_adapter
        or args.pre_temporal_physio_adapter
        or args.unfreeze_raw_motion_temporal_adapter
        or args.raw_motion_temporal_adapter
        or args.unfreeze_hr_anchored_rr_adapter
        or args.hr_anchored_rr_adapter
        or args.unfreeze_episodic_physio_whitening_adapter
        or args.episodic_physio_whitening_adapter
        or args.unfreeze_episode_context_physio_mixer
        or args.episode_context_physio_mixer
        or args.unfreeze_temporal_position_physio_adapter
        or args.temporal_position_physio_adapter
        or args.unfreeze_rate_query_physio_adapter
        or args.rate_query_physio_adapter
        or args.unfreeze_task_conditioned_physio_norm
        or args.task_conditioned_physio_norm
        or args.unfreeze_support_conditioning
        or args.support_conditioning
        or args.unfreeze_waveform_head
        or args.unfreeze_temporal_pyramid
        or args.unfreeze_temporal_refiner
        or args.unfreeze_representation_refiner
        or args.unfreeze_vital_representation_adapter
        or args.vital_representation_adapter
        or args.unfreeze_cross_task_representation_adapter
        or args.cross_task_representation_adapter
        or args.unfreeze_rr_only_slow_adapter
        or args.rr_only_slow_adapter
        or args.unfreeze_prototype_scalar_readout
        or args.prototype_scalar_readout
        or args.unfreeze_rr_lowfreq_branch
        or args.rr_lowfreq_branch
        or args.video_rate_branch
        or args.color_scalar_branch
        or args.hr_band_residual_branch
    ):
        trainable, total = apply_freeze_policy(model, args)
        print(
            "freeze_policy "
            f"freeze_non_scalar={bool(args.freeze_non_scalar)} "
            f"unfreeze_backbone_stem={bool(args.unfreeze_backbone_stem)} "
            f"unfreeze_backbone_last_blocks={int(args.unfreeze_backbone_last_blocks)} "
            f"unfreeze_backbone_encoder_head={bool(args.unfreeze_backbone_encoder_head)} "
            f"unfreeze_temporal_model={bool(args.unfreeze_temporal_model)} "
            f"unfreeze_pre_temporal_physio_adapter={bool(args.unfreeze_pre_temporal_physio_adapter)} "
            f"unfreeze_raw_motion_temporal_adapter={bool(args.unfreeze_raw_motion_temporal_adapter)} "
            f"unfreeze_hr_anchored_rr_adapter={bool(args.unfreeze_hr_anchored_rr_adapter)} "
            f"unfreeze_episodic_physio_whitening_adapter={bool(args.unfreeze_episodic_physio_whitening_adapter)} "
            f"unfreeze_episode_context_physio_mixer={bool(args.unfreeze_episode_context_physio_mixer)} "
            f"unfreeze_temporal_position_physio_adapter={bool(args.unfreeze_temporal_position_physio_adapter)} "
            f"unfreeze_rate_query_physio_adapter={bool(args.unfreeze_rate_query_physio_adapter)} "
            f"unfreeze_task_conditioned_physio_norm={bool(args.unfreeze_task_conditioned_physio_norm)} "
            f"unfreeze_support_conditioning={bool(args.unfreeze_support_conditioning)} "
            f"unfreeze_waveform_head={bool(args.unfreeze_waveform_head)} "
            f"unfreeze_temporal_pyramid={bool(args.unfreeze_temporal_pyramid)} "
            f"unfreeze_temporal_refiner={bool(args.unfreeze_temporal_refiner)} "
            f"unfreeze_representation_refiner={bool(args.unfreeze_representation_refiner)} "
            f"unfreeze_vital_representation_adapter={bool(args.unfreeze_vital_representation_adapter)} "
            f"unfreeze_cross_task_representation_adapter={bool(args.unfreeze_cross_task_representation_adapter)} "
            f"unfreeze_rr_only_slow_adapter={bool(args.unfreeze_rr_only_slow_adapter)} "
            f"unfreeze_prototype_scalar_readout={bool(args.unfreeze_prototype_scalar_readout)} "
            f"vital_representation_adapter={bool(args.vital_representation_adapter)} "
            f"cross_task_representation_adapter={bool(args.cross_task_representation_adapter)} "
            f"rr_only_slow_adapter={bool(args.rr_only_slow_adapter)} "
            f"prototype_scalar_readout={bool(args.prototype_scalar_readout)} "
            f"pre_temporal_physio_adapter={bool(args.pre_temporal_physio_adapter)} "
            f"episodic_physio_whitening_adapter={bool(args.episodic_physio_whitening_adapter)} "
            f"episode_context_physio_mixer={bool(args.episode_context_physio_mixer)} "
            f"temporal_position_physio_adapter={bool(args.temporal_position_physio_adapter)} "
            f"rate_query_physio_adapter={bool(args.rate_query_physio_adapter)} "
            f"task_conditioned_physio_norm={bool(args.task_conditioned_physio_norm)} "
            f"support_conditioning={bool(args.support_conditioning)} "
            f"rr_lowfreq_branch={bool(args.rr_lowfreq_branch)} "
            f"video_rate_branch={bool(args.video_rate_branch)} "
            f"color_scalar_branch={bool(args.color_scalar_branch)} "
            f"hr_band_residual_branch={bool(args.hr_band_residual_branch)} "
            f"train_scalar_tasks={args.train_scalar_tasks or '-'} "
            f"trainable_params={trainable} total_params={total}"
        )
    if args.freeze_scalar_pooled:
        frozen = 0
        for name, param in model.named_parameters():
            if ".pooled_head." in name:
                param.requires_grad = False
                frozen += param.numel()
        print(f"freeze_scalar_pooled frozen_params={frozen}")
    if args.freeze_scalar_heads:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith("scalar_heads."):
                param.requires_grad = False
                frozen += param.numel()
        print(f"freeze_scalar_heads frozen_params={frozen}")
    l2_sp_anchors = None
    if args.l2_sp_weight > 0.0:
        l2_sp_anchors = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        print(f"l2_sp enabled weight={float(args.l2_sp_weight):.5f} anchored_params={len(l2_sp_anchors)}")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    domain_adversary = None
    if args.domain_adversarial_weight > 0.0 and len(dataset_names) > 1:
        domain_adversary = DomainAdversary(
            feature_dim=args.feature_dim,
            hidden_dim=args.domain_adversarial_hidden_dim,
            num_domains=len(dataset_names),
        ).to(device)
        trainable_params = trainable_params + list(domain_adversary.parameters())
        print(
            "domain_adversary "
            f"enabled domains={dataset_names} "
            f"weight={float(args.domain_adversarial_weight):.5f} "
            f"lambda={float(args.domain_adversarial_lambda):.5f}"
        )
    task_stream_domain_adversary = None
    if args.task_stream_domain_adversarial_weight > 0.0 and len(dataset_names) > 1:
        task_stream_domain_adversary = TaskStreamDomainAdversary(
            feature_dim=args.feature_dim,
            hidden_dim=args.task_stream_domain_adversarial_hidden_dim,
            num_domains=len(dataset_names),
        ).to(device)
        trainable_params = trainable_params + list(task_stream_domain_adversary.parameters())
        print(
            "task_stream_domain_adversary "
            f"enabled domains={dataset_names} "
            f"weight={float(args.task_stream_domain_adversarial_weight):.5f} "
            f"lambda={float(args.task_stream_domain_adversarial_lambda):.5f}"
        )
    task_stream_subject_adversary = None
    if args.task_stream_subject_adversarial_weight > 0.0 and len(subject_names) > 1:
        task_stream_subject_adversary = TaskStreamDomainAdversary(
            feature_dim=args.feature_dim,
            hidden_dim=args.task_stream_subject_adversarial_hidden_dim,
            num_domains=len(subject_names),
        ).to(device)
        trainable_params = trainable_params + list(task_stream_subject_adversary.parameters())
        print(
            "task_stream_subject_adversary "
            f"enabled subjects={len(subject_names)} "
            f"weight={float(args.task_stream_subject_adversarial_weight):.5f} "
            f"lambda={float(args.task_stream_subject_adversarial_lambda):.5f}"
        )
    task_log_vars = None
    if args.adaptive_task_weights:
        task_log_vars = torch.nn.Parameter(
            torch.full((len(TASKS),), float(args.task_weight_logvar_init), device=device, dtype=torch.float32)
        )
        trainable_params = trainable_params + [task_log_vars]
        print(
            "adaptive_task_weights "
            f"enabled init={float(args.task_weight_logvar_init):.5f} "
            f"tasks={','.join(TASKS)}"
        )
    if not trainable_params:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    if ema is not None:
        print(f"ema enabled decay={float(args.ema_decay):.5f} tracked_params={len(ema.shadow)}")
    best_valid = float("inf")
    best_test = float("inf")
    best_saved = float("inf")
    best_path = output_dir / f"{args.model_name}_best.pth"
    last_path = output_dir / f"{args.model_name}_last.pth"
    history_rows: list[dict[str, float | int | str]] = []

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            args,
            device,
            task_log_vars=task_log_vars,
            domain_adversary=domain_adversary,
            task_stream_domain_adversary=task_stream_domain_adversary,
            task_stream_subject_adversary=task_stream_subject_adversary,
            subject_to_idx=subject_to_idx,
            l2_sp_anchors=l2_sp_anchors,
            ema=ema,
        )
        valid_metrics, _ = evaluate_with_optional_ema(model, valid_loader, args, device, ema=ema, task_log_vars=task_log_vars)
        valid_key = selection_value(valid_metrics, args.selection_metric)
        if valid_key < best_valid:
            best_valid = valid_key
        if not args.select_best_on_test and valid_key < best_saved:
            best_saved = valid_key
            torch.save(ema.state_dict(model) if ema is not None else model.state_dict(), best_path)
        print(f"epoch={epoch} train {format_metrics(train_metrics)}")
        print(f"epoch={epoch} valid {format_metrics(valid_metrics)} best_{args.selection_metric}={best_valid:.5f}")
        history_rows.append({"epoch": epoch, "split": "train", **train_metrics})
        history_rows.append({"epoch": epoch, "split": "valid", f"best_{args.selection_metric}": best_valid, **valid_metrics})
        if args.eval_test_every_epoch and test_loader is not None:
            test_metrics, _ = evaluate_with_optional_ema(model, test_loader, args, device, ema=ema, task_log_vars=task_log_vars)
            test_key = selection_value(test_metrics, args.selection_metric)
            if test_key < best_test:
                best_test = test_key
            if args.select_best_on_test and test_key < best_saved:
                best_saved = test_key
                torch.save(ema.state_dict(model) if ema is not None else model.state_dict(), best_path)
            print(f"epoch={epoch} test {format_metrics(test_metrics)} best_test_{args.selection_metric}={best_test:.5f}")
            history_rows.append({"epoch": epoch, "split": "test", f"best_test_{args.selection_metric}": best_test, **test_metrics})
        if task_log_vars is not None:
            task_weights_snapshot = {
                task: float(torch.exp(-task_log_vars[idx]).detach().cpu())
                for idx, task in enumerate(TASKS)
            }
            print(
                "task_weights "
                + " ".join(f"{task}={value:.5f}" for task, value in task_weights_snapshot.items())
            )
        write_metric_rows(output_dir / f"{args.model_name}_metrics_history.csv", history_rows)
    torch.save(ema.state_dict(model) if ema is not None else model.state_dict(), last_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_valid, valid_outputs = evaluate(model, valid_loader, args, device, save_outputs=True, task_log_vars=task_log_vars)
    output_pickle = output_dir / f"{args.model_name}_valid_outputs.pickle"
    with output_pickle.open("wb") as handle:
        pickle.dump(valid_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"best_checkpoint={best_path}")
    print(f"last_checkpoint={last_path}")
    print(
        f"best_selection_source={'test' if args.select_best_on_test else 'valid'} "
        f"best_saved_{args.selection_metric}={best_saved:.5f}"
    )
    print(f"valid_outputs={output_pickle}")
    print(f"final_valid {format_metrics(final_valid)}")
    final_rows = [{"split": "valid", "selection_source": "test" if args.select_best_on_test else "valid", **final_valid}]

    if test_loader is not None:
        test_metrics, test_outputs = evaluate(model, test_loader, args, device, save_outputs=True, task_log_vars=task_log_vars)
        test_pickle = output_dir / f"{args.model_name}_test_outputs.pickle"
        with test_pickle.open("wb") as handle:
            pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"test_outputs={test_pickle}")
        print(f"final_test {format_metrics(test_metrics)}")
        final_rows.append({"split": "test", "selection_source": "test" if args.select_best_on_test else "valid", **test_metrics})
    write_metric_rows(output_dir / f"{args.model_name}_final_metrics.csv", final_rows)


if __name__ == "__main__":
    main()

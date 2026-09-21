\
"""Train SpectroPhys with masked PPG/PR/HR/RR/SpO2 supervision."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from neural_methods.model.SpectroPhys import SpectroPhys


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
    )


def finite_float(value: str | float | int | None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


class MultiTaskClipDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], frames: int = 160, dataset_to_idx: dict[str, int] | None = None):
        self.rows = rows
        self.frames = int(frames)
        self.dataset_to_idx = dataset_to_idx or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        context_rows = row.get("_context_rows") or [row]
        video_chunks = [np.load(item["input_files"]).astype(np.float32)[: self.frames] for item in context_rows]
        video = np.concatenate(video_chunks, axis=0)
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
        raise ValueError("SpectroPhys did not return scalar outputs. Instantiate with scalar_tasks.")
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

    total = (
        args.ppg_weight * ppg_loss
        + scalar_loss
        + args.hr_from_ppg_weight * hr_freq_loss
        + args.recon_weight * recon_loss
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "ppg_loss": float(ppg_loss.detach().cpu()),
        "hr_from_ppg_loss": float(hr_freq_loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        **task_losses,
    }
    return total, logs


def train_epoch(model, loader, optimizer, args, device, task_log_vars: torch.Tensor | None = None):
    model.train()
    sums = {}
    count = 0
    for video, ppg, ppg_mask, scalar_targets, scalar_masks, dataset_ids, _, sort_indices in loader:
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
        )
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
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
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
    scalar_counts = {task: 0.0 for task in TASKS}
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
            scalar_counts[task] += float(mask.sum().cpu())
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
        else:
            metrics[f"{task}_mae"] = float("nan")
    for task, bins in scalar_bin_errors.items():
        bin_maes = [error_sum / error_count for error_sum, error_count in bins.values() if error_count > 0]
        metrics[f"{task}_balanced_mae"] = float(np.mean(bin_maes)) if bin_maes else float("nan")
    return metrics, outputs


def prepare_video(video, args, device):
    video = video.to(device, dtype=torch.float32)
    if args.raw_div255:
        video = video / 255.0
    return video


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
    model = SpectroPhys(
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        tcn_layers=args.tcn_layers,
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
        dataset_num_domains=args.dataset_num_domains,
        long_range_mixer=args.long_range_mixer,
        representation_refiner=args.representation_refiner,
        representation_bottleneck_dim=args.representation_bottleneck_dim,
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
    if args.unfreeze_waveform_head:
        _set_requires_grad(getattr(model, "regressor_head", None), True)
        _set_requires_grad(getattr(model, "frequency_waveform_decoder", None), True)
    if args.unfreeze_temporal_refiner:
        _set_requires_grad(getattr(model, "temporal_refiner", None), True)
    if args.unfreeze_representation_refiner:
        _set_requires_grad(getattr(model, "representation_refiner", None), True)
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
        )
        task_gate_prefixes = (
            "rate_residual_gates",
            "video_rate_gates",
            "color_scalar_gates",
            "hr_band_residual_gates",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--output-dir", default="runs/multitask_spectrophys")
    parser.add_argument("--model-name", default="multitask_spectrophys")
    parser.add_argument("--pretrained-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--context-clips", type=int, default=1)
    parser.add_argument("--context-stride", type=int, default=1)
    parser.add_argument("--eval-context-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-div255", action="store_true")
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--input-normalization", default="none")
    parser.add_argument("--waveform-head-type", default="multiscale")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tcn-layers", type=int, default=4)
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
    parser.add_argument("--dataset-num-domains", type=int, default=0)
    parser.add_argument("--representation-refiner", action="store_true")
    parser.add_argument("--long-range-mixer", action="store_true")
    parser.add_argument("--representation-bottleneck-dim", type=int, default=32)
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
    parser.add_argument("--recon-weight", type=float, default=0.02)
    parser.add_argument("--selection-metric", default="loss")
    parser.add_argument("--eval-test-every-epoch", action="store_true")
    parser.add_argument("--eval-hr-bin-width", type=float, default=8.0)
    parser.add_argument("--eval-rr-bin-width", type=float, default=1.0)
    parser.add_argument("--eval-bp-bin-width", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-valid-clips", type=int, default=0)
    parser.add_argument("--freeze-non-scalar", action="store_true")
    parser.add_argument("--freeze-scalar-heads", action="store_true")
    parser.add_argument("--freeze-scalar-pooled", action="store_true")
    parser.add_argument("--train-scalar-tasks", default="")
    parser.add_argument("--unfreeze-backbone-stem", action="store_true")
    parser.add_argument("--unfreeze-backbone-last-blocks", type=int, default=0)
    parser.add_argument("--unfreeze-backbone-encoder-head", action="store_true")
    parser.add_argument("--unfreeze-temporal-model", action="store_true")
    parser.add_argument("--unfreeze-waveform-head", action="store_true")
    parser.add_argument("--unfreeze-temporal-pyramid", action="store_true")
    parser.add_argument("--unfreeze-temporal-refiner", action="store_true")
    parser.add_argument("--unfreeze-representation-refiner", action="store_true")
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

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = limit_round_robin(read_manifest(args.train_csv), args.max_train_clips)
    valid_rows = limit_round_robin(read_manifest(args.valid_csv), args.max_valid_clips)
    test_rows = read_manifest(args.test_csv) if args.test_csv else []
    dataset_names = sorted({row.get("dataset", "") for row in (train_rows + valid_rows + test_rows)})
    dataset_to_idx = {name: idx for idx, name in enumerate(dataset_names)}
    if args.dataset_conditioning:
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
    )
    valid_loader = DataLoader(
        MultiTaskClipDataset(valid_rows, frames=args.frames, dataset_to_idx=dataset_to_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    test_loader = (
        DataLoader(
            MultiTaskClipDataset(test_rows, frames=args.frames, dataset_to_idx=dataset_to_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            drop_last=False,
        )
        if test_rows
        else None
    )

    model = make_model(args, device)
    if (
        args.freeze_non_scalar
        or args.unfreeze_backbone_stem
        or args.unfreeze_backbone_last_blocks > 0
        or args.unfreeze_backbone_encoder_head
        or args.unfreeze_temporal_model
        or args.unfreeze_waveform_head
        or args.unfreeze_temporal_pyramid
        or args.unfreeze_temporal_refiner
        or args.unfreeze_representation_refiner
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
            f"unfreeze_waveform_head={bool(args.unfreeze_waveform_head)} "
            f"unfreeze_temporal_pyramid={bool(args.unfreeze_temporal_pyramid)} "
            f"unfreeze_temporal_refiner={bool(args.unfreeze_temporal_refiner)} "
            f"unfreeze_representation_refiner={bool(args.unfreeze_representation_refiner)} "
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
    trainable_params = [param for param in model.parameters() if param.requires_grad]
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
    best_valid = float("inf")
    best_path = output_dir / f"{args.model_name}_best.pth"

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, args, device, task_log_vars=task_log_vars)
        valid_metrics, _ = evaluate(model, valid_loader, args, device, task_log_vars=task_log_vars)
        valid_key = selection_value(valid_metrics, args.selection_metric)
        if valid_key < best_valid:
            best_valid = valid_key
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch} train {format_metrics(train_metrics)}")
        print(f"epoch={epoch} valid {format_metrics(valid_metrics)} best_{args.selection_metric}={best_valid:.5f}")
        if args.eval_test_every_epoch and test_loader is not None:
            test_metrics, _ = evaluate(model, test_loader, args, device, task_log_vars=task_log_vars)
            print(f"epoch={epoch} test {format_metrics(test_metrics)}")
        if task_log_vars is not None:
            task_weights_snapshot = {
                task: float(torch.exp(-task_log_vars[idx]).detach().cpu())
                for idx, task in enumerate(TASKS)
            }
            print(
                "task_weights "
                + " ".join(f"{task}={value:.5f}" for task, value in task_weights_snapshot.items())
            )

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_valid, valid_outputs = evaluate(model, valid_loader, args, device, save_outputs=True, task_log_vars=task_log_vars)
    output_pickle = output_dir / f"{args.model_name}_valid_outputs.pickle"
    with output_pickle.open("wb") as handle:
        pickle.dump(valid_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"best_checkpoint={best_path}")
    print(f"valid_outputs={output_pickle}")
    print(f"final_valid {format_metrics(final_valid)}")

    if test_loader is not None:
        test_metrics, test_outputs = evaluate(model, test_loader, args, device, save_outputs=True, task_log_vars=task_log_vars)
        test_pickle = output_dir / f"{args.model_name}_test_outputs.pickle"
        with test_pickle.open("wb") as handle:
            pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"test_outputs={test_pickle}")
        print(f"final_test {format_metrics(test_metrics)}")


if __name__ == "__main__":
    main()

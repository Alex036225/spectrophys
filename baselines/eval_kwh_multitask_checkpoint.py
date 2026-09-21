\
"""Evaluate a PhaseNet-MT checkpoint on KWH cached clips.

KWH cached files provide video input and PPG waveform labels. This wrapper
derives HR/PR labels from the PPG label by FFT and masks RR/SpO2.
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
from torch.utils.data import DataLoader

from tools.train_multitask_phasenet import (
    MultiTaskClipDataset,
    evaluate,
    format_metrics,
    make_model,
    read_manifest,
)


def fft_bpm(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> float:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    signal = signal - np.mean(signal)
    if signal.size < 8 or not np.isfinite(signal).all():
        return float("nan")
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return float("nan")
    return float(freqs[mask][np.argmax(power[mask])] * 60.0)


def label_path_for_input(input_file: str) -> str:
    path = Path(input_file)
    match = re.match(r"(.+)_input(\d+)\.npy$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse input file name: {input_file}")
    return str(path.with_name(f"{match.group(1)}_label{match.group(2)}.npy"))


def subject_and_clip(input_file: str) -> tuple[str, int]:
    stem = Path(input_file).name
    match = re.match(r"(.+)_input(\d+)\.npy$", stem)
    if match is None:
        raise ValueError(f"Cannot parse input file name: {input_file}")
    subject = match.group(1)
    clip_index = int(match.group(2))
    clip_match = re.search(r"_clip(\d+)$", subject)
    if clip_match is not None:
        clip_index = int(clip_match.group(1))
    return subject, clip_index


def build_manifest(input_csv: Path, output_csv: Path, fs: float, low_hz: float, high_hz: float) -> None:
    with input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = [
        "input_files",
        "subject",
        "clip_index",
        "ppg_label_file",
        "label_polarity",
        "ppg_mask",
        "dataset",
        "pr_bpm",
        "pr_mask",
        "hr_bpm",
        "hr_mask",
        "rr_bpm",
        "rr_mask",
        "spo2_pct",
        "spo2_mask",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            input_file = row["input_files"]
            label_file = row.get("ppg_label_file") or label_path_for_input(input_file)
            subject = row.get("subject")
            clip_index = row.get("clip_index")
            if not subject or not clip_index:
                parsed_subject, parsed_clip = subject_and_clip(input_file)
                subject = subject or parsed_subject
                clip_index = clip_index or str(parsed_clip)
            bpm = float("nan")
            if Path(label_file).exists():
                bpm = fft_bpm(np.load(label_file)[:160], fs=fs, low_hz=low_hz, high_hz=high_hz)
            valid = math.isfinite(bpm)
            writer.writerow(
                {
                    "input_files": input_file,
                    "subject": subject,
                    "clip_index": clip_index,
                    "ppg_label_file": label_file,
                    "label_polarity": row.get("label_polarity", "1.0") or "1.0",
                    "ppg_mask": "1" if Path(label_file).exists() else "0",
                    "dataset": "KWH",
                    "pr_bpm": f"{bpm:.8f}" if valid else "",
                    "pr_mask": "1" if valid else "0",
                    "hr_bpm": f"{bpm:.8f}" if valid else "",
                    "hr_mask": "1" if valid else "0",
                    "rr_bpm": "",
                    "rr_mask": "0",
                    "spo2_pct": "",
                    "spo2_mask": "0",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="kwh_phasenet_mt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--low-hz", type=float, default=1.3)
    parser.add_argument("--high-hz", type=float, default=3.5)
    args_cli = parser.parse_args()

    output_dir = Path(args_cli.output_dir)
    manifest_csv = output_dir / f"{args_cli.name}_manifest.csv"
    build_manifest(Path(args_cli.input_csv), manifest_csv, fs=args_cli.fs, low_hz=args_cli.low_hz, high_hz=args_cli.high_hz)

    args = SimpleNamespace(
        pretrained_checkpoint=args_cli.checkpoint,
        frames=160,
        raw_div255=False,
        fs=args_cli.fs,
        input_normalization="channel_mean_center",
        waveform_head_type="gru",
        feature_dim=128,
        latent_dim=32,
        hidden_dim=128,
        tcn_layers=4,
        scalar_head_hidden_dim=128,
        scalar_head_type="pooled",
        rr_lowfreq_branch=False,
        rr_lowfreq_mode="replace",
        video_rate_branch=False,
        video_rate_mode="replace",
        color_scalar_branch=False,
        color_scalar_mode="residual",
        hr_band_residual_branch=False,
        encoder_motion_fusion=False,
        encoder_temporal_pyramid=False,
        encoder_color_motion_branch=False,
        mixstyle_stem=False,
        mixstyle_encoder=False,
        mixstyle_prob=0.5,
        mixstyle_alpha=0.1,
        dataset_conditioning=False,
        dataset_num_domains=0,
        long_range_mixer=False,
        representation_refiner=False,
        representation_bottleneck_dim=32,
        absolute_time_decoder=False,
        ppg_weight=0.0,
        pr_weight=1.0,
        hr_weight=1.5,
        rr_weight=0.0,
        spo2_weight=0.0,
        hr_from_ppg_weight=0.0,
        hr_fft_low_bpm=args_cli.low_hz * 60.0,
        hr_fft_high_bpm=args_cli.high_hz * 60.0,
        hr_fft_nfft=512,
        hr_fft_temperature=10.0,
        recon_weight=0.0,
        eval_hr_bin_width=8.0,
        eval_rr_bin_width=1.0,
    )
    device = torch.device(args_cli.device if torch.cuda.is_available() or not args_cli.device.startswith("cuda") else "cpu")
    rows = read_manifest(manifest_csv)
    print(f"device={device}")
    print(f"checkpoint={args_cli.checkpoint}")
    print(f"input_csv={args_cli.input_csv}")
    print(f"manifest_csv={manifest_csv}")
    print(f"rows={len(rows)} subjects={len({row.get('subject', '') for row in rows})}")
    loader = DataLoader(
        MultiTaskClipDataset(rows, frames=160),
        batch_size=args_cli.batch_size,
        shuffle=False,
        num_workers=args_cli.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = make_model(args, device)
    metrics, outputs = evaluate(model, loader, args, device, save_outputs=True)
    output_pickle = output_dir / f"{args_cli.name}_outputs.pickle"
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_pickle.open("wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"test_outputs={output_pickle}")
    print(f"final_test {format_metrics(metrics)}")


if __name__ == "__main__":
    main()

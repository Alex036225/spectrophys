\
"""Train SBP/DBP/MAP adapters on top of classical unsupervised rPPG waveforms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.ICA_POH import ICA_POH
from unsupervised_methods.methods.POS_WANG import POS_WANG


ROOT = Path("/public_hw/home/cit_yingxinlai/project/nature")
MANIFEST_DIR = Path("/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/JOINT/DataFileLists")
BP_PREFIX = "MultiTask_PPG_PR_HRpleth512_SpO2_BP_72x72_160"
METHODS = ("GREEN", "ICA", "CHROM", "POS")
TASKS = ("sbp", "dbp", "map")
DEFAULTS = {"sbp": 120.0, "dbp": 75.0, "map": 90.0}
SCALES = torch.tensor([25.0, 15.0, 20.0])


def parse_clip(path: str) -> tuple[str, int]:
    match = re.match(r"(.+)_input(\d+)\.npy$", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse cached clip name: {path}")
    return match.group(1), int(match.group(2))


def finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        subject, clip_index = parse_clip(row["input_files"])
        row.setdefault("subject", subject)
        row.setdefault("clip_index", str(clip_index))
    return rows


def prepare_video(video: np.ndarray) -> np.ndarray:
    video = np.asarray(video)
    if video.shape[-1] > 3:
        video = video[..., :3]
    if np.issubdtype(video.dtype, np.floating) and np.nanmax(video) <= 1.0:
        video = video * 255.0
    return video.astype(np.float32, copy=False)


def run_method(method: str, frames: np.ndarray, fs: float) -> np.ndarray:
    if method == "GREEN":
        return np.asarray(GREEN(frames), dtype=np.float32).reshape(-1)
    if method == "ICA":
        return np.asarray(ICA_POH(frames, fs), dtype=np.float32).reshape(-1)
    if method == "CHROM":
        return np.asarray(CHROME_DEHAAN(frames, fs), dtype=np.float32).reshape(-1)
    if method == "POS":
        return np.asarray(POS_WANG(frames, fs), dtype=np.float32).reshape(-1)
    raise ValueError(method)


def pad_or_trim(signal: np.ndarray, length: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if signal.size >= length:
        return signal[:length]
    if signal.size == 0:
        return np.zeros((length,), dtype=np.float32)
    return np.pad(signal, (0, length - signal.size), mode="edge")


def extract_bp(row: dict[str, str]) -> tuple[list[float], list[float]]:
    values = []
    masks = []
    for task in TASKS:
        value = finite_float(row.get(f"{task}_mmhg"))
        mask = 1.0 if row.get(f"{task}_mask", "0") == "1" and value is not None else 0.0
        values.append(DEFAULTS[task] if value is None else value)
        masks.append(mask)
    return values, masks


def build_arrays(rows: list[dict[str, str]], method: str, args, cache_path: Path):
    if cache_path.exists() and not args.rebuild_cache:
        data = np.load(cache_path, allow_pickle=True)
        return data["waveforms"], data["labels"], data["masks"], data["subjects"], data["clip_indices"]

    waveforms = []
    labels = []
    masks = []
    subjects = []
    clip_indices = []
    for idx, row in enumerate(rows):
        video = prepare_video(np.load(row["input_files"]))
        waveform = pad_or_trim(run_method(method, video, args.fs), args.frames)
        bp_values, bp_masks = extract_bp(row)
        waveforms.append(waveform)
        labels.append(bp_values)
        masks.append(bp_masks)
        subjects.append(row["subject"])
        clip_indices.append(int(row["clip_index"]))
        if (idx + 1) % 100 == 0:
            print(f"method={method} cached={idx + 1}/{len(rows)} split={cache_path.stem}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        waveforms=np.asarray(waveforms, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        masks=np.asarray(masks, dtype=np.float32),
        subjects=np.asarray(subjects, dtype=object),
        clip_indices=np.asarray(clip_indices, dtype=np.int64),
    )
    return (
        np.asarray(waveforms, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
        np.asarray(subjects, dtype=object),
        np.asarray(clip_indices, dtype=np.int64),
    )


def normalize_signal(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True, unbiased=False) + eps)


class WaveformBPHead(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(hidden * 8, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )
        self.register_buffer("low", torch.tensor([80.0, 45.0, 55.0]))
        self.register_buffer("span", torch.tensor([80.0, 55.0, 75.0]))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.net(normalize_signal(waveform).unsqueeze(1)) * self.span + self.low


def bp_loss(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return pred.sum() * 0.0
    scale = SCALES.to(pred.device).expand_as(pred)
    return F.smooth_l1_loss((pred[valid] - label[valid]) / scale[valid], torch.zeros_like(pred[valid]))


def metric_summary(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {"bp_count": int(len(pred))}
    for idx, task in enumerate(TASKS):
        diff = pred[:, idx] - label[:, idx]
        pearson = (
            float(np.corrcoef(pred[:, idx], label[:, idx])[0, 1])
            if len(pred) > 1 and np.std(pred[:, idx]) > 0 and np.std(label[:, idx]) > 0
            else float("nan")
        )
        metrics[f"{task}_mae"] = float(np.mean(np.abs(diff)))
        metrics[f"{task}_rmse"] = float(np.sqrt(np.mean(diff**2)))
        metrics[f"{task}_pearson"] = pearson
        metrics[f"{task}_pred_mean"] = float(np.mean(pred[:, idx]))
        metrics[f"{task}_label_mean"] = float(np.mean(label[:, idx]))
    metrics["bp_mae_mean"] = float(np.mean([metrics[f"{task}_mae"] for task in TASKS]))
    return metrics


def evaluate(model, waveforms, labels, masks, subjects, clip_indices, device, batch_size: int):
    model.eval()
    preds = []
    out_labels = []
    outputs = {}
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            end = start + batch_size
            wave = torch.from_numpy(waveforms[start:end]).to(device=device, dtype=torch.float32)
            pred = model(wave).detach().cpu().numpy()
            label = labels[start:end]
            mask = masks[start:end]
            valid = np.all(mask > 0.5, axis=1)
            preds.extend(pred[valid])
            out_labels.extend(label[valid])
            for i in range(len(pred)):
                outputs.setdefault(str(subjects[start + i]), {})[int(clip_indices[start + i])] = {
                    "bp_pred": {task: float(pred[i, j]) for j, task in enumerate(TASKS)},
                    "bp_label": {task: float(label[i, j]) for j, task in enumerate(TASKS)},
                    "bp_mask": {task: float(mask[i, j]) for j, task in enumerate(TASKS)},
                }
    return metric_summary(np.asarray(preds), np.asarray(out_labels)), outputs


def train_method(method: str, train_rows, test_rows, args, device):
    out_dir = Path(args.output_dir) / method.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.output_dir) / "cache"
    train = build_arrays(train_rows, method, args, cache_dir / f"{method.lower()}_trainval.npz")
    test = build_arrays(test_rows, method, args, cache_dir / f"{method.lower()}_test.npz")
    train_wave, train_label, train_mask, *_ = train
    test_wave, test_label, test_mask, test_subjects, test_clip_indices = test

    dataset = TensorDataset(
        torch.from_numpy(train_wave),
        torch.from_numpy(train_label),
        torch.from_numpy(train_mask),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = WaveformBPHead(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    best_epoch = -1
    best_path = out_dir / f"{method.lower()}_bp_adapter_best.pth"
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for wave, label, mask in loader:
            wave = wave.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(wave)
            loss = bp_loss(pred, label, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * wave.shape[0]
            count += wave.shape[0]
        train_metrics, _ = evaluate(model, train_wave, train_label, train_mask, train[3], train[4], device, args.batch_size)
        select = train_metrics["bp_mae_mean"]
        if select < best:
            best = select
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(f"method={method} epoch={epoch} loss={total / max(count, 1):.6f} train_bp_mae_mean={select:.6f} best={best:.6f}", flush=True)

    model.load_state_dict(torch.load(best_path, map_location=device))
    train_metrics, train_outputs = evaluate(model, train_wave, train_label, train_mask, train[3], train[4], device, args.batch_size)
    test_metrics, test_outputs = evaluate(model, test_wave, test_label, test_mask, test_subjects, test_clip_indices, device, args.batch_size)
    train_pickle = out_dir / f"{method.lower()}_bp_adapter_train_outputs.pickle"
    test_pickle = out_dir / f"{method.lower()}_bp_adapter_test_outputs.pickle"
    with train_pickle.open("wb") as handle:
        pickle.dump(train_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with test_pickle.open("wb") as handle:
        pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "method": method,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "train_outputs": str(train_pickle),
        "test_outputs": str(test_pickle),
        "final_train": train_metrics,
        "final_test": test_metrics,
    }
    summary_path = out_dir / f"{method.lower()}_bp_adapter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"method={method} final_test {test_metrics}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(MANIFEST_DIR / f"{BP_PREFIX}_zpu_trainval_novalid.csv"))
    parser.add_argument("--test-csv", default=str(MANIFEST_DIR / f"{BP_PREFIX}_zpu_test.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs/exp/unsupervised_bp_adapters_zpu0406"))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    train_rows = read_manifest(args.train_csv)
    test_rows = read_manifest(args.test_csv)
    summaries = []
    for method in args.methods:
        summaries.append(train_method(method.upper(), train_rows, test_rows, args, device))
    out_csv = Path(args.output_dir) / "unsupervised_bp_adapter_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        fieldnames = ["method", "best_epoch", "sbp_mae", "dbp_mae", "map_mae", "test_outputs", "summary"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            test = summary["final_test"]
            writer.writerow(
                {
                    "method": summary["method"],
                    "best_epoch": summary["best_epoch"],
                    "sbp_mae": test["sbp_mae"],
                    "dbp_mae": test["dbp_mae"],
                    "map_mae": test["map_mae"],
                    "test_outputs": summary["test_outputs"],
                    "summary": str(Path(args.output_dir) / summary["method"].lower() / f"{summary['method'].lower()}_bp_adapter_summary.json"),
                }
            )
    print(f"output_csv={out_csv}")


if __name__ == "__main__":
    main()

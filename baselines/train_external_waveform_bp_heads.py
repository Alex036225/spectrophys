\
"""Train SBP/DBP/MAP heads on top of existing external waveform adapters."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/public_hw/home/cit_yingxinlai/project/nature")
MANIFEST_DIR = Path("/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/JOINT/DataFileLists")
BP_PREFIX = "MultiTask_PPG_PR_HRpleth512_SpO2_BP_72x72_160"
TASKS = ("sbp", "dbp", "map")
DEFAULTS = {"sbp": 120.0, "dbp": 75.0, "map": 90.0}
SCALES = torch.tensor([25.0, 15.0, 20.0])


def load_symbol(path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


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


class BPClipDataset(Dataset):
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
        values = []
        masks = []
        for task in TASKS:
            value = finite_float(row.get(f"{task}_mmhg"))
            mask = 1.0 if row.get(f"{task}_mask", "0") == "1" and value is not None else 0.0
            values.append(DEFAULTS[task] if value is None else value)
            masks.append(mask)
        return (
            torch.from_numpy(video),
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
            row["subject"],
            int(row["clip_index"]),
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


class AdapterWithBP(nn.Module):
    def __init__(self, adapter: str, checkpoint: Path, frames: int):
        super().__init__()
        self.adapter = adapter
        if adapter == "fusionvitals":
            cls = load_symbol(ROOT / "tools/train_fusionvitals_face_adapter.py", "FusionVitalsFaceNet")
            self.base = cls(frames=frames)
        elif adapter == "boe_rppg_spo2":
            cls = load_symbol(ROOT / "tools/train_boe_rppg_spo2_adapter.py", "BoeRppgSpo2Net")
            self.base = cls(frames=frames, window_size=30)
        else:
            raise ValueError(f"Unsupported adapter: {adapter}")
        state = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = self.base.load_state_dict(state, strict=False)
        print(f"loaded_base={checkpoint} missing={len(missing)} unexpected={len(unexpected)}")
        for param in self.base.parameters():
            param.requires_grad = False
        self.bp_head = WaveformBPHead()

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.base(video)
        waveform = outputs[0]
        return waveform, self.bp_head(waveform)


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


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    model.base.eval()
    total_loss = 0.0
    count = 0
    preds = []
    labels = []
    outputs = {}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for video, bp, mask, subjects, clip_indices in loader:
            video = video.to(device=device, dtype=torch.float32, non_blocking=True)
            bp = bp.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            _, pred_bp = model(video)
            loss = bp_loss(pred_bp, bp, mask)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.bp_head.parameters(), 3.0)
                optimizer.step()
            batch = video.shape[0]
            total_loss += float(loss.detach().cpu()) * batch
            count += batch
            pred_cpu = pred_bp.detach().cpu().numpy()
            bp_cpu = bp.detach().cpu().numpy()
            mask_cpu = mask.detach().cpu().numpy()
            for i, subject in enumerate(subjects):
                if np.all(mask_cpu[i] > 0.5):
                    preds.append(pred_cpu[i])
                    labels.append(bp_cpu[i])
                outputs.setdefault(subject, {})[int(clip_indices[i])] = {
                    "bp_pred": {task: float(pred_cpu[i, j]) for j, task in enumerate(TASKS)},
                    "bp_label": {task: float(bp_cpu[i, j]) for j, task in enumerate(TASKS)},
                    "bp_mask": {task: float(mask_cpu[i, j]) for j, task in enumerate(TASKS)},
                }
    metrics = {"loss": total_loss / max(count, 1)}
    if preds:
        metrics.update(metric_summary(np.asarray(preds), np.asarray(labels)))
    return metrics, outputs


def fmt(metrics: dict[str, float]) -> str:
    return " ".join(
        f"{key}={value:.5f}" if isinstance(value, float) and math.isfinite(value) else f"{key}={value}"
        for key, value in sorted(metrics.items())
    )


def default_checkpoint(adapter: str) -> Path:
    if adapter == "fusionvitals":
        return ROOT / "runs/exp/fusionvitals_face/fusionvitals_face_subject19_zpu0406/fusionvitals_face_subject19_zpu0406_best.pth"
    if adapter == "boe_rppg_spo2":
        return ROOT / "runs/exp/boe_rppg_spo2/boe_rppg_spo2_subject19_zpu0406/boe_rppg_spo2_subject19_zpu0406_best.pth"
    raise ValueError(adapter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, choices=["fusionvitals", "boe_rppg_spo2"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--train-csv", default=str(MANIFEST_DIR / f"{BP_PREFIX}_zpu_trainval_novalid.csv"))
    parser.add_argument("--valid-csv", default=str(MANIFEST_DIR / f"{BP_PREFIX}_zpu_trainval_novalid.csv"))
    parser.add_argument("--test-csv", default=str(MANIFEST_DIR / f"{BP_PREFIX}_zpu_test.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs/exp/external_bp_heads_zpu0406"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir) / args.adapter
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.adapter)

    train_rows = read_manifest(args.train_csv)
    valid_rows = read_manifest(args.valid_csv)
    test_rows = read_manifest(args.test_csv)
    print(f"adapter={args.adapter} train_rows={len(train_rows)} valid_rows={len(valid_rows)} test_rows={len(test_rows)}")

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": args.device.startswith("cuda"), "drop_last": False}
    train_loader = DataLoader(BPClipDataset(train_rows, args.frames), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(BPClipDataset(valid_rows, args.frames), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(BPClipDataset(test_rows, args.frames), batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = AdapterWithBP(args.adapter, checkpoint, args.frames).to(device)
    optimizer = torch.optim.AdamW(model.bp_head.parameters(), lr=args.lr, weight_decay=1e-4)

    best = float("inf")
    best_epoch = -1
    best_path = output_dir / f"{args.adapter}_bphead_best.pth"
    for epoch in range(args.epochs):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, device, train=True)
        valid_metrics, _ = run_epoch(model, valid_loader, optimizer, device, train=False)
        select = valid_metrics.get("bp_mae_mean", float("inf"))
        if math.isfinite(select) and select < best:
            best = select
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch} train {fmt(train_metrics)}")
        print(f"epoch={epoch} valid {fmt(valid_metrics)} best_bp_mae_mean={best:.5f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    valid_metrics, valid_outputs = run_epoch(model, valid_loader, optimizer, device, train=False)
    test_metrics, test_outputs = run_epoch(model, test_loader, optimizer, device, train=False)
    valid_pickle = output_dir / f"{args.adapter}_bphead_valid_outputs.pickle"
    test_pickle = output_dir / f"{args.adapter}_bphead_test_outputs.pickle"
    with valid_pickle.open("wb") as handle:
        pickle.dump(valid_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with test_pickle.open("wb") as handle:
        pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "adapter": args.adapter,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "valid_outputs": str(valid_pickle),
        "test_outputs": str(test_pickle),
        "final_valid": valid_metrics,
        "final_test": test_metrics,
    }
    summary_path = output_dir / f"{args.adapter}_bphead_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"best_epoch={best_epoch}")
    print(f"best_checkpoint={best_path}")
    print(f"valid_outputs={valid_pickle}")
    print(f"test_outputs={test_pickle}")
    print(f"summary={summary_path}")
    print(f"final_valid {fmt(valid_metrics)}")
    print(f"final_test {fmt(test_metrics)}")


if __name__ == "__main__":
    main()

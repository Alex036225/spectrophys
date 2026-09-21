\
"""Add and train explicit SBP/DBP/MAP heads for standard rPPG models."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from neural_methods.model.BigSmall import BigSmall
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.MultiPhysNet import PhysNet_padding_Encoder_Decoder_MAX as MultiPhysNet
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.RhythmFormer import RhythmFormer


ROOT = Path("/public_hw/home/cit_yingxinlai/project/nature")
MANIFEST_DIR = Path("/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/JOINT/DataFileLists")
BP_PREFIX = "MultiTask_PPG_PR_HRpleth512_SpO2_BP_72x72_160"
MODEL_FILE_STEMS = {
    "physnet": "BENCH_PHYSNET_SOURCEZPU_ZPUTEST0406",
    "deepphys": "BENCH_DEEPPHYS_SOURCEZPU_ZPUTEST0406",
    "physformer": "BENCH_PHYSFORMER_SOURCEZPU_ZPUTEST0406",
    "rhythmformer": "BENCH_RHYTHMFORMER_SOURCEZPU_ZPUTEST0406",
    "bigsmall": "BENCH_BIGSMALL_SOURCEZPU_ZPUTEST0406",
    "efficientphys": "BENCH_EFFICIENTPHYS_SOURCEZPU_ZPUTEST0406",
    "multiphysnet": "BENCH_MULTIPHYSNET_SOURCEZPU_ZPUTEST0406",
}
BP_TASKS = ("sbp", "dbp", "map")


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
        subject, clip_idx = parse_clip(row["input_files"])
        row.setdefault("subject", subject)
        row.setdefault("clip_index", str(clip_idx))
        if not row.get("ppg_label_file"):
            row["ppg_label_file"] = row["input_files"].replace("input", "label")
        if not row.get("label_polarity"):
            row["label_polarity"] = "1.0"
        if not row.get("ppg_mask"):
            row["ppg_mask"] = "1"
        for task in BP_TASKS:
            row.setdefault(f"{task}_mask", "0")
    return rows


class CachedBPClipDataset(Dataset):
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
        video_tchw = np.transpose(video, (0, 3, 1, 2))

        ppg_mask = 1.0 if row.get("ppg_mask", "1") == "1" else 0.0
        if ppg_mask > 0.0 and Path(row["ppg_label_file"]).exists():
            ppg = np.load(row["ppg_label_file"]).astype(np.float32)[: self.frames]
            ppg = ppg * np.float32(float(row.get("label_polarity", "1.0") or 1.0))
        else:
            ppg = np.zeros((self.frames,), dtype=np.float32)
            ppg_mask = 0.0

        bp_values = []
        bp_masks = []
        defaults = {"sbp": 120.0, "dbp": 75.0, "map": 90.0}
        for task in BP_TASKS:
            value = finite_float(row.get(f"{task}_mmhg"))
            mask = 1.0 if row.get(f"{task}_mask", "0") == "1" and value is not None else 0.0
            bp_values.append(defaults[task] if value is None else value)
            bp_masks.append(mask)

        return (
            torch.from_numpy(video_tchw),
            torch.from_numpy(ppg),
            torch.tensor(ppg_mask, dtype=torch.float32),
            torch.tensor(bp_values, dtype=torch.float32),
            torch.tensor(bp_masks, dtype=torch.float32),
            row["subject"],
            int(row["clip_index"]),
        )


def build_sampler(rows: list[dict[str, str]], bp_weight: float) -> WeightedRandomSampler:
    weights = np.ones((len(rows),), dtype=np.float64)
    for idx, row in enumerate(rows):
        if any(row.get(f"{task}_mask", "0") == "1" for task in BP_TASKS):
            weights[idx] *= float(bp_weight)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(rows), replacement=True)


def normalize_signal(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True, unbiased=False) + eps)


def pad_1d_last(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() >= target_len:
        return x[:target_len]
    if x.numel() == 0:
        return torch.zeros((target_len,), dtype=x.dtype, device=x.device)
    return torch.cat((x, x[-1:].repeat(target_len - x.numel())), dim=0)


def neg_pearson_loss(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = normalize_signal(pred)
    label = normalize_signal(label)
    per_sample = 1.0 - torch.mean(pred * label, dim=1)
    return (per_sample * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def bp_loss(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return pred.sum() * 0.0
    scale = pred.new_tensor([25.0, 15.0, 20.0])
    return F.smooth_l1_loss((pred[valid] - label[valid]) / scale.expand_as(pred)[valid], torch.zeros_like(pred[valid]))


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
        x = normalize_signal(waveform).unsqueeze(1)
        return self.net(x) * self.span + self.low


def ensure_deepphys_channels(video_tchw: torch.Tensor) -> torch.Tensor:
    raw = video_tchw
    diff = torch.zeros_like(raw)
    diff[:, 1:] = (raw[:, 1:] - raw[:, :-1]) / (raw[:, 1:] + raw[:, :-1] + 1e-7)
    std = diff.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1, 1)
    diff = torch.nan_to_num(diff / torch.clamp(std, min=1e-6), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat((diff, raw), dim=2)


class StandardModelWithBP(nn.Module):
    def __init__(self, model_name: str, frames: int = 160):
        super().__init__()
        self.model_name = model_name.lower()
        self.frames = int(frames)
        if self.model_name == "physnet":
            self.base = PhysNet_padding_Encoder_Decoder_MAX(frames=frames)
        elif self.model_name == "deepphys":
            self.base = DeepPhys(img_size=72)
        elif self.model_name == "efficientphys":
            self.base = EfficientPhys(frame_depth=10, img_size=72)
        elif self.model_name == "physformer":
            self.base = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(frames, 72, 72),
                patches=(4, 4, 4),
                dim=96,
                ff_dim=144,
                num_heads=4,
                num_layers=12,
                dropout_rate=0.0,
                theta=0.7,
            )
        elif self.model_name == "rhythmformer":
            self.base = RhythmFormer()
        elif self.model_name == "bigsmall":
            self.base = BigSmall(n_segment=3)
        elif self.model_name == "multiphysnet":
            self.base = MultiPhysNet(frames=frames)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        self.bp_head = WaveformBPHead()

    def forward_waveform(self, video_tchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video_tchw.shape
        if self.model_name == "physnet":
            return self.base(video_tchw.permute(0, 2, 1, 3, 4))[0]
        if self.model_name == "multiphysnet":
            return self.base(video_tchw.permute(0, 2, 1, 3, 4))[0]
        if self.model_name == "physformer":
            return self.base(video_tchw.permute(0, 2, 1, 3, 4), 2.0)[0]
        if self.model_name == "rhythmformer":
            return self.base(video_tchw)
        if self.model_name == "deepphys":
            data = ensure_deepphys_channels(video_tchw).reshape(b * t, 6, h, w)
            return self.base(data).reshape(b, t)
        if self.model_name == "efficientphys":
            data = video_tchw.reshape(b * t, c, h, w)
            usable = (b * t) // 10 * 10
            data = data[:usable]
            last_frame = data[-1:].repeat(1, 1, 1, 1)
            pred = self.base(torch.cat((data, last_frame), dim=0)).reshape(-1)
            return pad_1d_last(pred, b * t).reshape(b, t)
        if self.model_name == "bigsmall":
            flat = video_tchw.reshape(b * t, c, h, w)
            big = F.interpolate(flat, size=(144, 144), mode="bilinear", align_corners=False)
            small = F.interpolate(flat, size=(9, 9), mode="bilinear", align_corners=False)
            usable = (b * t) // 3 * 3
            _, bvp, _ = self.base((big[:usable], small[:usable]))
            pred = bvp.reshape(-1)
            return pad_1d_last(pred, b * t).reshape(b, t)
        raise AssertionError(self.model_name)

    def forward(self, video_tchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self.forward_waveform(video_tchw)
        return waveform, self.bp_head(waveform)


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def find_base_checkpoint(model_name: str) -> Path | None:
    stem = MODEL_FILE_STEMS[model_name]
    pattern = f"runs/exp/benchmark_standard_models_zpu0406/{model_name}/**/PreTrainedModels/{stem}_Epoch*.pth"
    candidates = list(ROOT.glob(pattern))
    if not candidates:
        return None

    def epoch_num(path: Path) -> int:
        match = re.search(r"_Epoch(\d+)\.pth$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=epoch_num)


def maybe_load_base(model: StandardModelWithBP, model_name: str) -> None:
    path = find_base_checkpoint(model_name)
    if path is None or not path.exists():
        print(f"init_base_checkpoint=missing:{path}")
        return
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.base.load_state_dict(clean_state_dict(state), strict=False)
    print(f"init_base_checkpoint={path}")
    print(f"init_missing={len(missing)} init_unexpected={len(unexpected)}")


def bp_metric_summary(preds: list[list[float]], labels: list[list[float]]) -> dict[str, float]:
    pred = np.asarray(preds, dtype=np.float64)
    label = np.asarray(labels, dtype=np.float64)
    metrics: dict[str, float] = {"bp_count": int(len(pred))}
    for idx, task in enumerate(BP_TASKS):
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
    metrics["bp_mae_mean"] = float(np.mean([metrics[f"{task}_mae"] for task in BP_TASKS]))
    return metrics


def run_epoch(model, loader, optimizer, args, device, train: bool):
    model.train(train)
    sums = {"loss": 0.0, "ppg_loss": 0.0, "bp_loss": 0.0}
    count = 0
    preds: list[list[float]] = []
    labels: list[list[float]] = []
    outputs = {}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for video, ppg, ppg_mask, bp, bp_mask, subjects, clip_indices in loader:
            video = video.to(device=device, dtype=torch.float32, non_blocking=True)
            ppg = ppg.to(device=device, dtype=torch.float32, non_blocking=True)
            ppg_mask = ppg_mask.to(device=device, dtype=torch.float32, non_blocking=True)
            bp = bp.to(device=device, dtype=torch.float32, non_blocking=True)
            bp_mask = bp_mask.to(device=device, dtype=torch.float32, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred_ppg, pred_bp = model(video)
            loss_ppg = neg_pearson_loss(pred_ppg, ppg, ppg_mask)
            loss_bp = bp_loss(pred_bp, bp, bp_mask)
            total = args.ppg_weight * loss_ppg + args.bp_weight * loss_bp
            if train:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            batch = video.shape[0]
            count += batch
            sums["loss"] += float(total.detach().cpu()) * batch
            sums["ppg_loss"] += float(loss_ppg.detach().cpu()) * batch
            sums["bp_loss"] += float(loss_bp.detach().cpu()) * batch

            pred_ppg_cpu = pred_ppg.detach().cpu().numpy().astype(np.float32, copy=False)
            ppg_cpu = ppg.detach().cpu().numpy().astype(np.float32, copy=False)
            pred_bp_cpu = pred_bp.detach().cpu().numpy()
            bp_cpu = bp.detach().cpu().numpy()
            bp_mask_cpu = bp_mask.detach().cpu().numpy()
            for i, subject in enumerate(subjects):
                clip_idx = int(clip_indices[i])
                if np.all(bp_mask_cpu[i] > 0.5):
                    preds.append(pred_bp_cpu[i].astype(float).tolist())
                    labels.append(bp_cpu[i].astype(float).tolist())
                outputs.setdefault(subject, {})[clip_idx] = {
                    "waveform_pred": pred_ppg_cpu[i],
                    "waveform_label": ppg_cpu[i],
                    "bp_pred": {task: float(pred_bp_cpu[i, j]) for j, task in enumerate(BP_TASKS)},
                    "bp_label": {task: float(bp_cpu[i, j]) for j, task in enumerate(BP_TASKS)},
                    "bp_mask": {task: float(bp_mask_cpu[i, j]) for j, task in enumerate(BP_TASKS)},
                }
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    if preds:
        metrics.update(bp_metric_summary(preds, labels))
    else:
        metrics["bp_count"] = 0
        for task in BP_TASKS:
            for suffix in ("mae", "rmse", "pearson", "pred_mean", "label_mean"):
                metrics[f"{task}_{suffix}"] = float("nan")
        metrics["bp_mae_mean"] = float("nan")
    return metrics, outputs


def fmt(metrics: dict[str, float]) -> str:
    return " ".join(
        f"{k}={v:.5f}" if isinstance(v, float) and math.isfinite(v) else f"{k}={v}"
        for k, v in sorted(metrics.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_FILE_STEMS))
    parser.add_argument("--train-csv", default=str(MANIFEST_DIR / f"{BP_PREFIX}_zpu_trainval_novalid.csv"))
    parser.add_argument("--valid-csv", default=str(MANIFEST_DIR / f"{BP_PREFIX}_zpu_trainval_novalid.csv"))
    parser.add_argument("--test-csv", default=str(MANIFEST_DIR / f"{BP_PREFIX}_zpu_test.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs/exp/benchmark_bp_heads_zpu0406"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ppg-weight", type=float, default=0.2)
    parser.add_argument("--bp-weight", type=float, default=1.0)
    parser.add_argument("--bp-positive-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest(args.train_csv)
    valid_rows = read_manifest(args.valid_csv)
    test_rows = read_manifest(args.test_csv)
    print(
        f"model={args.model} train_rows={len(train_rows)} train_bp={sum(r.get('sbp_mask') == '1' for r in train_rows)} "
        f"valid_rows={len(valid_rows)} valid_bp={sum(r.get('sbp_mask') == '1' for r in valid_rows)} "
        f"test_rows={len(test_rows)} test_bp={sum(r.get('sbp_mask') == '1' for r in test_rows)}"
    )

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": args.device.startswith("cuda"), "drop_last": False}
    train_loader = DataLoader(
        CachedBPClipDataset(train_rows, frames=args.frames),
        batch_size=args.batch_size,
        sampler=build_sampler(train_rows, args.bp_positive_weight),
        **loader_kwargs,
    )
    valid_loader = DataLoader(CachedBPClipDataset(valid_rows, frames=args.frames), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(CachedBPClipDataset(test_rows, frames=args.frames), batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = StandardModelWithBP(args.model, frames=args.frames)
    maybe_load_base(model, args.model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_epoch = -1
    best_path = output_dir / f"{args.model}_bphead_best.pth"
    for epoch in range(args.epochs):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, args, device, train=True)
        valid_metrics, _ = run_epoch(model, valid_loader, optimizer, args, device, train=False)
        select = valid_metrics.get("bp_mae_mean", float("inf"))
        if math.isfinite(select) and select < best_metric:
            best_metric = select
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch} train {fmt(train_metrics)}")
        print(f"epoch={epoch} valid {fmt(valid_metrics)} best_bp_mae_mean={best_metric:.5f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    valid_metrics, valid_outputs = run_epoch(model, valid_loader, optimizer, args, device, train=False)
    test_metrics, test_outputs = run_epoch(model, test_loader, optimizer, args, device, train=False)
    valid_pickle = output_dir / f"{args.model}_bphead_valid_outputs.pickle"
    test_pickle = output_dir / f"{args.model}_bphead_test_outputs.pickle"
    with valid_pickle.open("wb") as handle:
        pickle.dump(valid_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with test_pickle.open("wb") as handle:
        pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "valid_outputs": str(valid_pickle),
        "test_outputs": str(test_pickle),
        "final_valid": valid_metrics,
        "final_test": test_metrics,
    }
    with (output_dir / f"{args.model}_bphead_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"best_epoch={best_epoch}")
    print(f"best_checkpoint={best_path}")
    print(f"valid_outputs={valid_pickle}")
    print(f"test_outputs={test_pickle}")
    print(f"final_valid {fmt(valid_metrics)}")
    print(f"final_test {fmt(test_metrics)}")


if __name__ == "__main__":
    main()

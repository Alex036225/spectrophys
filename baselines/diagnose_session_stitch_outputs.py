\
"""Diagnose why clip-level rPPG outputs fail or improve after session stitching."""

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.post_process import calculate_metric_per_video              

CLIP_RE = re.compile(r"^(?P<session>.+)_clip(?P<clip>\d+)$")


def _np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _z(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    std = np.std(x)
    if std < 1e-8:
        return x * 0.0
    return (x - np.mean(x)) / std


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _flatten(sort_dict):
    parts = []
    for k in sorted(sort_dict.keys(), key=lambda item: int(item)):
        parts.append(_np(sort_dict[k]))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def group(payload):
    sessions = defaultdict(list)
    labels = payload["labels"]
    for key in sorted(payload["predictions"].keys()):
        m = CLIP_RE.match(str(key))
        if m:
            sess = m.group("session")
            clip_idx = int(m.group("clip"))
        else:
            sess = str(key)
            clip_idx = 0
        sessions[sess].append((clip_idx, _flatten(payload["predictions"][key]), _flatten(labels[key]), str(key)))
    out = {}
    for sess, items in sessions.items():
        out[sess] = sorted(items, key=lambda item: item[0])
    return out


def hr(pred, label, fs, label_type):
    diff = label_type == "DiffNormalized"
    gt, pr, snr, macc = calculate_metric_per_video(
        pred, label, fs=fs, diff_flag=diff, hr_method="FFT", compute_snr=True, compute_macc=True
    )
    return float(gt), float(pr), float(snr), float(macc)


def corr(a, b):
    a = _z(a)
    b = _z(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def best_lag_corr(pred, label, max_lag):
    best = (-1e9, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            p = pred[-lag:]
            l = label[: len(p)]
        elif lag > 0:
            l = label[lag:]
            p = pred[: len(l)]
        else:
            p = pred
            l = label
        n = min(len(p), len(l))
        if n < 16:
            continue
        c = corr(p[:n], l[:n])
        if not math.isnan(c) and c > best[0]:
            best = (c, lag)
    return float(best[0]), int(best[1])


def abs_err(a, b):
    return float(abs(float(a) - float(b)))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "se": float("nan")}
    return {"mean": float(np.mean(values)), "se": float(np.std(values) / math.sqrt(len(values)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--fs", type=float, default=30)
    ap.add_argument("--label-type", default=None)
    ap.add_argument("--max-lag", type=int, default=30)
    args = ap.parse_args()

    payload = _load(args.pickle)
    label_type = args.label_type or payload.get("label_type", "Raw")
    sessions = group(payload)
    rows = []

    for sess, clips in sorted(sessions.items()):
        pred_clips = [c[1] for c in clips]
        label_clips = [c[2] for c in clips]
        pred_concat = np.concatenate(pred_clips)
        label_concat = np.concatenate(label_clips)
        pred_concat_zclip = np.concatenate([_z(x) for x in pred_clips])
        pred_concat_flip_oracle = []
        clip_pred_hrs = []
        clip_label_hrs = []
        clip_corrs = []
        clip_best_corrs = []
        clip_best_lags = []
        boundary_jumps = []

        for i, (p, l) in enumerate(zip(pred_clips, label_clips)):
            gt_c, pr_c, _, _ = hr(p, l, args.fs, label_type)
            gt_cf, pr_cf, _, _ = hr(-p, l, args.fs, label_type)
            if abs_err(pr_cf, gt_cf) < abs_err(pr_c, gt_c):
                pred_concat_flip_oracle.append(-p)
            else:
                pred_concat_flip_oracle.append(p)
            clip_pred_hrs.append(pr_c)
            clip_label_hrs.append(gt_c)
            clip_corrs.append(corr(p, l))
            bc, bl = best_lag_corr(p, l, args.max_lag)
            clip_best_corrs.append(bc)
            clip_best_lags.append(bl)
            if i + 1 < len(pred_clips):
                boundary_jumps.append(float(abs(_z(pred_clips[i])[-1] - _z(pred_clips[i + 1])[0])))

        pred_concat_flip_oracle = np.concatenate(pred_concat_flip_oracle)
        gt, pr, snr, macc = hr(pred_concat, label_concat, args.fs, label_type)
        gt_z, pr_z, _, _ = hr(pred_concat_zclip, label_concat, args.fs, label_type)
        gt_flip, pr_flip, _, _ = hr(-pred_concat, label_concat, args.fs, label_type)
        gt_oracle, pr_oracle, _, _ = hr(pred_concat_flip_oracle, label_concat, args.fs, label_type)
        best_corr, best_lag = best_lag_corr(pred_concat, label_concat, args.max_lag)

        rows.append({
            "session": sess,
            "num_clips": len(clips),
            "num_frames": int(len(pred_concat)),
            "label_hr_concat": gt,
            "pred_hr_concat": pr,
            "abs_error_concat": abs_err(pr, gt),
            "pred_hr_concat_zclip": pr_z,
            "abs_error_concat_zclip": abs_err(pr_z, gt_z),
            "pred_hr_concat_flip": pr_flip,
            "abs_error_concat_flip": abs_err(pr_flip, gt_flip),
            "pred_hr_concat_clip_oracle_flip": pr_oracle,
            "abs_error_concat_clip_oracle_flip": abs_err(pr_oracle, gt_oracle),
            "clip_hr_mean": float(np.mean(clip_pred_hrs)),
            "clip_hr_median": float(np.median(clip_pred_hrs)),
            "label_clip_hr_mean": float(np.mean(clip_label_hrs)),
            "abs_error_clip_hr_mean": abs_err(np.mean(clip_pred_hrs), np.mean(clip_label_hrs)),
            "abs_error_clip_hr_median_vs_label_mean": abs_err(np.median(clip_pred_hrs), np.mean(clip_label_hrs)),
            "wave_corr_concat": corr(pred_concat, label_concat),
            "wave_best_lag_corr_concat": best_corr,
            "wave_best_lag_frames_concat": best_lag,
            "clip_corr_mean": float(np.nanmean(clip_corrs)),
            "clip_best_lag_corr_mean": float(np.nanmean(clip_best_corrs)),
            "clip_best_lag_abs_mean": float(np.nanmean(np.abs(clip_best_lags))),
            "boundary_jump_mean": float(np.mean(boundary_jumps)) if boundary_jumps else 0.0,
            "snr_concat": snr,
            "macc_concat": macc,
        })

    summary = {
        "input_pickle": os.path.abspath(args.pickle),
        "num_sessions": len(rows),
        "label_type": label_type,
        "fs": args.fs,
    }
    for field in [
        "abs_error_concat",
        "abs_error_concat_zclip",
        "abs_error_concat_flip",
        "abs_error_concat_clip_oracle_flip",
        "abs_error_clip_hr_mean",
        "abs_error_clip_hr_median_vs_label_mean",
        "wave_corr_concat",
        "wave_best_lag_corr_concat",
        "clip_corr_mean",
        "clip_best_lag_corr_mean",
        "clip_best_lag_abs_mean",
        "boundary_jump_mean",
        "snr_concat",
        "macc_concat",
    ]:
        s = summarize([r[field] for r in rows if not math.isnan(float(r[field]))])
        summary[field + "_mean"] = s["mean"]
        summary[field + "_se"] = s["se"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    csv_path = args.out_prefix + "_diagnostic.csv"
    json_path = args.out_prefix + "_diagnostic_summary.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Saved CSV:", csv_path)
    print("Saved JSON:", json_path)


if __name__ == "__main__":
    main()

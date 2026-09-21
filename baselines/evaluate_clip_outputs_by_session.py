\
"""Evaluate clip-level rPPG outputs after stitching clips by session."""

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


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _flatten_inner(sort_dict):
    parts = []
    for sort_key in sorted(sort_dict.keys(), key=lambda item: int(item)):
        parts.append(_to_numpy(sort_dict[sort_key]))
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _load_outputs(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if "predictions" not in payload or "labels" not in payload:
        raise ValueError(f"{path} does not contain predictions/labels")
    return payload


def _group_by_session(predictions, labels):
    grouped_pred = defaultdict(list)
    grouped_label = defaultdict(list)
    ungrouped = []

    for clip_key in sorted(predictions.keys()):
        match = CLIP_RE.match(str(clip_key))
        if not match:
            ungrouped.append(str(clip_key))
            session_key = str(clip_key)
            clip_idx = 0
        else:
            session_key = match.group("session")
            clip_idx = int(match.group("clip"))

        if clip_key not in labels:
            raise KeyError(f"Missing label for prediction key {clip_key}")
        grouped_pred[session_key].append((clip_idx, _flatten_inner(predictions[clip_key])))
        grouped_label[session_key].append((clip_idx, _flatten_inner(labels[clip_key])))

    sessions = {}
    for session_key in sorted(grouped_pred.keys()):
        pred_items = sorted(grouped_pred[session_key], key=lambda item: item[0])
        label_items = sorted(grouped_label[session_key], key=lambda item: item[0])
        pred = np.concatenate([item[1] for item in pred_items], axis=0)
        label = np.concatenate([item[1] for item in label_items], axis=0)
        length = min(len(pred), len(label))
        sessions[session_key] = {
            "prediction": pred[:length],
            "label": label[:length],
            "num_clips": len(pred_items),
            "num_frames": length,
        }
    return sessions, ungrouped


def _diff_flag(label_type):
    if label_type in ("Raw", "Standardized"):
        return False
    if label_type == "DiffNormalized":
        return True
    raise ValueError(f"Unsupported label type: {label_type}")


def _safe_corr(pred_hr, label_hr):
    if len(pred_hr) < 2:
        return float("nan")
    if np.std(pred_hr) == 0 or np.std(label_hr) == 0:
        return float("nan")
    return float(np.corrcoef(pred_hr, label_hr)[0, 1])


def _standard_error(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    return float(np.std(values) / math.sqrt(len(values)))


def evaluate_sessions(sessions, fs, label_type, min_frames, compute_snr, compute_macc):
    rows = []
    for session_key, item in sessions.items():
        pred = item["prediction"]
        label = item["label"]
        if len(pred) < min_frames:
            continue
        hr_label, hr_pred, snr, macc = calculate_metric_per_video(
            pred,
            label,
            fs=fs,
            diff_flag=_diff_flag(label_type),
            hr_method="FFT",
            compute_snr=compute_snr,
            compute_macc=compute_macc,
        )
        rows.append({
            "session": session_key,
            "num_clips": item["num_clips"],
            "num_frames": item["num_frames"],
            "label_hr": float(hr_label),
            "pred_hr": float(hr_pred),
            "abs_error": float(abs(hr_pred - hr_label)),
            "squared_error": float((hr_pred - hr_label) ** 2),
            "ape": float(abs((hr_pred - hr_label) / hr_label) * 100) if hr_label != 0 else float("nan"),
            "snr": None if snr is None else float(snr),
            "macc": None if macc is None else float(macc),
        })
    return rows


def summarize(rows):
    pred_hr = np.asarray([row["pred_hr"] for row in rows], dtype=np.float64)
    label_hr = np.asarray([row["label_hr"] for row in rows], dtype=np.float64)
    abs_errors = np.asarray([row["abs_error"] for row in rows], dtype=np.float64)
    squared_errors = np.asarray([row["squared_error"] for row in rows], dtype=np.float64)
    ape = np.asarray([row["ape"] for row in rows], dtype=np.float64)
    snr = np.asarray([row["snr"] for row in rows if row["snr"] is not None], dtype=np.float64)
    macc = np.asarray([row["macc"] for row in rows if row["macc"] is not None], dtype=np.float64)

    return {
        "num_sessions": int(len(rows)),
        "mae": float(np.mean(abs_errors)),
        "mae_se": _standard_error(abs_errors),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "rmse_se": float(np.sqrt(np.std(squared_errors) / math.sqrt(len(squared_errors)))) if len(squared_errors) else float("nan"),
        "mape": float(np.nanmean(ape)),
        "mape_se": _standard_error(ape[~np.isnan(ape)]),
        "pearson": _safe_corr(pred_hr, label_hr),
        "snr": float(np.mean(snr)) if len(snr) else float("nan"),
        "snr_se": _standard_error(snr) if len(snr) else float("nan"),
        "macc": float(np.mean(macc)) if len(macc) else float("nan"),
        "macc_se": _standard_error(macc) if len(macc) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", required=True, help="Saved rPPG toolbox outputs pickle.")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for CSV/JSON summary.")
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--label-type", default=None, choices=["Raw", "Standardized", "DiffNormalized"])
    parser.add_argument("--min-frames", type=int, default=160)
    parser.add_argument("--no-snr", action="store_true")
    parser.add_argument("--no-macc", action="store_true")
    args = parser.parse_args()

    payload = _load_outputs(args.pickle)
    label_type = args.label_type or payload.get("label_type", "Raw")
    fs = float(args.fs or payload.get("fs", 30))

    sessions, ungrouped = _group_by_session(payload["predictions"], payload["labels"])
    rows = evaluate_sessions(
        sessions,
        fs=fs,
        label_type=label_type,
        min_frames=args.min_frames,
        compute_snr=not args.no_snr,
        compute_macc=not args.no_macc,
    )
    summary = summarize(rows)
    summary.update({
        "input_pickle": os.path.abspath(args.pickle),
        "fs": fs,
        "label_type": label_type,
        "num_input_keys": len(payload["predictions"]),
        "num_grouped_sessions_total": len(sessions),
        "num_ungrouped_keys": len(ungrouped),
        "min_frames": args.min_frames,
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    csv_path = args.out_prefix + "_session_metrics.csv"
    json_path = args.out_prefix + "_summary.json"

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["session"])
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Grouped {summary['num_input_keys']} clip keys into {summary['num_grouped_sessions_total']} sessions")
    print(f"Evaluated sessions: {summary['num_sessions']} (min_frames={args.min_frames})")
    print(f"FFT MAE (session): {summary['mae']} +/- {summary['mae_se']}")
    print(f"FFT RMSE (session): {summary['rmse']} +/- {summary['rmse_se']}")
    print(f"FFT MAPE (session): {summary['mape']} +/- {summary['mape_se']}")
    print(f"FFT Pearson (session): {summary['pearson']}")
    print(f"FFT SNR (session): {summary['snr']} +/- {summary['snr_se']} dB")
    print(f"FFT MACC (session): {summary['macc']} +/- {summary['macc_se']}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()

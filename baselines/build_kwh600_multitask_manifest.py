\
"""Build SpectroPhys-MT manifests for KWH 600-frame cached data."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def label_path_for_input(input_file: str) -> str:
    path = Path(input_file)
    match = re.match(r"(.+)_input(\d+)\.npy$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse KWH input name: {input_file}")
    return str(path.with_name(f"{match.group(1)}_label{match.group(2)}.npy"))


def subject_and_clip(input_file: str) -> tuple[str, int]:
    path = Path(input_file)
    match = re.match(r"(s\d+)_.*_([0-9]+)_input(\d+)\.npy$", path.name)
    if match is None:
        return path.stem, 0
    return match.group(1), int(match.group(2)) * 2 + int(match.group(3))


def build_one(input_csv: Path, output_csv: Path, ppg_dir: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ppg_dir.mkdir(parents=True, exist_ok=True)
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
        "sbp_mmhg",
        "sbp_mask",
        "dbp_mmhg",
        "dbp_mask",
        "map_mmhg",
        "map_mask",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            input_file = row["input_files"]
            label_file = Path(label_path_for_input(input_file))
            subject, clip_index = subject_and_clip(input_file)
            ppg_mask = label_file.exists()
            hr_value = np.nan
            rr_value = np.nan
            ppg_out = ppg_dir / f"{label_file.stem}_ppgcol0.npy"
            if ppg_mask:
                label = np.load(label_file)
                if label.ndim == 2:
                    ppg = label[:, 0].astype(np.float32)
                    hr_value = float(np.nanmedian(label[:, 4]))
                    rr_value = float(np.nanmedian(label[:, 5]))
                else:
                    ppg = label.astype(np.float32)
                if not ppg_out.exists():
                    np.save(ppg_out, ppg)
            hr_ok = np.isfinite(hr_value) and hr_value > 0
            rr_ok = np.isfinite(rr_value) and rr_value > 0
            writer.writerow(
                {
                    "input_files": input_file,
                    "subject": subject,
                    "clip_index": str(clip_index),
                    "ppg_label_file": str(ppg_out),
                    "label_polarity": "1.0",
                    "ppg_mask": "1" if ppg_mask else "0",
                    "dataset": "KWH",
                    "pr_bpm": f"{hr_value:.8f}" if hr_ok else "",
                    "pr_mask": "1" if hr_ok else "0",
                    "hr_bpm": f"{hr_value:.8f}" if hr_ok else "",
                    "hr_mask": "1" if hr_ok else "0",
                    "rr_bpm": f"{rr_value:.8f}" if rr_ok else "",
                    "rr_mask": "1" if rr_ok else "0",
                    "spo2_pct": "",
                    "spo2_mask": "0",
                    "sbp_mmhg": "",
                    "sbp_mask": "0",
                    "dbp_mmhg": "",
                    "dbp_mask": "0",
                    "map_mmhg": "",
                    "map_mask": "0",
                }
            )


def subtract_csv(a_csv: Path, b_csv: Path, output_csv: Path) -> None:
    with a_csv.open(newline="") as handle:
        a_rows = list(csv.DictReader(handle))
        fields = list(a_rows[0].keys()) if a_rows else ["input_files"]
    with b_csv.open(newline="") as handle:
        b_paths = {row["input_files"] for row in csv.DictReader(handle)}
    rows = [row for row in a_rows if row["input_files"] not in b_paths]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="/public_hw/share/cit_ztyu/zhaobo/KWH-rPPG_RAW_160_72x72_Y5F/DataFileLists")
    parser.add_argument("--output-dir", default="/public_hw/share/cit_ztyu/zhaobo/KWH-rPPG_RAW_160_72x72_Y5F/DataFileLists_MT600")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    ppg_dir = output_dir / "ppg_col0"
    raw_train = source_dir / "KWH-rPPG_RAW_160_72x72_Y5F_0.0_0.7.csv"
    raw_train_valid = source_dir / "KWH-rPPG_RAW_160_72x72_Y5F_0.0_0.8.csv"
    raw_valid = output_dir / "KWH-rPPG_RAW_160_72x72_Y5F_0.7_0.8.csv"
    raw_test = source_dir / "KWH-rPPG_RAW_160_72x72_Y5F_0.8_1.0.csv"
    raw_all = source_dir / "KWH-rPPG_RAW_160_72x72_Y5F_0.0_1.0.csv"
    subtract_csv(raw_train_valid, raw_train, raw_valid)
    specs = {
        "train_0.0_0.7": raw_train,
        "valid_0.7_0.8": raw_valid,
        "test_0.8_1.0": raw_test,
        "all_0.0_1.0": raw_all,
    }
    for name, csv_path in specs.items():
        out = output_dir / f"KWH_MT600_{name}.csv"
        build_one(csv_path, out, ppg_dir)
        with out.open(newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        print(f"{name}: {out} rows={count}")


if __name__ == "__main__":
    main()

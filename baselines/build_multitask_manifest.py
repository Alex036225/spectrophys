\
"""Build masked multi-task clip manifests for PhaseNet.

The source rPPG caches provide PPG waveform labels. PR/HR labels for these
clips are derived from the label waveform spectrum. ZPU/ZPH clips can additionally
receive monitor HR/PR/RR/SpO2 labels parsed from Mindray HL7 logs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_SOURCE_CACHES = {
    "UBFC": "/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/UBFC/UBFC-rPPG_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse",
    "PURE": "/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/PURE/PURE_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse",
    "BUAA": "/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/BUAA/BUAA_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse",
    "MMPD": "/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/MMPD/MMPD_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse",
}


FIELDNAMES = [
    "input_files",
    "ppg_label_file",
    "dataset",
    "subject",
    "clip_index",
    "label_polarity",
    "ppg_mask",
    "pr_bpm",
    "pr_mask",
    "pr_source",
    "hr_bpm",
    "hr_mask",
    "hr_source",
    "rr_bpm",
    "rr_mask",
    "rr_source",
    "spo2_pct",
    "spo2_mask",
    "spo2_source",
    "sbp_mmhg",
    "sbp_mask",
    "sbp_source",
    "dbp_mmhg",
    "dbp_mask",
    "dbp_source",
    "map_mmhg",
    "map_mask",
    "map_source",
]


def parse_clip(path: Path) -> tuple[str, int]:
    match = re.match(r"(.+)_input(\d+)\.npy$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse cached clip name: {path}")
    return match.group(1), int(match.group(2))


def subject_sort_key(subject: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", subject)
    if match:
        return int(match.group(1)), subject
    return 10**9, subject


def finite_float(value: str | float | int | None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def fft_bpm(signal: np.ndarray, fs: float, low_bpm: float, high_bpm: float, nfft: int = 0) -> float:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 16:
        return float("nan")
    values = values - np.nanmean(values)
    std = np.nanstd(values)
    if not math.isfinite(std) or std < 1e-8:
        return float("nan")
    values = values / std
    n_fft = max(int(nfft), values.size) if nfft else values.size
    window = np.hanning(values.size)
    spectrum = np.fft.rfft(values * window, n=n_fft)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mask = (freqs * 60.0 >= low_bpm) & (freqs * 60.0 <= high_bpm)
    if not np.any(mask):
        return float("nan")
    band_freqs = freqs[mask]
    band_power = power[mask]
    return float(band_freqs[int(np.argmax(band_power))] * 60.0)


def scalar_cell(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6g}"


def mask_cell(value: float | None) -> str:
    return "1" if value is not None and math.isfinite(float(value)) else "0"


def build_source_rows(dataset: str, cache_dir: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = []
    for input_path in sorted(cache_dir.glob("*_input*.npy")):
        label_path = Path(str(input_path).replace("input", "label"))
        if not label_path.exists():
            continue
        subject, clip_index = parse_clip(input_path)
        ppg = np.load(label_path)
        bpm = fft_bpm(ppg, args.fs, args.low_bpm, args.high_bpm, args.fft_nfft)
        bpm_value = bpm if math.isfinite(bpm) else None
        rows.append(
            {
                "input_files": str(input_path),
                "ppg_label_file": str(label_path),
                "dataset": dataset,
                "subject": subject,
                "clip_index": str(clip_index),
                "label_polarity": "1.0",
                "ppg_mask": "1",
                "pr_bpm": scalar_cell(bpm_value),
                "pr_mask": mask_cell(bpm_value),
                "pr_source": "ppg_fft" if bpm_value is not None else "",
                "hr_bpm": scalar_cell(bpm_value),
                "hr_mask": mask_cell(bpm_value),
                "hr_source": "ppg_fft" if bpm_value is not None else "",
                "rr_bpm": "",
                "rr_mask": "0",
                "rr_source": "",
                "spo2_pct": "",
                "spo2_mask": "0",
                "spo2_source": "",
                "sbp_mmhg": "",
                "sbp_mask": "0",
                "sbp_source": "",
                "dbp_mmhg": "",
                "dbp_mask": "0",
                "dbp_source": "",
                "map_mmhg": "",
                "map_mask": "0",
                "map_source": "",
            }
        )
    return rows


def parse_mindray_subject(path: Path, root: Path) -> str | None:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        match = re.fullmatch(r"subject(\d+)", part)
        if match:
            return f"subject{int(match.group(1))}"
    for part in parts:
        if part.lower() == "save":
            continue
        if part.isdigit():
            return f"subject{int(part)}"
    return None


def parse_repeated_numeric_value(value: str) -> float | None:
    if "^" in value:
        values = [finite_float(item) for item in value.split("^")]
        values = [item for item in values if item is not None]
        if not values:
            return None
        return float(np.median(values))
    return finite_float(value)


def valid_spo2(value: float | None) -> float | None:
    if value is None:
        return None
    if 50.0 <= value <= 100.0:
        return float(value)
    return None


def valid_bp(value: float | None, task: str) -> float | None:
    if value is None:
        return None
    ranges = {
        "sbp": (40.0, 260.0),
        "dbp": (20.0, 160.0),
        "map": (25.0, 180.0),
    }
    low, high = ranges[task]
    if low <= value <= high:
        return float(value)
    return None


def canonical_bp_session_key(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = Path(value).stem if isinstance(value, Path) else str(value)
    text = text.replace("_", "-").upper()
    text = re.sub(r"-(INPUT|LABEL)\d+$", "", text)
    text = re.sub(r"-BP$", "", text)
    match = re.search(r"([FM]\d+)-T(\d+)", text)
    if match:
        return f"{match.group(1)}-T{int(match.group(2))}"
    return None


def load_bp_waveform_series(root: Path) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    if not root.exists():
        print(f"skip missing bp waveform root: {root}")
        return series
    for bp_path in sorted(root.glob("**/*-BP.txt")):
        key = canonical_bp_session_key(bp_path)
        if key is None:
            continue
        try:
            values = np.loadtxt(bp_path, dtype=np.float64)
        except Exception as exc:
            print(f"warning: failed reading bp waveform {bp_path}: {exc}")
            continue
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            series[key] = values
    return series


def bp_values_for_clip(
    signal: np.ndarray,
    clip_index: int,
    max_clip_index: int,
) -> tuple[float | None, float | None, float | None]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 32:
        return None, None, None
    denom = max(max_clip_index + 1, 1)
    start = int(np.floor((clip_index / denom) * values.size))
    end = int(np.ceil(((clip_index + 1) / denom) * values.size))
    start = min(max(start, 0), values.size - 1)
    end = min(max(end, start + 1), values.size)
    segment = values[start:end]
    segment = segment[np.isfinite(segment)]
    if segment.size < 32:
        return None, None, None
    sbp = valid_bp(float(np.nanpercentile(segment, 95.0)), "sbp")
    dbp = valid_bp(float(np.nanpercentile(segment, 5.0)), "dbp")
    map_value = valid_bp(float(np.nanmean(segment)), "map")
    if map_value is None and sbp is not None and dbp is not None:
        map_value = valid_bp(dbp + (sbp - dbp) / 3.0, "map")
    return sbp, dbp, map_value


def row_bp_session_key(row: dict[str, str]) -> str | None:
    for field in ("subject", "input_files", "ppg_label_file"):
        key = canonical_bp_session_key(row.get(field))
        if key is not None:
            return key
    return None


def augment_rows_with_bp_waveforms(
    rows: list[dict[str, str]],
    bp_series: dict[str, np.ndarray],
) -> list[dict[str, str]]:
    if not bp_series:
        return rows
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row_bp_session_key(row)
        if key is not None and key in bp_series:
            grouped[key].append(row)
    for key, items in grouped.items():
        max_clip_index = max(int(row.get("clip_index") or 0) for row in items)
        signal = bp_series[key]
        for row in items:
            if row.get("sbp_mask") == "1" and row.get("dbp_mask") == "1":
                continue
            clip_index = int(row.get("clip_index") or 0)
            sbp, dbp, map_value = bp_values_for_clip(signal, clip_index, max_clip_index)
            if sbp is not None:
                row["sbp_mmhg"] = scalar_cell(sbp)
                row["sbp_mask"] = "1"
                row["sbp_source"] = "bp_waveform"
            if dbp is not None:
                row["dbp_mmhg"] = scalar_cell(dbp)
                row["dbp_mask"] = "1"
                row["dbp_source"] = "bp_waveform"
            if map_value is not None:
                row["map_mmhg"] = scalar_cell(map_value)
                row["map_mask"] = "1"
                row["map_source"] = "bp_waveform"
    return rows


def load_icu_label_series(root: Path) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if not root.exists():
        print(f"skip missing icu label root: {root}")
        return series
    for csv_path in sorted(root.glob("**/output_3tasks_*.csv")):
        if csv_path.name.startswith("._"):
            continue
        match = re.search(r"output_3tasks_(\d+)\.csv$", csv_path.name)
        if match is None:
            continue
        subject = f"subject{int(match.group(1))}"
        try:
            with csv_path.open(newline="", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    sbp = valid_bp(finite_float(row.get("sbp")), "sbp")
                    dbp = valid_bp(finite_float(row.get("dbp")), "dbp")
                    map_value = None
                    if sbp is not None and dbp is not None:
                        map_value = valid_bp(dbp + (sbp - dbp) / 3.0, "map")
                    if sbp is not None:
                        series[subject]["sbp"].append(sbp)
                    if dbp is not None:
                        series[subject]["dbp"].append(dbp)
                    if map_value is not None:
                        series[subject]["map"].append(map_value)
        except Exception as exc:
            print(f"warning: failed reading icu label csv {csv_path}: {exc}")
    return series


def augment_zpu_rows_with_icu_labels(
    rows: list[dict[str, str]],
    icu_series: dict[str, dict[str, list[float]]],
) -> list[dict[str, str]]:
    if not icu_series:
        return rows
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("subject") in icu_series:
            grouped[row["subject"]].append(row)
    for subject, items in grouped.items():
        max_clip_index = max(int(row.get("clip_index") or 0) for row in items)
        task_map = icu_series[subject]
        for row in items:
            clip_index = int(row.get("clip_index") or 0)
            for task, target_field in (
                ("sbp", "sbp_mmhg"),
                ("dbp", "dbp_mmhg"),
                ("map", "map_mmhg"),
            ):
                value = sequence_value_for_clip(task_map.get(task, []), clip_index, max_clip_index)
                if value is None:
                    continue
                row[target_field] = scalar_cell(value)
                row[f"{task}_mask"] = "1"
                row[f"{task}_source"] = "icu_output_3tasks"
    return rows


def parse_mindray_series(root: Path) -> dict[str, dict[str, list[float]]]:
    code_to_task = {
        "149530^MDC_PULS_OXIM_PULS_RATE^MDC": "pr",
        "147842^MDC_ECG_HEART_RATE^MDC": "hr",
        "151578^MDC_TTHOR_RESP_RATE^MDC": "rr",
        "150456^MDC_PULS_OXIM_SAT_O2^MDC": "spo2",
        "150021^MDC_PRESS_BLD_NONINV_SYS^MDC": "sbp",
        "150022^MDC_PRESS_BLD_NONINV_DIA^MDC": "dbp",
        "150023^MDC_PRESS_BLD_NONINV_MEAN^MDC": "map",
        "150033^MDC_PRESS_BLD_ART_SYS^MDC": "sbp",
        "150034^MDC_PRESS_BLD_ART_DIA^MDC": "dbp",
        "150035^MDC_PRESS_BLD_ART_MEAN^MDC": "map",
    }
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for log_path in sorted(root.glob("**/recorded_mindray.txt")):
        subject = parse_mindray_subject(log_path, root)
        if subject is None:
            continue
        with log_path.open(errors="ignore") as handle:
            for line in handle:
                if not line.startswith("OBX|"):
                    continue
                parts = line.rstrip("\n").split("|")
                if len(parts) <= 5:
                    continue
                code = parts[3]
                task = code_to_task.get(code)
                if task is None:
                    code_upper = code.upper()
                    if "PRESS" in code_upper or "NIBP" in code_upper or "BP" in code_upper:
                        if "SYS" in code_upper:
                            task = "sbp"
                        elif "DIA" in code_upper:
                            task = "dbp"
                        elif "MEAN" in code_upper or "MAP" in code_upper:
                            task = "map"
                if task is None:
                    continue
                value = parse_repeated_numeric_value(parts[5])
                if task == "spo2":
                    value = valid_spo2(value)
                elif task in {"sbp", "dbp", "map"}:
                    value = valid_bp(value, task)
                if value is not None:
                    grouped[subject][task].append(value)
    return grouped


def csv_column_task(fieldname: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "", fieldname.lower())
    if normalized in {"spo2", "spo2percent", "spo2pct", "sao2"}:
        return "spo2"
    if normalized in {"sbp", "sys", "systolic", "nibpsys", "nibpsystolic", "bpsys", "bloodpressuresys"}:
        return "sbp"
    if normalized in {"dbp", "dia", "diastolic", "nibpdia", "nibpdiastolic", "bpdia", "bloodpressuredia"}:
        return "dbp"
    if normalized in {"map", "meanbp", "nibpmean", "bpmean", "bloodpressuremean", "meanarterialpressure"}:
        return "map"
    return None


def parse_monitor_csv_series(root: Path) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for csv_path in sorted(root.glob("subject*/*.csv")):
        subject = parse_mindray_subject(csv_path, root)
        if subject is None:
            continue
        try:
            with csv_path.open(newline="", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                task_fields = [(field, csv_column_task(field)) for field in reader.fieldnames]
                task_fields = [(field, task) for field, task in task_fields if task is not None]
                if not task_fields:
                    continue
                for row in reader:
                    for field, task in task_fields:
                        value = finite_float(row.get(field))
                        if task == "spo2":
                            value = valid_spo2(value)
                        elif task in {"sbp", "dbp", "map"}:
                            value = valid_bp(value, task)
                        if value is not None:
                            grouped[subject][task].append(value)
        except Exception as exc:
            print(f"warning: failed reading monitor csv {csv_path}: {exc}")
    return grouped


def merge_series(*series_items: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    merged: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for series in series_items:
        for subject, task_map in series.items():
            for task, values in task_map.items():
                merged[subject][task].extend(values)
    return merged


def sequence_value_for_clip(values: list[float], clip_index: int, max_clip_index: int) -> float | None:
    if not values:
        return None
    seq = np.asarray(values, dtype=np.float64)
    if seq.size == 1:
        return float(seq[0])
    denom = max(max_clip_index + 1, 1)
    start = int(np.floor((clip_index / denom) * seq.size))
    end = int(np.ceil(((clip_index + 1) / denom) * seq.size))
    start = min(max(start, 0), seq.size - 1)
    end = min(max(end, start + 1), seq.size)
    return float(np.nanmedian(seq[start:end]))


def collect_zpu_inputs(args: argparse.Namespace) -> list[Path]:
    if args.zpu_file_list:
        inputs = []
        for csv_path in args.zpu_file_list:
            with Path(csv_path).open(newline="") as handle:
                reader = csv.DictReader(handle)
                if "input_files" not in reader.fieldnames:
                    raise ValueError(f"{csv_path} missing input_files column")
                inputs.extend(Path(row["input_files"]) for row in reader)
        return sorted(set(inputs))
    return sorted(Path(args.zpu_cache).glob("*_input*.npy"))


def build_zpu_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    zpu_cache = Path(args.zpu_cache)
    label_roots = [Path(item) for item in args.zpu_label_root]
    if not zpu_cache.exists():
        return []

    series = []
    for label_root in label_roots:
        if not label_root.exists():
            print(f"skip missing zpu label root: {label_root}")
            continue
        series.append(parse_mindray_series(label_root))
        series.append(parse_monitor_csv_series(label_root))
    mindray = merge_series(*series)
    inputs = collect_zpu_inputs(args)
    grouped_inputs: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for input_path in inputs:
        subject, clip_index = parse_clip(input_path)
        grouped_inputs[subject].append((clip_index, input_path))

    rows = []
    for subject, subject_items in grouped_inputs.items():
        max_clip_index = max(clip_index for clip_index, _ in subject_items)
        for clip_index, input_path in sorted(subject_items):
            label_path = Path(str(input_path).replace("input", "label"))
            bpm_value = None
            if label_path.exists():
                bpm = fft_bpm(np.load(label_path), args.fs, args.low_bpm, args.high_bpm, args.fft_nfft)
                bpm_value = bpm if math.isfinite(bpm) else None
            pr_value = sequence_value_for_clip(mindray.get(subject, {}).get("pr", []), clip_index, max_clip_index)
            hr_value = bpm_value
            rr_value = sequence_value_for_clip(mindray.get(subject, {}).get("rr", []), clip_index, max_clip_index)
            spo2_value = sequence_value_for_clip(mindray.get(subject, {}).get("spo2", []), clip_index, max_clip_index)
            sbp_value = sequence_value_for_clip(mindray.get(subject, {}).get("sbp", []), clip_index, max_clip_index)
            dbp_value = sequence_value_for_clip(mindray.get(subject, {}).get("dbp", []), clip_index, max_clip_index)
            map_value = sequence_value_for_clip(mindray.get(subject, {}).get("map", []), clip_index, max_clip_index)
            if pr_value is None:
                pr_value = bpm_value
            rows.append(
                {
                    "input_files": str(input_path),
                    "ppg_label_file": str(label_path),
                    "dataset": "ZPU",
                    "subject": subject,
                    "clip_index": str(clip_index),
                    "label_polarity": "1.0",
                    "ppg_mask": "1" if label_path.exists() else "0",
                    "pr_bpm": scalar_cell(pr_value),
                    "pr_mask": mask_cell(pr_value),
                    "pr_source": "mindray_pr" if subject in mindray and mindray[subject].get("pr") else "ppg_fft",
                    "hr_bpm": scalar_cell(hr_value),
                    "hr_mask": mask_cell(hr_value),
                    "hr_source": "pleth_fft" if hr_value is not None else "",
                    "rr_bpm": scalar_cell(rr_value),
                    "rr_mask": mask_cell(rr_value),
                    "rr_source": "mindray_rr" if subject in mindray and mindray[subject].get("rr") else "",
                    "spo2_pct": scalar_cell(spo2_value),
                    "spo2_mask": mask_cell(spo2_value),
                    "spo2_source": "monitor_spo2" if spo2_value is not None else "",
                    "sbp_mmhg": scalar_cell(sbp_value),
                    "sbp_mask": mask_cell(sbp_value),
                    "sbp_source": "monitor_bp" if sbp_value is not None else "",
                    "dbp_mmhg": scalar_cell(dbp_value),
                    "dbp_mask": mask_cell(dbp_value),
                    "dbp_source": "monitor_bp" if dbp_value is not None else "",
                    "map_mmhg": scalar_cell(map_value),
                    "map_mask": mask_cell(map_value),
                    "map_source": "monitor_bp" if map_value is not None else "",
                }
            )
    return rows


def split_source_rows(rows: list[dict[str, str]], valid_modulo: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["subject"])].append(row)
    train_rows, valid_rows = [], []
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for dataset, subject in grouped:
        by_dataset[dataset].append(subject)
    valid_subjects = set()
    for dataset, subjects in by_dataset.items():
        for idx, subject in enumerate(sorted(subjects, key=subject_sort_key)):
            if valid_modulo > 0 and idx % valid_modulo == 0:
                valid_subjects.add((dataset, subject))
    for key, items in grouped.items():
        target = valid_rows if key in valid_subjects else train_rows
        target.extend(sorted(items, key=lambda row: int(row["clip_index"])))
    return train_rows, valid_rows


def split_fraction_by_subject(
    rows: list[dict[str, str]],
    begin: float,
    end: float,
) -> list[dict[str, str]]:
    subjects = sorted({row["subject"] for row in rows}, key=subject_sort_key)
    if begin <= 0.0 and end >= 1.0:
        selected = set(subjects)
    else:
        start = int(begin * len(subjects))
        stop = int(end * len(subjects))
        selected = set(subjects[start:stop])
    return [row for row in rows if row["subject"] in selected]


def split_spo2_rows(rows: list[dict[str, str]], valid_modulo: int = 2) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    spo2_rows = [row for row in rows if row.get("spo2_mask") == "1"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in spo2_rows:
        grouped[row["subject"]].append(row)
    train_rows, valid_rows = [], []
    for idx, subject in enumerate(sorted(grouped, key=subject_sort_key)):
        target = valid_rows if valid_modulo > 0 and idx % valid_modulo == 1 else train_rows
        target.extend(sorted(grouped[subject], key=lambda row: int(row["clip_index"])))
    return train_rows, valid_rows


def spo2_only_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    oversampled = []
    for row in rows:
        if row.get("spo2_mask") != "1":
            continue
        item = dict(row)
        item["ppg_mask"] = "0"
        item["pr_mask"] = "0"
        item["hr_mask"] = "0"
        item["rr_mask"] = "0"
        item["sbp_mask"] = "0"
        item["dbp_mask"] = "0"
        item["map_mask"] = "0"
        oversampled.append(item)
    return oversampled


def merge_unique(*row_groups: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = []
    seen = set()
    for rows in row_groups:
        for row in rows:
            key = row["input_files"]
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def summarize(name: str, rows: list[dict[str, str]]) -> None:
    counts = defaultdict(int)
    for row in rows:
        counts[row["dataset"]] += 1
    parts = ", ".join(f"{dataset}={count}" for dataset, count in sorted(counts.items())) or "empty"
    mask_parts = []
    for task in ["ppg", "pr", "hr", "rr", "spo2", "sbp", "dbp", "map"]:
        mask_parts.append(f"{task}={sum(1 for row in rows if row.get(f'{task}_mask') == '1')}")
    print(f"{name}: rows={len(rows)} {parts}; masks {' '.join(mask_parts)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/public_hw/share/cit_ztyu/zhaobo/rPPG_dataset_processed/JOINT/DataFileLists")
    parser.add_argument("--prefix", default="MultiTask_PPG_PR_HR_SpO2_72x72_160")
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--low-bpm", type=float, default=45.0)
    parser.add_argument("--high-bpm", type=float, default=180.0)
    parser.add_argument("--fft-nfft", type=int, default=512)
    parser.add_argument("--source-valid-modulo", type=int, default=5)
    parser.add_argument("--zpu-cache", default="/public_hw/share/cit_ztyu/zhaobo/ZPH_Chunk160_Det30")
    parser.add_argument(
        "--zpu-label-root",
        action="append",
        default=[
            "/public_hw/share/cit_ztyu/zhaobo/macopoly/data2.9/MPU-ZPH-RPPG-ST",
            "/public_hw/share/cit_ztyu/zhaobo/zhuhai",
        ],
        help="Root containing ZPU/ZPH monitor labels. Can be repeated.",
    )
    parser.add_argument("--zpu-file-list", action="append", default=[])
    parser.add_argument("--zpu-train-begin", type=float, default=0.6)
    parser.add_argument("--zpu-train-end", type=float, default=0.8)
    parser.add_argument("--zpu-valid-begin", type=float, default=0.8)
    parser.add_argument("--zpu-valid-end", type=float, default=1.0)
    parser.add_argument("--zpu-test-begin", type=float, default=0.4)
    parser.add_argument("--zpu-test-end", type=float, default=0.6)
    parser.add_argument("--zpu-spo2-valid-modulo", type=int, default=2)
    parser.add_argument("--no-zpu-spo2-oversample", action="store_true")
    parser.add_argument(
        "--source-cache",
        action="append",
        default=[],
        help="Override/add source cache as DATASET=/path/to/cache. Defaults to UBFC/PURE/BUAA/MMPD.",
    )
    parser.add_argument(
        "--bp-waveform-root",
        action="append",
        default=[],
        help="Root containing per-session *-BP.txt pressure waveforms, e.g. V4V BP_raw_1KHz. Can be repeated.",
    )
    parser.add_argument(
        "--icu-label-root",
        action="append",
        default=[],
        help="Root containing ICU_LABEL output_3tasks_*.csv files with frame-level sbp/dbp labels. Can be repeated.",
    )
    args = parser.parse_args()

    source_caches = dict(DEFAULT_SOURCE_CACHES)
    for item in args.source_cache:
        if "=" not in item:
            raise ValueError(f"--source-cache must be DATASET=/path, got {item}")
        dataset, path = item.split("=", 1)
        source_caches[dataset] = path

    source_rows = []
    for dataset, cache in source_caches.items():
        cache_dir = Path(cache)
        if not cache_dir.exists():
            print(f"skip missing source cache: {dataset} {cache_dir}")
            continue
        source_rows.extend(build_source_rows(dataset, cache_dir, args))

    bp_series = {}
    for item in args.bp_waveform_root:
        bp_series.update(load_bp_waveform_series(Path(item)))
    if bp_series:
        source_rows = augment_rows_with_bp_waveforms(source_rows, bp_series)

    source_train, source_valid = split_source_rows(source_rows, args.source_valid_modulo)
    zpu_rows = build_zpu_rows(args)
    if bp_series:
        zpu_rows = augment_rows_with_bp_waveforms(zpu_rows, bp_series)
    icu_series = {}
    for item in args.icu_label_root:
        for subject, task_map in load_icu_label_series(Path(item)).items():
            if subject not in icu_series:
                icu_series[subject] = defaultdict(list)
            for task, values in task_map.items():
                icu_series[subject][task].extend(values)
    if icu_series:
        zpu_rows = augment_zpu_rows_with_icu_labels(zpu_rows, icu_series)
    zpu_train = split_fraction_by_subject(zpu_rows, args.zpu_train_begin, args.zpu_train_end)
    zpu_valid = split_fraction_by_subject(zpu_rows, args.zpu_valid_begin, args.zpu_valid_end)
    zpu_test = split_fraction_by_subject(zpu_rows, args.zpu_test_begin, args.zpu_test_end)
    zpu_spo2_train = spo2_only_rows(zpu_train)
    zpu_spo2_valid = []
    if args.no_zpu_spo2_oversample:
        zpu_spo2_train, zpu_spo2_valid = [], []

    output_dir = Path(args.output_dir)
    joint_train = merge_unique(source_train, zpu_train)
    joint_train.extend(zpu_spo2_train)
    joint_valid = merge_unique(source_valid, zpu_valid)
    outputs = {
        "source_train": source_train,
        "source_valid": source_valid,
        "zpu_all": zpu_rows,
        "zpu_train": zpu_train,
        "zpu_valid": zpu_valid,
        "zpu_test": zpu_test,
        "zpu_spo2_train": zpu_spo2_train,
        "zpu_spo2_valid": zpu_spo2_valid,
        "joint_train": joint_train,
        "joint_valid": joint_valid,
    }
    for suffix, rows in outputs.items():
        path = output_dir / f"{args.prefix}_{suffix}.csv"
        write_csv(path, rows)
        summarize(str(path), rows)


if __name__ == "__main__":
    main()

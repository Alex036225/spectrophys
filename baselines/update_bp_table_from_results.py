\
"""Update the ZPU benchmark LaTeX table with BP MAE results."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path("/public_hw/home/cit_yingxinlai/project/nature")
TABLE = ROOT / "reports/zpu0406_subject19_benchmark_table.tex"


def two(value) -> str:
    return f"{float(value):.2f}"


def replace_bp_columns(text: str, model: str, sbp: float, dbp: float, map_: float) -> str:
    pattern = re.compile(rf"^({re.escape(model)} & .*? & )(?:--|[0-9.]+) & (?:--|[0-9.]+) & (?:--|[0-9.]+) \\\\", re.MULTILINE)
    replacement = rf"\g<1>{two(sbp)} & {two(dbp)} & {two(map_)} \\\\"
    new_text, count = pattern.subn(replacement, text)
    if count != 1:
        raise RuntimeError(f"Expected one row for {model}, replaced {count}")
    return new_text


def load_standard_bp() -> dict[str, tuple[float, float, float]]:
    path = ROOT / "reports/benchmark_zpu0406_standard_bp_heads_results.csv"
    name_map = {
        "physnet": "PhysNet",
        "deepphys": "DeepPhys",
        "physformer": "PhysFormer",
        "rhythmformer": "RhythmFormer",
        "bigsmall": "BigSmall",
        "efficientphys": "EfficientPhys",
        "multiphysnet": "MultiPhysNet",
    }
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            out[name_map[row["model"]]] = (float(row["sbp_mae"]), float(row["dbp_mae"]), float(row["map_mae"]))
    return out


def load_json_result(path: str) -> tuple[float, float, float]:
    summary = json.loads((ROOT / path).read_text())
    test = summary["final_test"]
    return float(test["sbp_mae"]), float(test["dbp_mae"]), float(test["map_mae"])


def load_unsup_bp() -> dict[str, tuple[float, float, float]]:
    path = ROOT / "runs/exp/unsupervised_bp_adapters_zpu0406/unsupervised_bp_adapter_results.csv"
    out = {}
    if not path.exists():
        return out
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["method"]] = (float(row["sbp_mae"]), float(row["dbp_mae"]), float(row["map_mae"]))
    return out


def main() -> None:
    results = load_standard_bp()
    results.update(
        {
            "FusionVitals (RGB)": load_json_result("runs/exp/external_bp_heads_zpu0406/fusionvitals/fusionvitals_bphead_summary.json"),
            "rPPG-SpO2 (adapter)": load_json_result("runs/exp/external_bp_heads_zpu0406/boe_rppg_spo2/boe_rppg_spo2_bphead_summary.json"),
            "PhysMLE": load_json_result("runs/exp/physmle_bp_head_zpu0406/physmle_officialstmap_bphead/physmle_officialstmap_bphead_summary.json"),
        }
    )
    results.update(load_unsup_bp())

    text = TABLE.read_text()
    for model, values in results.items():
        text = replace_bp_columns(text, model, *values)
    TABLE.write_text(text)
    print(TABLE)
    for model in sorted(results):
        print(model, results[model])


if __name__ == "__main__":
    main()

\
"""Collect standard-model BP head benchmark summaries."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path("/public_hw/home/cit_yingxinlai/project/nature")
MODELS = ["physnet", "deepphys", "physformer", "rhythmformer", "bigsmall", "efficientphys", "multiphysnet"]


def read_metric(metrics: dict, key: str) -> float:
    value = metrics.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.6f}"


def main() -> None:
    rows = []
    for model in MODELS:
        summary_path = ROOT / "runs/exp/benchmark_bp_heads_zpu0406" / model / f"{model}_bphead_summary.json"
        if not summary_path.exists():
            rows.append({"model": model, "status": "missing", "summary": str(summary_path)})
            continue
        summary = json.loads(summary_path.read_text())
        test = summary.get("final_test", {})
        rows.append(
            {
                "model": model,
                "status": "ok",
                "best_epoch": summary.get("best_epoch", ""),
                "sbp_mae": fmt(read_metric(test, "sbp_mae")),
                "sbp_rmse": fmt(read_metric(test, "sbp_rmse")),
                "sbp_pearson": fmt(read_metric(test, "sbp_pearson")),
                "dbp_mae": fmt(read_metric(test, "dbp_mae")),
                "dbp_rmse": fmt(read_metric(test, "dbp_rmse")),
                "dbp_pearson": fmt(read_metric(test, "dbp_pearson")),
                "map_mae": fmt(read_metric(test, "map_mae")),
                "map_rmse": fmt(read_metric(test, "map_rmse")),
                "map_pearson": fmt(read_metric(test, "map_pearson")),
                "bp_count": int(read_metric(test, "bp_count")) if math.isfinite(read_metric(test, "bp_count")) else "",
                "best_checkpoint": summary.get("best_checkpoint", ""),
                "test_outputs": summary.get("test_outputs", ""),
                "summary": str(summary_path),
            }
        )

    out_path = ROOT / "reports/benchmark_zpu0406_standard_bp_heads_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "status",
        "best_epoch",
        "sbp_mae",
        "sbp_rmse",
        "sbp_pearson",
        "dbp_mae",
        "dbp_rmse",
        "dbp_pearson",
        "map_mae",
        "map_rmse",
        "map_pearson",
        "bp_count",
        "best_checkpoint",
        "test_outputs",
        "summary",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

import csv
import math
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    pickle_path = Path(
        "runs/exp/multitask_phasenet_bp_headonly_trainval_zpu0406/"
        "bp_headonly_train29_32_41_49_from_rrlowfreq_best/"
        "bp_headonly_train29_32_41_49_from_rrlowfreq_best_test_outputs.pickle"
    )
    out_path = Path("reports/zpu0406_subject19_phasenet_mt_bp_correlations.csv")
    obj = pickle.load(pickle_path.open("rb"))
    tasks = ["hr", "pr", "rr", "spo2", "sbp", "dbp", "map"]
    rows = []

    for task in tasks:
        preds = []
        labels = []
        for subject_data in obj.values():
            for item in subject_data.values():
                mask = item.get("scalar_masks", {}).get(task, 0.0)
                if mask <= 0:
                    continue
                pred = item.get("scalars_pred", {}).get(task)
                label = item.get("scalars_label", {}).get(task)
                if pred is None or label is None:
                    continue
                pred = float(pred)
                label = float(label)
                if not (math.isfinite(pred) and math.isfinite(label)):
                    continue
                preds.append(pred)
                labels.append(label)

        pred_arr = np.asarray(preds, dtype=float)
        label_arr = np.asarray(labels, dtype=float)
        if len(pred_arr) == 0:
            mae = rmse = pearson = pred_mean = label_mean = float("nan")
        else:
            err = pred_arr - label_arr
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            if len(pred_arr) > 1 and np.std(pred_arr) > 0 and np.std(label_arr) > 0:
                pearson = float(np.corrcoef(pred_arr, label_arr)[0, 1])
            else:
                pearson = float("nan")
            pred_mean = float(np.mean(pred_arr))
            label_mean = float(np.mean(label_arr))

        rows.append(
            {
                "task": task,
                "count": len(pred_arr),
                "mae": mae,
                "rmse": rmse,
                "pearson": pearson,
                "pred_mean": pred_mean,
                "label_mean": label_mean,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "count",
                "mae",
                "rmse",
                "pearson",
                "pred_mean",
                "label_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(out_path)
    for row in rows:
        print(
            "{task:>5s} count={count:d} mae={mae:.6f} rmse={rmse:.6f} "
            "pearson={pearson:.6f} pred_mean={pred_mean:.6f} "
            "label_mean={label_mean:.6f}".format(**row)
        )


if __name__ == "__main__":
    main()

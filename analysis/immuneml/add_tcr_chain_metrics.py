from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


POSITIVE_LABEL = {
    "cd_vs_control": "CD",
    "uc_vs_control": "UC",
    "cd_vs_uc": "CD",
}


def parse_task(task_name: str) -> dict[str, str]:
    for comparison in POSITIVE_LABEL:
        marker = f"_{comparison}_"
        if marker in task_name:
            chain_group, rest = task_name.split(marker, 1)
            if rest.endswith("_aa"):
                sequence_type = "aa"
                kmer = rest.removesuffix("_aa")
            elif rest.endswith("_nt"):
                sequence_type = "nt"
                kmer = rest.removesuffix("_nt")
            else:
                sequence_type = ""
                kmer = rest
            return {
                "instruction": task_name,
                "chain_group": chain_group,
                "comparison": comparison,
                "kmer": kmer,
                "sequence_type": sequence_type,
            }
    raise ValueError(f"Could not parse task name: {task_name}")


def score_prediction_file(path: Path) -> dict:
    task_name = path.parents[1].name
    split_name = path.parent.name
    split_index = int(split_name.replace("split_", ""))
    parsed = parse_task(task_name)
    positive = POSITIVE_LABEL[parsed["comparison"]]

    predictions = pd.read_csv(path)
    y_true = predictions["Diagnosis1_true"].astype(str)
    y_pred = predictions["Diagnosis1_pred"].astype(str)

    prob_col = f"prob_{positive}"
    if prob_col not in predictions.columns:
        raise ValueError(f"Missing {prob_col} in {path}")

    y_true_binary = (y_true == positive).astype(int)
    y_pred_binary = (y_pred == positive).astype(int)

    return {
        **parsed,
        "split_index": split_index,
        "positive_label": positive,
        "n_test": int(len(predictions)),
        "Diagnosis1_accuracy": accuracy_score(y_true, y_pred),
        "Diagnosis1_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "Diagnosis1_mcc": matthews_corrcoef(y_true, y_pred),
        "Diagnosis1_f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        "Diagnosis1_roc_auc": roc_auc_score(y_true_binary, predictions[prob_col]),
        "prediction_file": path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="chain_tcr_immuneml/sklearn_results")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    rows = [
        score_prediction_file(path)
        for path in sorted(result_root.glob("*/split_*/test_predictions.csv"))
    ]
    if not rows:
        raise SystemExit(f"No test_predictions.csv files found under {result_root}")

    detail = pd.DataFrame(rows).sort_values(
        ["chain_group", "comparison", "kmer", "split_index"]
    )
    detail_path = result_root / "all_split_scores_with_extra_metrics.csv"
    detail.to_csv(detail_path, index=False)

    metric_cols = [
        "Diagnosis1_accuracy",
        "Diagnosis1_balanced_accuracy",
        "Diagnosis1_mcc",
        "Diagnosis1_f1",
        "Diagnosis1_roc_auc",
        "n_test",
    ]
    aggregate = (
        detail.groupby(["chain_group", "comparison", "kmer", "sequence_type", "positive_label"], dropna=False)[
            metric_cols
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in aggregate.columns
    ]
    aggregate_path = result_root / "aggregate_scores_with_extra_metrics.csv"
    aggregate.to_csv(aggregate_path, index=False)

    best = aggregate.sort_values(
        ["comparison", "Diagnosis1_balanced_accuracy_mean"],
        ascending=[True, False],
    ).groupby("comparison").head(5)
    best_path = result_root / "best_models_by_comparison_with_extra_metrics.csv"
    best.to_csv(best_path, index=False)

    print(f"Wrote {detail_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {best_path}")
    cols = [
        "comparison",
        "chain_group",
        "kmer",
        "Diagnosis1_balanced_accuracy_mean",
        "Diagnosis1_mcc_mean",
        "Diagnosis1_f1_mean",
        "Diagnosis1_roc_auc_mean",
    ]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()

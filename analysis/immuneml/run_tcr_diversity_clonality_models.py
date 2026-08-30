from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CHAIN_GROUPS = [
    "tcr_alpha",
    "tcr_beta",
    "tcr_alpha_beta",
    "tcr_gamma",
    "tcr_delta",
    "tcr_gamma_delta",
]

COMPARISONS = ["cd_vs_control", "uc_vs_control", "cd_vs_uc"]

POSITIVE_LABEL = {
    "cd_vs_control": "CD",
    "uc_vs_control": "UC",
    "cd_vs_uc": "CD",
    "cd_inflamed_vs_noninflamed": "Inflamed",
    "uc_inflamed_vs_noninflamed": "Inflamed",
    "therapy_combined_response": "Nonresponder",
    "therapy_antitnf_response": "Nonresponder",
    "therapy_ustekinumab_response": "Responder",
    "therapy_vedolizumab_response": "Nonresponder",
}


def gini(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    sorted_values = np.sort(values.astype(float))
    total = sorted_values.sum()
    if total <= 0:
        return 0.0
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values) / (n * total)) - ((n + 1) / n))


def repertoire_features(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, sep="\t", usecols=["duplicate_count", "junction"])
    df = df.dropna(subset=["junction"])
    counts = pd.to_numeric(df["duplicate_count"], errors="coerce").fillna(1).to_numpy(dtype=float)
    counts = counts[counts > 0]
    richness = len(counts)
    total = float(counts.sum())
    if richness == 0 or total <= 0:
        return {
            "n_clonotypes": 0.0,
            "total_clone_count": 0.0,
            "log10_total_clone_count": 0.0,
            "shannon_entropy": 0.0,
            "pielou_evenness": 0.0,
            "clonality": 0.0,
            "simpson_index": 0.0,
            "inverse_simpson": 0.0,
            "gini_index": 0.0,
            "max_clone_frequency": 0.0,
            "top5_clone_frequency": 0.0,
            "top10_clone_frequency": 0.0,
            "singleton_fraction": 0.0,
        }

    frequencies = counts / total
    shannon = float(-np.sum(frequencies * np.log(frequencies)))
    evenness = float(shannon / np.log(richness)) if richness > 1 else 0.0
    simpson = float(np.sum(frequencies**2))
    sorted_freq = np.sort(frequencies)[::-1]

    return {
        "n_clonotypes": float(richness),
        "total_clone_count": total,
        "log10_total_clone_count": float(np.log10(total + 1)),
        "shannon_entropy": shannon,
        "pielou_evenness": evenness,
        "clonality": float(1 - evenness),
        "simpson_index": simpson,
        "inverse_simpson": float(1 / simpson) if simpson > 0 else 0.0,
        "gini_index": gini(counts),
        "max_clone_frequency": float(sorted_freq[0]),
        "top5_clone_frequency": float(sorted_freq[:5].sum()),
        "top10_clone_frequency": float(sorted_freq[:10].sum()),
        "singleton_fraction": float(np.mean(counts == 1)),
    }


def build_feature_table(input_root: Path, chain_group: str, comparison: str, label_column: str = "Diagnosis1") -> pd.DataFrame:
    group_root = input_root / chain_group
    metadata = pd.read_csv(group_root / f"metadata_{comparison}.csv")
    rows = []
    for row in metadata.itertuples(index=False):
        feature_row = {
            "Sample": row.Sample,
            "filename": row.filename,
            label_column: getattr(row, label_column),
        }
        feature_row.update(repertoire_features(group_root / "repertoires" / row.filename))
        rows.append(feature_row)
    return pd.DataFrame(rows)


def make_model(method: str, split_index: int):
    if method == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=5000, solver="liblinear", random_state=3000 + split_index),
                ),
            ]
        )
    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=4000 + split_index,
        )
    raise ValueError(f"Unknown method: {method}")


def run_task(
    input_root: Path,
    output_root: Path,
    chain_group: str,
    comparison: str,
    method: str,
    label_column: str,
) -> list[dict]:
    task_name = f"{chain_group}_{comparison}_diversity_clonality_{method}"
    task_root = output_root / task_name
    task_root.mkdir(parents=True, exist_ok=True)

    feature_table = build_feature_table(input_root, chain_group, comparison, label_column)
    feature_table.to_csv(task_root / "feature_table.csv", index=False)

    feature_columns = [
        col
        for col in feature_table.columns
        if col not in {"Sample", "filename", label_column}
    ]
    x = feature_table[feature_columns].to_numpy(dtype=float)
    y = feature_table[label_column].astype(str).to_numpy()
    sample_ids = feature_table["Sample"].astype(str).to_numpy()
    filenames = feature_table["filename"].astype(str).to_numpy()

    splitter = StratifiedShuffleSplit(n_splits=5, train_size=0.7, random_state=5000)
    positive_label = POSITIVE_LABEL[comparison]
    rows = []

    for split_index, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        split_root = task_root / f"split_{split_index}"
        split_root.mkdir(parents=True, exist_ok=True)

        model = make_model(method, split_index)
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])
        probabilities = model.predict_proba(x[test_idx])
        class_names = list(model.classes_)
        positive_index = class_names.index(positive_label)
        y_test_binary = (y[test_idx] == positive_label).astype(int)
        y_pred_binary = (y_pred == positive_label).astype(int)

        score_row = {
            "instruction": task_name,
            "chain_group": chain_group,
            "comparison": comparison,
            "model_family": "diversity_clonality",
            "feature_set": "diversity_clonality",
            "ml_method": method,
            "split_index": split_index,
            "positive_label": positive_label,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(feature_columns)),
            f"{label_column}_accuracy": accuracy_score(y[test_idx], y_pred),
            f"{label_column}_balanced_accuracy": balanced_accuracy_score(y[test_idx], y_pred),
            f"{label_column}_mcc": matthews_corrcoef(y[test_idx], y_pred),
            f"{label_column}_f1": f1_score(y_test_binary, y_pred_binary, zero_division=0),
            f"{label_column}_roc_auc": roc_auc_score(y_test_binary, probabilities[:, positive_index]),
        }
        rows.append(score_row)

        prediction_rows = []
        for local_index, sample_index in enumerate(test_idx):
            pred_row = {
                "Sample": sample_ids[sample_index],
                "filename": filenames[sample_index],
                f"{label_column}_true": y[test_idx][local_index],
                f"{label_column}_pred": y_pred[local_index],
            }
            for class_index, class_name in enumerate(class_names):
                pred_row[f"prob_{class_name}"] = probabilities[local_index, class_index]
            prediction_rows.append(pred_row)

        pd.DataFrame(prediction_rows).to_csv(split_root / "test_predictions.csv", index=False)
        pd.DataFrame([score_row]).to_csv(split_root / "ml_score.csv", index=False)
        with (split_root / "model_info.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "classes": class_names,
                    "feature_columns": feature_columns,
                    "train_samples": sample_ids[train_idx].tolist(),
                    "test_samples": sample_ids[test_idx].tolist(),
                },
                handle,
                indent=2,
            )

    pd.DataFrame(rows).to_csv(task_root / "ml_scores.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_tcr_immuneml/airr_input")
    parser.add_argument("--output-root", default="chain_tcr_immuneml/diversity_clonality_results")
    parser.add_argument("--label-column", default="Diagnosis1")
    parser.add_argument("--comparisons", nargs="+", default=COMPARISONS)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["logistic_regression", "random_forest"],
        choices=["logistic_regression", "random_forest"],
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for method in args.methods:
        for chain_group in CHAIN_GROUPS:
            for comparison in args.comparisons:
                print(f"Running {method} {chain_group} {comparison} diversity/clonality")
                all_rows.extend(run_task(input_root, output_root, chain_group, comparison, method, args.label_column))

    detail = pd.DataFrame(all_rows)
    detail.to_csv(output_root / "all_split_scores.csv", index=False)

    metric_prefix = args.label_column
    metric_columns = [
        f"{metric_prefix}_accuracy",
        f"{metric_prefix}_balanced_accuracy",
        f"{metric_prefix}_mcc",
        f"{metric_prefix}_f1",
        f"{metric_prefix}_roc_auc",
        "n_train",
        "n_test",
        "n_features",
    ]
    aggregate = (
        detail.groupby(["chain_group", "comparison", "model_family", "feature_set", "ml_method"], dropna=False)[
            metric_columns
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
    aggregate.to_csv(output_root / "aggregate_scores.csv", index=False)

    final_columns = [
        "comparison",
        "chain_group",
        "model_family",
        "feature_set",
        "ml_method",
        f"{metric_prefix}_balanced_accuracy_mean",
        f"{metric_prefix}_mcc_mean",
        f"{metric_prefix}_f1_mean",
        f"{metric_prefix}_roc_auc_mean",
    ]
    final = aggregate[final_columns].sort_values(["comparison", "chain_group", "ml_method"])
    final.to_csv(output_root / "final_metrics_table.csv", index=False)
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_tcr_diversity_clonality_models import POSITIVE_LABEL as DIV_POSITIVE_LABEL
from run_tcr_diversity_clonality_models import build_feature_table, make_model
from run_tcr_chain_kmer_models import (
    POSITIVE_LABEL,
    counters_to_matrix,
    read_sample_sequences,
    sample_counter,
)


COMPARISON_LABELS = {
    "cd_vs_control": "CD vs Control",
    "uc_vs_control": "UC vs Control",
    "cd_vs_uc": "CD vs UC",
    "cd_inflamed_vs_noninflamed": "CD Inflamed vs Noninflamed",
    "uc_inflamed_vs_noninflamed": "UC Inflamed vs Noninflamed",
    "therapy_combined_response": "Combined Therapy Nonresponder vs Responder",
    "therapy_antitnf_response": "AntiTNF Nonresponder vs Responder",
    "therapy_ustekinumab_response": "Ustekinumab Responder vs Nonresponder",
    "therapy_vedolizumab_response": "Vedolizumab Nonresponder vs Responder",
}


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value.lower()


def fit_kmer_model(input_root: Path, model_row: pd.Series, label_column: str):
    group_root = input_root / model_row["chain_group"]
    metadata = pd.read_csv(group_root / f"metadata_{model_row['comparison']}.csv")
    labels = metadata[label_column].astype(str).to_numpy()
    k = int(str(model_row["kmer"]).replace("k", ""))
    sequence_type = model_row["sequence_type"]

    counters = []
    for filename in metadata["filename"].astype(str):
        seqs = read_sample_sequences(group_root / "repertoires" / filename, sequence_type)
        counters.append(sample_counter(seqs, k))

    vocabulary = {
        kmer: index
        for index, kmer in enumerate(sorted({kmer for counter in counters for kmer in counter}))
    }
    x_raw = counters_to_matrix(counters, vocabulary)
    scaler = StandardScaler(with_mean=False, with_std=True)
    x_scaled = scaler.fit_transform(x_raw)

    model = LogisticRegression(max_iter=5000, solver="liblinear", random_state=9001)
    model.fit(x_scaled, labels)

    feature_names = np.array(sorted(vocabulary, key=vocabulary.get))
    return model, x_scaled, x_raw, labels, feature_names, metadata


def linear_shap_values(model: LogisticRegression, x_scaled: sparse.csr_matrix, positive_label: str) -> np.ndarray:
    x_dense = x_scaled.toarray()
    coef = model.coef_[0].copy()
    if list(model.classes_)[1] != positive_label:
        coef = -coef
    baseline = x_dense.mean(axis=0)
    return (x_dense - baseline) * coef


def fit_diversity_model(input_root: Path, model_row: pd.Series, label_column: str):
    feature_table = build_feature_table(input_root, model_row["chain_group"], model_row["comparison"], label_column)
    feature_columns = [
        col for col in feature_table.columns if col not in {"Sample", "filename", label_column}
    ]
    x_df = feature_table[feature_columns]
    labels = feature_table[label_column].astype(str).to_numpy()
    method = model_row["ml_method"]
    model = make_model(method, split_index=99)
    model.fit(x_df.to_numpy(dtype=float), labels)
    return model, x_df, labels, np.array(feature_columns), feature_table


def diversity_shap_values(model, x_df: pd.DataFrame, positive_label: str) -> np.ndarray:
    if isinstance(model, Pipeline):
        scaler = model.named_steps["scaler"]
        classifier = model.named_steps["classifier"]
        x_scaled = scaler.transform(x_df.to_numpy(dtype=float))
        coef = classifier.coef_[0].copy()
        if list(classifier.classes_)[1] != positive_label:
            coef = -coef
        baseline = x_scaled.mean(axis=0)
        return (x_scaled - baseline) * coef

    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_df.to_numpy(dtype=float))
        positive_index = list(model.classes_).index(positive_label)
        if isinstance(values, list):
            return values[positive_index]
        values = np.asarray(values)
        if values.ndim == 3:
            return values[:, :, positive_index]
        return values

    raise TypeError(f"Unsupported diversity model type: {type(model)}")


def write_importance(
    shap_values: np.ndarray,
    feature_values,
    feature_names: np.ndarray,
    output_prefix: Path,
    top_n: int,
) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    if sparse.issparse(feature_values):
        feature_mean = np.asarray(feature_values.mean(axis=0)).ravel()
    else:
        feature_mean = np.asarray(feature_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    importance = pd.DataFrame(
        {
            "feature": feature_names[order],
            "mean_abs_shap": mean_abs[order],
            "mean_feature_value": feature_mean[order],
        }
    )
    importance.to_csv(output_prefix.with_name(output_prefix.name + "_importance.csv"), index=False)
    return importance.head(top_n)


def save_shap_plots(
    shap_values: np.ndarray,
    feature_values,
    feature_names: np.ndarray,
    title: str,
    output_prefix: Path,
    max_display: int,
) -> None:
    if sparse.issparse(feature_values):
        values_for_plot = feature_values.toarray()
    elif isinstance(feature_values, pd.DataFrame):
        values_for_plot = feature_values
    else:
        values_for_plot = np.asarray(feature_values)

    plt.figure(figsize=(7.2, 5.2))
    shap.summary_plot(
        shap_values,
        values_for_plot,
        feature_names=feature_names.tolist(),
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(title + "\nMean absolute SHAP value", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_bar.png"), dpi=300, bbox_inches="tight")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_bar.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.2, 5.2))
    shap.summary_plot(
        shap_values,
        values_for_plot,
        feature_names=feature_names.tolist(),
        max_display=max_display,
        show=False,
    )
    plt.title(title + "\nSHAP summary", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_beeswarm.pdf"), bbox_inches="tight")
    plt.close()


def select_top_models(main_table: pd.DataFrame, top_n: int, metric_prefix: str) -> pd.DataFrame:
    metric_candidates = {
        "roc_auc": [f"{metric_prefix}_roc_auc_mean", "roc_auc_mean"],
        "balanced_accuracy": [f"{metric_prefix}_balanced_accuracy_mean", "balanced_accuracy_mean"],
        "mcc": [f"{metric_prefix}_mcc_mean", "mcc_mean"],
    }
    metric_cols = {}
    for metric_name, candidates in metric_candidates.items():
        for candidate in candidates:
            if candidate in main_table.columns:
                metric_cols[metric_name] = candidate
                break
        else:
            raise KeyError(f"Could not find a column for {metric_name}; tried {candidates}")

    sort_cols = [
        "comparison",
        metric_cols["roc_auc"],
        metric_cols["balanced_accuracy"],
        metric_cols["mcc"],
    ]
    selected = (
        main_table.sort_values(sort_cols, ascending=[True, False, False, False])
        .groupby("comparison", group_keys=False)
        .head(top_n)
        .copy()
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-table",
        default="chain_tcr_immuneml/combined_results/main_model_metrics_table_with_diversity.csv",
    )
    parser.add_argument("--input-root", default="chain_tcr_immuneml/airr_input")
    parser.add_argument("--output-dir", default="chain_tcr_immuneml/shap_top_models")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument("--label-column", default="Diagnosis1")
    parser.add_argument("--metric-prefix", default="Diagnosis1")
    args = parser.parse_args()

    main_table = pd.read_csv(args.main_table)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_top_models(main_table, args.top_n, args.metric_prefix)
    selected.to_csv(output_dir / "selected_top_models_for_shap.csv", index=False)

    all_importance = []
    for row in selected.itertuples(index=False):
        model_row = pd.Series(row._asdict())
        comparison = model_row["comparison"]
        positive_label = POSITIVE_LABEL.get(comparison)
        if positive_label is None:
            positive_label = DIV_POSITIVE_LABEL[comparison]
        comparison_dir = output_dir / comparison
        comparison_dir.mkdir(parents=True, exist_ok=True)

        label = model_row["model_label"]
        prefix = comparison_dir / safe_name(label)
        title = f"{COMPARISON_LABELS[comparison]}: {label}"
        print(f"Generating SHAP plots for {title}")

        if model_row["model_family"] == "cdr3_kmer":
            model, x_scaled, x_raw, labels, feature_names, metadata = fit_kmer_model(input_root, model_row, args.label_column)
            shap_values = linear_shap_values(model, x_scaled, positive_label)
            feature_values = x_raw
        else:
            model, x_df, labels, feature_names, feature_table = fit_diversity_model(input_root, model_row, args.label_column)
            shap_values = diversity_shap_values(model, x_df, positive_label)
            feature_values = x_df

        importance = write_importance(shap_values, feature_values, feature_names, prefix, args.max_display)
        importance.insert(0, "comparison", comparison)
        importance.insert(1, "model_label", label)
        importance.insert(2, "model_family", model_row["model_family"])
        all_importance.append(importance)
        save_shap_plots(shap_values, feature_values, feature_names, title, prefix, args.max_display)

    pd.concat(all_importance, ignore_index=True).to_csv(
        output_dir / "top_model_shap_importance_top_features.csv", index=False
    )
    print(f"Wrote SHAP outputs to {output_dir}")


if __name__ == "__main__":
    main()

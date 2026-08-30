from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTCOMES = {
    "diagnosis": {
        "metric_prefix": "Diagnosis1",
        "comparisons": ["cd_vs_control", "uc_vs_control", "cd_vs_uc"],
        "base": Path("chain_tcr_immuneml/combined_results/combined_all_model_metrics_table.csv"),
        "output": Path("chain_tcr_immuneml/advanced_immuneml_models/diagnosis_with_svm/combined_all_model_metrics_table.csv"),
    },
    "inflammation": {
        "metric_prefix": "Inflammation1",
        "comparisons": ["cd_inflamed_vs_noninflamed", "uc_inflamed_vs_noninflamed"],
        "base": Path("chain_tcr_immuneml/inflammation_results/combined_results/combined_all_model_metrics_table.csv"),
        "output": Path("chain_tcr_immuneml/advanced_immuneml_models/inflammation_with_svm/combined_all_model_metrics_table.csv"),
    },
    "therapy_response": {
        "metric_prefix": "TherapyResponse1",
        "comparisons": [
            "therapy_combined_response",
            "therapy_antitnf_response",
            "therapy_ustekinumab_response",
            "therapy_vedolizumab_response",
        ],
        "base": Path("chain_tcr_immuneml/therapy_response_results/combined_results/combined_all_model_metrics_table.csv"),
        "output": Path("chain_tcr_immuneml/advanced_immuneml_models/therapy_response_with_svm/combined_all_model_metrics_table.csv"),
    },
}


def normalize_base(base: pd.DataFrame, metric_prefix: str) -> pd.DataFrame:
    table = base.copy()
    if "feature_set" not in table.columns:
        table["feature_set"] = table["sequence_type"].str.upper() + "_cdr3_" + table["kmer"].str.upper()
    if "ml_method" not in table.columns:
        table["ml_method"] = "logistic_regression"
    for col in ["sequence_type", "kmer"]:
        if col not in table.columns:
            table[col] = pd.NA
    keep = [
        "comparison",
        "chain_group",
        "model_family",
        "feature_set",
        "sequence_type",
        "kmer",
        "ml_method",
        f"{metric_prefix}_balanced_accuracy_mean",
        f"{metric_prefix}_mcc_mean",
        f"{metric_prefix}_f1_mean",
        f"{metric_prefix}_roc_auc_mean",
    ]
    return table[keep]


def normalize_svm(svm: pd.DataFrame, metric_prefix: str, comparisons: list[str]) -> pd.DataFrame:
    table = svm[svm["comparison"].isin(comparisons)].copy()
    table = table.rename(
        columns={
            "balanced_accuracy_mean": f"{metric_prefix}_balanced_accuracy_mean",
            "mcc_mean": f"{metric_prefix}_mcc_mean",
            "f1_mean": f"{metric_prefix}_f1_mean",
            "roc_auc_mean": f"{metric_prefix}_roc_auc_mean",
        }
    )
    keep = [
        "comparison",
        "chain_group",
        "model_family",
        "feature_set",
        "sequence_type",
        "kmer",
        "ml_method",
        f"{metric_prefix}_balanced_accuracy_mean",
        f"{metric_prefix}_mcc_mean",
        f"{metric_prefix}_f1_mean",
        f"{metric_prefix}_roc_auc_mean",
    ]
    return table[keep]


def normalize_deeprc(deeprc: pd.DataFrame, metric_prefix: str, comparisons: list[str]) -> pd.DataFrame:
    table = deeprc[deeprc["comparison"].isin(comparisons)].copy()
    table = table.rename(
        columns={
            "balanced_accuracy_mean": f"{metric_prefix}_balanced_accuracy_mean",
            "mcc_mean": f"{metric_prefix}_mcc_mean",
            "f1_mean": f"{metric_prefix}_f1_mean",
            "roc_auc_mean": f"{metric_prefix}_roc_auc_mean",
        }
    )
    keep = [
        "comparison",
        "chain_group",
        "model_family",
        "feature_set",
        "sequence_type",
        "kmer",
        "ml_method",
        f"{metric_prefix}_balanced_accuracy_mean",
        f"{metric_prefix}_mcc_mean",
        f"{metric_prefix}_f1_mean",
        f"{metric_prefix}_roc_auc_mean",
    ]
    return table[keep]


def main() -> None:
    svm = pd.read_csv("chain_tcr_immuneml/advanced_immuneml_models/svm_kmer_results/final_metrics_table.csv")
    deeprc_path = Path("chain_tcr_immuneml/advanced_immuneml_models/deeprc_results/final_metrics_table.csv")
    deeprc = pd.read_csv(deeprc_path) if deeprc_path.exists() else pd.DataFrame()
    rows = []
    for outcome, config in OUTCOMES.items():
        metric_prefix = config["metric_prefix"]
        base = normalize_base(pd.read_csv(config["base"]), metric_prefix)
        svm_subset = normalize_svm(svm, metric_prefix, config["comparisons"])
        deeprc_subset = (
            normalize_deeprc(deeprc, metric_prefix, config["comparisons"])
            if not deeprc.empty
            else pd.DataFrame(columns=svm_subset.columns)
        )
        combined = pd.concat([base, svm_subset, deeprc_subset], ignore_index=True)
        config["output"].parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(config["output"], index=False)
        rows.append(
            {
                "outcome": outcome,
                "output": str(config["output"]),
                "base_rows": len(base),
                "svm_rows": len(svm_subset),
                "deeprc_rows": len(deeprc_subset),
                "combined_rows": len(combined),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv("chain_tcr_immuneml/advanced_immuneml_models/svm_merge_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

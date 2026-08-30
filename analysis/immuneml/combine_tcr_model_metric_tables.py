from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_kmer_metrics(path: Path, metric_prefix: str) -> pd.DataFrame:
    table = pd.read_csv(path).copy()
    table["model_family"] = "cdr3_kmer"
    table["feature_set"] = table["sequence_type"].str.upper() + "_cdr3_" + table["kmer"].str.upper()
    table["ml_method"] = "logistic_regression"
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


def read_diversity_metrics(path: Path, metric_prefix: str) -> pd.DataFrame:
    table = pd.read_csv(path).copy()
    table["sequence_type"] = pd.NA
    table["kmer"] = pd.NA
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--aa", required=True)
    parser.add_argument("--nt", required=True)
    parser.add_argument("--diversity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric-prefix", default="Diagnosis1")
    args = parser.parse_args()

    combined = pd.concat(
        [
            read_kmer_metrics(Path(args.aa), args.metric_prefix),
            read_kmer_metrics(Path(args.nt), args.metric_prefix),
            read_diversity_metrics(Path(args.diversity), args.metric_prefix),
        ],
        ignore_index=True,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(combined.groupby(["comparison", "model_family"]).size().to_string())


if __name__ == "__main__":
    main()

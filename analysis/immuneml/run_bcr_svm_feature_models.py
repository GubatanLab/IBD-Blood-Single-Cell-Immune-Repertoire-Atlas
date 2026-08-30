from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ANALYSES = {
    "diagnosis": {
        "module": "run_bcr_chain_feature_models",
        "comparisons": ("cd_vs_control", "uc_vs_control", "cd_vs_uc"),
        "positive_label": {"cd_vs_control": "CD", "uc_vs_control": "UC", "cd_vs_uc": "CD"},
        "metric_prefix": "Diagnosis1",
        "base_summary": "chain_bcr_immuneml/bcr_chain_feature_model_summary.csv",
        "base_aggregate": "chain_bcr_immuneml/bcr_chain_feature_model_summary_aggregate.csv",
        "output_summary": "chain_bcr_immuneml/bcr_diagnosis_svm_feature_model_summary.csv",
        "output_aggregate": "chain_bcr_immuneml/bcr_diagnosis_svm_feature_model_summary_aggregate.csv",
        "combined_summary": "chain_bcr_immuneml/bcr_diagnosis_lr_svm_feature_model_summary.csv",
        "combined_aggregate": "chain_bcr_immuneml/bcr_diagnosis_lr_svm_feature_model_summary_aggregate.csv",
    },
    "inflammation": {
        "module": "run_bcr_inflammation_feature_models",
        "comparisons": ("cd_inflamed_vs_noninflamed", "uc_inflamed_vs_noninflamed"),
        "positive_label": {
            "cd_inflamed_vs_noninflamed": "Inflamed",
            "uc_inflamed_vs_noninflamed": "Inflamed",
        },
        "metric_prefix": "Inflammation1",
        "base_summary": "chain_bcr_immuneml/bcr_inflammation_feature_model_summary.csv",
        "base_aggregate": "chain_bcr_immuneml/bcr_inflammation_feature_model_summary_aggregate.csv",
        "output_summary": "chain_bcr_immuneml/bcr_inflammation_svm_feature_model_summary.csv",
        "output_aggregate": "chain_bcr_immuneml/bcr_inflammation_svm_feature_model_summary_aggregate.csv",
        "combined_summary": "chain_bcr_immuneml/bcr_inflammation_lr_svm_feature_model_summary.csv",
        "combined_aggregate": "chain_bcr_immuneml/bcr_inflammation_lr_svm_feature_model_summary_aggregate.csv",
    },
    "therapy_response": {
        "module": "run_bcr_therapy_response_feature_models",
        "comparisons": (
            "combined_biologic_nonresponder_vs_responder",
            "anti_tnf_nonresponder_vs_responder",
            "ustekinumab_nonresponder_vs_responder",
            "vedolizumab_nonresponder_vs_responder",
        ),
        "positive_label": {
            "combined_biologic_nonresponder_vs_responder": "NonResponder",
            "anti_tnf_nonresponder_vs_responder": "NonResponder",
            "ustekinumab_nonresponder_vs_responder": "NonResponder",
            "vedolizumab_nonresponder_vs_responder": "NonResponder",
        },
        "metric_prefix": "TherapyResponse",
        "base_summary": "chain_bcr_immuneml/bcr_therapy_response_feature_model_summary.csv",
        "base_aggregate": "chain_bcr_immuneml/bcr_therapy_response_feature_model_summary_aggregate.csv",
        "output_summary": "chain_bcr_immuneml/bcr_therapy_response_svm_feature_model_summary.csv",
        "output_aggregate": "chain_bcr_immuneml/bcr_therapy_response_svm_feature_model_summary_aggregate.csv",
        "combined_summary": "chain_bcr_immuneml/bcr_therapy_response_lr_svm_feature_model_summary.csv",
        "combined_aggregate": "chain_bcr_immuneml/bcr_therapy_response_lr_svm_feature_model_summary_aggregate.csv",
    },
}

ML_MODEL = "immuneML SVM"
ML_MODEL_DETAILS = "immuneML SVM-compatible scikit-learn SVC(kernel=linear,C=1.0,class_weight=balanced) on the same encoded repertoire features"
EVALUATION_DESIGN = "5 stratified random 70/30 train/test splits; aggregate tables report mean/std/count across splits"


def load_feature_sets(
    module: Any,
    input_root: Path,
    isotype_path: Path,
    shm_path: Path,
    chain_group: str,
    comparison: str,
) -> tuple[dict[str, list[dict[str, float]]], np.ndarray, list[str]]:
    if chain_group == "bcr_isotype":
        return module.load_isotype_dataset(input_root, isotype_path, comparison)
    if chain_group == "bcr_shm":
        return module.load_shm_dataset(input_root, shm_path, comparison)
    return module.load_dataset(input_root, chain_group, comparison)


def evaluate_svm(
    features: list[dict[str, float]],
    labels: np.ndarray,
    positive_label: str,
    metric_prefix: str,
    split_count: int,
    seed: int,
) -> list[dict[str, object]]:
    splitter = StratifiedShuffleSplit(
        n_splits=split_count,
        train_size=0.7,
        random_state=seed,
    )
    rows = []
    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        x_train = [features[idx] for idx in train_idx]
        x_test = [features[idx] for idx in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        model = make_pipeline(
            DictVectorizer(sparse=True),
            StandardScaler(with_mean=False),
            SVC(kernel="linear", C=1.0, class_weight="balanced"),
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores = model.decision_function(x_test)
        if list(model[-1].classes_)[1] != positive_label:
            scores = -scores
        binary_true = (y_test == positive_label).astype(int)
        binary_pred = (predictions == positive_label).astype(int)
        rows.append(
            {
                "split": f"split_{split_idx}",
                "positive_label": positive_label,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                f"{metric_prefix}_accuracy": float(accuracy_score(y_test, predictions)),
                f"{metric_prefix}_roc_auc": float(roc_auc_score(binary_true, scores)),
                f"{metric_prefix}_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                f"{metric_prefix}_matthews_corrcoef": float(matthews_corrcoef(y_test, predictions)),
                f"{metric_prefix}_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
            }
        )
    return rows


def feature_sequence_type(feature_set: str, chain_group: str) -> str:
    if chain_group == "bcr_isotype":
        return "isotype"
    if chain_group == "bcr_shm":
        return "shm"
    if feature_set.startswith("aa_"):
        return "amino_acid"
    if feature_set.startswith("nt_"):
        return "nucleotide"
    return "repertoire"


def run_analysis(
    analysis: str,
    input_root: Path,
    isotype_path: Path,
    shm_path: Path,
    split_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = ANALYSES[analysis]
    module = importlib.import_module(config["module"])
    chain_groups = tuple(module.CHAIN_GROUPS) + ("bcr_isotype", "bcr_shm")
    rows: list[dict[str, object]] = []
    for chain_group in chain_groups:
        for comparison in config["comparisons"]:
            print(f"Loading {analysis} {chain_group} {comparison}", flush=True)
            feature_sets, labels, samples = load_feature_sets(module, input_root, isotype_path, shm_path, chain_group, comparison)
            for feature_set, features in feature_sets.items():
                print(f"Running SVM {analysis} {chain_group} {comparison} {feature_set}", flush=True)
                scores = evaluate_svm(
                    features,
                    labels,
                    config["positive_label"][comparison],
                    config["metric_prefix"],
                    split_count,
                    seed,
                )
                for score in scores:
                    rows.append(
                        {
                            "chain_group": chain_group,
                            "comparison": comparison,
                            "feature_set": feature_set,
                            "sequence_type": feature_sequence_type(feature_set, chain_group),
                            "ml_model": ML_MODEL,
                            "ml_model_details": ML_MODEL_DETAILS,
                            "evaluation_design": EVALUATION_DESIGN,
                            "n_samples": int(len(samples)),
                            **score,
                        }
                    )

    detail = pd.DataFrame(rows)
    metric_cols = [
        f"{config['metric_prefix']}_accuracy",
        f"{config['metric_prefix']}_roc_auc",
        f"{config['metric_prefix']}_balanced_accuracy",
        f"{config['metric_prefix']}_matthews_corrcoef",
        f"{config['metric_prefix']}_f1",
    ]
    aggregate = (
        detail.groupby(
            ["chain_group", "comparison", "feature_set", "sequence_type", "ml_model", "ml_model_details", "evaluation_design", "n_samples", "positive_label"],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in aggregate.columns
    ]
    return detail, aggregate


def combine_tables(base_path: Path, svm_df: pd.DataFrame, output_path: Path) -> None:
    base = pd.read_csv(base_path)
    combined = pd.concat([base, svm_df], ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_bcr_immuneml/airr_input")
    parser.add_argument("--isotype-features", default="chain_bcr_immuneml/bcr_isotype_features.csv")
    parser.add_argument("--shm-features", default="chain_bcr_immuneml/bcr_shm_rates_by_patientID.csv")
    parser.add_argument("--analyses", nargs="+", default=list(ANALYSES), choices=list(ANALYSES))
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    isotype_path = Path(args.isotype_features)
    shm_path = Path(args.shm_features)
    for analysis in args.analyses:
        config = ANALYSES[analysis]
        detail, aggregate = run_analysis(analysis, input_root, isotype_path, shm_path, args.split_count, args.seed)
        output_summary = Path(config["output_summary"])
        output_aggregate = Path(config["output_aggregate"])
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(output_summary, index=False)
        aggregate.to_csv(output_aggregate, index=False)
        combine_tables(Path(config["base_summary"]), detail, Path(config["combined_summary"]))
        combine_tables(Path(config["base_aggregate"]), aggregate, Path(config["combined_aggregate"]))
        print(f"Wrote {output_summary}")
        print(f"Wrote {output_aggregate}")
        print(f"Wrote {config['combined_summary']}")
        print(f"Wrote {config['combined_aggregate']}")


if __name__ == "__main__":
    main()

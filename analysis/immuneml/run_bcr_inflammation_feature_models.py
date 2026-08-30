from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
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

from run_bcr_chain_feature_models import (
    CHAIN_GROUPS,
    FEATURE_SETS,
    ISOTYPE_COLUMNS,
    ISOTYPE_FEATURE_SET,
    SHM_COLUMNS,
    SHM_FEATURE_SET,
    load_sample_features,
)


COMPARISONS = {
    "cd_inflamed_vs_noninflamed": "CD",
    "uc_inflamed_vs_noninflamed": "UC",
}
POSITIVE_LABEL = "Inflamed"
INFLAMMATION_LABELS = ("Inflamed", "Noninflamed")
METRIC_PREFIX = "Inflammation1"
ML_MODEL = "LogisticRegression"
ML_MODEL_DETAILS = "DictVectorizer + StandardScaler(with_mean=False) + LogisticRegression(solver=liblinear,max_iter=2000)"


def comparison_metadata(input_root: Path, chain_group: str, comparison: str) -> pd.DataFrame:
    diagnosis = COMPARISONS[comparison]
    metadata = pd.read_csv(input_root / chain_group / "metadata_all.csv")
    metadata = metadata.loc[
        metadata["Diagnosis1"].eq(diagnosis)
        & metadata["Inflammation1"].isin(INFLAMMATION_LABELS)
    ].copy()
    return metadata.reset_index(drop=True)


def load_dataset(
    input_root: Path,
    chain_group: str,
    comparison: str,
) -> tuple[dict[str, list[dict[str, float]]], np.ndarray, list[str]]:
    group_root = input_root / chain_group
    metadata = comparison_metadata(input_root, chain_group, comparison)
    feature_sets: dict[str, list[dict[str, float]]] = {feature_set: [] for feature_set in FEATURE_SETS}
    labels = []
    samples = []

    for row in metadata.itertuples(index=False):
        repertoire_path = group_root / "repertoires" / getattr(row, "filename")
        sample_features = load_sample_features(repertoire_path)
        for feature_set, features in sample_features.items():
            feature_sets[feature_set].append(features)
        labels.append(getattr(row, "Inflammation1"))
        samples.append(getattr(row, "Sample"))

    return feature_sets, np.asarray(labels), samples


def load_isotype_dataset(
    input_root: Path,
    isotype_path: Path,
    comparison: str,
) -> tuple[dict[str, list[dict[str, float]]], np.ndarray, list[str]]:
    metadata = comparison_metadata(input_root, "bcr_heavy", comparison)
    isotype = pd.read_csv(isotype_path)
    merged = metadata.merge(isotype[["SampleID", *ISOTYPE_COLUMNS]], on="SampleID", how="inner")
    if len(merged) != len(metadata):
        missing = sorted(set(metadata["SampleID"]) - set(merged["SampleID"]))
        raise ValueError(f"Missing isotype features for {comparison}: {missing[:10]}")

    features = [
        {f"iso_{column}": float(getattr(row, column)) for column in ISOTYPE_COLUMNS}
        for row in merged.itertuples(index=False)
    ]
    labels = merged["Inflammation1"].to_numpy()
    samples = merged["Sample"].astype(str).tolist()
    return {ISOTYPE_FEATURE_SET: features}, labels, samples


def load_shm_dataset(
    input_root: Path,
    shm_path: Path,
    comparison: str,
) -> tuple[dict[str, list[dict[str, float]]], np.ndarray, list[str]]:
    metadata = comparison_metadata(input_root, "bcr_heavy", comparison)
    shm = pd.read_csv(shm_path)
    merged = metadata.merge(shm[["SampleID", *SHM_COLUMNS]], on="SampleID", how="inner")
    if len(merged) != len(metadata):
        missing = sorted(set(metadata["SampleID"]) - set(merged["SampleID"]))
        raise ValueError(f"Missing SHM features for {comparison}: {missing[:10]}")

    features = [
        {f"shm_{column}": float(getattr(row, column)) for column in SHM_COLUMNS}
        for row in merged.itertuples(index=False)
    ]
    labels = merged["Inflammation1"].to_numpy()
    samples = merged["Sample"].astype(str).tolist()
    return {SHM_FEATURE_SET: features}, labels, samples


def evaluate_dataset(
    features: list[dict[str, float]],
    labels: np.ndarray,
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
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed + split_idx),
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        class_index = list(model[-1].classes_).index(POSITIVE_LABEL)
        probabilities = model.predict_proba(x_test)[:, class_index]
        binary_true = (y_test == POSITIVE_LABEL).astype(int)

        rows.append(
            {
                "split": f"split_{split_idx}",
                "positive_label": POSITIVE_LABEL,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                f"{METRIC_PREFIX}_accuracy": float(accuracy_score(y_test, predictions)),
                f"{METRIC_PREFIX}_roc_auc": float(roc_auc_score(binary_true, probabilities)),
                f"{METRIC_PREFIX}_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                f"{METRIC_PREFIX}_matthews_corrcoef": float(matthews_corrcoef(y_test, predictions)),
                f"{METRIC_PREFIX}_f1": float(f1_score(y_test, predictions, pos_label=POSITIVE_LABEL)),
            }
        )

    return rows


def append_scores(
    rows: list[dict[str, object]],
    chain_group: str,
    comparison: str,
    feature_set: str,
    sequence_type: str,
    sample_count: int,
    scores: list[dict[str, object]],
) -> None:
    for score in scores:
        rows.append(
            {
                "chain_group": chain_group,
                "comparison": comparison,
                "feature_set": feature_set,
                "sequence_type": sequence_type,
                "ml_model": ML_MODEL,
                "ml_model_details": ML_MODEL_DETAILS,
                "n_samples": int(sample_count),
                **score,
            }
        )


def sequence_type_for_feature_set(feature_set: str) -> str:
    if feature_set.startswith("aa_"):
        return "amino_acid"
    if feature_set.startswith("nt_"):
        return "nucleotide"
    return "repertoire"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_bcr_immuneml/airr_input")
    parser.add_argument("--output", default="chain_bcr_immuneml/bcr_inflammation_feature_model_summary.csv")
    parser.add_argument("--isotype-features", default="chain_bcr_immuneml/bcr_isotype_features.csv")
    parser.add_argument("--shm-features", default="chain_bcr_immuneml/bcr_shm_rates_by_patientID.csv")
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    rows: list[dict[str, object]] = []
    for chain_group in CHAIN_GROUPS:
        for comparison in COMPARISONS:
            print(f"Loading {chain_group} {comparison}", flush=True)
            feature_sets, labels, samples = load_dataset(input_root, chain_group, comparison)
            for feature_set, features in feature_sets.items():
                print(f"Running {chain_group} {comparison} {feature_set}", flush=True)
                append_scores(
                    rows,
                    chain_group,
                    comparison,
                    feature_set,
                    sequence_type_for_feature_set(feature_set),
                    len(samples),
                    evaluate_dataset(features, labels, args.split_count, args.seed),
                )

    isotype_path = Path(args.isotype_features)
    if isotype_path.exists():
        for comparison in COMPARISONS:
            print(f"Loading bcr_isotype {comparison}", flush=True)
            feature_sets, labels, samples = load_isotype_dataset(input_root, isotype_path, comparison)
            for feature_set, features in feature_sets.items():
                print(f"Running bcr_isotype {comparison} {feature_set}", flush=True)
                append_scores(
                    rows,
                    "bcr_isotype",
                    comparison,
                    feature_set,
                    "isotype",
                    len(samples),
                    evaluate_dataset(features, labels, args.split_count, args.seed),
                )
    else:
        print(f"Skipping isotype models; {isotype_path} does not exist", flush=True)

    shm_path = Path(args.shm_features)
    if shm_path.exists():
        for comparison in COMPARISONS:
            print(f"Loading bcr_shm {comparison}", flush=True)
            feature_sets, labels, samples = load_shm_dataset(input_root, shm_path, comparison)
            for feature_set, features in feature_sets.items():
                print(f"Running bcr_shm {comparison} {feature_set}", flush=True)
                append_scores(
                    rows,
                    "bcr_shm",
                    comparison,
                    feature_set,
                    "shm",
                    len(samples),
                    evaluate_dataset(features, labels, args.split_count, args.seed),
                )
    else:
        print(f"Skipping SHM models; {shm_path} does not exist", flush=True)

    detail = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)

    metric_cols = [
        f"{METRIC_PREFIX}_accuracy",
        f"{METRIC_PREFIX}_roc_auc",
        f"{METRIC_PREFIX}_balanced_accuracy",
        f"{METRIC_PREFIX}_matthews_corrcoef",
        f"{METRIC_PREFIX}_f1",
    ]
    aggregate = (
        detail.groupby(["chain_group", "comparison", "feature_set", "sequence_type", "ml_model", "ml_model_details", "n_samples", "positive_label"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in aggregate.columns
    ]
    aggregate_output = output.with_name(output.stem + "_aggregate.csv")
    aggregate.to_csv(aggregate_output, index=False)

    print(f"Wrote split-level summary to {output}")
    print(f"Wrote aggregate summary to {aggregate_output}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

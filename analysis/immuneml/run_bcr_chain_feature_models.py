from __future__ import annotations

import argparse
from collections import Counter
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


CHAIN_GROUPS = ("bcr_light", "bcr_heavy", "bcr_heavy_light")
EXTRA_GROUPS = ("bcr_isotype",)
COMPARISONS = ("cd_vs_control", "uc_vs_control", "cd_vs_uc")
POSITIVE_LABEL = {
    "cd_vs_control": "CD",
    "uc_vs_control": "UC",
    "cd_vs_uc": "CD",
}

FEATURE_SETS = (
    "aa_k3",
    "aa_k4",
    "nt_k3",
    "nt_k4",
    "repertoire_metrics",
    "aa_k3_plus_repertoire_metrics",
    "aa_k4_plus_repertoire_metrics",
    "nt_k3_plus_repertoire_metrics",
    "nt_k4_plus_repertoire_metrics",
)

ISOTYPE_FEATURE_SET = "isotype_proportions"
ISOTYPE_COLUMNS = ("IgA_prop", "IgG_prop", "IgM_prop", "IgD_prop")
SHM_FEATURE_SET = "shm_rates"
SHM_COLUMNS = ("shm_rate", "shm_rate_weighted")
ML_MODEL = "LogisticRegression"
ML_MODEL_DETAILS = "DictVectorizer + StandardScaler(with_mean=False) + LogisticRegression(solver=liblinear,max_iter=2000)"


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.sum(values) == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values) / np.sum(sorted_values) - (n + 1)) / n)


def prefixed(features: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in features.items()}


def sample_kmer_frequencies(df: pd.DataFrame, column: str, k: int) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0

    for sequence in df[column].dropna().astype(str).str.upper().drop_duplicates():
        if len(sequence) < k:
            continue
        for idx in range(len(sequence) - k + 1):
            counts[sequence[idx : idx + k]] += 1
            total += 1

    if total == 0:
        return {}
    return {kmer: count / total for kmer, count in counts.items()}


def repertoire_metrics(df: pd.DataFrame) -> dict[str, float]:
    grouped = df.groupby("junction_aa", dropna=True)["duplicate_count"].sum()
    clone_counts = grouped.to_numpy(dtype=float)
    richness = int(clone_counts.size)
    total = float(clone_counts.sum())

    if richness == 0 or total == 0:
        return {
            "richness": 0.0,
            "total_clone_count": 0.0,
            "shannon_entropy": 0.0,
            "shannon_evenness": 0.0,
            "clonality": 0.0,
            "simpson_diversity": 0.0,
            "inverse_simpson": 0.0,
            "gini": 0.0,
            "top_clone_fraction": 0.0,
            "top_10_clone_fraction": 0.0,
            "singleton_fraction": 0.0,
            "mean_clone_count": 0.0,
            "median_clone_count": 0.0,
            "cdr3_aa_length_mean": 0.0,
            "cdr3_aa_length_std": 0.0,
        }

    probabilities = clone_counts / total
    shannon = float(-np.sum(probabilities * np.log(probabilities)))
    evenness = float(shannon / np.log(richness)) if richness > 1 else 0.0
    simpson_concentration = float(np.sum(probabilities**2))
    lengths = grouped.index.to_series().astype(str).str.len().to_numpy(dtype=float)
    top_counts = np.sort(clone_counts)[::-1]

    return {
        "richness": float(richness),
        "total_clone_count": total,
        "shannon_entropy": shannon,
        "shannon_evenness": evenness,
        "clonality": float(1.0 - evenness),
        "simpson_diversity": float(1.0 - simpson_concentration),
        "inverse_simpson": float(1.0 / simpson_concentration) if simpson_concentration > 0 else 0.0,
        "gini": gini(clone_counts),
        "top_clone_fraction": float(top_counts[0] / total),
        "top_10_clone_fraction": float(top_counts[:10].sum() / total),
        "singleton_fraction": float(np.mean(clone_counts == 1)),
        "mean_clone_count": float(np.mean(clone_counts)),
        "median_clone_count": float(np.median(clone_counts)),
        "cdr3_aa_length_mean": float(np.average(lengths, weights=clone_counts)),
        "cdr3_aa_length_std": float(np.sqrt(np.average((lengths - np.average(lengths, weights=clone_counts)) ** 2, weights=clone_counts))),
    }


def load_sample_features(repertoire_path: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(repertoire_path, sep="\t", usecols=["junction", "junction_aa", "duplicate_count"])
    aa_k3 = prefixed(sample_kmer_frequencies(df, "junction_aa", 3), "aa_k3_")
    aa_k4 = prefixed(sample_kmer_frequencies(df, "junction_aa", 4), "aa_k4_")
    nt_k3 = prefixed(sample_kmer_frequencies(df, "junction", 3), "nt_k3_")
    nt_k4 = prefixed(sample_kmer_frequencies(df, "junction", 4), "nt_k4_")
    rep = prefixed(repertoire_metrics(df), "rep_")
    return {
        "aa_k3": aa_k3,
        "aa_k4": aa_k4,
        "nt_k3": nt_k3,
        "nt_k4": nt_k4,
        "repertoire_metrics": rep,
        "aa_k3_plus_repertoire_metrics": {**aa_k3, **rep},
        "aa_k4_plus_repertoire_metrics": {**aa_k4, **rep},
        "nt_k3_plus_repertoire_metrics": {**nt_k3, **rep},
        "nt_k4_plus_repertoire_metrics": {**nt_k4, **rep},
    }


def load_dataset(input_root: Path, chain_group: str, comparison: str) -> tuple[dict[str, list[dict[str, float]]], np.ndarray, list[str]]:
    group_root = input_root / chain_group
    metadata = pd.read_csv(group_root / f"metadata_{comparison}.csv")
    feature_sets: dict[str, list[dict[str, float]]] = {feature_set: [] for feature_set in FEATURE_SETS}
    labels = []
    samples = []

    for row in metadata.itertuples(index=False):
        repertoire_path = group_root / "repertoires" / getattr(row, "filename")
        sample_features = load_sample_features(repertoire_path)
        for feature_set, features in sample_features.items():
            feature_sets[feature_set].append(features)
        labels.append(getattr(row, "Diagnosis1"))
        samples.append(getattr(row, "Sample"))

    return feature_sets, np.asarray(labels), samples


def load_isotype_dataset(
    input_root: Path,
    isotype_path: Path,
    comparison: str,
) -> tuple[dict[str, list[dict[str, float]]], np.ndarray, list[str]]:
    metadata = pd.read_csv(input_root / "bcr_heavy" / f"metadata_{comparison}.csv")
    isotype = pd.read_csv(isotype_path)
    merged = metadata.merge(isotype[["SampleID", *ISOTYPE_COLUMNS]], on="SampleID", how="inner")
    if len(merged) != len(metadata):
        missing = sorted(set(metadata["SampleID"]) - set(merged["SampleID"]))
        raise ValueError(f"Missing isotype features for {comparison}: {missing[:10]}")

    features = [
        {f"iso_{column}": float(getattr(row, column)) for column in ISOTYPE_COLUMNS}
        for row in merged.itertuples(index=False)
    ]
    labels = merged["Diagnosis1"].to_numpy()
    samples = merged["Sample"].astype(str).tolist()
    return {ISOTYPE_FEATURE_SET: features}, labels, samples


def load_shm_dataset(
    input_root: Path,
    shm_path: Path,
    comparison: str,
) -> tuple[dict[str, list[dict[str, float]]], np.ndarray, list[str]]:
    metadata = pd.read_csv(input_root / "bcr_heavy" / f"metadata_{comparison}.csv")
    shm = pd.read_csv(shm_path)
    merged = metadata.merge(shm[["SampleID", *SHM_COLUMNS]], on="SampleID", how="inner")
    if len(merged) != len(metadata):
        missing = sorted(set(metadata["SampleID"]) - set(merged["SampleID"]))
        raise ValueError(f"Missing SHM features for {comparison}: {missing[:10]}")

    features = [
        {f"shm_{column}": float(getattr(row, column)) for column in SHM_COLUMNS}
        for row in merged.itertuples(index=False)
    ]
    labels = merged["Diagnosis1"].to_numpy()
    samples = merged["Sample"].astype(str).tolist()
    return {SHM_FEATURE_SET: features}, labels, samples


def evaluate_dataset(
    features: list[dict[str, float]],
    labels: np.ndarray,
    positive_label: str,
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
        class_index = list(model[-1].classes_).index(positive_label)
        probabilities = model.predict_proba(x_test)[:, class_index]
        binary_true = (y_test == positive_label).astype(int)

        rows.append(
            {
                "split": f"split_{split_idx}",
                "positive_label": positive_label,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "Diagnosis1_accuracy": float(accuracy_score(y_test, predictions)),
                "Diagnosis1_roc_auc": float(roc_auc_score(binary_true, probabilities)),
                "Diagnosis1_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "Diagnosis1_matthews_corrcoef": float(matthews_corrcoef(y_test, predictions)),
                "Diagnosis1_f1": float(f1_score(y_test, predictions, pos_label=positive_label)),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_bcr_immuneml/airr_input")
    parser.add_argument("--output", default="chain_bcr_immuneml/bcr_chain_feature_model_summary.csv")
    parser.add_argument("--isotype-features", default="chain_bcr_immuneml/bcr_isotype_features.csv")
    parser.add_argument("--shm-features", default="chain_bcr_immuneml/bcr_shm_rates_by_patientID.csv")
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    rows = []
    for chain_group in CHAIN_GROUPS:
        for comparison in COMPARISONS:
            print(f"Loading {chain_group} {comparison}", flush=True)
            feature_sets, labels, samples = load_dataset(input_root, chain_group, comparison)
            for feature_set, features in feature_sets.items():
                print(f"Running {chain_group} {comparison} {feature_set}", flush=True)
                for score in evaluate_dataset(
                    features,
                    labels,
                    POSITIVE_LABEL[comparison],
                    args.split_count,
                    args.seed,
                ):
                    rows.append(
                        {
                            "chain_group": chain_group,
                            "comparison": comparison,
                            "feature_set": feature_set,
                            "sequence_type": (
                                "amino_acid"
                                if feature_set.startswith("aa_")
                                else "nucleotide"
                                if feature_set.startswith("nt_")
                                else "repertoire"
                            ),
                            "ml_model": ML_MODEL,
                            "ml_model_details": ML_MODEL_DETAILS,
                            "n_samples": int(len(samples)),
                            **score,
                        }
                    )

    isotype_path = Path(args.isotype_features)
    if isotype_path.exists():
        for comparison in COMPARISONS:
            print(f"Loading bcr_isotype {comparison}", flush=True)
            feature_sets, labels, samples = load_isotype_dataset(input_root, isotype_path, comparison)
            for feature_set, features in feature_sets.items():
                print(f"Running bcr_isotype {comparison} {feature_set}", flush=True)
                for score in evaluate_dataset(
                    features,
                    labels,
                    POSITIVE_LABEL[comparison],
                    args.split_count,
                    args.seed,
                ):
                    rows.append(
                        {
                            "chain_group": "bcr_isotype",
                            "comparison": comparison,
                            "feature_set": feature_set,
                            "sequence_type": "isotype",
                            "ml_model": ML_MODEL,
                            "ml_model_details": ML_MODEL_DETAILS,
                            "n_samples": int(len(samples)),
                            **score,
                        }
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
                for score in evaluate_dataset(
                    features,
                    labels,
                    POSITIVE_LABEL[comparison],
                    args.split_count,
                    args.seed,
                ):
                    rows.append(
                        {
                            "chain_group": "bcr_shm",
                            "comparison": comparison,
                            "feature_set": feature_set,
                            "sequence_type": "shm",
                            "ml_model": ML_MODEL,
                            "ml_model_details": ML_MODEL_DETAILS,
                            "n_samples": int(len(samples)),
                            **score,
                        }
                    )
    else:
        print(f"Skipping SHM models; {shm_path} does not exist", flush=True)

    detail = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)

    metric_cols = [
        "Diagnosis1_accuracy",
        "Diagnosis1_roc_auc",
        "Diagnosis1_balanced_accuracy",
        "Diagnosis1_matthews_corrcoef",
        "Diagnosis1_f1",
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

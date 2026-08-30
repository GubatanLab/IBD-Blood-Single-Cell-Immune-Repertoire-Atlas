from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CHAIN_GROUPS = ("bcr_light", "bcr_heavy", "bcr_heavy_light")
COMPARISONS = ("cd_vs_control", "uc_vs_control", "cd_vs_uc")
K_VALUES = (3, 4)


def sample_kmer_frequencies(repertoire_path: Path, k: int) -> dict[str, float]:
    df = pd.read_csv(repertoire_path, sep="\t", usecols=["junction_aa"])
    counts: Counter[str] = Counter()
    total = 0

    for sequence in df["junction_aa"].dropna().astype(str).drop_duplicates():
        if len(sequence) < k:
            continue
        for idx in range(len(sequence) - k + 1):
            counts[sequence[idx : idx + k]] += 1
            total += 1

    if total == 0:
        return {}
    return {kmer: count / total for kmer, count in counts.items()}


def load_dataset(input_root: Path, chain_group: str, comparison: str, k: int) -> tuple[list[dict[str, float]], np.ndarray, list[str]]:
    group_root = input_root / chain_group
    metadata = pd.read_csv(group_root / f"metadata_{comparison}.csv")

    features = []
    labels = []
    samples = []
    for row in metadata.itertuples(index=False):
        repertoire_path = group_root / "repertoires" / getattr(row, "filename")
        features.append(sample_kmer_frequencies(repertoire_path, k))
        labels.append(getattr(row, "Diagnosis1"))
        samples.append(getattr(row, "Sample"))

    return features, np.asarray(labels), samples


def evaluate_dataset(features: list[dict[str, float]], labels: np.ndarray, split_count: int, seed: int) -> list[dict[str, object]]:
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

        rows.append(
            {
                "split": f"split_{split_idx}",
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "Diagnosis1_accuracy": float(accuracy_score(y_test, predictions)),
                "Diagnosis1_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_bcr_immuneml/airr_input")
    parser.add_argument("--output", default="chain_bcr_immuneml/bcr_chain_kmer_sklearn_summary.csv")
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    rows = []
    for chain_group in CHAIN_GROUPS:
        for comparison in COMPARISONS:
            for k in K_VALUES:
                print(f"Running {chain_group} {comparison} k{k}", flush=True)
                features, labels, samples = load_dataset(input_root, chain_group, comparison, k)
                for score in evaluate_dataset(features, labels, args.split_count, args.seed):
                    rows.append(
                        {
                            "chain_group": chain_group,
                            "comparison": comparison,
                            "kmer": f"k{k}",
                            "sequence_type": "aa",
                            "n_samples": int(len(samples)),
                            **score,
                        }
                    )

    detail = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)

    metric_cols = ["Diagnosis1_accuracy", "Diagnosis1_balanced_accuracy"]
    aggregate = (
        detail.groupby(["chain_group", "comparison", "kmer", "sequence_type", "n_samples"], dropna=False)[metric_cols]
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

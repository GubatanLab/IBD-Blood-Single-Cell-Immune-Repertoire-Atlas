from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_tcr_chain_kmer_models import CHAIN_GROUPS, POSITIVE_LABEL, kmers, read_sample_sequences


COMPARISONS = [
    "cd_vs_control",
    "uc_vs_control",
    "cd_vs_uc",
    "cd_inflamed_vs_noninflamed",
    "uc_inflamed_vs_noninflamed",
    "therapy_combined_response",
    "therapy_antitnf_response",
    "therapy_ustekinumab_response",
    "therapy_vedolizumab_response",
]

LABEL_COLUMN = {
    "cd_vs_control": "Diagnosis1",
    "uc_vs_control": "Diagnosis1",
    "cd_vs_uc": "Diagnosis1",
    "cd_inflamed_vs_noninflamed": "Inflammation1",
    "uc_inflamed_vs_noninflamed": "Inflammation1",
    "therapy_combined_response": "TherapyResponse1",
    "therapy_antitnf_response": "TherapyResponse1",
    "therapy_ustekinumab_response": "TherapyResponse1",
    "therapy_vedolizumab_response": "TherapyResponse1",
}


def sample_counter(sequences: list[str], k: int) -> Counter:
    counts: Counter = Counter()
    for sequence in sequences:
        counts.update(kmers(sequence, k))
    return counts


def counters_to_matrix(counters: list[Counter], vocabulary: dict[str, int]) -> sparse.csr_matrix:
    row_indices = []
    col_indices = []
    values = []
    for row_index, counter in enumerate(counters):
        total = sum(counter.values())
        if total == 0:
            continue
        for kmer, count in counter.items():
            col_index = vocabulary.get(kmer)
            if col_index is None:
                continue
            row_indices.append(row_index)
            col_indices.append(col_index)
            values.append(count / total)
    return sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(counters), len(vocabulary)),
        dtype=np.float64,
    )


def fit_transform(train_counters: list[Counter], test_counters: list[Counter]):
    vocabulary = {
        kmer: index
        for index, kmer in enumerate(sorted({kmer for counter in train_counters for kmer in counter}))
    }
    x_train = counters_to_matrix(train_counters, vocabulary)
    x_test = counters_to_matrix(test_counters, vocabulary)
    scaler = StandardScaler(with_mean=False, with_std=True)
    return scaler.fit_transform(x_train), scaler.transform(x_test), vocabulary


def run_task(
    input_root: Path,
    output_root: Path,
    chain_group: str,
    comparison: str,
    k: int,
    sequence_type: str,
) -> list[dict]:
    label_column = LABEL_COLUMN[comparison]
    group_root = input_root / chain_group
    metadata = pd.read_csv(group_root / f"metadata_{comparison}.csv")
    labels = metadata[label_column].astype(str).to_numpy()
    sample_ids = metadata["Sample"].astype(str).tolist()
    filenames = metadata["filename"].astype(str).tolist()

    sequences = [
        read_sample_sequences(group_root / "repertoires" / filename, sequence_type)
        for filename in filenames
    ]
    counters = [sample_counter(sample_sequences, k) for sample_sequences in sequences]

    task_name = f"{chain_group}_{comparison}_k{k}_{sequence_type}_svm"
    task_root = output_root / task_name
    task_root.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedShuffleSplit(n_splits=5, train_size=0.7, random_state=7000 + k)
    positive_label = POSITIVE_LABEL[comparison]
    rows = []

    for split_index, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        split_root = task_root / f"split_{split_index}"
        split_root.mkdir(parents=True, exist_ok=True)

        train_counters = [counters[i] for i in train_idx]
        test_counters = [counters[i] for i in test_idx]
        x_train, x_test, vocabulary = fit_transform(train_counters, test_counters)
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        model = LinearSVC(C=1.0, class_weight="balanced", random_state=8000 + split_index, max_iter=10000)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        scores = model.decision_function(x_test)
        class_names = list(model.classes_)
        if class_names[1] != positive_label:
            scores = -scores
        y_test_binary = (y_test == positive_label).astype(int)
        y_pred_binary = (y_pred == positive_label).astype(int)

        score_row = {
            "instruction": task_name,
            "chain_group": chain_group,
            "comparison": comparison,
            "model_family": "cdr3_kmer",
            "feature_set": f"{sequence_type.upper()}_cdr3_k{k}",
            "sequence_type": sequence_type,
            "kmer": f"k{k}",
            "ml_method": "svm_linear",
            "split_index": split_index,
            "positive_label": positive_label,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(vocabulary)),
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "f1": f1_score(y_test_binary, y_pred_binary, zero_division=0),
            "roc_auc": roc_auc_score(y_test_binary, scores),
        }
        rows.append(score_row)

        prediction_rows = []
        for local_index, sample_index in enumerate(test_idx):
            prediction_rows.append(
                {
                    "Sample": sample_ids[sample_index],
                    "filename": filenames[sample_index],
                    "true": y_test[local_index],
                    "pred": y_pred[local_index],
                    f"score_{positive_label}": scores[local_index],
                }
            )
        pd.DataFrame(prediction_rows).to_csv(split_root / "test_predictions.csv", index=False)
        pd.DataFrame([score_row]).to_csv(split_root / "ml_score.csv", index=False)
        with (split_root / "model_info.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "classes": class_names,
                    "positive_label": positive_label,
                    "n_features": len(vocabulary),
                    "train_samples": [sample_ids[i] for i in train_idx],
                    "test_samples": [sample_ids[i] for i in test_idx],
                },
                handle,
                indent=2,
            )

    pd.DataFrame(rows).to_csv(task_root / "ml_scores.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_tcr_immuneml/airr_input")
    parser.add_argument("--output-root", default="chain_tcr_immuneml/advanced_immuneml_models/svm_kmer_results")
    parser.add_argument("--sequence-types", nargs="+", default=["aa", "nt"], choices=["aa", "nt"])
    parser.add_argument("--comparisons", nargs="+", default=COMPARISONS)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for comparison in args.comparisons:
        for chain_group in CHAIN_GROUPS:
            for sequence_type in args.sequence_types:
                for k in (3, 4):
                    print(f"Running SVM {comparison} {chain_group} {sequence_type} k{k}")
                    all_rows.extend(run_task(input_root, output_root, chain_group, comparison, k, sequence_type))

    detail = pd.DataFrame(all_rows)
    detail.to_csv(output_root / "all_split_scores.csv", index=False)
    aggregate = (
        detail.groupby(
            ["comparison", "chain_group", "model_family", "feature_set", "sequence_type", "kmer", "ml_method"],
            dropna=False,
        )[["accuracy", "balanced_accuracy", "mcc", "f1", "roc_auc", "n_train", "n_test", "n_features"]]
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
    final = aggregate[
        [
            "comparison",
            "chain_group",
            "model_family",
            "feature_set",
            "sequence_type",
            "kmer",
            "ml_method",
            "balanced_accuracy_mean",
            "mcc_mean",
            "f1_mean",
            "roc_auc_mean",
        ]
    ].sort_values(["comparison", "chain_group", "sequence_type", "kmer"])
    final.to_csv(output_root / "final_metrics_table.csv", index=False)
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()

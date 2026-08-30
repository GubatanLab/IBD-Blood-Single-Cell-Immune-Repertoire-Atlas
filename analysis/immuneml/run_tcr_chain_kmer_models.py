from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
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


def kmers(sequence: str, k: int) -> list[str]:
    sequence = sequence.strip()
    if len(sequence) < k:
        return []
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def read_sample_sequences(repertoire_path: Path, sequence_type: str) -> list[str]:
    sequence_column = "junction_aa" if sequence_type == "aa" else "junction"
    df = pd.read_csv(repertoire_path, sep="\t", usecols=[sequence_column])
    seqs = df[sequence_column].dropna().astype(str)
    seqs = seqs[seqs.str.len() > 0]
    if sequence_type == "aa":
        seqs = seqs[~seqs.str.contains(r"\*", regex=True)]
    return seqs.tolist()


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
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, vocabulary


def run_task(
    input_root: Path,
    output_root: Path,
    chain_group: str,
    comparison: str,
    k: int,
    sequence_type: str,
    label_column: str,
) -> list[dict]:
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

    splitter = StratifiedShuffleSplit(n_splits=5, train_size=0.7, random_state=1000 + k)
    task_name = f"{chain_group}_{comparison}_k{k}_{sequence_type}"
    task_root = output_root / task_name
    task_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_index, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        split_root = task_root / f"split_{split_index}"
        split_root.mkdir(parents=True, exist_ok=True)

        train_counters = [counters[i] for i in train_idx]
        test_counters = [counters[i] for i in test_idx]
        x_train, x_test, vocabulary = fit_transform(train_counters, test_counters)
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        model = LogisticRegression(max_iter=5000, solver="liblinear", random_state=2000 + split_index)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        positive_label = POSITIVE_LABEL[comparison]
        positive_index = list(model.classes_).index(positive_label)
        y_test_binary = (y_test == positive_label).astype(int)
        y_pred_binary = (y_pred == positive_label).astype(int)

        score_row = {
            "instruction": task_name,
            "chain_group": chain_group,
            "comparison": comparison,
            "kmer": f"k{k}",
            "sequence_type": sequence_type,
            "split_index": split_index,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(vocabulary)),
            f"{label_column}_accuracy": accuracy_score(y_test, y_pred),
            f"{label_column}_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            f"{label_column}_mcc": matthews_corrcoef(y_test, y_pred),
            f"{label_column}_f1": f1_score(y_test_binary, y_pred_binary, zero_division=0),
            f"{label_column}_roc_auc": roc_auc_score(y_test_binary, probabilities[:, positive_index]),
        }
        rows.append(score_row)

        prediction_rows = []
        class_names = list(model.classes_)
        for local_index, sample_index in enumerate(test_idx):
            pred_row = {
                "Sample": sample_ids[sample_index],
                "filename": filenames[sample_index],
                f"{label_column}_true": y_test[local_index],
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
    parser.add_argument("--output-root", default="chain_tcr_immuneml/sklearn_results")
    parser.add_argument("--sequence-type", choices=["aa", "nt"], default="aa")
    parser.add_argument("--label-column", default="Diagnosis1")
    parser.add_argument("--comparisons", nargs="+", default=COMPARISONS)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for chain_group in CHAIN_GROUPS:
        for comparison in args.comparisons:
            for k in (3, 4):
                print(f"Running {chain_group} {comparison} k{k} {args.sequence_type}")
                all_rows.extend(
                    run_task(
                        input_root,
                        output_root,
                        chain_group,
                        comparison,
                        k,
                        args.sequence_type,
                        args.label_column,
                    )
                )

    detail = pd.DataFrame(all_rows)
    detail.to_csv(output_root / "all_split_scores.csv", index=False)
    metric_prefix = args.label_column
    aggregate = (
        detail.groupby(["chain_group", "comparison", "kmer", "sequence_type"], dropna=False)
        [[
            f"{metric_prefix}_accuracy",
            f"{metric_prefix}_balanced_accuracy",
            f"{metric_prefix}_mcc",
            f"{metric_prefix}_f1",
            f"{metric_prefix}_roc_auc",
            "n_train",
            "n_test",
            "n_features",
        ]]
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
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

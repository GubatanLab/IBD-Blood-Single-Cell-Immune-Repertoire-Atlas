from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from run_tcr_chain_kmer_models import CHAIN_GROUPS, POSITIVE_LABEL
from run_tcr_chain_kmer_svm_models import COMPARISONS, LABEL_COLUMN


DEFAULT_PYTHON = Path(r"C:/path/to/private-python-environment\envs\immuneml_py310_clean\python.exe")
RUNNER = Path("scripts/run_immuneml_numpy_compat.py")


def as_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def write_yaml_like(spec: dict, path: Path) -> None:
    try:
        import yaml

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(spec, handle, sort_keys=False)
    except Exception:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=2)


def make_spec(
    input_root: Path,
    chain_group: str,
    comparison: str,
    split_count: int,
    selection_split_count: int,
    sample_n_sequences: int,
    n_updates: int,
) -> dict:
    label_column = LABEL_COLUMN[comparison]
    group_root = input_root / chain_group
    return {
        "definitions": {
            "datasets": {
                "d": {
                    "format": "AIRR",
                    "params": {
                        "path": as_posix(group_root / "repertoires"),
                        "metadata_file": as_posix(group_root / f"metadata_{comparison}.csv"),
                        "is_repertoire": True,
                        "paired": False,
                        "separator": "\t",
                        "region_type": "IMGT_CDR3",
                        "label_columns": None,
                        "import_productive": True,
                        "import_with_stop_codon": False,
                        "import_out_of_frame": False,
                        "import_illegal_characters": False,
                        "import_empty_aa_sequences": False,
                        "import_empty_nt_sequences": True,
                        "import_unknown_productivity": True,
                    },
                }
            },
            "encodings": {"e": {"DeepRC": {}}},
            "ml_methods": {
                "m": {
                    "DeepRC": {
                        "validation_part": 0.2,
                        "add_positional_information": True,
                        "kernel_size": 9,
                        "n_kernels": 8,
                        "n_additional_convs": 0,
                        "n_attention_network_layers": 1,
                        "n_attention_network_units": 8,
                        "n_output_network_units": 8,
                        "consider_seq_counts": False,
                        "sequence_reduction_fraction": 0.1,
                        "reduction_mb_size": 1000,
                        "n_updates": n_updates,
                        "n_torch_threads": 1,
                        "learning_rate": 0.0001,
                        "l1_weight_decay": 0,
                        "l2_weight_decay": 0,
                        "evaluate_at": max(1, n_updates),
                        "sample_n_sequences": sample_n_sequences,
                        "training_batch_size": 2,
                        "n_workers": 1,
                        "sequence_counts_scaling_fn": "log",
                        "keep_dataset_in_ram": False,
                        "pytorch_device_name": "cpu",
                    }
                }
            },
        },
        "instructions": {
            "i": {
                "type": "TrainMLModel",
                "dataset": "d",
                "settings": [{"encoding": "e", "ml_method": "m", "preprocessing": None}],
                "assessment": {
                    "split_strategy": "random",
                    "split_count": split_count,
                    "training_percentage": 0.7,
                },
                "selection": {
                    "split_strategy": "random",
                    "split_count": selection_split_count,
                    "training_percentage": 0.7,
                },
                "labels": [label_column],
                "strategy": "GridSearch",
                "metrics": ["accuracy", "balanced_accuracy", "auc"],
                "optimization_metric": "balanced_accuracy",
                "number_of_processes": 1,
                "reports": [],
                "refit_optimal_model": False,
                "export_all_ml_settings": False,
                "sequence_type": "AMINO_ACID",
                "region_type": "IMGT_CDR3",
            }
        },
    }


def split_number(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"split_(\d+)", part)
        if match:
            return int(match.group(1))
    return 0


def safe_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true_binary)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true_binary, scores))
    except ValueError:
        return float("nan")


def extract_scores(job_dir: Path, chain_group: str, comparison: str) -> pd.DataFrame:
    label_column = LABEL_COLUMN[comparison]
    positive_label = POSITIVE_LABEL[comparison]
    prediction_files = sorted(
        path
        for path in (job_dir / "i").glob("split_*/**/test_predictions.csv")
        if path.parent.name.endswith("_optimal")
    )
    rows = []
    for prediction_file in prediction_files:
        predictions = pd.read_csv(prediction_file)
        true_col = f"{label_column}_true_class"
        pred_col = f"{label_column}_predicted_class"
        prob_col = f"{label_column}_{positive_label}_proba"
        if prob_col not in predictions.columns:
            candidates = [col for col in predictions.columns if col.endswith(f"_{positive_label}_proba")]
            prob_col = candidates[0] if candidates else None

        y_true = predictions[true_col].astype(str).to_numpy()
        y_pred = predictions[pred_col].astype(str).to_numpy()
        y_true_binary = (y_true == positive_label).astype(int)
        y_pred_binary = (y_pred == positive_label).astype(int)
        scores = (
            predictions[prob_col].astype(float).to_numpy()
            if prob_col is not None
            else y_pred_binary.astype(float)
        )
        rows.append(
            {
                "instruction": job_dir.name,
                "chain_group": chain_group,
                "comparison": comparison,
                "model_family": "deeprc",
                "feature_set": "aa_sequence_bag",
                "sequence_type": "aa",
                "kmer": pd.NA,
                "ml_method": "deeprc",
                "split_index": split_number(prediction_file),
                "positive_label": positive_label,
                "n_test": int(len(predictions)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                "roc_auc": safe_auc(y_true_binary, scores),
                "prediction_file": str(prediction_file),
            }
        )
    return pd.DataFrame(rows)


def run_job(
    python_exe: Path,
    spec_path: Path,
    job_dir: Path,
    log_path: Path,
) -> int:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    cmd = [str(python_exe), str(RUNNER), str(spec_path), str(job_dir), "--logging", "ERROR"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return int(process.returncode)


def aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    grouped = (
        detail.groupby(
            ["comparison", "chain_group", "model_family", "feature_set", "sequence_type", "kmer", "ml_method"],
            dropna=False,
        )[["accuracy", "balanced_accuracy", "mcc", "f1", "roc_auc", "n_test"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_tcr_immuneml/airr_input")
    parser.add_argument("--output-root", default="drb")
    parser.add_argument(
        "--publish-root",
        default="chain_tcr_immuneml/advanced_immuneml_models/deeprc_results",
    )
    parser.add_argument("--python-exe", default=str(DEFAULT_PYTHON))
    parser.add_argument("--comparisons", nargs="+", default=COMPARISONS)
    parser.add_argument("--chains", nargs="+", default=CHAIN_GROUPS)
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--selection-split-count", type=int, default=1)
    parser.add_argument("--sample-n-sequences", type=int, default=100)
    parser.add_argument("--n-updates", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    publish_root = Path(args.publish_root)
    specs_dir = output_root / "specs"
    jobs_dir = output_root / "jobs"
    logs_dir = output_root / "logs"
    for directory in (specs_dir, jobs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    publish_root.mkdir(parents=True, exist_ok=True)

    planned = [(comparison, chain) for comparison in args.comparisons for chain in args.chains]
    if args.max_jobs is not None:
        planned = planned[: args.max_jobs]

    def run_planned_job(job_index: int, comparison: str, chain_group: str) -> tuple[dict, pd.DataFrame]:
        job_id = f"j{job_index:03d}_{comparison}_{chain_group}".replace("therapy_", "tx_")
        spec_path = specs_dir / f"{job_id}.yml"
        job_dir = jobs_dir / job_id
        log_path = logs_dir / f"{job_id}.log"
        score_path = job_dir / "extracted_split_scores.csv"

        if args.rerun or not score_path.exists():
            spec = make_spec(
                input_root,
                chain_group,
                comparison,
                args.split_count,
                args.selection_split_count,
                args.sample_n_sequences,
                args.n_updates,
            )
            write_yaml_like(spec, spec_path)
            job_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running DeepRC {job_index}/{len(planned)}: {comparison} {chain_group}", flush=True)
            return_code = run_job(Path(args.python_exe), spec_path, job_dir, log_path)
        else:
            print(f"Using existing DeepRC scores: {comparison} {chain_group}", flush=True)
            return_code = 0

        scores = extract_scores(job_dir, chain_group, comparison)
        if not scores.empty:
            scores.to_csv(score_path, index=False)

        job_row = {
            "job_id": job_id,
            "comparison": comparison,
            "chain_group": chain_group,
            "return_code": return_code,
            "n_completed_splits": int(len(scores)),
            "spec": str(spec_path),
            "job_dir": str(job_dir),
            "log": str(log_path),
        }
        return job_row, scores

    results: list[tuple[dict, pd.DataFrame]] = []
    if args.workers <= 1:
        for job_index, (comparison, chain_group) in enumerate(planned, start=1):
            results.append(run_planned_job(job_index, comparison, chain_group))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_planned_job, job_index, comparison, chain_group)
                for job_index, (comparison, chain_group) in enumerate(planned, start=1)
            ]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda item: item[0]["job_id"])
    job_rows = [item[0] for item in results]
    all_scores = [item[1] for item in results if not item[1].empty]

    job_summary = pd.DataFrame(job_rows)
    job_summary.to_csv(output_root / "job_summary.csv", index=False)
    job_summary.to_csv(publish_root / "job_summary.csv", index=False)

    detail = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    detail.to_csv(output_root / "all_split_scores.csv", index=False)
    detail.to_csv(publish_root / "all_split_scores.csv", index=False)
    aggregate_scores = aggregate(detail)
    aggregate_scores.to_csv(output_root / "aggregate_scores.csv", index=False)
    aggregate_scores.to_csv(publish_root / "aggregate_scores.csv", index=False)

    metric_cols = [
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
    final = aggregate_scores[metric_cols].sort_values(["comparison", "chain_group"]) if not aggregate_scores.empty else pd.DataFrame(columns=metric_cols)
    final.to_csv(output_root / "final_metrics_table.csv", index=False)
    final.to_csv(publish_root / "final_metrics_table.csv", index=False)
    print(job_summary.to_string(index=False))
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()

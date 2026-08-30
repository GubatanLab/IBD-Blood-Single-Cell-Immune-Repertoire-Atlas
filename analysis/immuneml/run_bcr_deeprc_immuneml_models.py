from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


CHAIN_GROUPS = ("bcr_light", "bcr_heavy", "bcr_heavy_light")
DEFAULT_PYTHON = Path(r"C:/path/to/private-python-environment\python.exe")
RUNNER = Path("scripts/run_immuneml_numpy_compat.py")

OUTCOME_CONFIGS = {
    "diagnosis": {
        "comparisons": {
            "cd_vs_control": {"label_column": "Diagnosis1", "positive_label": "CD", "filter": lambda df: df["Diagnosis1"].isin(["CD", "Control"])},
            "uc_vs_control": {"label_column": "Diagnosis1", "positive_label": "UC", "filter": lambda df: df["Diagnosis1"].isin(["UC", "Control"])},
            "cd_vs_uc": {"label_column": "Diagnosis1", "positive_label": "CD", "filter": lambda df: df["Diagnosis1"].isin(["CD", "UC"])},
        },
        "metric_prefix": "Diagnosis1",
    },
    "inflammation": {
        "comparisons": {
            "cd_inflamed_vs_noninflamed": {"label_column": "Inflammation1", "positive_label": "Inflamed", "filter": lambda df: df["Diagnosis1"].eq("CD") & df["Inflammation1"].isin(["Inflamed", "Noninflamed"])},
            "uc_inflamed_vs_noninflamed": {"label_column": "Inflammation1", "positive_label": "Inflamed", "filter": lambda df: df["Diagnosis1"].eq("UC") & df["Inflammation1"].isin(["Inflamed", "Noninflamed"])},
        },
        "metric_prefix": "Inflammation1",
    },
    "therapy_response": {
        "comparisons": {
            "combined_biologic_nonresponder_vs_responder": {"label_column": "TherapyResponse", "positive_label": "NonResponder", "biologics": ("AntiTNF", "Ustekinumab", "Vedolizumab")},
            "anti_tnf_nonresponder_vs_responder": {"label_column": "TherapyResponse", "positive_label": "NonResponder", "biologics": ("AntiTNF",)},
            "ustekinumab_nonresponder_vs_responder": {"label_column": "TherapyResponse", "positive_label": "NonResponder", "biologics": ("Ustekinumab",)},
            "vedolizumab_nonresponder_vs_responder": {"label_column": "TherapyResponse", "positive_label": "NonResponder", "biologics": ("Vedolizumab",)},
        },
        "metric_prefix": "TherapyResponse",
    },
}


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


def comparison_metadata(input_root: Path, chain_group: str, outcome: str, comparison: str) -> pd.DataFrame:
    config = OUTCOME_CONFIGS[outcome]["comparisons"][comparison]
    metadata = pd.read_csv(input_root / chain_group / "metadata_all.csv")
    if outcome == "therapy_response":
        metadata = metadata.loc[
            metadata["Biologic"].isin(config["biologics"])
            & metadata["Inflammation1"].isin(["Inflamed", "Noninflamed"])
        ].copy()
        metadata["TherapyResponse"] = metadata["Inflammation1"].map(
            {"Inflamed": "NonResponder", "Noninflamed": "Responder"}
        )
    else:
        metadata = metadata.loc[config["filter"](metadata)].copy()
    return metadata.reset_index(drop=True)


def write_metadata(input_root: Path, metadata_root: Path, outcome: str, comparison: str, chain_group: str) -> Path:
    metadata = comparison_metadata(input_root, chain_group, outcome, comparison)
    out_dir = metadata_root / outcome / chain_group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metadata_{comparison}.csv"
    metadata.to_csv(out_path, index=False)
    return out_path


def make_spec(
    input_root: Path,
    metadata_path: Path,
    chain_group: str,
    outcome: str,
    comparison: str,
    split_count: int,
    selection_split_count: int,
    sample_n_sequences: int,
    n_updates: int,
) -> dict:
    config = OUTCOME_CONFIGS[outcome]["comparisons"][comparison]
    label_column = config["label_column"]
    group_root = input_root / chain_group
    return {
        "definitions": {
            "datasets": {
                "d": {
                    "format": "AIRR",
                    "params": {
                        "path": as_posix(group_root / "repertoires"),
                        "metadata_file": as_posix(metadata_path),
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
                "assessment": {"split_strategy": "random", "split_count": split_count, "training_percentage": 0.7},
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


def extract_scores(job_dir: Path, outcome: str, comparison: str, chain_group: str) -> pd.DataFrame:
    config = OUTCOME_CONFIGS[outcome]["comparisons"][comparison]
    label_column = config["label_column"]
    positive_label = config["positive_label"]
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
        scores = predictions[prob_col].astype(float).to_numpy() if prob_col else y_pred_binary.astype(float)
        rows.append(
            {
                "outcome": outcome,
                "chain_group": chain_group,
                "comparison": comparison,
                "model_family": "deeprc",
                "feature_set": "aa_sequence_bag",
                "sequence_type": "amino_acid",
                "kmer": pd.NA,
                "ml_model": "DeepRC",
                "ml_model_details": "immuneML DeepRC on amino-acid CDR3 sequence bags",
                "split_index": split_number(prediction_file),
                "positive_label": positive_label,
                "n_test": int(len(predictions)),
                f"{OUTCOME_CONFIGS[outcome]['metric_prefix']}_accuracy": float(accuracy_score(y_true, y_pred)),
                f"{OUTCOME_CONFIGS[outcome]['metric_prefix']}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                f"{OUTCOME_CONFIGS[outcome]['metric_prefix']}_matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
                f"{OUTCOME_CONFIGS[outcome]['metric_prefix']}_f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                f"{OUTCOME_CONFIGS[outcome]['metric_prefix']}_roc_auc": safe_auc(y_true_binary, scores),
                "prediction_file": str(prediction_file),
            }
        )
    return pd.DataFrame(rows)


def run_job(python_exe: Path, spec_path: Path, job_dir: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    cmd = [str(python_exe), str(RUNNER), str(spec_path), str(job_dir), "--logging", "ERROR"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return int(process.returncode)


def aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    metric_cols = [col for col in detail.columns if col.endswith(("_accuracy", "_balanced_accuracy", "_matthews_corrcoef", "_f1", "_roc_auc", "n_test"))]
    grouped = (
        detail.groupby(
            ["outcome", "comparison", "chain_group", "model_family", "feature_set", "sequence_type", "kmer", "ml_model", "ml_model_details"],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_bcr_immuneml/airr_input")
    parser.add_argument("--output-root", default="chain_bcr_immuneml/deeprc_work")
    parser.add_argument("--publish-root", default="chain_bcr_immuneml/deeprc_results")
    parser.add_argument("--python-exe", default=str(DEFAULT_PYTHON))
    parser.add_argument("--outcomes", nargs="+", default=list(OUTCOME_CONFIGS))
    parser.add_argument("--comparisons", nargs="+", default=None)
    parser.add_argument("--chains", nargs="+", default=list(CHAIN_GROUPS))
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
    metadata_root = output_root / "metadata"
    for directory in (specs_dir, jobs_dir, logs_dir, metadata_root, publish_root):
        directory.mkdir(parents=True, exist_ok=True)

    planned = []
    for outcome in args.outcomes:
        comparisons = args.comparisons or list(OUTCOME_CONFIGS[outcome]["comparisons"])
        for comparison in comparisons:
            if comparison not in OUTCOME_CONFIGS[outcome]["comparisons"]:
                continue
            for chain_group in args.chains:
                planned.append((outcome, comparison, chain_group))
    if args.max_jobs is not None:
        planned = planned[: args.max_jobs]

    def run_planned_job(job_index: int, outcome: str, comparison: str, chain_group: str) -> tuple[dict, pd.DataFrame]:
        job_id = f"j{job_index:03d}_{outcome}_{comparison}_{chain_group}".replace("therapy_response_", "tx_")
        spec_path = specs_dir / f"{job_id}.yml"
        job_dir = jobs_dir / job_id
        log_path = logs_dir / f"{job_id}.log"
        score_path = job_dir / "extracted_split_scores.csv"

        if args.rerun or not score_path.exists():
            metadata_path = write_metadata(input_root, metadata_root, outcome, comparison, chain_group)
            spec = make_spec(input_root, metadata_path, chain_group, outcome, comparison, args.split_count, args.selection_split_count, args.sample_n_sequences, args.n_updates)
            write_yaml_like(spec, spec_path)
            job_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running DeepRC {job_index}/{len(planned)}: {outcome} {comparison} {chain_group}", flush=True)
            return_code = run_job(Path(args.python_exe), spec_path, job_dir, log_path)
        else:
            print(f"Using existing DeepRC scores: {outcome} {comparison} {chain_group}", flush=True)
            return_code = 0

        scores = extract_scores(job_dir, outcome, comparison, chain_group)
        if not scores.empty:
            scores.to_csv(score_path, index=False)
        return {
            "job_id": job_id,
            "outcome": outcome,
            "comparison": comparison,
            "chain_group": chain_group,
            "return_code": return_code,
            "n_completed_splits": int(len(scores)),
            "spec": str(spec_path),
            "job_dir": str(job_dir),
            "log": str(log_path),
        }, scores

    results = []
    if args.workers <= 1:
        for job_index, (outcome, comparison, chain_group) in enumerate(planned, start=1):
            results.append(run_planned_job(job_index, outcome, comparison, chain_group))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_planned_job, job_index, outcome, comparison, chain_group)
                for job_index, (outcome, comparison, chain_group) in enumerate(planned, start=1)
            ]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda item: item[0]["job_id"])
    job_summary = pd.DataFrame([item[0] for item in results])
    nonempty_scores = [item[1] for item in results if not item[1].empty]
    detail = pd.concat(nonempty_scores, ignore_index=True) if nonempty_scores else pd.DataFrame()
    aggregate_scores = aggregate(detail)

    job_summary.to_csv(output_root / "job_summary.csv", index=False)
    detail.to_csv(output_root / "all_split_scores.csv", index=False)
    aggregate_scores.to_csv(output_root / "aggregate_scores.csv", index=False)
    job_summary.to_csv(publish_root / "job_summary.csv", index=False)
    detail.to_csv(publish_root / "all_split_scores.csv", index=False)
    aggregate_scores.to_csv(publish_root / "aggregate_scores.csv", index=False)
    print(job_summary.to_string(index=False))
    if not aggregate_scores.empty:
        print(aggregate_scores.to_string(index=False))


if __name__ == "__main__":
    main()

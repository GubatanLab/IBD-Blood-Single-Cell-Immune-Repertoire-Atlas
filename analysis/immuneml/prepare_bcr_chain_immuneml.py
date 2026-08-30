from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import pandas as pd
import rdata
import yaml


CHAIN_GROUPS = {
    "bcr_light": ["IGK", "IGL"],
    "bcr_heavy": ["IGH"],
    "bcr_heavy_light": ["IGH", "IGK", "IGL"],
}

COMPARISONS = {
    "cd_vs_control": ["CD", "Control"],
    "uc_vs_control": ["UC", "Control"],
    "cd_vs_uc": ["CD", "UC"],
}

AA_PATTERN = re.compile(r"^[A-Z]+$")


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def infer_locus(call: object) -> str | None:
    if pd.isna(call):
        return None
    match = re.match(r"^(IG[HKL])", str(call))
    return match.group(1) if match else None


def clean_airr_table(df: pd.DataFrame, loci: list[str], sample: str) -> pd.DataFrame:
    airr = pd.DataFrame(
        {
            "sequence_id": [f"{sample}_{idx}" for idx in range(1, len(df) + 1)],
            "sequence": df["Sequence"].fillna(df["CDR3.nt"]),
            "junction": df["CDR3.nt"],
            "junction_aa": df["CDR3.aa"],
            "v_call": df["V.name"],
            "d_call": df["D.name"],
            "j_call": df["J.name"],
            "duplicate_count": df["Clones"].fillna(1).astype(int),
            "productive": True,
        }
    )
    airr["locus"] = airr["v_call"].map(infer_locus)

    out = airr[airr["locus"].isin(loci)].copy()
    out = out.dropna(subset=["junction", "junction_aa"])
    out["junction"] = out["junction"].astype(str)
    out["junction_aa"] = out["junction_aa"].astype(str)
    out = out[out["junction"].str.len() > 0]
    out = out[out["junction_aa"].str.len() > 0]
    out = out[out["junction_aa"].str.match(AA_PATTERN)]
    out = out[~out["junction_aa"].str.contains(r"\*", regex=True)]
    return out


def write_metadata(metadata: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(path, index=False)


def build_spec(output_root: Path) -> dict:
    datasets = {}

    for group in CHAIN_GROUPS:
        for comparison in COMPARISONS:
            dataset_name = f"{group}_{comparison}"
            group_root = output_root / "airr_input" / group
            datasets[dataset_name] = {
                "format": "AIRR",
                "params": {
                    "import_empty_aa_sequences": False,
                    "import_empty_nt_sequences": True,
                    "import_illegal_characters": False,
                    "import_out_of_frame": False,
                    "import_productive": True,
                    "import_unknown_productivity": True,
                    "import_with_stop_codon": False,
                    "is_repertoire": True,
                    "label_columns": None,
                    "metadata_file": as_posix(group_root / f"metadata_{comparison}.csv"),
                    "paired": False,
                    "path": as_posix(group_root / "repertoires"),
                    "region_type": "IMGT_CDR3",
                    "separator": "\t",
                },
            }

    encodings = {}
    for sequence_suffix, sequence_type in (("aa", "AMINO_ACID"), ("nt", "NUCLEOTIDE")):
        for k in (3, 4):
            encoding_name = f"k{k}_{sequence_suffix}"
            encodings[encoding_name] = {
                "KmerFrequency": {
                    "k": k,
                    "k_left": 1,
                    "k_right": 1,
                    "max_gap": 0,
                    "min_gap": 0,
                    "name": encoding_name,
                    "normalization_type": "relative_frequency",
                    "reads": "unique",
                    "region_type": "imgt_cdr3",
                    "scale_to_unit_variance": True,
                    "scale_to_zero_mean": False,
                    "sequence_encoding": "CONTINUOUS_KMER",
                    "sequence_type": sequence_type,
                }
            }

    instructions = {}
    for group in CHAIN_GROUPS:
        for comparison in COMPARISONS:
            for sequence_suffix, sequence_type in (("aa", "AMINO_ACID"), ("nt", "NUCLEOTIDE")):
                for k in (3, 4):
                    encoding_name = f"k{k}_{sequence_suffix}"
                    instruction_name = f"{group}_{comparison}_{encoding_name}"
                    instructions[instruction_name] = {
                        "type": "TrainMLModel",
                        "settings": [
                            {
                                "encoding": encoding_name,
                                "ml_method": "logistic_regression",
                                "preprocessing": None,
                            }
                        ],
                        "assessment": {
                            "split_strategy": "random",
                            "split_count": 5,
                            "training_percentage": 0.7,
                        },
                        "selection": {
                            "split_strategy": "random",
                            "split_count": 1,
                            "training_percentage": 0.7,
                        },
                        "labels": ["Diagnosis1"],
                        "strategy": "GridSearch",
                        "metrics": ["auc", "balanced_accuracy", "f1_macro"],
                        "optimization_metric": "balanced_accuracy",
                        "number_of_processes": 1,
                        "reports": [],
                        "dataset": f"{group}_{comparison}",
                        "refit_optimal_model": False,
                        "export_all_ml_settings": False,
                        "sequence_type": sequence_type,
                        "region_type": "IMGT_CDR3",
                    }

    return {
        "definitions": {
            "datasets": datasets,
            "encodings": encodings,
            "example_weightings": {},
            "ml_methods": {
                "logistic_regression": {
                    "LogisticRegression": {},
                    "model_selection_cv": False,
                    "model_selection_n_folds": -1,
                }
            },
            "motifs": {},
            "preprocessing_sequences": {},
            "signals": {},
            "simulations": {},
        },
        "instructions": instructions,
    }


def load_bcr_rds(path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(path)

    data = {str(name): table.copy() for name, table in obj["data"].items()}
    metadata = obj["meta"].copy()
    return data, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-rds",
        default="C:/path/to/private-immuneml-results/BCR/IBDBCRonly.rds",
        help="BCR RDS containing data and meta entries.",
    )
    parser.add_argument(
        "--output-root",
        default="chain_bcr_immuneml",
        help="Output root for BCR chain-filtered AIRR inputs and immuneML spec.",
    )
    args = parser.parse_args()

    input_rds = Path(args.input_rds)
    output_root = Path(args.output_root)
    data, metadata = load_bcr_rds(input_rds)

    metadata = metadata.copy()
    metadata["filename"] = metadata["Sample"].astype(str) + ".tsv"

    summary_rows = []
    for group, loci in CHAIN_GROUPS.items():
        group_root = output_root / "airr_input" / group
        group_rep_root = group_root / "repertoires"
        group_rep_root.mkdir(parents=True, exist_ok=True)

        retained_rows = []
        for row in metadata.itertuples(index=False):
            sample = str(getattr(row, "Sample"))
            if sample not in data:
                continue

            filtered = clean_airr_table(data[sample], loci, sample)
            locus_counts = {locus: int((filtered["locus"] == locus).sum()) for locus in loci}
            if len(filtered) > 0:
                filtered.to_csv(
                    group_rep_root / getattr(row, "filename"),
                    sep="\t",
                    index=False,
                    quoting=csv.QUOTE_MINIMAL,
                )
                retained_rows.append(row._asdict())

            summary = {
                "chain_group": group,
                "sample": sample,
                "filename": getattr(row, "filename"),
                "diagnosis": getattr(row, "Diagnosis1"),
                "total_sequences": int(len(filtered)),
            }
            summary.update(locus_counts)
            summary_rows.append(summary)

        group_metadata = pd.DataFrame(retained_rows)
        write_metadata(group_metadata, group_root / "metadata_all.csv")

        for comparison, labels in COMPARISONS.items():
            comparison_metadata = group_metadata[group_metadata["Diagnosis1"].isin(labels)].copy()
            write_metadata(comparison_metadata, group_root / f"metadata_{comparison}.csv")

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_root / "chain_sequence_counts.csv", index=False)

    spec = build_spec(output_root)
    spec_path = output_root / "bcr_chain_diagnosis_k3_k4_aa.yaml"
    with spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)

    manifest = {
        "input_rds": as_posix(input_rds),
        "output_root": as_posix(output_root),
        "chain_groups": CHAIN_GROUPS,
        "comparisons": COMPARISONS,
        "spec_path": as_posix(spec_path),
        "note": "The source BCR tables do not include cell-level heavy-light pairing identifiers; bcr_heavy_light is a combined repertoire-level heavy+light chain set.",
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote BCR chain-filtered inputs to {output_root / 'airr_input'}")
    print(f"Wrote immuneML spec to {spec_path}")
    print(f"Wrote sequence counts to {output_root / 'chain_sequence_counts.csv'}")


if __name__ == "__main__":
    main()

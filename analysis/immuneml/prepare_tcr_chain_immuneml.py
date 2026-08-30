from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml


CHAIN_GROUPS = {
    "tcr_alpha": ["TRA"],
    "tcr_beta": ["TRB"],
    "tcr_alpha_beta": ["TRA", "TRB"],
    "tcr_gamma": ["TRG"],
    "tcr_delta": ["TRD"],
    "tcr_gamma_delta": ["TRG", "TRD"],
}

COMPARISONS = {
    "cd_vs_control": ["CD", "Control"],
    "uc_vs_control": ["UC", "Control"],
    "cd_vs_uc": ["CD", "UC"],
}


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def clean_airr_table(df: pd.DataFrame, loci: list[str]) -> pd.DataFrame:
    out = df[df["locus"].isin(loci)].copy()
    out = out.dropna(subset=["junction", "junction_aa"])
    out = out[out["junction"].astype(str).str.len() > 0]
    out = out[out["junction_aa"].astype(str).str.len() > 0]
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

    encodings = {
        f"k{k}_aa": {
            "KmerFrequency": {
                "k": k,
                "k_left": 1,
                "k_right": 1,
                "max_gap": 0,
                "min_gap": 0,
                "name": f"k{k}_aa",
                "normalization_type": "relative_frequency",
                "reads": "unique",
                "region_type": "imgt_cdr3",
                "scale_to_unit_variance": True,
                "scale_to_zero_mean": False,
                "sequence_encoding": "CONTINUOUS_KMER",
                "sequence_type": "AMINO_ACID",
            }
        }
        for k in (3, 4)
    }

    instructions = {}
    for group in CHAIN_GROUPS:
        for comparison in COMPARISONS:
            for k in (3, 4):
                instruction_name = f"{group}_{comparison}_k{k}_aa"
                instructions[instruction_name] = {
                    "type": "TrainMLModel",
                    "settings": [
                        {
                            "encoding": f"k{k}_aa",
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
                    "metrics": ["accuracy", "balanced_accuracy"],
                    "optimization_metric": "balanced_accuracy",
                    "number_of_processes": 1,
                    "reports": [],
                    "dataset": f"{group}_{comparison}",
                    "refit_optimal_model": False,
                    "export_all_ml_settings": False,
                    "sequence_type": "AMINO_ACID",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="C:/path/to/private-immuneml-results/immuneML_runs/airr_input/tcr",
        help="Existing TCR AIRR input root with repertoires/ and metadata_all.csv.",
    )
    parser.add_argument(
        "--output-root",
        default="chain_tcr_immuneml",
        help="Output root for chain-filtered AIRR inputs and immuneML spec.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    repertoire_root = input_root / "repertoires"
    metadata = pd.read_csv(input_root / "metadata_all.csv")

    summary_rows = []
    for group, loci in CHAIN_GROUPS.items():
        group_root = output_root / "airr_input" / group
        group_rep_root = group_root / "repertoires"
        group_rep_root.mkdir(parents=True, exist_ok=True)

        retained_rows = []
        for row in metadata.itertuples(index=False):
            filename = getattr(row, "filename")
            src = repertoire_root / filename
            df = pd.read_csv(src, sep="\t")
            filtered = clean_airr_table(df, loci)

            locus_counts = {locus: int((filtered["locus"] == locus).sum()) for locus in loci}
            if len(filtered) > 0:
                filtered.to_csv(group_rep_root / filename, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
                retained_rows.append(row._asdict())

            summary = {
                "chain_group": group,
                "sample": getattr(row, "Sample"),
                "filename": filename,
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
    spec_path = output_root / "tcr_chain_diagnosis_k3_k4_aa.yaml"
    with spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)

    manifest = {
        "input_root": as_posix(input_root),
        "output_root": as_posix(output_root),
        "chain_groups": CHAIN_GROUPS,
        "comparisons": COMPARISONS,
        "spec_path": as_posix(spec_path),
        "note": "Alpha/beta and gamma/delta groups are combined repertoire-level chain sets; the source AIRR files do not encode cell-level receptor pairing.",
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote chain-filtered inputs to {output_root / 'airr_input'}")
    print(f"Wrote immuneML spec to {spec_path}")
    print(f"Wrote sequence counts to {output_root / 'chain_sequence_counts.csv'}")


if __name__ == "__main__":
    main()

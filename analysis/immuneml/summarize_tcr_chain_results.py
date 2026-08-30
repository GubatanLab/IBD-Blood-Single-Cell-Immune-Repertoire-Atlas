from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_instruction(name: str) -> dict[str, str]:
    for comparison in ("cd_vs_control", "uc_vs_control", "cd_vs_uc"):
        marker = f"_{comparison}_"
        if marker in name:
            chain_group, rest = name.split(marker, 1)
            kmer = rest.replace("_aa", "")
            return {
                "chain_group": chain_group,
                "comparison": comparison,
                "kmer": kmer,
                "sequence_type": "aa",
            }
    return {
        "chain_group": "",
        "comparison": "",
        "kmer": "",
        "sequence_type": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="chain_tcr_immuneml/results")
    parser.add_argument("--output", default="chain_tcr_immuneml/tcr_chain_result_summary.csv")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    rows = []
    for score_path in result_root.glob("*/split_*/Diagnosis1_*_optimal/ml_score.csv"):
        instruction = score_path.parts[-4]
        split_name = score_path.parts[-3]
        parsed = parse_instruction(instruction)
        scores = pd.read_csv(score_path)
        for score_row in scores.to_dict("records"):
            rows.append(
                {
                    "instruction": instruction,
                    "split": split_name,
                    **parsed,
                    **score_row,
                    "score_file": score_path.as_posix(),
                }
            )

    if not rows:
        raise SystemExit(f"No ml_score.csv files found under {result_root}")

    detail = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)

    metric_cols = [col for col in detail.columns if col.startswith("Diagnosis1_")]
    grouped = (
        detail.groupby(["chain_group", "comparison", "kmer", "sequence_type"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]

    aggregate_output = output.with_name(output.stem + "_aggregate.csv")
    grouped.to_csv(aggregate_output, index=False)

    print(f"Wrote split-level summary to {output}")
    print(f"Wrote aggregate summary to {aggregate_output}")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd


CHAIN_GROUPS = [
    "tcr_alpha",
    "tcr_beta",
    "tcr_alpha_beta",
    "tcr_gamma",
    "tcr_delta",
    "tcr_gamma_delta",
]

COMPARISONS = {
    "cd_inflamed_vs_noninflamed": "CD",
    "uc_inflamed_vs_noninflamed": "UC",
}


def main() -> None:
    root = Path("chain_tcr_immuneml/airr_input")
    rows = []
    for chain_group in CHAIN_GROUPS:
        group_root = root / chain_group
        metadata = pd.read_csv(group_root / "metadata_all.csv")
        for comparison, diagnosis in COMPARISONS.items():
            subset = metadata[
                (metadata["Diagnosis1"] == diagnosis)
                & (metadata["Inflammation1"].isin(["Inflamed", "Noninflamed"]))
            ].copy()
            subset.to_csv(group_root / f"metadata_{comparison}.csv", index=False)
            counts = subset["Inflammation1"].value_counts().to_dict()
            rows.append({"chain_group": chain_group, "comparison": comparison, **counts})
    summary = pd.DataFrame(rows).fillna(0)
    summary.to_csv("chain_tcr_immuneml/inflammation_metadata_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

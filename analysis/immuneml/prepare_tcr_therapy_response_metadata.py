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
    "therapy_combined_response": ["AntiTNF", "Ustekinumab", "Vedolizumab"],
    "therapy_antitnf_response": ["AntiTNF"],
    "therapy_ustekinumab_response": ["Ustekinumab"],
    "therapy_vedolizumab_response": ["Vedolizumab"],
}

RESPONSE_LABELS = {
    "Inflamed": "Nonresponder",
    "Noninflamed": "Responder",
}


def main() -> None:
    root = Path("chain_tcr_immuneml/airr_input")
    rows = []

    for chain_group in CHAIN_GROUPS:
        group_root = root / chain_group
        metadata = pd.read_csv(group_root / "metadata_all.csv")

        for comparison, biologics in COMPARISONS.items():
            subset = metadata[
                metadata["Biologic"].isin(biologics)
                & metadata["Inflammation1"].isin(RESPONSE_LABELS)
            ].copy()
            subset["TherapyResponse1"] = subset["Inflammation1"].map(RESPONSE_LABELS)
            subset.to_csv(group_root / f"metadata_{comparison}.csv", index=False)

            counts = subset["TherapyResponse1"].value_counts().to_dict()
            rows.append(
                {
                    "chain_group": chain_group,
                    "comparison": comparison,
                    "biologics": "+".join(biologics),
                    "n": len(subset),
                    **counts,
                }
            )

    summary = pd.DataFrame(rows).fillna(0)
    summary.to_csv("chain_tcr_immuneml/therapy_response_metadata_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Leave-one-patient-out and patient-bootstrap checks for FDR-supported module edges."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import gene_module_microbiome_integration as gm


def main():
    meta, modules, layers = gm.load_inputs()
    primary = pd.read_csv(gm.RESULTS / "primary_gene_module_microbiome_associations.csv")
    selected = primary[primary["fdr_lt_0_10"]].copy()
    rng = np.random.default_rng(gm.SEED + 17)
    rows = []

    for edge in selected.itertuples():
        module_frame = modules[edge.compartment]
        outcome_frame = layers[edge.layer]
        joined = meta.join(module_frame, how="left").join(outcome_frame, how="left", rsuffix="__outcome")
        outcome = edge.outcome if edge.outcome not in module_frame.columns else edge.outcome + "__outcome"

        loo_betas = []
        for patient in joined.index:
            fit = gm.fit_association(joined.drop(index=patient), edge.module, outcome, ["ibdmd", "Batch"])
            loo_betas.append(fit["beta"])

        boot_betas = []
        for _ in range(1000):
            positions = rng.integers(0, len(joined), len(joined))
            sampled = joined.iloc[positions].copy()
            fit = gm.fit_association(sampled, edge.module, outcome, ["ibdmd", "Batch"])
            boot_betas.append(fit["beta"])

        loo = np.asarray(loo_betas, dtype=float)
        boot = np.asarray(boot_betas, dtype=float)
        target_sign = np.sign(edge.beta)
        rows.append({
            "compartment": edge.compartment,
            "layer": edge.layer,
            "module": edge.module,
            "outcome": edge.outcome,
            "original_beta": edge.beta,
            "original_p": edge.p_value,
            "original_q_family": edge.q_family,
            "loo_beta_min": np.nanmin(loo),
            "loo_beta_max": np.nanmax(loo),
            "loo_sign_concordance": np.nanmean(np.sign(loo) == target_sign),
            "bootstrap_beta_median": np.nanmedian(boot),
            "bootstrap_ci_low": np.nanquantile(boot, 0.025),
            "bootstrap_ci_high": np.nanquantile(boot, 0.975),
            "bootstrap_sign_concordance": np.nanmean(np.sign(boot) == target_sign),
        })

    stability = pd.DataFrame(rows).sort_values("original_q_family")
    stability.to_csv(gm.RESULTS / "fdr_edge_leave_one_out_bootstrap_stability.csv", index=False)

    # Leave-one-patient-out stability for the FDR-supported BCR–FACS block.
    block_rows = []
    for patient in meta.index:
        keep = meta.index != patient
        x = gm.residualize_matrix(modules["BCR"].loc[keep], meta.loc[keep], ["ibdmd", "Batch"])
        y = gm.residualize_matrix(layers["Bacterial_FACS"].loc[keep], meta.loc[keep], ["ibdmd", "Batch"])
        block_rows.append({"omitted_patient": patient, "rv": gm.rv_coefficient(x, y)})
    block_stability = pd.DataFrame(block_rows)
    block_stability.to_csv(gm.RESULTS / "bcr_facs_block_leave_one_out_stability.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1.7, 1]}, constrained_layout=True)
    plot_data = stability.copy()
    plot_data["label"] = (
        plot_data["compartment"].map(gm.pretty) + " | "
        + plot_data["module"].map(gm.pretty).str.slice(0, 30) + " → "
        + plot_data["outcome"].map(gm.pretty)
    )
    plot_data = plot_data.sort_values("original_beta")
    y = np.arange(len(plot_data))
    axes[0].errorbar(
        plot_data["bootstrap_beta_median"], y,
        xerr=[
            plot_data["bootstrap_beta_median"] - plot_data["bootstrap_ci_low"],
            plot_data["bootstrap_ci_high"] - plot_data["bootstrap_beta_median"],
        ],
        fmt="o", color="#0072B2", ecolor="#555555", capsize=3, label="Patient bootstrap",
    )
    axes[0].scatter(plot_data["original_beta"], y, marker="D", color="#D55E00", s=32, label="Original HC3 β", zorder=3)
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set_yticks(y, plot_data["label"])
    axes[0].set_xlabel("Standardized β")
    axes[0].set_title("A  FDR-edge patient-bootstrap stability", loc="left", weight="bold")
    axes[0].legend(frameon=False, loc="best")
    axes[0].grid(axis="y", visible=False)

    sns.histplot(block_stability["rv"], bins=12, color="#009E73", edgecolor="white", ax=axes[1])
    observed_block = 0.11393446389480187
    axes[1].axvline(observed_block, color="#D55E00", linewidth=1.5, label=f"Full cohort RV={observed_block:.3f}")
    axes[1].set_xlabel("Leave-one-patient-out RV")
    axes[1].set_ylabel("Omitted-patient analyses")
    axes[1].set_title("B  BCR–FACS block stability", loc="left", weight="bold")
    axes[1].legend(frameon=False)
    gm.save_figure(fig, "SupplementaryFigure9_gene_module_stability")

    report = [
        "# Stability checks for FDR-supported gene-module associations",
        "",
        "All six feature-level q<0.10 associations retained their direction in the leave-one-patient-out analysis if the sign-concordance value is 1.0. Bootstrap intervals use 1,000 patient-level resamples.",
        "",
    ]
    for row in stability.itertuples():
        report.append(
            f"- {gm.pretty(row.compartment)} {gm.pretty(row.module)} → {gm.pretty(row.layer)} {gm.pretty(row.outcome)}: "
            f"LOO β {row.loo_beta_min:.2f} to {row.loo_beta_max:.2f}, sign concordance {row.loo_sign_concordance:.1%}; "
            f"bootstrap median β={row.bootstrap_beta_median:.2f}, 95% interval {row.bootstrap_ci_low:.2f} to {row.bootstrap_ci_high:.2f}, "
            f"sign concordance {row.bootstrap_sign_concordance:.1%}."
        )
    report += [
        "",
        f"The BCR–bacterial-FACS block RV was {block_stability.rv.median():.3f} across leave-one-patient-out analyses "
        f"(range {block_stability.rv.min():.3f}–{block_stability.rv.max():.3f}).",
    ]
    (gm.ROOT / "STABILITY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()

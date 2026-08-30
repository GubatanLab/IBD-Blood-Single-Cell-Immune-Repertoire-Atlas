from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    r"C:/path/to/private-manuscript-workspace"
    r"\High Impact Additional Analyses\Priority Analyses"
)
GLIPH = ROOT / "IBDTCR_BetaOnly_GLIPH2"

RUNS = [
    ("CD_vs_Control", "CD", "Control"),
    ("UC_vs_Control", "UC", "Control"),
    ("CD_vs_UC", "CD", "UC"),
]


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def canonical_members(value: str) -> str:
    return " ".join(sorted(set(str(value).split())))


all_unique = []
summaries = []

for comparison, first_group, second_group in RUNS:
    run_dir = GLIPH / f"{comparison}_run_20260824"
    raw = pd.read_csv(run_dir / "cluster_participant_enrichment.csv")
    raw.insert(0, "comparison", comparison)
    raw["member_set"] = raw["members"].map(canonical_members)

    # GLIPH2 can emit several global-similarity tags for the same exact member
    # set. Treat the unique sequence-member set as the inferential unit.
    unique = (
        raw.sort_values(["fisher_p", "cluster_index"])
        .groupby("member_set", as_index=False, sort=False)
        .first()
    )
    tag_multiplicity = raw.groupby("member_set").size().rename("n_gliph2_tag_rows")
    unique = unique.merge(tag_multiplicity, on="member_set", how="left")
    unique["fdr_unique_member_set"] = bh_adjust(unique["fisher_p"])
    unique["first_group"] = first_group
    unique["second_group"] = second_group
    unique["enrichment_direction"] = np.where(
        unique["carrier_difference"] > 0,
        first_group,
        np.where(unique["carrier_difference"] < 0, second_group, "Equal"),
    )
    unique["significant_fdr05"] = unique["fdr_unique_member_set"] < 0.05
    unique = unique.sort_values(
        ["fdr_unique_member_set", "fisher_p", "member_set"]
    ).reset_index(drop=True)
    unique.insert(1, "unique_cluster_id", [f"{comparison}_U{i:05d}" for i in range(1, len(unique) + 1)])
    all_unique.append(unique)

    sig = unique.loc[unique["significant_fdr05"]]
    summaries.append(
        {
            "comparison": comparison,
            "source": "IBDTCR.rds",
            "chain_filter": "V.name begins TRBV",
            "gliph2_weighting": "one record per participant-sequence",
            "enrichment_unit": "participant carrier",
            "multiple_testing_unit": "unique sequence-member set",
            "n_participants_first_group": int(unique["n_participants_g1"].iloc[0]),
            "n_participants_second_group": int(unique["n_participants_g2"].iloc[0]),
            "n_gliph2_tag_rows": int(len(raw)),
            "n_unique_member_sets": int(len(unique)),
            "n_significant_unique_member_sets_fdr05": int(len(sig)),
            "n_first_group_enriched_fdr05": int((sig["carrier_difference"] > 0).sum()),
            "n_second_group_enriched_fdr05": int((sig["carrier_difference"] < 0).sum()),
            "minimum_unique_member_set_fdr": float(unique["fdr_unique_member_set"].min()),
        }
    )

# Shared-acquisition-series sensitivity for the only well-powered contrast with
# overlapping batches (CD versus UC in S1-S5).
sensitivity_comparison = "CD_vs_UC_shared_S1-S5"
sensitivity_dir = GLIPH / "CD_vs_UC_shared_S1-S5_run_20260824"
sensitivity_raw = pd.read_csv(sensitivity_dir / "cluster_participant_enrichment.csv")
sensitivity_raw.insert(0, "comparison", sensitivity_comparison)
sensitivity_raw["member_set"] = sensitivity_raw["members"].map(canonical_members)
sensitivity_unique = (
    sensitivity_raw.sort_values(["fisher_p", "cluster_index"])
    .groupby("member_set", as_index=False, sort=False)
    .first()
)
sensitivity_multiplicity = (
    sensitivity_raw.groupby("member_set").size().rename("n_gliph2_tag_rows")
)
sensitivity_unique = sensitivity_unique.merge(
    sensitivity_multiplicity, on="member_set", how="left"
)
sensitivity_unique["fdr_unique_member_set"] = bh_adjust(sensitivity_unique["fisher_p"])
sensitivity_unique["first_group"] = "CD"
sensitivity_unique["second_group"] = "UC"
sensitivity_unique["enrichment_direction"] = np.where(
    sensitivity_unique["carrier_difference"] > 0,
    "CD",
    np.where(sensitivity_unique["carrier_difference"] < 0, "UC", "Equal"),
)
sensitivity_unique["significant_fdr05"] = (
    sensitivity_unique["fdr_unique_member_set"] < 0.05
)
sensitivity_unique = sensitivity_unique.sort_values(
    ["fdr_unique_member_set", "fisher_p", "member_set"]
).reset_index(drop=True)
sensitivity_unique.insert(
    1,
    "unique_cluster_id",
    [f"CD_vs_UC_S1-S5_U{i:05d}" for i in range(1, len(sensitivity_unique) + 1)],
)
sensitivity_unique.to_csv(
    ROOT / "Table_PA5_IBDTCR_beta_only_GLIPH2_shared_batch_sensitivity.csv",
    index=False,
)
sensitivity_sig = sensitivity_unique.loc[sensitivity_unique["significant_fdr05"]]
summaries.append(
    {
        "comparison": sensitivity_comparison,
        "source": "IBDTCR.rds",
        "chain_filter": "V.name begins TRBV",
        "gliph2_weighting": "one record per participant-sequence",
        "enrichment_unit": "participant carrier",
        "multiple_testing_unit": "unique sequence-member set",
        "n_participants_first_group": int(sensitivity_unique["n_participants_g1"].iloc[0]),
        "n_participants_second_group": int(sensitivity_unique["n_participants_g2"].iloc[0]),
        "n_gliph2_tag_rows": int(len(sensitivity_raw)),
        "n_unique_member_sets": int(len(sensitivity_unique)),
        "n_significant_unique_member_sets_fdr05": int(len(sensitivity_sig)),
        "n_first_group_enriched_fdr05": int((sensitivity_sig["carrier_difference"] > 0).sum()),
        "n_second_group_enriched_fdr05": int((sensitivity_sig["carrier_difference"] < 0).sum()),
        "minimum_unique_member_set_fdr": float(
            sensitivity_unique["fdr_unique_member_set"].min()
        ),
    }
)

all_unique_df = pd.concat(all_unique, ignore_index=True)
summary_df = pd.DataFrame(summaries)
sig_df = all_unique_df.loc[all_unique_df["significant_fdr05"]].copy()

all_path = ROOT / "Table_PA5_IBDTCR_beta_only_GLIPH2_all_unique_clusters.csv"
sig_path = ROOT / "Table_PA5_IBDTCR_beta_only_GLIPH2_FDR05_unique_clusters.csv"
summary_path = ROOT / "Table_PA5_IBDTCR_beta_only_GLIPH2_summary.csv"
all_unique_df.to_csv(all_path, index=False)
sig_df.to_csv(sig_path, index=False)
summary_df.to_csv(summary_path, index=False)


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=False)

panel_titles = {
    "CD_vs_Control": "CD vs control",
    "UC_vs_Control": "UC vs control",
    "CD_vs_UC": "CD vs UC",
}

for ax, (comparison, first_group, second_group) in zip(axes, RUNS):
    dat = all_unique_df.loc[all_unique_df["comparison"] == comparison].copy()
    dat["minus_log10_fdr"] = -np.log10(dat["fdr_unique_member_set"].clip(lower=1e-300))
    nonsig = ~dat["significant_fdr05"]
    ax.scatter(
        dat.loc[nonsig, "carrier_difference"],
        dat.loc[nonsig, "minus_log10_fdr"],
        s=8,
        c="#b8b8b8",
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )
    sig_first = dat["significant_fdr05"] & (dat["carrier_difference"] > 0)
    sig_second = dat["significant_fdr05"] & (dat["carrier_difference"] < 0)
    ax.scatter(
        dat.loc[sig_first, "carrier_difference"],
        dat.loc[sig_first, "minus_log10_fdr"],
        s=27,
        marker="^",
        c="#c23b32",
        edgecolors="white",
        linewidths=0.35,
        label=f"{first_group} enriched",
        zorder=3,
    )
    ax.scatter(
        dat.loc[sig_second, "carrier_difference"],
        dat.loc[sig_second, "minus_log10_fdr"],
        s=25,
        marker="o",
        c="#3268a8",
        edgecolors="white",
        linewidths=0.35,
        label=f"{second_group} enriched",
        zorder=3,
    )
    ax.axhline(-np.log10(0.05), color="#555555", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.65)
    ax.set_title(panel_titles[comparison])
    ax.set_xlabel(f"Participant-carrier fraction difference\n({first_group} minus {second_group})")
    ax.set_ylabel("−log10(FDR)")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.05, max(1.45, dat["minus_log10_fdr"].max() * 1.12))

    label_dat = dat.loc[dat["significant_fdr05"]].nsmallest(1, "fdr_unique_member_set")
    for _, row in label_dat.iterrows():
        label = row["member_set"].split()[0]
        x_offset = 6 if row["carrier_difference"] <= 0 else -6
        horizontal_alignment = "left" if row["carrier_difference"] <= 0 else "right"
        ax.annotate(
            label,
            (row["carrier_difference"], row["minus_log10_fdr"]),
            xytext=(x_offset, 4),
            textcoords="offset points",
            fontsize=6.5,
            ha=horizontal_alignment,
            clip_on=True,
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=7)

fig.suptitle(
    "IBDTCR beta-chain GLIPH2 convergence: participant-level enrichment",
    fontsize=12,
    y=1.01,
)
fig.tight_layout()
png_path = ROOT / "Figure_PA5_IBDTCR_beta_only_GLIPH2_enrichment.png"
pdf_path = ROOT / "Figure_PA5_IBDTCR_beta_only_GLIPH2_enrichment.pdf"
fig.savefig(png_path, dpi=400, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

sdat = sensitivity_unique.copy()
sdat["minus_log10_fdr"] = -np.log10(sdat["fdr_unique_member_set"].clip(lower=1e-300))
fig_s, ax_s = plt.subplots(figsize=(4.8, 3.7))
nonsig = ~sdat["significant_fdr05"]
ax_s.scatter(
    sdat.loc[nonsig, "carrier_difference"],
    sdat.loc[nonsig, "minus_log10_fdr"],
    s=9,
    c="#b8b8b8",
    alpha=0.45,
    linewidths=0,
    rasterized=True,
)
for mask, marker, color, label in [
    (sdat["significant_fdr05"] & (sdat["carrier_difference"] > 0), "^", "#c23b32", "CD enriched"),
    (sdat["significant_fdr05"] & (sdat["carrier_difference"] < 0), "o", "#3268a8", "UC enriched"),
]:
    ax_s.scatter(
        sdat.loc[mask, "carrier_difference"],
        sdat.loc[mask, "minus_log10_fdr"],
        s=28,
        marker=marker,
        c=color,
        edgecolors="white",
        linewidths=0.35,
        label=label,
        zorder=3,
    )
ax_s.axhline(-np.log10(0.05), color="#555555", linestyle="--", linewidth=0.8)
ax_s.axvline(0, color="#777777", linewidth=0.65)
ax_s.set_title("CD vs UC, shared acquisition series S1–S5")
ax_s.set_xlabel("Participant-carrier fraction difference\n(CD minus UC)")
ax_s.set_ylabel("−log10(FDR)")
ax_s.set_ylim(-0.05, max(1.45, sdat["minus_log10_fdr"].max() * 1.12))
ax_s.grid(axis="y", color="#e5e5e5", linewidth=0.55)
ax_s.spines[["top", "right"]].set_visible(False)
if sdat["significant_fdr05"].any():
    top = sdat.loc[sdat["significant_fdr05"]].nsmallest(1, "fdr_unique_member_set").iloc[0]
    ax_s.annotate(
        top["member_set"].split()[0],
        (top["carrier_difference"], top["minus_log10_fdr"]),
        xytext=(6 if top["carrier_difference"] <= 0 else -6, 4),
        textcoords="offset points",
        fontsize=6.5,
        ha="left" if top["carrier_difference"] <= 0 else "right",
        clip_on=True,
    )
handles, labels = ax_s.get_legend_handles_labels()
if sdat["significant_fdr05"].any() and handles:
    ax_s.legend(handles, labels, loc="upper right", frameon=False, fontsize=7)
else:
    ax_s.text(
        0.98,
        0.95,
        "No clusters at FDR < 0.05",
        transform=ax_s.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
fig_s.tight_layout()
sensitivity_png = ROOT / "Figure_PA5b_IBDTCR_beta_only_GLIPH2_shared_batch_sensitivity.png"
sensitivity_pdf = ROOT / "Figure_PA5b_IBDTCR_beta_only_GLIPH2_shared_batch_sensitivity.pdf"
fig_s.savefig(sensitivity_png, dpi=400, bbox_inches="tight")
fig_s.savefig(sensitivity_pdf, bbox_inches="tight")
plt.close(fig_s)

result_manifest = {
    "source": str(
        Path(
            r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript"
            r"\Figure 2 TCR\IBDTCR.rds"
        )
    ),
    "chain_filter": "TRBV",
    "gliph2_weighting": "participant-sequence presence",
    "enrichment_unit": "participant carrier",
    "deduplication": "exact canonical sequence-member set",
    "summary_table": str(summary_path),
    "all_unique_clusters": str(all_path),
    "significant_unique_clusters": str(sig_path),
    "figure_png": str(png_path),
    "figure_pdf": str(pdf_path),
    "shared_batch_sensitivity_table": str(
        ROOT / "Table_PA5_IBDTCR_beta_only_GLIPH2_shared_batch_sensitivity.csv"
    ),
    "shared_batch_sensitivity_figure_png": str(sensitivity_png),
    "shared_batch_sensitivity_figure_pdf": str(sensitivity_pdf),
    "batch_audit_all_participants": str(
        GLIPH / "IBDTCR_participants_by_diagnosis_and_batch.csv"
    ),
    "batch_audit_TRBV_participants": str(
        GLIPH / "IBDTCR_TRBV_participants_by_diagnosis_and_batch.csv"
    ),
}
(GLIPH / "IBDTCR_GLIPH2_finalization_manifest.json").write_text(
    json.dumps(result_manifest, indent=2), encoding="utf-8"
)

deliverables = []
for path in sorted(ROOT.iterdir(), key=lambda item: item.name.lower()):
    if path.is_file() and path.name != "DELIVERABLES_INDEX.csv":
        deliverables.append(
            {"file": path.name, "type": path.suffix.lower().lstrip("."), "bytes": path.stat().st_size}
        )
pd.DataFrame(deliverables).to_csv(ROOT / "DELIVERABLES_INDEX.csv", index=False)

print(summary_df.to_string(index=False))

from __future__ import annotations

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import mannwhitneyu, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Cell Press Redrawn Figure Set"
MAIN = OUT / "Main Figures"
SRC = OUT / "Source Data"
LEG = OUT / "Legends"

BCR_DIR = Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 3 BCR")
PBMC_DIR = Path(r"C:/path/to/private-legacy-manuscript-assets")

DIAG = ["Control", "CD", "UC"]
COL = {"Control": "#6F6F6F", "CD": "#0072B2", "UC": "#D55E00"}
UP = "#B24745"
DOWN = "#2878A8"
NS = "#B9B9B9"
INK = "#222222"
GRID = "#D9D9D9"

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def bh(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    keep = np.isfinite(p)
    vals = p[keep]
    if not len(vals):
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adj = np.minimum.accumulate((ranked * len(vals) / np.arange(1, len(vals) + 1))[::-1])[::-1]
    restored = np.empty(len(vals))
    restored[order] = np.clip(adj, 0, 1)
    out[np.where(keep)[0]] = restored
    return out


def short_module(value: str) -> str:
    labels = {
        "IgA_mucosal_plasma_cell": "IgA mucosal plasma cell",
        "plasma_cell_inflammatory_antigen_presentation_UPR": "Plasma-cell antigen presentation/UPR",
        "plasmablast_plasma_cell_differentiation": "Plasmablast/plasma differentiation",
        "Plasmablast_plasma_differentiation": "Plasmablast/plasma differentiation",
        "IgG_inflammatory_plasma_cell": "IgG inflammatory plasma cell",
        "antibody_secretion_UPR": "Antibody secretion/UPR",
        "regulatory_B_cell_IL10_like": "IL-10-like regulatory B cell",
        "atypical_memory_CD11c_age_associated_like_B_cell": "CD11c+ atypical memory B cell",
        "mucosal_trafficking_retention_B_cell": "Mucosal trafficking/retention",
        "TLR_innate_inflammatory_B_cell": "TLR inflammatory B cell",
        "IFN_imprinted_B_cell": "IFN-imprinted B cell",
        "BCR_NFkB_activation": "BCR/NF-kB activation",
        "cell_cycle_proliferating_B_cell": "Cycling B cell",
        "Cell_cycle_proliferating_B_cell": "Cycling B cell",
        "IgM_IgD_unswitched_B_cell": "IgM/IgD unswitched B cell",
        "Resting_memory_B_cell": "Resting memory B cell",
        "Resting_naive_B_cell": "Resting naive B cell",
    }
    return labels.get(value, value.replace("_", " "))


def style_axis(ax, grid: str | None = None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(color=INK, labelcolor=INK, pad=2)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.14, y: float = 1.10):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def q_text(q: float) -> str:
    if not np.isfinite(q):
        return "q=NA"
    if q < 0.001:
        return "q<0.001"
    return f"q={q:.3f}"


def participant_panel(
    ax, frame: pd.DataFrame, value: str, ylabel: str,
    stats_y: float = 0.98, stats_inline: bool = False
):
    rng = np.random.default_rng(43)
    positions = np.arange(3)
    arrays = [frame.loc[frame.Diagnosis.eq(group), value].dropna().to_numpy() for group in DIAG]
    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "lw": 1.1},
        whiskerprops={"color": INK, "lw": 0.7},
        capprops={"color": INK, "lw": 0.7},
        boxprops={"edgecolor": "#777777", "lw": 0.7},
    )
    for patch, group in zip(bp["boxes"], DIAG):
        patch.set_facecolor(COL[group])
        patch.set_alpha(0.20)
    for xi, (group, vals) in enumerate(zip(DIAG, arrays)):
        jitter = rng.uniform(-0.14, 0.14, len(vals))
        ax.scatter(np.full(len(vals), xi) + jitter, vals, s=9, color=COL[group], alpha=0.55, edgecolor="none", zorder=3)
    pvals = []
    for vals in arrays[1:]:
        pvals.append(mannwhitneyu(vals, arrays[0], alternative="two-sided").pvalue)
    qvals = bh(np.asarray(pvals))
    stats_label = (
        f"CD {q_text(qvals[0])} · UC {q_text(qvals[1])}"
        if stats_inline else f"CD: {q_text(qvals[0])}\nUC: {q_text(qvals[1])}"
    )
    ax.text(0.98, stats_y, stats_label, transform=ax.transAxes,
            ha="right", va="bottom" if stats_inline else "top", fontsize=5.8 if stats_inline else 6.1, color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0})
    ax.set_xticks(positions, [f"{g}\n(n={len(v)})" for g, v in zip(DIAG, arrays)])
    ax.set_ylabel(ylabel)
    style_axis(ax, "y")
    return pd.DataFrame(
        {
            "metric": value,
            "comparison": ["CD vs Control", "UC vs Control"],
            "p_value": pvals,
            "FDR_within_metric": qvals,
        }
    )


def state_enrichment() -> tuple[pd.DataFrame, pd.DataFrame]:
    state_dir = BCR_DIR / "BCR Architecture Analyses/outputs_manuscript_cd_uc_control/clonotype_state_mapping"
    counts = pd.read_csv(state_dir / "clonotype_state_counts_by_sample.csv", low_memory=False)
    summary = pd.read_csv(state_dir / "clonotype_state_summary_by_sample.csv", low_memory=False)
    counts = counts.merge(summary[["SampleID", "clonotype_id", "total_cells"]], on=["SampleID", "clonotype_id"], how="left")
    counts["status"] = np.where(counts["total_cells"] > 1, "Expanded", "Singleton")
    agg = counts.groupby(["SampleID", "Diagnosis1", "status", "BcellState"], as_index=False)["n_cells"].sum()
    totals = agg.groupby(["SampleID", "status"], as_index=False)["n_cells"].sum().rename(columns={"n_cells": "status_total"})
    agg = agg.merge(totals, on=["SampleID", "status"], how="left")
    agg["fraction"] = agg["n_cells"] / agg["status_total"]
    wide = agg.pivot_table(
        index=["SampleID", "Diagnosis1", "BcellState"], columns="status", values=["fraction", "status_total"], fill_value=0
    ).reset_index()
    rows = []
    rng = np.random.default_rng(20260824)
    for state in counts.groupby("BcellState")["n_cells"].sum().sort_values(ascending=False).index:
        q = wide[wide["BcellState"].eq(state)].copy()
        eligible = (q[("status_total", "Expanded")] >= 5) & (q[("status_total", "Singleton")] >= 20)
        q = q[eligible]
        if len(q) < 15:
            continue
        expanded = q[("fraction", "Expanded")].to_numpy(float)
        singleton = q[("fraction", "Singleton")].to_numpy(float)
        delta = 100 * (expanded - singleton)
        boots = np.array([np.median(rng.choice(delta, size=len(delta), replace=True)) for _ in range(4000)])
        pval = wilcoxon(expanded, singleton, alternative="two-sided").pvalue if np.any(expanded != singleton) else 1.0
        rows.append(
            {
                "BcellState": state,
                "n_participants": len(q),
                "median_expanded_fraction": np.median(expanded),
                "median_singleton_fraction": np.median(singleton),
                "median_change_percentage_points": np.median(delta),
                "ci95_low": np.quantile(boots, 0.025),
                "ci95_high": np.quantile(boots, 0.975),
                "p_value": pval,
            }
        )
    stats = pd.DataFrame(rows)
    stats["FDR"] = bh(stats["p_value"].to_numpy())
    stats = stats.sort_values("median_change_percentage_points")
    return stats, agg


def corrected_isotypes() -> tuple[pd.DataFrame, pd.DataFrame]:
    iso = pd.read_csv(PBMC_DIR / "Figure 5/BCR_isotype_proportions_recommended_manuscript_sample_level_values.csv")
    iso = iso[~iso["isotype"].str.startswith("Total ")].copy()
    mapping = {
        "IGHA1": "IgA", "IGHA2": "IgA", "IGHG1": "IgG", "IGHG2": "IgG",
        "IGHG3": "IgG", "IGHG4": "IgG", "IGHM": "IgM", "IGHD": "IgD", "IGHE": "IgE",
    }
    iso["class"] = iso["isotype"].map(mapping)
    sample = iso.groupby(["SampleID", "Diagnosis1", "class"], as_index=False)["prop"].sum()
    sample["prop"] = sample["prop"] / sample.groupby("SampleID")["prop"].transform("sum")
    group = sample.groupby(["Diagnosis1", "class"], as_index=False)["prop"].mean()
    return sample, group


def build():
    MAIN.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    LEG.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(BCR_DIR / "clonality_metrics_IBDBCR_Immunarch.csv").rename(columns={"Diagnosis1": "Diagnosis"})
    expansion = pd.read_csv(BCR_DIR / "clonal_expansion_index_control_uc_cd_IBDBCR_values.csv").rename(columns={"Diagnosis1": "Diagnosis"})
    switched = pd.read_csv(
        BCR_DIR / "BCR Architecture Analyses/BCR isotype switching diagnosis comparisons/tables/bcr_isotype_switched_fraction_by_sample.csv"
    ).rename(columns={"Diagnosis1": "Diagnosis"})
    participant = metrics[["PatientID", "Diagnosis", "Clonality"]].merge(
        expansion[["PatientID", "ClonalExpansionIndex"]], on="PatientID", how="inner"
    ).merge(switched[["PatientID", "switched_fraction"]], on="PatientID", how="inner")

    states, state_values = state_enrichment()
    modules = pd.read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA1_clone_aware_pseudobulk_module_enrichment.csv")
    modules = modules[modules["modality"].str.upper().eq("BCR")].copy()
    modules["display"] = modules["module"].map(short_module)
    modules = modules.reindex(modules["signed_log10_FDR"].abs().sort_values(ascending=False).head(12).index)
    modules = modules.sort_values("signed_log10_FDR")
    genes = pd.read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA1_clone_aware_pseudobulk_DE_top50.csv")
    genes = genes[genes["modality"].str.upper().eq("BCR")].sort_values("FDR_global").head(8).copy()

    interactions = pd.read_csv(ROOT / "High Impact Additional Analyses/Clone State Interactions/Table_CSI3_formal_interaction_models.csv")
    bdiag = interactions[interactions["family"].eq("BCR_diagnosis")].copy()
    focus_pa1 = modules.loc[modules["FDR_global"] < 0.05, "module"].tolist()
    pa1_to_csi = {
        "plasmablast_plasma_cell_differentiation": "Plasmablast_plasma_differentiation",
        "cell_cycle_proliferating_B_cell": "Cell_cycle_proliferating_B_cell",
    }
    focus = [pa1_to_csi.get(module, module) for module in focus_pa1]
    consistency = bdiag[bdiag["contrast"].isin(["Control_expansion", "CD_expansion", "UC_expansion"]) & bdiag["module"].isin(focus)].copy()
    consistency["Diagnosis"] = consistency["contrast"].str.replace("_expansion", "", regex=False)
    consistency["display"] = consistency["module"].map(short_module)
    consistency["ci95_low"] = consistency["effect"] - 1.96 * consistency["SE"]
    consistency["ci95_high"] = consistency["effect"] + 1.96 * consistency["SE"]
    interaction_rows = bdiag[bdiag["contrast"].str.contains("interaction") & bdiag["module"].isin(focus)]
    interaction_min = interaction_rows.groupby("module")["FDR"].min().rename("minimum_diagnosis_interaction_FDR")
    consistency = consistency.merge(interaction_min, on="module", how="left")

    iso_sample, iso_group = corrected_isotypes()

    fig = plt.figure(figsize=(7.48, 8.0), facecolor="white")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.92, 1.16, 1.18], hspace=0.72, wspace=0.78,
                  left=0.09, right=0.98, top=0.97, bottom=0.075)

    # A: participant-level metrics.
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.43)
    metric_stats = []
    for j, (value, label) in enumerate(
        [("Clonality", "Clonality"), ("ClonalExpansionIndex", "Clonal expansion index"), ("switched_fraction", "Class-switched fraction")]
    ):
        ax = fig.add_subplot(sub[0, j])
        metric_stats.append(participant_panel(ax, participant, value, label))
        if j == 0:
            ax.set_title("Participant-level repertoire and maturation", loc="left", fontweight="bold", pad=5)
            panel_label(ax, "A", x=-0.25, y=1.14)

    # B: participant-level state enrichment.
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(states))
    colors = [UP if v > 0 else DOWN for v in states["median_change_percentage_points"]]
    for yi, (_, row), color in zip(y, states.iterrows(), colors):
        ax.plot([row.ci95_low, row.ci95_high], [yi, yi], color=color if row.FDR < 0.05 else NS, lw=1.1)
        ax.scatter(row.median_change_percentage_points, yi, s=25, color=color if row.FDR < 0.05 else "white",
                   edgecolor=color if row.FDR < 0.05 else NS, linewidth=0.8, zorder=3)
    ax.axvline(0, color="#777777", lw=0.7, ls="--")
    ax.set_yticks(y, [f"{s}\n(n={n})" for s, n in zip(states.BcellState, states.n_participants)])
    ax.set_xlabel("Median expanded − singleton fraction (percentage points)")
    ax.set_title("Expanded-clone B-cell-state enrichment", loc="left", fontweight="bold", pad=5)
    ax.text(0.99, 0.02, "filled: FDR < 0.05", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color="#555555")
    style_axis(ax, "x")
    panel_label(ax, "B", x=-0.30)

    # C: module-level evidence plus representative gene effects.
    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(len(modules))
    sig = modules["FDR_global"] < 0.05
    vals = modules["signed_log10_FDR"].to_numpy(float)
    colors = np.where(sig, np.where(vals > 0, UP, DOWN), NS)
    for yi, val, color in zip(y, vals, colors):
        ax.plot([0, val], [yi, yi], color=color, lw=1.0)
        ax.scatter(val, yi, s=23, color=color, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(0, color="#777777", lw=0.65)
    ax.axvline(-math.log10(0.05), color="#999999", lw=0.6, ls="--")
    ax.axvline(math.log10(20), color="#999999", lw=0.6, ls="--")
    ax.set_yticks(y, modules["display"])
    ax.set_xlabel("Signed −log10 global FDR\n← singleton enriched     expanded enriched →")
    ax.set_title("State-matched clone-aware programs", loc="left", fontweight="bold", pad=5)
    style_axis(ax, "x")
    panel_label(ax, "C", x=-0.30)

    # D: diagnosis-stratified formal estimates and interaction-FDR disclosure.
    ax = fig.add_subplot(gs[2, 0])
    order = [short_module(m) for m in focus]
    order = [x for x in order if x in set(consistency["display"])]
    offsets = {"Control": -0.18, "CD": 0.0, "UC": 0.18}
    for diagnosis in DIAG:
        q = consistency[consistency["Diagnosis"].eq(diagnosis)].set_index("display").reindex(order)
        yy = np.arange(len(order)) + offsets[diagnosis]
        ax.errorbar(q["effect"], yy, xerr=[q["effect"] - q["ci95_low"], q["ci95_high"] - q["effect"]],
                    fmt="o", ms=4.2, color=COL[diagnosis], ecolor=COL[diagnosis], elinewidth=0.9, capsize=1.8, label=diagnosis)
    ax.axvline(0, color="#777777", lw=0.7, ls="--")
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("Expanded − singleton module score (estimate ± 95% CI)")
    ax.set_title("No diagnosis-specific interaction", loc="left", fontweight="bold", pad=5)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.36), handletextpad=0.35, columnspacing=0.9)
    min_q = consistency["minimum_diagnosis_interaction_FDR"].min()
    ax.text(0.99, 0.98, f"all diagnosis interactions q ≥ {min_q:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0})
    style_axis(ax, "x")
    panel_label(ax, "D", x=-0.30)

    # E: corrected, complete isotype composition.
    ax = fig.add_subplot(gs[2, 1])
    classes = [c for c in ["IgM", "IgD", "IgA", "IgG", "IgE"] if c in iso_group["class"].unique()]
    palette = {"IgM": "#56B4E9", "IgD": "#999999", "IgA": "#009E73", "IgG": "#CC79A7", "IgE": "#E6B800"}
    comp = iso_group.pivot(index="Diagnosis1", columns="class", values="prop").fillna(0).reindex(DIAG)
    bottom = np.zeros(len(DIAG))
    for cls in classes:
        vals = comp[cls].to_numpy(float)
        ax.bar(np.arange(3), vals, bottom=bottom, width=0.64, color=palette[cls], label=cls)
        bottom += vals
    ax.set_xticks(np.arange(3), [f"{g}\n(n={iso_sample.loc[iso_sample.Diagnosis1.eq(g), 'SampleID'].nunique()})" for g in DIAG])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean participant repertoire fraction")
    ax.set_title("Heavy-chain isotype composition", loc="left", fontweight="bold", pad=5)
    ax.legend(frameon=False, ncol=len(classes), loc="lower center", bbox_to_anchor=(0.5, -0.31), handlelength=1.6, columnspacing=0.8)
    ax.text(0.99, 0.95, "subclasses summed; bars = 100%", transform=ax.transAxes, ha="right", va="top", fontsize=6.3,
            color="#555555", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0})
    style_axis(ax, "y")
    panel_label(ax, "E", x=-0.22)

    fig.savefig(MAIN / "Figure_4_Immunity_revised.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(MAIN / "Figure_4_Immunity_revised.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(MAIN / "Figure_4_Immunity_revised.tif", dpi=500, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    pd.concat(metric_stats, ignore_index=True).to_csv(SRC / "Figure4_revised_participant_metric_tests.csv", index=False)
    states.to_csv(SRC / "Figure4_revised_state_enrichment.csv", index=False)
    modules.to_csv(SRC / "Figure4_revised_clone_aware_modules.csv", index=False)
    genes.to_csv(SRC / "Figure4_revised_representative_genes.csv", index=False)
    consistency.to_csv(SRC / "Figure4_revised_diagnosis_consistency.csv", index=False)
    iso_sample.to_csv(SRC / "Figure4_revised_isotype_participant_values.csv", index=False)
    iso_group.to_csv(SRC / "Figure4_revised_isotype_group_means.csv", index=False)

    legend = """Figure 4. BCR clonal expansion is associated with antibody-secreting and class-switched B-cell programs. (A) Participant-level BCR clonality, clonal-expansion index, and class-switched fraction in Control, Crohn's disease (CD), and ulcerative colitis (UC). Points denote participants; boxes show the median and interquartile range, and whiskers extend to 1.5 times the interquartile range. Two-sided Mann-Whitney tests compare each disease group with Control; q values are Benjamini-Hochberg adjusted within each metric. (B) Participant-level enrichment of B-cell states among expanded relative to singleton clonotypes. For each participant, state fractions were calculated separately among expanded and singleton BCR-bearing cells. Participants were included for a state when they contributed at least 5 expanded and 20 singleton cells. Points show the median paired change in percentage points and lines show bootstrap 95% confidence intervals; filled points indicate Benjamini-Hochberg FDR < 0.05 across states. (C) State-matched, participant-paired pseudobulk module enrichment in expanded relative to singleton BCR-bearing cells, adjusted for RNA feature complexity and mitochondrial read fraction. Color is shown only for modules passing global FDR < 0.05; gray denotes nonsignificant modules. (D) Diagnosis-stratified estimates and 95% confidence intervals for clone-aware programs available in the formal interaction models. None of the diagnosis-by-expansion interactions passed FDR correction, providing no evidence that these expansion-associated effects differed by diagnosis. (E) Participant-level heavy-chain isotype proportions averaged by diagnosis. IGHA1 and IGHA2 were summed as IgA, and IGHG1-4 were summed as IgG before aggregation; each bar therefore sums to 100%. Expanded clonotypes were defined as clonotypes represented by more than one cell. BCR, B cell receptor; FDR, false discovery rate; UPR, unfolded protein response."""
    (LEG / "Figure_4_Immunity_revised_legend.txt").write_text(legend + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(r"C:/path/to/private-manuscript-workspace")
HI = ROOT / "High Impact Additional Analyses"
PA = HI / "Priority Analyses"
OUT = ROOT / "Revised Main Figure Additions"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260824
RNG = np.random.default_rng(SEED)

COLORS = {
    "TCR": "#2f6db0",
    "BCR": "#b85c38",
    "CD": "#2f6db0",
    "UC": "#c94f66",
    "Control": "#7b7b7b",
    "up": "#b83a55",
    "down": "#2f6db0",
    "single": "#666666",
    "paired": "#6a3d9a",
    "grid": "#dedede",
    "text": "#222222",
}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def finish_axis(ax: plt.Axes, xgrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.tick_params(width=0.8, length=3)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def pretty_module(value: str) -> str:
    mapping = {
        "EOMES_ZEB2_inflammatory_CD8_TRM_like": "EOMES-ZEB2 inflammatory CD8/TRM-like",
        "Effector_cytotoxicity": "Effector cytotoxicity",
        "Th1_Tc1_inflammatory": "Th1/Tc1 inflammation",
        "Activated_Treg_suppressive_T_cell": "Activated Treg suppressive",
        "Recent_TCR_stimulation_immediate_early": "Recent TCR stimulation",
        "Cell_cycle_clonal_proliferation": "Cell-cycle/clonal proliferation",
        "Th17_Tc17_IL23_axis": "Th17/Tc17-IL-23 axis",
        "plasmablast_plasma_cell_differentiation": "Plasmablast/plasma-cell differentiation",
        "antibody_secretion_UPR": "Antibody secretion/UPR",
        "IgA_mucosal_plasma_cell": "IgA mucosal plasma cell",
        "IgG_inflammatory_plasma_cell": "IgG inflammatory plasma cell",
        "plasma_cell_inflammatory_antigen_presentation_UPR": "Inflammatory antigen presentation/UPR",
        "cell_cycle_proliferating_B_cell": "Cell-cycle/proliferating B cell",
    }
    return mapping.get(value, value.replace("_", " "))


# ---------------------------------------------------------------------------
# Addition 1: clone-aware transcriptional programs
# ---------------------------------------------------------------------------
modules = pd.read_csv(PA / "Table_PA1_clone_aware_pseudobulk_module_enrichment.csv")
modules = modules.loc[modules["FDR_global"] < 0.05].copy()
modules["display_module"] = modules["module"].map(pretty_module)
modules["signed_significance"] = np.where(
    modules["Direction"].str.lower().eq("up"),
    -np.log10(modules["FDR_global"]),
    np.log10(modules["FDR_global"]),
)
modules.to_csv(OUT / "SourceData_Addition_1_clone_aware_programs.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), gridspec_kw={"wspace": 1.02})
for ax, modality, label in zip(axes, ["TCR", "BCR"], ["A", "B"]):
    dat = modules.loc[modules["modality"] == modality].copy()
    dat = dat.sort_values("signed_significance")
    y = np.arange(len(dat))
    colors = [COLORS["up"] if d == "Up" else COLORS["down"] for d in dat["Direction"]]
    sizes = np.repeat(52.0, len(dat))
    ax.hlines(y, 0, dat["signed_significance"], color=colors, linewidth=1.3, alpha=0.75)
    ax.scatter(
        dat["signed_significance"],
        y,
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax.axvline(0, color="#888888", linewidth=0.75)
    ax.axvline(-np.log10(0.05), color="#aaaaaa", linestyle="--", linewidth=0.65)
    ax.axvline(np.log10(0.05), color="#aaaaaa", linestyle="--", linewidth=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(dat["display_module"])
    ax.set_title(f"{modality} programs", loc="left", fontweight="bold")
    ax.set_xlabel("Signed -log10(global FDR)\nExpanded versus singleton")
    finish_axis(ax)
    panel_label(ax, label)

max_abs = max(abs(modules["signed_significance"].min()), abs(modules["signed_significance"].max()))
for ax in axes:
    ax.set_xlim(-max_abs * 1.08, max_abs * 1.08)

handles = [
    mpl.lines.Line2D([], [], marker="o", linestyle="", color=COLORS["up"], label="Higher in expanded"),
    mpl.lines.Line2D([], [], marker="o", linestyle="", color=COLORS["down"], label="Lower in expanded"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.subplots_adjust(bottom=0.21, left=0.31, right=0.98, top=0.92)
save_figure(fig, "Addition_1_clone_aware_programs")


# ---------------------------------------------------------------------------
# Addition 2: CD-specific T–B module coupling
# ---------------------------------------------------------------------------
coupling = pd.read_csv(HI / "Table_HI_CD_specific_T_B_module_coupling.csv")
interaction = pd.read_csv(HI / "Table_HI_T_B_module_interaction_tests.csv")
participants = pd.read_csv(HI / "Table_HI_integrated_participant_features.csv")
coupling.to_csv(OUT / "SourceData_Addition_2_coupling_effects.csv", index=False)


def rank_residuals(df: pd.DataFrame, target: str, covariates: list[str]) -> pd.Series:
    work = df[[target] + covariates].dropna().copy()
    y = rankdata(work[target].to_numpy(float), method="average")
    columns = [np.ones(len(work))]
    for cov in covariates:
        if work[cov].dtype == object:
            levels = sorted(work[cov].astype(str).unique())
            for level in levels[1:]:
                columns.append((work[cov].astype(str) == level).astype(float).to_numpy())
        else:
            v = work[cov].to_numpy(float)
            columns.append((v - np.nanmean(v)) / np.nanstd(v))
    x = np.column_stack(columns)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    residual = (residual - residual.mean()) / residual.std(ddof=1)
    return pd.Series(residual, index=work.index)


def bootstrap_line(x: np.ndarray, y: np.ndarray, n_boot: int = 1000):
    grid = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), 120)
    pred = np.empty((n_boot, len(grid)))
    for i in range(n_boot):
        idx = RNG.integers(0, len(x), len(x))
        coef = np.polyfit(x[idx], y[idx], 1)
        pred[i] = coef[0] * grid + coef[1]
    coef = np.polyfit(x, y, 1)
    return grid, coef[0] * grid + coef[1], np.quantile(pred, 0.025, axis=0), np.quantile(pred, 0.975, axis=0)


cd = participants.loc[participants["Diagnosis1"] == "CD"].copy()
cd["log_tcr_depth"] = np.log1p(cd["tcr_total_cells"])
cd["log_bcr_depth"] = np.log1p(cd["bcr_total_cells"])
base_covariates = ["Age", "Sex", "log_tcr_depth", "log_bcr_depth"]

scatter_rows = []
for outcome in ["bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module"]:
    valid = cd[["tcr_cytotoxic_expanded", outcome] + base_covariates].dropna().index
    tmp = cd.loc[valid].copy()
    tmp["x_residual"] = rank_residuals(tmp, "tcr_cytotoxic_expanded", base_covariates)
    tmp["y_residual"] = rank_residuals(tmp, outcome, base_covariates)
    tmp["outcome"] = outcome
    scatter_rows.append(tmp[["SampleID", "Diagnosis1", "outcome", "x_residual", "y_residual"]])
coupling_scatter = pd.concat(scatter_rows, ignore_index=True)
coupling_scatter.to_csv(OUT / "SourceData_Addition_2_coupling_scatter.csv", index=False)

fig = plt.figure(figsize=(10.2, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.55)
ax_forest = fig.add_subplot(gs[0, 0])
ax_iga = fig.add_subplot(gs[0, 1])
ax_plasma = fig.add_subplot(gs[0, 2])

outcome_order = ["bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module"]
diagnosis_order = ["Control", "CD", "UC"]
y_positions = {}
current = 0
for outcome in outcome_order:
    for diagnosis in diagnosis_order:
        y_positions[(outcome, diagnosis)] = current
        current += 1
    current += 0.65

for _, row in coupling.iterrows():
    y = y_positions[(row["outcome"], row["diagnosis"])]
    color = COLORS[row["diagnosis"]]
    ax_forest.errorbar(
        row["partial_rho"],
        y,
        xerr=[[row["partial_rho"] - row["ci_low"]], [row["ci_high"] - row["partial_rho"]]],
        fmt="o",
        color=color,
        ecolor=color,
        markersize=5.2,
        capsize=2,
        linewidth=1.1,
        zorder=3,
    )
ax_forest.axvline(0, color="#888888", linewidth=0.8)
ticks, labels = [], []
for outcome in outcome_order:
    for diagnosis in diagnosis_order:
        ticks.append(y_positions[(outcome, diagnosis)])
        labels.append(diagnosis)
ax_forest.set_yticks(ticks)
grouped_labels = [
    "IgA mucosal • Control",
    "CD",
    "UC",
    "Plasma differentiation • Control",
    "CD",
    "UC",
]
ax_forest.set_yticklabels(grouped_labels)
ax_forest.invert_yaxis()
ax_forest.set_xlim(-0.95, 0.95)
ax_forest.set_xlabel("Partial Spearman ρ (95% CI)")
ax_forest.set_title(
    "Diagnosis-specific coupling\nInteraction P=0.0025 (IgA); P=0.0013 (plasma)",
    loc="left",
    fontweight="bold",
    fontsize=9.2,
)
finish_axis(ax_forest)
panel_label(ax_forest, "A")

for ax, outcome, title, panel in [
    (ax_iga, "bcr_IgA_mucosal_module", "CD: IgA mucosal program", "B"),
    (ax_plasma, "bcr_plasma_differentiation_module", "CD: plasma-cell differentiation", "C"),
]:
    dat = coupling_scatter.loc[coupling_scatter["outcome"] == outcome]
    x = dat["x_residual"].to_numpy(float)
    y = dat["y_residual"].to_numpy(float)
    grid, fit, low, high = bootstrap_line(x, y)
    ax.fill_between(grid, low, high, color=COLORS["CD"], alpha=0.14, linewidth=0)
    ax.plot(grid, fit, color=COLORS["CD"], linewidth=1.5)
    ax.scatter(x, y, s=23, color=COLORS["CD"], alpha=0.72, edgecolors="white", linewidths=0.35)
    stat = coupling.loc[(coupling["outcome"] == outcome) & (coupling["diagnosis"] == "CD")].iloc[0]
    ax.text(
        0.04,
        0.96,
        f"partial ρ={stat['partial_rho']:.2f}\nFDR={stat['p_adj_within_six']:.2g}; n={int(stat['n'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Adjusted rank residual\nExpanded-TCR cytotoxicity")
    ax.set_ylabel("Adjusted rank residual\nB-cell program")
    finish_axis(ax, xgrid=False)
    panel_label(ax, panel)

fig.subplots_adjust(left=0.11, right=0.99, bottom=0.2, top=0.86)
save_figure(fig, "Addition_2_CD_specific_T_B_coupling")


# ---------------------------------------------------------------------------
# Addition 3: incremental value of receptor pairing
# ---------------------------------------------------------------------------
auc = pd.read_csv(HI / "Table_HI_paired_chain_nested_validation_summary.csv")
delta = pd.read_csv(HI / "Table_HI_paired_chain_incremental_value.csv")
auc_main = auc.loc[auc["representation"].isin(["single_chain", "dual_additive"])].copy()
delta_main = delta.loc[delta["comparison"] == "dual_additive vs single_chain"].copy()
auc_main.to_csv(OUT / "SourceData_Addition_3_paired_chain_AUC.csv", index=False)
delta_main.to_csv(OUT / "SourceData_Addition_3_paired_chain_delta_AUC.csv", index=False)

rows = [(m, t) for m in ["TCR", "BCR"] for t in ["CD vs Control", "UC vs Control", "CD vs UC"]]
y_map = {key: i for i, key in enumerate(rows)}
y_labels = [f"{m}: {t}" for m, t in rows]

fig, (ax_auc, ax_delta) = plt.subplots(1, 2, figsize=(9.0, 4.1), gridspec_kw={"wspace": 0.48})
for modality, task in rows:
    y = y_map[(modality, task)]
    sub = auc_main.loc[(auc_main["modality"] == modality) & (auc_main["task"] == task)]
    single = sub.loc[sub["representation"] == "single_chain"].iloc[0]
    paired = sub.loc[sub["representation"] == "dual_additive"].iloc[0]
    ax_auc.plot(
        [single["pooled_outer_fold_auc"], paired["pooled_outer_fold_auc"]],
        [y, y],
        color="#b7b7b7",
        linewidth=1.1,
        zorder=1,
    )
    for row, color, marker in [(single, COLORS["single"], "o"), (paired, COLORS["paired"], "s")]:
        ax_auc.errorbar(
            row["pooled_outer_fold_auc"],
            y,
            xerr=[
                [row["pooled_outer_fold_auc"] - row["pooled_auc_bootstrap_ci_low"]],
                [row["pooled_auc_bootstrap_ci_high"] - row["pooled_outer_fold_auc"]],
            ],
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=5,
            capsize=2,
            linewidth=1.05,
            zorder=3,
        )
ax_auc.axvline(0.5, color="#888888", linestyle="--", linewidth=0.8)
ax_auc.set_yticks(range(len(rows)))
ax_auc.set_yticklabels(y_labels)
ax_auc.invert_yaxis()
ax_auc.set_xlim(0.48, 1.015)
ax_auc.set_xlabel("Pooled outer-fold ROC AUC (95% CI)")
ax_auc.set_title("Matched participants and cells", loc="left", fontweight="bold")
finish_axis(ax_auc)
panel_label(ax_auc, "A")

for _, row in delta_main.iterrows():
    y = y_map[(row["modality"], row["task"])]
    color = COLORS[row["modality"]]
    significant = row["delta_auc_ci_low"] > 0 or row["delta_auc_ci_high"] < 0
    ax_delta.errorbar(
        row["delta_pooled_auc"],
        y,
        xerr=[
            [row["delta_pooled_auc"] - row["delta_auc_ci_low"]],
            [row["delta_auc_ci_high"] - row["delta_pooled_auc"]],
        ],
        fmt="D" if significant else "o",
        color=color,
        ecolor=color,
        markersize=5.2,
        capsize=2,
        linewidth=1.1,
        zorder=3,
    )
ax_delta.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
ax_delta.set_yticks(range(len(rows)))
ax_delta.set_yticklabels(y_labels)
ax_delta.invert_yaxis()
ax_delta.set_xlim(-0.07, 0.21)
ax_delta.set_xlabel("Δ pooled ROC AUC\nPaired-chain minus single-chain")
ax_delta.set_title("Incremental value of pairing", loc="left", fontweight="bold")
finish_axis(ax_delta)
panel_label(ax_delta, "B")

handles = [
    mpl.lines.Line2D([], [], marker="o", linestyle="", color=COLORS["single"], label="Single chain"),
    mpl.lines.Line2D([], [], marker="s", linestyle="", color=COLORS["paired"], label="Paired chains"),
    mpl.lines.Line2D([], [], marker="D", linestyle="", color=COLORS["TCR"], label="CI excludes zero"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.22, right=0.99, bottom=0.21, top=0.89)
save_figure(fig, "Addition_3_paired_chain_classification")


# ---------------------------------------------------------------------------
# Addition 4: IgG-inflammatory B-cell program and calprotectin
# ---------------------------------------------------------------------------
inflammation = pd.read_csv(PA / "Table_PA6_continuous_inflammation_analysis_dataset.csv")
inflammation = inflammation.loc[
    inflammation["Diagnosis1"].isin(["CD", "UC"])
    & inflammation["bcr_IgG_inflammatory_module"].notna()
    & inflammation["log1p_calprotectin"].notna()
].copy()
inflammation.to_csv(OUT / "SourceData_Addition_4_IgG_calprotectin.csv", index=False)

assoc = pd.read_csv(PA / "Table_PA6_continuous_calprotectin_associations.csv")
assoc_row = assoc.loc[
    (assoc["analysis"] == "all_IBD") & (assoc["metric"] == "bcr_IgG_inflammatory_module")
].iloc[0]

inflammation["calprotectin_rank_residual"] = rank_residuals(
    inflammation,
    "log1p_calprotectin",
    ["Diagnosis1", "Age", "Sex", "log_bcr_depth"],
)
inflammation["IgG_module_rank_residual"] = rank_residuals(
    inflammation,
    "bcr_IgG_inflammatory_module",
    ["Diagnosis1", "Age", "Sex", "log_bcr_depth"],
)

fig, (ax_raw, ax_adjusted) = plt.subplots(1, 2, figsize=(7.5, 3.55), gridspec_kw={"wspace": 0.42})
for diagnosis, marker in [("CD", "o"), ("UC", "^")]:
    dat = inflammation.loc[inflammation["Diagnosis1"] == diagnosis]
    ax_raw.scatter(
        dat["log1p_calprotectin"],
        dat["bcr_IgG_inflammatory_module"],
        s=22,
        marker=marker,
        color=COLORS[diagnosis],
        alpha=0.68,
        edgecolors="white",
        linewidths=0.3,
        label=diagnosis,
    )
xraw = inflammation["log1p_calprotectin"].to_numpy(float)
yraw = inflammation["bcr_IgG_inflammatory_module"].to_numpy(float)
grid, fit, low, high = bootstrap_line(xraw, yraw)
ax_raw.fill_between(grid, low, high, color="#666666", alpha=0.12, linewidth=0)
ax_raw.plot(grid, fit, color="#555555", linewidth=1.25)
ax_raw.set_xlabel("Fecal calprotectin, log1p(µg/g)")
ax_raw.set_ylabel("IgG-inflammatory B-cell module")
ax_raw.set_title("Observed participant values", loc="left", fontweight="bold")
ax_raw.legend(frameon=False, loc="upper left")
finish_axis(ax_raw, xgrid=False)
panel_label(ax_raw, "A")

x = inflammation["calprotectin_rank_residual"].to_numpy(float)
y = inflammation["IgG_module_rank_residual"].to_numpy(float)
grid, fit, low, high = bootstrap_line(x, y)
ax_adjusted.fill_between(grid, low, high, color=COLORS["BCR"], alpha=0.14, linewidth=0)
ax_adjusted.plot(grid, fit, color=COLORS["BCR"], linewidth=1.5)
for diagnosis, marker in [("CD", "o"), ("UC", "^")]:
    dat = inflammation.loc[inflammation["Diagnosis1"] == diagnosis]
    ax_adjusted.scatter(
        dat["calprotectin_rank_residual"],
        dat["IgG_module_rank_residual"],
        s=22,
        marker=marker,
        color=COLORS[diagnosis],
        alpha=0.68,
        edgecolors="white",
        linewidths=0.3,
    )
ax_adjusted.text(
    0.04,
    0.96,
    f"partial ρ={assoc_row['partial_spearman_rho']:.3f}\nFDR={assoc_row['FDR_global']:.3f}; n={int(assoc_row['n'])}",
    transform=ax_adjusted.transAxes,
    ha="left",
    va="top",
    fontsize=8,
)
ax_adjusted.set_xlabel("Adjusted rank residual\nFecal calprotectin")
ax_adjusted.set_ylabel("Adjusted rank residual\nIgG-inflammatory module")
ax_adjusted.set_title("Covariate-adjusted association", loc="left", fontweight="bold")
finish_axis(ax_adjusted, xgrid=False)
panel_label(ax_adjusted, "B")

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.2, top=0.88)
save_figure(fig, "Addition_4_IgG_calprotectin")


# Deliverables index
rows = []
for path in sorted(OUT.iterdir(), key=lambda p: p.name.lower()):
    if path.is_file() and path.name != "DELIVERABLES_INDEX.csv":
        rows.append({"file": path.name, "type": path.suffix.lstrip(".").lower(), "bytes": path.stat().st_size})
pd.DataFrame(rows).to_csv(OUT / "DELIVERABLES_INDEX.csv", index=False)

print(f"Created {len(rows)} main-figure addition deliverables in {OUT}")

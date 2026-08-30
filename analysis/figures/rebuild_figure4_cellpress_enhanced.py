from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import mannwhitneyu, norm, rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Cell Press Redrawn Figure Set"
MAIN = OUT / "Main Figures"
SRC = OUT / "Source Data"
LEG = OUT / "Legends"

BCR_DIR = Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 3 BCR")
PBMC_DIR = Path(r"C:/path/to/private-legacy-manuscript-assets")

DIAG = ["Control", "CD", "UC"]
DIAG_COLORS = {"Control": "#6F6F6F", "CD": "#0072B2", "UC": "#D55E00"}
UP = "#B24745"
DOWN = "#2878A8"
NS = "#B9B9B9"
INK = "#222222"
GRID = "#D9D9D9"

MODULE_GENES = {
    "IgA_mucosal_plasma_cell": ["IGHA1", "IGHA2", "JCHAIN", "MZB1", "XBP1", "SDC1", "TNFRSF17", "CCR10", "PRDM1", "IRF4"],
    "plasma_cell_inflammatory_antigen_presentation_UPR": [
        "XBP1", "MZB1", "SDC1", "JCHAIN", "DERL3", "HSPA5", "HSP90B1", "HLA-DRA", "HLA-DRB1",
        "HLA-DPA1", "HLA-DPB1", "CD74", "CXCR4", "IGHG1",
    ],
    "plasmablast_plasma_cell_differentiation": [
        "PRDM1", "XBP1", "IRF4", "MZB1", "SDC1", "JCHAIN", "SSR4", "FKBP11", "DERL3", "TNFRSF17",
        "SLAMF7", "CD38", "CD27", "SEC11C",
    ],
    "IgG_inflammatory_plasma_cell": ["IGHG1", "IGHG2", "IGHG3", "IGHG4", "JCHAIN", "MZB1", "XBP1", "SDC1", "PRDM1", "IRF4", "CXCR4"],
    "antibody_secretion_UPR": [
        "XBP1", "HSPA5", "HSP90B1", "HERPUD1", "ATF4", "ATF6", "DDIT3", "SEL1L", "DNAJB9", "PPIB",
        "CALR", "ERP44", "DERL3", "SEC61A1", "SSR4",
    ],
    "cell_cycle_proliferating_B_cell": [
        "MKI67", "TOP2A", "STMN1", "TYMS", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        "HMGB2", "CENPF", "UBE2C", "PCLAF",
    ],
    "BAFF_APRIL_survival_response": [
        "TNFRSF13B", "TNFRSF17", "TNFRSF13C", "BCL2", "MCL1", "BCL2A1",
        "CD40", "IL6R", "JCHAIN", "XBP1", "MZB1",
    ],
}

MODULE_LABELS = {
    "IgA_mucosal_plasma_cell": "IgA mucosal plasma cell",
    "plasma_cell_inflammatory_antigen_presentation_UPR": "Plasma-cell antigen presentation/UPR",
    "plasmablast_plasma_cell_differentiation": "Plasmablast/plasma differentiation",
    "IgG_inflammatory_plasma_cell": "IgG inflammatory plasma cell",
    "antibody_secretion_UPR": "Antibody secretion/UPR",
    "cell_cycle_proliferating_B_cell": "Cycling B cell",
    "BAFF_APRIL_survival_response": "BAFF/APRIL survival response",
}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.4,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def bh(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    keep = np.isfinite(p)
    vals = p[keep]
    if not len(vals):
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = np.minimum.accumulate((ranked * len(vals) / np.arange(1, len(vals) + 1))[::-1])[::-1]
    restored = np.empty(len(vals))
    restored[order] = np.clip(adjusted, 0, 1)
    out[np.where(keep)[0]] = restored
    return out


def style_axis(ax, grid: str | None = None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(color=INK, labelcolor=INK, pad=2)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.18, y: float = 1.10):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12.5, fontweight="bold", ha="left", va="top")


def q_label(q: float) -> str:
    if not np.isfinite(q):
        return "q=NA"
    if q < 0.001:
        return "q<0.001"
    return f"q={q:.3f}"


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 3000):
    if len(values) < 2:
        return np.nan, np.nan
    boots = np.array([np.median(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)])
    return np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def state_threshold_analysis() -> pd.DataFrame:
    state_dir = BCR_DIR / "BCR Architecture Analyses/outputs_manuscript_cd_uc_control/clonotype_state_mapping"
    counts = pd.read_csv(state_dir / "clonotype_state_counts_by_sample.csv", low_memory=False)
    summary = pd.read_csv(state_dir / "clonotype_state_summary_by_sample.csv", low_memory=False)
    counts = counts.merge(summary[["SampleID", "clonotype_id", "total_cells"]], on=["SampleID", "clonotype_id"], how="left")
    rng = np.random.default_rng(20260827)
    rows = []
    for threshold in (2, 3, 4):
        work = counts[(counts["total_cells"].eq(1)) | (counts["total_cells"].ge(threshold))].copy()
        work["status"] = np.where(work["total_cells"].ge(threshold), "Expanded", "Singleton")
        agg = work.groupby(["SampleID", "Diagnosis1", "status", "BcellState"], as_index=False)["n_cells"].sum()
        totals = agg.groupby(["SampleID", "status"], as_index=False)["n_cells"].sum().rename(columns={"n_cells": "status_total"})
        agg = agg.merge(totals, on=["SampleID", "status"], how="left")
        agg["fraction"] = agg["n_cells"] / agg["status_total"]
        wide = agg.pivot_table(
            index=["SampleID", "Diagnosis1", "BcellState"], columns="status", values=["fraction", "status_total"], fill_value=0
        ).reset_index()
        threshold_rows = []
        for state in counts["BcellState"].dropna().unique():
            subset = wide[wide["BcellState"].eq(state)].copy()
            subset = subset[(subset[("status_total", "Expanded")] >= 5) & (subset[("status_total", "Singleton")] >= 20)]
            if len(subset) < 10:
                continue
            expanded = subset[("fraction", "Expanded")].to_numpy(float)
            singleton = subset[("fraction", "Singleton")].to_numpy(float)
            delta = 100 * (expanded - singleton)
            low, high = bootstrap_median_ci(delta, rng)
            pval = wilcoxon(expanded, singleton, alternative="two-sided").pvalue if np.any(expanded != singleton) else 1.0
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "BcellState": state,
                    "n_participants": len(subset),
                    "median_change_pp": np.median(delta),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": pval,
                }
            )
        temp = pd.DataFrame(threshold_rows)
        temp["FDR_within_threshold"] = bh(temp["p_value"])
        rows.append(temp)
    return pd.concat(rows, ignore_index=True)


def program_effects() -> pd.DataFrame:
    enrichment = pd.read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA1_clone_aware_pseudobulk_module_enrichment.csv")
    enrichment = enrichment[
        (enrichment["modality"].str.upper() == "BCR")
        & ((enrichment["FDR_global"] < 0.05) | enrichment["module"].eq("BAFF_APRIL_survival_response"))
    ].copy()
    de = pd.read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA1_clone_aware_pseudobulk_DE_all_genes.csv", low_memory=False)
    de = de[(de["modality"].str.upper() == "BCR") & (de["coefficient"] == "statusExpanded")].set_index("gene")
    rows = []
    for _, record in enrichment.iterrows():
        module = record["module"]
        if module not in MODULE_GENES:
            continue
        genes = [g for g in MODULE_GENES[module] if g in de.index]
        if not genes:
            continue
        effects = de.loc[genes, "logFC"].astype(float).to_numpy()
        rows.append(
            {
                "module": module,
                "display": MODULE_LABELS[module],
                "direction": record["Direction"],
                "n_genes": len(genes),
                "median_member_gene_logFC": np.median(effects),
                "q1_member_gene_logFC": np.quantile(effects, 0.25),
                "q3_member_gene_logFC": np.quantile(effects, 0.75),
                "camera_global_FDR": record["FDR_global"],
            }
        )
    return pd.DataFrame(rows).sort_values("median_member_gene_logFC")


def representative_genes() -> pd.DataFrame:
    de = pd.read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA1_clone_aware_pseudobulk_DE_all_genes.csv", low_memory=False)
    de = de[(de["modality"].str.upper() == "BCR") & (de["coefficient"] == "statusExpanded")].copy()
    de = de[~de["gene"].astype(str).str.startswith("ENSG")].sort_values("FDR_global").head(8)
    de["SE"] = np.abs(de["logFC"] / de["t"].replace(0, np.nan))
    de["ci95_low"] = de["logFC"] - 1.96 * de["SE"]
    de["ci95_high"] = de["logFC"] + 1.96 * de["SE"]
    return de.sort_values("logFC")


def rank_residuals(frame: pd.DataFrame, target: str, covariates: list[str]) -> pd.Series:
    work = frame[[target] + covariates].dropna().copy()
    response = rankdata(work[target].to_numpy(float), method="average")
    columns = [np.ones(len(work))]
    for covariate in covariates:
        if work[covariate].dtype == object:
            levels = sorted(work[covariate].astype(str).unique())
            for level in levels[1:]:
                columns.append((work[covariate].astype(str) == level).astype(float).to_numpy())
        else:
            values = work[covariate].to_numpy(float)
            sd = np.nanstd(values)
            columns.append((values - np.nanmean(values)) / sd if sd > 0 else np.zeros(len(values)))
    design = np.column_stack(columns)
    residual = response - design @ np.linalg.lstsq(design, response, rcond=None)[0]
    residual = (residual - residual.mean()) / residual.std(ddof=1)
    return pd.Series(residual, index=work.index)


def coupling_scatter() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participant = pd.read_csv(ROOT / "High Impact Additional Analyses/Table_HI_integrated_participant_features.csv")
    coupling = pd.read_csv(ROOT / "High Impact Additional Analyses/Table_HI_CD_specific_T_B_module_coupling.csv")
    interactions = pd.read_csv(ROOT / "High Impact Additional Analyses/Table_HI_T_B_module_interaction_tests.csv")
    cd = participant[participant["Diagnosis1"].eq("CD")].copy()
    cd["log_tcr_depth"] = np.log1p(cd["tcr_total_cells"])
    cd["log_bcr_depth"] = np.log1p(cd["bcr_total_cells"])
    covariates = ["Age", "Sex", "log_tcr_depth", "log_bcr_depth"]
    rows = []
    for outcome in ["bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module"]:
        valid = cd[["tcr_cytotoxic_expanded", outcome] + covariates].dropna().index
        work = cd.loc[valid].copy()
        work["x_residual"] = rank_residuals(work, "tcr_cytotoxic_expanded", covariates)
        work["y_residual"] = rank_residuals(work, outcome, covariates)
        work["outcome"] = outcome
        rows.append(work[["SampleID", "Diagnosis1", "outcome", "x_residual", "y_residual"]])
    return pd.concat(rows, ignore_index=True), coupling, interactions


def bootstrap_regression_band(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = 1500):
    grid = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), 100)
    predictions = np.empty((n_boot, len(grid)))
    for i in range(n_boot):
        indices = rng.integers(0, len(x), len(x))
        coefficient = np.polyfit(x[indices], y[indices], 1)
        predictions[i] = coefficient[0] * grid + coefficient[1]
    coefficient = np.polyfit(x, y, 1)
    return grid, coefficient[0] * grid + coefficient[1], np.quantile(predictions, 0.025, axis=0), np.quantile(predictions, 0.975, axis=0)


def corrected_isotypes() -> tuple[pd.DataFrame, pd.DataFrame]:
    iso = pd.read_csv(PBMC_DIR / "Figure 5/BCR_isotype_proportions_recommended_manuscript_sample_level_values.csv")
    iso = iso[~iso["isotype"].str.startswith("Total ")].copy()
    mapping = {
        "IGHA1": "IgA", "IGHA2": "IgA", "IGHG1": "IgG", "IGHG2": "IgG", "IGHG3": "IgG", "IGHG4": "IgG",
        "IGHM": "IgM", "IGHD": "IgD", "IGHE": "IgE",
    }
    iso["class"] = iso["isotype"].map(mapping)
    sample_long = iso.groupby(["SampleID", "Diagnosis1", "class"], as_index=False)["prop"].sum()
    sample_long["prop"] = sample_long["prop"] / sample_long.groupby("SampleID")["prop"].transform("sum")
    wide = sample_long.pivot(index=["SampleID", "Diagnosis1"], columns="class", values="prop").fillna(0).reset_index()
    classes = ["IgM", "IgD", "IgA", "IgG"]
    parts = wide[classes].clip(lower=1e-6)
    clr = np.log(parts).sub(np.log(parts).mean(axis=1), axis=0)
    clr_frame = pd.concat([wide[["SampleID", "Diagnosis1"]].reset_index(drop=True), clr.reset_index(drop=True)], axis=1)
    metadata = pd.read_csv(ROOT / "High Impact Additional Analyses/Table_HI_integrated_participant_features.csv")
    clr_frame = clr_frame.merge(metadata[["SampleID", "Age", "Sex", "bcr_total_cells"]], on="SampleID", how="inner")
    clr_frame["Diagnosis1"] = pd.Categorical(clr_frame["Diagnosis1"], categories=DIAG)
    clr_frame["log_bcr_depth"] = np.log1p(clr_frame["bcr_total_cells"])
    rows = []
    for isotype in classes:
        clr_frame["response"] = clr_frame[isotype]
        model = smf.ols("response ~ C(Diagnosis1) + Age + C(Sex) + log_bcr_depth", data=clr_frame).fit(cov_type="HC3")
        for diagnosis in ["CD", "UC"]:
            term = f"C(Diagnosis1)[T.{diagnosis}]"
            rows.append(
                {
                    "isotype": isotype,
                    "comparison": f"{diagnosis} vs Control",
                    "diagnosis": diagnosis,
                    "adjusted_CLR_effect": model.params[term],
                    "SE_HC3": model.bse[term],
                    "ci95_low": model.conf_int().loc[term, 0],
                    "ci95_high": model.conf_int().loc[term, 1],
                    "p_value": model.pvalues[term],
                    "n": int(model.nobs),
                }
            )
    effects = pd.DataFrame(rows)
    effects["FDR_global"] = bh(effects["p_value"])
    return sample_long, effects


def switched_values_and_model() -> tuple[pd.DataFrame, pd.DataFrame]:
    switched = pd.read_csv(
        BCR_DIR / "BCR Architecture Analyses/BCR isotype switching diagnosis comparisons/tables/bcr_isotype_switched_fraction_by_sample.csv"
    ).rename(columns={"Diagnosis1": "Diagnosis"})
    metadata = pd.read_csv(ROOT / "High Impact Additional Analyses/Table_HI_integrated_participant_features.csv")
    frame = switched.merge(metadata[["SampleID", "Age", "Sex", "bcr_total_cells"]], on="SampleID", how="inner")
    frame["Diagnosis"] = pd.Categorical(frame["Diagnosis"], categories=DIAG)
    frame["log_bcr_depth"] = np.log1p(frame["bcr_total_cells"])
    frame["switch_logit"] = np.log((frame["switched_cells"] + 0.5) / (frame["total_isotyped"] - frame["switched_cells"] + 0.5))
    model = smf.ols("switch_logit ~ C(Diagnosis) + Age + C(Sex) + log_bcr_depth", data=frame).fit(cov_type="HC3")
    rows = []
    for diagnosis in ["CD", "UC"]:
        term = f"C(Diagnosis)[T.{diagnosis}]"
        rows.append(
            {
                "comparison": f"{diagnosis} vs Control",
                "diagnosis": diagnosis,
                "adjusted_log_odds_effect": model.params[term],
                "SE_HC3": model.bse[term],
                "ci95_low": model.conf_int().loc[term, 0],
                "ci95_high": model.conf_int().loc[term, 1],
                "p_value": model.pvalues[term],
                "n": int(model.nobs),
            }
        )
    effects = pd.DataFrame(rows)
    effects["FDR_within_switching"] = bh(effects["p_value"])
    return frame, effects


def raw_boxstrip(ax, frame: pd.DataFrame, group_col: str, value_col: str):
    rng = np.random.default_rng(421)
    arrays = [frame.loc[frame[group_col].astype(str).eq(group), value_col].dropna().to_numpy() for group in DIAG]
    boxes = ax.boxplot(
        arrays, positions=np.arange(3), widths=0.5, patch_artist=True, showfliers=False,
        medianprops={"color": INK, "lw": 1.0}, whiskerprops={"color": INK, "lw": 0.7},
        capprops={"color": INK, "lw": 0.7}, boxprops={"edgecolor": "#777777", "lw": 0.7},
    )
    for patch, group in zip(boxes["boxes"], DIAG):
        patch.set_facecolor(DIAG_COLORS[group])
        patch.set_alpha(0.2)
    for index, (group, values) in enumerate(zip(DIAG, arrays)):
        ax.scatter(index + rng.uniform(-0.13, 0.13, len(values)), values, s=7.5, color=DIAG_COLORS[group], alpha=0.5, edgecolor="none")
    ax.set_xticks(np.arange(3), [f"{group}\n(n={len(values)})" for group, values in zip(DIAG, arrays)])


def build():
    MAIN.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    LEG.mkdir(parents=True, exist_ok=True)

    state = state_threshold_analysis()
    programs = program_effects()
    genes = representative_genes()
    scatter, coupling, interactions = coupling_scatter()
    iso_sample, iso_effects = corrected_isotypes()
    switched, switched_effects = switched_values_and_model()

    fig = plt.figure(figsize=(7.48, 7.15), facecolor="white")
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.03], hspace=0.52, left=0.075, right=0.985, top=0.965, bottom=0.075)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.28, 1.15, 0.92], wspace=0.70)
    bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1.48, 1.0], wspace=0.58)

    # A: participant-level state enrichment with clone-threshold sensitivity.
    ax = fig.add_subplot(top[0, 0])
    main = state[state["threshold"].eq(2) & (state["n_participants"] >= 15)].copy()
    main = main.sort_values("median_change_pp")
    order = main["BcellState"].tolist()
    y = np.arange(len(order))
    for yi, (_, row) in zip(y, main.iterrows()):
        color = UP if row["median_change_pp"] > 0 else DOWN
        shown = color if row["FDR_within_threshold"] < 0.05 else NS
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=shown, lw=1.05)
        ax.scatter(row["median_change_pp"], yi, s=25, color=shown if row["FDR_within_threshold"] < 0.05 else "white",
                   edgecolor=shown, linewidth=0.8, zorder=4)
    offsets = {3: -0.13, 4: 0.13}
    markers = {3: "s", 4: "^"}
    for threshold in (3, 4):
        subset = state[state["threshold"].eq(threshold)].set_index("BcellState")
        for yi, state_name in enumerate(order):
            if state_name not in subset.index:
                continue
            row = subset.loc[state_name]
            color = UP if row["median_change_pp"] > 0 else DOWN
            ax.scatter(row["median_change_pp"], yi + offsets[threshold], s=14, marker=markers[threshold],
                       facecolor="white", edgecolor=color, linewidth=0.75, zorder=5)
    ax.axvline(0, color="#777777", lw=0.65, ls="--")
    ax.set_yticks(y, [f"{name}\n n={int(n)}" for name, n in zip(main["BcellState"], main["n_participants"])])
    ax.set_xlabel("Median expanded − singleton fraction\n(percentage points)")
    ax.set_title("State enrichment", loc="left", fontweight="bold", pad=4)
    ax.scatter([], [], s=22, marker="o", color="#555555", label="≥2 cells (95% CI)")
    ax.scatter([], [], s=14, marker="s", facecolor="white", edgecolor="#555555", label="≥3 cells")
    ax.scatter([], [], s=14, marker="^", facecolor="white", edgecolor="#555555", label="≥4 cells")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.98), handletextpad=0.35, labelspacing=0.22)
    style_axis(ax, "x")
    panel_label(ax, "A", x=-0.31)

    # B: effect-size summary of globally significant programs.
    ax = fig.add_subplot(top[0, 1])
    y = np.arange(len(programs))
    for yi, (_, row) in zip(y, programs.iterrows()):
        color = UP if row["direction"] == "Up" else DOWN
        ax.plot([row["q1_member_gene_logFC"], row["q3_member_gene_logFC"]], [yi, yi], color=color, lw=1.15)
        ax.scatter(row["median_member_gene_logFC"], yi, s=27, color=color, edgecolor="white", linewidth=0.35, zorder=4)
        ax.text(0.99, yi, q_label(row["camera_global_FDR"]), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.8, color="#555555")
    ax.axvline(0, color="#777777", lw=0.65, ls="--")
    ax.set_yticks(y, programs["display"])
    ax.set_xlabel("Median member-gene log2FC (IQR)\nexpanded versus singleton")
    ax.set_title("Clone-aware programs", loc="left", fontweight="bold", pad=4)
    style_axis(ax, "x")
    panel_label(ax, "B", x=-0.34)

    # C: representative adjusted gene effects.
    ax = fig.add_subplot(top[0, 2])
    y = np.arange(len(genes))
    for yi, (_, row) in zip(y, genes.iterrows()):
        significant = row["FDR_global"] < 0.05
        color = (UP if row["logFC"] > 0 else DOWN) if significant else NS
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=color, lw=1.0)
        ax.scatter(row["logFC"], yi, s=23, color=color if significant else "white", edgecolor=color, linewidth=0.8, zorder=4)
    ax.axvline(0, color="#777777", lw=0.65, ls="--")
    ax.set_yticks(y, genes["gene"])
    ax.set_xlabel("Adjusted log2FC (95% CI)")
    ax.set_title("Representative genes", loc="left", fontweight="bold", pad=4)
    style_axis(ax, "x")
    panel_label(ax, "C", x=-0.35)

    # D: CD-specific T-B coupling.
    dgrid = GridSpecFromSubplotSpec(1, 2, subplot_spec=bottom[0, 0], wspace=0.43)
    rng = np.random.default_rng(20260827)
    outcome_details = [
        ("bcr_IgA_mucosal_module", "IgA mucosal program"),
        ("bcr_plasma_differentiation_module", "Plasma-cell differentiation"),
    ]
    for index, (outcome, title) in enumerate(outcome_details):
        ax = fig.add_subplot(dgrid[0, index])
        data = scatter[scatter["outcome"].eq(outcome)]
        x = data["x_residual"].to_numpy(float)
        yv = data["y_residual"].to_numpy(float)
        grid, fit, low, high = bootstrap_regression_band(x, yv, rng)
        ax.fill_between(grid, low, high, color=DIAG_COLORS["CD"], alpha=0.13, linewidth=0)
        ax.plot(grid, fit, color=DIAG_COLORS["CD"], lw=1.35)
        ax.scatter(x, yv, s=14, color=DIAG_COLORS["CD"], alpha=0.66, edgecolor="white", linewidth=0.3)
        stat = coupling[(coupling["outcome"].eq(outcome)) & (coupling["diagnosis"].eq("CD"))].iloc[0]
        interaction = interactions[interactions["outcome"].eq(outcome)].iloc[0]
        ax.text(
            0.04, 0.97,
            f"partial ρ={stat['partial_rho']:.2f}\nFDR={stat['p_adj_within_six']:.2g}; n={int(stat['n'])}\ninteraction P={interaction['global_diagnosis_by_TCR_score_interaction_p']:.3g}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )
        if index == 0:
            ax.set_title("CD-specific T–B coupling\n" + title, loc="left", fontweight="bold", pad=4)
        else:
            ax.set_title("\n" + title, loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("Adjusted rank residual\nexpanded-TCR cytotoxicity")
        if index == 0:
            ax.set_ylabel("Adjusted rank residual\nB-cell program")
            panel_label(ax, "D", x=-0.27)
        style_axis(ax)

    # E: participant-level class switching and adjusted isotype composition.
    egrid = GridSpecFromSubplotSpec(2, 1, subplot_spec=bottom[0, 1], height_ratios=[0.95, 1.05], hspace=0.66)
    etop = GridSpecFromSubplotSpec(1, 2, subplot_spec=egrid[0, 0], width_ratios=[1.05, 0.95], wspace=0.48)
    ax = fig.add_subplot(etop[0, 0])
    raw_boxstrip(ax, switched, "Diagnosis", "switched_fraction")
    ax.set_ylabel("Class-switched fraction")
    ax.set_title("Antibody maturation", loc="left", fontweight="bold", pad=4)
    annotation = "\n".join(
        f"{row.diagnosis}: {q_label(row.FDR_within_switching)}" for row in switched_effects.itertuples(index=False)
    )
    ax.text(0.98, 0.98, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=5.9, color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0})
    style_axis(ax, "y")
    panel_label(ax, "E", x=-0.42, y=1.14)

    ax = fig.add_subplot(etop[0, 1])
    composition = iso_sample.groupby(["Diagnosis1", "class"], as_index=False)["prop"].mean().pivot(
        index="Diagnosis1", columns="class", values="prop"
    ).fillna(0).reindex(DIAG)
    classes = ["IgM", "IgD", "IgA", "IgG"]
    class_colors = {"IgM": "#56B4E9", "IgD": "#999999", "IgA": "#009E73", "IgG": "#CC79A7"}
    left = np.zeros(3)
    for isotype in classes:
        values = composition[isotype].to_numpy(float)
        ax.barh(np.arange(3), values, left=left, height=0.58, color=class_colors[isotype], label=isotype)
        left += values
    ax.set_xlim(0, 1.0)
    ax.set_yticks(np.arange(3), DIAG)
    ax.set_ylim(2.55, -0.85)
    ax.set_xlabel("Fraction")
    ax.set_title("Isotype composition", loc="left", fontweight="bold", pad=4)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.98), handlelength=1.25,
              columnspacing=0.65, handletextpad=0.35, labelspacing=0.18)
    style_axis(ax, "x")

    ax = fig.add_subplot(egrid[1, 0])
    isotype_order = ["IgG", "IgA", "IgD", "IgM"]
    positions = []
    labels = []
    for i, isotype in enumerate(isotype_order):
        for j, diagnosis in enumerate(["CD", "UC"]):
            row = iso_effects[(iso_effects["isotype"].eq(isotype)) & (iso_effects["diagnosis"].eq(diagnosis))].iloc[0]
            ypos = i * 2.0 + j * 0.55
            positions.append(ypos)
            labels.append(f"{isotype} · {diagnosis}" if j == 0 else f"      {diagnosis}")
            significant = row["FDR_global"] < 0.05
            color = DIAG_COLORS[diagnosis]
            ax.plot([row["ci95_low"], row["ci95_high"]], [ypos, ypos], color=color if significant else NS, lw=0.95)
            ax.scatter(row["adjusted_CLR_effect"], ypos, s=20, color=color if significant else "white",
                       edgecolor=color if significant else NS, linewidth=0.75, zorder=4)
    ax.axvline(0, color="#777777", lw=0.65, ls="--")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted centered-log-ratio effect\nversus Control (95% CI)")
    ax.text(0.98, 0.03, "filled: global FDR < 0.05", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.7, color="#555555")
    style_axis(ax, "x")

    stem = "Figure_4_CellPress_enhanced"
    fig.savefig(MAIN / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(MAIN / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(MAIN / f"{stem}.tif", dpi=500, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    state.to_csv(SRC / "Figure4_enhanced_state_threshold_sensitivity.csv", index=False)
    programs.to_csv(SRC / "Figure4_enhanced_program_effects.csv", index=False)
    genes.to_csv(SRC / "Figure4_enhanced_representative_gene_effects.csv", index=False)
    scatter.to_csv(SRC / "Figure4_enhanced_CD_T_B_coupling_scatter.csv", index=False)
    coupling.to_csv(SRC / "Figure4_enhanced_T_B_coupling_statistics.csv", index=False)
    interactions.to_csv(SRC / "Figure4_enhanced_T_B_interaction_tests.csv", index=False)
    iso_sample.to_csv(SRC / "Figure4_enhanced_isotype_participant_composition.csv", index=False)
    iso_effects.to_csv(SRC / "Figure4_enhanced_isotype_CLR_effects.csv", index=False)
    switched.to_csv(SRC / "Figure4_enhanced_class_switching_participant_values.csv", index=False)
    switched_effects.to_csv(SRC / "Figure4_enhanced_class_switching_adjusted_effects.csv", index=False)

    legend = """Figure 4. Expanded BCR clones acquire antibody-secreting programs and coordinate with cytotoxic T-cell activity in Crohn's disease. (A) Participant-level enrichment of B-cell states among expanded relative to singleton clonotypes. State fractions were calculated separately within expanded and singleton BCR-bearing cells. Circles show the primary expansion definition (clonotype size >=2 cells), with bootstrap 95% confidence intervals; squares and triangles show sensitivity analyses using thresholds of >=3 and >=4 cells. Participants were included for a state when they contributed at least 5 expanded and 20 singleton cells. Filled primary-analysis points indicate Benjamini-Hochberg FDR <0.05 across states. (B) Effect-size summary for five antibody- and plasma-cell programs passing global cameraPR FDR <0.05 in the state-matched, participant-paired pseudobulk analysis. Points and lines show the median and interquartile range of adjusted member-gene log2 fold changes; displayed q values are global cameraPR FDR values, which account for inter-gene correlation. (C) Representative adjusted gene-level effects from the same pseudobulk model. Points and lines show log2 fold changes and 95% confidence intervals; filled points indicate global FDR <0.05. Models adjusted for RNA feature complexity and mitochondrial read fraction. (D) Crohn's disease-specific coupling between expanded-TCR cytotoxicity and the BCR IgA mucosal plasma-cell or plasmablast/plasma-cell differentiation programs. Axes show rank residuals after adjustment for age, sex, TCR cell depth, and BCR cell depth. Lines show linear fits with participant-bootstrap 95% confidence bands. Partial Spearman correlations were corrected across six diagnosis-program tests; interaction P values test whether the TCR-BCR association differed by diagnosis. (E) Participant-level class-switched BCR fractions and adjusted immunoglobulin-class composition. Points denote participants; boxes show medians and interquartile ranges, with whiskers extending to 1.5 times the interquartile range. Class-switching q values derive from HC3-robust models of log odds adjusted for age, sex, and BCR depth. Horizontal bars show the participant-mean isotype composition and sum to 100%. Isotype effects are centered-log-ratio coefficients for CD or UC versus Control with HC3-robust 95% confidence intervals, adjusted for age, sex, and BCR depth; filled points indicate global FDR <0.05 across isotype contrasts. IGHA1 and IGHA2 were summed as IgA, and IGHG1-4 were summed as IgG before compositional analysis. BCR, B cell receptor; CD, Crohn's disease; CLR, centered log ratio; FDR, false discovery rate; TCR, T cell receptor; UC, ulcerative colitis; UPR, unfolded protein response."""
    (LEG / f"{stem}_legend.txt").write_text(legend + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()

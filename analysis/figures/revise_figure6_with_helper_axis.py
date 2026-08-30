#!/usr/bin/env python3
"""Rebuild main Figure 6 with the helper–B-cell axis and export its interaction audit."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from pypdf import PdfReader, PdfWriter
from scipy.stats import rankdata, spearmanr, t as student_t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Cell Press Redrawn Figure Set"
MAIN = OUT / "Main Figures"
SUPP = OUT / "Supplementary Figures"
SRC = OUT / "Source Data"
LEG = OUT / "Legends"
RA = ROOT / "High Impact Additional Analyses" / "Literature Guided Ranked Analyses"
HI = ROOT / "High Impact Additional Analyses"

for directory in (MAIN, SUPP, SRC, LEG):
    directory.mkdir(parents=True, exist_ok=True)


COL = {
    "Control": "#6F6F6F",
    "CD": "#0072B2",
    "UC": "#D55E00",
    "ink": "#222222",
    "muted": "#777777",
    "grid": "#D8D8D8",
    "purple": "#7B3294",
}
DIAG = ["Control", "CD", "UC"]

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
)


def style_axis(ax, grid_axis: str | None = None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["ink"])
    ax.spines["bottom"].set_color(COL["ink"])
    ax.tick_params(color=COL["ink"], labelcolor=COL["ink"], pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COL["grid"], lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.11):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11.5, fontweight="bold", va="top", ha="left")


def panel_title(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, loc="left", fontweight="bold", pad=14 if subtitle else 5)
    if subtitle:
        ax.text(0, 1.005, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=5.8, color=COL["muted"])


def save_figure(fig, stem: str, folder: Path):
    opts = {"bbox_inches": "tight", "pad_inches": 0.045, "facecolor": "white"}
    fig.savefig(folder / f"{stem}.pdf", **opts)
    fig.savefig(folder / f"{stem}.png", dpi=350, **opts)
    fig.savefig(folder / f"{stem}.tif", dpi=500, pil_kwargs={"compression": "tiff_lzw"}, **opts)
    plt.close(fig)


def add_heatmap(ax, matrix: pd.DataFrame, fdr: pd.DataFrame | None = None, vlim: float = 0.8,
                colorbar: bool = True, annotate_size: float = 5.2):
    values = matrix.to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            stars = ""
            if fdr is not None:
                q = float(fdr.iloc[i, j]) if pd.notna(fdr.iloc[i, j]) else np.nan
                if np.isfinite(q):
                    stars = "**" if q < 0.01 else ("*" if q < 0.05 else "")
            color = "white" if abs(val) >= 0.48 else COL["ink"]
            ax.text(j, i, f"{val:.2f}{stars}", ha="center", va="center",
                    fontsize=annotate_size, color=color)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
        cb.set_label("Partial Spearman ρ", fontsize=6.2)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=5.6, width=0.5, length=2)
    return im


def bootstrap_line(ax, x: np.ndarray, y: np.ndarray, color: str, seed: int):
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    grid = np.linspace(x.min(), x.max(), 140)
    coef = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(1500):
        idx = rng.integers(0, len(x), len(x))
        predictions.append(np.polyval(np.polyfit(x[idx], y[idx], 1), grid))
    predictions = np.asarray(predictions)
    ax.fill_between(grid, np.quantile(predictions, 0.025, axis=0),
                    np.quantile(predictions, 0.975, axis=0), color=color, alpha=0.14, lw=0)
    ax.plot(grid, np.polyval(coef, grid), color=color, lw=1.25)
    return x, y


def _numeric_design(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str]):
    continuous = predictors + ["Age", "log_tcr_depth", "log_bcr_depth"]
    train_columns = [np.ones(len(train))]
    test_columns = [np.ones(len(test))]
    for column in continuous:
        train_values = pd.to_numeric(train[column], errors="coerce").to_numpy(float)
        test_values = pd.to_numeric(test[column], errors="coerce").to_numpy(float)
        mean = train_values.mean()
        sd = train_values.std(ddof=1)
        sd = sd if np.isfinite(sd) and sd > 0 else 1.0
        train_columns.append((train_values - mean) / sd)
        test_columns.append((test_values - mean) / sd)
    train_columns.append((train["Sex"].astype(str).str.upper() == "M").astype(float).to_numpy())
    test_columns.append((test["Sex"].astype(str).str.upper() == "M").astype(float).to_numpy())
    return np.column_stack(train_columns), np.column_stack(test_columns)


def blocked_series_predictions(data: pd.DataFrame, outcome: str, predictors: list[str]):
    columns = ["SampleID", "acquisition_series", outcome, "Age", "Sex",
               "log_tcr_depth", "log_bcr_depth"] + predictors
    work = data[columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    rows = []
    for series in sorted(work["acquisition_series"].astype(str).unique()):
        train = work[work["acquisition_series"].astype(str) != series]
        test = work[work["acquisition_series"].astype(str) == series]
        if len(train) <= len(predictors) + 5 or len(test) == 0:
            continue
        x_train, x_test = _numeric_design(train, test, predictors)
        beta = np.linalg.lstsq(x_train, train[outcome].to_numpy(float), rcond=None)[0]
        prediction = x_test @ beta
        for (_, row), value in zip(test.iterrows(), prediction):
            rows.append({"SampleID": row.SampleID, "acquisition_series": series,
                         "observed": row[outcome], "predicted": value})
    result = pd.DataFrame(rows)
    rho = float(spearmanr(result["observed"], result["predicted"]).statistic) if len(result) > 3 else np.nan
    return result, rho


def partial_rank_correlation(data: pd.DataFrame, predictor: str, outcome: str, covariates: list[str]):
    columns = [predictor, outcome] + covariates
    work = data[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()

    def residual(column):
        y = rankdata(work[column].to_numpy(float), method="average")
        design = [np.ones(len(work))]
        for covariate in covariates:
            if work[covariate].dtype == object:
                levels = sorted(work[covariate].astype(str).unique())
                design.extend((work[covariate].astype(str) == level).astype(float).to_numpy()
                              for level in levels[1:])
            else:
                values = pd.to_numeric(work[covariate], errors="coerce").to_numpy(float)
                sd = values.std(ddof=1)
                design.append((values - values.mean()) / sd if sd > 0 else np.zeros(len(values)))
        matrix = np.column_stack(design)
        return y - matrix @ np.linalg.lstsq(matrix, y, rcond=None)[0]

    return float(np.corrcoef(residual(predictor), residual(outcome))[0, 1]), len(work)


def bootstrap_partial_rank(data: pd.DataFrame, predictor: str, outcome: str,
                           covariates: list[str], seed: int, n_boot: int = 2000):
    columns = [predictor, outcome] + covariates
    work = data[columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    estimate, n = partial_rank_correlation(work, predictor, outcome, covariates)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = work.iloc[rng.integers(0, len(work), len(work))].reset_index(drop=True)
        try:
            value, _ = partial_rank_correlation(sample, predictor, outcome, covariates)
            if np.isfinite(value):
                draws.append(value)
        except np.linalg.LinAlgError:
            continue
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), n


def backup_current_outputs():
    for suffix in ("pdf", "png", "tif"):
        source = MAIN / f"Figure_6.{suffix}"
        backup = MAIN / f"Figure_6_pre_helper_axis.{suffix}"
        if source.exists() and not backup.exists():
            copy2(source, backup)
    old_legend = LEG / "Figure_6_legend.txt"
    legend_backup = LEG / "Figure_6_pre_helper_axis_legend.txt"
    if old_legend.exists() and not legend_backup.exists():
        copy2(old_legend, legend_backup)
    combined = OUT / "Main_Figures_1-7_Cell_Press.pdf"
    combined_backup = OUT / "Main_Figures_1-7_Cell_Press_pre_helper_axis.pdf"
    if combined.exists() and not combined_backup.exists():
        copy2(combined, combined_backup)


def build_main_figure():
    global_corr = pd.read_csv(HI / "Table_HI_partial_spearman_TCR_BCR_coordination.csv")
    helper = pd.read_csv(RA / "Table_RA1_helper_B_partial_correlations.csv")
    helper_data = pd.read_csv(RA / "Table_RA1_helper_axis_analysis_dataset.csv")
    helper_interactions = pd.read_csv(RA / "Table_RA1_helper_B_CD_UC_interactions.csv")
    coupling = pd.read_csv(SRC / "Figure_6_coupling_effects.csv")
    residuals = pd.read_csv(SRC / "Figure_6_adjusted_rank_residuals.csv")
    sensitivity = pd.read_csv(SRC / "Figure_6_sensitivity.csv")
    permutation = pd.read_csv(SRC / "Figure_6_matched_pair_permutation.csv")
    integrated = pd.read_csv(HI / "Table_HI_integrated_participant_features.csv")
    integrated["log_tcr_depth"] = np.log1p(integrated["tcr_total_cells"])
    integrated["log_bcr_depth"] = np.log1p(integrated["bcr_total_cells"])

    # Preserve the complete source data used by each displayed panel.
    global_corr.to_csv(SRC / "Figure_6_panel_A_global_coordination.csv", index=False)
    coupling.to_csv(SRC / "Figure_6_panel_D_cytotoxic_coupling.csv", index=False)
    residuals.to_csv(SRC / "Figure_6_panels_EF_adjusted_rank_residuals.csv", index=False)
    sensitivity.to_csv(SRC / "Figure_6_panel_F_sensitivity.csv", index=False)
    permutation.to_csv(SRC / "Figure_6_panel_F_matched_permutation.csv", index=False)

    tcr_order = [
        "tcr_clonality", "tcr_gini", "tcr_expanded_cell_fraction",
        "tcr_multistate_clone_fraction", "tcr_cytotoxic_expanded",
    ]
    bcr_order = [
        "bcr_gini", "bcr_expanded_cell_fraction", "bcr_multistate_clone_fraction",
        "bcr_switched_fraction", "bcr_SHM_rate", "bcr_IgA_mucosal_module",
        "bcr_plasma_differentiation_module", "bcr_BAFF_APRIL_module",
    ]
    tcr_labels = {
        "tcr_clonality": "TCR clonality", "tcr_gini": "TCR Gini",
        "tcr_expanded_cell_fraction": "Expanded TCR-cell fraction",
        "tcr_multistate_clone_fraction": "Multistate TCR-clone fraction",
        "tcr_cytotoxic_expanded": "Expanded-TCR cytotoxic score",
    }
    bcr_labels = {
        "bcr_gini": "BCR\nGini", "bcr_expanded_cell_fraction": "Expanded BCR-\ncell fraction",
        "bcr_multistate_clone_fraction": "Multistate BCR-\nclone fraction",
        "bcr_switched_fraction": "Class-switched\nBCR fraction", "bcr_SHM_rate": "BCR SHM\nrate",
        "bcr_IgA_mucosal_module": "IgA mucosal\nprogram",
        "bcr_plasma_differentiation_module": "Plasma-cell\ndifferentiation",
        "bcr_BAFF_APRIL_module": "BAFF/APRIL\nprogram",
    }
    mat_a = global_corr.pivot(index="tcr_metric", columns="bcr_metric", values="partial_rho").reindex(
        index=tcr_order, columns=bcr_order
    )
    fdr_a = global_corr.pivot(index="tcr_metric", columns="bcr_metric", values="p_adj").reindex(
        index=tcr_order, columns=bcr_order
    )
    mat_a.index = [tcr_labels[x] for x in mat_a.index]
    mat_a.columns = [bcr_labels[x] for x in mat_a.columns]
    fdr_a.index, fdr_a.columns = mat_a.index, mat_a.columns

    selected_pairs = [
        ("cd4_mean_Tph_core", "b_mean_Plasmablast_plasma_differentiation", "Tph → plasma differentiation"),
        ("cd4_mean_Tph_core", "b_mean_IgA_mucosal_plasma_cell", "Tph → IgA mucosal plasma"),
        ("cd4_mean_Tph_core", "b_mean_B_cell_antigen_presentation", "Tph → B-cell antigen presentation"),
        ("cd4_mean_Tfh_core", "b_mean_Plasmablast_plasma_differentiation", "Tfh → plasma differentiation"),
        ("cd4_mean_Tfh_core", "b_mean_IgA_mucosal_plasma_cell", "Tfh → IgA mucosal plasma"),
        ("cd4_clone_delta_Tph_core", "b_mean_Plasmablast_plasma_differentiation", "Clone-associated Tph → plasma differentiation"),
        ("cd4_clone_delta_Tph_core", "b_mean_IgA_mucosal_plasma_cell", "Clone-associated Tph → IgA mucosal plasma"),
        ("cd4_clone_delta_Tfh_core", "b_mean_Plasmablast_plasma_differentiation", "Clone-associated Tfh → plasma differentiation"),
    ]
    rows_b = []
    for t_feature, b_feature, label in selected_pairs:
        ss = helper[(helper["T_feature"] == t_feature) & (helper["B_feature"] == b_feature)].copy()
        ss["pair_label"] = label
        rows_b.append(ss)
    helper_selected = pd.concat(rows_b, ignore_index=True)
    helper_selected.to_csv(SRC / "Figure_6_panel_B_helper_axis.csv", index=False)
    pair_order = [x[2] for x in selected_pairs]
    mat_b = helper_selected.pivot(index="pair_label", columns="Diagnosis", values="partial_rho").reindex(
        index=pair_order, columns=["CD", "UC", "Control"]
    )
    fdr_b = helper_selected.pivot(index="pair_label", columns="Diagnosis", values="FDR_within_diagnosis").reindex(
        index=pair_order, columns=["CD", "UC", "Control"]
    )

    cd_scatter = helper_data[helper_data["Diagnosis1"] == "CD"].copy()
    cd_scatter[["SampleID", "Diagnosis1", "cd4_mean_Tph_core",
                "b_mean_Plasmablast_plasma_differentiation"]].to_csv(
        SRC / "Figure_6_panel_C_helper_scatter.csv", index=False
    )

    # Directly test whether the CD cytotoxic-plasma association remains after accounting
    # for the broad helper axis. Rank-standardized coefficients keep the two predictors
    # on the same scale while preserving the primary age/sex/depth covariate structure.
    joint = helper_data[["SampleID", "acquisition_series", "cd4_mean_Tph_Tfh_help",
                         "b_mean_IgA_mucosal_plasma_cell",
                         "b_mean_Plasmablast_plasma_differentiation"]].merge(
        integrated[["SampleID", "Diagnosis1", "Age", "Sex", "log_tcr_depth", "log_bcr_depth",
                    "tcr_cytotoxic_expanded"]],
        on="SampleID", how="inner"
    )
    joint = joint[joint["Diagnosis1"] == "CD"].copy()
    joint_predictors = ["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"]
    joint_outcomes = ["b_mean_IgA_mucosal_plasma_cell",
                      "b_mean_Plasmablast_plasma_differentiation"]

    def joint_coefficients(data: pd.DataFrame, outcome: str):
        columns = [outcome] + joint_predictors + ["Age", "Sex", "log_tcr_depth", "log_bcr_depth"]
        work = data[columns].dropna().reset_index(drop=True)

        def ranked_standardized(series):
            values = pd.Series(series).rank(method="average").to_numpy(float)
            sd = values.std(ddof=1)
            return (values - values.mean()) / sd if sd > 0 else np.zeros(len(values))

        y = ranked_standardized(work[outcome])
        design = [np.ones(len(work))]
        for predictor in joint_predictors:
            design.append(ranked_standardized(work[predictor]))
        for covariate in ["Age", "log_tcr_depth", "log_bcr_depth"]:
            values = pd.to_numeric(work[covariate], errors="coerce").to_numpy(float)
            sd = values.std(ddof=1)
            design.append((values - values.mean()) / sd if sd > 0 else np.zeros(len(values)))
        design.append((work["Sex"].astype(str).str.upper() == "M").astype(float).to_numpy())
        beta = np.linalg.lstsq(np.column_stack(design), y, rcond=None)[0]
        return beta[1:3], len(work)

    joint_rows = []
    rng = np.random.default_rng(20260828)
    for outcome in joint_outcomes:
        complete_columns = [outcome] + joint_predictors + ["Age", "Sex", "log_tcr_depth", "log_bcr_depth"]
        work = joint[complete_columns].dropna().reset_index(drop=True)
        estimate, n_joint = joint_coefficients(work, outcome)
        draws = []
        for _ in range(3000):
            sample = work.iloc[rng.integers(0, len(work), len(work))].reset_index(drop=True)
            try:
                draw, _ = joint_coefficients(sample, outcome)
                if np.all(np.isfinite(draw)):
                    draws.append(draw)
            except np.linalg.LinAlgError:
                continue
        draws = np.asarray(draws)
        for index, predictor in enumerate(joint_predictors):
            joint_rows.append({
                "outcome": outcome,
                "predictor": predictor,
                "standardized_rank_beta": estimate[index],
                "ci_low": np.quantile(draws[:, index], 0.025),
                "ci_high": np.quantile(draws[:, index], 0.975),
                "n": n_joint,
                "bootstrap_draws": len(draws),
            })
    joint_results = pd.DataFrame(joint_rows)
    joint_results.to_csv(SRC / "Figure_6_panel_D_joint_helper_cytotoxic_model.csv", index=False)

    # Highest-priority validation: leave one acquisition series out, then quantify
    # the incremental held-out performance of the joint helper/cytotoxic model.
    validation_models = {
        "Helper only": ["cd4_mean_Tph_Tfh_help"],
        "Cytotoxic only": ["tcr_cytotoxic_expanded"],
        "Joint": ["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"],
    }
    validation_rows, validation_predictions = [], []
    validation_rng = np.random.default_rng(20260828)
    for outcome in joint_outcomes:
        model_rho = {}
        for model, predictors in validation_models.items():
            predictions, rho = blocked_series_predictions(joint, outcome, predictors)
            predictions["outcome"] = outcome
            predictions["model"] = model
            validation_predictions.append(predictions)
            model_rho[model] = rho
        observed_delta = model_rho["Joint"] - model_rho["Helper only"]
        null_delta = []
        complete = joint[["SampleID", "acquisition_series", outcome, "Age", "Sex",
                          "log_tcr_depth", "log_bcr_depth", "cd4_mean_Tph_Tfh_help",
                          "tcr_cytotoxic_expanded"]].dropna().reset_index(drop=True)
        for _ in range(2000):
            permuted = complete.copy()
            for series in permuted["acquisition_series"].astype(str).unique():
                mask = permuted["acquisition_series"].astype(str) == series
                permuted.loc[mask, "tcr_cytotoxic_expanded"] = validation_rng.permutation(
                    permuted.loc[mask, "tcr_cytotoxic_expanded"].to_numpy()
                )
            _, permuted_rho = blocked_series_predictions(
                permuted, outcome, ["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"]
            )
            null_delta.append(permuted_rho - model_rho["Helper only"])
        empirical_p = (1 + np.sum(np.asarray(null_delta) >= observed_delta)) / (len(null_delta) + 1)
        for model in validation_models:
            validation_rows.append({"outcome": outcome, "model": model,
                                    "held_out_spearman_rho": model_rho[model],
                                    "joint_minus_helper_delta": observed_delta if model == "Joint" else np.nan,
                                    "incremental_permutation_p": empirical_p if model == "Joint" else np.nan,
                                    "permutations": len(null_delta) if model == "Joint" else np.nan})
    validation_summary = pd.DataFrame(validation_rows)
    validation_prediction_data = pd.concat(validation_predictions, ignore_index=True)
    validation_summary.to_csv(SRC / "Figure_6_panel_G_blocked_validation_summary.csv", index=False)
    validation_prediction_data.to_csv(SRC / "Figure_6_panel_G_blocked_predictions.csv", index=False)

    fig = plt.figure(figsize=(7.48, 10.75), facecolor="white")
    gs = GridSpec(5, 6, figure=fig, height_ratios=[1.08, 1.48, 1.28, 1.70, 0.90],
                  hspace=0.92, wspace=1.40, left=0.20, right=0.985, top=0.910, bottom=0.060)

    ax = fig.add_subplot(gs[0, :])
    add_heatmap(ax, mat_a, fdr_a, vlim=0.8, colorbar=True, annotate_size=5.1)
    ax.tick_params(axis="x", pad=3)
    panel_title(ax, "Global repertoire and program coordination",
                "Covariate-adjusted participant-level correlations; *FDR<0.05, **FDR<0.01")
    panel_label(ax, "A", x=-0.10, y=1.23)

    ax = fig.add_subplot(gs[1, :4])
    add_heatmap(ax, mat_b, fdr_b, vlim=0.8, colorbar=False, annotate_size=5.0)
    ax.axhline(4.5, color="white", lw=2.0)
    panel_title(ax, "Helper T-cell–B-cell coordination across diagnoses",
                "Broad programs are shared; paired-clone estimates are underpowered")
    panel_label(ax, "B", x=-0.28, y=1.13)

    ax = fig.add_subplot(gs[1, 4:])
    x = pd.to_numeric(cd_scatter["cd4_mean_Tph_core"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(cd_scatter["b_mean_Plasmablast_plasma_differentiation"], errors="coerce").to_numpy(float)
    x, y = bootstrap_line(ax, x, y, COL["CD"], 20260828)
    ax.scatter(x, y, s=11, color=COL["CD"], alpha=0.55, edgecolor="white", lw=0.3, rasterized=True)
    row = helper[(helper["Diagnosis"] == "CD") & (helper["T_feature"] == "cd4_mean_Tph_core") &
                 (helper["B_feature"] == "b_mean_Plasmablast_plasma_differentiation")].iloc[0]
    ix = helper_interactions[(helper_interactions["T_feature"] == "cd4_mean_Tph_core") &
                             (helper_interactions["B_feature"] == "b_mean_Plasmablast_plasma_differentiation")].iloc[0]
    ax.text(0.03, 0.98,
            f"partial ρ={row.partial_rho:.2f}; FDR={row.FDR_within_diagnosis:.2g}; n={int(row.n)}\n"
            f"CD–UC interaction FDR={ix.FDR:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.7,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.86})
    ax.set_xlabel("Tph program score")
    ax.set_ylabel("Plasma-differentiation\nprogram score", labelpad=2)
    panel_title(ax, "Representative CD association")
    panel_label(ax, "C", x=-0.28, y=1.13)
    style_axis(ax)

    outcomes = ["bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module"]

    ax = fig.add_subplot(gs[2, :3])
    joint_order = [
        ("b_mean_IgA_mucosal_plasma_cell", "cd4_mean_Tph_Tfh_help", "IgA | helper axis"),
        ("b_mean_IgA_mucosal_plasma_cell", "tcr_cytotoxic_expanded", "IgA | expanded cytotoxic"),
        ("b_mean_Plasmablast_plasma_differentiation", "cd4_mean_Tph_Tfh_help", "Plasma | helper axis"),
        ("b_mean_Plasmablast_plasma_differentiation", "tcr_cytotoxic_expanded", "Plasma | expanded cytotoxic"),
    ]
    joint_y = [3.35, 2.55, 1.25, 0.45]
    for y0, (outcome, predictor, label) in zip(joint_y, joint_order):
        row_joint = joint_results[(joint_results["outcome"] == outcome) &
                                  (joint_results["predictor"] == predictor)].iloc[0]
        color = COL["purple"] if predictor == "cd4_mean_Tph_Tfh_help" else COL["CD"]
        ax.errorbar(row_joint.standardized_rank_beta, y0,
                    xerr=[[row_joint.standardized_rank_beta - row_joint.ci_low],
                          [row_joint.ci_high - row_joint.standardized_rank_beta]],
                    fmt="o", color=color, ms=4.3, capsize=2, lw=1.0, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(joint_y, [item[2] for item in joint_order])
    ax.set_xlim(-0.30, 0.78)
    ax.set_xlabel("Joint standardized rank coefficient (95% CI)")
    n_joint = int(joint_results["n"].min())
    panel_title(ax, "Complementary helper and cytotoxic axes in CD",
                f"Joint model; age, sex, and TCR/BCR depth adjusted; n={n_joint}")
    panel_label(ax, "D", x=-0.25, y=1.17)
    style_axis(ax, "x")

    ax = fig.add_subplot(gs[2, 3:])
    y_positions = [5.15, 4.25, 3.35, 1.55, 0.65, -0.25]
    ordered = []
    for outcome in outcomes:
        for diagnosis in DIAG:
            ordered.append(coupling[(coupling["outcome"] == outcome) &
                                     (coupling["diagnosis"] == diagnosis)].iloc[0])
    for y0, row_coupling in zip(y_positions, ordered):
        ax.errorbar(row_coupling.partial_rho, y0,
                    xerr=[[row_coupling.partial_rho - row_coupling.ci_low],
                          [row_coupling.ci_high - row_coupling.partial_rho]],
                    fmt="o", color=COL[row_coupling.diagnosis], ms=4.1, capsize=2, lw=0.95, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    forest_labels = [
        f"IgA | {d} (n={int(row_coupling.n)})" if i < 3 else f"Plasma | {d} (n={int(row_coupling.n)})"
        for i, (d, row_coupling) in enumerate(zip(DIAG + DIAG, ordered))
    ]
    ax.set_yticks(y_positions, forest_labels)
    ax.set_xlim(-0.90, 0.90)
    ax.set_xlabel("Partial Spearman ρ (95% CI)")
    ax.text(0.98, 0.49, "IgA interaction P=0.0025\nPlasma interaction P=0.0013",
            transform=ax.transAxes, fontsize=5.9, fontweight="bold", va="center", ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.90})
    panel_title(ax, "CD-specific cytotoxic-plasma extension")
    panel_label(ax, "E", x=-0.27, y=1.17)
    style_axis(ax, "x")

    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, :], width_ratios=[1.65, 1.0], wspace=0.86)
    ax = fig.add_subplot(sub[0, 0])
    model_order = ["Primary", "+ acquisition series", "+ inflammation status",
                   "+ biologic exposure", "+ cell-state composition"]
    sensitivity_ordered = []
    for outcome in outcomes:
        for model in model_order:
            sensitivity_ordered.append(sensitivity[(sensitivity["outcome"] == outcome) &
                                                   (sensitivity["model"] == model)].iloc[0])
    sensitivity_y = [9.2, 8.45, 7.70, 6.95, 6.20, 4.55, 3.80, 3.05, 2.30, 1.55]
    model_colors = {
        "Primary": COL["CD"], "+ acquisition series": "#009E73",
        "+ inflammation status": "#E69F00", "+ biologic exposure": "#CC79A7",
        "+ cell-state composition": "#7A7F87",
    }
    row_labels = []
    for y0, row_sensitivity in zip(sensitivity_y, sensitivity_ordered):
        ax.errorbar(row_sensitivity.rho, y0,
                    xerr=[[row_sensitivity.rho - row_sensitivity.ci_low],
                          [row_sensitivity.ci_high - row_sensitivity.rho]],
                    fmt="o", color=model_colors[row_sensitivity.model], ms=4.0,
                    capsize=2, lw=0.95, zorder=3)
        label = row_sensitivity.model
        if int(row_sensitivity.n) < 56:
            label += f" (n={int(row_sensitivity.n)})"
        row_labels.append(label)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(sensitivity_y, row_labels)
    ax.set_xlim(-0.12, 0.84)
    ax.set_xlabel("Partial Spearman ρ (95% bootstrap CI)")
    header_box = {"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.92}
    ax.text(0.02, 1.015, "IgA mucosal plasma-cell program", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", va="bottom", bbox=header_box)
    ax.text(0.02, 0.50, "Plasma-cell differentiation", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", va="center", bbox=header_box)
    ax.set_title("Clinical and cell-composition sensitivity", loc="left", fontweight="bold", pad=18)
    panel_label(ax, "F", x=-0.22, y=1.11)
    style_axis(ax, "x")

    ax = fig.add_subplot(sub[0, 1])
    perm_labels = {
        "bcr_IgA_mucosal_module": "IgA mucosal",
        "bcr_plasma_differentiation_module": "Plasma differentiation",
    }
    for y0, outcome in zip([1.0, 0.0], outcomes):
        row_perm = permutation[permutation["outcome"] == outcome].iloc[0]
        ax.hlines(y0, row_perm.null_ci_low, row_perm.null_ci_high,
                  color="#9AA0A6", lw=5.0, alpha=0.55, zorder=1)
        ax.scatter(row_perm.observed_rho, y0, s=32, color=COL["CD"],
                   edgecolor="white", lw=0.4, zorder=3)
        ax.text(0.02, y0, f"Pperm={row_perm.empirical_p:.3g}",
                ha="left", va="center", fontsize=5.9,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88})
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks([1.0, 0.0], [perm_labels[x] for x in outcomes])
    ax.set_xlim(-0.10, 0.72)
    ax.set_xlabel("Partial Spearman ρ")
    panel_title(ax, "Matched-participant permutation",
                "Gray: 95% within-series re-pairing null; blue: observed")
    style_axis(ax, "x")

    ax = fig.add_subplot(gs[4, :])
    validation_labels = {
        "b_mean_IgA_mucosal_plasma_cell": "IgA mucosal plasma-cell program",
        "b_mean_Plasmablast_plasma_differentiation": "Plasma-cell differentiation",
    }
    validation_colors = {"Helper only": COL["purple"], "Cytotoxic only": COL["CD"], "Joint": COL["ink"]}
    validation_markers = {"Helper only": "o", "Cytotoxic only": "s", "Joint": "D"}
    base_y = {joint_outcomes[0]: 1.0, joint_outcomes[1]: 0.0}
    offsets = {"Helper only": 0.16, "Cytotoxic only": 0.0, "Joint": -0.16}
    for outcome in joint_outcomes:
        for model in validation_models:
            row_validation = validation_summary[(validation_summary["outcome"] == outcome) &
                                                (validation_summary["model"] == model)].iloc[0]
            ax.scatter(row_validation.held_out_spearman_rho,
                       base_y[outcome] + offsets[model], s=31,
                       color=validation_colors[model], marker=validation_markers[model],
                       edgecolor="white", lw=0.45, zorder=3)
        joint_row = validation_summary[(validation_summary["outcome"] == outcome) &
                                       (validation_summary["model"] == "Joint")].iloc[0]
        ax.text(0.98, base_y[outcome],
                f"Δjoint-helper={joint_row.joint_minus_helper_delta:+.2f}; Pperm={joint_row.incremental_permutation_p:.3g}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.0,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.90})
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks([1.0, 0.0], [validation_labels[x] for x in joint_outcomes])
    ax.set_xlim(-0.35, 0.92)
    ax.set_xlabel("Leave-one-acquisition-series-out Spearman correlation")
    handles = [Line2D([0], [0], marker=validation_markers[m], color="none",
                      markerfacecolor=validation_colors[m], markeredgecolor="white",
                      markersize=5.5, label=m) for m in validation_models]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.00))
    panel_title(ax, "Blocked out-of-series validation of helper and cytotoxic models",
                "Models trained on all other acquisition series; cytotoxic predictor permuted within series")
    panel_label(ax, "G", x=-0.10, y=1.22)
    style_axis(ax, "x")

    fig.suptitle("Shared helper T-cell-B-cell coordination is augmented by CD-specific cytotoxic-plasma coupling",
                 x=0.52, y=0.988, fontsize=9.1, fontweight="bold")
    fig.text(0.52, 0.956, "Figures 2-3 T-cell axis + Figures 4-5 B-cell axis | 182 receptor-evaluable participants; 56 CD complete cases",
             ha="center", va="center", fontsize=6.1, color=COL["muted"])
    save_figure(fig, "Figure_6", MAIN)


def build_interaction_audit():
    interactions = pd.read_csv(RA / "Table_RA1_helper_B_CD_UC_interactions.csv").copy()
    interactions["ci_low"] = interactions["CD_minus_UC_interaction_beta"] - 1.96 * interactions["robust_SE"]
    interactions["ci_high"] = interactions["CD_minus_UC_interaction_beta"] + 1.96 * interactions["robust_SE"]
    interactions["label"] = interactions["T_label"] + " → " + interactions["B_label"]
    interactions.to_csv(SRC / "Figure_S12_helper_axis_interactions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.48, 5.15))
    data = interactions.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    ax.errorbar(data["CD_minus_UC_interaction_beta"], y,
                xerr=[data["CD_minus_UC_interaction_beta"] - data["ci_low"],
                      data["ci_high"] - data["CD_minus_UC_interaction_beta"]],
                fmt="o", color=COL["purple"], ms=4.2, lw=0.95, capsize=2)
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(y, data["label"])
    ax.set_xlabel("CD–UC interaction estimate (95% CI)")
    ax.set_title("Helper–B-cell interaction audit across shared acquisition series",
                 loc="left", fontsize=9, fontweight="bold", pad=10)
    ax.text(0.99, 1.01, "0 of 12 tests passed FDR < 0.05", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color=COL["muted"])
    for yi, (_, row) in enumerate(data.iterrows()):
        ax.text(1.82, yi, f"FDR={row.FDR:.2f}", ha="right", va="center",
                fontsize=5.7, color=COL["muted"],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.90})
    ax.set_xlim(-1.55, 1.86)
    style_axis(ax, "x")
    fig.subplots_adjust(left=0.39, right=0.98, top=0.90, bottom=0.13)
    save_figure(fig, "Figure_S12", SUPP)


def build_figure_s13():
    helper_data = pd.read_csv(RA / "Table_RA1_helper_axis_analysis_dataset.csv")
    integrated = pd.read_csv(HI / "Table_HI_integrated_participant_features.csv")
    integrated["log_tcr_depth"] = np.log1p(integrated["tcr_total_cells"])
    integrated["log_bcr_depth"] = np.log1p(integrated["bcr_total_cells"])
    keep_helper = [
        "SampleID", "Diagnosis1", "acquisition_series", "Biologic", "Calprotectin",
        "cd4_mean_Tph_Tfh_help", "b_mean_Atypical_memory_CD11c_like",
        "b_mean_B_cell_antigen_presentation", "b_mean_IgA_mucosal_plasma_cell",
        "b_mean_IgG_inflammatory_plasma_cell", "b_mean_Plasmablast_plasma_differentiation",
        "b_plasma_fraction", "b_iga_plasma_fraction",
    ]
    analysis = helper_data[keep_helper].merge(
        integrated[["SampleID", "Age", "Sex", "log_tcr_depth", "log_bcr_depth",
                    "tcr_cytotoxic_expanded"]], on="SampleID", how="inner"
    )
    analysis["log_calprotectin"] = np.log1p(pd.to_numeric(analysis["Calprotectin"], errors="coerce"))
    cd = analysis[analysis["Diagnosis1"] == "CD"].copy()
    outcomes = ["b_mean_IgA_mucosal_plasma_cell", "b_mean_Plasmablast_plasma_differentiation"]
    models = {
        "Covariates": [], "Helper": ["cd4_mean_Tph_Tfh_help"],
        "Cytotoxic": ["tcr_cytotoxic_expanded"],
        "Joint": ["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"],
    }

    def fitted_r2(data, outcome, predictors):
        columns = [outcome, "Age", "Sex", "log_tcr_depth", "log_bcr_depth"] + predictors
        work = data[columns].dropna().reset_index(drop=True)
        design, _ = _numeric_design(work, work, predictors)
        y = work[outcome].to_numpy(float)
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        denominator = np.sum((y - y.mean()) ** 2)
        return 1 - np.sum((y - fitted) ** 2) / denominator, len(work)

    r2_rows = []
    rng = np.random.default_rng(20260829)
    for outcome in outcomes:
        for model, predictors in models.items():
            complete = cd[[outcome, "Age", "Sex", "log_tcr_depth", "log_bcr_depth"] + predictors].dropna().reset_index(drop=True)
            estimate, n = fitted_r2(complete, outcome, predictors)
            draws = []
            for _ in range(1000):
                sample = complete.iloc[rng.integers(0, len(complete), len(complete))].reset_index(drop=True)
                try:
                    value, _ = fitted_r2(sample, outcome, predictors)
                    if np.isfinite(value):
                        draws.append(value)
                except np.linalg.LinAlgError:
                    continue
            r2_rows.append({"outcome": outcome, "model": model, "r2": estimate,
                            "ci_low": np.quantile(draws, 0.025), "ci_high": np.quantile(draws, 0.975), "n": n})
    r2_data = pd.DataFrame(r2_rows)
    r2_data.to_csv(SRC / "Figure_S13_panel_A_model_variance.csv", index=False)

    def rank_joint_beta(data, outcome):
        columns = [outcome, "cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded",
                   "Age", "Sex", "log_tcr_depth", "log_bcr_depth"]
        work = data[columns].dropna().reset_index(drop=True)
        standard_rank = lambda s: (rankdata(s.to_numpy(float)) - (len(s) + 1) / 2) / np.std(rankdata(s.to_numpy(float)), ddof=1)
        y = standard_rank(work[outcome])
        design = [np.ones(len(work)), standard_rank(work["cd4_mean_Tph_Tfh_help"]),
                  standard_rank(work["tcr_cytotoxic_expanded"])]
        for column in ["Age", "log_tcr_depth", "log_bcr_depth"]:
            values = work[column].to_numpy(float); sd = values.std(ddof=1)
            design.append((values - values.mean()) / sd if sd > 0 else np.zeros(len(values)))
        design.append((work["Sex"].astype(str).str.upper() == "M").astype(float).to_numpy())
        return np.linalg.lstsq(np.column_stack(design), y, rcond=None)[0][1:3], len(work)

    loo_rows = []
    for outcome in outcomes:
        full_beta, n = rank_joint_beta(cd, outcome)
        for predictor, value in zip(["Helper", "Cytotoxic"], full_beta):
            loo_rows.append({"outcome": outcome, "omitted_series": "None", "predictor": predictor,
                             "beta": value, "n": n})
        for series in sorted(cd["acquisition_series"].dropna().astype(str).unique()):
            retained = cd[cd["acquisition_series"].astype(str) != series]
            beta, retained_n = rank_joint_beta(retained, outcome)
            for predictor, value in zip(["Helper", "Cytotoxic"], beta):
                loo_rows.append({"outcome": outcome, "omitted_series": series, "predictor": predictor,
                                 "beta": value, "n": retained_n})
    loo_data = pd.DataFrame(loo_rows)
    loo_data.to_csv(SRC / "Figure_S13_panel_B_leave_one_series_joint_models.csv", index=False)

    covariates = ["Age", "Sex", "log_tcr_depth", "log_bcr_depth"]
    orthogonal_rows = []
    for outcome in ["b_iga_plasma_fraction", "b_plasma_fraction"]:
        for predictor in ["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"]:
            estimate, low, high, n = bootstrap_partial_rank(
                cd, predictor, outcome, covariates, 20260830 + len(orthogonal_rows), n_boot=1500
            )
            orthogonal_rows.append({"outcome": outcome, "predictor": predictor, "rho": estimate,
                                    "ci_low": low, "ci_high": high, "n": n})
    orthogonal = pd.DataFrame(orthogonal_rows)
    orthogonal.to_csv(SRC / "Figure_S13_panel_C_cell_composition_validation.csv", index=False)

    clinical_rows = []
    for diagnosis in ["CD", "UC"]:
        subset = analysis[analysis["Diagnosis1"] == diagnosis].copy()
        complete_score = subset[["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"]].dropna().index
        subset.loc[complete_score, "coordination_score"] = (
            pd.Series(rankdata(subset.loc[complete_score, "cd4_mean_Tph_Tfh_help"]), index=complete_score).rank(pct=True) +
            pd.Series(rankdata(subset.loc[complete_score, "tcr_cytotoxic_expanded"]), index=complete_score).rank(pct=True)
        ) / 2
        estimate, low, high, n = bootstrap_partial_rank(
            subset, "coordination_score", "log_calprotectin",
            covariates + ["Biologic"], 20260840 + len(clinical_rows), n_boot=1500
        )
        clinical_rows.append({"diagnosis": diagnosis, "rho": estimate, "ci_low": low, "ci_high": high, "n": n})
    clinical = pd.DataFrame(clinical_rows)
    clinical.to_csv(SRC / "Figure_S13_panel_D_clinical_coordination.csv", index=False)

    negative_outcomes = [
        "b_mean_Atypical_memory_CD11c_like", "b_mean_B_cell_antigen_presentation",
        "b_mean_IgA_mucosal_plasma_cell", "b_mean_IgG_inflammatory_plasma_cell",
        "b_mean_Plasmablast_plasma_differentiation",
    ]
    negative_rows = []
    for predictor in ["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"]:
        for outcome in negative_outcomes:
            rho, n = partial_rank_correlation(cd, predictor, outcome, covariates)
            degrees = max(n - len(covariates) - 2, 1)
            statistic = rho * np.sqrt(degrees / max(1 - rho ** 2, 1e-9))
            p_value = 2 * student_t.sf(abs(statistic), degrees)
            negative_rows.append({"predictor": predictor, "outcome": outcome,
                                  "rho": rho, "p_value": p_value, "n": n})
    negative = pd.DataFrame(negative_rows)
    order = np.argsort(negative["p_value"].to_numpy())
    ranked = negative["p_value"].to_numpy()[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    fdr = np.empty(len(negative)); fdr[order] = np.minimum(adjusted, 1.0)
    negative["FDR"] = fdr
    negative.to_csv(SRC / "Figure_S13_panel_E_specificity_matrix.csv", index=False)

    program_differences = pd.read_csv(SRC / "Figure_6_program_difference_tests.csv")
    program_differences.to_csv(SRC / "Figure_S13_panel_F_program_contrasts.csv", index=False)

    fig = plt.figure(figsize=(7.48, 9.55), facecolor="white")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.05, 1.05], hspace=0.72, wspace=0.68,
                  left=0.14, right=0.98, top=0.95, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0])
    model_colors = {"Covariates": "#B5B5B5", "Helper": COL["purple"],
                    "Cytotoxic": COL["CD"], "Joint": COL["ink"]}
    positions = np.arange(len(r2_data))[::-1]
    for y, (_, row) in zip(positions, r2_data.iterrows()):
        ax.errorbar(row.r2, y, xerr=[[row.r2 - row.ci_low], [row.ci_high - row.r2]],
                    fmt="o", color=model_colors[row.model], ms=4, capsize=2, lw=0.9)
    labels = [f"{'IgA' if 'IgA' in row.outcome else 'Plasma'} | {row.model}" for _, row in r2_data.iterrows()]
    ax.set_yticks(positions, labels); ax.set_xlabel("In-sample R² (95% bootstrap CI)")
    panel_title(ax, "Model variance and incremental information"); panel_label(ax, "A", x=-0.28)
    style_axis(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    plot_loo = loo_data[loo_data["omitted_series"] != "None"].copy()
    labels_b, y_b = [], []
    for idx, (outcome, predictor) in enumerate([(o, p) for o in outcomes for p in ["Helper", "Cytotoxic"]]):
        values = plot_loo[(plot_loo.outcome == outcome) & (plot_loo.predictor == predictor)].beta
        full = loo_data[(loo_data.outcome == outcome) & (loo_data.predictor == predictor) &
                        (loo_data.omitted_series == "None")].beta.iloc[0]
        y = 3 - idx; y_b.append(y)
        ax.hlines(y, values.min(), values.max(), color="#A7ABB0", lw=3.0)
        ax.scatter(full, y, color=COL["purple"] if predictor == "Helper" else COL["CD"], s=23, zorder=3)
        labels_b.append(f"{'IgA' if 'IgA' in outcome else 'Plasma'} | {predictor}")
    ax.axvline(0, color=COL["muted"], ls="--", lw=0.6); ax.set_yticks(y_b, labels_b)
    ax.set_xlabel("Joint-model coefficient; leave-one-series range")
    panel_title(ax, "Acquisition-series influence"); panel_label(ax, "B", x=-0.28); style_axis(ax, "x")

    ax = fig.add_subplot(gs[1, 0])
    labels_c = []
    for y, (_, row) in zip(range(len(orthogonal) - 1, -1, -1), orthogonal.iterrows()):
        color = COL["purple"] if row.predictor == "cd4_mean_Tph_Tfh_help" else COL["CD"]
        ax.errorbar(row.rho, y, xerr=[[row.rho - row.ci_low], [row.ci_high - row.rho]],
                    fmt="o", color=color, ms=4, capsize=2, lw=0.9)
        labels_c.append(f"{'IgA plasma fraction' if 'iga' in row.outcome else 'Plasma fraction'} | "
                        f"{'Helper' if row.predictor == 'cd4_mean_Tph_Tfh_help' else 'Cytotoxic'}")
    ax.axvline(0, color=COL["muted"], ls="--", lw=0.6)
    ax.set_yticks(range(len(orthogonal) - 1, -1, -1), labels_c); ax.set_xlabel("Partial Spearman ρ (95% CI)")
    panel_title(ax, "Orthogonal cell-composition validation"); panel_label(ax, "C", x=-0.28); style_axis(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    for y, (_, row) in zip([1, 0], clinical.iterrows()):
        ax.errorbar(row.rho, y, xerr=[[row.rho - row.ci_low], [row.ci_high - row.rho]],
                    fmt="o", color=COL[row.diagnosis], ms=4, capsize=2, lw=0.9)
    ax.axvline(0, color=COL["muted"], ls="--", lw=0.6); ax.set_yticks([1, 0], clinical.diagnosis)
    ax.set_xlabel("Partial Spearman ρ with log calprotectin (95% CI)")
    panel_title(ax, "Contemporaneous clinical association"); panel_label(ax, "D", x=-0.28); style_axis(ax, "x")

    ax = fig.add_subplot(gs[2, 0])
    outcome_labels = ["Atypical memory", "Antigen presentation", "IgA mucosal", "IgG inflammatory", "Plasma differentiation"]
    predictor_labels = ["Helper axis", "Expanded cytotoxic"]
    mat = negative.pivot(index="predictor", columns="outcome", values="rho").reindex(
        index=["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"], columns=negative_outcomes)
    fdr_mat = negative.pivot(index="predictor", columns="outcome", values="FDR").reindex(
        index=["cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"], columns=negative_outcomes)
    mat.index = predictor_labels; mat.columns = outcome_labels; fdr_mat.index = predictor_labels; fdr_mat.columns = outcome_labels
    add_heatmap(ax, mat, fdr_mat, vlim=0.8, colorbar=False, annotate_size=5.5)
    ax.set_xticklabels(outcome_labels, rotation=35, ha="right")
    panel_title(ax, "Cross-program specificity in CD"); panel_label(ax, "E", x=-0.28)

    ax = fig.add_subplot(gs[2, 1])
    diff_labels = ["Plasma-IgG", "Plasma-BAFF/APRIL", "IgA-IgG", "IgA-BAFF/APRIL"]
    y = np.arange(len(program_differences))[::-1]
    for yi, (_, row) in zip(y, program_differences.iterrows()):
        ax.errorbar(row.rho_difference, yi,
                    xerr=[[row.rho_difference - row.ci_low], [row.ci_high - row.rho_difference]],
                    fmt="o", color=COL["CD"], ms=4, capsize=2, lw=0.9)
        ax.text(0.53, yi, f"FDR={row.FDR_across_four:.2g}",
                ha="right", va="center", fontsize=5.7, color=COL["muted"],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.88})
    ax.axvline(0, color=COL["muted"], ls="--", lw=0.6); ax.set_yticks(y, diff_labels)
    ax.set_xlim(-0.22, 0.56)
    ax.set_xlabel("Participant-paired difference in ρ (95% CI)")
    panel_title(ax, "Formal between-program contrasts"); panel_label(ax, "F", x=-0.28); style_axis(ax, "x")

    fig.suptitle("Robustness, sensitivity, and specificity analyses",
                 fontsize=9.2, fontweight="bold", y=0.988)
    save_figure(fig, "Figure_S13", SUPP)


def write_legends():
    main_legend = (
        "Figure 6. Shared helper T-cell-B-cell coordination is augmented by Crohn's disease-specific cytotoxic-plasma coupling. "
        "(A) Participant-level partial Spearman correlations between global TCR and BCR repertoire or transcriptional "
        "features, adjusted for diagnosis, age, sex, and receptor depth. Asterisks denote Benjamini-Hochberg FDR "
        "<0.05 (*) or <0.01 (**). (B) Diagnosis-stratified partial Spearman correlations between prespecified CD4 "
        "helper and B-cell features after residualization for age, sex, log CD4-cell depth, and log B-cell depth. "
        "P values were obtained from 3,000 acquisition-series-blocked permutations and corrected within diagnosis. "
        "Clone-associated helper scores are expanded-minus-singleton differences calculated within participant and "
        "CD4 state; blank cells were not estimable. (C) Representative CD Tph versus plasma-differentiation association. "
        "The fitted line and band are a descriptive linear fit and participant-bootstrap 95% confidence interval; the "
        "annotation reports the covariate-adjusted partial correlation and formal shared-series CD-versus-UC interaction. "
        "(D) Joint Crohn's disease models of the IgA mucosal plasma-cell and plasmablast/plasma-cell-differentiation "
        "programs. The combined Tph/Tfh helper score and expanded-TCR cytotoxicity score were entered simultaneously "
        "with age, sex, log-transformed TCR depth, and log-transformed BCR depth. Points show standardized rank "
        "coefficients and bars show 95% confidence intervals from 3,000 participant bootstrap samples. "
        "(E) Diagnosis-stratified partial Spearman correlations between the expanded-TCR cytotoxicity score and IgA "
        "mucosal plasma-cell or plasmablast/plasma-cell-differentiation programs. Points and bars show estimates and "
        "participant-bootstrap 95% confidence intervals; primary models adjusted for age, sex, log TCR depth, and log "
        "BCR depth. (F) Robustness of the Crohn's disease cytotoxic-plasma estimates. The left panel adds acquisition "
        "series, objective inflammation status, biologic exposure, or participant-level CD8 Tem GZMB+ and IgM plasma-cell "
        "composition to the primary covariate set; bars show 95% participant-bootstrap confidence intervals. The right "
        "panel compares observed correlations with 95% empirical null intervals from 10,000 within-acquisition-series "
        "matched-participant re-pairings. Broad helper-T-cell-B-cell coordination was shared across diagnoses, whereas "
        "formal diagnosis interactions supported a CD-specific extension linking expanded cytotoxic T-cell activity to "
        "plasma-cell programs. (G) Leave-one-acquisition-series-out validation of helper-only, cytotoxic-only, and joint "
        "models for the two B-cell programs. Each acquisition series was predicted from models trained on all remaining "
        "series, and displayed values are Spearman correlations between pooled held-out predictions and observed scores. "
        "Annotations report the joint-minus-helper improvement and its one-sided empirical P value from 2,000 permutations "
        "of the cytotoxic predictor within acquisition series. These cross-sectional associations do not establish causal "
        "direction or mediation."
    )
    supp_legend = (
        "Figure S12. Helper–B-cell interaction audit across shared acquisition series. CD-versus-UC interaction "
        "estimates were calculated in shared acquisition series S1–S5 using heteroskedasticity-robust standard errors. "
        "Points show interaction coefficients and bars show 95% confidence intervals. P values were corrected across "
        "the 12 prespecified helper–B-cell feature pairs by the Benjamini-Hochberg method; no interaction passed FDR "
        "<0.05. Exact paired-alpha/beta clone-associated contrasts compare expanded and singleton cells within the same "
        "participant and annotated CD4 state."
    )
    s13_legend = (
        "Figure S13. Robustness, generalizability, and specificity of the helper/cytotoxic T-cell-B-cell coordination axes. "
        "(A) In-sample variance explained by covariates-only, helper-only, cytotoxic-only, and joint models for the IgA "
        "mucosal plasma-cell and plasmablast/plasma-cell-differentiation programs among participants with Crohn's disease; "
        "bars show 95% participant-bootstrap confidence intervals. (B) Full-sample joint-model coefficients and their "
        "ranges after omitting each acquisition series in turn. (C) Participant-level partial Spearman correlations of "
        "the helper and expanded-cytotoxic axes with independently measured IgA plasma-cell and total plasma-cell fractions, "
        "adjusted for age, sex, and TCR/BCR depth; bars show 95% participant-bootstrap confidence intervals. (D) Association "
        "of a prespecified mean-rank helper/cytotoxic coordination score with contemporaneous log-transformed fecal "
        "calprotectin in CD and UC, adjusted for age, sex, receptor depth, and biologic exposure. (E) Cross-program partial "
        "correlation matrix in CD. Asterisks denote Benjamini-Hochberg FDR <0.05 (*) or <0.01 (**) across the ten displayed "
        "tests. (F) Participant-paired differences between cytotoxic correlations for plasma/IgA programs and IgG-inflammatory "
        "or BAFF/APRIL programs; bars show 95% participant-bootstrap confidence intervals and annotations report FDR across "
        "the four contrasts. The available external gut cohort profiled CD8 T-cell programs but did not provide matched "
        "participant-level helper T-cell and B-cell measurements and therefore could not independently validate the Figure 6 "
        "cross-compartment coordination endpoint."
    )
    (LEG / "Figure_6_legend.txt").write_text(main_legend + "\n", encoding="utf-8")
    (LEG / "Figure_S12_legend.txt").write_text(supp_legend + "\n", encoding="utf-8")
    (LEG / "Figure_S13_legend.txt").write_text(s13_legend + "\n", encoding="utf-8")

    # Replace the existing Figure 6 paragraph and append S12 without disturbing other legends.
    all_path = LEG / "All_figure_legends.txt"
    if all_path.exists():
        paragraphs = [x.strip() for x in all_path.read_text(encoding="utf-8").split("\n\n") if x.strip()]
        updated = []
        found_main = False
        found_supp = False
        found_s13 = False
        for paragraph in paragraphs:
            if paragraph.startswith("Figure 6."):
                updated.append(main_legend)
                found_main = True
            elif paragraph.startswith("Figure S12."):
                updated.append(supp_legend)
                found_supp = True
            elif paragraph.startswith("Figure S13."):
                updated.append(s13_legend)
                found_s13 = True
            else:
                updated.append(paragraph)
        if not found_main:
            updated.append(main_legend)
        if not found_supp:
            updated.append(supp_legend)
        if not found_s13:
            updated.append(s13_legend)
        all_path.write_text("\n\n".join(updated) + "\n", encoding="utf-8")


def rebuild_combined_pdfs():
    writer = PdfWriter()
    for i in range(1, 8):
        path = MAIN / f"Figure_{i}.pdf"
        if path.exists():
            for page in PdfReader(str(path)).pages:
                writer.add_page(page)
    with (OUT / "Main_Figures_1-7_Cell_Press.pdf").open("wb") as handle:
        writer.write(handle)

    writer = PdfWriter()
    for i in range(1, 14):
        path = SUPP / f"Figure_S{i}.pdf"
        if path.exists():
            for page in PdfReader(str(path)).pages:
                writer.add_page(page)
    with (OUT / "Supplementary_Figures_S1-S12_Cell_Press.pdf").open("wb") as handle:
        writer.write(handle)


def main():
    backup_current_outputs()
    build_main_figure()
    build_interaction_audit()
    build_figure_s13()
    write_legends()
    rebuild_combined_pdfs()
    print("Rebuilt Figure 6 and Figures S12-S13 with source data and legends.")


if __name__ == "__main__":
    main()

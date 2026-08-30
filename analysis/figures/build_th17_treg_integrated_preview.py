#!/usr/bin/env python3
"""Build non-canonical main/supplement preview figures integrating Th17/Treg results."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives" / "Th17 Treg Integration"
PDF_OUT = ROOT / "output" / "pdf"
SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
TB = ROOT / "High Impact Additional Analyses" / "Th17 Treg B Helper Analyses"
PREVIEW.mkdir(parents=True, exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "Control": "#6F6F6F",
    "CD": "#0072B2",
    "UC": "#D55E00",
    "Tph/Tfh help": "#377EB8",
    "Pathogenic Th17": "#D95F02",
    "Conventional Th17": "#E69F00",
    "Suppressive Treg": "#1B9E77",
    "Reprogrammed Treg": "#66A61E",
    "ink": "#222222",
    "muted": "#777777",
    "grid": "#D8D8D8",
}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.0,
    "axes.titlesize": 7.7,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 5.8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
})


def style_axis(ax, grid=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["ink"])
    ax.spines["bottom"].set_color(COL["ink"])
    ax.tick_params(color=COL["ink"], labelcolor=COL["ink"], pad=2)
    if grid:
        ax.grid(axis=grid, color=COL["grid"], lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, label, title, x=-0.13):
    ax.text(x, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            ha="left", va="top")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)


def figure6_panel(ax, label, title, full_width=False):
    """Place Figure 6 letters and titles in a heading row above the axes."""
    title_x = 0.055 if full_width else 0.12
    heading_y = 1.08
    ax.text(0.0, heading_y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="top", clip_on=False)
    ax.text(title_x, heading_y, title, transform=ax.transAxes, fontsize=7.7,
            fontweight="bold", ha="left", va="top", clip_on=False)


def stars(q):
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def heatmap(ax, matrix, qmatrix=None, vlim=0.65, annotate=True, cbar=True, annot_size=4.5):
    arr = matrix.to_numpy(float)
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not np.isfinite(arr[i, j]):
                    continue
                q = np.nan if qmatrix is None else float(qmatrix.iloc[i, j])
                color = "white" if abs(arr[i, j]) > 0.40 else COL["ink"]
                ax.text(j, i, f"{arr[i, j]:.2f}{stars(q)}", ha="center", va="center",
                        fontsize=annot_size, color=color)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.014)
        cb.set_label("Partial Spearman rho", fontsize=5.5)
        cb.ax.tick_params(labelsize=5.0, length=2)
    return im


def save(fig, png_name, pdf_name):
    fig.savefig(PREVIEW / png_name, dpi=350, facecolor="white")
    fig.savefig(PDF_OUT / pdf_name, facecolor="white")
    plt.close(fig)


def clone_composition(ax, clone_axis, title=True):
    order = ["CD", "UC", "Control"]
    classes = ["Th17-only", "Treg-only", "Mixed Th17/Treg"]
    colors = ["#D95F02", "#1B9E77", "#7570B3"]
    comp = clone_axis.groupby(["Diagnosis1", "axis_clone_class"]).size().unstack(fill_value=0)
    comp = comp.reindex(index=order, columns=classes, fill_value=0)
    pct = comp.div(comp.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(order))
    for cls, color in zip(classes, colors):
        ax.bar(order, pct[cls], bottom=bottom, color=color, width=0.72, label=cls)
        bottom += pct[cls].to_numpy()
    for i, diagnosis in enumerate(order):
        ax.text(i, 101.5, f"mixed={int(comp.loc[diagnosis, 'Mixed Th17/Treg'])}",
                ha="center", va="bottom", fontsize=5.5, color="#5E4FA2")
    ax.set_ylim(0, 106)
    ax.set_ylabel("Exact paired axis clones (%)")
    ax.legend(frameon=False, fontsize=5.3, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(axis="x", visible=False)
    style_axis(ax, "y")
    if title:
        ax.set_title("Clone-state composition", loc="left", fontweight="bold", pad=5)


def sharing_null(ax, sharing, title=True, show_key=True):
    order = ["IBD", "CD", "UC", "Control"]
    d = sharing.set_index("group").reindex(order).reset_index()
    y = np.arange(len(d))[::-1]
    ax.errorbar(d["null_mean"], y,
                xerr=[d["null_mean"] - d["null_q025"], d["null_q975"] - d["null_mean"]],
                fmt="o", color="#777777", capsize=2.5, ms=3.5, label="Permutation null (95%)")
    ax.scatter(d["observed_mixed_clones"], y, marker="D", s=28, color="#7570B3",
               zorder=3, label="Observed")
    xmax = max(float(d["null_q975"].max()), float(d["observed_mixed_clones"].max())) + 5
    ax.set_xlim(-0.5, xmax)
    ax.set_yticks(y, order)
    ax.set_xlabel("Mixed exact clonotypes")
    for yi, (_, row) in zip(y, d.iterrows()):
        ax.text(xmax - 0.2, yi, f"P={row.two_sided_p:.3g}", ha="right", va="center", fontsize=5.4)
    if show_key:
        ax.text(0.0, 1.01, "Diamond: observed; circle/line: permutation null (95%)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=4.7, color=COL["muted"])
    style_axis(ax, "x")
    if title:
        ax.set_title("Within-participant state-label null", loc="left", fontweight="bold",
                     pad=15 if show_key else 5)


def build_revised_figure3(clone_axis, sharing):
    source_png = ROOT / "Cell Press Redrawn Figure Set" / "Main Figures" / "Figure_3.png"
    image = Image.open(source_png).convert("RGB")
    crop_y = int(image.height * 0.752)
    top = np.asarray(image.crop((0, 0, image.width, crop_y)))

    fig = plt.figure(figsize=(7.05, 8.35))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.95, 2.05], hspace=0.16,
                          left=0.055, right=0.985, top=0.995, bottom=0.080)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(top)
    ax_img.axis("off")
    bottom = gs[1, 0].subgridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.60)
    ax_l = fig.add_subplot(bottom[0, 0])
    clone_composition(ax_l, clone_axis)
    panel(ax_l, "F", "Th17/Treg clonotypes are preferentially state-restricted", x=-0.10)
    ax_r = fig.add_subplot(bottom[0, 1])
    sharing_null(ax_r, sharing)
    fig.text(0.52, 0.015,
             "Exact paired alpha-beta V+J+CDR3 amino-acid identity; 10,000 within-participant permutations",
             ha="center", va="bottom", fontsize=5.0, color=COL["muted"])
    save(fig, "Figure_3_Th17_Treg_integration_preview.png",
         "Figure_3_Th17_Treg_integration_preview.pdf")


def design_residuals(data, value, numeric, categorical):
    d = data.copy()
    y = stats.rankdata(pd.to_numeric(d[value], errors="coerce").to_numpy(float))
    parts = [pd.Series(1.0, index=d.index, name="intercept")]
    for c in numeric:
        parts.append(pd.to_numeric(d[c], errors="coerce").rename(c))
    parts.append(pd.get_dummies(d[categorical].fillna("Unknown").astype(str),
                                drop_first=True, dtype=float))
    x = pd.concat(parts, axis=1).to_numpy(float)
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def build_revised_figure6(corr, restraint, analysis, global_corr, cytotoxic, validation):
    # Preserve the Cell Press full-width canvas while adding vertical clearance
    # for panel B's rotated category labels and the D/E heading row.
    fig = plt.figure(figsize=(7.05, 9.45))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.95, 1.38, 1.16, 0.78],
                          width_ratios=[1.22, 0.78], hspace=0.82, wspace=0.55,
                          left=0.17, right=0.975, top=0.955, bottom=0.065)
    fig.suptitle(
        "Clonotype-restricted helper states coordinate with B-cell remodeling and CD-specific cytotoxic coupling",
        fontsize=9.2, fontweight="bold", y=0.988,
    )

    # A: global repertoire context.
    ax = fig.add_subplot(gs[0, :])
    t_order = ["tcr_clonality", "tcr_gini", "tcr_expanded_cell_fraction",
               "tcr_multistate_clone_fraction", "tcr_cytotoxic_expanded"]
    b_order = ["bcr_gini", "bcr_expanded_cell_fraction", "bcr_multistate_clone_fraction",
               "bcr_switched_fraction", "bcr_SHM_rate", "bcr_IgA_mucosal_module",
               "bcr_plasma_differentiation_module", "bcr_BAFF_APRIL_module"]
    t_labels = ["TCR clonality", "TCR Gini", "Expanded TCR-cell fraction",
                "Multistate TCR-clone fraction", "Expanded-TCR cytotoxic score"]
    b_labels = ["BCR\nGini", "Expanded BCR-\ncell fraction", "Multistate BCR-\nclone fraction",
                "Class-switched\nBCR", "BCR\nSHM", "IgA\nmucosal", "Plasma\ndifferentiation", "BAFF/APRIL"]
    m = global_corr.pivot(index="tcr_metric", columns="bcr_metric", values="partial_rho").reindex(
        index=t_order, columns=b_order)
    q = global_corr.pivot(index="tcr_metric", columns="bcr_metric", values="p_adj").reindex(
        index=t_order, columns=b_order)
    m.index, m.columns = t_labels, b_labels
    q.index, q.columns = t_labels, b_labels
    heatmap(ax, m, q, vlim=0.65, annot_size=4.4)
    ax.tick_params(axis="x", rotation=0, labelsize=5.4)
    ax.tick_params(axis="y", labelsize=5.7)
    figure6_panel(ax, "A", "Global repertoire and program coordination", full_width=True)

    # B: focused helper-state matrix.
    ax = fig.add_subplot(gs[1, 0])
    ibd = corr[corr.group.eq("IBD")]
    t_order = ["Pathogenic Th17", "Conventional Th17", "Suppressive Treg", "Tph/Tfh help"]
    b_order = ["Plasma differentiation", "IgA mucosal plasma", "IgG inflammatory plasma",
               "Atypical memory", "Antigen presentation"]
    m = ibd.pivot(index="T_label", columns="B_label", values="partial_rho").reindex(
        index=t_order, columns=b_order)
    q = ibd.pivot(index="T_label", columns="B_label", values="FDR_within_group").reindex(
        index=t_order, columns=b_order)
    # Panel A already supplies the shared partial-correlation color scale;
    # omitting the duplicate bar here protects the B/C inter-panel gutter.
    heatmap(ax, m, q, vlim=0.65, annot_size=4.5, cbar=False)
    ax.tick_params(axis="x", rotation=45, labelsize=5.2, pad=1)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=5.6)
    figure6_panel(ax, "B", "Helper programs track B-cell remodeling in IBD")

    # C: independent helper-state effects.
    ax = fig.add_subplot(gs[1, 1])
    outcomes = ["Plasma differentiation", "IgA mucosal plasma", "IgG inflammatory plasma",
                "Atypical memory", "Antigen presentation"]
    ypos = {label: i for i, label in enumerate(reversed(outcomes))}
    offsets = {"Tph/Tfh help": -0.18, "Pathogenic Th17": 0.0, "Suppressive Treg": 0.18}
    for tl, offset in offsets.items():
        d = restraint[restraint.T_label.eq(tl)]
        y = np.array([ypos[x] for x in d.B_label]) + offset
        ax.errorbar(d.standardized_beta, y,
                    xerr=[d.standardized_beta - d.ci_low, d.ci_high - d.standardized_beta],
                    fmt="o", ms=3.5, color=COL[tl], capsize=2, label=tl)
    ax.axvline(0, color=COL["ink"], lw=0.7)
    ax.set_yticks(range(len(outcomes)), list(reversed(outcomes)))
    ax.set_xlabel("Adjusted standardized beta")
    ax.legend(frameon=False, fontsize=4.4, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=3, columnspacing=0.55, handletextpad=0.25)
    style_axis(ax, "x")
    figure6_panel(ax, "C", "Independent helper contributions")

    # D: representative adjusted relationship.
    ax = fig.add_subplot(gs[2, 0])
    xcol = "mean_Tph_Tfh_help_all_CD4"
    ycol = "b_mean_B_cell_antigen_presentation"
    numeric = ["Age", "sex_male", "log_cd4_cells", "log_b_cells"]
    categorical = ["Diagnosis1", "Inflammation1", "Biologic", "acquisition_series"]
    cols = [xcol, ycol] + numeric + categorical
    d = analysis[analysis.Diagnosis1.isin(["CD", "UC"])][cols].replace([np.inf, -np.inf], np.nan).copy()
    for c in categorical:
        d[c] = d[c].fillna("Unknown")
    d = d.dropna(subset=[xcol, ycol] + numeric)
    d["x_resid"] = design_residuals(d, xcol, numeric, categorical)
    d["y_resid"] = design_residuals(d, ycol, numeric, categorical)
    for diagnosis in ["CD", "UC"]:
        z = d[d.Diagnosis1.eq(diagnosis)]
        ax.scatter(z.x_resid, z.y_resid, s=12, alpha=0.68, color=COL[diagnosis],
                   edgecolor="white", lw=0.25, label=diagnosis)
    coef = np.polyfit(d.x_resid, d.y_resid, 1)
    xx = np.linspace(d.x_resid.min(), d.x_resid.max(), 100)
    ax.plot(xx, np.polyval(coef, xx), color=COL["ink"], lw=1.0)
    effect = restraint[(restraint.T_label == "Tph/Tfh help") &
                       (restraint.B_label == "Antigen presentation")].iloc[0]
    ax.text(0.02, 0.97, f"joint beta={effect.standardized_beta:.2f}; FDR={effect.FDR:.2g}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8)
    ax.set_xlabel("Tph/Tfh-help residual rank")
    ax.set_ylabel("B-cell antigen-presentation residual rank")
    ax.legend(frameon=False, loc="lower right", fontsize=5.5)
    style_axis(ax, "both")
    figure6_panel(ax, "D", "Representative adjusted association")

    # E: retain the CD-specific cytotoxic-plasma extension.
    ax = fig.add_subplot(gs[2, 1])
    outcome_order = list(cytotoxic.outcome.drop_duplicates())
    positions = []
    labels = []
    y0 = 7
    for oi, outcome in enumerate(outcome_order):
        sub = cytotoxic[cytotoxic.outcome.eq(outcome)].set_index("diagnosis").reindex(["Control", "CD", "UC"])
        for j, (diagnosis, row) in enumerate(sub.iterrows()):
            y = y0 - oi * 4 - j
            positions.append(y)
            labels.append(f"{('IgA' if oi == 0 else 'Plasma')} | {diagnosis} (n={int(row.n)})")
            ax.plot([row.ci_low, row.ci_high], [y, y], color=COL[diagnosis], lw=1.0)
            ax.plot(row.partial_rho, y, "o", color=COL[diagnosis], ms=3.5)
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Partial Spearman rho (95% CI)")
    style_axis(ax, "x")
    figure6_panel(ax, "E", "Cytotoxic-plasma coupling")

    # F: blocked validation summary from the established Figure 6 model.
    ax = fig.add_subplot(gs[3, :])
    outcomes = list(validation.outcome.drop_duplicates())
    model_order = ["Helper only", "Cytotoxic only", "Joint"]
    model_color = {"Helper only": "#7B3294", "Cytotoxic only": "#0072B2", "Joint": "#222222"}
    offsets = {"Helper only": 0.18, "Cytotoxic only": 0.0, "Joint": -0.18}
    ybase = {outcomes[0]: 1, outcomes[1]: 0}
    for model in model_order:
        d = validation[validation.model.eq(model)]
        y = np.array([ybase[o] for o in d.outcome]) + offsets[model]
        ax.scatter(d.held_out_spearman_rho, y, s=24, marker="D" if model == "Joint" else "o",
                   color=model_color[model], label=model)
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks([1, 0], ["IgA mucosal plasma", "Plasma differentiation"])
    ax.set_xlabel("Leave-one-acquisition-series-out Spearman correlation")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=5.2)
    style_axis(ax, "x")
    figure6_panel(ax, "F", "Blocked out-of-series validation", full_width=True)

    save(fig, "Figure_6_Helper_state_integration_preview.png",
         "Figure_6_Helper_state_integration_preview.pdf")


def build_supplement_th17(clone_axis, sharing, size_models, delta_tests, corr, clinical, restraint):
    fig = plt.figure(figsize=(7.05, 10.0))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.08, 1.15, 0.95],
                          hspace=0.70, wspace=0.62, left=0.17, right=0.975,
                          top=0.965, bottom=0.055)
    fig.suptitle("Extended clone-aware Th17/Treg and B-cell coordination analyses",
                 fontsize=9.2, fontweight="bold", y=0.992)

    ax = fig.add_subplot(gs[0, 0])
    clone_composition(ax, clone_axis)
    panel(ax, "A", "Th17/Treg clone-state composition", x=-0.23)
    ax = fig.add_subplot(gs[0, 1])
    sharing_null(ax, sharing, title=False, show_key=False)
    panel(ax, "B", "Exact-clone sharing permutation", x=-0.20)
    ax.set_xlabel("Mixed exact clonotypes\n(diamond observed; circle/line permutation null, 95%)")

    module_labels = {
        "Th17_conventional": "Conventional Th17", "Th17_pathogenic": "Pathogenic Th17",
        "Treg_suppressive": "Suppressive Treg", "Treg_reprogramming": "Reprogrammed Treg",
    }
    group_colors = {"IBD": "#222222", "CD": COL["CD"], "UC": COL["UC"], "Control": COL["Control"]}

    ax = fig.add_subplot(gs[1, 0])
    groups = ["IBD", "CD", "UC", "Control"]
    offsets = dict(zip(groups, [0.24, 0.08, -0.08, -0.24]))
    mods = list(module_labels)
    ypos = {m: i for i, m in enumerate(reversed(mods))}
    for group in groups:
        d = size_models[size_models.group.eq(group)]
        y = np.array([ypos[m] for m in d.module]) + offsets[group]
        ax.errorbar(d.beta_per_doubling, y,
                    xerr=[d.beta_per_doubling - d.ci_low, d.ci_high - d.beta_per_doubling],
                    fmt="o", ms=3.0, capsize=2, color=group_colors[group], label=group)
    ax.axvline(0, color=COL["ink"], lw=0.7)
    ax.set_yticks(range(len(mods)), [module_labels[m] for m in reversed(mods)])
    ax.set_xlabel("Program change per clone-size doubling")
    ax.legend(frameon=False, ncol=4, fontsize=4.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), columnspacing=0.7, handletextpad=0.25)
    style_axis(ax, "x")
    panel(ax, "C", "Clone-size dose-response tests", x=-0.23)

    ax = fig.add_subplot(gs[1, 1])
    for group in groups:
        d = delta_tests[delta_tests.group.eq(group)]
        y = np.array([ypos[m] for m in d.module]) + offsets[group]
        ax.errorbar(d.mean_delta, y,
                    xerr=[d.mean_delta - d.bootstrap_ci_low, d.bootstrap_ci_high - d.mean_delta],
                    fmt="o", ms=3.0, capsize=2, color=group_colors[group], label=group)
    ax.axvline(0, color=COL["ink"], lw=0.7)
    ax.set_yticks(range(len(mods)), [module_labels[m] for m in reversed(mods)])
    ax.set_xlabel("Expanded - singleton program score")
    style_axis(ax, "x")
    panel(ax, "D", "State-matched expansion tests", x=-0.20)

    # E: diagnosis-specific matrices.
    holder = fig.add_subplot(gs[2, :])
    holder.axis("off")
    holder.text(-0.08, 1.17, "E", transform=holder.transAxes, fontsize=11, fontweight="bold", va="top")
    holder.text(0.0, 1.16, "Diagnosis-specific helper-B-cell coordination",
                transform=holder.transAxes, fontsize=7.7, fontweight="bold", ha="left", va="top")
    sub = gs[2, :].subgridspec(1, 3, wspace=0.58)
    t_order = ["Pathogenic Th17", "Conventional Th17", "Suppressive Treg", "Reprogrammed Treg", "Tph/Tfh help"]
    b_order = ["Plasma differentiation", "IgG inflammatory plasma", "Atypical memory", "Antigen presentation"]
    for j, group in enumerate(["CD", "UC", "Control"]):
        ax = fig.add_subplot(sub[0, j])
        d = corr[corr.group.eq(group)]
        m = d.pivot(index="T_label", columns="B_label", values="partial_rho").reindex(index=t_order, columns=b_order)
        q = d.pivot(index="T_label", columns="B_label", values="FDR_within_group").reindex(index=t_order, columns=b_order)
        heatmap(ax, m, q, vlim=0.70, cbar=(j == 2), annot_size=3.8)
        ax.set_title(group, fontweight="bold", pad=3)
        ax.tick_params(axis="x", rotation=55, labelsize=4.3)
        ax.tick_params(axis="y", labelsize=4.4)
        if j > 0:
            ax.set_yticklabels([])

    ax = fig.add_subplot(gs[3, 0])
    d = clinical.sort_values("partial_rho")
    y = np.arange(len(d))
    ax.errorbar(d.partial_rho, y,
                xerr=[d.partial_rho - d.ci_low, d.ci_high - d.partial_rho],
                fmt="o", color="#5E4FA2", capsize=2, ms=3.5)
    ax.axvline(0, color=COL["ink"], lw=0.7)
    ax.set_yticks(y, d.label)
    ax.set_xlabel("Partial rho with log1p calprotectin")
    style_axis(ax, "x")
    panel(ax, "F", "Clinical inflammatory-burden associations", x=-0.23)

    ax = fig.add_subplot(gs[3, 1])
    d = restraint.sort_values("FDR").head(8).copy()
    d["effect"] = d.T_label + " -> " + d.B_label
    d = d.sort_values("standardized_beta")
    y = np.arange(len(d))
    ax.hlines(y, d.loo_beta_min, d.loo_beta_max, color="#AAAAAA", lw=2)
    ax.scatter(d.standardized_beta, y, color=COL["ink"], s=18, zorder=3)
    ax.axvline(0, color=COL["ink"], lw=0.7)
    ax.set_yticks(y, d.effect, fontsize=4.7)
    ax.set_xlabel("Full beta; leave-one-series-out range")
    style_axis(ax, "x")
    panel(ax, "G", "Acquisition-series robustness", x=-0.20)

    save(fig, "Figure_S_Th17_Treg_extended_preview.png",
         "Figure_S_Th17_Treg_extended_preview.pdf")


def build_supplement_gamma_delta():
    source_png = ROOT / "Cell Press Redrawn Figure Set" / "Main Figures" / "Figure_3.png"
    image = Image.open(source_png).convert("RGB")
    crop_y = int(image.height * 0.746)
    crop = image.crop((0, crop_y + 45, image.width, image.height))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((0, 0, 1100, 72), fill="white")
    draw.rectangle((int(crop.width * 0.525), 0, int(crop.width * 0.595), 105), fill="white")

    fig = plt.figure(figsize=(7.05, 2.20))
    ax = fig.add_axes([0.01, 0.03, 0.98, 0.92])
    ax.imshow(np.asarray(crop))
    ax.axis("off")
    ax.text(0.005, 0.97, "A", transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    ax.text(0.055, 0.97, "Vγ9Vδ2 dominates paired γδ repertoires", transform=ax.transAxes,
            fontsize=7.1, fontweight="bold", va="top", ha="left")
    ax.text(0.515, 0.97, "B", transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    save(fig, "Figure_S_GammaDelta_relocated_preview.png",
         "Figure_S_GammaDelta_relocated_preview.pdf")


def main():
    clone_axis = pd.read_csv(TB / "Table_TB3_axis_clone_summary.csv")
    sharing = pd.read_csv(TB / "Table_TB6_clone_sharing_permutation.csv")
    size_models = pd.read_csv(TB / "Table_TB7_clone_size_models.csv")
    delta_tests = pd.read_csv(TB / "Table_TB8_expanded_singleton_tests.csv")
    corr = pd.read_csv(TB / "Table_TB10_targeted_T_B_correlations.csv")
    restraint = pd.read_csv(TB / "Table_TB11_regulatory_restraint_models.csv")
    clinical = pd.read_csv(TB / "Table_TB13_clinical_associations.csv")
    analysis = pd.read_csv(TB / "Table_TB9_T_B_analysis_dataset.csv")
    global_corr = pd.read_csv(SOURCE / "Figure_6_panel_A_global_coordination.csv")
    cytotoxic = pd.read_csv(SOURCE / "Figure_6_panel_D_cytotoxic_coupling.csv")
    validation = pd.read_csv(SOURCE / "Figure_6_panel_G_blocked_validation_summary.csv")

    build_revised_figure3(clone_axis, sharing)
    build_revised_figure6(corr, restraint, analysis, global_corr, cytotoxic, validation)
    build_supplement_th17(clone_axis, sharing, size_models, delta_tests, corr, clinical, restraint)
    build_supplement_gamma_delta()
    print(PREVIEW)


if __name__ == "__main__":
    main()

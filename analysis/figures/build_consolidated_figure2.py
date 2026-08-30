from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_trajectory_integrated_tcr_figures as base


TCR_METRICS = Path(
    r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 2 TCR\clonality_metrics_IBDTCR_Immunarch.csv"
)
EXACT = ROOT / "High Impact Additional Analyses" / "Table_HI_exact_clonotype_definition_metrics_by_participant.csv"
EXACT_TESTS = ROOT / "High Impact Additional Analyses" / "Table_HI_exact_clonotype_definition_pairwise_tests.csv"
DOSE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data" / "Figure2B_paired4_clone_size_dose_response_displayed.csv"
TRENDS = ROOT / "Cell Press Redrawn Figure Set" / "Source Data" / "Figure2B_paired4_clone_size_trend_tests.csv"

FIG2B_MODULES = {
    "EOMES_ZEB2_inflammatory_CD8_TRM_like": "EOMES–ZEB2",
    "Effector_cytotoxicity": "Cytotoxicity",
    "Th1_Tc1_inflammatory": "Th1/Tc1",
    "GZMK_inflammatory_memory": "GZMK inflammatory\nmemory",
}

TRAJECTORY_CMAP = LinearSegmentedColormap.from_list(
    "cellpress_trajectory",
    ["#31356E", "#3D65A5", "#2A9D8F", "#E9C46A", "#E76F51"],
    N=256,
)


def save_consolidated(fig):
    stem = base.MAIN / "Figure_2_consolidated"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=.04)
    fig.savefig(stem.with_suffix(".png"), dpi=350, facecolor="white", bbox_inches="tight", pad_inches=.04)
    fig.savefig(stem.with_suffix(".tif"), dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def diagnosis_boxstrip(ax, data, metric, ylabel):
    order = base.DIAG
    values = [data.loc[data.Diagnosis.eq(d), metric].dropna().to_numpy(float) for d in order]
    bp = ax.boxplot(
        values,
        positions=np.arange(3),
        widths=.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": base.COL["ink"], "lw": .8},
        whiskerprops={"color": base.COL["muted"], "lw": .65},
        capprops={"color": base.COL["muted"], "lw": .65},
        boxprops={"lw": .65, "edgecolor": base.COL["muted"]},
    )
    for patch, diagnosis in zip(bp["boxes"], order):
        patch.set_facecolor(base.COL[diagnosis])
        patch.set_alpha(.18)
    rng = np.random.default_rng(20260828)
    for x, diagnosis in enumerate(order):
        vals = data.loc[data.Diagnosis.eq(diagnosis), metric].dropna().to_numpy(float)
        ax.scatter(
            rng.normal(x, .055, len(vals)), vals, s=5.2, color=base.COL[diagnosis],
            alpha=.46, edgecolor="none", rasterized=True, zorder=3,
        )
    labels = [f"{d}\n(n={data.loc[data.Diagnosis.eq(d), metric].notna().sum()})" for d in order]
    ax.set_xticks(np.arange(3), labels, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    base.style_axis(ax, "y")


def plot_repertoire_overview(fig, spec, metrics, expanded):
    layout = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=spec, height_ratios=[.20, .80], hspace=.02, wspace=.57
    )
    header = fig.add_subplot(layout[0, :])
    header.set_axis_off()
    header.text(-.08, .98, "A", transform=header.transAxes, fontsize=12,
                fontweight="bold", va="top")
    header.text(0, .98, "Global TCR repertoire structure", transform=header.transAxes,
                fontsize=8, fontweight="bold", va="top")
    header.text(0, .20, "Descriptive overview; definition/depth sensitivity in Fig. S2",
                transform=header.transAxes, fontsize=5.0, color=base.COL["muted"], va="bottom")

    ax = fig.add_subplot(layout[1, 0])
    diagnosis_boxstrip(ax, metrics, "Clonality", "Clonality")
    ax.text(.03, .98, "Clonality", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.1, fontweight="bold")

    ax = fig.add_subplot(layout[1, 1])
    diagnosis_boxstrip(ax, metrics, "Shannon", "Shannon diversity")
    ax.text(.03, .98, "Diversity", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.1, fontweight="bold")

    ax = fig.add_subplot(layout[1, 2])
    diagnosis_boxstrip(ax, expanded, "expanded_cell_percent", "Expanded T cells (%)")
    ax.text(.03, .98, "Expansion burden", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.1, fontweight="bold")


def plot_dose_response(fig, spec, dose, trends):
    layout = GridSpecFromSubplotSpec(
        2, 4, subplot_spec=spec, height_ratios=[.20, .80], hspace=.02, wspace=.28
    )
    header = fig.add_subplot(layout[0, :])
    header.set_axis_off()
    header.text(-.08, .98, "B", transform=header.transAxes, fontsize=12,
                fontweight="bold", va="top")
    header.text(0, .98, "State-matched clone-size dose response", transform=header.transAxes,
                fontsize=8, fontweight="bold", va="top")
    handles = [Line2D([0], [0], color=base.COL["ink"], lw=1.25, marker="D", ms=3,
                      markerfacecolor="white", label="Pooled")]
    handles += [Line2D([0], [0], color=base.COL[d], lw=.8, marker="o", ms=2.5, label=d) for d in base.DIAG]
    header.legend(handles=handles, frameon=False, loc="lower right", bbox_to_anchor=(1, -.02),
                  ncol=4, handlelength=.9, columnspacing=.45, handletextpad=.20, fontsize=4.8)
    bins = ["Singleton", "2 cells", "3-4 cells", ">=5 cells"]
    x = np.arange(4)
    for j, (module, label) in enumerate(FIG2B_MODULES.items()):
        ax = fig.add_subplot(layout[1, j])
        dm = dose[dose.module.eq(module)]
        for diagnosis in base.DIAG:
            z = dm[dm.Diagnosis.eq(diagnosis)].set_index("clone_bin").reindex(bins)
            ax.plot(x, z["median"], color=base.COL[diagnosis], lw=.75, marker="o", ms=2.2, alpha=.72)
        pooled = dm[dm.Diagnosis.eq("Pooled")].set_index("clone_bin").reindex(bins)
        ax.errorbar(
            x, pooled["median"],
            yerr=[pooled["median"] - pooled.ci_low, pooled.ci_high - pooled["median"]],
            fmt="D-", ms=3.0, lw=1.15, color=base.COL["ink"], markerfacecolor="white", capsize=1.4,
        )
        trend = trends[(trends.module.eq(module)) & (trends.contrast.eq("Pooled ordered trend"))].iloc[0]
        ax.text(.03, .98, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=5.45, fontweight="bold")
        ax.text(.98, .05, f"slope {trend.estimate:+.3f}/bin\n{base.q_text(trend.FDR)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=4.25, color=base.COL["muted"])
        ax.axhline(0, color="#AAAAAA", lw=.55, ls="--")
        ax.set_xticks(x, ["1", "2", "3–4", "≥5"])
        ax.set_xlabel("Cells per clonotype", fontsize=5.4)
        if j == 0:
            ax.set_ylabel("State-matched score delta")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_ylim(-0.11, 0.26)
        ax.set_yticks([-0.10, 0.00, 0.10, 0.20])
        base.style_axis(ax, "y")


def plot_trajectory_umap(fig, spec, cells, primary):
    ax = fig.add_subplot(spec)
    show_parts = [group.sample(min(3300, len(group)), random_state=20260828)
                  for _, group in primary.groupby("state")]
    show = pd.concat(show_parts).sort_values("pseudotime")
    trm_cells = cells[cells.state.eq("CD8 Trm")]
    trm = trm_cells.sample(min(1200, len(trm_cells)), random_state=20260828)
    ax.scatter(trm.UMAP_1, trm.UMAP_2, s=.8, color="#CFC9DC", alpha=.52,
               linewidth=0, rasterized=True)
    sc = ax.scatter(show.UMAP_1, show.UMAP_2, c=show.pseudotime, cmap=TRAJECTORY_CMAP, s=.9,
                    alpha=.70, linewidth=0, rasterized=True)
    centers = cells.groupby("state")[["UMAP_1", "UMAP_2"]].median()
    primary_edges = [
        ("CD8 Naive", "CD8 Tcm CCR4-"), ("CD8 Tcm CCR4-", "CD8 Tem GZMK+"),
        ("CD8 Tem GZMK+", "CD8 Tem GZMB+"), ("CD8 Tem GZMB+", "CD8 Temra"),
        ("CD8 Tem GZMB+", "CD8 HLA-DR+"),
    ]
    for first, second in primary_edges:
        xy = centers.loc[[first, second]].to_numpy()
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=2.7, alpha=.82, zorder=4)
        ax.plot(xy[:, 0], xy[:, 1], color="#30383E", lw=.75, alpha=.76, zorder=5)
    xy = centers.loc[["CD8 Tem GZMK+", "CD8 Trm"]].to_numpy()
    ax.plot(xy[:, 0], xy[:, 1], color="#81799B", lw=.8, ls="--", zorder=5)
    root = centers.loc["CD8 Naive"]
    ax.scatter(root.UMAP_1, root.UMAP_2, marker="*", s=45, color="#F4A261",
               edgecolor="white", lw=.45, zorder=8)
    label_offsets = {
        "CD8 Naive": (-18, -2),
        "CD8 Tcm CCR4-": (10, 10),
        "CD8 Tem GZMK+": (13, -2),
        "CD8 Tem GZMB+": (15, -13),
        "CD8 Temra": (-2, -17),
        "CD8 HLA-DR+": (-24, -6),
        "CD8 Trm": (12, 13),
    }
    for state_name, row in centers.iterrows():
        ax.annotate(
            base.STATE_SHORT[state_name], xy=(row.UMAP_1, row.UMAP_2),
            xytext=label_offsets[state_name], textcoords="offset points",
            fontsize=4.55, ha="center", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": .30},
            arrowprops={"arrowstyle": "-", "color": "#6F6F6F", "lw": .35,
                        "shrinkA": 1.2, "shrinkB": 1.2},
            zorder=9,
        )
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=.052, pad=.07, aspect=25)
    cb.set_label("Consensus graph-geodesic pseudotime", labelpad=1)
    cb.ax.tick_params(labelsize=5.3, pad=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("CD8 UMAP 1")
    ax.set_ylabel("CD8 UMAP 2")
    ax.set_title("Rooted conventional-CD8 trajectory", loc="left", fontweight="bold", pad=4)
    base.panel_label(ax, "C", x=-.18, y=1.10)


def plot_state_positions(fig, spec, states):
    ax = fig.add_subplot(spec)
    ordered = states.sort_values("median_pseudotime")
    y = np.arange(len(ordered))[::-1]
    ax.hlines(y, ordered.q1_pseudotime, ordered.q3_pseudotime, color=base.COL["muted"], lw=1)
    ax.scatter(ordered.median_pseudotime, y, c=ordered.median_pseudotime, cmap=TRAJECTORY_CMAP,
               vmin=0, vmax=1, s=28, edgecolor="white", lw=.4, zorder=3)
    ax.set_yticks(y, [base.STATE_SHORT[s] for s in ordered.state])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Median pseudotime [IQR]")
    ax.set_title("Scalar-continuum positions", loc="left", fontweight="bold", pad=4)
    base.style_axis(ax, "x")
    base.panel_label(ax, "D", x=-.22, y=1.10)


def plot_program_kinetics(fig, spec, display, tests):
    container = fig.add_subplot(spec)
    container.set_axis_off()
    container.text(-.015, 1.10, "G", transform=container.transAxes, fontsize=12, fontweight="bold", va="top")
    container.text(.055, 1.10, "Expansion-associated program kinetics", transform=container.transAxes,
                   fontsize=8, fontweight="bold", va="top")
    container.text(.50, -.15, "Graph-geodesic pseudotime", transform=container.transAxes,
                   fontsize=7.2, ha="center", va="top")
    sub = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=.28)
    last_ax = None
    first_ax = None
    for j, module in enumerate(base.MODULES):
        ax = fig.add_subplot(sub[0, j], sharey=first_ax) if first_ax is not None else fig.add_subplot(sub[0, j])
        if first_ax is None:
            first_ax = ax
        last_ax = ax
        dm = display[display.module.eq(module)]
        for status in ["Singleton", "Expanded"]:
            z = dm[(dm.clone_status.eq(status)) & (dm.n_participants >= 8)].sort_values("pt_mid")
            ax.fill_between(z.pt_mid.to_numpy(float), z.ci_low.to_numpy(float), z.ci_high.to_numpy(float),
                            color=base.COL[status], alpha=.12, linewidth=0)
            ax.plot(z.pt_mid, z.median_score_z, color=base.COL[status], lw=1.25,
                    marker="s" if status == "Singleton" else "o", ms=2.8, label=status)
        test = tests[tests.module.eq(module)].iloc[0]
        ax.text(.03, .98, base.MODULE_LABELS[module], transform=ax.transAxes, ha="left", va="top",
                fontsize=5.7, fontweight="bold")
        ax.text(.03, .88,
                f"trajectory {base.q_text(test.trajectory_FDR)}\nexpansion×trajectory {base.q_text(test.expansion_by_trajectory_FDR)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=4.7, color=base.COL["muted"])
        ax.axhline(0, color="#AAAAAA", lw=.55)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .5, 1])
        if j == 0:
            ax.set_ylabel("Participant-level module score (z)")
        else:
            ax.tick_params(axis="y", labelleft=False)
        base.style_axis(ax, "y")
    last_ax.legend(frameon=False, loc="lower right")


def plot_diversity_supplement(fig, spec, metrics):
    container = fig.add_subplot(spec)
    container.set_axis_off()
    container.set_zorder(10)
    container.patch.set_alpha(0)
    container.text(-.035, 1.14, "A", transform=container.transAxes, fontsize=12,
                   fontweight="bold", va="top")
    container.text(0, 1.14, "Complementary TCR diversity metrics", transform=container.transAxes,
                   fontsize=8, fontweight="bold", va="top")
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=.46)
    for j, (metric, ylabel, title) in enumerate([
        ("Shannon", "Shannon diversity", "Shannon"),
        ("InvSimp", "Inverse Simpson diversity", "Inverse Simpson"),
        ("D50_Clones", "D50 clones", "D50"),
    ]):
        ax = fig.add_subplot(sub[0, j])
        diagnosis_boxstrip(ax, metrics, metric, ylabel)
        ax.text(.03, .98, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.1, fontweight="bold")


def sensitivity_rows(tests):
    specs = [
        ("TCR beta-only", "Original depth", 2, "β only ≥2"),
        ("TCR beta-only", "Original depth", 5, "β only ≥5"),
        ("TCR paired alpha-beta", "Original depth", 2, "paired αβ ≥2"),
        ("TCR beta-only", "Depth-normalized", 2, "β only ≥2, rarefied"),
    ]
    rows = []
    for receptor, analysis, threshold, label in specs:
        z = tests[
            tests.receptor_definition.eq(receptor)
            & tests.analysis.eq(analysis)
            & tests.threshold.eq(threshold)
            & tests.contrast.isin(["CD vs Control", "UC vs Control"])
        ].copy()
        z["spec"] = label
        rows.append(z)
    return pd.concat(rows, ignore_index=True), [x[3] for x in specs]


def plot_sensitivity_forest(ax, data, order, metric, title, panel):
    ybase = np.arange(len(order))[::-1]
    offsets = {"CD vs Control": .11, "UC vs Control": -.11}
    for contrast in ["CD vs Control", "UC vs Control"]:
        z = data[(data.metric.eq(metric)) & data.contrast.eq(contrast)].set_index("spec").reindex(order)
        y = ybase + offsets[contrast]
        color = base.COL["CD"] if contrast.startswith("CD") else base.COL["UC"]
        ax.hlines(y, z.ci_low, z.ci_high, color=color, lw=.9)
        ax.scatter(z.rank_biserial, y, color=color, s=18, edgecolor="white", lw=.35,
                   label=contrast.replace(" vs Control", ""), zorder=3)
    ax.axvline(0, color=base.COL["muted"], lw=.65, ls="--")
    ax.set_yticks(ybase, order)
    ax.set_xlim(-.56, .56)
    ax.set_xlabel("Rank-biserial effect vs Control (95% CI)")
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    base.style_axis(ax, "x")
    base.panel_label(ax, panel, x=-.25, y=1.08)
    if panel == "B":
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2,
                  handletextpad=.25, columnspacing=.55)


def plot_threshold_sensitivity(ax, exact):
    z = exact[
        exact.receptor_definition.eq("TCR beta-only")
        & exact.analysis.eq("Original depth")
        & exact.threshold.isin([2, 3, 5])
    ].copy()
    rows = []
    for (diagnosis, threshold), group in z.groupby(["Diagnosis1", "threshold"]):
        values = 100 * group.expanded_cell_fraction.dropna().to_numpy(float)
        med, lo, hi = base.bootstrap_median(values)
        rows.append({"Diagnosis": diagnosis, "threshold": threshold, "median": med,
                     "ci_low": lo, "ci_high": hi, "n": len(values)})
    summary = pd.DataFrame(rows)
    for diagnosis in base.DIAG:
        q = summary[summary.Diagnosis.eq(diagnosis)].sort_values("threshold")
        ax.errorbar(q.threshold, q["median"], yerr=[q["median"] - q.ci_low, q.ci_high - q["median"]],
                    color=base.COL[diagnosis], marker="o", ms=3.2, lw=1, capsize=1.5,
                    label=diagnosis)
    ax.set_xticks([2, 3, 5])
    ax.set_xlabel("Expansion threshold (cells)")
    ax.set_ylabel("Participant-median expanded-cell fraction (%)")
    ax.set_title("Threshold sensitivity", loc="left", x=.10, fontweight="bold", pad=4)
    ax.legend(frameon=False, loc="upper right")
    base.style_axis(ax, "y")
    base.panel_label(ax, "D", x=-.04, y=1.08)
    return summary


def build_repertoire_robustness_supplement(metrics, exact, exact_tests):
    sensitivity, order = sensitivity_rows(exact_tests)
    fig = plt.figure(figsize=(7.48, 5.55), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[.92, 1.08], hspace=.62, wspace=.70,
                  left=.075, right=.985, top=.965, bottom=.10)
    plot_diversity_supplement(fig, gs[0, :], metrics)
    ax = fig.add_subplot(gs[1, 0])
    plot_sensitivity_forest(ax, sensitivity, order, "clonality", "Clonality sensitivity", "B")
    ax = fig.add_subplot(gs[1, 1])
    plot_sensitivity_forest(ax, sensitivity, order, "expanded_cell_fraction",
                            "Expansion-burden sensitivity", "C")
    ax = fig.add_subplot(gs[1, 2])
    threshold_summary = plot_threshold_sensitivity(ax, exact)

    sensitivity.to_csv(base.SRC / "FigureS_TCR_repertoire_definition_depth_sensitivity.csv", index=False)
    threshold_summary.to_csv(base.SRC / "FigureS_TCR_repertoire_threshold_sensitivity.csv", index=False)
    base.save_figure(fig, "Figure_S_TCR_repertoire_structure_robustness", base.SUPP)

    legend = (
        "Figure S. Robustness of the global TCR repertoire overview. "
        "(A) Participant-level Shannon diversity, inverse Simpson diversity, and D50 clone counts. "
        "(B,C) Rank-biserial diagnosis effects and bootstrap 95% confidence intervals for clonality and expanded-cell burden under beta-only and exact paired alpha-beta clonotype definitions, expansion thresholds of at least 2 or at least 5 cells, and beta-only depth normalization by rarefaction to 140 cells. "
        "(D) Participant-median expanded-cell fraction across beta-chain expansion thresholds. These sensitivity analyses support the use of the main-figure repertoire overview as descriptive context and show that the trajectory-linked conclusions do not depend on a single repertoire definition or expansion threshold."
    )
    (base.LEG / "Figure_S_TCR_repertoire_structure_robustness_legend.txt").write_text(
        legend + "\n", encoding="utf-8"
    )


def build_consolidated_figure():
    metrics = pd.read_csv(TCR_METRICS).rename(columns={"PatientID": "SampleID"})
    exact_all = pd.read_csv(EXACT)
    exact_tests = pd.read_csv(EXACT_TESTS)
    expanded = exact_all
    expanded = expanded[
        expanded.receptor_definition.eq("TCR beta-only")
        & expanded.analysis.eq("Original depth")
        & expanded.threshold.eq(2)
    ].copy().rename(columns={"Diagnosis1": "Diagnosis"})
    expanded["expanded_cell_percent"] = 100 * expanded.expanded_cell_fraction
    dose = pd.read_csv(DOSE)
    trends = pd.read_csv(TRENDS)

    overview = pd.concat([
        metrics[["SampleID", "Diagnosis", "Clonality"]].rename(columns={"Clonality": "value"}).assign(metric="Clonality"),
        metrics[["SampleID", "Diagnosis", "Shannon"]].rename(columns={"Shannon": "value"}).assign(metric="Shannon diversity"),
        expanded[["SampleID", "Diagnosis", "expanded_cell_percent"]].rename(columns={"expanded_cell_percent": "value"}).assign(metric="Expanded T cells (%)"),
    ], ignore_index=True)
    overview.to_csv(base.SRC / "Figure2_consolidated_repertoire_overview.csv", index=False)
    dose.to_csv(base.SRC / "Figure2_consolidated_clone_size_dose_response.csv", index=False)
    trends.to_csv(base.SRC / "Figure2_consolidated_clone_size_trend_tests.csv", index=False)

    cells, primary, states, clone_bins, clone_tests, kinetics_display, kinetics_tests = base.prepare_primary_graph_outputs()

    fig = plt.figure(figsize=(7.48, 8.72), facecolor="white")
    outer = GridSpec(
        3, 6, figure=fig, height_ratios=[.88, 1.02, 1.05], hspace=.52, wspace=1.02,
        left=.075, right=.985, top=.975, bottom=.06,
    )
    plot_repertoire_overview(fig, outer[0, 0:3], metrics, expanded)
    plot_dose_response(fig, outer[0, 3:6], dose, trends)
    plot_trajectory_umap(fig, outer[1, 0:2], cells, primary)
    plot_state_positions(fig, outer[1, 2:4], states)

    ax = fig.add_subplot(outer[1, 4:6])
    base.plot_clone_summary(
        ax, clone_bins, clone_tests, "participant_median_pseudotime", ["1", "2", "3-4", ">=5"],
        "Clone size shifts trajectory position", "E", "Participant-median clone pseudotime", 0,
    )

    ax = fig.add_subplot(outer[2, 0:2])
    base.plot_clone_summary(
        ax, clone_bins, clone_tests, "participant_median_span", ["2", "3-4", ">=5"],
        "Larger clones span the continuum", "F", "Participant-median within-clone span", 1,
    )
    plot_program_kinetics(fig, outer[2, 2:6], kinetics_display, kinetics_tests)
    save_consolidated(fig)

    legend = (
        "Figure 2. TCR repertoire structure, clonal expansion, and conventional-CD8 differentiation in IBD. "
        "(A) Participant-level TCR clonality, Shannon diversity, and the percentage of T cells belonging to exact productive TCR-beta clonotypes represented by at least two cells. Boxes show medians and interquartile ranges; points represent participants. These distributions provide descriptive cohort context; clonotype-definition, depth-normalization, diversity-metric, and expansion-threshold sensitivities are shown in Fig. S2. "
        "(B) State-matched clone-size dose response for EOMES-ZEB2, cytotoxicity, Th1/Tc1, and GZMK inflammatory-memory programs across singleton, 2-cell, 3-4-cell, and at least 5-cell exact paired alpha-beta clonotypes. Diagnosis-specific trajectories are shown with pooled participant-bootstrap estimates and ordered-trend tests. "
        "(C) Conventional-CD8 UMAP colored by consensus graph-geodesic pseudotime; the skeleton summarizes the principal scalar continuum and the Trm branch. "
        "(D) Median and interquartile-range pseudotime positions for states included in the scalar continuum; Trm was retained as a branch and not forced onto the scalar axis. "
        "(E) Participant-level clone-median pseudotime across exact paired alpha-beta clone-size bins. "
        "(F) Participant-level median within-clone pseudotime span among expanded clonotypes. Diamonds and bars in E and F show pooled medians and participant-bootstrap 95% confidence intervals; annotations report participant-fixed-effect slopes per clone-size doubling with participant-clustered standard errors. "
        "(G) Participant-balanced EOMES-ZEB2, cytotoxicity, Th1/Tc1, and GZMK inflammatory-memory module kinetics for singleton and expanded exact paired alpha-beta clonotypes. Shading shows participant-bootstrap 95% confidence intervals; FDR values derive from participant-blocked spline models. Cross-sectional pseudotime represents transcriptional ordering and does not establish temporal progression."
    )
    (base.LEG / "Figure_2_consolidated_legend.txt").write_text(legend + "\n", encoding="utf-8")
    build_repertoire_robustness_supplement(metrics, exact_all, exact_tests)
    print(base.MAIN / "Figure_2_consolidated.png")


if __name__ == "__main__":
    build_consolidated_figure()

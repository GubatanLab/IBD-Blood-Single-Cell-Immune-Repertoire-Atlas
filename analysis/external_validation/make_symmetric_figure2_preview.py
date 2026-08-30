from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT.parent
SCRIPTS = MANUSCRIPT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_consolidated_figure2 as cfig
import build_trajectory_integrated_tcr_figures as base


OUT = ROOT / "Proposed Main Figure 2H Preview"
OUT.mkdir(exist_ok=True)
EXTERNAL = ROOT / "results" / "GSE261334_expansion_program_dose_response.csv"
EXTERNAL_STATS = ROOT / "results" / "External_validation_all_statistics.csv"

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.2,
    "axes.titlesize": 7.4,
    "axes.labelsize": 6.2,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 5.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def plot_external_validation(fig, spec, data):
    layout = GridSpecFromSubplotSpec(
        2, 12, subplot_spec=spec, height_ratios=[0.25, 0.75], hspace=0.02, wspace=0.18
    )
    header = fig.add_subplot(layout[0, :])
    header.set_axis_off()
    header.text(-0.037, 0.94, "H", transform=header.transAxes, fontsize=12,
                fontweight="bold", va="top")
    header.text(0.0, 0.94, "Independent longitudinal UC cohort, baseline",
                transform=header.transAxes, fontsize=8, fontweight="bold", va="top")
    header.text(0.0, 0.12,
                "Exact paired αβ clonotypes; n=10 participants; clone-size bins: 1, 2, 3–4, ≥5 cells",
                transform=header.transAxes, fontsize=5.0, color=base.COL["muted"], va="bottom")

    ax = fig.add_subplot(layout[1, 2:10])
    stat = fig.add_subplot(layout[1, 10:12], sharey=ax)
    stat.set_axis_off()

    rows = [
        ("Cytotoxic", "Cytotoxic", "#087FB6"),
        ("GZMK memory", "GZMK inflammatory memory", "#087FB6"),
        ("Th17 pathogenic", "Pathogenic Th17-like", "#087FB6"),
        ("Th17 conventional", "Conventional Th17", "#687782"),
        ("Naive/CM", "Naive/central memory", "#687782"),
    ]
    positions = np.arange(5)[::-1]
    rng = np.random.default_rng(20260829)

    ax.axvspan(-1.08, 0, color="#F3F5F7", zorder=-3)
    ax.axvspan(0, 1.08, color="#F3F8FB", zorder=-3)
    ax.axvline(0, color="#777777", lw=0.65, ls=(0, (3, 2)))
    for pos, (module, label, color) in zip(positions, rows):
        values = data.loc[data.module.eq(module), "rho"].to_numpy(float)
        ax.boxplot(
            values, positions=[pos], vert=False, widths=.42, patch_artist=True,
            showfliers=False, manage_ticks=False,
            medianprops={"color": "#17324D", "linewidth": .9},
            boxprops={"facecolor": color, "edgecolor": color, "linewidth": .65, "alpha": .20},
            whiskerprops={"color": color, "linewidth": .65},
            capprops={"color": color, "linewidth": .65},
        )
        ax.scatter(values, pos + rng.uniform(-.12, .12, len(values)), s=8.5,
                   color=color, edgecolor="white", linewidth=.25, zorder=4)
        stat.text(.06, pos, f"{np.median(values):+.2f}", ha="left", va="center",
                  fontsize=5.7, color="#253746")
        stat.text(.57, pos, "0.0056", ha="left", va="center",
                  fontsize=5.7, color="#253746")

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-.62, 4.62)
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_yticks(positions, [r[1] for r in rows])
    ax.set_xlabel("Clone-size/program correlation (participant Spearman ρ)", labelpad=1.5)
    ax.grid(axis="x", color="#D8DFE5", lw=.42)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(.7)
    ax.spines["bottom"].set_linewidth(.7)

    stat.set_ylim(ax.get_ylim())
    stat.text(.06, 4.62, "Median ρ", ha="left", va="bottom", fontsize=5.7,
              fontweight="bold", color="#253746")
    stat.text(.57, 4.62, "FDR", ha="left", va="bottom", fontsize=5.7,
              fontweight="bold", color="#253746")


def main():
    metrics = pd.read_csv(cfig.TCR_METRICS).rename(columns={"PatientID": "SampleID"})
    exact_all = pd.read_csv(cfig.EXACT)
    expanded = exact_all[
        exact_all.receptor_definition.eq("TCR beta-only")
        & exact_all.analysis.eq("Original depth")
        & exact_all.threshold.eq(2)
    ].copy().rename(columns={"Diagnosis1": "Diagnosis"})
    expanded["expanded_cell_percent"] = 100 * expanded.expanded_cell_fraction
    dose = pd.read_csv(cfig.DOSE)
    trends = pd.read_csv(cfig.TRENDS)
    cells, primary, states, clone_bins, clone_tests, kinetics_display, kinetics_tests = base.prepare_primary_graph_outputs()

    external = pd.read_csv(EXTERNAL)
    external = external[
        external.receptor.eq("TCR")
        & external.time.eq("baseline")
        & external.disease.eq("UC")
        & external.module.isin(["Cytotoxic", "GZMK memory", "Th17 pathogenic", "Th17 conventional", "Naive/CM"])
    ].copy()
    assert external.groupby("module").participant.nunique().eq(10).all()
    external.to_csv(OUT / "Figure2H_external_validation_participant_correlations.csv", index=False)

    external_stats = pd.read_csv(EXTERNAL_STATS)
    selected_comparisons = [
        "TCR: Cytotoxic",
        "TCR: GZMK memory",
        "TCR: Th17 pathogenic",
        "TCR: Th17 conventional",
        "TCR: Naive/CM",
    ]
    external_summary = external_stats[
        external_stats.family.eq("GSE261334 expansion dose response")
        & external_stats.comparison.isin(selected_comparisons)
    ].copy()
    external_summary.to_csv(OUT / "Figure2H_external_validation_summary.csv", index=False)

    # Independent nested grids allow the top, middle, and lower rows to use
    # different gutters. This keeps long labels from intruding into neighbors.
    fig = plt.figure(figsize=(7.48, 9.68), facecolor="white")
    outer = GridSpec(
        4, 1, figure=fig,
        height_ratios=[.88, 1.02, 1.02, .64],
        hspace=.58,
        left=.075, right=.985, top=.978, bottom=.050,
    )

    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=.24)
    cfig.plot_repertoire_overview(fig, top[0, 0], metrics, expanded)
    cfig.plot_dose_response(fig, top[0, 1], dose, trends)

    middle = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, 0], wspace=.30)
    cfig.plot_trajectory_umap(fig, middle[0, 0], cells, primary)
    cfig.plot_state_positions(fig, middle[0, 1], states)
    ax = fig.add_subplot(middle[0, 2])
    base.plot_clone_summary(
        ax, clone_bins, clone_tests, "participant_median_pseudotime", ["1", "2", "3-4", ">=5"],
        "Clone size shifts trajectory position", "E", "Participant-median clone pseudotime", 0,
    )
    lower = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2, 0], wspace=.23)
    ax = fig.add_subplot(lower[0, 0])
    base.plot_clone_summary(
        ax, clone_bins, clone_tests, "participant_median_span", ["2", "3-4", ">=5"],
        "Larger clones span the continuum", "F", "Participant-median within-clone span", 1,
    )
    cfig.plot_program_kinetics(fig, lower[0, 1:3], kinetics_display, kinetics_tests)
    plot_external_validation(fig, outer[3, 0], external)

    png = OUT / "Figure_2_symmetric_spaced_with_2H_PREVIEW.png"
    pdf = OUT / "Figure_2_symmetric_spaced_with_2H_PREVIEW.pdf"
    tif = OUT / "Figure_2_symmetric_spaced_with_2H_PREVIEW.tif"
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=.035)
    fig.savefig(png, dpi=350, facecolor="white", bbox_inches="tight", pad_inches=.035)
    fig.savefig(tif, dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"},
                bbox_inches="tight", pad_inches=.035)
    plt.close(fig)
    print(png)
    print(pdf)
    print(tif)


if __name__ == "__main__":
    main()

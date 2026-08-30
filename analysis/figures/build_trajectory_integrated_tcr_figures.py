from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "High Impact Additional Analyses" / "CD8 TCR Trajectory"
ORIGINAL = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
OUT = ROOT / "Trajectory Integrated Figure Set"
MAIN = OUT / "Main Figures"
SUPP = OUT / "Supplementary Figures"
SRC = OUT / "Source Data"
LEG = OUT / "Legends"
TEXT = OUT / "Manuscript Text"
for directory in (MAIN, SUPP, SRC, LEG, TEXT):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cd8_tcr_trajectory as trajectory_tools


COL = {
    "Control": "#6F6F6F",
    "CD": "#0072B2",
    "UC": "#D55E00",
    "Singleton": "#6F6F6F",
    "Expanded": "#D55E00",
    "positive": "#D55E00",
    "negative": "#0072B2",
    "ink": "#222222",
    "muted": "#777777",
    "grid": "#D9D9D9",
    "slingshot": "#009E73",
    "graph": "#7A4EAB",
}
DIAG = ["Control", "CD", "UC"]
MODULES = ["EOMES_ZEB2", "Cytotoxicity", "Th1_Tc1", "GZMK_inflammatory_memory"]
MODULE_LABELS = {
    "EOMES_ZEB2": "EOMES–ZEB2",
    "Cytotoxicity": "Cytotoxicity",
    "Th1_Tc1": "Th1/Tc1",
    "GZMK_inflammatory_memory": "GZMK inflammatory\nmemory",
}
ORIGINAL_MODULES = {
    "EOMES_ZEB2_inflammatory_CD8_TRM_like": "EOMES–ZEB2",
    "Effector_cytotoxicity": "Cytotoxicity",
    "Th1_Tc1_inflammatory": "Th1/Tc1",
}
STATE_SHORT = {
    "CD8 Naive": "Naive",
    "CD8 Tcm CCR4-": "Tcm CCR4−",
    "CD8 Tem GZMK+": "Tem GZMK+",
    "CD8 Tem GZMB+": "Tem GZMB+",
    "CD8 Temra": "Temra",
    "CD8 HLA-DR+": "HLA-DR+",
    "CD8 Trm": "Trm branch",
}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": .6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
RNG = np.random.default_rng(20260828)


def style_axis(ax, grid_axis: str | None = None):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COL["grid"], lw=.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x=-.16, y=1.10):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def save_figure(fig, stem: str, folder: Path):
    fig.savefig(folder / f"{stem}.pdf", facecolor="white", bbox_inches="tight", pad_inches=.04)
    fig.savefig(folder / f"{stem}.png", dpi=350, facecolor="white", bbox_inches="tight", pad_inches=.04)
    fig.savefig(folder / f"{stem}.tif", dpi=500, facecolor="white", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def bootstrap_median(values, n_boot=3000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    draws = np.median(values[RNG.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return float(np.median(values)), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def q_text(value):
    if value < 1e-4:
        return f"q={value:.1e}"
    return f"q={value:.3g}"


def prepare_primary_graph_outputs():
    cells = pd.read_csv(TRAJ / "Table_TJ2_cd8_cells_with_pseudotime.csv.gz", low_memory=False)
    primary = cells[cells.state.ne("CD8 Trm")].copy()
    states = trajectory_tools.summarize_states(primary)
    clones, clone_bins, clone_status, clone_tests = trajectory_tools.analyze_clones(primary)
    participant_bins, kinetics_display, kinetics_tests = trajectory_tools.program_kinetics(primary)

    primary.to_csv(SRC / "Figure2_primary_graph_cells.csv.gz", index=False)
    states.to_csv(SRC / "Figure2_primary_state_pseudotime.csv", index=False)
    clones.to_csv(SRC / "Figure2_primary_clone_trajectory.csv", index=False)
    clone_bins.to_csv(SRC / "Figure2_primary_participant_clone_bins.csv", index=False)
    clone_status.to_csv(SRC / "Figure2_primary_expanded_singleton.csv", index=False)
    clone_tests.to_csv(SRC / "Figure2_primary_clone_tests.csv", index=False)
    participant_bins.to_csv(SRC / "Figure2_primary_participant_program_kinetics.csv", index=False)
    kinetics_display.to_csv(SRC / "Figure2_primary_program_kinetics_display.csv", index=False)
    kinetics_tests.to_csv(SRC / "Figure2_primary_program_kinetics_tests.csv", index=False)
    return cells, primary, states, clone_bins, clone_tests, kinetics_display, kinetics_tests


def plot_clone_summary(ax, clone_bins, clone_tests, metric, order, title, panel, ylabel, test_row):
    values = [clone_bins.loc[clone_bins.clone_bin.eq(bin_name), metric].dropna().to_numpy(float) for bin_name in order]
    violin = ax.violinplot(values, positions=np.arange(len(order)), widths=.72, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#D9D9D9"); body.set_edgecolor("none"); body.set_alpha(.65)
    for x, bin_name in enumerate(order):
        group = clone_bins[clone_bins.clone_bin.eq(bin_name)]
        jitter = RNG.normal(x, .055, len(group))
        ax.scatter(jitter, group[metric], s=6.5, alpha=.34, edgecolor="none",
                   c=[COL.get(d, COL["muted"]) for d in group.Diagnosis1], rasterized=True)
        med, lo, hi = bootstrap_median(group[metric])
        ax.vlines(x, lo, hi, color=COL["ink"], lw=1.35, zorder=5)
        ax.scatter(x, med, marker="D", s=27, color=COL["positive"], edgecolor="white", lw=.4, zorder=6)
    test = clone_tests.iloc[test_row]
    ax.text(.02, .98, f"slope {test.estimate:+.3f} per doubling\n95% CI {test.ci_low:.3f}–{test.ci_high:.3f}; {q_text(test.FDR)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=COL["muted"])
    ax.set_xticks(np.arange(len(order)), [f"{b}\n(n={clone_bins.loc[clone_bins.clone_bin.eq(b), 'SampleID'].nunique()})" for b in order])
    ax.set_xlabel("Exact paired αβ clone size (cells)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    style_axis(ax, "y"); panel_label(ax, panel)


def plot_main_figure(cells, primary, states, clone_bins, clone_tests, kinetics_display, kinetics_tests):
    dose = pd.read_csv(ORIGINAL / "Figure2_clone_size_dose_response_displayed.csv")
    trends = pd.read_csv(ORIGINAL / "Figure2_clone_size_trend_tests.csv")

    fig = plt.figure(figsize=(7.48, 9.45), facecolor="white")
    outer = GridSpec(3, 6, figure=fig, height_ratios=[1.00, 1.08, .98], hspace=.76, wspace=1.02,
                     left=.075, right=.985, top=.975, bottom=.06)

    # A. Rooted graph-geodesic trajectory with the Trm branch displayed separately.
    ax = fig.add_subplot(outer[0, 0:2])
    show_parts = []
    for _, group in primary.groupby("state"):
        show_parts.append(group.sample(min(3300, len(group)), random_state=20260828))
    show = pd.concat(show_parts).sort_values("pseudotime")
    trm = cells[cells.state.eq("CD8 Trm")].sample(min(1200, cells.state.eq("CD8 Trm").sum()), random_state=20260828)
    ax.scatter(trm.UMAP_1, trm.UMAP_2, s=.8, color="#B8B8B8", alpha=.45, linewidth=0, rasterized=True)
    sc = ax.scatter(show.UMAP_1, show.UMAP_2, c=show.pseudotime, cmap="viridis", s=.9, alpha=.70, linewidth=0, rasterized=True)
    centers = cells.groupby("state")[["UMAP_1", "UMAP_2"]].median()
    primary_edges = [("CD8 Naive", "CD8 Tcm CCR4-"), ("CD8 Tcm CCR4-", "CD8 Tem GZMK+"),
                     ("CD8 Tem GZMK+", "CD8 Tem GZMB+"), ("CD8 Tem GZMB+", "CD8 Temra"),
                     ("CD8 Tem GZMB+", "CD8 HLA-DR+")]
    for first, second in primary_edges:
        xy = centers.loc[[first, second]].to_numpy()
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=2.7, alpha=.82, zorder=4)
        ax.plot(xy[:, 0], xy[:, 1], color="#30383E", lw=.75, alpha=.76, zorder=5)
    xy = centers.loc[["CD8 Tem GZMK+", "CD8 Trm"]].to_numpy()
    ax.plot(xy[:, 0], xy[:, 1], color=COL["muted"], lw=.75, ls="--", zorder=5)
    for state_name, row in centers.iterrows():
        ax.text(row.UMAP_1, row.UMAP_2, STATE_SHORT[state_name], fontsize=4.7, ha="center", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .72, "pad": .35}, zorder=7)
    root_xy = centers.loc["CD8 Naive"]
    ax.scatter(root_xy.UMAP_1, root_xy.UMAP_2, marker="*", s=45, color=COL["positive"], edgecolor="white", lw=.45, zorder=8)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=.052, pad=.07, aspect=25)
    cb.set_label("Consensus graph-geodesic pseudotime", labelpad=1); cb.ax.tick_params(labelsize=5.5, pad=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("CD8 UMAP 1"); ax.set_ylabel("CD8 UMAP 2")
    ax.set_title("Rooted conventional-CD8 trajectory", loc="left", fontweight="bold", pad=4)
    panel_label(ax, "A", x=-.18, y=1.08)

    # B. Scalar state positions; Trm is deliberately not forced onto this axis.
    ax = fig.add_subplot(outer[0, 2:4])
    ordered = states.sort_values("median_pseudotime")
    y = np.arange(len(ordered))[::-1]
    ax.hlines(y, ordered.q1_pseudotime, ordered.q3_pseudotime, color=COL["muted"], lw=1)
    ax.scatter(ordered.median_pseudotime, y, c=ordered.median_pseudotime, cmap="viridis", vmin=0, vmax=1,
               s=28, edgecolor="white", lw=.4, zorder=3)
    ax.set_yticks(y, [STATE_SHORT[s] for s in ordered.state]); ax.set_xlim(0, 1)
    ax.set_xlabel("Median pseudotime [IQR]")
    ax.text(.99, .98, "Trm branch shown in A", transform=ax.transAxes, ha="right", va="top", fontsize=5.2, color=COL["muted"])
    ax.set_title("Scalar-continuum positions", loc="left", fontweight="bold", pad=4)
    style_axis(ax, "x"); panel_label(ax, "B", x=-.22, y=1.08)

    ax = fig.add_subplot(outer[0, 4:6])
    plot_clone_summary(ax, clone_bins, clone_tests, "participant_median_pseudotime", ["1", "2", "3-4", ">=5"],
                       "Clone size shifts trajectory position", "C", "Participant-median clone pseudotime", 0)

    ax = fig.add_subplot(outer[1, 0:2])
    plot_clone_summary(ax, clone_bins, clone_tests, "participant_median_span", ["2", "3-4", ">=5"],
                       "Larger clones span more of the continuum", "D", "Participant-median within-clone span", 1)

    # E. Program kinetics, with participant-balanced uncertainty.
    container = fig.add_subplot(outer[1, 2:6]); container.set_axis_off()
    container.text(-.07, 1.15, "E", transform=container.transAxes, fontsize=12, fontweight="bold", va="top")
    container.text(0, 1.15, "Expansion-associated programs across pseudotime", transform=container.transAxes,
                   fontsize=8, fontweight="bold", va="top")
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, 2:6], wspace=.28)
    for j, module in enumerate(MODULES):
        ax = fig.add_subplot(sub[0, j])
        dm = kinetics_display[kinetics_display.module.eq(module)]
        for status in ["Singleton", "Expanded"]:
            z = dm[(dm.clone_status.eq(status)) & (dm.n_participants >= 8)].sort_values("pt_mid")
            ax.fill_between(z.pt_mid.to_numpy(float), z.ci_low.to_numpy(float), z.ci_high.to_numpy(float),
                            color=COL[status], alpha=.12, linewidth=0)
            ax.plot(z.pt_mid, z.median_score_z, color=COL[status], lw=1.25,
                    marker="s" if status == "Singleton" else "o", ms=2.8, label=status)
        test = kinetics_tests[kinetics_tests.module.eq(module)].iloc[0]
        ax.text(.03, .98, MODULE_LABELS[module], transform=ax.transAxes, ha="left", va="top", fontsize=6.4, fontweight="bold")
        ax.text(.03, .88, f"trajectory {q_text(test.trajectory_FDR)}\nexpansion×trajectory {q_text(test.expansion_by_trajectory_FDR)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=4.7, color=COL["muted"])
        ax.axhline(0, color="#AAAAAA", lw=.55); ax.set_xlim(0, 1)
        ax.set_xlabel("Graph-geodesic pseudotime")
        if j == 0: ax.set_ylabel("Participant-level module score (z)")
        else: ax.tick_params(axis="y", labelleft=False)
        style_axis(ax, "y")
    ax.legend(frameon=False, loc="lower right")

    # F. Existing state-matched CD4/CD8 dose response retained as orthogonal validation.
    container = fig.add_subplot(outer[2, :]); container.set_axis_off()
    container.text(-.035, 1.15, "F", transform=container.transAxes, fontsize=12, fontweight="bold", va="top")
    container.text(0, 1.15, "State-matched clone-size dose response across CD4 and CD8 compartments", transform=container.transAxes,
                   fontsize=8, fontweight="bold", va="top")
    handles = [Line2D([0], [0], color=COL["ink"], lw=1.25, marker="D", ms=3, markerfacecolor="white", label="Pooled")] + [
        Line2D([0], [0], color=COL[d], lw=.8, marker="o", ms=2.5, label=d) for d in DIAG]
    container.legend(handles=handles, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.18), ncol=4,
                     handlelength=1, columnspacing=.65, handletextpad=.25)
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2, :], wspace=.35)
    bins = ["Singleton", "2 cells", "3-4 cells", ">=5 cells"]
    x = np.arange(4)
    for j, (module, label) in enumerate(ORIGINAL_MODULES.items()):
        ax = fig.add_subplot(sub[0, j]); dm = dose[dose.module.eq(module)]
        for diagnosis in DIAG:
            z = dm[dm.Diagnosis.eq(diagnosis)].set_index("clone_bin").reindex(bins)
            ax.plot(x, z["median"], color=COL[diagnosis], lw=.8, marker="o", ms=2.4, alpha=.72)
        pooled = dm[dm.Diagnosis.eq("Pooled")].set_index("clone_bin").reindex(bins)
        ax.errorbar(x, pooled["median"], yerr=[pooled["median"]-pooled.ci_low, pooled.ci_high-pooled["median"]],
                    fmt="D-", ms=3.3, lw=1.2, color=COL["ink"], markerfacecolor="white", capsize=1.5)
        trend = trends[(trends.module.eq(module)) & (trends.contrast.eq("Pooled ordered trend"))].iloc[0]
        ax.text(.02, .98, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.4, fontweight="bold")
        ax.text(.98, .05, f"slope {trend.estimate:+.3f}/bin\n95% CI {trend.ci_low:.3f}–{trend.ci_high:.3f}; {q_text(trend.FDR)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=COL["muted"])
        ax.axhline(0, color="#AAAAAA", lw=.55, ls="--")
        ax.set_xticks(x, ["1", "2", "3–4", "≥5"]); ax.set_xlabel("Cells per clonotype")
        if j == 0: ax.set_ylabel("State-matched score delta")
        else: ax.tick_params(axis="y", labelleft=False)
        style_axis(ax, "y")

    save_figure(fig, "Figure_2_trajectory_integrated", MAIN)


def plot_validation_supplement(primary, states, clone_bins, clone_tests):
    sling_cells = pd.read_csv(TRAJ / "Table_ST2_cd8_cells_with_slingshot_pseudotime.csv.gz", low_memory=False)
    sling_states = pd.read_csv(TRAJ / "Table_ST3_slingshot_state_pseudotime_summary.csv")
    sling_bins = pd.read_csv(TRAJ / "Table_ST5_slingshot_participant_clone_bins.csv")
    sling_tests = pd.read_csv(TRAJ / "Table_ST7_slingshot_clone_tests.csv")
    lineages = pd.read_csv(TRAJ / "Table_ST1_cd8_slingshot_lineages.csv")
    root_qc = pd.read_csv(TRAJ / "Table_TJ5_trajectory_qc.csv")

    fig = plt.figure(figsize=(7.48, 7.55), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.02, .98], hspace=.58, wspace=.68,
                  left=.08, right=.985, top=.97, bottom=.08)

    ax = fig.add_subplot(gs[0, 0])
    show = sling_cells[sling_cells.state.ne("CD8 Trm")].sample(12000, random_state=20260828).sort_values("pseudotime")
    sc = ax.scatter(show.UMAP_1, show.UMAP_2, c=show.pseudotime, cmap="viridis", s=.8, alpha=.68, linewidth=0, rasterized=True)
    centers = sling_cells[sling_cells.state.ne("CD8 Trm")].groupby("state")[["UMAP_1", "UMAP_2"]].median()
    for path in lineages.cluster_order:
        names = [x.strip() for x in path.split("->")]
        xy = centers.loc[names].to_numpy()
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=2.5, alpha=.85, zorder=4)
        ax.plot(xy[:, 0], xy[:, 1], color="#30383E", lw=.7, alpha=.75, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("CD8 UMAP 1"); ax.set_ylabel("CD8 UMAP 2")
    ax.set_title("Orthogonal Slingshot lineages", loc="left", fontweight="bold")
    cb=fig.colorbar(sc,ax=ax,orientation="horizontal",fraction=.05,pad=.07); cb.set_label("Lineage-assigned pseudotime",labelpad=1); cb.ax.tick_params(labelsize=5.5,pad=1)
    panel_label(ax,"A",x=-.20,y=1.08)

    ax = fig.add_subplot(gs[0, 1])
    merged = clone_bins.merge(sling_bins, on=["SampleID","Diagnosis1","clone_bin"], suffixes=("_graph","_sling"))
    rho = spearmanr(merged.participant_median_pseudotime_graph, merged.participant_median_pseudotime_sling).statistic
    for bin_name, marker in zip(["1","2","3-4",">=5"],["o","s","^","D"]):
        z=merged[merged.clone_bin.eq(bin_name)]
        ax.scatter(z.participant_median_pseudotime_graph,z.participant_median_pseudotime_sling,s=11,alpha=.42,marker=marker,label=bin_name,edgecolor="none")
    limits=[0,1]; ax.plot(limits,limits,color=COL["muted"],lw=.7,ls="--"); ax.set_xlim(limits); ax.set_ylim(limits)
    ax.text(.03,.97,f"Spearman ρ={rho:.2f}; n={len(merged)} participant-bin pairs",transform=ax.transAxes,va="top",fontsize=5.5)
    ax.set_xlabel("Graph-geodesic participant median"); ax.set_ylabel("Slingshot participant median")
    ax.set_title("Participant-level method concordance",loc="left",fontweight="bold"); ax.legend(frameon=False,ncol=2,loc="lower right",title="Clone size")
    style_axis(ax,"both"); panel_label(ax,"B",x=-.22,y=1.08)

    ax = fig.add_subplot(gs[0, 2])
    compare=[]
    for method,table in [("Graph-geodesic",clone_tests),("Slingshot",sling_tests)]:
        for _,row in table.iterrows():
            compare.append({"method":method,"analysis":row.analysis,"estimate":row.estimate,"lo":row.ci_low,"hi":row.ci_high})
    compare=pd.DataFrame(compare)
    labels=["Clone position / doubling","Clone span / doubling","Expanded − singleton"]
    analyses=clone_tests.analysis.tolist(); y=np.arange(3)[::-1]
    for offset,(method,color) in zip([-.10,.10],[("Graph-geodesic",COL["graph"]),("Slingshot",COL["slingshot"]) ]):
        z=compare[compare.method.eq(method)].set_index("analysis").reindex(analyses)
        ax.hlines(y+offset,z.lo,z.hi,color=color,lw=1); ax.scatter(z.estimate,y+offset,s=20,color=color,edgecolor="white",lw=.35,label=method,zorder=3)
    ax.axvline(0,color=COL["muted"],lw=.6,ls="--"); ax.set_yticks(y,labels); ax.set_xlabel("Pseudotime effect (95% CI)")
    ax.set_title("Clone effects replicate across methods",loc="left",fontweight="bold"); ax.legend(frameon=False,loc="lower right")
    style_axis(ax,"x"); panel_label(ax,"C",x=-.26,y=1.08)

    ax = fig.add_subplot(gs[1, 0])
    merged_states=states[["state","median_pseudotime"]].merge(sling_states[["state","median_pseudotime"]],on="state",suffixes=("_graph","_sling"))
    order=merged_states.sort_values("median_pseudotime_graph").state.tolist(); y=np.arange(len(order))[::-1]
    z=merged_states.set_index("state").loc[order]
    for yi,(_,row) in zip(y,z.iterrows()):
        ax.plot([row.median_pseudotime_graph,row.median_pseudotime_sling],[yi,yi],color="#B8B8B8",lw=.8)
        ax.scatter(row.median_pseudotime_graph,yi,s=19,color=COL["graph"],zorder=3)
        ax.scatter(row.median_pseudotime_sling,yi,s=19,color=COL["slingshot"],marker="s",zorder=3)
    ax.set_yticks(y,[STATE_SHORT[s] for s in order]); ax.set_xlim(0,1); ax.set_xlabel("Median pseudotime")
    ax.set_title("Branching changes scalar state positions",loc="left",fontweight="bold"); style_axis(ax,"x"); panel_label(ax,"D",x=-.24,y=1.08)

    ax = fig.add_subplot(gs[1, 1])
    metrics=["Alternative-root pseudotime Spearman rho","Median pairwise rho across 12 naive roots","Minimum pairwise rho across 12 naive roots"]
    q=root_qc.set_index("metric").loc[metrics]
    yy=np.arange(3)[::-1]; ax.hlines(yy,0,q.value.astype(float),color="#B8B8B8",lw=2); ax.scatter(q.value.astype(float),yy,s=28,color=COL["graph"])
    ax.set_yticks(yy,["Alternative root","Median across 12 roots","Minimum across roots"]); ax.set_xlim(0,1); ax.set_xlabel("Spearman ρ")
    ax.set_title("Graph trajectory is root-robust",loc="left",fontweight="bold"); style_axis(ax,"x"); panel_label(ax,"E",x=-.25,y=1.08)

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0,1,"Interpretive boundary",fontweight="bold",va="top")
    ax.text(0,.84,"• Graph and Slingshot reproduce later clone\n  position and broader within-clone span.\n\n• Cell-level scalar pseudotime differs because the\n  CD8 manifold is branched.\n\n• Trm is therefore displayed as a branch and is\n  excluded from scalar-continuum effect estimates.\n\n• Cross-sectional pseudotime does not establish\n  temporal progression.",va="top",fontsize=6.7,linespacing=1.35)
    panel_label(ax,"F",x=-.14,y=1.08)

    merged.to_csv(SRC / "FigureS_TCR_method_participant_concordance.csv",index=False)
    compare.to_csv(SRC / "FigureS_TCR_method_clone_effects.csv",index=False)
    merged_states.to_csv(SRC / "FigureS_TCR_method_state_positions.csv",index=False)
    save_figure(fig,"Figure_S_TCR_trajectory_validation",SUPP)


def plot_expansion_robustness_supplement():
    state=pd.read_csv(ORIGINAL/"Figure2_state_enrichment_displayed.csv")
    deltas=pd.read_csv(ORIGINAL/"Figure2_participant_program_deltas_displayed.csv")
    tests=pd.read_csv(ORIGINAL/"Figure2_paired_program_delta_tests.csv")
    cons=pd.read_csv(ORIGINAL/"Figure2_diagnosis_conservation_displayed.csv")
    replication=pd.read_csv(ORIGINAL/"Figure2_series_replication_displayed.csv")

    fig=plt.figure(figsize=(7.48,6.9),facecolor="white")
    gs=GridSpec(2,2,figure=fig,hspace=.56,wspace=.58,left=.10,right=.98,top=.965,bottom=.085)

    ax=fig.add_subplot(gs[0,0]); z=state.sort_values("adjusted_log2_or"); y=np.arange(len(z))
    colors=[COL["positive"] if r.FDR<.05 and r.adjusted_log2_or>0 else COL["negative"] if r.FDR<.05 else "#A8A8A8" for r in z.itertuples()]
    ax.hlines(y,z.ci_low,z.ci_high,color=colors,lw=1); ax.scatter(z.adjusted_log2_or,y,c=colors,s=20,edgecolor="white",lw=.35,zorder=3)
    ax.axvline(0,color=COL["muted"],lw=.6,ls="--"); ax.set_yticks(y,z.state); ax.set_xlabel("Adjusted expanded-versus-singleton log2 OR")
    ax.set_title("Depth- and batch-adjusted state enrichment",loc="left",fontweight="bold"); style_axis(ax,"x"); panel_label(ax,"A",x=-.24)

    ax=fig.add_subplot(gs[0,1]); module_order=list(ORIGINAL_MODULES); ybase=np.arange(3)[::-1]
    for yi,module in zip(ybase,module_order):
        d=deltas[deltas.module.eq(module)]
        violin=ax.violinplot(d.delta.dropna(),positions=[yi],vert=False,widths=.6,showextrema=False)
        for body in violin["bodies"]: body.set_facecolor("#D9D9D9"); body.set_edgecolor("none"); body.set_alpha(.65)
        for diagnosis in DIAG:
            q=d[d.Diagnosis1.eq(diagnosis)]; ax.scatter(q.delta,RNG.normal(yi,.07,len(q)),s=6,alpha=.4,color=COL[diagnosis],edgecolor="none")
        t=tests[tests.module.eq(module)].iloc[0]; ax.hlines(yi+.22,t.ci_low,t.ci_high,color=COL["ink"],lw=1.4); ax.scatter(t.median_delta,yi+.22,marker="D",s=24,color=COL["positive"],edgecolor="white",lw=.4)
        ax.text(.99,yi,f"median {t.median_delta:.2f}; {q_text(t.FDR)}",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=5.3,color=COL["muted"])
    ax.axvline(0,color=COL["muted"],lw=.6,ls="--"); ax.set_yticks(ybase,[ORIGINAL_MODULES[m] for m in module_order]); ax.set_xlabel("Participant delta: expanded − singleton")
    ax.set_title("Participant-level program shifts",loc="left",fontweight="bold"); style_axis(ax,"x"); panel_label(ax,"B",x=-.22)

    ax=fig.add_subplot(gs[1,0]);
    ax.errorbar(cons.CD_expansion_effect,cons.UC_expansion_effect,xerr=1.96*cons.CD_expansion_SE,yerr=1.96*cons.UC_expansion_SE,fmt="none",ecolor="#BBBBBB",elinewidth=.6)
    colors=[COL["positive"] if q<.05 and x+y>0 else COL["negative"] if q<.05 else "#A8A8A8" for x,y,q in zip(cons.CD_expansion_effect,cons.UC_expansion_effect,cons.FDR_global)]
    ax.scatter(cons.CD_expansion_effect,cons.UC_expansion_effect,c=colors,s=22,edgecolor="white",lw=.35,zorder=3)
    lim=[-.21,.21]; ax.plot(lim,lim,color=COL["muted"],lw=.7,ls="--"); ax.axhline(0,color=COL["grid"],lw=.5); ax.axvline(0,color=COL["grid"],lw=.5); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Adjusted expansion effect in CD"); ax.set_ylabel("Adjusted expansion effect in UC"); ax.set_title("Program effects are diagnosis-conserved",loc="left",fontweight="bold")
    style_axis(ax,"both"); panel_label(ax,"C",x=-.20)

    ax=fig.add_subplot(gs[1,1]); modules=["EOMES_ZEB2_inflammatory_CD8_TRM_like","Effector_cytotoxicity"]
    series=[s for s in ["S1","S2","S3","S4","S5","S7","S8","S10"]]; y=np.arange(len(series))[::-1]
    for offset,(module,color,label) in zip([-.11,.11],[(modules[0],COL["positive"],"EOMES–ZEB2"),(modules[1],"#CC79A7","Cytotoxicity")]):
        q=replication[(replication.module.eq(module)) & (replication.series.isin(series))].set_index("series").reindex(series)
        ax.hlines(y+offset,q.ci_low,q.ci_high,color=color,lw=.8); ax.scatter(q.effect,y+offset,color=color,s=15,label=label,zorder=3)
    ax.axvline(0,color=COL["muted"],lw=.6,ls="--"); ax.set_yticks(y,series); ax.set_xlabel("Mean expanded − singleton score (95% CI)")
    ax.set_title("Replication across acquisition series",loc="left",fontweight="bold"); ax.legend(frameon=False,loc="lower right")
    style_axis(ax,"x"); panel_label(ax,"D",x=-.18)

    save_figure(fig,"Figure_S_TCR_expansion_robustness",SUPP)


def write_legends():
    figure2=(
        "Figure 2. TCR clonal expansion tracks conventional CD8 differentiation and amplifies cytotoxic and inflammatory programs. "
        "(A) Conventional CD8 UMAP colored by consensus graph-geodesic pseudotime. The root was selected from central, high-early-memory CD8-naive cells, and the displayed skeleton summarizes the principal scalar continuum and the Trm branch. "
        "(B) Median and interquartile-range pseudotime positions for states included in the scalar continuum; Trm was analyzed as a branch and was not forced onto the scalar axis. "
        "(C) Participant-level median clone pseudotime across exact paired alpha-beta clone-size bins. (D) Participant-level median within-clone pseudotime span among expanded clonotypes. Diamonds and bars show pooled medians and participant-bootstrap 95% confidence intervals; annotations report participant-fixed-effect slopes per clone-size doubling with participant-clustered standard errors. "
        "(E) Participant-balanced EOMES-ZEB2, cytotoxicity, and Th1/Tc1 module kinetics for singleton and expanded exact paired alpha-beta clonotypes. Shading shows participant-bootstrap 95% confidence intervals; FDR values derive from participant-blocked spline models. "
        "(F) Orthogonal state-matched clone-size dose response across CD4 and CD8 compartments. Cross-sectional pseudotime represents transcriptional ordering and does not establish temporal progression."
    )
    validation=(
        "Figure S. Orthogonal validation of the CD8 trajectory-TCR integration. (A) Participant-balanced Slingshot lineages in SCVI space. "
        "(B) Concordance of participant-by-clone-size median pseudotime between graph-geodesic and Slingshot analyses. (C) Clone-position, clone-span, and expanded-minus-singleton effects across methods. "
        "(D) Method-dependent scalar state positions illustrate the branching structure. (E) Stability of graph-geodesic pseudotime across alternative naive roots. (F) Prespecified interpretive boundary and treatment of the Trm branch."
    )
    robustness=(
        "Figure S. Robustness and generalizability of expansion-associated T-cell programs. (A) Diagnosis-, acquisition-series-, and receptor-depth-adjusted state-occupancy effects. "
        "(B) Participant-level expanded-minus-singleton program shifts. (C) Concordance of adjusted expansion effects in Crohn's disease and ulcerative colitis. (D) Acquisition-series replication of EOMES-ZEB2 and cytotoxicity effects."
    )
    (LEG/"Figure_2_trajectory_integrated_legend.txt").write_text(figure2+"\n",encoding="utf-8")
    (LEG/"Figure_S_TCR_trajectory_validation_legend.txt").write_text(validation+"\n",encoding="utf-8")
    (LEG/"Figure_S_TCR_expansion_robustness_legend.txt").write_text(robustness+"\n",encoding="utf-8")


def main():
    cells,primary,states,clone_bins,clone_tests,kinetics_display,kinetics_tests=prepare_primary_graph_outputs()
    plot_main_figure(cells,primary,states,clone_bins,clone_tests,kinetics_display,kinetics_tests)
    plot_validation_supplement(primary,states,clone_bins,clone_tests)
    plot_expansion_robustness_supplement()
    write_legends()
    print(clone_tests.to_string(index=False))
    print(kinetics_tests.to_string(index=False))
    print(OUT)


if __name__=="__main__":
    main()

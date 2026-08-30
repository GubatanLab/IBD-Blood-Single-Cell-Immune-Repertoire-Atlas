from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rebuild_figure4_cellpress_enhanced as base
import rebuild_figure4_immunity as immunity
import build_consolidated_figure2 as figure2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Trajectory Integrated Figure Set"
MAIN = OUT / "Main Figures"
SRC = OUT / "Source Data"
LEG = OUT / "Legends"
TRAJ = ROOT / "High Impact Additional Analyses" / "BCR Trajectory Priority"

TEAL = "#00858A"
GOLD = "#E69F00"
GREEN = "#3A986B"
MAGENTA = "#B24C7C"
GREY = "#6F777F"


def trajectory_program_summary(program_data: pd.DataFrame, n_boot: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(20260828)
    rows = []
    for (program, pt_bin), z in program_data.groupby(["program", "pt_bin"], observed=True):
        vals = z["score"].dropna().to_numpy(float)
        if not len(vals):
            continue
        sampled = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        boots = vals[sampled].mean(axis=1)
        rows.append({
            "program": program,
            "pt_bin": pt_bin,
            "n_participants": len(vals),
            "mean": vals.mean(),
            "ci95_low": np.quantile(boots, .025),
            "ci95_high": np.quantile(boots, .975),
        })
    out = pd.DataFrame(rows)
    order = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    out["pt_bin"] = pd.Categorical(out["pt_bin"], order, ordered=True)
    return out.sort_values(["program", "pt_bin"])


def build():
    MAIN.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    LEG.mkdir(parents=True, exist_ok=True)

    state = base.state_threshold_analysis()
    programs = base.program_effects()
    programs = programs[~programs["module"].eq("BAFF_APRIL_survival_response")].copy()
    genes = base.representative_genes()
    iso_sample, iso_effects = base.corrected_isotypes()
    switched, switched_effects = base.switched_values_and_model()
    metrics = pd.read_csv(immunity.BCR_DIR / "clonality_metrics_IBDBCR_Immunarch.csv").rename(columns={"Diagnosis1": "Diagnosis"})
    expansion = pd.read_csv(immunity.BCR_DIR / "clonal_expansion_index_control_uc_cd_IBDBCR_values.csv").rename(columns={"Diagnosis1": "Diagnosis"})
    participant_metrics = metrics[["PatientID", "Diagnosis", "Clonality"]].merge(
        expansion[["PatientID", "ClonalExpansionIndex"]], on="PatientID", how="inner"
    )

    trajectory = pd.read_csv(TRAJ / "Table_BT1_balanced_cell_pseudotime.csv.gz", low_memory=False)
    lineages = pd.read_csv(TRAJ / "Table_BT1_slingshot_lineages.csv")
    program_data = pd.read_csv(TRAJ / "Table_BT4_participant_program_trends.csv")
    program_summary = trajectory_program_summary(program_data)
    switch_count_tests = pd.read_csv(SRC / "Figure4G_class_switching_rank_and_betabinomial.csv")
    dm_tests = pd.read_csv(SRC / "Figure4G_dirichlet_multinomial_composition_tests.csv")

    fig = plt.figure(figsize=(7.48, 8.65), facecolor="white")
    outer = GridSpec(
        3, 1, figure=fig, height_ratios=[1.0, 1.08, .94], hspace=.36,
        left=.075, right=.985, top=.978, bottom=.06
    )
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[.86, 1.28, 1.10], wspace=.86)
    middle = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[.72, 1.35, 1.02], wspace=.55)
    bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], width_ratios=[.86, .72, 1.42], wspace=.58)

    # A. Participant-level BCR repertoire metrics.
    agrid = GridSpecFromSubplotSpec(2, 1, subplot_spec=top[0, 0], hspace=.60)
    metric_stats = []
    for j, (value, label) in enumerate([
        ("Clonality", "BCR clonality"),
        ("ClonalExpansionIndex", "Clonal expansion index"),
    ]):
        ax = fig.add_subplot(agrid[j, 0])
        metric_stats.append(immunity.participant_panel(
            ax, participant_metrics, value, label, stats_y=1.045, stats_inline=True
        ))
        ax.set_title(label, loc="left", fontweight="bold", fontsize=7.2, pad=0, y=1.21)
        ax.tick_params(axis="x", labelsize=5.7)
        if j == 0:
            ax.tick_params(axis="x", labelbottom=False)
            base.panel_label(ax, "A", x=-.34, y=1.17)

    # B. State enrichment with clone-threshold sensitivity.
    ax = fig.add_subplot(top[0, 1])
    primary = state[state["threshold"].eq(2) & (state["n_participants"] >= 15)].sort_values("median_change_pp")
    order = primary["BcellState"].tolist()
    y = np.arange(len(order))
    for yi, (_, row) in zip(y, primary.iterrows()):
        color = base.UP if row["median_change_pp"] > 0 else base.DOWN
        shown = color if row["FDR_within_threshold"] < .05 else base.NS
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=shown, lw=1.05)
        ax.scatter(row["median_change_pp"], yi, s=25,
                   color=shown if row["FDR_within_threshold"] < .05 else "white",
                   edgecolor=shown, linewidth=.8, zorder=4)
    for threshold, offset, marker in [(3, -.13, "s"), (4, .13, "^")]:
        subset = state[state["threshold"].eq(threshold)].set_index("BcellState")
        for yi, state_name in enumerate(order):
            if state_name not in subset.index:
                continue
            row = subset.loc[state_name]
            color = base.UP if row["median_change_pp"] > 0 else base.DOWN
            ax.scatter(row["median_change_pp"], yi + offset, s=14, marker=marker,
                       facecolor="white", edgecolor=color, linewidth=.75, zorder=5)
    ax.axvline(0, color="#777777", lw=.65, ls="--")
    ax.set_yticks(y, [f"{name}\n n={int(n)}" for name, n in zip(primary.BcellState, primary.n_participants)])
    ax.set_xlabel("Median expanded − singleton fraction\n(percentage points)")
    ax.set_title("State enrichment", loc="left", fontweight="bold", pad=4)
    ax.scatter([], [], s=22, marker="o", color="#555555", label="≥2 cells (95% CI)")
    ax.scatter([], [], s=14, marker="s", facecolor="white", edgecolor="#555555", label="≥3 cells")
    ax.scatter([], [], s=14, marker="^", facecolor="white", edgecolor="#555555", label="≥4 cells")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(.01, .98), handletextpad=.35, labelspacing=.22)
    base.style_axis(ax, "x"); base.panel_label(ax, "B", x=-.31)

    # C. Clone-aware program effects.
    ax = fig.add_subplot(top[0, 2])
    program_display_short = {
        "Antibody secretion/UPR": "Antibody secretion/UPR",
        "IgA mucosal plasma cell": "IgA mucosal plasma",
        "Plasmablast/plasma differentiation": "Plasma differentiation",
        "IgG inflammatory plasma cell": "IgG inflammatory plasma",
        "Plasma-cell antigen presentation/UPR": "Antigen presentation/UPR",
        "BAFF/APRIL survival response": "BAFF/APRIL survival",
    }
    y = np.arange(len(programs))
    for yi, (_, row) in zip(y, programs.iterrows()):
        color = base.UP if row["direction"] == "Up" else base.DOWN
        significant = row["camera_global_FDR"] < .05
        shown = color if significant else base.NS
        ax.plot([row["q1_member_gene_logFC"], row["q3_member_gene_logFC"]], [yi, yi], color=shown, lw=1.15)
        ax.scatter(row["median_member_gene_logFC"], yi, s=27,
                   color=shown if significant else "white", edgecolor=shown, linewidth=.75, zorder=4)
        ax.text(.99, yi, base.q_label(row["camera_global_FDR"]), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=5.8, color="#555555")
    effect_low = min(0, programs["q1_member_gene_logFC"].min())
    effect_high = max(0, programs["q3_member_gene_logFC"].max())
    effect_span = max(effect_high - effect_low, .1)
    ax.set_xlim(effect_low - .06 * effect_span, effect_high + .46 * effect_span)
    ax.text(.99, 1.01, "FDR q", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.5, color="#555555")
    ax.axvline(0, color="#777777", lw=.65, ls="--")
    ax.set_yticks(y, programs["display"].map(lambda value: program_display_short.get(value, value)))
    ax.set_xlabel("Median member-gene log2FC (IQR)\nexpanded versus singleton")
    ax.set_title("Clone-aware programs", loc="left", fontweight="bold", pad=4)
    base.style_axis(ax, "x"); base.panel_label(ax, "C", x=-.34)

    # D. Representative genes.
    ax = fig.add_subplot(middle[0, 0])
    y = np.arange(len(genes))
    for yi, (_, row) in zip(y, genes.iterrows()):
        significant = row["FDR_global"] < .05
        color = (base.UP if row["logFC"] > 0 else base.DOWN) if significant else base.NS
        ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=color, lw=1)
        ax.scatter(row["logFC"], yi, s=23, color=color if significant else "white",
                   edgecolor=color, linewidth=.8, zorder=4)
    ax.axvline(0, color="#777777", lw=.65, ls="--")
    ax.set_yticks(y, genes.gene)
    ax.set_xlabel("Adjusted log2FC (95% CI)")
    ax.set_title("Representative genes", loc="left", fontweight="bold", pad=4)
    base.style_axis(ax, "x"); base.panel_label(ax, "D", x=-.40)

    # E. Participant-balanced Slingshot trajectory.
    ax = fig.add_subplot(middle[0, 1])
    show = trajectory.sample(min(11500, len(trajectory)), random_state=20260828).sort_values("trajectory_pseudotime")
    sc = ax.scatter(show.UMAP_1, show.UMAP_2, c=show.trajectory_pseudotime,
                    cmap=figure2.TRAJECTORY_CMAP, vmin=0, vmax=1,
                    s=1.25, alpha=.67, linewidth=0, rasterized=True)
    centers = trajectory.groupby("state")[["UMAP_1", "UMAP_2"]].median()
    for path in lineages.cluster_order:
        names = [x.strip() for x in path.split("->")]
        xy = centers.loc[names].to_numpy()
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=3.1, alpha=.88, zorder=4)
        ax.plot(xy[:, 0], xy[:, 1], color="#30383E", lw=.8, alpha=.8, zorder=5)
    ax.scatter(centers.UMAP_1, centers.UMAP_2, s=15, facecolor="white", edgecolor="#30383E", lw=.55, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Participant-balanced B-cell Slingshot trajectory", loc="left", fontweight="bold", pad=4)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=.06, pad=.08, aspect=28)
    cb.set_label("Normalized Slingshot pseudotime", labelpad=2)
    cb.ax.tick_params(labelsize=6, pad=1)
    base.panel_label(ax, "E", x=-.16, y=1.08)

    # F. Participant-balanced effector-program trends.
    ax = fig.add_subplot(middle[0, 2])
    colors = {"IgA mucosal": GREEN, "Plasma differentiation": GOLD, "Antibody secretion/UPR": MAGENTA}
    for program in ["IgA mucosal", "Plasma differentiation", "Antibody secretion/UPR"]:
        z = program_summary[program_summary.program.eq(program)].sort_values("pt_bin")
        x = np.arange(len(z))
        ax.fill_between(x, z.ci95_low, z.ci95_high, color=colors[program], alpha=.14, linewidth=0)
        ax.plot(x, z["mean"], marker="o", ms=3, lw=1.35, color=colors[program], label=program)
    ax.axhline(0, color="#888888", lw=.6, ls="--")
    ax.set_xticks(np.arange(5), ["0", ".2", ".4", ".6", ".8–1"])
    ax.set_xlabel("Slingshot pseudotime interval")
    ax.set_ylabel("Participant-mean module score")
    # Add headroom so the program key occupies a dedicated non-data band.
    # Keeping the key inside the axes avoids collision with panel C above.
    ax.set_ylim(-0.65, 0.82)
    ax.set_title("Effector programs across pseudotime", loc="left", fontweight="bold", pad=4)
    ax.legend(
        frameon=False, loc="upper left", ncol=1,
        fontsize=5.2, handlelength=1.25, handletextpad=.35,
        labelspacing=.18, borderaxespad=.2,
    )
    base.style_axis(ax, "y"); base.panel_label(ax, "F", x=-.22, y=1.08)

    # G1. Observed participant-level class switching.
    ax = fig.add_subplot(bottom[0, 0])
    base.raw_boxstrip(ax, switched, "Diagnosis", "switched_fraction")
    ax.set_ylabel("Class-switched fraction")
    ax.set_title("Observed switching", loc="left", fontweight="bold", pad=4)
    base.style_axis(ax, "y"); base.panel_label(ax, "G", x=-.32, y=1.10)

    # G2. Covariate-adjusted beta-binomial switching effects.
    ax = fig.add_subplot(bottom[0, 1])
    cd_switch = switch_count_tests[switch_count_tests.diagnosis.eq("CD")].iloc[0]
    uc_switch = switch_count_tests[switch_count_tests.diagnosis.eq("UC")].iloc[0]
    for yi, row in enumerate([cd_switch, uc_switch]):
        diagnosis = row.diagnosis
        color = base.DIAG_COLORS[diagnosis]
        ax.plot([row.OR_ci95_low, row.OR_ci95_high], [yi, yi], color=color, lw=1.15)
        ax.scatter(row.adjusted_odds_ratio, yi, s=29, facecolor="white", edgecolor=color, lw=.9, zorder=4)
        ax.text(.98, yi, base.q_label(row.Holm_q), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=5.5, color="#555555",
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":.78, "pad":.5})
    ax.axvline(1, color="#777777", lw=.65, ls="--")
    ax.set_xscale("log"); ax.set_xlim(.72, 2.28)
    ax.set_xticks([.75, 1, 1.5, 2], ["0.75", "1", "1.5", "2"])
    ax.set_yticks([0, 1], ["CD", "UC"]); ax.set_ylim(1.55, -.55)
    ax.set_xlabel("Adjusted odds ratio\n(95% CI)")
    ax.set_title("Adjusted switching", loc="left", fontweight="bold", pad=4)
    base.style_axis(ax, "x")

    # G3. Adjusted compositional effects.
    dm_global = dm_tests[dm_tests.test.eq("Global diagnosis effect")].iloc[0]
    dm_cd = dm_tests[dm_tests.comparison.eq("CD vs Control")].iloc[0]
    dm_uc = dm_tests[dm_tests.comparison.eq("UC vs Control")].iloc[0]
    ax = fig.add_subplot(bottom[0, 2])
    offsets = {"CD": -.13, "UC": .13}
    for i, isotype in enumerate(["IgG", "IgA", "IgD", "IgM"]):
        for diagnosis in ["CD", "UC"]:
            row = iso_effects[(iso_effects.isotype.eq(isotype)) & (iso_effects.diagnosis.eq(diagnosis))].iloc[0]
            ypos = i + offsets[diagnosis]
            significant = row.FDR_global < .05
            color = base.DIAG_COLORS[diagnosis]
            ax.plot([row.ci95_low, row.ci95_high], [ypos, ypos], color=color, lw=.95, alpha=.9)
            ax.scatter(row.adjusted_CLR_effect, ypos, s=20, color=color if significant else "white",
                       edgecolor=color, linewidth=.8, zorder=4)
    for diagnosis in ["CD", "UC"]:
        ax.scatter([], [], s=20, facecolor="white", edgecolor=base.DIAG_COLORS[diagnosis],
                   linewidth=.8, label=f"{diagnosis} vs Control")
    ax.axvline(0, color="#777777", lw=.65, ls="--")
    ax.set_yticks(np.arange(4), ["IgG", "IgA", "IgD", "IgM"]); ax.invert_yaxis()
    ax.set_ylim(3.35, -.62)
    ax.set_xlabel("Adjusted centered-log-ratio effect\nversus Control (95% CI)")
    ax.set_title("Adjusted isotype composition", loc="left", fontweight="bold", pad=13)
    ax.text(0, 1.015, f"DM global P={dm_global.p_value:.3f}; CD q={dm_cd.Holm_q:.3f}; UC q={dm_uc.Holm_q:.3f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color="#555555")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(.01, .99), ncol=2,
              columnspacing=.65, handletextpad=.3, borderaxespad=0)
    base.style_axis(ax, "x")

    stem = "Figure_4_trajectory_harmonized"
    fig.savefig(MAIN / f"{stem}.pdf", bbox_inches="tight", pad_inches=.035)
    fig.savefig(MAIN / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=.035)
    fig.savefig(MAIN / f"{stem}.tif", dpi=500, pil_kwargs={"compression":"tiff_lzw"}, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)

    trajectory[["cell_id", "PatientID", "Diagnosis1", "state", "trajectory_pseudotime", "assigned_branch"]].to_csv(
        SRC / "Figure4_harmonized_cell_pseudotime.csv.gz", index=False)
    program_summary.to_csv(SRC / "Figure4_harmonized_program_trends.csv", index=False)
    participant_metrics.to_csv(SRC / "Figure4_harmonized_participant_BCR_metrics.csv", index=False)
    pd.concat(metric_stats, ignore_index=True).to_csv(
        SRC / "Figure4_harmonized_participant_BCR_metric_tests.csv", index=False
    )

    legend = (
        "Figure 4. BCR repertoire expansion is associated with antibody-secreting programs organized along a branching B-cell-state trajectory. "
        "(A) Participant-level BCR clonality and clonal-expansion index in Control, Crohn's disease (CD), and ulcerative colitis (UC). Points denote "
        "participants; boxes show medians and interquartile ranges, and whiskers extend to 1.5 times the interquartile range. Two-sided Mann–Whitney "
        "tests compare each disease group with Control; q values are Benjamini–Hochberg adjusted within each metric. "
        "(B) Participant-level enrichment of B-cell states among expanded relative to singleton clonotypes. Circles show the primary "
        "expansion definition (clonotype size ≥2 cells) with participant-bootstrap 95% confidence intervals; squares and triangles show "
        "sensitivity analyses using thresholds of ≥3 and ≥4 cells. Filled primary-analysis points indicate Benjamini–Hochberg FDR <0.05. "
        "(C) Effect-size summary for antibody- and plasma-cell programs passing global cameraPR FDR <0.05 in the state-matched, participant-paired pseudobulk "
        "analysis. Points and lines show the median and interquartile range of adjusted member-gene log2 fold changes. "
        "(D) Representative adjusted gene-level effects from the same pseudobulk model. Points and lines show log2 fold changes and 95% confidence "
        "intervals; filled points indicate global FDR <0.05. (E) Slingshot branching trajectory fitted in ten-dimensional SCVI latent space using a "
        "participant-by-state-balanced reference of 14,868 B cells from 249 participants. Cells are displayed in UMAP space and colored by normalized "
        "Slingshot pseudotime using the same low-to-high purple–blue–teal–gold–coral scale as the CD8 trajectory in Figure 2C; lines connect lineage-ordered state centroids for visualization. Transitional B cells were specified as the root, with atypical-memory "
        "and IgM-, IgA-, and IgG-plasma-cell terminal constraints. (F) Participant-balanced mean IgA-mucosal, plasma-differentiation, and antibody-secretion/UPR "
        "module scores across pseudotime intervals. Shading denotes participant-bootstrap 95% confidence intervals. (G) Participant-level class-switched BCR "
        "fractions, covariate-adjusted beta-binomial class-switching odds ratios, and covariate-adjusted centered-log-ratio isotype effects versus Control. The "
        "beta-binomial model used switched and unswitched counts and adjusted for age, sex, and log-transformed BCR depth; q values are Holm adjusted across the "
        "two diagnosis contrasts. Isotype composition was tested jointly by Dirichlet–multinomial regression with the same covariates; the global likelihood-ratio "
        "P value and Holm-adjusted diagnosis-specific joint-composition q values are shown once above the isotype forest. Individual isotype effects are displayed "
        "with diagnosis-colored 95% confidence intervals; none passed global Benjamini–Hochberg FDR correction across the eight centered-log-ratio contrasts. Cross-sectional "
        "pseudotime represents transcriptional ordering and does not establish temporal progression. BCR, B cell receptor; CD, Crohn's disease; CLR, centered "
        "log ratio; FDR, false discovery rate; UC, ulcerative colitis; UPR, unfolded protein response."
    )
    (LEG / f"{stem}_legend.txt").write_text(legend + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()

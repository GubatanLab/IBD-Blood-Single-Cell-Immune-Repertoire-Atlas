from __future__ import annotations

from pathlib import Path
from shutil import copy2
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import rankdata, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "Cell Press Redrawn Figure Set"
FINAL = ROOT / "Final 7-Figure Manuscript Set"
HI = ROOT / "High Impact Additional Analyses"
RA = HI / "Literature Guided Ranked Analyses"
BT = HI / "BCR Trajectory Priority"
TB = HI / "Th17 Treg B Helper Analyses"
PREVIEW = CANON / "Preview Alternatives" / "Immunity High-Impact Figures 2-6"
PNG_DIR = PREVIEW / "Figures"
LEG_DIR = PREVIEW / "Legends"
SRC_DIR = PREVIEW / "Source Data"
PDF_DIR = ROOT / "output" / "pdf" / "Immunity High-Impact Preview Figures 2-6"

for directory in (PNG_DIR, LEG_DIR, SRC_DIR, PDF_DIR):
    directory.mkdir(parents=True, exist_ok=True)


COL = {
    "Control": "#7A7A7A",
    "CD": "#167BB5",
    "UC": "#D95F02",
    "Tph": "#2A7FB8",
    "Tfh": "#7651A8",
    "Tph/Tfh help": "#2A7FB8",
    "Pathogenic Th17": "#D95F02",
    "Suppressive Treg": "#1B9E77",
    "IgA": "#E69F00",
    "IgG": "#CC79A7",
    "IgM": "#3268A8",
    "IgD": "#6C757D",
    "plasma": "#B64B4B",
}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.2,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.7,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.4,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="both", color="#D9D9D9", linewidth=0.45, alpha=0.8)
        ax.set_axisbelow(True)


def panel(ax, label, title):
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)


def fdr_mark(q):
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def save_figure(fig, number):
    pdf = PDF_DIR / f"Figure_{number}_immunity_preview.pdf"
    png = PNG_DIR / f"Figure_{number}_immunity_preview.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    return pdf, png


def clustered_slope(data, outcome, diagnosis):
    work = data[(data["Diagnosis1"] == diagnosis) & data[outcome].notna() &
                data["log2_clone_size"].notna()].copy()
    model = smf.ols(f"{outcome} ~ log2_clone_size", work).fit(
        cov_type="cluster", cov_kwds={"groups": work["SampleID"]}
    )
    ci = model.conf_int().loc["log2_clone_size"]
    return {
        "outcome": outcome,
        "Diagnosis": diagnosis,
        "estimate": model.params["log2_clone_size"],
        "ci_low": ci.iloc[0],
        "ci_high": ci.iloc[1],
        "p_value": model.pvalues["log2_clone_size"],
        "n_clonotypes": len(work),
        "n_participants": work["SampleID"].nunique(),
    }


def build_figure2():
    graph_path = FINAL / "Source Data" / "Figure2_primary_graph_cells.csv.gz"
    clone_path = FINAL / "Source Data" / "Figure2_primary_clone_trajectory.csv"
    dose_path = FINAL / "Source Data" / "Figure2_consolidated_clone_size_dose_response.csv"
    kinetics_path = FINAL / "Source Data" / "Figure2_primary_program_kinetics_display.csv"

    graph = pd.read_csv(graph_path)
    clones = pd.read_csv(clone_path)
    dose = pd.read_csv(dose_path)
    kinetics = pd.read_csv(kinetics_path)

    slope_rows = []
    for outcome in ("median_pseudotime", "pseudotime_span"):
        for diagnosis in ("Control", "CD", "UC"):
            slope_rows.append(clustered_slope(clones, outcome, diagnosis))
    slopes = pd.DataFrame(slope_rows)

    interaction_rows = []
    for outcome in ("median_pseudotime", "pseudotime_span"):
        work = clones[clones[outcome].notna() & clones["log2_clone_size"].notna()].copy()
        work["Diagnosis1"] = pd.Categorical(work["Diagnosis1"], ["CD", "Control", "UC"])
        model = smf.ols(f"{outcome} ~ log2_clone_size*C(Diagnosis1)", work).fit(
            cov_type="cluster", cov_kwds={"groups": work["SampleID"]}
        )
        for diagnosis, term in (
            ("Control vs CD", "log2_clone_size:C(Diagnosis1)[T.Control]"),
            ("UC vs CD", "log2_clone_size:C(Diagnosis1)[T.UC]"),
        ):
            ci = model.conf_int().loc[term]
            interaction_rows.append({
                "outcome": outcome, "contrast": diagnosis,
                "interaction_estimate": model.params[term],
                "ci_low": ci.iloc[0], "ci_high": ci.iloc[1],
                "p_value": model.pvalues[term],
            })
    interactions = pd.DataFrame(interaction_rows)
    slopes.to_csv(SRC_DIR / "Figure2_diagnosis_specific_clustered_clone_slopes.csv", index=False)
    interactions.to_csv(SRC_DIR / "Figure2_clone_size_by_diagnosis_interactions.csv", index=False)

    fig = plt.figure(figsize=(7.05, 8.25))
    fig.suptitle("Clonal expansion tracks a shared cytotoxic differentiation continuum",
                 fontsize=11.5, fontweight="bold", y=0.982)
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.08, 0.82, 0.92],
                     width_ratios=[1.03, 1.25], hspace=0.52, wspace=0.35,
                     left=0.105, right=0.985, bottom=0.07, top=0.93)

    ax_a = fig.add_subplot(outer[0, 0])
    rng = np.random.default_rng(20260829)
    ref = graph[graph["trajectory_reference"].fillna(False)].copy()
    if len(ref) > 16000:
        ref = ref.iloc[rng.choice(len(ref), 16000, replace=False)]
    sc = ax_a.scatter(ref["UMAP_1"], ref["UMAP_2"], c=ref["pseudotime"],
                      s=1.1, cmap="viridis", rasterized=True, alpha=0.78, linewidths=0)
    ax_a.set_xlabel("CD8 UMAP 1")
    ax_a.set_ylabel("CD8 UMAP 2")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    clean_axes(ax_a, grid=False)
    panel(ax_a, "A", "Rooted CD8 differentiation trajectory")
    cb = fig.colorbar(sc, ax=ax_a, orientation="horizontal", fraction=0.06, pad=0.08)
    cb.set_label("Graph-geodesic pseudotime")

    gs_b = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 1], wspace=0.38)
    module_map = {
        "EOMES_ZEB2_inflammatory_CD8_TRM_like": "EOMES-ZEB2",
        "Cytotoxicity": "Cytotoxicity",
        "Th1_Tc1": "Th1/Tc1",
    }
    bin_order = ["Singleton", "2 cells", "3-4 cells", ">=5 cells"]
    diag_order = ["Control", "CD", "UC"]
    legend_handles = legend_labels = None
    for j, (module, label) in enumerate(module_map.items()):
        ax = fig.add_subplot(gs_b[0, j])
        sub = dose[dose["module"] == module].copy()
        for diagnosis in diag_order:
            ss = sub[sub["Diagnosis"] == diagnosis].set_index("clone_bin").reindex(bin_order)
            x = np.arange(len(bin_order))
            ax.plot(x, ss["median"], marker="o", ms=3.2, color=COL[diagnosis], label=diagnosis)
            ax.fill_between(x, ss["ci_low"].astype(float), ss["ci_high"].astype(float),
                            color=COL[diagnosis], alpha=0.10, linewidth=0)
        ax.set_xticks(np.arange(4), ["1", "2", "3-4", ">=5"])
        ax.set_xlabel("Cells per clone")
        if j == 0:
            ax.set_ylabel("State-matched score delta")
            panel(ax, "B", "Clone-size dose response")
        else:
            ax.set_yticklabels([])
        ax.set_title(label, loc="left", fontweight="bold", fontsize=7.7, pad=3)
        clean_axes(ax)
        if j == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if j == 2:
            ax.legend(legend_handles, legend_labels, frameon=False, loc="upper left",
                      fontsize=5.5, handlelength=1.0, borderaxespad=0.25)

    ax_c = fig.add_subplot(outer[1, :])
    y_positions = []
    labels = []
    y = 0
    metric_labels = {
        "median_pseudotime": "Trajectory position",
        "pseudotime_span": "Within-clone span",
    }
    for outcome in ("median_pseudotime", "pseudotime_span"):
        for diagnosis in diag_order:
            row = slopes[(slopes.outcome == outcome) & (slopes.Diagnosis == diagnosis)].iloc[0]
            ax_c.errorbar(row.estimate, y, xerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]],
                          fmt="o", ms=4.2, capsize=2, color=COL[diagnosis], lw=1.2)
            y_positions.append(y)
            short_diag = "Ctl" if diagnosis == "Control" else diagnosis
            short_metric = "pos" if outcome == "median_pseudotime" else "span"
            labels.append(f"{short_diag} {short_metric} (n={int(row.n_participants)})")
            y += 1
        y += 0.55
    ax_c.axvline(0, color="black", lw=0.7)
    ax_c.set_yticks(y_positions, labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Change per clone-size doubling (cluster-robust 95% CI)")
    clean_axes(ax_c)
    panel(ax_c, "C", "Clone-size effects are conserved across diagnoses")
    p_uc_pt = interactions[(interactions.outcome == "median_pseudotime") &
                           (interactions.contrast == "UC vs CD")].p_value.iloc[0]
    p_uc_span = interactions[(interactions.outcome == "pseudotime_span") &
                             (interactions.contrast == "UC vs CD")].p_value.iloc[0]
    ax_c.text(0.99, 0.97, f"UC-CD interaction: position P={p_uc_pt:.2f}; span P={p_uc_span:.2f}",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=6.5,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2})

    gs_d = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2, :], wspace=0.38)
    kin_order = ["EOMES_ZEB2", "Cytotoxicity", "Th1_Tc1"]
    kin_labels = ["EOMES-ZEB2", "Cytotoxicity", "Th1/Tc1"]
    for j, (module, title) in enumerate(zip(kin_order, kin_labels)):
        ax = fig.add_subplot(gs_d[0, j])
        sub = kinetics[kinetics.module == module]
        for status, color, marker in (("Singleton", "#666666", "s"), ("Expanded", COL["UC"], "o")):
            ss = sub[sub.clone_status == status].sort_values("pt_mid")
            ax.plot(ss.pt_mid, ss.median_score_z, color=color, marker=marker, ms=3.1, label=status)
            ax.fill_between(ss.pt_mid.astype(float), ss.ci_low.astype(float), ss.ci_high.astype(float),
                            color=color, alpha=0.12, linewidth=0)
        ax.axhline(0, color="#777777", lw=0.55)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Pseudotime")
        if j == 0:
            ax.set_ylabel("Participant-level module score (z)")
            panel(ax, "D", "Expansion-associated program kinetics")
        else:
            ax.set_yticklabels([])
        ax.set_title(title, loc="left", fontsize=7.8, fontweight="bold", pad=3)
        clean_axes(ax)
    ax.legend(frameon=False, loc="lower right")

    save_figure(fig, 2)
    (LEG_DIR / "Figure_2_immunity_preview_legend.txt").write_text(
        "Figure 2. Clonal expansion tracks a shared cytotoxic differentiation continuum. "
        "(A) Participant-balanced CD8 trajectory. (B) State-matched module changes across exact paired alpha-beta clone-size bins. "
        "(C) Diagnosis-specific cluster-robust clone-size slopes, using participant as the clustering unit; formal interactions do not support diagnosis-specific slopes. "
        "(D) Participant-level program kinetics for singleton and expanded exact paired clones. Preview only; canonical Figure 2 was not changed.\n",
        encoding="utf-8"
    )


def bootstrap_summary(data, value, group_cols, draws=1000, seed=20260829):
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in data.groupby(group_cols, observed=True):
        vals = group[value].dropna().to_numpy(float)
        med = float(np.median(vals))
        boots = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(draws)] if len(vals) else [np.nan]
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row.update({"median": med, "ci_low": np.nanquantile(boots, 0.025),
                    "ci_high": np.nanquantile(boots, 0.975), "n": len(vals)})
        rows.append(row)
    return pd.DataFrame(rows)


def build_figure4():
    integrated = pd.read_csv(HI / "Table_HI_integrated_participant_features.csv")
    state = pd.read_csv(CANON / "Source Data" / "Figure4_revised_state_enrichment.csv")
    programs = pd.read_csv(CANON / "Source Data" / "Figure4_enhanced_program_effects.csv")
    bt_cells = pd.read_csv(BT / "Table_BT1_balanced_cell_pseudotime.csv.gz")
    trends = pd.read_csv(CANON / "Source Data" / "Figure4_trajectory_integrated_program_trends.csv")
    isotypes = pd.read_csv(BT / "Table_BT5_participant_isotype_trends.csv")
    primary = pd.read_csv(BT / "Table_BT7_primary_statistics.csv")
    connectivity = pd.read_csv(BT / "Table_BT8_clone_state_connectivity_permutation.csv")

    isotype_summary = bootstrap_summary(isotypes, "fraction", ["pt_bin", "isotype"], draws=800)
    isotype_summary.to_csv(SRC_DIR / "Figure4_isotype_trajectory_summary.csv", index=False)
    primary.to_csv(SRC_DIR / "Figure4_clone_size_trajectory_constraint.csv", index=False)
    connectivity.to_csv(SRC_DIR / "Figure4_clone_state_connectivity_null.csv", index=False)

    fig = plt.figure(figsize=(7.05, 8.35))
    fig.suptitle("Expanded BCRs concentrate in antibody-secreting states along an effector trajectory",
                 fontsize=11.2, fontweight="bold", y=0.982)
    outer = GridSpec(3, 2, figure=fig, height_ratios=[0.82, 1.0, 0.92],
                     width_ratios=[1, 1.08], hspace=0.52, wspace=0.40,
                     left=0.13, right=0.985, bottom=0.07, top=0.94)

    gs_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.42)
    metrics = [("bcr_gini", "BCR Gini"), ("bcr_expanded_cell_fraction", "Expanded BCR-cell fraction")]
    for j, (metric, label) in enumerate(metrics):
        ax = fig.add_subplot(gs_a[0, j])
        data = [integrated.loc[integrated.Diagnosis1 == d, metric].dropna().to_numpy() for d in ("Control", "CD", "UC")]
        bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.58, patch_artist=True, showfliers=False,
                        medianprops={"color": "#222222", "lw": 1})
        for patch, diagnosis in zip(bp["boxes"], ("Control", "CD", "UC")):
            patch.set_facecolor(COL[diagnosis]); patch.set_alpha(0.22); patch.set_edgecolor(COL[diagnosis])
        rng = np.random.default_rng(42+j)
        for x, (diagnosis, values) in enumerate(zip(("Control", "CD", "UC"), data)):
            ax.scatter(x+rng.normal(0, 0.06, len(values)), values, s=5, color=COL[diagnosis], alpha=0.42, linewidths=0)
        ax.set_xticks([0, 1, 2], ["Control", "CD", "UC"], rotation=25, ha="right")
        ax.set_ylabel(label)
        if j == 0:
            panel(ax, "A", "IBD BCR expansion")
        clean_axes(ax)

    ax_b = fig.add_subplot(outer[0, 1])
    splot = state.sort_values("median_change_percentage_points")
    y = np.arange(len(splot))
    colors = [COL["plasma"] if x > 0 else "#3B78A8" for x in splot.median_change_percentage_points]
    ax_b.errorbar(splot.median_change_percentage_points, y,
                  xerr=[splot.median_change_percentage_points-splot.ci95_low,
                        splot.ci95_high-splot.median_change_percentage_points],
                  fmt="none", ecolor="#777777", capsize=2, lw=1.15)
    ax_b.scatter(splot.median_change_percentage_points, y, c=colors, s=20, zorder=3)
    ax_b.axvline(0, color="black", lw=0.7)
    ax_b.set_yticks(y, splot.BcellState.str.replace(" B Cell", "", regex=False))
    ax_b.set_xlabel("Expanded - singleton state fraction (percentage points)")
    clean_axes(ax_b)
    panel(ax_b, "B", "Expanded clones favor plasma states")

    ax_c = fig.add_subplot(outer[1, 0])
    pplot = programs.sort_values("median_member_gene_logFC")
    short_programs = {
        "Plasma-cell antigen presentation/UPR": "Presentation/UPR",
        "IgG inflammatory plasma cell": "IgG plasma",
        "Plasmablast/plasma differentiation": "Plasma diff.",
        "IgA mucosal plasma cell": "IgA plasma",
        "Antibody secretion/UPR": "Secretion/UPR",
    }
    labels = pplot["display"].map(short_programs).fillna(pplot["display"])
    y = np.arange(len(pplot))
    ax_c.barh(y, pplot.median_member_gene_logFC, color=[COL["plasma"] if v > 0 else "#3B78A8" for v in pplot.median_member_gene_logFC], alpha=0.9)
    ax_c.axvline(0, color="black", lw=0.7)
    ax_c.set_yticks(y, labels)
    ax_c.set_xlabel("Median expanded-vs-singleton member-gene log2FC")
    clean_axes(ax_c)
    panel(ax_c, "C", "Clone-aware effector programs")

    ax_d = fig.add_subplot(outer[1, 1])
    rng = np.random.default_rng(7)
    plot_cells = bt_cells
    if len(plot_cells) > 15000:
        plot_cells = plot_cells.iloc[rng.choice(len(plot_cells), 15000, replace=False)]
    sc = ax_d.scatter(plot_cells.UMAP_1, plot_cells.UMAP_2,
                      c=plot_cells.trajectory_pseudotime, cmap="viridis", s=1.0,
                      alpha=0.72, linewidths=0, rasterized=True)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    ax_d.set_xlabel("B-cell UMAP 1"); ax_d.set_ylabel("B-cell UMAP 2")
    clean_axes(ax_d, grid=False)
    panel(ax_d, "D", "Participant-balanced B-cell trajectory")
    cb = fig.colorbar(sc, ax=ax_d, orientation="horizontal", fraction=0.06, pad=0.08)
    cb.set_label("Normalized pseudotime")

    ax_e = fig.add_subplot(outer[2, 0])
    order = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    trends = trends.copy()
    trends["pt_clean"] = trends.pt_bin.astype(str).str.replace("–", "-", regex=False)
    prog_colors = {"IgA mucosal": "#3A9D72", "Plasma differentiation": COL["UC"],
                   "Antibody secretion/UPR": "#B4457A"}
    for prog, color in prog_colors.items():
        ss = trends[trends.program == prog].set_index("pt_clean").reindex(order)
        x = np.arange(5)
        ax_e.plot(x, ss["mean"], marker="o", ms=3.3, color=color, label=prog)
        ax_e.fill_between(x, ss.ci95_low.astype(float), ss.ci95_high.astype(float), color=color, alpha=0.12, linewidth=0)
    ax_e.axhline(0, color="#777777", lw=0.6)
    ax_e.set_xticks(np.arange(5), ["0", ".2", ".4", ".6", ".8-1"])
    ax_e.set_xlabel("Pseudotime interval")
    ax_e.set_ylabel("Participant-mean module score")
    clean_axes(ax_e)
    panel(ax_e, "E", "Effector programs emerge along pseudotime")
    ax_e.legend(frameon=False, loc="upper left")

    ax_f = fig.add_subplot(outer[2, 1])
    iso_order = ["IgM", "IgD", "IgA", "IgG"]
    for iso in iso_order:
        ss = isotype_summary[isotype_summary.isotype == iso].copy()
        ss["pt_clean"] = ss.pt_bin.astype(str).str.replace("–", "-", regex=False)
        ss = ss.set_index("pt_clean").reindex(order)
        x = np.arange(5)
        ax_f.plot(x, ss["median"], marker="o", ms=3.0, color=COL[iso], label=iso)
        ax_f.fill_between(x, ss.ci_low.astype(float), ss.ci_high.astype(float), color=COL[iso], alpha=0.10, linewidth=0)
    ax_f.set_xticks(np.arange(5), ["0", ".2", ".4", ".6", ".8-1"])
    ax_f.set_xlabel("Pseudotime interval")
    ax_f.set_ylabel("Median participant isotype fraction")
    clean_axes(ax_f)
    panel(ax_f, "F", "Isotype composition along trajectory")
    ax_f.legend(frameon=False, ncol=2, loc="upper center")

    save_figure(fig, 4)
    (LEG_DIR / "Figure_4_immunity_preview_legend.txt").write_text(
        "Figure 4. Expanded BCRs concentrate in antibody-secreting states along an effector trajectory. "
        "(A) Participant-level BCR Gini and expanded-cell fraction. (B) Participant-matched expanded-minus-singleton state enrichment. "
        "(C) Clone-aware gene-program effects. (D) Participant-balanced B-cell trajectory. (E) Effector programs and (F) isotype composition across pseudotime. "
        "A separate matched-null analysis found sparse cross-state exact-clone sharing and no FDR-significant state pair; therefore the figure does not claim direct memory-to-plasma clone transitions. Preview only.\n",
        encoding="utf-8"
    )


def build_figure5():
    values = pd.read_csv(CANON / "Source Data" / "Figure_5_CDR_minus_FWR_participant_values.csv")
    points = pd.read_csv(CANON / "Source Data" / "Figure_5_adjusted_calprotectin_CDR_targeting_points.csv")
    effects = pd.read_csv(CANON / "Source Data" / "Figure_5_calprotectin_CDR_targeting_effects.csv")
    isotype = pd.read_csv(CANON / "Source Data" / "Figure_5_isotype_matched_effects.csv")
    architecture = pd.read_csv(CANON / "Source Data" / "Figure_5_standardized_lineage_architecture_effects.csv")
    iga = pd.read_csv(CANON / "Source Data" / "Figure_5_expanded_lineage_IgA_plasma_effects.csv")
    meta = pd.read_csv(CANON / "Source Data" / "Figure_5_expanded_lineage_maturation_landscape.csv")[["SampleID", "acquisition_series"]].drop_duplicates()
    data = points.merge(meta, on="SampleID", how="left")
    data["is_uc"] = (data.Diagnosis == "UC").astype(int)
    model = smf.ols("adjusted_CDR_targeting_rank ~ adjusted_log_calprotectin_rank*is_uc", data).fit(cov_type="HC3")
    term = "adjusted_log_calprotectin_rank:is_uc"
    ci = model.conf_int().loc[term]
    interaction = pd.DataFrame([{
        "contrast": "UC minus CD calprotectin slope",
        "estimate": model.params[term], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1],
        "p_value": model.pvalues[term], "n": len(data),
    }])
    loo_rows = []
    for diagnosis in ("CD", "UC"):
        subset = data[data.Diagnosis == diagnosis]
        full_rho = spearmanr(subset.adjusted_log_calprotectin_rank,
                             subset.adjusted_CDR_targeting_rank).statistic
        loo_rows.append({"Diagnosis": diagnosis, "omitted_series": "None", "rho": full_rho, "n": len(subset)})
        for series in sorted(subset.acquisition_series.dropna().unique()):
            work = subset[subset.acquisition_series != series]
            rho = spearmanr(work.adjusted_log_calprotectin_rank,
                            work.adjusted_CDR_targeting_rank).statistic
            loo_rows.append({"Diagnosis": diagnosis, "omitted_series": series, "rho": rho, "n": len(work)})
    loo = pd.DataFrame(loo_rows)
    interaction.to_csv(SRC_DIR / "Figure5_formal_diagnosis_interaction.csv", index=False)
    loo.to_csv(SRC_DIR / "Figure5_leave_one_acquisition_series_out.csv", index=False)

    fig = plt.figure(figsize=(7.05, 8.35))
    fig.suptitle("CDR-focused B-cell maturation tracks intestinal inflammation in ulcerative colitis",
                 fontsize=11.1, fontweight="bold", y=0.982)
    outer = GridSpec(3, 2, figure=fig, height_ratios=[0.86, 0.82, 0.93],
                     width_ratios=[0.83, 1.35], hspace=0.54, wspace=0.42,
                     left=0.13, right=0.985, bottom=0.07, top=0.94)

    ax_a = fig.add_subplot(outer[0, 0])
    groups = ("Control", "CD", "UC")
    vals = [values.loc[values.Diagnosis == d, "CDR_minus_FWR"].dropna().to_numpy() for d in groups]
    bp = ax_a.boxplot(vals, positions=[0, 1, 2], widths=0.55, patch_artist=True, showfliers=False,
                      medianprops={"color": "#222222", "lw": 1})
    for patch, diagnosis in zip(bp["boxes"], groups):
        patch.set_facecolor(COL[diagnosis]); patch.set_alpha(0.22); patch.set_edgecolor(COL[diagnosis])
    rng = np.random.default_rng(22)
    for x, (diagnosis, arr) in enumerate(zip(groups, vals)):
        ax_a.scatter(x+rng.normal(0, 0.06, len(arr)), arr, s=6, alpha=0.45,
                     color=COL[diagnosis], linewidths=0)
    ax_a.axhline(0, color="#555555", lw=0.6)
    ax_a.set_xticks([0, 1, 2], groups)
    ax_a.set_ylabel("CDR - framework SHM rate")
    clean_axes(ax_a)
    panel(ax_a, "A", "Preferential CDR targeting")

    gs_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.36)
    for j, diagnosis in enumerate(("CD", "UC")):
        ax = fig.add_subplot(gs_b[0, j])
        ss = data[data.Diagnosis == diagnosis]
        ax.scatter(ss.adjusted_log_calprotectin_rank, ss.adjusted_CDR_targeting_rank,
                   s=9, alpha=0.58, color=COL[diagnosis], linewidths=0)
        coef = np.polyfit(ss.adjusted_log_calprotectin_rank, ss.adjusted_CDR_targeting_rank, 1)
        xx = np.linspace(ss.adjusted_log_calprotectin_rank.min(), ss.adjusted_log_calprotectin_rank.max(), 100)
        ax.plot(xx, coef[0]*xx+coef[1], color=COL[diagnosis], lw=1.4)
        row = effects[effects.Diagnosis == diagnosis].iloc[0]
        ax.text(0.03, 0.96, f"partial rho={row.partial_spearman_rho:.2f}\nFDR={row.FDR:.3g}",
                transform=ax.transAxes, va="top", fontsize=6.6, color=COL[diagnosis], fontweight="bold")
        ax.set_xlabel("Adjusted log calprotectin rank")
        if j == 0:
            ax.set_ylabel("Adjusted CDR-targeting rank")
            panel(ax, "B", "Diagnosis-stratified inflammatory burden")
        else:
            ax.set_yticklabels([])
        ax.set_title(diagnosis, loc="left", fontweight="bold", color=COL[diagnosis], pad=3)
        clean_axes(ax)

    ax_c = fig.add_subplot(outer[1, 0])
    y = np.arange(3)
    rows = []
    for diagnosis in ("CD", "UC"):
        row = effects[effects.Diagnosis == diagnosis].iloc[0]
        rows.append((diagnosis, row.partial_spearman_rho, row.ci_low, row.ci_high, COL[diagnosis]))
    int_row = interaction.iloc[0]
    rows.append(("UC-CD interaction", int_row.estimate, int_row.ci_low, int_row.ci_high, "#222222"))
    for yy, (label, est, lo, hi, color) in enumerate(rows):
        ax_c.errorbar(est, yy, xerr=[[est-lo], [hi-est]], fmt="o", color=color, capsize=2, ms=4, lw=1.1)
    ax_c.axvline(0, color="black", lw=0.7)
    ax_c.set_yticks(y, [r[0] for r in rows])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Association estimate (95% CI)")
    clean_axes(ax_c)
    panel(ax_c, "C", "Formal diagnosis interaction")
    ax_c.text(0.98, 0.05, f"interaction P={int_row.p_value:.3f}", transform=ax_c.transAxes,
              ha="right", fontsize=6.6, fontweight="bold")

    ax_d = fig.add_subplot(outer[1, 1])
    series_order = sorted([x for x in loo.omitted_series.unique() if x != "None"])
    for diagnosis, offset in (("CD", -0.09), ("UC", 0.09)):
        ss = loo[(loo.Diagnosis == diagnosis) & (loo.omitted_series != "None")].set_index("omitted_series").reindex(series_order)
        ax_d.plot(np.arange(len(series_order))+offset, ss.rho, "o-", color=COL[diagnosis], ms=3.4, label=diagnosis)
        full = loo[(loo.Diagnosis == diagnosis) & (loo.omitted_series == "None")].rho.iloc[0]
        ax_d.axhline(full, color=COL[diagnosis], lw=0.8, ls="--", alpha=0.75)
    ax_d.axhline(0, color="black", lw=0.7)
    ax_d.set_xticks(np.arange(len(series_order)), series_order)
    ax_d.set_ylabel("Spearman rho after omitting series")
    ax_d.set_xlabel("Omitted acquisition series")
    clean_axes(ax_d)
    panel(ax_d, "D", "UC association is series-robust")
    ax_d.legend(frameon=False, ncol=2, loc="upper left")

    ax_e = fig.add_subplot(outer[2, 0])
    plot = isotype.copy()
    plot["diagnosis"] = plot.contrast.str.split().str[0]
    iso_order = ["IgM", "IgA", "IgG"]
    pos = []
    labels = []
    yy = 0
    for iso in iso_order:
        for diagnosis in ("CD", "UC"):
            row = plot[(plot.isotype_class == iso) & (plot.diagnosis == diagnosis)].iloc[0]
            ax_e.errorbar(row.median_difference, yy,
                          xerr=[[row.median_difference-row.ci_low], [row.ci_high-row.median_difference]],
                          fmt="o", color=COL[diagnosis], ms=3.8, capsize=2)
            pos.append(yy); labels.append(f"{iso} | {diagnosis}"); yy += 1
        yy += 0.35
    ax_e.axvline(0, color="black", lw=0.7)
    ax_e.set_yticks(pos, labels)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Isotype-matched SHM-rate difference vs control")
    clean_axes(ax_e)
    panel(ax_e, "E", "Isotype-matched remodeling")

    ax_f = fig.add_subplot(outer[2, 1])
    metrics = ["expanded_lineages", "mean_MST_branch_length", "mean_within_lineage_divergence"]
    metric_labels = {"expanded_lineages": "Expanded lineages", "mean_MST_branch_length": "Branch length",
                     "mean_within_lineage_divergence": "Divergence"}
    pos = []
    labels = []
    yy = 0
    for metric in metrics:
        for diagnosis in ("CD", "UC"):
            row = architecture[(architecture.metric == metric) & architecture.contrast.str.startswith(diagnosis)].iloc[0]
            est = row.standardized_median_difference
            ax_f.errorbar(est, yy, xerr=[[est-row.ci_low], [row.ci_high-est]], fmt="o",
                          color=COL[diagnosis], ms=3.8, capsize=2)
            pos.append(yy); labels.append(f"{metric_labels[metric]} | {diagnosis}"); yy += 1
        yy += 0.35
    for diagnosis in ("CD", "UC"):
        row = iga[iga.contrast.str.startswith(diagnosis)].iloc[0]
        est = row.median_fraction_difference
        ax_f.errorbar(est, yy, xerr=[[est-row.ci_low], [row.ci_high-est]], fmt="D",
                      color=COL[diagnosis], ms=3.8, capsize=2)
        pos.append(yy); labels.append(f"IgA plasma fraction | {diagnosis}"); yy += 1
    ax_f.axvline(0, color="black", lw=0.7)
    ax_f.set_yticks(pos, labels)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Disease-control effect (95% CI; standardized where indicated)")
    clean_axes(ax_f)
    panel(ax_f, "F", "Expanded-lineage maturation architecture")

    save_figure(fig, 5)
    (LEG_DIR / "Figure_5_immunity_preview_legend.txt").write_text(
        "Figure 5. CDR-focused B-cell maturation tracks intestinal inflammation in ulcerative colitis. "
        "(A) Participant-level CDR-minus-framework SHM targeting. (B) Covariate-adjusted residual-rank associations with calprotectin. "
        "(C) Diagnosis-specific partial correlations and the formal UC-minus-CD slope interaction. (D) Leave-one-acquisition-series-out robustness. "
        "(E) Isotype-matched SHM differences. (F) Expanded-lineage architecture and IgA plasma-state representation. Associations are not interpreted as antigen specificity. Preview only.\n",
        encoding="utf-8"
    )


def residualize_rank(data, outcome, predictor, covariates):
    work = data[[outcome, predictor] + covariates].dropna().copy()
    x = pd.Series(rankdata(work[predictor]), index=work.index)
    y = pd.Series(rankdata(work[outcome]), index=work.index)
    design = pd.get_dummies(work[covariates], columns=[c for c in covariates if work[c].dtype == object], drop_first=True, dtype=float)
    design = sm.add_constant(design.astype(float))
    work["x_resid"] = sm.OLS(x, design).fit().resid
    work["y_resid"] = sm.OLS(y, design).fit().resid
    if "Diagnosis1" in data.columns:
        work["Diagnosis1"] = data.loc[work.index, "Diagnosis1"]
    return work


def build_figure6():
    helper_corr = pd.read_csv(RA / "Table_RA1_helper_B_partial_correlations.csv")
    helper_data = pd.read_csv(RA / "Table_RA1_helper_axis_analysis_dataset.csv")
    targeted = pd.read_csv(TB / "Table_TB10_targeted_T_B_correlations.csv")
    independent = pd.read_csv(TB / "Table_TB11_regulatory_restraint_models.csv")
    coupling = pd.read_csv(CANON / "Source Data" / "Figure_6_coupling_effects.csv")
    validation = pd.read_csv(CANON / "Source Data" / "Figure_6_panel_G_blocked_validation_summary.csv")
    integrated = pd.read_csv(HI / "Table_HI_integrated_participant_features.csv")

    joint = integrated.merge(helper_data[["SampleID", "acquisition_series", "cd4_mean_Tph_Tfh_help",
                                          "b_mean_IgA_mucosal_plasma_cell",
                                          "b_mean_Plasmablast_plasma_differentiation",
                                          "b_mean_B_cell_antigen_presentation", "n_cd4_cells", "n_b_cells"]],
                             on="SampleID", how="inner")
    joint = joint[joint.Diagnosis1.isin(["CD", "UC"])].copy()
    joint["is_uc"] = (joint.Diagnosis1 == "UC").astype(int)
    joint["sex_male"] = (joint.Sex == "M").astype(int)
    joint["log_tcr_depth"] = np.log1p(joint.tcr_total_cells)
    joint["log_bcr_depth"] = np.log1p(joint.bcr_total_cells)

    interaction_rows = []
    for outcome in ("b_mean_IgA_mucosal_plasma_cell", "b_mean_Plasmablast_plasma_differentiation"):
        for predictor in ("cd4_mean_Tph_Tfh_help", "tcr_cytotoxic_expanded"):
            cols = [outcome, predictor, "is_uc", "Age", "sex_male", "log_tcr_depth", "log_bcr_depth", "acquisition_series"]
            work = joint[cols].dropna().copy()
            work["y"] = (rankdata(work[outcome]) - np.mean(rankdata(work[outcome]))) / np.std(rankdata(work[outcome]), ddof=1)
            work["x"] = (rankdata(work[predictor]) - np.mean(rankdata(work[predictor]))) / np.std(rankdata(work[predictor]), ddof=1)
            model = smf.ols("y ~ x*is_uc + Age + sex_male + log_tcr_depth + log_bcr_depth + C(acquisition_series)", work).fit(cov_type="HC3")
            term = "x:is_uc"
            ci = model.conf_int().loc[term]
            interaction_rows.append({
                "outcome": outcome, "predictor": predictor,
                "UC_minus_CD_interaction": model.params[term],
                "ci_low": ci.iloc[0], "ci_high": ci.iloc[1],
                "p_value": model.pvalues[term], "n": len(work),
            })
    interactions = pd.DataFrame(interaction_rows)
    interactions.to_csv(SRC_DIR / "Figure6_formal_CD_UC_interactions.csv", index=False)

    scatter_source = residualize_rank(
        helper_data[helper_data.Diagnosis1.isin(["CD", "UC"])].copy(),
        "b_mean_B_cell_antigen_presentation", "cd4_mean_Tph_Tfh_help",
        ["Age", "sex_male", "log_cd4_cells", "log_b_cells", "acquisition_series"]
    )
    scatter_source[["Diagnosis1", "x_resid", "y_resid"]].to_csv(
        SRC_DIR / "Figure6_adjusted_helper_antigen_presentation_residuals.csv", index=False)

    fig = plt.figure(figsize=(7.05, 8.45))
    fig.suptitle("Shared helper programs coordinate B-cell remodeling across IBD",
                 fontsize=11.4, fontweight="bold", y=0.982)
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.90, 0.82],
                     width_ratios=[1.08, 1], hspace=0.55, wspace=0.40,
                     left=0.145, right=0.985, bottom=0.07, top=0.94)

    ax_a = fig.add_subplot(outer[0, 0])
    t_order = ["Tph/Tfh help", "Pathogenic Th17", "Suppressive Treg"]
    b_order = ["Plasma differentiation", "IgA mucosal plasma", "IgG inflammatory plasma",
               "Atypical memory", "Antigen presentation"]
    target_ibd = targeted[(targeted.group == "IBD") & targeted.T_label.isin(t_order) & targeted.B_label.isin(b_order)]
    mat_df = target_ibd.pivot(
        index="T_label", columns="B_label", values="partial_rho").reindex(index=t_order, columns=b_order)
    q_df = target_ibd.pivot(
        index="T_label", columns="B_label", values="FDR_within_group").reindex(index=t_order, columns=b_order)
    mat = mat_df.to_numpy(float)
    qmatrix = q_df.to_numpy(float)
    im = ax_a.imshow(mat, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax_a.set_xticks(np.arange(5), ["Plasma\ndiff.", "IgA\nmucosal", "IgG\ninflamm.",
                                   "Atypical\nmemory", "Antigen\npresent."], rotation=0)
    ax_a.set_yticks(np.arange(3), t_order)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                color = "white" if abs(mat[i, j]) > 0.43 else "#222222"
                ax_a.text(j, i, f"{mat[i,j]:.2f}{fdr_mark(qmatrix[i,j])}", ha="center", va="center", fontsize=5.8, color=color)
    panel(ax_a, "A", "Targeted helper-B-cell coordination in IBD")
    cb = fig.colorbar(im, ax=ax_a, orientation="horizontal", fraction=0.055, pad=0.23)
    cb.set_label("Partial Spearman rho")

    ax_b = fig.add_subplot(outer[0, 1])
    outcome_order = ["Plasma differentiation", "IgA mucosal plasma", "IgG inflammatory plasma", "Atypical memory", "Antigen presentation"]
    pred_order = ["Tph/Tfh help", "Pathogenic Th17", "Suppressive Treg"]
    offsets = {"Tph/Tfh help": -0.18, "Pathogenic Th17": 0, "Suppressive Treg": 0.18}
    for i, outcome in enumerate(outcome_order):
        for pred in pred_order:
            row = independent[(independent.B_label == outcome) & (independent.T_label == pred)].iloc[0]
            yy = i + offsets[pred]
            ax_b.errorbar(row.standardized_beta, yy,
                          xerr=[[row.standardized_beta-row.ci_low], [row.ci_high-row.standardized_beta]],
                          fmt="o", color=COL[pred], ms=3.8, capsize=2, lw=1.0, label=pred if i == 0 else None)
    ax_b.axvline(0, color="black", lw=0.7)
    ax_b.set_yticks(np.arange(len(outcome_order)),
                    ["Plasma diff.", "IgA mucosal", "IgG inflammatory", "Atypical memory", "Antigen presentation"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Adjusted standardized beta (95% CI)")
    clean_axes(ax_b)
    panel(ax_b, "B", "Independent helper-state contributions")
    ax_b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
                handlelength=1, columnspacing=0.8)

    ax_c = fig.add_subplot(outer[1, 0])
    labels = []
    y = []
    for i, row in interactions.iterrows():
        pred = "Helper" if row.predictor == "cd4_mean_Tph_Tfh_help" else "Cytotoxic"
        out = "IgA mucosal" if row.outcome == "b_mean_IgA_mucosal_plasma_cell" else "Plasma differentiation"
        labels.append(f"{pred} | {'IgA' if 'IgA' in out else 'Plasma'}")
        y.append(i)
        color = COL["Tph/Tfh help"] if pred == "Helper" else "#444444"
        est = row.UC_minus_CD_interaction
        ax_c.errorbar(est, i, xerr=[[est-row.ci_low], [row.ci_high-est]], fmt="o", color=color, capsize=2, ms=4)
        ax_c.text(row.ci_high + 0.025, i, f"P={row.p_value:.2f}", va="center", fontsize=6.2)
    ax_c.axvline(0, color="black", lw=0.7)
    ax_c.set_yticks(y, labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("UC minus CD slope interaction (95% CI)")
    clean_axes(ax_c)
    panel(ax_c, "C", "No formal CD-UC interaction")

    ax_d = fig.add_subplot(outer[1, 1])
    for diagnosis in ("CD", "UC"):
        ss = scatter_source[scatter_source.Diagnosis1 == diagnosis]
        ax_d.scatter(ss.x_resid, ss.y_resid, color=COL[diagnosis], s=9, alpha=0.58, label=diagnosis, linewidths=0)
    coef = np.polyfit(scatter_source.x_resid, scatter_source.y_resid, 1)
    xx = np.linspace(scatter_source.x_resid.min(), scatter_source.x_resid.max(), 100)
    ax_d.plot(xx, coef[0]*xx+coef[1], color="#222222", lw=1.4)
    rho = spearmanr(scatter_source.x_resid, scatter_source.y_resid).statistic
    ax_d.text(0.03, 0.96, f"adjusted rho={rho:.2f}", transform=ax_d.transAxes, va="top", fontsize=6.6, fontweight="bold")
    ax_d.set_xlabel("Tph/Tfh-help residual rank")
    ax_d.set_ylabel("B-cell antigen-presentation residual rank")
    clean_axes(ax_d)
    panel(ax_d, "D", "Representative adjusted association")
    ax_d.legend(frameon=False, loc="lower right")

    ax_e = fig.add_subplot(outer[2, 0])
    outcome_order2 = ["bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module"]
    yy = 0
    pos = []
    labels2 = []
    for outcome in outcome_order2:
        for diagnosis in ("Control", "CD", "UC"):
            row = coupling[(coupling.outcome == outcome) & (coupling.diagnosis == diagnosis)].iloc[0]
            ax_e.errorbar(row.partial_rho, yy, xerr=[[row.partial_rho-row.ci_low], [row.ci_high-row.partial_rho]],
                          fmt="o", color=COL[diagnosis], ms=4, capsize=2)
            label = "IgA" if outcome == outcome_order2[0] else "Plasma"
            pos.append(yy); labels2.append(f"{label} | {diagnosis}"); yy += 1
        yy += 0.3
    ax_e.axvline(0, color="black", lw=0.7)
    ax_e.set_yticks(pos, labels2)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Expanded-cytotoxic TCR coupling rho (95% CI)")
    clean_axes(ax_e)
    panel(ax_e, "E", "Exploratory cytotoxic-plasma coupling")

    ax_f = fig.add_subplot(outer[2, 1])
    model_order = ["Helper only", "Cytotoxic only", "Joint"]
    model_color = {"Helper only": "#7B3294", "Cytotoxic only": COL["CD"], "Joint": "#222222"}
    outcome_labels = {"b_mean_IgA_mucosal_plasma_cell": "IgA mucosal plasma",
                      "b_mean_Plasmablast_plasma_differentiation": "Plasma differentiation"}
    for i, outcome in enumerate(outcome_labels):
        ss = validation[validation.outcome == outcome].set_index("model").reindex(model_order)
        for j, model_name in enumerate(model_order):
            marker = "D" if model_name == "Joint" else "o"
            ax_f.scatter(ss.loc[model_name, "held_out_spearman_rho"], i+(j-1)*0.13,
                         color=model_color[model_name], marker=marker, s=24,
                         label=model_name if i == 0 else None)
    ax_f.axvline(0, color="#777777", lw=0.7, ls="--")
    ax_f.set_yticks([0, 1], list(outcome_labels.values()))
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Leave-one-series-out Spearman correlation")
    clean_axes(ax_f)
    panel(ax_f, "F", "Blocked out-of-series validation")
    ax_f.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
                handlelength=1, columnspacing=0.8)

    save_figure(fig, 6)
    (LEG_DIR / "Figure_6_immunity_preview_legend.txt").write_text(
        "Figure 6. Shared helper programs coordinate B-cell remodeling across IBD. "
        "(A) Covariate-adjusted Tph/Tfh-B-cell program correlations. (B) Joint helper-state models adjusted for diagnosis, age, sex, T- and B-cell depth, and acquisition series. "
        "(C) Formal UC-minus-CD interactions for helper and expanded-cytotoxic associations; none is significant. "
        "(D) Adjusted participant-level Tph/Tfh-help and B-cell antigen-presentation relationship. "
        "(E) Diagnosis-stratified expanded-cytotoxic coupling, interpreted as exploratory because formal interactions are nonsignificant. "
        "(F) Leave-one-acquisition-series-out validation. Preview only; canonical Figure 6 was not changed.\n",
        encoding="utf-8"
    )


def copy_figure3():
    copy2(CANON / "Main Figures" / "Figure_3.pdf", PDF_DIR / "Figure_3_immunity_preview.pdf")
    copy2(CANON / "Main Figures" / "Figure_3.png", PNG_DIR / "Figure_3_immunity_preview.png")
    (LEG_DIR / "Figure_3_immunity_preview_note.txt").write_text(
        "Figure 3 is an exact unchanged copy of the canonical main figure, per instruction.\n",
        encoding="utf-8"
    )


def write_manifest():
    pd.DataFrame([
        {"figure": 2, "status": "revised preview", "canonical_changed": False,
         "main_new_analysis": "diagnosis-specific participant-clustered clone-size slopes and formal interactions"},
        {"figure": 3, "status": "unchanged canonical copy", "canonical_changed": False,
         "main_new_analysis": "none"},
        {"figure": 4, "status": "revised preview", "canonical_changed": False,
         "main_new_analysis": "BCR trajectory constraint and cross-state matched null retained in source data"},
        {"figure": 5, "status": "revised preview", "canonical_changed": False,
         "main_new_analysis": "formal UC-CD calprotectin slope interaction and leave-one-series-out robustness"},
        {"figure": 6, "status": "revised preview", "canonical_changed": False,
         "main_new_analysis": "formal CD-UC interactions for helper and cytotoxic coupling"},
    ]).to_csv(PREVIEW / "preview_manifest.csv", index=False)


def main():
    build_figure2()
    copy_figure3()
    build_figure4()
    build_figure5()
    build_figure6()
    write_manifest()
    print(f"Wrote preview figures to {PNG_DIR}")
    print(f"Wrote preview PDFs to {PDF_DIR}")
    print("Canonical main figures were not modified.")


if __name__ == "__main__":
    main()

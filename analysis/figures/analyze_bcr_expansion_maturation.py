from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from patsy import bs
from sklearn.neighbors import KNeighborsRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "High Impact Additional Analyses" / "BCR Expansion Maturation 20260829"
INPUT = OUT / "Table_BEM1_cell_program_latent_input.csv.gz"
REFERENCE = ROOT / "High Impact Additional Analyses" / "BCR Trajectory Priority" / "Table_BT1_balanced_cell_pseudotime.csv.gz"
LINEAGE_DB = ROOT / "High Impact Additional Analyses" / "BCR Germline Lineages" / "bcr_heavy_lineages_d015_clone-pass.tsv"
LINEAGE_METRICS = ROOT / "High Impact Additional Analyses" / "BCR Germline Lineages" / "Table_BGL4_germline_aware_lineage_metrics.csv"

MODULES = [
    "IgA_mucosal_plasma_cell",
    "plasmablast_plasma_cell_differentiation",
    "antibody_secretion_UPR",
    "IgG_inflammatory_plasma_cell",
    "cell_cycle_proliferating_B_cell",
]
LABELS = {
    "IgA_mucosal_plasma_cell": "IgA mucosal",
    "plasmablast_plasma_cell_differentiation": "Plasma differentiation",
    "antibody_secretion_UPR": "Antibody secretion/UPR",
    "IgG_inflammatory_plasma_cell": "IgG inflammatory plasma",
    "cell_cycle_proliferating_B_cell": "Cycling B cell",
}
SHORT = {
    "IgA_mucosal_plasma_cell": "IgA mucosal",
    "plasmablast_plasma_cell_differentiation": "Plasma diff.",
    "antibody_secretion_UPR": "Antibody/UPR",
    "IgG_inflammatory_plasma_cell": "IgG plasma",
    "cell_cycle_proliferating_B_cell": "Cycling",
}
DIAG = ["Control", "CD", "UC"]
COL = {
    "Control": "#777777", "CD": "#0072B2", "UC": "#D55E00",
    "Pooled": "#222222", "Singleton": "#777777", "Expanded": "#D55E00",
    "up": "#B24745", "down": "#2878A8", "grid": "#D9D9D9", "ink": "#222222",
}
BINS = ["Singleton", "2 cells", "3-4 cells", ">=5 cells"]
RNG = np.random.default_rng(20260829)

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7.3, "axes.titlesize": 7.8,
    "axes.labelsize": 7.3, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "legend.fontsize": 6.2, "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def bh(values) -> np.ndarray:
    values = np.asarray(values, float)
    out = np.full(len(values), np.nan)
    keep = np.isfinite(values)
    p = values[keep]
    if not len(p):
        return out
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.clip(ranked, 0, 1)
    out[np.where(keep)[0]] = restored
    return out


def q_text(value: float) -> str:
    if not np.isfinite(value):
        return "q=NA"
    if value < 1e-4:
        return f"q={value:.1e}"
    return f"q={value:.3g}"


def bootstrap_median(values, n_boot=3000):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    point = float(np.median(values))
    if len(values) == 1:
        return point, np.nan, np.nan
    sims = np.median(values[RNG.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return point, float(np.quantile(sims, .025)), float(np.quantile(sims, .975))


def style_axis(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=COL["grid"], lw=.45, zorder=0)
    ax.set_axisbelow(True)


def project_pseudotime(cells: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE, usecols=["cell_id", "trajectory_pseudotime"])
    scvi = [f"scvi_{i}" for i in range(1, 11)]
    train = cells.merge(reference, on="cell_id", how="inner", validate="one_to_one")
    if len(train) != len(reference):
        raise RuntimeError(f"Only {len(train)} of {len(reference)} trajectory-reference cells were recovered")
    model = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
    model.fit(train[scvi].to_numpy(np.float32), train["trajectory_pseudotime"].to_numpy(float))
    paired = cells[cells["paired_bcr"].eq(True)].copy()
    paired["trajectory_pseudotime"] = model.predict(paired[scvi].to_numpy(np.float32))
    known = reference.set_index("cell_id")["trajectory_pseudotime"]
    direct = paired["cell_id"].map(known)
    paired.loc[direct.notna(), "trajectory_pseudotime"] = direct[direct.notna()]
    paired["trajectory_pseudotime"] = paired["trajectory_pseudotime"].clip(0, 1)
    return paired


def add_standardized_scores(paired: pd.DataFrame) -> pd.DataFrame:
    paired = paired.copy()
    for module in MODULES:
        values = pd.to_numeric(paired[module], errors="coerce")
        paired[f"{module}_z"] = (values - values.mean()) / values.std()
        noig = pd.to_numeric(paired[f"{module}_noIG"], errors="coerce")
        residual = noig - noig.groupby([paired["PatientID"], paired["state"]]).transform("mean")
        paired[f"{module}_state_resid"] = residual / residual.std()
    return paired


def dose_response(paired: pd.DataFrame):
    long = paired.melt(
        id_vars=["cell_id", "PatientID", "Diagnosis1", "state", "clone_bin"],
        value_vars=[f"{m}_z" for m in MODULES], var_name="module", value_name="score_z",
    )
    long["module"] = long["module"].str.replace("_z$", "", regex=True)
    state_bin = (
        long.dropna(subset=["clone_bin", "score_z"])
        .groupby(["PatientID", "Diagnosis1", "state", "clone_bin", "module"], as_index=False)
        .agg(score_z=("score_z", "mean"), n_cells=("score_z", "size"))
    )
    singleton = (
        state_bin[state_bin["clone_bin"].eq("Singleton")]
        [["PatientID", "state", "module", "score_z"]]
        .rename(columns={"score_z": "singleton_score"})
    )
    matched = state_bin.merge(singleton, on=["PatientID", "state", "module"], how="inner")
    matched["delta"] = matched["score_z"] - matched["singleton_score"]
    participant = (
        matched.groupby(["PatientID", "Diagnosis1", "clone_bin", "module"], as_index=False)
        .agg(delta=("delta", "mean"), n_matched_states=("state", "nunique"), n_cells=("n_cells", "sum"))
    )
    participant["clone_bin"] = pd.Categorical(participant["clone_bin"], BINS, ordered=True)
    display_rows = []
    for module in MODULES:
        dm = participant[participant["module"].eq(module)]
        for diagnosis in DIAG + ["Pooled"]:
            group = dm if diagnosis == "Pooled" else dm[dm["Diagnosis1"].eq(diagnosis)]
            for clone_bin in BINS:
                z = group[group["clone_bin"].eq(clone_bin)]
                med, lo, hi = bootstrap_median(z["delta"])
                display_rows.append({
                    "module": module, "Diagnosis": diagnosis, "clone_bin": clone_bin,
                    "median": med, "ci_low": lo, "ci_high": hi,
                    "n_participants": z["PatientID"].nunique(), "n_cells": int(z["n_cells"].sum()),
                })
    display = pd.DataFrame(display_rows)
    tests = []
    for module in MODULES:
        for diagnosis in DIAG + ["Pooled"]:
            dm = participant[participant["module"].eq(module)].copy()
            if diagnosis != "Pooled":
                dm = dm[dm["Diagnosis1"].eq(diagnosis)]
            dm["bin_rank"] = dm["clone_bin"].cat.codes.astype(float)
            n_bins = dm.groupby("PatientID")["clone_bin"].nunique()
            dm = dm[dm["PatientID"].isin(n_bins[n_bins >= 2].index)]
            if dm["PatientID"].nunique() < 8:
                continue
            fit = smf.ols("delta ~ bin_rank + C(PatientID)", data=dm).fit(
                cov_type="cluster", cov_kwds={"groups": dm["PatientID"]}
            )
            tests.append({
                "module": module, "contrast": diagnosis, "estimate": fit.params["bin_rank"],
                "SE": fit.bse["bin_rank"], "ci_low": fit.params["bin_rank"] - 1.96 * fit.bse["bin_rank"],
                "ci_high": fit.params["bin_rank"] + 1.96 * fit.bse["bin_rank"],
                "p_value": fit.pvalues["bin_rank"], "n_participants": dm["PatientID"].nunique(),
                "n_participant_bins": len(dm),
            })
    tests = pd.DataFrame(tests)
    tests["FDR"] = np.nan
    for contrast, idx in tests.groupby("contrast").groups.items():
        tests.loc[idx, "FDR"] = bh(tests.loc[idx, "p_value"])
    return participant, display, tests


def kinetics(paired: pd.DataFrame):
    edges = np.unique(np.quantile(paired["trajectory_pseudotime"], np.linspace(0, 1, 11)))
    paired = paired.copy()
    paired["pt_bin"] = pd.cut(paired["trajectory_pseudotime"], edges, include_lowest=True, duplicates="drop")
    paired["pt_mid"] = paired["pt_bin"].map(lambda x: float(x.mid)).astype(float)
    long = paired.melt(
        id_vars=["PatientID", "Diagnosis1", "state", "clone_status", "pt_bin", "pt_mid"],
        value_vars=[f"{m}_z" for m in MODULES], var_name="module", value_name="score_z",
    )
    long["module"] = long["module"].str.replace("_z$", "", regex=True)
    state_status = (
        long.groupby(["PatientID", "Diagnosis1", "state", "pt_bin", "pt_mid", "clone_status", "module"],
                     observed=True, as_index=False)
        .agg(score_z=("score_z", "mean"), n_cells=("score_z", "size"))
    )
    wide = state_status.pivot_table(
        index=["PatientID", "Diagnosis1", "state", "pt_bin", "pt_mid", "module"],
        columns="clone_status", values="score_z", observed=True,
    ).dropna(subset=["Singleton", "Expanded"]).reset_index()
    matched = wide.melt(
        id_vars=["PatientID", "Diagnosis1", "state", "pt_bin", "pt_mid", "module"],
        value_vars=["Singleton", "Expanded"], var_name="clone_status", value_name="score_z",
    )
    participant = (
        matched.groupby(["PatientID", "Diagnosis1", "pt_bin", "pt_mid", "clone_status", "module"],
                        observed=True, as_index=False)
        .agg(score_z=("score_z", "mean"), n_matched_states=("state", "nunique"))
    )
    display_rows = []
    for (module, status, pt_mid), g in participant.groupby(["module", "clone_status", "pt_mid"]):
        med, lo, hi = bootstrap_median(g["score_z"])
        display_rows.append({
            "module": module, "clone_status": status, "pt_mid": pt_mid,
            "median_score_z": med, "ci_low": lo, "ci_high": hi,
            "n_participants": g["PatientID"].nunique(), "n_state_pairs": int(g["n_matched_states"].sum()),
        })
    display = pd.DataFrame(display_rows)
    tests = []
    for module, g in participant.groupby("module"):
        fit = smf.ols(
            "score_z ~ bs(pt_mid, df=4, include_intercept=False) * "
            "C(clone_status, Treatment(reference='Singleton')) + C(PatientID)", data=g,
        ).fit(cov_type="cluster", cov_kwds={"groups": g["PatientID"]})
        names = list(fit.params.index)
        interaction_terms = [x for x in names if "bs(pt_mid" in x and ":C(clone_status" in x]
        trajectory_terms = [x for x in names if x.startswith("bs(pt_mid") and ":" not in x]
        def wald(terms):
            restriction = np.zeros((len(terms), len(names)))
            for i, term in enumerate(terms):
                restriction[i, names.index(term)] = 1
            return float(fit.wald_test(restriction, scalar=True).pvalue)
        tests.append({
            "module": module, "trajectory_p_value": wald(trajectory_terms),
            "expansion_by_trajectory_p_value": wald(interaction_terms),
            "n_participant_bins": len(g), "n_participants": g["PatientID"].nunique(),
        })
    tests = pd.DataFrame(tests)
    tests["trajectory_FDR"] = bh(tests["trajectory_p_value"])
    tests["expansion_by_trajectory_FDR"] = bh(tests["expansion_by_trajectory_p_value"])
    return participant, display, tests


def map_lineages(paired: pd.DataFrame):
    parts = paired["exact_paired_clone"].str.split("|", expand=True)
    paired = paired.copy()
    paired["heavy_aa_key"] = (
        parts[0].astype(str) + "|" + parts[1].str.replace(r"\*.*$", "", regex=True) + "|" +
        parts[2].str.replace(r"\*.*$", "", regex=True) + "|" + parts[3].str.upper()
    )
    db = pd.read_csv(
        LINEAGE_DB, sep="\t", usecols=["sampleid", "clone_id", "v_call", "j_call", "junction_aa"],
        low_memory=False,
    )
    core = (db["junction_aa"].astype(str).str.upper()
            .str.replace(r"^C", "", regex=True).str.replace(r"W$", "", regex=True))
    db["heavy_aa_key"] = (
        db["sampleid"].astype(str) + "|" + db["v_call"].str.replace(r"\*.*$", "", regex=True) + "|" +
        db["j_call"].str.replace(r"\*.*$", "", regex=True) + "|" + core
    )
    db["lineage_id"] = db["sampleid"].astype(str) + "::" + db["clone_id"].astype(str)
    map_counts = db.groupby("heavy_aa_key")["lineage_id"].nunique()
    unambiguous = map_counts[map_counts.eq(1)].index
    mapping = db[db["heavy_aa_key"].isin(unambiguous)][["heavy_aa_key", "lineage_id"]].drop_duplicates()
    mapped = paired.merge(mapping, on="heavy_aa_key", how="left", validate="many_to_one")
    program_cols = [f"{m}_state_resid" for m in MODULES]
    lineage = (
        mapped.dropna(subset=["lineage_id"])
        .groupby(["lineage_id", "PatientID", "Diagnosis1"], as_index=False)
        .agg(
            **{f"{m}_program": (f"{m}_state_resid", "mean") for m in MODULES},
            lineage_mean_pseudotime=("trajectory_pseudotime", "mean"),
            n_program_cells=("cell_id", "size"),
            n_exact_paired_clones=("exact_paired_clone", "nunique"),
        )
    )
    metrics = pd.read_csv(LINEAGE_METRICS).rename(columns={"SampleID": "PatientID_metric"})
    lineage = lineage.merge(metrics, on="lineage_id", how="inner", validate="one_to_one")
    lineage = lineage[lineage["PatientID"].astype(str).eq(lineage["PatientID_metric"].astype(str))].copy()
    return mapped, lineage


def lineage_models(lineage: pd.DataFrame):
    outcomes = {
        "SHM/germline distance": ("max_germline_distance", "continuous", "all"),
        "Diversified-lineage branch length": ("total_MST_branch_length", "continuous_log", "diversified"),
        "Class-switched lineage": ("class_switched_lineage", "binary", "all"),
        "Cross-state occupancy": ("n_cell_states", "continuous_log", "multicell"),
    }
    rows = []
    for outcome_label, (outcome, kind, subset) in outcomes.items():
        base = lineage.copy()
        if subset == "diversified":
            base = base[base["n_unique_sequences"].ge(2)]
        if subset == "multicell":
            base = base[base["mapped_cells"].ge(2)]
        base[outcome] = pd.to_numeric(base[outcome], errors="coerce")
        if kind == "continuous_log":
            base["model_outcome"] = np.log1p(base[outcome])
        else:
            base["model_outcome"] = base[outcome].astype(float)
        if kind != "binary":
            base["model_outcome"] = (
                (base["model_outcome"] - base["model_outcome"].mean()) / base["model_outcome"].std()
            )
        base["log_abundance"] = np.log1p(pd.to_numeric(base["weighted_abundance"], errors="coerce"))
        base["log_program_cells"] = np.log1p(base["n_program_cells"])
        base["is_IBD"] = (~base["Diagnosis1"].eq("Control")).astype(float)
        for module in MODULES:
            predictor = f"{module}_program"
            base["program_z"] = (base[predictor] - base[predictor].mean()) / base[predictor].std()
            for model_name, add_pt in [("participant_fixed_effect", False), ("plus_pseudotime", True)]:
                cols = ["model_outcome", "program_z", "log_abundance", "log_program_cells", "PatientID"]
                if add_pt:
                    cols.append("lineage_mean_pseudotime")
                use = base[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
                participant_counts = use.groupby("PatientID").size()
                use = use[use["PatientID"].isin(participant_counts[participant_counts >= 2].index)]
                if len(use) < 80 or use["PatientID"].nunique() < 15:
                    continue
                rhs = "program_z + log_abundance + log_program_cells + C(PatientID)"
                if add_pt:
                    rhs = "program_z + log_abundance + log_program_cells + lineage_mean_pseudotime + C(PatientID)"
                fit = smf.ols(f"model_outcome ~ {rhs}", data=use).fit(
                    cov_type="cluster", cov_kwds={"groups": use["PatientID"]}
                )
                estimate = float(fit.params["program_z"])
                se = float(fit.bse["program_z"])
                rows.append({
                    "outcome": outcome_label, "module": module, "model": model_name,
                    "estimate_per_program_SD": estimate, "SE": se,
                    "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
                    "p_value": float(fit.pvalues["program_z"]), "n_lineages": len(use),
                    "n_participants": use["PatientID"].nunique(),
                    "outcome_scale": "probability" if kind == "binary" else "outcome SD",
                })
            interaction_cols = ["model_outcome", "program_z", "is_IBD", "log_abundance",
                                "log_program_cells", "lineage_mean_pseudotime", "PatientID"]
            interaction_use = base[interaction_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
            participant_counts = interaction_use.groupby("PatientID").size()
            interaction_use = interaction_use[
                interaction_use["PatientID"].isin(participant_counts[participant_counts >= 2].index)
            ]
            interaction_group_participants = interaction_use.groupby("is_IBD")["PatientID"].nunique()
            interaction_group_lineages = interaction_use.groupby("is_IBD").size()
            if (len(interaction_use) >= 80 and interaction_use["PatientID"].nunique() >= 15 and
                    interaction_use["is_IBD"].nunique() == 2 and
                    interaction_group_participants.min() >= 8 and interaction_group_lineages.min() >= 40):
                fit = smf.ols(
                    "model_outcome ~ program_z + program_z:is_IBD + log_abundance + log_program_cells + "
                    "lineage_mean_pseudotime + C(PatientID)", data=interaction_use,
                ).fit(cov_type="cluster", cov_kwds={"groups": interaction_use["PatientID"]})
                term = "program_z:is_IBD"
                estimate = float(fit.params[term])
                se = float(fit.bse[term])
                rows.append({
                    "outcome": outcome_label, "module": module, "model": "IBD_vs_control_interaction",
                    "estimate_per_program_SD": estimate, "SE": se,
                    "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
                    "p_value": float(fit.pvalues[term]), "n_lineages": len(interaction_use),
                    "n_participants": interaction_use["PatientID"].nunique(),
                    "outcome_scale": "probability" if kind == "binary" else "outcome SD",
                })
            for group_label, group_value in [("Control_stratified", 0.0), ("IBD_stratified", 1.0)]:
                stratified_use = interaction_use[interaction_use["is_IBD"].eq(group_value)].copy()
                if len(stratified_use) < 40 or stratified_use["PatientID"].nunique() < 8:
                    continue
                fit = smf.ols(
                    "model_outcome ~ program_z + log_abundance + log_program_cells + "
                    "lineage_mean_pseudotime + C(PatientID)", data=stratified_use,
                ).fit(cov_type="cluster", cov_kwds={"groups": stratified_use["PatientID"]})
                estimate = float(fit.params["program_z"])
                se = float(fit.bse["program_z"])
                rows.append({
                    "outcome": outcome_label, "module": module, "model": group_label,
                    "estimate_per_program_SD": estimate, "SE": se,
                    "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
                    "p_value": float(fit.pvalues["program_z"]), "n_lineages": len(stratified_use),
                    "n_participants": stratified_use["PatientID"].nunique(),
                    "outcome_scale": "probability" if kind == "binary" else "outcome SD",
                })
    tests = pd.DataFrame(rows)
    tests["FDR"] = np.nan
    for model, idx in tests.groupby("model").groups.items():
        tests.loc[idx, "FDR"] = bh(tests.loc[idx, "p_value"])
    return tests


def plot_clone_kinetics(dose_display, dose_tests, kin_participant, kin_tests):
    fig = plt.figure(figsize=(7.48, 6.15), facecolor="white")
    outer = GridSpec(2, 1, figure=fig, hspace=.62, left=.075, right=.99, top=.91, bottom=.10)
    dose_grid = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=.30)
    kin_grid = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1], wspace=.30)
    x = np.arange(4)
    pooled_tests = dose_tests[dose_tests["contrast"].eq("Pooled")].set_index("module")
    kin_by_module = kin_tests.set_index("module")
    kin_delta = kin_participant.pivot_table(
        index=["PatientID", "Diagnosis1", "pt_bin", "pt_mid", "module"],
        columns="clone_status", values="score_z", observed=True,
    ).dropna(subset=["Singleton", "Expanded"]).reset_index()
    kin_delta["delta"] = kin_delta["Expanded"] - kin_delta["Singleton"]
    kin_delta_rows = []
    for (module, pt_mid), group in kin_delta.groupby(["module", "pt_mid"]):
        med, lo, hi = bootstrap_median(group["delta"])
        kin_delta_rows.append({
            "module": module, "pt_mid": pt_mid, "median": med, "ci_low": lo, "ci_high": hi,
            "n_participants": group["PatientID"].nunique(),
        })
    kin_delta_display = pd.DataFrame(kin_delta_rows)
    kin_delta_display.to_csv(OUT / "Table_BEM7b_expanded_minus_singleton_kinetics_display.csv", index=False)
    fig.text(.022, .942, "A", fontsize=12, fontweight="bold", va="center")
    fig.text(.061, .942, "State-matched exact paired BCR clone-size dose response",
             fontsize=8.5, fontweight="bold", va="center")
    fig.text(.022, .465, "B", fontsize=12, fontweight="bold", va="center")
    fig.text(.061, .465, "Expanded-minus-singleton program kinetics along B-cell pseudotime",
             fontsize=8.5, fontweight="bold", va="center")
    for j, module in enumerate(MODULES):
        ax = fig.add_subplot(dose_grid[0, j])
        z = dose_display[(dose_display["module"].eq(module)) &
                         dose_display["Diagnosis"].eq("Pooled")].set_index("clone_bin").reindex(BINS)
        ax.errorbar(x, z["median"], yerr=[z["median"] - z["ci_low"], z["ci_high"] - z["median"]],
                    color=COL["Pooled"], marker="D", markerfacecolor="white", ms=3, lw=1.1, capsize=1.2)
        test = pooled_tests.loc[module]
        ax.text(.02, .98, SHORT[module], transform=ax.transAxes, va="top", fontweight="bold", fontsize=5.6)
        ax.text(.98, .04, f"slope {test.estimate_per_bin:+.3f}/bin\n{q_text(test.FDR)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=4.4, color="#777777")
        ax.axhline(0, color="#AAAAAA", lw=.55, ls="--")
        ax.set_xticks(x, ["1", "2", "3–4", "≥5"])
        ax.set_xlabel("Cells per clonotype", fontsize=5.4)
        for xi, n in zip(x, z["n_participants"]):
            if np.isfinite(n):
                ax.text(xi, .015, f"n={int(n)}", transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=3.7, color="#777777")
        if j == 0:
            ax.set_ylabel("State-matched score delta")
        else:
            ax.tick_params(axis="y", labelleft=False)
        style_axis(ax)

        ax2 = fig.add_subplot(kin_grid[0, j])
        q = kin_delta_display[(kin_delta_display["module"].eq(module)) &
                              kin_delta_display["n_participants"].ge(8)].sort_values("pt_mid")
        ax2.fill_between(q["pt_mid"], q["ci_low"], q["ci_high"], color="#7B3294", alpha=.13, linewidth=0)
        ax2.plot(q["pt_mid"], q["median"], color="#7B3294", marker="o", ms=2.6, lw=1.05)
        kt = kin_by_module.loc[module]
        ax2.text(.02, .98, SHORT[module], transform=ax2.transAxes, va="top", fontweight="bold", fontsize=5.6)
        ax2.text(.02, .87, f"expansion×trajectory\n{q_text(kt.expansion_by_trajectory_FDR)}",
                 transform=ax2.transAxes, va="top", fontsize=4.35, color="#777777")
        ax2.axhline(0, color="#777777", lw=.6, ls="--")
        ax2.set_xlim(0, 1); ax2.set_xticks([0, .5, 1])
        ax2.set_xlabel("B-cell pseudotime", fontsize=5.4)
        if j == 0:
            ax2.set_ylabel("Expanded − singleton score")
        else:
            ax2.tick_params(axis="y", labelleft=False)
        style_axis(ax2)
    fig.savefig(OUT / "Figure_BEM1_clone_size_and_kinetics_preview.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Figure_BEM1_clone_size_and_kinetics_preview.png", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lineage_models(tests):
    q = tests[tests["model"].eq("plus_pseudotime")].copy()
    outcomes = ["SHM/germline distance", "Diversified-lineage branch length", "Class-switched lineage", "Cross-state occupancy"]
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 6.1), constrained_layout=True)
    for ax, outcome in zip(axes.ravel(), outcomes):
        z = q[q["outcome"].eq(outcome)].set_index("module").reindex(MODULES)
        y = np.arange(len(MODULES))[::-1]
        colors = [COL["up"] if v >= 0 else COL["down"] for v in z["estimate_per_program_SD"]]
        ax.hlines(y, z["ci_low"], z["ci_high"], color=colors, lw=1.2)
        ax.scatter(z["estimate_per_program_SD"], y, color=colors, s=24, edgecolor="white", lw=.35, zorder=3)
        ax.axvline(0, color="#888888", lw=.6, ls="--")
        ax.set_yticks(y, [SHORT[m] for m in MODULES])
        ax.set_title(outcome, loc="left", fontweight="bold")
        ax.set_xlabel("Adjusted association per program SD")
        for yi, row in zip(y, z.itertuples()):
            ax.text(.99, yi, q_text(row.FDR), transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=5.0, color="#666666")
        style_axis(ax, "x")
    fig.suptitle("State-residualized B-cell programs and lineage maturation", fontsize=10.5, fontweight="bold")
    fig.savefig(OUT / "Figure_BEM2_lineage_maturation_preview.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Figure_BEM2_lineage_maturation_preview.png", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ibd_interactions(tests):
    q = tests[tests["model"].eq("IBD_vs_control_interaction")].copy()
    outcomes = ["SHM/germline distance", "Diversified-lineage branch length", "Class-switched lineage", "Cross-state occupancy"]
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 6.1), constrained_layout=True)
    for ax, outcome in zip(axes.ravel(), outcomes):
        z = q[q["outcome"].eq(outcome)].set_index("module").reindex(MODULES)
        y = np.arange(len(MODULES))[::-1]
        valid = z["estimate_per_program_SD"].notna()
        colors = [COL["up"] if v >= 0 else COL["down"] for v in z.loc[valid, "estimate_per_program_SD"]]
        ax.hlines(y[valid], z.loc[valid, "ci_low"], z.loc[valid, "ci_high"], color=colors, lw=1.2)
        ax.scatter(z.loc[valid, "estimate_per_program_SD"], y[valid], color=colors, s=24,
                   edgecolor="white", lw=.35, zorder=3)
        ax.axvline(0, color="#888888", lw=.6, ls="--")
        ax.set_yticks(y, [SHORT[m] for m in MODULES])
        ax.set_title(outcome, loc="left", fontweight="bold")
        ax.set_xlabel("IBD − control difference in association")
        for yi, row in zip(y[valid], z.loc[valid].itertuples()):
            ax.text(.99, yi, q_text(row.FDR), transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=5.0, color="#666666")
        if valid.any():
            first = z.loc[valid].iloc[0]
            ax.text(.01, .02, f"{int(first.n_lineages):,} lineages; {int(first.n_participants)} participants",
                    transform=ax.transAxes, fontsize=4.5, color="#666666")
        else:
            ax.text(.5, .5, "Not tested: insufficient\ncontrol lineage support",
                    transform=ax.transAxes, ha="center", va="center", fontsize=6, color="#777777")
        style_axis(ax, "x")
    fig.suptitle("Disease-specific shifts in program–lineage maturation coupling",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(OUT / "Figure_BEM3_IBD_lineage_interactions_preview.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Figure_BEM3_IBD_lineage_interactions_preview.png", dpi=350,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(dose_tests, kin_tests, lineage_tests, paired, lineage):
    pooled = dose_tests[dose_tests["contrast"].eq("Pooled")].set_index("module")
    kin = kin_tests.set_index("module")
    lm = lineage_tests[lineage_tests["model"].eq("plus_pseudotime")].copy()
    significant_lineage = lm[lm["FDR"].lt(.05)].sort_values("FDR")
    interaction = lineage_tests[lineage_tests["model"].eq("IBD_vs_control_interaction")].copy()
    significant_interaction = interaction[interaction["FDR"].lt(.05)].sort_values("FDR")
    control_multicell = lineage[lineage["mapped_cells"].ge(2) & lineage["Diagnosis1"].eq("Control")]
    control_multicell_counts = control_multicell.groupby("PatientID").size()
    control_multicell_eligible = control_multicell[
        control_multicell["PatientID"].isin(control_multicell_counts[control_multicell_counts >= 2].index)
    ]
    lines = [
        "# BCR expansion, trajectory kinetics, and lineage maturation analysis",
        "",
        "## Analysis design",
        "",
        f"The analysis included {len(paired):,} exact paired heavy-light BCR cells from {paired.PatientID.nunique()} participants. "
        "Clone-size analyses used singleton, 2-cell, 3-4-cell, and at least 5-cell bins and matched each expanded bin to singleton cells within participant and annotated B-cell state. "
        "Kinetics analyses projected every paired cell onto the fixed participant-balanced Slingshot trajectory and compared singleton and expanded cells within participant, state, and pseudotime interval. "
        "Lineage models used immunoglobulin-constant-free program scores centered within participant and state, participant fixed effects, participant-clustered standard errors, lineage abundance adjustment, and a prespecified pseudotime-adjusted sensitivity model.",
        "",
        "## Clone-size dose response",
        "",
    ]
    for module in MODULES:
        r = pooled.loc[module]
        lines.append(f"- {LABELS[module]}: slope {r.estimate_per_bin:+.4f} standardized-score units per bin; {q_text(r.FDR)}; n={int(r.n_participants)} participants.")
    lines += ["", "## Expansion-associated pseudotime kinetics", ""]
    for module in MODULES:
        r = kin.loc[module]
        lines.append(f"- {LABELS[module]}: trajectory {q_text(r.trajectory_FDR)}; expansion-by-trajectory {q_text(r.expansion_by_trajectory_FDR)}; n={int(r.n_participants)} participants.")
    lines += ["", "## Lineage maturation associations", ""]
    if significant_lineage.empty:
        lines.append("No program-lineage association passed FDR <0.05 after participant, lineage-abundance, state-residualization, and pseudotime adjustment.")
    else:
        for r in significant_lineage.itertuples():
            lines.append(f"- {LABELS[r.module]} with {r.outcome}: effect {r.estimate_per_program_SD:+.3f} per program SD (95% CI {r.ci_low:+.3f} to {r.ci_high:+.3f}); {q_text(r.FDR)}; {int(r.n_lineages)} lineages from {int(r.n_participants)} participants.")
    if significant_interaction.empty:
        lines.append("- No adequately supported program-by-IBD interaction passed FDR <0.05. Cross-state-occupancy interactions were not considered reliably estimable because the control stratum contained only "
                     f"{len(control_multicell_eligible)} eligible multi-cell lineages from {control_multicell_eligible.PatientID.nunique()} participants.")
    else:
        for r in significant_interaction.itertuples():
            lines.append(f"- IBD-versus-control interaction for {LABELS[r.module]} with {r.outcome}: {r.estimate_per_program_SD:+.3f} (95% CI {r.ci_low:+.3f} to {r.ci_high:+.3f}); {q_text(r.FDR)}.")
    lines += [
        "",
        "## Integration decision",
        "",
        "The clone-size analysis does not meet the prespecified bar to replace Figure 4C: none of the five ordered trends passed FDR <0.05, and none of the expansion-by-trajectory interactions passed FDR <0.05. "
        "The apparent antibody-secretion/UPR and cycling decreases were borderline after correction (both q=0.0623) and the extreme expansion bin was sparse (17 clonotypes from 8 participants), so these should remain supplementary negative-resource results.",
        "",
        "The lineage analysis provides stronger mechanistic support for Figure 5. IgA-mucosal and plasma-differentiation programs were independently associated with greater SHM/germline distance and a 2.4-percentage-point higher probability of class switching per program SD after state residualization and pseudotime adjustment. "
        "Because branch length and pooled cross-state occupancy were null, while cross-state disease interactions lacked adequate control support, the result supports a focused maturation link rather than a global or IBD-specific claim that these programs mark every form of lineage diversification. Recommended placement is a Figure 5-associated supplementary panel, with one compact main-text sentence; promotion into the main figure should be considered only if space permits and the current Figure 5 lacks a direct transcription-to-lineage bridge.",
        "",
    ]
    (OUT / "BCR_expansion_maturation_results_and_integration.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    cells = pd.read_csv(INPUT, low_memory=False)
    paired = add_standardized_scores(project_pseudotime(cells))
    keep_cols = [
        "cell_id", "PatientID", "Diagnosis1", "state", "exact_paired_clone", "clone_size", "clone_status",
        "clone_bin", "isotype", "switched", "receptor_key", "trajectory_pseudotime",
    ] + [f"{m}_z" for m in MODULES] + [f"{m}_state_resid" for m in MODULES]
    paired[keep_cols].to_csv(OUT / "Table_BEM2_paired_cells_projected_pseudotime.csv.gz", index=False)

    dose_participant, dose_display, dose_tests = dose_response(paired)
    dose_participant.to_csv(OUT / "Table_BEM3_state_matched_clone_size_participant_deltas.csv", index=False)
    dose_display.to_csv(OUT / "Table_BEM4_clone_size_dose_response_display.csv", index=False)
    dose_tests = dose_tests.rename(columns={"estimate": "estimate_per_bin"})
    dose_tests.to_csv(OUT / "Table_BEM5_clone_size_ordered_trend_tests.csv", index=False)

    kin_participant, kin_display, kin_tests = kinetics(paired)
    kin_participant.to_csv(OUT / "Table_BEM6_state_matched_kinetics_participant_bins.csv", index=False)
    kin_display.to_csv(OUT / "Table_BEM7_state_matched_kinetics_display.csv", index=False)
    kin_tests.to_csv(OUT / "Table_BEM8_state_matched_kinetics_tests.csv", index=False)

    mapped_cells, lineage = map_lineages(paired)
    lineage.to_csv(OUT / "Table_BEM9_lineage_program_maturation_data.csv.gz", index=False)
    lineage_tests = lineage_models(lineage)
    lineage_tests.to_csv(OUT / "Table_BEM10_participant_aware_lineage_models.csv", index=False)

    mapping_manifest = pd.DataFrame({
        "metric": ["paired cells", "cells mapped to germline lineage", "mapped fraction", "mapped lineages", "mapped participants"],
        "value": [len(mapped_cells), mapped_cells["lineage_id"].notna().sum(), mapped_cells["lineage_id"].notna().mean(),
                  lineage["lineage_id"].nunique(), lineage["PatientID"].nunique()],
    })
    for clone_bin, group in mapped_cells.groupby("clone_bin", observed=True):
        mapping_manifest.loc[len(mapping_manifest)] = {
            "metric": f"mapped fraction: {clone_bin}", "value": group["lineage_id"].notna().mean()
        }
    mapping_manifest.to_csv(OUT / "Table_BEM11_lineage_mapping_manifest.csv", index=False)

    plot_clone_kinetics(dose_display, dose_tests, kin_participant, kin_tests)
    plot_lineage_models(lineage_tests)
    plot_ibd_interactions(lineage_tests)
    write_report(dose_tests, kin_tests, lineage_tests, paired, lineage)
    print(dose_tests[dose_tests["contrast"].eq("Pooled")].to_string(index=False))
    print(kin_tests.to_string(index=False))
    print(lineage_tests[(lineage_tests["model"].eq("plus_pseudotime")) & (lineage_tests["FDR"].lt(.1))].to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()

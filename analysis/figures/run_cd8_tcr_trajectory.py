from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from patsy import bs
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "High Impact Additional Analyses" / "CD8 TCR Trajectory"
INPUT = OUT / "Table_TJ1_cd8_trajectory_input.csv.gz"
RNG = np.random.default_rng(20260828)

COL = {"Control": "#6F6F6F", "CD": "#0072B2", "UC": "#D55E00"}
STATUS_COL = {"Singleton": "#6F6F6F", "Expanded": "#D55E00"}
STATE_SHORT = {
    "CD8 Naive": "Naive",
    "CD8 Naive-IFN": "Naive-IFN",
    "CD8 Tcm CCR4-": "Tcm CCR4-",
    "CD8 Tem GZMK+": "Tem GZMK+",
    "CD8 Tem GZMB+": "Tem GZMB+",
    "CD8 Temra": "Temra",
    "CD8 HLA-DR+": "HLA-DR+",
    "CD8 Trm": "Trm",
    "CD8 Tmem KLRC2+": "Tmem KLRC2+",
}
MODULES = ["EOMES_ZEB2", "Cytotoxicity", "Th1_Tc1", "GZMK_inflammatory_memory"]
MODULE_LABELS = {
    "EOMES_ZEB2": "EOMES-ZEB2",
    "Cytotoxicity": "Cytotoxicity",
    "Th1_Tc1": "Th1/Tc1",
    "GZMK_inflammatory_memory": "GZMK inflammatory memory",
}


def bh(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    ok = values.notna()
    if not ok.any():
        return out
    p = values.loc[ok].to_numpy(float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(ranked)
    adj[order] = np.clip(ranked, 0, 1)
    out.loc[ok] = adj
    return out


def bootstrap_median(values: np.ndarray, n_boot: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    med = float(np.median(values))
    if len(values) < 2:
        return med, np.nan, np.nan
    sims = np.median(RNG.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return med, float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


def within_participant_slope(frame: pd.DataFrame, outcome: str, predictor: str) -> dict:
    use = frame[["SampleID", outcome, predictor]].dropna().copy()
    use["yw"] = use[outcome] - use.groupby("SampleID")[outcome].transform("mean")
    use["xw"] = use[predictor] - use.groupby("SampleID")[predictor].transform("mean")
    use = use[np.abs(use["xw"]) > 1e-12]
    fit = sm.OLS(use["yw"], use[["xw"]]).fit(
        cov_type="cluster", cov_kwds={"groups": use["SampleID"]}
    )
    estimate = float(fit.params["xw"])
    se = float(fit.bse["xw"])
    return {
        "estimate": estimate,
        "SE": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "p_value": float(fit.pvalues["xw"]),
        "n_clonotypes": len(use),
        "n_participants": use["SampleID"].nunique(),
    }


def fit_trajectory(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scvi_cols = [c for c in frame.columns if c.startswith("scvi_")]
    # Retain every paired-receptor cell in the fitted graph and add a
    # participant-by-state-stratified transcriptomic reference. This avoids an
    # expensive full 80k-cell eigendecomposition without projecting any cell
    # used in the repertoire-linked tests.
    shuffled = frame.sample(frac=1, random_state=20260828)
    stratified_index = (
        shuffled.groupby(["state", "SampleID"], sort=False).head(12).index
    )
    reference_mask = frame.index.isin(stratified_index) | frame["paired_alpha_beta"].to_numpy(bool)
    reference_index = np.flatnonzero(reference_mask)
    reference = frame.iloc[reference_index].copy()

    ref_x = reference[scvi_cols].to_numpy(np.float32)
    neighbor_model = NearestNeighbors(n_neighbors=36, metric="euclidean", n_jobs=-1).fit(ref_x)
    neighbor_distance, neighbor_index = neighbor_model.kneighbors(ref_x)
    # Remove self-neighbors and normalize distances by each cell's local scale.
    neighbor_distance = neighbor_distance[:, 1:]
    neighbor_index = neighbor_index[:, 1:]
    local_scale = np.median(neighbor_distance, axis=1)
    local_scale = np.clip(local_scale, 1e-8, None)
    normalized_distance = neighbor_distance / local_scale[:, None]
    row = np.repeat(np.arange(len(reference)), neighbor_index.shape[1])
    graph = coo_matrix(
        (normalized_distance.ravel(), (row, neighbor_index.ravel())),
        shape=(len(reference), len(reference)),
    ).tocsr()
    graph = graph.maximum(graph.T)

    naive = reference["state"].eq("CD8 Naive").to_numpy()
    early = reference["Early_memory"].to_numpy(float)
    cutoff = np.nanquantile(early[naive], 0.75)
    candidates = np.flatnonzero(naive & (early >= cutoff))
    center = np.nanmedian(ref_x[candidates], axis=0)
    distances = np.linalg.norm(ref_x[candidates] - center, axis=1)
    candidate_frame = pd.DataFrame({"idx": candidates, "distance": distances}).sort_values("distance")
    candidate_frame["SampleID"] = reference.iloc[candidate_frame["idx"]]["SampleID"].to_numpy()
    root_indices = candidate_frame.drop_duplicates("SampleID").head(12)["idx"].astype(int).to_numpy()
    root_primary, root_alt = int(root_indices[0]), int(root_indices[1])

    raw_distances = np.asarray(dijkstra(graph, directed=False, indices=root_indices), float)
    valid = np.all(np.isfinite(raw_distances), axis=0)
    root_pseudotimes = np.full_like(raw_distances, np.nan, dtype=float)
    for row_index in range(len(root_indices)):
        root_pseudotimes[row_index, valid] = (
            st.rankdata(raw_distances[row_index, valid], method="average") / valid.sum()
        )
    pt_primary = root_pseudotimes[0]
    pt_alt = root_pseudotimes[1]
    consensus_reference = np.nanmean(root_pseudotimes, axis=0)
    pairwise_root_rho = []
    for first in range(len(root_indices)):
        for second in range(first + 1, len(root_indices)):
            pairwise_root_rho.append(
                st.spearmanr(root_pseudotimes[first, valid], root_pseudotimes[second, valid]).statistic
            )

    all_x = frame[scvi_cols].to_numpy(np.float32)
    primary_model = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1).fit(
        ref_x[valid], pt_primary[valid]
    )
    alt_model = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1).fit(
        ref_x[valid], pt_alt[valid]
    )
    consensus_model = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1).fit(
        ref_x[valid], consensus_reference[valid]
    )
    primary_all = primary_model.predict(all_x)
    alt_all = alt_model.predict(all_x)
    consensus_all = consensus_model.predict(all_x)
    primary_all[reference_index[valid]] = pt_primary[valid]
    alt_all[reference_index[valid]] = pt_alt[valid]
    consensus_all[reference_index[valid]] = consensus_reference[valid]

    result = frame.copy()
    result["dpt_primary"] = primary_all
    result["dpt_alt_root"] = alt_all
    result["pseudotime"] = consensus_all
    result["trajectory_reference"] = reference_mask
    result["root_primary"] = False
    result["root_alt"] = False
    result.loc[result.index[reference_index[root_primary]], "root_primary"] = True
    result.loc[result.index[reference_index[root_alt]], "root_alt"] = True

    # State connectivity is summarized directly from cross-state kNN edges and
    # normalized by the geometric mean of state-specific edge opportunities.
    categories = sorted(reference["state"].unique())
    category_index = {state: i for i, state in enumerate(categories)}
    state_codes = reference["state"].map(category_index).to_numpy(int)
    edge_state_1 = state_codes[row]
    edge_state_2 = state_codes[neighbor_index.ravel()]
    counts = np.zeros((len(categories), len(categories)), dtype=float)
    np.add.at(counts, (edge_state_1, edge_state_2), 1)
    counts = counts + counts.T
    diagonal = np.clip(np.diag(counts), 1, None)
    conn = counts / np.sqrt(diagonal[:, None] * diagonal[None, :])
    np.fill_diagonal(conn, 0)
    paga_rows = []
    for i, first in enumerate(categories):
        for j in range(i + 1, len(categories)):
            paga_rows.append(
                {"state_1": first, "state_2": categories[j], "connectivity": float(conn[i, j])}
            )
    paga = pd.DataFrame(paga_rows).sort_values("connectivity", ascending=False)

    qc = pd.DataFrame(
        [
            {
                "metric": "Alternative-root pseudotime Spearman rho",
                "value": st.spearmanr(pt_primary[valid], pt_alt[valid]).statistic,
            },
            {
                "metric": "Median pairwise rho across 12 naive roots",
                "value": float(np.median(pairwise_root_rho)),
            },
            {
                "metric": "Minimum pairwise rho across 12 naive roots",
                "value": float(np.min(pairwise_root_rho)),
            },
            {
                "metric": "Consensus pseudotime vs early-memory score rho",
                "value": st.spearmanr(consensus_all, result["Early_memory"]).statistic,
            },
            {
                "metric": "Consensus pseudotime vs late-effector score rho",
                "value": st.spearmanr(consensus_all, result["Late_effector"]).statistic,
            },
            {"metric": "Cells in fitted trajectory reference", "value": int(reference_mask.sum())},
            {"metric": "Paired alpha-beta cells retained in reference", "value": int(reference["paired_alpha_beta"].sum())},
            {"metric": "Cells with finite pseudotime", "value": int(np.isfinite(consensus_all).sum())},
            {"metric": "Trajectory method", "value": "Consensus graph-geodesic pseudotime"},
        ]
    )
    return result, paga, qc


def summarize_states(cells: pd.DataFrame) -> pd.DataFrame:
    return (
        cells.groupby("state", as_index=False)
        .agg(
            n_cells=("cell", "size"),
            n_participants=("SampleID", "nunique"),
            median_pseudotime=("pseudotime", "median"),
            q1_pseudotime=("pseudotime", lambda x: x.quantile(0.25)),
            q3_pseudotime=("pseudotime", lambda x: x.quantile(0.75)),
            early_memory_score=("Early_memory", "median"),
            late_effector_score=("Late_effector", "median"),
        )
        .sort_values("median_pseudotime")
    )


def analyze_clones(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = cells[cells["paired_alpha_beta"] & cells["pseudotime"].notna()].copy()
    clones = (
        paired.groupby(["SampleID", "Diagnosis1", "clone_id"], as_index=False)
        .agg(
            clone_size=("cell", "size"),
            median_pseudotime=("pseudotime", "median"),
            q10_pseudotime=("pseudotime", lambda x: x.quantile(0.10)),
            q90_pseudotime=("pseudotime", lambda x: x.quantile(0.90)),
            n_states=("state", "nunique"),
        )
    )
    clones["pseudotime_span"] = clones["q90_pseudotime"] - clones["q10_pseudotime"]
    clones["log2_clone_size"] = np.log2(clones["clone_size"])
    clones["clone_status"] = np.where(clones["clone_size"] >= 2, "Expanded", "Singleton")
    clones["clone_bin"] = pd.cut(
        clones["clone_size"], bins=[0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", ">=5"]
    ).astype(str)

    bin_summary = (
        clones.groupby(["SampleID", "Diagnosis1", "clone_bin"], as_index=False)
        .agg(
            participant_median_pseudotime=("median_pseudotime", "median"),
            participant_median_span=("pseudotime_span", "median"),
            n_clonotypes=("clone_id", "size"),
        )
    )

    status = (
        clones.groupby(["SampleID", "Diagnosis1", "clone_status"], as_index=False)
        .agg(median_pseudotime=("median_pseudotime", "median"), n_clonotypes=("clone_id", "size"))
        .pivot(index=["SampleID", "Diagnosis1"], columns="clone_status", values="median_pseudotime")
        .dropna()
        .reset_index()
    )
    status["expanded_minus_singleton"] = status["Expanded"] - status["Singleton"]
    med, lo, hi = bootstrap_median(status["expanded_minus_singleton"].to_numpy())
    wilcox = st.wilcoxon(status["expanded_minus_singleton"], zero_method="wilcox")

    position = within_participant_slope(clones, "median_pseudotime", "log2_clone_size")
    span = within_participant_slope(
        clones[clones["clone_size"] >= 2], "pseudotime_span", "log2_clone_size"
    )
    tests = pd.DataFrame(
        [
            {"analysis": "Clone median pseudotime per log2 clone size", **position},
            {"analysis": "Clone pseudotime span per log2 clone size", **span},
            {
                "analysis": "Participant paired expanded-minus-singleton median pseudotime",
                "estimate": med,
                "SE": np.nan,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": float(wilcox.pvalue),
                "n_clonotypes": len(clones),
                "n_participants": len(status),
            },
        ]
    )
    tests["FDR"] = bh(tests["p_value"])
    return clones, bin_summary, status, tests


def program_kinetics(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = cells[cells["paired_alpha_beta"] & cells["pseudotime"].notna()].copy()
    for module in MODULES:
        paired[f"{module}_z"] = (cells.loc[paired.index, module] - cells[module].mean()) / cells[module].std()

    edges = np.quantile(paired["pseudotime"], np.linspace(0, 1, 11))
    edges = np.unique(edges)
    paired["pt_bin"] = pd.cut(paired["pseudotime"], bins=edges, include_lowest=True, duplicates="drop")
    paired["pt_mid"] = paired["pt_bin"].map(lambda x: float(x.mid)).astype(float)

    long = paired.melt(
        id_vars=["SampleID", "Diagnosis1", "clone_status", "pt_bin", "pt_mid"],
        value_vars=[f"{m}_z" for m in MODULES],
        var_name="module",
        value_name="score_z",
    )
    long["module"] = long["module"].str.replace("_z$", "", regex=True)
    participant_bins = (
        long.groupby(["SampleID", "Diagnosis1", "clone_status", "pt_bin", "pt_mid", "module"], observed=True, as_index=False)
        .agg(score_z=("score_z", "mean"), n_cells=("score_z", "size"))
    )

    display_rows = []
    for (module, status, pt_mid), group in participant_bins.groupby(["module", "clone_status", "pt_mid"]):
        values = group["score_z"].to_numpy(float)
        med, lo, hi = bootstrap_median(values, n_boot=3000)
        display_rows.append(
            {
                "module": module,
                "clone_status": status,
                "pt_mid": pt_mid,
                "median_score_z": med,
                "ci_low": lo,
                "ci_high": hi,
                "n_participants": group["SampleID"].nunique(),
                "n_cells": int(group["n_cells"].sum()),
            }
        )
    display = pd.DataFrame(display_rows)

    test_rows = []
    for module, group in participant_bins.groupby("module"):
        group = group.copy()
        fit = smf.ols(
            "score_z ~ bs(pt_mid, df=4, include_intercept=False) * C(clone_status, Treatment(reference='Singleton')) + C(SampleID)",
            data=group,
        ).fit(cov_type="cluster", cov_kwds={"groups": group["SampleID"]})
        terms = [
            name for name in fit.params.index
            if "bs(pt_mid" in name and ":C(clone_status" in name
        ]
        restriction = np.zeros((len(terms), len(fit.params)))
        for row, term in enumerate(terms):
            restriction[row, fit.params.index.get_loc(term)] = 1
        interaction = fit.wald_test(restriction, scalar=True)

        # The overall pseudotime association is tested in the singleton reference curve.
        main_terms = [
            name for name in fit.params.index
            if name.startswith("bs(pt_mid") and ":" not in name
        ]
        main_restriction = np.zeros((len(main_terms), len(fit.params)))
        for row, term in enumerate(main_terms):
            main_restriction[row, fit.params.index.get_loc(term)] = 1
        main_test = fit.wald_test(main_restriction, scalar=True)
        test_rows.append(
            {
                "module": module,
                "trajectory_p_value": float(main_test.pvalue),
                "expansion_by_trajectory_p_value": float(interaction.pvalue),
                "n_participant_bins": len(group),
                "n_participants": group["SampleID"].nunique(),
            }
        )
    tests = pd.DataFrame(test_rows)
    tests["trajectory_FDR"] = bh(tests["trajectory_p_value"])
    tests["expansion_by_trajectory_FDR"] = bh(tests["expansion_by_trajectory_p_value"])
    return participant_bins, display, tests


def plot_overview(
    cells: pd.DataFrame,
    states: pd.DataFrame,
    clone_bins: pd.DataFrame,
    clone_tests: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(7.5, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 0.88], hspace=0.48, wspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    chunks = []
    for _, group in cells.groupby("state"):
        chunks.append(group.sample(min(len(group), 4500), random_state=20260828))
    show = pd.concat(chunks).sort_values("pseudotime")
    points = ax.scatter(show["UMAP_1"], show["UMAP_2"], c=show["pseudotime"], cmap="viridis",
                        s=1.0, alpha=0.68, linewidth=0, rasterized=True)
    for state, group in cells.groupby("state"):
        ax.text(group["UMAP_1"].median(), group["UMAP_2"].median(), STATE_SHORT.get(state, state),
                fontsize=5.2, ha="center", va="center",
                path_effects=[], bbox={"fc": "white", "ec": "none", "alpha": 0.70, "pad": 0.4})
    root = cells[cells["root_primary"]]
    ax.scatter(root["UMAP_1"], root["UMAP_2"], marker="*", s=55, color="#D55E00",
               edgecolor="white", lw=0.5, zorder=5)
    cb = fig.colorbar(points, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Consensus graph-geodesic pseudotime")
    ax.set_xlabel("CD8 UMAP 1"); ax.set_ylabel("CD8 UMAP 2")
    ax.set_title("A  Conventional CD8 trajectory", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    ordered = states.sort_values("median_pseudotime")
    y = np.arange(len(ordered))[::-1]
    ax.hlines(y, ordered["q1_pseudotime"], ordered["q3_pseudotime"], color="#777777", lw=1.0)
    ax.scatter(ordered["median_pseudotime"], y, c=ordered["median_pseudotime"], cmap="viridis",
               vmin=0, vmax=1, s=30, edgecolor="white", lw=0.4, zorder=3)
    ax.set_yticks(y, [STATE_SHORT.get(x, x) for x in ordered["state"]])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Median pseudotime [IQR]")
    ax.set_title("B  State positions along the graph", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    order = ["1", "2", "3-4", ">=5"]
    values = [clone_bins.loc[clone_bins["clone_bin"] == b, "participant_median_pseudotime"].dropna().to_numpy() for b in order]
    violin = ax.violinplot(values, positions=np.arange(4), widths=0.76, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#D9D9D9"); body.set_edgecolor("none"); body.set_alpha(0.65)
    for x, b in enumerate(order):
        group = clone_bins[clone_bins["clone_bin"] == b]
        jitter = RNG.normal(x, 0.06, len(group))
        ax.scatter(jitter, group["participant_median_pseudotime"],
                   c=[COL.get(d, "#777777") for d in group["Diagnosis1"]], s=7, alpha=0.35,
                   edgecolor="none", rasterized=True)
        med, lo, hi = bootstrap_median(group["participant_median_pseudotime"].to_numpy(), 3000)
        ax.vlines(x, lo, hi, color="#222222", lw=1.5, zorder=5)
        ax.scatter(x, med, marker="D", s=30, color="#D55E00", edgecolor="white", lw=0.45, zorder=6)
    position = clone_tests.iloc[0]
    paired = clone_tests.iloc[2]
    ax.text(0.02, 0.98,
            f"Within-participant slope={position['estimate']:.3f} per doubling\n"
            f"95% CI {position['ci_low']:.3f} to {position['ci_high']:.3f}; q={position['FDR']:.2g}\n"
            f"Expanded-singleton median delta={paired['estimate']:.3f}; q={paired['FDR']:.2g}",
            transform=ax.transAxes, va="top", ha="left", fontsize=5.8)
    ax.set_xticks(np.arange(4), [f"{b}\n(n={clone_bins.loc[clone_bins['clone_bin']==b, 'SampleID'].nunique()})" for b in order])
    ax.set_xlabel("Exact paired alpha-beta clone size (cells)")
    ax.set_ylabel("Participant median clone pseudotime")
    ax.set_title("C  Larger clones occupy later positions", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    span_order = ["2", "3-4", ">=5"]
    span_values = [clone_bins.loc[clone_bins["clone_bin"] == b, "participant_median_span"].dropna().to_numpy() for b in span_order]
    violin = ax.violinplot(span_values, positions=np.arange(3), widths=0.76, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#D9D9D9"); body.set_edgecolor("none"); body.set_alpha(0.65)
    for x, b in enumerate(span_order):
        group = clone_bins[clone_bins["clone_bin"] == b]
        jitter = RNG.normal(x, 0.06, len(group))
        ax.scatter(jitter, group["participant_median_span"],
                   c=[COL.get(d, "#777777") for d in group["Diagnosis1"]], s=7, alpha=0.35,
                   edgecolor="none", rasterized=True)
        med, lo, hi = bootstrap_median(group["participant_median_span"].to_numpy(), 3000)
        ax.vlines(x, lo, hi, color="#222222", lw=1.5, zorder=5)
        ax.scatter(x, med, marker="D", s=30, color="#D55E00", edgecolor="white", lw=0.45, zorder=6)
    span = clone_tests.iloc[1]
    ax.text(0.02, 0.98,
            f"Within-participant slope={span['estimate']:.3f} per doubling\n"
            f"95% CI {span['ci_low']:.3f} to {span['ci_high']:.3f}; q={span['FDR']:.2g}",
            transform=ax.transAxes, va="top", ha="left", fontsize=5.8)
    ax.set_xticks(np.arange(3), [f"{b}\n(n={clone_bins.loc[clone_bins['clone_bin']==b, 'SampleID'].nunique()})" for b in span_order])
    ax.set_xlabel("Expanded clone size (cells)")
    ax.set_ylabel("Participant median within-clone pseudotime span")
    ax.set_title("D  Larger clones span more of the continuum", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_kinetics(display: pd.DataFrame, tests: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75), sharex=True, sharey=True)
    for ax, module in zip(axes, MODULES):
        dm = display[display["module"] == module]
        for status in ["Singleton", "Expanded"]:
            ds = dm[(dm["clone_status"] == status) & (dm["n_participants"] >= 8)].sort_values("pt_mid")
            ax.fill_between(ds["pt_mid"].to_numpy(float), ds["ci_low"].to_numpy(float),
                            ds["ci_high"].to_numpy(float), color=STATUS_COL[status], alpha=0.13, lw=0)
            ax.plot(ds["pt_mid"], ds["median_score_z"], color=STATUS_COL[status], lw=1.4,
                    marker="o" if status == "Expanded" else "s", ms=3.2, label=status)
        test = tests[tests["module"] == module].iloc[0]
        ax.text(0.03, 0.97,
                f"trajectory q={test['trajectory_FDR']:.2g}\nexpansion x trajectory q={test['expansion_by_trajectory_FDR']:.2g}",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.7)
        ax.axhline(0, color="#BBBBBB", lw=0.55)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Consensus graph-geodesic pseudotime")
        ax.set_title(MODULE_LABELS[module], fontweight="bold")
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Participant-level module score (z)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Expansion-associated program kinetics along the CD8 trajectory", x=0.01,
                 ha="left", fontweight="bold", fontsize=9)
    fig.subplots_adjust(top=0.80, wspace=0.18)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(INPUT, low_memory=False)
    cells, paga, trajectory_qc = fit_trajectory(cells)
    states = summarize_states(cells)
    clones, clone_bins, clone_status, clone_tests = analyze_clones(cells)
    participant_bins, kinetics_display, kinetics_tests = program_kinetics(cells)

    cells.to_csv(OUT / "Table_TJ2_cd8_cells_with_pseudotime.csv.gz", index=False)
    states.to_csv(OUT / "Table_TJ3_state_pseudotime_summary.csv", index=False)
    paga.to_csv(OUT / "Table_TJ4_paga_state_connectivity.csv", index=False)
    trajectory_qc.to_csv(OUT / "Table_TJ5_trajectory_qc.csv", index=False)
    clones.to_csv(OUT / "Table_TJ6_paired_alpha_beta_clone_trajectory.csv", index=False)
    clone_bins.to_csv(OUT / "Table_TJ7_participant_clone_bin_trajectory.csv", index=False)
    clone_status.to_csv(OUT / "Table_TJ8_participant_expanded_singleton_trajectory.csv", index=False)
    clone_tests.to_csv(OUT / "Table_TJ9_clone_trajectory_tests.csv", index=False)
    participant_bins.to_csv(OUT / "Table_TJ10_participant_program_kinetics.csv", index=False)
    kinetics_display.to_csv(OUT / "Table_TJ11_program_kinetics_display.csv", index=False)
    kinetics_tests.to_csv(OUT / "Table_TJ12_program_kinetics_tests.csv", index=False)

    plot_overview(cells, states, clone_bins, clone_tests, OUT / "Figure_TJ1_cd8_tcr_trajectory_overview.png")
    plot_kinetics(kinetics_display, kinetics_tests, OUT / "Figure_TJ2_program_kinetics.png")
    print(trajectory_qc.to_string(index=False))
    print(states[["state", "median_pseudotime", "q1_pseudotime", "q3_pseudotime"]].to_string(index=False))
    print(clone_tests.to_string(index=False))
    print(kinetics_tests.to_string(index=False))


if __name__ == "__main__":
    main()

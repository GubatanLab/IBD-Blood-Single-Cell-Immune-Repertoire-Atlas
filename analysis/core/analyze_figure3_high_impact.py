#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
TCR_DIR = ROOT / "High Impact Additional Analyses" / "Paired TCR Sequence State"
GLIPH_DIR = ROOT / "High Impact Additional Analyses" / "Priority Analyses"
OUT = ROOT / "High Impact Additional Analyses" / "Figure 3 High Impact Revision"
OUT.mkdir(parents=True, exist_ok=True)

NODES_PATH = TCR_DIR / "Table_TSS1_paired_alpha_beta_clone_sequence_state.csv"
EDGES_PATH = TCR_DIR / "Table_TSS3_paired_TCR_distance_graph_edges.csv"
GLIPH_PATH = GLIPH_DIR / "Table_PA5_IBDTCR_beta_only_GLIPH2_all_unique_clusters.csv"

MODULES = [
    "Th1_Tc1_inflammatory",
    "Effector_cytotoxicity",
    "EOMES_ZEB2_inflammatory_CD8_TRM_like",
    "Tissue_resident_mucosal_retention",
    "Gut_homing_intestinal_trafficking",
]

CONFIGS = [
    ("Baseline", "paired_vj", 5, 0.50, "unweighted"),
    ("k = 3", "paired_vj", 3, 0.50, "unweighted"),
    ("k = 10", "paired_vj", 10, 0.50, "unweighted"),
    ("Distance <= 0.40", "paired_vj", 5, 0.40, "unweighted"),
    ("Distance <= 0.60", "paired_vj", 5, 0.60, "unweighted"),
    ("CDR3 only", "paired_cdr3", 5, 0.50, "unweighted"),
    ("Alpha chain only", "alpha", 5, 0.50, "unweighted"),
    ("Beta chain only", "beta", 5, 0.50, "unweighted"),
    ("One neighbor", "paired_vj", 1, 0.50, "unweighted"),
    ("Degree weighted", "paired_vj", 5, 0.50, "degree"),
]

TRANSITION_CONFIGS = [
    ("Paired alpha-beta baseline", "paired_vj", "raw", "all"),
    ("Clone-size adjusted", "paired_vj", "clone_size_adjusted", "all"),
    ("Singleton clonotypes only", "paired_vj", "raw", "singleton"),
    ("Dominant-state adjusted", "paired_vj", "state_adjusted", "all"),
    ("Paired CDR3 only", "paired_cdr3", "raw", "all"),
    ("Alpha chain only", "alpha", "raw", "all"),
    ("Beta chain only", "beta", "raw", "all"),
]

PRIMARY_MODULES = [
    "EOMES_ZEB2_inflammatory_CD8_TRM_like",
    "Effector_cytotoxicity",
    "Th1_Tc1_inflammatory",
]


def kmer_tokens(sequence: str, prefix: str) -> list[str]:
    sequence = str(sequence)
    result: list[str] = []
    for k in (2, 3):
        result.extend(f"{prefix}{k}_{sequence[i:i+k]}" for i in range(max(0, len(sequence) - k + 1)))
    return result


def token_documents(frame: pd.DataFrame, feature_set: str) -> list[list[str]]:
    documents: list[list[str]] = []
    for row in frame.itertuples(index=False):
        tokens: list[str] = []
        if feature_set in {"paired_vj", "paired_cdr3", "alpha"}:
            tokens.extend(kmer_tokens(row.alpha_cdr3, "A"))
        if feature_set in {"paired_vj", "paired_cdr3", "beta"}:
            tokens.extend(kmer_tokens(row.beta_cdr3, "B"))
        if feature_set in {"paired_vj", "alpha"}:
            tokens.extend([f"AV_{row.alpha_v}", f"AV_{row.alpha_v}", f"AJ_{row.alpha_j}", f"AJ_{row.alpha_j}"])
        if feature_set in {"paired_vj", "beta"}:
            tokens.extend([f"BV_{row.beta_v}", f"BV_{row.beta_v}", f"BJ_{row.beta_j}", f"BJ_{row.beta_j}"])
        documents.append(tokens)
    return documents


def candidate_neighbors(frame: pd.DataFrame, feature_set: str):
    vectorizer = TfidfVectorizer(analyzer=lambda value: value, lowercase=False, norm="l2", min_df=2, dtype=np.float32)
    matrix = vectorizer.fit_transform(token_documents(frame, feature_set))
    n_neighbors = min(41, len(frame))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1).fit(matrix)
    return model.kneighbors(matrix, return_distance=True)


def derive_edges(frame: pd.DataFrame, distance, index, k_keep: int, max_distance: float) -> pd.DataFrame:
    exact = (
        frame.alpha_v.astype(str) + "|" + frame.alpha_j.astype(str) + "|" + frame.alpha_cdr3.astype(str) + "|"
        + frame.beta_v.astype(str) + "|" + frame.beta_j.astype(str) + "|" + frame.beta_cdr3.astype(str)
    ).to_numpy()
    participants = frame.SampleID.astype(str).to_numpy()
    rows = []
    for i in range(len(frame)):
        retained = 0
        for d, j in zip(distance[i, 1:], index[i, 1:]):
            j = int(j)
            if participants[i] == participants[j] or exact[i] == exact[j] or d > max_distance:
                continue
            a, b = (i, j) if i < j else (j, i)
            rows.append((a, b, float(d)))
            retained += 1
            if retained >= k_keep:
                break
    if not rows:
        return pd.DataFrame(columns=["i", "j", "distance"])
    return pd.DataFrame(rows, columns=["i", "j", "distance"]).drop_duplicates(["i", "j"]).reset_index(drop=True)


def participant_centered(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce")
    mean = values.groupby(frame.SampleID).transform("mean")
    sd = values.groupby(frame.SampleID).transform("std").replace(0, np.nan)
    return ((values - mean) / sd).fillna(0).to_numpy(float)


def clone_size_adjusted(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Remove a common within-participant log clone-size slope, then standardize within participant."""
    values = pd.to_numeric(frame[column], errors="coerce")
    clone_size = np.log1p(pd.to_numeric(frame.n_cells, errors="coerce").fillna(1))
    centered_values = values - values.groupby(frame.SampleID).transform("mean")
    centered_size = clone_size - clone_size.groupby(frame.SampleID).transform("mean")
    valid = centered_values.notna() & centered_size.notna()
    denominator = float(np.sum(np.square(centered_size[valid])))
    slope = float(np.sum(centered_values[valid] * centered_size[valid]) / denominator) if denominator > 0 else 0.0
    residual = centered_values - slope * centered_size
    residual_sd = residual.groupby(frame.SampleID).transform("std").replace(0, np.nan)
    return (residual / residual_sd).fillna(0).to_numpy(float)


def state_adjusted(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Remove dominant-state means, then standardize residual scores within participant."""
    values = pd.to_numeric(frame[column], errors="coerce")
    state_mean = values.groupby(frame.dominant_state.astype(str)).transform("mean")
    residual = values - state_mean
    participant_mean = residual.groupby(frame.SampleID).transform("mean")
    participant_sd = residual.groupby(frame.SampleID).transform("std").replace(0, np.nan)
    return ((residual - participant_mean) / participant_sd).fillna(0).to_numpy(float)


def transformed_values(frame: pd.DataFrame, column: str, adjustment: str) -> np.ndarray:
    if adjustment == "clone_size_adjusted":
        return clone_size_adjusted(frame, column)
    if adjustment == "state_adjusted":
        return state_adjusted(frame, column)
    return participant_centered(frame, column)


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 20:
        return np.nan
    x, y, weights = x[valid], y[valid], weights[valid]
    weights = weights / weights.sum()
    mx, my = np.sum(weights * x), np.sum(weights * y)
    vx, vy = np.sum(weights * (x - mx) ** 2), np.sum(weights * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(np.sum(weights * (x - mx) * (y - my)) / math.sqrt(vx * vy))


def graph_stat(values: np.ndarray, edges: pd.DataFrame, mode: str) -> float:
    ii, jj = edges.i.to_numpy(int), edges.j.to_numpy(int)
    if mode == "degree":
        degree = np.bincount(np.concatenate([ii, jj]), minlength=len(values)).astype(float)
        weights = 1 / np.sqrt(np.maximum(degree[ii] * degree[jj], 1))
    else:
        weights = np.ones(len(edges), float)
    return weighted_corr(values[ii], values[jj], weights)


def participant_jackknife(frame: pd.DataFrame, edges: pd.DataFrame, values: np.ndarray, mode: str):
    estimate = graph_stat(values, edges, mode)
    participants = frame.SampleID.astype(str).to_numpy()
    ii, jj = edges.i.to_numpy(int), edges.j.to_numpy(int)
    estimates = []
    for participant in pd.unique(participants):
        keep = (participants[ii] != participant) & (participants[jj] != participant)
        if keep.sum() < 20:
            continue
        estimates.append(graph_stat(values, edges.loc[keep].reset_index(drop=True), mode))
    estimates = np.asarray([x for x in estimates if np.isfinite(x)], float)
    if len(estimates) < 3:
        return estimate, np.nan, len(estimates)
    mean = estimates.mean()
    se = math.sqrt((len(estimates) - 1) / len(estimates) * np.sum((estimates - mean) ** 2))
    return estimate, se, len(estimates)


def random_effect_pool(frame: pd.DataFrame):
    frame = frame.dropna(subset=["edge_correlation", "participant_jackknife_se"])
    y = frame.edge_correlation.to_numpy(float)
    v = np.square(np.clip(frame.participant_jackknife_se.to_numpy(float), 1e-5, None))
    if len(y) == 0:
        return (np.nan,) * 7
    fixed_weights = 1 / v
    fixed = np.sum(fixed_weights * y) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * (y - fixed) ** 2))
    df = max(len(y) - 1, 0)
    c = np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights)
    tau2 = max(0.0, (q - df) / c) if df > 0 and c > 0 else 0.0
    weights = 1 / (v + tau2)
    estimate = float(np.sum(weights * y) / np.sum(weights))
    se = math.sqrt(1 / np.sum(weights))
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 and df > 0 else 0.0
    return estimate, estimate - 1.96 * se, estimate + 1.96 * se, tau2, i2, q, len(y)


def run_graph_robustness(nodes: pd.DataFrame):
    by_series = []
    for series, original in nodes.groupby("acquisition_series", sort=True):
        frame = original.reset_index(drop=True)
        neighbor_cache = {}
        for feature_set in sorted({x[1] for x in CONFIGS}):
            neighbor_cache[feature_set] = candidate_neighbors(frame, feature_set)
        edge_cache = {}
        for label, feature_set, k_keep, max_distance, mode in CONFIGS:
            edge_key = (feature_set, k_keep, max_distance)
            if edge_key not in edge_cache:
                edge_cache[edge_key] = derive_edges(frame, *neighbor_cache[feature_set], k_keep, max_distance)
            edges = edge_cache[edge_key]
            if len(edges) < 50:
                continue
            for module in MODULES:
                values = participant_centered(frame, module)
                estimate, se, n_jackknife = participant_jackknife(frame, edges, values, mode)
                by_series.append(
                    {
                        "configuration": label,
                        "feature_set": feature_set,
                        "k_keep": k_keep,
                        "max_distance": max_distance,
                        "weighting": mode,
                        "acquisition_series": series,
                        "module": module,
                        "n_nodes": len(frame),
                        "n_participants": frame.SampleID.nunique(),
                        "n_edges": len(edges),
                        "edge_correlation": estimate,
                        "participant_jackknife_se": se,
                        "n_jackknife_replicates": n_jackknife,
                    }
                )
        print(f"completed {series}", flush=True)
    by_series = pd.DataFrame(by_series)
    meta_rows = []
    for (configuration, module), frame in by_series.groupby(["configuration", "module"], sort=False):
        estimate, low, high, tau2, i2, q, n_series = random_effect_pool(frame)
        meta_rows.append(
            {
                "configuration": configuration,
                "module": module,
                "pooled_edge_correlation": estimate,
                "ci_low": low,
                "ci_high": high,
                "tau2": tau2,
                "I2_percent": i2,
                "Q": q,
                "n_series": n_series,
                "n_series_positive": int((frame.edge_correlation > 0).sum()),
                "total_edges": int(frame.n_edges.sum()),
            }
        )
    meta = pd.DataFrame(meta_rows)
    by_series.to_csv(OUT / "Table_F3R1_graph_robustness_by_series.csv", index=False)
    meta.to_csv(OUT / "Table_F3R2_graph_robustness_meta_analysis.csv", index=False)
    return by_series, meta


def run_transition_robustness(nodes: pd.DataFrame):
    """Sensitivity set that directly tests whether Figure 3 extends beyond Figure 2 clone-size effects."""
    by_series_rows = []
    for series, original in nodes.groupby("acquisition_series", sort=True):
        frames = {
            "all": original.reset_index(drop=True),
            "singleton": original[pd.to_numeric(original.n_cells, errors="coerce").fillna(1).eq(1)].reset_index(drop=True),
        }
        neighbor_cache = {}
        edge_cache = {}
        for label, feature_set, adjustment, subset in TRANSITION_CONFIGS:
            frame = frames[subset]
            neighbor_key = (subset, feature_set)
            if neighbor_key not in neighbor_cache:
                neighbor_cache[neighbor_key] = candidate_neighbors(frame, feature_set)
            edge_key = (subset, feature_set)
            if edge_key not in edge_cache:
                edge_cache[edge_key] = derive_edges(frame, *neighbor_cache[neighbor_key], 5, 0.50)
            edges = edge_cache[edge_key]
            if len(edges) < 50:
                continue
            for module in MODULES:
                values = transformed_values(frame, module, adjustment)
                estimate, se, n_jackknife = participant_jackknife(frame, edges, values, "unweighted")
                by_series_rows.append(
                    {
                        "configuration": label,
                        "feature_set": feature_set,
                        "score_adjustment": adjustment,
                        "node_subset": subset,
                        "acquisition_series": series,
                        "module": module,
                        "n_nodes": len(frame),
                        "n_participants": frame.SampleID.nunique(),
                        "n_edges": len(edges),
                        "edge_correlation": estimate,
                        "participant_jackknife_se": se,
                        "n_jackknife_replicates": n_jackknife,
                    }
                )
        print(f"completed transition sensitivities {series}", flush=True)

    by_series = pd.DataFrame(by_series_rows)
    meta_rows = []
    for (configuration, module), frame in by_series.groupby(["configuration", "module"], sort=False):
        estimate, low, high, tau2, i2, q, n_series = random_effect_pool(frame)
        first = frame.iloc[0]
        meta_rows.append(
            {
                "configuration": configuration,
                "feature_set": first.feature_set,
                "score_adjustment": first.score_adjustment,
                "node_subset": first.node_subset,
                "module": module,
                "pooled_edge_correlation": estimate,
                "ci_low": low,
                "ci_high": high,
                "tau2": tau2,
                "I2_percent": i2,
                "Q": q,
                "n_series": n_series,
                "n_series_positive": int((frame.edge_correlation > 0).sum()),
                "total_edges": int(frame.n_edges.sum()),
                "minimum_nodes_per_series": int(frame.n_nodes.min()),
                "maximum_nodes_per_series": int(frame.n_nodes.max()),
            }
        )
    meta = pd.DataFrame(meta_rows)
    by_series.to_csv(OUT / "Table_F3R14_transition_robustness_by_series.csv", index=False)
    meta.to_csv(OUT / "Table_F3R15_transition_robustness_meta_analysis.csv", index=False)
    source = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
    source.mkdir(parents=True, exist_ok=True)
    by_series.to_csv(source / "Figure_3_R14_transition_robustness_by_series.csv", index=False)
    meta.to_csv(source / "Figure_3_R15_transition_robustness_meta_analysis.csv", index=False)
    return by_series, meta


def run_primary_attenuation_bootstrap(
    nodes: pd.DataFrame,
    graph_edges: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 20260829,
):
    """Paired participant-block bootstrap tests of baseline-to-adjusted attenuation.

    Participants are resampled within acquisition series. The sequence graph remains fixed, and an edge receives the
    product of the bootstrap multiplicities of its two endpoint participants. Series are pooled with the fixed
    random-effects weights from the displayed transition analysis, preserving the paired covariance between baseline
    and adjusted estimates without adding a batch term.
    """
    by_series = pd.read_csv(OUT / "Table_F3R14_transition_robustness_by_series.csv", low_memory=False)
    meta = pd.read_csv(OUT / "Table_F3R15_transition_robustness_meta_analysis.csv", low_memory=False)
    comparison_adjustments = {
        "Clone-size adjusted": "clone_size_adjusted",
        "Dominant-state adjusted": "state_adjusted",
    }

    prepared = {}
    for series, original in nodes.groupby("acquisition_series", sort=True):
        frame = original.reset_index(drop=True)
        stored = graph_edges[graph_edges.acquisition_series.eq(series)]
        edges = baseline_edges_for_frame(frame, stored)
        if len(edges) < 50:
            continue
        participants = frame.SampleID.astype(str).to_numpy()
        unique_participants, participant_codes = np.unique(participants, return_inverse=True)
        ii, jj = edges.i.to_numpy(int), edges.j.to_numpy(int)
        values = {}
        for module in PRIMARY_MODULES:
            values[("Paired alpha-beta baseline", module)] = transformed_values(frame, module, "raw")
            for label, adjustment in comparison_adjustments.items():
                values[(label, module)] = transformed_values(frame, module, adjustment)
        prepared[str(series)] = {
            "ii": ii,
            "jj": jj,
            "endpoint1_participant": participant_codes[ii],
            "endpoint2_participant": participant_codes[jj],
            "n_participants": len(unique_participants),
            "values": values,
        }

    fixed_weights = {}
    for configuration in ["Paired alpha-beta baseline", *comparison_adjustments]:
        for module in PRIMARY_MODULES:
            subset = by_series[
                by_series.configuration.eq(configuration) & by_series.module.eq(module)
            ].copy()
            tau2 = float(
                meta.loc[
                    meta.configuration.eq(configuration) & meta.module.eq(module), "tau2"
                ].iloc[0]
            )
            subset["meta_weight"] = 1 / (
                np.square(np.clip(subset.participant_jackknife_se.to_numpy(float), 1e-5, None)) + tau2
            )
            fixed_weights[(configuration, module)] = dict(
                zip(subset.acquisition_series.astype(str), subset.meta_weight.to_numpy(float))
            )

    def fixed_pool(series_values: dict[str, float], configuration: str, module: str) -> float:
        weight_map = fixed_weights[(configuration, module)]
        retained = [
            (value, weight_map.get(series, np.nan))
            for series, value in series_values.items()
            if np.isfinite(value) and np.isfinite(weight_map.get(series, np.nan))
        ]
        if not retained:
            return np.nan
        values_array = np.asarray([item[0] for item in retained], float)
        weights_array = np.asarray([item[1] for item in retained], float)
        return float(np.sum(values_array * weights_array) / np.sum(weights_array))

    rng = np.random.default_rng(seed)
    boot_differences = {
        (comparison, module): np.full(n_bootstrap, np.nan, float)
        for comparison in comparison_adjustments
        for module in PRIMARY_MODULES
    }
    for bootstrap_index in range(n_bootstrap):
        replicate_estimates = {
            (configuration, module): {}
            for configuration in ["Paired alpha-beta baseline", *comparison_adjustments]
            for module in PRIMARY_MODULES
        }
        for series, data in prepared.items():
            participant_counts = rng.multinomial(
                data["n_participants"], np.repeat(1 / data["n_participants"], data["n_participants"])
            )
            edge_weights = (
                participant_counts[data["endpoint1_participant"]]
                * participant_counts[data["endpoint2_participant"]]
            ).astype(float)
            for key, values in data["values"].items():
                replicate_estimates[key][series] = weighted_corr(
                    values[data["ii"]], values[data["jj"]], edge_weights
                )
        for comparison in comparison_adjustments:
            for module in PRIMARY_MODULES:
                baseline_pool = fixed_pool(
                    replicate_estimates[("Paired alpha-beta baseline", module)],
                    "Paired alpha-beta baseline",
                    module,
                )
                comparison_pool = fixed_pool(
                    replicate_estimates[(comparison, module)], comparison, module
                )
                boot_differences[(comparison, module)][bootstrap_index] = baseline_pool - comparison_pool
        if (bootstrap_index + 1) % 250 == 0:
            print(f"completed attenuation bootstrap {bootstrap_index + 1}/{n_bootstrap}", flush=True)

    module_rows = []
    omnibus_rows = []
    for comparison in comparison_adjustments:
        comparison_p_values = []
        comparison_rows = []
        bootstrap_matrix = []
        observed_vector = []
        for module in PRIMARY_MODULES:
            baseline_row = meta[
                meta.configuration.eq("Paired alpha-beta baseline") & meta.module.eq(module)
            ].iloc[0]
            comparison_row = meta[
                meta.configuration.eq(comparison) & meta.module.eq(module)
            ].iloc[0]
            observed = float(baseline_row.pooled_edge_correlation - comparison_row.pooled_edge_correlation)
            values = boot_differences[(comparison, module)]
            values = values[np.isfinite(values)]
            bootstrap_se = float(np.std(values, ddof=1))
            wald_p = float(chi2.sf(np.square(observed / bootstrap_se), 1)) if bootstrap_se > 0 else np.nan
            comparison_p_values.append(wald_p)
            comparison_rows.append(
                {
                    "comparison": comparison,
                    "module": module,
                    "baseline_estimate": float(baseline_row.pooled_edge_correlation),
                    "comparison_estimate": float(comparison_row.pooled_edge_correlation),
                    "attenuation_baseline_minus_comparison": observed,
                    "bootstrap_ci_low": float(np.quantile(values, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_se": bootstrap_se,
                    "wald_p": wald_p,
                    "n_bootstrap": len(values),
                    "n_series": len(prepared),
                    "resampling_unit": "participant within acquisition series",
                    "batch_term_included": False,
                }
            )
            bootstrap_matrix.append(values)
            observed_vector.append(observed)
        adjusted = bh_adjust(comparison_p_values)
        for row, fdr in zip(comparison_rows, adjusted):
            row["FDR_within_comparison"] = float(fdr)
            module_rows.append(row)

        matrix = np.column_stack(bootstrap_matrix)
        covariance = np.cov(matrix, rowvar=False, ddof=1)
        observed_array = np.asarray(observed_vector, float)
        degrees_freedom = int(np.linalg.matrix_rank(covariance))
        statistic = float(observed_array @ np.linalg.pinv(covariance) @ observed_array)
        omnibus_rows.append(
            {
                "comparison": comparison,
                "wald_chi_square": statistic,
                "degrees_freedom": degrees_freedom,
                "omnibus_p": float(chi2.sf(statistic, degrees_freedom)),
                "n_bootstrap": n_bootstrap,
                "n_series": len(prepared),
                "resampling_unit": "participant within acquisition series",
                "batch_term_included": False,
            }
        )

    module_results = pd.DataFrame(module_rows)
    omnibus_results = pd.DataFrame(omnibus_rows)
    module_results.to_csv(OUT / "Table_F3R23_primary_attenuation_bootstrap.csv", index=False)
    omnibus_results.to_csv(OUT / "Table_F3R24_primary_attenuation_omnibus.csv", index=False)
    source = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
    source.mkdir(parents=True, exist_ok=True)
    module_results.to_csv(source / "Figure_3_R23_primary_attenuation_bootstrap.csv", index=False)
    omnibus_results.to_csv(source / "Figure_3_R24_primary_attenuation_omnibus.csv", index=False)
    return module_results, omnibus_results


def baseline_edges_for_frame(frame: pd.DataFrame, edge_rows: pd.DataFrame) -> pd.DataFrame:
    """Translate stored graph node IDs to local row indices without rebuilding the graph."""
    local = frame.reset_index(drop=True).copy()
    node_ids = local.SampleID.astype(str) + "::" + local.clone_id.astype(str)
    index = pd.Series(np.arange(len(local), dtype=int), index=node_ids).to_dict()
    result = edge_rows[["node1_id", "node2_id", "sequence_distance"]].copy()
    result["i"] = result.node1_id.map(index)
    result["j"] = result.node2_id.map(index)
    result = result.dropna(subset=["i", "j"]).copy()
    result[["i", "j"]] = result[["i", "j"]].astype(int)
    return result[["i", "j", "sequence_distance"]].rename(columns={"sequence_distance": "distance"}).reset_index(drop=True)


def run_diagnosis_convergence(nodes: pd.DataFrame, graph_edges: pd.DataFrame):
    """Estimate baseline sequence-program convergence within CD, UC, and control participants."""
    rows = []
    for diagnosis in ("CD", "UC", "Control"):
        for series, original in nodes[nodes.Diagnosis.eq(diagnosis)].groupby("acquisition_series", sort=True):
            frame = original.reset_index(drop=True)
            stored = graph_edges[
                graph_edges.acquisition_series.eq(series)
                & graph_edges.node1_diagnosis.eq(diagnosis)
                & graph_edges.node2_diagnosis.eq(diagnosis)
            ]
            edges = baseline_edges_for_frame(frame, stored)
            if len(edges) < 100:
                continue
            ii, jj = edges.i.to_numpy(int), edges.j.to_numpy(int)
            participants = frame.SampleID.astype(str).to_numpy()
            participants_with_edges = len(set(participants[ii]) | set(participants[jj]))
            if participants_with_edges < 6:
                continue
            for module in PRIMARY_MODULES:
                values = participant_centered(frame, module)
                estimate, se, n_jackknife = participant_jackknife(frame, edges, values, "unweighted")
                rows.append(
                    {
                        "diagnosis": diagnosis,
                        "acquisition_series": series,
                        "module": module,
                        "n_nodes": len(frame),
                        "n_participants": frame.SampleID.nunique(),
                        "n_participants_with_edges": participants_with_edges,
                        "n_edges": len(edges),
                        "edge_correlation": estimate,
                        "participant_jackknife_se": se,
                        "n_jackknife_replicates": n_jackknife,
                    }
                )
    by_series = pd.DataFrame(rows)
    meta_rows = []
    for (diagnosis, module), frame in by_series.groupby(["diagnosis", "module"], sort=False):
        estimate, low, high, tau2, i2, q, n_series = random_effect_pool(frame)
        meta_rows.append(
            {
                "diagnosis": diagnosis,
                "module": module,
                "pooled_edge_correlation": estimate,
                "ci_low": low,
                "ci_high": high,
                "tau2": tau2,
                "I2_percent": i2,
                "Q": q,
                "n_series": n_series,
                "n_series_positive": int((frame.edge_correlation > 0).sum()),
                "total_edges": int(frame.n_edges.sum()),
                "minimum_participants_with_edges": int(frame.n_participants_with_edges.min()),
                "maximum_participants_with_edges": int(frame.n_participants_with_edges.max()),
            }
        )
    meta = pd.DataFrame(meta_rows)
    by_series.to_csv(OUT / "Table_F3R18_diagnosis_convergence_by_series.csv", index=False)
    meta.to_csv(OUT / "Table_F3R19_diagnosis_convergence_meta_analysis.csv", index=False)
    source = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
    source.mkdir(parents=True, exist_ok=True)
    by_series.to_csv(source / "Figure_3_R18_diagnosis_convergence_by_series.csv", index=False)
    meta.to_csv(source / "Figure_3_R19_diagnosis_convergence_meta_analysis.csv", index=False)
    return by_series, meta


def run_diagnosis_interaction(
    nodes: pd.DataFrame, graph_edges: pd.DataFrame, diagnosis_by_series: pd.DataFrame, n_bootstrap: int = 2_000
):
    """Participant-block bootstrap test for any CD-versus-UC difference across the three primary programs."""
    diagnosis_data = {}
    eligible = diagnosis_by_series[["diagnosis", "acquisition_series"]].drop_duplicates()
    for diagnosis in ("CD", "UC"):
        series = set(eligible.loc[eligible.diagnosis.eq(diagnosis), "acquisition_series"])
        frame = nodes[nodes.Diagnosis.eq(diagnosis) & nodes.acquisition_series.isin(series)].reset_index(drop=True)
        stored = graph_edges[
            graph_edges.acquisition_series.isin(series)
            & graph_edges.node1_diagnosis.eq(diagnosis)
            & graph_edges.node2_diagnosis.eq(diagnosis)
        ]
        edges = baseline_edges_for_frame(frame, stored)
        ii, jj = edges.i.to_numpy(int), edges.j.to_numpy(int)
        node_participants = frame.SampleID.astype(str).to_numpy()
        edge_participants = sorted(set(node_participants[ii]) | set(node_participants[jj]))
        participant_index = {participant: index for index, participant in enumerate(edge_participants)}
        left_participant = np.asarray([participant_index[node_participants[index]] for index in ii], int)
        right_participant = np.asarray([participant_index[node_participants[index]] for index in jj], int)
        values = np.column_stack([participant_centered(frame, module) for module in PRIMARY_MODULES])
        point = np.asarray(
            [weighted_corr(values[ii, column], values[jj, column], np.ones(len(edges))) for column in range(3)],
            float,
        )
        diagnosis_data[diagnosis] = {
            "ii": ii,
            "jj": jj,
            "left_participant": left_participant,
            "right_participant": right_participant,
            "n_participants": len(edge_participants),
            "n_edges": len(edges),
            "values": values,
            "point": point,
        }

    rng = np.random.default_rng(20260828)
    bootstrap_differences = []
    for _ in range(n_bootstrap):
        estimates = {}
        for diagnosis in ("CD", "UC"):
            data = diagnosis_data[diagnosis]
            n_participants = data["n_participants"]
            sampled = rng.integers(0, n_participants, size=n_participants)
            multiplicity = np.bincount(sampled, minlength=n_participants).astype(float)
            weights = multiplicity[data["left_participant"]] * multiplicity[data["right_participant"]]
            estimates[diagnosis] = np.asarray(
                [
                    weighted_corr(
                        data["values"][data["ii"], column],
                        data["values"][data["jj"], column],
                        weights,
                    )
                    for column in range(3)
                ],
                float,
            )
        difference = estimates["UC"] - estimates["CD"]
        if np.all(np.isfinite(difference)):
            bootstrap_differences.append(difference)
    bootstrap_differences = np.asarray(bootstrap_differences, float)
    observed_difference = diagnosis_data["UC"]["point"] - diagnosis_data["CD"]["point"]
    covariance = np.cov(bootstrap_differences, rowvar=False, ddof=1)
    wald = float(observed_difference @ np.linalg.pinv(covariance) @ observed_difference)
    omnibus_p = float(chi2.sf(wald, df=len(PRIMARY_MODULES)))

    rows = []
    for column, module in enumerate(PRIMARY_MODULES):
        differences = bootstrap_differences[:, column]
        lower_tail = np.sum(differences <= 0)
        upper_tail = np.sum(differences >= 0)
        bootstrap_p = min(1.0, 2 * (1 + min(lower_tail, upper_tail)) / (len(differences) + 1))
        rows.append(
            {
                "module": module,
                "CD_aggregate_edge_correlation": diagnosis_data["CD"]["point"][column],
                "UC_aggregate_edge_correlation": diagnosis_data["UC"]["point"][column],
                "UC_minus_CD_difference": observed_difference[column],
                "difference_ci_low": float(np.quantile(differences, 0.025)),
                "difference_ci_high": float(np.quantile(differences, 0.975)),
                "difference_bootstrap_p": bootstrap_p,
                "omnibus_wald_statistic": wald,
                "omnibus_df": len(PRIMARY_MODULES),
                "omnibus_p": omnibus_p,
                "n_bootstrap": len(bootstrap_differences),
                "CD_participants": diagnosis_data["CD"]["n_participants"],
                "UC_participants": diagnosis_data["UC"]["n_participants"],
                "CD_edges": diagnosis_data["CD"]["n_edges"],
                "UC_edges": diagnosis_data["UC"]["n_edges"],
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "Table_F3R22_diagnosis_interaction_test.csv", index=False)
    source = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
    source.mkdir(parents=True, exist_ok=True)
    result.to_csv(source / "Figure_3_R22_diagnosis_interaction_test.csv", index=False)
    return result


def expected_inflammation_composition(edge_frame: pd.DataFrame, participant_frame: pd.DataFrame):
    """Exact expectation under participant-label exchangeability within diagnosis."""
    expected = {"Inflamed-Inflamed": 0.0, "Noninflamed-Noninflamed": 0.0, "Mixed": 0.0}
    total_edges = len(edge_frame)
    if total_edges == 0:
        return expected
    for diagnosis, diagnosis_edges in edge_frame.groupby("diagnosis"):
        participants = participant_frame[participant_frame.Diagnosis.eq(diagnosis)]
        n = len(participants)
        n_inflamed = int(participants.Inflammation.eq("Inflamed").sum())
        n_noninflamed = int(participants.Inflammation.eq("Noninflamed").sum())
        denominator = n * (n - 1)
        if denominator <= 0:
            continue
        weight = len(diagnosis_edges) / total_edges
        expected["Inflamed-Inflamed"] += weight * n_inflamed * (n_inflamed - 1) / denominator
        expected["Noninflamed-Noninflamed"] += weight * n_noninflamed * (n_noninflamed - 1) / denominator
        expected["Mixed"] += weight * 2 * n_inflamed * n_noninflamed / denominator
    return expected


def inflammation_category(left: str, right: str) -> str:
    if left == right == "Inflamed":
        return "Inflamed-Inflamed"
    if left == right == "Noninflamed":
        return "Noninflamed-Noninflamed"
    return "Mixed"


def bh_adjust(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, float)
    order = np.argsort(values_array)
    ranked = values_array[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty(len(values_array), float)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def run_inflammation_assortativity(nodes: pd.DataFrame, graph_edges: pd.DataFrame, n_permutations: int = 10_000):
    """Test whether IBD sequence neighbors share inflammatory status using participant-level inference."""
    participant_meta = (
        nodes[["SampleID", "Diagnosis", "Inflammation"]]
        .drop_duplicates("SampleID")
        .assign(SampleID=lambda frame: frame.SampleID.astype(str))
    )
    metadata = participant_meta.set_index("SampleID")
    edges = graph_edges.copy()
    edges["participant1"] = edges.node1_id.astype(str).str.split("::", n=1).str[0]
    edges["participant2"] = edges.node2_id.astype(str).str.split("::", n=1).str[0]
    edges["diagnosis1"] = edges.participant1.map(metadata.Diagnosis)
    edges["diagnosis2"] = edges.participant2.map(metadata.Diagnosis)
    edges["inflammation1"] = edges.participant1.map(metadata.Inflammation)
    edges["inflammation2"] = edges.participant2.map(metadata.Inflammation)
    edges = edges[
        edges.diagnosis1.isin(["CD", "UC"])
        & edges.diagnosis1.eq(edges.diagnosis2)
        & edges.inflammation1.isin(["Inflamed", "Noninflamed"])
        & edges.inflammation2.isin(["Inflamed", "Noninflamed"])
    ].copy()
    edges["diagnosis"] = edges.diagnosis1
    edges["category"] = [
        inflammation_category(left, right) for left, right in zip(edges.inflammation1, edges.inflammation2)
    ]

    rng = np.random.default_rng(20260828)
    summary_rows = []
    category_rows = []
    for stratum in ("Pooled IBD", "CD", "UC"):
        frame = edges if stratum == "Pooled IBD" else edges[edges.diagnosis.eq(stratum)]
        frame = frame.reset_index(drop=True)
        participant_ids = sorted(set(frame.participant1) | set(frame.participant2))
        participants = participant_meta[participant_meta.SampleID.isin(participant_ids)].copy()
        participant_index = {participant: index for index, participant in enumerate(participants.SampleID)}
        left_index = frame.participant1.map(participant_index).to_numpy(int)
        right_index = frame.participant2.map(participant_index).to_numpy(int)
        labels = participants.Inflammation.to_numpy(object)
        diagnosis_groups = [group.index.to_numpy(int) for _, group in participants.reset_index(drop=True).groupby("Diagnosis")]
        observed_same = float(np.mean(frame.inflammation1.eq(frame.inflammation2)))
        expected_composition = expected_inflammation_composition(frame, participants)
        expected_same = expected_composition["Inflamed-Inflamed"] + expected_composition["Noninflamed-Noninflamed"]
        observed_excess = observed_same - expected_same

        permutation_same = np.empty(n_permutations, float)
        permutation_counts = np.zeros((n_permutations, 3), float)
        for replicate in range(n_permutations):
            permuted = labels.copy()
            for positions in diagnosis_groups:
                permuted[positions] = rng.permutation(permuted[positions])
            left = permuted[left_index]
            right = permuted[right_index]
            same = left == right
            permutation_same[replicate] = same.mean()
            permutation_counts[replicate, 0] = np.sum((left == "Inflamed") & (right == "Inflamed"))
            permutation_counts[replicate, 1] = np.sum((left == "Noninflamed") & (right == "Noninflamed"))
            permutation_counts[replicate, 2] = np.sum(~same)
        null_mean = float(permutation_same.mean())
        permutation_p = float(
            (1 + np.sum(np.abs(permutation_same - null_mean) >= abs(observed_same - null_mean)))
            / (n_permutations + 1)
        )

        jackknife = []
        for participant in participant_ids:
            retained_edges = frame[
                frame.participant1.ne(participant) & frame.participant2.ne(participant)
            ].copy()
            retained_participants = participants[participants.SampleID.ne(participant)].copy()
            if len(retained_edges) < 20:
                continue
            retained_expected = expected_inflammation_composition(retained_edges, retained_participants)
            retained_same = float(np.mean(retained_edges.inflammation1.eq(retained_edges.inflammation2)))
            jackknife.append(
                retained_same - retained_expected["Inflamed-Inflamed"] - retained_expected["Noninflamed-Noninflamed"]
            )
        jackknife = np.asarray(jackknife, float)
        jackknife_mean = float(jackknife.mean())
        jackknife_se = math.sqrt(
            (len(jackknife) - 1) / len(jackknife) * np.sum(np.square(jackknife - jackknife_mean))
        )
        summary_rows.append(
            {
                "stratum": stratum,
                "n_participants": len(participants),
                "n_edges": len(frame),
                "observed_same_inflammation_fraction": observed_same,
                "expected_same_inflammation_fraction": expected_same,
                "excess_same_inflammation_fraction": observed_excess,
                "ci_low": observed_excess - 1.96 * jackknife_se,
                "ci_high": observed_excess + 1.96 * jackknife_se,
                "participant_jackknife_se": jackknife_se,
                "permutation_p": permutation_p,
                "n_permutations": n_permutations,
            }
        )

        category_names = ["Inflamed-Inflamed", "Noninflamed-Noninflamed", "Mixed"]
        observed_counts = frame.category.value_counts()
        category_p = []
        stratum_category_rows = []
        for category_index, category in enumerate(category_names):
            observed_count = int(observed_counts.get(category, 0))
            expected_fraction = expected_composition[category]
            expected_count = expected_fraction * len(frame)
            null_values = permutation_counts[:, category_index]
            category_null_mean = float(null_values.mean())
            p_value = float(
                (1 + np.sum(np.abs(null_values - category_null_mean) >= abs(observed_count - category_null_mean)))
                / (n_permutations + 1)
            )
            category_p.append(p_value)
            stratum_category_rows.append(
                {
                    "stratum": stratum,
                    "category": category,
                    "n_participants": len(participants),
                    "n_edges": len(frame),
                    "observed_count": observed_count,
                    "observed_fraction": observed_count / len(frame),
                    "expected_count": expected_count,
                    "expected_fraction": expected_fraction,
                    "observed_to_expected_ratio": observed_count / expected_count if expected_count > 0 else np.nan,
                    "permutation_null_count_low": float(np.quantile(null_values, 0.025)),
                    "permutation_null_count_high": float(np.quantile(null_values, 0.975)),
                    "permutation_p": p_value,
                    "n_permutations": n_permutations,
                }
            )
        adjusted = bh_adjust(category_p)
        for row, fdr in zip(stratum_category_rows, adjusted):
            row["permutation_FDR_within_stratum"] = fdr
            category_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    categories = pd.DataFrame(category_rows)
    summary.to_csv(OUT / "Table_F3R20_inflammation_assortativity_summary.csv", index=False)
    categories.to_csv(OUT / "Table_F3R21_inflammation_edge_composition.csv", index=False)
    source = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
    source.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source / "Figure_3_R20_inflammation_assortativity_summary.csv", index=False)
    categories.to_csv(source / "Figure_3_R21_inflammation_edge_composition.csv", index=False)
    return summary, categories


def run_ibd_replacement_analyses(nodes: pd.DataFrame):
    graph_edges = pd.read_csv(EDGES_PATH, low_memory=False)
    diagnosis_results = run_diagnosis_convergence(nodes, graph_edges)
    inflammation_results = run_inflammation_assortativity(nodes, graph_edges)
    diagnosis_interaction = run_diagnosis_interaction(nodes, graph_edges, diagnosis_results[0])
    return diagnosis_results, inflammation_results, diagnosis_interaction


def corrected_odds_ratio(a, b, c, d):
    cells = np.asarray([a, b, c, d], float)
    if np.any(cells == 0):
        cells += 0.5
    a, b, c, d = cells
    odds_ratio = a * d / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return odds_ratio, math.exp(math.log(odds_ratio) - 1.96 * se), math.exp(math.log(odds_ratio) + 1.96 * se)


def build_gliph_tables(nodes: pd.DataFrame):
    gliph = pd.read_csv(GLIPH_PATH, low_memory=False)
    gliph = gliph[gliph.significant_fdr05.astype(str).str.lower().eq("true")].drop_duplicates("unique_cluster_id").copy()
    effect_rows = []
    for row in gliph.itertuples(index=False):
        if row.enrichment_direction == row.first_group:
            enriched_group, other_group = row.first_group, row.second_group
            enriched_carriers, other_carriers = row.n_carriers_g1, row.n_carriers_g2
            enriched_total, other_total = row.n_participants_g1, row.n_participants_g2
        else:
            enriched_group, other_group = row.second_group, row.first_group
            enriched_carriers, other_carriers = row.n_carriers_g2, row.n_carriers_g1
            enriched_total, other_total = row.n_participants_g2, row.n_participants_g1
        odds_ratio, ci_low, ci_high = corrected_odds_ratio(
            enriched_carriers, enriched_total - enriched_carriers, other_carriers, other_total - other_carriers
        )
        effect_rows.append(
            {
                "comparison": row.comparison,
                "unique_cluster_id": row.unique_cluster_id,
                "tag": row.tag,
                "member_set": row.member_set,
                "members": row.members,
                "cluster_size": row.cluster_size,
                "enriched_group": enriched_group,
                "other_group": other_group,
                "enriched_carriers": enriched_carriers,
                "enriched_total": enriched_total,
                "other_carriers": other_carriers,
                "other_total": other_total,
                "enriched_carrier_fraction": enriched_carriers / enriched_total,
                "other_carrier_fraction": other_carriers / other_total,
                "carrier_fraction_difference": enriched_carriers / enriched_total - other_carriers / other_total,
                "odds_ratio_enriched_vs_other": odds_ratio,
                "or_ci_low": ci_low,
                "or_ci_high": ci_high,
                "fdr": row.fdr_unique_member_set,
            }
        )
    effects = pd.DataFrame(effect_rows).sort_values(["comparison", "fdr", "carrier_fraction_difference"], ascending=[True, True, False])
    effects["rank_within_comparison"] = effects.groupby("comparison").cumcount() + 1

    paired_sequences = set(nodes.beta_cdr3.dropna().astype(str))
    overlap_rows = []
    for row in effects.itertuples(index=False):
        members = str(row.members).split()
        overlap_rows.append(
            {
                "comparison": row.comparison,
                "unique_cluster_id": row.unique_cluster_id,
                "tag": row.tag,
                "n_member_sequences": len(members),
                "n_exact_beta_cdr3_matches_in_paired_graph": sum(member in paired_sequences for member in members),
            }
        )
    overlap = pd.DataFrame(overlap_rows)

    representative_pool = effects.drop_duplicates("member_set").sort_values(["fdr", "carrier_fraction_difference"]).copy()

    def select_distinct(frame: pd.DataFrame, count: int = 2) -> pd.DataFrame:
        selected = []
        selected_members: list[set[str]] = []
        for index, row in frame.iterrows():
            members = set(str(row.members).split())
            if any(len(members & existing) / max(1, len(members | existing)) >= 0.34 for existing in selected_members):
                continue
            selected.append(index)
            selected_members.append(members)
            if len(selected) >= count:
                break
        return frame.loc[selected]

    control = select_distinct(representative_pool[representative_pool.enriched_group.eq("Control")])
    uc = select_distinct(representative_pool[representative_pool.enriched_group.eq("UC")])
    representative = pd.concat([control, uc], ignore_index=True).drop_duplicates("member_set").head(4)
    motif_rows = []
    for motif_index, row in representative.reset_index(drop=True).iterrows():
        for sequence in str(row.members).split():
            motif_rows.append(
                {
                    "motif_index": motif_index + 1,
                    "comparison": row.comparison,
                    "unique_cluster_id": row.unique_cluster_id,
                    "tag": row.tag,
                    "enriched_group": row.enriched_group,
                    "fdr": row.fdr,
                    "cdr3_beta": sequence,
                }
            )
    motifs = pd.DataFrame(motif_rows)
    effects.to_csv(OUT / "Table_F3R3_GLIPH2_significant_cluster_effects.csv", index=False)
    motifs.to_csv(OUT / "Table_F3R4_GLIPH2_representative_motifs.csv", index=False)
    overlap.to_csv(OUT / "Table_F3R5_GLIPH2_paired_graph_overlap_audit.csv", index=False)
    return effects, motifs, overlap


def main():
    nodes = pd.read_csv(NODES_PATH, low_memory=False)
    run_graph_robustness(nodes)
    run_transition_robustness(nodes)
    run_primary_attenuation_bootstrap(nodes, pd.read_csv(EDGES_PATH, low_memory=False))
    run_ibd_replacement_analyses(nodes)
    print(OUT)


if __name__ == "__main__":
    import sys

    if "--transition-only" in sys.argv:
        run_transition_robustness(pd.read_csv(NODES_PATH, low_memory=False))
    elif "--panel3d-only" in sys.argv:
        panel3d_nodes = pd.read_csv(NODES_PATH, low_memory=False)
        run_primary_attenuation_bootstrap(panel3d_nodes, pd.read_csv(EDGES_PATH, low_memory=False))
    elif "--ibd-replacement-only" in sys.argv:
        run_ibd_replacement_analyses(pd.read_csv(NODES_PATH, low_memory=False))
    else:
        main()

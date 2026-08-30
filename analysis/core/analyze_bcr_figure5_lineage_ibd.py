#!/usr/bin/env python
"""Generate lineage-tree and IBD-focused analyses for Figures 5 and S12."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import mannwhitneyu, pearsonr, rankdata


MANUSCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = MANUSCRIPT_ROOT / "High Impact Additional Analyses" / "BCR Germline Lineages"
CLINICAL = MANUSCRIPT_ROOT / "High Impact Additional Analyses" / "Priority Analyses" / "Table_PA_clinical_metadata.csv"
DIAGNOSES = ["Control", "CD", "UC"]
DISEASES = ["CD", "UC"]
BOOTSTRAPS = 10000


def bh(values):
    p = np.asarray(values, dtype=float)
    valid = np.isfinite(p)
    result = np.full(len(p), np.nan)
    if not valid.any():
        return result
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty(len(ranked), dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def bootstrap_median_difference(x, y, seed, iterations=BOOTSTRAPS):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        differences[index] = np.median(rng.choice(x, len(x), replace=True)) - np.median(
            rng.choice(y, len(y), replace=True)
        )
    return np.quantile(differences, [0.025, 0.975])


def hdist(first, second):
    pairs = [
        (a, b)
        for a, b in zip(str(first).upper(), str(second).upper())
        if a in "ACGT" and b in "ACGT"
    ]
    return sum(a != b for a, b in pairs) / len(pairs) if pairs else np.nan


def residualized_ranks(data, target, covariates):
    columns = [target] + covariates
    work = data[columns].dropna().copy()
    outcome = rankdata(pd.to_numeric(work[target], errors="coerce"), method="average")
    design = [np.ones(len(work), dtype=float)]
    for covariate in covariates:
        values = work[covariate]
        if values.dtype == object or str(values.dtype).startswith("string"):
            encoded = pd.get_dummies(values.astype(str), drop_first=True, dtype=float)
            if encoded.shape[1]:
                design.extend(encoded[column].to_numpy(float) for column in encoded.columns)
        else:
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
            spread = np.nanstd(numeric)
            design.append((numeric - np.nanmean(numeric)) / spread if spread > 0 else np.zeros(len(numeric)))
    matrix = np.column_stack(design)
    residuals = outcome - matrix @ np.linalg.lstsq(matrix, outcome, rcond=None)[0]
    spread = residuals.std(ddof=1)
    residuals = residuals / spread if spread > 0 else residuals
    return pd.Series(residuals, index=work.index)


# Core lineage and annotation tables.
lineages = pd.read_csv(ROOT / "Table_BGL4_germline_aware_lineage_metrics.csv", low_memory=False)
observed_lineage_counts = pd.read_csv(
    ROOT / "Table_BGL21_germline_edge_free_lineage_divergence.csv",
    usecols=["lineage_id", "n_unique_observed_sequences"],
    low_memory=False,
)
lineages = lineages.merge(observed_lineage_counts, on="lineage_id", how="left")
lineage_states = pd.read_csv(ROOT / "Table_BGL6_lineage_transcriptional_state_composition.csv", low_memory=False)
lineage_lights = pd.read_csv(ROOT / "Table_BGL7_paired_light_chain_refinement.csv", low_memory=False)
participant_lineages = pd.read_csv(ROOT / "Table_BGL11_participant_lineage_metrics.csv", low_memory=False)
edge_free = pd.read_csv(ROOT / "Table_BGL22_participant_germline_edge_free_divergence.csv", low_memory=False)
valid_lineage_lights = lineage_lights[
    lineage_lights["light_id"].fillna("").astype(str).str.match(r"^IG[KL]V[^|]+\|IG[KL]J[^|]+\|.+")
].copy()
valid_light_lineages = set(valid_lineage_lights["lineage_id"])


# Prespecified representative-lineage selection: class-switched, paired-light
# annotated, non-truncated lineages with 4-10 distinct observed heavy
# alignments. Within
# each diagnosis, select the lineage closest to the pooled eligible medians for
# node count, log abundance, and germline-inclusive branch length.
eligible = lineages[
    lineages["n_unique_observed_sequences"].between(4, 10)
    & ~lineages["tree_truncated"].astype(bool)
    & lineages["mapped_cells"].gt(0)
    & lineages["lineage_id"].isin(valid_light_lineages)
    & lineages["class_switched_lineage"].astype(str).str.lower().eq("true")
].copy()
if set(eligible["Diagnosis"]) != set(DIAGNOSES):
    raise RuntimeError("Representative-lineage eligibility did not retain every diagnosis.")
eligible["log_abundance"] = np.log1p(eligible["weighted_abundance"])
selection_features = ["n_unique_observed_sequences", "log_abundance", "total_MST_branch_length"]
for feature in selection_features:
    center = eligible[feature].median()
    scale = eligible[feature].quantile(0.75) - eligible[feature].quantile(0.25)
    if not np.isfinite(scale) or scale == 0:
        scale = eligible[feature].std(ddof=1) or 1.0
    eligible[f"selection_z_{feature}"] = (eligible[feature] - center) / scale
eligible["selection_score"] = eligible[[f"selection_z_{feature}" for feature in selection_features]].abs().sum(axis=1)
representatives = (
    eligible.sort_values(["Diagnosis", "selection_score", "lineage_id"])
    .groupby("Diagnosis", as_index=False)
    .first()
    .set_index("Diagnosis")
    .reindex(DIAGNOSES)
    .reset_index()
)

dominant_lights = (
    valid_lineage_lights.sort_values(["lineage_id", "n_cells"], ascending=[True, False])
    .drop_duplicates("lineage_id")
    .rename(columns={"light_id": "dominant_light_id", "n_cells": "dominant_light_cells"})
)
dominant_states = (
    lineage_states.sort_values(["lineage_id", "n_cells"], ascending=[True, False])
    .drop_duplicates("lineage_id")
    .rename(columns={"cell_state": "dominant_cell_state", "n_cells": "dominant_state_cells"})
)
representatives = representatives.merge(
    dominant_lights[["lineage_id", "dominant_light_id", "dominant_light_cells"]], on="lineage_id", how="left"
).merge(
    dominant_states[["lineage_id", "dominant_cell_state", "dominant_state_cells"]], on="lineage_id", how="left"
)

db = pd.read_csv(
    ROOT / "bcr_heavy_lineages_d015_clone-pass.tsv",
    sep="\t",
    usecols=[
        "sequence_id", "sequence_alignment", "germline_alignment", "clone_id", "sampleid",
        "diagnosis", "count", "v_call", "j_call",
    ],
    low_memory=False,
)
db["lineage_id"] = db["sampleid"].astype(str) + "::" + db["clone_id"].astype(str)
representative_nodes = []
representative_edges = []
for _, selected in representatives.iterrows():
    group = db[db["lineage_id"].eq(selected["lineage_id"])].copy()
    observed = (
        group.groupby("sequence_alignment", as_index=False)
        .agg(sequence_id=("sequence_id", "first"), count=("count", "sum"))
        .sort_values(["count", "sequence_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    germline = str(group["germline_alignment"].dropna().iloc[0])
    sequences = [germline] + observed["sequence_alignment"].astype(str).tolist()
    distance_matrix = np.zeros((len(sequences), len(sequences)), dtype=float)
    for i in range(len(sequences)):
        for j in range(i):
            distance = hdist(sequences[i], sequences[j])
            if np.isfinite(distance) and distance == 0:
                distance = 1e-9
            distance_matrix[i, j] = distance_matrix[j, i] = distance if np.isfinite(distance) else 1.0
    tree = minimum_spanning_tree(distance_matrix).toarray()
    undirected_edges = []
    adjacency = {index: [] for index in range(len(sequences))}
    for i in range(len(sequences)):
        for j in range(len(sequences)):
            if tree[i, j] > 0:
                weight = float(tree[i, j])
                undirected_edges.append((i, j, weight))
                adjacency[i].append((j, weight))
                adjacency[j].append((i, weight))
    parent = {0: None}
    cumulative = {0: 0.0}
    queue = [0]
    order = [0]
    while queue:
        node = queue.pop(0)
        for neighbor, weight in adjacency[node]:
            if neighbor in parent:
                continue
            parent[neighbor] = node
            cumulative[neighbor] = cumulative[node] + weight
            queue.append(neighbor)
            order.append(neighbor)
    children = {node: [] for node in order}
    for node, parent_node in parent.items():
        if parent_node is not None:
            children[parent_node].append(node)
    leaves = [node for node in order if not children[node]]
    y_position = {node: float(index) for index, node in enumerate(leaves)}
    for node in reversed(order):
        if node not in y_position:
            y_position[node] = float(np.mean([y_position[child] for child in children[node]]))
    for node in order:
        if node == 0:
            sequence_id = "IMGT germline"
            count = 0.0
        else:
            sequence_id = observed.iloc[node - 1]["sequence_id"]
            count = float(observed.iloc[node - 1]["count"])
        representative_nodes.append(
            {
                "Diagnosis": selected["Diagnosis"],
                "lineage_id": selected["lineage_id"],
                "node_id": node,
                "node_type": "Germline" if node == 0 else "Observed",
                "sequence_id": sequence_id,
                "count": count,
                "x": cumulative[node],
                "y": y_position[node],
                "distance_to_germline": hdist(germline, sequences[node]) if node else 0.0,
            }
        )
    for first, second, weight in undirected_edges:
        representative_edges.append(
            {
                "Diagnosis": selected["Diagnosis"],
                "lineage_id": selected["lineage_id"],
                "source": first,
                "target": second,
                "edge_length": weight,
                "germline_edge": first == 0 or second == 0,
            }
        )

representative_nodes = pd.DataFrame(representative_nodes)
representative_edges = pd.DataFrame(representative_edges)


# Standardized cohort-level lineage architecture. Metrics are scaled to the
# control interquartile range so heterogeneous lineage quantities can share a
# single effect-size axis.
cohort = participant_lineages.merge(
    edge_free[["SampleID", "mean_within_lineage_divergence"]], on="SampleID", how="left"
)
cohort_metrics = {
    "expanded_lineages": "Expanded lineages per participant",
    "mean_MST_branch_length": "Germline-inclusive branch length",
    "mean_within_lineage_divergence": "Observed-only lineage divergence",
    "fraction_class_switched_lineages": "Class-switched lineage fraction",
}
cohort_effect_rows = []
cduc_rows = []
for metric_index, (metric, label) in enumerate(cohort_metrics.items()):
    control_values = pd.to_numeric(cohort.loc[cohort["Diagnosis"].eq("Control"), metric], errors="coerce").dropna()
    center = control_values.median()
    scale = control_values.quantile(0.75) - control_values.quantile(0.25)
    if not np.isfinite(scale) or scale == 0:
        scale = control_values.std(ddof=1) or 1.0
    cohort[f"standardized_{metric}"] = (pd.to_numeric(cohort[metric], errors="coerce") - center) / scale
    for disease_index, disease in enumerate(DISEASES):
        x = cohort.loc[cohort["Diagnosis"].eq(disease), f"standardized_{metric}"].dropna().to_numpy(float)
        y = cohort.loc[cohort["Diagnosis"].eq("Control"), f"standardized_{metric}"].dropna().to_numpy(float)
        low, high = bootstrap_median_difference(x, y, 20260828 + 100 * metric_index + disease_index)
        cohort_effect_rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "contrast": f"{disease} vs Control",
                "n_disease": len(x),
                "n_control": len(y),
                "standardized_median_difference": float(np.median(x) - np.median(y)),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
                "control_IQR_scale": float(scale),
            }
        )
    cd = cohort.loc[cohort["Diagnosis"].eq("CD"), f"standardized_{metric}"].dropna().to_numpy(float)
    uc = cohort.loc[cohort["Diagnosis"].eq("UC"), f"standardized_{metric}"].dropna().to_numpy(float)
    low, high = bootstrap_median_difference(cd, uc, 20261828 + metric_index)
    cduc_rows.append(
        {
            "metric": metric,
            "metric_label": label,
            "contrast": "CD vs UC",
            "n_CD": len(cd),
            "n_UC": len(uc),
            "standardized_median_difference": float(np.median(cd) - np.median(uc)),
            "ci_low": float(low),
            "ci_high": float(high),
            "p_value": float(mannwhitneyu(cd, uc, alternative="two-sided").pvalue),
        }
    )
cohort_effects = pd.DataFrame(cohort_effect_rows)
cohort_effects["FDR"] = bh(cohort_effects["p_value"])
cduc_effects = pd.DataFrame(cduc_rows)
cduc_effects["FDR"] = bh(cduc_effects["p_value"])


# IBD clinical relevance: objective inflammation and fecal calprotectin.
clinical = pd.read_csv(CLINICAL, low_memory=False).rename(columns={"Diagnosis1": "Diagnosis"})
clinical["acquisition_series"] = clinical["Batch"].astype(str).str.replace(r"[AB]$", "", regex=True)
clinical["Biologic"] = clinical["Biologic"].fillna("None").astype(str)
clinical["inflamed"] = clinical["Inflammation1"].eq("Inflamed").astype(int)
clinical["log_calprotectin"] = np.log1p(pd.to_numeric(clinical["Calprotectin"], errors="coerce"))

regional = pd.read_csv(ROOT / "Table_BGL10_participant_region_SHM.csv", low_memory=False)
regional["region_class"] = np.where(regional["region"].str.lower().str.startswith("cdr"), "CDR", "FWR")
targeting = regional.groupby(["SampleID", "region_class"])["weighted_shm_rate"].mean().unstack().reset_index()
targeting["CDR_minus_FWR"] = targeting["CDR"] - targeting["FWR"]
paired_shm = pd.read_csv(ROOT / "Table_BGL24_participant_paired_clonotype_SHM.csv", low_memory=False)
paired_shm = paired_shm[paired_shm["estimand"].eq("Exact paired H-L clonotypes")][
    ["SampleID", "overall_shm_rate"]
].rename(columns={"overall_shm_rate": "exact_paired_HL_shm"})

clinical_features = cohort.merge(targeting[["SampleID", "CDR_minus_FWR"]], on="SampleID", how="left").merge(
    paired_shm, on="SampleID", how="left"
).merge(
    clinical[["SampleID", "Diagnosis", "Inflammation1", "inflamed", "Biologic", "acquisition_series", "Calprotectin", "log_calprotectin"]],
    on=["SampleID", "Diagnosis"], how="left"
)
clinical_features["log_lineage_depth"] = np.log1p(clinical_features["n_lineages"])
clinical_metrics = {
    "expanded_lineages": "Expanded lineage burden",
    "mean_MST_branch_length": "Germline-inclusive branch length",
    "mean_within_lineage_divergence": "Observed-only divergence",
    "CDR_minus_FWR": "CDR targeting",
    "exact_paired_HL_shm": "Paired H-L heavy-chain SHM",
    "fraction_class_switched_lineages": "Class-switched lineage fraction",
}
adjustment = ["log_lineage_depth", "Biologic", "acquisition_series"]
inflammation_rows = []
calprotectin_rows = []
for diagnosis_index, diagnosis in enumerate(DISEASES):
    diagnosis_data = clinical_features[clinical_features["Diagnosis"].eq(diagnosis)].copy()
    for metric_index, (metric, label) in enumerate(clinical_metrics.items()):
        needed = [metric, "inflamed", "log_calprotectin"] + adjustment
        data = diagnosis_data[needed].dropna().copy()
        if len(data) < 20:
            continue
        outcome_residual = residualized_ranks(data, metric, adjustment)
        aligned = data.loc[outcome_residual.index].copy()
        aligned["outcome_residual"] = outcome_residual
        inflamed = aligned.loc[aligned["inflamed"].eq(1), "outcome_residual"].to_numpy(float)
        noninflamed = aligned.loc[aligned["inflamed"].eq(0), "outcome_residual"].to_numpy(float)
        if len(inflamed) >= 5 and len(noninflamed) >= 5:
            low, high = bootstrap_median_difference(
                inflamed, noninflamed, 20262828 + 100 * diagnosis_index + metric_index
            )
            inflammation_rows.append(
                {
                    "Diagnosis": diagnosis,
                    "metric": metric,
                    "metric_label": label,
                    "n_inflamed": len(inflamed),
                    "n_noninflamed": len(noninflamed),
                    "adjusted_median_difference": float(np.median(inflamed) - np.median(noninflamed)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "p_value": float(mannwhitneyu(inflamed, noninflamed, alternative="two-sided").pvalue),
                }
            )
        cal_residual = residualized_ranks(data, "log_calprotectin", adjustment)
        common = outcome_residual.index.intersection(cal_residual.index)
        if len(common) >= 20:
            x = cal_residual.loc[common].to_numpy(float)
            y = outcome_residual.loc[common].to_numpy(float)
            rho, p_value = pearsonr(x, y)
            rng = np.random.default_rng(20263828 + 100 * diagnosis_index + metric_index)
            boot = np.empty(5000, dtype=float)
            for iteration in range(len(boot)):
                index = rng.integers(0, len(common), len(common))
                boot[iteration] = np.corrcoef(x[index], y[index])[0, 1]
            calprotectin_rows.append(
                {
                    "Diagnosis": diagnosis,
                    "metric": metric,
                    "metric_label": label,
                    "n_participants": len(common),
                    "partial_spearman_rho": float(rho),
                    "ci_low": float(np.nanquantile(boot, 0.025)),
                    "ci_high": float(np.nanquantile(boot, 0.975)),
                    "p_value": float(p_value),
                }
            )
inflammation_effects = pd.DataFrame(inflammation_rows)
inflammation_effects["FDR"] = bh(inflammation_effects["p_value"])
calprotectin_effects = pd.DataFrame(calprotectin_rows)
calprotectin_effects["FDR"] = bh(calprotectin_effects["p_value"])


# Mucosal-relevant state composition among expanded, cell-mapped lineages.
mucosal_states = ["IgA Plasma B Cell", "IgG Plasma B Cell", "Switched memory B", "Atypical memory B"]
expanded_meta = lineages[lineages["n_unique_sequences"].ge(2)][["lineage_id", "SampleID", "Diagnosis"]]
expanded_state = lineage_states.merge(expanded_meta, on="lineage_id", how="inner")
state_total = expanded_state.groupby(["SampleID", "Diagnosis"], as_index=False)["n_cells"].sum().rename(
    columns={"n_cells": "expanded_lineage_mapped_cells"}
)
state_numerator = (
    expanded_state[expanded_state["cell_state"].isin(mucosal_states)]
    .groupby(["SampleID", "Diagnosis", "cell_state"], as_index=False)["n_cells"].sum()
)
state_grid = state_total.assign(key=1).merge(pd.DataFrame({"cell_state": mucosal_states, "key": 1}), on="key").drop(columns="key")
mucosal_participant = state_grid.merge(state_numerator, on=["SampleID", "Diagnosis", "cell_state"], how="left")
mucosal_participant["n_cells"] = mucosal_participant["n_cells"].fillna(0)
mucosal_participant["fraction"] = mucosal_participant["n_cells"] / mucosal_participant["expanded_lineage_mapped_cells"].replace(0, np.nan)
mucosal_rows = []
for state_index, state in enumerate(mucosal_states):
    data = mucosal_participant[mucosal_participant["cell_state"].eq(state)]
    control = data.loc[data["Diagnosis"].eq("Control"), "fraction"].dropna().to_numpy(float)
    for disease_index, disease in enumerate(DISEASES):
        disease_values = data.loc[data["Diagnosis"].eq(disease), "fraction"].dropna().to_numpy(float)
        if len(disease_values) < 3 or len(control) < 3:
            continue
        low, high = bootstrap_median_difference(
            disease_values, control, 20264828 + 100 * state_index + disease_index
        )
        mucosal_rows.append(
            {
                "cell_state": state,
                "contrast": f"{disease} vs Control",
                "n_disease": len(disease_values),
                "n_control": len(control),
                "median_fraction_difference": float(np.median(disease_values) - np.median(control)),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": float(mannwhitneyu(disease_values, control, alternative="two-sided").pvalue),
            }
        )
mucosal_effects = pd.DataFrame(mucosal_rows)
mucosal_effects["FDR"] = bh(mucosal_effects["p_value"])


# Paired-light-chain context: locus and the five most abundant V-gene families.
light = valid_lineage_lights.merge(lineages[["lineage_id", "SampleID", "Diagnosis"]], on="lineage_id", how="left")
light["light_locus"] = np.where(light["light_id"].str.startswith("IGK"), "IGK", "IGL")
light["light_v_gene"] = light["light_id"].str.split("|").str[0].str.replace(r"\*.*$", "", regex=True)
light["light_v_family"] = light["light_v_gene"].str.extract(r"^(IG[KL]V\d+)", expand=False)
top_families = light.groupby("light_v_family")["n_cells"].sum().nlargest(5).index.tolist()
light_features = ["IGL fraction"] + top_families
light_totals = light.groupby(["SampleID", "Diagnosis"], as_index=False)["n_cells"].sum().rename(columns={"n_cells": "paired_light_cells"})
light_participant_rows = []
for _, total_row in light_totals.iterrows():
    sample = total_row["SampleID"]
    diagnosis = total_row["Diagnosis"]
    denominator = float(total_row["paired_light_cells"])
    sample_light = light[light["SampleID"].eq(sample)]
    light_participant_rows.append(
        {
            "SampleID": sample,
            "Diagnosis": diagnosis,
            "feature": "IGL fraction",
            "fraction": float(sample_light.loc[sample_light["light_locus"].eq("IGL"), "n_cells"].sum() / denominator),
            "paired_light_cells": denominator,
        }
    )
    for family in top_families:
        light_participant_rows.append(
            {
                "SampleID": sample,
                "Diagnosis": diagnosis,
                "feature": family,
                "fraction": float(sample_light.loc[sample_light["light_v_family"].eq(family), "n_cells"].sum() / denominator),
                "paired_light_cells": denominator,
            }
        )
light_participant = pd.DataFrame(light_participant_rows)
light_rows = []
for feature_index, feature in enumerate(light_features):
    data = light_participant[light_participant["feature"].eq(feature)]
    control = data.loc[data["Diagnosis"].eq("Control"), "fraction"].to_numpy(float)
    for disease_index, disease in enumerate(DISEASES):
        disease_values = data.loc[data["Diagnosis"].eq(disease), "fraction"].to_numpy(float)
        low, high = bootstrap_median_difference(
            disease_values, control, 20265828 + 100 * feature_index + disease_index
        )
        light_rows.append(
            {
                "feature": feature,
                "contrast": f"{disease} vs Control",
                "n_disease": len(disease_values),
                "n_control": len(control),
                "median_fraction_difference": float(np.median(disease_values) - np.median(control)),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": float(mannwhitneyu(disease_values, control, alternative="two-sided").pvalue),
            }
        )
light_effects = pd.DataFrame(light_rows)
light_effects["FDR"] = bh(light_effects["p_value"])


representatives.to_csv(ROOT / "Table_BGL33_representative_lineage_selection.csv", index=False)
representative_nodes.to_csv(ROOT / "Table_BGL34_representative_lineage_nodes.csv", index=False)
representative_edges.to_csv(ROOT / "Table_BGL35_representative_lineage_edges.csv", index=False)
cohort_effects.to_csv(ROOT / "Table_BGL36_standardized_lineage_architecture_effects.csv", index=False)
cduc_effects.to_csv(ROOT / "Table_BGL37_CD_UC_lineage_concordance.csv", index=False)
clinical_features.to_csv(ROOT / "Table_BGL38_participant_IBD_clinical_lineage_features.csv", index=False)
inflammation_effects.to_csv(ROOT / "Table_BGL39_objective_inflammation_lineage_effects.csv", index=False)
calprotectin_effects.to_csv(ROOT / "Table_BGL40_calprotectin_lineage_partial_correlations.csv", index=False)
mucosal_participant.to_csv(ROOT / "Table_BGL41_expanded_lineage_mucosal_state_fractions.csv", index=False)
mucosal_effects.to_csv(ROOT / "Table_BGL42_expanded_lineage_mucosal_state_effects.csv", index=False)
light_participant.to_csv(ROOT / "Table_BGL43_paired_light_feature_fractions.csv", index=False)
light_effects.to_csv(ROOT / "Table_BGL44_paired_light_feature_effects.csv", index=False)

print("Representative lineages")
print(representatives[["Diagnosis", "lineage_id", "n_unique_observed_sequences", "weighted_abundance", "total_MST_branch_length", "isotype_classes", "dominant_light_id", "dominant_cell_state", "selection_score"]].to_string(index=False))
print("\nCohort lineage effects")
print(cohort_effects.to_string(index=False))
print("\nCD versus UC concordance")
print(cduc_effects.to_string(index=False))
print("\nObjective inflammation")
print(inflammation_effects.to_string(index=False))
print("\nCalprotectin")
print(calprotectin_effects.to_string(index=False))
print("\nExpanded-lineage mucosal states")
print(mucosal_effects.to_string(index=False))
print("\nPaired-light features")
print(light_effects.to_string(index=False))

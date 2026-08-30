#!/usr/bin/env python
"""Generate robustness tables for the enhanced Cell Press Figure 5."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


ROOT = Path(__file__).resolve().parent / "High Impact Additional Analyses" / "BCR Germline Lineages"
DIAGNOSES = ["Control", "CD", "UC"]
DISEASE_CONTRASTS = [("CD", "Control"), ("UC", "Control")]


def bh(values):
    p = np.asarray(values, dtype=float)
    valid = np.isfinite(p)
    result = np.full(len(p), np.nan)
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty(len(ranked), dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def bootstrap_median_difference(x, y, seed, iterations=10000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for i in range(iterations):
        differences[i] = np.median(rng.choice(x, len(x), replace=True)) - np.median(
            rng.choice(y, len(y), replace=True)
        )
    return np.quantile(differences, [0.025, 0.975])


def compare_groups(table, value, strata, seed_offset=0):
    rows = []
    grouped = table.groupby(strata, dropna=False, sort=False) if strata else [((), table)]
    for stratum_index, (key, data) in enumerate(grouped):
        if not isinstance(key, tuple):
            key = (key,)
        stratum_values = dict(zip(strata, key))
        for contrast_index, (disease, control) in enumerate(DISEASE_CONTRASTS):
            x = pd.to_numeric(data.loc[data["Diagnosis"] == disease, value], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(data.loc[data["Diagnosis"] == control, value], errors="coerce").dropna().to_numpy()
            if len(x) < 3 or len(y) < 3:
                continue
            low, high = bootstrap_median_difference(
                x, y, seed=20260827 + seed_offset + 100 * stratum_index + contrast_index
            )
            rows.append(
                {
                    **stratum_values,
                    "contrast": f"{disease} vs Control",
                    "n_disease": len(x),
                    "n_control": len(y),
                    "median_disease": float(np.median(x)),
                    "median_control": float(np.median(y)),
                    "median_difference": float(np.median(x) - np.median(y)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "p_value": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        result["FDR"] = bh(result["p_value"])
    return result


# Sequence-region mutation table joined to exact heavy-receptor metadata.
query = pd.read_csv(
    ROOT / "Table_BGL1_query_sequence_metadata.csv",
    usecols=["sequence_id", "SampleID", "Diagnosis", "count", "v_gene", "j_gene", "junction"],
    low_memory=False,
).drop_duplicates("sequence_id")
query["mapping_key"] = (
    query["SampleID"].astype(str)
    + "|"
    + query["v_gene"].astype(str)
    + "|"
    + query["j_gene"].astype(str)
    + "|"
    + query["junction"].astype(str).str.slice(3, -3)
)
regions = pd.read_csv(ROOT / "Table_BGL3_sequence_region_SHM_RS.csv", low_memory=False)
regions["informative_nt"] = pd.to_numeric(regions["informative_nt"], errors="coerce").fillna(0)
regions["mutations"] = pd.to_numeric(regions["mutations"], errors="coerce").fillna(0)
regions["count"] = pd.to_numeric(regions["count"], errors="coerce").fillna(1).clip(lower=1)


# Overall SHM with and without clonotype-abundance weighting.
clone_weight_rows = []
for (sample_id, diagnosis), group in regions.groupby(["SampleID", "Diagnosis"], sort=False):
    for weighting, weights in [
        ("Unique sequences", np.ones(len(group), dtype=float)),
        ("Abundance weighted", group["count"].to_numpy(float)),
    ]:
        denominator = float(np.sum(group["informative_nt"].to_numpy(float) * weights))
        numerator = float(np.sum(group["mutations"].to_numpy(float) * weights))
        clone_weight_rows.append(
            {
                "SampleID": sample_id,
                "Diagnosis": diagnosis,
                "weighting": weighting,
                "overall_shm_rate": numerator / denominator if denominator > 0 else np.nan,
                "informative_nt": denominator,
            }
        )
clone_weighting = pd.DataFrame(clone_weight_rows)
clone_weighting_effects = compare_groups(clone_weighting, "overall_shm_rate", ["weighting"], seed_offset=1000)


# Exact cell-linked isotype-stratified SHM.
cells = pd.read_csv(
    ROOT / "Table_BGL2_cell_state_isotype_light_mapping.csv",
    usecols=["receptor_key", "isotype", "n_cells"],
    low_memory=False,
)
cells["n_cells"] = pd.to_numeric(cells["n_cells"], errors="coerce").fillna(0)
cells["isotype_class"] = np.select(
    [
        cells["isotype"].astype(str).str.startswith("IGHM"),
        cells["isotype"].astype(str).str.startswith("IGHA"),
        cells["isotype"].astype(str).str.startswith("IGHG"),
    ],
    ["IgM", "IgA", "IgG"],
    default="Other",
)
cell_isotype = (
    cells[cells["isotype_class"].isin(["IgM", "IgA", "IgG"])]
    .groupby(["receptor_key", "isotype_class"], as_index=False)["n_cells"]
    .sum()
)
mapped = regions.merge(query[["sequence_id", "mapping_key"]], on="sequence_id", how="inner").merge(
    cell_isotype, left_on="mapping_key", right_on="receptor_key", how="inner"
)
mapped["weighted_mutations"] = mapped["mutations"] * mapped["n_cells"]
mapped["weighted_informative_nt"] = mapped["informative_nt"] * mapped["n_cells"]
isotype_shm = (
    mapped.groupby(["SampleID", "Diagnosis", "isotype_class"], as_index=False)
    .agg(
        weighted_mutations=("weighted_mutations", "sum"),
        weighted_informative_nt=("weighted_informative_nt", "sum"),
        mapped_cells=("n_cells", "sum"),
        mapped_sequences=("sequence_id", "nunique"),
    )
)
isotype_shm["shm_rate"] = isotype_shm["weighted_mutations"] / isotype_shm["weighted_informative_nt"].replace(0, np.nan)
isotype_effects = compare_groups(isotype_shm, "shm_rate", ["isotype_class"], seed_offset=2000)


# Paired-heavy-light resolution of heavy-chain SHM. Cell metadata do not contain
# a full light-chain variable-region alignment, so these analyses refine the
# clonotype denominator with the paired light V/J/CDR3 while retaining the
# germline-relative SHM measurement from the linked heavy sequence.
pair_cells = pd.read_csv(
    ROOT / "Table_BGL2_cell_state_isotype_light_mapping.csv",
    usecols=["receptor_key", "isotype", "light_v", "light_j", "light_cdr3", "n_cells"],
    low_memory=False,
)
pair_cells["n_cells"] = pd.to_numeric(pair_cells["n_cells"], errors="coerce").fillna(0)
for column in ["light_v", "light_j", "light_cdr3"]:
    pair_cells[column] = pair_cells[column].fillna("").astype(str).str.strip()
pair_cells = pair_cells[
    pair_cells["light_v"].ne("")
    & pair_cells["light_j"].ne("")
    & pair_cells["light_cdr3"].ne("")
    & pair_cells["n_cells"].gt(0)
].copy()
pair_cells["light_locus"] = np.where(pair_cells["light_v"].str.startswith("IGK"), "IGK", "IGL")
pair_cells["light_id"] = (
    pair_cells["light_locus"]
    + "|"
    + pair_cells["light_v"]
    + "|"
    + pair_cells["light_j"]
    + "|"
    + pair_cells["light_cdr3"]
)
pair_cells["isotype_class"] = np.select(
    [
        pair_cells["isotype"].astype(str).str.startswith("IGHM"),
        pair_cells["isotype"].astype(str).str.startswith("IGHA"),
        pair_cells["isotype"].astype(str).str.startswith("IGHG"),
    ],
    ["IgM", "IgA", "IgG"],
    default="Other",
)
pair_by_isotype = (
    pair_cells.groupby(["receptor_key", "light_id", "isotype_class"], as_index=False)["n_cells"].sum()
)
pair_overall = pair_by_isotype.groupby(["receptor_key", "light_id"], as_index=False)["n_cells"].sum()
pair_overall["paired_id"] = pair_overall["receptor_key"] + "||" + pair_overall["light_id"]

sequence_overall = (
    regions.groupby(["sequence_id", "SampleID", "Diagnosis"], as_index=False)
    .agg(mutations=("mutations", "sum"), informative_nt=("informative_nt", "sum"))
    .merge(query[["sequence_id", "mapping_key"]], on="sequence_id", how="inner")
)
heavy_receptors = (
    sequence_overall.groupby(["SampleID", "Diagnosis", "mapping_key"], as_index=False)
    .agg(
        mutations=("mutations", "sum"),
        informative_nt=("informative_nt", "sum"),
        n_unique_heavy_sequences=("sequence_id", "nunique"),
    )
)
heavy_receptors["shm_rate"] = heavy_receptors["mutations"] / heavy_receptors["informative_nt"].replace(0, np.nan)

paired_receptors = heavy_receptors.merge(pair_overall, left_on="mapping_key", right_on="receptor_key", how="inner")
paired_receptors = paired_receptors[paired_receptors["informative_nt"].gt(0)].copy()
light_counts = paired_receptors.groupby("mapping_key")["light_id"].nunique().rename("n_light_partners")
paired_receptors = paired_receptors.merge(light_counts, on="mapping_key", how="left")
paired_receptors["rank_within_heavy"] = paired_receptors.groupby("mapping_key")["n_cells"].rank(
    method="first", ascending=False
)


def pooled_participant_rates(data, estimand, weight_column=None):
    work = data.copy()
    weights = np.ones(len(work), dtype=float) if weight_column is None else work[weight_column].to_numpy(float)
    work["weighted_mutations"] = work["mutations"].to_numpy(float) * weights
    work["weighted_informative_nt"] = work["informative_nt"].to_numpy(float) * weights
    result = (
        work.groupby(["SampleID", "Diagnosis"], as_index=False)
        .agg(
            weighted_mutations=("weighted_mutations", "sum"),
            weighted_informative_nt=("weighted_informative_nt", "sum"),
            n_units=("mapping_key", "size"),
        )
    )
    result["overall_shm_rate"] = result["weighted_mutations"] / result["weighted_informative_nt"].replace(0, np.nan)
    result["estimand"] = estimand
    return result


paired_mapping_keys = set(paired_receptors["mapping_key"])
paired_heavy = heavy_receptors[heavy_receptors["mapping_key"].isin(paired_mapping_keys)].copy()
dominant_pairs = paired_receptors[paired_receptors["rank_within_heavy"].eq(1)].copy()
paired_participant_shm = pd.concat(
    [
        clone_weighting.loc[clone_weighting["weighting"].eq("Unique sequences")].rename(
            columns={"weighting": "estimand", "informative_nt": "weighted_informative_nt"}
        ).assign(estimand="All unique heavy sequences", n_units=np.nan),
        pooled_participant_rates(paired_heavy, "Paired-subset heavy clonotypes"),
        pooled_participant_rates(paired_receptors, "Exact paired H-L clonotypes"),
        pooled_participant_rates(paired_receptors, "Paired H-L cell-abundance weighted", "n_cells"),
        pooled_participant_rates(dominant_pairs, "Dominant-light sensitivity"),
    ],
    ignore_index=True,
    sort=False,
)
paired_participant_shm = paired_participant_shm[
    ["SampleID", "Diagnosis", "estimand", "overall_shm_rate", "weighted_informative_nt", "n_units"]
]
paired_clonotype_effects = compare_groups(
    paired_participant_shm, "overall_shm_rate", ["estimand"], seed_offset=4000
)

# Within-participant expanded-minus-singleton paired-clonotype contrasts.
pair_isotype = pair_by_isotype[pair_by_isotype["isotype_class"].isin(["IgM", "IgA", "IgG"])].copy()
pair_isotype["paired_id"] = pair_isotype["receptor_key"] + "||" + pair_isotype["light_id"]
pair_isotype = heavy_receptors.merge(pair_isotype, left_on="mapping_key", right_on="receptor_key", how="inner")
pair_isotype = pair_isotype[pair_isotype["informative_nt"].gt(0)].copy()
pair_isotype["expansion_status"] = np.where(pair_isotype["n_cells"].ge(2), "Expanded", "Singleton")
paired_status_shm = (
    pair_isotype.groupby(["SampleID", "Diagnosis", "isotype_class", "expansion_status"], as_index=False)
    .agg(
        mutations=("mutations", "sum"),
        informative_nt=("informative_nt", "sum"),
        n_paired_clonotypes=("paired_id", "nunique"),
        represented_cells=("n_cells", "sum"),
    )
)
paired_status_shm["shm_rate"] = paired_status_shm["mutations"] / paired_status_shm["informative_nt"].replace(0, np.nan)
paired_status_wide = paired_status_shm.pivot_table(
    index=["SampleID", "Diagnosis", "isotype_class"],
    columns="expansion_status",
    values=["shm_rate", "n_paired_clonotypes"],
    aggfunc="first",
).reset_index()
paired_status_wide.columns = [
    "_".join([str(item) for item in column if str(item) != ""]).strip("_")
    if isinstance(column, tuple)
    else column
    for column in paired_status_wide.columns
]
for required in ["shm_rate_Expanded", "shm_rate_Singleton", "n_paired_clonotypes_Expanded", "n_paired_clonotypes_Singleton"]:
    if required not in paired_status_wide:
        paired_status_wide[required] = np.nan
paired_status_deltas = paired_status_wide[
    paired_status_wide["n_paired_clonotypes_Expanded"].ge(1)
    & paired_status_wide["n_paired_clonotypes_Singleton"].ge(3)
].copy()
paired_status_deltas["expanded_minus_singleton_shm"] = (
    paired_status_deltas["shm_rate_Expanded"] - paired_status_deltas["shm_rate_Singleton"]
)

paired_status_tests = []
for stratum_index, ((diagnosis, isotype_class), data) in enumerate(
    paired_status_deltas.groupby(["Diagnosis", "isotype_class"], sort=False)
):
    values = data["expanded_minus_singleton_shm"].dropna().to_numpy(float)
    if len(values) < 5:
        continue
    rng = np.random.default_rng(20260827 + 5000 + stratum_index)
    boot = np.asarray([np.median(rng.choice(values, len(values), replace=True)) for _ in range(10000)])
    try:
        p_value = float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        p_value = 1.0
    paired_status_tests.append(
        {
            "Diagnosis": diagnosis,
            "isotype_class": isotype_class,
            "n_participants": len(values),
            "median_expanded_minus_singleton_shm": float(np.median(values)),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "p_value": p_value,
        }
    )
paired_status_tests = pd.DataFrame(paired_status_tests)
if len(paired_status_tests):
    paired_status_tests["FDR"] = bh(paired_status_tests["p_value"])
paired_status_diagnosis_effects = compare_groups(
    paired_status_deltas, "expanded_minus_singleton_shm", ["isotype_class"], seed_offset=6000
)

# Pairing coverage and multi-light ambiguity audit by participant.
paired_qc_units = (
    paired_receptors.groupby(["SampleID", "Diagnosis", "mapping_key"], as_index=False)
    .agg(
        n_light_partners=("light_id", "nunique"),
        exact_paired_clonotypes=("paired_id", "nunique"),
        paired_cells=("n_cells", "sum"),
    )
)
pairing_qc = (
    heavy_receptors.groupby(["SampleID", "Diagnosis"], as_index=False)
    .agg(total_heavy_clonotypes=("mapping_key", "nunique"))
    .merge(
        paired_qc_units.groupby(["SampleID", "Diagnosis"], as_index=False).agg(
            paired_heavy_clonotypes=("mapping_key", "nunique"),
            exact_paired_clonotypes=("exact_paired_clonotypes", "sum"),
            heavy_clonotypes_with_multiple_lights=("n_light_partners", lambda values: int(np.sum(np.asarray(values) > 1))),
            paired_cells=("paired_cells", "sum"),
        ),
        on=["SampleID", "Diagnosis"],
        how="left",
    )
)
for column in ["paired_heavy_clonotypes", "exact_paired_clonotypes", "heavy_clonotypes_with_multiple_lights", "paired_cells"]:
    pairing_qc[column] = pairing_qc[column].fillna(0)
pairing_qc["paired_heavy_fraction"] = pairing_qc["paired_heavy_clonotypes"] / pairing_qc["total_heavy_clonotypes"].replace(0, np.nan)
pairing_qc["multi_light_fraction_among_paired"] = (
    pairing_qc["heavy_clonotypes_with_multiple_lights"] / pairing_qc["paired_heavy_clonotypes"].replace(0, np.nan)
)

# Pairing-selection audit: paired versus unpaired heavy-clonotype SHM within
# each participant, followed by disease-control comparisons of that difference.
unpaired_heavy = heavy_receptors[~heavy_receptors["mapping_key"].isin(paired_mapping_keys)].copy()
pairing_selection_rates = pd.concat(
    [
        pooled_participant_rates(paired_heavy, "Paired heavy clonotypes"),
        pooled_participant_rates(unpaired_heavy, "Unpaired heavy clonotypes"),
    ],
    ignore_index=True,
)
pairing_selection_wide = pairing_selection_rates.pivot_table(
    index=["SampleID", "Diagnosis"], columns="estimand", values="overall_shm_rate", aggfunc="first"
).dropna().reset_index()
pairing_selection_wide["paired_minus_unpaired_shm"] = (
    pairing_selection_wide["Paired heavy clonotypes"] - pairing_selection_wide["Unpaired heavy clonotypes"]
)
pairing_selection_effects = compare_groups(
    pairing_selection_wide, "paired_minus_unpaired_shm", [], seed_offset=7000
)


# Germline-edge-free within-lineage divergence among lineages with >=2 distinct observed alignments.
db = pd.read_csv(
    ROOT / "bcr_heavy_lineages_d015_clone-pass.tsv",
    sep="\t",
    usecols=["sampleid", "diagnosis", "clone_id", "sequence_alignment"],
    low_memory=False,
)
db["lineage_id"] = db["sampleid"].astype(str) + "::" + db["clone_id"].astype(str)


def hdist(a, b):
    pairs = [(x, y) for x, y in zip(str(a).upper(), str(b).upper()) if x in "ACGT" and y in "ACGT"]
    return sum(x != y for x, y in pairs) / len(pairs) if pairs else np.nan


lineage_rows = []
for lineage_id, group in db.groupby("lineage_id", sort=False):
    sequences = list(dict.fromkeys(group["sequence_alignment"].dropna().astype(str)))
    if len(sequences) < 2:
        continue
    if len(sequences) > 50:
        indices = np.linspace(0, len(sequences) - 1, 50, dtype=int)
        selected = [sequences[index] for index in indices]
    else:
        selected = sequences
    distances = np.asarray([hdist(a, b) for a, b in combinations(selected, 2)], dtype=float)
    distances = distances[np.isfinite(distances)]
    if not len(distances):
        continue
    lineage_rows.append(
        {
            "lineage_id": lineage_id,
            "SampleID": group["sampleid"].iloc[0],
            "Diagnosis": group["diagnosis"].iloc[0],
            "n_unique_observed_sequences": len(sequences),
            "mean_pairwise_observed_divergence": float(np.mean(distances)),
            "median_pairwise_observed_divergence": float(np.median(distances)),
            "lineage_subsampled": len(sequences) > 50,
        }
    )
lineage_divergence = pd.DataFrame(lineage_rows)
participant_divergence = (
    lineage_divergence.groupby(["SampleID", "Diagnosis"], as_index=False)
    .agg(
        expanded_diversified_lineages=("lineage_id", "nunique"),
        mean_within_lineage_divergence=("mean_pairwise_observed_divergence", "mean"),
        median_within_lineage_divergence=("mean_pairwise_observed_divergence", "median"),
    )
)
participant_divergence_effects = compare_groups(
    participant_divergence, "mean_within_lineage_divergence", [], seed_offset=3000
)


clone_weighting.to_csv(ROOT / "Table_BGL17_participant_clone_weighting_SHM.csv", index=False)
clone_weighting_effects.to_csv(ROOT / "Table_BGL18_clone_weighting_SHM_effects.csv", index=False)
isotype_shm.to_csv(ROOT / "Table_BGL19_participant_isotype_matched_SHM.csv", index=False)
isotype_effects.to_csv(ROOT / "Table_BGL20_isotype_matched_SHM_effects.csv", index=False)
lineage_divergence.to_csv(ROOT / "Table_BGL21_germline_edge_free_lineage_divergence.csv", index=False)
participant_divergence.to_csv(ROOT / "Table_BGL22_participant_germline_edge_free_divergence.csv", index=False)
participant_divergence_effects.to_csv(ROOT / "Table_BGL23_germline_edge_free_divergence_effects.csv", index=False)
paired_participant_shm.to_csv(ROOT / "Table_BGL24_participant_paired_clonotype_SHM.csv", index=False)
paired_clonotype_effects.to_csv(ROOT / "Table_BGL25_paired_clonotype_SHM_effects.csv", index=False)
paired_status_shm.to_csv(ROOT / "Table_BGL26_participant_paired_expansion_SHM.csv", index=False)
paired_status_deltas.to_csv(ROOT / "Table_BGL27_paired_expanded_minus_singleton_SHM.csv", index=False)
paired_status_tests.to_csv(ROOT / "Table_BGL28_paired_expansion_SHM_tests.csv", index=False)
pairing_qc.to_csv(ROOT / "Table_BGL29_paired_clonotype_mapping_QC.csv", index=False)
paired_status_diagnosis_effects.to_csv(ROOT / "Table_BGL30_paired_expansion_diagnosis_effects.csv", index=False)
pairing_selection_wide.to_csv(ROOT / "Table_BGL31_participant_pairing_selection_SHM.csv", index=False)
pairing_selection_effects.to_csv(ROOT / "Table_BGL32_pairing_selection_SHM_effects.csv", index=False)

print("\nIsotype-matched SHM effects")
print(isotype_effects.to_string(index=False))
print("\nClone-weighting sensitivity")
print(clone_weighting_effects.to_string(index=False))
print("\nGermline-edge-free within-lineage divergence")
print(participant_divergence_effects.to_string(index=False))
print("\nPaired-clonotype-resolved heavy-chain SHM")
print(paired_clonotype_effects.to_string(index=False))
print("\nWithin-participant expanded-minus-singleton paired-clonotype SHM")
print(paired_status_tests.to_string(index=False))
print("\nPaired-clonotype mapping QC by diagnosis")
print(pairing_qc.groupby("Diagnosis")[["paired_heavy_fraction", "multi_light_fraction_among_paired"]].median().to_string())
print("\nDisease-control effects on the paired expanded-minus-singleton contrast")
print(paired_status_diagnosis_effects.to_string(index=False))
print("\nPairing-selection sensitivity")
print(pairing_selection_effects.to_string(index=False))

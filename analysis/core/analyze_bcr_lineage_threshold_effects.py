#!/usr/bin/env python
"""Recompute participant-level BCR lineage branch length across clone thresholds."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parent / "High Impact Additional Analyses" / "BCR Germline Lineages"
THRESHOLDS = {"d010": 0.10, "d015": 0.15, "d020": 0.20}
CONTRASTS = [("CD", "Control"), ("UC", "Control")]


def hdist(a, b):
    pairs = [(x, y) for x, y in zip(str(a).upper(), str(b).upper()) if x in "ACGT" and y in "ACGT"]
    return sum(x != y for x, y in pairs) / len(pairs) if pairs else 1.0


def lineage_mst_length(group):
    if len(group) < 1:
        return 0.0
    observed = group["sequence_alignment"].astype(str).tolist()[:100]
    nodes = [str(group["germline_alignment"].iloc[0])] + observed
    distances = np.zeros((len(nodes), len(nodes)), dtype=float)
    for i in range(len(nodes)):
        for j in range(i):
            distances[i, j] = distances[j, i] = hdist(nodes[i], nodes[j])
    return float(minimum_spanning_tree(distances).toarray().sum())


def bh(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p), dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
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


participant_rows = []
for label, threshold in THRESHOLDS.items():
    db = pd.read_csv(
        ROOT / f"bcr_heavy_lineages_{label}_clone-pass.tsv",
        sep="\t",
        usecols=[
            "sampleid",
            "diagnosis",
            "clone_id",
            "sequence_alignment",
            "germline_alignment",
        ],
        low_memory=False,
    )
    db["lineage_id"] = db["sampleid"].astype(str) + "::" + db["clone_id"].astype(str)
    lineage_rows = []
    for lineage_id, group in db.groupby("lineage_id", sort=False):
        lineage_rows.append(
            {
                "lineage_id": lineage_id,
                "SampleID": group["sampleid"].iloc[0],
                "Diagnosis": group["diagnosis"].iloc[0],
                "n_unique_sequences": len(group),
                "total_MST_branch_length": lineage_mst_length(group),
            }
        )
    lineages = pd.DataFrame(lineage_rows)
    for (sample_id, diagnosis), group in lineages.groupby(["SampleID", "Diagnosis"], sort=False):
        positive = group.loc[group["total_MST_branch_length"] > 0, "total_MST_branch_length"]
        participant_rows.append(
            {
                "threshold": threshold,
                "SampleID": sample_id,
                "Diagnosis": diagnosis,
                "n_lineages": len(group),
                "expanded_lineages": int((group["n_unique_sequences"] >= 2).sum()),
                "mean_MST_branch_length": float(positive.mean()) if len(positive) else 0.0,
            }
        )

participants = pd.DataFrame(participant_rows)
effect_rows = []
for threshold, table in participants.groupby("threshold", sort=True):
    for contrast_index, (disease, control) in enumerate(CONTRASTS):
        x = table.loc[table["Diagnosis"] == disease, "mean_MST_branch_length"].dropna().to_numpy()
        y = table.loc[table["Diagnosis"] == control, "mean_MST_branch_length"].dropna().to_numpy()
        low, high = bootstrap_median_difference(x, y, seed=20260824 + int(threshold * 100) + contrast_index)
        effect_rows.append(
            {
                "threshold": threshold,
                "contrast": f"{disease} vs Control",
                "n_disease": len(x),
                "n_control": len(y),
                "median_difference": float(np.median(x) - np.median(y)),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
            }
        )

effects = pd.DataFrame(effect_rows)
effects["FDR"] = bh(effects["p_value"])
participants.to_csv(ROOT / "Table_BGL15_threshold_participant_lineage_metrics.csv", index=False)
effects.to_csv(ROOT / "Table_BGL16_threshold_disease_effects.csv", index=False)
print(effects.to_string(index=False))

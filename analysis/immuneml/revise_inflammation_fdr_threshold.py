#!/usr/bin/env python
"""Reclassify completed inflammation-stratified results at FDR q<0.10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FDR_THRESHOLD = 0.10
RAW_P_THRESHOLD = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    p = args.results

    interactions = pd.read_csv(p / "within_diagnosis_motif_microbiome_interactions.csv", low_memory=False)
    three_way = pd.read_csv(p / "diagnosis_inflammation_motif_three_way_interactions.csv", low_memory=False)
    differential = pd.read_csv(p / "within_diagnosis_inflammation_differential.csv", low_memory=False)
    community = pd.read_csv(p / "community_concordance_by_diagnosis_inflammation.csv")
    multivariate = pd.read_csv(p / "multivariate_inflammation_separation.csv")
    community_primary = community[community["covariate_mode"].eq("source_batch")].copy()

    feature_sig = interactions[interactions["interaction_q_within_diagnosis"] < FDR_THRESHOLD].copy()
    feature_sig["significant_diagnosis_fdr_0_10"] = True
    feature_sig["significant_global_fdr_0_10"] = feature_sig["interaction_q_global"] < FDR_THRESHOLD
    feature_sig.to_csv(p / "significant_feature_interactions_fdr_0.10.csv", index=False)

    three_way_sig = three_way[three_way["q_global"] < FDR_THRESHOLD].copy()
    three_way_sig["significant_global_fdr_0_10"] = True
    three_way_sig.to_csv(p / "significant_three_way_interactions_fdr_0.10.csv", index=False)

    community_sig = community_primary[
        (community_primary["q_heterogeneity_diagnosis"] < FDR_THRESHOLD)
        | (community_primary["q_heterogeneity_global"] < FDR_THRESHOLD)
    ].copy()
    community_sig["significant_within_diagnosis_fdr_0_10"] = community_sig["q_heterogeneity_diagnosis"] < FDR_THRESHOLD
    community_sig["significant_global_fdr_0_10"] = community_sig["q_heterogeneity_global"] < FDR_THRESHOLD
    community_sig.to_csv(p / "significant_community_heterogeneity_fdr_0.10.csv", index=False)

    raw_feature = interactions[interactions["interaction_p"] < RAW_P_THRESHOLD].copy()
    raw_feature["passes_fdr_q_0_10_within_diagnosis"] = raw_feature["interaction_q_within_diagnosis"] < FDR_THRESHOLD
    raw_feature["passes_fdr_q_0_10_global"] = raw_feature["interaction_q_global"] < FDR_THRESHOLD
    raw_feature.to_csv(p / "raw_p_lt_0.01_feature_interactions.csv", index=False)

    raw_three_way = three_way[three_way["p_value"] < RAW_P_THRESHOLD].copy()
    raw_three_way["passes_fdr_q_0_10_global"] = raw_three_way["q_global"] < FDR_THRESHOLD
    raw_three_way.to_csv(p / "raw_p_lt_0.01_three_way_interactions.csv", index=False)

    raw_differential = differential[differential["p_value"] < RAW_P_THRESHOLD].copy()
    raw_differential["passes_fdr_q_0_10_within_class"] = raw_differential["q_within_diagnosis_class"] < FDR_THRESHOLD
    raw_differential["passes_fdr_q_0_10_global"] = raw_differential["q_global"] < FDR_THRESHOLD
    raw_differential.to_csv(p / "raw_p_lt_0.01_differential_features.csv", index=False)

    raw_community = community_primary[community_primary["p_heterogeneity"] < RAW_P_THRESHOLD].copy()
    raw_community["passes_fdr_q_0_10_within_diagnosis"] = raw_community["q_heterogeneity_diagnosis"] < FDR_THRESHOLD
    raw_community["passes_fdr_q_0_10_global"] = raw_community["q_heterogeneity_global"] < FDR_THRESHOLD
    raw_community.to_csv(p / "raw_p_lt_0.01_community_heterogeneity.csv", index=False)

    raw_multivariate = multivariate[multivariate["p_value"] < RAW_P_THRESHOLD].copy()
    raw_multivariate["passes_fdr_q_0_10_global"] = raw_multivariate["q_global"] < FDR_THRESHOLD
    raw_multivariate.to_csv(p / "raw_p_lt_0.01_multivariate.csv", index=False)

    counts = [
        ("Diagnosis-specific motif-by-inflammation interactions", "within diagnosis", len(interactions), int((interactions["interaction_q_within_diagnosis"] < FDR_THRESHOLD).sum())),
        ("Motif-by-inflammation interactions", "global", len(interactions), int((interactions["interaction_q_global"] < FDR_THRESHOLD).sum())),
        ("Diagnosis-by-inflammation-by-motif interactions", "global", len(three_way), int((three_way["q_global"] < FDR_THRESHOLD).sum())),
        ("Inflammation differential features", "within diagnosis/class", len(differential), int((differential["q_within_diagnosis_class"] < FDR_THRESHOLD).sum())),
        ("Inflammation differential features", "global", len(differential), int((differential["q_global"] < FDR_THRESHOLD).sum())),
        ("Community concordance heterogeneity", "within diagnosis", len(community_primary), int((community_primary["q_heterogeneity_diagnosis"] < FDR_THRESHOLD).sum())),
        ("Community concordance heterogeneity", "global", len(community_primary), int((community_primary["q_heterogeneity_global"] < FDR_THRESHOLD).sum())),
        ("Multivariate motif-block separation", "global", len(multivariate), int((multivariate["q_global"] < FDR_THRESHOLD).sum())),
    ]
    pd.DataFrame(counts, columns=["analysis_family", "fdr_scope", "tests", "significant_q_lt_0_10"]).to_csv(
        p / "fdr_0.10_significance_summary.csv", index=False
    )
    raw_counts = [
        ("Diagnosis-specific motif-by-inflammation interactions", len(interactions), len(raw_feature)),
        ("Diagnosis-by-inflammation-by-motif interactions", len(three_way), len(raw_three_way)),
        ("Inflammation differential features", len(differential), len(raw_differential)),
        ("Community concordance heterogeneity", len(community_primary), len(raw_community)),
        ("Multivariate motif-block separation", len(multivariate), len(raw_multivariate)),
    ]
    raw_summary = pd.DataFrame(raw_counts, columns=["analysis_family", "tests", "raw_p_lt_0_01"])
    raw_summary["percent_raw_p_lt_0_01"] = 100 * raw_summary["raw_p_lt_0_01"] / raw_summary["tests"]
    raw_summary.to_csv(p / "raw_p_0.01_significance_summary.csv", index=False)

    summary_path = p / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "fdr_threshold": FDR_THRESHOLD,
            "raw_p_exploratory_threshold": RAW_P_THRESHOLD,
            "raw_p_lt_0_01_interactions": int(len(raw_feature)),
            "raw_p_lt_0_01_three_way": int(len(raw_three_way)),
            "raw_p_lt_0_01_differential": int(len(raw_differential)),
            "raw_p_lt_0_01_community": int(len(raw_community)),
            "raw_p_lt_0_01_multivariate": int(len(raw_multivariate)),
            "interaction_q_diagnosis_lt_threshold": int((interactions["interaction_q_within_diagnosis"] < FDR_THRESHOLD).sum()),
            "interaction_q_global_lt_threshold": int((interactions["interaction_q_global"] < FDR_THRESHOLD).sum()),
            "three_way_q_global_lt_threshold": int((three_way["q_global"] < FDR_THRESHOLD).sum()),
            "differential_q_class_lt_threshold": int((differential["q_within_diagnosis_class"] < FDR_THRESHOLD).sum()),
            "differential_q_global_lt_threshold": int((differential["q_global"] < FDR_THRESHOLD).sum()),
            "community_heterogeneity_q_global_lt_threshold": int((community_primary["q_heterogeneity_global"] < FDR_THRESHOLD).sum()),
            "community_heterogeneity_q_diagnosis_lt_threshold": int((community_primary["q_heterogeneity_diagnosis"] < FDR_THRESHOLD).sum()),
            "multivariate_q_global_lt_threshold": int((multivariate["q_global"] < FDR_THRESHOLD).sum()),
        }
    )
    for old_key in [key for key in summary if key.endswith("lt_0_05")]:
        summary.pop(old_key)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

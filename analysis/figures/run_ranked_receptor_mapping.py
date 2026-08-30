#!/usr/bin/env python3
"""Ranked analyses 3-4: HLA-annotated TCR mapping and paired-BCR convergence."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "High Impact Additional Analyses"
OUT = BASE / "Literature Guided Ranked Analyses"
OUT.mkdir(parents=True, exist_ok=True)
VDJDB = Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 2 TCR\GLIPH2\gliph2_pairwise_tcronly\vdjdb-db-master\merged\vdjdb_merged_raw.tsv")
IMMUNO = Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 3 BCR\BCR Architecture Analyses\ImmunoMatch paired BCR enrichment PBMC2026\unique_paired_bcr_immunomatch_candidate_scored.csv")
HUAR_SUMMARY = Path(r"C:/path/to/private-legacy-manuscript-assets\Figure 5\BCR_p_lt_0p05_huARdb_match_per_query_summary.csv")
HUAR_PAIRED = Path(r"C:/path/to/private-legacy-manuscript-assets\Figure 5\BCR_p_lt_0p05_huARdb_exact_near_match_paired_heavy_light_only.csv")

sns.set_theme(style="whitegrid", context="talk")


def fdr_bh(pvalues):
    p = np.asarray(pvalues, float); out = np.full(p.shape, np.nan); ok = np.isfinite(p)
    if not ok.any(): return out
    x = p[ok]; o = np.argsort(x); y = x[o] * len(x) / np.arange(1, len(x) + 1)
    y = np.clip(np.minimum.accumulate(y[::-1])[::-1], 0, 1)
    rev = np.empty_like(o); rev[o] = np.arange(len(o)); out[ok] = y[rev]
    return out


def gene(x):
    return str(x).strip().split("*")[0] if pd.notna(x) else ""


def cdr3_core(x):
    if pd.isna(x): return ""
    s = str(x).strip().upper().replace(" ", "")
    if s.startswith("C") and len(s) > 4: s = s[1:]
    if s.endswith(("F", "W")) and len(s) > 4: s = s[:-1]
    return s


def series_from_batch(x):
    return str(x).rstrip("AB")


def tcr_reference_mapping():
    cohort = pd.read_csv(BASE / "Paired TCR Sequence State" /
                         "Table_TSS1_paired_alpha_beta_clone_sequence_state.csv")
    cohort["alpha_core"] = cohort.alpha_cdr3.map(cdr3_core)
    cohort["beta_core"] = cohort.beta_cdr3.map(cdr3_core)
    cohort["alpha_v_gene"] = cohort.alpha_v.map(gene)
    cohort["beta_v_gene"] = cohort.beta_v.map(gene)
    cohort["alpha_j_gene"] = cohort.alpha_j.map(gene)
    cohort["beta_j_gene"] = cohort.beta_j.map(gene)
    cohort["cohort_row"] = np.arange(len(cohort))

    usecols = ["cdr3.alpha", "v.alpha", "j.alpha", "cdr3.beta", "v.beta", "j.beta",
               "species", "mhc.a", "mhc.b", "mhc.class", "antigen.epitope",
               "antigen.gene", "antigen.species", "reference.id", "method.identification",
               "method.singlecell", "method.sequencing", "method.verification", "meta.tissue"]
    ref = pd.read_csv(VDJDB, sep="\t", usecols=usecols, low_memory=False)
    ref = ref[(ref.species == "HomoSapiens") & ref["cdr3.beta"].notna() &
              ref["mhc.a"].notna() & (ref["mhc.a"].astype(str).str.len() > 2)].copy()
    ref["alpha_core"] = ref["cdr3.alpha"].map(cdr3_core)
    ref["beta_core"] = ref["cdr3.beta"].map(cdr3_core)
    ref["alpha_v_gene"] = ref["v.alpha"].map(gene)
    ref["beta_v_gene"] = ref["v.beta"].map(gene)
    ref["alpha_j_gene"] = ref["j.alpha"].map(gene)
    ref["beta_j_gene"] = ref["j.beta"].map(gene)
    ref["ref_row"] = np.arange(len(ref))

    ref_cols = ["ref_row", "alpha_core", "beta_core", "alpha_v_gene", "beta_v_gene",
                "alpha_j_gene", "beta_j_gene", "mhc.a", "mhc.b", "mhc.class",
                "antigen.epitope", "antigen.gene", "antigen.species", "reference.id",
                "method.identification", "method.singlecell", "method.sequencing",
                "method.verification", "meta.tissue"]
    exact_ref = ref[(ref.alpha_core.str.len() > 0) & (ref.beta_core.str.len() > 0)][ref_cols]
    exact = cohort.merge(exact_ref, on=["alpha_core", "beta_core", "alpha_v_gene",
                                        "beta_v_gene"], suffixes=("", "_ref"))
    if len(exact):
        exact["match_tier"] = "exact paired alpha-beta + V genes"
        exact["reference_HLA_available"] = True
        exact["participant_HLA_compatibility"] = "not assessable: no cohort HLA genotype or GEX BAM"

    beta_ref = ref[ref.beta_core.str.len() > 0][ref_cols].drop(columns=["alpha_core", "alpha_v_gene"])
    beta = cohort.merge(beta_ref, on=["beta_core", "beta_v_gene"], suffixes=("", "_ref"))
    beta["match_tier"] = "exact beta + V gene; alpha/HLA unresolved"
    beta["reference_HLA_available"] = True
    beta["participant_HLA_compatibility"] = "not assessable: no cohort HLA genotype or GEX BAM"

    common = ["SampleID", "PatientID", "Diagnosis", "Inflammation", "Biologic", "Batch",
              "acquisition_series", "clone_id", "alpha_v", "alpha_j", "alpha_cdr3",
              "beta_v", "beta_j", "beta_cdr3", "n_cells", "dominant_state",
              "match_tier", "mhc.a", "mhc.b", "mhc.class", "antigen.epitope",
              "antigen.gene", "antigen.species", "reference.id", "method.identification",
              "method.singlecell", "method.sequencing", "method.verification", "meta.tissue",
              "participant_HLA_compatibility"]
    mappings = pd.concat([exact[common] if len(exact) else pd.DataFrame(columns=common),
                          beta[common]], ignore_index=True).drop_duplicates()
    mappings.to_csv(OUT / "Table_RA3_TCR_VDJdb_HLA_annotated_matches.csv", index=False)

    totals = cohort.groupby("SampleID").agg(total_paired_clone_cells=("n_cells", "sum"),
                                             total_paired_clones=("clone_id", "nunique")).reset_index()
    part = mappings.groupby(["SampleID", "Diagnosis", "match_tier", "antigen.species"], dropna=False).agg(
        matched_clone_cells=("n_cells", "sum"), matched_clones=("clone_id", "nunique"),
        reference_HLA=("mhc.a", lambda x: ";".join(sorted(set(map(str, x))))),
        epitopes=("antigen.epitope", lambda x: ";".join(sorted(set(map(str, x))))[:2000])
    ).reset_index().merge(totals, on="SampleID", how="left")
    part["matched_cell_fraction"] = part.matched_clone_cells / part.total_paired_clone_cells
    part.to_csv(OUT / "Table_RA3_TCR_reference_burden_by_participant.csv", index=False)

    all_part = cohort[["SampleID", "Diagnosis", "acquisition_series"]].drop_duplicates()
    tests = []
    for (tier, antigen), g in part.groupby(["match_tier", "antigen.species"], dropna=False):
        carriers = set(g.SampleID)
        ibd = all_part[all_part.Diagnosis.isin(["CD", "UC"])]
        cd = set(ibd.loc[ibd.Diagnosis == "CD", "SampleID"]); uc = set(ibd.loc[ibd.Diagnosis == "UC", "SampleID"])
        a, b = len(cd & carriers), len(cd - carriers); c, dd = len(uc & carriers), len(uc - carriers)
        if a + c < 3: continue
        odds, p = stats.fisher_exact([[a, b], [c, dd]])
        tests.append({"match_tier": tier, "antigen_species": antigen, "CD_carriers": a,
                      "CD_total": len(cd), "UC_carriers": c, "UC_total": len(uc),
                      "odds_ratio_CD_vs_UC": odds, "fisher_p": p})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["FDR"] = tests.groupby("match_tier")["fisher_p"].transform(fdr_bh)
    tests.to_csv(OUT / "Table_RA3_TCR_antigen_carrier_tests_CD_vs_UC.csv", index=False)

    # The published CAIT motif is reported separately and cannot be HLA-confirmed here.
    ca = cohort.copy()
    ca["CAIT_strict"] = (ca.alpha_v_gene.eq("TRAV12-1") & ca.alpha_j_gene.eq("TRAJ6") &
                         ca.alpha_core.str.match(r"^VV..A.GGSYIPT$", na=False))
    ca["CAIT_TRAV12_1_TRAJ6_family"] = (ca.alpha_v_gene.eq("TRAV12-1") &
                                        ca.alpha_j_gene.eq("TRAJ6") & ca.alpha_core.str.len().eq(13))
    ca_part = ca.groupby(["SampleID", "Diagnosis", "acquisition_series"], as_index=False).agg(
        CAIT_strict=("CAIT_strict", "max"), CAIT_TRAV12_1_TRAJ6_family=("CAIT_TRAV12_1_TRAJ6_family", "max"))
    ca_part.to_csv(OUT / "Table_RA3_CAIT_participant_screen.csv", index=False)

    manifest = pd.DataFrame([
        ["reference", "VDJdb merged raw, human entries with explicit reference HLA"],
        ["strict tier", "exact normalized alpha and beta CDR3 plus alpha and beta V genes"],
        ["sensitive tier", "exact normalized beta CDR3 plus beta V gene"],
        ["cohort HLA", "not available; no BAM/BAI or genotype files found in manuscript inputs"],
        ["interpretation", "reference-HLA annotated candidate mapping, not participant-HLA confirmed specificity"],
    ], columns=["item", "value"])
    manifest.to_csv(OUT / "Table_RA3_TCR_mapping_manifest.csv", index=False)

    if len(tests):
        top = tests.sort_values(["FDR", "fisher_p"]).head(15).copy()
        top["label"] = top.antigen_species.astype(str) + "\n" + top.match_tier.str.replace(";.*", "", regex=True)
        top["CD_fraction"] = top.CD_carriers / top.CD_total
        top["UC_fraction"] = top.UC_carriers / top.UC_total
        pl = top.melt(id_vars=["label", "FDR"], value_vars=["CD_fraction", "UC_fraction"],
                      var_name="Diagnosis", value_name="carrier_fraction")
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        sns.barplot(data=pl, y="label", x="carrier_fraction", hue="Diagnosis",
                    palette={"CD_fraction": "#d8832f", "UC_fraction": "#607bb2"}, ax=ax)
        ax.set_xlabel("Participant carrier fraction")
        ax.set_ylabel("")
        ax.set_title("Reference-HLA annotated TCR matches\n(participant HLA compatibility unavailable)")
        fig.savefig(OUT / "Figure_RA3_HLA_annotated_TCR_mapping.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / "Figure_RA3_HLA_annotated_TCR_mapping.pdf", bbox_inches="tight")
        plt.close(fig)
    return cohort, mappings, part, tests


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n); self.r = np.zeros(n, dtype=np.int8)
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return int(x)
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.r[a] < self.r[b]: a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]: self.r[a] += 1


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else 999


def annotate_bcr_phenotype(d):
    bgl = BASE / "BCR Germline Lineages"
    ann = pd.read_csv(OUT / "Table_RA4_exact_paired_BCR_cell_annotations.csv")
    q = pd.read_csv(bgl / "Table_BGL1_query_sequence_metadata.csv",
                    usecols=["SampleID", "sequence_id", "v_gene", "j_gene", "junction_aa", "receptor_key", "count"])
    q["heavy_core"] = q.junction_aa.map(cdr3_core)
    q["v_gene"] = q.v_gene.map(gene); q["j_gene"] = q.j_gene.map(gene)

    shm = pd.read_csv(bgl / "Table_BGL3_sequence_region_SHM_RS.csv",
                      usecols=["sequence_id", "informative_nt", "mutations"])
    shm = shm.groupby("sequence_id", as_index=False).agg(informative_nt=("informative_nt", "sum"),
                                                         mutations=("mutations", "sum"))
    qs = q.merge(shm, on="sequence_id", how="left")
    qs = qs.groupby(["SampleID", "v_gene", "j_gene", "heavy_core"], as_index=False).agg(
        informative_nt=("informative_nt", "sum"), mutations=("mutations", "sum"))
    qs["heavy_SHM_rate"] = qs.mutations / qs.informative_nt.replace(0, np.nan)
    ann = ann.merge(qs[["SampleID", "v_gene", "j_gene", "heavy_core", "heavy_SHM_rate"]],
                    on=["SampleID", "v_gene", "j_gene", "heavy_core"], how="left")
    return d.merge(ann, on=["SampleID", "v_gene", "j_gene", "heavy_core", "light_core"], how="left")


def bcr_convergence():
    d = pd.read_csv(BASE / "paired_BCR_clonotypes_for_nested_ML.csv")
    d["v_gene"] = d.v_call.map(gene); d["j_gene"] = d.j_call.map(gene)
    d["heavy_core"] = d.chain1_cdr3.map(cdr3_core); d["light_core"] = d.chain2_cdr3.map(cdr3_core)
    d["heavy_len"] = d.heavy_core.str.len(); d["light_len"] = d.light_core.str.len()
    d["row_id"] = np.arange(len(d))
    clinical = pd.read_csv(BASE / "Priority Analyses" / "Table_PA_clinical_metadata.csv")
    clinical["acquisition_series"] = clinical.Batch.map(series_from_batch)
    d = d.merge(clinical[["SampleID", "Inflammation1", "Biologic", "Calprotectin", "acquisition_series"]],
                on="SampleID", how="left")
    d = annotate_bcr_phenotype(d)

    uf = UnionFind(len(d)); edges = []
    for _, g in d.groupby(["v_gene", "j_gene", "heavy_len", "light_len"], sort=False):
        ix = g.index.to_numpy(); h = g.heavy_core.to_numpy(); l = g.light_core.to_numpy(); s = g.SampleID.to_numpy()
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                if s[a] == s[b]: continue
                hd, ld = hamming(h[a], h[b]), hamming(l[a], l[b])
                if hd <= 1 and ld <= 1 and hd + ld <= 2:
                    uf.union(ix[a], ix[b]); edges.append((ix[a], ix[b], hd, ld))
    roots = np.array([uf.find(i) for i in range(len(d))])
    d["component_root"] = roots
    n_part = d.groupby("component_root").SampleID.nunique()
    keep = set(n_part[n_part >= 2].index)
    conv = d[d.component_root.isin(keep)].copy()
    root_order = {r: i + 1 for i, r in enumerate(sorted(keep, key=lambda x: (-n_part[x], x)))}
    conv["convergence_cluster"] = conv.component_root.map(lambda x: f"BCRCONV_{root_order[x]:04d}")
    conv.to_csv(OUT / "Table_RA4_paired_BCR_convergent_cluster_members.csv", index=False)
    pd.DataFrame(edges, columns=["row1", "row2", "heavy_hamming", "light_hamming"]).to_csv(
        OUT / "Table_RA4_paired_BCR_convergence_edges.csv", index=False)

    total_by_dx = d[["SampleID", "Diagnosis1"]].drop_duplicates().groupby("Diagnosis1").SampleID.nunique().to_dict()
    summaries, tests = [], []
    for cid, g in conv.groupby("convergence_cluster"):
        participants = g[["SampleID", "Diagnosis1", "acquisition_series"]].drop_duplicates()
        counts = participants.Diagnosis1.value_counts().to_dict()
        series = sorted(participants.acquisition_series.dropna().astype(str).unique())
        exact_pairs = g.groupby(["heavy_core", "light_core"]).SampleID.nunique()
        summaries.append({
            "convergence_cluster": cid, "n_members": len(g), "n_participants": participants.SampleID.nunique(),
            "n_CD": counts.get("CD", 0), "n_UC": counts.get("UC", 0), "n_Control": counts.get("Control", 0),
            "n_acquisition_series": len(series), "acquisition_series": ";".join(series),
            "replicates_across_series": len(series) >= 2, "max_exact_pair_participants": int(exact_pairs.max()),
            "total_cells": g.n_cells.sum(), "max_clone_cells": g.n_cells.max(),
            "mean_switched_fraction": np.average(g.switched_fraction.fillna(0), weights=g.n_cells),
            "mean_plasma_fraction": np.average(g.plasma_fraction.fillna(0), weights=g.n_cells),
            "mean_heavy_SHM_rate": np.average(g.heavy_SHM_rate.fillna(0), weights=g.n_cells),
            "isotypes": ";".join(sorted(set(";".join(g.isotypes.dropna()).split(";"))))[:1000],
            "states": ";".join(sorted(set(";".join(g.states.dropna()).split(";"))))[:1000],
        })
        cd, uc = counts.get("CD", 0), counts.get("UC", 0)
        if cd + uc >= 3:
            odds, p = stats.fisher_exact([[cd, total_by_dx.get("CD", 0) - cd],
                                          [uc, total_by_dx.get("UC", 0) - uc]])
            tests.append({"convergence_cluster": cid, "CD_carriers": cd, "CD_total": total_by_dx.get("CD", 0),
                          "UC_carriers": uc, "UC_total": total_by_dx.get("UC", 0),
                          "odds_ratio_CD_vs_UC": odds, "fisher_p": p})
    summary = pd.DataFrame(summaries)
    test = pd.DataFrame(tests)
    if len(test):
        test["FDR"] = fdr_bh(test.fisher_p)
        summary = summary.merge(test, on="convergence_cluster", how="left")

    # Add full paired variable-region sequences and public-reference audit when available.
    im = pd.read_csv(IMMUNO)
    im["v_gene"] = im.heavy_v_trim.map(gene); im["j_gene"] = im.heavy_j_trim.map(gene)
    im["heavy_core"] = im.heavy_cdr3_core.map(cdr3_core); im["light_core"] = im.light_cdr3_core.map(cdr3_core)
    rep = conv.sort_values(["convergence_cluster", "n_cells"], ascending=[True, False]).drop_duplicates("convergence_cluster")
    rep = rep[["convergence_cluster", "SampleID", "v_gene", "j_gene", "heavy_core", "light_core", "n_cells"]]
    im_small = im.sort_values("n_cells", ascending=False).drop_duplicates(["v_gene", "j_gene", "heavy_core", "light_core"])
    rep = rep.merge(im_small[["v_gene", "j_gene", "heavy_core", "light_core", "pair_id", "heavy_vh_aa",
                              "light_vl_aa", "light_type", "light_v_trim", "light_j_trim",
                              "immunomatch_pairing_score"]],
                    on=["v_gene", "j_gene", "heavy_core", "light_core"], how="left")
    if HUAR_SUMMARY.exists():
        hu = pd.read_csv(HUAR_SUMMARY)
        hu = hu.groupby("pair_id", as_index=False).agg(n_huardb_hits=("n_huardb_hits", "max"),
                                                       n_huardb_paired_reference_hits=("n_huardb_paired_reference_hits", "max"),
                                                       huardb_categories=("top_huardb_sample_categories", "first"))
        rep = rep.merge(hu, on="pair_id", how="left")
    paired_public_rows = max(sum(1 for _ in open(HUAR_PAIRED, encoding="utf-8")) - 1, 0) if HUAR_PAIRED.exists() else 0
    rep["paired_huARdb_reference_file_rows"] = paired_public_rows
    summary = summary.merge(rep, on="convergence_cluster", how="left")

    summary["evidence_grade"] = np.select(
        [summary.replicates_across_series & (summary.n_participants >= 3),
         summary.replicates_across_series,
         summary.n_participants >= 3],
        ["A: multi-series, >=3 participants", "B: multi-series, 2 participants",
         "C: single-series, >=3 participants"],
        default="D: single-series, 2 participants")
    summary["priority_score"] = (
        np.log1p(summary.n_participants) + np.log1p(summary.max_clone_cells) +
        6 * summary.replicates_across_series.astype(float) + summary.mean_switched_fraction +
        summary.mean_plasma_fraction + 3 * summary.mean_heavy_SHM_rate.fillna(0) +
        np.minimum(-np.log10(summary.FDR.fillna(1).clip(lower=1e-6)), 6)
    )
    summary = summary.sort_values(["priority_score", "n_participants"], ascending=False)
    summary.to_csv(OUT / "Table_RA4_paired_BCR_convergence_clusters.csv", index=False)
    disease_candidates = summary[(summary.n_CD + summary.n_UC) > 0].copy()
    disease_candidates.head(50).to_csv(OUT / "Table_RA4_recombinant_antibody_priority_candidates.csv", index=False)
    if len(test): test.to_csv(OUT / "Table_RA4_BCR_convergence_CD_vs_UC_tests.csv", index=False)

    audit = pd.DataFrame([
        ["input clonotypes", len(d)], ["convergent clusters", len(summary)],
        ["convergent participants", conv.SampleID.nunique()],
        ["definition", "same heavy V and J; same heavy/light CDR3 lengths; <=1 amino-acid mismatch per chain and <=2 total; >=2 participants"],
        ["paired public huARdb hits in prior strict file", paired_public_rows],
        ["antigen interpretation", "sequence-convergent candidates for recombinant testing; public light-only hits are not treated as antigen specificity"],
    ], columns=["item", "value"])
    audit.to_csv(OUT / "Table_RA4_BCR_convergence_manifest.csv", index=False)

    top = disease_candidates.head(15).copy().sort_values("priority_score")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    y = np.arange(len(top))
    cd_frac = top.n_CD / max(total_by_dx.get("CD", 1), 1); uc_frac = top.n_UC / max(total_by_dx.get("UC", 1), 1)
    axes[0].barh(y - .18, cd_frac, height=.35, label="CD", color="#d8832f")
    axes[0].barh(y + .18, uc_frac, height=.35, label="UC", color="#607bb2")
    axes[0].set_yticks(y, top.convergence_cluster)
    axes[0].set_xlabel("Participant carrier fraction")
    axes[0].set_title("A  Cross-participant paired-BCR convergence")
    axes[0].legend()
    sc = axes[1].scatter(top.mean_heavy_SHM_rate, top.mean_plasma_fraction,
                         s=40 + 35 * top.n_participants, c=top.priority_score, cmap="viridis", edgecolor="black")
    label_ids = set(disease_candidates.head(10).convergence_cluster)
    for _, r in top[top.convergence_cluster.isin(label_ids)].iterrows():
        axes[1].annotate(r.convergence_cluster, (r.mean_heavy_SHM_rate, r.mean_plasma_fraction), fontsize=8)
    axes[1].set_xlabel("Weighted heavy-chain SHM rate")
    axes[1].set_ylabel("Plasma-state fraction")
    axes[1].set_title("B  Recombinant-antibody prioritization")
    fig.colorbar(sc, ax=axes[1], label="Priority score")
    fig.savefig(OUT / "Figure_RA4_paired_BCR_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure_RA4_paired_BCR_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    return d, conv, summary, test


if __name__ == "__main__":
    tcr_reference_mapping()
    bcr_convergence()
    print(f"Wrote ranked analyses 3-4 to {OUT}")

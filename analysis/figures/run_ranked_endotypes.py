#!/usr/bin/env python3
"""Ranked analysis 5: participant-level adaptive immune endotypes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "High Impact Additional Analyses"
OUT = BASE / "Literature Guided Ranked Analyses"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260828)
sns.set_theme(style="whitegrid", context="talk")


def fdr_bh(pvalues):
    p = np.asarray(pvalues, float); out = np.full(p.shape, np.nan); ok = np.isfinite(p)
    if not ok.any(): return out
    x = p[ok]; o = np.argsort(x); y = x[o] * len(x) / np.arange(1, len(x) + 1)
    y = np.clip(np.minimum.accumulate(y[::-1])[::-1], 0, 1)
    rev = np.empty_like(o); rev[o] = np.arange(len(o)); out[ok] = y[rev]
    return out


def robust_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x); q25, q75 = np.nanpercentile(x, [25, 75]); scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < 1e-8: scale = np.nanstd(x)
    if not np.isfinite(scale) or scale < 1e-8: scale = 1.0
    return (x - med) / scale


def residualize_matrix(x, cov):
    c = pd.get_dummies(cov, drop_first=True, dtype=float)
    c.insert(0, "intercept", 1.0)
    cm = c.to_numpy(float)
    return x - cm @ np.linalg.lstsq(cm, x, rcond=None)[0]


def cramer_v(tab):
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum(); r, k = tab.shape
    return np.sqrt(chi2 / max(n * min(k - 1, r - 1), 1))


def prepare_features():
    integrated = pd.read_csv(BASE / "Table_HI_integrated_participant_features.csv")
    helper = pd.read_csv(OUT / "Table_RA1_helper_axis_analysis_dataset.csv")
    clinical = pd.read_csv(BASE / "Priority Analyses" / "Table_PA_clinical_metadata.csv")
    bgl = pd.read_csv(BASE / "BCR Germline Lineages" / "Table_BGL11_participant_lineage_metrics.csv")
    pa3 = pd.read_csv(BASE / "Priority Analyses" / "Table_PA3_BCR_lineage_diversification_by_participant.csv")
    csi = pd.read_csv(BASE / "Clone State Interactions" / "Table_CSI2_participant_expanded_minus_singleton_deltas.csv")

    # Reference-mapping burdens are calculated from unique matched cohort clonotypes,
    # avoiding double counting the same clonotype across duplicate VDJdb records.
    tm = pd.read_csv(OUT / "Table_RA3_TCR_VDJdb_HLA_annotated_matches.csv")
    tm = tm.drop_duplicates(["SampleID", "match_tier", "clone_id"])
    tb = tm.groupby(["SampleID", "match_tier"], as_index=False).agg(
        reference_matched_clones=("clone_id", "nunique"), reference_matched_cells=("n_cells", "sum"))
    tb["tier"] = np.where(tb.match_tier.str.startswith("exact paired"), "strict_paired", "beta_only")
    tb = tb.pivot_table(index="SampleID", columns="tier",
                        values=["reference_matched_clones", "reference_matched_cells"], fill_value=0)
    tb.columns = [f"tcr_{a}_{b}" for a, b in tb.columns]
    tb = tb.reset_index()
    ca = pd.read_csv(OUT / "Table_RA3_CAIT_participant_screen.csv")

    cm = pd.read_csv(OUT / "Table_RA4_paired_BCR_convergent_cluster_members.csv")
    cb = cm.groupby("SampleID", as_index=False).agg(
        bcr_convergent_clusters=("convergence_cluster", "nunique"),
        bcr_convergent_clone_cells=("n_cells", "sum"),
        bcr_convergent_switched_mean=("switched_fraction", "mean"),
        bcr_convergent_plasma_mean=("plasma_fraction", "mean"),
        bcr_convergent_SHM_mean=("heavy_SHM_rate", "mean"))

    csi_sel = csi[csi.module.isin(["Effector_cytotoxicity", "Th1_Tc1_inflammatory",
                                   "Tph_Tfh_like_B_cell_help", "IgA_mucosal_plasma_cell",
                                   "Plasmablast_plasma_differentiation", "Atypical_memory_CD11c_like"])]
    csi_w = csi_sel.pivot_table(index="SampleID", columns=["modality", "module"], values="delta")
    csi_w.columns = [f"clone_delta_{a}_{b}" for a, b in csi_w.columns]
    csi_w = csi_w.reset_index()

    d = integrated.merge(helper.drop(columns=[c for c in helper.columns if c in integrated.columns and c != "SampleID"]),
                          on="SampleID", how="left")
    d = d.merge(clinical[["SampleID", "Calprotectin", "Inflammation1", "Biologic"]], on="SampleID", how="left",
                suffixes=("", "_clinical"))
    d = d.merge(bgl.drop(columns=["Diagnosis", "Inflammation"]), on="SampleID", how="left")
    pa_keep = ["SampleID", "fraction_diversified_lineages", "median_v_region_nt_divergence",
               "median_cdr_minus_fwr_divergence", "median_lineage_cdr3_variants", "max_lineage_cdr3_variants"]
    d = d.merge(pa3[pa_keep], on="SampleID", how="left")
    for extra in [tb, ca.drop(columns=["Diagnosis", "acquisition_series"]), cb, csi_w]:
        d = d.merge(extra, on="SampleID", how="left")
    for c in [x for x in d if x.startswith("tcr_reference_") or x.startswith("bcr_convergent_") or x.startswith("CAIT_")]:
        d[c] = d[c].fillna(0)
    d["acquisition_series"] = d.Batch.astype(str).str.replace(r"[AB]$", "", regex=True)
    d["log_tcr_depth"] = np.log1p(d.tcr_total_cells)
    d["log_bcr_depth"] = np.log1p(d.bcr_total_cells)
    d["sex_male"] = (d.Sex.astype(str).str.upper() == "M").astype(float)
    d["log_calprotectin"] = np.log1p(pd.to_numeric(d.Calprotectin, errors="coerce"))
    return d


def consensus_cluster(x, k, n_iter=160):
    n, p = x.shape
    co, seen = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n_iter):
        si = np.sort(RNG.choice(n, max(int(.82 * n), k * 5), replace=False))
        fi = np.sort(RNG.choice(p, max(int(.78 * p), min(4, p)), replace=False))
        km = KMeans(n_clusters=k, n_init=20, random_state=20260828 + i)
        lab = km.fit_predict(x[np.ix_(si, fi)])
        seen[np.ix_(si, si)] += 1
        for cl in range(k):
            q = si[lab == cl]
            co[np.ix_(q, q)] += 1
    con = np.divide(co, seen, out=np.zeros_like(co), where=seen > 0)
    np.fill_diagonal(con, 1)
    # Partition consensus profiles rather than cutting a dendrogram, which can
    # turn one or two extreme participants into apparently high-silhouette clusters.
    labels = KMeans(n_clusters=k, n_init=100, random_state=20260828 + 1000 + k).fit_predict(con)
    vals = con[np.triu_indices(n, 1)]
    pac = np.mean((vals > .1) & (vals < .9))
    within = np.mean([con[np.ix_(labels == c, labels == c)][np.triu_indices((labels == c).sum(), 1)].mean()
                      if (labels == c).sum() > 1 else 0 for c in range(k)])
    sil = silhouette_score(x, labels) if len(np.unique(labels)) > 1 else np.nan
    mins = min(np.bincount(labels)) if len(np.unique(labels)) == k else 0
    return labels, con, {"k": k, "silhouette": sil, "PAC": pac,
                         "within_consensus": within, "min_cluster_size": mins,
                         "score": sil + (1 - pac) + within - (0.4 if mins < .08 * n else 0)}


def run_endotypes():
    d = prepare_features()
    ibd = d[d.Diagnosis1.isin(["CD", "UC"])].copy().reset_index(drop=True)
    blocks = {
        "TCR": [c for c in d if c.startswith("tcr_") and c not in
                ["tcr_total_cells", "tcr_paired_cells"]] +
               [c for c in d if c.startswith("clone_delta_TCR_")] +
               ["CAIT_strict", "CAIT_TRAV12_1_TRAJ6_family"],
        "BCR": [c for c in d if c.startswith("bcr_") and c not in
                ["bcr_total_cells", "bcr_paired_cells"]] +
               ["n_lineages", "expanded_lineages", "mean_lineage_shannon", "mean_MST_branch_length",
                "fraction_class_switched_lineages", "fraction_mixed_unswitched_switched",
                "fraction_diversified_lineages", "median_v_region_nt_divergence",
                "median_cdr_minus_fwr_divergence", "median_lineage_cdr3_variants", "max_lineage_cdr3_variants"] +
               [c for c in d if c.startswith("clone_delta_BCR_")],
        "Helper_axis": [c for c in d if c.startswith("cd4_mean_") or c.startswith("cd4_clone_delta_")] +
                       ["cd4_tfh_fraction", "cd4_hladr_memory_fraction", "expanded_paired_ab_fraction",
                        "expanded_tfh_fraction_of_paired"] +
                       [c for c in d if c.startswith("b_mean_")] +
                       ["b_plasma_fraction", "b_iga_plasma_fraction", "b_igg_plasma_fraction",
                        "b_switched_memory_fraction", "b_atypical_memory_fraction"],
    }
    # Deduplicate names, retain features measured in >=60% with non-zero variance.
    blocks = {b: list(dict.fromkeys([c for c in cols if c in ibd])) for b, cols in blocks.items()}
    selected, block_labels, transformed = [], [], []
    feature_manifest = []
    cov = ibd[["Age", "sex_male", "log_tcr_depth", "log_bcr_depth", "acquisition_series"]].copy()
    for block, cols in blocks.items():
        clean = []
        for c in cols:
            x = pd.to_numeric(ibd[c], errors="coerce")
            if x.notna().mean() < .60 or x.nunique(dropna=True) < 3: continue
            x = x.fillna(x.median())
            log_used = bool((x.min() >= 0) and (abs(stats.skew(x)) > 1.2))
            if log_used: x = np.log1p(x)
            z = robust_z(x)
            clean.append((c, z, log_used))
        if not clean: continue
        # Remove almost exact duplicate features within a block.
        mat = np.column_stack([z for _, z, _ in clean])
        keep = []
        for j in range(mat.shape[1]):
            if not keep or np.nanmax(np.abs(np.corrcoef(mat[:, keep + [j]], rowvar=False)[-1, :-1])) < .97:
                keep.append(j)
        names = [clean[j][0] for j in keep]; mat = mat[:, keep]
        mat = residualize_matrix(mat, cov)
        mat = np.column_stack([robust_z(mat[:, j]) for j in range(mat.shape[1])])
        mat = np.clip(mat, -4, 4)
        s1 = np.linalg.svd(mat, full_matrices=False, compute_uv=False)[0]
        mat = mat / max(s1, 1e-8)  # Multiple-factor-analysis block balancing.
        transformed.append(mat); selected.extend(names); block_labels.extend([block] * len(names))
        for j, name in zip(keep, names):
            feature_manifest.append({"block": block, "feature": name, "log1p_transform": clean[j][2],
                                     "missing_fraction": float(pd.to_numeric(ibd[name], errors="coerce").isna().mean()),
                                     "block_first_singular_value": s1})
    x = np.column_stack(transformed)
    pca = PCA(n_components=min(12, x.shape[1], x.shape[0] - 1), random_state=20260828).fit(x)
    pcs = pca.transform(x)

    results = []; solutions = {}
    for k in range(2, 7):
        labels, con, metrics = consensus_cluster(x, k)
        results.append(metrics); solutions[k] = (labels, con)
    model = pd.DataFrame(results)
    eligible = model[model.min_cluster_size >= max(8, int(.06 * len(ibd)))]
    best_k = int((eligible if len(eligible) else model).sort_values("score", ascending=False).iloc[0].k)
    labels, consensus = solutions[best_k]
    order = sorted(range(best_k), key=lambda c: np.mean(pcs[labels == c, 0]))
    remap = {old: i + 1 for i, old in enumerate(order)}
    labels = np.array([remap[x] for x in labels])
    ibd["adaptive_endotype"] = [f"E{x}" for x in labels]
    ibd["endotype_numeric"] = labels
    confidence = []
    for i in range(len(ibd)):
        same = labels == labels[i]; same[i] = False
        within = consensus[i, same].mean() if same.any() else 0
        between = max([consensus[i, labels == c].mean() for c in set(labels) if c != labels[i]], default=0)
        confidence.append(within - between)
    ibd["assignment_confidence"] = confidence
    for j in range(min(6, pcs.shape[1])): ibd[f"adaptive_factor_{j+1}"] = pcs[:, j]
    ibd.to_csv(OUT / "Table_RA5_adaptive_endotype_assignments.csv", index=False)
    model["selected"] = model.k.eq(best_k)
    model.to_csv(OUT / "Table_RA5_endotype_model_selection.csv", index=False)

    loadings = pd.DataFrame(pca.components_.T, index=selected,
                            columns=[f"adaptive_factor_{i+1}" for i in range(pca.n_components_)])
    loadings.insert(0, "block", block_labels); loadings.insert(1, "feature", loadings.index)
    loadings.reset_index(drop=True).to_csv(OUT / "Table_RA5_adaptive_factor_loadings.csv", index=False)
    pd.DataFrame(feature_manifest).to_csv(OUT / "Table_RA5_feature_manifest.csv", index=False)
    pd.DataFrame({"adaptive_factor": [f"adaptive_factor_{i+1}" for i in range(pca.n_components_)],
                  "variance_explained": pca.explained_variance_ratio_,
                  "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)}).to_csv(
        OUT / "Table_RA5_factor_variance.csv", index=False)

    # Endotype-feature associations on the original interpretable scale.
    diff = []
    for feat in selected:
        vals = [pd.to_numeric(ibd.loc[ibd.adaptive_endotype == e, feat], errors="coerce").dropna()
                for e in sorted(ibd.adaptive_endotype.unique())]
        if min(map(len, vals)) < 3: continue
        kw = stats.kruskal(*vals)
        med = {e: np.median(v) for e, v in zip(sorted(ibd.adaptive_endotype.unique()), vals)}
        diff.append({"feature": feat, "block": block_labels[selected.index(feat)], "kruskal_p": kw.pvalue,
                     **{f"median_{e}": m for e, m in med.items()}})
    diff = pd.DataFrame(diff); diff["FDR"] = fdr_bh(diff.kruskal_p)
    diff.to_csv(OUT / "Table_RA5_endotype_feature_differences.csv", index=False)

    assoc = []
    categorical = ["Diagnosis1", "Inflammation1", "Biologic", "Sex", "acquisition_series"]
    for var in categorical:
        tab = pd.crosstab(ibd.adaptive_endotype, ibd[var])
        if tab.shape[0] < 2 or tab.shape[1] < 2: continue
        chi = stats.chi2_contingency(tab, correction=False)
        assoc.append({"variable": var, "type": "categorical", "n": int(tab.to_numpy().sum()),
                      "effect": cramer_v(tab), "effect_name": "Cramer's V", "p_value": chi.pvalue})
    for var in ["Age", "log_calprotectin"]:
        vals = [pd.to_numeric(ibd.loc[ibd.adaptive_endotype == e, var], errors="coerce").dropna()
                for e in sorted(ibd.adaptive_endotype.unique())]
        kw = stats.kruskal(*vals)
        assoc.append({"variable": var, "type": "continuous", "n": sum(map(len, vals)),
                      "effect": kw.statistic / max(sum(map(len, vals)) - 1, 1),
                      "effect_name": "epsilon-squared proxy", "p_value": kw.pvalue})
    assoc = pd.DataFrame(assoc); assoc["FDR"] = fdr_bh(assoc.p_value)
    assoc.to_csv(OUT / "Table_RA5_endotype_clinical_associations.csv", index=False)

    # Figure: factor map, consensus model selection, biological heatmap, clinical audit.
    fig = plt.figure(figsize=(19, 14), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    palette = {e: c for e, c in zip(sorted(ibd.adaptive_endotype.unique()), sns.color_palette("Set2", best_k))}
    for e, g in ibd.groupby("adaptive_endotype"):
        ax.scatter(g.adaptive_factor_1, g.adaptive_factor_2, s=60, alpha=.8, label=e, color=palette[e],
                   marker="o")
    ax.set_xlabel(f"Adaptive factor 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Adaptive factor 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("A  IBD adaptive immune endotypes")
    ax.legend(title="Endotype")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(model.k, model.silhouette, "o-", label="Silhouette")
    ax.plot(model.k, 1 - model.PAC, "o-", label="1 - PAC")
    ax.plot(model.k, model.within_consensus, "o-", label="Within consensus")
    ax.axvline(best_k, color="black", ls="--", label=f"Selected k={best_k}")
    ax.set_xlabel("Number of endotypes")
    ax.set_ylabel("Stability metric")
    ax.set_title("B  Consensus-clustering stability")
    ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    topf = diff.sort_values("FDR").head(24).feature.tolist()
    hm = ibd.groupby("adaptive_endotype")[topf].median().T
    hm = hm.apply(lambda r: robust_z(r), axis=1, result_type="expand")
    hm.columns = sorted(ibd.adaptive_endotype.unique())
    sns.heatmap(hm, cmap="vlag", center=0, cbar_kws={"label": "Median robust z"}, ax=ax)
    ax.set_title("C  Features defining endotypes")
    ax.set_xlabel(""); ax.set_ylabel("")

    ax = fig.add_subplot(gs[1, 1])
    aa = assoc.sort_values("FDR", ascending=False)
    ax.barh(aa.variable, -np.log10(aa.FDR.clip(lower=1e-6)), color="#6b5b95")
    ax.axvline(-np.log10(.05), color="black", ls="--")
    ax.set_xlabel("-log10 FDR")
    ax.set_title("D  Clinical and acquisition-series audit")
    fig.savefig(OUT / "Figure_RA5_adaptive_immune_endotypes.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure_RA5_adaptive_immune_endotypes.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = pd.DataFrame([
        ["population", f"IBD only (CD and UC), n={len(ibd)}; controls excluded because acquisition series is nearly confounded"],
        ["technical adjustment", "features residualized for age, sex, log TCR depth, log BCR depth, and acquisition series"],
        ["multi-view integration", "robust scaling followed by multiple-factor-analysis block balancing"],
        ["clustering", "160-iteration participant/feature subsampled consensus clustering; k selected from 2-6"],
        ["selected k", best_k],
        ["interpretation", "cross-sectional adaptive immune endotypes; not prospective treatment-response classes"],
    ], columns=["item", "value"])
    manifest.to_csv(OUT / "Table_RA5_endotype_manifest.csv", index=False)
    return ibd, model, loadings, diff, assoc


if __name__ == "__main__":
    run_endotypes()
    print(f"Wrote ranked analysis 5 to {OUT}")

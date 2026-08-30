#!/usr/bin/env python3
"""Ranked analyses 1-2: clone-aware helper axis and external gut validation."""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "High Impact Additional Analyses" / "Literature Guided Ranked Analyses"
EXT = OUT / "external" / "GSE148837"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260828)

sns.set_theme(style="whitegrid", context="talk")


def fdr_bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    ans = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return ans
    po = p[ok]
    order = np.argsort(po)
    ranked = po[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    rev = np.empty_like(order)
    rev[order] = np.arange(len(order))
    ans[ok] = q[rev]
    return ans


def residualize(v, cov):
    v = stats.rankdata(np.asarray(v, float))
    x = pd.get_dummies(cov, drop_first=True, dtype=float)
    x.insert(0, "intercept", 1.0)
    xx = x.to_numpy(float)
    return v - xx @ np.linalg.lstsq(xx, v, rcond=None)[0]


def partial_spearman_block(d, xcol, ycol, covars, block="acquisition_series", nperm=3000):
    cols = [xcol, ycol, block] + covars
    z = d[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(z) < 12 or z[xcol].nunique() < 4 or z[ycol].nunique() < 4:
        return len(z), np.nan, np.nan, np.nan
    rx = residualize(z[xcol], z[covars])
    ry = residualize(z[ycol], z[covars])
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perm = np.empty(nperm)
    blocks = [np.flatnonzero(z[block].astype(str).to_numpy() == b)
              for b in z[block].astype(str).unique()]
    for i in range(nperm):
        yp = ry.copy()
        for ix in blocks:
            if len(ix) > 1:
                yp[ix] = yp[RNG.permutation(ix)]
        perm[i] = np.corrcoef(rx, yp)[0, 1]
    p = (1 + np.sum(np.abs(perm) >= abs(rho))) / (nperm + 1)
    # Fisher interval is descriptive; the series-block permutation supplies p.
    zr = np.arctanh(np.clip(rho, -0.999, 0.999))
    se = 1 / math.sqrt(max(len(z) - len(covars) - 3, 1))
    lo, hi = np.tanh([zr - 1.96 * se, zr + 1.96 * se])
    return len(z), rho, p, (float(lo), float(hi))


def helper_axis():
    cd4 = pd.read_csv(OUT / "Table_RA1_CD4_helper_by_participant.csv")
    b = pd.read_csv(OUT / "Table_RA1_Bcell_response_by_participant.csv")
    clinical = pd.read_csv(ROOT / "High Impact Additional Analyses" / "Priority Analyses" /
                           "Table_PA_clinical_metadata.csv")
    integrated = pd.read_csv(ROOT / "High Impact Additional Analyses" /
                             "Table_HI_integrated_participant_features.csv")
    keep_int = ["SampleID", "tcr_paired_cells", "bcr_paired_cells", "bcr_SHM_rate",
                "bcr_switched_fraction", "bcr_IgA_fraction", "bcr_IgG_fraction"]
    d = cd4.merge(b, on=["SampleID", "Diagnosis1", "Inflammation1", "Biologic", "Batch",
                              "acquisition_series"], how="inner")
    d = d.merge(clinical.drop(columns=["Diagnosis1", "Batch", "Inflammation1", "Biologic"]),
                on="SampleID", how="left")
    d = d.merge(integrated[keep_int], on="SampleID", how="left")
    d["sex_male"] = (d["Sex"].astype(str).str.upper() == "M").astype(float)
    d["log_cd4_cells"] = np.log1p(d["cd4_cells"])
    d["log_b_cells"] = np.log1p(d["b_cells"])
    d["log_calprotectin"] = np.log1p(pd.to_numeric(d["Calprotectin"], errors="coerce"))
    d.to_csv(OUT / "Table_RA1_helper_axis_analysis_dataset.csv", index=False)

    pairs = [
        ("cd4_mean_Tph_core", "b_mean_Plasmablast_plasma_differentiation", "Tph", "Plasma differentiation"),
        ("cd4_mean_Tph_core", "b_mean_IgA_mucosal_plasma_cell", "Tph", "IgA mucosal plasma"),
        ("cd4_mean_Tph_core", "b_mean_Atypical_memory_CD11c_like", "Tph", "Atypical memory"),
        ("cd4_mean_Tph_core", "b_mean_B_cell_antigen_presentation", "Tph", "B-cell antigen presentation"),
        ("cd4_clone_delta_Tph_core", "b_mean_Plasmablast_plasma_differentiation", "Clone-associated Tph", "Plasma differentiation"),
        ("cd4_clone_delta_Tph_core", "b_mean_IgA_mucosal_plasma_cell", "Clone-associated Tph", "IgA mucosal plasma"),
        ("cd4_mean_Tfh_core", "b_mean_Plasmablast_plasma_differentiation", "Tfh", "Plasma differentiation"),
        ("cd4_mean_Tfh_core", "b_mean_IgA_mucosal_plasma_cell", "Tfh", "IgA mucosal plasma"),
        ("cd4_tfh_fraction", "b_switched_memory_fraction", "Tfh fraction", "Switched memory fraction"),
        ("cd4_tfh_fraction", "bcr_SHM_rate", "Tfh fraction", "BCR SHM"),
        ("cd4_clone_delta_Tfh_core", "b_mean_Plasmablast_plasma_differentiation", "Clone-associated Tfh", "Plasma differentiation"),
        ("expanded_tfh_fraction_of_paired", "b_mean_IgA_mucosal_plasma_cell", "Expanded Tfh fraction", "IgA mucosal plasma"),
    ]
    cov = ["Age", "sex_male", "log_cd4_cells", "log_b_cells"]
    rows = []
    for dx in ["CD", "UC", "Control"]:
        dd = d[d.Diagnosis1 == dx]
        for x, y, xl, yl in pairs:
            n, rho, p, ci = partial_spearman_block(dd, x, y, cov)
            rows.append({"Diagnosis": dx, "T_feature": x, "B_feature": y,
                         "T_label": xl, "B_label": yl, "n": n, "partial_rho": rho,
                         "ci_low": ci[0] if isinstance(ci, tuple) else np.nan,
                         "ci_high": ci[1] if isinstance(ci, tuple) else np.nan,
                         "series_block_permutation_p": p})
    cor = pd.DataFrame(rows)
    cor["FDR_within_diagnosis"] = cor.groupby("Diagnosis")["series_block_permutation_p"].transform(fdr_bh)
    cor.to_csv(OUT / "Table_RA1_helper_B_partial_correlations.csv", index=False)

    # CD-versus-UC interaction analysis is restricted to the shared S1-S5 acquisition series.
    ibd = d[(d.Diagnosis1.isin(["CD", "UC"])) &
            (d.acquisition_series.isin(["S1", "S2", "S3", "S4", "S5"]))].copy()
    interaction_rows = []
    import statsmodels.formula.api as smf
    for x, y, xl, yl in pairs:
        z = ibd[[x, y, "Diagnosis1", "Age", "sex_male", "log_cd4_cells", "log_b_cells",
                 "acquisition_series"]].dropna().copy()
        if len(z) < 30:
            continue
        z["x_z"] = stats.zscore(z[x])
        z["y_z"] = stats.zscore(z[y])
        fit = smf.ols("y_z ~ x_z * C(Diagnosis1, Treatment(reference='UC')) + Age + sex_male + "
                      "log_cd4_cells + log_b_cells + C(acquisition_series)", data=z).fit(cov_type="HC3")
        term = "x_z:C(Diagnosis1, Treatment(reference='UC'))[T.CD]"
        interaction_rows.append({"T_feature": x, "B_feature": y, "T_label": xl, "B_label": yl,
                                 "n": len(z), "CD_minus_UC_interaction_beta": fit.params.get(term, np.nan),
                                 "robust_SE": fit.bse.get(term, np.nan),
                                 "p_value": fit.pvalues.get(term, np.nan)})
    inter = pd.DataFrame(interaction_rows)
    inter["FDR"] = fdr_bh(inter.p_value)
    inter.to_csv(OUT / "Table_RA1_helper_B_CD_UC_interactions.csv", index=False)

    # Prespecified continuous clinical associations within IBD.
    clinical_features = [
        "cd4_mean_Tph_core", "cd4_clone_delta_Tph_core", "cd4_mean_Tfh_core",
        "cd4_clone_delta_Tfh_core", "cd4_tfh_fraction",
        "b_mean_Plasmablast_plasma_differentiation", "b_mean_IgA_mucosal_plasma_cell",
    ]
    crows = []
    cov2 = ["Age", "sex_male", "log_cd4_cells", "log_b_cells", "Diagnosis1", "Biologic"]
    for feat in clinical_features:
        n, rho, p, ci = partial_spearman_block(d[d.Diagnosis1.isin(["CD", "UC"])], feat,
                                                "log_calprotectin", cov2)
        crows.append({"feature": feat, "outcome": "log1p_calprotectin", "n": n,
                      "partial_rho": rho, "ci_low": ci[0] if isinstance(ci, tuple) else np.nan,
                      "ci_high": ci[1] if isinstance(ci, tuple) else np.nan,
                      "series_block_permutation_p": p})
    clinical_out = pd.DataFrame(crows)
    clinical_out["FDR"] = fdr_bh(clinical_out.series_block_permutation_p)
    clinical_out.to_csv(OUT / "Table_RA1_helper_axis_clinical_associations.csv", index=False)

    # Figure: diagnosis-specific targeted correlations plus top relationship scatter.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    h = cor.pivot_table(index=["T_label", "B_label"], columns="Diagnosis", values="partial_rho")
    h = h.reindex(columns=["CD", "UC", "Control"])
    sns.heatmap(h, cmap="vlag", center=0, vmin=-0.7, vmax=0.7, annot=True, fmt=".2f",
                cbar_kws={"label": "Partial Spearman rho"}, ax=axes[0])
    axes[0].set_title("A  Clone-aware helper–B coupling")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("T feature / B feature")

    top = cor[(cor.Diagnosis != "Control") & cor.partial_rho.notna()].sort_values(
        ["FDR_within_diagnosis", "series_block_permutation_p"]).iloc[0]
    td = d[d.Diagnosis1 == top.Diagnosis].dropna(subset=[top.T_feature, top.B_feature])
    sns.regplot(data=td, x=top.T_feature, y=top.B_feature, scatter_kws={"s": 45, "alpha": .75},
                line_kws={"color": "black"}, ax=axes[1])
    axes[1].set_title(f"B  Strongest prespecified link ({top.Diagnosis})\n"
                      f"rho={top.partial_rho:.2f}, FDR={top.FDR_within_diagnosis:.3g}")
    axes[1].set_xlabel(top.T_label)
    axes[1].set_ylabel(top.B_label)

    plot_i = inter.sort_values("FDR").head(10).sort_values("CD_minus_UC_interaction_beta")
    yy = np.arange(len(plot_i))
    axes[2].errorbar(plot_i.CD_minus_UC_interaction_beta, yy,
                     xerr=1.96 * plot_i.robust_SE, fmt="o", color="#7a3e9d", capsize=3)
    axes[2].axvline(0, color="black", lw=1)
    axes[2].set_yticks(yy, [f"{a} → {b}" for a, b in zip(plot_i.T_label, plot_i.B_label)], fontsize=9)
    axes[2].set_xlabel("CD–UC interaction beta (95% CI)")
    axes[2].set_title("C  Shared-series interaction audit")
    fig.savefig(OUT / "Figure_RA1_clone_aware_helper_B_axis.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure_RA1_clone_aware_helper_B_axis.pdf", bbox_inches="tight")
    plt.close(fig)
    return d, cor, inter


EXTERNAL_MODULES = {
    "Effector_cytotoxicity": ["NKG7", "GNLY", "PRF1", "GZMB", "GZMA", "GZMH", "GZMK", "GZMM", "CTSW", "CST7", "FGFBP2", "CCL5", "CCL4", "CCL3", "IFNG", "FASLG", "KLRD1", "KLRG1"],
    "Th1_Tc1_inflammatory": ["TBX21", "STAT4", "CXCR3", "IFNG", "TNF", "IL12RB2", "CCL5", "CCL4", "NKG7", "GZMB", "PRF1", "CXCR6", "BHLHE40"],
    "EOMES_ZEB2_inflammatory_CD8_TRM_like": ["EOMES", "ZEB2", "GZMB", "GZMH", "PRF1", "NKG7", "CX3CR1", "KLRG1", "TBX21", "CCL5", "CST7", "FGFBP2"],
    "Tissue_resident_mucosal_retention": ["CD69", "ITGAE", "ITGA1", "CXCR6", "ZNF683", "RUNX3", "PRDM1", "RGS1", "CD101", "DUSP6", "AHR", "CCR6"],
    "Gut_homing_intestinal_trafficking": ["ITGA4", "ITGB7", "ITGAE", "CCR9", "CCR6", "CXCR3", "CXCR6", "SELPLG", "S1PR1", "KLF2", "SELL", "GPR183", "CD69"],
}


def read_tsv_gz(path):
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n\r").split("\t") for line in handle]


def external_gut_validation():
    sample_meta = pd.read_csv(EXT / "GSE148837_citeseq_sample_meta_data.csv.gz")
    sample_meta.columns = [c.strip() for c in sample_meta.columns]
    sample_meta["hash_no"] = sample_meta.Hashtag.str.extract(r"#(\d+)").astype(int)
    audit, pool_tables = [], []
    for pool_no in range(1, 5):
        pool = f"Pool{pool_no}"
        p = EXT / pool
        features = pd.DataFrame(read_tsv_gz(p / "features.tsv.gz"), columns=["id", "name", "feature_type"])
        mat = mmread(str(p / "matrix.mtx.gz")).tocsr()
        gene_ix = np.flatnonzero(features.feature_type.to_numpy() == "Gene Expression")
        gene_names = features.name.to_numpy()[gene_ix]
        gene_mat = mat[gene_ix, :].tocsc()
        libs = np.asarray(gene_mat.sum(axis=0)).ravel(); detected = np.diff(gene_mat.indptr)
        kept = np.flatnonzero((libs >= 200) & (detected >= 100))
        hto_lookup = {f"HTO{i}": i for i in range(1, 6)}
        hto_ix = [i for i, n in enumerate(features.name.astype(str)) if n in hto_lookup]
        hto_names = features.name.iloc[hto_ix].tolist(); hto = mat[hto_ix, :][:, kept].toarray()
        order = np.argsort(hto, axis=0); top_ix = order[-1, :]
        top = hto[top_ix, np.arange(hto.shape[1])]; second = hto[order[-2, :], np.arange(hto.shape[1])]
        confident = (top >= 5) & ((top + 1) / (second + 1) >= 2)
        hash_num = np.array([hto_lookup[hto_names[i]] for i in top_ix])
        eligible = sample_meta[sample_meta["Pool No."].astype(str).str.contains(f"Pool {pool_no}")]
        hmap = dict(zip(eligible.hash_no, eligible["Sample ID"]))
        sample_id = np.array([hmap.get(int(h), None) for h in hash_num], dtype=object)
        assigned = confident & pd.notna(sample_id); cells = kept[assigned]
        audit.append({"pool": pool, "matrix_barcodes": mat.shape[1], "rna_qc_cells": len(kept),
                      "confident_sample_assignments": int(assigned.sum()),
                      "assignment_fraction_of_rna_qc": assigned.mean() if len(assigned) else np.nan})
        sdf = pd.DataFrame({"SampleID_external": sample_id[assigned]})
        for module, genes in EXTERNAL_MODULES.items():
            local_rows = [np.flatnonzero(gene_names == g)[0] for g in genes if np.any(gene_names == g)]
            sub = gene_mat[local_rows, :][:, cells].toarray().astype(float)
            sdf[module] = np.log1p(sub / libs[cells][None, :] * 1e4).mean(axis=0)
        a = sdf.groupby("SampleID_external").agg({**{m: "mean" for m in EXTERNAL_MODULES},
                                                   "SampleID_external": "size"})
        a = a.rename(columns={"SampleID_external": "n_cells"}).reset_index()
        a["pool"] = pool
        pool_tables.append(a)
    pool_df = pd.concat(pool_tables, ignore_index=True)
    pool_df = pool_df.merge(sample_meta[["Sample ID", "Diagnosis", "UCEIS Score", "Age", "Gender"]],
                            left_on="SampleID_external", right_on="Sample ID", how="left")
    # S12 and S13 are inflamed/non-inflamed biopsies from the same individual
    # (identical demographics, comorbidities, and medication metadata).
    pool_df["DonorID_external"] = pool_df.SampleID_external.replace({"S13": "S12"})
    pool_df.to_csv(OUT / "Table_RA2_GSE148837_pool_level_module_scores.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "Table_RA2_GSE148837_assignment_audit.csv", index=False)

    donor = pool_df.groupby(["DonorID_external", "Diagnosis"], as_index=False).agg(
        {**{m: "mean" for m in EXTERNAL_MODULES}, "n_cells": "sum"})
    donor = donor.rename(columns={"DonorID_external": "SampleID_external"})
    donor.to_csv(OUT / "Table_RA2_GSE148837_donor_module_scores.csv", index=False)
    test_rows = []
    for module in EXTERNAL_MODULES:
        uc = donor.loc[donor.Diagnosis == "Ulcerative Colitis", module].dropna()
        hc = donor.loc[donor.Diagnosis == "Healthy", module].dropna()
        if len(uc) and len(hc):
            u = stats.mannwhitneyu(uc, hc, alternative="two-sided")
            delta = (sum(a > b for a in uc for b in hc) - sum(a < b for a in uc for b in hc)) / (len(uc) * len(hc))
            test_rows.append({"module": module, "n_UC": len(uc), "n_healthy": len(hc),
                              "median_UC": np.median(uc), "median_healthy": np.median(hc),
                              "median_difference": np.median(uc) - np.median(hc),
                              "cliffs_delta": delta, "mann_whitney_p": u.pvalue})
    tests = pd.DataFrame(test_rows)
    tests["FDR"] = fdr_bh(tests.mann_whitney_p)
    tests.to_csv(OUT / "Table_RA2_GSE148837_UC_vs_healthy_tests.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    long = donor[donor.Diagnosis.isin(["Ulcerative Colitis", "Healthy"])].melt(
        id_vars=["SampleID_external", "Diagnosis"], value_vars=list(EXTERNAL_MODULES),
        var_name="module", value_name="score")
    sns.stripplot(data=long, x="module", y="score", hue="Diagnosis", dodge=True, size=7,
                  palette={"Ulcerative Colitis": "#c43c39", "Healthy": "#3b78b4"}, ax=axes[0])
    axes[0].tick_params(axis="x", rotation=35, labelsize=9)
    axes[0].set_title("A  Independent colonic CD8+ cohort (GSE148837)")
    axes[0].set_ylabel("Donor mean log-normalized module score")
    axes[0].set_xlabel("")
    axes[0].legend(title="")
    y = np.arange(len(tests))
    axes[1].barh(y, tests.cliffs_delta, color=np.where(tests.cliffs_delta >= 0, "#c43c39", "#3b78b4"))
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_yticks(y, tests.module.str.replace("_", " "), fontsize=10)
    axes[1].set_xlabel("Cliff's delta (UC > healthy)")
    axes[1].set_title("B  External effect sizes")
    for i, row in tests.iterrows():
        axes[1].text(row.cliffs_delta, i, f"  FDR={row.FDR:.3g}", va="center", fontsize=9)
    fig.savefig(OUT / "Figure_RA2_external_gut_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure_RA2_external_gut_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    return donor, tests


if __name__ == "__main__":
    helper_axis()
    external_gut_validation()
    print(f"Wrote ranked analyses 1-2 to {OUT}")

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import pearsonr, rankdata
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

SEED = 20260824
rng = np.random.default_rng(SEED)

ROOT = Path(r"C:/path/to/private-manuscript-workspace")
HI = ROOT / "High Impact Additional Analyses"
OUT = HI / "Priority Analyses"
OUT.mkdir(parents=True, exist_ok=True)


def stratified_bootstrap(y, scores, n_boot=5000):
    y = np.asarray(y, dtype=int)
    scores = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    idx_by_class = [np.flatnonzero(y == c) for c in np.unique(y)]
    vals = {k: [] for k in scores}
    for _ in range(n_boot):
        ii = np.concatenate([rng.choice(x, len(x), replace=True) for x in idx_by_class])
        for k, v in scores.items():
            vals[k].append(roc_auc_score(y[ii], v[ii]))
    return {k: np.asarray(v) for k, v in vals.items()}


def calibration_metrics(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    x = logit(p).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs").fit(x, y)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "calibration_intercept": float(cal.intercept_[0]),
        "calibration_slope": float(cal.coef_[0, 0]),
        "brier_score": float(brier_score_loss(y, p)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
    }


def run_late_fusion():
    pred = pd.read_csv(HI / "Table_HI_paired_chain_nested_outer_fold_predictions.csv")
    pred = pred[pred.representation.eq("dual_additive")].copy()
    a = pred[pred.modality.eq("TCR")].rename(columns={"probability": "p_tcr"})
    b = pred[pred.modality.eq("BCR")].rename(columns={"probability": "p_bcr"})
    keys = ["task", "outer_fold", "SampleID", "truth"]
    m = a[keys + ["p_tcr"]].merge(b[keys + ["p_bcr"]], on=keys, validate="one_to_one")
    m["p_fusion"] = 0.5 * (m.p_tcr + m.p_bcr)
    m.to_csv(OUT / "Table_PA4_joint_TCR_BCR_outer_fold_predictions.csv", index=False)

    rows, delta_rows, perm_rows = [], [], []
    pooled_all = []
    for task, d in m.groupby("task", sort=False):
        pooled = d.groupby(["SampleID", "truth"], as_index=False)[["p_tcr", "p_bcr", "p_fusion"]].mean()
        pooled["task"] = task
        pooled_all.append(pooled)
        y = pooled.truth.to_numpy(int)
        scores = {k: pooled[k].to_numpy(float) for k in ["p_tcr", "p_bcr", "p_fusion"]}
        boots = stratified_bootstrap(y, scores)
        observed = {k: roc_auc_score(y, v) for k, v in scores.items()}
        for k, auc in observed.items():
            lo, hi = np.quantile(boots[k], [0.025, 0.975])
            metric = calibration_metrics(y, scores[k])
            rows.append({"task": task, "model": k, "pooled_auc": auc, "ci_low": lo, "ci_high": hi,
                         "n": len(y), **metric})
        for ref in ["p_tcr", "p_bcr"]:
            delta = boots["p_fusion"] - boots[ref]
            delta_rows.append({
                "task": task,
                "comparison": f"fusion minus {ref[2:].upper()}",
                "delta_auc": observed["p_fusion"] - observed[ref],
                "ci_low": np.quantile(delta, 0.025),
                "ci_high": np.quantile(delta, 0.975),
                "bootstrap_p_two_sided": 2 * min(np.mean(delta <= 0), np.mean(delta >= 0)),
            })
        null = np.array([roc_auc_score(rng.permutation(y), scores["p_fusion"]) for _ in range(5000)])
        perm_rows.append({"task": task, "observed_auc": observed["p_fusion"], "null_mean_auc": null.mean(),
                          "null_95_low": np.quantile(null, .025), "null_95_high": np.quantile(null, .975),
                          "permutation_p": (1 + np.sum(null >= observed["p_fusion"])) / (1 + len(null))})
    pd.DataFrame(rows).to_csv(OUT / "Table_PA4_joint_TCR_BCR_nested_validation_summary.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(OUT / "Table_PA4_joint_TCR_BCR_incremental_value.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(OUT / "Table_PA4_joint_TCR_BCR_permutation_null.csv", index=False)
    pd.concat(pooled_all).to_csv(OUT / "Table_PA4_joint_TCR_BCR_pooled_predictions.csv", index=False)

    summary = pd.DataFrame(rows)
    deltas = pd.DataFrame(delta_rows)
    tasks = list(summary.task.drop_duplicates())
    colors = {"p_tcr": "#2B6CB0", "p_bcr": "#C2415D", "p_fusion": "#6B46C1"}
    labels = {"p_tcr": "TCR", "p_bcr": "BCR", "p_fusion": "Joint TCR+BCR"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    x0 = np.arange(len(tasks))
    for j, model in enumerate(["p_tcr", "p_bcr", "p_fusion"]):
        q = summary[summary.model.eq(model)].set_index("task").loc[tasks]
        x = x0 + (j - 1) * .22
        axes[0].errorbar(x, q.pooled_auc, yerr=[q.pooled_auc - q.ci_low, q.ci_high - q.pooled_auc],
                         fmt="o", color=colors[model], capsize=3, label=labels[model])
    axes[0].axhline(.5, color="0.7", lw=.8)
    axes[0].set_xticks(x0, tasks, rotation=20, ha="right")
    axes[0].set_ylim(0.45, 1.02)
    axes[0].set_ylabel("Pooled out-of-fold ROC AUC")
    axes[0].set_title("A  Nested participant-level classification", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    for j, ref in enumerate(["TCR", "BCR"]):
        q = deltas[deltas.comparison.eq(f"fusion minus {ref}")].set_index("task").loc[tasks]
        x = x0 + (j - .5) * .22
        axes[1].errorbar(x, q.delta_auc, yerr=[q.delta_auc - q.ci_low, q.ci_high - q.delta_auc],
                         fmt="o", capsize=3, label=f"Joint minus {ref}", color=["#2B6CB0", "#C2415D"][j])
    axes[1].axhline(0, color="0.55", lw=.8)
    axes[1].set_xticks(x0, tasks, rotation=20, ha="right")
    axes[1].set_ylabel("Incremental ROC AUC")
    axes[1].set_title("B  Incremental value of joint modeling", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "Figure_PA3_joint_TCR_BCR_late_fusion.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "Figure_PA3_joint_TCR_BCR_late_fusion.pdf", bbox_inches="tight")
    plt.close(fig)


def residualized_spearman(df, x, y, covariates):
    cols = [x, y] + covariates
    d = df[cols].copy().dropna()
    if len(d) < 15 or d[x].nunique() < 4 or d[y].nunique() < 4:
        return np.nan, np.nan, len(d)
    Xparts = [np.ones((len(d), 1))]
    for c in covariates:
        if pd.api.types.is_numeric_dtype(d[c]):
            Xparts.append(rankdata(d[c]).reshape(-1, 1))
        else:
            Xparts.append(pd.get_dummies(d[c].astype(str), drop_first=True, dtype=float).to_numpy())
    X = np.column_stack(Xparts)
    rx, ry = rankdata(d[x]), rankdata(d[y])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    rho, p = pearsonr(ex, ey)
    return rho, p, len(d)


def build_antigen_convergence():
    gliph_root = Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 2 TCR\GLIPH2\gliph2_pairwise_tcronly")
    members = pd.read_csv(Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 2 TCR\TCR Architecture Analyses\outputs\gliph2_fdr05_annotationL2_20260601_081428\significant_gliph2_clusters_fdr05_members.csv"))
    members["SampleID"] = members.patient.astype(str).str.split(":").str[0]
    members["Diagnosis1"] = members.patient.astype(str).str.split(":").str[-1]
    members["cluster_key"] = members.comparison.astype(str) + "|||" + members.tag.astype(str)
    members = members.drop_duplicates(["comparison", "tag", "SampleID", "CDR3b"])

    annotations = []
    denominators = []
    for folder in ["CD_vs_control", "UC_vs_control", "CD_vs_UC"]:
        f = gliph_root / folder / "with_iedb_cluster_annotation_summary.csv"
        if f.exists():
            annotations.append(pd.read_csv(f))
        f2 = gliph_root / folder / "input_sequences.csv"
        if f2.exists():
            z = pd.read_csv(f2)
            z["comparison"] = folder
            z["SampleID"] = z.patient.astype(str).str.split(":").str[0]
            denominators.append(z.drop_duplicates(["comparison", "SampleID", "CDR3b"]))
    annotations = pd.concat(annotations, ignore_index=True) if annotations else pd.DataFrame()
    denom = pd.concat(denominators, ignore_index=True)
    denom_n = denom.groupby(["comparison", "SampleID"]).CDR3b.nunique().rename("n_input_unique_cdr3").reset_index()

    burden = members.groupby(["comparison", "SampleID", "Diagnosis1"]).agg(
        n_significant_cluster_cdr3=("CDR3b", "nunique"),
        n_significant_clusters=("tag", "nunique"),
    ).reset_index().merge(denom_n, on=["comparison", "SampleID"], how="left")
    burden["significant_cluster_cdr3_fraction"] = burden.n_significant_cluster_cdr3 / burden.n_input_unique_cdr3

    # Add zero-burden participants represented in each GLIPH comparison.
    base = denom[["comparison", "SampleID", "group"]].drop_duplicates().rename(columns={"group": "Diagnosis1"})
    burden = base.merge(burden, on=["comparison", "SampleID", "Diagnosis1"], how="left")
    for c in ["n_significant_cluster_cdr3", "n_significant_clusters", "significant_cluster_cdr3_fraction"]:
        burden[c] = burden[c].fillna(0)
    burden = burden.merge(denom_n, on=["comparison", "SampleID"], how="left", suffixes=("", "_base"))
    burden["n_input_unique_cdr3"] = burden["n_input_unique_cdr3"].fillna(burden.pop("n_input_unique_cdr3_base"))

    # Map beta-chain sequences to participant-level dominant transcriptional states.
    states = pd.read_csv(HI / "Table_HI_TCR_clonotype_state_breadth.csv")
    states["CDR3b"] = states.clonotype_id.astype(str).str.split("|").str[-1]
    sm = members.merge(states[["SampleID", "CDR3b", "dominant_state", "total_cells"]], on=["SampleID", "CDR3b"], how="left")
    state_summary = sm.dropna(subset=["dominant_state"]).groupby(["comparison", "tag", "dominant_state"]).total_cells.sum().rename("mapped_cells").reset_index()
    top_state = state_summary.sort_values("mapped_cells", ascending=False).drop_duplicates(["comparison", "tag"])

    cluster = members.groupby(["comparison", "tag", "cluster_index", "type"]).agg(
        n_unique_sequences=("CDR3b", "nunique"), n_participants=("SampleID", "nunique"),
        fisher_or=("fisher_or", "first"), fdr=("fdr", "first"), disease_fraction=("prop_g1", "first")
    ).reset_index()
    if len(annotations):
        annotations = annotations.rename(columns={"cluster_tag": "tag"})
        cluster = cluster.merge(annotations, on=["comparison", "tag"], how="left")
    cluster = cluster.merge(top_state[["comparison", "tag", "dominant_state", "mapped_cells"]], on=["comparison", "tag"], how="left")

    members.to_csv(OUT / "Table_PA5_GLIPH2_significant_cluster_members_unique.csv", index=False)
    cluster.to_csv(OUT / "Table_PA5_GLIPH2_clusters_antigen_state_annotation.csv", index=False)
    burden.to_csv(OUT / "Table_PA5_GLIPH2_convergent_burden_by_participant.csv", index=False)
    return burden, cluster


def run_continuous_inflammation(burden):
    clinical = pd.read_csv(OUT / "Table_PA_clinical_metadata.csv")
    integ = pd.read_csv(HI / "Table_HI_integrated_participant_features.csv")
    exact = pd.read_csv(HI / "Table_HI_exact_clonotype_definition_metrics_by_participant.csv")
    exact = exact[(exact.threshold == 2) & exact.analysis.eq("Original depth") &
                  exact.receptor_definition.isin(["TCR paired alpha-beta", "BCR paired heavy-light"])]
    ex = exact.pivot(index="SampleID", columns="receptor_definition",
                     values=["clonality", "expanded_cell_fraction", "total_cells"])
    ex.columns = [f"{a}_{'tcr_pair' if 'TCR' in b else 'bcr_pair'}" for a, b in ex.columns]
    ex = ex.reset_index()
    lineage = pd.read_csv(OUT / "Table_PA3_BCR_lineage_diversification_by_participant.csv")
    d = clinical.merge(integ, on=["SampleID", "Diagnosis1"], how="left", suffixes=("", "_integ"))
    d = d.merge(ex, on="SampleID", how="left").merge(
        lineage[["SampleID", "fraction_diversified_lineages", "median_v_region_nt_divergence", "median_cdr_minus_fwr_divergence"]],
        on="SampleID", how="left")
    d = d[d.Diagnosis1.isin(["CD", "UC"]) & d.Calprotectin.notna() & (d.Calprotectin >= 0)].copy()
    d["log1p_calprotectin"] = np.log1p(d.Calprotectin)
    d["log_tcr_depth"] = np.log1p(d.total_cells_tcr_pair)
    d["log_bcr_depth"] = np.log1p(d.total_cells_bcr_pair)
    metrics = [
        "clonality_tcr_pair", "expanded_cell_fraction_tcr_pair",
        "clonality_bcr_pair", "expanded_cell_fraction_bcr_pair",
        "tcr_cytotoxic_expanded", "tcr_gut_homing_expanded",
        "bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module",
        "bcr_IgG_inflammatory_module", "bcr_BAFF_APRIL_module",
        "fraction_diversified_lineages", "median_v_region_nt_divergence",
        "median_cdr_minus_fwr_divergence",
    ]
    rows = []
    for metric in metrics:
        depth = "log_tcr_depth" if ("tcr" in metric.lower()) else "log_bcr_depth"
        rho, p, n = residualized_spearman(d, metric, "log1p_calprotectin", ["Diagnosis1", "Age", "Sex", depth])
        rows.append({"analysis": "all_IBD", "metric": metric, "partial_spearman_rho": rho, "p_value": p, "n": n})
        for dx in ["CD", "UC"]:
            q = d[d.Diagnosis1.eq(dx)]
            rho, p, n = residualized_spearman(q, metric, "log1p_calprotectin", ["Age", "Sex", depth])
            rows.append({"analysis": dx, "metric": metric, "partial_spearman_rho": rho, "p_value": p, "n": n})

    # Add convergent-cluster burden, evaluated within its defining pairwise comparison.
    b = burden.merge(clinical, on=["SampleID", "Diagnosis1"], how="left")
    b = b[b.Diagnosis1.isin(["CD", "UC"]) & b.Calprotectin.notna()].copy()
    b["log1p_calprotectin"] = np.log1p(b.Calprotectin)
    b["log_depth"] = np.log1p(b.n_input_unique_cdr3)
    for comp, q in b.groupby("comparison"):
        rho, p, n = residualized_spearman(q, "significant_cluster_cdr3_fraction", "log1p_calprotectin", ["Diagnosis1", "Age", "Sex", "log_depth"])
        rows.append({"analysis": f"GLIPH2_{comp}", "metric": "significant_cluster_cdr3_fraction",
                     "partial_spearman_rho": rho, "p_value": p, "n": n})
    res = pd.DataFrame(rows)
    res["FDR_global"] = res.p_value.rank(method="first")  # placeholder overwritten below
    mask = res.p_value.notna()
    p = res.loc[mask, "p_value"].to_numpy()
    order = np.argsort(p)
    adj = np.empty(len(p))
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj[order] = np.minimum(ranked, 1)
    res.loc[mask, "FDR_global"] = adj
    res.to_csv(OUT / "Table_PA6_continuous_calprotectin_associations.csv", index=False)
    d.to_csv(OUT / "Table_PA6_continuous_inflammation_analysis_dataset.csv", index=False)

    # Compact forest plot of the all-IBD estimates.
    q = res[res.analysis.eq("all_IBD")].sort_values("partial_spearman_rho")
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    y = np.arange(len(q))
    ax.axvline(0, color="0.65", lw=.8)
    colors = np.where(q.FDR_global < .05, "#6B46C1", "#777777")
    ax.scatter(q.partial_spearman_rho, y, c=colors, s=28)
    ax.set_yticks(y, [x.replace("_", " ") for x in q.metric])
    ax.set_xlabel("Partial Spearman correlation with log1p calprotectin")
    ax.set_title("Continuous inflammation associations", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "Figure_PA4_continuous_inflammation_associations.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "Figure_PA4_continuous_inflammation_associations.pdf", bbox_inches="tight")
    plt.close(fig)

    # Coupling between convergent TCR burden and B-cell programs.
    merged = burden.merge(integ, on=["SampleID", "Diagnosis1"], how="left")
    merged["log_tcr_depth"] = np.log1p(merged.n_input_unique_cdr3)
    bmetrics = ["bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module",
                "bcr_IgG_inflammatory_module", "bcr_BAFF_APRIL_module"]
    cr = []
    for comp, q in merged.groupby("comparison"):
        for metric in bmetrics:
            rho, p, n = residualized_spearman(q, "significant_cluster_cdr3_fraction", metric,
                                              ["Diagnosis1", "Age", "Sex", "log_tcr_depth", "bcr_total_cells"])
            cr.append({"comparison": comp, "bcr_metric": metric, "partial_spearman_rho": rho, "p_value": p, "n": n})
    cr = pd.DataFrame(cr)
    mask = cr.p_value.notna(); p = cr.loc[mask, "p_value"].to_numpy(); order = np.argsort(p)
    adj = np.empty(len(p)); ranked = p[order] * len(p) / np.arange(1, len(p)+1); ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj[order] = np.minimum(ranked, 1); cr.loc[mask, "FDR_global"] = adj
    cr.to_csv(OUT / "Table_PA5_GLIPH2_TCR_BCR_program_coupling.csv", index=False)
    return res, cr


def write_summary():
    de = pd.read_csv(OUT / "Table_PA1_clone_aware_pseudobulk_DE_all_genes.csv")
    prox = pd.read_csv(OUT / "Table_PA2_receptor_transcriptome_proximity_vs_null.csv")
    lin = pd.read_csv(OUT / "Table_PA3_BCR_lineage_group_tests.csv")
    fusion = pd.read_csv(OUT / "Table_PA4_joint_TCR_BCR_nested_validation_summary.csv")
    delta = pd.read_csv(OUT / "Table_PA4_joint_TCR_BCR_incremental_value.csv")
    cal = pd.read_csv(OUT / "Table_PA6_continuous_calprotectin_associations.csv")
    coupling = pd.read_csv(OUT / "Table_PA5_GLIPH2_TCR_BCR_program_coupling.csv")
    lines = ["# Priority paired-repertoire analyses: results summary", "", "## Clone-aware pseudobulk expression", ""]
    for mod, q in de.groupby("modality"):
        sig = q[(q.coefficient == "statusExpanded") & (q.FDR_global < .05)].sort_values("FDR_global")
        top = ", ".join(sig.head(8).gene.astype(str)) if len(sig) else "none"
        lines.append(f"- {mod}: {len(sig)} expansion-associated genes at FDR < 0.05; leading genes: {top}.")
    lines += ["", "## Receptor-transcriptome proximity", "",
              "- Exact paired clone-mates did not show reproducibly greater latent-space proximity than participant- and cell-state-matched null sets after FDR correction. This is a negative sensitivity result.",
              "", "## BCR lineage diversification", ""]
    siglin = lin[lin.FDR_global < .05]
    lines.append(f"- {len(siglin)} of {len(lin)} diagnosis contrasts passed global FDR correction using unique-sequence lineage metrics.")
    lines += ["", "## Joint TCR-BCR late fusion", ""]
    for _, r in fusion[fusion.model.eq("p_fusion")].iterrows():
        dt = delta[(delta.task == r.task) & delta.comparison.eq("fusion minus TCR")].iloc[0]
        lines.append(f"- {r.task}: joint AUC {r.pooled_auc:.3f} (95% CI {r.ci_low:.3f}-{r.ci_high:.3f}); delta versus TCR {dt.delta_auc:.3f} (95% CI {dt.ci_low:.3f}-{dt.ci_high:.3f}).")
    lines += ["", "## Continuous inflammation and convergent-receptor coupling", ""]
    sigcal = cal[cal.FDR_global < .05].sort_values("FDR_global")
    if len(sigcal):
        for _, r in sigcal.head(10).iterrows():
            lines.append(f"- {r.analysis}, {r.metric}: partial rho {r.partial_spearman_rho:.3f}, n={int(r.n)}, FDR={r.FDR_global:.3g}.")
    else:
        lines.append("- No calprotectin association passed global FDR correction.")
    sigc = coupling[coupling.FDR_global < .05].sort_values("FDR_global")
    if len(sigc):
        for _, r in sigc.iterrows():
            lines.append(f"- {r.comparison}, convergent TCR burden versus {r.bcr_metric}: partial rho {r.partial_spearman_rho:.3f}, FDR={r.FDR_global:.3g}.")
    else:
        lines.append("- No GLIPH2 convergent-burden/B-cell module association passed global FDR correction.")
    lines += ["", "## Submission recommendation", "",
              "- Main-text candidates should be limited to findings that remain significant with participant-level analysis and exact or explicitly chain-specific clonotype definitions.",
              "- The AIRR abundance-encoding discrepancy remains unresolved; no analysis here uses the AIRR Clones field as a cell-count denominator.",
              "- Antigen labels are putative database overlaps and BCR regional-divergence metrics are not formal germline-based selection estimates."]
    (OUT / "PRIORITY_ANALYSES_RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_late_fusion()
    burden, cluster = build_antigen_convergence()
    run_continuous_inflammation(burden)
    write_summary()
    manifest = {
        "completed": [
            "clone-aware pseudobulk differential expression",
            "receptor-transcriptome proximity",
            "participant-specific BCR lineage diversification",
            "joint TCR-BCR late-fusion classification",
            "continuous calprotectin associations",
            "GLIPH2 antigen/state annotation and TCR-BCR program coupling",
        ],
        "seed": SEED,
    }
    (OUT / "priority_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Stage 2 complete")

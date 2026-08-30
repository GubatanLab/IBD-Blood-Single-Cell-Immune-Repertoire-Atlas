from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"C:/path/to/private-manuscript-workspace")
HI = ROOT / "High Impact Additional Analyses"
OUT = HI / "Priority Analyses"
BETA = OUT / "BetaOnly_GLIPH2"


def fmt_p(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.3g}"


# Corrected beta-only GLIPH2 results.
parts = []
for folder in ["CD_vs_Control", "UC_vs_Control", "CD_vs_UC"]:
    d = pd.read_csv(BETA / folder / "cluster_group_enrichment.csv")
    d["comparison"] = folder
    parts.append(d)
beta = pd.concat(parts, ignore_index=True)
beta.to_csv(OUT / "Table_PA5_beta_only_GLIPH2_all_clusters.csv", index=False)
beta[beta.fdr < .05].to_csv(OUT / "Table_PA5_beta_only_GLIPH2_FDR05_clusters.csv", index=False)

# Mixed-chain audit of the earlier file labeled as TCR-only.
mixed = pd.read_csv(OUT / "Table_PA5_GLIPH2_significant_cluster_members_unique.csv")
mixed["chain_family"] = mixed.TRBV.astype(str).str.extract(r"^(TR[ABGD])", expand=False).fillna("Other")
audit = mixed.groupby("chain_family").size().rename("n_records").reset_index()
audit["fraction"] = audit.n_records / audit.n_records.sum()
audit.to_csv(OUT / "Table_PA5_GLIPH2_input_chain_audit.csv", index=False)

summaries = pd.concat(
    [pd.read_csv(BETA / f / "summary.csv") for f in ["CD_vs_Control", "UC_vs_Control", "CD_vs_UC"]],
    ignore_index=True,
)
direction = beta[beta.fdr < .05].groupby("comparison").agg(
    n_significant=("tag", "size"),
    n_group1_enriched=("prop_g1", lambda x: int((x > .5).sum())),
    n_group2_enriched=("prop_g1", lambda x: int((x < .5).sum())),
).reset_index()
summaries = summaries.merge(direction, on="comparison", how="left").fillna(0)
summaries.to_csv(OUT / "Table_PA5_beta_only_GLIPH2_summary.csv", index=False)

# Supplementary beta-only convergence plot.
plotd = beta.copy()
plotd["log2_or"] = np.log2(plotd.fisher_or.clip(lower=1e-3, upper=1e3))
plotd["minus_log10_fdr"] = -np.log10(plotd.fdr.clip(lower=1e-300))
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True, sharey=True)
for ax, comp in zip(axes, ["CD_vs_Control", "UC_vs_Control", "CD_vs_UC"]):
    q = plotd[plotd.comparison.eq(comp)]
    sig = q.fdr < .05
    ax.scatter(q.loc[~sig, "log2_or"], q.loc[~sig, "minus_log10_fdr"], s=7, c="#AAAAAA", alpha=.35, linewidths=0)
    if sig.any():
        colors = np.where(q.loc[sig, "log2_or"] > 0, "#C2415D", "#2B6CB0")
        ax.scatter(q.loc[sig, "log2_or"], q.loc[sig, "minus_log10_fdr"], s=20, c=colors, alpha=.9, linewidths=0)
    ax.axhline(-np.log10(.05), color="0.5", lw=.8, ls="--")
    ax.axvline(0, color="0.65", lw=.7)
    ax.set_title(comp.replace("_", " "), fontsize=10, fontweight="bold")
    ax.set_xlabel("log2 odds ratio\n(first group enriched →)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("-log10 FDR")
fig.suptitle("Exact cell-defined beta-chain GLIPH2 enrichment", x=.01, ha="left", fontsize=12, fontweight="bold")
fig.savefig(OUT / "Figure_PA5_beta_only_GLIPH2_enrichment.png", dpi=600, bbox_inches="tight")
fig.savefig(OUT / "Figure_PA5_beta_only_GLIPH2_enrichment.pdf", bbox_inches="tight")
plt.close(fig)

# Load all central results.
module = pd.read_csv(OUT / "Table_PA1_clone_aware_pseudobulk_module_enrichment.csv")
de = pd.read_csv(OUT / "Table_PA1_clone_aware_pseudobulk_DE_all_genes.csv")
pb = pd.read_csv(OUT / "Table_PA1_pseudobulk_sample_manifest.csv")
prox0 = pd.read_csv(OUT / "Table_PA2_receptor_transcriptome_proximity_vs_null.csv")
proxg = pd.read_csv(OUT / "Table_PA2_receptor_transcriptome_proximity_group_tests.csv")
line_unadj = pd.read_csv(OUT / "Table_PA3_BCR_lineage_group_tests.csv")
line_adj = pd.read_csv(OUT / "Table_PA3_BCR_lineage_depth_adjusted_models.csv")
fusion = pd.read_csv(OUT / "Table_PA4_joint_TCR_BCR_nested_validation_summary.csv")
fusion_delta = pd.read_csv(OUT / "Table_PA4_joint_TCR_BCR_incremental_value.csv")
cal = pd.read_csv(OUT / "Table_PA6_continuous_calprotectin_associations.csv")
tb = pd.read_csv(HI / "Table_HI_CD_specific_T_B_module_coupling.csv")
tb_int = pd.read_csv(HI / "Table_HI_T_B_module_interaction_tests.csv")
pair = pd.read_csv(HI / "Table_HI_paired_chain_incremental_value.csv")

def module_row(modality, name):
    return module[(module.modality == modality) & (module.module == name)].iloc[0]

def de_row(modality, gene):
    return de[(de.modality == modality) & (de.gene == gene)].iloc[0]

pb_n = pb.groupby(["modality", "Diagnosis1"]).SampleID.nunique().unstack(fill_value=0)
beta_trb_fraction = float(audit.loc[audit.chain_family.eq("TRB"), "fraction"].iloc[0])

report = f"""# All priority paired TCR/BCR–single-cell analyses

Completed August 24, 2026

## Executive interpretation

The strongest new result is a clone-aware transcriptional analysis using state-matched, participant-paired pseudobulks. Expanded TCR-bearing cells preferentially expressed EOMES–ZEB2/TRM-like, effector-cytotoxic, and Th1/Tc1 programs, whereas expanded BCR-bearing cells preferentially expressed IgA mucosal plasma-cell, plasmablast/plasma-cell, antibody-secretion/UPR, IgG-inflammatory plasma-cell, and plasma-cell inflammatory antigen-presentation programs. These results provide participant-level support for the manuscript’s central claim that clonal expansion is linked to coherent effector states rather than merely to shifts in cell abundance.

The previously identified Crohn’s disease–specific coupling between expanded-TCR cytotoxicity and B-cell IgA/plasma differentiation remains the strongest cross-compartment result. Continuous inflammation analysis adds a modest but FDR-significant association between the B-cell IgG-inflammatory module and fecal calprotectin.

Several anticipated high-impact analyses were negative or exposed source-data problems. Joint TCR+BCR late fusion did not improve disease-control classification beyond TCR alone. Exact paired clone-mates were not more transcriptionally proximal than matched null cells. BCR lineage differences were not robust after sequence-depth adjustment. Most importantly, the prior files labeled “TCR-only” for GLIPH2 were 99.7% non-TRB records. Correct beta-only GLIPH2 found no disease-enriched clusters at FDR < 0.05; all significant disease-control clusters were control enriched.

## 1. Clone-aware, cell-state-matched pseudobulk expression

Expansion was defined from exact TRBV–TRBJ–CDR3 amino-acid identity for T cells and IGHV–IGHJ–CDR3 amino-acid identity for B cells. Expanded and singleton cells were matched within participant and annotated cell state before aggregation. The final biological unit was the participant-status pseudobulk. Models additionally adjusted for RNA feature complexity and mitochondrial read fraction.

- TCR analysis included {int(pb_n.loc['TCR','CD'])} CD, {int(pb_n.loc['TCR','UC'])} UC, and {int(pb_n.loc['TCR','Control'])} control participants.
- BCR analysis included {int(pb_n.loc['BCR','CD'])} CD, {int(pb_n.loc['BCR','UC'])} UC, and {int(pb_n.loc['BCR','Control'])} control participants.
- TCR program enrichment: EOMES–ZEB2 inflammatory CD8/TRM-like, FDR {fmt_p(module_row('TCR','EOMES_ZEB2_inflammatory_CD8_TRM_like').FDR_global)}; effector cytotoxicity, FDR {fmt_p(module_row('TCR','Effector_cytotoxicity').FDR_global)}; Th1/Tc1 inflammation, FDR {fmt_p(module_row('TCR','Th1_Tc1_inflammatory').FDR_global)}.
- Representative TCR genes: FGFBP2 log2FC {de_row('TCR','FGFBP2').logFC:.3f}, FDR {fmt_p(de_row('TCR','FGFBP2').FDR_global)}; TCF7 log2FC {de_row('TCR','TCF7').logFC:.3f}, FDR {fmt_p(de_row('TCR','TCF7').FDR_global)}; CCR7 log2FC {de_row('TCR','CCR7').logFC:.3f}, FDR {fmt_p(de_row('TCR','CCR7').FDR_global)}.
- BCR program enrichment: IgA mucosal plasma cell, FDR {fmt_p(module_row('BCR','IgA_mucosal_plasma_cell').FDR_global)}; plasma-cell inflammatory antigen presentation/UPR, FDR {fmt_p(module_row('BCR','plasma_cell_inflammatory_antigen_presentation_UPR').FDR_global)}; plasmablast/plasma differentiation, FDR {fmt_p(module_row('BCR','plasmablast_plasma_cell_differentiation').FDR_global)}; antibody secretion/UPR, FDR {fmt_p(module_row('BCR','antibody_secretion_UPR').FDR_global)}; IgG inflammatory plasma cell, FDR {fmt_p(module_row('BCR','IgG_inflammatory_plasma_cell').FDR_global)}.
- Representative BCR genes: BANK1 log2FC {de_row('BCR','BANK1').logFC:.3f}, FDR {fmt_p(de_row('BCR','BANK1').FDR_global)}; MS4A1 log2FC {de_row('BCR','MS4A1').logFC:.3f}, FDR {fmt_p(de_row('BCR','MS4A1').FDR_global)}; CCDC88A log2FC {de_row('BCR','CCDC88A').logFC:.3f}, FDR {fmt_p(de_row('BCR','CCDC88A').FDR_global)}.

Interpretation: expanded T-cell receptors are associated with an effector/cytotoxic program and loss of naive/central-memory features; expanded B-cell receptors are associated with antibody-secreting and mucosal plasma-cell programs. This should be integrated into the main Results.

## 2. Crohn’s disease–specific T–B program coupling

- Expanded-TCR cytotoxicity versus IgA mucosal plasma-cell module in CD: partial rho 0.495, 95% CI 0.231–0.689, n=56, adjusted P=0.000621.
- Expanded-TCR cytotoxicity versus plasmablast/plasma differentiation in CD: partial rho 0.589, 95% CI 0.394–0.727, n=56, adjusted P=0.000011.
- Diagnosis-by-TCR-score interactions: P=0.00251 and P=0.00127, respectively.

Interpretation: this is the most mechanistically distinctive integrative result and should remain in the main manuscript.

## 3. Paired-chain and joint-repertoire classification

Matched-cell nested modeling showed that alpha-chain information improved TCR disease-control classification beyond beta alone: CD versus control AUC 0.807 to 0.914 (delta 0.107, 95% CI 0.023–0.194) and UC versus control AUC 0.932 to 0.979 (delta 0.047, approximately 95% CI 0.000–0.111). BCR heavy-plus-light gains were smaller and their delta-AUC intervals included zero.

Equal-weight late fusion of independently nested paired-chain TCR and BCR predictions did not add disease-control performance beyond TCR alone:

- CD versus control: joint AUC 0.839 (95% CI 0.751–0.912); delta versus TCR −0.075 (95% CI −0.175–0.022).
- UC versus control: joint AUC 0.888 (95% CI 0.813–0.948); delta versus TCR −0.091 (95% CI −0.155 to −0.039).
- CD versus UC: joint AUC 0.773 (95% CI 0.698–0.840); delta versus TCR 0.026 (95% CI −0.032–0.083).

Interpretation: receptor pairing adds information within the TCR compartment, but simply combining TCR and BCR classifier probabilities is not synergistic. Use paired-chain TCR results in the main figure; retain late fusion as a supplement/negative analysis.

## 4. Continuous inflammation

Across 158 participants with IBD, the BCR IgG-inflammatory plasma-cell module correlated with log-transformed fecal calprotectin after adjustment for diagnosis, age, sex, and receptor depth (partial rho 0.267; FDR 0.0282). No exact paired-repertoire clonality or expansion metric passed global FDR correction.

Interpretation: this is a clinically relevant continuous association, but the effect is modest. It is appropriate as a small main panel or a prominent supplementary result, not as a stand-alone biomarker claim.

## 5. Receptor–transcriptome neighborhoods

Exact paired clone-mates were compared with participant- and cell-state-matched null sets in latent transcriptomic space. Neither TCR nor BCR clone-mate proximity differed from the matched null after FDR correction, and no diagnosis contrast was significant.

Interpretation: expanded clones occupy recognizable annotated programs, but exact clone identity does not create additional within-state transcriptomic neighborhood structure detectable here. Supplement only.

## 6. BCR lineage diversification and regional-divergence proxy

Participant-specific heavy-chain lineages were defined using IGHV, IGHJ, CDR3 length, and ≤15% CDR3 amino-acid distance. Two unadjusted contrasts for median V-region divergence passed global FDR, but all diagnosis coefficients were null after adjustment for unique heavy-sequence depth, age, sex, and batch (all adjusted-model FDR ≥ {line_adj.FDR.min():.3f}).

The source tables do not contain a germline alignment. Therefore, formal BASELINe-style selection inference was not possible. The supplied CDR-versus-framework measure is a within-lineage consensus divergence proxy and must not be labeled as selection pressure.

Interpretation: do not include as a main finding; retain as a sensitivity analysis and clearly state the germline limitation.

## 7. Corrected beta-only GLIPH2 and antigen-convergence audit

The earlier significant-cluster file labeled as “TCR-only” contained only {beta_trb_fraction*100:.2f}% TRB records; the remainder were TRA, TRG, or TRD. Exact beta-chain inputs were therefore rebuilt directly from cell-level dominant TRB calls.

- CD versus control: 115 participants, 54,552 unique beta CDR3s, 21 FDR-significant clusters; all 21 were control enriched and none was CD enriched.
- UC versus control: 91 participants, 41,074 unique beta CDR3s, 21 FDR-significant clusters; all 21 were control enriched and none was UC enriched.
- CD versus UC: 158 participants, 66,913 unique beta CDR3s, no FDR-significant clusters.

Interpretation: the current data do not support disease-enriched beta-chain convergence. Do not carry forward prior disease-enriched GLIPH2 claims or putative antigen annotations from the mixed-chain analysis. If gamma-delta convergence is biologically intended, it should be presented as a separately prespecified analysis with chain-correct Methods and independent multiple-testing control.

## 8. Repertoire abundance audit

The pre-existing AIRR objects encode controls with unit `Clones` counts and IBD samples with non-unit counts. Exact cell-barcode reconstruction does not reproduce broad IBD-associated increases in clonality or expansion. Until the field is traced to cells versus UMIs versus reads and all groups are rebuilt identically, AIRR-abundance-derived expansion claims should not be treated as submission-ready.

## Recommended manuscript integration

### Main manuscript

1. Clone-aware program enrichment: expanded TCRs align with EOMES–ZEB2/cytotoxic/Th1 programs, and expanded BCRs align with IgA/plasma/UPR/IgG-inflammatory programs.
2. CD-specific coupling between expanded-TCR cytotoxicity and IgA/plasma-cell differentiation.
3. Incremental value of paired alpha-plus-beta TCR features in nested participant-level classification.
4. Optional small panel: IgG-inflammatory B-cell module versus continuous fecal calprotectin.

### Supplement

1. Exact paired clone-state breadth and receptor–transcriptome proximity null analyses.
2. Joint TCR+BCR late-fusion classification.
3. BCR lineage-diversification depth sensitivity.
4. Corrected beta-only GLIPH2 results, including the absence of disease-enriched clusters.
5. Exact clonotype-definition, threshold, and depth sensitivity.

### Remove or rewrite before submission

1. Remove claims of disease-enriched beta-chain GLIPH2 clusters from the mixed-chain output.
2. Avoid formal BCR “selection” language without germline-aligned V-region sequences.
3. Avoid claiming that joint TCR+BCR classifiers outperform TCR models.
4. Resolve the AIRR `Clones` encoding discrepancy before retaining broad disease-associated clonality or clone-size conclusions.
5. Keep all therapy analyses framed as contemporaneous post-treatment response-status classification, not prediction.
"""

(OUT / "ALL_PRIORITY_ANALYSES_REPORT.md").write_text(report, encoding="utf-8")

deliverables = []
for p in sorted(OUT.glob("*")):
    if p.is_file():
        deliverables.append({"file": p.name, "type": p.suffix.lower().lstrip("."), "bytes": p.stat().st_size})
pd.DataFrame(deliverables).to_csv(OUT / "DELIVERABLES_INDEX.csv", index=False)

manifest_path = OUT / "priority_analysis_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
manifest["corrected_beta_only_gliph2"] = {
    "status": "completed",
    "mixed_chain_TRB_fraction": beta_trb_fraction,
    "disease_enriched_clusters_FDR05": 0,
}
manifest["final_report"] = str(OUT / "ALL_PRIORITY_ANALYSES_REPORT.md")
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Priority analyses finalized")

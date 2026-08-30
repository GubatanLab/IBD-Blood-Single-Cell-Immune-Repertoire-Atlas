# External validation results — exploratory, not integrated

Generated 2026-08-29. Canonical manuscript and figure assets were not modified.

## Analysis framework

- Exact paired TRA+TRB and IGH+IGK/IGL clonotypes were reconstructed from public V(D)J files.
- Transcriptional programs used the manuscript's locked TCR and BCR gene signatures.
- The inferential unit was the participant or participant-sample, not the cell.
- Mann–Whitney tests were used for independent disease groups, Wilcoxon signed-rank tests for paired longitudinal or cross-compartment contrasts, and Benjamini–Hochberg correction within each prespecified analysis family.
- Results with FDR < 0.05 are treated as statistically supported. FDR 0.05–0.10 is reported as suggestive only.

## 1. Longitudinal PBMC cohort (GSE261334)

Five healthy participants and ten participants with UC were profiled; UC samples were paired at baseline and week 6.

The strongest validation was a clone-size dose response among exact paired TCR clonotypes at UC baseline. Increasing clone size tracked cytotoxicity (median participant Spearman rho = 1.00), GZMK inflammatory memory (rho = 0.80), pathogenic Th17-like activation (rho = 0.60), Treg reprogramming (rho = 0.40), tissue-resident memory (rho = 0.30), cycling (rho = 0.40), and Tph/Tfh help (rho = 0.60); each passed family FDR < 0.05. In contrast, increasing clone size tracked loss of naive/central-memory (rho = -1.00), conventional Th17 (rho = -0.97), and broad Th17/IL23 (rho = -0.90) programs. These results support selective expansion of GZMK/cytotoxic/pathogenic-memory clonotypes rather than generalized expansion of canonical IL-17-producing cells.

At week 6, TCR expanded-cell fraction fell by a median 0.088, cytotoxic score by 0.273, pathogenic Th17 score by 0.087, and GZMK memory score by 0.138, while naive/central-memory score rose by 0.155. These changes were nominally significant but only suggestive after family correction (FDR 0.059–0.082). The broad Th17/IL23 score rose by 0.053 (FDR = 0.059), demonstrating that the pathogenic/GZMK and canonical IL23-axis signatures do not move as a single program.

No baseline UC-versus-healthy program difference passed FDR correction. Using one baseline sample per participant, activated-Treg and plasma-differentiation scores were positively correlated (rho = 0.571, nominal p = 0.026, FDR = 0.417), with a weaker Tph/Tfh–plasma-differentiation association (rho = 0.402, p = 0.137, FDR = 0.493). Neither cross-receptor association survived correction.

## 2. Blood–mucosa repertoire cohort (GSE125527)

Fifteen participants with rectal and/or blood/ileal sampling were analyzed. Exact paired TCR clonotypes did not show a corrected disease-associated increase in Th17 or Treg expansion. In rectum, the median expanded-cell fraction was 0.093 in UC versus 0.115 in healthy samples for Th17-classified cells (p = 0.189, FDR = 0.379), and 0.137 versus 0.134 for Treg-classified cells (p = 0.613, FDR = 0.919). Mixed Th17–Treg expanded clonotypes were numerically more frequent in UC rectum (0.224 versus 0.136; p = 0.148, FDR = 0.379), but this was not statistically supported.

The fraction of rectal exact paired TCR clonotypes also found in blood was 0.0225 in UC versus 0.0105 in healthy participants (difference 0.0120; p = 0.234, FDR = 0.469). Exact paired BCR sharing was near zero in both groups.

BCR expansion-bin analyses did not validate an expansion-driven plasma program. UC-versus-healthy differences in the participant-level expansion slopes were positive for plasma differentiation, antibody secretion/UPR, IgA mucosal, and IgG inflammatory programs, but all had FDR = 0.424. Among expanded BCR lineages, class-switch diversity was modestly higher in UC in rectum (+0.0338; p = 0.117, FDR = 0.281) and PBMC (+0.0993; p = 0.140, FDR = 0.281), again without corrected significance. True SHM could not be tested because the public receptor tables lack germline-alignment fields.

## 3. Blood–colon–MLN TCR traffic (GSE301689)

Across three participants with Crohn disease, 85, 21, and 47 exact paired TCR clonotypes, respectively, were shared between blood and colon. In inflamed colon, blood-shared clonotypes had a median GZMK-memory score 0.308 higher than colon-private clonotypes. Activated-Treg and Tph/Tfh-help scores were 0.132 and 0.101 lower, respectively; the broad Th17/IL23 difference was only +0.026. With n = 3, the smallest attainable two-sided signed-rank p value was 0.25 and no program passed FDR correction. These data are directionally consistent with preferential circulation of GZMK inflammatory-memory clones, but are descriptive rather than confirmatory.

## 4. Independent sequence-motif validation

The 2026 JCI Insight study's independent European cohort strongly validated two discovery motifs, `%IGSGANV` and `RD%LYG` (each 21/32 IBD versus 5/24 healthy; OR = 7.255, p = 0.0012, FDR = 0.003). However, those motifs were rare in the current internal PBMC repertoire: `%IGSGANV` occurred in 1/158 IBD and 0/24 healthy participants, and `RD%LYG` in 4/158 and 0/24, respectively. None of the five JCI motifs was enriched internally after correction, and none recurred as an exact significant GLIPH2 tag. Thus, the external publication validates the general existence of convergent IBD TCR motifs, but it does not independently validate the manuscript's specific sequence groups.

## 5. Analyses that could not be executed without changing the question

- SCP1690 cell-level files could be enumerated but not downloaded without portal authentication.
- No reproducible public cell-level accession was found for the multi-organ Crohn B-cell atlas.
- Donor-level response labels were not exposed for GSE261334, so response-stratified longitudinal validation was not possible.
- A frozen diagnostic/response classifier could not be transferred because the canonical workspace contains summary/SHAP outputs but no serialized estimator, coefficients, or preprocessing object. No replacement model was retrained.

## Interpretation

The external evidence most strongly reinforces a focused claim: exact paired T-cell clonal expansion preferentially marks GZMK inflammatory-memory, cytotoxic, and pathogenic/reprogrammed states, and those features trend toward contraction during treatment. It does not support a broad claim that canonical Th17 or Treg clonotypes are more expanded in IBD, nor that BCR expansion itself drives plasma-cell transcription. Cross-tissue and B-cell lineage findings are useful as directional supplementary evidence but are not ready for a main-figure claim without a larger independent cohort or authenticated SCP1690 analysis.

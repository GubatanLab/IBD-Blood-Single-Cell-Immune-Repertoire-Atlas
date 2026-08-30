# Reproducibility and provenance

## Analysis levels

The manuscript deliberately separates cell-level visualization from participant-level inference. Unless a legend states otherwise, statistical comparisons, correlations, bootstraps, permutations, and cross-validation splits are participant-aware.

## Canonical definitions

- Exact paired alpha-beta TCR clonotypes: concordant productive alpha and beta V gene, J gene, and CDR3 amino-acid identities within participant.
- Exact paired gamma-delta clonotypes: the analogous paired gamma/delta definition.
- Exact paired BCR clonotypes: concordant productive heavy and light V gene, J gene, and CDR3 amino-acid identities within participant.
- Expanded clonotype: at least two observed cells, with threshold sensitivities where reported.
- Sequence neighborhoods: cross-participant, non-identical receptor neighbors; exact receptor pairs are excluded.

## Recommended rerun sequence

1. Receptor QC and state annotation audit.
2. Figure 1 atlas exports and differential-abundance summaries.
3. Figure 2 TCR clone-size dose response and CD8 trajectory.
4. Figure 3 sequence-state graph, robustness tests, and paired gamma-delta analyses.
5. Figure 4 BCR expansion, pseudobulk programs, trajectory, and class-switching models.
6. Figure 5 SHM and germline-rooted lineage analyses.
7. Figure 6 participant-level T-cell–B-cell coordination and blocked validation.
8. Figure 7 immuneML preparation, model fitting, nested validation, permutation, and SHAP summaries.
9. External validation and supplementary sensitivity analyses.
10. Figure rendering and repository audit.

## Environment provenance

The historical analyses used multiple R and Python environments and did not retain one complete cross-language lockfile. The included environment manifests therefore document required package families without asserting an unrecoverable exact package state. For final archival release, export the validated environments (`conda env export --from-history`, `pip freeze`, and `renv::snapshot()`) after rerunning the full workflow.

## Path provenance

The original execution scripts contained machine-specific Windows roots. These were replaced with neutral `C:/path/to/private-*` placeholders. `tools/relocate_paths.py` performs a controlled literal replacement from `config/paths.json`; it does not fetch data or modify files outside the cloned repository.

## Claims and limitations

- Cross-sectional pseudotime is transcriptional ordering, not observed temporal progression.
- T-cell–B-cell correlations show coordinated participant-level states, not direct cell-cell interaction or mediation.
- BCR germline-inclusive branch length reflects accumulated maturation and must be interpreted separately from observed-only diversification.
- Clinical classifiers quantify contemporaneous information in repertoire features; therapy analyses are not prospective treatment-response predictions.
- SHAP-ranked k-mers are model attributions, not stable antigen-specificity assignments.


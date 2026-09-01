# T-helper–B-cell interactome analysis

This directory contains the executed MultiNicheNet provenance used to nominate
bidirectional helper-T-cell–B-cell signaling programs in Figure 6G.

## Workflow

Run the scripts from the repository root in this order:

1. `audit_multinichenet_t_b_inputs.R` audits state coverage and sample metadata.
2. `run_multinichenet_t_b_coupling.R` performs the four prespecified
   MultiNicheNet comparisons.
3. `postprocess_multinichenet_t_b_coupling.R` assigns canonical signaling
   families and builds communication/program association summaries.
4. `run_clone_aware_multinichenet_validation.R` tests prioritized ligand and
   receptor expression in expanded versus singleton clonotypes.

The workflows require controlled CD4 and B-lineage Seurat objects and the
official human NicheNet ligand–receptor and ligand–target priors. Configure:

- `IBD_PRIVATE_WORKSPACE`: root containing the controlled manuscript inputs.
- `IBD_MULTINICHENET_PRIORS`: directory containing the two NicheNet prior RDS
  files. If unset, the workflow uses
  `<IBD_PRIVATE_WORKSPACE>/analysis_inputs/multinichenet`.
- `IBD_MULTINICHENET_OUTPUT`: output directory. If unset, results are written
  to `analysis_outputs/multinichenet` under the current working directory.

Raw single-cell objects, participant-level communication tables, and large RDS
intermediates are intentionally excluded. The aggregate table plotted in Figure
6G is available at
`source_data/main/Figure_6_panel_G_multinichenet_interactions.csv`. A concise
analysis report is available in `docs/multinichenet-analysis.md`.

MultiNicheNet infers expression-compatible communication; these results do not
establish physical contact, peptide–TCR specificity, signaling flux, mediation,
or causality.

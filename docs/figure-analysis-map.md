# Figure-to-analysis map

## Main figures

| Figure | Analysis focus | Principal workflows |
|---|---|---|
| 1 | Cohort design, clone-aware atlas, receptor recovery, Milo differential abundance | `analysis/figures/extract_figure1_umaps_from_seurat.R`, `extract_figure1_clone_and_participant_effects.R`, `extract_figure1_robustness_data.R`, `build_cell_press_redrawn_figures.py` |
| 2 | TCR clonality, clone-size dose response, CD8 pseudotime, GZMK/cytotoxic/pathogenic-memory states | `analysis/figures/export_figure2_clone_size_dose_response.R`, `export_figure2b_paired4_dose_response.R`, `run_cd8_tcr_trajectory.py`, `run_cd8_slingshot_validation.R`, `build_consolidated_figure2.py` |
| 3 | Cross-participant paired-TCR sequence convergence and paired gamma-delta architecture | `analysis/core/export_paired_tcr_sequence_state.R`, `run_tcr_sequence_state_graph.py`, `analyze_figure3_high_impact.py`, `export_paired_gammadelta_figure3.R`, `analyze_paired_gammadelta_figure3.py` |
| 4 | BCR clonal focusing, expansion-linked programs, B-cell trajectory, class switching | `analysis/figures/export_bcr_expansion_maturation_inputs.R`, `analyze_bcr_expansion_maturation.py`, `run_bcr_trajectory_priority.R`, `run_figure4c_mixed_effects_robustness.R`, `run_figure4g_count_models.R`, `run_figure4g_dirichlet_multinomial.py` |
| 5 | Heavy-chain SHM, germline-rooted lineages, observed-only diversification, clinical context | `analysis/core/prepare_bcr_germline_inputs.R`, `analyze_bcr_germline_lineages.py`, `run_bcr_lineage_diversification.R`, `analyze_bcr_figure5_lineage_ibd.py`, `analyze_bcr_figure5_enhancements.py`, `analyze_bcr_lineage_threshold_effects.py` |
| 6 | Participant-level helper/regulatory and cytotoxic T-cell–B-cell coordination plus bidirectional ligand–receptor prioritization | `analysis/core/run_high_impact_integrative_analyses.R`, `run_cd_specific_module_coupling.py`, `analysis/figures/export_th17_treg_b_helper.R`, `run_th17_treg_b_helper.py`, `analysis/interactome/audit_multinichenet_t_b_inputs.R`, `analysis/interactome/run_multinichenet_t_b_coupling.R`, `analysis/interactome/postprocess_multinichenet_t_b_coupling.R`, `analysis/interactome/run_clone_aware_multinichenet_validation.R`, `analysis/figures/build_figure6_with_multinichenet.py` |
| 7 | immuneML diagnosis, inflammation, and biologic-response benchmarks | `analysis/immuneml/prepare_tcr_chain_immuneml.py`, `prepare_bcr_chain_immuneml.py`, `run_tcr_chain_kmer_svm_models.py`, `run_bcr_svm_feature_models.py`, `analysis/core/run_ml_sensitivity_validation.py`, `analysis/figures/build_figure7_immunity_revision.py` |

## Supplementary figures

| Figure | Supporting analyses | Principal workflows |
|---|---|---|
| S1 | Receptor recovery and adaptive-state marker validation | Figure 1 export and robustness workflows |
| S2 | TCR diversity, clone-state robustness, interaction tests, series replication, external motif context | `analysis/core/run_exact_clonotype_definition_sensitivity.R`, `run_clone_state_interactions.R`, `run_beta_only_gliph2_exact.R`, `run_gliph2_ibdtcr_shared_batch_sensitivity.R` |
| S3 | Independent longitudinal UC program validation | `analysis/external_validation/run_external_validation.py` |
| S4 | BCR repertoire, expansion programs, isotype, and SHM context | Figure 4 export/model workflows and `analysis/figures/redraw_s2_s4_cell_press.py` |
| S5 | BCR lineage reconstruction and threshold sensitivity | Figure 5 lineage workflows and `analysis/core/summarize_bcr_lineage_thresholds.py` |
| S6 | Transcription-to-lineage maturation bridge and clinical context | `analysis/core/analyze_bcr_figure5_enhancements.py`, `analysis/figures/build_bcr_trajectory_priority_figures.py` |
| S7 | Global TCR–BCR covariation and robustness | `analysis/core/run_high_impact_integrative_analyses.R`, `run_cd_specific_module_coupling.py` |
| S8 | Th17/Treg clone architecture and helper–B-cell covariation | `analysis/figures/export_th17_treg_b_helper.R`, `run_th17_treg_b_helper.py`, `audit_th17_treg_helper_inputs.R` |
| S9 | External mucosal and cross-tissue comparisons | `analysis/external_validation/run_external_validation.py` |
| S10 | Nested ML, permutation, calibration, and optimism | `analysis/core/run_ml_sensitivity_validation.py`, `analysis/ml_validation/build_supplementary_figures.py` |

The final PDFs and complete legends are in `figures/`. Aggregate panel values are in `source_data/`; controlled inputs are described in `data/README.md`.

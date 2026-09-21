# Paper update — 20 September 2026

## Included artifacts

- Current saved manuscript and its 64-reference bibliography.
- Seven reviewed main-figure PDFs and ten reviewed supplementary-figure PDFs.
- All figure and table legends extracted from that manuscript.
- Supplementary Tables S1-S7, with original table values and workbook formatting preserved.
- Current aggregate Figure 7 performance and SHAP source tables, including complete therapy-feature attributions.
- Executed figure-layout workflows and helper imports used by the current Figure 7 workflow.

## Figure interpretation

Figure S2 consists of A-D on one page: repertoire diversity, clone-size composition, paired-alpha-beta cell-state composition, and independent motif context. Former D-O panels were removed; the former motif panel P is now D. The current manuscript and legends define which analyses remain part of the paper.

Figure 2A's expanded-cell fraction uses exact productive TCR-beta V/J/CDR3 amino-acid identity within participant. The paired-alpha-beta analyses and CDR3-based repertoire summaries use different definitions; their fractions and diversity estimates are not interchangeable.

Figure 5 distinguishes accumulated germline divergence from diversification among observed lineage members. Figure 6 includes the helper-B-cell interactome panel, whose prioritized interactions remain inferred. Figure 7 distinguishes fully nested diagnosis benchmarks, conditional post-selection clinical-state estimates, and descriptive SHAP attributions from selected-model refits.

## Figure-assembly provenance

The dated scripts preserve the executed layout steps and require the matching local intermediate figure assets. They are not a one-command reconstruction from the committed PDFs alone.

1. Start from the retained supplementary artwork after removal of former S2D-O and relabeling of former S2P to D.
2. `analysis/figures/standardize_supplement_panel_geometry_20260920.py` standardizes the supplementary panel geometry using the archived artwork and review renders.
3. `analysis/figures/compact_s2_single_page_20260920.py` lays out S2A-D on one page.
4. `analysis/figures/build_figure7_clinical_translation.py` supplies the current clinical-state estimates and selected-feature summaries from authorized inputs. Its helper modules retain their historical filenames.
5. `analysis/figures/rebuild_figure7_layout_review_20260920.py` redraws the numerical panels from existing source CSVs without refitting models.
6. `analysis/figures/review_main_figures_20260920.py` applies the final typography and boundary corrections to the matching local source PDFs.

Original scientific values were preserved when copying this snapshot. Workstation paths in the Figure 7 selection-source provenance were replaced with neutral placeholders, and text line endings follow repository conventions. Analyses and model fitting were not rerun during the repository update. Primary cohort files and participant-level predictions remain outside Git. Neutral path placeholders follow `config/paths.example.json`; the historical typography workflows use installed Windows Arial fonts.

## Current Figure 7 numerical sources

Under `source_data/clinical_models/`:

- `Figure_7BDF_performance_source_data.csv`: 24 aggregate endpoint/model performance records.
- `Figure_7CEG_SHAP_top6_source_data.csv`: 96 displayed selected-feature attribution records.
- `Figure_7G_complete_therapy_SHAP_source_data.csv`: 160 therapy-feature attribution records, including supplementary attribution context.

Older clinical-model source files remain available as provenance and should not override the current figure legends.

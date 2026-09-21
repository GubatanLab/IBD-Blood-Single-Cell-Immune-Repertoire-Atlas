# IBD Blood Single-Cell Immune Repertoire Atlas

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22866361.svg)](https://doi.org/10.5281/zenodo.22866361)

Analysis code and figure workflows for the manuscript **“Single-cell immune repertoire atlas maps coordinated circulating adaptive immune states in inflammatory bowel disease.”**

This repository links single-cell transcriptomes to paired TCR and BCR repertoires across Crohn's disease, ulcerative colitis, and non-IBD controls. It contains the analysis provenance for all seven main figures, ten supplementary figures, external-validation analyses, BCR lineage reconstruction, TCR sequence-neighborhood analyses, and the immuneML clinical-classification benchmarks.

## Current paper snapshot — 20 September 2026

The current manuscript, 64-reference bibliography, reviewed Figures 1–7 and S1–S10, manuscript-matched legends, and Supplementary Tables S1–S7 are included. Figure S2 now contains panels A–D on one page. Its former D–O panels are no longer part of the current supplement; the former motif panel P is now D.

See [the release notes](docs/paper-update-2026-09-20.md), [manuscript](manuscript/README.md), and [supplementary tables](tables/README.md). The current Figure 7 numerical sources are named `Figure_7BDF`, `Figure_7CEG`, and `Figure_7G`; older source files are retained as historical provenance.

## Repository status

The repository is publicly available. The first archival release is [v1.0.0](https://github.com/GubatanLab/IBD-Blood-Single-Cell-Immune-Repertoire-Atlas/releases/tag/v1.0.0), with version DOI [10.5281/zenodo.22866361](https://doi.org/10.5281/zenodo.22866361).

Public access does not change the existing reuse terms: no open-source license is granted. Contact the Gubatan Lab before redistributing code or figures. Third-party materials retain their applicable terms.

See [public-release notes](docs/public-release-v1.0.0.md) and the [current data and code availability statement](docs/data-and-code-availability.md). The included manuscript is a dated scientific snapshot; its pending-release wording predates this archival release.

## Contents

- `analysis/core/`: clone-aware TCR/BCR, sequence-neighborhood, lineage, pseudobulk, and sensitivity analyses
- `analysis/interactome/`: MultiNicheNet T-helper–B-cell communication and clone-aware validation workflows
- `analysis/figures/`: canonical and supporting figure-generation workflows
- `analysis/external_validation/`: public-cohort validation workflows
- `analysis/immuneml/`: immuneML input preparation, model fitting, evaluation, and SHAP summaries
- `analysis/ml_validation/`: nested-validation, permutation, calibration, and optimism analyses
- `immuneml/`: reusable TCR and BCR immuneML specifications, including chain-specific and paired-chain models
- `source_data/`: aggregate, non-identifying source outputs used in displayed panels
- `figures/main/`: canonical main-figure PDFs and legends
- `figures/supplementary/`: canonical supplementary-figure PDFs and legends
- `manuscript/`: current manuscript snapshot and bibliography
- `tables/`: supplementary tables and their current legends
- `docs/figure-analysis-map.md`: panel-to-code map for Figures 1–7 and Figures S1–S10
- `docs/reproducibility.md`: execution order, validation hierarchy, and provenance limitations
- `data/README.md`: controlled-data policy and required input classes

## Reproducibility model

The repository separates three layers:

1. **Controlled inputs** — single-cell expression objects, paired AIRR repertoires, and participant-level clinical metadata. These are not committed.
2. **Analysis provenance** — the R, Python, and immuneML workflows used to derive the manuscript results.
3. **Shareable outputs** — aggregate source data, figure legends, and publication figures.

The original analysis scripts were retained to preserve executed provenance. Known machine-specific roots were replaced with neutral placeholders. After placing authorized inputs on a secure system, copy `config/paths.example.json` to `config/paths.json`, enter the local roots, and run:

```powershell
python tools/relocate_paths.py --config config/paths.json --apply
```

Review the resulting path-only diff before executing analyses. The scripts do not download or reconstruct controlled cohort data.

## Suggested execution order

1. Prepare quality-controlled transcriptomic and paired-receptor objects described in `data/README.md`.
2. Run receptor audits and figure-specific exports in `analysis/core/` and `analysis/figures/`.
3. Run TCR clone-state, trajectory, and sequence-neighborhood workflows for Figures 2–3.
4. Run BCR expansion, trajectory, SHM, and lineage workflows for Figures 4–5.
5. Run participant-level T–B coordination and MultiNicheNet interactome workflows for Figure 6.
6. Prepare AIRR repertoires and execute immuneML specifications for Figure 7.
7. Run nested-validation and external-validation workflows before assembling final figures.

The complete panel-level order is in `workflow_manifest.tsv` and `docs/figure-analysis-map.md`.

## Environment

Python and R package manifests are provided in `environment/`. The historical workflows span several environments, so the manifests describe compatible package families rather than claiming a single exact lockfile that was not recorded during the original analyses. immuneML should be run in an isolated environment because its dependency constraints may differ from the single-cell analysis environment.

## Data availability

Primary cohort inputs are intentionally excluded from this GitHub repository. Sequencing datasets will be deposited in the NIH National Center for Biotechnology Information Gene Expression Omnibus (NCBI GEO) at the time of publication. An accession has not yet been assigned here. Linked clinical metadata remain subject to consent, institutional review board approvals, and applicable data-use restrictions.

Public external-validation analyses use the accessions documented in `data/README.md` and the manuscript. Aggregate source data in this repository must not be treated as a substitute for the primary data.

## Validation

Run the repository audit before committing or sharing:

```powershell
python tools/validate_repository.py
```

The audit checks expected figures/configurations, forbidden raw-data formats, machine-specific user paths, likely credentials, and internal participant identifiers in shareable source-data directories.

## Citation

Cite the specific archived release used in an analysis:

Gubatan, John Mark. (2026). *IBD Blood Single-Cell Immune Repertoire Atlas* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.22866361

`CITATION.cff` provides the software citation in GitHub and retains the repository's existing software-author attribution. Full manuscript authorship is provided in the included manuscript. Cite the associated manuscript separately when discussing its scientific findings; this software DOI does not imply journal acceptance or a published article DOI.

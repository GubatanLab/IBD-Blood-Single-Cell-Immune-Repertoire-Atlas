# IBD Blood Single-Cell Immune Repertoire Atlas

Analysis code and figure workflows for the manuscript **“Single-cell immune repertoire profiling reveals coordinated systemic adaptive immune remodeling in inflammatory bowel disease.”**

This repository links single-cell transcriptomes to paired TCR and BCR repertoires across Crohn's disease, ulcerative colitis, and non-IBD controls. It contains the analysis provenance for all seven main figures, ten supplementary figures, external-validation analyses, BCR lineage reconstruction, TCR sequence-neighborhood analyses, and the immuneML clinical-classification benchmarks.

## Repository status

The repository is private while the manuscript and controlled participant-level data remain under review. No open-source license is granted at this stage. Contact the Gubatan Lab before redistributing code or figures.

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

Controlled cohort inputs are intentionally excluded. Public external-validation analyses use the accessions documented in `data/README.md` and the manuscript. Aggregate source data in this repository must not be treated as a substitute for the controlled primary data.

## Validation

Run the repository audit before committing or sharing:

```powershell
python tools/validate_repository.py
```

The audit checks expected figures/configurations, forbidden raw-data formats, machine-specific user paths, likely credentials, and internal participant identifiers in shareable source-data directories.

## Citation

Until a journal citation or preprint identifier is available, cite the associated manuscript title and this repository. `CITATION.cff` will surface the software citation in GitHub.

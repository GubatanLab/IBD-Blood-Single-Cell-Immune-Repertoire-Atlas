# Data inputs and access

Primary cohort data are controlled and are not stored in this repository.

## Required controlled inputs

The workflows expect authorized local copies of:

- quality-controlled PBMC, CD4/Treg, CD8, and B-cell single-cell objects with count matrices, state annotations, acquisition series, and latent/UMAP reductions;
- productive paired TCR alpha-beta and gamma-delta calls linked to cells and participants;
- productive paired BCR heavy-light calls, isotype annotations, germline alignments, and linked cell states;
- participant-level diagnosis, objective intestinal inflammation, biologic exposure/response status, age, sex, receptor depth, and other covariates specified in STAR Methods;
- immuneML-compatible AIRR repertoire files generated from the controlled receptor tables.

Participant identifiers must remain pseudonymous and inside an approved secure environment. Do not commit AIRR repertoires, cell-level receptor tables, expression matrices, row-level clinical metadata, outer-fold participant predictions, or trained models.

## Public external-validation inputs

The external-validation workflows use public datasets documented in the manuscript:

- `GSE261334`: longitudinal ulcerative-colitis blood cohort used for locked T-cell/B-cell program validation;
- `GSE125527`: rectal immune-repertoire cohort used for mucosal clone-state comparisons;
- `GSE301689`: paired blood/colon cohort used for descriptive cross-tissue clone occupancy;
- previously published GLIPH/GLIPH2 motif resources used only for external motif context.

Downloaded public data should be placed outside the Git repository and referenced through `config/paths.json`.

## Shareable outputs

`source_data/` contains aggregate panel values and public-cohort derivatives selected for reproducibility. Internal sample-level tables were deliberately excluded even when identifiers were pseudonymous.


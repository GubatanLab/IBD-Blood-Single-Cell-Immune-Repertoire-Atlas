# immuneML workflows

This directory contains the reusable model specifications supporting Figure 7 and Figure S10.

## TCR specifications

- `tcr/tcr_chain_diagnosis_k3_k4_aa.yaml`
- `tcr/tcr_chain_diagnosis_k3_k4_nt.yaml`
- `tcr/deeprc_specs/`: diagnosis, inflammation, and biologic-response tasks across alpha, beta, alpha-beta, gamma, delta, and gamma-delta chain compartments

## BCR specifications

- `bcr/bcr_heavy.yaml`
- `bcr/bcr_light.yaml`
- `bcr/bcr_heavy_light.yaml`

The BCR specifications are the pruned chain-specific versions used for model comparison. Preparation and execution scripts are in `analysis/immuneml/`.

## Validation hierarchy

- Diagnosis benchmarks use participant-level fully nested cross-validation.
- Inflammation and biologic-response panels report participant-level out-of-fold, conditional post-selection estimates.
- Fixed-pipeline permutation nulls, calibration, and historical-screen optimism are summarized in Figure S10.
- SHAP values describe full-cohort refit models and do not establish stable antigen specificity.

## Excluded artifacts

Generated immuneML job trees, raw AIRR repertoires, fold-level predictions, checkpoints, serialized estimators, and participant-named YAML files are excluded. Recreate them only inside an approved controlled-data environment.


# Source-data scope

This directory contains aggregate values supporting displayed manuscript panels.

- `main/`: TCR clone-state, sequence-neighborhood, BCR/T–B coordination, Figure 6G MultiNicheNet, and external-validation summaries
- `supplementary/`: BCR lineage and public external-validation summaries
- `clinical_models/`: aggregate immuneML performance, SHAP, nested-validation, permutation, and calibration summaries

Internal participant-level values, cell-level tables, receptor sequences, outer-fold predictions, and clinical metadata are not included. Public external-validation tables can contain the public dataset's participant or sample labels and are retained solely to make the published validation analyses auditable.

`main/Figure_6_panel_G_multinichenet_interactions.csv` contains the aggregate,
canonical ligand–receptor interactions displayed in Figure 6G. Participant-level
communication scores and cell-level expression matrices remain controlled.

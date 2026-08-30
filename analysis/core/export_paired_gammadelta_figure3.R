#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(readr)
})

root <- "C:/path/to/private-manuscript-workspace"
outdir <- file.path(root, "High Impact Additional Analyses", "Figure 3 High Impact Revision")
srcdir <- file.path(root, "Cell Press Redrawn Figure Set", "Source Data")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(srcdir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds"
)

modules <- list(
  Effector_cytotoxicity = c(
    "NKG7", "GNLY", "PRF1", "GZMB", "GZMA", "GZMH", "GZMK", "GZMM", "CTSW",
    "CST7", "FGFBP2", "CCL5", "CCL4", "CCL3", "IFNG", "FASLG", "KLRD1", "KLRG1"
  ),
  Th1_Tc1_inflammatory = c(
    "TBX21", "STAT4", "CXCR3", "IFNG", "TNF", "IL12RB2", "CCL5", "CCL4", "NKG7",
    "GZMB", "PRF1", "CXCR6", "BHLHE40"
  ),
  EOMES_ZEB2_inflammatory_CD8_TRM_like = c(
    "EOMES", "ZEB2", "GZMB", "GZMH", "PRF1", "NKG7", "CX3CR1", "KLRG1", "TBX21",
    "CCL5", "CST7", "FGFBP2"
  ),
  Tissue_resident_mucosal_retention = c(
    "CD69", "ITGAE", "ITGA1", "CXCR6", "ZNF683", "RUNX3", "PRDM1", "RGS1", "CD101",
    "DUSP6", "AHR", "CCR6"
  ),
  Gut_homing_intestinal_trafficking = c(
    "ITGA4", "ITGB7", "ITGAE", "CCR9", "CCR6", "CXCR3", "CXCR6", "SELPLG", "S1PR1",
    "KLF2", "SELL", "GPR183", "CD69"
  )
)

valid_translation <- function(x) {
  !is.na(x) & nzchar(trimws(as.character(x))) & grepl("^[A-Z]+$", as.character(x)) &
    !grepl("[*]", as.character(x))
}

strip_allele <- function(x) sub("\\*.*$", "", as.character(x))

extract_one <- function(path, compartment) {
  obj <- readRDS(path)
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)

  keep <-
    grepl("^TRGC", as.character(md$TCR_Alpha_Gamma_C_gene_Dominant)) &
    grepl("^TRDC", as.character(md$TCR_Beta_Delta_C_gene_Dominant)) &
    valid_translation(md$TCR_Alpha_Gamma_CDR3_Translation_Dominant) &
    valid_translation(md$TCR_Beta_Delta_CDR3_Translation_Dominant) &
    !is.na(md$SampleID) & nzchar(as.character(md$SampleID)) &
    !is.na(md$AnnotationLevel2)

  md <- md[keep, , drop = FALSE]
  expr <- LayerData(obj[["RNA"]], layer = "data")[, md$cell, drop = FALSE]
  scores <- vapply(
    modules,
    function(genes) {
      present <- intersect(genes, rownames(expr))
      if (length(present) == 0) return(rep(NA_real_, ncol(expr)))
      Matrix::colMeans(expr[present, , drop = FALSE])
    },
    numeric(ncol(expr))
  )
  scores <- as.data.frame(scores, check.names = FALSE)

  result <- bind_cols(md, scores) %>%
    transmute(
      cell = as.character(cell),
      SampleID = as.character(SampleID),
      PatientID = as.character(PatientID),
      Diagnosis = as.character(Diagnosis1),
      Inflammation = as.character(Inflammation1),
      Biologic = as.character(Biologic),
      acquisition_series = sub("[AB]$", "", as.character(Batch)),
      compartment = compartment,
      state = as.character(AnnotationLevel2),
      gamma_c = strip_allele(TCR_Alpha_Gamma_C_gene_Dominant),
      gamma_v = strip_allele(TCR_Alpha_Gamma_V_gene_Dominant),
      gamma_j = strip_allele(TCR_Alpha_Gamma_J_gene_Dominant),
      gamma_cdr3 = as.character(TCR_Alpha_Gamma_CDR3_Translation_Dominant),
      delta_c = strip_allele(TCR_Beta_Delta_C_gene_Dominant),
      delta_v = strip_allele(TCR_Beta_Delta_V_gene_Dominant),
      delta_j = strip_allele(TCR_Beta_Delta_J_gene_Dominant),
      delta_cdr3 = as.character(TCR_Beta_Delta_CDR3_Translation_Dominant),
      across(all_of(names(modules)))
    ) %>%
    mutate(
      clone_id = paste(gamma_v, gamma_j, gamma_cdr3, delta_v, delta_j, delta_cdr3, sep = "|"),
      participant_clone_id = paste(SampleID, clone_id, sep = "::"),
      receptor_group = if_else(
        gamma_v == "TRGV9" & delta_v == "TRDV2",
        "TRGV9-TRDV2",
        "Other paired gamma-delta"
      )
    )

  rm(expr, obj)
  invisible(gc())
  result
}

cells <- bind_rows(lapply(names(paths), function(name) extract_one(paths[[name]], name)))

clone_sizes <- cells %>%
  count(SampleID, clone_id, name = "n_cells")

publicity <- cells %>%
  distinct(SampleID, clone_id) %>%
  count(clone_id, name = "n_participant_carriers")

cells <- cells %>%
  left_join(clone_sizes, by = c("SampleID", "clone_id")) %>%
  left_join(publicity, by = "clone_id") %>%
  mutate(
    expansion_class = if_else(n_cells >= 2, "Expanded", "Singleton"),
    sharing_class = if_else(n_participant_carriers >= 2, "Public exact", "Private exact")
  )

dominant_state <- cells %>%
  count(SampleID, clone_id, state, name = "state_cells") %>%
  group_by(SampleID, clone_id) %>%
  arrange(desc(state_cells), state, .by_group = TRUE) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  select(SampleID, clone_id, dominant_state = state)

clonotypes <- cells %>%
  group_by(
    SampleID, PatientID, Diagnosis, Inflammation, Biologic, acquisition_series, clone_id,
    gamma_v, gamma_j, gamma_cdr3, delta_v, delta_j, delta_cdr3, receptor_group
  ) %>%
  summarise(
    n_cells = n(),
    n_compartments = n_distinct(compartment),
    n_states = n_distinct(state),
    n_participant_carriers = first(n_participant_carriers),
    across(all_of(names(modules)), function(x) mean(x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  left_join(dominant_state, by = c("SampleID", "clone_id")) %>%
  mutate(
    expansion_class = if_else(n_cells >= 2, "Expanded", "Singleton"),
    sharing_class = if_else(n_participant_carriers >= 2, "Public exact", "Private exact")
  )

vpair_summary <- cells %>%
  group_by(gamma_v, delta_v) %>%
  summarise(
    n_cells = n(),
    n_participants = n_distinct(SampleID),
    n_participant_clonotypes = n_distinct(participant_clone_id),
    .groups = "drop"
  ) %>%
  mutate(cell_fraction = n_cells / sum(n_cells)) %>%
  arrange(desc(n_cells), gamma_v, delta_v)

state_summary <- cells %>%
  group_by(receptor_group, state) %>%
  summarise(
    n_cells = n(),
    n_participants = n_distinct(SampleID),
    n_participant_clonotypes = n_distinct(participant_clone_id),
    .groups = "drop"
  ) %>%
  group_by(receptor_group) %>%
  mutate(cell_fraction_within_group = n_cells / sum(n_cells)) %>%
  ungroup() %>%
  arrange(receptor_group, desc(n_cells), state)

diagnosis_summary <- cells %>%
  group_by(Diagnosis, receptor_group) %>%
  summarise(
    n_cells = n(),
    n_participants = n_distinct(SampleID),
    n_participant_clonotypes = n_distinct(participant_clone_id),
    .groups = "drop"
  ) %>%
  group_by(Diagnosis) %>%
  mutate(cell_fraction_within_diagnosis = n_cells / sum(n_cells)) %>%
  ungroup()

qc_summary <- tibble(
  metric = c(
    "productive_paired_gamma_delta_cells",
    "participants",
    "participant_specific_clonotypes",
    "unique_paired_receptors",
    "expanded_participant_specific_clonotypes",
    "participants_with_expanded_clonotype",
    "public_exact_receptors",
    "maximum_exact_receptor_carriers",
    "TRGV9_TRDV2_cells",
    "TRGV9_TRDV2_participants",
    "cells_with_gdT_transcriptional_annotation"
  ),
  value = c(
    nrow(cells),
    n_distinct(cells$SampleID),
    nrow(clonotypes),
    n_distinct(cells$clone_id),
    sum(clonotypes$n_cells >= 2),
    n_distinct(clonotypes$SampleID[clonotypes$n_cells >= 2]),
    sum(publicity$n_participant_carriers >= 2),
    max(publicity$n_participant_carriers),
    sum(cells$receptor_group == "TRGV9-TRDV2"),
    n_distinct(cells$SampleID[cells$receptor_group == "TRGV9-TRDV2"]),
    sum(cells$state == "gdT")
  )
)

outputs <- list(
  "Table_F3R6_paired_gamma_delta_cells.csv" = cells,
  "Table_F3R7_paired_gamma_delta_clonotypes.csv" = clonotypes,
  "Table_F3R8_gamma_delta_V_pair_summary.csv" = vpair_summary,
  "Table_F3R9_gamma_delta_state_summary.csv" = state_summary,
  "Table_F3R10_gamma_delta_diagnosis_summary.csv" = diagnosis_summary,
  "Table_F3R11_gamma_delta_QC_summary.csv" = qc_summary
)

for (filename in names(outputs)) {
  write_csv(outputs[[filename]], file.path(outdir, filename))
  write_csv(outputs[[filename]], file.path(srcdir, sub("^Table_F3R", "Figure_3_R", filename)))
}

message(
  "Exported ", nrow(cells), " productive paired gamma-delta cells and ", nrow(clonotypes),
  " participant-specific clonotypes"
)

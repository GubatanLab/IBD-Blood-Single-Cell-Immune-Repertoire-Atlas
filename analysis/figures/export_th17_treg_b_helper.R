#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(readr)
})

options(stringsAsFactors = FALSE)
set.seed(20260829)

root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
outdir <- file.path(root, "High Impact Additional Analyses", "Th17 Treg B Helper Analyses")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

cd4_path <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 2 TCR/CD42026.rds"
stopifnot(file.exists(cd4_path))

t_modules <- list(
  Th17_conventional = c("RORC", "CCR6", "KLRB1", "IL7R", "IL23R", "CCL20", "IL17A", "IL17F"),
  Th17_pathogenic = c("TBX21", "IFNG", "CXCR3", "GZMK", "CSF2", "CCL5"),
  Treg_suppressive = c("FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2", "TNFRSF18", "LRRC32", "ENTPD1"),
  Treg_reprogramming = c("RORC", "KLRB1", "CCR6", "TBX21", "IFNG", "CXCR3", "GZMK", "CCL5"),
  Tph_Tfh_help = c("CXCL13", "PDCD1", "ICOS", "MAF", "IL21", "TOX2", "TIGIT", "CD200", "SLAMF6", "BATF", "CD40LG", "CXCR5", "BCL6", "SH2D1A")
)

valid <- function(x) {
  x <- trimws(as.character(x))
  !is.na(x) & nzchar(x) & !toupper(x) %in% c("NA", "NONE", "NULL")
}

mean_or_na <- function(x) if (length(x) && any(is.finite(x))) mean(x, na.rm = TRUE) else NA_real_
series_from_batch <- function(x) sub("[AB]$", "", as.character(x))

module_scores <- function(obj, cells, modules) {
  expr <- LayerData(obj[["RNA"]], layer = "data")[, cells, drop = FALSE]
  scores <- lapply(modules, function(gs) {
    genes <- intersect(gs, rownames(expr))
    if (!length(genes)) return(rep(NA_real_, length(cells)))
    Matrix::colMeans(expr[genes, , drop = FALSE])
  })
  scores <- as.data.frame(scores, check.names = FALSE)
  rownames(scores) <- cells
  attr(scores, "genes_detected") <- vapply(modules, function(gs) {
    paste(intersect(gs, rownames(expr)), collapse = ";")
  }, character(1))
  scores
}

message("Reading CD4 object")
cd4 <- readRDS(cd4_path)
tm <- as.data.frame(cd4[[]], stringsAsFactors = FALSE)
tm$cell <- rownames(tm)
tm$state <- as.character(tm$AnnotationLevel2)
tm$SampleID <- as.character(tm$SampleID)
tm$acquisition_series <- series_from_batch(tm$Batch)
tm$axis_state <- ifelse(tm$state %in% c("CD4 Th17", "CD4 Th1/Th17"), "Th17-like",
                        ifelse(grepl("^TReg", tm$state), "Treg", "Other CD4"))

fields <- c(
  "TCR_Alpha_Gamma_V_gene_Dominant", "TCR_Alpha_Gamma_J_gene_Dominant",
  "TCR_Alpha_Gamma_CDR3_Translation_Dominant", "TCR_Beta_Delta_V_gene_Dominant",
  "TCR_Beta_Delta_J_gene_Dominant", "TCR_Beta_Delta_CDR3_Translation_Dominant"
)
paired <- Reduce(`&`, lapply(fields, function(k) valid(tm[[k]]))) &
  grepl("^TRA", tm$TCR_Alpha_Gamma_V_gene_Dominant) &
  grepl("^TRB", tm$TCR_Beta_Delta_V_gene_Dominant) & valid(tm$SampleID)
tm$paired_clone_id <- NA_character_
tm$paired_clone_id[paired] <- apply(tm[paired, fields, drop = FALSE], 1, paste, collapse = "|")

clone_sizes <- tm %>%
  filter(valid(paired_clone_id), valid(SampleID)) %>%
  count(SampleID, paired_clone_id, name = "full_cd4_clone_size")
tm <- tm %>% left_join(clone_sizes, by = c("SampleID", "paired_clone_id"))
tm$full_cd4_clone_size[is.na(tm$full_cd4_clone_size)] <- 0L
tm$clone_status <- ifelse(tm$full_cd4_clone_size >= 2L, "Expanded",
                          ifelse(tm$full_cd4_clone_size == 1L, "Singleton", NA_character_))

message("Scoring prespecified T-cell programs")
ts <- module_scores(cd4, tm$cell, t_modules)
t_genes <- attr(ts, "genes_detected")
tm <- bind_cols(tm, ts[tm$cell, , drop = FALSE])
module_names <- names(t_modules)

axis_cells <- tm %>%
  filter(axis_state %in% c("Th17-like", "Treg"), valid(paired_clone_id)) %>%
  select(cell, SampleID, PatientID, Diagnosis1, Inflammation1, Biologic, Batch,
         acquisition_series, Age, Sex, state, axis_state, paired_clone_id,
         full_cd4_clone_size, clone_status, all_of(module_names))
write_csv(axis_cells, gzfile(file.path(outdir, "Table_TB1_paired_Th17_Treg_cells.csv.gz")))

clone_state <- axis_cells %>%
  group_by(SampleID, Diagnosis1, acquisition_series, paired_clone_id,
           full_cd4_clone_size, state, axis_state) %>%
  summarise(n_state_cells = n(), across(all_of(module_names), mean_or_na), .groups = "drop")
write_csv(clone_state, file.path(outdir, "Table_TB2_clone_state_programs.csv"))

clone_axis <- axis_cells %>%
  group_by(SampleID, Diagnosis1, acquisition_series, paired_clone_id, full_cd4_clone_size) %>%
  summarise(
    n_axis_cells = n(),
    n_th17_cells = sum(axis_state == "Th17-like"),
    n_treg_cells = sum(axis_state == "Treg"),
    axis_clone_class = case_when(
      n_th17_cells > 0 & n_treg_cells > 0 ~ "Mixed Th17/Treg",
      n_th17_cells > 0 ~ "Th17-only",
      TRUE ~ "Treg-only"
    ),
    across(all_of(module_names), mean_or_na),
    .groups = "drop"
  )
write_csv(clone_axis, file.path(outdir, "Table_TB3_axis_clone_summary.csv"))

participant_base <- tm %>%
  filter(valid(SampleID)) %>%
  group_by(SampleID) %>%
  summarise(
    Diagnosis1 = first(Diagnosis1), Inflammation1 = first(Inflammation1),
    Biologic = first(Biologic), Batch = first(Batch),
    acquisition_series = first(acquisition_series), Age = first(Age), Sex = first(Sex),
    cd4_cells = n(),
    th17_cells = sum(axis_state == "Th17-like"),
    treg_cells = sum(axis_state == "Treg"),
    th17_fraction = mean(axis_state == "Th17-like"),
    treg_fraction = mean(axis_state == "Treg"),
    mean_Th17_conventional_in_Th17 = mean_or_na(Th17_conventional[axis_state == "Th17-like"]),
    mean_Th17_pathogenic_in_Th17 = mean_or_na(Th17_pathogenic[axis_state == "Th17-like"]),
    mean_Treg_suppressive_in_Treg = mean_or_na(Treg_suppressive[axis_state == "Treg"]),
    mean_Treg_reprogramming_in_Treg = mean_or_na(Treg_reprogramming[axis_state == "Treg"]),
    mean_Tph_Tfh_help_all_CD4 = mean_or_na(Tph_Tfh_help),
    mean_Tph_Tfh_help_in_Th17 = mean_or_na(Tph_Tfh_help[axis_state == "Th17-like"]),
    mean_Tph_Tfh_help_in_Treg = mean_or_na(Tph_Tfh_help[axis_state == "Treg"]),
    paired_axis_cells = sum(axis_state != "Other CD4" & full_cd4_clone_size > 0),
    expanded_axis_cell_fraction = ifelse(sum(axis_state != "Other CD4" & full_cd4_clone_size > 0) > 0,
      mean(full_cd4_clone_size[axis_state != "Other CD4" & full_cd4_clone_size > 0] >= 2), NA_real_),
    expanded_th17_cell_fraction = ifelse(sum(axis_state == "Th17-like" & full_cd4_clone_size > 0) > 0,
      mean(full_cd4_clone_size[axis_state == "Th17-like" & full_cd4_clone_size > 0] >= 2), NA_real_),
    expanded_treg_cell_fraction = ifelse(sum(axis_state == "Treg" & full_cd4_clone_size > 0) > 0,
      mean(full_cd4_clone_size[axis_state == "Treg" & full_cd4_clone_size > 0] >= 2), NA_real_),
    .groups = "drop"
  )

participant_clones <- clone_axis %>%
  group_by(SampleID) %>%
  summarise(
    axis_clones = n(),
    expanded_axis_clones = sum(full_cd4_clone_size >= 2),
    mixed_axis_clones = sum(axis_clone_class == "Mixed Th17/Treg"),
    mixed_axis_clone_fraction = mixed_axis_clones / axis_clones,
    mixed_axis_cell_fraction = sum(n_axis_cells[axis_clone_class == "Mixed Th17/Treg"]) / sum(n_axis_cells),
    .groups = "drop"
  )
participant <- participant_base %>% left_join(participant_clones, by = "SampleID")
write_csv(participant, file.path(outdir, "Table_TB4_participant_T_features.csv"))

state_delta <- axis_cells %>%
  filter(clone_status %in% c("Expanded", "Singleton")) %>%
  pivot_longer(cols = all_of(module_names), names_to = "module", values_to = "score") %>%
  group_by(SampleID, Diagnosis1, acquisition_series, state, axis_state, module, clone_status) %>%
  summarise(score = mean_or_na(score), n_cells = n(), .groups = "drop") %>%
  group_by(SampleID, state, module) %>%
  filter(n_distinct(clone_status) == 2L) %>%
  ungroup() %>%
  select(-n_cells) %>%
  pivot_wider(names_from = clone_status, values_from = score) %>%
  mutate(expanded_minus_singleton = Expanded - Singleton)
write_csv(state_delta, file.path(outdir, "Table_TB5_state_matched_expanded_singleton_deltas.csv"))

manifest <- tibble(
  compartment = "CD4",
  module = names(t_modules),
  genes_requested = vapply(t_modules, paste, collapse = ";", character(1)),
  genes_detected = unname(t_genes),
  score = "Mean log-normalized RNA expression across detected prespecified genes"
)
write_csv(manifest, file.path(outdir, "Table_TB0_module_manifest.csv"))

audit <- tibble(
  item = c("CD4 object", "Th17-like states", "Treg states", "clonotype definition",
           "expanded definition", "sharing null", "primary analysis unit"),
  value = c(cd4_path, "CD4 Th17; CD4 Th1/Th17", "all AnnotationLevel2 labels beginning TReg",
            "Exact paired alpha/beta V+J+CDR3 amino-acid identity within participant",
            ">=2 CD4 cells", "State-label permutation within participant preserving clone sizes and state abundance",
            "Participant")
)
write_csv(audit, file.path(outdir, "Table_TB0_analysis_audit.csv"))

message("Wrote Th17/Treg clonotype exports to ", outdir)

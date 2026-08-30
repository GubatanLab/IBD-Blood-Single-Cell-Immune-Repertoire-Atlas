#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(readr)
})

options(stringsAsFactors = FALSE)
set.seed(20260828)

root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
outdir <- file.path(root, "High Impact Additional Analyses", "Literature Guided Ranked Analyses")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 2 TCR/CD42026.rds",
  B = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026.rds"
)
stopifnot(all(file.exists(paths)))

t_modules <- list(
  Tph_core = c("CXCL13", "PDCD1", "ICOS", "MAF", "IL21", "TOX2", "TIGIT", "CD200", "SLAMF6", "BATF", "CD40LG"),
  Tfh_core = c("CXCR5", "BCL6", "PDCD1", "ICOS", "MAF", "IL21", "SH2D1A", "TOX2", "TIGIT", "CD40LG"),
  Tph_Tfh_help = c("CXCL13", "PDCD1", "ICOS", "MAF", "IL21", "TOX2", "TIGIT", "CD200", "SLAMF6", "BATF", "CD40LG", "CXCR5", "BCL6", "SH2D1A")
)

b_modules <- list(
  Plasmablast_plasma_differentiation = c("PRDM1", "XBP1", "IRF4", "MZB1", "SDC1", "JCHAIN", "SSR4", "FKBP11", "DERL3", "TNFRSF17", "SLAMF7", "CD38", "SEC11C"),
  IgA_mucosal_plasma_cell = c("IGHA1", "IGHA2", "JCHAIN", "MZB1", "XBP1", "SDC1", "TNFRSF17", "CCR10", "PRDM1", "IRF4"),
  IgG_inflammatory_plasma_cell = c("IGHG1", "IGHG2", "IGHG3", "IGHG4", "JCHAIN", "MZB1", "XBP1", "SDC1", "PRDM1", "IRF4", "CXCR4"),
  Atypical_memory_CD11c_like = c("FCRL5", "FCRL4", "ITGAX", "TBX21", "ZEB2", "CXCR3", "DUSP4", "HOPX", "TLR7", "TLR9", "FCGR2B", "CD86", "CD80"),
  B_cell_antigen_presentation = c("HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "CD74", "CIITA", "HLA-DMA", "HLA-DMB", "CD86", "CD80", "CD40")
)

valid <- function(x) {
  x <- trimws(as.character(x))
  !is.na(x) & nzchar(x) & !toupper(x) %in% c("NA", "NONE", "NULL")
}

series_from_batch <- function(x) sub("[AB]$", "", as.character(x))

module_scores <- function(obj, cells, modules) {
  expr <- LayerData(obj[["RNA"]], layer = "data")[, cells, drop = FALSE]
  ans <- lapply(modules, function(gs) {
    genes <- intersect(gs, rownames(expr))
    if (length(genes) < 3L) return(rep(NA_real_, length(cells)))
    Matrix::colMeans(expr[genes, , drop = FALSE])
  })
  ans <- as.data.frame(ans, check.names = FALSE)
  rownames(ans) <- cells
  attr(ans, "genes_detected") <- vapply(modules, function(gs) {
    paste(intersect(gs, rownames(expr)), collapse = ";")
  }, character(1))
  ans
}

message("Reading and scoring CD4 object")
cd4 <- readRDS(paths[["CD4"]])
tm <- as.data.frame(cd4[[]], stringsAsFactors = FALSE)
tm$cell <- rownames(tm)
tm$state <- as.character(tm$AnnotationLevel2)
tm$SampleID <- as.character(tm$SampleID)
tm$acquisition_series <- series_from_batch(tm$Batch)

fields <- c(
  "TCR_Alpha_Gamma_V_gene_Dominant", "TCR_Alpha_Gamma_J_gene_Dominant",
  "TCR_Alpha_Gamma_CDR3_Translation_Dominant", "TCR_Beta_Delta_V_gene_Dominant",
  "TCR_Beta_Delta_J_gene_Dominant", "TCR_Beta_Delta_CDR3_Translation_Dominant"
)
paired <- Reduce(`&`, lapply(fields, function(k) valid(tm[[k]]))) &
  grepl("^TRA", tm$TCR_Alpha_Gamma_V_gene_Dominant) &
  grepl("^TRB", tm$TCR_Beta_Delta_V_gene_Dominant)
tm$paired_clone_id <- NA_character_
tm$paired_clone_id[paired] <- apply(tm[paired, fields, drop = FALSE], 1, paste, collapse = "|")
clone_sizes <- tm %>% filter(valid(paired_clone_id), valid(SampleID)) %>%
  count(SampleID, paired_clone_id, name = "paired_clone_size")
tm <- tm %>% left_join(clone_sizes, by = c("SampleID", "paired_clone_id"))
tm$paired_clone_size[is.na(tm$paired_clone_size)] <- 0L
tm$clone_status <- ifelse(tm$paired_clone_size >= 2L, "Expanded",
                          ifelse(tm$paired_clone_size == 1L, "Singleton", NA_character_))

ts <- module_scores(cd4, tm$cell, t_modules)
t_genes <- attr(ts, "genes_detected")
tm <- bind_cols(tm, ts[tm$cell, , drop = FALSE])
tm$Tph_minus_Tfh <- as.numeric(scale(tm$Tph_core)) - as.numeric(scale(tm$Tfh_core))

t_module_names <- c(names(t_modules), "Tph_minus_Tfh")
t_long <- tm %>%
  select(cell, SampleID, Diagnosis1, Inflammation1, Biologic, Batch,
         acquisition_series, state, paired_clone_size, clone_status,
         all_of(t_module_names)) %>%
  pivot_longer(cols = all_of(t_module_names), names_to = "module", values_to = "score")

t_overall <- t_long %>%
  filter(valid(SampleID)) %>%
  group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series, module) %>%
  summarise(mean_score = mean(score, na.rm = TRUE), n_cd4_cells = n(), .groups = "drop") %>%
  pivot_wider(names_from = module, values_from = mean_score, names_prefix = "cd4_mean_")

t_state_fraction <- tm %>% filter(valid(SampleID)) %>%
  group_by(SampleID) %>%
  summarise(
    cd4_cells = n(),
    cd4_tfh_fraction = mean(state == "CD4 Tfh"),
    cd4_hladr_memory_fraction = mean(state == "CD4 HLA-DR+ memory"),
    paired_ab_cells = sum(paired_clone_size > 0),
    expanded_paired_ab_fraction = ifelse(sum(paired_clone_size > 0) > 0,
      mean(paired_clone_size[paired_clone_size > 0] >= 2), NA_real_),
    expanded_tfh_fraction_of_paired = ifelse(sum(paired_clone_size > 0) > 0,
      mean(state[paired_clone_size > 0] == "CD4 Tfh" & paired_clone_size[paired_clone_size > 0] >= 2), NA_real_),
    .groups = "drop"
  )

# Within each participant and annotated CD4 state, compare exact paired-clone
# expanded cells with singleton cells. Equal state weights avoid compositional
# differences being mistaken for clone-associated expression.
t_state_delta <- t_long %>%
  filter(valid(SampleID), clone_status %in% c("Expanded", "Singleton")) %>%
  group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch,
           acquisition_series, state, module, clone_status) %>%
  summarise(score = mean(score, na.rm = TRUE), n_cells = n(), .groups = "drop") %>%
  group_by(SampleID, state, module) %>%
  filter(n_distinct(clone_status) == 2L) %>% ungroup() %>%
  select(-n_cells) %>%
  pivot_wider(names_from = clone_status, values_from = score) %>%
  mutate(delta = Expanded - Singleton)

t_delta <- t_state_delta %>%
  group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series, module) %>%
  summarise(clone_delta = mean(delta, na.rm = TRUE), matched_states = n(), .groups = "drop") %>%
  pivot_wider(names_from = module, values_from = c(clone_delta, matched_states),
              names_glue = "cd4_{.value}_{module}")

t_part <- t_overall %>% left_join(t_state_fraction, by = "SampleID") %>%
  left_join(t_delta, by = c("SampleID", "Diagnosis1", "Inflammation1", "Biologic", "Batch", "acquisition_series"))
write_csv(t_part, file.path(outdir, "Table_RA1_CD4_helper_by_participant.csv"))
write_csv(t_state_delta, file.path(outdir, "Table_RA1_CD4_state_matched_clone_deltas.csv"))
rm(cd4, tm, ts, t_long, t_state_delta); invisible(gc())

message("Reading and scoring B-cell object")
b <- readRDS(paths[["B"]])
bm <- as.data.frame(b[[]], stringsAsFactors = FALSE)
bm$cell <- rownames(bm)
bm$state <- as.character(bm$AnnotationLevel2)
bm$SampleID <- as.character(bm$SampleID)
bm$acquisition_series <- series_from_batch(bm$Batch)
bs <- module_scores(b, bm$cell, b_modules)
b_genes <- attr(bs, "genes_detected")
bm <- bind_cols(bm, bs[bm$cell, , drop = FALSE])

b_long <- bm %>%
  select(SampleID, Diagnosis1, Inflammation1, Biologic, Batch,
         acquisition_series, state, all_of(names(b_modules))) %>%
  pivot_longer(cols = all_of(names(b_modules)), names_to = "module", values_to = "score")
b_score <- b_long %>% filter(valid(SampleID)) %>%
  group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series, module) %>%
  summarise(mean_score = mean(score, na.rm = TRUE), n_b_cells = n(), .groups = "drop") %>%
  pivot_wider(names_from = module, values_from = mean_score, names_prefix = "b_mean_")
b_fraction <- bm %>% filter(valid(SampleID)) %>% group_by(SampleID) %>%
  summarise(
    b_cells = n(),
    b_plasma_fraction = mean(grepl("Plasma B Cell$", state)),
    b_iga_plasma_fraction = mean(state == "IgA Plasma B Cell"),
    b_igg_plasma_fraction = mean(state == "IgG Plasma B Cell"),
    b_igm_plasma_fraction = mean(state == "IgM Plasma B Cell"),
    b_switched_memory_fraction = mean(state == "Switched memory B"),
    b_atypical_memory_fraction = mean(state == "Atypical memory B"),
    .groups = "drop"
  )
b_part <- b_score %>% left_join(b_fraction, by = "SampleID")
write_csv(b_part, file.path(outdir, "Table_RA1_Bcell_response_by_participant.csv"))
rm(b, bm, bs, b_long); invisible(gc())

manifest <- bind_rows(
  tibble(compartment = "CD4", module = names(t_modules), genes_requested = vapply(t_modules, paste, collapse = ";", character(1)), genes_detected = unname(t_genes)),
  tibble(compartment = "B", module = names(b_modules), genes_requested = vapply(b_modules, paste, collapse = ";", character(1)), genes_detected = unname(b_genes))
)
write_csv(manifest, file.path(outdir, "Table_RA0_helper_axis_module_manifest.csv"))

audit <- tibble(
  item = c("CD4 object", "B-cell object", "clonotype definition", "expanded definition", "clone-aware comparison", "analysis unit"),
  value = c(paths[["CD4"]], paths[["B"]],
            "exact paired alpha/beta V+J+CDR3 amino-acid identity within participant",
            ">=2 cells", "expanded minus singleton within participant and CD4 state; equal state weights",
            "participant")
)
write_csv(audit, file.path(outdir, "Table_RA0_helper_axis_audit.csv"))
message("Wrote ranked helper-axis exports to ", outdir)

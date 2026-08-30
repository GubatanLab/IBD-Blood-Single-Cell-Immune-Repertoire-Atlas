#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(readr)
})

root <- "C:/path/to/private-manuscript-workspace"
outdir <- file.path(root, "Cell Press Redrawn Figure Set", "Source Data")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds"
)
stopifnot(all(file.exists(paths)))

modules <- list(
  EOMES_ZEB2_inflammatory_CD8_TRM_like = c(
    "EOMES","ZEB2","GZMB","GZMH","PRF1","NKG7","CX3CR1","KLRG1","TBX21","CCL5","CST7","FGFBP2"
  ),
  Effector_cytotoxicity = c(
    "NKG7","GNLY","PRF1","GZMB","GZMA","GZMH","GZMK","GZMM","CTSW","CST7","FGFBP2","CCL5","CCL4","CCL3","IFNG","FASLG","KLRD1","KLRG1"
  ),
  Th1_Tc1_inflammatory = c(
    "TBX21","STAT4","CXCR3","IFNG","TNF","IL12RB2","CCL5","CCL4","NKG7","GZMB","PRF1","CXCR6","BHLHE40"
  ),
  GZMK_inflammatory_memory = c(
    "GZMK","GZMA","CCL5","CCL4","CCL4L2","XCL1","XCL2","NKG7","DUSP2","CRTAM","EOMES","CXCR3","IL7R"
  )
)

valid <- function(x) {
  !is.na(x) & nzchar(trimws(as.character(x))) & toupper(trimws(as.character(x))) != "NA"
}

metadata_one <- function(path, compartment) {
  obj <- readRDS(path)
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  keep <- grepl("^TRAC", md$TCR_Alpha_Gamma_C_gene_Dominant) &
    grepl("^TRBC", md$TCR_Beta_Delta_C_gene_Dominant) &
    valid(md$TCR_Alpha_Gamma_V_gene_Dominant) &
    valid(md$TCR_Alpha_Gamma_J_gene_Dominant) &
    valid(md$TCR_Alpha_Gamma_CDR3_Translation_Dominant) &
    valid(md$TCR_Beta_Delta_V_gene_Dominant) &
    valid(md$TCR_Beta_Delta_J_gene_Dominant) &
    valid(md$TCR_Beta_Delta_CDR3_Translation_Dominant) &
    valid(md$SampleID) & valid(md$AnnotationLevel2)
  md <- md[keep, , drop = FALSE]
  ans <- md %>% transmute(
    cell,
    SampleID = as.character(SampleID),
    Diagnosis1 = as.character(Diagnosis1),
    compartment = compartment,
    state = as.character(AnnotationLevel2),
    clone_id = paste(
      TCR_Alpha_Gamma_V_gene_Dominant,
      TCR_Alpha_Gamma_J_gene_Dominant,
      TCR_Alpha_Gamma_CDR3_Translation_Dominant,
      TCR_Beta_Delta_V_gene_Dominant,
      TCR_Beta_Delta_J_gene_Dominant,
      TCR_Beta_Delta_CDR3_Translation_Dominant,
      sep = "|"
    )
  )
  rm(obj)
  invisible(gc())
  ans
}

metadata <- bind_rows(
  metadata_one(paths[["CD4"]], "CD4"),
  metadata_one(paths[["CD8"]], "CD8")
)
clone_sizes <- metadata %>% count(SampleID, clone_id, name = "clone_size")
metadata <- metadata %>%
  left_join(clone_sizes, by = c("SampleID", "clone_id")) %>%
  mutate(clone_bin = case_when(
    clone_size == 1L ~ "Singleton",
    clone_size == 2L ~ "2 cells",
    clone_size %in% 3:4 ~ "3-4 cells",
    clone_size >= 5L ~ ">=5 cells",
    TRUE ~ NA_character_
  ))

score_one <- function(path, compartment) {
  message("Scoring Figure 2B paired-alpha/beta programs in ", compartment)
  obj <- readRDS(path)
  md <- metadata %>% filter(.data$compartment == .env$compartment)
  expr <- LayerData(obj[["RNA"]], layer = "data")
  genes <- intersect(unique(unlist(modules)), rownames(expr))
  expr <- expr[genes, md$cell, drop = FALSE]
  scores <- vapply(modules, function(gs) {
    present <- intersect(gs, rownames(expr))
    Matrix::colMeans(expr[present, , drop = FALSE])
  }, numeric(ncol(expr)))
  scores <- as.data.frame(scores, check.names = FALSE)
  scores$cell <- colnames(expr)
  ans <- md %>% left_join(scores, by = "cell") %>%
    pivot_longer(cols = all_of(names(modules)), names_to = "module", values_to = "score") %>%
    group_by(SampleID, Diagnosis1, compartment, state, clone_bin, module) %>%
    summarise(score = mean(score, na.rm = TRUE), n_cells = n(), .groups = "drop")
  rm(expr, scores, obj)
  invisible(gc())
  ans
}

state_bin <- bind_rows(
  score_one(paths[["CD4"]], "CD4"),
  score_one(paths[["CD8"]], "CD8")
)

matched <- state_bin %>%
  group_by(SampleID, compartment, state, module) %>%
  filter(any(clone_bin == "Singleton")) %>%
  mutate(singleton_score = score[clone_bin == "Singleton"][1], delta = score - singleton_score) %>%
  ungroup()

participant <- matched %>%
  group_by(SampleID, Diagnosis1, compartment, clone_bin, module) %>%
  summarise(delta = mean(delta), n_matched_states = n(), .groups = "drop") %>%
  group_by(SampleID, Diagnosis1, clone_bin, module) %>%
  summarise(
    delta = mean(delta),
    n_matched_states = sum(n_matched_states),
    n_compartments = n_distinct(compartment),
    .groups = "drop"
  ) %>%
  mutate(clone_bin = factor(clone_bin, levels = c("Singleton", "2 cells", "3-4 cells", ">=5 cells"))) %>%
  arrange(module, Diagnosis1, SampleID, clone_bin)

write_csv(participant, file.path(outdir, "Figure2B_paired4_clone_size_participant_deltas.csv"))
write_csv(tibble(
  item = c("clonotype definition", "state matching", "programs", "integration status"),
  value = c(
    "within-participant exact productive paired alpha-beta V+J+CDR3 amino-acid identity",
    "clone-size-bin score minus singleton score within participant, compartment, and annotated state; equal state and compartment weights",
    paste(names(modules), collapse = "; "),
    "canonical Figure 2B revision requested 2026-08-29"
  )
), file.path(outdir, "Figure2B_paired4_analysis_manifest.csv"))

message("Wrote strict paired-alpha/beta Figure 2B source data")

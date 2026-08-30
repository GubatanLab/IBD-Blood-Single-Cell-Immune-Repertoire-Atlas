#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(dplyr)
  library(readr)
})

root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
outdir <- file.path(root, "High Impact Additional Analyses", "Literature Guided Ranked Analyses")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
path <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026.rds"

valid <- function(x) {
  x <- trimws(as.character(x))
  !is.na(x) & nzchar(x) & !toupper(x) %in% c("NA", "NONE", "NULL")
}
gene <- function(x) sub("\\*.*$", "", as.character(x))
core <- function(x) {
  x <- toupper(trimws(as.character(x)))
  x <- sub("^C", "", x)
  x <- sub("[FW]$", "", x)
  x
}

message("Reading B-cell object")
obj <- readRDS(path)
m <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
fields <- c("BCR_Heavy_V_gene_Dominant", "BCR_Heavy_J_gene_Dominant",
            "BCR_Heavy_CDR3_Translation_Dominant", "BCR_Light_V_gene_Dominant",
            "BCR_Light_J_gene_Dominant", "BCR_Light_CDR3_Translation_Dominant")
ok <- valid(m$SampleID) & Reduce(`&`, lapply(fields, function(k) valid(m[[k]]))) &
  grepl("^IGH", m$BCR_Heavy_V_gene_Dominant) &
  grepl("^IG[KL]", m$BCR_Light_V_gene_Dominant)
m <- m[ok, , drop = FALSE]
m$v_gene <- gene(m$BCR_Heavy_V_gene_Dominant)
m$j_gene <- gene(m$BCR_Heavy_J_gene_Dominant)
m$heavy_core <- core(m$BCR_Heavy_CDR3_Translation_Dominant)
m$light_core <- core(m$BCR_Light_CDR3_Translation_Dominant)
m$light_v_gene <- gene(m$BCR_Light_V_gene_Dominant)
m$light_j_gene <- gene(m$BCR_Light_J_gene_Dominant)
m$isotype <- as.character(m$BCR_Heavy_C_gene_Dominant)
m$state <- as.character(m$AnnotationLevel2)
m$switched <- !m$isotype %in% c("IGHM", "IGHD")
m$plasma <- grepl("Plasma B Cell$", m$state)

ans <- m %>%
  group_by(SampleID, v_gene, j_gene, heavy_core, light_core) %>%
  summarise(
    mapped_b_cells = n(),
    switched_fraction = mean(switched),
    plasma_fraction = mean(plasma),
    isotypes = paste(sort(unique(isotype)), collapse = ";"),
    states = paste(sort(unique(state)), collapse = ";"),
    light_v_genes = paste(sort(unique(light_v_gene)), collapse = ";"),
    light_j_genes = paste(sort(unique(light_j_gene)), collapse = ";"),
    .groups = "drop"
  )
write_csv(ans, file.path(outdir, "Table_RA4_exact_paired_BCR_cell_annotations.csv"))
message("Wrote ", nrow(ans), " paired-BCR annotation rows")

#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(readr)
})

root <- "C:/path/to/private-manuscript-workspace"
object_path <- "C:/path/to/private-user-home/OneDrive/Desktop/BCR Module Scores/BCell2026.rds"
reference_path <- file.path(
  root, "High Impact Additional Analyses", "BCR Trajectory Priority",
  "Table_BT1_balanced_cell_pseudotime.csv.gz"
)
outdir <- file.path(root, "High Impact Additional Analyses", "BCR Expansion Maturation 20260829")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

modules <- list(
  IgA_mucosal_plasma_cell = c(
    "IGHA1", "IGHA2", "JCHAIN", "MZB1", "XBP1", "SDC1", "TNFRSF17", "CCR10", "PRDM1", "IRF4"
  ),
  plasmablast_plasma_cell_differentiation = c(
    "PRDM1", "XBP1", "IRF4", "MZB1", "SDC1", "JCHAIN", "SSR4", "FKBP11", "DERL3",
    "TNFRSF17", "SLAMF7", "CD38", "CD27", "SEC11C"
  ),
  antibody_secretion_UPR = c(
    "XBP1", "HSPA5", "HSP90B1", "HERPUD1", "ATF4", "ATF6", "DDIT3", "SEL1L", "DNAJB9",
    "PPIB", "CALR", "ERP44", "DERL3", "SEC61A1", "SSR4"
  ),
  IgG_inflammatory_plasma_cell = c(
    "IGHG1", "IGHG2", "IGHG3", "IGHG4", "JCHAIN", "MZB1", "XBP1", "SDC1", "PRDM1", "IRF4", "CXCR4"
  ),
  cell_cycle_proliferating_B_cell = c(
    "MKI67", "TOP2A", "STMN1", "TYMS", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6",
    "MCM7", "HMGB2", "CENPF", "UBE2C", "PCLAF"
  )
)
modules_noig <- lapply(modules, function(x) x[!grepl("^IGH", x)])

valid <- function(x) {
  !is.na(x) & nzchar(trimws(as.character(x))) & toupper(trimws(as.character(x))) != "NA"
}
core_gene <- function(x) sub("\\*.*$", "", as.character(x))

message("Loading B-cell object")
obj <- readRDS(object_path)
md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
md$cell_id <- rownames(md)

paired <- as.logical(md$BCR_Paired_Chains) &
  valid(md$PatientID) & valid(md$AnnotationLevel3) &
  valid(md$BCR_Heavy_V_gene_Dominant) & valid(md$BCR_Heavy_J_gene_Dominant) &
  valid(md$BCR_Heavy_CDR3_Translation_Dominant) &
  valid(md$BCR_Light_V_gene_Dominant) & valid(md$BCR_Light_J_gene_Dominant) &
  valid(md$BCR_Light_CDR3_Translation_Dominant)

md$exact_paired_clone <- NA_character_
md$exact_paired_clone[paired] <- paste(
  md$PatientID[paired],
  md$BCR_Heavy_V_gene_Dominant[paired], md$BCR_Heavy_J_gene_Dominant[paired],
  md$BCR_Heavy_CDR3_Translation_Dominant[paired],
  md$BCR_Light_V_gene_Dominant[paired], md$BCR_Light_J_gene_Dominant[paired],
  md$BCR_Light_CDR3_Translation_Dominant[paired], sep = "|"
)
sizes <- md %>% filter(paired) %>% count(exact_paired_clone, name = "clone_size")
md <- md %>% left_join(sizes, by = "exact_paired_clone") %>% mutate(
  paired_bcr = paired,
  clone_status = case_when(
    paired_bcr & clone_size == 1L ~ "Singleton",
    paired_bcr & clone_size >= 2L ~ "Expanded",
    TRUE ~ NA_character_
  ),
  clone_bin = case_when(
    paired_bcr & clone_size == 1L ~ "Singleton",
    paired_bcr & clone_size == 2L ~ "2 cells",
    paired_bcr & clone_size %in% 3:4 ~ "3-4 cells",
    paired_bcr & clone_size >= 5L ~ ">=5 cells",
    TRUE ~ NA_character_
  ),
  receptor_key = ifelse(
    paired_bcr & valid(BCR_Heavy_CDR3_Nucleotide_Dominant),
    paste(
      PatientID, core_gene(BCR_Heavy_V_gene_Dominant), core_gene(BCR_Heavy_J_gene_Dominant),
      toupper(BCR_Heavy_CDR3_Nucleotide_Dominant), sep = "|"
    ),
    NA_character_
  ),
  state = as.character(AnnotationLevel3),
  isotype = case_when(
    grepl("^IGHA", BCR_Heavy_C_gene_Dominant) ~ "IgA",
    grepl("^IGHG", BCR_Heavy_C_gene_Dominant) ~ "IgG",
    grepl("^IGHM", BCR_Heavy_C_gene_Dominant) ~ "IgM",
    grepl("^IGHD", BCR_Heavy_C_gene_Dominant) ~ "IgD",
    grepl("^IGHE", BCR_Heavy_C_gene_Dominant) ~ "IgE",
    TRUE ~ NA_character_
  ),
  switched = ifelse(isotype %in% c("IgA", "IgG", "IgE"), 1,
                    ifelse(isotype %in% c("IgM", "IgD"), 0, NA_real_))
)

reference_ids <- read_csv(reference_path, show_col_types = FALSE, col_select = cell_id)$cell_id
use_ids <- union(reference_ids, md$cell_id[md$paired_bcr])
use <- md %>% filter(cell_id %in% use_ids)

message("Scoring five prespecified B-cell programs")
expr <- LayerData(obj[["RNA"]], layer = "data")
expr <- expr[, use$cell_id, drop = FALSE]
scores <- vapply(modules, function(gs) {
  genes <- intersect(gs, rownames(expr))
  if (length(genes) < 3L) return(rep(NA_real_, ncol(expr)))
  Matrix::colMeans(expr[genes, , drop = FALSE])
}, numeric(ncol(expr)))
scores <- as.data.frame(scores, check.names = FALSE)
scores$cell_id <- colnames(expr)
scores_noig <- vapply(modules_noig, function(gs) {
  genes <- intersect(gs, rownames(expr))
  if (length(genes) < 3L) return(rep(NA_real_, ncol(expr)))
  Matrix::colMeans(expr[genes, , drop = FALSE])
}, numeric(ncol(expr)))
colnames(scores_noig) <- paste0(colnames(scores_noig), "_noIG")
scores_noig <- as.data.frame(scores_noig, check.names = FALSE)
scores_noig$cell_id <- colnames(expr)

latent <- Embeddings(obj, "SCVI_50")[use$cell_id, seq_len(10), drop = FALSE]
colnames(latent) <- paste0("scvi_", seq_len(ncol(latent)))

batch_value <- if ("Batch" %in% colnames(use)) as.character(use$Batch) else NA_character_
sample_value <- if ("SampleID" %in% colnames(use)) as.character(use$SampleID) else as.character(use$PatientID)

output <- use %>% transmute(
  cell_id,
  PatientID = as.character(PatientID),
  SampleID = sample_value,
  Diagnosis1 = as.character(Diagnosis1),
  Batch = batch_value,
  acquisition_series = sub("[AB]$", "", batch_value),
  state,
  paired_bcr,
  exact_paired_clone,
  clone_size,
  clone_status,
  clone_bin,
  receptor_key,
  isotype,
  switched
) %>% left_join(scores, by = "cell_id") %>%
  left_join(scores_noig, by = "cell_id") %>%
  bind_cols(as.data.frame(latent))

write_csv(output, gzfile(file.path(outdir, "Table_BEM1_cell_program_latent_input.csv.gz")))
write_csv(tibble(
  module = names(modules),
  genes = vapply(modules, paste, collapse = ";", FUN.VALUE = character(1)),
  genes_without_immunoglobulin_constants = vapply(
    modules, function(x) paste(x[!grepl("^IGH", x)], collapse = ";"), FUN.VALUE = character(1)
  )
), file.path(outdir, "Table_BEM0_module_definitions.csv"))
write_csv(tibble(
  metric = c("object_cells", "trajectory_reference_cells", "exact_paired_cells", "exact_paired_clonotypes", "participants"),
  value = c(nrow(md), length(reference_ids), sum(md$paired_bcr), n_distinct(md$exact_paired_clone[md$paired_bcr]), n_distinct(md$PatientID[md$paired_bcr]))
), file.path(outdir, "Table_BEM0_input_manifest.csv"))

message("Wrote BCR expansion-maturation inputs to ", outdir)

#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(readr)
})

root <- "C:/path/to/private-manuscript-workspace"
input <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds"
out_dir <- file.path(root, "High Impact Additional Analyses", "CD8 TCR Trajectory")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

stopifnot(file.exists(input))
obj <- readRDS(input)
md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
md$cell <- rownames(md)

excluded_states <- c(
  "MAIT", "gdT", "CD8 Proliferative", "CD8 Naive-IFN", "CD8 Tmem KLRC2+"
)
keep <- !is.na(md$AnnotationLevel2) &
  nzchar(md$AnnotationLevel2) &
  !md$AnnotationLevel2 %in% excluded_states
md <- md[keep, , drop = FALSE]

valid <- function(x) {
  !is.na(x) & nzchar(trimws(as.character(x))) &
    toupper(trimws(as.character(x))) != "NA"
}

paired_ab <-
  grepl("^TRAC", md$TCR_Alpha_Gamma_C_gene_Dominant) &
  grepl("^TRBC", md$TCR_Beta_Delta_C_gene_Dominant) &
  valid(md$TCR_Alpha_Gamma_V_gene_Dominant) &
  valid(md$TCR_Alpha_Gamma_J_gene_Dominant) &
  valid(md$TCR_Alpha_Gamma_CDR3_Translation_Dominant) &
  valid(md$TCR_Beta_Delta_V_gene_Dominant) &
  valid(md$TCR_Beta_Delta_J_gene_Dominant) &
  valid(md$TCR_Beta_Delta_CDR3_Translation_Dominant) &
  valid(md$SampleID)

md$paired_alpha_beta <- paired_ab
md$clone_id <- NA_character_
md$clone_id[paired_ab] <- paste(
  md$TCR_Alpha_Gamma_V_gene_Dominant[paired_ab],
  md$TCR_Alpha_Gamma_J_gene_Dominant[paired_ab],
  md$TCR_Alpha_Gamma_CDR3_Translation_Dominant[paired_ab],
  md$TCR_Beta_Delta_V_gene_Dominant[paired_ab],
  md$TCR_Beta_Delta_J_gene_Dominant[paired_ab],
  md$TCR_Beta_Delta_CDR3_Translation_Dominant[paired_ab],
  sep = "|"
)

clone_sizes <- md %>%
  filter(paired_alpha_beta) %>%
  count(SampleID, clone_id, name = "clone_size_cd8")
md <- md %>%
  left_join(clone_sizes, by = c("SampleID", "clone_id")) %>%
  mutate(
    clone_size_cd8 = if_else(paired_alpha_beta, as.integer(clone_size_cd8), NA_integer_),
    clone_status = case_when(
      !paired_alpha_beta ~ "No paired alpha-beta",
      clone_size_cd8 == 1L ~ "Singleton",
      clone_size_cd8 >= 2L ~ "Expanded",
      TRUE ~ NA_character_
    ),
    clone_bin = case_when(
      !paired_alpha_beta ~ "No paired alpha-beta",
      clone_size_cd8 == 1L ~ "1",
      clone_size_cd8 == 2L ~ "2",
      clone_size_cd8 %in% 3:4 ~ "3-4",
      clone_size_cd8 >= 5L ~ ">=5",
      TRUE ~ NA_character_
    )
  )

modules <- list(
  EOMES_ZEB2 = c(
    "EOMES", "ZEB2", "GZMB", "GZMH", "PRF1", "NKG7", "CX3CR1", "KLRG1",
    "TBX21", "CCL5", "CST7", "FGFBP2"
  ),
  Cytotoxicity = c(
    "NKG7", "GNLY", "PRF1", "GZMB", "GZMA", "GZMH", "GZMK", "GZMM", "CTSW",
    "CST7", "FGFBP2", "CCL5", "CCL4", "CCL3", "IFNG", "FASLG", "KLRD1", "KLRG1"
  ),
  Th1_Tc1 = c(
    "TBX21", "STAT4", "CXCR3", "IFNG", "TNF", "IL12RB2", "CCL5", "CCL4", "NKG7",
    "GZMB", "PRF1", "CXCR6", "BHLHE40"
  ),
  GZMK_inflammatory_memory = c(
    "GZMK", "GZMA", "CCL5", "CCL4", "CCL4L2", "XCL1", "XCL2", "NKG7",
    "DUSP2", "CRTAM", "EOMES", "CXCR3", "IL7R"
  ),
  Early_memory = c("TCF7", "CCR7", "LEF1", "MAL", "LTB", "IL7R", "NOSIP"),
  Late_effector = c("GNLY", "NKG7", "GZMB", "PRF1", "CCL5", "FGFBP2", "CX3CR1")
)

expr <- LayerData(obj[["RNA"]], layer = "data")
expr <- expr[, md$cell, drop = FALSE]
scores <- vapply(modules, function(gs) {
  present <- intersect(gs, rownames(expr))
  if (!length(present)) return(rep(NA_real_, ncol(expr)))
  Matrix::colMeans(expr[present, , drop = FALSE])
}, numeric(ncol(expr)))
scores <- as.data.frame(scores, check.names = FALSE)
scores$cell <- colnames(expr)

embedding <- Embeddings(obj, "SCVI_50E400")[md$cell, 1:20, drop = FALSE]
colnames(embedding) <- paste0("scvi_", seq_len(ncol(embedding)))
umap <- Embeddings(obj, "umap")[md$cell, , drop = FALSE]
colnames(umap) <- c("UMAP_1", "UMAP_2")

output <- md %>%
  transmute(
    cell,
    SampleID = as.character(SampleID),
    Diagnosis1 = as.character(Diagnosis1),
    Batch = as.character(Batch),
    Biologic = as.character(Biologic),
    state = as.character(AnnotationLevel2),
    paired_alpha_beta,
    clone_id,
    clone_size_cd8,
    clone_status,
    clone_bin
  ) %>%
  left_join(scores, by = "cell") %>%
  bind_cols(as.data.frame(embedding), as.data.frame(umap))

write_csv(output, file.path(out_dir, "Table_TJ1_cd8_trajectory_input.csv.gz"))

manifest <- tibble(
  metric = c(
    "All conventional CD8 cells", "Paired alpha-beta cells",
    "Participants", "Expanded paired alpha-beta clonotypes"
  ),
  value = c(
    nrow(output), sum(output$paired_alpha_beta), n_distinct(output$SampleID),
    output %>% filter(paired_alpha_beta, clone_size_cd8 >= 2L) %>%
      distinct(SampleID, clone_id) %>% nrow()
  )
)
write_csv(manifest, file.path(out_dir, "Table_TJ0_input_manifest.csv"))

message("Wrote trajectory inputs to: ", out_dir)

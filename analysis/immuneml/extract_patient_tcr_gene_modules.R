#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(dplyr)
  library(readr)
})

source_root <- "C:/path/to/private-user-home/OneDrive/Desktop/TCR Module Scores"
output_dir <- "outputs/gene_module_microbiome_integration_no_age_sex_20260801/data"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

objects <- c(
  TCR_CD4 = file.path(source_root, "CD4T2026_scvi50_umap.rds"),
  TCR_CD8 = file.path(source_root, "CD8T2026_scvi50_epoch400_umap.rds")
)

module_genes <- list(
  Naive_central_memory = c("CCR7", "SELL", "TCF7", "LEF1", "IL7R", "LTB", "MAL", "NOSIP", "SATB1", "BACH2", "KLF2", "S1PR1", "CD27", "CD28", "BCL2"),
  Recent_TCR_stimulation_immediate_early = c("NR4A1", "NR4A2", "NR4A3", "EGR1", "EGR2", "EGR3", "FOS", "JUN", "JUNB", "DUSP1", "DUSP2", "DUSP4", "DUSP5", "CD69", "IL2RA", "NFKBIA", "NFKBIZ", "REL", "IRF4", "BATF"),
  Effector_cytotoxicity = c("NKG7", "GNLY", "PRF1", "GZMB", "GZMA", "GZMH", "GZMK", "GZMM", "CTSW", "CST7", "FGFBP2", "CCL5", "CCL4", "CCL3", "IFNG", "FASLG", "KLRD1", "KLRG1"),
  GZMK_inflammatory_memory_T = c("GZMK", "GZMA", "CCL5", "CCL4", "CCL4L2", "XCL1", "XCL2", "NKG7", "DUSP2", "CRTAM", "EOMES", "CXCR3", "IL7R"),
  Th1_Tc1_inflammatory = c("TBX21", "STAT4", "CXCR3", "IFNG", "TNF", "IL12RB2", "CCL5", "CCL4", "NKG7", "GZMB", "PRF1", "CXCR6", "BHLHE40"),
  Th17_Tc17_IL23_axis = c("RORC", "CCR6", "IL23R", "IL17A", "IL17F", "IL22", "IL26", "KLRB1", "AHR", "CCL20", "IL1R1", "IL21", "CXCR6", "LTB"),
  Chronic_stimulation_exhaustion_like = c("PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "TOX2", "ENTPD1", "SLAMF6", "TNFRSF9", "BATF", "EOMES", "PRDM1", "LAYN", "CXCL13"),
  Tissue_resident_memory_mucosal_retention = c("CD69", "ITGAE", "ITGA1", "CXCR6", "ZNF683", "RUNX3", "PRDM1", "RGS1", "CD101", "DUSP6", "AHR", "CCR6"),
  EOMES_ZEB2_inflammatory_CD8_TRM_like = c("EOMES", "ZEB2", "GZMB", "GZMH", "PRF1", "NKG7", "CX3CR1", "KLRG1", "TBX21", "CCL5", "CST7", "FGFBP2"),
  Gut_homing_intestinal_trafficking = c("ITGA4", "ITGB7", "ITGAE", "CCR9", "CCR6", "CXCR3", "CXCR6", "SELPLG", "S1PR1", "KLF2", "SELL", "GPR183", "CD69"),
  Tph_Tfh_like_B_cell_help = c("CXCL13", "PDCD1", "ICOS", "MAF", "TOX2", "IL21", "CD40LG", "SLAMF6", "TIGIT", "CD200", "CXCR5", "BCL6", "SH2D1A"),
  Activated_Treg_suppressive_T_cell = c("FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2", "IKZF4", "TNFRSF18", "TNFRSF4", "BATF", "CCR8", "LAYN", "ENTPD1", "IL10", "AREG"),
  TNF_NFkB_inflammatory_activation = c("TNF", "NFKBIA", "NFKBIZ", "TNFAIP3", "REL", "RELB", "BIRC3", "ICOS", "CD40LG", "CCL3", "CCL4", "JUNB", "DUSP1"),
  Type_I_II_interferon_response = c("IFI6", "IFI27", "IFI44L", "ISG15", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "IFIT1", "IFIT2", "IFIT3", "RSAD2", "STAT1", "IRF7", "ISG20", "CXCL10"),
  MAIT_like_unconventional_T_cell = c("SLC4A10", "KLRB1", "ZBTB16", "RORC", "IL18RAP", "CCR6", "DPP4", "CXCR6", "IL7R", "GZMK", "IFNG", "NKG7"),
  Cell_cycle_clonal_proliferation = c("MKI67", "TOP2A", "STMN1", "TYMS", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "HMGB2", "CENPF", "UBE2C", "PCLAF"),
  Stress_dissociation_response = c("FOS", "JUN", "JUNB", "ATF3", "DUSP1", "HSPA1A", "HSPA1B", "HSPH1", "DNAJB1", "HSP90AA1", "PPP1R15A", "IER2")
)

pick_sample_column <- function(md) {
  candidates <- c("SampleID", "Sample_Name", "Sample_Tag", "sample_id", "subject_id", "PatientID")
  for (candidate in candidates) {
    if (candidate %in% colnames(md)) {
      values <- trimws(as.character(md[[candidate]]))
      if (sum(!is.na(values) & values != "") > 0L) return(candidate)
    }
  }
  stop("No patient/sample identifier found in Seurat metadata")
}

extract_one <- function(compartment, path) {
  message("Loading ", compartment, ": ", path)
  seu <- readRDS(path)
  md <- seu@meta.data
  sample_col <- pick_sample_column(md)
  sample_id <- trimws(as.character(md[[sample_col]]))
  sample_id <- sub("_.*$", "", sample_id)

  assay <- DefaultAssay(seu)
  expr <- tryCatch(
    GetAssayData(seu, assay = assay, layer = "data"),
    error = function(e) GetAssayData(seu, assay = assay, slot = "data")
  )
  common_cells <- intersect(rownames(md), colnames(expr))
  md <- md[common_cells, , drop = FALSE]
  expr <- expr[, common_cells, drop = FALSE]
  sample_id <- sample_id[match(common_cells, rownames(seu@meta.data))]

  score_list <- lapply(module_genes, function(genes) {
    keep <- intersect(genes, rownames(expr))
    if (length(keep) == 0L) return(rep(NA_real_, ncol(expr)))
    as.numeric(Matrix::colMeans(expr[keep, , drop = FALSE]))
  })
  score_df <- as.data.frame(score_list, stringsAsFactors = FALSE)
  names(score_df) <- paste0("module_", names(module_genes))
  score_df$PatientID <- sample_id
  score_df <- score_df[!is.na(score_df$PatientID) & grepl("^P[0-9]+$", score_df$PatientID), , drop = FALSE]

  module_cols <- grep("^module_", names(score_df), value = TRUE)
  patient <- score_df %>%
    group_by(PatientID) %>%
    summarise(n_cells = n(), across(all_of(module_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
  patient$compartment <- compartment

  present <- vapply(module_genes, function(g) sum(g %in% rownames(expr)), integer(1))
  qc <- tibble(
    compartment = compartment,
    assay = assay,
    sample_column = sample_col,
    module = names(module_genes),
    n_genes_defined = lengths(module_genes),
    n_genes_present = present
  )
  rm(seu, expr, md, score_df)
  invisible(gc())
  list(patient = patient, qc = qc)
}

res <- lapply(names(objects), function(name) extract_one(name, objects[[name]]))
patient_long <- bind_rows(lapply(res, `[[`, "patient"))
qc <- bind_rows(lapply(res, `[[`, "qc"))

write_csv(patient_long, file.path(output_dir, "tcr_patient_gene_module_scores_long_compartment.csv"))
write_csv(qc, file.path(output_dir, "tcr_gene_module_extraction_qc.csv"))

for (compartment_name in names(objects)) {
  out <- patient_long %>% filter(.data$compartment == .env$compartment_name) %>% select(-compartment)
  write_csv(out, file.path(output_dir, paste0(tolower(compartment_name), "_patient_gene_module_scores.csv")))
}

message("Patient-level TCR gene-module extraction complete")

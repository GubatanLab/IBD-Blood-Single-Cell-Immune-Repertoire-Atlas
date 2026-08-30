options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))

input <- "C:/path/to/private-legacy-manuscript-assets/Figure 1/UMAP/PBMC2026_scvi50_umap.rds"
out_dir <- file.path(getwd(), "Cell Press Redrawn Figure Set", "Source Data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(input)
md <- obj@meta.data

chr <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  trimws(x)
}

nonblank <- function(x) {
  x <- chr(x)
  nzchar(x) & !tolower(x) %in% c("na", "nan", "none", "null")
}

first_nonblank <- function(x) {
  x <- chr(x)
  hit <- x[nonblank(x)]
  if (length(hit)) hit[[1]] else ""
}

patient_id <- chr(md$PatientID)
patient_id[!nonblank(patient_id)] <- chr(md$SampleID[!nonblank(patient_id)])
keep_patient <- nonblank(patient_id)

participant <- do.call(rbind, lapply(split(seq_len(nrow(md))[keep_patient], patient_id[keep_patient]), function(ix) {
  data.frame(
    PatientID = patient_id[ix[[1]]],
    Diagnosis1 = first_nonblank(md$Diagnosis1[ix]),
    Inflammation1 = first_nonblank(md$Inflammation1[ix]),
    Biologic = first_nonblank(md$Biologic[ix]),
    Batch = first_nonblank(md$Batch[ix]),
    stringsAsFactors = FALSE
  )
}))
participant$acquisition_series <- sub("[AB]$", "", participant$Batch)
participant$biologic_exposed <- !tolower(participant$Biologic) %in% c("", "none", "no")
write.csv(participant, file.path(out_dir, "Figure1_participant_summary.csv"), row.names = FALSE)

ann1 <- chr(md$AnnotationLevel1)
ann2 <- chr(md$AnnotationLevel2)
is_b <- grepl("B Cell|Plasma B|memory B|Naive B|Transitional B", ann2, ignore.case = TRUE) |
  grepl("^B( Cell)?$|Plasma", ann1, ignore.case = TRUE)
is_t <- grepl("^(CD4|CD8|TReg|MAIT|gdT)", ann2, ignore.case = TRUE) |
  grepl("^T( Cell)?$", ann1, ignore.case = TRUE)
is_t <- is_t & !is_b

ag_v <- chr(md$TCR_Alpha_Gamma_V_gene_Dominant)
ag_cdr3 <- chr(md$TCR_Alpha_Gamma_CDR3_Translation_Dominant)
bd_v <- chr(md$TCR_Beta_Delta_V_gene_Dominant)
bd_cdr3 <- chr(md$TCR_Beta_Delta_CDR3_Translation_Dominant)
heavy_cdr3 <- chr(md$BCR_Heavy_CDR3_Translation_Dominant)
light_cdr3 <- chr(md$BCR_Light_CDR3_Translation_Dominant)

has_alpha <- nonblank(ag_cdr3) & grepl("^TRA", ag_v)
has_gamma <- nonblank(ag_cdr3) & grepl("^TRG", ag_v)
has_beta <- nonblank(bd_cdr3) & grepl("^TRB", bd_v)
has_delta <- nonblank(bd_cdr3) & grepl("^TRD", bd_v)
has_heavy <- nonblank(heavy_cdr3)
has_light <- nonblank(light_cdr3)

productive_tcr <- has_alpha | has_beta | has_gamma | has_delta
paired_ab <- has_alpha & has_beta
paired_gd <- has_gamma & has_delta
productive_bcr <- has_heavy | has_light
paired_hl <- has_heavy & has_light

participant_count <- function(flag) length(unique(patient_id[flag & keep_patient]))
qc <- rbind(
  data.frame(compartment = "TCR", metric = "Quality-controlled T cells", numerator = sum(is_t), denominator = sum(is_t), participants = participant_count(is_t)),
  data.frame(compartment = "TCR", metric = "Productive TCR", numerator = sum(is_t & productive_tcr), denominator = sum(is_t), participants = participant_count(is_t & productive_tcr)),
  data.frame(compartment = "TCR", metric = "Paired alpha-beta", numerator = sum(is_t & paired_ab), denominator = sum(is_t), participants = participant_count(is_t & paired_ab)),
  data.frame(compartment = "TCR", metric = "Paired gamma-delta", numerator = sum(is_t & paired_gd), denominator = sum(is_t), participants = participant_count(is_t & paired_gd)),
  data.frame(compartment = "BCR", metric = "Quality-controlled B cells", numerator = sum(is_b), denominator = sum(is_b), participants = participant_count(is_b)),
  data.frame(compartment = "BCR", metric = "Productive BCR", numerator = sum(is_b & productive_bcr), denominator = sum(is_b), participants = participant_count(is_b & productive_bcr)),
  data.frame(compartment = "BCR", metric = "Paired heavy-light", numerator = sum(is_b & paired_hl), denominator = sum(is_b), participants = participant_count(is_b & paired_hl))
)
qc$percent <- 100 * qc$numerator / qc$denominator
write.csv(qc, file.path(out_dir, "Figure1_receptor_qc_summary.csv"), row.names = FALSE)

# Participant is the biological unit for all quantitative receptor-recovery
# summaries shown in the main figure. Participants without cells in the
# corresponding lineage are retained with an undefined denominator and are not
# plotted for that metric.
recovery_flags <- data.frame(
  t_cells = as.integer(is_t),
  productive_tcr_n = as.integer(is_t & productive_tcr),
  paired_alpha_beta_n = as.integer(is_t & paired_ab),
  paired_gamma_delta_n = as.integer(is_t & paired_gd),
  b_cells = as.integer(is_b),
  productive_bcr_n = as.integer(is_b & productive_bcr),
  paired_heavy_light_n = as.integer(is_b & paired_hl)
)
recovery_matrix <- rowsum(
  as.matrix(recovery_flags[keep_patient, , drop = FALSE]),
  group = patient_id[keep_patient],
  reorder = FALSE
)
recovery <- data.frame(PatientID = rownames(recovery_matrix), recovery_matrix, row.names = NULL)
recovery <- merge(recovery, participant, by = "PatientID", all.x = TRUE, sort = FALSE)
safe_pct <- function(numerator, denominator) ifelse(denominator > 0, 100 * numerator / denominator, NA_real_)
recovery$productive_tcr_pct <- safe_pct(recovery$productive_tcr_n, recovery$t_cells)
recovery$paired_alpha_beta_pct <- safe_pct(recovery$paired_alpha_beta_n, recovery$t_cells)
recovery$paired_gamma_delta_pct <- safe_pct(recovery$paired_gamma_delta_n, recovery$t_cells)
recovery$productive_bcr_pct <- safe_pct(recovery$productive_bcr_n, recovery$b_cells)
recovery$paired_heavy_light_pct <- safe_pct(recovery$paired_heavy_light_n, recovery$b_cells)
recovery$repertoire_evaluable <- recovery$productive_tcr_n > 0 | recovery$productive_bcr_n > 0
write.csv(recovery, file.path(out_dir, "Figure1_participant_receptor_recovery.csv"), row.names = FALSE)

major_lineage <- vapply(ann2, function(s) {
  s_lower <- tolower(s)
  if (grepl("plasma", s_lower)) return("Plasma cells")
  if (grepl(" b", s_lower) || grepl("^b ", s_lower) || grepl("memory b|naive b|transitional b", s_lower)) return("B cells")
  if (grepl("treg", s_lower)) return("Treg")
  if (grepl("mait|gdt", s_lower)) return("Unconventional T")
  if (grepl("cd8", s_lower)) return("CD8 T")
  if (grepl("cd4|tfh", s_lower)) return("CD4 T")
  "Other"
}, character(1))
major_keep <- major_lineage != "Other"
major_summary <- do.call(rbind, lapply(split(seq_len(nrow(md))[major_keep], major_lineage[major_keep]), function(ix) {
  data.frame(
    major_lineage = major_lineage[ix[[1]]],
    total_cells = length(ix),
    participants = length(unique(patient_id[ix][keep_patient[ix]])),
    stringsAsFactors = FALSE
  )
}))
write.csv(major_summary, file.path(out_dir, "Figure1_major_lineage_summary.csv"), row.names = FALSE)

state <- ann2
valid_state <- nonblank(state) & (is_t | is_b)
coverage <- do.call(rbind, lapply(split(seq_len(nrow(md))[valid_state], state[valid_state]), function(ix) {
  data.frame(
    cell_state = state[ix[[1]]],
    lineage = if (mean(is_b[ix]) > 0.5) "B cell" else "T cell",
    total_cells = length(ix),
    paired_alpha_beta_n = sum(paired_ab[ix]),
    paired_gamma_delta_n = sum(paired_gd[ix]),
    paired_heavy_light_n = sum(paired_hl[ix]),
    paired_alpha_beta_pct = 100 * mean(paired_ab[ix]),
    paired_gamma_delta_pct = 100 * mean(paired_gd[ix]),
    paired_heavy_light_pct = 100 * mean(paired_hl[ix]),
    participants = length(unique(patient_id[ix][keep_patient[ix]])),
    stringsAsFactors = FALSE
  )
}))
write.csv(coverage, file.path(out_dir, "Figure1_receptor_coverage_by_state.csv"), row.names = FALSE)

# Canonical marker summaries use the complete lineage-resolved objects rather
# than the downsampled coordinates used for display.
lineage_inputs <- c(
  CD4 = "C:/path/to/private-legacy-manuscript-assets/Figure 1/UMAP/CD4T2026_scvi50_umap.rds",
  CD8 = "C:/path/to/private-legacy-manuscript-assets/Figure 1/UMAP/CD8T2026_scvi50_epoch400_umap.rds",
  B = "C:/path/to/private-legacy-manuscript-assets/Figure 1/UMAP/BCell2026_scvi50_umap.rds"
)
selected_states <- list(
  CD4 = c("CD4 Naive", "CD4 Tfh", "CD4 Th17", "TReg Cytotoxic", "CD4 Temra"),
  CD8 = c("CD8 Naive", "CD8 Tcm CCR4-", "CD8 HLA-DR+", "CD8 Temra", "CD8 Proliferative"),
  B = c("Naive B", "Switched memory B", "Atypical memory B", "IgA Plasma B Cell", "IgG Plasma B Cell")
)
marker_sets <- list(
  `CD4/Treg identity` = c("CCR7", "LTB", "IL7R", "CXCR5", "PDCD1", "RORA", "KLRB1", "FOXP3", "CTLA4"),
  `Effector/activation` = c("GZMK", "NKG7", "CCL5", "GZMB", "PRF1", "HLA-DRA", "MKI67"),
  `B/plasma identity` = c("MS4A1", "CD79A", "TCL1A", "IGHD", "CD27", "FCRL5", "TBX21", "MZB1", "JCHAIN", "XBP1")
)
markers <- unlist(marker_sets, use.names = FALSE)

select_data_assay <- function(x) {
  preferred <- intersect(c("RNA", "SCT", "Corrected"), Assays(x))
  candidates <- unique(c(preferred, Assays(x)))
  for (assay in candidates) {
    if ("data" %in% Layers(x[[assay]])) return(assay)
  }
  stop("No normalized data layer found in lineage object")
}

marker_rows <- list()
panel_rows <- list()
for (lineage_name in names(lineage_inputs)) {
  lineage_obj <- readRDS(lineage_inputs[[lineage_name]])
  lineage_md <- lineage_obj@meta.data
  state_col <- if ("AnnotationLevel2" %in% colnames(lineage_md)) "AnnotationLevel2" else stop("AnnotationLevel2 missing")
  lineage_patient <- chr(lineage_md$PatientID)
  if ("SampleID" %in% colnames(lineage_md)) {
    lineage_patient[!nonblank(lineage_patient)] <- chr(lineage_md$SampleID[!nonblank(lineage_patient)])
  }
  panel_rows[[lineage_name]] <- data.frame(
    panel = lineage_name,
    total_cells = ncol(lineage_obj),
    participants = length(unique(lineage_patient[nonblank(lineage_patient)])),
    stringsAsFactors = FALSE
  )
  assay <- select_data_assay(lineage_obj)
  expr <- GetAssayData(lineage_obj, assay = assay, layer = "data")
  available <- intersect(markers, rownames(expr))
  state_values <- chr(lineage_md[[state_col]])
  for (cell_state in selected_states[[lineage_name]]) {
    ix <- which(state_values == cell_state)
    if (!length(ix)) next
    state_avg <- Matrix::rowMeans(expr[available, ix, drop = FALSE])
    state_pct <- 100 * Matrix::rowMeans(expr[available, ix, drop = FALSE] > 0)
    marker_rows[[paste(lineage_name, cell_state, sep = "|")]] <- data.frame(
      lineage = lineage_name,
      cell_state = cell_state,
      gene = available,
      avg_expression = as.numeric(state_avg),
      pct_expressing = as.numeric(state_pct),
      state_cells = length(ix),
      state_participants = length(unique(lineage_patient[ix][nonblank(lineage_patient[ix])])),
      stringsAsFactors = FALSE
    )
  }
  rm(lineage_obj, lineage_md, expr)
  invisible(gc())
}
marker_summary <- do.call(rbind, marker_rows)
scale_gene <- function(x) {
  if (length(unique(x[is.finite(x)])) < 2) return(rep(0, length(x)))
  as.numeric(scale(x))
}
marker_summary$avg_expression_scaled <- ave(marker_summary$avg_expression, marker_summary$gene, FUN = scale_gene)
marker_summary$avg_expression_scaled <- pmax(-2.5, pmin(2.5, marker_summary$avg_expression_scaled))
marker_summary$marker_group <- unname(vapply(marker_summary$gene, function(g) {
  names(marker_sets)[which(vapply(marker_sets, function(x) g %in% x, logical(1)))[1]]
}, character(1)))
write.csv(marker_summary, file.path(out_dir, "Figure1_canonical_marker_dotplot.csv"), row.names = FALSE)
write.csv(do.call(rbind, panel_rows), file.path(out_dir, "Figure1_lineage_panel_summary.csv"), row.names = FALSE)

cat("participants:", nrow(participant), "\n")
print(table(participant$Diagnosis1, useNA = "ifany"))
print(qc)

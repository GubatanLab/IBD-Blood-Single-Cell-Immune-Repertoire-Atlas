options(stringsAsFactors = FALSE)

source_dir <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD Immune Repertoire Manuscript/Figure 3 TCR Repertoire Architecture"
seurat_lib <- "C:/path/to/private-user-home/OneDrive/Desktop/PBMC2026 Seurat Objects/R-4.5.3/library"
.libPaths(c(seurat_lib, .libPaths()))
suppressPackageStartupMessages(library(SeuratObject))

output_dir <- file.path(getwd(), "TRBV9 state analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

clean <- function(x) {
  x <- trimws(as.character(x))
  x[x %in% c("", "NA", "None", "none", "null", "NULL")] <- NA_character_
  x
}

strip_allele <- function(x) sub("\\*[0-9]+$", "", clean(x))

extract_meta <- function(filename, lineage) {
  obj <- readRDS(file.path(source_dir, filename))
  md <- obj@meta.data
  needed <- c(
    "PatientID", "SampleID", "Diagnosis1", "Inflammation1", "AnnotationLevel2",
    "TCR_Beta_Delta_V_gene_Dominant", "TCR_Beta_Delta_J_gene_Dominant",
    "TCR_Beta_Delta_CDR3_Translation_Dominant"
  )
  stopifnot(all(needed %in% names(md)))
  out <- data.frame(
    cell_id = rownames(md),
    lineage = lineage,
    PatientID = clean(md$PatientID),
    SampleID = clean(md$SampleID),
    Diagnosis = clean(md$Diagnosis1),
    Inflammation = clean(md$Inflammation1),
    T_cell_state = clean(md$AnnotationLevel2),
    TRBV = strip_allele(md$TCR_Beta_Delta_V_gene_Dominant),
    TRBJ = strip_allele(md$TCR_Beta_Delta_J_gene_Dominant),
    CDR3_aa = clean(md$TCR_Beta_Delta_CDR3_Translation_Dominant),
    stringsAsFactors = FALSE
  )
  rm(obj, md)
  gc(verbose = FALSE)
  out
}

cd4 <- extract_meta("CD4TCellRepertoire.rds", "CD4")
cd8 <- extract_meta("CD8TCellRepertoire.rds", "CD8")
stopifnot(length(intersect(cd4$cell_id, cd8$cell_id)) == 0L)
cells <- rbind(cd4, cd8)
rm(cd4, cd8)
gc(verbose = FALSE)

beta <- cells[
  !is.na(cells$PatientID) & !is.na(cells$T_cell_state) & !is.na(cells$TRBV) &
    grepl("^TRBV", cells$TRBV, ignore.case = TRUE) & !is.na(cells$CDR3_aa),
]
beta$is_TRBV9 <- grepl("^TRBV9$", beta$TRBV, ignore.case = TRUE)
beta$clonotype_id <- paste(beta$TRBV, beta$CDR3_aa, ifelse(is.na(beta$TRBJ), "TRBJ-NA", beta$TRBJ), sep = "|")
trbv9 <- beta[beta$is_TRBV9, ]
stopifnot(nrow(trbv9) > 0L)

write.csv(trbv9, file.path(output_dir, "TRBV9_cell_level_metadata.csv"), row.names = FALSE, na = "")

patient_clone_counts <- aggregate(
  cell_id ~ PatientID + Diagnosis + Inflammation + clonotype_id,
  trbv9, length
)
names(patient_clone_counts)[names(patient_clone_counts) == "cell_id"] <- "n_cells"
patient_clone_counts$clone_size_class <- cut(
  patient_clone_counts$n_cells,
  breaks = c(0, 1, 5, 20, 100, Inf),
  labels = c("Singleton", "Small (2-5)", "Medium (6-20)", "Large (21-100)", "Hyperexpanded (>100)"),
  right = TRUE
)
write.csv(patient_clone_counts, file.path(output_dir, "TRBV9_patient_clonotype_sizes.csv"), row.names = FALSE, na = "")

clone_cells <- aggregate(cell_id ~ clonotype_id, trbv9, length)
names(clone_cells)[2] <- "n_cells"
clone_patients <- aggregate(PatientID ~ clonotype_id, trbv9, function(x) length(unique(x)))
names(clone_patients)[2] <- "n_patients"
clone_states <- aggregate(T_cell_state ~ clonotype_id, trbv9, function(x) length(unique(x)))
names(clone_states)[2] <- "n_states"
clone_diag <- aggregate(Diagnosis ~ clonotype_id, trbv9, function(x) paste(sort(unique(x)), collapse = ";"))
clone_lineage <- aggregate(lineage ~ clonotype_id, trbv9, function(x) paste(sort(unique(x)), collapse = ";"))
clone_summary <- Reduce(function(x, y) merge(x, y, by = "clonotype_id", all = TRUE),
                        list(clone_cells, clone_patients, clone_states, clone_diag, clone_lineage))
clone_summary <- clone_summary[order(-clone_summary$n_cells, -clone_summary$n_patients, clone_summary$clonotype_id), ]
clone_summary$public_across_patients <- clone_summary$n_patients >= 2
write.csv(clone_summary, file.path(output_dir, "TRBV9_clonotype_summary.csv"), row.names = FALSE, na = "")

top_ids <- head(clone_summary$clonotype_id, 30)
top_state <- aggregate(cell_id ~ clonotype_id + T_cell_state, trbv9[trbv9$clonotype_id %in% top_ids, ], length)
names(top_state)[3] <- "n_cells"
top_totals <- aggregate(n_cells ~ clonotype_id, top_state, sum)
names(top_totals)[2] <- "clonotype_total_cells"
top_state <- merge(top_state, top_totals, by = "clonotype_id")
top_state$pct_clonotype_cells <- 100 * top_state$n_cells / top_state$clonotype_total_cells
top_state$rank <- match(top_state$clonotype_id, top_ids)
top_state <- top_state[order(top_state$rank, -top_state$n_cells), ]
write.csv(top_state, file.path(output_dir, "TRBV9_top30_clonotype_state_distribution.csv"), row.names = FALSE)

state_total <- aggregate(cell_id ~ T_cell_state, beta, length)
names(state_total)[2] <- "all_beta_cells"
state_trbv9_cells <- aggregate(cell_id ~ T_cell_state, trbv9, length)
names(state_trbv9_cells)[2] <- "trbv9_cells"
state_trbv9_clones <- aggregate(clonotype_id ~ T_cell_state, trbv9, function(x) length(unique(x)))
names(state_trbv9_clones)[2] <- "unique_trbv9_clonotypes"
state_patients <- aggregate(PatientID ~ T_cell_state, trbv9, function(x) length(unique(x)))
names(state_patients)[2] <- "patients_with_trbv9"
state_summary <- Reduce(function(x, y) merge(x, y, by = "T_cell_state", all = TRUE),
                        list(state_total, state_trbv9_cells, state_trbv9_clones, state_patients))
state_summary[is.na(state_summary)] <- 0
state_summary$pct_all_trbv9_cells <- 100 * state_summary$trbv9_cells / nrow(trbv9)
state_summary$trbv9_prevalence_pct <- 100 * state_summary$trbv9_cells / state_summary$all_beta_cells
state_summary <- state_summary[order(-state_summary$trbv9_cells), ]

state_diag_total <- aggregate(cell_id ~ T_cell_state + Diagnosis, beta, length)
names(state_diag_total)[3] <- "all_beta_cells"
state_diag_9 <- aggregate(cell_id ~ T_cell_state + Diagnosis, trbv9, length)
names(state_diag_9)[3] <- "trbv9_cells"
state_by_diagnosis <- merge(state_diag_total, state_diag_9, by = c("T_cell_state", "Diagnosis"), all.x = TRUE)
state_by_diagnosis$trbv9_cells[is.na(state_by_diagnosis$trbv9_cells)] <- 0
state_by_diagnosis$trbv9_prevalence_pct <- 100 * state_by_diagnosis$trbv9_cells / state_by_diagnosis$all_beta_cells
write.csv(state_by_diagnosis, file.path(output_dir, "TRBV9_state_by_diagnosis.csv"), row.names = FALSE)

patients <- sort(unique(beta$PatientID))
states <- sort(unique(beta$T_cell_state))
patient_state <- expand.grid(PatientID = patients, T_cell_state = states, stringsAsFactors = FALSE)
patient_info <- unique(beta[, c("PatientID", "Diagnosis", "Inflammation")])
patient_info <- patient_info[!duplicated(patient_info$PatientID), ]
patient_state <- merge(patient_state, patient_info, by = "PatientID", all.x = TRUE)

ps_total <- aggregate(cell_id ~ PatientID + T_cell_state, beta, length)
names(ps_total)[3] <- "state_beta_cells"
ps_9 <- aggregate(cell_id ~ PatientID + T_cell_state, trbv9, length)
names(ps_9)[3] <- "state_trbv9_cells"
patient_total <- aggregate(cell_id ~ PatientID, beta, length)
names(patient_total)[2] <- "patient_beta_cells"
patient_9 <- aggregate(cell_id ~ PatientID, trbv9, length)
names(patient_9)[2] <- "patient_trbv9_cells"
patient_state <- merge(patient_state, ps_total, by = c("PatientID", "T_cell_state"), all.x = TRUE)
patient_state <- merge(patient_state, ps_9, by = c("PatientID", "T_cell_state"), all.x = TRUE)
patient_state <- merge(patient_state, patient_total, by = "PatientID", all.x = TRUE)
patient_state <- merge(patient_state, patient_9, by = "PatientID", all.x = TRUE)
patient_state$state_beta_cells[is.na(patient_state$state_beta_cells)] <- 0
patient_state$state_trbv9_cells[is.na(patient_state$state_trbv9_cells)] <- 0
patient_state$patient_trbv9_cells[is.na(patient_state$patient_trbv9_cells)] <- 0
patient_state$rest_beta_cells <- patient_state$patient_beta_cells - patient_state$state_beta_cells
patient_state$rest_trbv9_cells <- patient_state$patient_trbv9_cells - patient_state$state_trbv9_cells
patient_state$state_trbv9_pct <- ifelse(patient_state$state_beta_cells > 0,
                                        100 * patient_state$state_trbv9_cells / patient_state$state_beta_cells, NA_real_)
patient_state$rest_trbv9_pct <- ifelse(patient_state$rest_beta_cells > 0,
                                       100 * patient_state$rest_trbv9_cells / patient_state$rest_beta_cells, NA_real_)
write.csv(patient_state, file.path(output_dir, "TRBV9_patient_state_prevalence.csv"), row.names = FALSE, na = "")

state_test <- function(state) {
  d <- patient_state[patient_state$T_cell_state == state & patient_state$state_beta_cells >= 5 &
                       patient_state$rest_beta_cells >= 20 & is.finite(patient_state$state_trbv9_pct) &
                       is.finite(patient_state$rest_trbv9_pct), ]
  if (nrow(d) < 5) {
    return(data.frame(T_cell_state = state, n_patients_tested = nrow(d), median_state_pct = NA,
                      median_rest_pct = NA, median_paired_difference_pct = NA, wilcoxon_V = NA,
                      p_value = NA))
  }
  wt <- suppressWarnings(wilcox.test(d$state_trbv9_pct, d$rest_trbv9_pct, paired = TRUE, exact = FALSE))
  data.frame(
    T_cell_state = state,
    n_patients_tested = nrow(d),
    median_state_pct = median(d$state_trbv9_pct),
    median_rest_pct = median(d$rest_trbv9_pct),
    median_paired_difference_pct = median(d$state_trbv9_pct - d$rest_trbv9_pct),
    wilcoxon_V = unname(wt$statistic),
    p_value = wt$p.value
  )
}
state_tests <- do.call(rbind, lapply(states, state_test))
state_tests$fdr_bh <- p.adjust(state_tests$p_value, method = "BH")
state_tests <- state_tests[order(state_tests$fdr_bh, -abs(state_tests$median_paired_difference_pct), na.last = TRUE), ]
state_summary <- merge(state_summary, state_tests, by = "T_cell_state", all.x = TRUE)
state_summary <- state_summary[order(-state_summary$trbv9_cells), ]
write.csv(state_summary, file.path(output_dir, "TRBV9_T_cell_state_summary_and_enrichment.csv"), row.names = FALSE, na = "")

size_levels <- levels(patient_clone_counts$clone_size_class)
diag_levels <- c("Control", "UC", "CD")
size_cells <- aggregate(n_cells ~ Diagnosis + clone_size_class, patient_clone_counts, sum)
size_cells <- merge(expand.grid(Diagnosis = diag_levels, clone_size_class = size_levels, stringsAsFactors = FALSE),
                    size_cells, by = c("Diagnosis", "clone_size_class"), all.x = TRUE)
size_cells$n_cells[is.na(size_cells$n_cells)] <- 0
size_cells$cell_fraction <- ave(size_cells$n_cells, size_cells$Diagnosis, FUN = function(x) x / sum(x))
write.csv(size_cells, file.path(output_dir, "TRBV9_clone_size_distribution_by_diagnosis.csv"), row.names = FALSE)

overview <- data.frame(
  metric = c(
    "All beta-chain cells with clonotype", "TRBV9 cells", "TRBV9 patient-clonotype units",
    "Unique TRBV9 clonotypes", "Patients with TRBV9", "Public TRBV9 clonotypes (>=2 patients)",
    "TRBV9 cells in CD4 object", "TRBV9 cells in CD8 object"
  ),
  value = c(
    nrow(beta), nrow(trbv9), nrow(patient_clone_counts), nrow(clone_summary),
    length(unique(trbv9$PatientID)), sum(clone_summary$n_patients >= 2),
    sum(trbv9$lineage == "CD4"), sum(trbv9$lineage == "CD8")
  )
)
write.csv(overview, file.path(output_dir, "TRBV9_overview.csv"), row.names = FALSE)

state_plot <- state_summary[order(state_summary$trbv9_prevalence_pct), ]
png(file.path(output_dir, "TRBV9_T_cell_state_distribution.png"), width = 3000, height = 2100, res = 300, bg = "white")
par(mfrow = c(1, 2), mar = c(5.5, 10.5, 4, 1.2), oma = c(0, 0, 2.2, 0), las = 1)
barplot(state_plot$trbv9_prevalence_pct, names.arg = state_plot$T_cell_state, horiz = TRUE,
        col = "#4C78A8", border = NA, cex.names = 0.72,
        xlab = "TRBV9 prevalence (%)", main = "State-specific prevalence", cex.main = 0.95)
abline(v = 100 * nrow(trbv9) / nrow(beta), lty = 2, col = "#B22222", lwd = 1.5)
legend("bottomright", legend = "Overall TRBV9 prevalence", lty = 2, col = "#B22222", bty = "n", cex = 0.72)

comp_plot <- state_summary[order(state_summary$pct_all_trbv9_cells), ]
barplot(comp_plot$pct_all_trbv9_cells, names.arg = comp_plot$T_cell_state, horiz = TRUE,
        col = "#72B7B2", border = NA, cex.names = 0.72,
        xlab = "Share of TRBV9 cells (%)", main = "Cell-state composition", cex.main = 0.95)
mtext("TRBV9 distribution and associated T-cell states", outer = TRUE, side = 3, font = 2, cex = 1.25)
dev.off()

png(file.path(output_dir, "TRBV9_clone_size_distribution.png"), width = 2400, height = 1700, res = 300, bg = "white")
mat <- xtabs(cell_fraction ~ clone_size_class + Diagnosis, size_cells)
mat <- mat[size_levels, diag_levels, drop = FALSE]
cols <- c("#D9D9D9", "#A6CEE3", "#1F78B4", "#FB9A99", "#E31A1C")
barplot(mat, col = cols, border = "white", ylab = "Fraction of TRBV9 cells", xlab = NULL,
        main = "TRBV9 clone-size distribution by diagnosis", ylim = c(0, 1), las = 1)
legend("topright", legend = size_levels, fill = cols, border = NA, bty = "n", cex = 0.78)
dev.off()

heat_states <- head(state_summary$T_cell_state[order(-state_summary$trbv9_cells)], 12)
heat_ids <- head(top_ids, 20)
heat <- matrix(0, nrow = length(heat_ids), ncol = length(heat_states), dimnames = list(heat_ids, heat_states))
for (i in seq_len(nrow(top_state))) {
  if (top_state$clonotype_id[i] %in% heat_ids && top_state$T_cell_state[i] %in% heat_states) {
    heat[top_state$clonotype_id[i], top_state$T_cell_state[i]] <- top_state$pct_clonotype_cells[i]
  }
}
short_ids <- sub("^TRBV9\\|", "", rownames(heat))
short_ids <- sub("\\|TRBJ", " | J", short_ids)
rownames(heat) <- short_ids
png(file.path(output_dir, "TRBV9_top20_clonotype_state_heatmap.png"), width = 3000, height = 2400, res = 300, bg = "white")
par(mar = c(11, 12, 4, 5))
pal <- colorRampPalette(c("white", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"))(100)
image(seq_len(ncol(heat)), seq_len(nrow(heat)), t(heat[nrow(heat):1, , drop = FALSE]),
      col = pal, axes = FALSE, xlab = "", ylab = "", main = "Top TRBV9 clonotypes across T-cell states")
axis(1, at = seq_len(ncol(heat)), labels = colnames(heat), las = 2, cex.axis = 0.72)
axis(2, at = seq_len(nrow(heat)), labels = rev(rownames(heat)), las = 2, cex.axis = 0.62)
box()
dev.off()

notes <- c(
  "TRBV9 TCR clonotype and T-cell-state analysis",
  "",
  "Definitions:",
  "- Beta clonotype = allele-collapsed TRBV gene + beta-chain CDR3 amino-acid sequence + allele-collapsed TRBJ gene.",
  "- TRBV9 includes dominant beta chains annotated as TRBV9, with alleles collapsed.",
  "- State = AnnotationLevel2 from the CD4 and CD8 repertoire-linked Seurat objects.",
  "- Clone size is the number of cells for a clonotype within a patient, pooled across CD4/CD8 states.",
  "- State enrichment compares, within each patient, TRBV9 prevalence in a state versus all other states using a paired Wilcoxon test.",
  "- State tests require >=5 beta-chain cells in the state and >=20 outside it; FDR is BH-adjusted across states.",
  "",
  paste("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  capture.output(sessionInfo())
)
writeLines(notes, file.path(output_dir, "TRBV9_state_analysis_notes.txt"))

print(overview)
print(head(state_summary[order(state_summary$fdr_bh), ], 12))

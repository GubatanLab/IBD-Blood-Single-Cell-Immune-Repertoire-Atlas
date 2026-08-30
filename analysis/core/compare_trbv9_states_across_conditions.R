options(stringsAsFactors = FALSE)

input_file <- file.path(getwd(), "TRBV9 state analysis", "TRBV9_patient_state_prevalence.csv")
output_dir <- file.path(getwd(), "TRBV9 state analysis", "condition comparisons")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

d <- read.csv(input_file, check.names = FALSE)
d <- d[is.finite(d$state_trbv9_pct) & d$state_beta_cells >= 5, ]

comparisons <- list(
  IBD_vs_Control = list(
    keep = d$Diagnosis %in% c("Control", "UC", "CD"),
    group = ifelse(d$Diagnosis == "Control", "Control", "IBD"),
    group1 = "IBD", group2 = "Control", label = "IBD vs Control"
  ),
  UC_vs_Control = list(
    keep = d$Diagnosis %in% c("Control", "UC"), group = d$Diagnosis,
    group1 = "UC", group2 = "Control", label = "UC vs Control"
  ),
  CD_vs_Control = list(
    keep = d$Diagnosis %in% c("Control", "CD"), group = d$Diagnosis,
    group1 = "CD", group2 = "Control", label = "CD vs Control"
  ),
  UC_Inflamed_vs_Noninflamed = list(
    keep = d$Diagnosis == "UC" & d$Inflammation %in% c("Inflamed", "Noninflamed"),
    group = d$Inflammation, group1 = "Inflamed", group2 = "Noninflamed",
    label = "UC: Inflamed vs Noninflamed"
  ),
  CD_Inflamed_vs_Noninflamed = list(
    keep = d$Diagnosis == "CD" & d$Inflammation %in% c("Inflamed", "Noninflamed"),
    group = d$Inflammation, group1 = "Inflamed", group2 = "Noninflamed",
    label = "CD: Inflamed vs Noninflamed"
  )
)

cliffs_delta <- function(x, y) mean(outer(x, y, function(a, b) sign(a - b)))

run_state <- function(comp_id, state) {
  comp <- comparisons[[comp_id]]
  keep <- comp$keep & d$T_cell_state == state
  x <- d$state_trbv9_pct[keep & comp$group == comp$group1]
  y <- d$state_trbv9_pct[keep & comp$group == comp$group2]
  detect_x <- d$state_trbv9_cells[keep & comp$group == comp$group1] > 0
  detect_y <- d$state_trbv9_cells[keep & comp$group == comp$group2] > 0
  if (length(x) < 5 || length(y) < 5) {
    return(data.frame(
      comparison_id = comp_id, comparison = comp$label, T_cell_state = state,
      group1 = comp$group1, group2 = comp$group2, n_group1 = length(x), n_group2 = length(y),
      median_group1_pct = if (length(x)) median(x) else NA, median_group2_pct = if (length(y)) median(y) else NA,
      median_difference_pct = NA, cliffs_delta = NA, p_value = NA,
      detection_group1_pct = if (length(detect_x)) 100 * mean(detect_x) else NA,
      detection_group2_pct = if (length(detect_y)) 100 * mean(detect_y) else NA,
      detection_p_value = NA
    ))
  }
  wt <- suppressWarnings(wilcox.test(x, y, exact = FALSE))
  ftab <- matrix(c(sum(detect_x), sum(!detect_x), sum(detect_y), sum(!detect_y)), nrow = 2, byrow = TRUE)
  fp <- suppressWarnings(fisher.test(ftab)$p.value)
  data.frame(
    comparison_id = comp_id, comparison = comp$label, T_cell_state = state,
    group1 = comp$group1, group2 = comp$group2, n_group1 = length(x), n_group2 = length(y),
    median_group1_pct = median(x), median_group2_pct = median(y),
    median_difference_pct = median(x) - median(y), cliffs_delta = cliffs_delta(x, y),
    p_value = wt$p.value,
    detection_group1_pct = 100 * mean(detect_x), detection_group2_pct = 100 * mean(detect_y),
    detection_p_value = fp
  )
}

states <- sort(unique(d$T_cell_state))
results <- do.call(rbind, lapply(names(comparisons), function(comp_id) {
  do.call(rbind, lapply(states, run_state, comp_id = comp_id))
}))
results$fdr_bh_within_comparison <- ave(results$p_value, results$comparison_id,
                                        FUN = function(x) p.adjust(x, method = "BH"))
results$detection_fdr_bh_within_comparison <- ave(results$detection_p_value, results$comparison_id,
                                                  FUN = function(x) p.adjust(x, method = "BH"))
results$fdr_bh_global <- p.adjust(results$p_value, method = "BH")
results <- results[order(results$comparison_id, results$fdr_bh_within_comparison,
                         -abs(results$median_difference_pct), na.last = TRUE), ]
write.csv(results, file.path(output_dir, "TRBV9_state_condition_comparison_statistics.csv"), row.names = FALSE, na = "")

# Non-parametric omnibus diagnosis test within each state.
run_kw <- function(state) {
  ds <- d[d$T_cell_state == state & d$Diagnosis %in% c("Control", "UC", "CD"), ]
  group_n <- table(factor(ds$Diagnosis, levels = c("Control", "UC", "CD")))
  if (any(group_n < 5)) {
    return(data.frame(
      T_cell_state = state, n_control = group_n["Control"], n_uc = group_n["UC"], n_cd = group_n["CD"],
      median_control_pct = NA, median_uc_pct = NA, median_cd_pct = NA,
      kruskal_wallis_chisq = NA, df = NA, p_value = NA
    ))
  }
  kt <- kruskal.test(state_trbv9_pct ~ Diagnosis, data = ds)
  data.frame(
    T_cell_state = state,
    n_control = group_n["Control"], n_uc = group_n["UC"], n_cd = group_n["CD"],
    median_control_pct = median(ds$state_trbv9_pct[ds$Diagnosis == "Control"]),
    median_uc_pct = median(ds$state_trbv9_pct[ds$Diagnosis == "UC"]),
    median_cd_pct = median(ds$state_trbv9_pct[ds$Diagnosis == "CD"]),
    kruskal_wallis_chisq = unname(kt$statistic), df = unname(kt$parameter), p_value = kt$p.value
  )
}
kw_results <- do.call(rbind, lapply(states, run_kw))
kw_results$fdr_bh_across_states <- p.adjust(kw_results$p_value, method = "BH")
kw_results <- kw_results[order(kw_results$fdr_bh_across_states, kw_results$p_value, na.last = TRUE), ]
write.csv(kw_results, file.path(output_dir, "TRBV9_state_global_kruskal_wallis.csv"), row.names = FALSE, na = "")

medians <- aggregate(state_trbv9_pct ~ T_cell_state + Diagnosis, d, median)
names(medians)[3] <- "median_patient_TRBV9_pct"
ns <- aggregate(PatientID ~ T_cell_state + Diagnosis, d, length)
names(ns)[3] <- "n_patients"
diagnosis_summary <- merge(medians, ns, by = c("T_cell_state", "Diagnosis"))
write.csv(diagnosis_summary, file.path(output_dir, "TRBV9_state_medians_by_diagnosis.csv"), row.names = FALSE)

inflamed <- d[d$Diagnosis %in% c("UC", "CD") & d$Inflammation %in% c("Inflamed", "Noninflamed"), ]
inflamed$condition <- paste(inflamed$Diagnosis, inflamed$Inflammation, sep = "-")
imed <- aggregate(state_trbv9_pct ~ T_cell_state + condition, inflamed, median)
names(imed)[3] <- "median_patient_TRBV9_pct"
ins <- aggregate(PatientID ~ T_cell_state + condition, inflamed, length)
names(ins)[3] <- "n_patients"
inflammation_summary <- merge(imed, ins, by = c("T_cell_state", "condition"))
write.csv(inflammation_summary, file.path(output_dir, "TRBV9_state_medians_by_inflammation.csv"), row.names = FALSE)

valid <- results[is.finite(results$median_difference_pct), ]
effect_rank_matrix <- xtabs(median_difference_pct ~ T_cell_state + comparison_id, valid)
state_order <- rownames(effect_rank_matrix)[order(-apply(abs(effect_rank_matrix), 1, max, na.rm = TRUE))]
comp_order <- names(comparisons)
mat <- matrix(NA_real_, nrow = length(state_order), ncol = length(comp_order),
              dimnames = list(state_order, comp_order))
fdr <- mat
for (i in seq_len(nrow(valid))) {
  mat[valid$T_cell_state[i], valid$comparison_id[i]] <- valid$median_difference_pct[i]
  fdr[valid$T_cell_state[i], valid$comparison_id[i]] <- valid$fdr_bh_within_comparison[i]
}
write.csv(data.frame(T_cell_state = rownames(mat), mat, check.names = FALSE),
          file.path(output_dir, "TRBV9_state_condition_effect_matrix.csv"), row.names = FALSE, na = "")

short_comp <- c(
  IBD_vs_Control = "IBD-Control", UC_vs_Control = "UC-Control", CD_vs_Control = "CD-Control",
  UC_Inflamed_vs_Noninflamed = "UC Infl-Noninfl", CD_Inflamed_vs_Noninflamed = "CD Infl-Noninfl"
)
colnames(mat) <- unname(short_comp[colnames(mat)])
colnames(fdr) <- colnames(mat)

png(file.path(output_dir, "TRBV9_state_condition_effect_heatmap.png"), width = 2400, height = 2600, res = 300, bg = "white")
par(mar = c(9, 10, 4, 3))
lim <- max(abs(mat), na.rm = TRUE)
pal <- colorRampPalette(c("#2166AC", "#D1E5F0", "white", "#FDDBC7", "#B2182B"))(101)
z <- mat[nrow(mat):1, , drop = FALSE]
image(seq_len(ncol(z)), seq_len(nrow(z)), t(z), col = pal, zlim = c(-lim, lim), axes = FALSE,
      xlab = "", ylab = "", main = "TRBV9 prevalence differences across IBD conditions")
axis(1, at = seq_len(ncol(z)), labels = colnames(z), las = 2, cex.axis = 0.78)
axis(2, at = seq_len(nrow(z)), labels = rev(rownames(mat)), las = 2, cex.axis = 0.72)
for (ri in seq_len(nrow(mat))) {
  for (ci in seq_len(ncol(mat))) {
    if (is.finite(fdr[ri, ci]) && fdr[ri, ci] < 0.05) {
      text(ci, nrow(mat) - ri + 1, "*", cex = 1.15, font = 2)
    }
  }
}
mtext("Red: higher in first group; blue: lower in first group; * within-comparison FDR < 0.05",
      side = 1, line = 7.2, cex = 0.72)
box()
dev.off()

top_states <- c("CD4 Naive", "CD4 Tfh", "CD4 Th17", "CD8 Tcm CCR4-", "CD8 Naive",
                "TReg Cytotoxic", "CD4 Temra", "CD8 HLA-DR+", "CD4 Terminal effector")
top_states <- top_states[top_states %in% states]
cols <- c(Control = "#8C8C8C", UC = "#D95F02", CD = "#1B9E77")
png(file.path(output_dir, "TRBV9_key_states_by_diagnosis.png"), width = 2700, height = 2700, res = 300, bg = "white")
par(mfrow = c(3, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0), las = 1)
set.seed(20260810)
for (state in top_states) {
  ds <- d[d$T_cell_state == state & d$Diagnosis %in% c("Control", "UC", "CD"), ]
  vals <- lapply(c("Control", "UC", "CD"), function(g) ds$state_trbv9_pct[ds$Diagnosis == g])
  ymax <- max(unlist(vals), na.rm = TRUE)
  boxplot(vals, names = c("Control", "UC", "CD"), col = cols, outline = FALSE,
          ylab = "TRBV9 prevalence (%)", main = state, cex.main = 0.92, ylim = c(0, max(0.5, ymax * 1.08)))
  for (j in 1:3) points(jitter(rep(j, length(vals[[j]])), amount = 0.07), vals[[j]], pch = 21,
                         bg = adjustcolor("white", 0.65), col = "#333333", cex = 0.55)
}
mtext("TRBV9 prevalence in key T-cell states by diagnosis", outer = TRUE, side = 3, line = 1.2, font = 2, cex = 1.25)
dev.off()

notes <- c(
  "TRBV9 state comparisons across IBD conditions",
  "",
  "- Outcome: percentage of beta-chain cells using TRBV9 within each patient and AnnotationLevel2 state.",
  "- Patient-state strata require at least five beta-chain cells.",
  "- Two-sided Wilcoxon rank-sum tests compare patient-level percentages.",
  "- Omnibus Control-UC-CD differences are tested with Kruskal-Wallis tests within each state.",
  "- FDR is Benjamini-Hochberg adjusted across states separately within each clinical comparison; a global FDR is also reported.",
  "- Fisher tests compare whether TRBV9 is detected at all in each patient-state stratum.",
  paste("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"))
)
writeLines(notes, file.path(output_dir, "TRBV9_state_condition_analysis_notes.txt"))

print(results[is.finite(results$fdr_bh_within_comparison) & results$fdr_bh_within_comparison < 0.10, ])

options(stringsAsFactors = FALSE)

source_path <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD Immune Repertoire Manuscript/Figure 3 TCR Repertoire Architecture/IBDTCR.rds"
output_dir <- file.path(getwd(), "TRBV9 analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(source_path)
meta <- obj$meta

required_meta <- c("Sample", "SampleID", "PatientID", "Diagnosis1", "Inflammation1")
stopifnot(all(required_meta %in% names(meta)))
stopifnot(!anyDuplicated(meta$PatientID))
stopifnot(all(meta$Sample %in% names(obj$data)))

summarize_sample <- function(sample_name) {
  d <- obj$data[[sample_name]]
  v <- as.character(d$V.name)
  clones <- suppressWarnings(as.numeric(d$Clones))
  clones[is.na(clones)] <- 0
  beta <- !is.na(v) & grepl("^TRBV", v, ignore.case = TRUE)
  trbv9 <- beta & grepl("^TRBV9(?:\\*|$)", v, ignore.case = TRUE)
  beta_cells <- sum(clones[beta])
  trbv9_cells <- sum(clones[trbv9])
  beta_clonotypes <- sum(beta)
  trbv9_clonotypes <- sum(trbv9)
  data.frame(
    Sample = sample_name,
    beta_cells = beta_cells,
    trbv9_cells = trbv9_cells,
    trbv9_cell_fraction = if (beta_cells > 0) trbv9_cells / beta_cells else NA_real_,
    beta_clonotypes = beta_clonotypes,
    trbv9_clonotypes = trbv9_clonotypes,
    trbv9_clonotype_fraction = if (beta_clonotypes > 0) trbv9_clonotypes / beta_clonotypes else NA_real_
  )
}

sample_usage <- do.call(rbind, lapply(meta$Sample, summarize_sample))
sample_usage <- merge(meta, sample_usage, by = "Sample", all.x = TRUE, sort = FALSE)
sample_usage <- sample_usage[match(meta$Sample, sample_usage$Sample), ]
stopifnot(identical(sample_usage$Sample, meta$Sample))

comparisons <- list(
  IBD_vs_Control = list(
    keep = sample_usage$Diagnosis1 %in% c("UC", "CD", "Control"),
    group = ifelse(sample_usage$Diagnosis1 == "Control", "Control", "IBD"),
    group1 = "IBD", group2 = "Control",
    label = "IBD vs Control"
  ),
  UC_vs_Control = list(
    keep = sample_usage$Diagnosis1 %in% c("UC", "Control"),
    group = sample_usage$Diagnosis1,
    group1 = "UC", group2 = "Control",
    label = "UC vs Control"
  ),
  CD_vs_Control = list(
    keep = sample_usage$Diagnosis1 %in% c("CD", "Control"),
    group = sample_usage$Diagnosis1,
    group1 = "CD", group2 = "Control",
    label = "CD vs Control"
  ),
  UC_Inflamed_vs_Noninflamed = list(
    keep = sample_usage$Diagnosis1 == "UC" & sample_usage$Inflammation1 %in% c("Inflamed", "Noninflamed"),
    group = sample_usage$Inflammation1,
    group1 = "Inflamed", group2 = "Noninflamed",
    label = "UC: Inflamed vs Noninflamed"
  ),
  CD_Inflamed_vs_Noninflamed = list(
    keep = sample_usage$Diagnosis1 == "CD" & sample_usage$Inflammation1 %in% c("Inflamed", "Noninflamed"),
    group = sample_usage$Inflammation1,
    group1 = "Inflamed", group2 = "Noninflamed",
    label = "CD: Inflamed vs Noninflamed"
  )
)

cliffs_delta <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  if (!length(x) || !length(y)) return(NA_real_)
  mean(outer(x, y, FUN = function(a, b) sign(a - b)))
}

summarize_vector <- function(x) {
  x <- x[is.finite(x)]
  c(n = length(x), median = median(x), q1 = unname(quantile(x, 0.25)), q3 = unname(quantile(x, 0.75)), mean = mean(x))
}

run_comparison <- function(comp_name, metric) {
  comp <- comparisons[[comp_name]]
  keep <- comp$keep & is.finite(sample_usage[[metric]])
  g <- comp$group[keep]
  x <- sample_usage[[metric]][keep & comp$group == comp$group1]
  y <- sample_usage[[metric]][keep & comp$group == comp$group2]
  sx <- summarize_vector(x)
  sy <- summarize_vector(y)
  wt <- wilcox.test(x, y, exact = FALSE, conf.int = TRUE, conf.level = 0.95)
  data.frame(
    comparison_id = comp_name,
    comparison = comp$label,
    metric = metric,
    group1 = comp$group1,
    group2 = comp$group2,
    n_group1 = sx["n"],
    n_group2 = sy["n"],
    median_group1 = sx["median"],
    q1_group1 = sx["q1"],
    q3_group1 = sx["q3"],
    median_group2 = sy["median"],
    q1_group2 = sy["q1"],
    q3_group2 = sy["q3"],
    median_difference = sx["median"] - sy["median"],
    wilcoxon_W = unname(wt$statistic),
    p_value = wt$p.value,
    cliffs_delta = cliffs_delta(x, y),
    stringsAsFactors = FALSE
  )
}

metrics <- c("trbv9_cell_fraction", "trbv9_clonotype_fraction")
results <- do.call(rbind, lapply(metrics, function(metric) {
  do.call(rbind, lapply(names(comparisons), run_comparison, metric = metric))
}))
results$fdr_bh_within_metric <- ave(results$p_value, results$metric, FUN = function(x) p.adjust(x, method = "BH"))

write.csv(sample_usage, file.path(output_dir, "TRBV9_sample_level_usage.csv"), row.names = FALSE, na = "")
write.csv(results, file.path(output_dir, "TRBV9_comparison_statistics.csv"), row.names = FALSE, na = "")

plot_rows <- list()
for (comp_name in names(comparisons)) {
  comp <- comparisons[[comp_name]]
  keep <- comp$keep & is.finite(sample_usage$trbv9_cell_fraction)
  plot_rows[[comp_name]] <- data.frame(
    comparison_id = comp_name,
    comparison = comp$label,
    group = comp$group[keep],
    value = 100 * sample_usage$trbv9_clonotype_fraction[keep],
    stringsAsFactors = FALSE
  )
}
plot_data <- do.call(rbind, plot_rows)
write.csv(plot_data, file.path(output_dir, "TRBV9_clonotype_plot_source_data.csv"), row.names = FALSE)

ann <- results[results$metric == "trbv9_clonotype_fraction", ]
ann$annotation <- ifelse(
  ann$fdr_bh_within_metric < 0.001, "FDR < 0.001",
  paste0("FDR = ", formatC(ann$fdr_bh_within_metric, format = "g", digits = 2))
)

group_colors <- c(
  Control = "#8C8C8C", IBD = "#7570B3", UC = "#D95F02", CD = "#1B9E77",
  Noninflamed = "#80B1D3", Inflamed = "#FB8072"
)

draw_figure <- function() {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  par(mfrow = c(3, 2), mar = c(4.5, 4.5, 3.3, 1.0), oma = c(0, 0, 3.8, 0), las = 1)
  set.seed(20260810)
  for (i in seq_along(comparisons)) {
    comp_name <- names(comparisons)[i]
    comp <- comparisons[[comp_name]]
    d <- plot_data[plot_data$comparison_id == comp_name, ]
    groups <- c(comp$group2, comp$group1)
    vals <- lapply(groups, function(g) d$value[d$group == g])
    ymax <- max(unlist(vals), na.rm = TRUE)
    upper <- max(ymax * 1.23, 0.12)
    boxplot(
      vals, names = groups, col = unname(group_colors[groups]), border = "#333333",
      outline = FALSE, ylim = c(0, upper), ylab = "TRBV9 clonotypes (%)", main = comp$label,
      cex.main = 0.94, cex.axis = 0.88, cex.lab = 0.9
    )
    for (j in seq_along(vals)) {
      points(jitter(rep(j, length(vals[[j]])), amount = 0.09), vals[[j]], pch = 21,
             bg = adjustcolor("white", alpha.f = 0.68), col = "#222222", cex = 0.72)
    }
    a <- ann[ann$comparison_id == comp_name, ]
    segments(1, upper * 0.89, 2, upper * 0.89, lwd = 1)
    segments(c(1, 2), upper * 0.86, c(1, 2), upper * 0.89, lwd = 1)
    text(1.5, upper * 0.95, a$annotation, font = 2, cex = 0.84)
    mtext(sprintf("n = %d vs %d", length(vals[[2]]), length(vals[[1]])), side = 3, line = 0.15, cex = 0.72, col = "#555555")
  }
  plot.new()
  mtext("TRBV9 clonotypes in the IBD T-cell repertoire", outer = TRUE, side = 3, line = 2.0, font = 2, cex = 1.35)
  mtext("Each point is one patient; FDR is BH-adjusted across five comparisons", outer = TRUE, side = 3, line = 0.65, cex = 0.85, col = "#444444")
}

png(file.path(output_dir, "TRBV9_clonotype_comparisons.png"), width = 2700, height = 3000, res = 300, bg = "white")
draw_figure()
dev.off()

tiff(file.path(output_dir, "TRBV9_clonotype_comparisons.tiff"), width = 2700, height = 3000, res = 300, compression = "lzw", bg = "white")
draw_figure()
dev.off()

pdf(file.path(output_dir, "TRBV9_clonotype_comparisons.pdf"), width = 9, height = 10, useDingbats = FALSE)
draw_figure()
dev.off()

session_lines <- c(
  "Analysis definition:",
  "- Primary clonotype metric: unique TRBV9 clonotype rows divided by all TRBV clonotype rows, per patient sample.",
  "- Abundance sensitivity metric: sum(Clones) for TRBV9*01/*02/*03 divided by sum(Clones) for all TRBV genes, per patient sample.",
  "- Tests: two-sided Wilcoxon rank-sum tests; Benjamini-Hochberg correction across the five comparisons within each metric.",
  "- Effect size: Cliff's delta, oriented as group1 minus group2.",
  "- Biological unit: patient (182 unique PatientID values; no repeated patients).",
  "",
  paste("Source:", source_path),
  paste("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  "",
  capture.output(sessionInfo())
)
writeLines(session_lines, file.path(output_dir, "TRBV9_analysis_notes.txt"))

print(results)

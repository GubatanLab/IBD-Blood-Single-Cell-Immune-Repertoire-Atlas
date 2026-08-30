suppressPackageStartupMessages({
  library(scoper)
  library(ggplot2)
})

input_root <- "chain_bcr_immuneml/airr_input/bcr_heavy_light"
output_root <- "chain_bcr_immuneml/scoper_bcr_clustering"
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

threshold <- 0.15
method <- "nt"
linkage <- "single"

comparisons <- list(
  cd_vs_control = c("CD", "Control"),
  uc_vs_control = c("UC", "Control"),
  cd_vs_uc = c("CD", "UC")
)
positive_group <- c(
  cd_vs_control = "CD",
  uc_vs_control = "UC",
  cd_vs_uc = "CD"
)

metrics_to_test <- c(
  "n_sequences",
  "n_clones",
  "clone_richness_per_100_sequences",
  "expanded_clone_count",
  "expanded_clone_fraction",
  "dominant_clone_fraction",
  "top10_clone_fraction",
  "shannon_entropy",
  "shannon_evenness",
  "clonality",
  "simpson_diversity",
  "inverse_simpson",
  "gini",
  "mean_clone_size",
  "median_clone_size",
  "heavy_light_clone_fraction"
)

pretty_metric <- c(
  n_sequences = "Sequences",
  n_clones = "Clones",
  clone_richness_per_100_sequences = "Clones per 100 seq",
  expanded_clone_count = "Expanded clones",
  expanded_clone_fraction = "Expanded clone fraction",
  dominant_clone_fraction = "Dominant clone fraction",
  top10_clone_fraction = "Top 10 clone fraction",
  shannon_entropy = "Shannon entropy",
  shannon_evenness = "Shannon evenness",
  clonality = "Clonality",
  simpson_diversity = "Simpson diversity",
  inverse_simpson = "Inverse Simpson",
  gini = "Gini",
  mean_clone_size = "Mean clone size",
  median_clone_size = "Median clone size",
  heavy_light_clone_fraction = "Heavy+light clone fraction"
)

comparison_label <- c(
  cd_vs_control = "CD vs control",
  uc_vs_control = "UC vs control",
  cd_vs_uc = "CD vs UC"
)

gini <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x) & x >= 0]
  if (length(x) == 0 || sum(x) == 0) return(0)
  x <- sort(x)
  n <- length(x)
  sum((2 * seq_len(n) - n - 1) * x) / (n * sum(x))
}

safe_wilcox <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  if (length(x) == 0 || length(y) == 0) return(NA_real_)
  if (length(unique(c(x, y))) < 2) return(NA_real_)
  suppressWarnings(wilcox.test(x, y, exact = FALSE)$p.value)
}

clone_metrics <- function(assignments) {
  counts <- aggregate(duplicate_count ~ clone_uid, assignments, sum)
  names(counts)[2] <- "clone_size"
  seq_counts <- as.numeric(counts$clone_size)
  total <- sum(seq_counts)
  richness <- length(seq_counts)

  if (richness == 0 || total == 0) {
    return(data.frame(
      n_sequences = 0,
      n_clones = 0,
      clone_richness_per_100_sequences = 0,
      expanded_clone_count = 0,
      expanded_clone_fraction = 0,
      dominant_clone_fraction = 0,
      top10_clone_fraction = 0,
      shannon_entropy = 0,
      shannon_evenness = 0,
      clonality = 0,
      simpson_diversity = 0,
      inverse_simpson = 0,
      gini = 0,
      mean_clone_size = 0,
      median_clone_size = 0,
      heavy_light_clone_fraction = 0
    ))
  }

  probabilities <- seq_counts / total
  shannon <- -sum(probabilities * log(probabilities))
  evenness <- if (richness > 1) shannon / log(richness) else 0
  simpson_concentration <- sum(probabilities ^ 2)
  clone_loci <- split(assignments$locus, assignments$clone_uid)
  has_heavy_light <- vapply(
    clone_loci,
    function(x) any(x == "IGH") && any(x %in% c("IGK", "IGL")),
    logical(1)
  )

  data.frame(
    n_sequences = nrow(assignments),
    n_clones = richness,
    clone_richness_per_100_sequences = richness / nrow(assignments) * 100,
    expanded_clone_count = sum(seq_counts > 1),
    expanded_clone_fraction = mean(seq_counts > 1),
    dominant_clone_fraction = max(seq_counts) / total,
    top10_clone_fraction = sum(sort(seq_counts, decreasing = TRUE)[seq_len(min(10, richness))]) / total,
    shannon_entropy = shannon,
    shannon_evenness = evenness,
    clonality = 1 - evenness,
    simpson_diversity = 1 - simpson_concentration,
    inverse_simpson = if (simpson_concentration > 0) 1 / simpson_concentration else 0,
    gini = gini(seq_counts),
    mean_clone_size = mean(seq_counts),
    median_clone_size = median(seq_counts),
    heavy_light_clone_fraction = mean(has_heavy_light)
  )
}

metadata <- read.csv(file.path(input_root, "metadata_all.csv"), stringsAsFactors = FALSE)
all_assignments <- list()
metric_rows <- list()
status_rows <- list()

for (idx in seq_len(nrow(metadata))) {
  meta <- metadata[idx, ]
  rep_path <- file.path(input_root, "repertoires", meta$filename)
  message(sprintf("Clustering %s (%s)", meta$SampleID, meta$Diagnosis1))

  db <- read.delim(rep_path, stringsAsFactors = FALSE, check.names = FALSE)
  db$SampleID <- meta$SampleID
  db$PatientID <- meta$PatientID
  db$Diagnosis1 <- meta$Diagnosis1
  db$filename <- meta$filename
  db$duplicate_count <- suppressWarnings(as.numeric(db$duplicate_count))
  db$duplicate_count[!is.finite(db$duplicate_count) | db$duplicate_count < 1] <- 1
  db <- db[!is.na(db$junction) & db$junction != "" &
             !is.na(db$junction_aa) & db$junction_aa != "" &
             !is.na(db$v_call) & db$v_call != "" &
             !is.na(db$j_call) & db$j_call != "" &
             db$locus %in% c("IGH", "IGK", "IGL"), ]
  if ("productive" %in% names(db)) {
    db <- db[tolower(as.character(db$productive)) %in% c("true", "t", "1"), ]
  }

  status <- data.frame(
    SampleID = meta$SampleID,
    PatientID = meta$PatientID,
    Diagnosis1 = meta$Diagnosis1,
    n_input_rows = nrow(db),
    status = "ok",
    message = "",
    stringsAsFactors = FALSE
  )

  clustered <- tryCatch(
    hierarchicalClones(
      db,
      threshold = threshold,
      method = method,
      linkage = linkage,
      normalize = "len",
      junction = "junction",
      v_call = "v_call",
      j_call = "j_call",
      clone = "clone_id",
      fields = "SampleID",
      locus = "locus",
      only_heavy = FALSE,
      split_light = TRUE,
      nproc = 1,
      verbose = FALSE
    ),
    error = function(e) e
  )

  if (inherits(clustered, "error")) {
    status$status <- "failed"
    status$message <- conditionMessage(clustered)
    status_rows[[length(status_rows) + 1]] <- status
    next
  }

  clustered$clone_uid <- paste(clustered$SampleID, clustered$clone_id, sep = "__")
  all_assignments[[length(all_assignments) + 1]] <- clustered

  metrics <- clone_metrics(clustered)
  metrics$SampleID <- meta$SampleID
  metrics$PatientID <- meta$PatientID
  metrics$Diagnosis1 <- meta$Diagnosis1
  metrics$n_heavy <- sum(clustered$locus == "IGH")
  metrics$n_light <- sum(clustered$locus %in% c("IGK", "IGL"))
  metric_rows[[length(metric_rows) + 1]] <- metrics
  status_rows[[length(status_rows) + 1]] <- status
}

assignments <- do.call(rbind, all_assignments)
patient_metrics <- do.call(rbind, metric_rows)
status <- do.call(rbind, status_rows)

write.table(
  assignments,
  file.path(output_root, "bcr_scoper_hierarchical_clone_assignments.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
write.csv(patient_metrics, file.path(output_root, "bcr_scoper_patient_clonotype_metrics.csv"), row.names = FALSE)
write.csv(status, file.path(output_root, "bcr_scoper_clustering_status.csv"), row.names = FALSE)

test_rows <- list()
for (comparison in names(comparisons)) {
  groups <- comparisons[[comparison]]
  subset_metrics <- patient_metrics[patient_metrics$Diagnosis1 %in% groups, ]
  pos <- positive_group[[comparison]]
  neg <- setdiff(groups, pos)
  for (metric in metrics_to_test) {
    x <- subset_metrics[subset_metrics$Diagnosis1 == pos, metric]
    y <- subset_metrics[subset_metrics$Diagnosis1 == neg, metric]
    test_rows[[length(test_rows) + 1]] <- data.frame(
      comparison = comparison,
      comparison_label = comparison_label[[comparison]],
      positive_group = pos,
      reference_group = neg,
      metric = metric,
      metric_label = pretty_metric[[metric]],
      n_positive = sum(subset_metrics$Diagnosis1 == pos),
      n_reference = sum(subset_metrics$Diagnosis1 == neg),
      positive_mean = mean(x, na.rm = TRUE),
      reference_mean = mean(y, na.rm = TRUE),
      mean_difference = mean(x, na.rm = TRUE) - mean(y, na.rm = TRUE),
      positive_median = median(x, na.rm = TRUE),
      reference_median = median(y, na.rm = TRUE),
      median_difference = median(x, na.rm = TRUE) - median(y, na.rm = TRUE),
      wilcox_p = safe_wilcox(x, y),
      stringsAsFactors = FALSE
    )
  }
}

tests <- do.call(rbind, test_rows)
tests$wilcox_fdr <- ave(tests$wilcox_p, tests$comparison, FUN = function(x) p.adjust(x, method = "BH"))
tests$signed_log10_fdr <- -log10(pmax(tests$wilcox_fdr, .Machine$double.xmin)) * sign(tests$mean_difference)
write.csv(tests, file.path(output_root, "bcr_scoper_comparison_tests.csv"), row.names = FALSE)

note <- c(
  "# BCR SCOPer Hierarchical Clustering",
  "",
  paste0("Method: scoper::hierarchicalClones, method=", method, ", linkage=", linkage,
         ", normalized nucleotide Hamming distance threshold=", threshold, "."),
  "",
  "Input: chain_bcr_immuneml/airr_input/bcr_heavy_light repertoires.",
  "",
  "Important limitation: the source BCR AIRR tables do not include cell-level heavy-light pairing identifiers. SCOPer therefore runs in bulk mode and reports that it retains heavy chains only despite the heavy+light input repertoire. These results should be interpreted as SCOPer hierarchical heavy-chain clonotype clustering from the paired heavy+light input folder, not true single-cell paired-chain heavy/light clonotyping.",
  "",
  "In the clone-assignment output, n_light is expected to be zero for all patients because of this SCOPer bulk-mode fallback. No artificial heavy-light pairing is inferred from row order.",
  "",
  "Spectral clustering was not used because the prepared AIRR input lacks the sequence_alignment and germline_alignment fields required by scoper::spectralClones."
)
writeLines(note, file.path(output_root, "bcr_scoper_method_note.md"))

heatmap_df <- tests
heatmap_df$metric_label <- factor(heatmap_df$metric_label, levels = unname(pretty_metric[metrics_to_test]))
heatmap_df$comparison_label <- factor(heatmap_df$comparison_label, levels = unname(comparison_label[names(comparisons)]))

p_heat <- ggplot(heatmap_df, aes(x = comparison_label, y = metric_label, fill = signed_log10_fdr)) +
  geom_tile(color = "white", linewidth = 0.35) +
  geom_text(aes(label = sprintf("%.2f", mean_difference)), size = 2.6) +
  scale_fill_gradient2(
    low = "#3B6FB6",
    mid = "white",
    high = "#B33A3A",
    midpoint = 0,
    name = "Signed -log10(FDR)"
  ) +
  labs(
    title = "BCR SCOPer Hierarchical Clustering: CD, UC, and Control Comparisons",
    x = NULL,
    y = NULL,
    caption = "Tile labels show mean difference: positive group minus reference group."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 35, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(file.path(output_root, "bcr_scoper_comparison_metric_heatmap.png"), p_heat, width = 8.5, height = 6.8, dpi = 600)
ggsave(file.path(output_root, "bcr_scoper_comparison_metric_heatmap.pdf"), p_heat, width = 8.5, height = 6.8)
ggsave(file.path(output_root, "bcr_scoper_comparison_metric_heatmap.svg"), p_heat, width = 8.5, height = 6.8)

box_metrics <- c("n_clones", "shannon_entropy", "clonality", "dominant_clone_fraction", "top10_clone_fraction", "heavy_light_clone_fraction")
box_df <- do.call(
  rbind,
  lapply(names(comparisons), function(comparison) {
    groups <- comparisons[[comparison]]
    subset_metrics <- patient_metrics[patient_metrics$Diagnosis1 %in% groups, ]
    do.call(
      rbind,
      lapply(box_metrics, function(metric) {
        data.frame(
          comparison = comparison_label[[comparison]],
          diagnosis = subset_metrics$Diagnosis1,
          metric = pretty_metric[[metric]],
          value = subset_metrics[[metric]],
          stringsAsFactors = FALSE
        )
      })
    )
  })
)
box_df$comparison <- factor(box_df$comparison, levels = unname(comparison_label[names(comparisons)]))
box_df$metric <- factor(box_df$metric, levels = unname(pretty_metric[box_metrics]))

p_box <- ggplot(box_df, aes(x = diagnosis, y = value, fill = diagnosis)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.72, width = 0.65) +
  geom_jitter(width = 0.15, height = 0, size = 0.8, alpha = 0.55) +
  facet_grid(metric ~ comparison, scales = "free_y") +
  scale_fill_manual(values = c(CD = "#B33A3A", UC = "#3B6FB6", Control = "#5A8F54")) +
  labs(
    title = "BCR SCOPer Clonotype Metrics by Diagnosis Contrast",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 35, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(file.path(output_root, "bcr_scoper_clonotype_metric_boxplots.png"), p_box, width = 10.5, height = 9.5, dpi = 600)
ggsave(file.path(output_root, "bcr_scoper_clonotype_metric_boxplots.pdf"), p_box, width = 10.5, height = 9.5)
ggsave(file.path(output_root, "bcr_scoper_clonotype_metric_boxplots.svg"), p_box, width = 10.5, height = 9.5)

message("Wrote SCOPer clustering outputs to ", output_root)

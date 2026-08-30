#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(scales)
})

set.seed(20260824)
outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds",
  BCR = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026_scvi50_umap.rds"
)
stopifnot(all(file.exists(paths)))

valid_string <- function(x) !is.na(x) & trimws(as.character(x)) != "" & toupper(as.character(x)) != "NA"

load_meta <- function(path) {
  x <- readRDS(path)
  md <- as.data.frame(x[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  rm(x); gc(verbose = FALSE)
  md
}

message("Loading T-cell metadata...")
tmeta <- bind_rows(load_meta(paths[["CD4"]]), load_meta(paths[["CD8"]]))
if ("High_Quality_Cell" %in% names(tmeta)) {
  hq <- as.character(tmeta$High_Quality_Cell)
  keep_hq <- is.na(hq) | tolower(hq) %in% c("true", "t", "1", "high_quality", "high quality")
  tmeta <- tmeta[keep_hq, , drop = FALSE]
}
if ("Total_VDJ_Molecule_Count" %in% names(tmeta)) {
  mol <- suppressWarnings(as.numeric(tmeta$Total_VDJ_Molecule_Count))
  tmeta <- tmeta[is.na(mol) | mol >= 4, , drop = FALSE]
}

tcr_beta <- tmeta %>%
  filter(
    grepl("^TRBC", TCR_Beta_Delta_C_gene_Dominant),
    valid_string(TCR_Beta_Delta_V_gene_Dominant), valid_string(TCR_Beta_Delta_J_gene_Dominant),
    valid_string(TCR_Beta_Delta_CDR3_Translation_Dominant), valid_string(SampleID), valid_string(Diagnosis1)
  ) %>%
  transmute(
    SampleID = as.character(SampleID), Diagnosis1 = as.character(Diagnosis1), cell,
    clonotype_id = paste(TCR_Beta_Delta_V_gene_Dominant, TCR_Beta_Delta_J_gene_Dominant,
                         TCR_Beta_Delta_CDR3_Translation_Dominant, sep = "|")
  )

tcr_pair <- tmeta %>%
  filter(
    grepl("^TRAC", TCR_Alpha_Gamma_C_gene_Dominant), grepl("^TRBC", TCR_Beta_Delta_C_gene_Dominant),
    valid_string(TCR_Alpha_Gamma_V_gene_Dominant), valid_string(TCR_Alpha_Gamma_J_gene_Dominant),
    valid_string(TCR_Alpha_Gamma_CDR3_Translation_Dominant),
    valid_string(TCR_Beta_Delta_V_gene_Dominant), valid_string(TCR_Beta_Delta_J_gene_Dominant),
    valid_string(TCR_Beta_Delta_CDR3_Translation_Dominant), valid_string(SampleID), valid_string(Diagnosis1)
  ) %>%
  transmute(
    SampleID = as.character(SampleID), Diagnosis1 = as.character(Diagnosis1), cell,
    clonotype_id = paste(
      TCR_Alpha_Gamma_V_gene_Dominant, TCR_Alpha_Gamma_J_gene_Dominant, TCR_Alpha_Gamma_CDR3_Translation_Dominant,
      TCR_Beta_Delta_V_gene_Dominant, TCR_Beta_Delta_J_gene_Dominant, TCR_Beta_Delta_CDR3_Translation_Dominant, sep = "|"
    )
  )
rm(tmeta); gc(verbose = FALSE)

message("Loading B-cell metadata...")
bmeta <- load_meta(paths[["BCR"]])
if ("High_Quality_Cell" %in% names(bmeta)) {
  hq <- as.character(bmeta$High_Quality_Cell)
  keep_hq <- is.na(hq) | tolower(hq) %in% c("true", "t", "1", "high_quality", "high quality")
  bmeta <- bmeta[keep_hq, , drop = FALSE]
}
if ("Total_VDJ_Molecule_Count" %in% names(bmeta)) {
  mol <- suppressWarnings(as.numeric(bmeta$Total_VDJ_Molecule_Count))
  bmeta <- bmeta[is.na(mol) | mol >= 4, , drop = FALSE]
}

bcr_heavy <- bmeta %>%
  filter(
    grepl("^IGH", BCR_Heavy_C_gene_Dominant), valid_string(BCR_Heavy_V_gene_Dominant),
    valid_string(BCR_Heavy_J_gene_Dominant), valid_string(BCR_Heavy_CDR3_Translation_Dominant),
    valid_string(SampleID), valid_string(Diagnosis1)
  ) %>%
  transmute(
    SampleID = as.character(SampleID), Diagnosis1 = as.character(Diagnosis1), cell,
    clonotype_id = paste(BCR_Heavy_V_gene_Dominant, BCR_Heavy_J_gene_Dominant,
                         BCR_Heavy_CDR3_Translation_Dominant, sep = "|")
  )

bcr_pair <- bmeta %>%
  filter(
    grepl("^IGH", BCR_Heavy_C_gene_Dominant), grepl("^IG[KL]", BCR_Light_C_gene_Dominant),
    valid_string(BCR_Heavy_V_gene_Dominant), valid_string(BCR_Heavy_J_gene_Dominant),
    valid_string(BCR_Heavy_CDR3_Translation_Dominant), valid_string(BCR_Light_V_gene_Dominant),
    valid_string(BCR_Light_J_gene_Dominant), valid_string(BCR_Light_CDR3_Translation_Dominant),
    valid_string(SampleID), valid_string(Diagnosis1)
  ) %>%
  transmute(
    SampleID = as.character(SampleID), Diagnosis1 = as.character(Diagnosis1), cell,
    clonotype_id = paste(
      BCR_Heavy_V_gene_Dominant, BCR_Heavy_J_gene_Dominant, BCR_Heavy_CDR3_Translation_Dominant,
      BCR_Light_V_gene_Dominant, BCR_Light_J_gene_Dominant, BCR_Light_CDR3_Translation_Dominant, sep = "|"
    )
  )
rm(bmeta); gc(verbose = FALSE)

gini_counts <- function(counts) {
  x <- sort(as.numeric(counts)); n <- length(x)
  if (!n || sum(x) == 0) return(NA_real_)
  sum((2 * seq_len(n) - n - 1) * x) / (n * sum(x))
}

clone_metrics <- function(counts, threshold = 2L) {
  counts <- as.numeric(counts[counts > 0]); p <- counts / sum(counts)
  sh <- -sum(p * log(p)); evenness <- if (length(counts) > 1) sh / log(length(counts)) else 1
  tibble(
    total_cells = sum(counts), richness = length(counts), clonality = 1 - evenness,
    gini = gini_counts(counts),
    expanded_cell_fraction = sum(counts[counts >= threshold]) / sum(counts),
    expanded_clonotype_fraction = mean(counts >= threshold)
  )
}

downsample_counts <- function(counts, target) {
  counts <- as.integer(counts[counts > 0]); total <- sum(counts)
  if (total < target) return(NULL)
  out <- integer(length(counts)); draw <- target; rem_total <- total
  if (length(counts) == 1L) return(target)
  for (i in seq_len(length(counts) - 1L)) {
    out[i] <- rhyper(1, counts[i], rem_total - counts[i], draw)
    draw <- draw - out[i]; rem_total <- rem_total - counts[i]
  }
  out[length(counts)] <- draw
  out[out > 0]
}

process_cells <- function(cells, receptor_definition, B = 100L) {
  clone_counts <- cells %>% count(SampleID, Diagnosis1, clonotype_id, name = "n_cells")
  md <- clone_counts %>% distinct(SampleID, Diagnosis1)
  lists <- split(clone_counts$n_cells, clone_counts$SampleID)
  raw <- bind_rows(lapply(c(2L, 3L, 5L), function(thr) {
    bind_rows(lapply(names(lists), function(sid) bind_cols(tibble(SampleID = sid, threshold = thr), clone_metrics(lists[[sid]], thr))))
  })) %>% left_join(md, by = "SampleID") %>% mutate(receptor_definition = receptor_definition, analysis = "Original depth")
  target <- max(10, floor(quantile(raw$total_cells[raw$threshold == 2], .10, na.rm = TRUE) / 10) * 10)
  rare <- bind_rows(lapply(names(lists), function(sid) {
    counts <- lists[[sid]]
    if (sum(counts) < target) return(NULL)
    sims <- bind_rows(lapply(seq_len(B), function(i) clone_metrics(downsample_counts(counts, target), 2L)))
    sims %>% summarise(across(everything(), \(x) mean(x, na.rm = TRUE))) %>% mutate(SampleID = sid)
  })) %>% left_join(md, by = "SampleID") %>%
    mutate(receptor_definition = receptor_definition, analysis = "Depth-normalized", threshold = 2L, downsample_depth = target)
  list(raw = raw, rare = rare, clone_counts = clone_counts, target = target)
}

sets <- list(
  `TCR beta-only` = process_cells(tcr_beta, "TCR beta-only"),
  `TCR paired alpha-beta` = process_cells(tcr_pair, "TCR paired alpha-beta"),
  `BCR heavy-only` = process_cells(bcr_heavy, "BCR heavy-only"),
  `BCR paired heavy-light` = process_cells(bcr_pair, "BCR paired heavy-light")
)

metrics <- bind_rows(lapply(sets, function(x) bind_rows(x$raw, x$rare)))
clone_counts <- bind_rows(lapply(names(sets), function(nm) sets[[nm]]$clone_counts %>% mutate(receptor_definition = nm)))
write_csv(metrics, file.path(outdir, "Table_HI_exact_clonotype_definition_metrics_by_participant.csv"))
write_csv(clone_counts, file.path(outdir, "Table_HI_exact_clonotype_counts.csv"))

rank_biserial <- function(x, y) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  u <- as.numeric(wilcox.test(x, y, exact = FALSE)$statistic)
  2 * u / (length(x) * length(y)) - 1
}
bootstrap_rbc <- function(x, y, B = 1000L) {
  z <- replicate(B, rank_biserial(sample(x, length(x), TRUE), sample(y, length(y), TRUE)))
  quantile(z, c(.025, .975), na.rm = TRUE)
}
test_block <- function(d) {
  contrasts <- list(c("CD", "Control"), c("UC", "Control"), c("CD", "UC"))
  rows <- list(); k <- 1L
  for (metric in c("clonality", "gini", "expanded_cell_fraction", "expanded_clonotype_fraction")) for (cc in contrasts) {
    x <- d[d$Diagnosis1 == cc[1], metric, drop = TRUE]; y <- d[d$Diagnosis1 == cc[2], metric, drop = TRUE]
    x <- x[is.finite(x)]; y <- y[is.finite(y)]
    if (length(x) < 3 || length(y) < 3) next
    ci <- bootstrap_rbc(x, y)
    rows[[k]] <- tibble(metric = metric, contrast = paste(cc, collapse = " vs "), n_group1 = length(x), n_group2 = length(y),
                        median_group1 = median(x), median_group2 = median(y), rank_biserial = rank_biserial(x, y),
                        ci_low = ci[1], ci_high = ci[2], p_value = wilcox.test(x, y, exact = FALSE)$p.value)
    k <- k + 1L
  }
  bind_rows(rows)
}

tests <- metrics %>%
  group_by(receptor_definition, analysis, threshold, downsample_depth) %>%
  group_modify(~test_block(.x)) %>% ungroup()
tests$p_adj <- p.adjust(tests$p_value, "BH")
write_csv(tests, file.path(outdir, "Table_HI_exact_clonotype_definition_pairwise_tests.csv"))

plot_data <- tests %>%
  filter(metric %in% c("clonality", "expanded_cell_fraction")) %>%
  mutate(
    metric = recode(metric, clonality = "Clonality", expanded_cell_fraction = "Expanded-cell fraction"),
    setting = ifelse(analysis == "Original depth", paste0("Threshold >=", threshold), paste0("Rarefied n=", downsample_depth)),
    receptor_definition = factor(receptor_definition, levels = c("TCR beta-only", "TCR paired alpha-beta", "BCR heavy-only", "BCR paired heavy-light")),
    contrast = factor(contrast, levels = c("CD vs Control", "UC vs Control", "CD vs UC"))
  )

p <- ggplot(plot_data, aes(rank_biserial, setting, xmin = ci_low, xmax = ci_high, color = contrast)) +
  geom_vline(xintercept = 0, linetype = 2, color = "grey60") +
  geom_errorbar(orientation = "y", width = .18, position = position_dodge(width = .55)) +
  geom_point(position = position_dodge(width = .55), size = 1.8) +
  facet_grid(metric ~ receptor_definition, scales = "free_y", space = "free_y") +
  scale_color_manual(values = c("CD vs Control" = "#2B6CB0", "UC vs Control" = "#C2415D", "CD vs UC" = "#555555")) +
  labs(
    title = "Clonal expansion is evaluated across exact receptor definitions and sampling depth",
    subtitle = "Clonotypes include identical V and J calls plus CDR3 amino-acid sequence; bars are bootstrap 95% confidence intervals",
    x = "Rank-biserial effect (first group higher ->)", y = NULL, color = NULL
  ) +
  theme_classic(base_size = 9) +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .35), strip.background = element_blank(),
        strip.text = element_text(face = "bold", size = 8), legend.position = "bottom", axis.text = element_text(color = "black"))
ggsave(file.path(outdir, "Figure_HI3_exact_clonotype_robustness.pdf"), p, width = 12.5, height = 7.8, device = cairo_pdf)
ggsave(file.path(outdir, "Figure_HI3_exact_clonotype_robustness.png"), p, width = 12.5, height = 7.8, dpi = 400)

manifest <- tibble(
  receptor_definition = names(sets),
  cells = c(nrow(tcr_beta), nrow(tcr_pair), nrow(bcr_heavy), nrow(bcr_pair)),
  participants = c(n_distinct(tcr_beta$SampleID), n_distinct(tcr_pair$SampleID), n_distinct(bcr_heavy$SampleID), n_distinct(bcr_pair$SampleID)),
  rarefaction_depth = vapply(sets, function(x) x$target, numeric(1)),
  definition = c("TRBV+TRBJ+CDR3aa", "TRAV+TRAJ+CDR3aa and TRBV+TRBJ+CDR3aa",
                 "IGHV+IGHJ+CDR3aa", "IGHV+IGHJ+CDR3aa and IGLV/IGKV+IGLJ/IGKJ+CDR3aa")
)
write_csv(manifest, file.path(outdir, "exact_clonotype_definition_manifest.csv"))
message("Completed exact clonotype-definition sensitivity analysis.")

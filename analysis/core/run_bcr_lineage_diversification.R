#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(igraph)
})

set.seed(20260824)
outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses/Priority Analyses"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
path <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/IBDBCR.rds"
obj <- readRDS(path)

meta <- obj$meta %>% mutate(SampleID = as.character(SampleID), Diagnosis1 = as.character(Diagnosis1))
meta <- meta %>% select(any_of(c("SampleID", "Diagnosis1", "Age", "Sex", "Batch", "Calprotectin", "Inflammation1"))) %>% distinct(SampleID, .keep_all = TRUE)

valid <- function(x) !is.na(x) & trimws(as.character(x)) != "" & toupper(trimws(as.character(x))) != "NA"
strip_allele <- function(x) sub("\\*.*$", "", as.character(x))

records <- bind_rows(lapply(names(obj$data), function(nm) {
  z <- obj$data[[nm]]
  sid <- sub("_BCR_AIRR$", "", nm)
  needed <- c("V.name", "J.name", "CDR3.nt", "CDR3.aa", "FR1.nt", "CDR1.nt", "FR2.nt", "CDR2.nt", "FR3.nt",
              "FR1.aa", "CDR1.aa", "FR2.aa", "CDR2.aa", "FR3.aa")
  for (v in setdiff(needed, names(z))) z[[v]] <- NA_character_
  z %>% transmute(
    SampleID = sid, V = strip_allele(V.name), J = strip_allele(J.name),
    cdr3_nt = toupper(as.character(`CDR3.nt`)), cdr3_aa = toupper(as.character(`CDR3.aa`)),
    v_nt = paste0(FR1.nt, CDR1.nt, FR2.nt, CDR2.nt, FR3.nt),
    cdr_aa = paste0(CDR1.aa, CDR2.aa), fwr_aa = paste0(FR1.aa, FR2.aa, FR3.aa)
  )
})) %>%
  left_join(meta, by = "SampleID") %>%
  filter(grepl("^IGHV", V), grepl("^IGHJ", J), valid(cdr3_aa), grepl("^[A-Z]+$", cdr3_aa),
         valid(SampleID), Diagnosis1 %in% c("Control", "UC", "CD")) %>%
  distinct(SampleID, V, J, cdr3_nt, cdr3_aa, v_nt, cdr_aa, fwr_aa, .keep_all = TRUE) %>%
  mutate(cdr3_len = nchar(cdr3_aa), seed_group = paste(SampleID, V, J, cdr3_len, sep = "|||"))

hamming_norm <- function(a, b) {
  if (is.na(a) || is.na(b) || nchar(a) != nchar(b) || nchar(a) == 0) return(NA_real_)
  sum(strsplit(a, "")[[1]] != strsplit(b, "")[[1]]) / nchar(a)
}

cluster_group <- function(d, threshold = .15) {
  seqs <- d$cdr3_aa
  n <- length(seqs)
  if (n == 1L) return(rep(1L, 1L))
  dm <- as.matrix(adist(seqs, seqs)) / nchar(seqs[1])
  diag(dm) <- Inf
  ed <- which(dm <= threshold, arr.ind = TRUE)
  ed <- ed[ed[, 1] < ed[, 2], , drop = FALSE]
  if (!nrow(ed)) return(seq_len(n))
  g <- make_empty_graph(n = n, directed = FALSE)
  g <- add_edges(g, as.vector(t(ed)))
  components(g)$membership
}

message("Clustering participant-specific heavy-chain lineages...")
split_groups <- split(records, records$seed_group)
lineage_records <- bind_rows(lapply(split_groups, function(d) {
  d$component <- cluster_group(d)
  d$lineage_id <- paste(d$seed_group, d$component, sep = "|||L")
  d
}))

consensus_string <- function(x) {
  x <- x[valid(x)]
  if (!length(x)) return(NA_character_)
  lens <- nchar(x); L <- as.integer(names(sort(table(lens), decreasing = TRUE))[1])
  x <- x[lens == L]
  m <- do.call(rbind, strsplit(x, ""))
  paste0(apply(m, 2, function(v) names(sort(table(v), decreasing = TRUE))[1]), collapse = "")
}
mean_divergence <- function(x) {
  x <- unique(x[valid(x)])
  if (length(x) < 2) return(0)
  cons <- consensus_string(x)
  x <- x[nchar(x) == nchar(cons)]
  if (!length(x)) return(NA_real_)
  mean(vapply(x, hamming_norm, numeric(1), b = cons), na.rm = TRUE)
}

lineages <- lineage_records %>% group_by(SampleID, Diagnosis1, lineage_id, V, J, cdr3_len) %>%
  summarise(
    n_unique_cdr3 = n_distinct(cdr3_aa), n_unique_v_nt = n_distinct(v_nt[valid(v_nt)]),
    cdr3_aa_divergence = mean_divergence(cdr3_aa), v_region_nt_divergence = mean_divergence(v_nt),
    cdr_aa_divergence = mean_divergence(cdr_aa), fwr_aa_divergence = mean_divergence(fwr_aa),
    cdr_minus_fwr_divergence = cdr_aa_divergence - fwr_aa_divergence,
    .groups = "drop"
  ) %>% mutate(diversified = n_unique_cdr3 >= 2 | n_unique_v_nt >= 2)

participant <- lineages %>% group_by(SampleID, Diagnosis1) %>% summarise(
  n_lineages = n(), n_diversified_lineages = sum(diversified), fraction_diversified_lineages = mean(diversified),
  median_lineage_cdr3_variants = median(n_unique_cdr3), max_lineage_cdr3_variants = max(n_unique_cdr3),
  median_v_region_nt_divergence = median(v_region_nt_divergence[diversified], na.rm = TRUE),
  median_cdr_minus_fwr_divergence = median(cdr_minus_fwr_divergence[diversified], na.rm = TRUE),
  .groups = "drop") %>%
  left_join(records %>% count(SampleID, name = "n_unique_heavy_records"), by = "SampleID") %>%
  left_join(meta, by = c("SampleID", "Diagnosis1"))
participant[!is.finite(participant$median_v_region_nt_divergence), "median_v_region_nt_divergence"] <- NA
participant[!is.finite(participant$median_cdr_minus_fwr_divergence), "median_cdr_minus_fwr_divergence"] <- NA

metrics <- c("fraction_diversified_lineages", "median_lineage_cdr3_variants", "max_lineage_cdr3_variants",
             "median_v_region_nt_divergence", "median_cdr_minus_fwr_divergence")
contrasts <- list(c("CD", "Control"), c("UC", "Control"), c("CD", "UC"))
tests <- bind_rows(lapply(metrics, function(v) bind_rows(lapply(contrasts, function(g) {
  a <- participant[[v]][participant$Diagnosis1 == g[1]]; b <- participant[[v]][participant$Diagnosis1 == g[2]]
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  tibble(metric = v, contrast = paste(g, collapse = " vs "), n1 = length(a), n2 = length(b),
         median1 = median(a), median2 = median(b),
         p_value = if(length(a) >= 3 && length(b) >= 3) wilcox.test(a, b, exact = FALSE)$p.value else NA_real_)
})))) %>% group_by(metric) %>% mutate(FDR_within_metric = p.adjust(p_value, "BH")) %>% ungroup() %>%
  mutate(FDR_global = p.adjust(p_value, "BH"))

# Depth-adjusted linear sensitivity models; coefficients are diagnosis contrasts after
# accounting for unique heavy-sequence depth, age, sex, and batch when estimable.
adjusted <- bind_rows(lapply(metrics, function(v) {
  d <- participant %>% filter(is.finite(.data[[v]]), is.finite(n_unique_heavy_records), Diagnosis1 %in% c("Control", "UC", "CD")) %>%
    mutate(Diagnosis1 = relevel(factor(Diagnosis1), "Control"), Sex = factor(Sex), Batch = factor(Batch))
  rhs <- c("Diagnosis1", "log10(n_unique_heavy_records + 1)")
  if (sum(is.finite(d$Age)) > .8*nrow(d)) rhs <- c(rhs, "Age")
  if (n_distinct(na.omit(d$Sex)) > 1) rhs <- c(rhs, "Sex")
  if (n_distinct(na.omit(d$Batch)) > 1) rhs <- c(rhs, "Batch")
  fit <- lm(as.formula(paste(v, "~", paste(rhs, collapse = "+"))), data = d)
  co <- summary(fit)$coefficients
  ix <- grep("^Diagnosis1", rownames(co))
  tibble(metric = v, term = rownames(co)[ix], estimate = co[ix, 1], std_error = co[ix, 2], p_value = co[ix, 4], n = nobs(fit))
})) %>% mutate(FDR = p.adjust(p_value, "BH"))

write_csv(lineages, file.path(outdir, "Table_PA3_BCR_lineages.csv"))
write_csv(participant, file.path(outdir, "Table_PA3_BCR_lineage_diversification_by_participant.csv"))
write_csv(tests, file.path(outdir, "Table_PA3_BCR_lineage_group_tests.csv"))
write_csv(adjusted, file.path(outdir, "Table_PA3_BCR_lineage_depth_adjusted_models.csv"))

plotd <- participant %>% select(SampleID, Diagnosis1, all_of(metrics)) %>% pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
    fraction_diversified_lineages = "Diversified lineage fraction",
    median_lineage_cdr3_variants = "Median CDR3 variants / lineage",
    max_lineage_cdr3_variants = "Maximum CDR3 variants / lineage",
    median_v_region_nt_divergence = "Median V-region nucleotide divergence",
    median_cdr_minus_fwr_divergence = "CDR minus framework AA divergence"))
p <- ggplot(plotd, aes(Diagnosis1, value, color = Diagnosis1)) + geom_boxplot(outlier.shape = NA, width = .55) +
  geom_jitter(width = .14, alpha = .5, size = .9) + facet_wrap(~metric, scales = "free_y", ncol = 3) +
  scale_color_manual(values = c(Control = "#777777", UC = "#C2415D", CD = "#2B6CB0")) +
  labs(x = NULL, y = NULL, title = "Participant-specific BCR lineage diversification") + theme_classic(base_size = 10) +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"), legend.position = "none")
ggsave(file.path(outdir, "Figure_PA2_BCR_lineage_diversification.png"), p, width = 9.2, height = 6.4, dpi = 600)
ggsave(file.path(outdir, "Figure_PA2_BCR_lineage_diversification.pdf"), p, width = 9.2, height = 6.4)

writeLines(c(
  "Lineages were defined within participants by shared IGHV, IGHJ, and CDR3 amino-acid length, followed by single-linkage clustering at <=15% CDR3 amino-acid distance.",
  "All metrics are based on unique heavy-chain sequence records rather than the AIRR Clones abundance field.",
  "V-region and regional amino-acid divergence are measured from the within-lineage consensus, not a germline sequence.",
  "Accordingly, CDR-minus-framework divergence is an exploratory regional-diversification proxy and must not be described as formal BASELINe selection inference."
), file.path(outdir, "Table_PA3_BCR_lineage_method_notes.txt"))
message("BCR lineage analysis complete")

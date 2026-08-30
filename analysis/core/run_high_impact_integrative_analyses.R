#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

set.seed(20260824)

outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

tcr_root <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 2 TCR"
tcr_map_root <- file.path(tcr_root, "TCR Architecture Analyses/outputs/updated_tcr_clonotype_state_mapping_20260628_123244")
bcr_root <- "C:/path/to/private-user-home/OneDrive/Desktop/BCR Module Scores"

paths <- list(
  tcr_rds = file.path(tcr_root, "IBDTCR.rds"),
  bcr_rds = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/IBDBCR.rds",
  tcr_metrics = file.path(tcr_root, "clonality_metrics_IBDTCR_Immunarch.csv"),
  bcr_metrics = file.path(bcr_root, "outputs_manuscript_cd_uc_control/tables/figure3_sample_diversity.csv"),
  bcr_cells = file.path(bcr_root, "outputs_manuscript_cd_uc_control/clonotype_state_mapping/cell_to_clonotype_state_map.csv"),
  bcr_modules = file.path(bcr_root, "outputs_bcr_total_clonotypes_gene_modules/tables/sample_level_bcr_clonotype_module_scores.csv"),
  bcr_isotype = file.path(bcr_root, "outputs_manuscript_cd_uc_control/tables/figure5_isotype_by_sample.csv"),
  bcr_shm = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 3 BCR/BCR Architecture Analyses/SHM/outputs_manuscript_cd_uc_control/tables/figure6_shm_immunarch_by_sample.csv",
  cd4_seu = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  cd8_seu = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds"
)

stopifnot(all(vapply(paths, file.exists, logical(1))))

theme_manuscript <- function(base_size = 10) {
  theme_classic(base_size = base_size) +
    theme(
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black"),
      plot.title = element_text(face = "bold", size = base_size + 1),
      plot.subtitle = element_text(size = base_size - 1),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold"),
      legend.title = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.35)
    )
}

diag_cols <- c(Control = "#7A7A7A", CD = "#2B6CB0", UC = "#C2415D")

read_tcr_cells <- function() {
  message("Loading T-cell metadata and paired clonotype maps...")
  load_one <- function(seu_path, lineage) {
    seu <- readRDS(seu_path)
    md <- as.data.frame(seu[[]], stringsAsFactors = FALSE)
    md$cell <- rownames(md)
    keep <- intersect(c("cell", "SampleID", "PatientID", "Batch", "Age", "Sex", "Calprotectin", "Inflammation1", "Biologic"), names(md))
    md <- md[, keep, drop = FALSE]
    rm(seu); gc(verbose = FALSE)
    mp <- read_csv(file.path(tcr_map_root, lineage, "paired_alpha_beta/cell_level_clonotype_annotationL2.csv"), show_col_types = FALSE)
    mp %>%
      left_join(md, by = "cell") %>%
      mutate(lineage = lineage)
  }
  bind_rows(load_one(paths$cd4_seu, "CD4"), load_one(paths$cd8_seu, "CD8")) %>%
    filter(!is.na(SampleID), !is.na(clonotype_id), clonotype_id != "")
}

entropy_from_counts <- function(n) {
  p <- n / sum(n)
  -sum(p[p > 0] * log(p[p > 0]))
}

make_clone_state_summary <- function(cells, sample_col, diagnosis_col, clone_col, state_col, modality, nsim = 250L) {
  cells <- cells %>%
    transmute(
      SampleID = as.character(.data[[sample_col]]),
      Diagnosis1 = as.character(.data[[diagnosis_col]]),
      clonotype_id = as.character(.data[[clone_col]]),
      state = as.character(.data[[state_col]])
    ) %>%
    filter(!is.na(SampleID), SampleID != "", !is.na(clonotype_id), clonotype_id != "", !is.na(state), state != "")

  state_counts <- cells %>% count(SampleID, Diagnosis1, clonotype_id, state, name = "n_cells")
  clone_summary <- state_counts %>%
    group_by(SampleID, Diagnosis1, clonotype_id) %>%
    summarise(
      total_cells = sum(n_cells),
      n_states = n(),
      dominant_state = state[which.max(n_cells)][1],
      dominant_state_frac = max(n_cells) / sum(n_cells),
      state_entropy = entropy_from_counts(n_cells),
      effective_states = exp(state_entropy),
      .groups = "drop"
    )

  # Participant- and size-specific null: random state assignment under each participant's
  # observed state availability. This controls the strong mechanical effect of clone size.
  null_rows <- vector("list", 0)
  idx <- 1L
  for (sid in unique(clone_summary$SampleID)) {
    csub <- cells[cells$SampleID == sid, , drop = FALSE]
    probs <- prop.table(table(csub$state))
    sizes <- sort(unique(clone_summary$total_cells[clone_summary$SampleID == sid]))
    for (sz in sizes) {
      sims <- replicate(nsim, as.vector(rmultinom(1, size = sz, prob = probs)))
      if (is.null(dim(sims))) sims <- matrix(sims, ncol = 1)
      sim_nstates <- colSums(sims > 0)
      sim_entropy <- apply(sims, 2, entropy_from_counts)
      null_rows[[idx]] <- tibble(
        SampleID = sid,
        total_cells = sz,
        null_n_states = mean(sim_nstates),
        null_n_states_sd = sd(sim_nstates),
        null_entropy = mean(sim_entropy),
        null_entropy_sd = sd(sim_entropy)
      )
      idx <- idx + 1L
    }
  }
  null_df <- bind_rows(null_rows)
  clone_summary <- clone_summary %>%
    left_join(null_df, by = c("SampleID", "total_cells")) %>%
    mutate(
      breadth_excess = n_states - null_n_states,
      entropy_excess = state_entropy - null_entropy,
      multistate = n_states > 1,
      modality = modality
    )

  participant_summary <- bind_rows(lapply(c(2L, 5L), function(thr) {
    clone_summary %>%
      filter(total_cells >= thr) %>%
      group_by(SampleID, Diagnosis1) %>%
      summarise(
        threshold = thr,
        n_clonotypes = n(),
        median_clone_size = median(total_cells),
        multistate_fraction = mean(multistate),
        median_n_states = median(n_states),
        median_breadth_excess = median(breadth_excess),
        median_entropy_excess = median(entropy_excess),
        median_dominant_state_fraction = median(dominant_state_frac),
        .groups = "drop"
      )
  })) %>% mutate(modality = modality)

  list(cells = cells, clone = clone_summary, participant = participant_summary)
}

rank_biserial <- function(x, y) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  if (length(x) == 0L || length(y) == 0L) return(NA_real_)
  u <- as.numeric(wilcox.test(x, y, exact = FALSE)$statistic)
  2 * u / (length(x) * length(y)) - 1
}

bootstrap_rbc <- function(x, y, B = 1000L) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  vals <- replicate(B, rank_biserial(sample(x, length(x), TRUE), sample(y, length(y), TRUE)))
  quantile(vals, c(0.025, 0.975), na.rm = TRUE)
}

pairwise_tests <- function(d, value_cols, group_col = "Diagnosis1", extra = list()) {
  contrasts <- list(c("CD", "Control"), c("UC", "Control"), c("CD", "UC"))
  rows <- list(); k <- 1L
  for (v in value_cols) for (cc in contrasts) {
    x <- d[d[[group_col]] == cc[1], v, drop = TRUE]
    y <- d[d[[group_col]] == cc[2], v, drop = TRUE]
    x <- x[is.finite(x)]; y <- y[is.finite(y)]
    if (length(x) < 3L || length(y) < 3L) next
    ci <- bootstrap_rbc(x, y)
    rows[[k]] <- tibble(
      metric = v, contrast = paste(cc, collapse = " vs "),
      n_group1 = length(x), n_group2 = length(y),
      median_group1 = median(x), median_group2 = median(y),
      rank_biserial = rank_biserial(x, y), ci_low = ci[1], ci_high = ci[2],
      p_value = wilcox.test(x, y, exact = FALSE)$p.value
    )
    k <- k + 1L
  }
  out <- bind_rows(rows)
  out$p_adj <- p.adjust(out$p_value, method = "BH")
  for (nm in names(extra)) out[[nm]] <- extra[[nm]]
  out
}

message("Building paired clonotype state-breadth datasets...")
tcr_cells <- read_tcr_cells()
tcr_state <- make_clone_state_summary(tcr_cells, "SampleID", "Diagnosis1", "clonotype_id", "AnnotationLevel2", "TCR")

bcr_cells <- read_csv(paths$bcr_cells, show_col_types = FALSE)
bcr_state <- make_clone_state_summary(bcr_cells, "SampleID", "Diagnosis1", "clonotype_id", "BcellState", "BCR")

write_csv(tcr_state$clone, file.path(outdir, "Table_HI_TCR_clonotype_state_breadth.csv"))
write_csv(bcr_state$clone, file.path(outdir, "Table_HI_BCR_clonotype_state_breadth.csv"))
state_participant <- bind_rows(tcr_state$participant, bcr_state$participant)
write_csv(state_participant, file.path(outdir, "Table_HI_clonotype_state_breadth_by_participant.csv"))
state_tests <- state_participant %>%
  group_by(modality, threshold) %>%
  group_modify(~pairwise_tests(.x, c("multistate_fraction", "median_breadth_excess", "median_entropy_excess"))) %>%
  ungroup()
write_csv(state_tests, file.path(outdir, "Table_HI_clonotype_state_breadth_pairwise_tests.csv"))

# Export clone-pair tables for matched-cell machine-learning analysis.
tcr_pair_export <- tcr_cells %>%
  count(SampleID, Diagnosis1, clonotype_id, name = "n_cells") %>%
  separate(clonotype_id, into = c("chain1_cdr3", "chain2_cdr3"), sep = "\\|", extra = "merge", fill = "right", remove = FALSE) %>%
  filter(!is.na(chain1_cdr3), chain1_cdr3 != "", !is.na(chain2_cdr3), chain2_cdr3 != "") %>%
  mutate(modality = "TCR")
bcr_pair_export <- bcr_cells %>%
  count(SampleID, Diagnosis1, clonotype_id, name = "n_cells") %>%
  separate(clonotype_id, into = c("v_call", "j_call", "chain1_cdr3", "chain2_cdr3"), sep = "\\|", extra = "merge", fill = "right", remove = FALSE) %>%
  filter(!is.na(chain1_cdr3), chain1_cdr3 != "", !is.na(chain2_cdr3), chain2_cdr3 != "") %>%
  mutate(modality = "BCR")
write_csv(tcr_pair_export, file.path(outdir, "paired_TCR_clonotypes_for_nested_ML.csv"))
write_csv(bcr_pair_export, file.path(outdir, "paired_BCR_clonotypes_for_nested_ML.csv"))

message("Building participant-level TCR-BCR coordination table...")
tcr_metrics <- read_csv(paths$tcr_metrics, show_col_types = FALSE) %>%
  transmute(SampleID = as.character(PatientID), Diagnosis1 = Diagnosis,
            tcr_clonality = Clonality, tcr_gini = Gini,
            tcr_shannon = Shannon, tcr_inv_simpson = InvSimp)
bcr_metrics <- read_csv(paths$bcr_metrics, show_col_types = FALSE) %>%
  transmute(SampleID, Diagnosis1, bcr_gini = gini, bcr_shannon = shannon,
            bcr_inv_simpson = inv_simpson, bcr_total_cells = total_clonal_cells,
            bcr_singleton_fraction = singleton_frac)

tcr_cell_features <- tcr_cells %>%
  left_join(tcr_state$clone %>% select(SampleID, clonotype_id, total_cells), by = c("SampleID", "clonotype_id")) %>%
  mutate(expanded_calc = total_cells >= 2) %>%
  group_by(SampleID, Diagnosis1) %>%
  summarise(
    tcr_paired_cells = n(),
    tcr_expanded_cell_fraction = mean(expanded_calc),
    tcr_cytotoxic_expanded = ifelse(any(expanded_calc), mean(module_cytotoxic_TNK[expanded_calc], na.rm = TRUE), NA_real_),
    tcr_gut_homing_expanded = ifelse(any(expanded_calc), mean(module_gut_homing_trafficking[expanded_calc], na.rm = TRUE), NA_real_),
    .groups = "drop"
  )
tcr_breadth <- tcr_state$participant %>% filter(threshold == 2) %>%
  transmute(SampleID, tcr_multistate_clone_fraction = multistate_fraction,
            tcr_breadth_excess = median_breadth_excess)

bcr_cell_features <- bcr_cells %>%
  left_join(bcr_state$clone %>% select(SampleID, clonotype_id, total_cells), by = c("SampleID", "clonotype_id")) %>%
  mutate(expanded_calc = total_cells >= 2) %>%
  group_by(SampleID, Diagnosis1) %>%
  summarise(bcr_paired_cells = n(), bcr_expanded_cell_fraction = mean(expanded_calc), .groups = "drop")
bcr_breadth <- bcr_state$participant %>% filter(threshold == 2) %>%
  transmute(SampleID, bcr_multistate_clone_fraction = multistate_fraction,
            bcr_breadth_excess = median_breadth_excess)

bcr_modules <- read_csv(paths$bcr_modules, show_col_types = FALSE) %>%
  transmute(
    SampleID = sample_id,
    bcr_IgA_mucosal_module = module_IgA_mucosal_plasma_cell,
    bcr_plasma_differentiation_module = module_plasmablast_plasma_cell_differentiation,
    bcr_IgG_inflammatory_module = module_IgG_inflammatory_plasma_cell,
    bcr_BAFF_APRIL_module = module_BAFF_APRIL_survival_response
  )

iso <- read_csv(paths$bcr_isotype, show_col_types = FALSE) %>%
  filter(isotype %in% c("IgA", "IgG", "IgM", "IgD")) %>%
  group_by(SampleID, Diagnosis1) %>%
  summarise(
    bcr_switched_fraction = sum(n[isotype %in% c("IgA", "IgG")]) / sum(n),
    bcr_IgA_fraction = sum(n[isotype == "IgA"]) / sum(n),
    bcr_IgG_fraction = sum(n[isotype == "IgG"]) / sum(n),
    .groups = "drop"
  )
bcr_shm <- read_csv(paths$bcr_shm, show_col_types = FALSE) %>%
  transmute(SampleID, bcr_SHM_rate = SHM_Rate)

meta <- readRDS(paths$tcr_rds)$meta %>%
  transmute(SampleID = as.character(SampleID), Diagnosis1, Batch, Age = as.numeric(Age), Sex,
            tcr_total_cells = NA_real_)

# Fill total TCR observations directly from clonotype counts.
tcr_obj_tmp <- readRDS(paths$tcr_rds)
tcr_depth <- tibble(
  Sample = names(tcr_obj_tmp$data),
  tcr_total_cells = vapply(tcr_obj_tmp$data, function(z) sum(z$Clones, na.rm = TRUE), numeric(1))
) %>% mutate(SampleID = sub("_TCR_AIRR$", "", Sample)) %>% select(-Sample)
rm(tcr_obj_tmp); gc(verbose = FALSE)
meta <- meta %>% select(-tcr_total_cells) %>% left_join(tcr_depth, by = "SampleID")

participant <- meta %>%
  left_join(tcr_metrics, by = c("SampleID", "Diagnosis1")) %>%
  left_join(tcr_cell_features, by = c("SampleID", "Diagnosis1")) %>%
  left_join(tcr_breadth, by = "SampleID") %>%
  left_join(bcr_metrics, by = c("SampleID", "Diagnosis1")) %>%
  left_join(bcr_cell_features, by = c("SampleID", "Diagnosis1")) %>%
  left_join(bcr_breadth, by = "SampleID") %>%
  left_join(bcr_modules, by = "SampleID") %>%
  left_join(iso, by = c("SampleID", "Diagnosis1")) %>%
  left_join(bcr_shm, by = "SampleID")
write_csv(participant, file.path(outdir, "Table_HI_integrated_participant_features.csv"))

partial_spearman <- function(d, x, y, B = 500L) {
  keep <- complete.cases(d[, c(x, y, "Diagnosis1", "Age", "Sex", "tcr_total_cells", "bcr_total_cells")])
  z <- d[keep, , drop = FALSE]
  calc <- function(q) {
    rx <- rank(q[[x]], ties.method = "average")
    ry <- rank(q[[y]], ties.method = "average")
    cov <- model.matrix(~ Diagnosis1 + Age + Sex + log10(tcr_total_cells + 1) + log10(bcr_total_cells + 1), data = q)
    ex <- residuals(lm.fit(cov, rx)); ey <- residuals(lm.fit(cov, ry))
    cor(ex, ey)
  }
  rho <- calc(z)
  # Diagnosis-stratified bootstrap preserves case mix.
  boots <- replicate(B, {
    idx <- unlist(lapply(split(seq_len(nrow(z)), z$Diagnosis1), function(ii) sample(ii, length(ii), replace = TRUE)))
    tryCatch(calc(z[idx, , drop = FALSE]), error = function(e) NA_real_)
  })
  cov <- model.matrix(~ Diagnosis1 + Age + Sex + log10(tcr_total_cells + 1) + log10(bcr_total_cells + 1), data = z)
  ex <- residuals(lm.fit(cov, rank(z[[x]]))); ey <- residuals(lm.fit(cov, rank(z[[y]])))
  p <- cor.test(ex, ey, method = "pearson")$p.value
  tibble(tcr_metric = x, bcr_metric = y, n = nrow(z), partial_rho = rho,
         ci_low = quantile(boots, .025, na.rm = TRUE), ci_high = quantile(boots, .975, na.rm = TRUE), p_value = p)
}

tcr_corr_vars <- c("tcr_clonality", "tcr_gini", "tcr_expanded_cell_fraction", "tcr_multistate_clone_fraction", "tcr_cytotoxic_expanded")
bcr_corr_vars <- c("bcr_gini", "bcr_expanded_cell_fraction", "bcr_multistate_clone_fraction", "bcr_switched_fraction",
                   "bcr_SHM_rate", "bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module", "bcr_BAFF_APRIL_module")
corr <- bind_rows(lapply(tcr_corr_vars, function(x) bind_rows(lapply(bcr_corr_vars, function(y) partial_spearman(participant, x, y)))))
corr$p_adj <- p.adjust(corr$p_value, method = "BH")
write_csv(corr, file.path(outdir, "Table_HI_partial_spearman_TCR_BCR_coordination.csv"))

message("Running expansion-threshold and depth-normalization sensitivity analyses...")
gini_counts <- function(counts) {
  x <- sort(as.numeric(counts)); n <- length(x)
  if (!n || sum(x) == 0) return(NA_real_)
  sum((2 * seq_len(n) - n - 1) * x) / (n * sum(x))
}
clone_metrics <- function(counts, threshold = 2L) {
  counts <- as.numeric(counts[counts > 0])
  p <- counts / sum(counts)
  sh <- -sum(p * log(p))
  ns <- if (length(counts) > 1) sh / log(length(counts)) else 0
  c(total_cells = sum(counts), richness = length(counts), clonality = 1 - ns,
    gini = gini_counts(counts), expanded_cell_fraction = sum(counts[counts >= threshold]) / sum(counts),
    expanded_clonotype_fraction = mean(counts >= threshold))
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
process_paired_repertoire <- function(pair_df, modality, B = 100L) {
  md <- pair_df %>% distinct(SampleID, Diagnosis1)
  clone_list <- split(pair_df$n_cells, pair_df$SampleID)
  raw <- bind_rows(lapply(c(2L, 3L, 5L), function(thr) {
    bind_rows(lapply(names(clone_list), function(nm) {
      bind_cols(tibble(SampleID = nm, threshold = thr), tibble::as_tibble_row(clone_metrics(clone_list[[nm]], thr)))
    }))
  })) %>%
    mutate(modality = modality) %>%
    left_join(md, by = "SampleID")
  totals <- raw %>% filter(threshold == 2) %>% pull(total_cells)
  target <- max(10, floor(as.numeric(quantile(totals, .10, na.rm = TRUE)) / 10) * 10)
  rare <- bind_rows(lapply(names(clone_list), function(nm) {
    counts <- clone_list[[nm]]
    if (sum(counts) < target) return(NULL)
    sims <- bind_rows(lapply(seq_len(B), function(i) tibble::as_tibble_row(clone_metrics(downsample_counts(counts, target), 2L))))
    sims %>% summarise(across(everything(), \(x) mean(x, na.rm = TRUE))) %>% mutate(SampleID = nm)
  })) %>%
    mutate(modality = modality, threshold = 2L, downsample_depth = target) %>%
    left_join(md, by = "SampleID")
  list(raw = raw, rare = rare, target = target)
}

tcr_rob <- process_paired_repertoire(tcr_pair_export, "TCR", B = 100L)
bcr_rob <- process_paired_repertoire(bcr_pair_export, "BCR", B = 100L)
rob_raw <- bind_rows(tcr_rob$raw, bcr_rob$raw)
rob_rare <- bind_rows(tcr_rob$rare, bcr_rob$rare)
write_csv(rob_raw, file.path(outdir, "Table_HI_expansion_threshold_metrics_by_participant.csv"))
write_csv(rob_rare, file.path(outdir, "Table_HI_depth_normalized_metrics_by_participant.csv"))

rob_tests_raw <- rob_raw %>%
  group_by(modality, threshold) %>%
  group_modify(~pairwise_tests(.x, c("clonality", "gini", "expanded_cell_fraction", "expanded_clonotype_fraction"))) %>%
  ungroup() %>% mutate(analysis = "Original depth")
rob_tests_rare <- rob_rare %>%
  group_by(modality, threshold, downsample_depth) %>%
  group_modify(~pairwise_tests(.x, c("clonality", "gini", "expanded_cell_fraction"))) %>%
  ungroup() %>% mutate(analysis = "Depth-normalized")
rob_tests <- bind_rows(rob_tests_raw, rob_tests_rare)
rob_tests$p_adj_global <- p.adjust(rob_tests$p_value, method = "BH")
write_csv(rob_tests, file.path(outdir, "Table_HI_repertoire_robustness_pairwise_tests.csv"))

# ------------------------ Figures ------------------------
message("Generating figures...")
pretty_metric <- c(
  tcr_clonality = "TCR clonality", tcr_gini = "TCR Gini", tcr_expanded_cell_fraction = "Expanded TCR-cell fraction",
  tcr_multistate_clone_fraction = "Multistate TCR-clone fraction", tcr_cytotoxic_expanded = "Expanded-TCR cytotoxic score",
  bcr_gini = "BCR Gini", bcr_expanded_cell_fraction = "Expanded BCR-cell fraction",
  bcr_multistate_clone_fraction = "Multistate BCR-clone fraction", bcr_switched_fraction = "Class-switched BCR fraction",
  bcr_SHM_rate = "BCR SHM rate", bcr_IgA_mucosal_module = "IgA mucosal module",
  bcr_plasma_differentiation_module = "Plasma-cell differentiation module", bcr_BAFF_APRIL_module = "BAFF/APRIL module"
)

corr_plot <- corr %>%
  mutate(
    tcr_label = factor(pretty_metric[tcr_metric], levels = rev(pretty_metric[tcr_corr_vars])),
    bcr_label = factor(pretty_metric[bcr_metric], levels = pretty_metric[bcr_corr_vars]),
    sig = case_when(p_adj < .01 ~ "**", p_adj < .05 ~ "*", TRUE ~ "")
  )
p_heat <- ggplot(corr_plot, aes(bcr_label, tcr_label, fill = partial_rho)) +
  geom_tile(color = "white", linewidth = .5) +
  geom_text(aes(label = paste0(sprintf("%.2f", partial_rho), sig)), size = 2.7) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0, limits = c(-1, 1), name = "Partial rho") +
  labs(title = "A  Within-participant TCR-BCR coordination", x = NULL, y = NULL,
       subtitle = "Partial Spearman correlations adjusted for diagnosis, age, sex, and receptor depth") +
  theme_manuscript(9) + theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank())

best_pairs <- corr %>% arrange(p_adj, desc(abs(partial_rho))) %>% slice_head(n = 3)
scatter_panel <- function(x, y, label) {
  ann <- corr %>% filter(tcr_metric == x, bcr_metric == y) %>% slice(1)
  ggplot(participant, aes(.data[[x]], .data[[y]], color = Diagnosis1)) +
    geom_point(alpha = .72, size = 1.6) +
    geom_smooth(method = "lm", se = TRUE, color = "#333333", fill = "#BDBDBD", linewidth = .5) +
    scale_color_manual(values = diag_cols, na.translate = FALSE) +
    labs(title = label, x = pretty_metric[[x]], y = pretty_metric[[y]],
         subtitle = sprintf("partial rho = %.2f; FDR = %.3g; n = %d", ann$partial_rho, ann$p_adj, ann$n)) +
    theme_manuscript(9) + theme(legend.position = "bottom")
}
scatter_list <- lapply(seq_len(nrow(best_pairs)), function(i) scatter_panel(best_pairs$tcr_metric[i], best_pairs$bcr_metric[i], paste0(LETTERS[i + 1], "  ", pretty_metric[[best_pairs$tcr_metric[i]]], " vs ", pretty_metric[[best_pairs$bcr_metric[i]]])))
fig1 <- p_heat / wrap_plots(scatter_list, nrow = 1) + plot_layout(heights = c(1.35, 1))
ggsave(file.path(outdir, "Figure_HI1_TCR_BCR_coordination.pdf"), fig1, width = 11, height = 8.5, device = cairo_pdf)
ggsave(file.path(outdir, "Figure_HI1_TCR_BCR_coordination.png"), fig1, width = 11, height = 8.5, dpi = 400)

state_long <- state_participant %>%
  filter(threshold %in% c(2, 5)) %>%
  mutate(Diagnosis1 = factor(Diagnosis1, levels = c("Control", "CD", "UC")), threshold_label = paste0("Clones >=", threshold, " cells"))
p_state1 <- ggplot(state_long, aes(Diagnosis1, median_breadth_excess, color = Diagnosis1)) +
  geom_hline(yintercept = 0, linetype = 2, color = "grey55") +
  geom_boxplot(outlier.shape = NA, width = .62, color = "black", fill = "white") +
  geom_jitter(width = .15, alpha = .55, size = 1.1) +
  facet_grid(modality ~ threshold_label, scales = "free_y") +
  scale_color_manual(values = diag_cols, guide = "none") +
  labs(title = "A  Clone-size-adjusted state breadth", x = NULL, y = "Median excess number of states per clonotype") + theme_manuscript(9)
p_state2 <- ggplot(state_long, aes(Diagnosis1, multistate_fraction, color = Diagnosis1)) +
  geom_boxplot(outlier.shape = NA, width = .62, color = "black", fill = "white") +
  geom_jitter(width = .15, alpha = .55, size = 1.1) +
  facet_grid(modality ~ threshold_label, scales = "free_y") +
  scale_color_manual(values = diag_cols, guide = "none") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "B  Multistate occupancy by expanded clonotypes", x = NULL, y = "Multistate clonotype fraction") + theme_manuscript(9)
state_effect <- state_tests %>%
  filter(metric %in% c("median_breadth_excess", "multistate_fraction")) %>%
  mutate(metric_label = recode(metric, median_breadth_excess = "Null-adjusted breadth", multistate_fraction = "Multistate fraction"),
         contrast = factor(contrast, levels = c("CD vs Control", "UC vs Control", "CD vs UC")))
p_state3 <- ggplot(state_effect, aes(rank_biserial, contrast, xmin = ci_low, xmax = ci_high, color = modality)) +
  geom_vline(xintercept = 0, linetype = 2, color = "grey60") +
  geom_errorbarh(height = .18, position = position_dodge(width = .5)) +
  geom_point(position = position_dodge(width = .5), size = 2) +
  facet_grid(metric_label ~ threshold, labeller = labeller(threshold = function(x) paste0("Threshold >=", x))) +
  scale_color_manual(values = c(TCR = "#7A3E9D", BCR = "#D17C20")) +
  labs(title = "C  Diagnosis effect sizes", x = "Rank-biserial effect (first group higher ->)", y = NULL) +
  theme_manuscript(9) + theme(legend.position = "bottom")
fig2 <- p_state1 / p_state2 / p_state3 + plot_layout(heights = c(1, 1, .9))
ggsave(file.path(outdir, "Figure_HI2_clonotype_state_breadth.pdf"), fig2, width = 10, height = 11, device = cairo_pdf)
ggsave(file.path(outdir, "Figure_HI2_clonotype_state_breadth.png"), fig2, width = 10, height = 11, dpi = 400)

rob_plot_data <- rob_tests %>%
  filter(metric %in% c("expanded_cell_fraction", "clonality", "gini")) %>%
  mutate(
    metric_label = recode(metric, expanded_cell_fraction = "Expanded-cell fraction", clonality = "Clonality", gini = "Gini coefficient"),
    analysis_label = ifelse(analysis == "Original depth", paste0("Threshold >=", threshold), paste0("Depth-normalized (n=", downsample_depth, ")")),
    analysis_label = factor(analysis_label, levels = unique(analysis_label))
  )
p_rob <- ggplot(rob_plot_data, aes(rank_biserial, analysis_label, xmin = ci_low, xmax = ci_high, color = contrast)) +
  geom_vline(xintercept = 0, linetype = 2, color = "grey60") +
  geom_errorbarh(height = .18, position = position_dodge(width = .55)) +
  geom_point(position = position_dodge(width = .55), size = 1.8) +
  facet_grid(metric_label ~ modality, scales = "free_y", space = "free_y") +
  scale_color_manual(values = c("CD vs Control" = "#2B6CB0", "UC vs Control" = "#C2415D", "CD vs UC" = "#5B5B5B")) +
  labs(title = "Repertoire effects are evaluated across expansion thresholds and normalized depth",
       subtitle = "Points are rank-biserial effects; bars are bootstrap 95% confidence intervals",
       x = "Rank-biserial effect (first group higher ->)", y = NULL) +
  theme_manuscript(9) + theme(legend.position = "bottom")
ggsave(file.path(outdir, "Figure_HI3_repertoire_robustness.pdf"), p_rob, width = 10.5, height = 8.5, device = cairo_pdf)
ggsave(file.path(outdir, "Figure_HI3_repertoire_robustness.png"), p_rob, width = 10.5, height = 8.5, dpi = 400)

manifest <- tibble(
  analysis = c("TCR-BCR coordination", "Clonotype state breadth", "Expansion-threshold robustness", "Depth-normalized repertoire robustness", "Paired-chain ML input export"),
  statistical_unit = c("Participant", "Participant summaries of clonotypes", "Participant", "Participant", "Participant"),
  key_adjustment = c("Diagnosis, age, sex, TCR depth, BCR depth", "Participant- and clone-size-specific multinomial null", "Paired clonotypes; thresholds >=2, >=3, >=5", paste0("Paired clonotypes; exact without-replacement rarefaction; TCR n=", tcr_rob$target, ", BCR n=", bcr_rob$target), "Same paired cells for single- and paired-chain models"),
  seed = 20260824
)
write_csv(manifest, file.path(outdir, "analysis_manifest.csv"))

message("Completed high-impact integrative analyses. Outputs: ", outdir)

#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(readr)
})

set.seed(20260829)

root <- "C:/path/to/private-manuscript-workspace"
outdir <- file.path(root, "High Impact Prior Findings Integration Preview 20260829", "source_data")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds"
)
stopifnot(all(file.exists(paths)))

modules <- list(
  Naive_central_memory = c("CCR7","SELL","TCF7","LEF1","IL7R","LTB","MAL","NOSIP","SATB1","BACH2","KLF2","S1PR1","CD27","CD28","BCL2"),
  Recent_TCR_stimulation = c("NR4A1","NR4A2","NR4A3","EGR1","EGR2","EGR3","FOS","JUN","JUNB","DUSP1","DUSP2","DUSP4","DUSP5","CD69","IL2RA","NFKBIA","NFKBIZ","REL","IRF4","BATF"),
  Effector_cytotoxicity = c("NKG7","GNLY","PRF1","GZMB","GZMA","GZMH","GZMK","GZMM","CTSW","CST7","FGFBP2","CCL5","CCL4","CCL3","IFNG","FASLG","KLRD1","KLRG1"),
  GZMK_inflammatory_memory = c("GZMK","GZMA","CCL5","CCL4","CCL4L2","XCL1","XCL2","NKG7","DUSP2","CRTAM","EOMES","CXCR3","IL7R"),
  Th1_Tc1_inflammatory = c("TBX21","STAT4","CXCR3","IFNG","TNF","IL12RB2","CCL5","CCL4","NKG7","GZMB","PRF1","CXCR6","BHLHE40"),
  Chronic_stimulation_exhaustion_like = c("PDCD1","LAG3","HAVCR2","TIGIT","CTLA4","TOX","TOX2","ENTPD1","SLAMF6","TNFRSF9","BATF","EOMES","PRDM1","LAYN","CXCL13"),
  Tissue_resident_mucosal_retention = c("CD69","ITGAE","ITGA1","CXCR6","ZNF683","RUNX3","PRDM1","RGS1","CD101","DUSP6","AHR","CCR6"),
  EOMES_ZEB2_inflammatory_CD8_TRM_like = c("EOMES","ZEB2","GZMB","GZMH","PRF1","NKG7","CX3CR1","KLRG1","TBX21","CCL5","CST7","FGFBP2"),
  Gut_homing_intestinal_trafficking = c("ITGA4","ITGB7","ITGAE","CCR9","CCR6","CXCR3","CXCR6","SELPLG","S1PR1","KLF2","SELL","GPR183","CD69"),
  Tph_Tfh_like_B_cell_help = c("CXCL13","PDCD1","ICOS","MAF","TOX2","IL21","CD40LG","SLAMF6","TIGIT","CD200","CXCR5","BCL6","SH2D1A"),
  MAIT_like = c("SLC4A10","KLRB1","ZBTB16","RORC","IL18RAP","CCR6","DPP4","CXCR6","IL7R","GZMK","IFNG","NKG7"),
  Cell_cycle_clonal_proliferation = c("MKI67","TOP2A","STMN1","TYMS","PCNA","MCM2","MCM3","MCM4","MCM5","MCM6","MCM7","HMGB2","CENPF","UBE2C","PCLAF")
)

valid <- function(x) {
  !is.na(x) & nzchar(trimws(as.character(x))) & toupper(trimws(as.character(x))) != "NA"
}

metadata_one <- function(path, compartment) {
  obj <- readRDS(path)
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  keep <- grepl("^TRAC", md$TCR_Alpha_Gamma_C_gene_Dominant) &
    grepl("^TRBC", md$TCR_Beta_Delta_C_gene_Dominant) &
    valid(md$TCR_Alpha_Gamma_V_gene_Dominant) &
    valid(md$TCR_Alpha_Gamma_J_gene_Dominant) &
    valid(md$TCR_Alpha_Gamma_CDR3_Translation_Dominant) &
    valid(md$TCR_Beta_Delta_V_gene_Dominant) &
    valid(md$TCR_Beta_Delta_J_gene_Dominant) &
    valid(md$TCR_Beta_Delta_CDR3_Translation_Dominant) &
    valid(md$SampleID) & valid(md$AnnotationLevel2)
  md <- md[keep, , drop = FALSE]
  ans <- md %>% transmute(
    cell,
    SampleID = as.character(SampleID),
    Diagnosis1 = as.character(Diagnosis1),
    Inflammation1 = as.character(Inflammation1),
    Biologic = as.character(Biologic),
    Batch = as.character(Batch),
    acquisition_series = sub("[AB]$", "", as.character(Batch)),
    compartment = compartment,
    state = as.character(AnnotationLevel2),
    clone_id = paste(
      TCR_Alpha_Gamma_V_gene_Dominant,
      TCR_Alpha_Gamma_J_gene_Dominant,
      TCR_Alpha_Gamma_CDR3_Translation_Dominant,
      TCR_Beta_Delta_V_gene_Dominant,
      TCR_Beta_Delta_J_gene_Dominant,
      TCR_Beta_Delta_CDR3_Translation_Dominant,
      sep = "|"
    )
  )
  rm(obj)
  invisible(gc())
  ans
}

metadata <- bind_rows(
  metadata_one(paths[["CD4"]], "CD4"),
  metadata_one(paths[["CD8"]], "CD8")
)
clone_sizes <- metadata %>% count(SampleID, clone_id, name = "clone_size")
metadata <- metadata %>% left_join(clone_sizes, by = c("SampleID", "clone_id")) %>%
  mutate(status = ifelse(clone_size >= 2L, "Expanded", "Singleton"))

score_one <- function(path, compartment) {
  message("Scoring exact paired alpha-beta clonotypes in ", compartment)
  obj <- readRDS(path)
  md <- metadata %>% filter(.data$compartment == .env$compartment)
  expr <- LayerData(obj[["RNA"]], layer = "data")
  genes <- intersect(unique(unlist(modules)), rownames(expr))
  expr <- expr[genes, md$cell, drop = FALSE]
  scores <- vapply(modules, function(gs) {
    present <- intersect(gs, rownames(expr))
    if (length(present) < 3L) rep(NA_real_, ncol(expr)) else Matrix::colMeans(expr[present, , drop = FALSE])
  }, numeric(ncol(expr)))
  scores <- as.data.frame(scores, check.names = FALSE)
  scores$cell <- colnames(expr)
  ans <- md %>% left_join(scores, by = "cell") %>%
    pivot_longer(cols = all_of(names(modules)), names_to = "module", values_to = "score") %>%
    group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series,
             compartment, state, status, module) %>%
    summarise(score = mean(score, na.rm = TRUE), n_cells = n(), .groups = "drop") %>%
    group_by(SampleID, compartment, state, module) %>%
    filter(n_distinct(status) == 2L) %>%
    ungroup()
  rm(expr, scores, obj)
  invisible(gc())
  ans
}

state_scores <- bind_rows(
  score_one(paths[["CD4"]], "CD4"),
  score_one(paths[["CD8"]], "CD8")
)

state_paired <- state_scores %>%
  select(-n_cells) %>%
  pivot_wider(names_from = status, values_from = score) %>%
  filter(is.finite(Expanded), is.finite(Singleton)) %>%
  mutate(delta = Expanded - Singleton)

participant_compartment <- state_paired %>%
  group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series,
           compartment, module) %>%
  summarise(
    Singleton = mean(Singleton),
    Expanded = mean(Expanded),
    delta = mean(delta),
    n_matched_states = n(),
    .groups = "drop"
  )

participant <- participant_compartment %>%
  group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series, module) %>%
  summarise(
    Singleton = mean(Singleton),
    Expanded = mean(Expanded),
    delta = mean(delta),
    n_matched_states = sum(n_matched_states),
    n_compartments = n_distinct(compartment),
    .groups = "drop"
  )

bh <- function(p) p.adjust(p, method = "BH")

bootstrap_median <- function(x, n_boot = 5000L) {
  x <- x[is.finite(x)]
  sims <- replicate(n_boot, median(sample(x, length(x), replace = TRUE)))
  c(low = unname(quantile(sims, 0.025)), high = unname(quantile(sims, 0.975)))
}

test_group <- function(dat, label) {
  bind_rows(lapply(names(modules), function(module_name) {
    x <- dat$delta[dat$module == module_name]
    x <- x[is.finite(x)]
    ci <- bootstrap_median(x)
    wt <- if (length(unique(x)) > 1L) wilcox.test(x, mu = 0, exact = FALSE) else list(p.value = NA_real_)
    tibble(
      stratum = label,
      module = module_name,
      n_participants = length(x),
      median_delta = median(x),
      ci_low = ci[["low"]],
      ci_high = ci[["high"]],
      p_value = wt$p.value
    )
  })) %>% mutate(FDR = bh(p_value))
}

tests <- bind_rows(
  test_group(participant %>% filter(Diagnosis1 %in% c("CD", "UC")), "Pooled IBD"),
  test_group(participant %>% filter(Diagnosis1 == "CD"), "CD"),
  test_group(participant %>% filter(Diagnosis1 == "UC"), "UC"),
  test_group(participant %>% filter(Diagnosis1 == "Control"), "Control")
)

series_effects <- participant %>%
  filter(Diagnosis1 %in% c("CD", "UC")) %>%
  group_by(module, acquisition_series) %>%
  summarise(
    n_participants = n(),
    mean_delta = mean(delta),
    se = ifelse(n() >= 3L, sd(delta) / sqrt(n()), NA_real_),
    p_value = ifelse(n() >= 3L, t.test(delta, mu = 0)$p.value, NA_real_),
    .groups = "drop"
  )

random_effect_one <- function(d) {
  d <- d %>% filter(is.finite(mean_delta), is.finite(se), se > 0)
  if (nrow(d) < 2L) return(tibble(k = nrow(d), effect = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p_value = NA_real_, tau2 = NA_real_))
  yi <- d$mean_delta
  vi <- d$se^2
  wi <- 1 / vi
  fixed <- sum(wi * yi) / sum(wi)
  q <- sum(wi * (yi - fixed)^2)
  cval <- sum(wi) - sum(wi^2) / sum(wi)
  tau2 <- max(0, (q - (length(yi) - 1)) / cval)
  wr <- 1 / (vi + tau2)
  effect <- sum(wr * yi) / sum(wr)
  se <- sqrt(1 / sum(wr))
  z <- effect / se
  p <- 2 * pnorm(-abs(z))
  tibble(k = nrow(d), effect = effect, se = se, ci_low = effect - 1.96 * se,
         ci_high = effect + 1.96 * se, p_value = p, tau2 = tau2)
}

random_effects <- bind_rows(lapply(names(modules), function(module_name) {
  random_effect_one(series_effects %>% filter(module == module_name)) %>% mutate(module = module_name, .before = 1)
})) %>% mutate(FDR = bh(p_value))

leave_one_series <- bind_rows(lapply(names(modules), function(module_name) {
  d <- participant %>% filter(Diagnosis1 %in% c("CD", "UC"), module == module_name)
  series <- sort(unique(d$acquisition_series))
  vals <- vapply(series, function(s) median(d$delta[d$acquisition_series != s], na.rm = TRUE), numeric(1))
  tibble(
    module = module_name,
    n_series = length(series),
    min_leave_one_series_median = min(vals),
    max_leave_one_series_median = max(vals),
    n_positive_series_means = sum(series_effects$module == module_name & series_effects$mean_delta > 0, na.rm = TRUE),
    n_estimable_series = sum(series_effects$module == module_name & is.finite(series_effects$mean_delta))
  )
}))

write_csv(state_paired, file.path(outdir, "Table_PI1_state_matched_exact_paired_alpha_beta_scores.csv"))
write_csv(participant, file.path(outdir, "Table_PI2_participant_exact_paired_alpha_beta_program_deltas.csv"))
write_csv(tests, file.path(outdir, "Table_PI3_program_tests_by_diagnosis.csv"))
write_csv(series_effects, file.path(outdir, "Table_PI4_acquisition_series_program_effects.csv"))
write_csv(random_effects, file.path(outdir, "Table_PI5_random_effects_program_meta_analysis.csv"))
write_csv(leave_one_series, file.path(outdir, "Table_PI6_leave_one_series_robustness.csv"))
write_csv(tibble(
  item = c("analysis date", "primary unit", "clonotype definition", "expansion threshold", "state matching", "program family", "multiplicity", "canonical files"),
  value = c(
    "2026-08-29",
    "participant",
    "within-participant exact productive paired alpha-beta V+J+CDR3 amino-acid identity",
    "expanded >=2 cells; singleton =1 cell",
    "expanded and singleton cells compared within participant, lineage compartment, and annotated state; equal state and compartment weights",
    "12 prespecified T-cell programs inherited from the prior analysis",
    "Benjamini-Hochberg correction across 12 programs within each displayed stratum",
    "not modified"
  )
), file.path(outdir, "Analysis_manifest_prior_findings_integration.csv"))

message("Wrote non-canonical prior-findings integration outputs to ", outdir)

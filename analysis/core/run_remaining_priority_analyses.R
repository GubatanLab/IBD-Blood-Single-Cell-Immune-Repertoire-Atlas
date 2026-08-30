#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(edgeR)
  library(limma)
  library(patchwork)
})

set.seed(20260824)

outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses/Priority Analyses"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds",
  BCR = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026_scvi50_umap.rds"
)
stopifnot(all(file.exists(paths)))

valid_string <- function(x) !is.na(x) & trimws(as.character(x)) != "" & toupper(trimws(as.character(x))) != "NA"

make_clone_id <- function(md, modality) {
  if (modality == "TCR") {
    ok <- grepl("^TRAC", md$TCR_Alpha_Gamma_C_gene_Dominant) &
      grepl("^TRBC", md$TCR_Beta_Delta_C_gene_Dominant) &
      valid_string(md$TCR_Alpha_Gamma_V_gene_Dominant) & valid_string(md$TCR_Alpha_Gamma_J_gene_Dominant) &
      valid_string(md$TCR_Alpha_Gamma_CDR3_Translation_Dominant) &
      valid_string(md$TCR_Beta_Delta_V_gene_Dominant) & valid_string(md$TCR_Beta_Delta_J_gene_Dominant) &
      valid_string(md$TCR_Beta_Delta_CDR3_Translation_Dominant)
    id <- rep(NA_character_, nrow(md))
    id[ok] <- do.call(paste, c(md[ok, c(
      "TCR_Alpha_Gamma_V_gene_Dominant", "TCR_Alpha_Gamma_J_gene_Dominant", "TCR_Alpha_Gamma_CDR3_Translation_Dominant",
      "TCR_Beta_Delta_V_gene_Dominant", "TCR_Beta_Delta_J_gene_Dominant", "TCR_Beta_Delta_CDR3_Translation_Dominant"
    )], sep = "|"))
  } else {
    ok <- grepl("^IGH", md$BCR_Heavy_C_gene_Dominant) & grepl("^IG[KL]", md$BCR_Light_C_gene_Dominant) &
      valid_string(md$BCR_Heavy_V_gene_Dominant) & valid_string(md$BCR_Heavy_J_gene_Dominant) &
      valid_string(md$BCR_Heavy_CDR3_Translation_Dominant) &
      valid_string(md$BCR_Light_V_gene_Dominant) & valid_string(md$BCR_Light_J_gene_Dominant) &
      valid_string(md$BCR_Light_CDR3_Translation_Dominant)
    id <- rep(NA_character_, nrow(md))
    id[ok] <- do.call(paste, c(md[ok, c(
      "BCR_Heavy_V_gene_Dominant", "BCR_Heavy_J_gene_Dominant", "BCR_Heavy_CDR3_Translation_Dominant",
      "BCR_Light_V_gene_Dominant", "BCR_Light_J_gene_Dominant", "BCR_Light_CDR3_Translation_Dominant"
    )], sep = "|"))
  }
  id
}

make_primary_clone_id <- function(md, modality) {
  if (modality == "TCR") {
    ok <- grepl("^TRBC", md$TCR_Beta_Delta_C_gene_Dominant) &
      valid_string(md$TCR_Beta_Delta_V_gene_Dominant) & valid_string(md$TCR_Beta_Delta_J_gene_Dominant) &
      valid_string(md$TCR_Beta_Delta_CDR3_Translation_Dominant)
    id <- rep(NA_character_, nrow(md))
    id[ok] <- do.call(paste, c(md[ok, c("TCR_Beta_Delta_V_gene_Dominant", "TCR_Beta_Delta_J_gene_Dominant",
                                               "TCR_Beta_Delta_CDR3_Translation_Dominant")], sep = "|"))
  } else {
    ok <- grepl("^IGH", md$BCR_Heavy_C_gene_Dominant) & valid_string(md$BCR_Heavy_V_gene_Dominant) &
      valid_string(md$BCR_Heavy_J_gene_Dominant) & valid_string(md$BCR_Heavy_CDR3_Translation_Dominant)
    id <- rep(NA_character_, nrow(md))
    id[ok] <- do.call(paste, c(md[ok, c("BCR_Heavy_V_gene_Dominant", "BCR_Heavy_J_gene_Dominant",
                                               "BCR_Heavy_CDR3_Translation_Dominant")], sep = "|"))
  }
  id
}

aggregate_pseudobulk <- function(obj, object_name, modality, min_cells = 5L) {
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  # Use the exact beta-chain or heavy-chain identity for expansion status. This
  # retains paired cells where present while avoiding loss of power from rare
  # two-chain identities; the paired definition is reserved for proximity tests.
  md$clone_id <- make_primary_clone_id(md, modality)
  md$state <- as.character(md$AnnotationLevel2)
  md$SampleID <- as.character(md$SampleID)
  md$Diagnosis1 <- as.character(md$Diagnosis1)
  keep <- valid_string(md$clone_id) & valid_string(md$SampleID) & valid_string(md$Diagnosis1) & valid_string(md$state)
  if ("High_Quality_Cell" %in% names(md)) {
    hq <- tolower(as.character(md$High_Quality_Cell))
    keep <- keep & (is.na(hq) | hq %in% c("true", "t", "1", "high_quality", "high quality"))
  }
  md <- md[keep, , drop = FALSE]
  clone_sizes <- md %>% count(SampleID, clone_id, name = "clone_size")
  md <- md %>% left_join(clone_sizes, by = c("SampleID", "clone_id")) %>%
    mutate(status = ifelse(clone_size >= 2, "Expanded", "Singleton"), lineage = object_name)

  # Match expanded and singleton cells within participant and cell state. Each
  # state-status pseudobulk is scaled to the smaller cell count before states
  # are summed, using all available cells without random downsampling. The final
  # unit is participant-by-status, so states are not treated as replicates.
  pair_keys <- md %>% count(SampleID, Diagnosis1, state, status, name = "n") %>%
    group_by(SampleID, Diagnosis1, state) %>% filter(n_distinct(status) == 2) %>%
    mutate(n_match = min(n)) %>% ungroup()
  md <- md %>% inner_join(pair_keys %>% select(SampleID, Diagnosis1, state, status, n, n_match),
                          by = c("SampleID", "Diagnosis1", "state", "status")) %>%
    mutate(state_group = paste(SampleID, state, status, object_name, sep = "|||"),
           final_group = paste(SampleID, status, object_name, sep = "|||"),
           qc_log_nfeature = if ("nFeature_RNA" %in% names(.)) log1p(as.numeric(nFeature_RNA)) else NA_real_,
           qc_percent_mito = if ("percent.mt" %in% names(.)) as.numeric(percent.mt) else NA_real_)
  state_samples <- md %>% distinct(state_group, final_group, SampleID, Diagnosis1, state, status, lineage, n, n_match) %>%
    left_join(md %>% group_by(state_group) %>% summarise(qc_log_nfeature = mean(qc_log_nfeature, na.rm = TRUE),
                                                        qc_percent_mito = mean(qc_percent_mito, na.rm = TRUE), .groups = "drop"),
              by = "state_group") %>% arrange(state_group)
  counts <- LayerData(obj[["RNA"]], layer = "counts")[, md$cell, drop = FALSE]
  f <- factor(md$state_group, levels = state_samples$state_group)
  mm_state <- sparse.model.matrix(~ 0 + f)
  colnames(mm_state) <- levels(f)
  state_counts <- counts %*% mm_state
  state_counts <- sweep(state_counts, 2, state_samples$n_match / state_samples$n, `*`)
  lev <- unique(state_samples$final_group)
  mm_final <- sparse.model.matrix(~0 + factor(state_samples$final_group, levels = lev))
  colnames(mm_final) <- lev
  agg <- state_counts %*% mm_final
  gs <- state_samples %>% group_by(final_group, SampleID, Diagnosis1, status, lineage) %>%
    summarise(n_cells = sum(n_match),
              qc_log_nfeature = weighted.mean(qc_log_nfeature, n_match, na.rm = TRUE),
              qc_percent_mito = weighted.mean(qc_percent_mito, n_match, na.rm = TRUE), .groups = "drop") %>%
    group_by(SampleID) %>% filter(n_distinct(status) == 2, min(n_cells) >= min_cells) %>% ungroup() %>% arrange(final_group)
  agg <- agg[, gs$final_group, drop = FALSE]
  colnames(agg) <- gs$final_group
  gs$group <- gs$final_group
  if (!nrow(gs)) stop("No state-matched participant pseudobulk groups for ", object_name)
  list(counts = agg, samples = gs, cell_meta = md)
}

neighborhood_null <- function(obj, object_name, modality, n_perm = 100L, max_cells = 60L) {
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  md$clone_id <- make_clone_id(md, modality)
  md$state <- as.character(md$AnnotationLevel2)
  md$SampleID <- as.character(md$SampleID)
  md$Diagnosis1 <- as.character(md$Diagnosis1)
  keep <- valid_string(md$clone_id) & valid_string(md$SampleID) & valid_string(md$Diagnosis1) & valid_string(md$state)
  md <- md[keep, , drop = FALSE]
  clone_sizes <- md %>% count(SampleID, clone_id, name = "clone_size")
  md <- md %>% left_join(clone_sizes, by = c("SampleID", "clone_id")) %>% filter(clone_size >= 2)
  reduction <- if ("SCVI_50" %in% names(obj@reductions)) "SCVI_50" else if ("SCVI" %in% names(obj@reductions)) "SCVI" else names(obj@reductions)[1]
  emb <- Embeddings(obj, reduction = reduction)[md$cell, , drop = FALSE]
  # Standardize dimensions so one high-variance latent dimension does not dominate.
  emb <- scale(emb)
  emb[!is.finite(emb)] <- 0

  pool_index <- split(seq_len(nrow(md)), paste(md$SampleID, md$state, sep = "|||"))
  clone_index <- split(seq_len(nrow(md)), paste(md$SampleID, md$clone_id, sep = "|||"))
  rows <- vector("list", length(clone_index)); z <- 1L
  median_dist <- function(ii) {
    if (length(ii) > max_cells) ii <- sample(ii, max_cells)
    if (length(ii) < 2) return(NA_real_)
    median(as.numeric(dist(emb[ii, , drop = FALSE])), na.rm = TRUE)
  }
  for (key in names(clone_index)) {
    ii <- clone_index[[key]]
    obs <- median_dist(ii)
    if (!is.finite(obs)) next
    null <- replicate(n_perm, {
      jj <- integer(length(ii))
      for (q in seq_along(ii)) {
        k <- paste(md$SampleID[ii[q]], md$state[ii[q]], sep = "|||")
        cand <- pool_index[[k]]
        cand <- cand[md$clone_id[cand] != md$clone_id[ii[q]]]
        if (!length(cand)) return(NA_real_)
        jj[q] <- sample(cand, 1)
      }
      median_dist(jj)
    })
    null <- null[is.finite(null)]
    if (length(null) < 20 || sd(null) == 0) next
    rows[[z]] <- tibble(
      modality = modality, lineage = object_name, SampleID = md$SampleID[ii[1]], Diagnosis1 = md$Diagnosis1[ii[1]],
      clone_id = md$clone_id[ii[1]], clone_size = length(ii), n_states = n_distinct(md$state[ii]),
      observed_distance = obs, null_mean = mean(null), null_sd = sd(null),
      proximity_z = (mean(null) - obs) / sd(null),
      empirical_p = (1 + sum(null <= obs)) / (1 + length(null))
    )
    z <- z + 1L
  }
  bind_rows(rows)
}

run_voom <- function(counts, samples, modality) {
  samples$status <- factor(samples$status, levels = c("Singleton", "Expanded"))
  samples$Diagnosis1 <- factor(samples$Diagnosis1, levels = c("Control", "UC", "CD"))
  samples$SampleID <- factor(samples$SampleID)
  design <- model.matrix(~ 0 + SampleID + status + scale(qc_log_nfeature) + scale(qc_percent_mito), data = samples)
  if (qr(design)$rank < ncol(design)) {
    # Remove redundant columns while retaining the estimable expansion terms.
    keep <- qr(design)$pivot[seq_len(qr(design)$rank)]
    design <- design[, sort(keep), drop = FALSE]
  }
  y <- DGEList(counts = counts)
  keep_gene <- filterByExpr(y, design = design, min.count = 10)
  y <- calcNormFactors(y[keep_gene, , keep.lib.sizes = FALSE])
  v <- voom(y, design, plot = FALSE)
  fit <- eBayes(lmFit(v, design), robust = TRUE)
  cn <- colnames(design)
  status_col <- grep("^statusExpanded$", cn, value = TRUE)
  coef_names <- status_col
  coef_names <- coef_names[coef_names %in% cn]
  out <- bind_rows(lapply(coef_names, function(co) {
    tt <- topTable(fit, coef = co, number = Inf, sort.by = "P")
    tt$gene <- rownames(tt)
    tt$coefficient <- co
    as_tibble(tt)
  }))
  out$modality <- modality
  out
}

pb_list <- list(TCR = list(), BCR = list())
neigh_list <- list()
clinical_list <- list()

for (nm in names(paths)) {
  message("Loading ", nm, "...")
  obj <- readRDS(paths[[nm]])
  modality <- ifelse(nm == "BCR", "BCR", "TCR")
  pb <- aggregate_pseudobulk(obj, nm, modality)
  pb_list[[modality]][[nm]] <- pb[c("counts", "samples")]
  message("Computing receptor-transcriptome proximity null for ", nm, "...")
  neigh_list[[nm]] <- neighborhood_null(obj, nm, modality)
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  clinical_list[[nm]] <- md %>%
    transmute(SampleID = as.character(SampleID), Diagnosis1 = as.character(Diagnosis1), Age = suppressWarnings(as.numeric(Age)),
              Sex = as.character(Sex), Batch = as.character(Batch), Calprotectin = suppressWarnings(as.numeric(Calprotectin)),
              Inflammation1 = as.character(Inflammation1), Biologic = as.character(Biologic)) %>% distinct()
  rm(md, obj, pb); gc()
}

combine_pb <- function(x) {
  genes <- Reduce(intersect, lapply(x, function(z) rownames(z$counts)))
  raw_counts <- do.call(cbind, lapply(x, function(z) z$counts[genes, , drop = FALSE]))
  raw_samples <- bind_rows(lapply(x, `[[`, "samples"))
  raw_samples <- raw_samples[match(colnames(raw_counts), raw_samples$group), , drop = FALSE] %>%
    mutate(combined_group = paste(SampleID, status, sep = "|||"))
  lev <- unique(raw_samples$combined_group)
  mm <- sparse.model.matrix(~0 + factor(raw_samples$combined_group, levels = lev))
  colnames(mm) <- lev
  counts <- raw_counts %*% mm
  samples <- raw_samples %>% group_by(combined_group, SampleID, Diagnosis1, status) %>%
    summarise(lineage = paste(sort(unique(lineage)), collapse = "+"),
              qc_log_nfeature = sum(qc_log_nfeature * n_cells, na.rm = TRUE) / sum(n_cells),
              qc_percent_mito = sum(qc_percent_mito * n_cells, na.rm = TRUE) / sum(n_cells),
              n_cells = sum(n_cells), .groups = "drop")
  samples <- samples[match(colnames(counts), samples$combined_group), , drop = FALSE]
  samples$group <- samples$combined_group
  list(counts = counts, samples = samples)
}

message("Running clone-aware pseudobulk differential expression...")
tcr_pb <- combine_pb(pb_list$TCR)
bcr_pb <- combine_pb(pb_list$BCR)
tcr_de <- run_voom(tcr_pb$counts, tcr_pb$samples, "TCR")
bcr_de <- run_voom(bcr_pb$counts, bcr_pb$samples, "BCR")
de <- bind_rows(tcr_de, bcr_de) %>% group_by(modality, coefficient) %>% mutate(FDR_global = p.adjust(P.Value, "BH")) %>% ungroup()
write_csv(de, file.path(outdir, "Table_PA1_clone_aware_pseudobulk_DE_all_genes.csv"))
write_csv(de %>% group_by(modality, coefficient) %>% arrange(P.Value, .by_group = TRUE) %>% slice_head(n = 50) %>% ungroup(),
          file.path(outdir, "Table_PA1_clone_aware_pseudobulk_DE_top50.csv"))
write_csv(bind_rows(tcr_pb$samples %>% mutate(modality = "TCR"), bcr_pb$samples %>% mutate(modality = "BCR")),
          file.path(outdir, "Table_PA1_pseudobulk_sample_manifest.csv"))

neigh <- bind_rows(neigh_list)
neigh_participant <- neigh %>% group_by(modality, lineage, SampleID, Diagnosis1) %>%
  summarise(n_expanded_clones = n(), median_proximity_z = median(proximity_z, na.rm = TRUE),
            fraction_empirical_p_lt_005 = mean(empirical_p < .05), .groups = "drop")
neigh_tests <- neigh_participant %>% group_by(modality) %>% group_modify(~{
  d <- .x
  groups <- list(c("CD", "Control"), c("UC", "Control"), c("CD", "UC"))
  bind_rows(lapply(groups, function(g) {
    a <- d$median_proximity_z[d$Diagnosis1 == g[1]]; b <- d$median_proximity_z[d$Diagnosis1 == g[2]]
    tibble(contrast = paste(g, collapse = " vs "), n1 = length(a), n2 = length(b),
           median1 = median(a, na.rm = TRUE), median2 = median(b, na.rm = TRUE),
           p_value = if (length(a) >= 3 && length(b) >= 3) wilcox.test(a, b, exact = FALSE)$p.value else NA_real_)
  }))
}) %>% ungroup() %>% mutate(FDR = p.adjust(p_value, "BH"))
overall_neigh <- neigh_participant %>% group_by(modality, Diagnosis1) %>%
  summarise(n = n(), median_z = median(median_proximity_z),
            p_vs_zero = if(n() >= 5) wilcox.test(median_proximity_z, mu = 0, exact = FALSE)$p.value else NA_real_, .groups = "drop") %>%
  mutate(FDR = p.adjust(p_vs_zero, "BH"))
write_csv(neigh, file.path(outdir, "Table_PA2_receptor_transcriptome_proximity_by_clone.csv"))
write_csv(neigh_participant, file.path(outdir, "Table_PA2_receptor_transcriptome_proximity_by_participant.csv"))
write_csv(neigh_tests, file.path(outdir, "Table_PA2_receptor_transcriptome_proximity_group_tests.csv"))
write_csv(overall_neigh, file.path(outdir, "Table_PA2_receptor_transcriptome_proximity_vs_null.csv"))

clinical <- bind_rows(clinical_list) %>% group_by(SampleID) %>% summarise(
  Diagnosis1 = first(na.omit(Diagnosis1)), Age = first(na.omit(Age)), Sex = first(na.omit(Sex)), Batch = first(na.omit(Batch)),
  Calprotectin = first(na.omit(Calprotectin)), Inflammation1 = first(na.omit(Inflammation1)), Biologic = first(na.omit(Biologic)), .groups = "drop")
write_csv(clinical, file.path(outdir, "Table_PA_clinical_metadata.csv"))

# Publication figure: top expansion-associated genes and receptor-transcriptome proximity.
top_de <- de %>% filter(coefficient == "statusExpanded") %>% group_by(modality) %>%
  arrange(P.Value, .by_group = TRUE) %>% slice_head(n = 12) %>% ungroup() %>%
  mutate(sig = FDR_global < .05, gene = factor(gene, levels = rev(unique(gene))))
p1 <- ggplot(top_de, aes(logFC, gene, shape = sig, color = modality)) +
  geom_vline(xintercept = 0, color = "grey75", linewidth = .35) + geom_point(size = 2.2) +
  facet_wrap(~modality, scales = "free_y", ncol = 2) +
  scale_color_manual(values = c(TCR = "#2B6CB0", BCR = "#C2415D")) +
  labs(x = "Expanded versus singleton log2 fold change", y = NULL, shape = "FDR < 0.05", color = NULL,
       title = "A  Clone-aware pseudobulk expression") + theme_classic(base_size = 10) +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"), legend.position = "bottom")
p2 <- ggplot(neigh_participant, aes(Diagnosis1, median_proximity_z, color = Diagnosis1)) +
  geom_hline(yintercept = 0, color = "grey65", linewidth = .35) + geom_boxplot(outlier.shape = NA, width = .55) +
  geom_jitter(width = .13, alpha = .55, size = 1.1) + facet_wrap(~modality, scales = "free_y") +
  scale_color_manual(values = c(Control = "#777777", UC = "#C2415D", CD = "#2B6CB0")) +
  labs(x = NULL, y = "Clone-mate transcriptomic proximity (null z)", title = "B  Receptor-transcriptome organization") +
  theme_classic(base_size = 10) + theme(strip.background = element_blank(), strip.text = element_text(face = "bold"), legend.position = "none")
fig <- p1 / p2 + plot_layout(heights = c(1.35, 1))
ggsave(file.path(outdir, "Figure_PA1_clone_aware_expression_and_receptor_transcriptome_proximity.png"), fig, width = 9.2, height = 9.2, dpi = 600)
ggsave(file.path(outdir, "Figure_PA1_clone_aware_expression_and_receptor_transcriptome_proximity.pdf"), fig, width = 9.2, height = 9.2)

manifest <- tibble(
  analysis = c("Clone-aware pseudobulk DE", "Receptor-transcriptome proximity"),
  status = c("completed", "completed"),
  primary_unit = c("participant-state-status pseudobulk", "participant median across exact expanded clonotypes"),
  output_prefix = c("Table_PA1", "Table_PA2")
)
write_csv(manifest, file.path(outdir, "priority_analysis_manifest_stage1.csv"))
message("Stage 1 complete: ", outdir)

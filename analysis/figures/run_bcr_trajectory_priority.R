#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(SingleCellExperiment)
  library(slingshot)
  library(dplyr)
  library(readr)
})

set.seed(20260828)

root <- "C:/path/to/private-manuscript-workspace"
object_path <- "C:/path/to/private-user-home/OneDrive/Desktop/BCR Module Scores/BCell2026.rds"
outdir <- file.path(root, "High Impact Additional Analyses", "BCR Trajectory Priority")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

message("Loading B-cell object...")
obj <- readRDS(object_path)
md <- obj@meta.data %>% as.data.frame() %>% mutate(cell_id = rownames(.))

required <- c(
  "PatientID", "Diagnosis1", "AnnotationLevel3", "BCR_Paired_Chains",
  "BCR_Heavy_V_gene_Dominant", "BCR_Heavy_J_gene_Dominant", "BCR_Heavy_C_gene_Dominant",
  "BCR_Heavy_CDR3_Nucleotide_Dominant", "BCR_Heavy_CDR3_Translation_Dominant",
  "BCR_Light_V_gene_Dominant", "BCR_Light_J_gene_Dominant", "BCR_Light_CDR3_Translation_Dominant"
)
stopifnot(all(required %in% colnames(md)))

clean_gene <- function(x) sub("\\*.*$", "", as.character(x))
nonempty <- function(x) !is.na(x) & as.character(x) != ""

md <- md %>%
  mutate(
    state = as.character(AnnotationLevel3),
    paired_bcr = as.logical(BCR_Paired_Chains),
    exact_paired_clone = ifelse(
      paired_bcr &
        nonempty(BCR_Heavy_V_gene_Dominant) & nonempty(BCR_Heavy_J_gene_Dominant) &
        nonempty(BCR_Heavy_CDR3_Translation_Dominant) & nonempty(BCR_Light_V_gene_Dominant) &
        nonempty(BCR_Light_J_gene_Dominant) & nonempty(BCR_Light_CDR3_Translation_Dominant),
      paste(
        PatientID,
        BCR_Heavy_V_gene_Dominant, BCR_Heavy_J_gene_Dominant, BCR_Heavy_CDR3_Translation_Dominant,
        BCR_Light_V_gene_Dominant, BCR_Light_J_gene_Dominant, BCR_Light_CDR3_Translation_Dominant,
        sep = "|"
      ),
      NA_character_
    ),
    receptor_key = ifelse(
      nonempty(BCR_Heavy_V_gene_Dominant) & nonempty(BCR_Heavy_J_gene_Dominant) &
        nonempty(BCR_Heavy_CDR3_Nucleotide_Dominant),
      paste(
        PatientID, clean_gene(BCR_Heavy_V_gene_Dominant), clean_gene(BCR_Heavy_J_gene_Dominant),
        BCR_Heavy_CDR3_Nucleotide_Dominant, sep = "|"
      ),
      NA_character_
    ),
    shm_match_key = ifelse(
      nonempty(BCR_Heavy_V_gene_Dominant) & nonempty(BCR_Heavy_J_gene_Dominant) &
        nonempty(BCR_Heavy_CDR3_Translation_Dominant),
      paste(
        PatientID, clean_gene(BCR_Heavy_V_gene_Dominant), clean_gene(BCR_Heavy_J_gene_Dominant),
        toupper(BCR_Heavy_CDR3_Translation_Dominant), sep = "|"
      ),
      NA_character_
    ),
    isotype = case_when(
      grepl("^IGHM", BCR_Heavy_C_gene_Dominant) ~ "IgM",
      grepl("^IGHD", BCR_Heavy_C_gene_Dominant) ~ "IgD",
      grepl("^IGHA", BCR_Heavy_C_gene_Dominant) ~ "IgA",
      grepl("^IGHG", BCR_Heavy_C_gene_Dominant) ~ "IgG",
      grepl("^IGHE", BCR_Heavy_C_gene_Dominant) ~ "IgE",
      TRUE ~ NA_character_
    )
  )

clone_sizes <- md %>%
  filter(nonempty(exact_paired_clone)) %>%
  count(exact_paired_clone, name = "full_clone_size")
md <- md %>%
  left_join(clone_sizes, by = "exact_paired_clone") %>%
  mutate(
    full_clone_size = ifelse(is.na(full_clone_size), 0L, full_clone_size),
    clone_bin = case_when(
      full_clone_size == 1 ~ "Singleton",
      full_clone_size >= 2 & full_clone_size <= 5 ~ "2-5",
      full_clone_size >= 6 & full_clone_size <= 20 ~ "6-20",
      full_clone_size > 20 ~ ">20",
      TRUE ~ NA_character_
    ),
    switched = ifelse(isotype %in% c("IgA", "IgG", "IgE"), 1, ifelse(isotype %in% c("IgM", "IgD"), 0, NA_real_))
  )

message("Mapping germline-relative SHM to receptor keys...")
bgl1 <- read_csv(
  file.path(root, "High Impact Additional Analyses", "BCR Germline Lineages", "Table_BGL1_query_sequence_metadata.csv"),
  show_col_types = FALSE
)
bgl3 <- read_csv(
  file.path(root, "High Impact Additional Analyses", "BCR Germline Lineages", "Table_BGL3_sequence_region_SHM_RS.csv"),
  show_col_types = FALSE
)
shm_by_sequence <- bgl3 %>%
  group_by(sequence_id) %>%
  summarise(
    informative_nt = sum(informative_nt, na.rm = TRUE),
    mutations = sum(mutations, na.rm = TRUE),
    total_shm = ifelse(informative_nt > 0, mutations / informative_nt, NA_real_),
    .groups = "drop"
  )
shm_map <- bgl1 %>%
  mutate(
    cdr3_aa_core = toupper(sub("W$", "", sub("^C", "", junction_aa))),
    shm_match_key = paste(PatientID, clean_gene(v_gene), clean_gene(j_gene), cdr3_aa_core, sep = "|")
  ) %>%
  select(sequence_id, shm_match_key, count) %>%
  inner_join(shm_by_sequence, by = "sequence_id") %>%
  group_by(shm_match_key) %>%
  summarise(
    total_shm = weighted.mean(total_shm, w = pmax(count, 1), na.rm = TRUE),
    .groups = "drop"
  )
md <- md %>% left_join(shm_map, by = "shm_match_key")

message("Creating participant-state-balanced reference...")
balanced <- md %>%
  filter(nonempty(PatientID), nonempty(state)) %>%
  group_by(PatientID, state) %>%
  slice_sample(n = 8L, replace = FALSE) %>%
  ungroup()

balanced_cells <- balanced$cell_id
latent <- Embeddings(obj, "SCVI_50")[balanced_cells, seq_len(10), drop = FALSE]
umap <- Embeddings(obj, "umap")[balanced_cells, , drop = FALSE]

message("Fitting branching Slingshot principal curves in SCVI latent space...")
terminal_states <- c("Atypical memory B", "IgM Plasma B Cell", "IgA Plasma B Cell", "IgG Plasma B Cell")
sds <- slingshot(
  latent,
  clusterLabels = balanced$state,
  start.clus = "Transitional B",
  end.clus = terminal_states,
  approx_points = 150,
  reweight = TRUE
)
lineages <- slingLineages(sds)
lineage_names <- vapply(lineages, function(x) tail(x, 1), character(1))
lineage_names <- make.unique(lineage_names)

pt_raw <- slingPseudotime(sds, na = FALSE)
weights <- slingCurveWeights(sds)
pt_norm <- apply(pt_raw, 2, function(x) {
  span <- diff(range(x, finite = TRUE))
  if (span > 0) (x - min(x, na.rm = TRUE)) / span else x * 0
})
colnames(pt_norm) <- lineage_names
colnames(weights) <- lineage_names
best_branch_index <- max.col(weights, ties.method = "first")
best_branch <- lineage_names[best_branch_index]
weighted_pt <- rowSums(pt_norm * weights, na.rm = TRUE) / pmax(rowSums(weights, na.rm = TRUE), 1e-8)

message("Computing focused transcriptional program scores...")
module_genes <- list(
  IgA_mucosal = c("IGHA1", "IGHA2", "JCHAIN", "MZB1", "XBP1", "SDC1", "TNFRSF17", "CCR10", "PRDM1", "IRF4"),
  Plasma_differentiation = c("PRDM1", "XBP1", "IRF4", "MZB1", "SDC1", "JCHAIN", "SSR4", "FKBP11", "DERL3", "TNFRSF17", "SLAMF7", "CD38", "CD27", "SEC11C"),
  Antibody_secretion_UPR = c("XBP1", "HSPA5", "HSP90B1", "HERPUD1", "ATF4", "ATF6", "DDIT3", "SEL1L", "DNAJB9", "PPIB", "CALR", "ERP44", "DERL3", "SEC61A1", "SSR4")
)
expr <- GetAssayData(obj, assay = "RNA", layer = "data")
module_scores <- list()
for (module in names(module_genes)) {
  genes <- intersect(module_genes[[module]], rownames(expr))
  values <- Matrix::colMeans(expr[genes, balanced_cells, drop = FALSE])
  module_scores[[module]] <- as.numeric(scale(values))
}

trajectory_cells <- balanced %>%
  transmute(
    cell_id, PatientID, Diagnosis1, state, paired_bcr, exact_paired_clone, full_clone_size, clone_bin,
    isotype, switched, total_shm,
    UMAP_1 = umap[, 1], UMAP_2 = umap[, 2],
    trajectory_pseudotime = weighted_pt,
    assigned_branch = best_branch,
    max_branch_weight = apply(weights, 1, max, na.rm = TRUE),
    IgA_mucosal_score = module_scores$IgA_mucosal,
    plasma_differentiation_score = module_scores$Plasma_differentiation,
    antibody_secretion_UPR_score = module_scores$Antibody_secretion_UPR
  )

for (j in seq_len(ncol(pt_norm))) {
  trajectory_cells[[paste0("pseudotime_", j)]] <- pt_norm[, j]
  trajectory_cells[[paste0("weight_", j)]] <- weights[, j]
}

lineage_table <- bind_rows(lapply(seq_along(lineages), function(i) {
  tibble(
    lineage_index = i,
    terminal_state = lineage_names[i],
    cluster_order = paste(lineages[[i]], collapse = " -> "),
    n_cells_positive_weight = sum(weights[, i] > 0, na.rm = TRUE)
  )
}))

paired_full <- md %>%
  filter(paired_bcr, nonempty(exact_paired_clone)) %>%
  select(cell_id, PatientID, Diagnosis1, state, exact_paired_clone, full_clone_size, clone_bin, isotype, switched, total_shm)

write_csv(trajectory_cells, gzfile(file.path(outdir, "Table_BT1_balanced_cell_pseudotime.csv.gz")))
write_csv(lineage_table, file.path(outdir, "Table_BT1_slingshot_lineages.csv"))
write_csv(paired_full, gzfile(file.path(outdir, "Table_BT2_full_paired_clone_state_metadata.csv.gz")))

manifest <- tibble(
  metric = c(
    "all_object_cells", "balanced_reference_cells", "participants", "paired_cells_in_reference",
    "cells_with_SHM_in_reference", "number_of_lineages"
  ),
  value = c(
    nrow(md), nrow(trajectory_cells), n_distinct(trajectory_cells$PatientID), sum(trajectory_cells$paired_bcr, na.rm = TRUE),
    sum(is.finite(trajectory_cells$total_shm)), nrow(lineage_table)
  )
)
write_csv(manifest, file.path(outdir, "Table_BT0_analysis_manifest.csv"))
saveRDS(sds, file.path(outdir, "BT1_slingshot_fit.rds"))

message("Trajectory analysis complete: ", outdir)
print(lineage_table)
print(manifest)

#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(S4Vectors)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
  library(multinichenetr)
  library(nichenetr)
})

options(stringsAsFactors = FALSE)
set.seed(20260831)

root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
private_root <- Sys.getenv("IBD_PRIVATE_WORKSPACE", unset = "C:/path/to/private-manuscript-workspace")
prior_root <- Sys.getenv(
  "IBD_MULTINICHENET_PRIORS",
  unset = file.path(private_root, "analysis_inputs", "multinichenet")
)
outdir <- Sys.getenv(
  "IBD_MULTINICHENET_OUTPUT",
  unset = file.path(root, "analysis_outputs", "multinichenet")
)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
logfile <- file.path(outdir, "MN_core_run_log.txt")

log_msg <- function(...) {
  z <- paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "-", paste(..., collapse = " "))
  cat(z, "\n")
  cat(z, "\n", file = logfile, append = TRUE)
  flush.console()
}

paths <- c(
  CD4 = file.path(private_root, "Figure 2 TCR", "CD42026.rds"),
  B = file.path(private_root, "Figure 1", "Figure 1", "BCell2026.rds")
)
network_paths <- c(
  lr = file.path(prior_root, "lr_network_human_21122021.rds"),
  ligand_target = file.path(prior_root, "ligand_target_matrix_nsga2r_final_human.rds")
)
stopifnot(all(file.exists(paths)), all(file.exists(network_paths)))

series_from_batch <- function(x) sub("[AB]$", "", as.character(x))
valid <- function(x) !is.na(x) & nzchar(trimws(as.character(x)))

map_cd4 <- function(x) {
  case_when(
    x == "CD4 Tfh" ~ "Tfh",
    x %in% c("CD4 Th17", "CD4 Th1/Th17") ~ "Th17",
    grepl("^TReg", x) ~ "Treg",
    TRUE ~ NA_character_
  )
}

map_b <- function(x) {
  case_when(
    x %in% c("Naive B", "Naive-IFN B", "Transitional B", "CD5+ B Cell") ~ "B_NaiveTransitional",
    x %in% c("Switched memory B", "Non-switched memory B") ~ "B_Memory",
    x == "Atypical memory B" ~ "B_AtypicalMemory",
    x == "IgM Plasma B Cell" ~ "Plasma_IgM",
    x %in% c("IgA Plasma B Cell", "IgG Plasma B Cell") ~ "Plasma_Switched",
    TRUE ~ NA_character_
  )
}

extract_compartment <- function(compartment, path, mapper) {
  log_msg("Reading", compartment, "object")
  obj <- readRDS(path)
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$original_cell <- rownames(md)
  md$original_state <- as.character(md$AnnotationLevel2)
  md$celltype <- mapper(md$original_state)
  keep <- which(!is.na(md$celltype) & valid(md$SampleID))
  md <- md[keep, , drop = FALSE]
  cells <- md$original_cell
  counts <- LayerData(obj[["RNA"]], layer = "counts")[, cells, drop = FALSE]
  new_cells <- paste(compartment, cells, sep = "::")
  colnames(counts) <- new_cells
  rownames(md) <- new_cells
  md$cell <- new_cells
  md$compartment <- compartment
  md$SampleID <- as.character(md$SampleID)
  md$PatientID <- as.character(md$PatientID)
  md$Diagnosis1 <- as.character(md$Diagnosis1)
  md$Inflammation1 <- as.character(md$Inflammation1)
  md$Biologic <- as.character(md$Biologic)
  md$Batch <- as.character(md$Batch)
  md$acquisition_series <- series_from_batch(md$Batch)
  md$Age <- suppressWarnings(as.numeric(md$Age))
  md$Sex <- as.character(md$Sex)
  rm(obj)
  invisible(gc())
  list(counts = counts, metadata = md)
}

log_msg("Loading official NicheNet human priors")
lr_network <- readRDS(network_paths[["lr"]])
ligand_target_matrix <- readRDS(network_paths[["ligand_target"]])
prior_genes_raw <- unique(c(
  rownames(ligand_target_matrix), colnames(ligand_target_matrix),
  if ("from" %in% colnames(lr_network)) lr_network$from else lr_network$ligand,
  if ("to" %in% colnames(lr_network)) lr_network$to else lr_network$receptor
))
if (all(c("from", "to") %in% colnames(lr_network))) {
  lr_network$from <- make.names(lr_network$from)
  lr_network$to <- make.names(lr_network$to)
} else if (all(c("ligand", "receptor") %in% colnames(lr_network))) {
  lr_network$ligand <- make.names(lr_network$ligand)
  lr_network$receptor <- make.names(lr_network$receptor)
} else {
  stop("Unexpected ligand-receptor network columns")
}
rownames(ligand_target_matrix) <- make.names(rownames(ligand_target_matrix))
colnames(ligand_target_matrix) <- make.names(colnames(ligand_target_matrix))

cd4 <- extract_compartment("CD4", paths[["CD4"]], map_cd4)
b <- extract_compartment("B", paths[["B"]], map_b)
stopifnot(identical(rownames(cd4$counts), rownames(b$counts)))

keep_raw_genes <- intersect(rownames(cd4$counts), prior_genes_raw)
cd4$counts <- cd4$counts[keep_raw_genes, , drop = FALSE]
b$counts <- b$counts[keep_raw_genes, , drop = FALSE]
safe_genes <- make.names(keep_raw_genes)
if (anyDuplicated(safe_genes)) stop("Gene symbols are not unique after make.names")
rownames(cd4$counts) <- safe_genes
rownames(b$counts) <- safe_genes
keep_genes <- safe_genes
log_msg("Combining", length(keep_genes), "prior-supported genes across",
        ncol(cd4$counts), "CD4 and", ncol(b$counts), "B-lineage cells")

counts <- cbind(cd4$counts[keep_genes, , drop = FALSE], b$counts[keep_genes, , drop = FALSE])
metadata <- bind_rows(cd4$metadata, b$metadata)
metadata <- metadata[colnames(counts), , drop = FALSE]
stopifnot(identical(rownames(metadata), colnames(counts)))

sce <- SingleCellExperiment(
  assays = list(counts = counts),
  colData = DataFrame(metadata)
)
rm(counts, cd4, b)
invisible(gc())

write_csv(
  as.data.frame(colData(sce)) %>%
    count(SampleID, PatientID, Diagnosis1, Inflammation1, Biologic,
          acquisition_series, celltype, compartment, name = "n_cells"),
  file.path(outdir, "Table_MN6_modeled_sample_celltype_counts.csv")
)

write_csv(
  tibble(
    item = c("CD4 object", "B object", "LR network", "ligand-target matrix",
             "modeled genes", "modeled cells", "minimum cells per sample-state",
             "minimum samples per group", "logFC threshold", "fraction cutoff",
             "minimum sample proportion"),
    value = c(paths[["CD4"]], paths[["B"]], network_paths[["lr"]],
              network_paths[["ligand_target"]], nrow(sce), ncol(sce), 5, 4,
              0.25, 0.05, 0.5)
  ),
  file.path(outdir, "Table_MN7_core_analysis_manifest.csv")
)

helper_types <- c("Tfh", "Th17", "Treg")
b_types <- c("B_NaiveTransitional", "B_Memory", "B_AtypicalMemory", "Plasma_IgM", "Plasma_Switched")
all_types <- c(helper_types, b_types)

sample_metadata <- as.data.frame(colData(sce)) %>%
  distinct(SampleID, Diagnosis1, Inflammation1, acquisition_series)

shared_series <- function(data, group_col, groups) {
  data %>%
    filter(.data[[group_col]] %in% groups) %>%
    distinct(SampleID, acquisition_series, .data[[group_col]]) %>%
    count(acquisition_series, .data[[group_col]], name = "n") %>%
    group_by(acquisition_series) %>%
    filter(n_distinct(.data[[group_col]]) == length(groups), all(n >= 1)) %>%
    pull(acquisition_series) %>% unique()
}

cd_series <- shared_series(filter(sample_metadata, Diagnosis1 == "CD"), "Inflammation1",
                           c("Inflamed", "Noninflamed"))
uc_series <- shared_series(filter(sample_metadata, Diagnosis1 == "UC"), "Inflammation1",
                           c("Inflamed", "Noninflamed"))
cduc_series <- shared_series(filter(sample_metadata, Diagnosis1 %in% c("CD", "UC")), "Diagnosis1",
                             c("CD", "UC"))

specs <- list(
  CD_inflammation = list(
    keep = with(as.data.frame(colData(sce)), Diagnosis1 == "CD" &
      Inflammation1 %in% c("Inflamed", "Noninflamed") & acquisition_series %in% cd_series),
    group = "Inflammation1",
    contrasts = c("Inflamed-Noninflamed"),
    main_groups = c("Inflamed"),
    covariates = "acquisition_series"
  ),
  UC_inflammation = list(
    keep = with(as.data.frame(colData(sce)), Diagnosis1 == "UC" &
      Inflammation1 %in% c("Inflamed", "Noninflamed") & acquisition_series %in% uc_series),
    group = "Inflammation1",
    contrasts = c("Inflamed-Noninflamed"),
    main_groups = c("Inflamed"),
    covariates = "acquisition_series"
  ),
  CD_vs_UC_shared_series = list(
    keep = with(as.data.frame(colData(sce)), Diagnosis1 %in% c("CD", "UC") &
      acquisition_series %in% cduc_series),
    group = "Diagnosis1",
    contrasts = c("CD-UC"),
    main_groups = c("CD"),
    covariates = "acquisition_series"
  ),
  S6_diagnosis = list(
    keep = with(as.data.frame(colData(sce)), Diagnosis1 %in% c("CD", "UC", "Control") &
      acquisition_series == "S6"),
    group = "Diagnosis1",
    contrasts = c("CD-Control", "UC-Control", "CD-UC"),
    main_groups = c("CD", "UC", "CD"),
    covariates = NA_character_
  )
)

write_csv(
  bind_rows(lapply(names(specs), function(nm) {
    z <- specs[[nm]]
    as.data.frame(colData(sce))[z$keep, , drop = FALSE] %>%
      distinct(SampleID, .data[[z$group]], acquisition_series) %>%
      count(group = .data[[z$group]], acquisition_series, name = "samples") %>%
      mutate(analysis = nm, .before = 1)
  })),
  file.path(outdir, "Table_MN8_analysis_sample_inventory.csv")
)

run_one <- function(name, spec) {
  output_rds <- file.path(outdir, paste0("MN_core_", name, ".rds"))
  if (file.exists(output_rds)) {
    log_msg("Skipping completed analysis", name)
    return(invisible(NULL))
  }

  sub <- sce[, spec$keep]
  colData(sub)$group_id <- as.character(colData(sub)[[spec$group]])
  samples_by_group <- as.data.frame(colData(sub)) %>%
    distinct(SampleID, group_id) %>% count(group_id, name = "samples")
  if (any(samples_by_group$samples < 4)) stop(name, ": fewer than four samples in a group")

  contrasts_oi <- paste0("'", spec$contrasts, "'", collapse = ",")
  contrast_tbl <- tibble(contrast = spec$contrasts, group = spec$main_groups)
  covariates <- if (all(is.na(spec$covariates))) NA else spec$covariates

  log_msg("Starting", name, "with", ncol(sub), "cells and", n_distinct(colData(sub)$SampleID), "samples")
  output <- multi_nichenet_analysis(
    sce = sub,
    celltype_id = "celltype",
    sample_id = "SampleID",
    group_id = "group_id",
    batches = NA,
    covariates = covariates,
    lr_network = lr_network,
    ligand_target_matrix = ligand_target_matrix,
    contrasts_oi = contrasts_oi,
    contrast_tbl = contrast_tbl,
    senders_oi = all_types,
    receivers_oi = all_types,
    fraction_cutoff = 0.05,
    min_sample_prop = 0.5,
    scenario = "regular",
    ligand_activity_down = FALSE,
    assay_oi_pb = "counts",
    fun_oi_pb = "sum",
    de_method_oi = "edgeR",
    min_cells = 5,
    logFC_threshold = 0.25,
    p_val_threshold = 0.05,
    p_val_adj = FALSE,
    empirical_pval = TRUE,
    top_n_target = 250,
    verbose = TRUE,
    n.cores = 4,
    return_lr_prod_matrix = FALSE,
    findMarkers = FALSE,
    top_n_LR = 2500
  )
  saveRDS(output, output_rds, compress = FALSE)

  for (tbl_name in names(output$prioritization_tables)) {
    tbl <- output$prioritization_tables[[tbl_name]]
    if (is.data.frame(tbl)) {
      write_csv(tbl, gzfile(file.path(outdir, paste0("Table_MN_", name, "_", tbl_name, ".csv.gz"))))
    }
  }

  group_tbl <- as.data.frame(output$prioritization_tables$group_prioritization_tbl) %>%
    mutate(direction = case_when(
      sender %in% helper_types & receiver %in% b_types ~ "T_helper_to_B",
      sender %in% b_types & receiver %in% helper_types ~ "B_to_T_helper",
      TRUE ~ "within_compartment"
    ))
  write_csv(
    filter(group_tbl, direction != "within_compartment"),
    file.path(outdir, paste0("Table_MN_", name, "_cross_lineage_prioritization.csv"))
  )
  log_msg("Completed", name, "with", nrow(group_tbl), "prioritized interaction rows")
  rm(output, group_tbl, sub)
  invisible(gc())
}

for (nm in names(specs)) run_one(nm, specs[[nm]])
log_msg("All MultiNicheNet core analyses completed")

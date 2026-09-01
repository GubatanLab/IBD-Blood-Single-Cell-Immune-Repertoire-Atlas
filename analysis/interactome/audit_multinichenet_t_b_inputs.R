#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(readr)
  library(tidyr)
})

options(stringsAsFactors = FALSE)

root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
private_root <- Sys.getenv("IBD_PRIVATE_WORKSPACE", unset = "C:/path/to/private-manuscript-workspace")
outdir <- Sys.getenv(
  "IBD_MULTINICHENET_OUTPUT",
  unset = file.path(root, "analysis_outputs", "multinichenet")
)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = file.path(private_root, "Figure 2 TCR", "CD42026.rds"),
  B = file.path(private_root, "Figure 1", "Figure 1", "BCell2026.rds")
)
stopifnot(all(file.exists(paths)))

series_from_batch <- function(x) sub("[AB]$", "", as.character(x))

audit_one <- function(compartment, path) {
  message("Reading ", compartment, ": ", path)
  obj <- readRDS(path)
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  md$compartment <- compartment
  md$state <- as.character(md$AnnotationLevel2)
  md$SampleID <- as.character(md$SampleID)
  md$PatientID <- as.character(md$PatientID)
  md$Diagnosis1 <- as.character(md$Diagnosis1)
  md$Inflammation1 <- as.character(md$Inflammation1)
  md$Biologic <- as.character(md$Biologic)
  md$Batch <- as.character(md$Batch)
  md$acquisition_series <- series_from_batch(md$Batch)

  assay <- obj[["RNA"]]
  layer_names <- Layers(assay)
  count_layers <- grep("^counts($|\\.)", layer_names, value = TRUE)
  count_cells <- vapply(count_layers, function(z) ncol(LayerData(assay, layer = z)), integer(1))
  count_features <- vapply(count_layers, function(z) nrow(LayerData(assay, layer = z)), integer(1))

  object_summary <- tibble(
    compartment = compartment,
    path = path,
    cells = ncol(obj),
    features = nrow(obj),
    samples = n_distinct(md$SampleID[!is.na(md$SampleID) & nzchar(md$SampleID)]),
    patients = n_distinct(md$PatientID[!is.na(md$PatientID) & nzchar(md$PatientID)]),
    layers = paste(layer_names, collapse = ";"),
    count_layers = paste(count_layers, collapse = ";"),
    count_layer_cells = paste(count_cells, collapse = ";"),
    count_layer_features = paste(count_features, collapse = ";")
  )

  coverage <- md %>%
    filter(!is.na(SampleID), nzchar(SampleID), !is.na(state), nzchar(state)) %>%
    count(compartment, SampleID, PatientID, Diagnosis1, Inflammation1, Biologic,
          Batch, acquisition_series, state, name = "n_cells")

  list(obj = obj, md = md, summary = object_summary, coverage = coverage)
}

cd4 <- audit_one("CD4", paths[["CD4"]])
b <- audit_one("B", paths[["B"]])

write_csv(bind_rows(cd4$summary, b$summary), file.path(outdir, "Table_MN0_object_audit.csv"))
coverage <- bind_rows(cd4$coverage, b$coverage)
write_csv(coverage, file.path(outdir, "Table_MN1_sample_state_cell_counts.csv"))

state_summary <- coverage %>%
  group_by(compartment, state) %>%
  summarise(
    cells = sum(n_cells),
    samples_any = n_distinct(SampleID),
    samples_ge5 = n_distinct(SampleID[n_cells >= 5]),
    samples_ge10 = n_distinct(SampleID[n_cells >= 10]),
    median_cells_nonzero = median(n_cells),
    .groups = "drop"
  ) %>% arrange(compartment, desc(cells))
write_csv(state_summary, file.path(outdir, "Table_MN2_state_coverage_summary.csv"))

diagnosis_coverage <- coverage %>%
  group_by(compartment, state, Diagnosis1) %>%
  summarise(
    samples_any = n_distinct(SampleID),
    samples_ge5 = n_distinct(SampleID[n_cells >= 5]),
    samples_ge10 = n_distinct(SampleID[n_cells >= 10]),
    cells = sum(n_cells),
    .groups = "drop"
  )
write_csv(diagnosis_coverage, file.path(outdir, "Table_MN3_state_coverage_by_diagnosis.csv"))

sample_map <- bind_rows(cd4$md, b$md) %>%
  filter(!is.na(SampleID), nzchar(SampleID)) %>%
  distinct(SampleID, PatientID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series) %>%
  group_by(SampleID) %>%
  summarise(
    records = n(),
    patients = n_distinct(PatientID),
    diagnoses = n_distinct(Diagnosis1),
    inflammation_labels = n_distinct(Inflammation1),
    biologic_labels = n_distinct(Biologic),
    batches = n_distinct(Batch),
    acquisition_series = n_distinct(acquisition_series),
    PatientID = paste(sort(unique(PatientID)), collapse = ";"),
    Diagnosis1 = paste(sort(unique(Diagnosis1)), collapse = ";"),
    Inflammation1 = paste(sort(unique(Inflammation1)), collapse = ";"),
    Biologic = paste(sort(unique(Biologic)), collapse = ";"),
    Batch = paste(sort(unique(Batch)), collapse = ";"),
    series = paste(sort(unique(acquisition_series)), collapse = ";"),
    .groups = "drop"
  )
write_csv(sample_map, file.path(outdir, "Table_MN4_sample_metadata_consistency.csv"))

cross_tab <- bind_rows(cd4$md, b$md) %>%
  filter(!is.na(SampleID), nzchar(SampleID)) %>%
  distinct(SampleID, Diagnosis1, acquisition_series) %>%
  count(Diagnosis1, acquisition_series, name = "samples")
write_csv(cross_tab, file.path(outdir, "Table_MN5_diagnosis_by_acquisition_series.csv"))

summary_lines <- c(
  paste0("CD4 cells: ", ncol(cd4$obj), "; B cells: ", ncol(b$obj)),
  paste0("CD4 states: ", n_distinct(cd4$md$state), "; B states: ", n_distinct(b$md$state)),
  paste0("Unique samples across objects: ", n_distinct(c(cd4$md$SampleID, b$md$SampleID))),
  paste0("Samples with inconsistent metadata: ", sum(sample_map$patients > 1 | sample_map$diagnoses > 1 |
    sample_map$inflammation_labels > 1 | sample_map$biologic_labels > 1 | sample_map$batches > 1)),
  paste0("CD4/B cell barcode overlap: ", length(intersect(cd4$md$cell, b$md$cell)))
)
writeLines(summary_lines, file.path(outdir, "MN_input_audit_summary.txt"))
cat(paste(summary_lines, collapse = "\n"), "\n")

options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(SeuratObject))

out_dir <- file.path(getwd(), "Cell Press Redrawn Figure Set", "Source Data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

objects <- data.frame(
  panel = c("PBMC", "CD4", "CD8", "B"),
  path = c(
    "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/PBMC2026_SCVI.RDS",
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/CD4T2026_scvi50_umap_umap_SCVI_50_filtered.rds",
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/CD8T2026_scvi50_epoch400_umap_umap_SCVI_50E400_filtered.rds",
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/BCell2026_scvi50_umap_umap_SCVI_50_filtered.rds"
  ),
  reduction = c("scvi50.umap", "umap_SCVI_50", "umap_SCVI_50E400", "umap"),
  cap = c(220000L, 120000L, 95005L, 119174L),
  output = c(
    "UMAP_PBMC_from_Seurat.csv", "UMAP_CD4_from_Seurat.csv",
    "UMAP_CD8_from_Seurat.csv", "UMAP_B_from_Seurat.csv"
  )
)

stratified_sample <- function(data, cap, seed, minimum_per_stratum = 250L) {
  if (nrow(data) <= cap) return(data)
  set.seed(seed)
  strata <- interaction(data$AnnotationLevel2, data$Diagnosis1, drop = TRUE, lex.order = TRUE)
  groups <- split(seq_len(nrow(data)), strata)
  guaranteed <- unlist(
    lapply(groups, function(indices) sample(indices, min(length(indices), minimum_per_stratum))),
    use.names = FALSE
  )
  guaranteed <- unique(guaranteed)
  if (length(guaranteed) > cap) guaranteed <- sample(guaranteed, cap)
  remaining_n <- cap - length(guaranteed)
  if (remaining_n > 0L) {
    pool <- setdiff(seq_len(nrow(data)), guaranteed)
    additional <- sample(pool, min(remaining_n, length(pool)))
    guaranteed <- c(guaranteed, additional)
  }
  data[sort(guaranteed), , drop = FALSE]
}

manifest <- vector("list", nrow(objects))

for (i in seq_len(nrow(objects))) {
  spec <- objects[i, ]
  message("Reading ", spec$panel, " Seurat object: ", spec$path)
  object <- readRDS(spec$path)
  if (!inherits(object, "Seurat")) stop(spec$path, " is not a Seurat object")
  if (!spec$reduction %in% Reductions(object)) {
    stop("Reduction ", spec$reduction, " not found in ", spec$path)
  }

  embedding <- Embeddings(object, reduction = spec$reduction)
  metadata <- object[[]][rownames(embedding), , drop = FALSE]
  required <- c("AnnotationLevel2", "Diagnosis1", "PatientID")
  missing_columns <- setdiff(required, colnames(metadata))
  if (length(missing_columns)) stop("Missing metadata: ", paste(missing_columns, collapse = ", "))

  exported <- data.frame(
    cell = rownames(embedding),
    UMAP_1 = as.numeric(embedding[, 1]),
    UMAP_2 = as.numeric(embedding[, 2]),
    AnnotationLevel2 = as.character(metadata$AnnotationLevel2),
    Diagnosis1 = as.character(metadata$Diagnosis1),
    PatientID = as.character(metadata$PatientID),
    Batch = if ("Batch" %in% colnames(metadata)) as.character(metadata$Batch) else NA_character_,
    stringsAsFactors = FALSE
  )
  exported <- exported[
    is.finite(exported$UMAP_1) & is.finite(exported$UMAP_2) &
      !is.na(exported$AnnotationLevel2) & nzchar(exported$AnnotationLevel2),
    , drop = FALSE
  ]
  exported$Diagnosis1[is.na(exported$Diagnosis1) | !nzchar(exported$Diagnosis1)] <- "Unknown"
  plotted <- stratified_sample(exported, as.integer(spec$cap), seed = 20260825L + i)
  output_path <- file.path(out_dir, spec$output)
  write.csv(plotted, output_path, row.names = FALSE, quote = TRUE)

  info <- file.info(spec$path)
  manifest[[i]] <- data.frame(
    panel = spec$panel,
    seurat_object = spec$path,
    object_modified = format(info$mtime, "%Y-%m-%d %H:%M:%S"),
    object_size_bytes = as.numeric(info$size),
    reduction = spec$reduction,
    object_cells = ncol(object),
    embedding_cells = nrow(embedding),
    exported_cells = nrow(plotted),
    participants = length(unique(exported$PatientID[!is.na(exported$PatientID) & nzchar(exported$PatientID)])),
    level2_states = length(unique(exported$AnnotationLevel2)),
    UMAP_1_min = min(exported$UMAP_1),
    UMAP_1_max = max(exported$UMAP_1),
    UMAP_2_min = min(exported$UMAP_2),
    UMAP_2_max = max(exported$UMAP_2),
    source_data_file = output_path,
    stringsAsFactors = FALSE
  )
  message("Exported ", format(nrow(plotted), big.mark = ","), " of ",
          format(nrow(exported), big.mark = ","), " cells using ", spec$reduction)
  rm(object, embedding, metadata, exported, plotted)
  invisible(gc())
}

manifest <- do.call(rbind, manifest)
write.csv(
  manifest,
  file.path(out_dir, "Figure1_UMAP_Seurat_object_manifest.csv"),
  row.names = FALSE,
  quote = TRUE
)
print(manifest[, c("panel", "reduction", "object_cells", "exported_cells", "participants", "level2_states")])

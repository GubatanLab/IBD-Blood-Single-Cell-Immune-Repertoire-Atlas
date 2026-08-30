options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(SeuratObject))

root <- getwd()
source_dir <- file.path(root, "Cell Press Redrawn Figure Set", "Source Data")
preview_dir <- file.path(
  root, "Cell Press Redrawn Figure Set", "Preview Alternatives", "Source Data"
)
dir.create(preview_dir, recursive = TRUE, showWarnings = FALSE)

objects <- data.frame(
  panel = c("CD4", "CD8", "B"),
  path = c(
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/CD4T2026_scvi50_umap_umap_SCVI_50_filtered.rds",
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/CD8T2026_scvi50_epoch400_umap_umap_SCVI_50E400_filtered.rds",
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/BCell2026_scvi50_umap_umap_SCVI_50_filtered.rds"
  ),
  input = c(
    "UMAP_CD4_from_Seurat.csv",
    "UMAP_CD8_from_Seurat.csv",
    "UMAP_B_from_Seurat.csv"
  ),
  output = c(
    "Figure_1E_CD4_Level3_UMAP.csv",
    "Figure_1F_CD8_Level3_UMAP.csv",
    "Figure_1G_B_Level3_UMAP.csv"
  )
)

manifest <- vector("list", nrow(objects))

for (i in seq_len(nrow(objects))) {
  spec <- objects[i, ]
  message("Reading ", spec$panel, " object")
  object <- readRDS(spec$path)
  metadata <- object[[]]
  if (!"AnnotationLevel3" %in% colnames(metadata)) {
    stop("AnnotationLevel3 missing from ", spec$path)
  }

  input_path <- file.path(source_dir, spec$input)
  exported <- read.csv(input_path, check.names = FALSE)
  match_index <- match(exported$cell, rownames(metadata))
  if (anyNA(match_index)) {
    stop(sum(is.na(match_index)), " preview cells were not found in ", spec$panel, " metadata")
  }

  exported$AnnotationLevel3 <- as.character(metadata$AnnotationLevel3[match_index])
  if (any(is.na(exported$AnnotationLevel3) | !nzchar(exported$AnnotationLevel3))) {
    stop("Missing Level 3 annotations in ", spec$panel, " preview data")
  }

  output_path <- file.path(preview_dir, spec$output)
  write.csv(exported, output_path, row.names = FALSE, quote = TRUE)

  original_equal <- identical(
    as.character(exported$AnnotationLevel2),
    as.character(exported$AnnotationLevel3)
  )
  manifest[[i]] <- data.frame(
    panel = spec$panel,
    source_object = spec$path,
    source_umap = input_path,
    output = output_path,
    preview_cells = nrow(exported),
    participants = length(unique(exported$PatientID[!is.na(exported$PatientID) & nzchar(exported$PatientID)])),
    level3_states = length(unique(exported$AnnotationLevel3)),
    level2_identical_to_level3 = original_equal,
    stringsAsFactors = FALSE
  )

  rm(object, metadata, exported)
  invisible(gc())
}

manifest <- do.call(rbind, manifest)
write.csv(
  manifest,
  file.path(preview_dir, "Figure_1EFG_Level3_preview_manifest.csv"),
  row.names = FALSE,
  quote = TRUE
)
print(manifest[, c(
  "panel", "preview_cells", "participants", "level3_states",
  "level2_identical_to_level3"
)])

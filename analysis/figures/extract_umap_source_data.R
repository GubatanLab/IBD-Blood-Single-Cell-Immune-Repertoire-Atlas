options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(Seurat))

out_dir <- file.path(getwd(), "Cell Press Redrawn Figure Set", "Source Data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

inputs <- c(
  PBMC = "C:/path/to/private-legacy-manuscript-assets/Figure 1/UMAP/PBMC2026_scvi50_umap.rds",
  CD4 = "C:/path/to/private-legacy-manuscript-assets/Figure 1/UMAP/CD4T2026_scvi50_umap.rds",
  CD8 = "C:/path/to/private-legacy-manuscript-assets/Figure 1/UMAP/CD8T2026_scvi50_epoch400_umap.rds",
  B = "C:/path/to/private-legacy-manuscript-assets/Figure 1/UMAP/BCell2026_scvi50_umap.rds"
)

set.seed(20260824)
for (nm in names(inputs)) {
  obj <- readRDS(inputs[[nm]])
  emb <- Embeddings(obj[["umap"]])
  md <- obj@meta.data[rownames(emb), , drop = FALSE]
  annotation <- if ("AnnotationLevel2" %in% colnames(md)) md$AnnotationLevel2 else md[["Annotation Level2"]]
  df <- data.frame(
    cell = rownames(emb),
    UMAP_1 = emb[, 1],
    UMAP_2 = emb[, 2],
    AnnotationLevel2 = as.character(annotation),
    Diagnosis1 = as.character(md$Diagnosis1),
    SampleID = as.character(md$SampleID),
    stringsAsFactors = FALSE
  )
  cap <- if (nm == "PBMC") 120000L else 70000L
  if (nrow(df) > cap) {
    groups <- interaction(df$AnnotationLevel2, df$Diagnosis1, drop = TRUE)
    index <- split(seq_len(nrow(df)), groups)
    allocation <- pmax(100L, floor(as.numeric(cap) * as.numeric(lengths(index)) / nrow(df)))
    keep <- unlist(Map(function(ix, n) sample(ix, min(length(ix), n)), index, allocation), use.names = FALSE)
    if (length(keep) > cap) keep <- sample(keep, cap)
    df <- df[keep, , drop = FALSE]
  }
  write.csv(df, file.path(out_dir, paste0("UMAP_", nm, "_downsampled.csv")), row.names = FALSE)
  rm(obj, md, emb, df)
  invisible(gc())
}

pbmc <- readRDS(inputs[["PBMC"]])
cohort <- unique(pbmc@meta.data[, intersect(c("SampleID", "PatientID", "Diagnosis1"), colnames(pbmc@meta.data)), drop = FALSE])
write.csv(cohort, file.path(out_dir, "cohort_participants_from_PBMC.csv"), row.names = FALSE)

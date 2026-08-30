options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(SeuratObject))

objects <- data.frame(
  panel = c("CD4", "CD8", "B"),
  path = c(
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/CD4T2026_scvi50_umap_umap_SCVI_50_filtered.rds",
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/CD8T2026_scvi50_epoch400_umap_umap_SCVI_50E400_filtered.rds",
    "C:/path/to/private-user-home/OneDrive/Desktop/UMAP Repertoire/scvi_umap_outputs/BCell2026_scvi50_umap_umap_SCVI_50_filtered.rds"
  )
)

for (i in seq_len(nrow(objects))) {
  spec <- objects[i, ]
  message("Reading ", spec$panel)
  object <- readRDS(spec$path)
  metadata <- object[[]]
  annotation_columns <- grep(
    "annot|cell.?type|cluster|level|identity|ident",
    colnames(metadata),
    value = TRUE,
    ignore.case = TRUE
  )
  cat("\n### ", spec$panel, " metadata columns\n", sep = "")
  print(annotation_columns)
  for (column in annotation_columns) {
    values <- as.character(metadata[[column]])
    values <- sort(unique(values[!is.na(values) & nzchar(values)]))
    cat("\n", column, " (", length(values), "):\n", sep = "")
    print(values)
  }
  rm(object, metadata)
  invisible(gc())
}

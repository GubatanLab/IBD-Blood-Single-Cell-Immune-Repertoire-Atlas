args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("Provide one or more RDS paths")

suppressPackageStartupMessages(library(SeuratObject))

for (path in args) {
  cat("\nFILE\t", path, "\n", sep = "")
  object <- readRDS(path)
  cat("CLASS\t", paste(class(object), collapse = ","), "\n", sep = "")
  cat("DIM\t", paste(dim(object), collapse = "x"), "\n", sep = "")
  cat("ASSAYS\t", paste(Assays(object), collapse = ","), "\n", sep = "")
  reductions <- Reductions(object)
  cat("REDUCTIONS\t", paste(reductions, collapse = ","), "\n", sep = "")
  for (reduction in reductions) {
    embedding <- Embeddings(object, reduction = reduction)
    cat("REDUCTION_DIM\t", reduction, "\t", paste(dim(embedding), collapse = "x"), "\n", sep = "")
  }
  metadata <- object[[]]
  candidate_pattern <- paste(
    c("annot", "celltype", "cell_type", "level", "diagn", "disease", "patient",
      "participant", "sample", "series", "batch", "orig.ident", "scvi", "cluster"),
    collapse = "|"
  )
  candidate_columns <- grep(candidate_pattern, colnames(metadata), value = TRUE, ignore.case = TRUE)
  cat("METADATA_COLUMNS\t", paste(colnames(metadata), collapse = ","), "\n", sep = "")
  cat("CANDIDATE_COLUMNS\t", paste(candidate_columns, collapse = ","), "\n", sep = "")
  for (column in candidate_columns) {
    values <- metadata[[column]]
    unique_values <- unique(as.character(values[!is.na(values)]))
    cat("COLUMN\t", column, "\tN_UNIQUE=", length(unique_values), "\tVALUES=",
        paste(head(unique_values, 30), collapse = "|"), "\n", sep = "")
  }
  rm(object, metadata)
  invisible(gc())
}

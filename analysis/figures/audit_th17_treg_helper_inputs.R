#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
})

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 2 TCR/CD42026.rds",
  B = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026.rds"
)

genes <- unique(c(
  "RORC", "CCR6", "KLRB1", "IL7R", "IL23R", "CCL20", "IL17A", "IL17F",
  "TBX21", "IFNG", "CXCR3", "GZMK", "CSF2", "CCL5",
  "FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2", "TNFRSF18", "LRRC32", "ENTPD1",
  "CXCL13", "PDCD1", "ICOS", "MAF", "IL21", "TOX2", "CD40LG"
))

for (nm in names(paths)) {
  cat("\n===", nm, "===\n")
  stopifnot(file.exists(paths[[nm]]))
  obj <- readRDS(paths[[nm]])
  md <- obj[[]]
  cat("class:", paste(class(obj), collapse = ", "), "\n")
  cat("cells:", ncol(obj), " features:", nrow(obj), "\n")
  cat("assays:", paste(Assays(obj), collapse = ", "), "\n")
  cat("metadata columns:\n", paste(colnames(md), collapse = "\n"), "\n")
  if ("AnnotationLevel2" %in% colnames(md)) {
    cat("AnnotationLevel2 counts:\n")
    print(sort(table(md$AnnotationLevel2, useNA = "ifany"), decreasing = TRUE))
  }
  if ("AnnotationLevel3" %in% colnames(md)) {
    cat("AnnotationLevel3 counts:\n")
    print(sort(table(md$AnnotationLevel3, useNA = "ifany"), decreasing = TRUE))
  }
  if ("Diagnosis1" %in% colnames(md)) {
    cat("Diagnosis counts:\n")
    print(table(md$Diagnosis1, useNA = "ifany"))
  }
  if ("RNA" %in% Assays(obj)) {
    expr <- LayerData(obj[["RNA"]], layer = "data")
    cat("target genes present:\n", paste(intersect(genes, rownames(expr)), collapse = ";"), "\n")
    cat("target genes absent:\n", paste(setdiff(genes, rownames(expr)), collapse = ";"), "\n")
  }
  rm(obj, md)
  invisible(gc())
}

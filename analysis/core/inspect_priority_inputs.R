#!/usr/bin/env Rscript

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds",
  BCR = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026_scvi50_umap.rds"
)

suppressPackageStartupMessages(library(SeuratObject))

for (nm in names(paths)) {
  cat("\n=====", nm, "=====\n")
  x <- readRDS(paths[[nm]])
  cat("class:", paste(class(x), collapse = ", "), "\n")
  if (inherits(x, "Seurat")) {
    cat("dims:", nrow(x), "genes x", ncol(x), "cells\n")
    cat("assays:", paste(names(x@assays), collapse = ", "), "\n")
    cat("default assay:", DefaultAssay(x), "\n")
    cat("layers by assay:\n")
    for (a in names(x@assays)) {
      cat(" ", a, ":", paste(Layers(x[[a]]), collapse = ", "), "\n")
    }
    md <- x[[]]
    wanted <- grep("Sample|Patient|Diagn|Age|Sex|Calprotect|Inflam|Biologic|Response|Site|Batch|Annotation|module|Module|CDR3|gene_Dominant|SHM|Mutation|Isotype|IGH", names(md), value = TRUE, ignore.case = TRUE)
    cat("selected metadata columns (", length(wanted), "):\n", paste(wanted, collapse = "\n"), "\n", sep = "")
    cat("metadata sample values:\n")
    for (v in intersect(wanted, c("SampleID","PatientID","Diagnosis1","Calprotectin","Inflammation1","Biologic","BiologicResponse","Response","AnnotationLevel2","BcellState","Batch","Site"))) {
      vals <- unique(as.character(md[[v]])); vals <- vals[!is.na(vals)]
      cat(" ", v, " [", length(vals), "]: ", paste(head(vals, 12), collapse = ", "), "\n", sep = "")
    }
  } else {
    str(x, max.level = 1)
  }
  rm(x); gc()
}

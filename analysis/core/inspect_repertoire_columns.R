#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(SeuratObject))

ibd <- readRDS("C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/IBDBCR.rds")
cat("IBDBCR names:", paste(names(ibd), collapse=", "), "\n")
if (is.list(ibd) && "data" %in% names(ibd)) {
  cat("n repertoires:", length(ibd$data), "\n")
  cat("first data columns:\n", paste(names(ibd$data[[1]]), collapse="\n"), "\n", sep="")
  print(utils::head(ibd$data[[1]], 2))
}

x <- readRDS("C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026_scvi50_umap.rds")
md <- x[[]]
cat("B-cell mutation/alignment metadata:\n")
cat(paste(grep("germ|mut|align|identity|shm|sequence", names(md), value=TRUE, ignore.case=TRUE), collapse="\n"), "\n")
cat("reductions:", paste(names(x@reductions), collapse=", "), "\n")
for (r in names(x@reductions)) cat(r, nrow(Embeddings(x, r)), ncol(Embeddings(x, r)), "\n")

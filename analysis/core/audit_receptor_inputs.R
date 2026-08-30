options(width = 220)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds",
  B = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026_scvi50_umap.rds",
  TCR = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/IBDTCR.rds",
  BCR = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/IBDBCR.rds",
  BCRonly = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 3 BCR/IBDBCRonly.rds"
)
paths <- paths[c("CD8", "TCR", "BCR", "BCRonly")]

nonnull_rate <- function(x) {
  y <- trimws(as.character(x))
  mean(!is.na(x) & nzchar(y) & !tolower(y) %in% c("na", "nan", "none", "unknown"))
}

for (nm in names(paths)) {
  cat("\n=====", nm, "=====\n", paths[[nm]], "\n")
  if (!file.exists(paths[[nm]])) { cat("MISSING\n"); next }
  x <- readRDS(paths[[nm]])
  cat("class:", paste(class(x), collapse = ","), "\n")
  if (inherits(x, "Seurat")) {
    md <- x@meta.data
    cat("cells:", nrow(md), "features:", nrow(x), "\n")
    hits <- grep("(?i)(hla|mhc|tcr|bcr|cdr|v_gene|j_gene|d_gene|sequence|nucleotide|translation|framework|germline|alignment|isotype|constant|sample|patient|batch|diagnos|inflam|biologic|medication|level2|celltype)", colnames(md), value = TRUE, perl = TRUE)
    cat("metadata columns (matched):\n", paste(hits, collapse = "\n"), "\n")
    if (length(hits)) {
      rates <- sort(vapply(md[hits], nonnull_rate, numeric(1)), decreasing = TRUE)
      cat("nonmissing rates:\n")
      print(rates)
    }
    for (v in intersect(c("Diagnosis1","Inflammation1","Inflammation2","Batch","Biologic","Medication","Level2","CellType","celltype"), colnames(md))) {
      cat("table", v, "\n")
      print(sort(table(md[[v]], useNA = "ifany"), decreasing = TRUE))
    }
    rm(md)
  } else if (is.list(x)) {
    cat("top-level names:", paste(names(x), collapse = ", "), "\n")
    for (n in names(x)) {
      z <- x[[n]]
      if (is.data.frame(z) || data.table::is.data.table(z)) {
        cat("component", n, "rows", nrow(z), "cols", ncol(z), "\n")
        cat("columns:", paste(colnames(z), collapse = " | "), "\n")
        cat("first-row selected sequence fields:\n")
        hh <- grep("(?i)(sequence|germline|cdr|framework|fwr|align|v[.]name|j[.]name|v_gene|j_gene|isotype)", colnames(z), value = TRUE, perl = TRUE)
        if (length(hh) && nrow(z)) print(z[1, hh, drop = FALSE])
      } else if (is.list(z) && length(z) && (is.data.frame(z[[1]]) || data.table::is.data.table(z[[1]]))) {
        cat("component", n, "is a list of", length(z), "tables\n")
        cat("first table", names(z)[1], "rows", nrow(z[[1]]), "cols", ncol(z[[1]]), "\n")
        cat("columns:", paste(colnames(z[[1]]), collapse = " | "), "\n")
        hh <- grep("(?i)(sequence|germline|cdr|framework|fwr|align|v[.]name|j[.]name|v_gene|j_gene|isotype)", colnames(z[[1]]), value = TRUE, perl = TRUE)
        if (length(hh) && nrow(z[[1]])) print(z[[1]][1, hh, drop = FALSE])
      }
    }
  } else {
    str(x, max.level = 2)
  }
  rm(x); invisible(gc())
}

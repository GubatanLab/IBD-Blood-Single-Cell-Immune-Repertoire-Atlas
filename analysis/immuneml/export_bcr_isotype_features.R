root <- "C:/path/to/private-immuneml-workspace"
seurat_path <- "C:/path/to/private-immuneml-results/BCR/BCell2026_scvi50_umap.rds"
out_path <- file.path(root, "chain_bcr_immuneml/bcr_isotype_features.csv")
value_path <- file.path(root, "chain_bcr_immuneml/bcr_isotype_c_gene_values.csv")

classify_isotype <- function(c_gene) {
  x <- toupper(as.character(c_gene))
  ifelse(
    grepl("IGHA", x), "IgA",
    ifelse(
      grepl("IGHG", x), "IgG",
      ifelse(
        grepl("IGHM", x), "IgM",
        ifelse(grepl("IGHD", x), "IgD", NA_character_)
      )
    )
  )
}

obj <- readRDS(seurat_path)
md <- obj@meta.data

needed <- c(
  "SampleID", "Sample_Name", "Diagnosis1", "Diagnosis2",
  "BCR_Heavy_C_gene_Dominant", "BCR_Heavy_Molecule_Count"
)
missing <- setdiff(needed, colnames(md))
if (length(missing) > 0) {
  stop("Missing required Seurat metadata columns: ", paste(missing, collapse = ", "))
}

heavy <- md[, needed]
heavy$isotype_class <- classify_isotype(heavy$BCR_Heavy_C_gene_Dominant)
heavy$weight <- suppressWarnings(as.numeric(heavy$BCR_Heavy_Molecule_Count))
heavy$weight[is.na(heavy$weight) | heavy$weight <= 0] <- 1
heavy <- heavy[!is.na(heavy$SampleID) & !is.na(heavy$isotype_class), ]

value_counts <- as.data.frame(sort(table(md$BCR_Heavy_C_gene_Dominant, useNA = "ifany"), decreasing = TRUE))
colnames(value_counts) <- c("BCR_Heavy_C_gene_Dominant", "cell_count")
write.csv(value_counts, value_path, row.names = FALSE)

classes <- c("IgA", "IgG", "IgM", "IgD")
sample_ids <- sort(unique(as.character(heavy$SampleID)))
rows <- lapply(sample_ids, function(sample_id) {
  subset <- heavy[as.character(heavy$SampleID) == sample_id, ]
  diagnosis1 <- names(sort(table(subset$Diagnosis1), decreasing = TRUE))[1]
  diagnosis2 <- names(sort(table(subset$Diagnosis2), decreasing = TRUE))[1]
  sample_name <- names(sort(table(subset$Sample_Name), decreasing = TRUE))[1]

  cell_counts <- table(factor(subset$isotype_class, levels = classes))
  molecule_counts <- tapply(subset$weight, factor(subset$isotype_class, levels = classes), sum)
  molecule_counts[is.na(molecule_counts)] <- 0

  cell_total <- sum(cell_counts)
  molecule_total <- sum(molecule_counts)
  cell_props <- as.numeric(cell_counts) / ifelse(cell_total > 0, cell_total, 1)
  molecule_props <- as.numeric(molecule_counts) / ifelse(molecule_total > 0, molecule_total, 1)

  data.frame(
    SampleID = sample_id,
    Sample_Name = sample_name,
    Diagnosis1 = diagnosis1,
    Diagnosis2 = diagnosis2,
    bcr_heavy_isotype_cells = as.integer(cell_total),
    bcr_heavy_isotype_molecules = as.numeric(molecule_total),
    IgA_prop = molecule_props[1],
    IgG_prop = molecule_props[2],
    IgM_prop = molecule_props[3],
    IgD_prop = molecule_props[4],
    IgA_cell_prop = cell_props[1],
    IgG_cell_prop = cell_props[2],
    IgM_cell_prop = cell_props[3],
    IgD_cell_prop = cell_props[4],
    stringsAsFactors = FALSE
  )
})

features <- do.call(rbind, rows)
write.csv(features, out_path, row.names = FALSE)

cat("Wrote", out_path, "with", nrow(features), "samples\n")
cat("Wrote", value_path, "\n")
cat("Isotype feature summary:\n")
print(summary(features[, c("IgA_prop", "IgG_prop", "IgM_prop", "IgD_prop")]))

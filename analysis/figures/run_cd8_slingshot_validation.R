suppressPackageStartupMessages({
  library(data.table)
  library(slingshot)
})

set.seed(20260828)

root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
out <- file.path(root, "High Impact Additional Analyses", "CD8 TCR Trajectory")
input_file <- file.path(out, "Table_TJ1_cd8_trajectory_input.csv.gz")

cells <- fread(input_file, showProgress = FALSE)
scvi_cols <- paste0("scvi_", seq_len(20))

# The scalar primary trajectory is restricted to the circulating conventional
# CD8 continuum. Trm is retained as a branch-specific sensitivity population,
# because neither graph distance nor Slingshot supports forcing it onto the
# same one-dimensional axis. Within the primary continuum, retain every paired
# alpha-beta cell and add up to 12 reference cells per participant and state.
primary_cells <- cells[state != "CD8 Trm"]
primary_cells[, random_order := runif(.N)]
stratified <- primary_cells[order(random_order), head(.I, 12), by = .(state, SampleID)]$V1
reference_index <- sort(unique(c(stratified, which(primary_cells$paired_alpha_beta))))
reference <- primary_cells[reference_index]

latent <- as.matrix(reference[, ..scvi_cols])
storage.mode(latent) <- "double"
cluster_labels <- factor(reference$state)

sds <- slingshot(
  latent,
  clusterLabels = cluster_labels,
  start.clus = "CD8 Naive",
  end.clus = c("CD8 HLA-DR+", "CD8 Temra", "CD8 Tem GZMB+"),
  stretch = 0,
  reweight = TRUE,
  reassign = TRUE,
  shrink = TRUE
)

pt <- slingPseudotime(sds, na = FALSE)
weights <- slingCurveWeights(sds)
pt[!is.finite(pt)] <- NA_real_
weights[!is.finite(weights)] <- 0

weighted_numerator <- rowSums(pt * weights, na.rm = TRUE)
weighted_denominator <- rowSums(weights * is.finite(pt), na.rm = TRUE)
consensus <- weighted_numerator / weighted_denominator
consensus[!is.finite(consensus)] <- NA_real_
finite <- is.finite(consensus)
consensus[finite] <- rank(consensus[finite], ties.method = "average") / sum(finite)

assigned_lineage <- apply(weights, 1, function(x) if (all(!is.finite(x)) || max(x, na.rm = TRUE) <= 0) NA_integer_ else which.max(x))
max_weight <- apply(weights, 1, function(x) if (all(!is.finite(x))) NA_real_ else max(x, na.rm = TRUE))

reference_output <- data.table(
  cell = reference$cell,
  SampleID = reference$SampleID,
  state = reference$state,
  paired_alpha_beta = reference$paired_alpha_beta,
  slingshot_pseudotime = consensus,
  assigned_lineage = assigned_lineage,
  maximum_lineage_weight = max_weight
)
for (i in seq_len(ncol(pt))) {
  reference_output[[paste0("lineage_", i, "_pseudotime")]] <- pt[, i]
  reference_output[[paste0("lineage_", i, "_weight")]] <- weights[, i]
}
fwrite(reference_output, file.path(out, "Table_ST1_cd8_slingshot_reference.csv.gz"))

lineages <- slingLineages(sds)
lineage_output <- rbindlist(lapply(seq_along(lineages), function(i) {
  data.table(lineage = i, cluster_order = paste(lineages[[i]], collapse = " -> "))
}))
fwrite(lineage_output, file.path(out, "Table_ST1_cd8_slingshot_lineages.csv"))

qc <- data.table(
  metric = c(
    "Slingshot reference cells",
    "Paired alpha-beta cells retained in reference",
    "Finite Slingshot reference pseudotimes",
    "Slingshot lineages",
    "Root cluster",
    "Terminal clusters",
    "Branch-specific state excluded from scalar primary trajectory"
  ),
  value = c(
    nrow(reference),
    sum(reference$paired_alpha_beta),
    sum(finite),
    length(lineages),
    "CD8 Naive",
    "CD8 HLA-DR+; CD8 Temra; CD8 Tem GZMB+",
    "CD8 Trm"
  )
)
fwrite(qc, file.path(out, "Table_ST1_cd8_slingshot_qc.csv"))

cat("Reference cells:", nrow(reference), "\n")
cat("Paired alpha-beta reference cells:", sum(reference$paired_alpha_beta), "\n")
cat("Finite pseudotimes:", sum(finite), "\n")
print(lineage_output)

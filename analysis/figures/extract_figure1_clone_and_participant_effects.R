options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(SeuratObject))

root <- getwd()
source_dir <- file.path(root, "Cell Press Redrawn Figure Set", "Source Data")
pbmc_path <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/PBMC2026_SCVI.RDS"

chr <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  trimws(x)
}

nonblank <- function(x) {
  x <- chr(x)
  nzchar(x) & !tolower(x) %in% c("na", "nan", "none", "null")
}

first_nonblank <- function(x) {
  x <- chr(x)
  hit <- x[nonblank(x)]
  if (length(hit)) hit[[1]] else ""
}

message("Reading PBMC object: ", pbmc_path)
object <- readRDS(pbmc_path)
md <- object@meta.data
cells <- rownames(md)

patient <- chr(md$PatientID)
patient[!nonblank(patient)] <- chr(md$SampleID[!nonblank(patient)])

ag_v <- chr(md$TCR_Alpha_Gamma_V_gene_Dominant)
ag_j <- chr(md$TCR_Alpha_Gamma_J_gene_Dominant)
ag_cdr3 <- chr(md$TCR_Alpha_Gamma_CDR3_Translation_Dominant)
bd_v <- chr(md$TCR_Beta_Delta_V_gene_Dominant)
bd_j <- chr(md$TCR_Beta_Delta_J_gene_Dominant)
bd_cdr3 <- chr(md$TCR_Beta_Delta_CDR3_Translation_Dominant)

heavy_v <- chr(md$BCR_Heavy_V_gene_Dominant)
heavy_j <- chr(md$BCR_Heavy_J_gene_Dominant)
heavy_cdr3 <- chr(md$BCR_Heavy_CDR3_Translation_Dominant)
light_v <- chr(md$BCR_Light_V_gene_Dominant)
light_j <- chr(md$BCR_Light_J_gene_Dominant)
light_cdr3 <- chr(md$BCR_Light_CDR3_Translation_Dominant)

paired_ab <- nonblank(patient) & grepl("^TRA", ag_v) & grepl("^TRB", bd_v) &
  nonblank(ag_j) & nonblank(bd_j) & nonblank(ag_cdr3) & nonblank(bd_cdr3)
paired_gd <- nonblank(patient) & grepl("^TRG", ag_v) & grepl("^TRD", bd_v) &
  nonblank(ag_j) & nonblank(bd_j) & nonblank(ag_cdr3) & nonblank(bd_cdr3)
paired_hl <- nonblank(patient) & nonblank(heavy_v) & nonblank(heavy_j) & nonblank(heavy_cdr3) &
  nonblank(light_v) & nonblank(light_j) & nonblank(light_cdr3)

tcr_id <- rep(NA_character_, nrow(md))
tcr_id[paired_ab] <- paste(
  patient[paired_ab], "AB", ag_v[paired_ab], ag_j[paired_ab], ag_cdr3[paired_ab],
  bd_v[paired_ab], bd_j[paired_ab], bd_cdr3[paired_ab], sep = "|"
)
tcr_id[paired_gd] <- paste(
  patient[paired_gd], "GD", ag_v[paired_gd], ag_j[paired_gd], ag_cdr3[paired_gd],
  bd_v[paired_gd], bd_j[paired_gd], bd_cdr3[paired_gd], sep = "|"
)
bcr_id <- rep(NA_character_, nrow(md))
bcr_id[paired_hl] <- paste(
  patient[paired_hl], heavy_v[paired_hl], heavy_j[paired_hl], heavy_cdr3[paired_hl],
  light_v[paired_hl], light_j[paired_hl], light_cdr3[paired_hl], sep = "|"
)

tcr_size <- integer(nrow(md))
bcr_size <- integer(nrow(md))
tcr_keep <- which(!is.na(tcr_id))
bcr_keep <- which(!is.na(bcr_id))
tcr_size[tcr_keep] <- ave(rep.int(1L, length(tcr_keep)), tcr_id[tcr_keep], FUN = sum)
bcr_size[bcr_keep] <- ave(rep.int(1L, length(bcr_keep)), bcr_id[bcr_keep], FUN = sum)

panel_files <- c(
  CD4 = "UMAP_CD4_from_Seurat.csv",
  CD8 = "UMAP_CD8_from_Seurat.csv",
  B = "UMAP_B_from_Seurat.csv"
)
overlay_rows <- list()
for (panel in names(panel_files)) {
  plotted <- read.csv(file.path(source_dir, panel_files[[panel]]), check.names = FALSE)
  index <- match(plotted$cell, cells)
  if (anyNA(index)) {
    warning(panel, ": ", sum(is.na(index)), " displayed cells did not match the PBMC metadata")
  }
  if (panel == "B") {
    size <- bcr_size[index]
    family <- ifelse(size > 0, "heavy-light BCR", "unpaired")
  } else {
    size <- tcr_size[index]
    family <- ifelse(paired_ab[index], "alpha-beta TCR", ifelse(paired_gd[index], "gamma-delta TCR", "unpaired"))
  }
  overlay_rows[[panel]] <- data.frame(
    cell = plotted$cell,
    panel = panel,
    paired_family = family,
    paired_clone_size = size,
    expanded_paired = size >= 2,
    stringsAsFactors = FALSE
  )
  message(panel, ": matched ", sum(!is.na(index)), "/", length(index),
          "; expanded paired cells = ", sum(size >= 2, na.rm = TRUE))
}
overlay <- do.call(rbind, overlay_rows)
write.csv(overlay, file.path(source_dir, "Figure1_clone_engagement_umap.csv"), row.names = FALSE)

state <- chr(md$AnnotationLevel2)
diagnosis <- chr(md$Diagnosis1)
batch <- chr(md$Batch)
lineage <- rep("Other", nrow(md))
lineage[grepl("^(CD4|TReg)", state)] <- "CD4/Treg"
lineage[grepl("^CD8", state) | state %in% c("MAIT", "gdT")] <- "CD8/innate-like T"
lineage[grepl("B Cell|Plasma B|memory B|Naive B|Naive-IFN B|Transitional B", state, ignore.case = TRUE)] <- "B/plasma"

summarize_clone_engagement <- function(panel, panel_index, clonotype_id, clone_size) {
  paired <- panel_index & !is.na(clonotype_id) & clone_size > 0
  expanded <- panel_index & !is.na(clonotype_id) & clone_size >= 2
  data.frame(
    panel = panel,
    total_cells = sum(panel_index),
    participants = length(unique(patient[panel_index & nonblank(patient)])),
    paired_cells = sum(paired),
    paired_participants = length(unique(patient[paired & nonblank(patient)])),
    expanded_cells = sum(expanded),
    expanded_clonotypes = length(unique(clonotype_id[expanded])),
    expanded_participants = length(unique(patient[expanded & nonblank(patient)])),
    expanded_pct_of_paired = if (sum(paired)) 100 * sum(expanded) / sum(paired) else NA_real_,
    stringsAsFactors = FALSE
  )
}

clone_summary <- rbind(
  summarize_clone_engagement("CD4", lineage == "CD4/Treg", tcr_id, tcr_size),
  summarize_clone_engagement("CD8", lineage == "CD8/innate-like T", tcr_id, tcr_size),
  summarize_clone_engagement("B", lineage == "B/plasma", bcr_id, bcr_size)
)
write.csv(
  clone_summary,
  file.path(source_dir, "Figure1_expanded_clone_summary.csv"),
  row.names = FALSE
)

selected_states <- c(
  "CD4 Naive", "TReg Cytotoxic",
  "CD8 Naive", "CD8 Tem GZMB+",
  "IgM Plasma B Cell", "Switched memory B"
)

valid <- nonblank(patient) & lineage != "Other"
participant_lineage <- split(which(valid), interaction(patient[valid], lineage[valid], drop = TRUE))
abundance_rows <- lapply(participant_lineage, function(ix) {
  total <- length(ix)
  data.frame(
    PatientID = patient[ix[[1]]],
    Diagnosis1 = first_nonblank(diagnosis[ix]),
    acquisition_series = sub("[AB]$", "", first_nonblank(batch[ix])),
    lineage = lineage[ix[[1]]],
    cell_state = selected_states,
    state_cells = vapply(selected_states, function(s) sum(state[ix] == s), integer(1)),
    lineage_cells = total,
    stringsAsFactors = FALSE
  )
})
abundance <- do.call(rbind, abundance_rows)
state_lineage <- c(
  "CD4 Naive" = "CD4/Treg", "TReg Cytotoxic" = "CD4/Treg",
  "CD8 Naive" = "CD8/innate-like T", "CD8 Tem GZMB+" = "CD8/innate-like T",
  "IgM Plasma B Cell" = "B/plasma", "Switched memory B" = "B/plasma"
)
abundance <- abundance[abundance$lineage == unname(state_lineage[abundance$cell_state]), , drop = FALSE]
abundance$state_fraction_pct <- 100 * abundance$state_cells / abundance$lineage_cells
abundance$logit_fraction <- log((abundance$state_cells + 0.5) / (abundance$lineage_cells - abundance$state_cells + 0.5))
write.csv(abundance, file.path(source_dir, "Figure1_participant_state_abundance.csv"), row.names = FALSE)

cat("Wrote clone engagement, full-data clone summary, and participant-level state abundance source data.\n")

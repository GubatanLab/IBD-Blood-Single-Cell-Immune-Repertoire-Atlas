#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(turboGliph)
  library(SeuratObject)
  library(dplyr)
  library(stringr)
  library(readr)
  library(tibble)
})

set.seed(20260824)
out_root <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses/Priority Analyses/BetaOnly_GLIPH2"
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds"
)

valid <- function(x) !is.na(x) & trimws(as.character(x)) != "" & toupper(trimws(as.character(x))) != "NA"
pieces <- list()
for (nm in names(paths)) {
  message("Loading exact ", nm, " beta-chain calls")
  x <- readRDS(paths[[nm]])
  md <- as.data.frame(x[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  pieces[[nm]] <- md %>%
    filter(grepl("^TRBC", TCR_Beta_Delta_C_gene_Dominant),
           grepl("^TRBV", TCR_Beta_Delta_V_gene_Dominant),
           valid(TCR_Beta_Delta_CDR3_Translation_Dominant),
           valid(SampleID), Diagnosis1 %in% c("CD", "UC", "Control")) %>%
    transmute(cell, SampleID = as.character(SampleID), Diagnosis1 = as.character(Diagnosis1),
              CDR3b = toupper(as.character(TCR_Beta_Delta_CDR3_Translation_Dominant)),
              TRBV = as.character(TCR_Beta_Delta_V_gene_Dominant)) %>%
    mutate(CDR3b = ifelse(startsWith(CDR3b, "C"), CDR3b, paste0("C", CDR3b)),
           CDR3b = ifelse(endsWith(CDR3b, "F"), CDR3b, paste0(CDR3b, "F"))) %>%
    filter(grepl("^C[A-Z]+F$", CDR3b), !grepl("\\*|X", CDR3b))
  rm(x, md); gc()
}
all_cells <- bind_rows(pieces) %>% distinct(cell, .keep_all = TRUE)
all_receptors <- all_cells %>% count(SampleID, Diagnosis1, CDR3b, TRBV, name = "counts") %>%
  mutate(patient = paste0(SampleID, ":", Diagnosis1), group = Diagnosis1)
write_csv(all_receptors, file.path(out_root, "exact_beta_receptors_all_participants.csv"))

score_clusters <- function(res, input_df, g1, g2) {
  cl <- res[["cluster_properties"]]
  if (is.null(cl) || nrow(cl) == 0) return(tibble())
  # Enrichment is based on participant-by-sequence presence, preventing large
  # repertoires or expanded clones from dominating the contingency table.
  seq_group <- input_df %>% select(CDR3b, SampleID, group) %>% distinct()
  bg <- seq_group %>% count(group, name = "bg_n")
  bg1 <- bg$bg_n[bg$group == g1]; bg2 <- bg$bg_n[bg$group == g2]
  if (!length(bg1)) bg1 <- 0; if (!length(bg2)) bg2 <- 0
  bind_rows(lapply(seq_len(nrow(cl)), function(i) {
    members <- unique(str_split(str_squish(cl$members[i]), "\\s+")[[1]])
    mtab <- seq_group %>% filter(CDR3b %in% members) %>% count(group, name = "n")
    a <- ifelse(any(mtab$group == g1), mtab$n[mtab$group == g1], 0)
    b <- ifelse(any(mtab$group == g2), mtab$n[mtab$group == g2], 0)
    ft <- fisher.test(matrix(c(a, b, max(bg1-a, 0), max(bg2-b, 0)), nrow = 2))
    tibble(cluster_index = i, tag = cl$tag[i], type = cl$type[i], cluster_size = cl$cluster_size[i],
           members = cl$members[i], n_g1 = a, n_g2 = b, prop_g1 = ifelse(a+b > 0, a/(a+b), NA_real_),
           fisher_p = ft$p.value, fisher_or = unname(ft$estimate))
  })) %>% mutate(fdr = p.adjust(fisher_p, "BH")) %>% arrange(fdr, fisher_p)
}

comparisons <- list(c("CD", "Control"), c("UC", "Control"), c("CD", "UC"))
for (cmp in comparisons) {
  g1 <- cmp[1]; g2 <- cmp[2]; label <- paste0(g1, "_vs_", g2)
  outdir <- file.path(out_root, label); dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  in_df <- all_receptors %>% filter(group %in% c(g1, g2))
  write_csv(in_df, file.path(outdir, "input_sequences.csv"))
  message("Running exact beta-only GLIPH2: ", label, " (", nrow(in_df), " participant-sequence rows)")
  res <- gliph2(cdr3_sequences = in_df %>% select(CDR3b, TRBV, patient, counts),
                result_folder = outdir, sim_depth = 100, n_cores = 4)
  saveRDS(res, file.path(outdir, "gliph2_result.rds"))
  enrich <- score_clusters(res, in_df, g1, g2)
  write_csv(enrich, file.path(outdir, "cluster_group_enrichment.csv"))
  write_csv(tibble(
    comparison = label, n_input_rows = nrow(in_df), n_unique_cdr3 = n_distinct(in_df$CDR3b),
    n_participants = n_distinct(in_df$SampleID),
    n_clusters = ifelse(is.null(res[["cluster_properties"]]), 0, nrow(res[["cluster_properties"]])),
    n_clusters_fdr_lt_0_05 = ifelse("fdr" %in% names(enrich), sum(enrich$fdr < .05, na.rm = TRUE), 0)
  ), file.path(outdir, "summary.csv"))
}
message("Exact beta-only GLIPH2 complete")

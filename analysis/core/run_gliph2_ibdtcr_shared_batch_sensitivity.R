#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(turboGliph)
  library(dplyr)
  library(stringr)
  library(readr)
  library(tibble)
})

set.seed(20260824)
input_rds <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 2 TCR/IBDTCR.rds"
out_root <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses/Priority Analyses/IBDTCR_BetaOnly_GLIPH2"
run_dir <- file.path(out_root, "CD_vs_UC_shared_S1-S5_run_20260824")
dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)

ibd <- readRDS(input_rds)
meta <- as_tibble(ibd$meta) %>%
  transmute(
    Sample = as.character(Sample),
    SampleID = as.character(PatientID),
    Diagnosis1 = case_when(
      str_to_lower(as.character(Diagnosis1)) == "control" ~ "Control",
      str_to_lower(as.character(Diagnosis1)) == "cd" ~ "CD",
      str_to_lower(as.character(Diagnosis1)) == "uc" ~ "UC",
      TRUE ~ as.character(Diagnosis1)
    ),
    Batch = as.character(Batch),
    BatchSeries = str_remove(Batch, "[AB]$")
  )

write_csv(
  meta %>% count(Diagnosis1, Batch, BatchSeries, name = "n_participants"),
  file.path(out_root, "IBDTCR_participants_by_diagnosis_and_batch.csv")
)

valid <- function(x) !is.na(x) & trimws(as.character(x)) != "" & toupper(trimws(as.character(x))) != "NA"

pieces <- lapply(seq_len(nrow(meta)), function(i) {
  dd <- ibd$data[[meta$Sample[i]]]
  if (is.null(dd) || !nrow(dd)) return(NULL)
  tibble(
    SampleID = meta$SampleID[i],
    Diagnosis1 = meta$Diagnosis1[i],
    Batch = meta$Batch[i],
    BatchSeries = meta$BatchSeries[i],
    CDR3b = toupper(as.character(dd[["CDR3.aa"]])),
    TRBV = as.character(dd[["V.name"]])
  ) %>%
    filter(grepl("^TRBV", TRBV), valid(CDR3b), grepl("^C[A-Z]+F$", CDR3b), !grepl("\\*|X", CDR3b))
})

receptors <- bind_rows(pieces) %>%
  group_by(SampleID, Diagnosis1, Batch, BatchSeries, CDR3b, TRBV) %>%
  summarise(.groups = "drop") %>%
  mutate(patient = paste0(SampleID, ":", Diagnosis1), group = Diagnosis1, counts = 1)

write_csv(
  receptors %>% distinct(SampleID, Diagnosis1, Batch, BatchSeries) %>%
    count(Diagnosis1, Batch, BatchSeries, name = "n_participants_with_TRBV"),
  file.path(out_root, "IBDTCR_TRBV_participants_by_diagnosis_and_batch.csv")
)

input_df <- receptors %>%
  filter(Diagnosis1 %in% c("CD", "UC"), BatchSeries %in% paste0("S", 1:5))
write_csv(input_df, file.path(run_dir, "input_sequences.csv"))

message("Running shared-batch IBDTCR beta-only GLIPH2: CD vs UC, S1-S5")
res <- gliph2(
  cdr3_sequences = input_df %>% select(CDR3b, TRBV, patient, counts),
  result_folder = run_dir,
  sim_depth = 100,
  n_cores = 4
)
saveRDS(res, file.path(run_dir, "gliph2_result.rds"))

score_clusters_participant <- function(res, input_df) {
  cl <- res[["cluster_properties"]]
  participant_group <- input_df %>% distinct(SampleID, group)
  totals <- participant_group %>% count(group, name = "n_participants")
  n1 <- totals$n_participants[totals$group == "CD"]
  n2 <- totals$n_participants[totals$group == "UC"]
  seq_participant <- input_df %>% distinct(CDR3b, SampleID, group)

  bind_rows(lapply(seq_len(nrow(cl)), function(i) {
    members <- unique(str_split(str_squish(as.character(cl$members[i])), "\\s+")[[1]])
    carriers <- seq_participant %>% filter(CDR3b %in% members) %>% distinct(SampleID, group)
    a <- n_distinct(carriers$SampleID[carriers$group == "CD"])
    b <- n_distinct(carriers$SampleID[carriers$group == "UC"])
    ft <- fisher.test(matrix(c(a, b, n1 - a, n2 - b), nrow = 2), alternative = "two.sided")
    tibble(
      cluster_index = i,
      tag = as.character(cl$tag[i]),
      type = as.character(cl$type[i]),
      cluster_size = as.numeric(cl$cluster_size[i]),
      members = as.character(cl$members[i]),
      n_carriers_g1 = a,
      n_carriers_g2 = b,
      n_participants_g1 = n1,
      n_participants_g2 = n2,
      carrier_fraction_g1 = a / n1,
      carrier_fraction_g2 = b / n2,
      carrier_difference = a / n1 - b / n2,
      fisher_p = ft$p.value,
      fisher_or = unname(ft$estimate)
    )
  })) %>% mutate(fdr = p.adjust(fisher_p, method = "BH")) %>% arrange(fdr, fisher_p)
}

enrich <- score_clusters_participant(res, input_df)
write_csv(enrich, file.path(run_dir, "cluster_participant_enrichment.csv"))
write_csv(
  tibble(
    comparison = "CD_vs_UC_shared_S1-S5",
    source = "IBDTCR.rds",
    restriction = "BatchSeries S1-S5",
    n_input_rows = nrow(input_df),
    n_unique_cdr3 = n_distinct(input_df$CDR3b),
    n_CD_participants = n_distinct(input_df$SampleID[input_df$group == "CD"]),
    n_UC_participants = n_distinct(input_df$SampleID[input_df$group == "UC"]),
    n_clusters = nrow(res[["cluster_properties"]]),
    n_tag_rows_fdr_lt_0_05 = sum(enrich$fdr < .05, na.rm = TRUE)
  ),
  file.path(run_dir, "summary.csv")
)

message("Shared-batch sensitivity complete")

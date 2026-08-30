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
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

ibd <- readRDS(input_rds)
meta <- as_tibble(ibd$meta) %>%
  mutate(
    Diagnosis1 = as.character(Diagnosis1),
    DiagnosisNorm = case_when(
      str_to_lower(Diagnosis1) == "control" ~ "Control",
      str_to_lower(Diagnosis1) == "cd" ~ "CD",
      str_to_lower(Diagnosis1) == "uc" ~ "UC",
      TRUE ~ Diagnosis1
    ),
    Participant = as.character(PatientID)
  )

valid <- function(x) !is.na(x) & trimws(as.character(x)) != "" & toupper(trimws(as.character(x))) != "NA"

message("Extracting TRBV records from IBDTCR.rds...")
pieces <- lapply(seq_len(nrow(meta)), function(i) {
  sample_name <- as.character(meta$Sample[i])
  dd <- ibd$data[[sample_name]]
  if (is.null(dd) || !nrow(dd)) return(NULL)
  tibble(
    SampleID = meta$Participant[i],
    Diagnosis1 = meta$DiagnosisNorm[i],
    CDR3b = toupper(as.character(dd[["CDR3.aa"]])),
    TRBV = as.character(dd[["V.name"]]),
    native_clones = suppressWarnings(as.numeric(dd[["Clones"]]))
  ) %>%
    filter(Diagnosis1 %in% c("Control", "UC", "CD"), grepl("^TRBV", TRBV),
           valid(CDR3b), grepl("^C[A-Z]+F$", CDR3b), !grepl("\\*|X", CDR3b))
})

receptors <- bind_rows(pieces) %>%
  group_by(SampleID, Diagnosis1, CDR3b, TRBV) %>%
  summarise(native_clones = sum(native_clones, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    patient = paste0(SampleID, ":", Diagnosis1),
    group = Diagnosis1,
    counts = 1
  )

write_csv(receptors, file.path(out_root, "IBDTCR_TRBV_participant_sequence_table.csv"))
write_csv(receptors %>% count(Diagnosis1, name = "n_participant_sequence_records") %>%
            left_join(receptors %>% group_by(Diagnosis1) %>% summarise(
              n_participants = n_distinct(SampleID), n_unique_cdr3 = n_distinct(CDR3b),
              native_clones_nonunit_fraction = mean(native_clones != 1, na.rm = TRUE), .groups = "drop"),
                      by = "Diagnosis1"),
          file.path(out_root, "IBDTCR_TRBV_input_audit.csv"))

score_clusters_participant <- function(res, input_df, g1, g2) {
  cl <- res[["cluster_properties"]]
  if (is.null(cl) || !nrow(cl)) return(tibble())
  participant_group <- input_df %>% distinct(SampleID, group)
  totals <- participant_group %>% count(group, name = "n_participants")
  n1 <- totals$n_participants[totals$group == g1]
  n2 <- totals$n_participants[totals$group == g2]
  seq_participant <- input_df %>% distinct(CDR3b, SampleID, group)

  bind_rows(lapply(seq_len(nrow(cl)), function(i) {
    members <- unique(str_split(str_squish(as.character(cl$members[i])), "\\s+")[[1]])
    carriers <- seq_participant %>% filter(CDR3b %in% members) %>% distinct(SampleID, group)
    a <- n_distinct(carriers$SampleID[carriers$group == g1])
    b <- n_distinct(carriers$SampleID[carriers$group == g2])
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

comparisons <- list(c("CD", "Control"), c("UC", "Control"), c("CD", "UC"))
for (cmp in comparisons) {
  g1 <- cmp[1]; g2 <- cmp[2]
  label <- paste0(g1, "_vs_", g2)
  run_dir <- file.path(out_root, paste0(label, "_run_20260824"))
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  input_df <- receptors %>% filter(group %in% c(g1, g2))
  write_csv(input_df, file.path(run_dir, "input_sequences.csv"))
  message("Running IBDTCR beta-only GLIPH2: ", label, " (", nrow(input_df), " participant-sequence rows)")
  res <- gliph2(
    cdr3_sequences = input_df %>% select(CDR3b, TRBV, patient, counts),
    result_folder = run_dir,
    sim_depth = 100,
    n_cores = 4
  )
  saveRDS(res, file.path(run_dir, "gliph2_result.rds"))
  enrich <- score_clusters_participant(res, input_df, g1, g2)
  write_csv(enrich, file.path(run_dir, "cluster_participant_enrichment.csv"))
  write_csv(tibble(
    comparison = label,
    source = "IBDTCR.rds",
    chain_filter = "V.name begins TRBV",
    gliph_weighting = "one per participant-sequence record",
    enrichment_unit = "participant carrier",
    n_input_rows = nrow(input_df),
    n_unique_cdr3 = n_distinct(input_df$CDR3b),
    n_participants = n_distinct(input_df$SampleID),
    n_clusters = ifelse(is.null(res[["cluster_properties"]]), 0, nrow(res[["cluster_properties"]])),
    n_clusters_fdr_lt_0_05 = ifelse("fdr" %in% names(enrich), sum(enrich$fdr < .05, na.rm = TRUE), 0),
    n_first_group_enriched_fdr_lt_0_05 = ifelse("fdr" %in% names(enrich), sum(enrich$fdr < .05 & enrich$carrier_difference > 0, na.rm = TRUE), 0),
    n_second_group_enriched_fdr_lt_0_05 = ifelse("fdr" %in% names(enrich), sum(enrich$fdr < .05 & enrich$carrier_difference < 0, na.rm = TRUE), 0)
  ), file.path(run_dir, "summary.csv"))
}

message("IBDTCR beta-only GLIPH2 complete")

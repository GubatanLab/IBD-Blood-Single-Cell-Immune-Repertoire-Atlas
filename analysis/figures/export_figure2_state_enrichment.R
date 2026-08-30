#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(dplyr)
  library(tidyr)
  library(readr)
})

set.seed(20260824)

root <- "C:/path/to/private-manuscript-workspace"
out_dir <- file.path(root, "Cell Press Redrawn Figure Set", "Source Data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds"
)
stopifnot(all(file.exists(paths)))

valid <- function(x) {
  !is.na(x) & nzchar(trimws(as.character(x))) & toupper(trimws(as.character(x))) != "NA"
}

read_metadata <- function(path, compartment) {
  obj <- readRDS(path)
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  keep <- grepl("^TRBC", md$TCR_Beta_Delta_C_gene_Dominant) &
    valid(md$TCR_Beta_Delta_V_gene_Dominant) &
    valid(md$TCR_Beta_Delta_J_gene_Dominant) &
    valid(md$TCR_Beta_Delta_CDR3_Translation_Dominant) &
    valid(md$SampleID) & valid(md$AnnotationLevel2)
  md <- md[keep, , drop = FALSE]
  md$clone_id <- paste(
    md$TCR_Beta_Delta_V_gene_Dominant,
    md$TCR_Beta_Delta_J_gene_Dominant,
    md$TCR_Beta_Delta_CDR3_Translation_Dominant,
    sep = "|"
  )
  md %>%
    transmute(
      cell,
      SampleID = as.character(SampleID),
      Diagnosis1 = as.character(Diagnosis1),
      Batch = as.character(Batch),
      compartment = compartment,
      state = as.character(AnnotationLevel2),
      clone_id
    )
}

cells <- bind_rows(
  read_metadata(paths[["CD4"]], "CD4"),
  read_metadata(paths[["CD8"]], "CD8")
) %>%
  group_by(SampleID, clone_id) %>%
  mutate(clone_size = n(), status = if_else(clone_size >= 2L, "Expanded", "Singleton")) %>%
  ungroup()

# Restrict the main panel to well-represented states while retaining all eligible
# participants. The state list is chosen without reference to the effect estimate.
top_states <- cells %>%
  count(state, name = "receptor_cells") %>%
  arrange(desc(receptor_cells)) %>%
  slice_head(n = 12L) %>%
  pull(state)

participant_totals <- cells %>%
  count(SampleID, Diagnosis1, Batch, status, name = "status_total") %>%
  pivot_wider(names_from = status, values_from = status_total, values_fill = 0) %>%
  filter(Expanded > 0, Singleton > 0)

state_counts <- cells %>%
  filter(state %in% top_states) %>%
  count(SampleID, Diagnosis1, Batch, status, state, name = "state_cells") %>%
  complete(
    nesting(SampleID, Diagnosis1, Batch),
    status = c("Expanded", "Singleton"),
    state = top_states,
    fill = list(state_cells = 0L)
  ) %>%
  inner_join(
    participant_totals %>%
      pivot_longer(c(Expanded, Singleton), names_to = "status", values_to = "status_total"),
    by = c("SampleID", "Diagnosis1", "Batch", "status")
  ) %>%
  mutate(other_cells = status_total - state_cells) %>%
  select(SampleID, Diagnosis1, Batch, state, status, state_cells, other_cells, status_total) %>%
  pivot_wider(
    names_from = status,
    values_from = c(state_cells, other_cells, status_total),
    names_sep = "_"
  ) %>%
  mutate(
    # Haldane-Anscombe correction makes zero cells finite without discarding
    # low-abundance states. Positive values indicate enrichment in expanded clones.
    log2_odds_ratio = log2(
      ((state_cells_Expanded + 0.5) * (other_cells_Singleton + 0.5)) /
        ((other_cells_Expanded + 0.5) * (state_cells_Singleton + 0.5))
    )
  ) %>%
  # A state absent from both expanded and singleton compartments contains no
  # information about state preference and would otherwise inherit an artifact
  # from the unequal compartment totals.
  filter(state_cells_Expanded + state_cells_Singleton > 0)

bootstrap_median <- function(x, n_boot = 4000L) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) return(c(ci_low = NA_real_, ci_high = NA_real_))
  sims <- replicate(n_boot, median(sample(x, length(x), replace = TRUE)))
  setNames(as.numeric(quantile(sims, c(0.025, 0.975), na.rm = TRUE)), c("ci_low", "ci_high"))
}

summary_tbl <- state_counts %>%
  group_by(state) %>%
  summarise(
    n_participants = sum(is.finite(log2_odds_ratio)),
    median_log2_or = median(log2_odds_ratio, na.rm = TRUE),
    ci_low = bootstrap_median(log2_odds_ratio)[["ci_low"]],
    ci_high = bootstrap_median(log2_odds_ratio)[["ci_high"]],
    p_value = if (sum(is.finite(log2_odds_ratio)) >= 5L && any(log2_odds_ratio != 0, na.rm = TRUE))
      suppressWarnings(wilcox.test(log2_odds_ratio, mu = 0, exact = FALSE)$p.value) else NA_real_,
    expanded_cells = sum(state_cells_Expanded),
    singleton_cells = sum(state_cells_Singleton),
    .groups = "drop"
  ) %>%
  mutate(FDR = p.adjust(p_value, method = "BH")) %>%
  arrange(desc(median_log2_or))

write_csv(state_counts, file.path(out_dir, "Figure2_state_enrichment_by_participant.csv"))
write_csv(summary_tbl, file.path(out_dir, "Figure2_state_enrichment_summary.csv"))

message("Wrote participant-level Figure 2 state-enrichment data to: ", out_dir)

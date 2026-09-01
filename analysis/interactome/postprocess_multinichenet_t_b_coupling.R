suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(ggplot2)
  library(scales)
})

root <- normalizePath(".", winslash = "/")
private_root <- Sys.getenv("IBD_PRIVATE_WORKSPACE", unset = "C:/path/to/private-manuscript-workspace")
out_dir <- Sys.getenv(
  "IBD_MULTINICHENET_OUTPUT",
  unset = file.path(root, "analysis_outputs", "multinichenet")
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

analyses <- c("CD_inflammation", "UC_inflammation", "CD_vs_UC_shared_series", "S6_diagnosis")
b_groups <- c("B_NaiveTransitional", "B_Memory", "B_AtypicalMemory", "Plasma_IgM", "Plasma_Switched")
t_groups <- c("Tfh", "Th17", "Treg")

family_lr <- function(ligand, receptor, sender, receiver) {
  case_when(
    ligand == "CD40LG" & str_detect(receptor, "^CD40($|[._])") ~ "CD40 help",
    ligand == "IL21" & str_detect(receptor, "^IL21R") ~ "IL-21 help",
    ligand == "CXCL13" & str_detect(receptor, "CXCR5") ~ "CXCL13-CXCR5",
    ligand == "ICOSLG" & str_detect(receptor, "^ICOS($|[._])") ~ "ICOS costimulation",
    ligand %in% c("CD80", "CD86") & str_detect(receptor, "CD28|CTLA4") ~ "B-cell costimulation",
    ligand %in% c("TGFB1", "TGFB2", "TGFB3") & str_detect(receptor, "TGFBR") ~ "TGF-beta",
    ligand == "IFNG" & str_detect(receptor, "IFNGR") ~ "IFN-gamma",
    ligand == "IL4" & str_detect(receptor, "IL4R") ~ "IL-4",
    ligand == "IL10" & str_detect(receptor, "IL10R") ~ "IL-10",
    ligand %in% c("LTA", "LTB", "TNF") & str_detect(receptor, "LTBR|TNFRSF1A|TNFRSF1B") ~ "TNF/lymphotoxin",
    ligand %in% c("TNFSF13", "TNFSF13B") & str_detect(receptor, "TNFRSF13B|TNFRSF13C|TNFRSF17") ~ "APRIL/BAFF",
    ligand == "TNFSF4" & str_detect(receptor, "TNFRSF4") ~ "OX40",
    ligand %in% c("CD274", "PDCD1LG2") & str_detect(receptor, "PDCD1") ~ "PD-1 checkpoint",
    ligand == "ICAM1" & str_detect(receptor, "ITGAL|ITGB2") ~ "ICAM-1 adhesion",
    ligand == "VCAM1" & str_detect(receptor, "ITGA4|ITGB1") ~ "VCAM-1 adhesion",
    sender %in% b_groups & str_detect(ligand, "^HLA[.]D") & receptor == "CD4" ~ "MHC-II-CD4",
    TRUE ~ NA_character_
  )
}

read_cross <- function(a) {
  p <- file.path(out_dir, paste0("Table_MN_", a, "_cross_lineage_prioritization.csv"))
  read_csv(p, show_col_types = FALSE) %>%
    mutate(analysis = a,
           direction = if_else(sender %in% t_groups, "T helper -> B", "B -> T helper"),
           pathway_family = family_lr(ligand, receptor, sender, receiver)) %>%
    group_by(analysis, contrast, group, direction) %>%
    arrange(desc(prioritization_score), .by_group = TRUE) %>%
    mutate(rank_within_contrast_direction = row_number()) %>%
    ungroup()
}

cross <- map_dfr(analyses, read_cross)
write_csv(cross, file.path(out_dir, "Table_MN9_all_cross_lineage_interactions.csv"))

top_unbiased <- cross %>%
  group_by(analysis, contrast, group, direction) %>%
  slice_max(prioritization_score, n = 25, with_ties = FALSE) %>%
  ungroup()
write_csv(top_unbiased, file.path(out_dir, "Table_MN10_top25_cross_lineage_interactions.csv"))

canonical <- cross %>% filter(!is.na(pathway_family))
write_csv(canonical, file.path(out_dir, "Table_MN11_canonical_helper_interactions.csv"))

recurrence <- canonical %>%
  group_by(direction, pathway_family, ligand, receptor, sender, receiver) %>%
  summarise(n_analyses = n_distinct(analysis), n_contrasts = n_distinct(paste(analysis, contrast, group)),
            max_score = max(prioritization_score, na.rm = TRUE),
            median_score = median(prioritization_score, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(n_analyses), desc(max_score))
write_csv(recurrence, file.path(out_dir, "Table_MN12_canonical_interaction_recurrence.csv"))

selected_general <- top_unbiased %>%
  transmute(analysis, contrast, group, direction, score_type = "Top-25 cross-lineage composite", id)
selected_pathway <- canonical %>%
  group_by(analysis, contrast, group, direction, pathway_family) %>%
  slice_max(prioritization_score, n = 10, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(analysis, contrast, group, direction, score_type = pathway_family, id)
selected <- bind_rows(selected_general, selected_pathway) %>% distinct()

score_one <- function(a) {
  p <- file.path(out_dir, paste0("Table_MN_", a, "_sample_prioritization_tbl.csv.gz"))
  sel <- selected %>% filter(analysis == a) %>% rename(discovery_group = group)
  read_csv(p, show_col_types = FALSE,
           col_select = any_of(c("sample", "group", "id", "scaled_LR_prod", "scaled_LR_pb_prod",
                                 "keep_sender_receiver", "n_cells_sender", "n_cells_receiver"))) %>%
    filter(is.na(keep_sender_receiver) | keep_sender_receiver == "Sender & Receiver present") %>%
    rename(sample_group = group) %>%
    group_by(sample, sample_group, id) %>%
    summarise(across(c(scaled_LR_prod, scaled_LR_pb_prod), ~ mean(.x, na.rm = TRUE)),
              n_cells_sender = max(n_cells_sender, na.rm = TRUE),
              n_cells_receiver = max(n_cells_receiver, na.rm = TRUE), .groups = "drop") %>%
    inner_join(sel, by = "id", relationship = "many-to-many") %>%
    group_by(analysis, contrast, discovery_group, sample_group, sample, direction, score_type) %>%
    summarise(communication_score = mean(scaled_LR_pb_prod, na.rm = TRUE),
              expression_product_score = mean(scaled_LR_prod, na.rm = TRUE),
              n_interactions_scored = n_distinct(id),
              sender_cells = max(n_cells_sender, na.rm = TRUE),
              receiver_cells = max(n_cells_receiver, na.rm = TRUE), .groups = "drop") %>%
    mutate(across(c(communication_score, expression_product_score, sender_cells, receiver_cells),
                  ~ if_else(is.infinite(.x), NA_real_, .x)))
}

scores <- map_dfr(analyses, score_one)
scores <- scores %>% rename(group = discovery_group)
write_csv(scores, file.path(out_dir, "Table_MN13_participant_communication_scores.csv"))

tb <- read_csv(file.path(private_root, "High Impact Additional Analyses", "Th17 Treg B Helper Analyses",
                         "Table_TB9_T_B_analysis_dataset.csv"), show_col_types = FALSE)
bgl <- read_csv(file.path(private_root, "High Impact Additional Analyses", "BCR Germline Lineages",
                          "Table_BGL38_participant_IBD_clinical_lineage_features.csv"), show_col_types = FALSE) %>%
  select(SampleID, expanded_lineages, fraction_class_switched_lineages,
         mean_within_lineage_divergence, CDR_minus_FWR, exact_paired_HL_shm) %>%
  rename_with(~ paste0("lineage_", .x), -SampleID)
dat <- scores %>% rename(SampleID = sample) %>% left_join(tb, by = "SampleID") %>% left_join(bgl, by = "SampleID")
write_csv(dat, file.path(out_dir, "Table_MN14_communication_repertoire_integrated_dataset.csv"))

b_outcomes <- c(
  "b_mean_Plasmablast_plasma_differentiation", "b_mean_IgA_mucosal_plasma_cell",
  "b_mean_IgG_inflammatory_plasma_cell", "b_mean_Atypical_memory_CD11c_like",
  "b_mean_B_cell_antigen_presentation", "bcr_switched_fraction", "bcr_SHM_rate",
  "bcr_IgA_fraction", "bcr_IgG_fraction", "lineage_expanded_lineages",
  "lineage_fraction_class_switched_lineages", "lineage_mean_within_lineage_divergence",
  "lineage_CDR_minus_FWR", "lineage_exact_paired_HL_shm")
t_outcomes <- c(
  "mean_Tph_Tfh_help_all_CD4", "mean_Th17_conventional_in_Th17",
  "mean_Th17_pathogenic_in_Th17", "mean_Treg_suppressive_in_Treg",
  "mean_Treg_reprogramming_in_Treg", "expanded_axis_cell_fraction",
  "expanded_th17_cell_fraction", "expanded_treg_cell_fraction", "mixed_axis_cell_fraction")

partial_rank <- function(d, outcome) {
  cov_names <- switch(unique(d$analysis),
    CD_inflammation = c("Inflammation1", "acquisition_series", "log_cd4_cells", "log_b_cells"),
    UC_inflammation = c("Inflammation1", "acquisition_series", "log_cd4_cells", "log_b_cells"),
    CD_vs_UC_shared_series = c("Diagnosis1", "acquisition_series", "log_cd4_cells", "log_b_cells"),
    S6_diagnosis = c("Diagnosis1", "log_cd4_cells", "log_b_cells"))
  vars <- unique(c("communication_score", outcome, cov_names))
  z <- d %>% select(any_of(vars)) %>% filter(if_all(everything(), ~ !is.na(.x)))
  if (nrow(z) < 15 || n_distinct(z$communication_score) < 3 || n_distinct(z[[outcome]]) < 3)
    return(tibble(n = nrow(z), partial_spearman_rho = NA_real_, p_value = NA_real_, ci_low = NA_real_, ci_high = NA_real_, covariate_df = NA_integer_))
  cov_names <- cov_names[vapply(z[cov_names], function(x) n_distinct(x) > 1, logical(1))]
  cov_formula <- reformulate(cov_names)
  X <- model.matrix(cov_formula, data = z)
  rx <- lm.fit(X, rank(z$communication_score, ties.method = "average"))$residuals
  ry <- lm.fit(X, rank(z[[outcome]], ties.method = "average"))$residuals
  r <- suppressWarnings(cor(rx, ry))
  k <- qr(X)$rank - 1L
  df <- nrow(z) - k - 2L
  tstat <- r * sqrt(df / max(1e-12, 1 - r^2))
  p <- 2 * pt(abs(tstat), df = df, lower.tail = FALSE)
  se <- 1 / sqrt(max(1, nrow(z) - k - 3))
  ci <- tanh(atanh(max(-.999999, min(.999999, r))) + c(-1, 1) * 1.96 * se)
  tibble(n = nrow(z), partial_spearman_rho = r, p_value = p, ci_low = ci[1], ci_high = ci[2], covariate_df = k)
}

assoc_dat <- dat %>%
  filter(!(analysis == "S6_diagnosis" & contrast == "CD-Control") | Diagnosis1 %in% c("CD", "Control"),
         !(analysis == "S6_diagnosis" & contrast == "UC-Control") | Diagnosis1 %in% c("UC", "Control"),
         !(analysis == "S6_diagnosis" & contrast == "CD-UC") | Diagnosis1 %in% c("CD", "UC"))

assoc <- assoc_dat %>%
  group_by(analysis, contrast, group, direction, score_type) %>%
  group_modify(function(df, key) {
    outs <- if (unique(key$direction) == "T helper -> B") b_outcomes else t_outcomes
    map_dfr(outs, function(o) partial_rank(df, o) %>% mutate(outcome = o))
  }, .keep = TRUE) %>%
  ungroup() %>%
  group_by(analysis, direction) %>%
  mutate(fdr = p.adjust(p_value, method = "BH")) %>%
  ungroup() %>%
  arrange(fdr, p_value)
write_csv(assoc, file.path(out_dir, "Table_MN15_partial_spearman_communication_repertoire_associations.csv"))

# Ligand-supported targets for the strongest canonical interactions.
headline <- canonical %>%
  group_by(analysis, contrast, group, direction, pathway_family) %>%
  slice_max(prioritization_score, n = 3, with_ties = FALSE) %>%
  ungroup() %>% select(analysis, contrast, group, direction, pathway_family, sender, receiver, ligand, receptor, prioritization_score)
targets <- map_dfr(analyses, function(a) {
  read_csv(file.path(out_dir, paste0("Table_MN_", a, "_ligand_activities_target_de_tbl.csv.gz")), show_col_types = FALSE) %>%
    mutate(analysis = a)
})
target_support <- headline %>%
  inner_join(targets, by = c("analysis", "contrast", "receiver", "ligand"), relationship = "many-to-many") %>%
  filter(direction_regulation == "up", p_val < 0.05) %>%
  group_by(analysis, contrast, group, direction, pathway_family, sender, receiver, ligand, receptor) %>%
  slice_max(abs(logFC) * ligand_target_weight, n = 10, with_ties = FALSE) %>%
  ungroup()
write_csv(target_support, file.path(out_dir, "Table_MN16_canonical_ligand_supported_targets.csv"))

# Figure 1: canonical interactions across endpoints.
plot_dat <- canonical %>%
  mutate(endpoint = paste(analysis, contrast, group, sep = " | "),
         interaction = paste(sender, paste0(ligand, "-", receptor), receiver, sep = " : ")) %>%
  group_by(direction, interaction) %>%
  mutate(recurrence_n = n_distinct(analysis)) %>% ungroup() %>%
  group_by(direction) %>%
  filter(dense_rank(desc(recurrence_n * 2 + prioritization_score)) <= 18) %>%
  ungroup()
p1 <- ggplot(plot_dat, aes(endpoint, reorder(interaction, prioritization_score), color = prioritization_score, size = fraction_expressing_ligand_receptor)) +
  geom_point(alpha = .9) + facet_wrap(~ direction, scales = "free_y", ncol = 1) +
  scale_color_viridis_c(option = "C", limits = c(0, 1)) + scale_size(range = c(1.5, 6)) +
  labs(x = NULL, y = NULL, color = "Priority score", size = "Expressing fraction",
       title = "Canonical T-helper-B-cell communication programs") +
  theme_bw(base_size = 9) + theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right")
ggsave(file.path(out_dir, "Figure_MN1_canonical_interactions.png"), p1, width = 13, height = 11, dpi = 400)
ggsave(file.path(out_dir, "Figure_MN1_canonical_interactions.pdf"), p1, width = 13, height = 11)

# Figure 2: strongest adjusted associations linking inferred communication to repertoire state.
pa <- assoc %>% filter(!is.na(partial_spearman_rho), n >= 20) %>%
  group_by(direction) %>% slice_min(p_value, n = 18, with_ties = FALSE) %>% ungroup() %>%
  mutate(label = paste(score_type, outcome, sep = " -> "))
p2 <- ggplot(pa, aes(partial_spearman_rho, reorder(label, partial_spearman_rho), color = fdr < .1)) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = .4) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), width = .2, orientation = "y") + geom_point(size = 2.5) +
  facet_wrap(~ direction, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c(`TRUE` = "#B2182B", `FALSE` = "#2166AC"), labels = c(`TRUE` = "FDR < 0.10", `FALSE` = "FDR >= 0.10")) +
  labs(x = "Adjusted rank association (rho)", y = NULL, color = NULL,
       title = "Participant-level communication-repertoire coupling") +
  theme_bw(base_size = 9) + theme(legend.position = "top")
ggsave(file.path(out_dir, "Figure_MN2_communication_repertoire_associations.png"), p2, width = 12, height = 10, dpi = 400)
ggsave(file.path(out_dir, "Figure_MN2_communication_repertoire_associations.pdf"), p2, width = 12, height = 10)

checks <- tibble(
  item = c("Cross-lineage rows", "Canonical rows", "Participant score rows", "Unique scored participants", "Association tests", "Target-support rows"),
  value = c(nrow(cross), nrow(canonical), nrow(scores), n_distinct(scores$sample), nrow(assoc), nrow(target_support)))
write_csv(checks, file.path(out_dir, "Table_MN17_postprocessing_checks.csv"))

write_lines(c(
  paste("Postprocessing completed:", Sys.time()),
  paste("Cross-lineage interactions:", nrow(cross)),
  paste("Canonical interactions:", nrow(canonical)),
  paste("Unique participants scored:", n_distinct(scores$sample)),
  paste("Association tests:", nrow(assoc)),
  paste("Associations at FDR < 0.10:", sum(assoc$fdr < .10, na.rm = TRUE)),
  paste("Canonical ligand-supported targets:", nrow(target_support))
), file.path(out_dir, "MN_postprocessing_log.txt"))

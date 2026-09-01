suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
})

root <- normalizePath(".", winslash = "/")
private_root <- Sys.getenv("IBD_PRIVATE_WORKSPACE", unset = "C:/path/to/private-manuscript-workspace")
out_dir <- Sys.getenv(
  "IBD_MULTINICHENET_OUTPUT",
  unset = file.path(root, "analysis_outputs", "multinichenet")
)
canonical <- read_csv(file.path(out_dir, "Table_MN11_canonical_helper_interactions.csv"), show_col_types = FALSE)

b_groups <- c("B_NaiveTransitional", "B_Memory", "B_AtypicalMemory", "Plasma_IgM", "Plasma_Switched")
t_groups <- c("Tfh", "Th17", "Treg")

gene_evidence <- bind_rows(
  canonical %>% transmute(compartment = if_else(sender %in% t_groups, "CD4", "B"), role = "ligand", gene = ligand,
                          pathway_family, prioritization_score, analysis),
  canonical %>% transmute(compartment = if_else(receiver %in% t_groups, "CD4", "B"), role = "receptor", gene = receptor,
                          pathway_family, prioritization_score, analysis)
) %>%
  group_by(compartment, role, gene, pathway_family) %>%
  summarise(max_priority = max(prioritization_score, na.rm = TRUE), n_analyses = n_distinct(analysis), .groups = "drop") %>%
  group_by(compartment, role) %>%
  arrange(desc(n_analyses), desc(max_priority), .by_group = TRUE) %>%
  slice_head(n = 30) %>% ungroup()
write_csv(gene_evidence, file.path(out_dir, "Table_MN18_clone_validation_gene_panel.csv"))

map_cd4 <- function(x) case_when(
  x == "CD4 Tfh" ~ "Tfh",
  x %in% c("CD4 Th17", "CD4 Th1/Th17") ~ "Th17",
  grepl("^TReg", x) ~ "Treg",
  TRUE ~ NA_character_)
map_b <- function(x) case_when(
  x %in% c("Naive B", "Naive-IFN B", "Transitional B", "CD5+ B Cell") ~ "B_NaiveTransitional",
  x %in% c("Switched memory B", "Non-switched memory B") ~ "B_Memory",
  x == "Atypical memory B" ~ "B_AtypicalMemory",
  x == "IgM Plasma B Cell" ~ "Plasma_IgM",
  x %in% c("IgA Plasma B Cell", "IgG Plasma B Cell") ~ "Plasma_Switched",
  TRUE ~ NA_character_)

paths <- c(
  CD4 = file.path(private_root, "Figure 2 TCR", "CD42026.rds"),
  B = file.path(private_root, "Figure 1", "Figure 1", "BCell2026.rds"))

process_compartment <- function(compartment) {
  message("Clone-aware validation: ", compartment)
  obj <- readRDS(paths[[compartment]])
  md <- as.data.frame(obj[[]], stringsAsFactors = FALSE)
  md$cell <- rownames(md)
  md$celltype <- if (compartment == "CD4") map_cd4(as.character(md$AnnotationLevel2)) else map_b(as.character(md$AnnotationLevel2))
  cdr <- if (compartment == "CD4") "TCR_Beta_Delta_CDR3_Nucleotide_Dominant" else "BCR_Heavy_CDR3_Nucleotide_Dominant"
  vg <- if (compartment == "CD4") "TCR_Beta_Delta_V_gene_Dominant" else "BCR_Heavy_V_gene_Dominant"
  jg <- if (compartment == "CD4") "TCR_Beta_Delta_J_gene_Dominant" else "BCR_Heavy_J_gene_Dominant"
  valid_chain <- !is.na(md[[cdr]]) & nzchar(trimws(md[[cdr]])) & !md[[cdr]] %in% c("None", "NA", "nan")
  md$clone_key <- ifelse(valid_chain, paste(md$SampleID, md[[vg]], md[[jg]], md[[cdr]], sep = "|"), NA_character_)
  md <- md %>% filter(!is.na(celltype), !is.na(SampleID), !is.na(clone_key)) %>%
    add_count(SampleID, clone_key, name = "clone_size") %>%
    mutate(clone_status = if_else(clone_size >= 2, "Expanded", "Singleton"),
           acquisition_series = sub("[AB]$", "", as.character(Batch)))
  genes <- intersect(unique(gene_evidence$gene[gene_evidence$compartment == compartment]), rownames(obj[["RNA"]]))
  mat <- LayerData(obj[["RNA"]], layer = "data")[genes, md$cell, drop = FALSE]
  summaries <- map_dfr(seq_along(genes), function(i) {
    v <- as.numeric(mat[i, ])
    tibble(SampleID = md$SampleID, Diagnosis1 = md$Diagnosis1, Inflammation1 = md$Inflammation1,
           acquisition_series = md$acquisition_series, celltype = md$celltype,
           clone_status = md$clone_status, gene = genes[i], expression = v) %>%
      group_by(SampleID, Diagnosis1, Inflammation1, acquisition_series, celltype, clone_status, gene) %>%
      summarise(n_cells = n(), mean_expression = mean(expression), pct_expression = mean(expression > 0), .groups = "drop")
  }) %>% mutate(compartment = compartment)
  rm(obj, mat); gc()
  summaries
}

cell_summary <- bind_rows(process_compartment("CD4"), process_compartment("B"))
write_csv(cell_summary, file.path(out_dir, "Table_MN19_clone_status_expression_by_participant_state.csv"))

paired <- cell_summary %>%
  filter(n_cells >= 5) %>%
  select(compartment, SampleID, Diagnosis1, Inflammation1, acquisition_series, celltype, gene,
         clone_status, n_cells, mean_expression, pct_expression) %>%
  pivot_wider(names_from = clone_status, values_from = c(n_cells, mean_expression, pct_expression)) %>%
  filter(!is.na(mean_expression_Expanded), !is.na(mean_expression_Singleton)) %>%
  mutate(delta_mean_expression = mean_expression_Expanded - mean_expression_Singleton,
         delta_pct_expression = pct_expression_Expanded - pct_expression_Singleton)
write_csv(paired, file.path(out_dir, "Table_MN20_within_participant_expanded_singleton_deltas.csv"))

test_one <- function(d) {
  if (nrow(d) < 8 || all(d$delta_mean_expression == 0))
    return(tibble(n_participants = nrow(d), median_delta_expression = median(d$delta_mean_expression),
                  median_delta_pct = median(d$delta_pct_expression), p_value = NA_real_))
  w <- suppressWarnings(wilcox.test(d$delta_mean_expression, mu = 0, paired = FALSE, exact = FALSE))
  tibble(n_participants = nrow(d), median_delta_expression = median(d$delta_mean_expression),
         median_delta_pct = median(d$delta_pct_expression), p_value = w$p.value)
}

base_tests <- bind_rows(
  paired %>% mutate(stratum = "All") %>% group_by(compartment, celltype, gene, stratum) %>% group_modify(~ test_one(.x)),
  paired %>% filter(Diagnosis1 %in% c("CD", "UC")) %>% mutate(stratum = Diagnosis1) %>%
    group_by(compartment, celltype, gene, stratum) %>% group_modify(~ test_one(.x))
) %>% ungroup() %>%
  group_by(compartment, stratum) %>% mutate(fdr = p.adjust(p_value, method = "BH")) %>% ungroup() %>%
  arrange(fdr, p_value)
tests <- base_tests %>%
  left_join(gene_evidence, by = c("compartment", "gene"), relationship = "many-to-many")
write_csv(tests, file.path(out_dir, "Table_MN21_clone_aware_ligand_receptor_expression_tests.csv"))

pd <- tests %>% filter(stratum == "All", !is.na(p_value), n_participants >= 10) %>%
  distinct(compartment, celltype, gene, median_delta_expression, n_participants, p_value, fdr) %>%
  group_by(compartment) %>% slice_min(p_value, n = 25, with_ties = FALSE) %>% ungroup() %>%
  mutate(label = paste(celltype, gene, sep = " : "))
p <- ggplot(pd, aes(median_delta_expression, reorder(label, median_delta_expression), color = fdr < .1)) +
  geom_vline(xintercept = 0, color = "grey70") + geom_point(size = 2.5) +
  facet_wrap(~ compartment, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c(`TRUE` = "#B2182B", `FALSE` = "#2166AC")) +
  labs(x = "Median within-participant expression difference (expanded - singleton)", y = NULL,
       color = "FDR < 0.10", title = "Clone-aware validation of prioritized communication genes") +
  theme_bw(base_size = 9) + theme(legend.position = "top")
ggsave(file.path(out_dir, "Figure_MN3_clone_aware_expression.png"), p, width = 10, height = 10, dpi = 400)
ggsave(file.path(out_dir, "Figure_MN3_clone_aware_expression.pdf"), p, width = 10, height = 10)

write_lines(c(
  paste("Clone-aware validation completed:", Sys.time()),
  paste("Genes tested:", n_distinct(tests$gene)),
  paste("Within-participant comparisons:", nrow(paired)),
  paste("Unique tests at FDR < 0.10:", sum(base_tests$fdr < .1, na.rm = TRUE))
), file.path(out_dir, "MN_clone_validation_log.txt"))

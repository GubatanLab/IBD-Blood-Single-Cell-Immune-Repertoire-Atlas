#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(limma)
  library(dplyr)
  library(readr)
  library(ggplot2)
})

outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses/Priority Analyses"
de <- read_csv(file.path(outdir, "Table_PA1_clone_aware_pseudobulk_DE_all_genes.csv"), show_col_types = FALSE)

extract_assignment <- function(path, target) {
  ex <- parse(path)
  for (e in ex) {
    if (is.call(e) && as.character(e[[1]])[1] %in% c("<-", "=") && identical(e[[2]], as.name(target))) {
      return(eval(e[[3]], envir = baseenv()))
    }
  }
  stop("Could not find ", target, " in ", path)
}

tcr_sets <- extract_assignment(
  "C:/path/to/private-user-home/OneDrive/Desktop/TCR Module Scores/compare_tcr_clonotype_module_scores.R",
  "module_genes"
)
bcr_sets <- extract_assignment(
  "C:/path/to/private-user-home/OneDrive/Desktop/BCR Module Scores/run_bcr_module_expansion_stars.R",
  "modules"
)

run_one <- function(modality, sets) {
  d <- de %>% filter(.data$modality == modality, coefficient == "statusExpanded")
  stat <- d$t; names(stat) <- d$gene
  index <- lapply(sets, function(g) intersect(g, names(stat)))
  index <- index[lengths(index) >= 3]
  ans <- cameraPR(stat, index = index, use.ranks = TRUE, inter.gene.cor = 0.01, sort = FALSE)
  ans$module <- rownames(ans)
  ans$modality <- modality
  ans$n_genes_tested <- lengths(index)[ans$module]
  as_tibble(ans)
}

res <- bind_rows(run_one("TCR", tcr_sets), run_one("BCR", bcr_sets)) %>%
  group_by(modality) %>% mutate(FDR_within_modality = p.adjust(PValue, "BH")) %>% ungroup() %>%
  mutate(FDR_global = p.adjust(PValue, "BH"),
         signed_log10_FDR = ifelse(Direction == "Up", 1, -1) * -log10(pmax(FDR_global, 1e-300)))
write_csv(res, file.path(outdir, "Table_PA1_clone_aware_pseudobulk_module_enrichment.csv"))

p <- res %>% group_by(modality) %>% arrange(FDR_global, .by_group = TRUE) %>% slice_head(n = 12) %>% ungroup() %>%
  mutate(module_label = gsub("_", " ", module),
         module_label = factor(module_label, levels = rev(unique(module_label)))) %>%
  ggplot(aes(signed_log10_FDR, module_label, color = Direction)) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = .35) + geom_point(size = 2.2) +
  facet_wrap(~modality, scales = "free_y") + scale_color_manual(values = c(Up = "#C2415D", Down = "#2B6CB0")) +
  labs(x = "Signed -log10 FDR (expanded versus singleton)", y = NULL,
       title = "Predefined transcriptional programs associated with receptor expansion", color = NULL) +
  theme_classic(base_size = 10) + theme(strip.background = element_blank(), strip.text = element_text(face = "bold"), legend.position = "bottom")
ggsave(file.path(outdir, "Figure_PA1b_clone_aware_module_enrichment.png"), p, width = 9.0, height = 6.5, dpi = 600)
ggsave(file.path(outdir, "Figure_PA1b_clone_aware_module_enrichment.pdf"), p, width = 9.0, height = 6.5)
message("Module enrichment complete")

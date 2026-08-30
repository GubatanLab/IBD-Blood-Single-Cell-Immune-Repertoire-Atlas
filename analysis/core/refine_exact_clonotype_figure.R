suppressPackageStartupMessages({library(dplyr); library(readr); library(ggplot2)})
outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses"
tests <- read_csv(file.path(outdir, "Table_HI_exact_clonotype_definition_pairwise_tests.csv"), show_col_types = FALSE)
plot_data <- tests %>%
  filter(metric %in% c("clonality", "expanded_cell_fraction")) %>%
  mutate(
    metric_label = recode(metric, clonality = "Clonality", expanded_cell_fraction = "Expanded-cell fraction"),
    setting = ifelse(analysis == "Original depth", paste0("Threshold >=", threshold), paste0("Rarefied n=", downsample_depth)),
    receptor_definition = factor(receptor_definition, levels = c("TCR beta-only", "TCR paired alpha-beta", "BCR heavy-only", "BCR paired heavy-light")),
    contrast = factor(contrast, levels = c("CD vs Control", "UC vs Control", "CD vs UC")),
    panel = paste(metric_label, receptor_definition, sep = "___")
  )

p <- ggplot(plot_data, aes(rank_biserial, setting, xmin = ci_low, xmax = ci_high, color = contrast)) +
  geom_vline(xintercept = 0, linetype = 2, color = "grey60") +
  geom_errorbar(orientation = "y", width = .18, position = position_dodge(width = .55)) +
  geom_point(position = position_dodge(width = .55), size = 1.8) +
  facet_wrap(vars(metric_label, receptor_definition), ncol = 4, scales = "free_y") +
  scale_color_manual(values = c("CD vs Control" = "#2B6CB0", "UC vs Control" = "#C2415D", "CD vs UC" = "#555555")) +
  labs(
    title = "Clonal expansion is evaluated across exact receptor definitions and sampling depth",
    subtitle = "Clonotypes require identical V and J calls plus CDR3 amino-acid sequence; bars are bootstrap 95% confidence intervals",
    x = "Rank-biserial effect (first group higher ->)", y = NULL, color = NULL
  ) +
  theme_classic(base_size = 9) +
  theme(panel.border = element_rect(color = "black", fill = NA, linewidth = .35), strip.background = element_blank(),
        strip.text = element_text(face = "bold", size = 8), legend.position = "bottom", axis.text = element_text(color = "black"))
ggsave(file.path(outdir, "Figure_HI3_exact_clonotype_robustness.pdf"), p, width = 12.5, height = 7.2, device = cairo_pdf)
ggsave(file.path(outdir, "Figure_HI3_exact_clonotype_robustness.png"), p, width = 12.5, height = 7.2, dpi = 400)

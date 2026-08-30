#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(glmmTMB)
  library(emmeans)
  library(dplyr)
  library(readr)
})

set.seed(20260828)

root <- "C:/path/to/private-manuscript-workspace"
switch_path <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 3 BCR/BCR Architecture Analyses/BCR isotype switching diagnosis comparisons/tables/bcr_isotype_switched_fraction_by_sample.csv"
metadata_path <- file.path(root, "High Impact Additional Analyses", "Table_HI_integrated_participant_features.csv")
outdir <- file.path(root, "Trajectory Integrated Figure Set", "Source Data")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

switched <- read_csv(switch_path, show_col_types = FALSE) %>%
  rename(Diagnosis = Diagnosis1) %>%
  mutate(
    Diagnosis = factor(Diagnosis, levels = c("Control", "CD", "UC")),
    unswitched_cells = total_isotyped - switched_cells
  )
metadata <- read_csv(metadata_path, show_col_types = FALSE) %>%
  select(SampleID, Age, Sex, bcr_total_cells)
frame <- switched %>%
  inner_join(metadata, by = "SampleID") %>%
  mutate(
    Age_z = as.numeric(scale(Age)),
    log_bcr_depth_z = as.numeric(scale(log1p(bcr_total_cells))),
    Sex = factor(Sex)
  )

fit <- glmmTMB(
  cbind(switched_cells, unswitched_cells) ~ Diagnosis + Age_z + Sex + log_bcr_depth_z,
  data = frame,
  family = betabinomial(link = "logit")
)

emm <- emmeans(fit, ~ Diagnosis)
contr <- contrast(emm, method = "trt.vs.ctrl", ref = 1, adjust = "none")
contr_df <- as.data.frame(summary(contr, infer = c(TRUE, TRUE))) %>%
  mutate(
    diagnosis = c("CD", "UC"),
    comparison = paste(diagnosis, "vs Control"),
    adjusted_odds_ratio = exp(estimate),
    OR_ci95_low = exp(asymp.LCL),
    OR_ci95_high = exp(asymp.UCL),
    p_value = p.value,
    Holm_q = p.adjust(p_value, method = "holm")
  ) %>%
  select(comparison, diagnosis, adjusted_log_odds = estimate, adjusted_odds_ratio,
         OR_ci95_low, OR_ci95_high, p_value, Holm_q)

control <- frame %>% filter(Diagnosis == "Control") %>% pull(switched_fraction)
rank_rows <- lapply(c("CD", "UC"), function(dx) {
  values <- frame %>% filter(Diagnosis == dx) %>% pull(switched_fraction)
  boot <- replicate(
    10000,
    median(sample(values, length(values), replace = TRUE)) -
      median(sample(control, length(control), replace = TRUE))
  )
  tibble(
    diagnosis = dx,
    comparison = paste(dx, "vs Control"),
    median_difference = median(values) - median(control),
    median_difference_ci95_low = quantile(boot, 0.025),
    median_difference_ci95_high = quantile(boot, 0.975),
    rank_p_value = wilcox.test(values, control, alternative = "two.sided", exact = FALSE)$p.value
  )
}) %>% bind_rows() %>% mutate(rank_Holm_q = p.adjust(rank_p_value, method = "holm"))

result <- rank_rows %>% left_join(contr_df, by = c("diagnosis", "comparison"))
write_csv(result, file.path(outdir, "Figure4G_class_switching_rank_and_betabinomial.csv"))

group_summary <- frame %>%
  group_by(Diagnosis) %>%
  summarise(
    n_participants = n(),
    median_switched_fraction = median(switched_fraction),
    mean_switched_fraction = mean(switched_fraction),
    .groups = "drop"
  )
write_csv(group_summary, file.path(outdir, "Figure4G_class_switching_group_summary.csv"))
saveRDS(fit, file.path(outdir, "Figure4G_betabinomial_fit.rds"))

print(result)
print(summary(fit))

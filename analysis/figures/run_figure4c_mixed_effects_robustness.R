suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lmerTest)
  library(sandwich)
})

set.seed(20260828)
root <- normalizePath(".", mustWork=TRUE)
input <- file.path(root, "High Impact Additional Analyses", "Clone State Interactions", "Table_CSI1_state_matched_module_scores.csv")
output <- file.path(root, "Trajectory Integrated Figure Set", "Source Data", "Figure4C_mixed_effects_cluster_bootstrap.csv")

selected <- c(
  "Plasmablast_plasma_differentiation",
  "IgA_mucosal_plasma_cell",
  "IgG_inflammatory_plasma_cell",
  "BAFF_APRIL_survival_response",
  "B_cell_antigen_presentation"
)

dat <- read_csv(input, show_col_types=FALSE) %>%
  filter(compartment == "BCR", module %in% selected) %>%
  mutate(
    Diagnosis=factor(Diagnosis1, levels=c("Control", "CD", "UC")),
    Biologic=factor(ifelse(is.na(Biologic) | Biologic == "", "Unknown", Biologic)),
    Series=factor(acquisition_series),
    state=factor(state)
  )

n_boot <- 2000L

fit_module <- function(module_name) {
  d <- droplevels(filter(dat, module == module_name))
  mixed <- lmer(delta ~ Diagnosis + state + Biologic + Series + (1 | SampleID), data=d, REML=FALSE)
  mixed_coefs <- as.data.frame(coef(summary(mixed)))
  mixed_coefs$term <- rownames(mixed_coefs)
  primary <- lm(delta ~ Diagnosis + state + Biologic + Series, data=d)
  primary_vcov <- vcovCL(primary, cluster=d$SampleID, type="HC3", cadjust=TRUE)
  primary_coef <- coef(primary)

  participant_groups <- split(unique(d[c("SampleID", "Diagnosis1")]), unique(d[c("SampleID", "Diagnosis1")])$Diagnosis1)
  boot <- matrix(NA_real_, nrow=n_boot, ncol=2, dimnames=list(NULL, c("DiagnosisCD", "DiagnosisUC")))
  for (b in seq_len(n_boot)) {
    sampled <- bind_rows(lapply(names(participant_groups), function(dx) {
      ids <- participant_groups[[dx]]$SampleID
      tibble(SampleID=sample(ids, length(ids), replace=TRUE), copy=seq_along(ids), diagnosis=dx)
    }))
    bd <- bind_rows(lapply(seq_len(nrow(sampled)), function(i) {
      z <- d[d$SampleID == sampled$SampleID[i], , drop=FALSE]
      z$BootstrapID <- paste0(sampled$diagnosis[i], "_", sampled$copy[i])
      z
    })) %>% droplevels()
    bf <- tryCatch(lm(delta ~ Diagnosis + state + Biologic + Series, data=bd), error=function(e) NULL)
    if (!is.null(bf)) {
      bb <- coef(bf)
      for (term in colnames(boot)) if (term %in% names(bb)) boot[b, term] <- bb[[term]]
    }
  }

  bind_rows(lapply(c("DiagnosisCD", "DiagnosisUC"), function(term) {
    label <- ifelse(term == "DiagnosisCD", "CD_vs_Control", "UC_vs_Control")
    vals <- boot[, term]
    vals <- vals[is.finite(vals)]
    mixed_row <- mixed_coefs[mixed_coefs$term == term, , drop=FALSE]
    estimate <- unname(primary_coef[[term]])
    robust_se <- sqrt(primary_vcov[term, term])
    robust_df <- n_distinct(d$SampleID) - 1
    tibble(
      module=module_name,
      contrast=label,
      effect=estimate,
      cluster_HC3_SE=robust_se,
      cluster_HC3_df=robust_df,
      cluster_HC3_ci95_low=estimate - qt(.975, robust_df) * robust_se,
      cluster_HC3_ci95_high=estimate + qt(.975, robust_df) * robust_se,
      cluster_HC3_p=2 * pt(-abs(estimate / robust_se), df=robust_df),
      bootstrap_replicates=length(vals),
      bootstrap_ci95_low=unname(quantile(vals, .025)),
      bootstrap_ci95_high=unname(quantile(vals, .975)),
      bootstrap_p=2 * min(mean(vals <= 0), mean(vals >= 0)),
      mixed_effect=mixed_row$Estimate,
      mixed_SE=mixed_row$`Std. Error`,
      mixed_p=mixed_row$`Pr(>|t|)`,
      n_participants=n_distinct(d$SampleID),
      n_state_pairs=nrow(d),
      singular_fit=lme4::isSingular(mixed)
    )
  }))
}

results <- bind_rows(lapply(selected, fit_module)) %>%
  mutate(
    cluster_HC3_FDR=p.adjust(cluster_HC3_p, method="BH"),
    bootstrap_FDR=p.adjust(bootstrap_p, method="BH")
  )

write_csv(results, output)
message("Wrote: ", output)

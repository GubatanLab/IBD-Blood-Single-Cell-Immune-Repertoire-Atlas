suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(limma)
})

root <- normalizePath(".", mustWork=TRUE)
input <- file.path(root, "High Impact Additional Analyses", "Clone State Interactions", "Table_CSI2_participant_expanded_minus_singleton_deltas.csv")
output <- file.path(root, "Trajectory Integrated Figure Set", "Source Data", "Figure4C_diagnosis_stratified_program_effects.csv")

selected <- c(
  "Plasmablast_plasma_differentiation",
  "IgA_mucosal_plasma_cell",
  "IgG_inflammatory_plasma_cell",
  "BAFF_APRIL_survival_response",
  "B_cell_antigen_presentation"
)

d <- read_csv(input, show_col_types=FALSE) %>%
  filter(modality == "BCR", module %in% selected) %>%
  mutate(
    Diagnosis=factor(Diagnosis1, levels=c("Control", "CD", "UC")),
    Biologic=factor(ifelse(is.na(Biologic) | Biologic == "", "Unknown", Biologic)),
    Series=factor(acquisition_series)
  )

wide <- d %>% select(SampleID, module, delta) %>% pivot_wider(names_from=SampleID, values_from=delta)
mods <- wide$module
mat <- as.matrix(wide[, -1])
rownames(mat) <- mods
meta <- d %>% distinct(SampleID, .keep_all=TRUE) %>% arrange(match(SampleID, colnames(mat)))
mat <- mat[, meta$SampleID, drop=FALSE]

design <- model.matrix(~0 + Diagnosis + Biologic + Series, data=meta)
q <- qr(design)
if (q$rank < ncol(design)) design <- design[, sort(q$pivot[seq_len(q$rank)]), drop=FALSE]
fit <- eBayes(lmFit(mat, design), robust=TRUE)

defs <- c(
  Control="DiagnosisControl",
  CD="DiagnosisCD",
  UC="DiagnosisUC",
  CD_vs_Control="DiagnosisCD-DiagnosisControl",
  UC_vs_Control="DiagnosisUC-DiagnosisControl"
)

results <- bind_rows(lapply(names(defs), function(label) {
  cc <- makeContrasts(contrasts=defs[[label]], levels=design)
  ff <- eBayes(contrasts.fit(fit, cc), robust=TRUE)
  tt <- topTable(ff, number=Inf, sort.by="none")
  tibble(
    contrast=label,
    module=rownames(tt),
    effect=tt$logFC,
    SE=abs(tt$logFC / tt$t),
    ci95_low=tt$logFC - 1.96 * abs(tt$logFC / tt$t),
    ci95_high=tt$logFC + 1.96 * abs(tt$logFC / tt$t),
    p_value=tt$P.Value
  )
})) %>%
  group_by(contrast) %>% mutate(FDR=p.adjust(p_value, method="BH")) %>% ungroup()

dir.create(dirname(output), recursive=TRUE, showWarnings=FALSE)
write_csv(results, output)
message("Wrote: ", output)

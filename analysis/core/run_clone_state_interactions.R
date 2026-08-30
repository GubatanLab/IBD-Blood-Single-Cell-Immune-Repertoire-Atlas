#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(limma)
})

set.seed(20260824)
outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses/Clone State Interactions"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

paths <- c(
  CD4 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
  CD8 = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds",
  BCR = "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026_scvi50_umap.rds"
)
stopifnot(all(file.exists(paths)))

t_modules <- list(
  Naive_central_memory=c("CCR7","SELL","TCF7","LEF1","IL7R","LTB","MAL","NOSIP","SATB1","BACH2","KLF2","S1PR1","CD27","CD28","BCL2"),
  Recent_TCR_stimulation=c("NR4A1","NR4A2","NR4A3","EGR1","EGR2","EGR3","FOS","JUN","JUNB","DUSP1","DUSP2","DUSP4","DUSP5","CD69","IL2RA","NFKBIA","NFKBIZ","REL","IRF4","BATF"),
  Effector_cytotoxicity=c("NKG7","GNLY","PRF1","GZMB","GZMA","GZMH","GZMK","GZMM","CTSW","CST7","FGFBP2","CCL5","CCL4","CCL3","IFNG","FASLG","KLRD1","KLRG1"),
  GZMK_inflammatory_memory=c("GZMK","GZMA","CCL5","CCL4","CCL4L2","XCL1","XCL2","NKG7","DUSP2","CRTAM","EOMES","CXCR3","IL7R"),
  Th1_Tc1_inflammatory=c("TBX21","STAT4","CXCR3","IFNG","TNF","IL12RB2","CCL5","CCL4","NKG7","GZMB","PRF1","CXCR6","BHLHE40"),
  Chronic_stimulation_exhaustion_like=c("PDCD1","LAG3","HAVCR2","TIGIT","CTLA4","TOX","TOX2","ENTPD1","SLAMF6","TNFRSF9","BATF","EOMES","PRDM1","LAYN","CXCL13"),
  Tissue_resident_mucosal_retention=c("CD69","ITGAE","ITGA1","CXCR6","ZNF683","RUNX3","PRDM1","RGS1","CD101","DUSP6","AHR","CCR6"),
  EOMES_ZEB2_inflammatory_CD8_TRM_like=c("EOMES","ZEB2","GZMB","GZMH","PRF1","NKG7","CX3CR1","KLRG1","TBX21","CCL5","CST7","FGFBP2"),
  Gut_homing_intestinal_trafficking=c("ITGA4","ITGB7","ITGAE","CCR9","CCR6","CXCR3","CXCR6","SELPLG","S1PR1","KLF2","SELL","GPR183","CD69"),
  Tph_Tfh_like_B_cell_help=c("CXCL13","PDCD1","ICOS","MAF","TOX2","IL21","CD40LG","SLAMF6","TIGIT","CD200","CXCR5","BCL6","SH2D1A"),
  MAIT_like=c("SLC4A10","KLRB1","ZBTB16","RORC","IL18RAP","CCR6","DPP4","CXCR6","IL7R","GZMK","IFNG","NKG7"),
  Cell_cycle_clonal_proliferation=c("MKI67","TOP2A","STMN1","TYMS","PCNA","MCM2","MCM3","MCM4","MCM5","MCM6","MCM7","HMGB2","CENPF","UBE2C","PCLAF")
)

b_modules <- list(
  Resting_naive_B_cell=c("IGHD","IGHM","TCL1A","IL4R","FCER2","CD72","BACH2","CCR7","SELL","CD22","MS4A1","CD79A","CD79B","BANK1","BCL2"),
  Resting_memory_B_cell=c("CD27","TNFRSF13B","AIM2","GPR183","CD80","CD86","BANK1","MS4A1","CD79A","CD37","HLA-DRA","HLA-DPA1"),
  Atypical_memory_CD11c_like=c("FCRL5","FCRL4","ITGAX","TBX21","ZEB2","CXCR3","DUSP4","LYN","HOPX","TLR7","TLR9","FCGR2B","CD86","CD80"),
  Plasmablast_plasma_differentiation=c("PRDM1","XBP1","IRF4","MZB1","SDC1","JCHAIN","SSR4","FKBP11","DERL3","TNFRSF17","SLAMF7","CD38","CD27","SEC11C"),
  IgA_mucosal_plasma_cell=c("IGHA1","IGHA2","JCHAIN","MZB1","XBP1","SDC1","TNFRSF17","CCR10","PRDM1","IRF4"),
  IgG_inflammatory_plasma_cell=c("IGHG1","IGHG2","IGHG3","IGHG4","JCHAIN","MZB1","XBP1","SDC1","PRDM1","IRF4","CXCR4"),
  IgM_IgD_unswitched_B_cell=c("IGHM","IGHD","TCL1A","FCER2","CD72","CCR7","SELL","BACH2","MS4A1","CD79A","CD79B"),
  B_cell_antigen_presentation=c("HLA-DRA","HLA-DRB1","HLA-DPA1","HLA-DPB1","HLA-DQA1","HLA-DQB1","CD74","CIITA","HLA-DMA","HLA-DMB","CD86","CD80","CD40"),
  BCR_NFkB_activation=c("CD69","NR4A1","NR4A2","DUSP2","DUSP4","FOS","JUNB","NFKBIA","NFKBIZ","TNFAIP3","REL","BCL2A1","CD83","CD86","IRF4"),
  TLR_inflammatory_B_cell=c("TLR7","TLR9","MYD88","IRF5","NFKBIA","NFKBIZ","REL","TNFAIP3","CD69","JUNB","FOS","DUSP1","CCL3","CCL4"),
  BAFF_APRIL_survival_response=c("TNFRSF13B","TNFRSF17","TNFRSF13C","BCL2","MCL1","BCL2A1","CD40","IL6R","JCHAIN","XBP1","MZB1"),
  Mucosal_trafficking_retention_B_cell=c("CXCR4","CXCR5","CCR6","CCR7","CCR10","ITGA4","ITGB7","SELL","S1PR1","S1PR2","GPR183","CD69"),
  Cell_cycle_proliferating_B_cell=c("MKI67","TOP2A","STMN1","TYMS","PCNA","MCM2","MCM3","MCM4","MCM5","MCM6","MCM7","HMGB2","CENPF","UBE2C","PCLAF")
)

valid <- function(x) !is.na(x) & nzchar(trimws(as.character(x))) & toupper(trimws(as.character(x))) != "NA"

clone_id <- function(md, modality) {
  if (modality == "TCR") {
    ok <- grepl("^TRBC", md$TCR_Beta_Delta_C_gene_Dominant) & valid(md$TCR_Beta_Delta_V_gene_Dominant) &
      valid(md$TCR_Beta_Delta_J_gene_Dominant) & valid(md$TCR_Beta_Delta_CDR3_Translation_Dominant)
    id <- rep(NA_character_, nrow(md))
    id[ok] <- paste(md$TCR_Beta_Delta_V_gene_Dominant[ok], md$TCR_Beta_Delta_J_gene_Dominant[ok],
                    md$TCR_Beta_Delta_CDR3_Translation_Dominant[ok], sep="|")
  } else {
    ok <- grepl("^IGH", md$BCR_Heavy_C_gene_Dominant) & valid(md$BCR_Heavy_V_gene_Dominant) &
      valid(md$BCR_Heavy_J_gene_Dominant) & valid(md$BCR_Heavy_CDR3_Translation_Dominant)
    id <- rep(NA_character_, nrow(md))
    id[ok] <- paste(md$BCR_Heavy_V_gene_Dominant[ok], md$BCR_Heavy_J_gene_Dominant[ok],
                    md$BCR_Heavy_CDR3_Translation_Dominant[ok], sep="|")
  }
  id
}

score_and_match <- function(path, compartment, modality, modules) {
  message("Loading/scoring ", compartment)
  obj <- readRDS(path)
  md <- as.data.frame(obj[[]], stringsAsFactors=FALSE)
  md$cell <- rownames(md)
  md$clone_id <- clone_id(md, modality)
  md$state <- as.character(md$AnnotationLevel2)
  md$SampleID <- as.character(md$SampleID)
  keep <- valid(md$clone_id) & valid(md$state) & valid(md$SampleID)
  md <- md[keep, , drop=FALSE]
  cs <- md %>% count(SampleID, clone_id, name="clone_size")
  md <- md %>% left_join(cs, by=c("SampleID","clone_id")) %>%
    mutate(status=ifelse(clone_size >= 2, "Expanded", "Singleton"), compartment=compartment,
           acquisition_series=sub("[AB]$", "", as.character(Batch)))
  expr <- LayerData(obj[["RNA"]], layer="data")[, md$cell, drop=FALSE]
  score_mat <- vapply(modules, function(gs) {
    g <- intersect(gs, rownames(expr))
    if (length(g) < 3) rep(NA_real_, ncol(expr)) else Matrix::colMeans(expr[g, , drop=FALSE])
  }, numeric(ncol(expr)))
  score_mat <- as.data.frame(score_mat, check.names=FALSE)
  score_mat$cell <- md$cell
  long <- bind_cols(md, score_mat %>% select(-cell)) %>%
    pivot_longer(cols=all_of(names(modules)), names_to="module", values_to="score") %>%
    group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series, compartment, state, status, module) %>%
    summarise(score=mean(score, na.rm=TRUE), n_cells=n(), .groups="drop") %>%
    group_by(SampleID, compartment, state, module) %>% filter(n_distinct(status)==2) %>% ungroup()
  # Equal state weights remove cell-composition differences; each state must contain both statuses.
  paired <- long %>% select(-n_cells) %>% pivot_wider(names_from=status, values_from=score) %>%
    filter(is.finite(Expanded), is.finite(Singleton)) %>% mutate(delta=Expanded-Singleton)
  participant <- paired %>% group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series, compartment, module) %>%
    summarise(delta=mean(delta), n_matched_states=n(), .groups="drop")
  rm(expr, obj); invisible(gc())
  list(state=paired, participant=participant)
}

res <- list()
for (nm in names(paths)) res[[nm]] <- score_and_match(paths[[nm]], nm, ifelse(nm=="BCR","BCR","TCR"), if (nm=="BCR") b_modules else t_modules)
state_tbl <- bind_rows(lapply(res, `[[`, "state"))
part0 <- bind_rows(lapply(res, `[[`, "participant"))
# CD4 and CD8 remain separate state spaces until participant deltas are combined with equal compartment weights.
part <- part0 %>% mutate(modality=ifelse(compartment=="BCR","BCR","TCR")) %>%
  group_by(SampleID, Diagnosis1, Inflammation1, Biologic, Batch, acquisition_series, modality, module) %>%
  summarise(delta=mean(delta), n_matched_states=sum(n_matched_states), n_compartments=n_distinct(compartment), .groups="drop")

fit_limma <- function(d, formula, contrast_defs, family) {
  if (!nrow(d)) return(tibble())
  wide <- d %>% select(SampleID, module, delta) %>% pivot_wider(names_from=SampleID, values_from=delta)
  mods <- wide$module; mat <- as.matrix(wide[,-1]); rownames(mat) <- mods
  meta <- d %>% distinct(SampleID, .keep_all=TRUE) %>% arrange(match(SampleID, colnames(mat)))
  mat <- mat[, meta$SampleID, drop=FALSE]
  design <- model.matrix(formula, data=meta)
  # Keep an estimable basis; report which requested contrasts cannot be represented.
  q <- qr(design); if (q$rank < ncol(design)) design <- design[, sort(q$pivot[seq_len(q$rank)]), drop=FALSE]
  fit <- eBayes(lmFit(mat, design), robust=TRUE)
  bind_rows(lapply(names(contrast_defs), function(nm) {
    expr <- contrast_defs[[nm]]
    cc <- tryCatch(makeContrasts(contrasts=expr, levels=design), error=function(e) NULL)
    if (is.null(cc)) return(tibble(family=family, contrast=nm, estimable=FALSE, module=mods))
    ff <- eBayes(contrasts.fit(fit, cc), robust=TRUE)
    tt <- topTable(ff, number=Inf, sort.by="none")
    tibble(family=family, contrast=nm, estimable=TRUE, module=rownames(tt), effect=tt$logFC,
           SE=tt$logFC/tt$t, t=tt$t, p_value=tt$P.Value)
  })) %>% group_by(family, contrast) %>% mutate(FDR=p.adjust(p_value,"BH")) %>% ungroup()
}

all_tests <- list()
for (modality in c("TCR","BCR")) {
  d <- part %>% filter(.data$modality==.env$modality) %>% mutate(
    Diagnosis=factor(Diagnosis1, levels=c("Control","CD","UC")),
    Biologic=factor(ifelse(is.na(Biologic)|Biologic=="", "Unknown", Biologic)),
    Series=factor(acquisition_series))
  all_tests[[paste0(modality,"_dx")]] <- fit_limma(d, ~0+Diagnosis+Biologic+Series,
    c(Control_expansion="DiagnosisControl", CD_expansion="DiagnosisCD", UC_expansion="DiagnosisUC",
      CD_vs_Control_interaction="DiagnosisCD-DiagnosisControl", UC_vs_Control_interaction="DiagnosisUC-DiagnosisControl",
      CD_vs_UC_interaction="DiagnosisCD-DiagnosisUC"), paste0(modality,"_diagnosis"))

  di <- d %>% filter(Diagnosis1 %in% c("CD","UC")) %>% mutate(Inflammation=factor(Inflammation1, levels=c("Noninflamed","Inflamed")))
  all_tests[[paste0(modality,"_ibdinf")]] <- fit_limma(di, ~0+Inflammation+Diagnosis+Biologic+Series,
    c(IBD_noninflamed_expansion="InflammationNoninflamed", IBD_inflamed_expansion="InflammationInflamed",
      inflammation_interaction="InflammationInflamed-InflammationNoninflamed"), paste0(modality,"_IBD_inflammation"))
  for (dx in c("CD","UC")) {
    dd <- di %>% filter(Diagnosis1==dx)
    all_tests[[paste(modality,dx,sep="_")]] <- fit_limma(dd, ~0+Inflammation+Biologic+Series,
      c(noninflamed_expansion="InflammationNoninflamed", inflamed_expansion="InflammationInflamed",
        inflamed_vs_noninflamed_interaction="InflammationInflamed-InflammationNoninflamed"), paste0(modality,"_",dx,"_inflammation"))
  }
}
tests <- bind_rows(all_tests)

meta_random <- function(y, se) {
  ok <- is.finite(y); y<-y[ok]; k<-length(y)
  # Acquisition series are treated as the replication unit. An unweighted
  # t-based synthesis is deliberately conservative for sparse/unequal series.
  if (k<3) return(c(k=k,effect=NA,se=NA,t=NA,p=NA,tau2=NA,I2=NA))
  mu<-mean(y); ser<-sd(y)/sqrt(k); tt<-ifelse(is.finite(ser)&&ser>0,mu/ser,NA_real_)
  c(k=k,effect=mu,se=ser,t=tt,p=ifelse(is.finite(tt),2*pt(-abs(tt),df=k-1),NA_real_),tau2=var(y),I2=NA_real_)
}

series_effects <- part %>% group_by(modality,module,Diagnosis1,Inflammation1,acquisition_series) %>%
  summarise(n=n(), effect=mean(delta), se=sd(delta)/sqrt(n()), .groups="drop") %>% filter(n>=2,is.finite(effect))
series_meta <- series_effects %>% group_by(modality,module,Diagnosis1,Inflammation1) %>%
  summarise(tmp=list(meta_random(effect,se)), .groups="drop") %>% unnest_wider(tmp) %>%
  group_by(modality,Diagnosis1,Inflammation1) %>% mutate(FDR=p.adjust(p,"BH")) %>% ungroup()

# Cross-compartment coupling: participant-level selected T and B expansion deltas.
t_sel <- part %>% filter(modality=="TCR", module %in% c("Effector_cytotoxicity","EOMES_ZEB2_inflammatory_CD8_TRM_like","Th1_Tc1_inflammatory")) %>%
  select(SampleID,Diagnosis1,Inflammation1,Biologic,acquisition_series,T_module=module,T_delta=delta)
b_sel <- part %>% filter(modality=="BCR", module %in% c("Plasmablast_plasma_differentiation","IgA_mucosal_plasma_cell","IgG_inflammatory_plasma_cell")) %>%
  select(SampleID,B_module=module,B_delta=delta)
coupling_dat <- inner_join(t_sel,b_sel,by="SampleID")
coupling <- coupling_dat %>% group_by(T_module,B_module,Diagnosis1) %>% group_modify(~{
  d=.x; if(nrow(d)<6) return(tibble(n=nrow(d),rho=NA_real_,p_value=NA_real_))
  ct=cor.test(d$T_delta,d$B_delta,method="spearman",exact=FALSE); tibble(n=nrow(d),rho=unname(ct$estimate),p_value=ct$p.value)
}) %>% ungroup() %>% group_by(Diagnosis1) %>% mutate(FDR=p.adjust(p_value,"BH")) %>% ungroup()

write_csv(state_tbl, file.path(outdir,"Table_CSI1_state_matched_module_scores.csv"))
write_csv(part, file.path(outdir,"Table_CSI2_participant_expanded_minus_singleton_deltas.csv"))
write_csv(tests, file.path(outdir,"Table_CSI3_formal_interaction_models.csv"))
write_csv(series_effects, file.path(outdir,"Table_CSI4_acquisition_series_effects.csv"))
write_csv(series_meta, file.path(outdir,"Table_CSI5_random_effects_meta_analysis.csv"))
write_csv(coupling, file.path(outdir,"Table_CSI6_T_B_cross_compartment_coupling.csv"))

manifest <- tibble(
  item=c("analysis_date","primary_unit","expansion_definition","state_matching","diagnosis_model","inflammation_model","biologic_covariate","acquisition_replication"),
  value=c("2026-08-24","participant expanded-minus-singleton delta","exact productive beta-chain (TCR) or heavy-chain (BCR) V+J+CDR3 amino acid; expanded >=2 cells","only participant-state strata containing both expanded and singleton cells; equal state weights","delta ~ diagnosis + biologic class + acquisition series","delta ~ inflammation + diagnosis + biologic class + acquisition series; diagnosis-stratified models also fit","post-treatment biologic class entered as covariate; no prospective-response claim","series-specific effects and random-effects meta-analysis where >=2 estimable series")
)
write_csv(manifest,file.path(outdir,"Analysis_manifest_clone_state_interactions.csv"))
message("Wrote clone-state interaction results to: ",outdir)

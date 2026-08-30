#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(SeuratObject);library(Matrix);library(dplyr);library(readr)})

outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses/Paired TCR Sequence State"
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
paths <- c(
 CD4="C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD4T2026_scvi30_epoch400_umap.rds",
 CD8="C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/CD8T2026_scvi50_epoch400_umap.rds")
mods <- list(
 Effector_cytotoxicity=c("NKG7","GNLY","PRF1","GZMB","GZMA","GZMH","GZMK","GZMM","CTSW","CST7","FGFBP2","CCL5","CCL4","CCL3","IFNG","FASLG","KLRD1","KLRG1"),
 Th1_Tc1_inflammatory=c("TBX21","STAT4","CXCR3","IFNG","TNF","IL12RB2","CCL5","CCL4","NKG7","GZMB","PRF1","CXCR6","BHLHE40"),
 EOMES_ZEB2_inflammatory_CD8_TRM_like=c("EOMES","ZEB2","GZMB","GZMH","PRF1","NKG7","CX3CR1","KLRG1","TBX21","CCL5","CST7","FGFBP2"),
 Tissue_resident_mucosal_retention=c("CD69","ITGAE","ITGA1","CXCR6","ZNF683","RUNX3","PRDM1","RGS1","CD101","DUSP6","AHR","CCR6"),
 Gut_homing_intestinal_trafficking=c("ITGA4","ITGB7","ITGAE","CCR9","CCR6","CXCR3","CXCR6","SELPLG","S1PR1","KLF2","SELL","GPR183","CD69"))
valid <- function(x) !is.na(x)&nzchar(trimws(as.character(x)))&grepl("^[A-Z]+$",as.character(x))&!grepl("[*]",as.character(x))

one <- function(path,comp){
 obj<-readRDS(path); md<-as.data.frame(obj[[]],stringsAsFactors=FALSE); md$cell<-rownames(md)
 keep<-grepl("^TRAC",md$TCR_Alpha_Gamma_C_gene_Dominant)&grepl("^TRBC",md$TCR_Beta_Delta_C_gene_Dominant)&
  valid(md$TCR_Alpha_Gamma_CDR3_Translation_Dominant)&valid(md$TCR_Beta_Delta_CDR3_Translation_Dominant)&
  !is.na(md$SampleID)&nzchar(as.character(md$SampleID))&!is.na(md$AnnotationLevel2)
 md<-md[keep,,drop=FALSE]
 expr<-LayerData(obj[["RNA"]],layer="data")[,md$cell,drop=FALSE]
 sm<-vapply(mods,function(gs){g<-intersect(gs,rownames(expr));Matrix::colMeans(expr[g,,drop=FALSE])},numeric(ncol(expr)))
 sm<-as.data.frame(sm,check.names=FALSE)
 d<-bind_cols(md,sm)%>%transmute(
  SampleID=as.character(SampleID),PatientID=as.character(PatientID),Diagnosis=as.character(Diagnosis1),
  Inflammation=as.character(Inflammation1),Biologic=as.character(Biologic),Batch=as.character(Batch),
  acquisition_series=sub("[AB]$","",as.character(Batch)),compartment=comp,state=as.character(AnnotationLevel2),
  alpha_v=as.character(TCR_Alpha_Gamma_V_gene_Dominant),alpha_j=as.character(TCR_Alpha_Gamma_J_gene_Dominant),
  alpha_cdr3=as.character(TCR_Alpha_Gamma_CDR3_Translation_Dominant),
  beta_v=as.character(TCR_Beta_Delta_V_gene_Dominant),beta_j=as.character(TCR_Beta_Delta_J_gene_Dominant),
  beta_cdr3=as.character(TCR_Beta_Delta_CDR3_Translation_Dominant),across(all_of(names(mods))))%>%
  mutate(clone_id=paste(alpha_v,alpha_j,alpha_cdr3,beta_v,beta_j,beta_cdr3,sep="|"))
 rm(expr,obj);invisible(gc());d
}
d<-bind_rows(one(paths[["CD4"]],"CD4"),one(paths[["CD8"]],"CD8"))
state_occ<-d%>%count(SampleID,clone_id,state,name="state_cells")%>%group_by(SampleID,clone_id)%>%
 mutate(state_fraction=state_cells/sum(state_cells),state_rank=min_rank(desc(state_cells)))%>%ungroup()
clone<-d%>%group_by(SampleID,PatientID,Diagnosis,Inflammation,Biologic,Batch,acquisition_series,clone_id,
 alpha_v,alpha_j,alpha_cdr3,beta_v,beta_j,beta_cdr3)%>%
 summarise(n_cells=n(),n_compartments=n_distinct(compartment),n_states=n_distinct(state),across(all_of(names(mods)),mean),.groups="drop")%>%
 left_join(state_occ%>%filter(state_rank==1)%>%group_by(SampleID,clone_id)%>%slice_head(n=1)%>%ungroup()%>%
  select(SampleID,clone_id,dominant_state=state,dominant_state_fraction=state_fraction),by=c("SampleID","clone_id"))
write_csv(clone,file.path(outdir,"Table_TSS1_paired_alpha_beta_clone_sequence_state.csv"))
write_csv(state_occ,file.path(outdir,"Table_TSS2_clone_state_occupancy.csv"))
write_csv(tibble(item=c("HLA genotype","graph nodes","sequence features","transcriptome features","permutation strata"),
 value=c("Not available; analysis is exploratory and not HLA-stratified","participant-specific exact paired alpha-beta clonotypes","paired CDR3 amino-acid similarity plus alpha/beta V/J usage","mean module scores per clonotype","receptor/state labels permuted within participant and acquisition series")),file.path(outdir,"Analysis_manifest_TCR_sequence_state.csv"))
message("Exported ",nrow(clone)," paired alpha-beta clone nodes")

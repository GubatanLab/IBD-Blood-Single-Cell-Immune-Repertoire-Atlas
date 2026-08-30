#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(dplyr);library(readr);library(SeuratObject)})
outdir <- "C:/path/to/private-manuscript-workspace/High Impact Additional Analyses/BCR Germline Lineages"
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
airr_path <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 3 BCR/IBDBCRonly.rds"
b_path <- "C:/path/to/private-user-home/OneDrive/Desktop/IBD SingleCell Repertoire Manuscript/Figure 1/Figure 1/BCell2026_scvi50_umap.rds"
x<-readRDS(airr_path); meta<-as.data.frame(x$meta,stringsAsFactors=FALSE)
rows<-lapply(names(x$data),function(s){
 d<-as.data.frame(x$data[[s]],stringsAsFactors=FALSE);m<-meta[match(s,meta$Sample),,drop=FALSE]
 if(!nrow(d)||!nrow(m))return(NULL)
 d%>%filter(grepl("^IGHV",V.name),grepl("^IGHJ",J.name),!is.na(Sequence),grepl("^[ACGTN]+$",toupper(Sequence)),
             nchar(Sequence)>=250,!is.na(CDR3.nt),grepl("^[ACGT]+$",toupper(CDR3.nt)))%>%
  transmute(SampleID=as.character(m$SampleID),PatientID=as.character(m$PatientID),Diagnosis=as.character(m$Diagnosis1),
   Inflammation=as.character(m$Inflammation1),Biologic=as.character(m$Biologic),Batch=as.character(m$Batch),
   acquisition_series=sub("[AB]$","",as.character(m$Batch)),count=suppressWarnings(as.numeric(Clones)),proportion=suppressWarnings(as.numeric(Proportion)),
   sequence=toupper(Sequence),v_call_original=as.character(V.name),d_call_original=as.character(D.name),j_call_original=as.character(J.name),
   junction= toupper(CDR3.nt),junction_aa=as.character(CDR3.aa),
   fwr1=FR1.nt,cdr1=CDR1.nt,fwr2=FR2.nt,cdr2=CDR2.nt,fwr3=FR3.nt,fwr4=FR4.nt)
})
q<-bind_rows(rows)%>%mutate(count=ifelse(is.finite(count)&count>0,count,1),
 v_gene=sub("[*].*$","",v_call_original),j_gene=sub("[*].*$","",j_call_original),
 receptor_key=paste(SampleID,v_gene,j_gene,junction,sep="|"))%>%
 group_by(SampleID,sequence)%>%summarise(across(c(PatientID,Diagnosis,Inflammation,Biologic,Batch,acquisition_series,v_call_original,d_call_original,j_call_original,junction,junction_aa,fwr1,cdr1,fwr2,cdr2,fwr3,fwr4,v_gene,j_gene,receptor_key),first),count=sum(count),proportion=sum(proportion,na.rm=TRUE),.groups="drop")%>%
 mutate(sequence_id=sprintf("BCR_%07d",row_number()),sequence_length=nchar(sequence))

writeLines(as.vector(rbind(paste0(">",q$sequence_id),q$sequence)),file.path(outdir,"bcr_heavy_queries.fasta"))
write_csv(q,file.path(outdir,"Table_BGL1_query_sequence_metadata.csv"))

obj<-readRDS(b_path);md<-as.data.frame(obj[[]],stringsAsFactors=FALSE)
good<-grepl("^IGH",md$BCR_Heavy_C_gene_Dominant)&!is.na(md$SampleID)&!is.na(md$BCR_Heavy_CDR3_Nucleotide_Dominant)&
 grepl("^[ACGT]+$",toupper(md$BCR_Heavy_CDR3_Nucleotide_Dominant))
cm<-md[good,,drop=FALSE]%>%transmute(SampleID=as.character(SampleID),
 v_gene=sub("[*].*$","",as.character(BCR_Heavy_V_gene_Dominant)),j_gene=sub("[*].*$","",as.character(BCR_Heavy_J_gene_Dominant)),
 junction=toupper(as.character(BCR_Heavy_CDR3_Nucleotide_Dominant)),isotype=as.character(BCR_Heavy_C_gene_Dominant),
 light_v=as.character(BCR_Light_V_gene_Dominant),light_j=as.character(BCR_Light_J_gene_Dominant),light_cdr3=as.character(BCR_Light_CDR3_Translation_Dominant),
 cell_state=as.character(AnnotationLevel2))%>%mutate(receptor_key=paste(SampleID,v_gene,j_gene,junction,sep="|"))%>%
 count(receptor_key,isotype,light_v,light_j,light_cdr3,cell_state,name="n_cells")
write_csv(cm,file.path(outdir,"Table_BGL2_cell_state_isotype_light_mapping.csv"))

audit<-tibble(metric=c("heavy_queries","participants","median_sequence_length","fraction_with_complete_FWR1_CDR1_FWR2_CDR2_FWR3_FWR4","cell_mapping_rows","HLA_available"),
 value=c(nrow(q),n_distinct(q$SampleID),median(q$sequence_length),mean(complete.cases(q[,c("fwr1","cdr1","fwr2","cdr2","fwr3","fwr4")])),nrow(cm),FALSE))
write_csv(audit,file.path(outdir,"Input_audit_BCR_germline_lineages.csv"))
message("Prepared ",nrow(q)," unique heavy-chain query sequences")

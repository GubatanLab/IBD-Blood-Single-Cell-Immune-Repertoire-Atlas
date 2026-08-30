#!/usr/bin/env python
from pathlib import Path
from itertools import combinations
import math
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import mannwhitneyu

root=Path(r"C:/path/to/private-manuscript-workspace\High Impact Additional Analyses\BCR Germline Lineages")
db=pd.read_csv(root/"bcr_heavy_lineages_d015_clone-pass.tsv",sep="\t",low_memory=False)
cells=pd.read_csv(root/"Table_BGL2_cell_state_isotype_light_mapping.csv",low_memory=False)
db=db.rename(columns={"sampleid":"SampleID","patientid":"PatientID","diagnosis":"Diagnosis","inflammation":"Inflammation","biologic":"Biologic","batch":"Batch"})
db["lineage_id"]=db.SampleID.astype(str)+"::"+db.clone_id.astype(str)
db["count"]=pd.to_numeric(db["count"],errors="coerce").fillna(1).clip(lower=1)
db["v_gene_core"]=db.v_call.astype(str).str.replace(r"\*.*$","",regex=True)
db["j_gene_core"]=db.j_call.astype(str).str.replace(r"\*.*$","",regex=True)
db["junction_core"]=db.junction.astype(str).str.slice(3,-3)
db["mapping_key"]=db.SampleID.astype(str)+"|"+db.v_gene_core+"|"+db.j_gene_core+"|"+db.junction_core

regions=["fwr1","cdr1","fwr2","cdr2","fwr3","fwr4"]
def region_metrics(row,region):
    obs=str(row.get(region,""));sa=str(row.get("sequence_alignment",""));ga=str(row.get("germline_alignment",""))
    if obs in ("","nan","None") or sa in ("nan","None") or ga in ("nan","None"):return (np.nan,np.nan,np.nan,np.nan,np.nan)
    pos=sa.find(obs)
    if pos<0:return (np.nan,np.nan,np.nan,np.nan,np.nan)
    germ=ga[pos:pos+len(obs)]
    pairs=[(a,b) for a,b in zip(obs.upper(),germ.upper()) if a in "ACGT" and b in "ACGT"]
    info=len(pairs);muts=sum(a!=b for a,b in pairs);rate=muts/info if info else np.nan
    # Descriptive replacement/silent codon counts only; no selection inference.
    o="".join(a for a,b in zip(obs.upper(),germ.upper()) if a in "ACGT" and b in "ACGT")
    g="".join(b for a,b in zip(obs.upper(),germ.upper()) if a in "ACGT" and b in "ACGT")
    rep=sil=0
    for k in range(0,min(len(o),len(g))-2,3):
        oc=o[k:k+3];gc=g[k:k+3]
        if oc==gc:continue
        try: oa=str(Seq(oc).translate());gaaa=str(Seq(gc).translate())
        except Exception:continue
        if "*" in (oa,gaaa):continue
        if oa==gaaa:sil+=1
        else:rep+=1
    return info,muts,rate,rep,sil

seq_rows=[]
for r in db.itertuples(index=False):
    base={"sequence_id":r.sequence_id,"lineage_id":r.lineage_id,"SampleID":r.SampleID,"Diagnosis":r.Diagnosis,"Inflammation":r.Inflammation,"count":r.count}
    rd=r._asdict()
    for reg in regions:
        info,muts,rate,rep,sil=region_metrics(rd,reg)
        seq_rows.append({**base,"region":reg,"informative_nt":info,"mutations":muts,"shm_rate":rate,"replacement_codons":rep,"silent_codons":sil})
seqreg=pd.DataFrame(seq_rows)

def hdist(a,b):
    pairs=[(x,y) for x,y in zip(str(a).upper(),str(b).upper()) if x in "ACGT" and y in "ACGT"]
    return sum(x!=y for x,y in pairs)/len(pairs) if pairs else 1.0

lineages=[]
for lid,g in db.groupby("lineage_id",sort=False):
    weights=g["count"].to_numpy(float);p=weights/weights.sum();shannon=float(-np.sum(p*np.log(p))) if len(p)>1 else 0.0
    seqs=g.sequence_alignment.astype(str).tolist();germ=str(g.germline_alignment.iloc[0]);use=seqs[:100]
    nodes=[germ]+use
    if len(nodes)>1:
        dm=np.zeros((len(nodes),len(nodes)),float)
        for i in range(len(nodes)):
            for j in range(i):dm[i,j]=dm[j,i]=hdist(nodes[i],nodes[j])
        mst=minimum_spanning_tree(dm).toarray();total=float(mst.sum());maxg=float(max(dm[0,1:]))
    else:total=maxg=0.0
    lineages.append(dict(lineage_id=lid,SampleID=g.SampleID.iloc[0],Diagnosis=g.Diagnosis.iloc[0],Inflammation=g.Inflammation.iloc[0],Biologic=g.Biologic.iloc[0],acquisition_series=g.acquisition_series.iloc[0],
      n_unique_sequences=len(g),weighted_abundance=float(weights.sum()),lineage_shannon=shannon,total_MST_branch_length=total,max_germline_distance=maxg,tree_nodes_used=len(use),tree_truncated=len(seqs)>100,
      v_gene=str(g.v_call.iloc[0]).split("*")[0],j_gene=str(g.j_call.iloc[0]).split("*")[0]))
lin=pd.DataFrame(lineages)

# Map isotype, paired light chain and transcriptomic state by the exact heavy receptor key.
mapbase=db[["lineage_id","mapping_key","SampleID","Diagnosis","Inflammation"]].drop_duplicates(["lineage_id","mapping_key"])
mapd=mapbase.merge(cells,left_on="mapping_key",right_on="receptor_key",how="left")
mapd["isotype_class"]=np.select([mapd.isotype.astype(str).str.startswith("IGHA"),mapd.isotype.astype(str).str.startswith("IGHG"),mapd.isotype.astype(str).str.startswith("IGHM"),mapd.isotype.astype(str).str.startswith("IGHD"),mapd.isotype.astype(str).str.startswith("IGHE")],["IgA","IgG","IgM","IgD","IgE"],default="Unknown")
mapd["light_id"]=mapd.light_v.fillna("").astype(str)+"|"+mapd.light_j.fillna("").astype(str)+"|"+mapd.light_cdr3.fillna("").astype(str)
iso=mapd.dropna(subset=["n_cells"]).groupby(["lineage_id","isotype_class"],as_index=False).n_cells.sum()
state=mapd.dropna(subset=["n_cells"]).groupby(["lineage_id","cell_state"],as_index=False).n_cells.sum()
light=mapd.dropna(subset=["n_cells"]).groupby(["lineage_id","light_id"],as_index=False).n_cells.sum()

def iso_summary(g):
    cats=set(g.loc[g.n_cells>0,"isotype_class"]);return pd.Series({"mapped_cells":g.n_cells.sum(),"n_isotype_classes":len(cats),"isotype_classes":";".join(sorted(cats)),"class_switched_lineage":bool(cats&{"IgA","IgG","IgE"}),"mixed_unswitched_switched":bool(cats&{"IgM","IgD"}) and bool(cats&{"IgA","IgG","IgE"})})
isum=iso.groupby("lineage_id").apply(iso_summary,include_groups=False).reset_index()
lsum=light.groupby("lineage_id").agg(n_exact_light_refinements=("light_id","nunique"),light_cells=("n_cells","sum"),dominant_light_cells=("n_cells","max")).reset_index()
lsum["dominant_light_fraction"]=lsum.dominant_light_cells/lsum.light_cells
ssum=state.groupby("lineage_id").cell_state.nunique().rename("n_cell_states").reset_index()
lin=lin.merge(isum,on="lineage_id",how="left").merge(lsum[["lineage_id","n_exact_light_refinements","dominant_light_fraction"]],on="lineage_id",how="left").merge(ssum,on="lineage_id",how="left")
lin["mapped_cells"]=lin.mapped_cells.fillna(0);lin["class_switched_lineage"]=lin.class_switched_lineage.fillna(False);lin["mixed_unswitched_switched"]=lin.mixed_unswitched_switched.fillna(False)

traj=[]
for lid,g in iso.groupby("lineage_id"):
    cats=sorted(set(g.loc[g.n_cells>0,"isotype_class"])-{"Unknown"})
    for a,b in combinations(cats,2):traj.append((lid,a,b))
traj=pd.DataFrame(traj,columns=["lineage_id","isotype_1","isotype_2"]) if traj else pd.DataFrame(columns=["lineage_id","isotype_1","isotype_2"])
traj_summary=traj.merge(lin[["lineage_id","Diagnosis"]],on="lineage_id",how="left").groupby(["Diagnosis","isotype_1","isotype_2"],as_index=False).lineage_id.nunique().rename(columns={"lineage_id":"n_lineages"})

plasma=state.pivot_table(index="lineage_id",columns="cell_state",values="n_cells",aggfunc="sum",fill_value=0).reset_index()
for c in ["IgA Plasma B Cell","IgG Plasma B Cell"]:
    if c not in plasma:plasma[c]=0
shared=plasma[(plasma["IgA Plasma B Cell"]>0)&(plasma["IgG Plasma B Cell"]>0)].merge(lin,on="lineage_id",how="left")

sample_mut=seqreg.groupby(["SampleID","Diagnosis","Inflammation","region"],as_index=False).apply(lambda g:pd.Series({"weighted_shm_rate":np.average(g.shm_rate.fillna(0),weights=g["count"]*g.informative_nt.fillna(0)) if np.sum(g["count"]*g.informative_nt.fillna(0))>0 else np.nan,"replacement_codons":g.replacement_codons.sum(),"silent_codons":g.silent_codons.sum(),"informative_sequences":g.shm_rate.notna().sum()}),include_groups=False)
sample_lin=lin.groupby(["SampleID","Diagnosis","Inflammation"],as_index=False).agg(n_lineages=("lineage_id","nunique"),expanded_lineages=("n_unique_sequences",lambda x:int(np.sum(x>=2))),mean_lineage_shannon=("lineage_shannon","mean"),mean_MST_branch_length=("total_MST_branch_length",lambda x:float(np.mean(x[x>0])) if np.any(x>0) else 0),fraction_class_switched_lineages=("class_switched_lineage","mean"),fraction_mixed_unswitched_switched=("mixed_unswitched_switched","mean"))

tests=[]
for table,name,metrics in [(sample_mut,"mutation",["weighted_shm_rate"]),(sample_lin,"lineage",["expanded_lineages","mean_lineage_shannon","mean_MST_branch_length","fraction_class_switched_lineages","fraction_mixed_unswitched_switched"])]:
    extra=[None] if name=="lineage" else sorted(table.region.unique())
    for reg in extra:
        d=table if reg is None else table[table.region==reg]
        for metric in metrics:
            for a,b in [("CD","Control"),("UC","Control"),("CD","UC")]:
                x=d.loc[d.Diagnosis==a,metric].dropna();y=d.loc[d.Diagnosis==b,metric].dropna()
                p=mannwhitneyu(x,y,alternative="two-sided").pvalue if len(x)>=3 and len(y)>=3 else np.nan
                tests.append(dict(analysis=name,region=reg,metric=metric,contrast=f"{a} vs {b}",n1=len(x),n2=len(y),median1=x.median(),median2=y.median(),p_value=p))
tests=pd.DataFrame(tests)
def bh(s):
    p=s.to_numpy(float);ok=np.isfinite(p);q=np.full(len(p),np.nan);order=np.argsort(p[ok]);vals=p[ok][order];adj=np.minimum.accumulate((vals*len(vals)/np.arange(1,len(vals)+1))[::-1])[::-1];tmp=np.empty(len(vals));tmp[order]=np.minimum(adj,1);q[ok]=tmp;return q
tests["FDR"]=tests.groupby(["analysis","metric"],group_keys=False).p_value.transform(bh)

seqreg.to_csv(root/"Table_BGL3_sequence_region_SHM_RS.csv",index=False)
lin.to_csv(root/"Table_BGL4_germline_aware_lineage_metrics.csv",index=False)
iso.to_csv(root/"Table_BGL5_lineage_isotype_composition.csv",index=False)
state.to_csv(root/"Table_BGL6_lineage_transcriptional_state_composition.csv",index=False)
light.to_csv(root/"Table_BGL7_paired_light_chain_refinement.csv",index=False)
traj_summary.to_csv(root/"Table_BGL8_class_switch_cooccupancy_network.csv",index=False)
shared.to_csv(root/"Table_BGL9_shared_IgA_IgG_plasma_state_lineages.csv",index=False)
sample_mut.to_csv(root/"Table_BGL10_participant_region_SHM.csv",index=False)
sample_lin.to_csv(root/"Table_BGL11_participant_lineage_metrics.csv",index=False)
tests.to_csv(root/"Table_BGL12_diagnosis_comparisons.csv",index=False)
pd.DataFrame({"item":["IgBLAST","germline reference","Change-O parsing","lineage definition","regional SHM","replacement/silent","phylogeny","class switching","paired light refinement","terminal-branch limitation","BASELINe"],"value":["NCBI IgBLAST 1.22.0","IMGT human IG reference directory downloaded 2026-08-24","MakeDb extended AIRR output; 49,356 pass/93 fail","within participant; shared V/J gene and junction length; normalized nucleotide Hamming <=0.15; single linkage","observed-versus-IMGT germline mismatches in FWR1/CDR1/FWR2/CDR2/FWR3/FWR4","descriptive codon counts only","minimum-spanning tree over IMGT-aligned heavy sequences plus germline root; max 100 observed nodes/lineage","cross-sectional isotype co-occupancy, not temporal direction","exact paired light V/J/CDR3 subdivisions linked from single-cell metadata","cell-level metadata lacks full heavy variable sequence; IgA/IgG plasma co-occupancy can be tested, but terminal branch identity cannot be assigned","not performed"]}).to_csv(root/"Analysis_manifest_BCR_germline_lineages.csv",index=False)
print(f"sequences={len(db)} lineages={len(lin)} expanded_lineages={(lin.n_unique_sequences>=2).sum()} mapped_lineages={(lin.mapped_cells>0).sum()} shared_IgA_IgG_plasma={len(shared)}")

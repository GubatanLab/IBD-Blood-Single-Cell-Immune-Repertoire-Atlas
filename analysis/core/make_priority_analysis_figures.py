from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

base=Path("C:/path/to/private-manuscript-workspace/High Impact Additional Analyses")
out=base/"Requested Priority Analyses Figures";out.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({"font.family":"Arial","font.size":8,"axes.titlesize":9,"axes.labelsize":8,"pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":0.7})
colors={"Control":"#6F6F6F","CD":"#0072B2","UC":"#D55E00","TCR":"#0072B2","BCR":"#D55E00"}

def panel(ax,label):ax.text(-0.28,1.10,label,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top",ha="left",clip_on=False)
def bhstar(q):return "***" if q<.001 else "**" if q<.01 else "*" if q<.05 else ""

# Figure S1: clone-state interactions
p=base/"Clone State Interactions";t=pd.read_csv(p/"Table_CSI3_formal_interaction_models.csv")
fig,axs=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
selT=["Naive_central_memory","Effector_cytotoxicity","Th1_Tc1_inflammatory","EOMES_ZEB2_inflammatory_CD8_TRM_like","Tissue_resident_mucosal_retention","Gut_homing_intestinal_trafficking"]
selB=["Resting_naive_B_cell","Resting_memory_B_cell","Plasmablast_plasma_differentiation","IgA_mucosal_plasma_cell","IgG_inflammatory_plasma_cell","B_cell_antigen_presentation","BAFF_APRIL_survival_response"]
def heat(ax,fam,mods,contrasts,title):
 d=t[(t.family==fam)&t.module.isin(mods)&t.contrast.isin(contrasts)].copy();pv=d.pivot(index="module",columns="contrast",values="effect").reindex(mods)[contrasts];q=d.pivot(index="module",columns="contrast",values="FDR").reindex(mods)[contrasts]
 im=ax.imshow(pv,cmap="RdBu_r",norm=TwoSlopeNorm(vcenter=0,vmin=-max(.05,np.nanmax(abs(pv.values))),vmax=max(.05,np.nanmax(abs(pv.values)))),aspect="auto")
 ax.set_xticks(range(len(contrasts)),[x.replace("_expansion","").replace("_"," ") for x in contrasts],rotation=30,ha="right");ax.set_yticks(range(len(mods)),[x.replace("_"," ") for x in mods])
 for i in range(len(mods)):
  for j in range(len(contrasts)):
   if np.isfinite(q.iloc[i,j]):ax.text(j,i,bhstar(q.iloc[i,j]),ha="center",va="center",fontweight="bold")
 ax.set_title(title);fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label="Expanded − singleton score")
heat(axs[0,0],"TCR_diagnosis",selT,["Control_expansion","CD_expansion","UC_expansion"],"T-cell clonal programs by diagnosis");panel(axs[0,0],"A")
heat(axs[0,1],"BCR_diagnosis",selB,["Control_expansion","CD_expansion","UC_expansion"],"B-cell clonal programs by diagnosis");panel(axs[0,1],"B")
ints=t[t.contrast.str.contains("interaction",na=False)].sort_values("p_value").head(16).copy();ints["label"]=ints.module.str.replace("_"," ")+" | "+ints.contrast.str.replace("_"," ")
axs[1,0].axvline(0,color="#777",lw=.7);y=np.arange(len(ints));axs[1,0].errorbar(ints.effect,y,xerr=1.96*ints.SE,fmt="o",ms=3,color="#333",ecolor="#999",lw=.8);axs[1,0].set_yticks(y,ints.label);axs[1,0].invert_yaxis();axs[1,0].set_xlabel("Interaction effect (95% CI)");axs[1,0].set_title("Formal diagnosis/inflammation interactions\n(no contrast passed FDR < 0.05)");panel(axs[1,0],"C")
se=pd.read_csv(p/"Table_CSI4_acquisition_series_effects.csv");keys=[("TCR","EOMES_ZEB2_inflammatory_CD8_TRM_like","CD","Inflamed"),("TCR","EOMES_ZEB2_inflammatory_CD8_TRM_like","UC","Inflamed"),("TCR","Effector_cytotoxicity","CD","Inflamed"),("BCR","B_cell_antigen_presentation","CD","Inflamed")]
for k,(mo,mod,dx,inf) in enumerate(keys):
 d=se[(se.modality==mo)&(se.module==mod)&(se.Diagnosis1==dx)&(se.Inflammation1==inf)];x=np.full(len(d),k)+np.linspace(-.12,.12,max(len(d),1));axs[1,1].scatter(x,d.effect,s=18,color=colors[mo],alpha=.8);axs[1,1].plot([k-.22,k+.22],[d.effect.mean(),d.effect.mean()],color="black",lw=1.3)
axs[1,1].axhline(0,color="#777",lw=.7);axs[1,1].set_xticks(range(len(keys)),["CD EOMES–ZEB2","UC EOMES–ZEB2","CD cytotoxicity","CD antigen presentation"],rotation=25,ha="right");axs[1,1].set_ylabel("Series-specific expanded − singleton score");axs[1,1].set_title("Replication across acquisition series");panel(axs[1,1],"D")
fig.savefig(out/"Figure_S_clone_state_interactions.pdf",bbox_inches="tight");fig.savefig(out/"Figure_S_clone_state_interactions.png",dpi=600,bbox_inches="tight");plt.close(fig)

# Figure S2: paired TCR sequence-state graph
p=base/"Paired TCR Sequence State";tt=pd.read_csv(p/"Table_TSS4_sequence_state_permutation_tests_by_series.csv");meta=pd.read_csv(p/"Table_TSS5_sequence_state_replication_meta_analysis.csv")
fig,axs=plt.subplots(1,3,figsize=(11,4.1),constrained_layout=True)
d=tt[(tt.stratum=="All")&(tt.module!="Dominant_state_concordance")];pv=d.pivot(index="module",columns="acquisition_series",values="observed_edge_correlation");pv=pv.reindex([x for x in ["Effector_cytotoxicity","Th1_Tc1_inflammatory","EOMES_ZEB2_inflammatory_CD8_TRM_like","Tissue_resident_mucosal_retention","Gut_homing_intestinal_trafficking"] if x in pv.index])
im=axs[0].imshow(pv,cmap="RdBu_r",norm=TwoSlopeNorm(vcenter=0,vmin=-max(.02,np.nanmax(abs(pv.values))),vmax=max(.02,np.nanmax(abs(pv.values)))),aspect="auto");axs[0].set_xticks(range(len(pv.columns)),pv.columns,rotation=45);axs[0].set_yticks(range(len(pv.index)),[x.replace("_"," ") for x in pv.index]);axs[0].set_title("Sequence-neighbor program concordance");fig.colorbar(im,ax=axs[0],fraction=.05,pad=.03,label="Edge correlation");panel(axs[0],"A")
mm=meta[(meta.module!="Dominant_state_concordance")].sort_values(["stratum","meta_z"]);strata=[x for x in ["All","CD","UC"] if x in mm.stratum.unique()];mods=list(pv.index);w=.23
for j,st in enumerate(strata):
 x=mm[mm.stratum==st].set_index("module").reindex(mods);axs[1].barh(np.arange(len(mods))+(j-1)*w,x.meta_z,height=w,label=st,color=["#555555",colors["CD"],colors["UC"]][j])
axs[1].axvline(0,color="#777",lw=.7);axs[1].set_yticks(range(len(mods)),[x.replace("_"," ") for x in mods]);axs[1].set_xlabel("Weighted Stouffer z across series");axs[1].set_title("Replicated sequence–state associations");axs[1].legend(frameon=False);panel(axs[1],"B")
ds=tt[(tt.module=="Dominant_state_concordance")&(tt.stratum=="All")].sort_values("acquisition_series");x=np.arange(len(ds));axs[2].scatter(x,ds.observed_edge_correlation,label="Observed",color="#333");axs[2].scatter(x,ds.null_mean,label="Permutation mean",facecolor="white",edgecolor="#777");axs[2].vlines(x,ds.null_mean,ds.observed_edge_correlation,color="#bbb",lw=.8);axs[2].set_xticks(x,ds.acquisition_series,rotation=45);axs[2].set_ylabel("Same dominant-state fraction");axs[2].set_title("Exact state-label concordance\nnot replicated after permutation");axs[2].legend(frameon=False);panel(axs[2],"C")
fig.savefig(out/"Figure_S_paired_TCR_sequence_state.pdf",bbox_inches="tight");fig.savefig(out/"Figure_S_paired_TCR_sequence_state.png",dpi=600,bbox_inches="tight");plt.close(fig)

# Figure S3: BCR germline-aware lineages
p=base/"BCR Germline Lineages";sm=pd.read_csv(p/"Table_BGL10_participant_region_SHM.csv");sl=pd.read_csv(p/"Table_BGL11_participant_lineage_metrics.csv");sens=pd.read_csv(p/"Table_BGL13_lineage_threshold_sensitivity_summary.csv");shared=pd.read_csv(p/"Table_BGL9_shared_IgA_IgG_plasma_state_lineages.csv")
fig,axs=plt.subplots(2,2,figsize=(9.5,7.5),constrained_layout=True)
regs=["fwr1","cdr1","fwr2","cdr2","fwr3","fwr4"];dxs=["Control","CD","UC"]
for j,dx in enumerate(dxs):
 vals=[sm[(sm.Diagnosis==dx)&(sm.region==r)].weighted_shm_rate.dropna().to_numpy() for r in regs];pos=np.arange(len(regs))+(j-1)*.23
 bp=axs[0,0].boxplot(vals,positions=pos,widths=.2,patch_artist=True,showfliers=False,medianprops={"color":"black","lw":.7},boxprops={"facecolor":colors[dx],"alpha":.65,"linewidth":.6},whiskerprops={"linewidth":.6},capprops={"linewidth":.6})
axs[0,0].set_xticks(range(len(regs)),[x.upper() for x in regs]);axs[0,0].set_ylabel("Germline-relative SHM rate");axs[0,0].set_title("Regional heavy-chain somatic hypermutation");axs[0,0].legend([mpl.patches.Patch(color=colors[x]) for x in dxs],dxs,frameon=False,ncol=3);panel(axs[0,0],"A")
vals=[sl[sl.Diagnosis==dx].mean_MST_branch_length.dropna() for dx in dxs];bp=axs[0,1].boxplot(vals,labels=dxs,patch_artist=True,showfliers=False);[b.set_facecolor(colors[dx]) for b,dx in zip(bp["boxes"],dxs)];axs[0,1].set_ylabel("Mean lineage MST branch length");axs[0,1].set_title("Participant-level lineage diversification");panel(axs[0,1],"B")
ax=axs[1,0];ax.plot(sens.threshold,sens.lineages,marker="o",color="#333",label="All lineages");ax.set_xlabel("Normalized junction-distance threshold");ax.set_ylabel("Lineages",color="#333");ax2=ax.twinx();ax2.plot(sens.threshold,sens.expanded_lineages,marker="s",color=colors["UC"],label="Expanded lineages");ax2.set_ylabel("Expanded lineages",color=colors["UC"]);ax.set_title("Lineage-definition sensitivity");panel(ax,"C")
cnt=shared.groupby("Diagnosis").lineage_id.nunique().reindex(dxs,fill_value=0);axs[1,1].bar(dxs,cnt.values,color=[colors[x] for x in dxs]);axs[1,1].set_ylabel("Shared IgA- and IgG-plasma-state lineages");axs[1,1].set_title("Cross-state lineage occupancy is rare");panel(axs[1,1],"D")
fig.savefig(out/"Figure_S_BCR_germline_lineages.pdf",bbox_inches="tight");fig.savefig(out/"Figure_S_BCR_germline_lineages.png",dpi=600,bbox_inches="tight");plt.close(fig)

(out/"Figure_legends.txt").write_text("""Figure S1. Disease-specific clone–state interactions. (A,B) Participant-paired, cell-state-matched expanded-minus-singleton module-score effects by diagnosis for TCR and BCR compartments. Asterisks denote BH-adjusted FDR: *<0.05, **<0.01, ***<0.001. (C) Formal expansion-by-diagnosis or expansion-by-inflammation interaction estimates with 95% confidence intervals; no interaction contrast passed FDR <0.05. (D) Acquisition-series effects for selected programs.\n\nFigure S2. Paired alpha-beta TCR sequence neighborhoods align with cytotoxic inflammatory programs. (A) Correlation of participant-centered module scores across cross-participant paired-TCR sequence-neighbor edges in each acquisition series. Exact paired clonotypes were excluded. (B) Weighted Stouffer meta-analysis across acquisition series. (C) Dominant cell-state-label concordance versus participant-stratified permutation expectations. HLA genotype was unavailable; results are exploratory and V/J-aware, not HLA-stratified.\n\nFigure S3. Germline-aware BCR lineage reconstruction. (A) Participant-level SHM rates calculated from IgBLAST/IMGT observed-germline alignments in framework and CDR regions. (B) Mean minimum-spanning-tree branch length across participant-specific lineages. (C) Stability of lineage counts across normalized junction-distance thresholds. (D) Lineages containing both IgA-plasma and IgG-plasma transcriptional states. Full heavy-variable sequences were not cell linked, preventing assignment of these states to terminal phylogenetic branches. Descriptive replacement/silent counts are provided in the tables; BASELINe selection inference was not performed.""",encoding="utf-8")
print(out)

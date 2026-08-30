from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

ROOT=Path(__file__).resolve().parent
s=pd.read_csv(ROOT/"Table_S_primary_model_validation_summary.csv")
n=pd.read_csv(ROOT/"Table_S_permutation_null_distributions.csv")
p=pd.read_csv(ROOT/"Table_S_nested_outer_fold_predictions.csv")
o=pd.read_csv(ROOT/"Table_S_model_selection_optimism.csv")
COL={"pbmc":"#2878B5","tcr":"#D6814E","bcr":"#4C9B78"}
LAB={"pbmc":"PBMC composition","tcr":"TCR","bcr":"BCR"}
TASK={"cd_vs_control":"CD vs control","uc_vs_control":"UC vs control","cd_vs_uc":"CD vs UC"}
s["label"]=s.modality.map(LAB)+"\n"+s.task.map(TASK)
s["yerr_low"]=s.roc_auc_median-s.roc_auc_ci_low;s["yerr_high"]=s.roc_auc_ci_high-s.roc_auc_median
sns.set_theme(style="whitegrid",font_scale=.9);plt.rcParams.update({"font.family":"Arial","pdf.fonttype":42})

fig,axs=plt.subplots(1,2,figsize=(12.5,5.2),constrained_layout=True)
ax=axs[0]
for j,r in s.reset_index(drop=True).iterrows():
    ax.errorbar(j,r.roc_auc_median,yerr=[[r.yerr_low],[r.yerr_high]],fmt='o',ms=7,capsize=3,color=COL[r.modality])
ax.axhline(.5,color="#888",ls="--",lw=1);ax.set_ylim(.4,1.03);ax.set_xticks(range(len(s)),s.label,rotation=38,ha="right");ax.set_ylabel("Median outer-fold ROC AUC (95% CI)");ax.set_title("A  Fully nested participant-level validation",loc="left",fontweight="bold")
ax=axs[1]
order=[f"{m}|{t}" for m in ["pbmc","tcr","bcr"] for t in ["cd_vs_control","uc_vs_control","cd_vs_uc"]]
n["key"]=n.modality+"|"+n.task
s["key"]=s.modality+"|"+s.task
sns.violinplot(data=n,x="key",y="median_outer_auc",order=order,color="#D8DDE2",inner=None,cut=0,ax=ax)
for j,k in enumerate(order):
    r=s[s.key.eq(k)].iloc[0];ax.scatter(j,r.roc_auc_median,color=COL[r.modality],s=45,zorder=3)
ax.axhline(.5,color="#888",ls="--",lw=1);ax.set_ylim(.25,1.03);ax.set_xticks(range(len(order)),[LAB[k.split('|')[0]]+"\n"+TASK[k.split('|')[1]] for k in order],rotation=38,ha="right");ax.set_xlabel("");ax.set_ylabel("ROC AUC");ax.set_title("B  Permutation-derived null distributions",loc="left",fontweight="bold")
fig.savefig(ROOT/"Figure_S_ML_nested_and_permutation.pdf",bbox_inches="tight");fig.savefig(ROOT/"Figure_S_ML_nested_and_permutation.png",dpi=400,bbox_inches="tight");plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11.8,5),constrained_layout=True)
ax=axs[0]
q=p.groupby(["modality","task","participant","truth"],as_index=False).probability.mean()
for mod in ["pbmc","tcr","bcr"]:
    z=q[(q.modality==mod)&(q.task=="cd_vs_control")]; frac,mean=calibration_curve(z.truth,z.probability,n_bins=6,strategy="quantile");ax.plot(mean,frac,marker="o",label=LAB[mod],color=COL[mod])
ax.plot([0,1],[0,1],ls="--",color="#888");ax.set(xlabel="Mean predicted probability",ylabel="Observed event fraction",title="A  Calibration: CD versus control");ax.legend(frameon=False)
ax=axs[1]; oo=o.copy();oo["label"]=oo.modality.map(LAB)+"\n"+oo.task.map(TASK);colors=[COL[x] for x in oo.modality];ax.bar(range(len(oo)),oo.apparent_optimism,color=colors);ax.axhline(0,color="#777",lw=1);ax.set_xticks(range(len(oo)),oo.label,rotation=35,ha="right");ax.set_ylabel("Best historical AUC − nested primary-model AUC");ax.set_title("B  Apparent model-selection optimism")
fig.savefig(ROOT/"Figure_S_ML_calibration_and_optimism.pdf",bbox_inches="tight");fig.savefig(ROOT/"Figure_S_ML_calibration_and_optimism.png",dpi=400,bbox_inches="tight");plt.close(fig)

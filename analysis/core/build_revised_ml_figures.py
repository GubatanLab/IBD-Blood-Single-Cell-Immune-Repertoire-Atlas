from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT=Path(r"C:/path/to/private-manuscript-workspace\Revised ML Figures")
OUT.mkdir(parents=True,exist_ok=True)
BLUE="#2878B5"; ORANGE="#E07A3F"; GREEN="#3A9D77"; PURPLE="#7A65A8"; GRAY="#68737D"; RED="#C44E52"
sns.set_theme(style="whitegrid",font_scale=.85)
plt.rcParams.update({"font.family":"Arial","axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})

def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",dpi=400,bbox_inches="tight")
    plt.close(fig)

def letter(ax,s): ax.text(-.14,1.08,s,transform=ax.transAxes,fontsize=13,fontweight="bold",va="top")
def chance(ax): ax.axhline(.5,color="#999999",lw=1,ls="--",zorder=0); ax.set_ylim(.4,1.03)
def clean_task(x):
    direct={"cd_vs_control":"CD vs control","uc_vs_control":"UC vs control","cd_vs_uc":"CD vs UC","cd_inflamed_vs_noninflamed":"CD inflammation","uc_inflamed_vs_noninflamed":"UC inflammation","therapy_combined_response":"All biologics","therapy_antitnf_response":"Anti-TNF","therapy_ustekinumab_response":"Ustekinumab","therapy_vedolizumab_response":"Vedolizumab"}
    if x in direct:return direct[x]
    y=str(x).lower()
    if "anti_tnf" in y:return "Anti-TNF"
    if "combined" in y:return "All biologics"
    if "ustek" in y:return "Ustekinumab"
    if "vedol" in y:return "Vedolizumab"
    return str(x).replace("_"," ")

def parse_old_result(path):
    txt=Path(path).read_text(encoding="utf-8")
    best=float(re.search(r"balanced accuracy of ([0-9.]+)",txt).group(1))
    obj=json.loads(txt[txt.index('{'):])
    perf=[]
    for k,v in obj.items():
        m=v.get("average_metric_scores",{})
        if m: perf.append((k,m.get("Balanced Accuracy",np.nan),m.get("Matthews Correlation",np.nan),m.get("F1",np.nan)))
    feat=obj.get("rf",obj.get("logreg",{})).get("average_feature_importance",{})
    return best,pd.DataFrame(perf,columns=["model","BA","MCC","F1"]),pd.Series(feat).sort_values(ascending=False)

def figure2():
    root=Path(r"C:/path/to/private-user-home\OneDrive\Desktop\ML scRNA IBD PBMC Freq")
    tasks={"CD vs control":"results_CD_vs_control.txt","UC vs control":"results_UC_vs_control.txt","CD inflammation":"results_CD_inflamed_vs_noninflamed.txt","UC inflammation":"results_UC_inflamed_vs_noninflamed.txt"}
    parsed={k:parse_old_result(root/v) for k,v in tasks.items()}
    diag=[]
    for t in ["CD vs control","UC vs control"]:
        _,d,_=parsed[t]; d=d.nlargest(4,"BA"); d["task"]=t; diag.append(d)
    diag=pd.concat(diag)
    infl=pd.DataFrame({"task":["CD inflammation","UC inflammation"],"BA":[parsed["CD inflammation"][0],parsed["UC inflammation"][0]]})
    anti=pd.read_csv(r"C:/path/to/private-user-home\OneDrive\Desktop\DDW2026\DDW 2026 Oral Presentation\autogluon_antitnf_nonresponder_celltype_only\leaderboard_balacc_mcc_f1_celltype_only.csv")
    therapy=pd.DataFrame({"task":["Anti-TNF"],"BA":[anti.balanced_accuracy.max()]})
    fig,axs=plt.subplots(2,3,figsize=(13.2,7.7),constrained_layout=True)
    ax=axs[0,0]; sns.barplot(diag,x="task",y="BA",hue="model",ax=ax,palette=[BLUE,ORANGE,GREEN,PURPLE]); chance(ax); ax.set(title="Diagnosis classification",xlabel="",ylabel="Balanced accuracy"); ax.tick_params(axis="x",rotation=18); ax.legend(title="Model",fontsize=7,title_fontsize=8,loc="lower left"); letter(ax,"A")
    ax=axs[0,1]; f=pd.concat([parsed[t][2].head(6).rename(t) for t in ["CD vs control","UC vs control"]],axis=1).fillna(0); f=f.loc[f.max(axis=1).sort_values().tail(10).index]; sns.heatmap(f,cmap="Blues",ax=ax,cbar_kws={"label":"RF importance"}); ax.set(title="Diagnosis-associated cell states",xlabel="",ylabel=""); letter(ax,"B")
    ax=axs[0,2]; vals=pd.DataFrame({"domain":["Diagnosis","Inflammation","6-month response status"],"BA":[diag.groupby('task').BA.max().mean(),infl.BA.mean(),therapy.BA.mean()]}); sns.barplot(vals,x="domain",y="BA",color=BLUE,ax=ax); chance(ax); ax.set(title="Performance across clinical domains",xlabel="",ylabel="Mean balanced accuracy"); ax.tick_params(axis="x",rotation=22); letter(ax,"C")
    ax=axs[1,0]; sns.barplot(infl,x="task",y="BA",color=ORANGE,ax=ax); chance(ax); ax.set(title="Tissue-inflammation classification",xlabel="",ylabel="Balanced accuracy"); ax.tick_params(axis="x",rotation=18); letter(ax,"D")
    ax=axs[1,1]; sns.barplot(therapy,x="task",y="BA",color=GREEN,ax=ax); chance(ax); ax.set(title="Six-month response-status classification",xlabel="",ylabel="Balanced accuracy"); ax.tick_params(axis="x",rotation=22); ax.text(.02,.02,"Post-treatment associations; exploratory",transform=ax.transAxes,fontsize=8,color=GRAY); letter(ax,"E")
    ax=axs[1,2]; feats=pd.concat([parsed[t][2].head(5).rename(t) for t in ["CD inflammation","UC inflammation"]],axis=1).fillna(0); feats=feats.loc[feats.max(axis=1).sort_values().tail(9).index]; sns.heatmap(feats,cmap="Oranges",ax=ax,cbar_kws={"label":"RF importance"}); ax.set(title="Inflammation-associated cell states",xlabel="",ylabel=""); letter(ax,"F")
    save(fig,"Figure_2_revised_ML")
    pd.concat({k:v[1] for k,v in parsed.items()}).to_csv(OUT/"Figure_2_source_data.csv")

def load_tcr():
    base=Path(r"C:/path/to/private-immuneml-results\TCR Results")
    d=pd.read_csv(base/r"advanced_immuneml_models\diagnosis_with_svm\tcr_diagnosis_all_models_summary_with_svm_plotted_values.csv")
    i=pd.read_csv(base/r"advanced_immuneml_models\inflammation_with_svm\tcr_inflammation_all_models_summary_with_svm_plotted_values.csv")
    parts=[]
    for sub in [r"therapy_response_results\sklearn_results_aa\aggregate_scores.csv",r"therapy_response_results\sklearn_results_nt\aggregate_scores.csv"]:
        x=pd.read_csv(base/sub); x=x.rename(columns={"TherapyResponse1_roc_auc_mean":"roc_auc_mean","TherapyResponse1_roc_auc_std":"roc_auc_std"}); x["model_family"]="cdr3_kmer"; parts.append(x)
    t=pd.concat(parts,ignore_index=True)
    shap=[]
    for sub in [r"shap_top_models\top_model_shap_importance_top_features.csv",r"inflammation_results\shap_top_models\top_model_shap_importance_top_features.csv",r"therapy_response_results\shap_top_models\top_model_shap_importance_top_features.csv"]:
        p=base/sub
        if p.exists(): shap.append(pd.read_csv(p))
    return d,i,t,pd.concat(shap,ignore_index=True)

def repertoire_figure(kind):
    if kind=="TCR":
        d,i,t,sh=load_tcr(); chain_col="chain_group"; metric="roc_auc_mean"
    else:
        base=Path(r"C:/path/to/private-immuneml-results\BCR Results\tables")
        d=pd.read_csv(base/"bcr_diagnosis_lr_svm_feature_model_summary_aggregate.csv").rename(columns={"Diagnosis1_roc_auc_mean":"roc_auc_mean","Diagnosis1_roc_auc_std":"roc_auc_std"})
        i=pd.read_csv(base/"bcr_inflammation_lr_svm_feature_model_summary_aggregate.csv").rename(columns={"Inflammation1_roc_auc_mean":"roc_auc_mean","Inflammation1_roc_auc_std":"roc_auc_std"})
        t=pd.read_csv(base/"bcr_therapy_response_lr_svm_feature_model_summary_aggregate.csv").rename(columns={"TherapyResponse_roc_auc_mean":"roc_auc_mean","TherapyResponse_roc_auc_std":"roc_auc_std"})
        for x in (d,i,t): x["model_family"]=np.where(x.feature_set.astype(str).str.contains("repertoire_metrics|diversity",regex=True),"diversity_or_combined","cdr3_kmer")
        chain_col="chain_group"; metric="roc_auc_mean"
        sh=pd.DataFrame(columns=["comparison","feature","mean_abs_shap"])
        for p in Path(r"C:/path/to/private-immuneml-results\BCR Results").rglob("*shap*top*features*.csv"):
            try:
                z=pd.read_csv(p)
                if {"feature","mean_abs_shap"}.issubset(z.columns): sh=pd.concat([sh,z],ignore_index=True)
            except: pass
    def top(df,n=1): return df.sort_values(metric).groupby("comparison",as_index=False).tail(n)
    fig,axs=plt.subplots(2,3,figsize=(13.2,7.7),constrained_layout=True)
    ax=axs[0,0]; ax.axis("off"); nodes=[("Participant-level\nrepertoires",.12),("CDR3 / repertoire\nfeatures",.50),("LR · SVM · RF"+(" · DeepRC" if kind=="TCR" else ""),.86)];
    for txt,x in nodes: ax.text(x,.55,txt,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",fc="#EAF2F8",ec=BLUE),transform=ax.transAxes)
    for x in [.26,.64]: ax.annotate("",xy=(x+.1,.55),xytext=(x,.55),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",color=GRAY,lw=1.5))
    ax.text(.5,.18,"Participant-level splits → held-out ROC AUC",ha="center",transform=ax.transAxes,color=GRAY); ax.set_title(f"{kind} modeling workflow"); letter(ax,"A")
    ax=axs[0,1]; z=top(d); z=z.assign(task=z.comparison.map(clean_task),chain=z[chain_col].str.replace(('tcr_' if kind=='TCR' else 'bcr_'),'').str.replace('_',' ')); sns.barplot(z,x="task",y=metric,hue="chain",ax=ax); chance(ax); ax.set(title="Diagnosis: best model by task",xlabel="",ylabel="ROC AUC"); ax.tick_params(axis="x",rotation=20); ax.legend(title="Receptor",fontsize=7,title_fontsize=8); letter(ax,"B")
    ax=axs[0,2]; q=d.copy(); q["class"]=np.where(q.model_family.astype(str).str.contains("diversity"),"Diversity/clonality","CDR3 sequence"); q=top(q.groupby(["comparison","class"],as_index=False)[metric].max(),1); sns.barplot(q,x="comparison",y=metric,hue="class",ax=ax,palette=[ORANGE,BLUE]); chance(ax); ax.set(title="Feature-class ablation",xlabel="",ylabel="ROC AUC"); ax.set_xticklabels([clean_task(x.get_text()) for x in ax.get_xticklabels()],rotation=20); ax.legend(title="Feature class",fontsize=7); letter(ax,"C")
    ax=axs[1,0]
    if kind=="TCR":
        s=sh[sh.comparison.astype(str).str.contains("cd_vs_control|uc_vs_control",regex=True)] if len(sh) else sh
        if len(s): s=s.groupby("feature",as_index=False).mean(numeric_only=True).nlargest(10,"mean_abs_shap").sort_values("mean_abs_shap"); ax.barh(s.feature,s.mean_abs_shap,color=PURPLE)
        ax.set(title="Top diagnosis model features",xlabel="Mean |SHAP value|",ylabel="")
    else:
        paired=d[d[chain_col].eq("bcr_heavy_light")].groupby("comparison")[metric].max(); single=d[d[chain_col].isin(["bcr_heavy","bcr_light"])].groupby("comparison")[metric].max(); inc=(paired-single).dropna(); ax.bar(inc.index.map(clean_task),inc.values,color=PURPLE); ax.axhline(0,color=GRAY,lw=1); ax.set(title="Incremental value of paired heavy-light features",xlabel="",ylabel="Δ ROC AUC vs best single chain"); ax.tick_params(axis="x",rotation=20)
    letter(ax,"D")
    ax=axs[1,1]; zz=pd.concat([top(i).assign(domain="Inflammation"),top(t).assign(domain="6-month response status")]); zz=zz.assign(task=zz.comparison.map(clean_task)); sns.barplot(zz,x="task",y=metric,hue="domain",ax=ax,palette=[ORANGE,GREEN]); chance(ax); ax.set(title="Clinical-state associations",xlabel="",ylabel="ROC AUC"); ax.tick_params(axis="x",rotation=25); ax.legend(title="Domain",fontsize=7); letter(ax,"E")
    ax=axs[1,2]; allx=pd.concat([d.assign(domain="Diagnosis"),i.assign(domain="Inflammation"),t.assign(domain="Response status")]); mat=allx.groupby([chain_col,"domain"])[metric].max().unstack().fillna(.5); mat.index=mat.index.str.replace(('tcr_' if kind=='TCR' else 'bcr_'),'').str.replace('_',' '); sns.heatmap(mat,cmap="YlGnBu",vmin=.5,vmax=1,annot=True,fmt=".2f",ax=ax,cbar_kws={"label":"Best ROC AUC"}); ax.set(title="Receptor representation across domains",xlabel="",ylabel=""); letter(ax,"F")
    save(fig,f"Figure_{4 if kind=='TCR' else 6}_revised_ML")
    pd.concat([d.assign(section="diagnosis"),i.assign(section="inflammation"),t.assign(section="response_status")],ignore_index=True).to_csv(OUT/f"Figure_{4 if kind=='TCR' else 6}_source_data.csv",index=False)

if __name__=="__main__":
    figure2(); repertoire_figure("TCR"); repertoire_figure("BCR")

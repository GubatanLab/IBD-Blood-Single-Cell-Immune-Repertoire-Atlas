from pathlib import Path
from collections import Counter
import warnings, json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, accuracy_score,
    brier_score_loss, confusion_matrix)
import statsmodels.api as sm

warnings.filterwarnings("ignore")
SEED=20260710
OUT=Path(r"C:/path/to/private-manuscript-workspace\ML Sensitivity Validation")
OUT.mkdir(parents=True,exist_ok=True)
PBMC=Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 2 MiloR\ML Celltype\pbmc2026_scvi_l2_freq_by_sample.csv")
AIRR=Path(r"C:/path/to/private-immuneml-results\immuneML_runs\airr_input")
TASKS=[("cd_vs_control","CD","Control"),("uc_vs_control","UC","Control"),("cd_vs_uc","CD","UC")]

def kmers(s,k=4):
    s=str(s)
    return [s[i:i+k] for i in range(max(0,len(s)-k+1)) if "*" not in s[i:i+k] and "X" not in s[i:i+k]]

def read_repertoire(modality,k=4):
    md=pd.read_csv(AIRR/modality/"metadata_all.csv")
    rdir=AIRR/modality/"repertoires"
    rows=[]
    for _,r in md.iterrows():
        p=rdir/r.filename
        if not p.exists(): continue
        z=pd.read_csv(p,sep="\t",usecols=lambda c:c in {"junction_aa","locus","duplicate_count","productive"})
        z=z[z.productive.astype(str).str.upper().isin(["T","TRUE","1"])]
        if modality=="tcr": z=z[z.locus.isin(["TRA","TRB"])]
        else: z=z[z.locus.eq("IGH")]
        c=Counter()
        for _,q in z.iterrows():
            wt=float(q.get("duplicate_count",1) or 1)
            for token in kmers(q.junction_aa,k): c[token]+=wt
        tot=sum(c.values()) or 1
        rows.append({"SampleID":r.SampleID,"Diagnosis1":r.Diagnosis1,"Batch":r.Batch,"features":{x:v/tot for x,v in c.items()}})
    return pd.DataFrame(rows)

def pbmc_data():
    d=pd.read_csv(PBMC)
    feats=[c for c in d if c.startswith("L2_")]
    return d,feats

def model_for(modality,n_features):
    if modality=="pbmc":
        pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("select",SelectKBest(f_classif)),("scale",StandardScaler()),("clf",LogisticRegression(max_iter=5000,class_weight="balanced",solver="liblinear",random_state=SEED))])
        grid={"select__k":[min(25,n_features),"all"],"clf__C":[.1,1,10]}
    else:
        pipe=Pipeline([("vec",DictVectorizer(sparse=True)),("select",SelectPercentile(chi2)),("scale",StandardScaler(with_mean=False)),("clf",LogisticRegression(max_iter=5000,class_weight="balanced",solver="liblinear",random_state=SEED))])
        grid={"select__percentile":[25,100],"clf__C":[.1,1,10]}
    return pipe,grid

def metrics(y,p,threshold=.5):
    yh=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    return {"roc_auc":roc_auc_score(y,p),"balanced_accuracy":balanced_accuracy_score(y,yh),"accuracy":accuracy_score(y,yh),"brier":brier_score_loss(y,p),"sensitivity":tp/(tp+fn) if tp+fn else np.nan,"specificity":tn/(tn+fp) if tn+fp else np.nan}

def calibration(y,p):
    x=logit(np.clip(p,1e-6,1-1e-6)); fit=sm.GLM(y,sm.add_constant(x),family=sm.families.Binomial()).fit()
    return float(fit.params[0]),float(fit.params[1])

def bootstrap_ci(vals,B=4000):
    vals=np.asarray(vals,float); rng=np.random.default_rng(SEED); meds=[np.median(rng.choice(vals,len(vals),replace=True)) for _ in range(B)]
    return np.median(vals),np.quantile(meds,.025),np.quantile(meds,.975)

def nested_run(modality,task,pos,neg,X,y,ids,n_features):
    outer=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=SEED)
    pipe,grid=model_for(modality,n_features); folds=[]; preds=[]
    for j,(tr,te) in enumerate(outer.split(np.zeros(len(y)),y),1):
        inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED+j)
        gs=GridSearchCV(pipe,grid,scoring="roc_auc",cv=inner,n_jobs=-1,refit=True)
        gs.fit(X[tr] if modality=="pbmc" else [X[q] for q in tr],y[tr])
        pp=gs.predict_proba(X[te] if modality=="pbmc" else [X[q] for q in te])[:,1]
        m=metrics(y[te],pp); m.update({"modality":modality,"task":task,"outer_fold":j,"n_test":len(te),"best_params":json.dumps(gs.best_params_,sort_keys=True)})
        folds.append(m)
        preds.extend({"modality":modality,"task":task,"outer_fold":j,"participant":ids[q],"truth":y[q],"probability":v} for q,v in zip(te,pp))
    fd=pd.DataFrame(folds); pr=pd.DataFrame(preds)
    pooled=pr.groupby(["modality","task","participant","truth"],as_index=False).probability.mean()
    pm=metrics(pooled.truth.values,pooled.probability.values); ci0,ci1=calibration(pooled.truth.values,pooled.probability.values); pm.update({"calibration_intercept":ci0,"calibration_slope":ci1})
    summary={"modality":modality,"task":task,"positive_class":pos,"negative_class":neg,"n":len(y),"n_positive":int(y.sum()),"n_negative":int((1-y).sum())}
    for met in ["roc_auc","balanced_accuracy","brier","sensitivity","specificity"]:
        med,lo,hi=bootstrap_ci(fd[met]); summary.update({f"{met}_median":med,f"{met}_ci_low":lo,f"{met}_ci_high":hi})
    summary.update({f"pooled_{k}":v for k,v in pm.items()})
    return fd,pr,pd.DataFrame([summary])

def permutation_null(modality,task,X,y,n_features,n_perm=25):
    rng=np.random.default_rng(SEED); pipe,_=model_for(modality,n_features); outer=StratifiedKFold(5,shuffle=True,random_state=SEED); out=[]
    if modality=="pbmc": pipe.set_params(select__k=min(25,n_features),clf__C=1)
    else: pipe.set_params(select__percentile=25,clf__C=1)
    for b in range(n_perm):
        yp=rng.permutation(y); aucs=[]
        for j,(tr,te) in enumerate(outer.split(np.zeros(len(yp)),yp)):
            fit=clone(pipe).fit(X[tr] if modality=="pbmc" else [X[q] for q in tr],yp[tr])
            pp=fit.predict_proba(X[te] if modality=="pbmc" else [X[q] for q in te])[:,1]
            aucs.append(roc_auc_score(yp[te],pp))
        out.append({"modality":modality,"task":task,"permutation":b+1,"median_outer_auc":np.median(aucs),"mean_outer_auc":np.mean(aucs)})
    return pd.DataFrame(out)

def audit_batch(md):
    tab=pd.crosstab(md.Batch,md.Diagnosis1)
    rows=[]
    for b,r in tab.iterrows():
        present=(r>0).sum(); rows.append({"batch":b,**{f"n_{c}":int(r.get(c,0)) for c in ["Control","CD","UC"]},"n_diagnoses_present":int(present),"eligible_as_multiclass_site_test":present==3})
    return pd.DataFrame(rows)

def selection_optimism(nested_summary):
    rows=[]
    sources={
      "tcr":Path(r"C:/path/to/private-immuneml-results\TCR Results\advanced_immuneml_models\diagnosis_with_svm\tcr_diagnosis_all_models_summary_with_svm_plotted_values.csv"),
      "bcr":Path(r"C:/path/to/private-immuneml-results\BCR Results\tables\bcr_diagnosis_lr_svm_feature_model_summary_aggregate.csv")}
    for mod,p in sources.items():
        d=pd.read_csv(p); auc="roc_auc_mean" if "roc_auc_mean" in d else "Diagnosis1_roc_auc_mean"
        for task,g in d.groupby("comparison"):
            s=nested_summary.query("modality==@mod and task==@task")
            if s.empty: continue
            best=float(g[auc].max()); nested=float(s.iloc[0].pooled_roc_auc)
            rows.append({"modality":mod,"task":task,"candidate_models_evaluated":len(g),"reported_best_candidate_auc":best,"nested_primary_model_pooled_auc":nested,"apparent_optimism":best-nested})
    return pd.DataFrame(rows)

def main():
    pb,feat=pbmc_data(); tcr=read_repertoire("tcr",4); bcr=read_repertoire("bcr",4)
    audit_batch(pd.read_csv(AIRR/"tcr"/"metadata_all.csv")).to_csv(OUT/"Table_S_batch_confounding_audit.csv",index=False)
    allfold=[]; allpred=[]; allsum=[]; allnull=[]
    for modality,d in [("pbmc",pb),("tcr",tcr),("bcr",bcr)]:
        for task,pos,neg in TASKS:
            print(f"Running {modality}: {task}",flush=True)
            q=d[d.Diagnosis1.isin([pos,neg])].copy(); y=(q.Diagnosis1==pos).astype(int).to_numpy(); ids=q.SampleID.astype(str).to_numpy()
            X=q[feat].to_numpy(float) if modality=="pbmc" else q.features.tolist(); nf=len(feat) if modality=="pbmc" else 100
            fd,pr,su=nested_run(modality,task,pos,neg,X,y,ids,nf); allfold.append(fd); allpred.append(pr); allsum.append(su)
            allnull.append(permutation_null(modality,task,X,y,nf))
            print(f"Completed {modality}: {task}",flush=True)
    folds=pd.concat(allfold,ignore_index=True); preds=pd.concat(allpred,ignore_index=True); summary=pd.concat(allsum,ignore_index=True); null=pd.concat(allnull,ignore_index=True)
    folds.to_csv(OUT/"Table_S_nested_outer_fold_metrics.csv",index=False); preds.to_csv(OUT/"Table_S_nested_outer_fold_predictions.csv",index=False); summary.to_csv(OUT/"Table_S_primary_model_validation_summary.csv",index=False); null.to_csv(OUT/"Table_S_permutation_null_distributions.csv",index=False)
    opt=selection_optimism(summary); opt.to_csv(OUT/"Table_S_model_selection_optimism.csv",index=False)
    rows=[]
    for _,s in summary.iterrows():
        n=null[(null.modality==s.modality)&(null.task==s.task)].median_outer_auc
        rows.append({"modality":s.modality,"task":s.task,"observed_median_outer_auc":s.roc_auc_median,"null_median":n.median(),"null_95th_percentile":n.quantile(.95),"permutation_p":(1+(n>=s.roc_auc_median).sum())/(1+len(n))})
    pd.DataFrame(rows).to_csv(OUT/"Table_S_permutation_test_summary.csv",index=False)
    (OUT/"analysis_manifest.json").write_text(json.dumps({"seed":SEED,"outer_resampling":"5-fold stratified CV repeated 3 times","inner_resampling":"3-fold stratified CV","permutations":25,"permutation_pipeline":"Prespecified fixed pipeline: C=1; 25 PBMC features or 25% of receptor k-mers; five-fold participant-level CV","primary_models":{"PBMC":"L2 cell-frequency logistic regression","TCR":"TRA+TRB amino-acid CDR3 4-mer logistic regression","BCR":"IGH amino-acid CDR3 4-mer logistic regression"},"site_validation":"Not performed: no site variable; Batch is diagnosis-confounded."},indent=2))

if __name__=="__main__": main()

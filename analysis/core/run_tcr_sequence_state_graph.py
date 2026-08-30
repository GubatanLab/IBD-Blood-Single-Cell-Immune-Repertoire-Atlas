#!/usr/bin/env python
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

SEED=20260824
N_PERM=5000
K_CAND=40
K_KEEP=5
MAX_DIST=0.50
rng=np.random.default_rng(SEED)
root=Path(r"C:/path/to/private-manuscript-workspace\High Impact Additional Analyses\Paired TCR Sequence State")
inp=root/"Table_TSS1_paired_alpha_beta_clone_sequence_state.csv"
df=pd.read_csv(inp)
modules=["Effector_cytotoxicity","Th1_Tc1_inflammatory","EOMES_ZEB2_inflammatory_CD8_TRM_like","Tissue_resident_mucosal_retention","Gut_homing_intestinal_trafficking"]

def toks(row):
    out=[]
    for prefix,seq in (("A",str(row.alpha_cdr3)),("B",str(row.beta_cdr3))):
        for k in (2,3):
            out.extend([f"{prefix}{k}_{seq[i:i+k]}" for i in range(max(0,len(seq)-k+1))])
    for label,val in (("AV",row.alpha_v),("AJ",row.alpha_j),("BV",row.beta_v),("BJ",row.beta_j)):
        out.extend([f"{label}_{val}"]*2)
    return out

def make_graph(d):
    if len(d)<20 or d.SampleID.nunique()<2: return pd.DataFrame()
    vec=TfidfVectorizer(analyzer=toks,lowercase=False,norm="l2",min_df=2,dtype=np.float32)
    X=vec.fit_transform(d.itertuples(index=False))
    k=min(K_CAND+1,len(d))
    nn=NearestNeighbors(n_neighbors=k,metric="cosine",algorithm="brute",n_jobs=-1).fit(X)
    dist,idx=nn.kneighbors(X,return_distance=True)
    exact=(d.alpha_v.astype(str)+"|"+d.alpha_j.astype(str)+"|"+d.alpha_cdr3.astype(str)+"|"+
           d.beta_v.astype(str)+"|"+d.beta_j.astype(str)+"|"+d.beta_cdr3.astype(str)).to_numpy()
    pats=d.SampleID.astype(str).to_numpy(); rows=[]
    for i in range(len(d)):
        kept=0
        for ds,j in zip(dist[i,1:],idx[i,1:]):
            if pats[i]==pats[j] or exact[i]==exact[j] or ds>MAX_DIST: continue
            a,b=(i,int(j)) if i<int(j) else (int(j),i)
            rows.append((a,b,float(ds))); kept+=1
            if kept>=K_KEEP: break
    if not rows:return pd.DataFrame()
    e=pd.DataFrame(rows,columns=["i","j","sequence_distance"]).drop_duplicates(["i","j"])
    return e

def within_participant_z(d,col):
    x=d[col].astype(float)
    mu=x.groupby(d.SampleID).transform("mean")
    sd=x.groupby(d.SampleID).transform("std").replace(0,np.nan)
    return ((x-mu)/sd).fillna(0).to_numpy()

def stat_corr(x,e):
    a=x[e.i.to_numpy()];b=x[e.j.to_numpy()]
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<20 or np.std(a[ok])==0 or np.std(b[ok])==0:return np.nan
    return float(np.corrcoef(a[ok],b[ok])[0,1])

def perm_indices(d):
    groups=[np.asarray(v,dtype=int) for v in d.groupby("SampleID",sort=False).indices.values()]
    p=np.arange(len(d))
    for g in groups:p[g]=rng.permutation(g)
    return p

def test_graph(d,e,series,stratum):
    rows=[]
    if len(e)<100:return rows
    for mod in modules:
        x=within_participant_z(d,mod);obs=stat_corr(x,e)
        null=np.array([stat_corr(x[perm_indices(d)],e) for _ in range(N_PERM)])
        null=null[np.isfinite(null)]
        p=(1+np.sum(np.abs(null)>=abs(obs)))/(1+len(null)) if np.isfinite(obs) and len(null) else np.nan
        rows.append(dict(acquisition_series=series,stratum=stratum,module=mod,n_nodes=len(d),n_participants=d.SampleID.nunique(),n_edges=len(e),observed_edge_correlation=obs,null_mean=float(np.mean(null)),null_sd=float(np.std(null,ddof=1)),empirical_p=p,n_permutations=len(null)))
    # Dominant-state concordance is an interpretable secondary endpoint.
    labels=d.dominant_state.astype(str).to_numpy();obs=float(np.mean(labels[e.i.to_numpy()]==labels[e.j.to_numpy()]))
    null=[]
    for _ in range(N_PERM):
        pp=perm_indices(d);null.append(np.mean(labels[pp][e.i.to_numpy()]==labels[pp][e.j.to_numpy()]))
    null=np.asarray(null);p=(1+np.sum(null>=obs))/(1+len(null))
    rows.append(dict(acquisition_series=series,stratum=stratum,module="Dominant_state_concordance",n_nodes=len(d),n_participants=d.SampleID.nunique(),n_edges=len(e),observed_edge_correlation=obs,null_mean=float(null.mean()),null_sd=float(null.std(ddof=1)),empirical_p=p,n_permutations=len(null)))
    return rows

all_edges=[];tests=[]
for series,d0 in df.groupby("acquisition_series",sort=True):
    d0=d0.reset_index(drop=True)
    e0=make_graph(d0)
    if e0.empty:continue
    ee=e0.copy();ee["acquisition_series"]=series
    ee["node1_id"]=d0.iloc[ee.i].SampleID.to_numpy()+"::"+d0.iloc[ee.i].clone_id.to_numpy()
    ee["node2_id"]=d0.iloc[ee.j].SampleID.to_numpy()+"::"+d0.iloc[ee.j].clone_id.to_numpy()
    ee["node1_diagnosis"]=d0.iloc[ee.i].Diagnosis.to_numpy();ee["node2_diagnosis"]=d0.iloc[ee.j].Diagnosis.to_numpy()
    all_edges.append(ee.drop(columns=["i","j"]))
    tests.extend(test_graph(d0,e0,series,"All"))
    for dx in ("CD","UC","Control"):
        ids=np.flatnonzero(d0.Diagnosis.to_numpy()==dx)
        if len(ids)<100 or d0.iloc[ids].SampleID.nunique()<3:continue
        mapper=np.full(len(d0),-1,dtype=int);mapper[ids]=np.arange(len(ids))
        ex=e0[e0.i.isin(ids)&e0.j.isin(ids)].copy()
        if len(ex)<100:continue
        ex["i"]=mapper[ex.i.to_numpy()];ex["j"]=mapper[ex.j.to_numpy()]
        tests.extend(test_graph(d0.iloc[ids].reset_index(drop=True),ex.reset_index(drop=True),series,dx))

edge=pd.concat(all_edges,ignore_index=True) if all_edges else pd.DataFrame()
tt=pd.DataFrame(tests)
if len(tt):
    tt["FDR_within_series_stratum"]=tt.groupby(["acquisition_series","stratum"])["empirical_p"].transform(lambda x:np.minimum(1,x.rank(method="max")/len(x)*x.sort_values().to_numpy() if False else x))
    # Explicit Benjamini-Hochberg implementation preserving row order.
    def bh(s):
        p=s.to_numpy(float);order=np.argsort(p);q=np.empty(len(p));rank=np.arange(1,len(p)+1);vals=np.minimum.accumulate((p[order]*len(p)/rank)[::-1])[::-1];q[order]=np.minimum(vals,1);return q
    tt["FDR_within_series_stratum"]=tt.groupby(["acquisition_series","stratum"],group_keys=False)["empirical_p"].transform(bh)

meta=[]
for (stratum,mod),g in tt.groupby(["stratum","module"]):
    g=g[np.isfinite(g.empirical_p)&np.isfinite(g.observed_edge_correlation)]
    if len(g)<3:continue
    z=np.sign(g.observed_edge_correlation.to_numpy())*norm.isf(np.clip(g.empirical_p.to_numpy()/2,1e-12,1-1e-12))
    w=np.sqrt(g.n_edges.to_numpy())
    zm=float(np.sum(w*z)/math.sqrt(np.sum(w*w)));p=float(2*norm.sf(abs(zm)))
    meta.append(dict(stratum=stratum,module=mod,n_series=len(g),meta_z=zm,meta_p=p,n_series_positive=int(np.sum(g.observed_edge_correlation>0)),n_series_nominal_p_lt_005=int(np.sum(g.empirical_p<.05)),total_edges=int(g.n_edges.sum())))
meta=pd.DataFrame(meta)
if len(meta):meta["meta_FDR"]=meta.groupby("stratum",group_keys=False)["meta_p"].transform(bh)

edge.to_csv(root/"Table_TSS3_paired_TCR_distance_graph_edges.csv",index=False)
tt.to_csv(root/"Table_TSS4_sequence_state_permutation_tests_by_series.csv",index=False)
meta.to_csv(root/"Table_TSS5_sequence_state_replication_meta_analysis.csv",index=False)
pd.DataFrame({"item":["distance representation","neighbor graph","exact-clone handling","HLA","permutation","multiplicity","replication"],"value":["paired alpha/beta CDR3 2- and 3-mer TF-IDF plus alpha/beta V/J tokens; cosine distance","up to 5 nearest cross-participant neighbors among 40 candidates; maximum distance 0.50","identical paired receptors excluded","participant HLA genotype unavailable; exploratory V/J-aware analysis only",f"{N_PERM} label permutations within participant and acquisition series","BH FDR within each series/stratum and across meta-analyzed programs","weighted Stouffer synthesis; meta-analysis required >=3 acquisition series"]}).to_csv(root/"Analysis_manifest_TCR_graph.csv",index=False)
print(f"nodes={len(df)} edges={len(edge)} series_tests={len(tt)} meta_tests={len(meta)}")

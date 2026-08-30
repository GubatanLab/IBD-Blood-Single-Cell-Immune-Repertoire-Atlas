from pathlib import Path
import pandas as pd
root=Path(r"C:/path/to/private-manuscript-workspace\High Impact Additional Analyses\BCR Germline Lineages")
rows=[]
assign=[]
for label in ("d010","d015","d020"):
 d=pd.read_csv(root/f"bcr_heavy_lineages_{label}_clone-pass.tsv",sep="\t",usecols=["sequence_id","clone_id","sampleid","diagnosis"],low_memory=False)
 d["lineage_id"]=d.sampleid.astype(str)+"::"+d.clone_id.astype(str)
 s=d.groupby(["sampleid","diagnosis","lineage_id"]).size().rename("n_sequences").reset_index()
 rows.append(dict(threshold=float(label[-3:])/100,records=len(d),lineages=d.lineage_id.nunique(),expanded_lineages=(s.n_sequences>=2).sum(),max_lineage_size=s.n_sequences.max(),median_expanded_size=s.loc[s.n_sequences>=2,"n_sequences"].median()))
 assign.append(d[["sequence_id","lineage_id"]].rename(columns={"lineage_id":f"lineage_{label}"}))
out=assign[0].merge(assign[1],on="sequence_id").merge(assign[2],on="sequence_id")
pd.DataFrame(rows).to_csv(root/"Table_BGL13_lineage_threshold_sensitivity_summary.csv",index=False)
out.to_csv(root/"Table_BGL14_lineage_threshold_assignments.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))

from pathlib import Path
import pandas as pd
root=Path(r"C:/path/to/private-manuscript-workspace\High Impact Additional Analyses\BCR Germline Lineages")
db=pd.read_csv(root/"bcr_heavy_changeo_db-pass.tsv",sep="\t",low_memory=False)
meta=pd.read_csv(root/"Table_BGL1_query_sequence_metadata.csv",low_memory=False)
keep=["sequence_id","SampleID","PatientID","Diagnosis","Inflammation","Biologic","Batch","acquisition_series","count","proportion","receptor_key","v_call_original","j_call_original"]
out=db.merge(meta[keep],on="sequence_id",how="inner",validate="one_to_one")
out.to_csv(root/"bcr_heavy_changeo_with_metadata.tsv",sep="\t",index=False)
print(len(db),len(out),out.SampleID.nunique())

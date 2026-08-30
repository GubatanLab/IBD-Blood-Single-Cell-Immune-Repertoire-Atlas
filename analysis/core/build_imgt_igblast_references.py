from pathlib import Path
from Bio import SeqIO

root=Path(r"C:/path/to/private-manuscript-workspace\analysis_tools")
src=root/"imgt_human_ig_2026-08-24"
out=root/"igblast_db_imgt_2026-08-24"
out.mkdir(parents=True,exist_ok=True)

def convert(files,dest):
    seen=set();records=[]
    for fn in files:
        for r in SeqIO.parse(src/fn,"fasta"):
            parts=r.description.split("|")
            gene=parts[1].strip() if len(parts)>1 else r.id
            if gene in seen:continue
            seen.add(gene);r.id=gene;r.name=gene;r.description="";r.seq=type(r.seq)(str(r.seq).replace(".","").upper())
            records.append(r)
    SeqIO.write(records,out/dest,"fasta")
    print(dest,len(records))

convert(["IGHV.fasta","IGKV.fasta","IGLV.fasta"],"human_gl_V.fasta")
convert(["IGHD.fasta"],"human_gl_D.fasta")
convert(["IGHJ.fasta","IGKJ.fasta","IGLJ.fasta"],"human_gl_J.fasta")

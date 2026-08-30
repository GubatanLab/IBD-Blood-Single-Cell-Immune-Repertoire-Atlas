from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

BLUE = "#167DB7"
ORANGE = "#E56B00"
GREEN = "#009E73"
GREY = "#707070"
LIGHT = "#E8EEF3"

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


T_MODULES = {
    "Naive/CM": "CCR7 SELL TCF7 LEF1 IL7R LTB MAL NOSIP SATB1 BACH2 KLF2 S1PR1 CD27 CD28 BCL2".split(),
    "Cytotoxic": "NKG7 GNLY PRF1 GZMB GZMA GZMH GZMK GZMM CTSW CST7 FGFBP2 CCL5 CCL4 CCL3 IFNG FASLG KLRD1 KLRG1".split(),
    "GZMK memory": "GZMK GZMA CCL5 CCL4 CCL4L2 XCL1 XCL2 NKG7 DUSP2 CRTAM EOMES CXCR3 IL7R".split(),
    "Th17/IL23": "RORC CCR6 IL23R IL17A IL17F IL22 IL26 KLRB1 AHR CCL20 IL1R1 IL21 CXCR6 LTB".split(),
    "TRM": "CD69 ITGAE ITGA1 CXCR6 ZNF683 RUNX3 PRDM1 RGS1 CD101 DUSP6 AHR CCR6".split(),
    "Gut homing": "ITGA4 ITGB7 ITGAE CCR9 CCR6 CXCR3 CXCR6 SELPLG S1PR1 KLF2 SELL GPR183 CD69".split(),
    "Tph/Tfh help": "CXCL13 PDCD1 ICOS MAF TOX2 IL21 CD40LG SLAMF6 TIGIT CD200 CXCR5 BCL6 SH2D1A".split(),
    "Activated Treg": "FOXP3 IL2RA CTLA4 TIGIT IKZF2 IKZF4 TNFRSF18 TNFRSF4 BATF CCR8 LAYN ENTPD1 IL10 AREG".split(),
    "Cycling T": "MKI67 TOP2A STMN1 TYMS PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 HMGB2 CENPF UBE2C PCLAF".split(),
}

B_MODULES = {
    "Naive B": "IGHD IGHM TCL1A IL4R FCER2 CD72 BACH2 CCR7 SELL CD22 MS4A1 CD79A CD79B BANK1 BCL2".split(),
    "Memory B": "CD27 TNFRSF13B AIM2 GPR183 CD80 CD86 BANK1 MS4A1 CD79A CD37 HLA-DRA HLA-DPA1".split(),
    "Atypical memory": "FCRL5 FCRL4 ITGAX TBX21 ZEB2 CXCR3 DUSP4 LYN HOPX TLR7 TLR9 FCGR2B CD86 CD80".split(),
    "Plasma differentiation": "PRDM1 XBP1 IRF4 MZB1 SDC1 JCHAIN SSR4 FKBP11 DERL3 TNFRSF17 SLAMF7 CD38 CD27 SEC11C".split(),
    "Antibody secretion/UPR": "XBP1 HSPA5 HSP90B1 HERPUD1 ATF4 ATF6 DDIT3 SEL1L DNAJB9 PPIB CALR ERP44 DERL3 SEC61A1 SSR4".split(),
    "IgA mucosal": "IGHA1 IGHA2 JCHAIN MZB1 XBP1 SDC1 TNFRSF17 CCR10 PRDM1 IRF4".split(),
    "IgG inflammatory": "IGHG1 IGHG2 IGHG3 IGHG4 JCHAIN MZB1 XBP1 SDC1 PRDM1 IRF4 CXCR4".split(),
    "Inflammatory AP/UPR": "XBP1 MZB1 SDC1 JCHAIN DERL3 HSPA5 HSP90B1 HLA-DRA HLA-DRB1 HLA-DPA1 HLA-DPB1 CD74 CXCR4 IGHG1".split(),
    "Cycling B": "MKI67 TOP2A STMN1 TYMS PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 HMGB2 CENPF UBE2C PCLAF".split(),
}

FOCUSED_T = {
    "Th17 conventional": "RORC CCR6 KLRB1 IL7R IL23R CCL20 IL17A IL17F".split(),
    "Th17 pathogenic": "TBX21 IFNG CXCR3 GZMK CSF2 CCL5".split(),
    "Treg suppressive": "FOXP3 IL2RA CTLA4 TIGIT IKZF2 TNFRSF18 LRRC32 ENTPD1".split(),
    "Treg reprogramming": "RORC KLRB1 CCR6 TBX21 IFNG CXCR3 GZMK CCL5".split(),
}


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def read_10x_scores(path: Path, modules: dict[str, list[str]], extra_features=()):
    """Read a 10x H5, returning library-normalized locked-module scores and feature counts."""
    requested = sorted(set(sum(modules.values(), []) + list(extra_features)))
    with h5py.File(path, "r") as h:
        g = h["matrix"]
        names = decode(g["features"]["name"][:])
        types = decode(g["features"]["feature_type"][:])
        barcodes = decode(g["barcodes"][:])
        indptr = g["indptr"][:]
        indices = g["indices"][:]
        data = g["data"][:].astype(np.float32)
        shape = tuple(g["shape"][:])
    mat = sparse.csc_matrix((data, indices, indptr), shape=shape)
    gex_idx = np.where(types == "Gene Expression")[0]
    lib = np.asarray(mat[gex_idx, :].sum(axis=0)).ravel().astype(np.float32)
    lib[lib == 0] = 1
    by_name = {}
    for gene in requested:
        idx = np.where(names == gene)[0]
        if len(idx):
            by_name[gene] = np.asarray(mat[idx[0], :].todense()).ravel().astype(np.float32)
    scores = pd.DataFrame(index=barcodes)
    for label, genes in modules.items():
        present = [g for g in genes if g in by_name]
        if present:
            vals = np.vstack([np.log1p(by_name[g] / lib * 1e4) for g in present])
            scores[label] = vals.mean(axis=0)
        else:
            scores[label] = np.nan
    feature_counts = pd.DataFrame(index=barcodes)
    for feature in extra_features:
        idx = np.where(names == feature)[0]
        feature_counts[feature] = np.asarray(mat[idx[0], :].todense()).ravel() if len(idx) else 0
    return scores, feature_counts


def choose_chain(df, prefixes):
    x = df[df["chain"].astype(str).str.startswith(tuple(prefixes))].copy()
    if x.empty:
        return pd.DataFrame()
    for c in ("umis", "duplicate_count", "consensus_count"):
        if c in x:
            x["_weight"] = pd.to_numeric(x[c], errors="coerce").fillna(0)
            break
    else:
        x["_weight"] = 1
    x = x.sort_values(["barcode", "_weight"], ascending=[True, False]).drop_duplicates("barcode")
    x["chain_token"] = x["v_gene"].fillna("").astype(str) + "|" + x["j_gene"].fillna("").astype(str) + "|" + x["cdr3"].fillna("").astype(str)
    return x.set_index("barcode")


def paired_contigs(path: Path, receptor: str):
    x = pd.read_csv(path)
    if "productive" in x:
        x = x[x["productive"].astype(str).str.lower().isin(["true", "t", "1"])]
    if "is_cell" in x:
        x = x[x["is_cell"].astype(str).str.lower().isin(["true", "t", "1"])]
    if "high_confidence" in x:
        x = x[x["high_confidence"].astype(str).str.lower().isin(["true", "t", "1"])]
    if receptor == "TCR":
        a, b = choose_chain(x, ["TRA"]), choose_chain(x, ["TRB"])
    else:
        a, b = choose_chain(x, ["IGH"]), choose_chain(x, ["IGK", "IGL"])
    cells = a.index.intersection(b.index)
    out = pd.DataFrame(index=cells)
    out["clonotype"] = a.loc[cells, "chain_token"] + "||" + b.loc[cells, "chain_token"]
    out["chain1_c"] = a.loc[cells, "c_gene"].astype(str) if "c_gene" in a else ""
    out["chain1_v"] = a.loc[cells, "v_gene"].astype(str)
    out["chain1_j"] = a.loc[cells, "j_gene"].astype(str)
    out["chain1_cdr3"] = a.loc[cells, "cdr3"].astype(str)
    out["chain2_v"] = b.loc[cells, "v_gene"].astype(str)
    out["chain2_j"] = b.loc[cells, "j_gene"].astype(str)
    out["chain2_cdr3"] = b.loc[cells, "cdr3"].astype(str)
    out.index.name = "barcode"
    return out


def add_expansion(x):
    sizes = x["clonotype"].value_counts()
    x = x.copy()
    x["clone_size"] = x["clonotype"].map(sizes).astype(int)
    x["expansion_bin"] = pd.cut(x["clone_size"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3–4", "≥5"])
    x["expanded"] = x["clone_size"] >= 2
    return x


def participant_two_group(df, value, group, family, comparison, min_per_group=3):
    labels = sorted(df[group].dropna().unique())
    if set(labels) == {"healthy", "UC"}:
        labels = ["healthy", "UC"]
    elif set(labels) == {"healthy", "IBD"}:
        labels = ["healthy", "IBD"]
    vals = [df.loc[df[group] == label, value].dropna().to_numpy() for label in labels]
    if len(labels) != 2 or min(map(len, vals)) < min_per_group:
        return None
    a = df.loc[df[group] == labels[0], value].dropna().to_numpy()
    b = df.loc[df[group] == labels[1], value].dropna().to_numpy()
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "family": family, "comparison": comparison, "contrast": f"{labels[1]} - {labels[0]}",
        "effect": float(np.median(b) - np.median(a)), "p": float(u.pvalue),
        "n1": len(a), "n2": len(b), "group1": labels[0], "group2": labels[1],
        "median1": float(np.median(a)), "median2": float(np.median(b)),
    }


def paired_test(df, before, after, value, family, comparison):
    pvt = df.pivot_table(index="participant", columns="time", values=value, aggfunc="mean")
    if before not in pvt or after not in pvt:
        return None
    pvt = pvt[[before, after]].dropna()
    if len(pvt) < 3:
        return None
    try:
        w = stats.wilcoxon(pvt[after], pvt[before], alternative="two-sided")
        p = float(w.pvalue)
    except ValueError:
        p = 1.0
    return {
        "family": family, "comparison": comparison, "contrast": f"{after} - {before}",
        "effect": float(np.median(pvt[after] - pvt[before])), "p": p,
        "n1": len(pvt), "n2": len(pvt), "group1": before, "group2": after,
        "median1": float(np.median(pvt[before])), "median2": float(np.median(pvt[after])),
    }


def analyze_gse261():
    folder = ROOT / "GSE261334"
    h5s = sorted(folder.glob("GSM*_PBMC_filtered_counts.h5"))
    sample_rows, bin_rows, stats_rows, corr_rows = [], [], [], []
    for i, h5 in enumerate(h5s, 1):
        stem = re.sub(r"^GSM\d+_", "", h5.name).replace("_PBMC_filtered_counts.h5", "")
        m = re.match(r"(HC-\d+|IBD-\d+)(?:_([12]))?$", stem)
        if not m:
            continue
        participant = m.group(1)
        time = "baseline" if m.group(2) in (None, "1") else "week6"
        disease = "healthy" if participant.startswith("HC") else "UC"
        modules = {**T_MODULES, **B_MODULES, **FOCUSED_T}
        scores, _ = read_10x_scores(h5, modules)
        for receptor, suffix in (("TCR", "vdj_t_contig_annotations.csv.gz"), ("BCR", "vdj_b_contig_annotations.csv.gz")):
            contig = Path(str(h5).replace("filtered_counts.h5", suffix))
            pairs = paired_contigs(contig, receptor)
            common = scores.index.intersection(pairs.index)
            dat = add_expansion(pairs.loc[common].join(scores.loc[common]))
            module_set = list(T_MODULES) + list(FOCUSED_T) if receptor == "TCR" else list(B_MODULES)
            row = {"cohort": "GSE261334", "sample": stem, "participant": participant,
                   "time": time, "disease": disease, "receptor": receptor,
                   "n_paired_cells": len(dat), "n_clonotypes": dat["clonotype"].nunique(),
                   "expanded_cell_fraction": dat["expanded"].mean() if len(dat) else np.nan}
            for mod in module_set:
                row[mod] = dat[mod].median() if len(dat) else np.nan
            sample_rows.append(row)
            for mod in module_set:
                for eb, g in dat.groupby("expansion_bin", observed=True):
                    bin_rows.append({"cohort": "GSE261334", "sample": stem, "participant": participant,
                                     "time": time, "disease": disease, "receptor": receptor,
                                     "module": mod, "expansion_bin": str(eb), "value": g[mod].median(),
                                     "n_cells": len(g)})
        print(f"GSE261334 {i}/{len(h5s)} {stem}", flush=True)
    sample = pd.DataFrame(sample_rows)
    bins = pd.DataFrame(bin_rows)
    sample.to_csv(OUT / "GSE261334_participant_sample_summary.csv", index=False)
    bins.to_csv(OUT / "GSE261334_expansion_bin_programs.csv", index=False)

    # Baseline disease differences and paired induction changes.
    for receptor, mods in (("TCR", list(T_MODULES) + list(FOCUSED_T)), ("BCR", list(B_MODULES))):
        base = sample[(sample.receptor == receptor) & (sample.time == "baseline")]
        for mod in mods + ["expanded_cell_fraction"]:
            r = participant_two_group(base, mod, "disease", "GSE261334 baseline", f"{receptor}: {mod}")
            if r: stats_rows.append(r)
        uc = sample[(sample.receptor == receptor) & (sample.disease == "UC")]
        for mod in mods + ["expanded_cell_fraction"]:
            r = paired_test(uc, "baseline", "week6", mod, "GSE261334 longitudinal", f"{receptor}: {mod}")
            if r: stats_rows.append(r)

    # Donor-level expansion-bin dose response, Spearman over bin medians.
    order = {"1": 0, "2": 1, "3–4": 2, "≥5": 3}
    dose = []
    for (receptor, mod, sample_id), g in bins.groupby(["receptor", "module", "sample"]):
        g = g.assign(x=g.expansion_bin.map(order)).dropna(subset=["x", "value"])
        if len(g) >= 3:
            rho, _ = stats.spearmanr(g.x, g.value)
            dose.append({"receptor": receptor, "module": mod, "sample": sample_id,
                         "participant": g.participant.iloc[0], "time": g.time.iloc[0],
                         "disease": g.disease.iloc[0], "rho": rho})
    dose = pd.DataFrame(dose)
    dose.to_csv(OUT / "GSE261334_expansion_program_dose_response.csv", index=False)
    for (receptor, mod), g in dose[(dose.time == "baseline") & (dose.disease == "UC")].groupby(["receptor", "module"]):
        if len(g) >= 5:
            try: p = stats.wilcoxon(g.rho, alternative="two-sided").pvalue
            except ValueError: p = 1.0
            stats_rows.append({"family": "GSE261334 expansion dose response", "comparison": f"{receptor}: {mod}",
                               "contrast": "Spearman rho across 1, 2, 3–4, ≥5 cells",
                               "effect": float(np.median(g.rho)), "p": float(p), "n1": len(g), "n2": np.nan,
                               "group1": "UC baseline", "group2": "", "median1": float(np.median(g.rho)), "median2": np.nan})

    # Cross-receptor coordination using participant-sample medians.
    # Use one sample per participant for cross-receptor coordination; longitudinal
    # week-6 samples are excluded to avoid treating repeated measures as independent.
    wide = sample[sample.time == "baseline"].pivot_table(index=["participant", "disease"], columns="receptor", aggfunc="first")
    for tm in ["GZMK memory", "Th17/IL23", "Activated Treg", "Tph/Tfh help"]:
        for bm in ["Plasma differentiation", "Antibody secretion/UPR", "IgA mucosal", "IgG inflammatory"]:
            try:
                x = wide[(tm, "TCR")]
                y = wide[(bm, "BCR")]
            except KeyError:
                continue
            ok = x.notna() & y.notna()
            rho, p = stats.spearmanr(x[ok], y[ok]) if ok.sum() >= 5 else (np.nan, np.nan)
            corr_rows.append({"T_module": tm, "B_module": bm, "rho": rho, "p": p, "n": int(ok.sum())})
    corr = pd.DataFrame(corr_rows)
    corr["fdr"] = bh(corr.p)
    corr.to_csv(OUT / "GSE261334_T_B_coordination.csv", index=False)
    return sample, bins, dose, pd.DataFrame(stats_rows), corr


def sample_key(path: Path):
    m = re.match(r"GSM\d+_(.+?)_(?:cell-gene_UMI_table|full-length_productive_[BT]CR_table)", path.name)
    return m.group(1) if m else None


def paired_gse125(path: Path, receptor: str):
    x = pd.read_csv(path, sep="\t")
    x = x.rename(columns={"cell_id": "barcode", "v_gene": "v_gene", "j_gene": "j_gene", "cdr3": "cdr3"})
    x["chain"] = x["v_gene"].fillna("").astype(str).str.extract(r"^(TR[AB]|IGH|IGK|IGL)")[0].fillna("")
    if receptor == "TCR":
        a, b = choose_chain(x, ["TRA"]), choose_chain(x, ["TRB"])
    else:
        a, b = choose_chain(x, ["IGH"]), choose_chain(x, ["IGK", "IGL"])
    cells = a.index.intersection(b.index)
    out = pd.DataFrame(index=cells)
    out["clonotype"] = a.loc[cells, "chain_token"] + "||" + b.loc[cells, "chain_token"]
    out["heavy_isotype"] = a.loc[cells, "c_gene"].fillna("").astype(str) if "c_gene" in a else ""
    out["celltype"] = a.loc[cells, "celltype"].fillna("").astype(str) if "celltype" in a else ""
    out.index.name = "barcode"
    return out


def read_wide_scores(path: Path, barcodes, modules):
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    genes = sorted(set(sum(modules.values(), [])))
    present = [g for g in genes if g in header]
    use = [header[0]] + present
    # Targeted CPM denominator is the sum across all locked-signature genes. It is
    # deterministic and avoids loading the 10,000-gene wide matrices into memory.
    x = pd.read_csv(path, sep="\t", usecols=use)
    x = x[x[header[0]].astype(str).isin(set(barcodes))].set_index(header[0])
    denom = x[present].sum(axis=1).to_numpy(float)
    denom[denom == 0] = 1
    out = pd.DataFrame(index=x.index)
    for label, gs in modules.items():
        pp = [g for g in gs if g in x]
        out[label] = np.log1p(x[pp].to_numpy(float) / denom[:, None] * 1e4).mean(axis=1) if pp else np.nan
    return out


def analyze_gse125():
    folder = ROOT / "GSE125527"
    expr = {sample_key(p): p for p in folder.glob("*cell-gene_UMI_table.tsv.gz")}
    tcr = {sample_key(p): p for p in folder.glob("*TCR_table.tsv.gz")}
    bcr = {sample_key(p): p for p in folder.glob("*BCR_table.tsv.gz")}
    cell_rows, sample_rows, bbin_rows, traffic_rows, lineage_rows = [], [], [], [], []
    all_t_clones, all_b_clones = [], []
    for i, key in enumerate(sorted(expr), 1):
        for receptor, files, modules in (("TCR", tcr, {**T_MODULES, **FOCUSED_T}), ("BCR", bcr, B_MODULES)):
            if key not in files:
                continue
            pairs = paired_gse125(files[key], receptor)
            scores = read_wide_scores(expr[key], pairs.index, modules)
            common = scores.index.intersection(pairs.index)
            dat = add_expansion(pairs.loc[common].join(scores.loc[common]))
            parts = key.split("_")
            participant, tissue = parts[0], parts[1]
            disease = "UC" if participant.startswith("U") else "healthy"
            dat = dat.assign(participant=participant, tissue=tissue, disease=disease, sample=key, receptor=receptor)
            if receptor == "TCR":
                z = dat[["Th17 conventional", "Treg suppressive"]].apply(lambda c: (c-c.mean())/(c.std()+1e-8))
                strength = z.max(axis=1)
                dat["program_state"] = np.where(strength < 0, "other", np.where(z["Th17 conventional"] >= z["Treg suppressive"], "Th17", "Treg"))
                all_t_clones.append(dat[["participant", "tissue", "disease", "clonotype", "program_state", "expanded", "clone_size"]].reset_index())
                for st, g in dat[dat.program_state.isin(["Th17", "Treg"])].groupby("program_state"):
                    sample_rows.append({"cohort": "GSE125527", "sample": key, "participant": participant,
                                        "tissue": tissue, "disease": disease, "measure": f"{st} expanded-cell fraction",
                                        "value": g.expanded.mean(), "n_cells": len(g)})
                expanded_clones = dat[dat.clone_size >= 2].groupby("clonotype").program_state.agg(lambda s: set(s))
                mixed = expanded_clones.map(lambda s: "Th17" in s and "Treg" in s)
                sample_rows.append({"cohort": "GSE125527", "sample": key, "participant": participant,
                                    "tissue": tissue, "disease": disease, "measure": "mixed Th17-Treg expanded-clone fraction",
                                    "value": mixed.mean() if len(mixed) else np.nan, "n_cells": len(dat)})
            else:
                all_b_clones.append(dat[["participant", "tissue", "disease", "clonotype", "heavy_isotype", "expanded", "clone_size"] + list(B_MODULES)].reset_index())
                for mod in ["Plasma differentiation", "Antibody secretion/UPR", "IgA mucosal", "IgG inflammatory", "Cycling B"]:
                    for eb, g in dat.groupby("expansion_bin", observed=True):
                        bbin_rows.append({"participant": participant, "tissue": tissue, "disease": disease,
                                          "module": mod, "expansion_bin": str(eb), "value": g[mod].median(), "n_cells": len(g)})
                # Constant-region diversity within exact paired V/J/CDR3 lineages.
                for clone, g in dat.groupby("clonotype"):
                    dom = g[list(B_MODULES)].idxmax(axis=1)
                    lineage_rows.append({"participant": participant, "tissue": tissue, "disease": disease,
                                         "clonotype": clone, "clone_size": len(g),
                                         "n_isotypes": g.heavy_isotype.replace("", np.nan).nunique(),
                                         "n_program_states": dom.nunique(),
                                         "class_switched": g.heavy_isotype.str.contains("IGHA|IGHG|IGHE").any()})
            print(f"GSE125527 {i}/{len(expr)} {key} {receptor}", flush=True)

    sample = pd.DataFrame(sample_rows)
    bbin = pd.DataFrame(bbin_rows)
    lineage = pd.DataFrame(lineage_rows)
    tclone = pd.concat(all_t_clones, ignore_index=True) if all_t_clones else pd.DataFrame()
    bclone = pd.concat(all_b_clones, ignore_index=True) if all_b_clones else pd.DataFrame()

    # Cross-compartment exact-clonotype traffic by participant.
    for receptor, dc in (("TCR", tclone), ("BCR", bclone)):
        for participant, g in dc.groupby("participant"):
            tissue_sets = {t: set(q.clonotype) for t, q in g.groupby("tissue")}
            if "R" in tissue_sets and "pBMC" in tissue_sets:
                inter = tissue_sets["R"] & tissue_sets["pBMC"]
                traffic_rows.append({"participant": participant, "disease": g.disease.iloc[0], "receptor": receptor,
                                     "comparison": "PBMC–rectum", "n_shared": len(inter),
                                     "jaccard": len(inter) / max(1, len(tissue_sets["R"] | tissue_sets["pBMC"])),
                                     "rectum_fraction_shared": len(inter) / max(1, len(tissue_sets["R"]))})
            if "I" in tissue_sets and "pBMC" in tissue_sets:
                inter = tissue_sets["I"] & tissue_sets["pBMC"]
                traffic_rows.append({"participant": participant, "disease": g.disease.iloc[0], "receptor": receptor,
                                     "comparison": "PBMC–ileum", "n_shared": len(inter),
                                     "jaccard": len(inter) / max(1, len(tissue_sets["I"] | tissue_sets["pBMC"])),
                                     "rectum_fraction_shared": len(inter) / max(1, len(tissue_sets["I"]))})
    traffic = pd.DataFrame(traffic_rows)

    stats_rows = []
    for (tissue, measure), g in sample.groupby(["tissue", "measure"]):
        r = participant_two_group(g, "value", "disease", "GSE125527 T-cell programs", f"{tissue}: {measure}")
        if r: stats_rows.append(r)
    for (receptor, comp), g in traffic.groupby(["receptor", "comparison"]):
        r = participant_two_group(g, "rectum_fraction_shared", "disease", "GSE125527 clone traffic", f"{receptor}: {comp}")
        if r: stats_rows.append(r)
    # Expansion-bin program dose response per participant/tissue.
    order = {"1": 0, "2": 1, "3–4": 2, "≥5": 3}
    bdose = []
    for (participant, tissue, disease, mod), g in bbin.groupby(["participant", "tissue", "disease", "module"]):
        g = g.assign(x=g.expansion_bin.map(order)).dropna(subset=["x", "value"])
        if len(g) >= 3:
            rho, _ = stats.spearmanr(g.x, g.value)
            bdose.append({"participant": participant, "tissue": tissue, "disease": disease, "module": mod, "rho": rho})
    bdose = pd.DataFrame(bdose)
    for (tissue, mod), g in bdose.groupby(["tissue", "module"]):
        r = participant_two_group(g, "rho", "disease", "GSE125527 BCR expansion kinetics", f"{tissue}: {mod}")
        if r: stats_rows.append(r)
    # Lineage maturation: participant-level fractions among expanded lineages.
    lin = lineage[lineage.clone_size >= 2].groupby(["participant", "tissue", "disease"]).agg(
        class_switch_fraction=("n_isotypes", lambda x: np.mean(x >= 2)),
        cross_program_fraction=("n_program_states", lambda x: np.mean(x >= 2)),
        expanded_lineages=("clonotype", "nunique"),
    ).reset_index()
    for tissue, g in lin.groupby("tissue"):
        for value in ["class_switch_fraction", "cross_program_fraction"]:
            r = participant_two_group(g, value, "disease", "GSE125527 BCR lineage maturation", f"{tissue}: {value}")
            if r: stats_rows.append(r)

    for name, df in (("GSE125527_T_cell_sample_metrics.csv", sample),
                     ("GSE125527_BCR_expansion_program_bins.csv", bbin),
                     ("GSE125527_cross_compartment_clone_traffic.csv", traffic),
                     ("GSE125527_BCR_lineage_metrics.csv", lin),
                     ("GSE125527_BCR_expansion_dose_response.csv", bdose)):
        df.to_csv(OUT / name, index=False)
    return sample, bbin, traffic, lin, bdose, pd.DataFrame(stats_rows)


def paired_airr(path: Path):
    x = pd.read_csv(path, sep="\t")
    x = x[x["productive"].astype(str).str.lower().isin(["true", "t", "1"])]
    x = x.rename(columns={"cell_id": "barcode", "v_call": "v_gene", "j_call": "j_gene", "junction_aa": "cdr3"})
    x["chain"] = x["v_gene"].fillna("").astype(str).str.extract(r"^(TR[AB])")[0].fillna("")
    a, b = choose_chain(x, ["TRA"]), choose_chain(x, ["TRB"])
    cells = a.index.intersection(b.index)
    out = pd.DataFrame(index=cells)
    out["clonotype"] = a.loc[cells, "chain_token"] + "||" + b.loc[cells, "chain_token"]
    out.index.name = "barcode"
    return out


def analyze_gse301():
    folder = ROOT / "GSE301689"
    rows, clone_rows, stats_rows = [], [], []
    hto_map = {"Hashtag1": "mild colon", "Hashtag2": "inflamed colon", "Hashtag3": "MLN", "Hashtag4": "blood",
               "Hashtag_1": "mild colon", "Hashtag_2": "inflamed colon", "Hashtag_3": "MLN", "Hashtag_4": "blood"}
    for donor in ("CD10", "CD11", "CD12"):
        h5 = folder / f"GSE301689_{donor}_CITE_filtered_feature_bc_matrix.h5"
        airr = next(folder.glob(f"GSM*_{donor}_CITE_airr_rearrangement.tsv.gz"))
        scores, hto = read_10x_scores(h5, T_MODULES, list(hto_map))
        pairs = paired_airr(airr)
        common = scores.index.intersection(pairs.index)
        dat = add_expansion(pairs.loc[common].join(scores.loc[common]).join(hto.loc[common]))
        hv = dat[list(hto_map)].to_numpy(float)
        order = np.argsort(hv, axis=1)
        top = hv[np.arange(len(hv)), order[:, -1]]
        second = hv[np.arange(len(hv)), order[:, -2]]
        names = np.asarray(list(hto_map))[order[:, -1]]
        dat["compartment"] = [hto_map[n] for n in names]
        dat.loc[(top < 5) | ((top + 1) / (second + 1) < 2), "compartment"] = "unassigned"
        clone_comp = dat[dat.compartment != "unassigned"].groupby("clonotype").compartment.agg(set)
        for clone, comps in clone_comp.items():
            clone_rows.append({"participant": donor, "clonotype": clone, "n_compartments": len(comps),
                               "compartments": ";".join(sorted(comps)), "blood_colon_shared": "blood" in comps and bool({"mild colon", "inflamed colon"} & comps)})
        for comp in ["mild colon", "inflamed colon", "MLN"]:
            q = dat[dat.compartment == comp].copy()
            if q.empty: continue
            shared = set(clone_comp[clone_comp.map(lambda s: "blood" in s)].index)
            q["traffic"] = np.where(q.clonotype.isin(shared), "blood-shared", "tissue-private")
            for mod in ["GZMK memory", "Th17/IL23", "TRM", "Gut homing", "Tph/Tfh help", "Activated Treg", "Cycling T"]:
                for traffic, gg in q.groupby("traffic"):
                    rows.append({"participant": donor, "compartment": comp, "module": mod, "traffic": traffic,
                                 "value": gg[mod].median(), "n_cells": len(gg), "n_clonotypes": gg.clonotype.nunique()})
        print(f"GSE301689 {donor}", flush=True)
    metrics = pd.DataFrame(rows)
    clones = pd.DataFrame(clone_rows)
    for (comp, mod), g in metrics.groupby(["compartment", "module"]):
        pvt = g.pivot_table(index="participant", columns="traffic", values="value").dropna()
        if len(pvt) >= 3 and {"blood-shared", "tissue-private"}.issubset(pvt.columns):
            try: p = stats.wilcoxon(pvt["blood-shared"], pvt["tissue-private"]).pvalue
            except ValueError: p = 1.0
            stats_rows.append({"family": "GSE301689 cross-tissue programs", "comparison": f"{comp}: {mod}",
                               "contrast": "blood-shared - tissue-private", "effect": float(np.median(pvt["blood-shared"]-pvt["tissue-private"])),
                               "p": float(p), "n1": len(pvt), "n2": len(pvt), "group1": "tissue-private", "group2": "blood-shared",
                               "median1": float(np.median(pvt["tissue-private"])), "median2": float(np.median(pvt["blood-shared"]))})
    metrics.to_csv(OUT / "GSE301689_cross_tissue_program_metrics.csv", index=False)
    clones.to_csv(OUT / "GSE301689_clone_compartment_occupancy.csv", index=False)
    return metrics, clones, pd.DataFrame(stats_rows)


def motif_regex(motif):
    return re.compile(re.escape(motif).replace("%", "."))


def analyze_jci():
    internal = MANUSCRIPT / "High Impact Additional Analyses" / "Priority Analyses" / "BetaOnly_GLIPH2"
    seq = pd.read_csv(internal / "exact_beta_receptors_all_participants.csv")
    motifs = ["%IGSGANV", "RD%LYG", "S%LDGYE", "S%RGATGE", "SESG%GQP"]
    external = {
        "%IGSGANV": (21, 32, 5, 24, 7.255, 0.0012),
        "RD%LYG": (21, 32, 5, 24, 7.255, 0.0012),
        "S%LDGYE": (18, 32, 20, 24, 0.257, 0.0441),
        "S%RGATGE": (np.nan, 32, np.nan, 24, np.nan, 0.5813),
        "SESG%GQP": (np.nan, 32, np.nan, 24, np.nan, 0.2820),
    }
    rows = []
    participant = seq[["patient", "group"]].drop_duplicates()
    for motif in motifs:
        rx = motif_regex(motif)
        carriers = set(seq.loc[seq.CDR3b.astype(str).map(lambda s: bool(rx.search(s))), "patient"])
        participant["carrier"] = participant.patient.isin(carriers)
        ibd = participant.group.isin(["CD", "UC"])
        a = int((participant.carrier & ibd).sum()); b = int((~participant.carrier & ibd).sum())
        c = int((participant.carrier & ~ibd).sum()); d = int((~participant.carrier & ~ibd).sum())
        _, p = stats.fisher_exact([[a, b], [c, d]])
        # Haldane–Anscombe correction keeps rare zero-cell motifs finite for plotting.
        orr = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        e = external[motif]
        rows.append({"motif": motif, "internal_IBD_carriers": a, "internal_IBD_n": a+b,
                     "internal_HC_carriers": c, "internal_HC_n": c+d, "internal_OR": orr,
                     "internal_p": p, "external_IBD_carriers": e[0], "external_IBD_n": e[1],
                     "external_HC_carriers": e[2], "external_HC_n": e[3], "external_OR": e[4], "external_p": e[5]})
    out = pd.DataFrame(rows)
    out["internal_fdr"] = bh(out.internal_p)
    out["external_fdr"] = bh(out.external_p)
    # Exact GLIPH tag recurrence in the manuscript's beta-only runs.
    tags = []
    for f in internal.glob("*/cluster_group_enrichment.csv"):
        x = pd.read_csv(f, usecols=["tag", "fdr", "fisher_or"])
        for motif in motifs:
            q = x[x.tag == motif]
            if len(q):
                tags.append({"comparison": f.parent.name, "motif": motif, "n_tag_rows": len(q),
                             "minimum_fdr": q.fdr.min(), "maximum_OR": q.fisher_or.replace(np.inf, np.nan).max()})
    pd.DataFrame(tags).to_csv(OUT / "JCI_exact_GLIPH_tag_recurrence.csv", index=False)
    out.to_csv(OUT / "JCI_motif_external_validation.csv", index=False)
    return out


def finalize_stats(*frames):
    stats_df = pd.concat([x for x in frames if x is not None and len(x)], ignore_index=True)
    stats_df["fdr"] = stats_df.groupby("family")["p"].transform(bh)
    stats_df.to_csv(OUT / "External_validation_all_statistics.csv", index=False)
    return stats_df


def forest(ax, df, title, color=BLUE, xmax=None):
    q = df.sort_values("effect").copy()
    y = np.arange(len(q))
    ax.axvline(0, color="#999999", lw=.8, ls="--")
    ax.scatter(q.effect, y, s=25, color=np.where(q.fdr < .05, color, "white"), edgecolor=color, zorder=3)
    ax.set_yticks(y, q.comparison.str.replace("TCR: ", "", regex=False).str.replace("BCR: ", "", regex=False))
    ax.set_title(title, loc="left", weight="bold")
    ax.set_xlabel("Median participant-level effect")
    if xmax is not None: ax.set_xlim(-xmax, xmax)
    for yi, (_, r) in enumerate(q.iterrows()):
        marker = "*" if r.fdr < .05 else ("†" if r.fdr < .1 else "")
        if marker:
            ax.text(r.effect, yi, f"  {marker}" if r.effect >= 0 else f"{marker}  ", va="center", ha="left" if r.effect >= 0 else "right", color=color)


def make_figures(g261_sample, g261_bins, g261_dose, g261_stats, corr,
                 g125_sample, g125_bins, traffic, lineage, bdose, g125_stats,
                 g301_metrics, g301_clones, g301_stats, motifs, all_stats):
    # Figure EV1: independent longitudinal PBMC validation.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    q = all_stats[(all_stats.family == "GSE261334 baseline") & all_stats.comparison.str.contains("GZMK|Th17|Treg|Tph|Plasma|UPR|IgA|IgG", regex=True)]
    forest(axes[0,0], q, "A  Baseline UC versus healthy", BLUE)
    q = all_stats[(all_stats.family == "GSE261334 longitudinal") & all_stats.comparison.str.contains("GZMK|Th17|Treg|Tph|Plasma|UPR|IgA|IgG", regex=True)]
    forest(axes[0,1], q, "B  Week 6 change from baseline", GREEN)
    q = g261_dose[(g261_dose.time == "baseline") & (g261_dose.disease == "UC") & g261_dose.module.isin(["GZMK memory","Th17/IL23","Activated Treg","Plasma differentiation","Antibody secretion/UPR","IgA mucosal","IgG inflammatory"])]
    sns.boxplot(data=q, y="module", x="rho", hue="receptor", palette={"TCR": BLUE, "BCR": ORANGE}, ax=axes[1,0], fliersize=0, linewidth=.8)
    sns.stripplot(data=q, y="module", x="rho", hue="receptor", dodge=True, palette={"TCR": BLUE, "BCR": ORANGE}, ax=axes[1,0], size=3, alpha=.7)
    axes[1,0].axvline(0, color="#999", ls="--", lw=.8); axes[1,0].set_title("C  Clone-size dose response", loc="left", weight="bold"); axes[1,0].set_xlabel("Participant Spearman rho"); axes[1,0].set_ylabel("")
    handles, labels = axes[1,0].get_legend_handles_labels(); axes[1,0].legend(handles[:2], labels[:2], frameon=False, title="")
    cm = corr.pivot(index="T_module", columns="B_module", values="rho")
    sns.heatmap(cm, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", cbar_kws={"label":"Spearman rho"}, ax=axes[1,1])
    axes[1,1].set_title("D  Coordinated T–B programs", loc="left", weight="bold"); axes[1,1].set_xlabel(""); axes[1,1].set_ylabel("")
    save_figure(fig, "ExternalValidation_Figure1_Longitudinal_PBMC")

    # Figure EV2: mucosal and cross-compartment replication.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    q = g125_sample[(g125_sample.tissue == "R") & g125_sample.measure.str.contains("expanded-cell")]
    sns.boxplot(data=q, x="measure", y="value", hue="disease", palette={"healthy": GREY, "UC": BLUE}, ax=axes[0,0], fliersize=0)
    sns.stripplot(data=q, x="measure", y="value", hue="disease", dodge=True, palette={"healthy": GREY, "UC": BLUE}, ax=axes[0,0], size=4)
    axes[0,0].set_title("A  Rectal Th17/Treg clone expansion", loc="left", weight="bold"); axes[0,0].set_xlabel(""); axes[0,0].set_ylabel("Expanded-cell fraction"); axes[0,0].tick_params(axis="x", rotation=20)
    h,l=axes[0,0].get_legend_handles_labels(); axes[0,0].legend(h[:2],l[:2],frameon=False)
    q = traffic[traffic.comparison == "PBMC–rectum"]
    sns.boxplot(data=q, x="receptor", y="rectum_fraction_shared", hue="disease", palette={"healthy": GREY, "UC": GREEN}, ax=axes[0,1], fliersize=0)
    sns.stripplot(data=q, x="receptor", y="rectum_fraction_shared", hue="disease", dodge=True, palette={"healthy": GREY, "UC": GREEN}, ax=axes[0,1], size=4)
    axes[0,1].set_title("B  Exact paired clones shared with blood", loc="left", weight="bold"); axes[0,1].set_ylabel("Fraction of rectal clonotypes"); axes[0,1].set_xlabel(""); h,l=axes[0,1].get_legend_handles_labels(); axes[0,1].legend(h[:2],l[:2],frameon=False)
    q = bdose[(bdose.tissue == "R") & bdose.module.isin(["Plasma differentiation","Antibody secretion/UPR","IgA mucosal","IgG inflammatory","Cycling B"])]
    sns.boxplot(data=q, y="module", x="rho", hue="disease", palette={"healthy": GREY,"UC": ORANGE}, ax=axes[1,0], fliersize=0)
    sns.stripplot(data=q, y="module", x="rho", hue="disease", dodge=True, palette={"healthy": GREY,"UC": ORANGE}, ax=axes[1,0], size=3)
    axes[1,0].axvline(0,color="#999",ls="--",lw=.8); axes[1,0].set_title("C  BCR expansion–program kinetics",loc="left",weight="bold"); axes[1,0].set_xlabel("Participant Spearman rho"); axes[1,0].set_ylabel(""); h,l=axes[1,0].get_legend_handles_labels(); axes[1,0].legend(h[:2],l[:2],frameon=False)
    q = g301_metrics[g301_metrics.compartment == "inflamed colon"].pivot_table(index=["participant","module"],columns="traffic",values="value").dropna().reset_index()
    q["difference"] = q["blood-shared"] - q["tissue-private"]
    q = q[q.module.isin(["GZMK memory","Th17/IL23","TRM","Gut homing","Tph/Tfh help","Activated Treg"])]
    sns.pointplot(data=q, y="module", x="difference", errorbar=None, color=GREEN, ax=axes[1,1])
    sns.stripplot(data=q, y="module", x="difference", color=GREEN, ax=axes[1,1], size=4, alpha=.75)
    axes[1,1].axvline(0,color="#999",ls="--",lw=.8); axes[1,1].set_title("D  Blood-shared versus colon-private clones",loc="left",weight="bold"); axes[1,1].set_xlabel("Median program-score difference (n=3)"); axes[1,1].set_ylabel("")
    save_figure(fig, "ExternalValidation_Figure2_Mucosal_Clonotypes")

    # Figure EV3: convergence and feasibility/audit.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    m = motifs.copy(); y=np.arange(len(m))
    axes[0].axvline(1,color="#999",ls="--",lw=.8)
    for label,col,off,field in (("Internal PBMC cohort",BLUE,-.12,"internal_OR"),("Independent European cohort",ORANGE,.12,"external_OR")):
        vals=m[field].replace([np.inf,-np.inf],np.nan)
        axes[0].scatter(vals,y+off,label=label,color=col,s=32)
    axes[0].set_xscale("log"); axes[0].set_yticks(y,m.motif); axes[0].invert_yaxis(); axes[0].set_xlabel("IBD versus healthy carrier odds ratio (log scale)"); axes[0].set_title("A  Independent TCR motif validation",loc="left",weight="bold"); axes[0].legend(frameon=False,fontsize=7)
    audit = pd.DataFrame([
        ["GSE261334", "Longitudinal PBMC + paired TCR/BCR", "Executable", "5 healthy; 10 UC, paired baseline/week 6"],
        ["GSE125527", "Blood–rectum/ileum paired repertoires", "Executable", "15 participants"],
        ["GSE301689", "Blood–colon–MLN paired TCR traffic", "Descriptive", "3 Crohn disease participants"],
        ["JCI Insight 2026", "Five GLIPH2 specificity motifs", "Executable", "Independent 32 IBD / 24 healthy validation"],
        ["SCP1690", "Colon plasma-cell repertoire", "Access-limited", "Portal requires authenticated download"],
        ["Multi-organ CD B atlas", "Cross-organ B-cell state/lineage validation", "Not executable", "No public cell-level accession located"],
        ["Frozen classifier", "Diagnosis/response transfer", "Not executable", "No serialized weights or preprocessing object"],
    ], columns=["Dataset","Analysis","Status","Reason"])
    audit.to_csv(OUT/"External_validation_feasibility_audit.csv",index=False)
    axes[1].axis("off"); axes[1].set_title("B  External-validation audit",loc="left",weight="bold")
    tab=axes[1].table(cellText=audit[["Dataset","Status"]].values,colLabels=["Resource","Status"],loc="center",cellLoc="left",colLoc="left",colWidths=[.57,.35])
    tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1,1.35)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor("white"); cell.set_facecolor(LIGHT if r%2 else "white")
        if r==0: cell.set_facecolor("#D9EAF7"); cell.set_text_props(weight="bold")
    save_figure(fig, "ExternalValidation_Figure3_Sequence_Convergence_Audit")


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(stats_df, motifs, g301_clones):
    sig = stats_df.sort_values(["fdr", "p"]).head(30)
    payload = {
        "generated": "2026-08-29",
        "canonical_assets_modified": False,
        "analysis_unit": "participant or participant-sample",
        "multiple_testing": "Benjamini-Hochberg within prespecified analysis family",
        "top_results": sig.replace({np.nan: None}).to_dict("records"),
        "motif_results": motifs.replace({np.nan: None, np.inf: "Infinity"}).to_dict("records"),
        "gse301_blood_colon_shared_clonotypes": int(g301_clones.blood_colon_shared.sum()),
        "limitations": [
            "GSE261334 public metadata do not expose donor-level response labels.",
            "GSE125527 matrices were normalized to total counts across the locked module-gene universe because full 10,000-gene row sums are not supplied separately.",
            "GSE301689 contains only three Crohn disease donors; cross-tissue effects are descriptive and paired.",
            "SCP1690 public file streaming requires authentication; no analysis was fabricated from inaccessible cell-level data.",
            "The multi-organ Crohn B-cell study did not expose a public cell-level accession that could be harmonized reproducibly.",
            "Frozen classifier transfer was not attempted because serialized model weights and preprocessing objects were not found; no model was retrained.",
            "GSE125527 lacks germline-alignment fields, preventing true SHM validation; class switching and cross-program lineage occupancy were tested instead.",
        ],
    }
    (OUT / "External_validation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    g261_sample, g261_bins, g261_dose, g261_stats, corr = analyze_gse261()
    g125_sample, g125_bins, traffic, lineage, bdose, g125_stats = analyze_gse125()
    g301_metrics, g301_clones, g301_stats = analyze_gse301()
    motifs = analyze_jci()
    all_stats = finalize_stats(g261_stats, g125_stats, g301_stats)
    make_figures(g261_sample, g261_bins, g261_dose, g261_stats, corr,
                 g125_sample, g125_bins, traffic, lineage, bdose, g125_stats,
                 g301_metrics, g301_clones, g301_stats, motifs, all_stats)
    write_summary(all_stats, motifs, g301_clones)
    print(f"COMPLETE: {OUT}", flush=True)


if __name__ == "__main__":
    main()

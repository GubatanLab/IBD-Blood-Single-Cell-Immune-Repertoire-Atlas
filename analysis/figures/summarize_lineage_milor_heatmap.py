from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
OUT.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "UC vs control": Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD MiloR\UCvsControls\UCControlSCVI_milo_da_results.csv"),
    "CD vs control": Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD MiloR\CDvsControls\CDControlSCVI_milo_da_results.csv"),
    "CD vs UC": Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD MiloR\CDvsUC\CDvsUCSCVI_milo_da_results.csv"),
}


def lineage(state: str):
    state = str(state)
    if state.startswith(("CD4", "TReg")):
        return "CD4/Treg"
    if state.startswith("CD8") or state in {"MAIT", "gdT"}:
        return "CD8/innate-like T"
    b_states = (
        "Naive B", "Naive-IFN B", "Transitional B", "CD5+ B Cell",
        "Atypical memory B", "Switched memory B", "Non-switched memory B",
    )
    if state.startswith(b_states) or "Plasma B" in state:
        return "B/plasma"
    return None


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.nan
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")])


rows = []
for comparison, path in INPUTS.items():
    data = pd.read_csv(path, low_memory=False)
    data["lineage"] = data["Celltype"].map(lineage)
    data = data[data["lineage"].notna()].copy()
    data["annotation_purity_pass"] = pd.to_numeric(data["Celltype_fraction"], errors="coerce") >= 0.20
    data = data[data["annotation_purity_pass"]]
    data["significant"] = pd.to_numeric(data["SpatialFDR"], errors="coerce") < 0.05
    for (lineage_name, state), group in data.groupby(["lineage", "Celltype"], sort=False):
        significant = group[group["significant"]]
        n_total = len(group)
        n_sig = len(significant)
        n_positive = int((significant["logFC"] > 0).sum())
        n_negative = int((significant["logFC"] < 0).sum())
        dominant = max(n_positive, n_negative) / n_sig if n_sig else np.nan
        wm = weighted_median(significant["logFC"], significant["Celltype_fraction"]) if n_sig else np.nan
        rows.append(
            {
                "lineage": lineage_name,
                "cell_state": state,
                "comparison": comparison,
                "effect_definition": "positive log2 FC favors first-named group",
                "annotation_purity_threshold": 0.20,
                "n_neighborhoods": n_total,
                "n_significant_neighborhoods": n_sig,
                "fraction_significant": n_sig / n_total if n_total else np.nan,
                "n_significant_positive": n_positive,
                "n_significant_negative": n_negative,
                "directional_concordance": dominant,
                "median_significant_logFC": float(significant["logFC"].median()) if n_sig else np.nan,
                "purity_weighted_median_significant_logFC": wm,
                "minimum_SpatialFDR": float(significant["SpatialFDR"].min()) if n_sig else np.nan,
                "display_value": wm if n_sig >= 5 else np.nan,
                "display_eligible": n_sig >= 5,
            }
        )

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "Figure1_lineage_miloR_heatmap_summary.csv", index=False)
print(summary.groupby(["lineage", "comparison"])["display_eligible"].sum())

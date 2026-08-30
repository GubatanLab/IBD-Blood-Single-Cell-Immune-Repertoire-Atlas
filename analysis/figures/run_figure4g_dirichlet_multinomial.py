from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2


ROOT = Path(r"C:/path/to/private-manuscript-workspace")
ISO_PATH = Path(r"C:/path/to/private-legacy-manuscript-assets\Figure 5\BCR_isotype_proportions_recommended_manuscript_sample_level_values.csv")
SWITCH_PATH = Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 3 BCR\BCR Architecture Analyses\BCR isotype switching diagnosis comparisons\tables\bcr_isotype_switched_fraction_by_sample.csv")
META_PATH = ROOT / "High Impact Additional Analyses" / "Table_HI_integrated_participant_features.csv"
OUT = ROOT / "Trajectory Integrated Figure Set" / "Source Data"
OUT.mkdir(parents=True, exist_ok=True)

mapping = {
    "IGHM": "IgM", "IGHD": "IgD", "IGHA1": "IgA", "IGHA2": "IgA",
    "IGHG1": "IgG", "IGHG2": "IgG", "IGHG3": "IgG", "IGHG4": "IgG",
}
classes = ["IgM", "IgD", "IgA", "IgG"]

iso = pd.read_csv(ISO_PATH)
sw = pd.read_csv(SWITCH_PATH)[["SampleID", "Diagnosis1", "total_isotyped"]]
iso = iso[~iso.isotype.str.startswith("Total ")].copy()
iso["class"] = iso.isotype.map(mapping)
iso = iso[iso["class"].notna()].merge(sw, on=["SampleID", "Diagnosis1"], how="inner")
iso["count"] = np.rint(iso.prop * iso.total_isotyped).astype(int)
counts = iso.groupby(["SampleID", "Diagnosis1", "class"], as_index=False)["count"].sum()
wide = counts.pivot_table(index=["SampleID", "Diagnosis1"], columns="class", values="count", fill_value=0).reset_index()
for c in classes:
    if c not in wide:
        wide[c] = 0

meta = pd.read_csv(META_PATH)[["SampleID", "Age", "Sex", "bcr_total_cells"]]
d = wide.merge(meta, on="SampleID", how="inner")
d["Age_z"] = (d.Age - d.Age.mean()) / d.Age.std(ddof=0)
d["log_depth_z"] = (np.log1p(d.bcr_total_cells) - np.log1p(d.bcr_total_cells).mean()) / np.log1p(d.bcr_total_cells).std(ddof=0)
sex = pd.get_dummies(d.Sex, drop_first=True, dtype=float)

y = d[classes].to_numpy(float)


def design(include_cd=True, include_uc=True):
    cols = [np.ones(len(d)), d.Age_z.to_numpy(), d.log_depth_z.to_numpy()]
    names = ["Intercept", "Age_z", "log_depth_z"]
    for col in sex.columns:
        cols.append(sex[col].to_numpy())
        names.append(f"Sex_{col}")
    if include_cd:
        cols.append((d.Diagnosis1 == "CD").astype(float).to_numpy())
        names.append("Diagnosis_CD")
    if include_uc:
        cols.append((d.Diagnosis1 == "UC").astype(float).to_numpy())
        names.append("Diagnosis_UC")
    return np.column_stack(cols), names


def fit_dm(x):
    p = x.shape[1]
    k = y.shape[1]

    def objective(theta):
        beta = theta[:-1].reshape(p, k - 1)
        phi = np.exp(theta[-1])
        eta = x @ beta
        logits = np.column_stack([np.zeros(len(x)), eta])
        probs = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        alpha = np.clip(phi * probs, 1e-10, None)
        n = y.sum(axis=1)
        ll = (
            gammaln(phi) - gammaln(n + phi)
            + np.sum(gammaln(y + alpha) - gammaln(alpha), axis=1)
            + gammaln(n + 1) - np.sum(gammaln(y + 1), axis=1)
        )
        return -float(ll.sum())

    init = np.zeros(p * (k - 1) + 1)
    init[-1] = np.log(20)
    result = minimize(objective, init, method="L-BFGS-B", options={"maxiter": 4000, "ftol": 1e-11})
    if not result.success:
        raise RuntimeError(result.message)
    return result, -result.fun


x_null, names_null = design(False, False)
x_full, names_full = design(True, True)
x_no_cd, _ = design(False, True)
x_no_uc, _ = design(True, False)

fit_null, ll_null = fit_dm(x_null)
fit_full, ll_full = fit_dm(x_full)
fit_no_cd, ll_no_cd = fit_dm(x_no_cd)
fit_no_uc, ll_no_uc = fit_dm(x_no_uc)

global_lr = 2 * (ll_full - ll_null)
global_df = 2 * (y.shape[1] - 1)
global_p = chi2.sf(global_lr, global_df)

rows = [{
    "test": "Global diagnosis effect",
    "comparison": "CD and UC jointly versus Control",
    "logLik_full": ll_full,
    "logLik_reduced": ll_null,
    "LR_statistic": global_lr,
    "df": global_df,
    "p_value": global_p,
}]
for label, ll_red in [("CD vs Control", ll_no_cd), ("UC vs Control", ll_no_uc)]:
    lr = 2 * (ll_full - ll_red)
    rows.append({
        "test": "Diagnosis-specific joint composition effect",
        "comparison": label,
        "logLik_full": ll_full,
        "logLik_reduced": ll_red,
        "LR_statistic": lr,
        "df": y.shape[1] - 1,
        "p_value": chi2.sf(lr, y.shape[1] - 1),
    })
result = pd.DataFrame(rows)
mask = result.test.eq("Diagnosis-specific joint composition effect")
pv = result.loc[mask, "p_value"].to_numpy()
order = np.argsort(pv)
adj_sorted = np.maximum.accumulate(pv[order] * (len(pv) - np.arange(len(pv))))
adj = np.empty_like(pv)
adj[order] = np.minimum(adj_sorted, 1)
result.loc[mask, "Holm_q"] = adj
result.to_csv(OUT / "Figure4G_dirichlet_multinomial_composition_tests.csv", index=False)

pd.DataFrame({
    "metric": ["participants", "isotypes", "full_model_logLik", "dispersion_phi"],
    "value": [len(d), len(classes), ll_full, np.exp(fit_full.x[-1])],
}).to_csv(OUT / "Figure4G_dirichlet_multinomial_model_summary.csv", index=False)

print(result.to_string(index=False))

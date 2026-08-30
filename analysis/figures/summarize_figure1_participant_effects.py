from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
INPUT = SOURCE / "Figure1_participant_state_abundance.csv"
OUTPUT = SOURCE / "Figure1_participant_state_effects.csv"

STATE_ORDER = [
    "CD4 Naive",
    "TReg Cytotoxic",
    "CD8 Naive",
    "CD8 Tem GZMB+",
    "Switched memory B",
    "IgM Plasma B Cell",
]
COMPARISONS = ["UC vs control", "CD vs control", "CD vs UC"]


def coefficient_name(parameters, diagnosis):
    matches = [name for name in parameters if "Diagnosis1" in name and f"[T.{diagnosis}]" in name]
    if len(matches) != 1:
        raise ValueError(f"Unable to identify the {diagnosis} diagnosis coefficient: {matches}")
    return matches[0]


data = pd.read_csv(INPUT)
data = data[data["Diagnosis1"].isin(["Control", "CD", "UC"])].copy()
data["Diagnosis1"] = pd.Categorical(data["Diagnosis1"], categories=["Control", "CD", "UC"])
data["acquisition_series"] = data["acquisition_series"].fillna("Unknown").astype(str)

rows = []
for state in STATE_ORDER:
    state_data = data[data["cell_state"].eq(state)].copy()
    model = smf.ols(
        "logit_fraction ~ C(Diagnosis1, Treatment(reference='Control')) + C(acquisition_series)",
        data=state_data,
    ).fit(cov_type="HC3")
    parameters = list(model.params.index)
    cd_name = coefficient_name(parameters, "CD")
    uc_name = coefficient_name(parameters, "UC")
    covariance = model.cov_params()

    contrast_vectors = {}
    for comparison in COMPARISONS:
        vector = pd.Series(0.0, index=parameters)
        if comparison == "UC vs control":
            vector[uc_name] = 1.0
        elif comparison == "CD vs control":
            vector[cd_name] = 1.0
        else:
            vector[cd_name] = 1.0
            vector[uc_name] = -1.0
        contrast_vectors[comparison] = vector

    counts = state_data.groupby("Diagnosis1", observed=False)["PatientID"].nunique()
    for comparison, vector in contrast_vectors.items():
        estimate = float(vector @ model.params)
        variance = float(vector @ covariance @ vector)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        z_value = estimate / standard_error if standard_error > 0 else np.nan
        p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        rows.append(
            {
                "cell_state": state,
                "lineage": state_data["lineage"].iloc[0],
                "comparison": comparison,
                "effect": estimate,
                "standard_error": standard_error,
                "ci_low": estimate - 1.96 * standard_error,
                "ci_high": estimate + 1.96 * standard_error,
                "p_value": p_value,
                "n_control": int(counts.get("Control", 0)),
                "n_cd": int(counts.get("CD", 0)),
                "n_uc": int(counts.get("UC", 0)),
                "model": "participant-level OLS of logit-transformed within-lineage fraction; acquisition-series adjusted; HC3 covariance",
            }
        )

result = pd.DataFrame(rows)
valid = result["p_value"].notna()
result.loc[valid, "adjusted_p_value"] = multipletests(result.loc[valid, "p_value"], method="fdr_bh")[1]
result.to_csv(OUTPUT, index=False)
print(result[["cell_state", "comparison", "effect", "ci_low", "ci_high", "adjusted_p_value"]].to_string(index=False))

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
CSI = ROOT / "High Impact Additional Analyses" / "Clone State Interactions"
RNG = np.random.default_rng(20260827)

SELECTED = [
    "EOMES_ZEB2_inflammatory_CD8_TRM_like",
    "Effector_cytotoxicity",
    "Th1_Tc1_inflammatory",
]


def bh(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    ok = values.notna()
    if not ok.any():
        return out
    p = values[ok].to_numpy(float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0, 1)
    out.loc[ok] = adjusted
    return out


def estimate_linear_contrast(fit, contrast: np.ndarray) -> tuple[float, float, float, float, float]:
    beta = np.asarray(fit.params)
    cov = np.asarray(fit.cov_params())
    estimate = float(contrast @ beta)
    se = float(np.sqrt(max(contrast @ cov @ contrast, 0)))
    z = estimate / se if se > 0 else np.nan
    p = float(2 * st.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return estimate, se, estimate - 1.96 * se, estimate + 1.96 * se, p


def mean_design(fit, frame: pd.DataFrame) -> np.ndarray:
    design_info = fit.model.data.design_info
    matrix = patsy.build_design_matrices([design_info], frame, return_type="dataframe")[0]
    return matrix.to_numpy(float).mean(axis=0)


def clone_size_trends() -> None:
    dose = pd.read_csv(SRC / "Figure2_clone_size_module_dose_response.csv")
    dose = dose[dose["module"].isin(SELECTED)].copy()
    dose["clone_ord"] = dose["clone_bin"].map(
        {"Singleton": 0.0, "2 cells": 1.0, "3-4 cells": 2.0, ">=5 cells": 3.0}
    )
    rows: list[dict] = []
    for module, data in dose.groupby("module"):
        data = data.dropna(subset=["delta", "clone_ord", "SampleID", "Diagnosis1"]).copy()
        pooled = smf.ols("delta ~ 0 + C(SampleID) + clone_ord", data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data["SampleID"]}
        )
        name = "clone_ord"
        rows.append(
            {
                "module": module,
                "contrast": "Pooled ordered trend",
                "estimate": pooled.params[name],
                "SE": pooled.bse[name],
                "p_value": pooled.pvalues[name],
                "n_participants": data["SampleID"].nunique(),
            }
        )

        by_dx = smf.ols(
            "delta ~ 0 + C(SampleID) + clone_ord:C(Diagnosis1)", data=data
        ).fit(cov_type="cluster", cov_kwds={"groups": data["SampleID"]})
        slopes = {}
        for diagnosis in ["Control", "CD", "UC"]:
            term = f"clone_ord:C(Diagnosis1)[{diagnosis}]"
            slopes[diagnosis] = term
            rows.append(
                {
                    "module": module,
                    "contrast": f"{diagnosis} ordered trend",
                    "estimate": by_dx.params[term],
                    "SE": by_dx.bse[term],
                    "p_value": by_dx.pvalues[term],
                    "n_participants": data.loc[data["Diagnosis1"] == diagnosis, "SampleID"].nunique(),
                }
            )
        for first, second in [("CD", "Control"), ("UC", "Control"), ("CD", "UC")]:
            contrast = np.zeros(len(by_dx.params))
            contrast[by_dx.params.index.get_loc(slopes[first])] = 1
            contrast[by_dx.params.index.get_loc(slopes[second])] = -1
            est, se, _, _, p = estimate_linear_contrast(by_dx, contrast)
            rows.append(
                {
                    "module": module,
                    "contrast": f"{first} vs {second} trend interaction",
                    "estimate": est,
                    "SE": se,
                    "p_value": p,
                    "n_participants": data["SampleID"].nunique(),
                }
            )

    result = pd.DataFrame(rows)
    result["ci_low"] = result["estimate"] - 1.96 * result["SE"]
    result["ci_high"] = result["estimate"] + 1.96 * result["SE"]
    result["FDR"] = np.nan
    for family, mask in {
        "pooled": result["contrast"].eq("Pooled ordered trend"),
        "diagnosis": result["contrast"].str.endswith("ordered trend") & ~result["contrast"].eq("Pooled ordered trend"),
        "interaction": result["contrast"].str.contains("interaction"),
    }.items():
        result.loc[mask, "FDR"] = bh(result.loc[mask, "p_value"])
    result.to_csv(SRC / "Figure2_clone_size_trend_tests.csv", index=False)


def adjusted_state_enrichment() -> None:
    data = pd.read_csv(SRC / "Figure2_state_enrichment_by_participant.csv")
    data["receptor_depth"] = data["status_total_Expanded"] + data["status_total_Singleton"]
    rows = []
    for state, frame in data.groupby("state"):
        frame = frame.dropna(subset=["log2_odds_ratio", "Diagnosis1", "Batch", "receptor_depth"]).copy()
        fit = smf.ols(
            "log2_odds_ratio ~ C(Diagnosis1) + C(Batch) + np.log1p(receptor_depth)", data=frame
        ).fit(cov_type="HC3")
        contrast = mean_design(fit, frame)
        est, se, lo, hi, p = estimate_linear_contrast(fit, contrast)
        rows.append(
            {
                "state": state,
                "n_participants": frame["SampleID"].nunique(),
                "adjusted_log2_or": est,
                "SE": se,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "median_receptor_depth": frame["receptor_depth"].median(),
            }
        )
    result = pd.DataFrame(rows).sort_values("adjusted_log2_or", ascending=False)
    result["FDR"] = bh(result["p_value"])
    result.to_csv(SRC / "Figure2_state_enrichment_depth_batch_adjusted.csv", index=False)


def paired_program_tests() -> None:
    state = pd.read_csv(CSI / "Table_CSI1_state_matched_module_scores.csv")
    state = state[state["module"].isin(SELECTED)].copy()
    paired = (
        state.groupby(["SampleID", "Diagnosis1", "module", "compartment"], as_index=False)[
            ["Singleton", "Expanded"]
        ]
        .mean()
        .groupby(["SampleID", "Diagnosis1", "module"], as_index=False)[["Singleton", "Expanded"]]
        .mean()
    )
    paired["delta"] = paired["Expanded"] - paired["Singleton"]
    paired.to_csv(SRC / "Figure2_participant_paired_program_deltas.csv", index=False)
    rows = []
    for module, frame in paired.groupby("module"):
        values = frame["delta"].dropna().to_numpy()
        bootstrap = np.median(RNG.choice(values, size=(5000, len(values)), replace=True), axis=1)
        test = st.wilcoxon(values, zero_method="wilcox", correction=False, alternative="two-sided")
        rows.append(
            {
                "module": module,
                "n_participants": len(values),
                "median_delta": np.median(values),
                "ci_low": np.quantile(bootstrap, 0.025),
                "ci_high": np.quantile(bootstrap, 0.975),
                "p_value": test.pvalue,
            }
        )
    tests = pd.DataFrame(rows)
    tests["FDR"] = bh(tests["p_value"])
    tests.to_csv(SRC / "Figure2_paired_program_delta_tests.csv", index=False)


def adjusted_series_replication() -> None:
    data = pd.read_csv(CSI / "Table_CSI2_participant_expanded_minus_singleton_deltas.csv")
    data = data[(data["modality"] == "TCR") & (data["module"].isin(SELECTED[:2]))].copy()
    rows = []
    for module, frame in data.groupby("module"):
        frame = frame.dropna(subset=["delta", "acquisition_series", "Diagnosis1"]).copy()
        series = sorted(frame["acquisition_series"].unique(), key=lambda x: int(str(x).replace("S", "")))
        module_rows = []
        for acquisition_series in series:
            observed = frame.loc[frame["acquisition_series"] == acquisition_series, "delta"].to_numpy(float)
            n = len(observed)
            est = float(np.mean(observed))
            se = float(st.sem(observed))
            critical = float(st.t.ppf(0.975, n - 1))
            lo, hi = est - critical * se, est + critical * se
            p = float(st.ttest_1samp(observed, 0).pvalue)
            module_rows.append(
                {
                    "module": module,
                    "series": acquisition_series,
                    "n": n,
                    "effect": est,
                    "SE": se,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p,
                }
            )

        observed = frame["delta"].to_numpy(float)
        est = float(np.mean(observed))
        se = float(st.sem(observed))
        critical = float(st.t.ppf(0.975, len(observed) - 1))
        lo, hi = est - critical * se, est + critical * se
        p = float(st.ttest_1samp(observed, 0).pvalue)
        leave_out = [
            float(frame.loc[frame["acquisition_series"] != omitted, "delta"].mean())
            for omitted in series
        ]

        estimates = np.asarray([row["effect"] for row in module_rows])
        variances = np.asarray([row["SE"] ** 2 for row in module_rows])
        weights = 1 / np.clip(variances, 1e-12, None)
        q = float(np.sum(weights * (estimates - np.sum(weights * estimates) / np.sum(weights)) ** 2))
        df = max(len(estimates) - 1, 1)
        i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
        module_rows.append(
            {
                "module": module,
                "series": "Pooled",
                "n": frame["SampleID"].nunique(),
                "effect": est,
                "SE": se,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "I2": i2,
                "leave_one_series_out_low": min(leave_out),
                "leave_one_series_out_high": max(leave_out),
                "estimand": "Participant mean expanded-minus-singleton delta",
            }
        )
        rows.extend(module_rows)
    result = pd.DataFrame(rows)
    result["FDR"] = bh(result["p_value"])
    result.to_csv(SRC / "Figure2_series_replication_meta_analysis.csv", index=False)


def interaction_summary() -> None:
    interactions = pd.read_csv(CSI / "Table_CSI3_formal_interaction_models.csv")
    dx = interactions[
        (interactions["family"] == "TCR_diagnosis")
        & interactions["contrast"].str.contains("interaction", na=False)
    ].copy()
    pd.DataFrame(
        [
            {
                "n_tests": len(dx),
                "minimum_interaction_FDR": dx["FDR"].min(),
                "minimum_interaction_p": dx["p_value"].min(),
                "n_interaction_FDR_lt_0_05": int((dx["FDR"] < 0.05).sum()),
            }
        ]
    ).to_csv(SRC / "Figure2_diagnosis_interaction_summary.csv", index=False)


def main() -> None:
    clone_size_trends()
    adjusted_state_enrichment()
    paired_program_tests()
    adjusted_series_replication()
    interaction_summary()
    print(SRC)


if __name__ == "__main__":
    main()

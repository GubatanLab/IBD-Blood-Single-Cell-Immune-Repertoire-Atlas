from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
RAW = SRC / "Figure2B_paired4_clone_size_participant_deltas.csv"
RNG = np.random.default_rng(20260829)

MODULES = [
    "EOMES_ZEB2_inflammatory_CD8_TRM_like",
    "Effector_cytotoxicity",
    "Th1_Tc1_inflammatory",
    "GZMK_inflammatory_memory",
]
BINS = ["Singleton", "2 cells", "3-4 cells", ">=5 cells"]


def bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    x = p[ok].to_numpy(float)
    order = np.argsort(x)
    ranked = x[order] * len(x) / np.arange(1, len(x) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0, 1)
    out.loc[ok] = adjusted
    return out


def bootstrap_median(values: np.ndarray, n_boot: int = 4000) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    sims = np.median(RNG.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return float(np.median(values)), float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


def main() -> None:
    data = pd.read_csv(RAW)
    data = data[data["module"].isin(MODULES) & data["clone_bin"].isin(BINS)].copy()
    rows: list[dict] = []
    for module in MODULES:
        dm = data[data["module"].eq(module)]
        for diagnosis in ["Control", "CD", "UC", "Pooled"]:
            dd = dm if diagnosis == "Pooled" else dm[dm["Diagnosis1"].eq(diagnosis)]
            for clone_bin in BINS:
                values = dd.loc[dd["clone_bin"].eq(clone_bin), "delta"].dropna().to_numpy(float)
                median, lo, hi = bootstrap_median(values)
                rows.append(
                    {
                        "module": module,
                        "Diagnosis": diagnosis,
                        "clone_bin": clone_bin,
                        "n": len(values),
                        "median": median,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(SRC / "Figure2B_paired4_clone_size_dose_response_displayed.csv", index=False)

    data["clone_ord"] = data["clone_bin"].map({b: i for i, b in enumerate(BINS)}).astype(float)
    trends: list[dict] = []
    for module in MODULES:
        dm = data[data["module"].eq(module)].dropna(subset=["delta", "clone_ord", "SampleID"])
        fit = smf.ols("delta ~ 0 + C(SampleID) + clone_ord", data=dm).fit(
            cov_type="cluster", cov_kwds={"groups": dm["SampleID"]}
        )
        estimate = float(fit.params["clone_ord"])
        se = float(fit.bse["clone_ord"])
        trends.append(
            {
                "module": module,
                "contrast": "Pooled ordered trend",
                "estimate": estimate,
                "SE": se,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
                "p_value": float(fit.pvalues["clone_ord"]),
                "n_participants": int(dm["SampleID"].nunique()),
            }
        )
    trend = pd.DataFrame(trends)
    trend["FDR"] = bh(trend["p_value"])
    trend.to_csv(SRC / "Figure2B_paired4_clone_size_trend_tests.csv", index=False)
    print(summary.groupby(["module", "Diagnosis"])["n"].max().unstack(fill_value=0))
    print(trend.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Clone-aware Th17/Treg analyses and participant-level helper-B coupling."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "High Impact Additional Analyses" / "Th17 Treg B Helper Analyses"
PDF_OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260829)

N_PERM_SHARING = 10000
N_PERM_CORRELATION = 5000

T_LABELS = {
    "mean_Th17_pathogenic_in_Th17": "Pathogenic Th17",
    "mean_Th17_conventional_in_Th17": "Conventional Th17",
    "mean_Treg_suppressive_in_Treg": "Suppressive Treg",
    "mean_Treg_reprogramming_in_Treg": "Reprogrammed Treg",
    "mean_Tph_Tfh_help_all_CD4": "Tph/Tfh help",
}

B_LABELS = {
    "b_mean_Plasmablast_plasma_differentiation": "Plasma differentiation",
    "b_mean_IgA_mucosal_plasma_cell": "IgA mucosal plasma",
    "b_mean_IgG_inflammatory_plasma_cell": "IgG inflammatory plasma",
    "b_mean_Atypical_memory_CD11c_like": "Atypical memory",
    "b_mean_B_cell_antigen_presentation": "Antigen presentation",
    "bcr_switched_fraction": "Class-switched BCR",
    "bcr_SHM_rate": "BCR SHM",
}


def fdr_bh(values):
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    po = p[ok]
    order = np.argsort(po)
    ranked = po[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    undo = np.empty_like(order)
    undo[order] = np.arange(len(order))
    out[ok] = q[undo]
    return out


def zscore_safe(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=1)
    return (x - np.nanmean(x)) / sd if np.isfinite(sd) and sd > 0 else np.zeros_like(x)


def design_matrix(df, numeric, categorical):
    parts = [pd.Series(1.0, index=df.index, name="intercept")]
    for col in numeric:
        parts.append(pd.to_numeric(df[col], errors="coerce").rename(col))
    if categorical:
        cats = df[categorical].fillna("Unknown").astype(str)
        parts.append(pd.get_dummies(cats, drop_first=True, dtype=float))
    return pd.concat(parts, axis=1).astype(float)


def residualize_rank(df, value, numeric_covars, categorical_covars):
    y = stats.rankdata(pd.to_numeric(df[value], errors="coerce").to_numpy(float))
    x = design_matrix(df, numeric_covars, categorical_covars).to_numpy(float)
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def partial_spearman_block(
    data, xcol, ycol, numeric_covars, categorical_covars,
    block="acquisition_series", nperm=N_PERM_CORRELATION,
):
    cols = list(dict.fromkeys([xcol, ycol, block] + numeric_covars + categorical_covars))
    z = data[cols].replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).copy()
    z[block] = z[block].fillna("Unknown").astype(str)
    for c in categorical_covars:
        z[c] = z[c].fillna("Unknown").astype(str)
    z[xcol] = pd.to_numeric(z[xcol], errors="coerce")
    z[ycol] = pd.to_numeric(z[ycol], errors="coerce")
    for c in numeric_covars:
        z[c] = pd.to_numeric(z[c], errors="coerce")
    z = z.dropna(subset=[xcol, ycol] + numeric_covars)
    if len(z) < 20 or z[xcol].nunique() < 4 or z[ycol].nunique() < 4:
        return len(z), np.nan, np.nan, np.nan, np.nan, None
    rx = residualize_rank(z, xcol, numeric_covars, categorical_covars)
    ry = residualize_rank(z, ycol, numeric_covars, categorical_covars)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    groups = [np.flatnonzero(z[block].to_numpy() == b) for b in z[block].unique()]
    null = np.empty(nperm)
    for i in range(nperm):
        yp = ry.copy()
        for ix in groups:
            if len(ix) > 1:
                yp[ix] = yp[RNG.permutation(ix)]
        null[i] = np.corrcoef(rx, yp)[0, 1]
    p = (1 + np.sum(np.abs(null) >= abs(rho))) / (nperm + 1)
    zr = np.arctanh(np.clip(rho, -0.999, 0.999))
    df_resid = max(len(z) - len(numeric_covars) - len(categorical_covars) - 3, 1)
    se = 1 / math.sqrt(df_resid)
    lo, hi = np.tanh([zr - 1.96 * se, zr + 1.96 * se])
    return len(z), rho, float(lo), float(hi), float(p), (z, rx, ry)


def clone_sharing_permutation(axis_cells):
    d = axis_cells.copy()
    d["clone_key"] = d["SampleID"].astype(str) + "::" + d["paired_clone_id"].astype(str)
    clone_code, clone_names = pd.factorize(d["clone_key"], sort=True)
    sample_groups = [np.asarray(ix, dtype=int) for ix in d.groupby("SampleID", sort=False).indices.values()]
    is_th17 = (d["axis_state"].astype(str) == "Th17-like").to_numpy(np.int8)
    total = np.bincount(clone_code, minlength=len(clone_names))
    clone_dx = d.groupby("clone_key", sort=True)["Diagnosis1"].first().reindex(clone_names).to_numpy()
    masks = {
        "All": np.ones(len(clone_names), dtype=bool),
        "IBD": np.isin(clone_dx, ["CD", "UC"]),
        "CD": clone_dx == "CD",
        "UC": clone_dx == "UC",
        "Control": clone_dx == "Control",
    }

    def count_mixed(labels):
        n_th17 = np.bincount(clone_code, weights=labels, minlength=len(clone_names))
        mixed = (n_th17 > 0) & (n_th17 < total)
        return np.array([mixed[masks[g]].sum() for g in masks], dtype=float)

    observed = count_mixed(is_th17)
    null = np.empty((N_PERM_SHARING, len(masks)), dtype=float)
    for i in range(N_PERM_SHARING):
        perm = is_th17.copy()
        for ix in sample_groups:
            if len(ix) > 1:
                perm[ix] = perm[RNG.permutation(ix)]
        null[i] = count_mixed(perm)

    rows = []
    for j, group in enumerate(masks):
        mu, sd = null[:, j].mean(), null[:, j].std(ddof=1)
        p_enrich = (1 + np.sum(null[:, j] >= observed[j])) / (N_PERM_SHARING + 1)
        p_deplete = (1 + np.sum(null[:, j] <= observed[j])) / (N_PERM_SHARING + 1)
        rows.append({
            "group": group,
            "observed_mixed_clones": int(observed[j]),
            "null_mean": mu,
            "null_sd": sd,
            "null_q025": np.quantile(null[:, j], 0.025),
            "null_q975": np.quantile(null[:, j], 0.975),
            "observed_to_null_ratio": observed[j] / mu if mu > 0 else np.nan,
            "z_score": (observed[j] - mu) / sd if sd > 0 else np.nan,
            "enrichment_p": p_enrich,
            "depletion_p": p_deplete,
            "two_sided_p": min(1.0, 2 * min(p_enrich, p_deplete)),
            "permutations": N_PERM_SHARING,
        })
    out = pd.DataFrame(rows)
    out["FDR"] = fdr_bh(out["two_sided_p"])
    out.to_csv(OUT / "Table_TB6_clone_sharing_permutation.csv", index=False)
    return out


def clone_size_models(clone_state):
    specs = {
        "Th17_conventional": "Th17-like",
        "Th17_pathogenic": "Th17-like",
        "Treg_suppressive": "Treg",
        "Treg_reprogramming": "Treg",
    }
    rows = []
    for module, axis in specs.items():
        base = clone_state[clone_state["axis_state"] == axis].copy()
        base["score"] = pd.to_numeric(base[module], errors="coerce")
        base["log2_clone_size"] = np.log2(pd.to_numeric(base["full_cd4_clone_size"], errors="coerce"))
        for group in ["All", "IBD", "CD", "UC", "Control"]:
            z = base.copy()
            if group == "IBD":
                z = z[z["Diagnosis1"].isin(["CD", "UC"])]
            elif group != "All":
                z = z[z["Diagnosis1"] == group]
            z = z.dropna(subset=["score", "log2_clone_size", "SampleID", "state"])
            if len(z) < 50 or z["SampleID"].nunique() < 10 or z["log2_clone_size"].nunique() < 2:
                continue
            try:
                fit = smf.ols("score ~ log2_clone_size + C(state) + C(SampleID)", data=z).fit(
                    cov_type="cluster", cov_kwds={"groups": z["SampleID"]}
                )
            except Exception:
                fit = smf.ols("score ~ log2_clone_size + C(state) + C(SampleID)", data=z).fit(cov_type="HC3")
            rows.append({
                "module": module, "axis_state": axis, "group": group,
                "n_clone_states": len(z), "n_participants": z["SampleID"].nunique(),
                "beta_per_doubling": fit.params.get("log2_clone_size", np.nan),
                "robust_SE": fit.bse.get("log2_clone_size", np.nan),
                "ci_low": fit.conf_int().loc["log2_clone_size", 0] if "log2_clone_size" in fit.params else np.nan,
                "ci_high": fit.conf_int().loc["log2_clone_size", 1] if "log2_clone_size" in fit.params else np.nan,
                "p_value": fit.pvalues.get("log2_clone_size", np.nan),
            })
    out = pd.DataFrame(rows)
    out["FDR_within_group"] = out.groupby("group")["p_value"].transform(fdr_bh)
    out.to_csv(OUT / "Table_TB7_clone_size_models.csv", index=False)
    return out


def expanded_singleton_tests(delta):
    relevant = {
        "Th17_conventional": "Th17-like", "Th17_pathogenic": "Th17-like",
        "Treg_suppressive": "Treg", "Treg_reprogramming": "Treg",
    }
    z = delta[delta.apply(lambda r: relevant.get(r["module"]) == r["axis_state"], axis=1)].copy()
    participant = z.groupby(["SampleID", "Diagnosis1", "acquisition_series", "module"], as_index=False)[
        "expanded_minus_singleton"
    ].mean()
    rows = []
    for module in relevant:
        for group in ["All", "IBD", "CD", "UC", "Control"]:
            d = participant[participant["module"] == module]
            if group == "IBD":
                d = d[d["Diagnosis1"].isin(["CD", "UC"])]
            elif group != "All":
                d = d[d["Diagnosis1"] == group]
            x = pd.to_numeric(d["expanded_minus_singleton"], errors="coerce").dropna().to_numpy()
            if len(x) < 5:
                continue
            try:
                p = stats.wilcoxon(x, alternative="two-sided").pvalue
            except ValueError:
                p = 1.0
            boot = np.array([np.mean(RNG.choice(x, size=len(x), replace=True)) for _ in range(5000)])
            rows.append({
                "module": module, "group": group, "n_participants": len(x),
                "mean_delta": np.mean(x), "median_delta": np.median(x),
                "bootstrap_ci_low": np.quantile(boot, 0.025),
                "bootstrap_ci_high": np.quantile(boot, 0.975),
                "wilcoxon_p": p,
            })
    out = pd.DataFrame(rows)
    out["FDR_within_group"] = out.groupby("group")["wilcoxon_p"].transform(fdr_bh)
    out.to_csv(OUT / "Table_TB8_expanded_singleton_tests.csv", index=False)
    return out


def build_analysis_dataset():
    t = pd.read_csv(OUT / "Table_TB4_participant_T_features.csv")
    ra = ROOT / "High Impact Additional Analyses" / "Literature Guided Ranked Analyses"
    b = pd.read_csv(ra / "Table_RA1_Bcell_response_by_participant.csv")
    integrated = pd.read_csv(ROOT / "High Impact Additional Analyses" / "Table_HI_integrated_participant_features.csv")
    clinical = pd.read_csv(ROOT / "High Impact Additional Analyses" / "Priority Analyses" / "Table_PA_clinical_metadata.csv")
    bcols = ["SampleID", "b_cells"] + [c for c in B_LABELS if c.startswith("b_mean_")]
    icols = ["SampleID", "bcr_switched_fraction", "bcr_SHM_rate", "bcr_IgA_fraction", "bcr_IgG_fraction"]
    d = t.merge(b[bcols], on="SampleID", how="inner")
    d = d.merge(integrated[icols], on="SampleID", how="left")
    d = d.merge(clinical[["SampleID", "Calprotectin"]], on="SampleID", how="left")
    d["sex_male"] = (d["Sex"].astype(str).str.upper() == "M").astype(float)
    d["log_cd4_cells"] = np.log1p(pd.to_numeric(d["cd4_cells"], errors="coerce"))
    d["log_b_cells"] = np.log1p(pd.to_numeric(d["b_cells"], errors="coerce"))
    d["log_calprotectin"] = np.log1p(pd.to_numeric(d["Calprotectin"], errors="coerce"))
    d.to_csv(OUT / "Table_TB9_T_B_analysis_dataset.csv", index=False)
    return d


def targeted_correlations(data):
    t_features = [
        "mean_Th17_pathogenic_in_Th17", "mean_Th17_conventional_in_Th17",
        "mean_Treg_suppressive_in_Treg", "mean_Treg_reprogramming_in_Treg",
        "mean_Tph_Tfh_help_all_CD4",
    ]
    b_features = list(B_LABELS)
    rows = []
    for group in ["IBD", "CD", "UC", "Control"]:
        d = data[data["Diagnosis1"].isin(["CD", "UC"])] if group == "IBD" else data[data["Diagnosis1"] == group]
        numeric = ["Age", "sex_male", "log_cd4_cells", "log_b_cells"]
        categorical = ["Inflammation1", "Biologic", "acquisition_series"] + (
            ["Diagnosis1"] if group == "IBD" else []
        )
        for x in t_features:
            for y in b_features:
                n, rho, lo, hi, p, _ = partial_spearman_block(d, x, y, numeric, categorical)
                rows.append({
                    "group": group, "T_feature": x, "T_label": T_LABELS[x],
                    "B_feature": y, "B_label": B_LABELS[y], "n": n,
                    "partial_rho": rho, "ci_low": lo, "ci_high": hi,
                    "series_block_permutation_p": p,
                })
    out = pd.DataFrame(rows)
    out["FDR_within_group"] = out.groupby("group")["series_block_permutation_p"].transform(fdr_bh)
    out.to_csv(OUT / "Table_TB10_targeted_T_B_correlations.csv", index=False)
    return out


def fit_restraint_model(z, outcome, leave_out=None):
    features = [
        "mean_Tph_Tfh_help_all_CD4",
        "mean_Th17_pathogenic_in_Th17",
        "mean_Treg_suppressive_in_Treg",
    ]
    cols = [outcome] + features + ["Age", "sex_male", "log_cd4_cells", "log_b_cells",
                                      "Diagnosis1", "Inflammation1", "Biologic", "acquisition_series"]
    d = z[cols].copy().replace([np.inf, -np.inf], np.nan)
    if leave_out is not None:
        d = d[d["acquisition_series"].astype(str) != str(leave_out)]
    for c in ["Diagnosis1", "Inflammation1", "Biologic", "acquisition_series"]:
        d[c] = d[c].fillna("Unknown").astype(str)
    d = d.dropna(subset=[outcome] + features + ["Age", "sex_male", "log_cd4_cells", "log_b_cells"])
    if len(d) < 40:
        return None, d
    d["y_z"] = zscore_safe(d[outcome])
    xnames = []
    for i, feat in enumerate(features):
        name = f"x{i}_z"
        d[name] = zscore_safe(d[feat])
        xnames.append(name)
    formula = "y_z ~ " + " + ".join(xnames) + (
        " + Age + sex_male + log_cd4_cells + log_b_cells + C(Diagnosis1)"
        " + C(Inflammation1) + C(Biologic) + C(acquisition_series)"
    )
    return smf.ols(formula, data=d).fit(cov_type="HC3"), d


def regulatory_restraint_models(data):
    ibd = data[data["Diagnosis1"].isin(["CD", "UC"])].copy()
    outcomes = [c for c in B_LABELS if c.startswith("b_mean_")]
    features = [
        "mean_Tph_Tfh_help_all_CD4",
        "mean_Th17_pathogenic_in_Th17",
        "mean_Treg_suppressive_in_Treg",
    ]
    rows, loo = [], []
    for outcome in outcomes:
        fit, z = fit_restraint_model(ibd, outcome)
        if fit is None:
            continue
        for i, feat in enumerate(features):
            term = f"x{i}_z"
            rows.append({
                "B_feature": outcome, "B_label": B_LABELS[outcome],
                "T_feature": feat, "T_label": T_LABELS[feat],
                "n": len(z), "standardized_beta": fit.params.get(term, np.nan),
                "robust_SE": fit.bse.get(term, np.nan),
                "ci_low": fit.conf_int().loc[term, 0], "ci_high": fit.conf_int().loc[term, 1],
                "p_value": fit.pvalues.get(term, np.nan),
            })
        for series in sorted(ibd["acquisition_series"].dropna().astype(str).unique()):
            fit_loo, z_loo = fit_restraint_model(ibd, outcome, leave_out=series)
            if fit_loo is None:
                continue
            for i, feat in enumerate(features):
                term = f"x{i}_z"
                loo.append({
                    "B_feature": outcome, "B_label": B_LABELS[outcome],
                    "T_feature": feat, "T_label": T_LABELS[feat],
                    "left_out_series": series, "n": len(z_loo),
                    "standardized_beta": fit_loo.params.get(term, np.nan),
                })
    out = pd.DataFrame(rows)
    out["FDR"] = fdr_bh(out["p_value"])
    loo = pd.DataFrame(loo)
    if not loo.empty:
        summary = loo.groupby(["B_feature", "B_label", "T_feature", "T_label"], as_index=False).agg(
            loo_beta_min=("standardized_beta", "min"),
            loo_beta_max=("standardized_beta", "max"),
            loo_median_beta=("standardized_beta", "median"),
            positive_fraction=("standardized_beta", lambda x: np.mean(np.asarray(x) > 0)),
            n_leave_one_series_out=("standardized_beta", "size"),
        )
        out = out.merge(summary, on=["B_feature", "B_label", "T_feature", "T_label"], how="left")
    out.to_csv(OUT / "Table_TB11_regulatory_restraint_models.csv", index=False)
    loo.to_csv(OUT / "Table_TB12_leave_one_series_out.csv", index=False)
    return out, loo


def clinical_associations(data):
    ibd = data[data["Diagnosis1"].isin(["CD", "UC"])].copy()
    features = list(T_LABELS)
    rows = []
    numeric = ["Age", "sex_male", "log_cd4_cells", "log_b_cells"]
    categorical = ["Diagnosis1", "Inflammation1", "Biologic", "acquisition_series"]
    for x in features:
        n, rho, lo, hi, p, _ = partial_spearman_block(
            ibd, x, "log_calprotectin", numeric, categorical
        )
        rows.append({
            "feature": x, "label": T_LABELS[x], "outcome": "log1p_calprotectin",
            "n": n, "partial_rho": rho, "ci_low": lo, "ci_high": hi,
            "series_block_permutation_p": p,
        })
    out = pd.DataFrame(rows)
    out["FDR"] = fdr_bh(out["series_block_permutation_p"])
    out.to_csv(OUT / "Table_TB13_clinical_associations.csv", index=False)
    return out


def star(q):
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def build_figure_h1(clone_axis, sharing, size_models, delta_tests):
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    fig = plt.figure(figsize=(7.05, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    comp = clone_axis.groupby(["Diagnosis1", "axis_clone_class"]).size().unstack(fill_value=0)
    order = ["CD", "UC", "Control"]
    classes = ["Th17-only", "Treg-only", "Mixed Th17/Treg"]
    comp = comp.reindex(index=order, columns=classes, fill_value=0)
    pct = comp.div(comp.sum(axis=1), axis=0) * 100
    colors = ["#D95F02", "#1B9E77", "#7570B3"]
    bottom = np.zeros(len(pct))
    for cls, color in zip(classes, colors):
        ax_a.bar(order, pct[cls], bottom=bottom, color=color, label=cls, width=0.72)
        bottom += pct[cls].to_numpy()
    for i, dx in enumerate(order):
        ax_a.text(i, 101.5, f"mixed={int(comp.loc[dx, 'Mixed Th17/Treg'])}",
                  ha="center", va="bottom", fontsize=6.2, color="#5E4FA2")
    ax_a.set_ylim(0, 106)
    ax_a.set_ylabel("Exact paired axis clones (%)")
    ax_a.set_title("A  Th17/Treg clone-state composition", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax_a.grid(axis="x", visible=False)

    sh = sharing[sharing["group"].isin(["IBD", "CD", "UC", "Control"])].copy()
    y = np.arange(len(sh))
    ax_b.errorbar(sh["null_mean"], y,
                  xerr=[sh["null_mean"] - sh["null_q025"], sh["null_q975"] - sh["null_mean"]],
                  fmt="o", color="#777777", capsize=3, label="Permutation null (95%)")
    ax_b.scatter(sh["observed_mixed_clones"], y, marker="D", s=34, color="#7570B3", label="Observed")
    ax_b.set_yticks(y, sh["group"])
    ax_b.set_xlabel("Mixed exact clones")
    ax_b.set_title("B  Exact-clone sharing null", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=6.2, loc="upper right")
    xmax = max(float(sh["null_q975"].max()), float(sh["observed_mixed_clones"].max())) + 5
    ax_b.set_xlim(-0.5, xmax)
    for i, r in sh.reset_index(drop=True).iterrows():
        ax_b.text(xmax - 0.3, i, f"P={r['two_sided_p']:.3g}",
                  ha="right", va="center", fontsize=6.3)

    sm = size_models[size_models["group"] == "IBD"].copy()
    sm["label"] = sm["module"].map({
        "Th17_conventional": "Conventional Th17", "Th17_pathogenic": "Pathogenic Th17",
        "Treg_suppressive": "Suppressive Treg", "Treg_reprogramming": "Reprogrammed Treg",
    })
    sm = sm.sort_values("beta_per_doubling")
    y = np.arange(len(sm))
    ax_c.errorbar(sm["beta_per_doubling"], y,
                  xerr=[sm["beta_per_doubling"] - sm["ci_low"], sm["ci_high"] - sm["beta_per_doubling"]],
                  fmt="o", color="#2C7FB8", capsize=3)
    ax_c.axvline(0, color="black", lw=0.8)
    ax_c.set_yticks(y, sm["label"])
    ax_c.set_xlabel("Program change per clone-size doubling")
    ax_c.set_title("C  Clone-size dose response in IBD", loc="left", fontweight="bold")
    for i, r in sm.reset_index(drop=True).iterrows():
        ax_c.text(r["ci_high"] + 0.002, i, star(r["FDR_within_group"]), va="center", fontsize=8)

    dt = delta_tests[delta_tests["group"] == "IBD"].copy()
    dt["label"] = dt["module"].map({
        "Th17_conventional": "Conventional Th17", "Th17_pathogenic": "Pathogenic Th17",
        "Treg_suppressive": "Suppressive Treg", "Treg_reprogramming": "Reprogrammed Treg",
    })
    dt = dt.sort_values("mean_delta")
    y = np.arange(len(dt))
    ax_d.errorbar(dt["mean_delta"], y,
                  xerr=[dt["mean_delta"] - dt["bootstrap_ci_low"], dt["bootstrap_ci_high"] - dt["mean_delta"]],
                  fmt="o", color="#D95F02", capsize=3)
    ax_d.axvline(0, color="black", lw=0.8)
    ax_d.set_yticks(y, [f"{a} (n={n})" for a, n in zip(dt["label"], dt["n_participants"])])
    ax_d.set_xlabel("Expanded - singleton score")
    ax_d.set_title("D  State-matched expansion", loc="left", fontweight="bold")
    for i, r in dt.reset_index(drop=True).iterrows():
        ax_d.text(r["bootstrap_ci_high"] + 0.002, i, star(r["FDR_within_group"]), va="center", fontsize=8)

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.tick_params(labelsize=7)
    fig.savefig(OUT / "Figure_H1_Th17_Treg_clonotype_architecture.png", dpi=400)
    fig.savefig(PDF_OUT / "Figure_H1_Th17_Treg_clonotype_architecture.pdf")
    plt.close(fig)


def build_figure_h2(data, correlations, restraint):
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    fig = plt.figure(figsize=(7.05, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1])
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    ibd = correlations[correlations["group"] == "IBD"].copy()
    t_order = ["Pathogenic Th17", "Conventional Th17", "Suppressive Treg", "Reprogrammed Treg", "Tph/Tfh help"]
    b_order = list(B_LABELS.values())
    mat = ibd.pivot(index="T_label", columns="B_label", values="partial_rho").reindex(index=t_order, columns=b_order)
    ann = mat.copy().astype(object)
    for i, tl in enumerate(mat.index):
        for j, bl in enumerate(mat.columns):
            row = ibd[(ibd["T_label"] == tl) & (ibd["B_label"] == bl)]
            ann.iloc[i, j] = "" if row.empty or not np.isfinite(mat.iloc[i, j]) else f"{mat.iloc[i, j]:.2f}{star(row.iloc[0]['FDR_within_group'])}"
    sns.heatmap(mat, cmap="vlag", center=0, vmin=-0.55, vmax=0.55, annot=ann, fmt="",
                annot_kws={"fontsize": 5.8}, cbar_kws={"label": "Partial Spearman rho", "shrink": 0.7}, ax=ax_a)
    ax_a.set_title("A  Targeted T-cell-B-cell coupling in IBD", loc="left", fontweight="bold")
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.tick_params(axis="x", rotation=55, labelsize=6.3)
    ax_a.tick_params(axis="y", rotation=0, labelsize=6.5)

    rr = restraint.copy()
    tcolors = {"Tph/Tfh help": "#377EB8", "Pathogenic Th17": "#D95F02", "Suppressive Treg": "#1B9E77"}
    outcomes = list(reversed(["Plasma differentiation", "IgA mucosal plasma", "IgG inflammatory plasma", "Atypical memory", "Antigen presentation"]))
    ypos = {label: i for i, label in enumerate(outcomes)}
    offsets = {"Tph/Tfh help": -0.18, "Pathogenic Th17": 0, "Suppressive Treg": 0.18}
    for tl in offsets:
        z = rr[rr["T_label"] == tl]
        yy = np.array([ypos[x] for x in z["B_label"]]) + offsets[tl]
        ax_b.errorbar(z["standardized_beta"], yy,
                      xerr=[z["standardized_beta"] - z["ci_low"], z["ci_high"] - z["standardized_beta"]],
                      fmt="o", color=tcolors[tl], capsize=2, label=tl, ms=4)
    ax_b.axvline(0, color="black", lw=0.8)
    ax_b.set_yticks(range(len(outcomes)), outcomes)
    ax_b.set_xlabel("Adjusted standardized beta")
    ax_b.set_title("B  Independent helper model", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=5.7, loc="upper center",
                bbox_to_anchor=(0.5, -0.14), ncol=3, columnspacing=0.8, handletextpad=0.3)
    ax_b.tick_params(labelsize=6.5)

    top = ibd.dropna(subset=["partial_rho", "FDR_within_group"]).sort_values(
        ["FDR_within_group", "series_block_permutation_p"]
    ).iloc[0]
    numeric = ["Age", "sex_male", "log_cd4_cells", "log_b_cells"]
    categorical = ["Inflammation1", "Biologic", "Diagnosis1"]
    subset = data[data["Diagnosis1"].isin(["CD", "UC"])]
    _, _, _, _, _, extra = partial_spearman_block(
        subset, top["T_feature"], top["B_feature"], numeric, categorical, nperm=250
    )
    z, rx, ry = extra
    plot = pd.DataFrame({"T residual rank": rx, "B residual rank": ry, "Diagnosis": z["Diagnosis1"].to_numpy()})
    palette = {"CD": "#D95F02", "UC": "#7570B3"}
    sns.scatterplot(data=plot, x="T residual rank", y="B residual rank", hue="Diagnosis",
                    palette=palette, s=24, alpha=0.8, ax=ax_c)
    sns.regplot(data=plot, x="T residual rank", y="B residual rank", scatter=False,
                line_kws={"color": "black", "lw": 1}, ax=ax_c)
    ax_c.set_title(f"C  Strongest prespecified association\n{top['T_label']} vs {top['B_label']}",
                   loc="left", fontweight="bold")
    ax_c.text(0.02, 0.98, f"rho={top['partial_rho']:.2f}; FDR={top['FDR_within_group']:.3g}",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=7)
    ax_c.legend(frameon=False, fontsize=6.5)
    ax_c.tick_params(labelsize=7)

    robust = restraint.copy()
    robust["effect"] = robust["T_label"] + " -> " + robust["B_label"]
    robust = robust.sort_values("FDR").head(8).sort_values("standardized_beta")
    y = np.arange(len(robust))
    ax_d.hlines(y, robust["loo_beta_min"], robust["loo_beta_max"], color="#AAAAAA", lw=2)
    ax_d.scatter(robust["standardized_beta"], y, color="#222222", s=22, zorder=3)
    ax_d.axvline(0, color="black", lw=0.8)
    ax_d.set_yticks(y, robust["effect"], fontsize=5.7)
    ax_d.set_xlabel("Full beta; leave-one-series-out range")
    ax_d.set_title("D  Acquisition-series robustness", loc="left", fontweight="bold")
    ax_d.tick_params(axis="x", labelsize=7)

    fig.savefig(OUT / "Figure_H2_Th17_Treg_B_helper_coupling.png", dpi=400)
    fig.savefig(PDF_OUT / "Figure_H2_Th17_Treg_B_helper_coupling.pdf")
    plt.close(fig)


def write_summary(sharing, size_models, delta_tests, correlations, restraint, clinical):
    lines = [
        "Th17/Treg clonotype and B-helper analysis summary",
        "================================================",
        "",
        "Exact paired alpha/beta clonotypes; participant is the inferential unit.",
        "Sharing null: 10,000 within-participant state-label permutations.",
        "T-B associations: rank residualization plus 5,000 acquisition-series-blocked permutations.",
        "",
    ]
    for _, r in sharing.iterrows():
        lines.append(
            f"{r.group} mixed clones: observed {r.observed_mixed_clones}, null mean {r.null_mean:.2f}, "
            f"ratio {r.observed_to_null_ratio:.2f}, two-sided P={r.two_sided_p:.4g}."
        )
    lines.extend(["", "IBD clone-size effects:"])
    for _, r in size_models[size_models.group == "IBD"].iterrows():
        lines.append(
            f"- {r.module}: beta/doubling={r.beta_per_doubling:.4f}, 95% CI "
            f"[{r.ci_low:.4f}, {r.ci_high:.4f}], FDR={r.FDR_within_group:.4g}."
        )
    lines.extend(["", "Top pooled-IBD T-B associations:"])
    for _, r in correlations[correlations.group == "IBD"].sort_values("FDR_within_group").head(8).iterrows():
        lines.append(
            f"- {r.T_label} vs {r.B_label}: rho={r.partial_rho:.3f}, n={int(r.n)}, "
            f"FDR={r.FDR_within_group:.4g}."
        )
    lines.extend(["", "Independent helper-state model terms with FDR < 0.10:"])
    sig = restraint[restraint.FDR < 0.10].sort_values("FDR")
    if sig.empty:
        lines.append("- None.")
    else:
        for _, r in sig.iterrows():
            lines.append(
                f"- {r.T_label} -> {r.B_label}: beta={r.standardized_beta:.3f}, "
                f"95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}], FDR={r.FDR:.4g}."
            )
    lines.extend(["", "Clinical associations:"])
    for _, r in clinical.sort_values("FDR").iterrows():
        lines.append(f"- {r.label}: rho={r.partial_rho:.3f}, n={int(r.n)}, FDR={r.FDR:.4g}.")
    (OUT / "Analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    axis_cells = pd.read_csv(OUT / "Table_TB1_paired_Th17_Treg_cells.csv.gz")
    clone_state = pd.read_csv(OUT / "Table_TB2_clone_state_programs.csv")
    clone_axis = pd.read_csv(OUT / "Table_TB3_axis_clone_summary.csv")
    delta = pd.read_csv(OUT / "Table_TB5_state_matched_expanded_singleton_deltas.csv")

    sharing = clone_sharing_permutation(axis_cells)
    size_models = clone_size_models(clone_state)
    delta_tests = expanded_singleton_tests(delta)
    data = build_analysis_dataset()
    correlations = targeted_correlations(data)
    restraint, _ = regulatory_restraint_models(data)
    clinical = clinical_associations(data)
    build_figure_h1(clone_axis, sharing, size_models, delta_tests)
    build_figure_h2(data, correlations, restraint)
    write_summary(sharing, size_models, delta_tests, correlations, restraint, clinical)
    print(f"Wrote analyses to {OUT}")
    print(f"Wrote PDFs to {PDF_OUT}")


if __name__ == "__main__":
    main()

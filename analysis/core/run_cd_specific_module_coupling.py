from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

SEED = 20260824
rng = np.random.default_rng(SEED)
OUT = Path(r"C:/path/to/private-manuscript-workspace\High Impact Additional Analyses")
DATA = OUT / "Table_HI_integrated_participant_features.csv"
X = "tcr_cytotoxic_expanded"
OUTCOMES = {
    "bcr_IgA_mucosal_module": "IgA mucosal plasma-cell module",
    "bcr_plasma_differentiation_module": "Plasmablast/plasma-cell differentiation module",
}
GROUPS = ["Control", "CD", "UC"]
COLORS = {"Control": "#777777", "CD": "#2B6CB0", "UC": "#C2415D"}


def partial_spearman(group, x, y):
    z = group[[x, y, "Age", "Sex", "tcr_total_cells", "bcr_total_cells"]].dropna().copy()
    z["rx"] = rankdata(z[x])
    z["ry"] = rankdata(z[y])
    cov = pd.get_dummies(z[["Sex"]], drop_first=True, dtype=float)
    cov["Age"] = z.Age.to_numpy(float)
    cov["log_tcr_depth"] = np.log10(z.tcr_total_cells.to_numpy(float) + 1)
    cov["log_bcr_depth"] = np.log10(z.bcr_total_cells.to_numpy(float) + 1)
    cov.insert(0, "intercept", 1.0)
    Xcov = cov.to_numpy(float)
    ex = z.rx.to_numpy() - Xcov @ np.linalg.lstsq(Xcov, z.rx.to_numpy(), rcond=None)[0]
    ey = z.ry.to_numpy() - Xcov @ np.linalg.lstsq(Xcov, z.ry.to_numpy(), rcond=None)[0]
    rho, p = pearsonr(ex, ey)
    boots = []
    for _ in range(2000):
        ii = rng.choice(np.arange(len(z)), len(z), replace=True)
        q = z.iloc[ii].copy()
        c = pd.get_dummies(q[["Sex"]], drop_first=True, dtype=float)
        c["Age"] = q.Age.to_numpy(float)
        c["log_tcr_depth"] = np.log10(q.tcr_total_cells.to_numpy(float) + 1)
        c["log_bcr_depth"] = np.log10(q.bcr_total_cells.to_numpy(float) + 1)
        c.insert(0, "intercept", 1.0)
        C = c.to_numpy(float)
        try:
            bx = rankdata(q[x]); by = rankdata(q[y])
            rx = bx - C @ np.linalg.lstsq(C, bx, rcond=None)[0]
            ry = by - C @ np.linalg.lstsq(C, by, rcond=None)[0]
            boots.append(np.corrcoef(rx, ry)[0, 1])
        except Exception:
            pass
    return len(z), rho, p, np.nanquantile(boots, 0.025), np.nanquantile(boots, 0.975)


def main():
    d = pd.read_csv(DATA)
    rows = []
    interaction_rows = []
    for outcome, label in OUTCOMES.items():
        for diagnosis in GROUPS:
            n, rho, p, lo, hi = partial_spearman(d[d.Diagnosis1 == diagnosis], X, outcome)
            rows.append({"outcome": outcome, "outcome_label": label, "diagnosis": diagnosis,
                         "n": n, "partial_rho": rho, "ci_low": lo, "ci_high": hi, "p_value": p})
        z = d[["Diagnosis1", "Age", "Sex", "tcr_total_cells", "bcr_total_cells", X, outcome]].dropna().copy()
        z["rank_x"] = rankdata(z[X]); z["rank_y"] = rankdata(z[outcome])
        z["log_tcr_depth"] = np.log10(z.tcr_total_cells + 1)
        z["log_bcr_depth"] = np.log10(z.bcr_total_cells + 1)
        reduced = smf.ols("rank_y ~ rank_x + C(Diagnosis1) + Age + C(Sex) + log_tcr_depth + log_bcr_depth", z).fit()
        full = smf.ols("rank_y ~ rank_x*C(Diagnosis1) + Age + C(Sex) + log_tcr_depth + log_bcr_depth", z).fit()
        comparison = anova_lm(reduced, full)
        interaction_rows.append({"outcome": outcome, "outcome_label": label, "n": len(z),
                                 "global_diagnosis_by_TCR_score_interaction_p": comparison.iloc[1]["Pr(>F)"],
                                 "model_r_squared": full.rsquared})
    stats = pd.DataFrame(rows)
    stats["p_adj_within_six"] = np.minimum(stats.p_value * len(stats), 1.0)
    interactions = pd.DataFrame(interaction_rows)
    stats.to_csv(OUT / "Table_HI_CD_specific_T_B_module_coupling.csv", index=False)
    interactions.to_csv(OUT / "Table_HI_T_B_module_interaction_tests.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.3), constrained_layout=True)
    for row_i, (outcome, label) in enumerate(OUTCOMES.items()):
        interaction_p = interactions.loc[interactions.outcome == outcome, "global_diagnosis_by_TCR_score_interaction_p"].iloc[0]
        for col_i, diagnosis in enumerate(GROUPS):
            ax = axes[row_i, col_i]
            z = d[d.Diagnosis1 == diagnosis][[X, outcome]].dropna()
            ax.scatter(z[X], z[outcome], s=22, alpha=0.72, color=COLORS[diagnosis], edgecolors="none")
            if len(z) >= 3 and z[X].nunique() > 1:
                coef = np.polyfit(z[X], z[outcome], 1)
                xx = np.linspace(z[X].min(), z[X].max(), 100)
                ax.plot(xx, coef[0] * xx + coef[1], color="#333333", linewidth=1.2)
            s = stats[(stats.outcome == outcome) & (stats.diagnosis == diagnosis)].iloc[0]
            ax.text(0.03, 0.97, f"partial rho={s.partial_rho:.2f}\n95% CI {s.ci_low:.2f} to {s.ci_high:.2f}\nn={int(s.n)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8)
            ax.set_title(diagnosis, fontweight="bold")
            ax.set_xlabel("Expanded-TCR cytotoxic score")
            ax.set_ylabel(label if col_i == 0 else "")
            ax.spines[["top", "right"]].set_visible(False)
        axes[row_i, 1].text(0.5, 1.16, f"Diagnosis x TCR-score interaction P={interaction_p:.3g}",
                            transform=axes[row_i, 1].transAxes, ha="center", va="bottom", fontsize=9)
    fig.suptitle("Cytotoxic T-cell and plasma-cell programs are coupled selectively in Crohn's disease",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "Figure_HI1b_CD_specific_T_B_module_coupling.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_HI1b_CD_specific_T_B_module_coupling.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

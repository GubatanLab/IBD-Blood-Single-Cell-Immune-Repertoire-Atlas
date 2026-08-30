from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PREVIEW = ROOT / "Proposed Main Figure 2H Preview"
PREVIEW.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

source = pd.read_csv(RESULTS / "GSE261334_expansion_program_dose_response.csv")

display_order = [
    ("Cytotoxic", "Cytotoxic", "#087FB6"),
    ("GZMK memory", "GZMK inflammatory memory", "#087FB6"),
    ("Th17 pathogenic", "Pathogenic Th17-like", "#087FB6"),
    ("Th17 conventional", "Conventional Th17", "#687782"),
    ("Naive/CM", "Naive/central memory", "#687782"),
]

data = source[
    (source["receptor"] == "TCR")
    & (source["time"] == "baseline")
    & (source["disease"] == "UC")
    & source["module"].isin([x[0] for x in display_order])
].copy()

assert data["participant"].nunique() == 10
assert data.groupby("module")["participant"].nunique().eq(10).all()

fdr = {
    "Cytotoxic": 0.005580,
    "GZMK memory": 0.005580,
    "Th17 pathogenic": 0.005580,
    "Th17 conventional": 0.005580,
    "Naive/CM": 0.005580,
}

fig, ax = plt.subplots(figsize=(5.45, 3.35))
fig.subplots_adjust(left=0.32, right=0.94, top=0.79, bottom=0.20)

positions = np.arange(len(display_order))[::-1]
rng = np.random.default_rng(20260829)

for pos, (module, label, color) in zip(positions, display_order):
    values = data.loc[data["module"] == module, "rho"].to_numpy()
    bp = ax.boxplot(
        values,
        positions=[pos],
        vert=False,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "#17324D", "linewidth": 1.5},
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1.0, "alpha": 0.20},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
    )
    jitter = rng.uniform(-0.13, 0.13, len(values))
    ax.scatter(
        values,
        pos + jitter,
        s=24,
        facecolor=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
        zorder=4,
    )
    med = float(np.median(values))
    ax.text(
        1.10,
        pos,
        f"{med:+.2f}       {fdr[module]:.4f}",
        ha="left",
        va="center",
        fontsize=7.2,
        color="#253746",
    )

ax.axvspan(-1.12, 0, color="#F3F5F7", zorder=-3)
ax.axvspan(0, 1.06, color="#F3F8FB", zorder=-3)
ax.axvline(0, color="#7A7A7A", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)

ax.set_xlim(-1.12, 1.52)
ax.set_ylim(-0.65, 4.65)
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_yticks(positions, [x[1] for x in display_order])
ax.set_xlabel("Clone-size/program correlation (participant Spearman ρ)")
ax.set_ylabel("")
ax.grid(axis="x", color="#D7DEE4", linewidth=0.55, alpha=0.8)
ax.set_axisbelow(True)

ax.text(-0.20, 1.16, "H", fontsize=15, fontweight="bold", ha="left", va="bottom", transform=ax.transAxes)
ax.text(
    -0.10,
    1.16,
    "Independent longitudinal UC cohort, baseline",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom",
    transform=ax.transAxes,
)
ax.text(
    -0.10,
    1.06,
    "Exact paired αβ clonotypes | n = 10 participants | bins: 1, 2, 3–4, ≥5 cells",
    fontsize=7.2,
    color="#53626F",
    ha="left",
    va="bottom",
    transform=ax.transAxes,
)
ax.text(1.10, 4.45, "Median ρ      FDR", fontsize=7.2, fontweight="bold", color="#253746", ha="left", va="bottom")
ax.text(-1.05, -0.53, "← lower in expanded clones", fontsize=6.8, color="#687782", ha="left", va="center")
ax.text(0.04, -0.53, "higher in expanded clones →", fontsize=6.8, color="#087FB6", ha="left", va="center")

for spine in ("left", "bottom"):
    ax.spines[spine].set_linewidth(0.9)

png = PREVIEW / "Proposed_Figure_2H_external_validation.png"
pdf = PREVIEW / "Proposed_Figure_2H_external_validation.pdf"
fig.savefig(png, dpi=400, facecolor="white")
fig.savefig(pdf, facecolor="white")
plt.close(fig)

print(png)
print(pdf)

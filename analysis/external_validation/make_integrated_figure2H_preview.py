from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT.parent
RESULTS = ROOT / "results"
OUT = ROOT / "Proposed Main Figure 2H Preview"
OUT.mkdir(exist_ok=True)

CANONICAL = MANUSCRIPT / "Final 7-Figure Manuscript Set" / "Main Figures" / "Figure_2.png"
DATA = RESULTS / "GSE261334_expansion_program_dose_response.csv"

DPI = 350
PAGE_W = 2581
PANEL_H = 540
MAIN_W = 2425

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.2,
    "axes.labelsize": 6.2,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 6.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

x = pd.read_csv(DATA)
order = [
    ("Cytotoxic", "Cytotoxic", "#087FB6"),
    ("GZMK memory", "GZMK inflammatory memory", "#087FB6"),
    ("Th17 pathogenic", "Pathogenic Th17-like", "#087FB6"),
    ("Th17 conventional", "Conventional Th17", "#687782"),
    ("Naive/CM", "Naive/central memory", "#687782"),
]
q = x[
    (x.receptor == "TCR")
    & (x.time == "baseline")
    & (x.disease == "UC")
    & x.module.isin([z[0] for z in order])
].copy()
assert q.groupby("module").participant.nunique().eq(10).all()

# Render the compact fourth-row panel at the same physical width/DPI as Figure 2.
fig = plt.figure(figsize=(PAGE_W / DPI, PANEL_H / DPI), dpi=DPI, facecolor="white")
ax = fig.add_axes([0.245, 0.20, 0.505, 0.47])
positions = np.arange(5)[::-1]
rng = np.random.default_rng(20260829)

ax.axvspan(-1.08, 0, color="#F3F5F7", zorder=-3)
ax.axvspan(0, 1.04, color="#F3F8FB", zorder=-3)
ax.axvline(0, color="#777777", lw=0.65, ls=(0, (3, 2)), zorder=1)

for pos, (module, label, color) in zip(positions, order):
    values = q.loc[q.module == module, "rho"].to_numpy()
    ax.boxplot(
        values,
        positions=[pos],
        vert=False,
        widths=0.43,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "#17324D", "linewidth": 0.9},
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.65, "alpha": 0.19},
        whiskerprops={"color": color, "linewidth": 0.65},
        capprops={"color": color, "linewidth": 0.65},
    )
    ax.scatter(
        values,
        pos + rng.uniform(-0.12, 0.12, len(values)),
        s=8,
        color=color,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.93,
        zorder=4,
    )
    ax.text(1.10, pos, f"{np.median(values):+.2f}", ha="left", va="center", fontsize=6.1, color="#253746")
    ax.text(1.48, pos, "0.0056", ha="left", va="center", fontsize=6.1, color="#253746")

ax.set_xlim(-1.10, 1.66)
ax.set_ylim(-0.58, 4.58)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks(positions, [z[1] for z in order])
ax.set_xlabel("Clone-size/program correlation (participant Spearman ρ)", labelpad=1.5)
ax.grid(axis="x", color="#D8DFE5", lw=0.4)
ax.set_axisbelow(True)
ax.spines["left"].set_linewidth(0.65)
ax.spines["bottom"].set_linewidth(0.65)
ax.tick_params(length=2.5, width=0.6)

fig.text(0.018, 0.93, "H", fontsize=13.5, fontweight="bold", ha="left", va="top")
fig.text(0.055, 0.93, "Independent longitudinal UC cohort, baseline", fontsize=8.6, fontweight="bold", ha="left", va="top")
fig.text(0.055, 0.79, "Exact paired αβ clonotypes | n = 10 participants | clone-size bins: 1, 2, 3–4, ≥5 cells", fontsize=5.9, color="#586875", ha="left", va="top")
fig.text(0.670, 0.72, "Median ρ", fontsize=6.1, fontweight="bold", color="#253746", ha="left")
fig.text(0.738, 0.72, "FDR", fontsize=6.1, fontweight="bold", color="#253746", ha="left")

strip_path = OUT / "_integrated_2H_strip.png"
fig.savefig(strip_path, dpi=DPI, facecolor="white")
plt.close(fig)

main = Image.open(CANONICAL).convert("RGB")
main_h = round(main.height * MAIN_W / main.width)
main = main.resize((MAIN_W, main_h), Image.Resampling.LANCZOS)
strip = Image.open(strip_path).convert("RGB")

gap = 8
canvas = Image.new("RGB", (PAGE_W, main_h + gap + PANEL_H), "white")
canvas.paste(main, ((PAGE_W - MAIN_W) // 2, 0))
canvas.paste(strip, (0, main_h + gap))

png = OUT / "Figure_2_with_proposed_2H_INTEGRATION_PREVIEW.png"
pdf = OUT / "Figure_2_with_proposed_2H_INTEGRATION_PREVIEW.pdf"
canvas.save(png, dpi=(DPI, DPI), optimize=True)
canvas.save(pdf, "PDF", resolution=DPI)
strip_path.unlink()

print(png)
print(pdf)
print(f"pixels={canvas.width}x{canvas.height}; inches={canvas.width/DPI:.2f}x{canvas.height/DPI:.2f}")

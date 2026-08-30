from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(r"C:/path/to/private-manuscript-workspace")
SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data" / "Figure1_lineage_miloR_heatmap_summary.csv"
OUT = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives"
PNG_OUT = OUT / "Figure_1J_original_clean_preview.png"
WEBP_OUT = OUT / "Figure_1J_original_clean_preview.webp"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


STATE_ORDER = {
    "CD4/Treg": [
        "CD4 Naive",
        "CD4 Naive-IFN",
        "CD4 Tfh",
        "CD4 Th1",
        "CD4 Th1/Th17",
        "CD4 Th17",
        "CD4 Th2",
        "CD4 Th22",
        "CD4 HLA-DR+ memory",
        "CD4 Temra",
        "CD4 Terminal effector",
        "TReg Naive",
        "TReg Memory",
        "TReg KLRB1+RORC+",
        "TReg Cytotoxic",
    ],
    "CD8/innate-like T": [
        "CD8 Naive",
        "CD8 Naive-IFN",
        "CD8 Tcm CCR4-",
        "CD8 Tem GZMK+",
        "CD8 Tem GZMB+",
        "CD8 Temra",
        "CD8 Trm",
        "CD8 HLA-DR+",
        "CD8 Proliferative",
        "CD8 Tmem KLRC2+",
        "MAIT",
        "gdT",
    ],
    "B/plasma": [
        "Transitional B",
        "CD5+ B Cell",
        "Naive B",
        "Naive-IFN B",
        "Non-switched memory B",
        "Switched memory B",
        "Atypical memory B",
        "IgM Plasma B Cell",
        "IgA Plasma B Cell",
        "IgG Plasma B Cell",
    ],
}

TITLE = {
    "CD4/Treg": "CD4 T Cells",
    "CD8/innate-like T": "CD8 T Cells",
    "B/plasma": "B Cells",
}

TITLE_COLOR = {
    "CD4/Treg": "#7656A8",
    "CD8/innate-like T": "#168C78",
    "B/plasma": "#C23B78",
}

COMPARISONS = ["UC vs control", "CD vs control", "CD vs UC"]
COMPARISON_LABELS = ["UC vs\ncontrol", "CD vs\ncontrol", "CD vs\nUC"]


def heatmap(ax, summary, lineage, norm, cmap):
    data = summary[summary["lineage"].eq(lineage)].copy()
    eligible = data.groupby("cell_state")["display_eligible"].max()
    states = [state for state in STATE_ORDER[lineage] if bool(eligible.get(state, False))]
    effect = data.pivot_table(index="cell_state", columns="comparison", values="display_value", aggfunc="first")
    matrix = effect.reindex(index=states, columns=COMPARISONS).to_numpy(float)

    masked = np.ma.masked_invalid(matrix)
    ax.set_facecolor("#EEEEEE")
    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    ax.set_xticks(range(3), COMPARISON_LABELS)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=6.6, length=0, pad=3)
    ax.set_yticks(range(len(states)), states)
    ax.tick_params(axis="y", labelsize=6.25, length=0, pad=3)

    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(states), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.85)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)

    # Open Cell Press-style matrices: no upper rule and no surrounding border.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Preserve the original heatmap values and layout but omit cell count annotations.
    ax.set_title(TITLE[lineage], fontweight="bold", fontsize=8.8, pad=19, color=TITLE_COLOR[lineage])
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SOURCE)

    fig = plt.figure(figsize=(9.2, 4.05), facecolor="white")
    container = fig.add_axes([0.055, 0.16, 0.90, 0.71])
    container.set_axis_off()
    container.text(-0.045, 1.13, "J", transform=container.transAxes, fontsize=15.5, fontweight="bold", ha="left", va="top")
    container.text(0.0, 1.12, "MiloR differential abundance", transform=container.transAxes, fontsize=10.2, fontweight="bold", ha="left", va="top")
    container.text(0.995, 1.105, "SpatialFDR < 0.05", transform=container.transAxes, fontsize=7.0, color="#6B7280", ha="right", va="top")

    positions = {
        "CD4/Treg": [0.00, 0.10, 0.31, 0.79],
        "CD8/innate-like T": [0.415, 0.10, 0.27, 0.79],
        "B/plasma": [0.785, 0.10, 0.18, 0.79],
    }
    norm = mpl.colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#EEEEEE")

    image = None
    for lineage, position in positions.items():
        ax = container.inset_axes(position)
        image = heatmap(ax, summary, lineage, norm, cmap)

    cax = container.inset_axes([0.982, 0.22, 0.015, 0.49])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_ticks([-5, 0, 5])
    colorbar.ax.tick_params(labelsize=6.5, width=0.45, length=2)
    colorbar.outline.set_linewidth(0.45)
    cax.set_title("median\nlog2 FC", fontsize=6.4, pad=4, color="#374151")

    container.text(
        0.0,
        -0.015,
        "Color: annotation-fraction-weighted median log2 FC among significant neighborhoods; positive values favor the first-named group; gray indicates insufficient significant neighborhoods.",
        transform=container.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="#6B7280",
    )

    fig.savefig(PNG_OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    with Image.open(PNG_OUT) as image_file:
        width = 1800
        height = round(image_file.height * width / image_file.width)
        image_file.resize((width, height), Image.Resampling.LANCZOS).save(WEBP_OUT, "WEBP", quality=92, method=6)

    print(PNG_OUT)
    print(WEBP_OUT)


if __name__ == "__main__":
    main()

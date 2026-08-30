from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(r"C:/path/to/private-manuscript-workspace")
SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
PREVIEW = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives"

I_SOURCE = SOURCE / "Figure1_participant_state_effects.csv"
J_SOURCE = PREVIEW / "Figure_1J_revised_preview_source.csv"
PNG_OUT = PREVIEW / "Figure_1IJ_revised_preview.png"
WEBP_OUT = PREVIEW / "Figure_1IJ_revised_preview.webp"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.3,
        "axes.titlesize": 10.2,
        "axes.labelsize": 8.7,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

LINEAGE_COLORS = {
    "CD4/Treg": "#6B4C9A",
    "CD8/innate-like T": "#148A8A",
    "B/plasma": "#B23A77",
}
COMPARISONS = ["UC vs control", "CD vs control", "CD vs UC"]
COMPARISON_LABELS = {
    "UC vs control": "UC vs control",
    "CD vs control": "CD vs control",
    "CD vs UC": "CD vs UC",
}
COMPARISON_MARKERS = {
    "UC vs control": "o",
    "CD vs control": "s",
    "CD vs UC": "^",
}
COMPARISON_OFFSETS = {
    "UC vs control": -0.19,
    "CD vs control": 0.0,
    "CD vs UC": 0.19,
}

I_STATES = [
    "CD4 Naive",
    "TReg Cytotoxic",
    "CD8 Naive",
    "CD8 Tem GZMB+",
    "Switched memory B",
    "IgM Plasma B Cell",
]
I_LABELS = {
    "CD4 Naive": "CD4 naive",
    "TReg Cytotoxic": "Cytotoxic Treg",
    "CD8 Naive": "CD8 naive",
    "CD8 Tem GZMB+": "CD8 GZMB+ memory",
    "Switched memory B": "Switched-memory B",
    "IgM Plasma B Cell": "IgM plasma cell",
}

J_STATES = {
    "CD4/Treg": ["CD4 Naive", "CD4 Th17", "CD4 Temra", "TReg Cytotoxic"],
    "CD8/innate-like T": ["CD8 Naive", "CD8 Tem GZMB+", "CD8 HLA-DR+", "MAIT"],
    "B/plasma": [
        "Naive B",
        "Switched memory B",
        "Atypical memory B",
        "IgM Plasma B Cell",
        "IgA Plasma B Cell",
        "IgG Plasma B Cell",
    ],
}
J_LABELS = {
    "CD4 Naive": "CD4 naive",
    "CD4 Th17": "CD4 Th17",
    "CD4 Temra": "CD4 TEMRA",
    "TReg Cytotoxic": "Cytotoxic Treg",
    "CD8 Naive": "CD8 naive",
    "CD8 Tem GZMB+": "CD8 GZMB+ memory",
    "CD8 HLA-DR+": "CD8 HLA-DR+",
    "MAIT": "MAIT",
    "Naive B": "Naive B",
    "Switched memory B": "Switched-memory B",
    "Atypical memory B": "Atypical-memory B",
    "IgM Plasma B Cell": "IgM plasma cell",
    "IgA Plasma B Cell": "IgA plasma cell",
    "IgG Plasma B Cell": "IgG plasma cell",
}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.65, color="#4B5563")


def plot_panel_i(ax, data):
    ax.text(-0.14, 1.13, "I", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    ax.text(
        0,
        1.13,
        "Participant-level differential abundance",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0,
        1.035,
        "Prespecified lineage states evaluated with acquisition-series-adjusted models",
        transform=ax.transAxes,
        fontsize=7.8,
        color="#4B5563",
        va="top",
    )

    y_positions = {state: len(I_STATES) - 1 - idx for idx, state in enumerate(I_STATES)}
    for state in I_STATES:
        state_data = data[data["cell_state"] == state]
        lineage = state_data["lineage"].iloc[0]
        color = LINEAGE_COLORS[lineage]
        for comparison in COMPARISONS:
            row = state_data[state_data["comparison"] == comparison].iloc[0]
            y = y_positions[state] + COMPARISON_OFFSETS[comparison]
            significant = row["adjusted_p_value"] < 0.05
            ax.errorbar(
                row["effect"],
                y,
                xerr=[[row["effect"] - row["ci_low"]], [row["ci_high"] - row["effect"]]],
                fmt=COMPARISON_MARKERS[comparison],
                ms=5.5,
                mfc=color if significant else "white",
                mec=color,
                mew=1.1,
                ecolor=color,
                elinewidth=1.05,
                capsize=2.2,
                capthick=0.85,
                zorder=3,
            )

    ax.axvline(0, color="#6B7280", lw=0.8, ls=(0, (2.2, 2.2)), zorder=1)
    ax.axhline(3.5, color="#D1D5DB", lw=0.7)
    ax.axhline(1.5, color="#D1D5DB", lw=0.7)
    ax.set_yticks([y_positions[s] for s in I_STATES])
    ax.set_yticklabels([I_LABELS[s] for s in I_STATES])
    for tick, state in zip(ax.get_yticklabels(), I_STATES):
        lineage = data.loc[data["cell_state"] == state, "lineage"].iloc[0]
        tick.set_color(LINEAGE_COLORS[lineage])
        tick.set_fontweight("bold")
    ax.set_ylim(-0.55, len(I_STATES) - 0.45)
    ax.set_xlim(-3.05, 4.05)
    ax.set_xticks(np.arange(-3, 5, 1))
    ax.set_xlabel("Difference in logit-transformed within-lineage fraction (95% CI)", labelpad=5)
    ax.grid(axis="x", color="#E5E7EB", lw=0.55, zorder=0)
    style_axis(ax)

    contrast_handles = [
        Line2D(
            [0],
            [0],
            marker=COMPARISON_MARKERS[c],
            color="#374151",
            markerfacecolor="white",
            markeredgecolor="#374151",
            linewidth=0,
            markersize=5.4,
            label=COMPARISON_LABELS[c],
        )
        for c in COMPARISONS
    ]
    contrast_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="#374151",
                markerfacecolor="#374151",
                linewidth=0,
                markersize=5.4,
                label="FDR < 0.05",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#374151",
                markerfacecolor="white",
                markeredgecolor="#374151",
                linewidth=0,
                markersize=5.4,
                label="FDR ≥ 0.05",
            ),
        ]
    )
    ax.legend(
        handles=contrast_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        frameon=False,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0,
    )
    ax.text(
        1.0,
        -0.29,
        "HC3 95% CIs; Benjamini–Hochberg FDR across 18 contrasts",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#6B7280",
    )


def plot_panel_j(fig, subspec, data):
    container = fig.add_subplot(subspec)
    container.axis("off")
    container.text(-0.025, 1.12, "J", transform=container.transAxes, fontsize=16, fontweight="bold", va="top")
    container.text(
        0.025,
        1.12,
        "Neighborhood-level differential abundance",
        transform=container.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )
    container.text(
        0.025,
        1.035,
        "Effect direction and breadth across Milo neighborhoods, aligned to the participant-level tests in I",
        transform=container.transAxes,
        fontsize=7.8,
        color="#4B5563",
        va="top",
    )

    gs = subspec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.16], wspace=0.46)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    norm = mpl.colors.Normalize(vmin=-4, vmax=4)
    cmap = mpl.colormaps["RdBu_r"]

    for ax, lineage in zip(axes, J_STATES):
        states = J_STATES[lineage]
        subset = data[data["lineage"] == lineage].copy()
        y_positions = {state: len(states) - 1 - idx for idx, state in enumerate(states)}

        for _, row in subset.iterrows():
            if row["cell_state"] not in y_positions or row["comparison"] not in COMPARISONS:
                continue
            x = COMPARISONS.index(row["comparison"])
            y = y_positions[row["cell_state"]]
            if not bool(row["coverage_adequate"]):
                ax.scatter(x, y, marker="x", s=42, c="#9CA3AF", linewidths=1.15, zorder=4)
                continue
            size = 42 + 245 * float(row["fraction_significant"])
            ax.scatter(
                x,
                y,
                s=size,
                c=[cmap(norm(np.clip(row["weighted_median_all_logFC"], -4, 4)))],
                edgecolors="black" if bool(row["participant_tested"]) else "white",
                linewidths=1.25 if bool(row["participant_tested"]) else 0.6,
                zorder=3,
            )
            if bool(row["participant_confirmed"]):
                ax.scatter(
                    x + 0.23,
                    y + 0.20,
                    marker=(5, 1),
                    s=29,
                    c="black",
                    linewidths=0,
                    zorder=5,
                )

        ax.set_title(
            {"CD4/Treg": "CD4 T Cells", "CD8/innate-like T": "CD8 T Cells", "B/plasma": "B Cells"}[lineage],
            color=LINEAGE_COLORS[lineage],
            fontweight="bold",
            pad=8,
        )
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-0.55, len(states) - 0.45)
        ax.set_xticks(range(3))
        ax.set_xticklabels([COMPARISON_LABELS[c] for c in COMPARISONS], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks([y_positions[s] for s in states])
        ax.set_yticklabels([J_LABELS[s] for s in states])
        ax.tick_params(axis="y", colors=LINEAGE_COLORS[lineage])
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.set_axisbelow(True)
        ax.grid(color="#E5E7EB", lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_color("#9CA3AF")
        ax.tick_params(length=0)

    cax = container.inset_axes([0.055, -0.245, 0.25, 0.035])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_ticks([])
    cb.outline.set_linewidth(0.55)
    container.text(0.055, -0.286, "Lower", transform=container.transAxes, ha="left", va="top", fontsize=6.8, color="#4B5563")
    container.text(0.305, -0.286, "Higher", transform=container.transAxes, ha="right", va="top", fontsize=6.8, color="#4B5563")
    container.text(
        0.18,
        -0.145,
        "Weighted median log2 fold change",
        transform=container.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color="#4B5563",
    )

    size_ax = container.inset_axes([0.37, -0.285, 0.27, 0.16])
    size_ax.axis("off")
    for x, frac, label in zip([0.13, 0.46, 0.82], [0.1, 0.5, 0.9], ["Narrow", "Moderate", "Broad"]):
        size_ax.scatter(x, 0.55, s=42 + 245 * frac, c="#D1D5DB", edgecolors="white", linewidths=0.6)
        size_ax.text(x, 0.03, label, ha="center", va="bottom", fontsize=6.8, color="#4B5563")
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)
    container.text(
        0.505,
        -0.145,
        "Neighborhoods with SpatialFDR < 0.05",
        transform=container.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color="#4B5563",
    )

    key_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D1D5DB", markeredgecolor="black", markeredgewidth=1.2, markersize=7.5, label="Tested in I"),
        Line2D([0], [0], marker="$★$", color="black", linestyle="none", markersize=7.5, label="FDR < 0.05 + concordant"),
        Line2D([0], [0], marker="x", color="#9CA3AF", linestyle="none", markersize=6.5, label="Low neighborhood coverage"),
    ]
    container.legend(
        handles=key_handles,
        loc="lower right",
        bbox_to_anchor=(1.005, -0.30),
        frameon=False,
        ncol=1,
        labelspacing=0.45,
        handletextpad=0.45,
        borderaxespad=0,
        fontsize=7.0,
    )
    container.text(
        1.0,
        -0.39,
        "Neighborhood SpatialFDR is the Milo inferential unit; participant-level models are shown separately in I.",
        transform=container.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#6B7280",
    )


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    i_data = pd.read_csv(I_SOURCE)
    j_data = pd.read_csv(J_SOURCE)

    fig = plt.figure(figsize=(10.8, 8.15), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.34], hspace=0.56, left=0.16, right=0.975, top=0.94, bottom=0.155)
    ax_i = fig.add_subplot(outer[0])
    plot_panel_i(ax_i, i_data)
    plot_panel_j(fig, outer[1], j_data)

    fig.savefig(PNG_OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    with Image.open(PNG_OUT) as image:
        width = 1800
        height = round(image.height * width / image.width)
        image.resize((width, height), Image.Resampling.LANCZOS).save(WEBP_OUT, "WEBP", quality=92, method=6)
    print(PNG_OUT)
    print(WEBP_OUT)


if __name__ == "__main__":
    main()

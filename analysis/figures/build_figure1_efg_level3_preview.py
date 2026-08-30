from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives" / "Source Data"
MAIN_SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
OUT = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1C1C1C"
MUTED = "#5D5D5D"

LEVEL3_COLORS = {
    # CD4/Treg
    "CD4 Naive": "#4EA5D9",
    "CD4 Naive-IFN": "#86C8E5",
    "CD4 HLA-DR+ memory": "#AA4499",
    "CD4 Tfh": "#E5A32A",
    "CD4 Th1": "#5E3C99",
    "CD4 Th1/Th17": "#8073AC",
    "CD4 Th17": "#2A9D61",
    "CD4 Th22": "#78C679",
    "CD4 Th2": "#F28E2B",
    "CD4 Temra": "#D95F02",
    "CD4 Terminal effector": "#A63603",
    "TReg Naive": "#E78AC3",
    "TReg Memory": "#C51B7D",
    "TReg KLRB1+RORC+": "#8C6BB1",
    "TReg Cytotoxic": "#54278F",
    # CD8/innate-like T
    "CD8 Naive": "#4EA5D9",
    "CD8 Naive-IFN": "#86C8E5",
    "CD8 Tcm CCR4-": "#2B6CB0",
    "CD8 Tem GZMK+": "#38A169",
    "CD8 Tem GZMB+": "#E67E22",
    "CD8 Temra": "#C0392B",
    "CD8 Trm": "#805AD5",
    "CD8 Tmem KLRC2+": "#319795",
    "CD8 HLA-DR+": "#9B2C2C",
    "CD8 Proliferative": "#4A5568",
    "MAIT": "#6B8E23",
    "gdT": "#B7791F",
    # B/plasma
    "Naive B": "#4EA5D9",
    "Naive-IFN B": "#86C8E5",
    "Transitional B": "#EDC948",
    "CD5+ B Cell": "#F28E2B",
    "Non-switched memory B": "#59A14F",
    "Switched memory B": "#00876C",
    "Atypical memory B": "#B07AA1",
    "IgM Plasma B Cell": "#9C755F",
    "IgA Plasma B Cell": "#E15759",
    "IgG Plasma B Cell": "#7B2CBF",
}

PANEL_ORDERS = {
    "CD4": [
        "CD4 Naive", "CD4 Naive-IFN", "CD4 HLA-DR+ memory", "CD4 Tfh",
        "CD4 Th1", "CD4 Th1/Th17", "CD4 Th17", "CD4 Th22", "CD4 Th2",
        "CD4 Temra", "CD4 Terminal effector", "TReg Naive", "TReg Memory",
        "TReg KLRB1+RORC+", "TReg Cytotoxic",
    ],
    "CD8": [
        "CD8 Naive", "CD8 Naive-IFN", "CD8 Tcm CCR4-", "CD8 Tem GZMK+",
        "CD8 Tem GZMB+", "CD8 Temra", "CD8 Trm", "CD8 Tmem KLRC2+",
        "CD8 HLA-DR+", "CD8 Proliferative", "MAIT", "gdT",
    ],
    "B": [
        "Naive B", "Naive-IFN B", "Transitional B", "CD5+ B Cell",
        "Non-switched memory B", "Switched memory B", "Atypical memory B",
        "IgM Plasma B Cell", "IgA Plasma B Cell", "IgG Plasma B Cell",
    ],
}

CLONE_STYLES = [
    ("2-5", 2, 5, "#E07A1F", 1.0),
    ("6-20", 6, 20, "#C23B73", 1.25),
    ("21-100", 21, 100, "#7A3E9D", 1.55),
    (">100", 101, np.inf, "#32115F", 1.9),
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.5,
            "axes.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def stratified_background(data: pd.DataFrame, cap: int = 60000) -> pd.DataFrame:
    if len(data) <= cap:
        return data
    fraction = cap / len(data)
    sampled = []
    for index, (_, group) in enumerate(
        data.groupby(["AnnotationLevel3", "Diagnosis1"], observed=True, sort=True)
    ):
        target = max(1, min(len(group), int(round(len(group) * fraction))))
        sampled.append(group.sample(target, random_state=20260827 + index))
    output = pd.concat(sampled, ignore_index=False)
    if len(output) > cap:
        output = output.sample(cap, random_state=20260827)
    return output


def load_panel(panel: str, filename: str, overlay: pd.DataFrame) -> pd.DataFrame:
    data = pd.read_csv(SOURCE / filename)
    use = overlay.loc[
        overlay["panel"].eq(panel),
        ["cell", "paired_clone_size", "expanded_paired"],
    ].copy()
    data = data.merge(use, on="cell", how="left", validate="one_to_one")
    data["expanded_paired"] = data["expanded_paired"].fillna(False).astype(bool)
    data["paired_clone_size"] = pd.to_numeric(data["paired_clone_size"], errors="coerce").fillna(0)
    return data


def draw_umap(ax: plt.Axes, data: pd.DataFrame, panel: str) -> None:
    order = [state for state in PANEL_ORDERS[panel] if state in set(data["AnnotationLevel3"])]
    background = stratified_background(data)
    for state in order:
        subset = background.loc[background["AnnotationLevel3"].eq(state)]
        ax.scatter(
            subset["UMAP_1"], subset["UMAP_2"],
            s=0.38, c=LEVEL3_COLORS[state], alpha=0.46,
            edgecolors="none", rasterized=True, zorder=1,
        )

    expanded = data.loc[data["expanded_paired"]].copy()
    expanded["clone_bin"] = pd.cut(
        expanded["paired_clone_size"],
        bins=[1, 5, 20, 100, np.inf],
        labels=[item[0] for item in CLONE_STYLES],
        right=True,
    )
    if len(expanded) > 8000:
        sampled = []
        for index, (_, group) in enumerate(
            expanded.groupby(["AnnotationLevel3", "clone_bin"], observed=True, sort=True)
        ):
            sampled.append(group.sample(min(len(group), 180), random_state=20260827 + index))
        expanded = pd.concat(sampled, ignore_index=False)
    for label, lower, upper, color, multiplier in CLONE_STYLES:
        subset = expanded.loc[
            expanded["paired_clone_size"].ge(lower)
            & expanded["paired_clone_size"].le(upper)
        ]
        if subset.empty:
            continue
        ax.scatter(
            subset["UMAP_1"], subset["UMAP_2"],
            s=1.3 * multiplier, c=color, alpha=0.92,
            edgecolors="white", linewidths=0.12, rasterized=False, zorder=3,
        )

    xlo, xhi = np.nanquantile(data["UMAP_1"], [0.003, 0.997])
    ylo, yhi = np.nanquantile(data["UMAP_2"], [0.003, 0.997])
    xspan = xhi - xlo
    yspan = yhi - ylo
    # Reserve a small, asymmetric non-data corner for the orientation glyph.
    # This keeps the axes legible without covering low-coordinate cells.
    left_pad = max(xspan * 0.15, 0.50)
    right_pad = max(xspan * 0.04, 0.25)
    bottom_pad = max(yspan * 0.15, 0.50)
    top_pad = max(yspan * 0.04, 0.25)
    ax.set_xlim(xlo - left_pad, xhi + right_pad)
    ax.set_ylim(ylo - bottom_pad, yhi + top_pad)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_umap_axis_glyph(ax)


def draw_umap_axis_glyph(ax: plt.Axes, *, x: float = 0.028, y: float = 0.032,
                         length: float = 0.095, color: str = "#343434") -> None:
    """Add the same small UMAP orientation marker used in the main PBMC atlas."""
    halo = [pe.withStroke(linewidth=1.45, foreground="white")]
    arrow = {
        "arrowstyle": "-|>",
        "mutation_scale": 5.2,
        "lw": 0.62,
        "color": color,
        "shrinkA": 0,
        "shrinkB": 0,
        "path_effects": halo,
    }
    ax.annotate("", xy=(x + length, y), xytext=(x, y), xycoords=ax.transAxes,
                arrowprops=arrow, annotation_clip=False, zorder=30)
    ax.annotate("", xy=(x, y + length), xytext=(x, y), xycoords=ax.transAxes,
                arrowprops=arrow, annotation_clip=False, zorder=30)
    label_x = ax.text(x + length * 0.52, y - 0.020, "UMAP 1", transform=ax.transAxes,
                      ha="center", va="top", fontsize=3.2, color=color, zorder=31)
    label_y = ax.text(x - 0.020, y + length * 0.52, "UMAP 2", transform=ax.transAxes,
                      ha="right", va="center", rotation=90, fontsize=3.2, color=color, zorder=31)
    label_x.set_path_effects(halo)
    label_y.set_path_effects(halo)


def draw_state_legend(ax: plt.Axes, panel: str) -> None:
    ax.set_axis_off()
    states = PANEL_ORDERS[panel]
    ax.text(0.01, 0.985, "Original Level 3", ha="left", va="top",
            fontsize=5.5, fontweight="bold", color=INK, transform=ax.transAxes)
    top = 0.91
    bottom = 0.055
    positions = np.linspace(top, bottom, len(states))
    for y, state in zip(positions, states):
        display_state = {
            "Non-switched memory B": "Non-switched\nmemory B",
        }.get(state, state)
        ax.scatter([0.055], [y], s=13, c=LEVEL3_COLORS[state], edgecolors="none",
                   transform=ax.transAxes, clip_on=False)
        ax.text(0.14, y, display_state, ha="left", va="center", fontsize=4.85,
                color=INK, transform=ax.transAxes)


def build_preview() -> Path:
    configure_style()
    overlay = pd.read_csv(MAIN_SOURCE / "Figure1_clone_engagement_umap.csv")
    summary = pd.read_csv(MAIN_SOURCE / "Figure1_lineage_panel_summary.csv").set_index("panel")

    panels = [
        ("E", "CD4", "CD4 T Cells", "Figure_1E_CD4_Level3_UMAP.csv"),
        ("F", "CD8", "CD8 T Cells", "Figure_1F_CD8_Level3_UMAP.csv"),
        ("G", "B", "B Cells", "Figure_1G_B_Level3_UMAP.csv"),
    ]

    fig = plt.figure(figsize=(7.48, 3.05), facecolor="white")
    outer = GridSpec(1, 3, figure=fig, left=0.025, right=0.995, top=0.87, bottom=0.18, wspace=0.16)

    for position, (letter, panel, title, filename) in enumerate(panels):
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0, position], width_ratios=[1.38, 1.12], wspace=0.02
        )
        umap_ax = fig.add_subplot(inner[0, 0])
        legend_ax = fig.add_subplot(inner[0, 1])
        data = load_panel(panel, filename, overlay)
        draw_umap(umap_ax, data, panel)
        draw_state_legend(legend_ax, panel)

        row = summary.loc[panel]
        umap_ax.text(-0.10, 1.14, letter, transform=umap_ax.transAxes, ha="left", va="top",
                     fontsize=10.5, fontweight="bold", color=INK, clip_on=False)
        umap_ax.text(0.00, 1.12, title, transform=umap_ax.transAxes, ha="left", va="top",
                     fontsize=7.2, fontweight="bold", color=INK, clip_on=False)
        umap_ax.text(
            0.00, 1.045,
            f"{int(row.total_cells):,} cells | {int(row.participants)} participants",
            transform=umap_ax.transAxes, ha="left", va="top",
            fontsize=5.3, color=MUTED, clip_on=False,
        )

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=3.1 + 0.45 * index,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.3,
            label=label,
        )
        for index, (label, _, _, color, _) in enumerate(CLONE_STYLES)
    ]
    fig.text(0.325, 0.075, "Expanded paired clonotype size (cells)", ha="right", va="center",
             fontsize=5.7, color=INK)
    fig.legend(
        handles=handles, loc="center left", bbox_to_anchor=(0.33, 0.075),
        ncol=4, frameon=False, fontsize=5.5, handletextpad=0.25,
        columnspacing=0.85, borderaxespad=0,
    )
    fig.text(
        0.995, 0.018,
        "Preview only: Level 3 annotations are read directly from the original lineage Seurat objects; "
        "current Figure 1 is unchanged.",
        ha="right", va="bottom", fontsize=4.8, color=MUTED,
    )

    png_path = OUT / "Figure_1EFG_Level3_unlabeled_UMAP_preview.png"
    fig.savefig(png_path, dpi=400, facecolor="white", bbox_inches=None)
    plt.close(fig)

    web_path = OUT / "Figure_1EFG_Level3_unlabeled_UMAP_preview.webp"
    with Image.open(png_path) as image:
        max_width = 1800
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        image.save(web_path, format="WEBP", quality=92, method=6)
    return png_path


if __name__ == "__main__":
    print(build_preview())

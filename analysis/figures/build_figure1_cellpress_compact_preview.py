from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_cell_press_redrawn_figures as base
from build_figure1_efg_level3_preview import (
    CLONE_STYLES,
    draw_state_legend,
    draw_umap,
)


OUT = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives"
OUT.mkdir(parents=True, exist_ok=True)


def panel_letter(ax, letter: str, x: float = -0.10, y: float = 1.045) -> None:
    ax.text(
        x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
        ha="left", va="top", clip_on=False, color=base.COL["ink"],
    )


def cohort_panel_compact(ax, participants) -> None:
    y = [2, 1, 0]
    totals = participants.groupby("Diagnosis1").size().reindex(base.DIAG).fillna(0).astype(int)
    inflamed = (
        participants[participants["Inflammation1"].eq("Inflamed")]
        .groupby("Diagnosis1").size().reindex(base.DIAG).fillna(0).astype(int)
    )
    for yi, group in zip(y, base.DIAG):
        total = totals[group]
        ax.barh(yi, total, height=0.56, color=base.COL[group], alpha=0.22,
                edgecolor=base.COL[group], lw=0.8)
        if inflamed[group] > 0:
            ax.barh(yi, inflamed[group], height=0.56, color=base.COL[group],
                    alpha=0.82, edgecolor="none")
        ax.text(total + 3, yi, f"n={total}", va="center", fontsize=6.6,
                fontweight="bold", color=base.COL[group])
    ax.set_yticks(y, base.DIAG)
    ax.set_xlim(0, max(totals) * 1.43)
    ax.set_xlabel("Participants", labelpad=2)
    base.style_axis(ax, "x")


def workflow_panel_undistorted(ax) -> None:
    """Draw the Figure 1B raster at its native cropped aspect ratio."""
    ax.set_axis_off()
    workflow = plt.imread(base.ASSET / "Figure_1B_ChatGPT_new_study_schematic_v6.png")
    height = workflow.shape[0]
    workflow = workflow[int(height * 0.08):int(height * 0.86), :, :]
    aspect = workflow.shape[1] / workflow.shape[0]
    ax.imshow(
        workflow, extent=(0.0, aspect, 0.0, 1.0), aspect="equal",
        interpolation="lanczos", origin="upper",
    )
    x_positions = np.asarray([0.095, 0.285, 0.485, 0.690, 0.895]) * aspect
    labels = [
        "Cohort\nn=249", "PBMC\nisolation", "BD Rhapsody\nmicrowell capture",
        "WTA + paired\nTCR/BCR", "Cell state +\nclonotype",
    ]
    for x, label in zip(x_positions, labels):
        ax.text(
            x, -0.055, label, ha="center", va="top", fontsize=5.5,
            linespacing=0.95, fontweight="bold", color=base.COL["ink"], clip_on=False,
        )
    ax.set_xlim(0.0, aspect)
    ax.set_ylim(0.0, 1.0)


def remove_matching_text(fig, prefixes: tuple[str, ...]) -> None:
    axes = []
    pending = list(fig.axes)
    while pending:
        ax = pending.pop()
        if ax in axes:
            continue
        axes.append(ax)
        pending.extend(getattr(ax, "child_axes", []))
    for ax in axes:
        for text in list(ax.texts):
            value = text.get_text().strip()
            if any(value.startswith(prefix) for prefix in prefixes):
                text.remove()


def main() -> None:
    participants = base.read_csv(base.SRC / "Figure1_participant_summary.csv")
    recovery = base.read_csv(base.SRC / "Figure1_participant_receptor_recovery.csv")
    panel_summary = base.read_csv(base.SRC / "Figure1_lineage_panel_summary.csv").set_index("panel")
    milo_summary = base.read_csv(base.SRC / "Figure1_lineage_miloR_heatmap_summary.csv")
    clone_overlay = base.read_csv(base.SRC / "Figure1_clone_engagement_umap.csv")
    marker_data = base.read_csv(base.SRC / "Figure1_canonical_marker_dotplot.csv")
    pbmc = base.read_csv(base.SRC / "UMAP_PBMC_from_Seurat.csv")

    level3_source = base.OUT / "Preview Alternatives" / "Source Data"
    cd4 = base.read_csv(level3_source / "Figure_1E_CD4_Level3_UMAP.csv")
    cd8 = base.read_csv(level3_source / "Figure_1F_CD8_Level3_UMAP.csv")
    bcell = base.read_csv(level3_source / "Figure_1G_B_Level3_UMAP.csv")
    overlay_columns = ["cell", "expanded_paired", "paired_family", "paired_clone_size"]
    cd4 = cd4.merge(clone_overlay[clone_overlay["panel"].eq("CD4")][overlay_columns], on="cell", how="left")
    cd8 = cd8.merge(clone_overlay[clone_overlay["panel"].eq("CD8")][overlay_columns], on="cell", how="left")
    bcell = bcell.merge(clone_overlay[clone_overlay["panel"].eq("B")][overlay_columns], on="cell", how="left")
    clone_summary = base.summarize_figure1h_clone_engagement({"CD4": cd4, "CD8": cd8, "B": bcell})

    # 172 mm x 225 mm Cell Press-oriented full-page preview.
    fig = plt.figure(figsize=(6.772, 8.858))
    gs = GridSpec(
        5, 1, figure=fig,
        height_ratios=[0.72, 1.52, 1.08, 1.34, 1.20],
        hspace=0.31, left=0.115, right=0.978, top=0.982, bottom=0.038,
    )
    top_row = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, 0], width_ratios=[1.80, 4.20], wspace=0.17,
    )
    atlas_row = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 0], width_ratios=[3.92, 2.08], wspace=0.19,
    )
    lineage_row = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, 0], wspace=0.22)

    ax_a = fig.add_subplot(top_row[0, 0])
    cohort_panel_compact(ax_a, participants)
    panel_letter(ax_a, "A", x=-0.15, y=1.06)

    ax_b = fig.add_subplot(top_row[0, 1])
    workflow_panel_undistorted(ax_b)
    panel_letter(ax_b, "B", x=-0.075, y=1.06)
    remove_matching_text(fig, ("Barcode-linked RNA",))

    ax_c = fig.add_subplot(atlas_row[0, 0])
    base.full_pbmc_level2_umap(ax_c, pbmc)
    panel_letter(ax_c, "C", x=-0.075, y=1.05)

    ax_d = fig.add_subplot(atlas_row[0, 1])
    base.participant_receptor_recovery_panel(ax_d, recovery)
    panel_letter(ax_d, "D", x=-0.12, y=1.05)
    remove_matching_text(fig, ("249 profiled", "expanded 0-4% scale"))

    lineage_panels = [
        ("E", "CD4", "CD4 T Cells", cd4, lineage_row[0, 0]),
        ("F", "CD8", "CD8 T Cells", cd8, lineage_row[0, 1]),
        ("G", "B", "B Cells", bcell, lineage_row[0, 2]),
    ]
    for letter, panel, title, data, subplot_spec in lineage_panels:
        widths = [1.78, 0.72] if panel in {"CD4", "CD8"} else [1.45, 1.05]
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplot_spec, width_ratios=widths, wspace=0.055,
        )
        umap_ax = fig.add_subplot(inner[0, 0])
        legend_ax = fig.add_subplot(inner[0, 1])
        draw_umap(umap_ax, data, panel)
        draw_state_legend(legend_ax, panel)
        panel_letter(umap_ax, letter, x=-0.16, y=1.10)
        umap_ax.text(
            0.0, 1.085, title, transform=umap_ax.transAxes, ha="left", va="top",
            fontsize=6.9, fontweight="bold", color=base.COL["ink"], clip_on=False,
        )
    remove_matching_text(fig, ("Original Level 3",))

    clone_handles = [
        Line2D([0], [0], marker="o", ls="", ms=3.0 + 0.42 * index,
               markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.38,
               label=label)
        for index, (label, _, _, color, _) in enumerate(CLONE_STYLES)
    ]
    lineage_box = gs[2, 0].get_position(fig)
    clone_y = lineage_box.y0 - 0.012
    fig.text(0.45, clone_y, "Expanded paired clonotype size:", ha="right", va="center",
             fontsize=4.6, color=base.COL["ink"])
    fig.legend(
        handles=clone_handles, frameon=False, loc="center left",
        bbox_to_anchor=(0.455, clone_y), ncol=4, fontsize=4.45,
        handletextpad=0.16, columnspacing=0.48, borderaxespad=0,
    )

    ax_h = fig.add_subplot(gs[3, 0])
    # Lift the dot matrices to create a clean shared-legend band below the
    # rotated gene labels on the shorter Cell Press canvas.
    for spec in base.FIGURE1H_FACETS.values():
        for position_key in ("marker_position", "clone_position"):
            position = list(spec[position_key])
            position[1] = 0.315
            position[3] = 0.525
            spec[position_key] = position
    base.compact_canonical_marker_dotplot(ax_h, marker_data, clone_summary)
    panel_letter(ax_h, "H", x=-0.065, y=1.03)
    remove_matching_text(fig, ("Expanded fraction among paired-receptor cells",))

    ax_i = fig.add_subplot(gs[4, 0])
    base.unified_milor_panel(ax_i, milo_summary)
    panel_letter(ax_i, "I", x=-0.055, y=1.035)
    remove_matching_text(
        fig,
        ("MiloR differential abundance", "SpatialFDR < 0.05", "Color: median log2 FC"),
    )

    out = OUT / "Figure_1_CellPress_compact_preview.png"
    fig.savefig(OUT / "Figure_1_CellPress_compact_preview.pdf", facecolor="white")
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(
        OUT / "Figure_1_CellPress_compact_preview.tif", dpi=500, facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

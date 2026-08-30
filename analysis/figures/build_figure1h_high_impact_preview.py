from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(r"C:/path/to/private-manuscript-workspace")
SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
LEVEL3_SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives" / "Source Data"
OUT = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives"

MARKER_SOURCE = SOURCE / "Figure1_canonical_marker_dotplot.csv"
CLONE_SOURCE = SOURCE / "Figure1_clone_engagement_umap.csv"
PNG_OUT = OUT / "Figure_1H_high_impact_preview.png"
WEBP_OUT = OUT / "Figure_1H_high_impact_preview.webp"
SUMMARY_OUT = OUT / "Figure_1H_clone_engagement_preview_source.csv"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


FACETS = {
    "CD4": {
        "title": "CD4 T Cells",
        "color": "#7656A8",
        "file": "Figure_1E_CD4_Level3_UMAP.csv",
        "states": ["CD4 Naive", "CD4 Tfh", "CD4 Th17", "CD4 Temra", "TReg Cytotoxic"],
        "labels": {
            "CD4 Naive": "CD4 naive",
            "CD4 Tfh": "CD4 Tfh",
            "CD4 Th17": "CD4 Th17",
            "CD4 Temra": "CD4 TEMRA",
            "TReg Cytotoxic": "Cytotoxic Treg",
        },
        "genes": ["CCR7", "IL7R", "CXCR5", "PDCD1", "KLRB1", "RORA", "FOXP3", "CTLA4"],
    },
    "CD8": {
        "title": "CD8 T Cells",
        "color": "#168C78",
        "file": "Figure_1F_CD8_Level3_UMAP.csv",
        "states": ["CD8 Naive", "CD8 Tcm CCR4-", "CD8 Temra", "CD8 HLA-DR+", "CD8 Proliferative"],
        "labels": {
            "CD8 Naive": "CD8 naive",
            "CD8 Tcm CCR4-": "CD8 Tcm CCR4−",
            "CD8 Temra": "CD8 TEMRA",
            "CD8 HLA-DR+": "CD8 HLA-DR+",
            "CD8 Proliferative": "CD8 proliferative",
        },
        "genes": ["CCR7", "LTB", "GZMK", "NKG7", "CCL5", "GZMB", "PRF1", "HLA-DRA", "MKI67"],
    },
    "B": {
        "title": "B Cells",
        "color": "#C23B78",
        "file": "Figure_1G_B_Level3_UMAP.csv",
        "states": ["Naive B", "Switched memory B", "Atypical memory B", "IgA Plasma B Cell", "IgG Plasma B Cell"],
        "labels": {
            "Naive B": "Naive B",
            "Switched memory B": "Switched-memory B",
            "Atypical memory B": "Atypical-memory B",
            "IgA Plasma B Cell": "IgA plasma cell",
            "IgG Plasma B Cell": "IgG plasma cell",
        },
        "genes": ["MS4A1", "CD79A", "TCL1A", "IGHD", "CD27", "FCRL5", "TBX21", "MZB1", "JCHAIN", "XBP1"],
    },
}


def prepare_clone_summary():
    clone = pd.read_csv(CLONE_SOURCE, usecols=["cell", "panel", "paired_family", "paired_clone_size", "expanded_paired"])
    clone["paired"] = clone["paired_family"].astype(str).ne("unpaired") & (pd.to_numeric(clone["paired_clone_size"], errors="coerce").fillna(0) >= 1)
    clone["expanded"] = clone["expanded_paired"].astype(str).str.lower().eq("true")

    rows = []
    for panel, spec in FACETS.items():
        state = pd.read_csv(
            LEVEL3_SOURCE / spec["file"],
            usecols=["cell", "PatientID", "AnnotationLevel3"],
        )
        merged = state.merge(clone[clone["panel"].eq(panel)][["cell", "paired", "expanded"]], on="cell", how="left")
        merged[["paired", "expanded"]] = merged[["paired", "expanded"]].fillna(False).astype(bool)
        merged = merged[merged["AnnotationLevel3"].isin(spec["states"])].copy()

        participant = (
            merged.groupby(["AnnotationLevel3", "PatientID"], observed=True)
            .agg(total_cells=("cell", "size"), paired_cells=("paired", "sum"), expanded_cells=("expanded", "sum"))
            .reset_index()
        )
        participant["paired_recovery_pct"] = 100 * participant["paired_cells"] / participant["total_cells"]
        participant["expanded_among_paired_pct"] = np.where(
            participant["paired_cells"] > 0,
            100 * participant["expanded_cells"] / participant["paired_cells"],
            np.nan,
        )

        for state_name in spec["states"]:
            subset = participant[participant["AnnotationLevel3"].eq(state_name)]
            paired_subset = subset["paired_recovery_pct"].dropna()
            expanded_subset = subset["expanded_among_paired_pct"].dropna()
            rows.append(
                {
                    "panel": panel,
                    "cell_state": state_name,
                    "participants_with_state": int(subset["PatientID"].nunique()),
                    "participants_with_paired_receptor": int((subset["paired_cells"] > 0).sum()),
                    "paired_recovery_median_pct": float(paired_subset.median()) if len(paired_subset) else np.nan,
                    "paired_recovery_q1_pct": float(paired_subset.quantile(0.25)) if len(paired_subset) else np.nan,
                    "paired_recovery_q3_pct": float(paired_subset.quantile(0.75)) if len(paired_subset) else np.nan,
                    "expanded_among_paired_median_pct": float(expanded_subset.median()) if len(expanded_subset) else np.nan,
                    "expanded_among_paired_q1_pct": float(expanded_subset.quantile(0.25)) if len(expanded_subset) else np.nan,
                    "expanded_among_paired_q3_pct": float(expanded_subset.quantile(0.75)) if len(expanded_subset) else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_OUT, index=False)
    return summary


def marker_size(percent):
    return 10 + 85 * np.sqrt(np.clip(percent, 0, 100) / 100)


def clone_size(percent):
    return 18 + 155 * np.sqrt(np.clip(percent, 0, 100) / 100)


def draw_marker_matrix(ax, marker_data, panel, spec):
    states = spec["states"]
    genes = spec["genes"]
    data = marker_data[marker_data["cell_state"].isin(states) & marker_data["gene"].isin(genes)].copy()
    data["x"] = data["gene"].map({gene: i for i, gene in enumerate(genes)})
    data["y"] = data["cell_state"].map({state: i for i, state in enumerate(states)})

    norm = mpl.colors.Normalize(vmin=-2.5, vmax=2.5)
    scatter = ax.scatter(
        data["x"],
        data["y"],
        s=marker_size(data["pct_expressing"].to_numpy(float)),
        c=data["avg_expression_scaled"],
        cmap="RdBu_r",
        norm=norm,
        edgecolor="#555555",
        linewidth=0.32,
        zorder=3,
    )

    ax.set_xticks(range(len(genes)), genes, rotation=58, ha="right", rotation_mode="anchor")
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("italic")
    ax.set_yticks(range(len(states)), [spec["labels"][state] for state in states])
    for tick in ax.get_yticklabels():
        tick.set_color(spec["color"])
        tick.set_fontweight("bold")
    ax.set_xlim(-0.55, len(genes) - 0.45)
    ax.set_ylim(len(states) - 0.50, -0.50)
    ax.set_axisbelow(True)
    ax.grid(color="#E5E7EB", linewidth=0.55)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(spec["title"], color=spec["color"], fontweight="bold", pad=11)
    return scatter


def draw_clone_strip(ax, clone_summary, panel, spec):
    states = spec["states"]
    data = clone_summary[clone_summary["panel"].eq(panel)].set_index("cell_state").reindex(states)
    metrics = ["paired_recovery_median_pct", "expanded_among_paired_median_pct"]
    xlabels = ["Paired", "Expanded"]
    for yi, state in enumerate(states):
        for xi, metric in enumerate(metrics):
            value = data.loc[state, metric]
            if not np.isfinite(value):
                ax.scatter(xi, yi, marker="x", s=34, color="#9CA3AF", linewidth=1.0, zorder=3)
                continue
            ax.scatter(
                xi,
                yi,
                s=clone_size(value),
                facecolor=spec["color"],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.88,
                zorder=3,
            )
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(len(states) - 0.50, -0.50)
    ax.set_xticks([0, 1], xlabels)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=3, labelsize=6.2)
    ax.set_axisbelow(True)
    ax.grid(color="#E5E7EB", linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Clone engagement", color="#374151", fontsize=7.6, fontweight="bold", pad=10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    marker_data = pd.read_csv(MARKER_SOURCE)
    clone_summary = prepare_clone_summary()

    fig = plt.figure(figsize=(14.5, 5.25), facecolor="white")
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.00, 1.03, 1.12],
        wspace=0.38,
        left=0.09,
        right=0.982,
        top=0.77,
        bottom=0.24,
    )

    last_scatter = None
    for index, (panel, spec) in enumerate(FACETS.items()):
        inner = outer[0, index].subgridspec(1, 2, width_ratios=[len(spec["genes"]), 2.35], wspace=0.08)
        marker_ax = fig.add_subplot(inner[0, 0])
        clone_ax = fig.add_subplot(inner[0, 1])
        last_scatter = draw_marker_matrix(marker_ax, marker_data, panel, spec)
        draw_clone_strip(clone_ax, clone_summary, panel, spec)

    fig.text(0.025, 0.945, "H", fontsize=17, fontweight="bold", ha="left", va="top")
    fig.text(0.065, 0.945, "Canonical identity and clone engagement", fontsize=12.2, fontweight="bold", ha="left", va="top")
    fig.text(
        0.065,
        0.885,
        "Level 3 transcriptional states aligned to participant-level paired-receptor recovery and expanded-clonotype occupancy",
        fontsize=8.2,
        color="#4B5563",
        ha="left",
        va="top",
    )

    cax = fig.add_axes([0.095, 0.095, 0.20, 0.025])
    colorbar = fig.colorbar(last_scatter, cax=cax, orientation="horizontal")
    colorbar.set_ticks([-2.5, 0, 2.5])
    colorbar.ax.tick_params(labelsize=6.7, length=2, pad=1)
    colorbar.outline.set_linewidth(0.5)
    fig.text(0.195, 0.145, "Scaled mean expression", fontsize=7.2, color="#4B5563", ha="center")

    expression_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#555555",
               markersize=np.sqrt(marker_size(value)), label=f"{value}%")
        for value in (25, 75)
    ]
    fig.legend(
        handles=expression_handles,
        title="Cells expressing",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.325, 0.065),
        ncol=2,
        fontsize=6.8,
        title_fontsize=7.2,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    clone_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#6B7280", markeredgecolor="white",
               markersize=np.sqrt(clone_size(value)), label=f"{value}%")
        for value in (10, 50, 90)
    ]
    fig.legend(
        handles=clone_handles,
        title="Participant median",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.525, 0.065),
        ncol=3,
        fontsize=6.8,
        title_fontsize=7.2,
        columnspacing=0.75,
        handletextpad=0.30,
    )
    fig.text(
        0.982,
        0.105,
        "Expanded fraction is calculated among paired-receptor cells; × indicates insufficient paired-receptor support.",
        fontsize=6.8,
        color="#6B7280",
        ha="right",
        va="center",
    )

    fig.savefig(PNG_OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    with Image.open(PNG_OUT) as image:
        width = 1900
        height = round(image.height * width / image.width)
        image.resize((width, height), Image.Resampling.LANCZOS).save(WEBP_OUT, "WEBP", quality=92, method=6)

    print(PNG_OUT)
    print(WEBP_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()

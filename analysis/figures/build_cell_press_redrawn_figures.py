from __future__ import annotations

from pathlib import Path
import colorsys
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
import logomaker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pypdf import PdfReader, PdfWriter
from scipy.stats import mannwhitneyu, pearsonr, rankdata


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Cell Press Redrawn Figure Set"
MAIN = OUT / "Main Figures"
SUPP = OUT / "Supplementary Figures"
SRC = OUT / "Source Data"
LEG = OUT / "Legends"
ASSET = OUT / "Graphical Assets"
for directory in (MAIN, SUPP, SRC, LEG, ASSET):
    directory.mkdir(parents=True, exist_ok=True)

TCR_DIR = Path(r"C:\Users\johng\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 2 TCR")
BCR_DIR = Path(r"C:\Users\johng\OneDrive\Desktop\IBD SingleCell Repertoire Manuscript\Figure 3 BCR")
PBMC_DIR = Path(r"C:\Users\johng\OneDrive\Desktop\IBD PBMC Immune Repertoire Manuscript Figures")

# Okabe-Ito-derived palette; group identity is reinforced by position and labels.
COL = {
    "Control": "#6F6F6F",
    "CD": "#0072B2",
    "UC": "#D55E00",
    "TCR": "#0072B2",
    "BCR": "#D55E00",
    "Joint": "#009E73",
    "negative": "#0072B2",
    "positive": "#D55E00",
    "ink": "#222222",
    "muted": "#777777",
    "grid": "#D8D8D8",
}
DIAG = ["Control", "CD", "UC"]

# Ordered clone-size palette used consistently across the adaptive UMAPs.
# A single-hue magenta-purple sequence distinguishes clone magnitude from the
# gray-blue-orange diagnosis colors used in the preceding recovery panel.
CLONE_SIZE_STYLES = [
    ("Small (2-5)", 1, 5, "#D9A6D1", 0.86),
    ("Medium (6-20)", 5, 20, "#C45BAA", 1.05),
    ("Large (21-100)", 20, 100, "#8E3C97", 1.34),
    ("Hyperexpanded (>100)", 100, np.inf, "#4B1D6B", 1.72),
]

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
)


def read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def clean_label(value: str) -> str:
    return str(value).replace("_", " ").replace("NFkB", "NF-kB").replace(" Tc1 ", "/Tc1 ")


def style_axis(ax, grid_axis: str | None = None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["ink"])
    ax.spines["bottom"].set_color(COL["ink"])
    ax.tick_params(color=COL["ink"], labelcolor=COL["ink"], pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COL["grid"], lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.16, y: float = 1.10):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def panel_title(ax, title: str):
    ax.set_title(title, loc="left", fontweight="bold", pad=4)


def figure5_heading(ax, label: str, title: str, full_width: bool = False):
    """Draw a Figure 5 panel letter and title above, not beside, the y-axis."""
    title_x = 0.055 if full_width else 0.10
    # Keep the panel letter in the heading gutter, clearly left of the y-axis.
    # A smaller offset is sufficient for full-width panels because their axes
    # span the complete Cell Press canvas.
    label_x = -0.025 if full_width else -0.055
    heading_y = 1.105
    ax.text(label_x, heading_y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", clip_on=False)
    ax.text(title_x, heading_y, title, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left", clip_on=False)


def save_figure(fig, stem: str, folder: Path, fixed_canvas: bool = False):
    # Retain all scientific labels in the exported artwork. The 7.48-inch
    # design grid is optimized for 190-mm full-width placement; the tight crop
    # removes unused canvas while preventing labels from being clipped.
    crop = {} if fixed_canvas else {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(folder / f"{stem}.pdf", facecolor="white", **crop)
    fig.savefig(folder / f"{stem}.png", dpi=300, facecolor="white", **crop)
    fig.savefig(
        folder / f"{stem}.tif",
        dpi=500,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
        **crop,
    )
    plt.close(fig)


def add_group_legend(ax, loc="upper right", ncol=3):
    handles = [
        Line2D([0], [0], marker="o", ls="", ms=4.5, color=COL[d], label=d)
        for d in DIAG
    ]
    ax.legend(handles=handles, frameon=False, loc=loc, ncol=ncol, handletextpad=0.3, columnspacing=0.8)


def boxstrip(ax, df: pd.DataFrame, value: str, group: str = "Diagnosis", order=DIAG, ylabel=""):
    values = [pd.to_numeric(df.loc[df[group] == g, value], errors="coerce").dropna().to_numpy() for g in order]
    bp = ax.boxplot(
        values,
        positions=np.arange(len(order)),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COL["ink"], "lw": 1.0},
        whiskerprops={"color": COL["ink"], "lw": 0.7},
        capprops={"color": COL["ink"], "lw": 0.7},
        boxprops={"color": COL["ink"], "lw": 0.7},
    )
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(COL[g])
        patch.set_alpha(0.23)
    rng = np.random.default_rng(20260824)
    for i, (g, vals) in enumerate(zip(order, values)):
        x = rng.normal(i, 0.055, len(vals))
        ax.scatter(x, vals, s=7, color=COL[g], alpha=0.62, edgecolor="none", zorder=3, rasterized=True)
    ax.set_xticks(range(len(order)), [f"{g}\n(n={len(v)})" for g, v in zip(order, values)])
    ax.set_ylabel(ylabel)
    style_axis(ax, "y")


def heatmap(ax, matrix: pd.DataFrame, cmap="RdBu_r", center=0, cbar_label="", annotate=False, vlim=None,
            show_cbar=True):
    values = matrix.to_numpy(dtype=float)
    if vlim is None and center == 0:
        vlim = np.nanpercentile(np.abs(values), 95)
        if vlim == 0 or not np.isfinite(vlim):
            vlim = 1
    kwargs = {"cmap": cmap, "aspect": "auto", "interpolation": "nearest"}
    if center == 0:
        kwargs.update(vmin=-vlim, vmax=vlim)
    im = ax.imshow(values, **kwargs)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), [clean_label(x) for x in matrix.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotate and matrix.shape[0] * matrix.shape[1] <= 60:
        ann_lim = vlim if vlim is not None else np.nanmax(np.abs(values))
        if not np.isfinite(ann_lim) or ann_lim == 0:
            ann_lim = 1
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(val) > 0.58 * ann_lim else COL["ink"])
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
        cb.set_label(cbar_label, fontsize=7)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    return im


def concise_module_label(value):
    labels = {
        "EOMES_ZEB2_inflammatory_CD8_TRM_like": "EOMES-ZEB2 CD8 TRM-like",
        "Tissue_resident_mucosal_retention": "TRM/mucosal retention",
        "Gut_homing_intestinal_trafficking": "Gut homing",
        "Th1_Tc1_inflammatory": "Th1/Tc1",
        "Effector_cytotoxicity": "Cytotoxicity",
        "Chronic_stimulation_exhaustion_like": "Chronic stimulation",
        "GZMK_inflammatory_memory": "GZMK inflammatory memory",
        "Naive_central_memory": "Naive/central memory",
        "Tph_Tfh_like_B_cell_help": "Tph/Tfh-like B-cell help",
        "Dominant_state_concordance": "Dominant-state concordance",
    }
    return labels.get(str(value), clean_label(value))


def lollipop(ax, labels, values, threshold=None, xlabel=""):
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]
    y = np.arange(len(values))
    colors = np.where(values >= 0, COL["positive"], COL["negative"])
    ax.hlines(y, 0, values, colors=colors, lw=1.1)
    ax.scatter(values, y, c=colors, s=21, zorder=3, edgecolor="white", lw=0.4)
    ax.axvline(0, color=COL["ink"], lw=0.6)
    if threshold is not None:
        ax.axvline(threshold, color=COL["muted"], lw=0.5, ls="--")
        ax.axvline(-threshold, color=COL["muted"], lw=0.5, ls="--")
    ax.set_yticks(y, [textwrap.fill(clean_label(x), 28) for x in labels])
    ax.set_xlabel(xlabel)
    style_axis(ax, "x")


def forest(ax, labels, estimate, low, high, xlabel="Effect estimate", colors=None):
    labels = list(labels)
    estimate = np.asarray(estimate, float)
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    y = np.arange(len(labels))[::-1]
    if colors is None:
        colors = [COL["ink"]] * len(labels)
    for yi, est, lo, hi, color in zip(y, estimate, low, high, colors):
        ax.plot([lo, hi], [yi, yi], color=color, lw=0.9)
        ax.scatter(est, yi, s=20, color=color, edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.6, ls="--")
    ax.set_yticks(y, [textwrap.fill(str(x), 38) for x in labels])
    ax.set_xlabel(xlabel)
    style_axis(ax, "x")


def scatter_fit(ax, x, y, color, xlabel, ylabel, annotation=None):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=12, color=color, alpha=0.68, edgecolor="none", rasterized=True)
    if len(x) >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, color=COL["ink"], lw=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax)
    if annotation:
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=6.5)


def merge_pdfs(paths: list[Path], destination: Path):
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)
    with destination.open("wb") as stream:
        writer.write(stream)


def workflow_panel(ax):
    ax.set_axis_off()
    labels = ["PBMC\nisolation", "Single-cell\nRNA-seq", "Paired\nTCR/BCR", "Cell state +\nclonotype"]
    x = np.linspace(0.03, 0.78, 4)
    for i, (xi, label) in enumerate(zip(x, labels)):
        box = FancyBboxPatch((xi, 0.34), 0.17, 0.28, boxstyle="round,pad=0.02,rounding_size=0.02",
                             fc="#F3F3F3", ec=COL["ink"], lw=0.7)
        ax.add_patch(box)
        ax.text(xi + 0.085, 0.48, label, ha="center", va="center", fontsize=7)
        if i < 3:
            ax.add_patch(FancyArrowPatch((xi + 0.17, 0.48), (x[i + 1], 0.48), arrowstyle="-|>",
                                         mutation_scale=8, color=COL["muted"], lw=0.8))
    ax.text(0.03, 0.18, "Participant-level analyses", fontsize=7, fontweight="bold")
    ax.text(0.03, 0.10, "Acquisition series retained for replication and sensitivity analyses", fontsize=6.5, color=COL["muted"])


def umap_panel(ax, df, title, max_labels=12, direct_labels=0):
    cats = pd.Series(df["AnnotationLevel2"].astype(str).fillna("Unknown")).unique().tolist()
    cmap = plt.get_cmap("turbo", max(2, len(cats)))
    mapping = {cat: cmap(i) for i, cat in enumerate(cats)}
    for cat in cats:
        sub = df[df["AnnotationLevel2"].astype(str) == cat]
        ax.scatter(sub["UMAP_1"], sub["UMAP_2"], s=0.45, color=mapping[cat], alpha=0.55,
                   edgecolor="none", rasterized=True)
    ax.set_title(title, fontweight="bold", pad=3)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if direct_labels:
        top = df["AnnotationLevel2"].astype(str).value_counts().head(direct_labels).index
        for cat in top:
            sub = df[df["AnnotationLevel2"].astype(str) == cat]
            ax.text(sub["UMAP_1"].median(), sub["UMAP_2"].median(), textwrap.shorten(cat, 18, placeholder="…"),
                    fontsize=5.5, ha="center", va="center", color=COL["ink"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5})
    if len(cats) <= max_labels:
        handles = [Line2D([0], [0], marker="o", ls="", ms=3.5, color=mapping[c], label=c) for c in cats]
        ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=5.7,
                  handletextpad=0.2, labelspacing=0.2)


def state_family_color(state: str):
    s = str(state).lower()
    if "plasma" in s or "plasmablast" in s:
        return "#D55E00"
    if "atypical" in s:
        return "#CC79A7"
    if "memory b" in s:
        return "#009E73"
    if "naive b" in s or "naive-ifn b" in s:
        return "#56B4E9"
    if "transitional" in s or "cd5+ b" in s:
        return "#E69F00"
    if "treg" in s:
        return "#CC79A7"
    if "mait" in s or "gdt" in s or "gamma" in s:
        return "#009E73"
    if "th17" in s or "th22" in s:
        return "#7CAE00"
    if "tfh" in s or "th2" in s:
        return "#E69F00"
    if "hla-dr" in s or "cytotoxic" in s or "terminal" in s or "temra" in s or "gzmb" in s or "prolifer" in s:
        return "#D55E00"
    if "naive" in s or "tcm" in s:
        return "#56B4E9"
    if "cd8" in s:
        return "#0072B2"
    if "cd4" in s:
        return "#648FFF"
    return "#7A7A7A"


def semantic_state_colors(states):
    """Return high-contrast, biologically organized colors for adaptive states."""
    grouped = {}
    for state in states:
        grouped.setdefault(state_family_color(state), []).append(state)
    colors = {}
    for base, group_states in grouped.items():
        group_states = sorted(group_states)
        hue, lightness, saturation = colorsys.rgb_to_hls(*mpl.colors.to_rgb(base))
        count = max(len(group_states), 2)
        offsets = np.linspace(-0.025, 0.025, count)
        lightness_values = np.linspace(max(0.36, lightness - 0.09), min(0.68, lightness + 0.11), count)
        for state, offset, state_lightness in zip(group_states, offsets, lightness_values):
            rgb = colorsys.hls_to_rgb((hue + offset) % 1.0, state_lightness, min(0.82, saturation + 0.08))
            colors[state] = mpl.colors.to_hex(rgb)
    return colors


def biological_umap_panel(ax, df, title, subtitle=None, direct_labels=4, label_states=None,
                          clone_label="Expanded paired clone", clone_summary=None,
                          lineage_header=None, background_cap=75000):
    cats = df["AnnotationLevel2"].astype(str)
    order = cats.value_counts().index.tolist()
    state_colors = semantic_state_colors(order)

    # Use the same maximum background density in all three lineage panels.
    # Sampling is stratified by cell state and diagnosis and affects display only.
    background = df
    if len(df) > background_cap:
        fraction = background_cap / len(df)
        strata = ["AnnotationLevel2"] + (["Diagnosis1"] if "Diagnosis1" in df.columns else [])
        sampled_groups = []
        for sample_index, (_, group) in enumerate(df.groupby(strata, sort=True, observed=True)):
            target = min(len(group), max(1, int(round(len(group) * fraction))))
            sampled_groups.append(group.sample(target, random_state=20260826 + sample_index))
        background = pd.concat(sampled_groups, ignore_index=False)

    for cat in order:
        sub = background[background["AnnotationLevel2"].astype(str).eq(cat)]
        ax.scatter(sub["UMAP_1"], sub["UMAP_2"], s=0.44, color=state_colors[cat], alpha=0.17,
                   edgecolor="none", rasterized=True)
    expanded = df[df.get("expanded_paired", False).fillna(False).astype(bool)] if "expanded_paired" in df else df.iloc[0:0]
    if not expanded.empty:
        expanded_total_display_embedding = len(expanded)
        expanded = expanded.copy()
        expanded["clone_size_bin"] = pd.cut(
            pd.to_numeric(expanded["paired_clone_size"], errors="coerce"),
            bins=[1, 5, 20, 100, np.inf],
            labels=[style[0] for style in CLONE_SIZE_STYLES],
            right=True,
        )
        expanded_display = expanded
        overlay_sampled = expanded_total_display_embedding > 8000
        if overlay_sampled:
            sampled_groups = []
            for sample_index, (_, group) in enumerate(
                expanded.groupby(["AnnotationLevel2", "clone_size_bin"], observed=True, sort=True)
            ):
                sampled_groups.append(group.sample(min(len(group), 180), random_state=20260826 + sample_index))
            expanded_display = pd.concat(sampled_groups, ignore_index=False)
        base_size = 4.20 if expanded_total_display_embedding < 500 else (
            1.55 if expanded_total_display_embedding < 5000 else 1.02
        )
        # Plot small-to-large categories so the largest clones remain visible.
        for label, _, _, color, size_multiplier in CLONE_SIZE_STYLES:
            sub = expanded_display[expanded_display["clone_size_bin"].astype(str).eq(label)]
            if sub.empty:
                continue
            ax.scatter(
                sub["UMAP_1"], sub["UMAP_2"], s=base_size * size_multiplier, color=color,
                alpha=0.98 if expanded_total_display_embedding < 500 else (
                    0.94 if expanded_total_display_embedding < 5000 else 0.90
                ),
                edgecolor="white", linewidth=0.20, rasterized=False, zorder=4,
            )
    ax.set_title(title, fontweight="bold", fontsize=7.3, pad=9)
    if subtitle:
        ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.4, color=COL["muted"])
    if lineage_header:
        header_color = COL["TCR"] if lineage_header == "TCR" else COL["BCR"]
        ax.text(0.0, 1.105, lineage_header, transform=ax.transAxes, ha="left", va="center",
                fontsize=5.5, fontweight="bold", color=header_color, clip_on=False)
        ax.plot([0.12, 1.0], [1.105, 1.105], transform=ax.transAxes, color=header_color,
                lw=0.85, alpha=0.76, clip_on=False)
    if not expanded.empty and clone_summary is not None:
        summary = clone_summary
        full_expanded = int(summary["expanded_cells"])
        full_clonotypes = int(summary["expanded_clonotypes"])
        full_participants = int(summary["expanded_participants"])
        expanded_pct = float(summary["expanded_pct_of_paired"])
        summary_lines = [
            f"Full data: {full_expanded:,} expanded cells | {full_clonotypes:,} clonotypes",
            f"{full_participants} participants | {expanded_pct:.1f}% of paired-receptor cells",
        ]
        overlay_shown = len(expanded_display)
        if expanded_total_display_embedding != full_expanded or overlay_sampled:
            summary_lines.append(f"Display overlay: {overlay_shown:,}/{full_expanded:,} expanded cells")
        ax.text(
            0.5, 0.010, "\n".join(summary_lines), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=4.20, linespacing=1.05, color=COL["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.45}, zorder=12,
        )
    elif not expanded.empty:
        ax.text(
            0.01, 0.015, f"{clone_label}: n={expanded_total_display_embedding:,}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=4.7, color=COL["ink"],
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")], zorder=12,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    selected_labels = [cat for cat in (label_states or order[:direct_labels]) if cat in set(order)]
    display_labels = {
        "TReg Cytotoxic": "Cytotoxic Treg",
        "CD8 Tem GZMB+": "GZMB+ effector",
        "CD8 HLA-DR+": "HLA-DR+ CD8",
        "Switched memory B": "Switched memory",
        "Atypical memory B": "Atypical memory",
        "IgM Plasma B Cell": "IgM plasma",
        "IgA Plasma B Cell": "IgA plasma",
        "IgG Plasma B Cell": "IgG plasma",
    }
    label_rows = []
    for cat in selected_labels:
        sub = df[cats == cat]
        label_rows.append([cat, float(sub["UMAP_1"].median()), float(sub["UMAP_2"].median())])
    y_range = max(float(df["UMAP_2"].max() - df["UMAP_2"].min()), 1)
    x_range = max(float(df["UMAP_1"].max() - df["UMAP_1"].min()), 1)
    x_center = float(df["UMAP_1"].median())
    y_center = float(df["UMAP_2"].median())
    gap = 0.045 * y_range
    label_rows.sort(key=lambda x: x[2])
    placed = []
    for label_index, (cat, x0, y0) in enumerate(label_rows):
        direction = np.array([(x0 - x_center) / x_range, (y0 - y_center) / y_range], dtype=float)
        length = float(np.linalg.norm(direction))
        if length < 0.075:
            angle = 2 * np.pi * label_index / max(len(label_rows), 1)
            direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            direction /= length
        x_label = x0 + direction[0] * 0.055 * x_range
        y_label = y0 + direction[1] * 0.055 * y_range
        manual_offsets = {
            "CD8 Tem GZMB+": (-0.045, 0.045),
            "CD8 HLA-DR+": (0.035, -0.050),
            "Atypical memory B": (-0.075, 0.000),
            "IgM Plasma B Cell": (0.000, 0.045),
            "IgA Plasma B Cell": (-0.080, 0.090),
            "IgG Plasma B Cell": (-0.145, -0.045),
            "Switched memory B": (0.040, 0.015),
        }
        dx, dy = manual_offsets.get(cat, (0.0, 0.0))
        x_label += dx * x_range
        y_label += dy * y_range
        for _, xp, yp in placed:
            if abs(x_label - xp) < 0.28 * x_range and abs(y_label - yp) < gap:
                y_label = yp + gap
        placed.append((cat, x_label, y_label))
        annotation = ax.annotate(
            display_labels.get(cat, cat), xy=(x0, y0), xytext=(x_label, y_label),
            fontsize=5.05, fontweight="bold", ha="center", va="center",
            color=state_colors.get(cat, COL["ink"]),
            arrowprops={"arrowstyle": "-", "color": COL["muted"], "lw": 0.38},
            zorder=10,
        )
        annotation.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    xlo, xhi = np.nanquantile(df["UMAP_1"], [0.003, 0.997])
    ylo, yhi = np.nanquantile(df["UMAP_2"], [0.003, 0.997])
    xpad = max((xhi - xlo) * 0.070, 0.6)
    ypad = max((yhi - ylo) * 0.045, 0.5)
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)


def adaptive_lineage(state: str):
    s = str(state).lower()
    if "plasma" in s:
        return "Plasma cells"
    if " b" in s or s.startswith("b ") or "memory b" in s or "naive b" in s or "transitional b" in s:
        return "B cells"
    if "treg" in s:
        return "Treg"
    if "mait" in s or "gdt" in s:
        return "Unconventional T"
    if "cd8" in s:
        return "CD8 T"
    if "cd4" in s or "tfh" in s:
        return "CD4 T"
    return "Other"


def integrated_adaptive_umap(ax, pbmc, major_summary):
    lineage_colors = {
        "CD4 T": "#648FFF", "CD8 T": "#0072B2", "Treg": "#CC79A7",
        "Unconventional T": "#009E73", "B cells": "#E69F00", "Plasma cells": "#D55E00",
    }
    data = pbmc.copy()
    data["major_lineage"] = data["AnnotationLevel2"].map(adaptive_lineage)
    data = data[data["major_lineage"].isin(lineage_colors)]
    offsets = {
        "CD4 T": (-0.2, 0.5), "CD8 T": (1.5, -0.9), "Treg": (-0.2, -0.6),
        "Unconventional T": (-1.8, 1.2), "B cells": (0.0, -0.7), "Plasma cells": (0.0, 0.65),
    }
    for lineage in lineage_colors:
        sub = data[data["major_lineage"] == lineage]
        ax.scatter(sub["UMAP_1"], sub["UMAP_2"], s=0.44, color=lineage_colors[lineage], alpha=0.52,
                   edgecolor="none", rasterized=True)
        if len(sub):
            x0, y0 = float(sub["UMAP_1"].median()), float(sub["UMAP_2"].median())
            dx, dy = offsets[lineage]
            ax.annotate(lineage, xy=(x0, y0), xytext=(x0 + dx, y0 + dy), fontsize=5.7,
                        ha="center", va="center", color=COL["ink"],
                        arrowprops={"arrowstyle": "-", "color": COL["muted"], "lw": 0.35},
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.45})
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    xlo, xhi = np.nanquantile(data["UMAP_1"], [0.003, 0.997])
    ylo, yhi = np.nanquantile(data["UMAP_2"], [0.003, 0.997])
    ax.set_xlim(xlo - 0.03 * (xhi - xlo), xhi + 0.03 * (xhi - xlo))
    ax.set_ylim(ylo - 0.03 * (yhi - ylo), yhi + 0.03 * (yhi - ylo))


def pbmc_state_family(state: str):
    s = str(state).lower()
    if "monocyte" in s:
        return "Monocytes"
    if s.startswith("cdc") or s == "pdc":
        return "Dendritic cells"
    if s.startswith("cd4") or s.startswith("treg"):
        return "CD4/Treg"
    if s.startswith("cd8") or s in {"mait", "gdt"}:
        return "CD8/innate-like T"
    if " b" in s or s.endswith(" b") or "plasma" in s or s.startswith("naive b") or s.startswith("transitional b"):
        return "B/plasma"
    if s.startswith("nk") or s.startswith("ilc"):
        return "NK/ILC"
    return "Platelet/progenitor"


def pbmc_level2_colors(states):
    family_bases = {
        "Monocytes": "#A56624",
        "Dendritic cells": "#C6535A",
        "CD4/Treg": "#7656A8",
        "CD8/innate-like T": "#168C78",
        "B/plasma": "#C23B78",
        "NK/ILC": "#536575",
        "Platelet/progenitor": "#8C8174",
    }
    grouped = {}
    for state in states:
        grouped.setdefault(pbmc_state_family(state), []).append(state)
    colors = {}
    for family, family_states in grouped.items():
        family_states = sorted(family_states)
        base_rgb = mpl.colors.to_rgb(family_bases[family])
        hue, lightness, saturation = colorsys.rgb_to_hls(*base_rgb)
        count = max(len(family_states), 2)
        hue_offsets = np.linspace(-0.045, 0.045, count)
        lightness_values = np.linspace(0.38, 0.70, count)
        for state, hue_offset, state_lightness in zip(family_states, hue_offsets, lightness_values):
            rgb = colorsys.hls_to_rgb((hue + hue_offset) % 1.0, state_lightness, min(0.74, saturation + 0.10))
            colors[state] = mpl.colors.to_hex(rgb)
    return colors


def representative_umap_point(sub, x_col="UMAP_1", y_col="UMAP_2"):
    """Return an observed point nearest the robust two-dimensional center."""
    xy = sub[[x_col, y_col]].to_numpy(float)
    center = np.nanmedian(xy, axis=0)
    scale = np.nanpercentile(np.abs(xy - center), 75, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    distance = np.sum(((xy - center) / scale) ** 2, axis=1)
    return tuple(xy[int(np.nanargmin(distance))])


def repel_umap_annotations(ax, annotations, max_iter=120):
    """Resolve label collisions in display coordinates while retaining arrows."""
    fig = ax.figure
    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [a.get_window_extent(renderer).expanded(1.08, 1.22) for a in annotations]
        shifts = np.zeros((len(annotations), 2), dtype=float)
        collisions = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                collisions += 1
                ci = np.array([(boxes[i].x0 + boxes[i].x1) / 2, (boxes[i].y0 + boxes[i].y1) / 2])
                cj = np.array([(boxes[j].x0 + boxes[j].x1) / 2, (boxes[j].y0 + boxes[j].y1) / 2])
                direction = ci - cj
                if np.linalg.norm(direction) < 0.1:
                    direction = np.array([1.0 if (i + j) % 2 else -1.0, 1.0])
                direction /= np.linalg.norm(direction)
                overlap_x = min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0)
                overlap_y = min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0)
                magnitude = min(4.0, 1.0 + 0.25 * min(overlap_x, overlap_y))
                shifts[i] += direction * magnitude
                shifts[j] -= direction * magnitude
        if not collisions:
            break
        axes_box = ax.get_window_extent()
        for ann, shift in zip(annotations, shifts):
            if not np.any(shift):
                continue
            current = ax.transData.transform(ann.get_position())
            moved = current + np.clip(shift, -5, 5)
            moved[0] = np.clip(moved[0], axes_box.x0 + 18, axes_box.x1 - 18)
            moved[1] = np.clip(moved[1], axes_box.y0 + 10, axes_box.y1 - 10)
            ann.set_position(ax.transData.inverted().transform(moved))


def full_pbmc_level2_umap(container_ax, pbmc, mode="main"):
    """PBMC atlas with a main-panel or complete supplementary annotation design."""
    container_ax.set_axis_off()
    supplemental = mode == "supplement"
    if supplemental:
        plot_ax = container_ax.inset_axes([0.00, 0.03, 0.68, 0.94])
        legend_ax = container_ax.inset_axes([0.70, 0.03, 0.30, 0.94])
        legend_ax.set_axis_off()
    else:
        # Slightly oversize and left-anchor the main atlas so the embedding
        # occupies more of panel C and sits closer to the figure margin.
        plot_ax = container_ax.inset_axes([-0.035, -0.020, 1.035, 1.010])
        legend_ax = None

    data = pbmc.dropna(subset=["UMAP_1", "UMAP_2", "AnnotationLevel2"]).copy()
    data["AnnotationLevel2"] = data["AnnotationLevel2"].astype(str)
    # The original PBMC embedding is taller than it is wide. For the landscape
    # main-panel slot, apply a rigid 90-degree rotation and reflection; this
    # preserves all inter-point distances and prevents aspect-ratio distortion.
    if supplemental:
        data["_plot_x"] = data["UMAP_1"]
        data["_plot_y"] = data["UMAP_2"]
        x_label, y_label = "UMAP 1", "UMAP 2"
    else:
        # A rigid -74-degree rotation aligns the long axis of the embedding
        # with the landscape panel. This preserves inter-point distances while
        # reducing non-data whitespace compared with the prior -90-degree view.
        theta = np.deg2rad(-74.0)
        data["_plot_x"] = data["UMAP_1"] * np.cos(theta) - data["UMAP_2"] * np.sin(theta)
        data["_plot_y"] = data["UMAP_1"] * np.sin(theta) + data["UMAP_2"] * np.cos(theta)
        x_label, y_label = "UMAP 1", "UMAP 2"
    counts = data["AnnotationLevel2"].value_counts()
    states = counts.index.tolist()
    colors = pbmc_level2_colors(states)

    # Abundant states form a quiet background; rare states are drawn last with
    # larger, more opaque marks so small populations remain visible.
    for state in states:
        sub = data[data["AnnotationLevel2"].eq(state)]
        n = int(counts[state])
        if n < 500:
            point_size, alpha = (0.64 if supplemental else 0.68), 0.82
        elif n < 1500:
            point_size, alpha = (0.48 if supplemental else 0.50), 0.66
        else:
            point_size, alpha = (0.30 if supplemental else 0.34), 0.39
        plot_ax.scatter(sub["_plot_x"], sub["_plot_y"], s=point_size, color=colors[state], alpha=alpha,
                        edgecolor="none", rasterized=True)

    label_map = {
        "Intermediate monocytes": "Intermediate mono.",
        "TReg Cytotoxic": "Cytotoxic Treg",
        "NK CD56dim CD57-": "NK CD56dim",
        "NK CD56bright": "NK CD56bright",
        "Switched memory B": "Switched memory B",
        "IgM Plasma B Cell": "IgM plasma",
    }
    if supplemental:
        anchor_states = [
            "CD14 Monocyte", "CD16 Monocyte", "Intermediate monocytes", "cDC2", "pDC",
            "CD4 Naive", "CD4 Tfh", "CD4 Th17", "TReg Cytotoxic",
            "CD8 Naive", "CD8 Temra", "MAIT", "NK CD56bright", "NK CD56dim CD57-",
            "Naive B", "Switched memory B", "IgM Plasma B Cell", "Platelets",
            "cDC1", "CD4 Temra", "CD8 HLA-DR+", "gdT", "Atypical memory B",
            "IgA Plasma B Cell", "ILC1",
        ]
    else:
        anchor_states = [
            "CD14 Monocyte", "pDC", "CD4 Naive", "TReg Cytotoxic", "CD8 Naive",
            "CD8 Temra", "MAIT", "Naive B", "IgM Plasma B Cell", "Platelets",
        ]
    label_states = [state for state in anchor_states if state in counts.index]

    xlo, xhi = np.nanquantile(data["_plot_x"], [0.002, 0.998])
    ylo, yhi = np.nanquantile(data["_plot_y"], [0.002, 0.998])
    xspan, yspan = xhi - xlo, yhi - ylo
    xcenter, ycenter = np.nanmedian(data[["_plot_x", "_plot_y"]].to_numpy(float), axis=0)
    plot_padding = 0.035 if supplemental else 0.010
    plot_ax.set_xlim(xlo - plot_padding * xspan, xhi + plot_padding * xspan)
    plot_ax.set_ylim(ylo - plot_padding * yspan, yhi + plot_padding * yspan)
    plot_ax.set_aspect("equal", adjustable="box", anchor="C" if supplemental else "W")

    annotations = []
    for index, state in enumerate(label_states):
        sub = data[data["AnnotationLevel2"].eq(state)]
        x0, y0 = representative_umap_point(sub, "_plot_x", "_plot_y")
        xtext, ytext = x0, y0
        if not supplemental:
            # Offset labels radially from the atlas center and retain leader
            # lines, reducing occlusion of compact central neighborhoods.
            direction = np.array([(x0 - xcenter) / xspan, (y0 - ycenter) / yspan], dtype=float)
            length = float(np.linalg.norm(direction))
            if length < 0.035:
                angle = 2 * np.pi * index / max(len(label_states), 1)
                direction = np.array([np.cos(angle), np.sin(angle)])
                length = 1.0
            direction /= length
            xtext = x0 + direction[0] * 0.065 * xspan
            ytext = y0 + direction[1] * 0.065 * yspan
        ann = plot_ax.annotate(
            label_map.get(state, state), xy=(x0, y0), xytext=(xtext, ytext),
            fontsize=5.3 if supplemental else 5.0, ha="center", va="center", color=COL["ink"],
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.32, "shrinkA": 2.5, "shrinkB": 1.5},
            zorder=10,
        )
        ann.set_path_effects([pe.withStroke(linewidth=1.45 if supplemental else 1.2, foreground="white")])
        annotations.append(ann)

    if supplemental:
        plot_ax.set_xlabel(x_label, labelpad=-1)
        plot_ax.set_ylabel(y_label, labelpad=-1)
    else:
        plot_ax.set_xlabel("")
        plot_ax.set_ylabel("")
        draw_umap_axis_glyph(plot_ax, x=0.045, y=0.055, length=0.095)
    plot_ax.set_xticks([]); plot_ax.set_yticks([])
    for spine in plot_ax.spines.values():
        spine.set_visible(False)
    repel_umap_annotations(plot_ax, annotations, max_iter=160 if supplemental else 120)

    family_order = ["Monocytes", "Dendritic cells", "CD4/Treg", "CD8/innate-like T", "B/plasma", "NK/ILC", "Platelet/progenitor"]
    family_colors = {
        "Monocytes": "#A56624", "Dendritic cells": "#C6535A", "CD4/Treg": "#7656A8",
        "CD8/innate-like T": "#168C78", "B/plasma": "#C23B78", "NK/ILC": "#536575",
        "Platelet/progenitor": "#8C8174",
    }
    if supplemental:
        legend_states = sorted(states, key=lambda x: (family_order.index(pbmc_state_family(x)), x))
        rows = int(math.ceil(len(legend_states) / 2))
        for i, state in enumerate(legend_states):
            col = i // rows
            row = i % rows
            x = col * 0.50
            y = 0.965 - row * 0.0350
            legend_ax.scatter([x + 0.012], [y], s=10, color=colors[state], edgecolor="none")
            legend_ax.text(x + 0.032, y, state, fontsize=4.8, ha="left", va="center", color=COL["ink"])
        legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
        legend_ax.text(0.0, 1.005, f"Complete key: {len(states)} level 2 states", fontsize=6.1,
                       fontweight="bold", ha="left", va="bottom")
    else:
        handles = [Line2D([0], [0], marker="o", ls="", ms=3.6, color=family_colors[f], label=f) for f in family_order]
        container_ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.50, -0.155),
                            ncol=7, fontsize=4.05, handletextpad=0.14, columnspacing=0.48,
                            labelspacing=0.25)


def draw_umap_axis_glyph(ax, *, x=0.055, y=0.060, length=0.13, color="#343434"):
    """Draw a compact, consistent UMAP orientation glyph in axes coordinates."""
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
                      ha="center", va="top", fontsize=3.6, color=color, zorder=31)
    label_y = ax.text(x - 0.020, y + length * 0.52, "UMAP 2", transform=ax.transAxes,
                      ha="right", va="center", rotation=90, fontsize=3.6, color=color, zorder=31)
    label_x.set_path_effects(halo)
    label_y.set_path_effects(halo)


def figure1_workflow(ax):
    ax.set_axis_off()
    # ChatGPT-generated plate-based workflow art. Labels remain code-native so
    # terminology is exact, editable, and legible at final journal dimensions.
    workflow_path = ASSET / "Figure_1B_ChatGPT_new_study_schematic_v6.png"
    workflow = plt.imread(workflow_path)
    # Remove the generous top/bottom whitespace from the generated 2:1 canvas
    # and map the scientific artwork into the wide Figure 1B panel.
    h = workflow.shape[0]
    workflow = workflow[int(h * 0.08):int(h * 0.86), :, :]
    # The cropped artwork and panel have nearly identical aspect ratios. Fill the
    # panel with the image and place labels in the inter-row whitespace so the
    # artwork is not horizontally stretched to make room for text.
    ax.imshow(workflow, extent=(0.00, 1.00, 0.00, 1.00), aspect="auto", interpolation="lanczos")

    x_positions = [0.095, 0.285, 0.485, 0.690, 0.895]
    labels = [
        "Cohort\nn=249", "PBMC\nisolation", "BD Rhapsody\nmicrowell capture",
        "WTA + paired\nTCR/BCR", "Cell state +\nclonotype",
    ]
    for x, label in zip(x_positions, labels):
        ax.text(x, -0.055, label, ha="center", va="top", fontsize=5.5, linespacing=0.95,
                fontweight="bold", color=COL["ink"], clip_on=False)
    ax.text(0.5, -0.315, "Barcode-linked RNA and receptor profiles  |  participant-level inference  |  10 acquisition series",
            ha="center", va="bottom", fontsize=5.8, color=COL["muted"], clip_on=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def cohort_panel(ax, participants):
    y = np.arange(3)[::-1]
    totals = participants.groupby("Diagnosis1").size().reindex(DIAG).fillna(0).astype(int)
    inflamed = participants[participants["Inflammation1"].eq("Inflamed")].groupby("Diagnosis1").size().reindex(DIAG).fillna(0).astype(int)
    exposed = participants[participants["biologic_exposed"].astype(str).str.lower().eq("true")].groupby("Diagnosis1").size().reindex(DIAG).fillna(0).astype(int)
    for yi, group in zip(y, DIAG):
        total = totals[group]
        ax.barh(yi, total, height=0.56, color=COL[group], alpha=0.22, edgecolor=COL[group], lw=0.8)
        if inflamed[group] > 0:
            ax.barh(yi, inflamed[group], height=0.56, color=COL[group], alpha=0.82, edgecolor="none")
        ax.text(total + 3, yi + 0.08, f"n={total}", va="center", fontsize=7, fontweight="bold", color=COL[group])
        detail = "non-IBD control" if group == "Control" else f"{inflamed[group]} inflamed; {exposed[group]} exposed"
        ax.text(total + 3, yi - 0.18, detail, va="center", fontsize=5.9, color=COL["muted"])
    ax.set_yticks(y, DIAG)
    ax.set_xlim(0, max(totals) * 1.50)
    ax.set_xlabel("Participants")
    style_axis(ax, "x")


def receptor_qc_panel(ax, qc):
    plot = qc[~qc["metric"].str.startswith("Quality")].copy()
    order = ["Productive TCR", "Paired alpha-beta", "Paired gamma-delta", "Productive BCR", "Paired heavy-light"]
    plot["metric"] = pd.Categorical(plot["metric"], order, ordered=True)
    plot = plot.sort_values("metric")
    y = np.arange(len(plot))[::-1]
    colors = [COL[x] for x in plot["compartment"]]
    ax.barh(y, plot["percent"], color=colors, alpha=0.82, height=0.58)
    for yi, (_, row) in zip(y, plot.iterrows()):
        x = max(float(row["percent"]) + 0.7, 1.1)
        ax.text(x, yi, f'{int(row["numerator"]):,} cells; {int(row["participants"])} participants',
                va="center", fontsize=5.7, color=COL["ink"])
    labels = [str(x).replace("alpha-beta", "alpha-beta").replace("gamma-delta", "gamma-delta").replace("heavy-light", "heavy-light") for x in plot["metric"]]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Receptor-positive cells within lineage (%)")
    ax.set_xlim(0, 57)
    ax.axhline(1.5, color=COL["grid"], lw=0.7)
    style_axis(ax, "x")


def participant_receptor_recovery_panel(ax, recovery):
    """Participant-level receptor QC with an expanded gamma-delta scale."""

    # Participants without any productive TCR or BCR were not receptor-evaluable.
    # Excluding them here prevents absent receptor libraries from being displayed as
    # biological zeroes. All five metrics therefore use the same 182 participants.
    if "repertoire_evaluable" in recovery.columns:
        evaluable = recovery["repertoire_evaluable"].astype(str).str.lower().isin(["true", "1"])
    else:
        evaluable = (
            pd.to_numeric(recovery["productive_tcr_n"], errors="coerce").fillna(0).gt(0)
            | pd.to_numeric(recovery["productive_bcr_n"], errors="coerce").fillna(0).gt(0)
        )
    plot = recovery.loc[evaluable].copy()

    ax.set_axis_off()
    ax.text(
        0.0, 0.965, f"249 profiled | {len(plot)} receptor-evaluable | pooled median [IQR]",
        transform=ax.transAxes, ha="left", va="top", fontsize=4.75, color=COL["muted"],
    )

    diagnosis_counts = plot["Diagnosis1"].value_counts()
    legend_handles = [
        Line2D(
            [0], [0], marker="o", ls="", ms=3.6, markerfacecolor=COL[diagnosis],
            markeredgecolor="white", markeredgewidth=0.25,
            label=f"{diagnosis} n={int(diagnosis_counts.get(diagnosis, 0))}",
        )
        for diagnosis in DIAG
    ]
    ax.legend(
        handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.905),
        ncol=3, fontsize=4.45, handletextpad=0.15, columnspacing=0.48, borderaxespad=0,
    )

    def draw_block(block_ax, metrics, xlim, xticks, show_xlabel=False, expanded=False,
                   show_xticklabels=True):
        row_positions = np.arange(len(metrics))[::-1]
        rng = np.random.default_rng(20260824 + len(metrics) + int(xlim[1]))
        diagnosis_offsets = {"Control": -0.13, "CD": 0.0, "UC": 0.13}
        tick_labels = []

        for yi, (label, column, denominator) in zip(row_positions, metrics):
            values = pd.to_numeric(plot[column], errors="coerce")
            valid_all = values.notna() & pd.to_numeric(plot[denominator], errors="coerce").gt(0)
            pooled = values[valid_all].to_numpy(float)
            q1, median, q3 = np.nanpercentile(pooled, [25, 50, 75])

            decimals = 2 if expanded else 1
            tick_labels.append(
                f"{label}\n{median:.{decimals}f} [{q1:.{decimals}f}-{q3:.{decimals}f}]"
            )
            for diagnosis in DIAG:
                valid = valid_all & plot["Diagnosis1"].eq(diagnosis)
                diagnosis_values = values[valid].to_numpy(float)
                jitter = rng.normal(0, 0.020 if len(metrics) == 1 else 0.026, len(diagnosis_values))
                block_ax.scatter(
                    diagnosis_values, yi + diagnosis_offsets[diagnosis] + jitter,
                    s=6.8, color=COL[diagnosis], alpha=0.54, edgecolor="white",
                    linewidth=0.12, rasterized=True, zorder=2,
                )

            block_ax.plot([q1, q3], [yi, yi], color=COL["ink"], lw=1.25, zorder=4)
            block_ax.scatter(
                [median], [yi], marker="D", s=18, facecolor="white",
                edgecolor=COL["ink"], lw=0.75, zorder=5,
            )

        block_ax.set_yticks(row_positions, tick_labels)
        block_ax.set_ylim(-0.45, len(metrics) - 0.55)
        block_ax.set_xlim(*xlim)
        block_ax.set_xticks(xticks)
        block_ax.tick_params(axis="y", labelsize=5.15, length=0, pad=2.5)
        block_ax.tick_params(axis="x", labelsize=5.0, length=2, pad=1.5)
        if not show_xticklabels:
            block_ax.tick_params(axis="x", labelbottom=False, length=0)
        style_axis(block_ax, "x")
        block_ax.spines["left"].set_visible(False)
        if show_xlabel:
            block_ax.set_xlabel("Receptor-positive cells within lineage (%)", fontsize=5.2, labelpad=2)
        if expanded:
            block_ax.text(
                0.99, 0.98, "expanded 0-4% scale", transform=block_ax.transAxes,
                ha="right", va="top", fontsize=4.35, color=COL["muted"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
            )

    ax.text(0.0, 0.825, "TCR", transform=ax.transAxes, ha="left", va="center",
            fontsize=5.7, fontweight="bold", color=COL["TCR"])
    ax.plot([0.105, 1.0], [0.825, 0.825], transform=ax.transAxes, color=COL["TCR"], lw=0.8, alpha=0.72)
    tcr_ax = ax.inset_axes([0.0, 0.555, 1.0, 0.235])
    draw_block(
        tcr_ax,
        [("Productive", "productive_tcr_pct", "t_cells"),
         ("Paired alpha-beta", "paired_alpha_beta_pct", "t_cells")],
        (0, 100), [0, 25, 50, 75, 100], show_xticklabels=False,
    )

    gamma_ax = ax.inset_axes([0.0, 0.405, 1.0, 0.105])
    draw_block(
        gamma_ax,
        [("Paired gamma-delta", "paired_gamma_delta_pct", "t_cells")],
        (0, 4), [0, 1, 2, 3, 4], expanded=True,
    )

    ax.text(0.0, 0.335, "BCR", transform=ax.transAxes, ha="left", va="center",
            fontsize=5.7, fontweight="bold", color=COL["BCR"])
    ax.plot([0.105, 1.0], [0.335, 0.335], transform=ax.transAxes, color=COL["BCR"], lw=0.8, alpha=0.72)
    bcr_ax = ax.inset_axes([0.0, 0.055, 1.0, 0.235])
    draw_block(
        bcr_ax,
        [("Productive", "productive_bcr_pct", "b_cells"),
         ("Paired heavy-light", "paired_heavy_light_pct", "b_cells")],
        (0, 100), [0, 25, 50, 75, 100], show_xlabel=True,
    )


def canonical_marker_dotplot(ax, marker_data):
    states = [
        "CD4 Naive", "CD4 Tfh", "CD4 Th17", "TReg Cytotoxic", "CD4 Temra",
        "CD8 Naive", "CD8 Tcm CCR4-", "CD8 HLA-DR+", "CD8 Temra", "CD8 Proliferative",
        "Naive B", "Switched memory B", "Atypical memory B", "IgA Plasma B Cell", "IgG Plasma B Cell",
    ]
    genes = [
        "CCR7", "LTB", "IL7R", "CXCR5", "PDCD1", "RORA", "KLRB1", "FOXP3", "CTLA4",
        "GZMK", "NKG7", "CCL5", "GZMB", "PRF1", "HLA-DRA", "MKI67",
        "MS4A1", "CD79A", "TCL1A", "IGHD", "CD27", "FCRL5", "TBX21", "MZB1", "JCHAIN", "XBP1",
    ]
    data = marker_data[marker_data["cell_state"].isin(states) & marker_data["gene"].isin(genes)].copy()
    state_index = {v: i for i, v in enumerate(states)}
    gene_index = {v: i for i, v in enumerate(genes)}
    data["x"] = data["gene"].map(gene_index)
    data["y"] = data["cell_state"].map(state_index)
    norm = mpl.colors.Normalize(vmin=-2.5, vmax=2.5)
    sizes = 8 + 68 * np.sqrt(np.clip(data["pct_expressing"].to_numpy(float), 0, 100) / 100)
    sc = ax.scatter(data["x"], data["y"], s=sizes, c=data["avg_expression_scaled"],
                    cmap="RdBu_r", norm=norm, edgecolor="#555555", lw=0.22)
    ax.set_xticks(range(len(genes)), genes, rotation=58, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(states)), states)
    ax.invert_yaxis()
    ax.set_xlim(-0.7, len(genes) - 0.3)
    ax.grid(color="#E3E3E3", lw=0.35)
    ax.set_axisbelow(True)
    for edge in (8.5, 15.5):
        ax.axvline(edge, color=COL["muted"], lw=0.75)
    for edge in (4.5, 9.5):
        ax.axhline(edge, color=COL["muted"], lw=0.75)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", labelsize=5.3, length=0)
    ax.tick_params(axis="y", labelsize=5.8, length=0)
    transform = ax.get_xaxis_transform()
    ax.text(4.0, 1.015, "CD4/Treg identity", transform=transform, ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    ax.text(12.0, 1.015, "Effector/activation", transform=transform, ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    ax.text(20.5, 1.015, "B/plasma identity", transform=transform, ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    cb = plt.colorbar(sc, ax=ax, fraction=0.024, pad=0.018)
    cb.set_label("Scaled average expression", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=2)
    cb.outline.set_linewidth(0.45)
    size_handles = [
        plt.scatter([], [], s=8 + 68 * math.sqrt(p / 100), facecolor="white", edgecolor="#555555", lw=0.35, label=f"{p}%")
        for p in (25, 50, 75)
    ]
    ax.legend(handles=size_handles, title="Cells expressing", frameon=False, loc="upper right",
              bbox_to_anchor=(0.995, 1.18), ncol=3, fontsize=5.2, title_fontsize=5.4,
              handletextpad=0.10, columnspacing=0.45, labelspacing=0.25)


FIGURE1H_FACETS = {
    "CD4": {
        "title": "CD4 T Cells", "color": "#7656A8",
        "states": ["CD4 Naive", "CD4 Tfh", "CD4 Th17", "CD4 Temra", "TReg Cytotoxic"],
        "labels": ["Naive", "Tfh", "Th17", "TEMRA", "Cytotoxic Treg"],
        "genes": ["CCR7", "IL7R", "CXCR5", "PDCD1", "KLRB1", "RORA", "FOXP3", "CTLA4"],
        "marker_position": [0.000, 0.245, 0.245, 0.620],
        "clone_position": [0.252, 0.245, 0.066, 0.620],
    },
    "CD8": {
        "title": "CD8 T Cells", "color": "#168C78",
        "states": ["CD8 Naive", "CD8 Tcm CCR4-", "CD8 Temra", "CD8 HLA-DR+", "CD8 Proliferative"],
        "labels": ["Naive", "Tcm CCR4−", "TEMRA", "HLA-DR+", "Proliferative"],
        "genes": ["CCR7", "LTB", "GZMK", "NKG7", "CCL5", "GZMB", "PRF1", "HLA-DRA", "MKI67"],
        "marker_position": [0.350, 0.245, 0.260, 0.620],
        "clone_position": [0.617, 0.245, 0.066, 0.620],
    },
    "B": {
        "title": "B Cells", "color": "#C23B78",
        "states": ["Naive B", "Switched memory B", "Atypical memory B", "IgA Plasma B Cell", "IgG Plasma B Cell"],
        "labels": ["Naive", "Switched", "Atypical", "IgA plasma", "IgG plasma"],
        "genes": ["MS4A1", "CD79A", "TCL1A", "IGHD", "CD27", "FCRL5", "TBX21", "MZB1", "JCHAIN", "XBP1"],
        "marker_position": [0.745, 0.245, 0.190, 0.620],
        "clone_position": [0.942, 0.245, 0.058, 0.620],
    },
}


def summarize_figure1h_clone_engagement(panel_frames):
    rows = []
    for panel, spec in FIGURE1H_FACETS.items():
        data = panel_frames[panel].copy()
        data["paired"] = (
            data["paired_family"].astype(str).ne("unpaired")
            & (pd.to_numeric(data["paired_clone_size"], errors="coerce").fillna(0) >= 1)
        )
        data["expanded"] = data["expanded_paired"].astype(str).str.lower().isin(["true", "1"])
        data = data[data["AnnotationLevel3"].isin(spec["states"])].copy()
        participant = (
            data.groupby(["AnnotationLevel3", "PatientID"], observed=True)
            .agg(total_cells=("cell", "size"), paired_cells=("paired", "sum"), expanded_cells=("expanded", "sum"))
            .reset_index()
        )
        participant["paired_recovery_pct"] = 100 * participant["paired_cells"] / participant["total_cells"]
        participant["expanded_among_paired_pct"] = np.where(
            participant["paired_cells"] > 0,
            100 * participant["expanded_cells"] / participant["paired_cells"],
            np.nan,
        )
        for state in spec["states"]:
            subset = participant[participant["AnnotationLevel3"].eq(state)]
            paired = subset["paired_recovery_pct"].dropna()
            expanded = subset["expanded_among_paired_pct"].dropna()
            rows.append({
                "panel": panel,
                "cell_state": state,
                "participants_with_state": int(subset["PatientID"].nunique()),
                "participants_with_paired_receptor": int((subset["paired_cells"] > 0).sum()),
                "paired_recovery_median_pct": float(paired.median()) if len(paired) else np.nan,
                "paired_recovery_q1_pct": float(paired.quantile(0.25)) if len(paired) else np.nan,
                "paired_recovery_q3_pct": float(paired.quantile(0.75)) if len(paired) else np.nan,
                "expanded_among_paired_median_pct": float(expanded.median()) if len(expanded) else np.nan,
                "expanded_among_paired_q1_pct": float(expanded.quantile(0.25)) if len(expanded) else np.nan,
                "expanded_among_paired_q3_pct": float(expanded.quantile(0.75)) if len(expanded) else np.nan,
            })
    return pd.DataFrame(rows)


def compact_canonical_marker_dotplot(container_ax, marker_data, clone_summary):
    container_ax.set_axis_off()
    norm = mpl.colors.Normalize(vmin=-2.5, vmax=2.5)
    scatter = None
    for panel, spec in FIGURE1H_FACETS.items():
        states, genes = spec["states"], spec["genes"]
        data = marker_data[marker_data["cell_state"].isin(states) & marker_data["gene"].isin(genes)].copy()
        data["x"] = data["gene"].map({gene: index for index, gene in enumerate(genes)})
        data["y"] = data["cell_state"].map({state: index for index, state in enumerate(states)})
        marker_ax = container_ax.inset_axes(spec["marker_position"])
        sizes = 3.2 + 28 * np.sqrt(np.clip(data["pct_expressing"].to_numpy(float), 0, 100) / 100)
        scatter = marker_ax.scatter(
            data["x"], data["y"], s=sizes, c=data["avg_expression_scaled"], cmap="RdBu_r", norm=norm,
            edgecolor="#555555", lw=0.20, zorder=3,
        )
        marker_ax.set_xticks(range(len(genes)), genes, rotation=58, ha="right", rotation_mode="anchor")
        for tick in marker_ax.get_xticklabels():
            tick.set_fontstyle("italic")
        marker_ax.set_yticks(range(len(states)), spec["labels"])
        marker_ax.set_xlim(-0.52, len(genes) - 0.48)
        marker_ax.set_ylim(len(states) - 0.50, -0.50)
        marker_ax.grid(color="#E5E5E5", lw=0.32)
        marker_ax.set_axisbelow(True)
        marker_ax.tick_params(axis="x", labelsize=4.5, length=0, pad=1)
        marker_ax.tick_params(axis="y", labelsize=4.5, length=0, pad=1.5)
        for tick in marker_ax.get_yticklabels():
            tick.set_color(spec["color"])
            tick.set_fontweight("bold")
        for spine in marker_ax.spines.values():
            spine.set_visible(False)
        marker_ax.set_title(spec["title"], color=spec["color"], fontsize=5.7, fontweight="bold", pad=8)

        clone_ax = container_ax.inset_axes(spec["clone_position"])
        clone_data = clone_summary[clone_summary["panel"].eq(panel)].set_index("cell_state").reindex(states)
        for yi, state in enumerate(states):
            for xi, metric in enumerate(["paired_recovery_median_pct", "expanded_among_paired_median_pct"]):
                value = clone_data.loc[state, metric]
                if not np.isfinite(value):
                    clone_ax.scatter(xi, yi, marker="x", s=14, color="#9A9A9A", lw=0.75, zorder=3)
                    continue
                size = 5 + 47 * np.sqrt(np.clip(value, 0, 100) / 100)
                clone_ax.scatter(xi, yi, s=size, color=spec["color"], edgecolor="white", lw=0.35, alpha=0.90, zorder=3)
        clone_ax.set_xlim(-0.25, 1.25)
        clone_ax.set_ylim(len(states) - 0.50, -0.50)
        clone_ax.set_xticks([0, 1], ["Paired", "Expanded"])
        clone_ax.xaxis.tick_top()
        clone_ax.set_yticks([])
        clone_ax.grid(color="#E5E5E5", lw=0.32)
        clone_ax.set_axisbelow(True)
        clone_ax.tick_params(axis="x", labelsize=4.0, length=0, pad=1.5)
        for spine in clone_ax.spines.values():
            spine.set_visible(False)

    cax = container_ax.inset_axes([0.015, 0.050, 0.175, 0.030])
    colorbar = container_ax.figure.colorbar(scatter, cax=cax, orientation="horizontal")
    colorbar.set_ticks([-2.5, 0, 2.5])
    colorbar.ax.tick_params(labelsize=4.1, length=1.5, pad=1)
    colorbar.outline.set_linewidth(0.35)
    container_ax.text(0.102, 0.105, "Scaled mean expression", transform=container_ax.transAxes,
                      ha="center", va="bottom", fontsize=4.35, color=COL["muted"])

    expression_handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555",
               markersize=np.sqrt(3.2 + 28 * math.sqrt(value / 100)), label=f"{value}%")
        for value in (25, 75)
    ]
    expression_legend = container_ax.legend(
        handles=expression_handles, title="Cells expressing", frameon=False, loc="lower left",
        bbox_to_anchor=(0.225, 0.010), ncol=2, fontsize=4.0, title_fontsize=4.2,
        handletextpad=0.18, columnspacing=0.40, borderaxespad=0,
    )
    container_ax.add_artist(expression_legend)
    clone_handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor="#727985", markeredgecolor="white",
               markersize=np.sqrt(5 + 47 * math.sqrt(value / 100)), label=f"{value}%")
        for value in (10, 50, 90)
    ]
    container_ax.legend(
        handles=clone_handles, title="Participant median", frameon=False, loc="lower left",
        bbox_to_anchor=(0.480, 0.010), ncol=3, fontsize=4.0, title_fontsize=4.2,
        handletextpad=0.16, columnspacing=0.38, borderaxespad=0,
    )
    container_ax.text(
        1.0, 0.040, "Expanded fraction among paired-receptor cells",
        transform=container_ax.transAxes, ha="right", va="bottom", fontsize=4.1, color=COL["muted"],
    )


def participant_state_effect_panel(ax, effects):
    state_order = [
        "CD4 Naive", "TReg Cytotoxic", "CD8 Naive", "CD8 Tem GZMB+",
        "Switched memory B", "IgM Plasma B Cell",
    ]
    state_labels = {
        "CD4 Naive": "CD4 naive", "TReg Cytotoxic": "Cytotoxic Treg",
        "CD8 Naive": "CD8 naive", "CD8 Tem GZMB+": "CD8 GZMB+",
        "Switched memory B": "Memory B", "IgM Plasma B Cell": "IgM plasma",
    }
    lineage_colors = {"CD4/Treg": "#7656A8", "CD8/innate-like T": "#168C78", "B/plasma": "#C23B78"}
    comparison_styles = {
        "UC vs control": ("o", -0.22), "CD vs control": ("s", 0.0), "CD vs UC": ("^", 0.22),
    }
    y_positions = {state: index for index, state in enumerate(state_order)}
    for _, row in effects.iterrows():
        if row["cell_state"] not in y_positions or row["comparison"] not in comparison_styles:
            continue
        marker, offset = comparison_styles[row["comparison"]]
        y = y_positions[row["cell_state"]] + offset
        color = lineage_colors[row["lineage"]]
        ax.errorbar(
            row["effect"], y,
            xerr=[[row["effect"] - row["ci_low"]], [row["ci_high"] - row["effect"]]],
            fmt=marker, markersize=3.2, color=color, markeredgecolor="white", markeredgewidth=0.35,
            elinewidth=0.72, capsize=1.6, capthick=0.62, zorder=3,
        )
        if float(row.get("adjusted_p_value", 1.0)) < 0.05:
            ax.text(row["ci_high"] + 0.10, y, "*", fontsize=6.0, color=color, va="center", ha="left")
    ax.axvline(0, color="#666666", lw=0.65)
    ax.set_yticks(range(len(state_order)), [state_labels[state] for state in state_order])
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted log-odds difference", fontsize=5.7)
    ax.tick_params(axis="x", labelsize=5.1, pad=1)
    ax.tick_params(axis="y", labelsize=5.0, length=0, pad=2)
    ax.grid(axis="x", color="#E1E1E1", lw=0.38)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    for tick, color in zip(ax.get_yticklabels(), ["#7656A8"] * 2 + ["#168C78"] * 2 + ["#C23B78"] * 2):
        tick.set_color(color)
    finite = effects[["ci_low", "ci_high"]].to_numpy(float)
    low, high = np.nanmin(finite), np.nanmax(finite)
    span = max(high - low, 1.0)
    ax.set_xlim(low - 0.05 * span, high + 0.15 * span)
    handles = [
        Line2D([0], [0], marker=marker, ls="", ms=3.2, color="#555555", markerfacecolor="#555555", label=label)
        for label, (marker, _) in comparison_styles.items()
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.50, 1.19),
              ncol=3, fontsize=4.25, handletextpad=0.15, columnspacing=0.35)
    ax.text(0.0, -0.18, "Participant-level; series-adjusted; HC3 95% CI; *FDR < 0.05",
            transform=ax.transAxes, fontsize=4.6, color=COL["muted"], ha="left", va="top")


MILO_STATE_ORDER = {
    "CD4/Treg": [
        "CD4 Naive", "CD4 Naive-IFN", "CD4 Tfh", "CD4 Th1", "CD4 Th1/Th17", "CD4 Th17",
        "CD4 Th2", "CD4 Th22", "CD4 HLA-DR+ memory", "CD4 Temra", "CD4 Terminal effector",
        "TReg Naive", "TReg Memory", "TReg KLRB1+RORC+", "TReg Cytotoxic",
    ],
    "CD8/innate-like T": [
        "CD8 Naive", "CD8 Naive-IFN", "CD8 Tcm CCR4-", "CD8 Tem GZMK+", "CD8 Tem GZMB+",
        "CD8 Temra", "CD8 Trm", "CD8 HLA-DR+", "CD8 Proliferative", "CD8 Tmem KLRC2+", "MAIT", "gdT",
    ],
    "B/plasma": [
        "Transitional B", "CD5+ B Cell", "Naive B", "Naive-IFN B", "Non-switched memory B",
        "Switched memory B", "Atypical memory B", "IgM Plasma B Cell", "IgA Plasma B Cell", "IgG Plasma B Cell",
    ],
}


def compact_count(value):
    value = int(value)
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def lineage_milor_heatmap(ax, summary, lineage, norm, cmap):
    comparisons = ["UC vs control", "CD vs control", "CD vs UC"]
    data = summary[summary["lineage"].eq(lineage)].copy()
    eligible = data.groupby("cell_state")["display_eligible"].max()
    states = [state for state in MILO_STATE_ORDER[lineage] if bool(eligible.get(state, False))]
    effect = data.pivot_table(index="cell_state", columns="comparison", values="display_value", aggfunc="first")
    matrix = effect.reindex(index=states, columns=comparisons).to_numpy(float)
    masked = np.ma.masked_invalid(matrix)
    ax.set_facecolor("#EEEEEE")
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    ax.set_xticks(range(3), ["UC vs\ncontrol", "CD vs\ncontrol", "CD vs\nUC"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=5.3, length=0, pad=2)
    ax.set_yticks(range(len(states)), states)
    ax.tick_params(axis="y", labelsize=5.15, length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(states), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = {"CD4/Treg": "CD4 T Cells", "CD8/innate-like T": "CD8 T Cells", "B/plasma": "B Cells"}[lineage]
    title_color = {"CD4/Treg": "#7656A8", "CD8/innate-like T": "#168C78", "B/plasma": "#C23B78"}[lineage]
    ax.set_title(title, fontweight="bold", fontsize=7.0, pad=18, color=title_color)
    return im


def unified_milor_panel(container_ax, summary):
    container_ax.set_axis_off()
    container_ax.text(0.0, 1.035, "MiloR differential abundance", transform=container_ax.transAxes,
                      ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=COL["ink"])
    container_ax.text(0.995, 1.035, "SpatialFDR < 0.05", transform=container_ax.transAxes,
                      ha="right", va="bottom", fontsize=5.4, color=COL["muted"])
    positions = {
        "CD4/Treg": [0.00, 0.13, 0.28, 0.69],
        "CD8/innate-like T": [0.42, 0.13, 0.24, 0.69],
        "B/plasma": [0.78, 0.13, 0.17, 0.69],
    }
    norm = mpl.colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#EEEEEE")
    image = None
    for lineage, position in positions.items():
        heat_ax = container_ax.inset_axes(position)
        image = lineage_milor_heatmap(heat_ax, summary, lineage, norm, cmap)
    cax = container_ax.inset_axes([0.978, 0.20, 0.018, 0.50])
    colorbar = container_ax.figure.colorbar(image, cax=cax)
    colorbar.set_label("")
    cax.set_title("median\nlog2 FC", fontsize=5.0, pad=2, color=COL["ink"])
    colorbar.ax.tick_params(labelsize=5.2, width=0.45, length=2)
    colorbar.outline.set_linewidth(0.45)
    container_ax.text(
        0.0, 0.015,
        "Color: median log2 FC among significant neighborhoods; positive values favor the first-named group. "
        "Gray: fewer than 5 significant neighborhoods.",
        transform=container_ax.transAxes, ha="left", va="bottom", fontsize=5.2, color=COL["muted"]
    )


def receptor_coverage_dotplot(ax, coverage):
    selected = [
        "CD4 Naive", "CD4 Tfh", "CD4 Th17", "TReg Cytotoxic",
        "CD8 Naive", "CD8 Tcm CCR4-", "CD8 Temra", "CD8 HLA-DR+", "MAIT", "gdT",
        "Naive B", "Switched memory B", "Atypical memory B", "IgA Plasma B Cell", "IgG Plasma B Cell",
    ]
    data = coverage.set_index("cell_state").reindex(selected).dropna(subset=["total_cells"]).reset_index()
    columns = [
        ("Paired alpha-beta", "paired_alpha_beta_n", "paired_alpha_beta_pct"),
        ("Paired gamma-delta", "paired_gamma_delta_n", "paired_gamma_delta_pct"),
        ("Paired heavy-light", "paired_heavy_light_n", "paired_heavy_light_pct"),
    ]
    max_n = max(float(data[c].max()) for _, c, _ in columns)
    norm = mpl.colors.Normalize(vmin=0, vmax=50)
    cmap = plt.get_cmap("YlGnBu")
    for yi, row in data.iterrows():
        for xi, (_, n_col, pct_col) in enumerate(columns):
            is_b = row["lineage"] == "B cell"
            appropriate = (xi == 2 and is_b) or (xi < 2 and not is_b)
            if not appropriate:
                continue
            n = float(row[n_col]); pct = float(row[pct_col])
            size = 10 + 125 * math.sqrt(max(n, 0) / max_n) if max_n else 10
            ax.scatter(xi, yi, s=size, color=cmap(norm(pct)), edgecolor=COL["ink"], lw=0.35)
    ax.set_xticks(range(3), [x[0] for x in columns])
    ax.set_yticks(range(len(data)), data["cell_state"])
    ax.invert_yaxis()
    ax.set_xlim(-0.55, 2.55)
    ax.grid(axis="x", color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.025)
    cb.set_label("Cells with paired receptor (%)", fontsize=7)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    size_handles = []
    for n in (100, 1000, 10000):
        size = 10 + 125 * math.sqrt(n / max_n) if max_n else 10
        size_handles.append(plt.scatter([], [], s=size, facecolor="white", edgecolor=COL["ink"], lw=0.5, label=f"{n:,}"))
    ax.legend(handles=size_handles, title="Paired cells", frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02),
              fontsize=6, title_fontsize=6)


def build_figure_1():
    from build_figure1_efg_level3_preview import (
        CLONE_STYLES as LEVEL3_CLONE_STYLES,
        draw_state_legend as draw_level3_state_legend,
        draw_umap as draw_level3_umap,
    )

    participants = read_csv(SRC / "Figure1_participant_summary.csv")
    qc = read_csv(SRC / "Figure1_receptor_qc_summary.csv")
    recovery = read_csv(SRC / "Figure1_participant_receptor_recovery.csv")
    coverage = read_csv(SRC / "Figure1_receptor_coverage_by_state.csv")
    major_summary = read_csv(SRC / "Figure1_major_lineage_summary.csv")
    panel_summary = read_csv(SRC / "Figure1_lineage_panel_summary.csv").set_index("panel")
    milo_summary = read_csv(SRC / "Figure1_lineage_miloR_heatmap_summary.csv")
    clone_overlay = read_csv(SRC / "Figure1_clone_engagement_umap.csv")
    clone_summary = read_csv(SRC / "Figure1_expanded_clone_summary.csv").set_index("panel")
    marker_data = read_csv(SRC / "Figure1_canonical_marker_dotplot.csv")
    pbmc = read_csv(SRC / "UMAP_PBMC_from_Seurat.csv")
    level3_source = OUT / "Preview Alternatives" / "Source Data"
    cd4 = read_csv(level3_source / "Figure_1E_CD4_Level3_UMAP.csv")
    cd8 = read_csv(level3_source / "Figure_1F_CD8_Level3_UMAP.csv")
    bcell = read_csv(level3_source / "Figure_1G_B_Level3_UMAP.csv")

    overlay_columns = ["cell", "expanded_paired", "paired_family", "paired_clone_size"]
    cd4 = cd4.merge(clone_overlay[clone_overlay["panel"].eq("CD4")][overlay_columns], on="cell", how="left")
    cd8 = cd8.merge(clone_overlay[clone_overlay["panel"].eq("CD8")][overlay_columns], on="cell", how="left")
    bcell = bcell.merge(clone_overlay[clone_overlay["panel"].eq("B")][overlay_columns], on="cell", how="left")
    marker_clone_summary = summarize_figure1h_clone_engagement({"CD4": cd4, "CD8": cd8, "B": bcell})
    marker_clone_summary.to_csv(SRC / "Figure1_clone_engagement_by_state.csv", index=False)

    fig = plt.figure(figsize=(7.48, 11.70))
    # Use row-specific grids so the gutters are optically consistent across
    # two- and three-panel rows. A single six-column grid made the C-D gutter
    # feel pinched while leaving excessive dead space elsewhere.
    gs = GridSpec(
        5, 1, figure=fig, height_ratios=[0.78, 1.50, 1.22, 1.55, 1.42], hspace=0.44,
        left=0.14, right=0.985, top=0.978, bottom=0.040,
    )
    top_row = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, 0], width_ratios=[1.85, 4.15], wspace=0.18,
    )
    atlas_row = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1, 0], width_ratios=[4.05, 1.95], wspace=0.18,
    )
    lineage_row = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[2, 0], wspace=0.32,
    )

    ax = fig.add_subplot(top_row[0, 0])
    cohort_panel(ax, participants)
    panel_title(ax, "Cohort and clinical context")
    # Figure 5 uses long vertical axis labels. Keep panel letters in the
    # narrow gutter immediately before each title rather than over the
    # y-label zone used by the global default.
    panel_label(ax, "A", x=-0.065, y=1.10)

    ax = fig.add_subplot(top_row[0, 1])
    figure1_workflow(ax)
    panel_title(ax, "Paired multi-omic workflow")
    panel_label(ax, "B", x=-0.08)

    atlas_ax = fig.add_subplot(atlas_row[0, 0])
    atlas_position = atlas_ax.get_position()
    atlas_ax.set_position([
        atlas_position.x0 - 0.005,
        atlas_position.y0 - 0.008,
        atlas_position.width + 0.010,
        atlas_position.height + 0.016,
    ])
    full_pbmc_level2_umap(atlas_ax, pbmc)
    panel_title(atlas_ax, "PBMC atlas")
    atlas_ax.text(1.0, 1.015, "1,130,665 cells | 249 participants | 53 level 2 states",
                  transform=atlas_ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=COL["muted"])
    panel_label(atlas_ax, "C", x=-0.08, y=1.08)

    ax = fig.add_subplot(atlas_row[0, 1])
    participant_receptor_recovery_panel(ax, recovery)
    panel_title(ax, "Participant receptor recovery")
    panel_label(ax, "D", x=-0.16)

    lineage_panels = [
        ("E", "CD4", "CD4 T Cells", cd4, lineage_row[0, 0]),
        ("F", "CD8", "CD8 T Cells", cd8, lineage_row[0, 1]),
        ("G", "B", "B Cells", bcell, lineage_row[0, 2]),
    ]
    lineage_headings = []
    for letter, panel, title, data, subplot_spec in lineage_panels:
        panel_widths = [1.82, 0.68] if panel in {"CD4", "CD8"} else [1.65, 0.85]
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplot_spec, width_ratios=panel_widths, wspace=0.07
        )
        umap_ax = fig.add_subplot(inner[0, 0])
        legend_ax = fig.add_subplot(inner[0, 1])
        draw_level3_umap(umap_ax, data, panel)
        draw_level3_state_legend(legend_ax, panel)
        lineage_headings.append((letter, title, umap_ax))

    # Equal-aspect UMAP axes can have different physical heights. Draw all
    # three headings in figure coordinates so E-G share one exact baseline.
    fig.canvas.draw()
    lineage_top = lineage_row[0, 0].get_position(fig).y1 + 0.004
    for letter, title, umap_ax in lineage_headings:
        title_x = umap_ax.get_position().x0
        fig.text(
            title_x - 0.025, lineage_top, letter, ha="left", va="top",
            fontsize=12, fontweight="bold", color=COL["ink"],
        )
        fig.text(
            title_x, lineage_top, title, ha="left", va="top",
            fontsize=7.2, fontweight="bold", color=COL["ink"],
        )

    clone_handles = [
        Line2D(
            [0], [0], marker="o", ls="", ms=4.2 + 0.55 * index,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.40,
            label=label,
        )
        for index, (label, _, _, color, _) in enumerate(LEVEL3_CLONE_STYLES)
    ]
    lineage_left = lineage_row[0, 0].get_position(fig).x0
    lineage_right = lineage_row[0, 2].get_position(fig).x1
    lineage_bottom = lineage_row[0, 0].get_position(fig).y0
    clone_legend = fig.legend(
        handles=clone_handles,
        title="Expanded paired clonotype size (cells)",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=((lineage_left + lineage_right) / 2, lineage_bottom - 0.004),
        ncol=4,
        fontsize=5.6,
        title_fontsize=5.8,
        handletextpad=0.28,
        columnspacing=0.88,
        labelspacing=0.20,
        borderaxespad=0,
    )
    clone_legend._legend_box.align = "center"

    marker_ax = fig.add_subplot(gs[3, 0])
    compact_canonical_marker_dotplot(marker_ax, marker_data, marker_clone_summary)
    marker_ax.set_title("Canonical identity and clone engagement", loc="left", fontweight="bold", pad=18)
    panel_label(marker_ax, "H", x=-0.08, y=1.10)

    milo_ax = fig.add_subplot(gs[4, 0])
    unified_milor_panel(milo_ax, milo_summary)
    panel_label(milo_ax, "I", x=-0.065, y=1.055)
    save_figure(fig, "Figure_1", MAIN, fixed_canvas=True)


def build_figure_s9():
    coverage = read_csv(SRC / "Figure1_receptor_coverage_by_state.csv")
    fig = plt.figure(figsize=(7.48, 5.7))
    gs = GridSpec(1, 1, figure=fig, left=0.20, right=0.94, top=0.93, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    receptor_coverage_dotplot(ax, coverage)
    panel_title(ax, "Paired-receptor recovery across representative adaptive immune-cell states")
    panel_label(ax, "A", x=-0.18, y=1.06)
    save_figure(fig, "Figure_S9", SUPP, fixed_canvas=True)


def build_figure_s10():
    pbmc = read_csv(SRC / "UMAP_PBMC_from_Seurat.csv")
    fig = plt.figure(figsize=(7.48, 5.8))
    gs = GridSpec(1, 1, figure=fig, left=0.075, right=0.985, top=0.92, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    full_pbmc_level2_umap(ax, pbmc, mode="supplement")
    panel_title(ax, "Full PBMC UMAP with complete level 2 annotation key")
    panel_label(ax, "A", x=-0.06, y=1.05)
    save_figure(fig, "Figure_S10", SUPP, fixed_canvas=True)


def build_figure_s11():
    markers = read_csv(SRC / "Figure1_canonical_marker_dotplot.csv")
    fig = plt.figure(figsize=(7.48, 5.0))
    gs = GridSpec(1, 1, figure=fig, left=0.14, right=0.97, top=0.88, bottom=0.17)
    ax = fig.add_subplot(gs[0, 0])
    canonical_marker_dotplot(ax, markers)
    ax.set_title("Canonical marker validation of adaptive immune-cell states", loc="left", fontweight="bold", pad=23)
    panel_label(ax, "A", x=-0.10, y=1.17)
    save_figure(fig, "Figure_S11", SUPP, fixed_canvas=True)


def build_figure_2_visual_previous():
    metrics = read_csv(TCR_DIR / "clonality_metrics_IBDTCR_Immunarch.csv").rename(columns={"Diagnosis": "Diagnosis"})
    exact = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_exact_clonotype_definition_metrics_by_participant.csv")
    exact_tests = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_exact_clonotype_definition_pairwise_tests.csv")
    state = read_csv(SRC / "Figure2_state_enrichment_summary.csv")
    modules = read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA1_clone_aware_pseudobulk_module_enrichment.csv")
    deltas = read_csv(ROOT / "High Impact Additional Analyses/Clone State Interactions/Table_CSI2_participant_expanded_minus_singleton_deltas.csv")
    interactions = read_csv(ROOT / "High Impact Additional Analyses/Clone State Interactions/Table_CSI3_formal_interaction_models.csv")
    breadth = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_clonotype_state_breadth_pairwise_tests.csv")

    fig = plt.figure(figsize=(7.48, 9.0))
    outer = GridSpec(3, 2, figure=fig, height_ratios=[0.82, 1.55, 1.08], hspace=0.66, wspace=0.82,
                     left=0.10, right=0.98, top=0.97, bottom=0.065)

    # A. Retain two complementary metrics; Gini is omitted here because it is
    # visually and conceptually redundant with clonality.
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.42)
    for j, (metric, label) in enumerate([("Clonality", "Clonality"), ("NormShannon", "Shannon evenness")]):
        ax = fig.add_subplot(sub[0, j])
        boxstrip(ax, metrics, metric, ylabel="")
        ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold")
        for collection in ax.collections:
            collection.set_alpha(0.46)
        ax.tick_params(axis="x", rotation=35)
        if j == 0:
            panel_label(ax, "A", x=-0.36)
            panel_title(ax, "Repertoire structure")

    # B. Replace boundary-prone clone-size bins with exact expanded-cell burden
    # and definition/depth sensitivity estimates.
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], width_ratios=[0.92, 1.22], wspace=0.62)
    ax = fig.add_subplot(sub[0, 0])
    primary = exact[(exact["receptor_definition"] == "TCR beta-only") &
                    (exact["analysis"] == "Original depth") & (exact["threshold"] == 2)].copy()
    primary = primary.rename(columns={"Diagnosis1": "Diagnosis"})
    primary["expanded_cell_percent"] = 100 * primary["expanded_cell_fraction"]
    boxstrip(ax, primary, "expanded_cell_percent", ylabel="Expanded T cells (%)")
    for collection in ax.collections:
        collection.set_alpha(0.46)
    ax.tick_params(axis="x", rotation=35)
    panel_label(ax, "B", x=-0.52)
    panel_title(ax, "Expansion burden")

    ax = fig.add_subplot(sub[0, 1])
    specs = [
        ("TCR beta-only", "Original depth", 2, "Beta2"),
        ("TCR beta-only", "Original depth", 5, "Beta5"),
        ("TCR paired alpha-beta", "Original depth", 2, "Paired2"),
        ("TCR beta-only", "Depth-normalized", 2, "Depth2"),
    ]
    rows = []
    for receptor, analysis, threshold, label in specs:
        ss = exact_tests[(exact_tests["receptor_definition"] == receptor) &
                         (exact_tests["analysis"] == analysis) &
                         (exact_tests["threshold"] == threshold) &
                         (exact_tests["metric"] == "expanded_cell_fraction") &
                         (exact_tests["contrast"].isin(["CD vs Control", "UC vs Control"]))].copy()
        ss["spec"] = label
        rows.append(ss)
    robust = pd.concat(rows, ignore_index=True)
    spec_order = [x[3] for x in specs]
    ybase = np.arange(len(spec_order))[::-1]
    offsets = {"CD vs Control": 0.12, "UC vs Control": -0.12}
    for contrast in ["CD vs Control", "UC vs Control"]:
        ss = robust[robust["contrast"] == contrast].set_index("spec").reindex(spec_order)
        y = ybase + offsets[contrast]
        color = COL["CD"] if contrast.startswith("CD") else COL["UC"]
        ax.hlines(y, ss["ci_low"], ss["ci_high"], color=color, lw=0.9)
        ax.scatter(ss["rank_biserial"], y, color=color, s=18, edgecolor="white", lw=0.35, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.6, ls="--")
    ax.set_yticks(ybase, ["Beta clone >=2", "Beta clone >=5", "Paired alpha-beta >=2", "Depth-matched beta >=2"])
    ax.set_xlabel("Rank-biserial effect vs control")
    style_axis(ax, "x")
    ax.tick_params(axis="y", labelsize=5.4)
    panel_title(ax, "Definition/depth sensitivity")
    robust.to_csv(SRC / "Figure2_expansion_robustness_displayed.csv", index=False)

    # C. Direct expanded-versus-singleton state enrichment rather than raw occupancy.
    ax = fig.add_subplot(outer[1, 0])
    sc = state.sort_values("median_log2_or", ascending=False)
    state_labels = [f"{s} (n={int(n)})" for s, n in zip(sc["state"], sc["n_participants"])]
    state_colors = [COL["positive"] if q < 0.05 and e > 0 else
                    COL["negative"] if q < 0.05 and e < 0 else "#A8A8A8"
                    for e, q in zip(sc["median_log2_or"], sc["FDR"])]
    forest(ax, state_labels, sc["median_log2_or"], sc["ci_low"], sc["ci_high"],
           xlabel="Median within-participant log2 odds ratio", colors=state_colors)
    for yi in np.arange(len(sc))[::-1][::2]:
        ax.axhspan(yi - 0.5, yi + 0.5, color="#F5F5F5", zorder=-2)
    ax.tick_params(axis="y", labelsize=5.8)
    panel_title(ax, "State enrichment: expanded vs singleton")
    panel_label(ax, "C")

    # D. Show participant-level effect magnitude with bootstrap intervals.
    ax = fig.add_subplot(outer[1, 1])
    dd = deltas[deltas["modality"].str.upper() == "TCR"].copy()
    rng = np.random.default_rng(20260824)
    module_rows = []
    module_fdr = modules[modules["modality"].str.upper() == "TCR"].set_index("module")["FDR_global"]
    for module, group in dd.groupby("module"):
        values = pd.to_numeric(group["delta"], errors="coerce").dropna().to_numpy()
        boot = np.median(rng.choice(values, size=(4000, len(values)), replace=True), axis=1)
        module_rows.append({"module": module, "n": len(values), "effect": np.median(values),
                            "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
                            "FDR_global": module_fdr.get(module, np.nan)})
    md = pd.DataFrame(module_rows).sort_values("effect", ascending=False)
    md["label"] = [concise_module_label(m) for m in md["module"]]
    md["color"] = [COL["positive"] if q < 0.05 and e > 0 else
                   COL["negative"] if q < 0.05 and e < 0 else "#A8A8A8"
                   for e, q in zip(md["effect"], md["FDR_global"])]
    forest(ax, md["label"], md["effect"], md["ci_low"], md["ci_high"],
           xlabel="Median expanded - singleton module score", colors=md["color"])
    for yi in np.arange(len(md))[::-1][::2]:
        ax.axhspan(yi - 0.5, yi + 0.5, color="#F5F5F5", zorder=-2)
    ax.tick_params(axis="y", labelsize=5.4)
    panel_title(ax, "Clone-aware transcriptional programs (n=175)")
    panel_label(ax, "D")
    md.to_csv(SRC / "Figure2_module_effects_displayed.csv", index=False)

    # E. Adjusted diagnosis-specific estimates and the formal interaction result.
    ax = fig.add_subplot(outer[2, 0])
    selected = ["EOMES_ZEB2_inflammatory_CD8_TRM_like", "Effector_cytotoxicity",
                "Th1_Tc1_inflammatory", "Naive_central_memory"]
    ie = interactions[(interactions["family"] == "TCR_diagnosis") &
                      (interactions["module"].isin(selected)) &
                      (interactions["contrast"].isin(["Control_expansion", "CD_expansion", "UC_expansion"]))].copy()
    ie["Diagnosis"] = ie["contrast"].str.replace("_expansion", "", regex=False)
    ybase = np.arange(len(selected))[::-1]
    offsets = {"Control": 0.18, "CD": 0.0, "UC": -0.18}
    for diagnosis in DIAG:
        ss = ie[ie["Diagnosis"] == diagnosis].set_index("module").reindex(selected)
        y = ybase + offsets[diagnosis]
        lo = ss["effect"] - 1.96 * ss["SE"]
        hi = ss["effect"] + 1.96 * ss["SE"]
        line_width = 0.65 if diagnosis == "Control" else 0.9
        alpha = 0.62 if diagnosis == "Control" else 1.0
        ax.hlines(y, lo, hi, color=COL[diagnosis], lw=line_width, alpha=alpha)
        ax.scatter(ss["effect"], y, color=COL[diagnosis], s=15 if diagnosis == "Control" else 18,
                   alpha=alpha, edgecolor="white", lw=0.35,
                   label=diagnosis, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.6, ls="--")
    ax.set_yticks(ybase, [concise_module_label(x) for x in selected])
    ax.set_xlabel("Adjusted expanded - singleton score")
    ax.legend(frameon=False, ncol=3, loc="lower right", handletextpad=0.25, columnspacing=0.7)
    style_axis(ax, "x")
    interaction_min = interactions[(interactions["family"] == "TCR_diagnosis") &
                                   interactions["contrast"].str.contains("interaction")]["FDR"].min()
    panel_title(ax, "Programs conserved across diagnoses (no interaction)")
    panel_label(ax, "E")

    # F. Display the >=2-cell primary analysis with the sparse >=5-cell sensitivity.
    ax = fig.add_subplot(outer[2, 1])
    bb = breadth[(breadth["modality"] == "TCR") &
                 (breadth["threshold"].isin([2, 5])) &
                 (breadth["metric"] == "median_entropy_excess")].copy()
    contrast_order = ["CD vs Control", "UC vs Control", "CD vs UC"]
    bb["contrast_order"] = bb["contrast"].map({x: i for i, x in enumerate(contrast_order)})
    bb = bb.sort_values(["threshold", "contrast_order"])
    positions = {2: np.array([7, 6, 5]), 5: np.array([2, 1, 0])}
    tick_pos, tick_labels = [], []
    for threshold in [2, 5]:
        ss = bb[bb["threshold"] == threshold].set_index("contrast").reindex(contrast_order)
        y = positions[threshold]
        color = COL["ink"] if threshold == 2 else "#999999"
        ax.hlines(y, ss["ci_low"], ss["ci_high"], color=color, lw=0.9)
        ax.scatter(ss["rank_biserial"], y, color=color, s=19, edgecolor="white", lw=0.35, zorder=3)
        tick_pos.extend(y.tolist())
        tick_labels.extend([f"{c} (n={int(n1)}/{int(n2)})" for c, n1, n2 in
                            zip(contrast_order, ss["n_group1"], ss["n_group2"])])
    ax.axvline(0, color=COL["muted"], lw=0.6, ls="--")
    ax.axhline(3.5, color=COL["grid"], lw=0.7)
    ax.set_yticks(tick_pos, tick_labels)
    ax.set_ylim(-0.55, 8.25)
    ax.text(0.01, 0.98, "Primary: clones >=2 cells", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.8, fontweight="bold", color=COL["ink"])
    ax.text(0.01, 0.42, "Sensitivity: clones >=5 cells", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.8, fontweight="bold", color="#777777")
    ax.set_xlabel("Rank-biserial effect: entropy excess")
    style_axis(ax, "x")
    ax.tick_params(axis="y", labelsize=5.4)
    qmin = pd.to_numeric(bb["p_adj"], errors="coerce").min()
    panel_title(ax, "State-breadth sensitivity (all adjusted p > 0.3)")
    panel_label(ax, "F")
    bb.to_csv(SRC / "Figure2_state_breadth_displayed.csv", index=False)
    save_figure(fig, "Figure_2", MAIN)


def build_figure_2():
    """High-priority, mechanism-first Figure 2 redesign."""
    metrics = read_csv(TCR_DIR / "clonality_metrics_IBDTCR_Immunarch.csv")
    exact = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_exact_clonotype_definition_metrics_by_participant.csv")
    dose = read_csv(SRC / "Figure2_clone_size_module_dose_response.csv")
    trends = read_csv(SRC / "Figure2_clone_size_trend_tests.csv")
    state = read_csv(SRC / "Figure2_state_enrichment_depth_batch_adjusted.csv")
    paired_deltas = read_csv(SRC / "Figure2_participant_paired_program_deltas.csv")
    paired_tests = read_csv(SRC / "Figure2_paired_program_delta_tests.csv")
    replication = read_csv(SRC / "Figure2_series_replication_meta_analysis.csv")
    interaction_summary = read_csv(SRC / "Figure2_diagnosis_interaction_summary.csv")
    interactions = read_csv(ROOT / "High Impact Additional Analyses/Clone State Interactions/Table_CSI3_formal_interaction_models.csv")
    modules = read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA1_clone_aware_pseudobulk_module_enrichment.csv")

    selected_modules = [
        "EOMES_ZEB2_inflammatory_CD8_TRM_like",
        "Effector_cytotoxicity",
        "Th1_Tc1_inflammatory",
    ]
    module_titles = {
        "EOMES_ZEB2_inflammatory_CD8_TRM_like": "EOMES-ZEB2",
        "Effector_cytotoxicity": "Cytotoxicity",
        "Th1_Tc1_inflammatory": "Th1/Tc1",
    }

    def q_label(value):
        if value < 1e-4:
            return "q<0.0001"
        if value < 0.01:
            return f"q={value:.3f}"
        return f"q={value:.2f}"

    fig = plt.figure(figsize=(7.48, 8.92))
    outer = GridSpec(
        3, 2, figure=fig, height_ratios=[0.94, 1.48, 1.14], hspace=0.70, wspace=0.78,
        left=0.10, right=0.98, top=0.965, bottom=0.065,
    )

    # A. Compact repertoire and expansion overview.
    container = fig.add_subplot(outer[0, 0]); container.set_axis_off()
    container.text(-0.16, 1.08, "A", transform=container.transAxes, fontsize=12, fontweight="bold", va="top")
    container.text(0, 1.08, "Repertoire overview", transform=container.transAxes,
                   fontsize=8, fontweight="bold", va="top")
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.42)
    ax = fig.add_subplot(sub[0, 0])
    boxstrip(ax, metrics, "Clonality", ylabel="Clonality")
    for collection in ax.collections: collection.set_alpha(0.46)
    ax.tick_params(axis="x", rotation=35)
    ax.text(0.02, 0.98, "Repertoire clonality", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.2, fontweight="bold")
    ax = fig.add_subplot(sub[0, 1])
    primary = exact[(exact["receptor_definition"] == "TCR beta-only") &
                    (exact["analysis"] == "Original depth") & (exact["threshold"] == 2)].copy()
    primary = primary.rename(columns={"Diagnosis1": "Diagnosis"})
    primary["expanded_cell_percent"] = 100 * primary["expanded_cell_fraction"]
    boxstrip(ax, primary, "expanded_cell_percent", ylabel="Expanded T cells (%)")
    for collection in ax.collections: collection.set_alpha(0.46)
    ax.tick_params(axis="x", rotation=35)
    ax.text(0.02, 0.98, "Exact beta clones >=2 cells", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.2, fontweight="bold")

    # B. Clone-size dose response, emphasizing the pooled trajectory while
    # retaining diagnosis-specific patterns and formal trend inference.
    container = fig.add_subplot(outer[0, 1]); container.set_axis_off()
    container.text(-0.16, 1.08, "B", transform=container.transAxes, fontsize=12, fontweight="bold", va="top")
    container.text(0, 1.08, "Clone-size dose response", transform=container.transAxes,
                   fontsize=7.6, fontweight="bold", va="top")
    pooled_handle = Line2D([0], [0], color=COL["ink"], lw=1.4, marker="D", ms=3.2,
                           markerfacecolor="white", label="Pooled")
    diagnosis_handles = [Line2D([0], [0], color=COL[d], lw=0.9, marker="o", ms=2.8, label=d) for d in DIAG]
    container.legend(handles=[pooled_handle] + diagnosis_handles, frameon=False, loc="upper right",
                     bbox_to_anchor=(1.02, 1.15), ncol=4, fontsize=4.45,
                     handlelength=0.9, handletextpad=0.16, columnspacing=0.30)
    min_interaction_q = trends[trends["contrast"].str.contains("interaction")]["FDR"].min()
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 1], wspace=0.36)
    bins = ["Singleton", "2 cells", "3-4 cells", ">=5 cells"]
    x = np.arange(len(bins))
    rng = np.random.default_rng(20260824)
    dose_rows = []
    shared_bin_n = None
    for j, module in enumerate(selected_modules):
        ax = fig.add_subplot(sub[0, j])
        dm = dose[dose["module"] == module].copy()
        for band in [1, 3]:
            ax.axvspan(band - 0.5, band + 0.5, color="#F5F5F5", zorder=-3)

        # Diagnosis-specific medians provide contextual trajectories without
        # competing visually with the pooled effect and its uncertainty.
        for diagnosis in DIAG:
            med, lo, hi, ns = [], [], [], []
            for clone_bin in bins:
                vals = pd.to_numeric(dm[(dm["Diagnosis1"] == diagnosis) &
                                        (dm["clone_bin"] == clone_bin)]["delta"], errors="coerce").dropna().to_numpy()
                ns.append(len(vals))
                if len(vals):
                    boot = np.median(rng.choice(vals, size=(2500, len(vals)), replace=True), axis=1)
                    med.append(float(np.median(vals)))
                    lo.append(float(np.quantile(boot, 0.025)))
                    hi.append(float(np.quantile(boot, 0.975)))
                else:
                    med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                dose_rows.append({"module": module, "Diagnosis": diagnosis, "clone_bin": clone_bin,
                                  "n": len(vals), "median": med[-1], "ci_low": lo[-1], "ci_high": hi[-1]})
            offset = {"Control": -0.08, "CD": 0.0, "UC": 0.08}[diagnosis]
            ax.plot(x + offset, med, color=COL[diagnosis], lw=0.82, alpha=0.72,
                    marker="o", ms=2.7, markeredgewidth=0, zorder=3)

        # Pooled medians and bootstrap intervals are visually dominant because
        # the diagnosis-by-trend tests do not support heterogeneous slopes.
        pooled_med, pooled_lo, pooled_hi, pooled_n = [], [], [], []
        for clone_bin in bins:
            vals = pd.to_numeric(dm.loc[dm["clone_bin"] == clone_bin, "delta"], errors="coerce").dropna().to_numpy()
            pooled_n.append(len(vals))
            boot = np.median(rng.choice(vals, size=(4000, len(vals)), replace=True), axis=1)
            pooled_med.append(float(np.median(vals)))
            pooled_lo.append(float(np.quantile(boot, 0.025)))
            pooled_hi.append(float(np.quantile(boot, 0.975)))
            dose_rows.append({"module": module, "Diagnosis": "Pooled", "clone_bin": clone_bin,
                              "n": len(vals), "median": pooled_med[-1],
                              "ci_low": pooled_lo[-1], "ci_high": pooled_hi[-1]})
        if shared_bin_n is None:
            shared_bin_n = pooled_n
        ax.errorbar(x, pooled_med,
                    yerr=[np.asarray(pooled_med) - np.asarray(pooled_lo),
                          np.asarray(pooled_hi) - np.asarray(pooled_med)],
                    fmt="D-", color=COL["ink"], markerfacecolor="white", markeredgecolor=COL["ink"],
                    markeredgewidth=0.65, ms=3.8, lw=1.35, ecolor="#555555", elinewidth=0.75,
                    capsize=1.6, zorder=5)
        ax.axhline(0, color=COL["muted"], lw=0.55, ls="--")
        ax.set_xticks(x, ["1", "2", "3-4", ">=5"], rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=5.1, pad=1.5)
        ax.text(0.03, 0.965, module_titles[module], transform=ax.transAxes, ha="left", va="top",
                fontsize=5.7, fontweight="bold")
        trend = trends[(trends["module"] == module) &
                       (trends["contrast"] == "Pooled ordered trend")].iloc[0]
        annotation = (f"Trend={trend['estimate']:.3f}/bin\n"
                      f"95% CI {trend['ci_low']:.3f}-{trend['ci_high']:.3f}\n{q_label(trend['FDR'])}")
        ax.text(0.97, 0.055, annotation, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.55, color=COL["muted"], linespacing=1.05,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#E5E5E5",
                      "lw": 0.35, "alpha": 0.92})
        if j == 0: ax.set_ylabel("State-matched score delta")
        ax.set_ylim(-0.008, 0.292)
        ax.set_xlim(-0.24, 3.24)
        ax.set_yticks([0, 0.10, 0.20])
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)
        style_axis(ax, "y")
    container.text(0.5, -0.18, "Cells per clonotype", transform=container.transAxes,
                   ha="center", va="top", fontsize=7.0)
    container.text(0.5, -0.28,
                   "Participants/bin: " + " | ".join(str(n) for n in shared_bin_n) +
                   f"; diagnosis x trend min q={min_interaction_q:.2f}",
                   transform=container.transAxes, ha="center", va="top", fontsize=4.7, color=COL["muted"])
    pd.DataFrame(dose_rows).to_csv(SRC / "Figure2_clone_size_dose_response_displayed.csv", index=False)

    # C. Expanded-versus-singleton state enrichment, adjusted for major technical covariates.
    ax = fig.add_subplot(outer[1, 0])
    sc = state.sort_values("adjusted_log2_or", ascending=False)
    labels = sc["state"].tolist()
    colors = [COL["positive"] if q < 0.05 and e > 0 else COL["negative"] if q < 0.05 and e < 0 else "#A8A8A8"
              for e, q in zip(sc["adjusted_log2_or"], sc["FDR"])]
    forest(ax, labels, sc["adjusted_log2_or"], sc["ci_low"], sc["ci_high"],
           xlabel="Adjusted expanded-versus-singleton log2 odds ratio", colors=colors)
    for yi in np.arange(len(sc))[::-1][::2]: ax.axhspan(yi-0.5, yi+0.5, color="#F5F5F5", zorder=-2)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.text(0.99, 0.99, f"n={int(sc['n_participants'].min())}-{int(sc['n_participants'].max())}; diagnosis, batch, depth adjusted",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.0, color=COL["muted"])
    panel_title(ax, "Depth- and batch-adjusted state enrichment")
    panel_label(ax, "C")
    sc.to_csv(SRC / "Figure2_state_enrichment_displayed.csv", index=False)

    # D. Participant-level paired differences display magnitude and heterogeneity directly.
    ax = fig.add_subplot(outer[1, 1])
    rng_delta = np.random.default_rng(20260827)
    y_positions = np.arange(len(selected_modules))[::-1]
    for y, module in zip(y_positions, selected_modules):
        pm = paired_deltas[paired_deltas["module"] == module].copy()
        values = pm["delta"].dropna().to_numpy(float)
        violin = ax.violinplot(values, positions=[y], vert=False, widths=0.62,
                               showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#D9D9D9"); body.set_edgecolor("none"); body.set_alpha(0.62)
        for diagnosis in DIAG:
            subset = pm.loc[pm["Diagnosis1"] == diagnosis, "delta"].dropna().to_numpy(float)
            yy = rng_delta.normal(y, 0.085, len(subset))
            ax.scatter(subset, yy, s=7, color=COL[diagnosis], alpha=0.42,
                       edgecolor="none", rasterized=True, zorder=3)
        test = paired_tests[paired_tests["module"] == module].iloc[0]
        ax.hlines(y + 0.25, test["ci_low"], test["ci_high"], color=COL["ink"], lw=1.8, zorder=5)
        ax.scatter(test["median_delta"], y + 0.25, marker="D", s=27, color=COL["positive"],
                   edgecolor="white", lw=0.45, zorder=6)
        ax.text(0.99, (y + 0.25) / (len(selected_modules) - 1 + 0.60),
                f"median={test['median_delta']:.2f}; {q_label(test['FDR'])}",
                transform=ax.transAxes, ha="right", va="center", fontsize=5.1, color=COL["muted"])
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(y_positions, [module_titles[m] for m in selected_modules])
    ax.set_ylim(-0.43, len(selected_modules) - 0.40)
    ax.set_xlim(-0.32, 1.15)
    ax.set_xlabel("Participant delta: expanded - singleton score")
    ax.tick_params(axis="y", labelsize=6.3)
    style_axis(ax, "x")
    panel_title(ax, "Paired participant program shifts")
    panel_label(ax, "D")
    paired_deltas.to_csv(SRC / "Figure2_participant_program_deltas_displayed.csv", index=False)

    # E. Diagnosis conservation with uncertainty in both dimensions.
    ax = fig.add_subplot(outer[2, 0])
    ie = interactions[(interactions["family"] == "TCR_diagnosis") &
                      (interactions["contrast"].isin(["CD_expansion", "UC_expansion"]))].copy()
    cons = ie.pivot(index="module", columns="contrast", values=["effect", "SE"]).dropna().reset_index()
    cons.columns = ["module" if col[0] == "module" else f"{col[1]}_{col[0]}" for col in cons.columns]
    fdr_map = modules[modules["modality"].str.upper() == "TCR"].set_index("module")["FDR_global"]
    cons["FDR_global"] = cons["module"].map(fdr_map)
    cons_colors = [COL["positive"] if q < 0.05 and (x+y) > 0 else COL["negative"] if q < 0.05 else "#A8A8A8"
                   for x, y, q in zip(cons["CD_expansion_effect"], cons["UC_expansion_effect"], cons["FDR_global"])]
    ax.errorbar(cons["CD_expansion_effect"], cons["UC_expansion_effect"],
                xerr=1.96 * cons["CD_expansion_SE"], yerr=1.96 * cons["UC_expansion_SE"],
                fmt="none", ecolor="#B8B8B8", elinewidth=0.55, alpha=0.72, zorder=1)
    ax.scatter(cons["CD_expansion_effect"], cons["UC_expansion_effect"], s=27, c=cons_colors,
               edgecolor="white", lw=0.4, zorder=3)
    lower = min((cons["CD_expansion_effect"] - 1.96 * cons["CD_expansion_SE"]).min(),
                (cons["UC_expansion_effect"] - 1.96 * cons["UC_expansion_SE"]).min())
    upper = max((cons["CD_expansion_effect"] + 1.96 * cons["CD_expansion_SE"]).max(),
                (cons["UC_expansion_effect"] + 1.96 * cons["UC_expansion_SE"]).max())
    limits = [lower - 0.015, upper + 0.015]
    ax.plot(limits, limits, color=COL["muted"], ls="--", lw=0.7)
    ax.axhline(0, color=COL["grid"], lw=0.5); ax.axvline(0, color=COL["grid"], lw=0.5)
    ax.set_xlim(limits); ax.set_ylim(limits)
    label_offsets = {
        "EOMES_ZEB2_inflammatory_CD8_TRM_like": (5, 3),
        "Effector_cytotoxicity": (5, -10),
        "Th1_Tc1_inflammatory": (5, 7),
        "Naive_central_memory": (5, 3),
    }
    for module in selected_modules + ["Naive_central_memory"]:
        row = cons[cons["module"] == module]
        if len(row):
            ax.annotate(concise_module_label(module), (row["CD_expansion_effect"].iloc[0], row["UC_expansion_effect"].iloc[0]),
                        xytext=label_offsets[module], textcoords="offset points", fontsize=5.4)
    rho = cons[["CD_expansion_effect", "UC_expansion_effect"]].corr(method="spearman").iloc[0, 1]
    min_interaction_q = float(interaction_summary["minimum_interaction_FDR"].iloc[0])
    ax.text(0.03, 0.97, f"Spearman rho={rho:.2f}\nDiagnosis interaction: min q={min_interaction_q:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.7,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.86})
    ax.set_xlabel("Adjusted expansion effect in CD"); ax.set_ylabel("Adjusted expansion effect in UC")
    style_axis(ax, "both")
    ax.set_aspect("equal", adjustable="box")
    panel_title(ax, "Program effects conserved in CD and UC")
    panel_label(ax, "E")
    cons.to_csv(SRC / "Figure2_diagnosis_conservation_displayed.csv", index=False)

    # F. Acquisition-series replication with pooled heterogeneity and leave-one-series-out checks.
    ax = fig.add_subplot(outer[2, 1])
    rep_modules = selected_modules[:2]
    rep_colors = {rep_modules[0]: COL["positive"], rep_modules[1]: "#CC79A7"}
    rep = replication[replication["module"].isin(rep_modules)].copy()
    series_order = sorted([s for s in rep["series"].unique() if s != "Pooled"], key=lambda z: int(str(z).replace("S", ""))) + ["Pooled"]
    ybase = np.arange(len(series_order))[::-1]
    pooled_y = ybase[series_order.index("Pooled")]
    ax.axhspan(pooled_y - 0.45, pooled_y + 0.45, color="#F1F1F1", zorder=-2)
    for module, offset in zip(rep_modules, [0.12, -0.12]):
        rr = rep[rep["module"] == module].set_index("series").reindex(series_order)
        y = ybase + offset
        for k, series in enumerate(series_order):
            is_pooled = series == "Pooled"
            ax.hlines(y[k], rr.loc[series, "ci_low"], rr.loc[series, "ci_high"],
                      color=rep_colors[module], lw=1.6 if is_pooled else 0.85)
            ax.scatter(rr.loc[series, "effect"], y[k], color=rep_colors[module],
                       marker="D" if is_pooled else "o", s=31 if is_pooled else 18,
                       edgecolor="white", lw=0.4, zorder=3,
                       label=module_titles[module] if k == 0 else None)
    ax.axvline(0, color=COL["muted"], lw=0.6, ls="--")
    ns = rep[rep["module"] == rep_modules[0]].set_index("series").reindex(series_order)["n"]
    ax.set_yticks(ybase, [f"{s} (n={int(n)})" for s, n in zip(series_order, ns)])
    ax.set_xlabel("Mean expanded - singleton score (95% CI)")
    ax.legend(frameon=False, loc="upper right", fontsize=5.4, handletextpad=0.25)
    pooled = rep[rep["series"] == "Pooled"].set_index("module")
    robustness = (
        f"Pooled robustness\n"
        f"EOMES: I2={pooled.loc[rep_modules[0], 'I2']:.0f}%; LOO {pooled.loc[rep_modules[0], 'leave_one_series_out_low']:.2f}-{pooled.loc[rep_modules[0], 'leave_one_series_out_high']:.2f}\n"
        f"Cytotoxicity: I2={pooled.loc[rep_modules[1], 'I2']:.0f}%; LOO {pooled.loc[rep_modules[1], 'leave_one_series_out_low']:.2f}-{pooled.loc[rep_modules[1], 'leave_one_series_out_high']:.2f}"
    )
    ax.text(0.99, 0.025, robustness, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=4.8, color=COL["muted"],
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.88})
    style_axis(ax, "x")
    panel_title(ax, "Replication across acquisition series")
    panel_label(ax, "F")
    rep.to_csv(SRC / "Figure2_series_replication_displayed.csv", index=False)

    save_figure(fig, "Figure_2", MAIN)


def build_figure_3():
    from figure3_high_impact import build as build_high_impact_figure_3
    return build_high_impact_figure_3(
        ROOT, MAIN, COL, concise_module_label, heatmap, style_axis, panel_title, panel_label, save_figure
    )

    perm = read_csv(ROOT / "High Impact Additional Analyses/Paired TCR Sequence State/Table_TSS4_sequence_state_permutation_tests_by_series.csv")
    meta = read_csv(ROOT / "High Impact Additional Analyses/Paired TCR Sequence State/Table_TSS5_sequence_state_replication_meta_analysis.csv")
    edges = read_csv(ROOT / "High Impact Additional Analyses/Paired TCR Sequence State/Table_TSS3_paired_TCR_distance_graph_edges.csv")
    nodes = read_csv(ROOT / "High Impact Additional Analyses/Paired TCR Sequence State/Table_TSS1_paired_alpha_beta_clone_sequence_state.csv")
    gliph = read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA5_IBDTCR_beta_only_GLIPH2_all_unique_clusters.csv")
    gliph_summary = read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA5_IBDTCR_beta_only_GLIPH2_summary.csv")

    module_order = [
        "Th1_Tc1_inflammatory", "Effector_cytotoxicity", "EOMES_ZEB2_inflammatory_CD8_TRM_like",
        "Tissue_resident_mucosal_retention", "Gut_homing_intestinal_trafficking",
    ]
    series_order = sorted(perm["acquisition_series"].unique(), key=lambda x: int(str(x).replace("S", "")))

    def random_effect_pool(frame):
        """Pool correlations using permutation-null SDs as series-level uncertainty."""
        frame = frame.dropna(subset=["observed_edge_correlation", "null_sd"])
        y = frame["observed_edge_correlation"].to_numpy(float)
        v = np.square(frame["null_sd"].clip(lower=1e-6).to_numpy(float))
        if len(y) == 0:
            return np.nan, np.nan, np.nan
        fixed_w = 1 / v
        fixed = np.sum(fixed_w * y) / np.sum(fixed_w)
        q = np.sum(fixed_w * np.square(y - fixed))
        c = np.sum(fixed_w) - np.sum(np.square(fixed_w)) / np.sum(fixed_w)
        tau2 = max(0.0, (q - (len(y) - 1)) / c) if len(y) > 1 and c > 0 else 0.0
        w = 1 / (v + tau2)
        estimate = np.sum(w * y) / np.sum(w)
        se = math.sqrt(1 / np.sum(w))
        return estimate, estimate - 1.96 * se, estimate + 1.96 * se

    fig = plt.figure(figsize=(7.48, 8.35))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.86, 0.98, 0.78], width_ratios=[0.92, 1.0, 1.28],
                  hspace=0.68, wspace=0.78, left=0.075, right=0.985, top=0.97, bottom=0.065)

    # A: compact visual definition of the analysis.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.04, 0.69, 0.92, 0.20, "Paired alpha-beta TCRs", "CDR3 2-/3-mers + V/J genes"),
        (0.04, 0.39, 0.92, 0.20, "Cross-participant neighbors", "cosine distance <=0.50; up to 5"),
        (0.04, 0.09, 0.92, 0.20, "Program concordance", "within series; participant-stratified null"),
    ]
    for x, y, w, h, title, subtitle in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                    fc="#F3F6F8", ec=COL["muted"], lw=0.7))
        ax.text(x + w / 2, y + 0.125, title, ha="center", va="center", fontsize=7.3, fontweight="bold")
        ax.text(x + w / 2, y + 0.055, subtitle, ha="center", va="center", fontsize=6.1, color=COL["muted"])
    for y0, y1 in [(0.69, 0.59), (0.39, 0.29)]:
        ax.add_patch(FancyArrowPatch((0.50, y0), (0.50, y1), arrowstyle="-|>", mutation_scale=8,
                                     color=COL["ink"], lw=0.7))
    ax.text(0.50, 0.01, "Exact paired receptors excluded", ha="center", va="bottom", fontsize=6.2, color=COL["muted"])
    panel_title(ax, "Analysis framework")
    panel_label(ax, "A")

    # B: representative, prespecified-series cross-participant component.
    ax = fig.add_subplot(gs[0, 1])
    example_series = "S3"
    ee = edges[edges["acquisition_series"] == example_series].copy()
    graph = nx.from_pandas_edgelist(ee, "node1_id", "node2_id", edge_attr="sequence_distance")
    components = list(nx.connected_components(graph))
    component = max(components, key=lambda c: (len({x.split("::", 1)[0] for x in c}), len(c)))
    graph = graph.subgraph(component).copy()
    nn = nodes.copy()
    nn["node_id"] = nn["SampleID"].astype(str) + "::" + nn["clone_id"].astype(str)
    nn["Th1_z"] = nn.groupby("SampleID")["Th1_Tc1_inflammatory"].transform(
        lambda x: ((x - x.mean()) / x.std()).fillna(0)
    )
    node_map = nn.set_index("node_id")
    values = np.array([node_map.loc[n, "Th1_z"] if n in node_map.index else 0 for n in graph.nodes()], float)
    sizes = np.array([node_map.loc[n, "n_cells"] if n in node_map.index else 1 for n in graph.nodes()], float)
    pos = nx.spring_layout(graph, seed=20260824, weight=None, k=0.55)
    nx.draw_networkx_edges(graph, pos, ax=ax, width=0.55, alpha=0.36, edge_color=COL["muted"])
    vmax = max(1.5, np.nanpercentile(np.abs(values), 95))
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           node_size=18 + 7 * np.sqrt(np.maximum(sizes, 1)), edgecolors="white", linewidths=0.45)
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    cax = ax.inset_axes([0.42, 0.035, 0.52, 0.035])
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_label("Participant-centered Th1/Tc1 score", fontsize=5.7, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    ax.text(0.02, 0.94, f"{len(graph)} clonotypes; {len({x.split('::', 1)[0] for x in graph})} participants",
            transform=ax.transAxes, fontsize=6.1, color=COL["muted"])
    ax.axis("off")
    panel_title(ax, f"Cross-participant neighborhood ({example_series})")
    panel_label(ax, "B", x=-0.065, y=1.10)

    # C: all-series effect-size map, with multiplicity-controlled cells marked.
    ax = fig.add_subplot(gs[0, 2])
    all_perm = perm[(perm["stratum"] == "All") & perm["module"].isin(module_order)].copy()
    mat = all_perm.pivot(index="module", columns="acquisition_series", values="observed_edge_correlation")
    mat = mat.reindex(index=module_order, columns=series_order)
    im = heatmap(ax, mat.rename(index={m: concise_module_label(m) for m in module_order}),
                 cbar_label="Edge correlation", show_cbar=True, vlim=0.18)
    qmat = all_perm.pivot(index="module", columns="acquisition_series", values="FDR_within_series_stratum").reindex(
        index=module_order, columns=series_order)
    for i in range(qmat.shape[0]):
        for j in range(qmat.shape[1]):
            if pd.notna(qmat.iloc[i, j]) and qmat.iloc[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white" if abs(mat.iloc[i, j]) > 0.10 else COL["ink"])
    ax.tick_params(axis="y", labelsize=6.2)
    ax.text(1.0, -0.20, "* within-series FDR < 0.05", transform=ax.transAxes, ha="right", fontsize=5.8)
    panel_title(ax, "Program concordance across series")
    panel_label(ax, "C")

    # D: pooled biological effect sizes, rather than significance-only z scores.
    ax = fig.add_subplot(gs[1, 0])
    pooled = []
    for mod in module_order:
        g = all_perm[all_perm["module"] == mod]
        estimate, low, high = random_effect_pool(g)
        pooled.append((mod, estimate, low, high, int((g["observed_edge_correlation"] > 0).sum()), len(g)))
    y = np.arange(len(pooled))[::-1]
    for yi, (_, estimate, low, high, _, _) in zip(y, pooled):
        ax.plot([low, high], [yi, yi], color=COL["TCR"], lw=1.2)
        ax.plot(estimate, yi, "o", color=COL["TCR"], ms=4)
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(y, [concise_module_label(x[0]) for x in pooled])
    ax.tick_params(axis="y", labelsize=6.1)
    ax.set_xlabel("Pooled edge correlation (95% CI)")
    style_axis(ax, "x")
    for yi, (_, _, _, _, npos, ntotal) in zip(y, pooled):
        ax.text(1.02, yi, f"{npos}/{ntotal} positive", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=5.3, clip_on=False)
    panel_title(ax, "Random-effects program estimates")
    panel_label(ax, "D", x=-0.065, y=1.10)

    # E: diagnosis-restricted and leave-one-series-out robustness for primary programs.
    ax = fig.add_subplot(gs[1, 1])
    primary = module_order[:3]
    y = np.arange(len(primary))[::-1]
    offsets = {"All": 0.16, "CD": 0.0, "UC": -0.16}
    markers = {"All": "s", "CD": "o", "UC": "^"}
    colors = {"All": COL["ink"], "CD": COL["CD"], "UC": COL["UC"]}
    for yi, mod in zip(y, primary):
        g_all = perm[(perm["stratum"] == "All") & (perm["module"] == mod)]
        loo = []
        for series in g_all["acquisition_series"].unique():
            loo.append(random_effect_pool(g_all[g_all["acquisition_series"] != series])[0])
        ax.plot([min(loo), max(loo)], [yi + offsets["All"], yi + offsets["All"]], color="#A7A7A7", lw=2.3, zorder=1)
        for stratum in ("All", "CD", "UC"):
            g = perm[(perm["stratum"] == stratum) & (perm["module"] == mod)]
            estimate = random_effect_pool(g)[0]
            ax.plot(estimate, yi + offsets[stratum], marker=markers[stratum], color=colors[stratum],
                    ms=4.2, ls="", zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(y, [concise_module_label(m) for m in primary])
    ax.tick_params(axis="y", labelsize=6.1)
    ax.set_xlabel("Pooled edge correlation")
    style_axis(ax, "x")
    ax.text(0.03, 0.55, "■  All\n●  CD\n▲  UC\n—  LOO range",
            transform=ax.transAxes, ha="left", va="center", fontsize=5.4, color=COL["ink"], linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5})
    panel_title(ax, "Subset and series sensitivity")
    panel_label(ax, "E")

    # F: exact labels are a negative specificity endpoint.
    ax = fig.add_subplot(gs[1, 2])
    dom = perm[(perm["module"] == "Dominant_state_concordance") & (perm["stratum"] == "All")].copy()
    dom["order"] = dom["acquisition_series"].map({s: i for i, s in enumerate(series_order)})
    dom = dom.sort_values("order")
    delta = dom["observed_edge_correlation"] - dom["null_mean"]
    err = 1.96 * dom["null_sd"]
    x = np.arange(len(dom))
    ax.errorbar(x, delta, yerr=err, fmt="o", ms=3.8, color=COL["ink"], ecolor="#A7A7A7",
                elinewidth=0.9, capsize=1.8)
    ax.axhline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_xticks(x, dom["acquisition_series"], rotation=45, ha="right")
    ax.set_ylabel("Observed - permutation mean")
    ax.set_xlabel("Acquisition series")
    ax.text(0.98, 0.96, "Meta-analysis FDR > 0.05", transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
    style_axis(ax, "y")
    panel_title(ax, "Discrete state labels do not replicate")
    panel_label(ax, "F")

    # G: full-cohort, participant-carrier beta-chain GLIPH2 enrichment.
    comparisons = ["CD_vs_Control", "UC_vs_Control", "CD_vs_UC"]
    contrast_labels = {
        "CD_vs_Control": ("CD vs Control", "Control", "CD"),
        "UC_vs_Control": ("UC vs Control", "Control", "UC"),
        "CD_vs_UC": ("CD vs UC", "UC", "CD"),
    }
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :2], wspace=0.46)
    for j, comparison in enumerate(comparisons):
        ax = fig.add_subplot(sub[0, j])
        gg = gliph[gliph["comparison"] == comparison].drop_duplicates("unique_cluster_id").copy()
        yv = -np.log10(gg["fdr_unique_member_set"].clip(lower=1e-12))
        sig = gg["significant_fdr05"].astype(str).str.lower().eq("true")
        point_colors = [COL.get(str(direction), COL["muted"]) if is_sig else "#BEBEBE"
                        for direction, is_sig in zip(gg["enrichment_direction"], sig)]
        ax.scatter(gg["carrier_difference"], yv, s=np.where(sig, 15, 5), c=point_colors,
                   alpha=np.where(sig, 0.95, 0.42), edgecolor="none", rasterized=True)
        ax.axhline(-math.log10(0.05), color=COL["muted"], lw=0.6, ls="--")
        ax.axvline(0, color=COL["muted"], lw=0.6)
        title, left_group, right_group = contrast_labels[comparison]
        ax.set_title(title, fontsize=7.2, fontweight="bold", pad=3)
        ax.set_xlabel(f"← {left_group} enriched    {right_group} enriched →", fontsize=5.8)
        if j == 0:
            ax.set_ylabel("-log10 FDR")
            ax.text(0.0, 1.24, "Participant-carrier GLIPH2 enrichment", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, fontweight="bold", clip_on=False)
            panel_label(ax, "G", x=-0.26, y=1.27)
        style_axis(ax, "both")
        ax.tick_params(labelsize=5.8)

    # H: direction-aware count summary of the full-cohort GLIPH2 tests.
    ax = fig.add_subplot(gs[2, 2])
    summary = gliph_summary[gliph_summary["comparison"].isin(comparisons)].copy()
    x = np.arange(len(summary))
    count = summary["n_significant_unique_member_sets_fdr05"].to_numpy(float)
    enriched = []
    for _, row in summary.iterrows():
        first_group, second_group = {
            "CD_vs_Control": ("CD", "Control"), "UC_vs_Control": ("UC", "Control"), "CD_vs_UC": ("CD", "UC")
        }[row["comparison"]]
        enriched.append(first_group if row["n_first_group_enriched_fdr05"] > 0 else second_group)
    ax.bar(x, count, color=[COL.get(str(group), COL["muted"]) for group in enriched], width=0.62)
    ax.set_xticks(x, ["CD vs\nControl", "UC vs\nControl", "CD vs UC"])
    ax.set_ylabel("FDR-significant clusters")
    ax.set_ylim(0, max(count) + 2.2)
    for xi, value, group in zip(x, count, enriched):
        ax.text(xi, value + 0.25, f"{int(value)}\n{group} enriched", ha="center", va="bottom", fontsize=5.8)
    style_axis(ax, "y")
    panel_title(ax, "Diagnosis-associated beta-chain clusters")
    panel_label(ax, "H")
    save_figure(fig, "Figure_3", MAIN)


def build_figure_4():
    metrics = read_csv(BCR_DIR / "clonality_metrics_IBDBCR_Immunarch.csv")
    metrics = metrics.rename(columns={"Diagnosis1": "Diagnosis"})
    expansion = read_csv(BCR_DIR / "clonal_expansion_index_control_uc_cd_IBDBCR_values.csv")
    expansion = expansion.rename(columns={"Diagnosis1": "Diagnosis", "diagnosis": "Diagnosis"})
    if "Diagnosis" not in expansion.columns:
        expansion["Diagnosis"] = expansion.get("Group", "")
    state_path = BCR_DIR / "BCR Architecture Analyses/outputs_manuscript_cd_uc_control/clonotype_state_mapping/expanded_clonotype_state_composition_by_group.csv"
    state = read_csv(state_path)
    switched_path = BCR_DIR / "BCR Architecture Analyses/BCR isotype switching diagnosis comparisons/tables/bcr_isotype_switched_fraction_by_sample.csv"
    switched = read_csv(switched_path).rename(columns={"Diagnosis1": "Diagnosis", "diagnosis": "Diagnosis"})
    isotype = read_csv(PBMC_DIR / "Figure 5/BCR_isotype_proportions_recommended_manuscript_sample_level_values.csv")
    modules = read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA1_clone_aware_pseudobulk_module_enrichment.csv")
    deltas = read_csv(ROOT / "High Impact Additional Analyses/Clone State Interactions/Table_CSI2_participant_expanded_minus_singleton_deltas.csv")

    fig = plt.figure(figsize=(7.48, 8.7))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.18, 1.1], hspace=0.62, wspace=0.88,
                  left=0.09, right=0.98, top=0.97, bottom=0.07)
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0], wspace=0.55)
    metric_cols = [("Clonality", "Clonality"), ("Gini", "Gini coefficient"), ("Shannon", "Shannon diversity")]
    for j, (metric, label) in enumerate(metric_cols):
        ax = fig.add_subplot(sub[0, j])
        boxstrip(ax, metrics, metric, ylabel=label)
        ax.tick_params(axis="x", rotation=35)
        if j == 0:
            panel_title(ax, "Participant-level repertoire structure")
            panel_label(ax, "A", x=-0.55)

    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.45)
    numeric = expansion.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric if "exp" in c.lower() or "startrac" in c.lower()]
    if not candidates:
        candidates = numeric[:1]
    ax = fig.add_subplot(sub[0, 0])
    boxstrip(ax, expansion, candidates[0], ylabel=clean_label(candidates[0]))
    ax.tick_params(axis="x", rotation=35)
    panel_title(ax, "Clonal expansion")
    panel_label(ax, "B", x=-0.45)
    ax = fig.add_subplot(sub[0, 1])
    fraction_col = next((c for c in switched.columns if "fraction" in c.lower()), None)
    if fraction_col:
        boxstrip(ax, switched, fraction_col, ylabel="Class-switched fraction")
        ax.tick_params(axis="x", rotation=35)
    else:
        ax.set_axis_off()

    ax = fig.add_subplot(gs[1, 0])
    top_states = state.groupby("BcellState")["n_cells"].sum().nlargest(10).index
    mat = state[state["BcellState"].isin(top_states)].pivot_table(index="BcellState", columns="Diagnosis1",
                                                                   values="frac", aggfunc="mean").fillna(0)
    mat = mat.reindex(columns=DIAG)
    heatmap(ax, mat, cmap="YlGnBu", center=None, cbar_label="")
    panel_title(ax, "Expanded-clone B-cell states")
    panel_label(ax, "C", x=-0.065, y=1.10)

    ax = fig.add_subplot(gs[1, 1])
    dm = modules[modules["modality"].str.upper() == "BCR"].copy()
    dm = dm.reindex(dm["signed_log10_FDR"].abs().sort_values(ascending=False).head(12).index)
    lollipop(ax, dm["module"], dm["signed_log10_FDR"], threshold=-math.log10(0.05),
             xlabel="Signed -log10 FDR (expanded versus singleton)")
    ax.tick_params(axis="y", labelsize=6.2)
    panel_title(ax, "Clone-aware transcriptional programs")
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[2, 0])
    dd = deltas[deltas["modality"].str.upper() == "BCR"].copy()
    selected = dd.groupby("module")["delta"].apply(lambda x: x.abs().median()).nlargest(8).index
    mat = dd[dd["module"].isin(selected)].pivot_table(index="module", columns="Diagnosis1", values="delta", aggfunc="median")
    mat = mat.reindex(columns=DIAG)
    heatmap(ax, mat, cbar_label="", annotate=True)
    panel_title(ax, "Diagnosis-stratified clonal programs")
    panel_label(ax, "E")

    ax = fig.add_subplot(gs[2, 1])
    iso = isotype.copy()
    iso["class"] = iso["isotype"].replace({"IGHA1": "IgA", "IGHA2": "IgA", "IGHG1": "IgG", "IGHG2": "IgG",
                                               "IGHG3": "IgG", "IGHG4": "IgG", "IGHM": "IgM", "IGHD": "IgD", "IGHE": "IgE"})
    comp = iso.groupby(["Diagnosis1", "class"])["prop"].mean().unstack(fill_value=0).reindex(DIAG)
    comp = comp[[c for c in ["IgM", "IgD", "IgA", "IgG", "IgE"] if c in comp.columns]]
    colors = ["#56B4E9", "#999999", "#009E73", "#CC79A7", "#F0E442"][: len(comp.columns)]
    bottom = np.zeros(len(comp))
    for c, color in zip(comp.columns, colors):
        ax.bar(range(len(comp)), comp[c], bottom=bottom, color=color, width=0.65, label=c)
        bottom += comp[c].to_numpy()
    ax.set_xticks(range(len(comp)), comp.index)
    ax.set_ylabel("Mean repertoire fraction")
    ax.legend(frameon=False, ncol=min(5, len(comp.columns)), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    style_axis(ax, "y")
    panel_title(ax, "Heavy-chain isotype composition")
    panel_label(ax, "F")
    save_figure(fig, "Figure_4", MAIN)


def build_figure_5():
    shm = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL10_participant_region_SHM.csv")
    comparisons = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL12_diagnosis_comparisons.csv")
    isotype_effects = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL20_isotype_matched_SHM_effects.csv")
    paired_effects = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL25_paired_clonotype_SHM_effects.csv")
    paired_expansion_tests = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL28_paired_expansion_SHM_tests.csv")
    pairing_qc = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL29_paired_clonotype_mapping_QC.csv")
    paired_expansion_diagnosis = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL30_paired_expansion_diagnosis_effects.csv")
    pairing_selection = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL32_pairing_selection_SHM_effects.csv")
    representative_selection = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL33_representative_lineage_selection.csv")
    representative_nodes = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL34_representative_lineage_nodes.csv")
    representative_edges = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL35_representative_lineage_edges.csv")
    lineage_architecture = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL36_standardized_lineage_architecture_effects.csv")
    lineage_cduc = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL37_CD_UC_lineage_concordance.csv")
    lineage_metrics = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL4_germline_aware_lineage_metrics.csv")
    lineage_isotypes = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL5_lineage_isotype_composition.csv")
    lineage_divergence = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL21_germline_edge_free_lineage_divergence.csv")
    clinical_features = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL38_participant_IBD_clinical_lineage_features.csv")
    # Preserve the explicit untreated category; pandas otherwise interprets
    # the literal string "None" as missing on CSV import.
    clinical_features["Biologic"] = clinical_features["Biologic"].fillna("None").astype(str)
    calprotectin_effects = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL40_calprotectin_lineage_partial_correlations.csv")
    mucosal_effects = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL42_expanded_lineage_mucosal_state_effects.csv")

    def bootstrap_median_difference(x, y, seed, iterations=10000):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
        y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(float)
        rng = np.random.default_rng(seed)
        diffs = np.empty(iterations, dtype=float)
        for i in range(iterations):
            diffs[i] = np.median(rng.choice(x, len(x), replace=True)) - np.median(
                rng.choice(y, len(y), replace=True)
            )
        return np.median(x) - np.median(y), *np.quantile(diffs, [0.025, 0.975])

    # Within-participant CDR targeting is less sensitive than absolute SHM to
    # global mutation-rate calibration differences across acquisition series.
    regional = shm.copy()
    regional["region_class"] = np.where(regional["region"].str.lower().str.startswith("cdr"), "CDR", "FWR")
    targeting = (
        regional.groupby(["SampleID", "Diagnosis", "region_class"])["weighted_shm_rate"]
        .mean().unstack("region_class").dropna().reset_index()
    )
    targeting["CDR_minus_FWR"] = targeting["CDR"] - targeting["FWR"]
    targeting_effect_rows = []
    control_targeting = targeting.loc[targeting["Diagnosis"] == "Control", "CDR_minus_FWR"]
    for j, disease in enumerate(["CD", "UC"]):
        disease_targeting = targeting.loc[targeting["Diagnosis"] == disease, "CDR_minus_FWR"]
        effect, low, high = bootstrap_median_difference(disease_targeting, control_targeting, 20260824 + j)
        p_value = mannwhitneyu(disease_targeting, control_targeting, alternative="two-sided").pvalue
        targeting_effect_rows.append({"contrast": f"{disease} vs Control", "median_difference": effect,
                                      "ci_low": low, "ci_high": high, "p_value": p_value})
    targeting_effects = pd.DataFrame(targeting_effect_rows)
    p_values = targeting_effects["p_value"].to_numpy(float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    targeting_effects["FDR"] = adjusted

    # Expanded-lineage maturation landscape. Observed-only divergence is
    # defined only for lineages with at least two distinct observed heavy
    # alignments; points are therefore descriptive lineage-level summaries.
    dominant_isotype = (
        lineage_isotypes.sort_values(["lineage_id", "n_cells", "isotype_class"], ascending=[True, False, True])
        .drop_duplicates("lineage_id")[["lineage_id", "isotype_class"]]
        .rename(columns={"isotype_class": "dominant_isotype"})
    )
    maturation_landscape = (
        lineage_metrics.merge(
            lineage_divergence[["lineage_id", "n_unique_observed_sequences", "mean_pairwise_observed_divergence"]],
            on="lineage_id", how="inner",
        )
        .merge(dominant_isotype, on="lineage_id", how="left")
    )
    maturation_landscape["dominant_isotype"] = maturation_landscape["dominant_isotype"].fillna("Unmapped")
    maturation_landscape["representative_lineage"] = maturation_landscape["lineage_id"].isin(
        representative_selection["lineage_id"]
    )

    # Adjusted rank-residual points underlying the calprotectin association.
    # This mirrors the prespecified participant-level analysis in Table BGL40.
    def residualized_ranks(data, target, covariates):
        columns = [target] + covariates
        work = data[columns].dropna().copy()
        outcome = rankdata(pd.to_numeric(work[target], errors="coerce"), method="average")
        design = [np.ones(len(work), dtype=float)]
        for covariate in covariates:
            values = work[covariate]
            if values.dtype == object or str(values.dtype).startswith("string"):
                encoded = pd.get_dummies(values.astype(str), drop_first=True, dtype=float)
                design.extend(encoded[column].to_numpy(float) for column in encoded.columns)
            else:
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
                spread = np.nanstd(numeric)
                design.append((numeric - np.nanmean(numeric)) / spread if spread > 0 else np.zeros(len(numeric)))
        matrix = np.column_stack(design)
        residuals = outcome - matrix @ np.linalg.lstsq(matrix, outcome, rcond=None)[0]
        spread = residuals.std(ddof=1)
        return pd.Series(residuals / spread if spread > 0 else residuals, index=work.index)

    calprotectin_points = []
    adjustment = ["log_lineage_depth", "Biologic", "acquisition_series"]
    for diagnosis in ["CD", "UC"]:
        data = clinical_features[clinical_features["Diagnosis"].eq(diagnosis)].copy()
        x_residual = residualized_ranks(data, "log_calprotectin", adjustment)
        y_residual = residualized_ranks(data, "CDR_minus_FWR", adjustment)
        common = x_residual.index.intersection(y_residual.index)
        for index in common:
            calprotectin_points.append({
                "SampleID": data.loc[index, "SampleID"],
                "Diagnosis": diagnosis,
                "adjusted_log_calprotectin_rank": x_residual.loc[index],
                "adjusted_CDR_targeting_rank": y_residual.loc[index],
            })
    calprotectin_points = pd.DataFrame(calprotectin_points)

    targeting.to_csv(SRC / "Figure_5_CDR_minus_FWR_participant_values.csv", index=False)
    targeting_effects.to_csv(SRC / "Figure_5_CDR_minus_FWR_effects.csv", index=False)
    isotype_effects.to_csv(SRC / "Figure_5_isotype_matched_effects.csv", index=False)
    paired_effects.to_csv(SRC / "Figure_5_paired_clonotype_SHM_effects.csv", index=False)
    paired_expansion_tests.to_csv(SRC / "Figure_5_paired_expansion_SHM_tests.csv", index=False)
    pairing_qc.to_csv(SRC / "Figure_5_paired_clonotype_mapping_QC.csv", index=False)
    paired_expansion_diagnosis.to_csv(SRC / "Figure_5_paired_expansion_diagnosis_effects.csv", index=False)
    pairing_selection.to_csv(SRC / "Figure_5_pairing_selection_SHM_effects.csv", index=False)
    representative_selection.to_csv(SRC / "Figure_5_representative_lineage_selection.csv", index=False)
    representative_nodes.to_csv(SRC / "Figure_5_representative_lineage_nodes.csv", index=False)
    representative_edges.to_csv(SRC / "Figure_5_representative_lineage_edges.csv", index=False)
    lineage_architecture.to_csv(SRC / "Figure_5_standardized_lineage_architecture_effects.csv", index=False)
    lineage_cduc.to_csv(SRC / "Figure_5_CD_UC_lineage_concordance.csv", index=False)
    maturation_landscape.to_csv(SRC / "Figure_5_expanded_lineage_maturation_landscape.csv", index=False)
    calprotectin_points.to_csv(SRC / "Figure_5_adjusted_calprotectin_CDR_targeting_points.csv", index=False)
    calprotectin_effects[calprotectin_effects["metric"].eq("CDR_minus_FWR")].to_csv(
        SRC / "Figure_5_calprotectin_CDR_targeting_effects.csv", index=False
    )
    mucosal_effects[mucosal_effects["cell_state"].eq("IgA Plasma B Cell")].to_csv(
        SRC / "Figure_5_expanded_lineage_IgA_plasma_effects.csv", index=False
    )

    fig = plt.figure(figsize=(7.48, 10.45))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[0.96, 0.96, 0.92, 1.12, 1.02],
                  hspace=0.82, wspace=0.66, left=0.095, right=0.98, top=0.972, bottom=0.055)
    ax = fig.add_subplot(gs[0, 0])
    regions = [r for r in ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "fwr4"] if r in shm["region"].str.lower().unique()]
    if not regions:
        regions = list(shm["region"].dropna().unique())
    x = np.arange(len(regions))
    width = 0.24
    rng = np.random.default_rng(20260824)
    for k, d in enumerate(DIAG):
        dd = shm[shm["Diagnosis"] == d]
        med = dd.groupby("region")["weighted_shm_rate"].median().reindex(regions)
        q1 = dd.groupby("region")["weighted_shm_rate"].quantile(0.25).reindex(regions)
        q3 = dd.groupby("region")["weighted_shm_rate"].quantile(0.75).reindex(regions)
        for j, region in enumerate(regions):
            vals = pd.to_numeric(dd.loc[dd["region"] == region, "weighted_shm_rate"], errors="coerce").dropna()
            jitter = rng.normal(0, 0.020, len(vals))
            ax.scatter(np.full(len(vals), x[j] + (k - 1) * width) + jitter, vals, s=3.8,
                       color=COL[d], alpha=0.16, edgecolor="none", rasterized=True, zorder=2)
            above = vals > 0.28
            if above.any():
                ax.scatter(np.full(int(above.sum()), x[j] + (k - 1) * width), np.full(int(above.sum()), 0.274),
                           marker="^", s=17, color=COL[d], edgecolor="white", lw=0.35, zorder=5)
        ax.errorbar(x + (k - 1) * width, med, yerr=[med - q1, q3 - med], fmt="o", color=COL[d], ms=3.8,
                    capsize=2, lw=0.85, label=d, zorder=4)
    ax.set_xticks(x, [r.upper() for r in regions])
    ax.set_ylabel("Germline-relative SHM rate")
    ax.set_ylim(0, 0.28)
    ax.text(0.02, 0.98, "Triangle: UC CDR2 = 0.402", transform=ax.transAxes, va="top", ha="left",
            fontsize=5.8, color=COL["muted"])
    ax.legend(frameon=False, ncol=3)
    style_axis(ax, "y")
    figure5_heading(ax, "A", "Region-wide heavy-chain SHM increase")

    ax = fig.add_subplot(gs[0, 1])
    isotype_order = ["IgM", "IgA", "IgG"]
    y = np.arange(len(isotype_order))[::-1]
    offsets = {"CD": 0.13, "UC": -0.13}
    for disease in ["CD", "UC"]:
        dd = isotype_effects[isotype_effects["contrast"] == f"{disease} vs Control"].set_index("isotype_class").reindex(isotype_order)
        yy = y + offsets[disease]
        ax.hlines(yy, dd["ci_low"], dd["ci_high"], color=COL[disease], lw=1.0)
        for yi, (_, row) in zip(yy, dd.iterrows()):
            significant = row["FDR"] < 0.05
            ax.scatter(row["median_difference"], yi, s=24, facecolor=COL[disease] if significant else "white",
                       edgecolor=COL[disease], lw=1.0, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(y, isotype_order)
    ax.set_xlabel("Median SHM-rate difference (95% CI)")
    handles = [
        Line2D([0], [0], marker="o", ls="", ms=4.5, markerfacecolor="white", markeredgecolor=COL[d],
               label=f"{d} - Control") for d in ["CD", "UC"]
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    style_axis(ax, "x")
    figure5_heading(ax, "B", "Isotype-matched SHM remodeling")

    ax = fig.add_subplot(gs[1, 1])
    boxstrip(ax, targeting, "CDR_minus_FWR", ylabel="CDR - FWR SHM rate")
    ax.axhline(0, color=COL["muted"], lw=0.65, ls="--")
    effect_text = []
    for _, row in targeting_effects.iterrows():
        disease = row["contrast"].split()[0]
        effect_text.append(f"{disease} - Control: median diff={row['median_difference']:.3f}; FDR={row['FDR']:.2g}")
    ax.text(0.02, 0.98, "\n".join(effect_text), transform=ax.transAxes, va="top", ha="left",
            fontsize=6.2, color=COL["muted"])
    figure5_heading(ax, "D", "Preferential CDR-targeted mutation")

    ax = fig.add_subplot(gs[1, 0])
    estimand_order = [
        "All unique heavy sequences",
        "Paired-subset heavy clonotypes",
        "Exact paired H-L clonotypes",
        "Paired H-L cell-abundance weighted",
    ]
    y = np.arange(len(estimand_order))[::-1]
    for disease in ["CD", "UC"]:
        dd = paired_effects[paired_effects["contrast"] == f"{disease} vs Control"].set_index("estimand").reindex(estimand_order)
        yy = y + offsets[disease]
        ax.hlines(yy, dd["ci_low"], dd["ci_high"], color=COL[disease], lw=1.0)
        ax.scatter(dd["median_difference"], yy, color=COL[disease], s=24, edgecolor="white", lw=0.4,
                   label=f"{disease} - Control", zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(y, ["All unique heavy\nsequences", "Paired-subset heavy\nclonotypes",
                      "Exact paired H-L\nclonotypes", "Paired H-L\ncell-weighted"])
    ax.set_ylim(-0.45, 3.70)
    ax.set_xlabel("Median overall SHM-rate difference (95% CI)")
    displayed = paired_effects[paired_effects["estimand"].isin(estimand_order)]
    ax.text(0.98, 0.98, f"Displayed contrasts FDR <= {displayed['FDR'].max():.2g}", transform=ax.transAxes,
            va="top", ha="right", fontsize=6.0, color=COL["muted"])
    style_axis(ax, "x")
    figure5_heading(ax, "C", "Paired-clonotype-resolved heavy-chain SHM")

    tree_container = fig.add_subplot(gs[2, 0])
    tree_container.set_axis_off()
    # Enlarge the representative-lineage panel into the generous inter-panel
    # gutters while preserving the fixed Cell Press page dimensions and the
    # allocation of panel F. The nested axes are positioned explicitly so the
    # lineage artwork, rather than blank padding, occupies the added space.
    tree_position = tree_container.get_position()
    tree_container.set_position([
        tree_position.x0,
        tree_position.y0 - 0.006,
        tree_position.width * 1.12,
        tree_position.height * 1.10,
    ])
    figure5_heading(tree_container, "E", "Representative germline-rooted lineage graphs")
    tree_position = tree_container.get_position()
    tree_gap = 0.006
    tree_width = (tree_position.width - 2 * tree_gap) / 3
    global_xmax = max(0.01, representative_nodes["x"].max())
    for tree_index, diagnosis in enumerate(DIAG):
        tree_ax = fig.add_axes([
            tree_position.x0 + tree_index * (tree_width + tree_gap),
            tree_position.y0,
            tree_width,
            tree_position.height,
        ])
        selected = representative_selection[representative_selection["Diagnosis"].eq(diagnosis)].iloc[0]
        nodes = representative_nodes[representative_nodes["Diagnosis"].eq(diagnosis)].set_index("node_id")
        edges = representative_edges[representative_edges["Diagnosis"].eq(diagnosis)]
        ymin, ymax = nodes["y"].min(), nodes["y"].max()
        yscale = max(1.0, ymax - ymin)
        normalized_y = (nodes["y"] - ymin) / yscale
        display_y = 0.16 + 0.70 * normalized_y
        # Center shorter examples on the common x scale. Translation preserves
        # branch-length comparability while removing one-sided empty space.
        local_xmax = max(0.0, nodes["x"].max())
        x_offset = 0.5 * (global_xmax - local_xmax)
        for _, edge in edges.iterrows():
            source = nodes.loc[int(edge["source"])]
            target = nodes.loc[int(edge["target"])]
            tree_ax.plot([source["x"] + x_offset, target["x"] + x_offset],
                         [display_y.loc[int(edge["source"])], display_y.loc[int(edge["target"])]],
                         color=COL["muted"] if bool(edge["germline_edge"]) else COL[diagnosis],
                         lw=0.9 if bool(edge["germline_edge"]) else 1.05,
                         ls="--" if bool(edge["germline_edge"]) else "-", zorder=1)
        observed = nodes[nodes["node_type"].eq("Observed")]
        sizes = 14 + 8 * np.sqrt(observed["count"].clip(lower=1))
        tree_ax.scatter(observed["x"] + x_offset, display_y.loc[observed.index], s=sizes,
                        color=COL[diagnosis], alpha=0.82, edgecolor="white", lw=0.45, zorder=3)
        root_node = nodes[nodes["node_type"].eq("Germline")].iloc[0]
        root_id = nodes[nodes["node_type"].eq("Germline")].index[0]
        tree_ax.scatter(root_node["x"] + x_offset, display_y.loc[root_id], marker="D", s=27,
                        color=COL["ink"], edgecolor="white", lw=0.45, zorder=4)
        light_v = str(selected["dominant_light_id"]).split("|")[0].replace("nan", "unmapped")
        annotation = (f"{selected['SampleID']} | {selected['v_gene']}/{selected['j_gene']}\n"
                      f"{int(selected['n_unique_observed_sequences'])} seq; {int(selected['weighted_abundance'])} cells | {selected['isotype_classes']}\n"
                      f"{light_v} | {selected['dominant_cell_state']}")
        tree_ax.text(0.5, 0.94, diagnosis, transform=tree_ax.transAxes, color=COL[diagnosis],
                     fontsize=7.0, fontweight="bold", ha="center", va="top")
        tree_ax.text(0.5, 0.015, annotation, transform=tree_ax.transAxes, ha="center", va="bottom",
                     fontsize=4.35, linespacing=1.10)
        tree_ax.set_xlim(-0.02 * global_xmax, global_xmax * 1.02)
        tree_ax.set_ylim(0, 1)
        tree_ax.set_xticks([]); tree_ax.set_yticks([])
        for spine in tree_ax.spines.values():
            spine.set_visible(False)

    ax = fig.add_subplot(gs[2, 1])
    metric_order = [
        "expanded_lineages",
        "mean_MST_branch_length",
        "mean_within_lineage_divergence",
        "fraction_class_switched_lineages",
    ]
    labels = [
        "Expanded lineages",
        "Germline-inclusive\nbranch length",
        "Observed-only\ndivergence",
        "Class-switched\nlineage fraction",
    ]
    ybase = np.arange(len(metric_order))[::-1]
    for disease in ["CD", "UC"]:
        data = lineage_architecture[lineage_architecture["contrast"].eq(f"{disease} vs Control")].set_index("metric").reindex(metric_order)
        yy = ybase + offsets[disease]
        ax.hlines(yy, data["ci_low"], data["ci_high"], color=COL[disease], lw=1.0)
        for yvalue, (_, row) in zip(yy, data.iterrows()):
            significant = row["FDR"] < 0.05
            ax.scatter(row["standardized_median_difference"], yvalue, s=24,
                       facecolor=COL[disease] if significant else "white", edgecolor=COL[disease], lw=1.0, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(ybase, labels)
    ax.set_xlabel("Median difference (control-IQR units; 95% CI)")
    ax.text(0.98, 0.98, f"CD vs UC: all FDR >= {lineage_cduc['FDR'].min():.2g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=COL["muted"])
    style_axis(ax, "x")
    figure5_heading(ax, "F", "Shared IBD lineage architecture")

    # G: Descriptive lineage-level maturation geometry. Shared axes and
    # marginal rugs make the shift in lineage occupancy visible without
    # substituting lineage-level points for participant-level inference.
    landscape_container = fig.add_subplot(gs[3, :])
    landscape_container.set_axis_off()
    figure5_heading(landscape_container, "G", "Expanded-lineage maturation landscape", full_width=True)
    landscape_grid = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[3, :],
                                             height_ratios=[0.15, 0.85], hspace=0.08, wspace=0.30)
    isotype_colors = {"IgM": "#56B4E9", "IgA": "#009E73", "IgG": "#CC79A7", "IgD": "#999999", "Unmapped": "#C8C8C8"}
    xmax = max(0.25, maturation_landscape["total_MST_branch_length"].quantile(0.995))
    ymax = max(0.05, maturation_landscape["mean_pairwise_observed_divergence"].quantile(0.995))
    for index, diagnosis in enumerate(DIAG):
        ax = fig.add_subplot(landscape_grid[1, index])
        data = maturation_landscape[maturation_landscape["Diagnosis"].eq(diagnosis)].copy()
        data["display_size"] = 5.5 + 8.5 * np.sqrt(np.log1p(data["weighted_abundance"]).clip(lower=0))
        for isotype in ["Unmapped", "IgD", "IgM", "IgA", "IgG"]:
            subset = data[data["dominant_isotype"].eq(isotype)]
            if subset.empty:
                continue
            ax.scatter(subset["total_MST_branch_length"], subset["mean_pairwise_observed_divergence"],
                       s=subset["display_size"], color=isotype_colors.get(isotype, "#B8B8B8"),
                       alpha=0.43, edgecolor="none", rasterized=True, zorder=2)
        selected = data[data["representative_lineage"]]
        if not selected.empty:
            ax.scatter(selected["total_MST_branch_length"], selected["mean_pairwise_observed_divergence"],
                       marker="*", s=62, color=COL[diagnosis], edgecolor="white", lw=0.7, zorder=5)
        ax.axvline(data["total_MST_branch_length"].median(), color=COL[diagnosis], lw=0.65, ls=":", alpha=0.8)
        ax.axhline(data["mean_pairwise_observed_divergence"].median(), color=COL[diagnosis], lw=0.65, ls=":", alpha=0.8)
        # Compact marginal rugs preserve density information without expanding
        # the already information-rich main figure.
        ax.plot(data["total_MST_branch_length"], np.full(len(data), ymax * 0.985), "|",
                color=COL[diagnosis], ms=2.0, mew=0.35, alpha=0.18, clip_on=True)
        ax.plot(np.full(len(data), xmax * 0.985), data["mean_pairwise_observed_divergence"], "_",
                color=COL[diagnosis], ms=2.0, mew=0.35, alpha=0.18, clip_on=True)
        ax.set_xlim(-0.02 * xmax, xmax)
        ax.set_ylim(-0.02 * ymax, ymax)
        ax.set_title(f"{diagnosis} (n={len(data):,} lineages)", color=COL[diagnosis], fontsize=7.0,
                     fontweight="bold", pad=2)
        ax.set_xlabel("Germline-inclusive branch length")
        if index == 0:
            ax.set_ylabel("Observed-only divergence")
        else:
            ax.set_yticklabels([])
        style_axis(ax, "both")
    landscape_container.text(0.995, 1.015, "Star: panel E lineage; dotted lines: medians",
                             transform=landscape_container.transAxes, ha="right", va="bottom",
                             fontsize=5.6, color=COL["muted"])
    landscape_container.legend(
        handles=[Line2D([0], [0], marker="o", ls="", ms=4.3, markerfacecolor=isotype_colors[key],
                        markeredgecolor="none", label=key) for key in ["IgM", "IgA", "IgG", "IgD", "Unmapped"]],
        frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.60, 1.10), columnspacing=1.0,
        handletextpad=0.35,
    )

    # H: The adjusted participant-level clinical relationship, shown for both
    # IBD diagnoses so the UC association is not presented without its null CD
    # comparator.
    clinical_container = fig.add_subplot(gs[4, :])
    clinical_container.set_axis_off()
    figure5_heading(clinical_container, "H", "BCR maturation and intestinal inflammatory burden", full_width=True)
    clinical_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[4, :],
                                            height_ratios=[0.15, 0.85], hspace=0.08, wspace=0.38)
    clinical_effect = calprotectin_effects[calprotectin_effects["metric"].eq("CDR_minus_FWR")].set_index("Diagnosis")
    rng_fit = np.random.default_rng(20260828)
    for index, diagnosis in enumerate(["CD", "UC"]):
        ax = fig.add_subplot(clinical_grid[1, index])
        data = calprotectin_points[calprotectin_points["Diagnosis"].eq(diagnosis)]
        xdata = data["adjusted_log_calprotectin_rank"].to_numpy(float)
        ydata = data["adjusted_CDR_targeting_rank"].to_numpy(float)
        ax.scatter(xdata, ydata, s=13, color=COL[diagnosis], alpha=0.55,
                   edgecolor="white", lw=0.3, zorder=3)
        xgrid = np.linspace(xdata.min(), xdata.max(), 100)
        fit = np.polyfit(xdata, ydata, 1)
        ax.plot(xgrid, np.polyval(fit, xgrid), color=COL[diagnosis], lw=1.25, zorder=4)
        bootstrap_lines = np.empty((1000, len(xgrid)), dtype=float)
        for iteration in range(len(bootstrap_lines)):
            sample_index = rng_fit.integers(0, len(data), len(data))
            bootstrap_lines[iteration] = np.polyval(np.polyfit(xdata[sample_index], ydata[sample_index], 1), xgrid)
        low, high = np.nanquantile(bootstrap_lines, [0.025, 0.975], axis=0)
        ax.fill_between(xgrid, low, high, color=COL[diagnosis], alpha=0.13, lw=0, zorder=1)
        result = clinical_effect.loc[diagnosis]
        ax.text(0.03, 0.96,
                f"partial rho={result['partial_spearman_rho']:.3f}; FDR={result['FDR']:.3g}\nn={int(result['n_participants'])}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
                color=COL[diagnosis], fontweight="bold" if result["FDR"] < 0.05 else "normal")
        ax.set_title(diagnosis, color=COL[diagnosis], fontsize=7.0, fontweight="bold", pad=2)
        ax.set_xlabel("Adjusted log calprotectin rank")
        if index == 0:
            ax.set_ylabel("Adjusted CDR-targeting rank")
        style_axis(ax, "both")

    save_figure(fig, "Figure_5", MAIN)


def build_figure_6():
    integrated_out = ROOT / "Trajectory Integrated Figure Set"
    integrated_main = integrated_out / "Main Figures"
    integrated_src = integrated_out / "Source Data"
    integrated_leg = integrated_out / "Legends"
    for directory in (integrated_main, integrated_src, integrated_leg):
        directory.mkdir(parents=True, exist_ok=True)
    features = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_integrated_participant_features.csv")
    coupling = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_CD_specific_T_B_module_coupling.csv")
    clinical = read_csv(ROOT / "High Impact Additional Analyses/Priority Analyses/Table_PA_clinical_metadata.csv")
    state_abundance = read_csv(SRC / "Figure1_participant_state_abundance.csv")
    clinical = clinical[["SampleID", "Inflammation1", "Biologic"]].drop_duplicates("SampleID")
    state_covariates = (state_abundance[state_abundance["cell_state"].isin(["CD8 Tem GZMB+", "IgM Plasma B Cell"])]
                        .pivot(index="PatientID", columns="cell_state", values="logit_fraction")
                        .rename(columns={"CD8 Tem GZMB+": "cd8_tem_gzmb_logit_fraction",
                                         "IgM Plasma B Cell": "igm_plasma_logit_fraction"})
                        .reset_index().rename(columns={"PatientID": "SampleID"}))
    features = features.copy()
    features = features.merge(clinical, on="SampleID", how="left")
    features = features.merge(state_covariates, on="SampleID", how="left")
    trajectory_cells = read_csv(
        integrated_src / "Figure2_primary_graph_cells.csv.gz",
        usecols=["SampleID", "paired_alpha_beta", "clone_status", "pseudotime"],
    )
    trajectory_covariate = (
        trajectory_cells[
            trajectory_cells["paired_alpha_beta"].astype(bool)
            & trajectory_cells["clone_status"].eq("Expanded")
        ]
        .groupby("SampleID", as_index=False)["pseudotime"]
        .median()
        .rename(columns={"pseudotime": "cd8_expanded_median_pseudotime"})
    )
    features = features.merge(trajectory_covariate, on="SampleID", how="left")
    features["log_tcr_depth"] = np.log1p(features["tcr_total_cells"])
    features["log_bcr_depth"] = np.log1p(features["bcr_total_cells"])
    features["acquisition_series"] = features["Batch"].astype(str).str.replace(r"[AB]$", "", regex=True)
    base_covariates = ["Age", "Sex", "log_tcr_depth", "log_bcr_depth"]
    outcomes = ["bcr_IgA_mucosal_module", "bcr_plasma_differentiation_module"]
    context_outcomes = outcomes + ["bcr_IgG_inflammatory_module", "bcr_BAFF_APRIL_module"]
    outcome_labels = {
        "bcr_IgA_mucosal_module": "IgA mucosal plasma-cell program",
        "bcr_plasma_differentiation_module": "Plasmablast/plasma-cell differentiation",
        "bcr_IgG_inflammatory_module": "IgG inflammatory program",
        "bcr_BAFF_APRIL_module": "BAFF/APRIL program",
    }

    def residualized_ranks(data, target, covariates):
        work = data[[target] + covariates].dropna().copy()
        y = rankdata(work[target].to_numpy(float), method="average")
        columns = [np.ones(len(work))]
        for covariate in covariates:
            if work[covariate].dtype == object:
                levels = sorted(work[covariate].astype(str).unique())
                columns.extend((work[covariate].astype(str) == level).astype(float).to_numpy()
                               for level in levels[1:])
            else:
                value = work[covariate].to_numpy(float)
                sd = np.nanstd(value)
                columns.append((value - np.nanmean(value)) / sd if sd > 0 else np.zeros(len(value)))
        design = np.column_stack(columns)
        residuals = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        residuals = (residuals - residuals.mean()) / residuals.std(ddof=1)
        return pd.Series(residuals, index=work.index)

    def partial_rho(data, outcome, covariates):
        work = data[["tcr_cytotoxic_expanded", outcome] + covariates].dropna().copy()
        x = residualized_ranks(work, "tcr_cytotoxic_expanded", covariates)
        y = residualized_ranks(work, outcome, covariates)
        common = x.index.intersection(y.index)
        return float(np.corrcoef(x.loc[common], y.loc[common])[0, 1]), len(common)

    def partial_rho_test(data, outcome, covariates):
        work = data[["tcr_cytotoxic_expanded", outcome] + covariates].dropna().copy()
        x = residualized_ranks(work, "tcr_cytotoxic_expanded", covariates)
        y = residualized_ranks(work, outcome, covariates)
        common = x.index.intersection(y.index)
        result = pearsonr(x.loc[common], y.loc[common])
        return float(result.statistic), float(result.pvalue), len(common)

    def bh_adjust(p_values):
        p_values = np.asarray(p_values, dtype=float)
        order = np.argsort(p_values)
        ranked = p_values[order]
        adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
        output = np.empty_like(adjusted)
        output[order] = np.minimum(adjusted, 1.0)
        return output

    def bootstrap_partial_rho(data, outcome, covariates, n_boot=2000):
        work = data[["tcr_cytotoxic_expanded", outcome] + covariates].dropna().reset_index(drop=True)
        rng = np.random.default_rng(20260824)
        estimates = []
        for _ in range(n_boot):
            sample = work.iloc[rng.integers(0, len(work), len(work))].reset_index(drop=True)
            try:
                estimate, _ = partial_rho(sample, outcome, covariates)
                if np.isfinite(estimate):
                    estimates.append(estimate)
            except np.linalg.LinAlgError:
                continue
        return np.quantile(estimates, [0.025, 0.975])

    def stratified_residual_permutation(data, outcome, covariates, strata="acquisition_series", n_perm=10000):
        columns = ["SampleID", "tcr_cytotoxic_expanded", outcome, strata] + covariates
        work = data[columns].dropna().copy()
        x = residualized_ranks(work, "tcr_cytotoxic_expanded", covariates)
        y = residualized_ranks(work, outcome, covariates)
        work = work.loc[x.index.intersection(y.index)].copy()
        x_values = x.loc[work.index].to_numpy(float)
        y_values = y.loc[work.index].to_numpy(float)
        observed = float(np.corrcoef(x_values, y_values)[0, 1])
        groups = [np.flatnonzero(work[strata].to_numpy() == level)
                  for level in sorted(work[strata].astype(str).unique())]
        rng = np.random.default_rng(20260827 + context_outcomes.index(outcome))
        null = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            permuted = y_values.copy()
            for indices in groups:
                permuted[indices] = rng.permutation(permuted[indices])
            null[i] = np.corrcoef(x_values, permuted)[0, 1]
        p_value = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
        return observed, null, float(p_value), len(work)

    def paired_program_bootstrap(data, n_boot=5000):
        work = data[["tcr_cytotoxic_expanded"] + context_outcomes + base_covariates].dropna().reset_index(drop=True)
        observed = {outcome: partial_rho(work, outcome, base_covariates)[0] for outcome in context_outcomes}
        comparisons = [
            ("bcr_plasma_differentiation_module", "bcr_IgG_inflammatory_module", "Plasma differentiation - IgG inflammatory"),
            ("bcr_plasma_differentiation_module", "bcr_BAFF_APRIL_module", "Plasma differentiation - BAFF/APRIL"),
            ("bcr_IgA_mucosal_module", "bcr_IgG_inflammatory_module", "IgA mucosal - IgG inflammatory"),
            ("bcr_IgA_mucosal_module", "bcr_BAFF_APRIL_module", "IgA mucosal - BAFF/APRIL"),
        ]
        rng = np.random.default_rng(20260827)
        draws = {label: [] for _, _, label in comparisons}
        for _ in range(n_boot):
            sample = work.iloc[rng.integers(0, len(work), len(work))].reset_index(drop=True)
            try:
                estimates = {outcome: partial_rho(sample, outcome, base_covariates)[0]
                             for outcome in context_outcomes}
            except np.linalg.LinAlgError:
                continue
            for first, second, label in comparisons:
                value = estimates[first] - estimates[second]
                if np.isfinite(value):
                    draws[label].append(value)
        rows = []
        for first, second, label in comparisons:
            values = np.asarray(draws[label])
            difference = observed[first] - observed[second]
            p_value = min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))
            rows.append({"comparison": label, "first_outcome": first, "second_outcome": second,
                         "rho_difference": difference, "ci_low": np.quantile(values, 0.025),
                         "ci_high": np.quantile(values, 0.975), "bootstrap_p": p_value,
                         "n": len(work), "bootstrap_draws": len(values)})
        result = pd.DataFrame(rows)
        result["FDR_across_four"] = bh_adjust(result["bootstrap_p"])
        return result

    cd = features[features["Diagnosis1"] == "CD"].copy()
    residual_rows = []
    sensitivity_rows = []
    for outcome in outcomes:
        work = cd[["SampleID", "tcr_cytotoxic_expanded", outcome] + base_covariates].dropna().copy()
        work["x_residual"] = residualized_ranks(work, "tcr_cytotoxic_expanded", base_covariates)
        work["y_residual"] = residualized_ranks(work, outcome, base_covariates)
        work["outcome"] = outcome
        residual_rows.append(work[["SampleID", "outcome", "x_residual", "y_residual"]])

        primary = coupling[(coupling["outcome"] == outcome) & (coupling["diagnosis"] == "CD")].iloc[0]
        sensitivity_rows.append({"outcome": outcome, "model": "Primary", "rho": primary.partial_rho,
                                 "ci_low": primary.ci_low, "ci_high": primary.ci_high, "n": int(primary.n)})
        primary_index = len(sensitivity_rows) - 1
        batch_covariates = base_covariates + ["Batch"]
        batch_rho, batch_n = partial_rho(cd, outcome, batch_covariates)
        batch_ci = bootstrap_partial_rho(cd, outcome, batch_covariates)
        sensitivity_rows.append({"outcome": outcome, "model": "+ acquisition series", "rho": batch_rho,
                                 "ci_low": batch_ci[0], "ci_high": batch_ci[1], "n": batch_n})

        inflammation_covariates = base_covariates + ["Inflammation1"]
        inflammation_rho, inflammation_n = partial_rho(cd, outcome, inflammation_covariates)
        inflammation_ci = bootstrap_partial_rho(cd, outcome, inflammation_covariates)
        sensitivity_rows.append({"outcome": outcome, "model": "+ inflammation status", "rho": inflammation_rho,
                                 "ci_low": inflammation_ci[0], "ci_high": inflammation_ci[1], "n": inflammation_n})

        biologic_covariates = base_covariates + ["Biologic"]
        biologic_rho, biologic_n = partial_rho(cd, outcome, biologic_covariates)
        biologic_ci = bootstrap_partial_rho(cd, outcome, biologic_covariates)
        sensitivity_rows.append({"outcome": outcome, "model": "+ biologic exposure", "rho": biologic_rho,
                                 "ci_low": biologic_ci[0], "ci_high": biologic_ci[1], "n": biologic_n})

        composition_covariates = base_covariates + ["cd8_tem_gzmb_logit_fraction", "igm_plasma_logit_fraction"]
        composition_rho, composition_n = partial_rho(cd, outcome, composition_covariates)
        composition_ci = bootstrap_partial_rho(cd, outcome, composition_covariates)
        sensitivity_rows.append({"outcome": outcome, "model": "+ cell-state composition", "rho": composition_rho,
                                 "ci_low": composition_ci[0], "ci_high": composition_ci[1], "n": composition_n})

        trajectory_covariates = base_covariates + ["cd8_expanded_median_pseudotime"]
        trajectory_rho, trajectory_n = partial_rho(cd, outcome, trajectory_covariates)
        trajectory_ci = bootstrap_partial_rho(cd, outcome, trajectory_covariates)
        sensitivity_rows.append({"outcome": outcome, "model": "+ expanded-clone trajectory position", "rho": trajectory_rho,
                                 "ci_low": trajectory_ci[0], "ci_high": trajectory_ci[1], "n": trajectory_n})

        loo = []
        valid = cd[["tcr_cytotoxic_expanded", outcome] + base_covariates].dropna()
        for held_out in valid.index:
            estimate, _ = partial_rho(valid.drop(index=held_out), outcome, base_covariates)
            loo.append(estimate)
        sensitivity_rows[primary_index]["loo_participant_low"] = min(loo)
        sensitivity_rows[primary_index]["loo_participant_high"] = max(loo)

        series_loo = []
        for series in sorted(valid.join(cd[["acquisition_series"]])["acquisition_series"].dropna().unique()):
            retained = cd.loc[cd["acquisition_series"] != series]
            estimate, _ = partial_rho(retained, outcome, base_covariates)
            series_loo.append(estimate)
        sensitivity_rows[primary_index]["loo_series_low"] = min(series_loo)
        sensitivity_rows[primary_index]["loo_series_high"] = max(series_loo)

    residual_data = pd.concat(residual_rows, ignore_index=True)
    sensitivity = pd.DataFrame(sensitivity_rows)
    context_rows = []
    for outcome in context_outcomes:
        estimate, p_value, n = partial_rho_test(cd, outcome, base_covariates)
        ci_low, ci_high = bootstrap_partial_rho(cd, outcome, base_covariates)
        context_rows.append({"outcome": outcome, "label": outcome_labels[outcome], "rho": estimate,
                             "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value, "n": n})
    context = pd.DataFrame(context_rows)
    context["FDR_across_four"] = bh_adjust(context["p_value"])
    permutation_rows = []
    permutation_null = {}
    for outcome in outcomes:
        observed, null, p_value, n = stratified_residual_permutation(cd, outcome, base_covariates)
        permutation_null[outcome] = null
        permutation_rows.append({"outcome": outcome, "observed_rho": observed,
                                 "null_ci_low": np.quantile(null, 0.025),
                                 "null_ci_high": np.quantile(null, 0.975),
                                 "empirical_p": p_value, "n": n, "permutations": len(null),
                                 "permutation_strata": "acquisition_series"})
    permutation = pd.DataFrame(permutation_rows)
    program_differences = paired_program_bootstrap(cd)
    coupling.to_csv(integrated_src / "Figure6_trajectory_adjusted_coupling_effects.csv", index=False)
    residual_data.to_csv(integrated_src / "Figure6_trajectory_adjusted_rank_residuals.csv", index=False)
    sensitivity.to_csv(integrated_src / "Figure6_trajectory_adjusted_sensitivity.csv", index=False)
    context.to_csv(integrated_src / "Figure6_trajectory_adjusted_cross_program_context.csv", index=False)
    permutation.to_csv(integrated_src / "Figure6_trajectory_adjusted_matched_pair_permutation.csv", index=False)
    program_differences.to_csv(integrated_src / "Figure6_trajectory_adjusted_program_difference_tests.csv", index=False)
    trajectory_covariate.to_csv(integrated_src / "Figure6_cd8_expanded_clone_trajectory_covariate.csv", index=False)

    fig = plt.figure(figsize=(7.48, 8.35))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[0.90, 1.0, 1.26], hspace=0.76, wspace=0.90,
                  left=0.10, right=0.98, top=0.96, bottom=0.08)

    ax = fig.add_subplot(gs[0, :3])
    y_positions = [5.4, 4.4, 3.4, 2.0, 1.0, 0.0]
    labels = ["Control (n=17)", "CD (n=56)", "UC (n=54)",
              "Control (n=17)", "CD (n=56)", "UC (n=54)"]
    ordered_rows = []
    for outcome in outcomes:
        for diagnosis in DIAG:
            ordered_rows.append(coupling[(coupling["outcome"] == outcome) &
                                         (coupling["diagnosis"] == diagnosis)].iloc[0])
    for y, row in zip(y_positions, ordered_rows):
        color = COL[row.diagnosis]
        ax.errorbar(row.partial_rho, y,
                    xerr=[[row.partial_rho - row.ci_low], [row.ci_high - row.partial_rho]],
                    fmt="o", ms=4.3, lw=1.05, capsize=2, color=color, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(-0.9, 0.9)
    ax.set_xlabel("Partial Spearman ρ (95% CI)")
    header_box = dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.8)
    ax.text(0.01, 1.015, "IgA mucosal plasma-cell program | interaction P=0.0025",
            transform=ax.transAxes, fontsize=6.1, va="bottom", fontweight="bold", bbox=header_box)
    ax.text(0.01, 0.50, "Plasma-cell differentiation | interaction P=0.0013",
            transform=ax.transAxes, fontsize=6.1, va="center", fontweight="bold", bbox=header_box)
    style_axis(ax, "x")
    ax.set_title("Diagnosis-stratified cross-compartment coupling", loc="left", fontweight="bold", pad=26)
    panel_label(ax, "A", x=-0.18, y=1.18)

    ax = fig.add_subplot(gs[0, 3:])
    y_perm = np.arange(len(outcomes))[::-1]
    for y, outcome in zip(y_perm, outcomes):
        null = permutation_null[outcome]
        violin = ax.violinplot(null, positions=[y], vert=False, widths=0.58,
                               showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#C9CDD3")
            body.set_edgecolor("#8B929C")
            body.set_alpha(0.75)
            body.set_linewidth(0.6)
        row = permutation[permutation["outcome"] == outcome].iloc[0]
        ax.hlines(y, row.null_ci_low, row.null_ci_high, color=COL["muted"], lw=2.2, zorder=3)
        ax.scatter(row.observed_rho, y, s=34, color=COL["CD"], edgecolor="white", lw=0.45, zorder=4)
        ax.text(0.98, y, f"Pperm={row.empirical_p:.3g}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.5, color=COL["ink"])
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(y_perm, ["IgA mucosal\nplasma-cell", "Plasma-cell\ndifferentiation"])
    ax.set_xlim(-0.72, 0.76)
    ax.set_xlabel("Partial Spearman ρ")
    ax.text(0.02, 0.02, "Gray: 10,000 within-series re-pairings\nBlue: observed matched participants",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.1, color=COL["muted"])
    panel_title(ax, "Matched-pair permutation validation")
    style_axis(ax, "x")
    panel_label(ax, "B", x=-0.25, y=1.10)

    for j, outcome in enumerate(outcomes):
        ax = fig.add_subplot(gs[1, j * 3:(j + 1) * 3])
        data = residual_data[residual_data["outcome"] == outcome]
        x = data["x_residual"].to_numpy(float)
        y = data["y_residual"].to_numpy(float)
        grid = np.linspace(x.min(), x.max(), 120)
        coef = np.polyfit(x, y, 1)
        rng = np.random.default_rng(20260824 + j)
        predictions = []
        for _ in range(2000):
            idx = rng.integers(0, len(x), len(x))
            boot_coef = np.polyfit(x[idx], y[idx], 1)
            predictions.append(np.polyval(boot_coef, grid))
        predictions = np.asarray(predictions)
        ax.fill_between(grid, np.quantile(predictions, 0.025, axis=0),
                        np.quantile(predictions, 0.975, axis=0), color=COL["CD"], alpha=0.14, lw=0)
        ax.plot(grid, np.polyval(coef, grid), color=COL["CD"], lw=1.35)
        ax.scatter(x, y, s=20, color=COL["CD"], alpha=0.72, edgecolor="white", linewidth=0.35)
        stat = coupling[(coupling["outcome"] == outcome) & (coupling["diagnosis"] == "CD")].iloc[0]
        ax.text(0.03, 0.97, f"ρ={stat.partial_rho:.2f}; adjusted P={stat.p_adj_within_six:.2g}; n={int(stat.n)}",
                transform=ax.transAxes, va="top", fontsize=6.7)
        ax.set_xlabel("Adjusted expanded-cytotoxicity rank residual")
        ax.set_ylabel("Adjusted B-cell-program rank residual")
        panel_title(ax, "CD: " + outcome_labels[outcome])
        style_axis(ax)
        panel_label(ax, "C" if j == 0 else "D", x=-0.17, y=1.12)

    ax = fig.add_subplot(gs[2, :4])
    positions = [11.3, 10.5, 9.7, 8.9, 8.1, 7.3, 5.6, 4.8, 4.0, 3.2, 2.4, 1.6]
    model_colors = {
        "Primary": COL["CD"],
        "+ acquisition series": COL["Joint"],
        "+ inflammation status": "#E69F00",
        "+ biologic exposure": "#CC79A7",
        "+ cell-state composition": "#7A7F87",
        "+ expanded-clone trajectory position": "#7A4EAB",
    }
    row_labels = []
    for y, (_, row) in zip(positions, sensitivity.iterrows()):
        color = model_colors[row.model]
        ax.errorbar(row.rho, y, xerr=[[row.rho - row.ci_low], [row.ci_high - row.rho]],
                    fmt="o", ms=4.3, lw=1.05, capsize=2, color=color, zorder=3)
        label = row.model
        if int(row.n) < 56:
            label += f" (n={int(row.n)})"
        row_labels.append(label)
        if row.model == "Primary":
            ax.text(0.98, y + 0.21,
                    f"leave-one-series-out ρ: {row.loo_series_low:.2f} to {row.loo_series_high:.2f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.2,
                    color=COL["muted"])
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(positions, row_labels)
    ax.set_xlim(-0.12, 0.84)
    ax.set_xlabel("Partial Spearman ρ (95% participant-bootstrap CI)")
    ax.text(0.02, 1.015, "IgA mucosal plasma-cell program", transform=ax.transAxes,
            fontsize=7, fontweight="bold", va="bottom", bbox=header_box)
    ax.text(0.02, 0.50, "Plasma-cell differentiation", transform=ax.transAxes,
            fontsize=7, fontweight="bold", va="center", bbox=header_box)
    style_axis(ax, "x")
    panel_label(ax, "E", x=-0.10, y=1.08)

    ax = fig.add_subplot(gs[2, 4:])
    y_difference = np.arange(len(program_differences))[::-1]
    difference_labels = ["Plasma - IgG", "Plasma - BAFF/APRIL",
                         "IgA - IgG", "IgA - BAFF/APRIL"]
    for y, (_, row) in zip(y_difference, program_differences.iterrows()):
        ax.errorbar(row.rho_difference, y,
                    xerr=[[row.rho_difference - row.ci_low], [row.ci_high - row.rho_difference]],
                    fmt="o", ms=4.4, lw=1.05, capsize=2, color=COL["CD"], zorder=3)
        ax.text(0.59, y, f"FDR={row.FDR_across_four:.2g}",
                ha="right", va="center", fontsize=6.0, color=COL["muted"], bbox=header_box)
    ax.axvline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_yticks(y_difference, difference_labels)
    ax.set_xlim(-0.42, 0.62)
    ax.set_xlabel("Paired difference in ρ (95% CI)")
    panel_title(ax, "Formal between-program contrasts")
    style_axis(ax, "x")
    panel_label(ax, "F", x=-0.28, y=1.08)
    save_figure(fig, "Figure_6_trajectory_adjusted", integrated_main)
    legend = (
        "Figure 6. Diagnosis-specific coupling links cytotoxic T-cell expansion to B-cell activation programs in Crohn's disease. "
        "(A) Diagnosis-stratified partial Spearman correlations between the expanded-TCR cytotoxicity score and the IgA mucosal plasma-cell or plasmablast/plasma-cell differentiation program. "
        "(B) Acquisition-series-stratified matched-participant permutation validation. (C and D) Covariate-adjusted rank-residual plots among participants with Crohn's disease. "
        "(E) Sensitivity analyses adding acquisition series, objective inflammation, biologic exposure, cell-state composition, or the participant-median graph-geodesic position of cells in expanded exact paired alpha-beta clonotypes to the primary age-, sex-, TCR-depth-, and BCR-depth-adjusted model. "
        "(F) Formal participant-paired contrasts between plasma-cell programs and IgG-inflammatory or BAFF/APRIL programs. Cross-sectional pseudotime represents transcriptional ordering and does not establish temporal progression."
    )
    (integrated_leg / "Figure_6_trajectory_adjusted_legend.txt").write_text(legend + "\n", encoding="utf-8")


def build_figure_7():
    nested = read_csv(ROOT / "Recreated Figure 6/Figure_6_panel_B_nested_diagnosis_source_data.csv")
    curves = read_csv(ROOT / "Recreated Figure 6/Figure_6_panel_C_curves_source_data.csv")
    calibration = read_csv(ROOT / "Recreated Figure 6/Figure_6_panel_D_calibration_source_data.csv")
    clinical = read_csv(ROOT / "Recreated Figure 6/Figure_6_panel_E_clinical_state_screen_source_data.csv")
    paired = read_csv(ROOT / "Recreated Figure 6/Figure_6_panel_F_paired_chain_source_data.csv")

    fig = plt.figure(figsize=(7.48, 8.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.9, 1.08, 1.1], hspace=0.62, wspace=0.45,
                  left=0.09, right=0.98, top=0.97, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0])
    workflow_panel(ax)
    panel_title(ax, "Fully nested participant-level validation")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    nn = nested[nested["modality"].isin(["tcr", "bcr"])].copy()
    tasks = ["cd_vs_control", "uc_vs_control", "cd_vs_uc"]
    ybase = np.arange(len(tasks))[::-1]
    for offset, mod in [(-0.10, "tcr"), (0.10, "bcr")]:
        sub = nn[nn["modality"] == mod].set_index("task").reindex(tasks)
        ax.errorbar(sub["roc_auc_median"], ybase + offset,
                    xerr=[sub["roc_auc_median"] - sub["roc_auc_ci_low"], sub["roc_auc_ci_high"] - sub["roc_auc_median"]],
                    fmt="o", color=COL[mod.upper()], ms=4, capsize=2, lw=0.8, label=mod.upper())
    ax.axvline(0.5, color=COL["muted"], lw=0.6, ls="--")
    ax.set_yticks(ybase, [t.replace("_", " ") for t in tasks])
    ax.set_xlabel("Median outer-fold ROC AUC (95% CI)")
    ax.set_xlim(0.45, 1.02)
    ax.legend(frameon=False)
    style_axis(ax, "x")
    panel_title(ax, "Nested diagnosis classification")
    panel_label(ax, "B")

    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.45)
    for j, curve in enumerate(["ROC", "PR"]):
        ax = fig.add_subplot(sub[0, j])
        for mod in ["TCR", "BCR"]:
            cc = curves[(curves["curve"] == curve) & (curves["modality"] == mod)]
            ax.plot(cc["x"], cc["y"], color=COL[mod], lw=1.2, label=mod)
        if curve == "ROC":
            ax.plot([0, 1], [0, 1], color=COL["muted"], ls="--", lw=0.7)
            ax.set_xlabel("False-positive rate")
            ax.set_ylabel("True-positive rate")
        else:
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        style_axis(ax)
        if j == 0:
            panel_label(ax, "C", x=-0.42)
            ax.legend(frameon=False, loc="lower right")
        panel_title(ax, curve)

    ax = fig.add_subplot(gs[1, 1])
    for mod in ["TCR", "BCR"]:
        cc = calibration[(calibration["task"] == "cd_vs_control") & (calibration["modality"] == mod)].sort_values("mean_predicted")
        ax.plot(cc["mean_predicted"], cc["observed_fraction"], "o-", color=COL[mod], lw=0.9, ms=3.5, label=mod)
    ax.plot([0, 1], [0, 1], color=COL["muted"], ls="--", lw=0.7)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    style_axis(ax)
    panel_title(ax, "Calibration: CD versus control")
    panel_label(ax, "D")

    ax = fig.add_subplot(gs[2, 0])
    clinical = clinical.copy()
    clinical["comparison_label"] = clinical["comparison"].str.replace("_", " ")
    order = clinical.groupby("comparison_label")["auc"].max().sort_values().index
    ymap = {v: i for i, v in enumerate(order)}
    offsets = {"TCR": -0.12, "BCR": 0.12, "TCR+BCR": 0}
    markers = {"TCR": "o", "BCR": "s", "TCR+BCR": "D"}
    for mod in ["TCR", "BCR", "TCR+BCR"]:
        cc = clinical[clinical["receptor_model"] == mod]
        ax.scatter(cc["auc"], [ymap[x] + offsets[mod] for x in cc["comparison_label"]], s=22, marker=markers[mod],
                   facecolor="white", edgecolor=COL.get(mod, COL["Joint"]), lw=1, label=mod)
    ax.axvline(0.5, color=COL["muted"], lw=0.6, ls="--")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Exploratory cross-validated ROC AUC")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, "x")
    panel_title(ax, "Clinical-state and six-month response-status screens")
    panel_label(ax, "E")

    ax = fig.add_subplot(gs[2, 1])
    tasks = list(dict.fromkeys(paired["task"]))
    for i, task in enumerate(tasks):
        cc = paired[paired["task"] == task]
        for j, rep in enumerate(["Single-chain", "Paired interaction"]):
            row = cc[cc["representation"] == rep]
            if row.empty:
                continue
            r = row.iloc[0]
            color = COL["muted"] if rep == "Single-chain" else COL["Joint"]
            y = len(tasks) - 1 - i + (j - 0.5) * 0.16
            ax.plot([r.ci_low, r.ci_high], [y, y], color=color, lw=0.9)
            ax.scatter(r.auc, y, color=color, s=20, marker="o" if j == 0 else "D")
    ax.axvline(0.5, color=COL["muted"], lw=0.6, ls="--")
    ax.set_yticks(range(len(tasks))[::-1], tasks)
    ax.set_xlabel("Pooled outer-fold ROC AUC (95% CI)")
    style_axis(ax, "x")
    panel_title(ax, "Incremental information from paired chains")
    panel_label(ax, "F")
    save_figure(fig, "Figure_7", MAIN)


def milo_summary(ax, df, title, letter):
    df = df.copy()
    df["significant"] = pd.to_numeric(df["SpatialFDR"], errors="coerce") < 0.05
    top = df.groupby("Celltype").agg(mean_logFC=("logFC", "mean"), sig_n=("significant", "sum"), n=("Nhood", "count"))
    top = top.sort_values("mean_logFC").iloc[np.linspace(0, max(0, len(top) - 1), min(18, len(top))).astype(int)]
    y = np.arange(len(top))
    colors = np.where(top["mean_logFC"] >= 0, COL["positive"], COL["negative"])
    ax.hlines(y, 0, top["mean_logFC"], color=colors, lw=1)
    ax.scatter(top["mean_logFC"], y, s=10 + 3 * np.sqrt(top["sig_n"]), c=colors, edgecolor="white", lw=0.35)
    ax.axvline(0, color=COL["muted"], lw=0.6)
    ax.set_yticks(y, top.index)
    ax.set_xlabel("Mean neighborhood log2 fold change")
    style_axis(ax, "x")
    panel_title(ax, title)
    panel_label(ax, letter)


def build_figure_s1():
    pbmc = read_csv(SRC / "UMAP_PBMC_from_Seurat.csv")
    milo_cd = read_csv(PBMC_DIR / "Supplementary Data/Supplementary Table 1 CDControlSCVI_milo_da_results.csv")
    milo_uc = read_csv(PBMC_DIR / "Supplementary Data/Supplementary Table 2 UCControlSCVI_milo_da_results.csv")
    fig = plt.figure(figsize=(7.48, 8.7))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.12], hspace=0.52, wspace=0.42,
                  left=0.08, right=0.98, top=0.97, bottom=0.06)
    ax = fig.add_subplot(gs[0, :])
    umap_panel(ax, pbmc, "PBMC atlas", max_labels=0)
    panel_label(ax, "A", x=-0.07)
    for j, d in enumerate(DIAG):
        ax = fig.add_subplot(gs[1, j])
        umap_panel(ax, pbmc[pbmc["Diagnosis1"] == d], d, max_labels=0)
        panel_label(ax, chr(ord("B") + j))
    ax = fig.add_subplot(gs[2, :2])
    milo_summary(ax, milo_cd, "Milo differential abundance: CD versus control", "E")
    ax = fig.add_subplot(gs[2, 2])
    milo_summary(ax, milo_uc, "UC versus control", "F")
    save_figure(fig, "Figure_S1", SUPP)


def build_figure_s2():
    metrics = read_csv(TCR_DIR / "clonality_metrics_IBDTCR_Immunarch.csv")
    threshold = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_expansion_threshold_metrics_by_participant.csv")
    threshold = threshold[threshold["modality"] == "TCR"]
    vj = read_csv(TCR_DIR / "vj_usage_IBDTCR_sample_level.csv")
    state = read_csv(TCR_DIR / "celltype_distribution_expanded_clones_IBDTCR_table.csv")
    fig = plt.figure(figsize=(7.48, 9.1))
    gs = GridSpec(3, 2, figure=fig, hspace=0.62, wspace=0.48, left=0.09, right=0.98, top=0.97, bottom=0.06)
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.42)
    for j, (m, lab) in enumerate([("Shannon", "Shannon diversity"), ("InvSimp", "Inverse Simpson"), ("D50_Clones", "D50 clones")]):
        ax = fig.add_subplot(sub[0, j]); boxstrip(ax, metrics, m, ylabel=lab); ax.tick_params(axis="x", rotation=30)
        panel_label(ax, chr(ord("A") + j), x=-0.35)
    ax = fig.add_subplot(gs[1, 0])
    for d in DIAG:
        ss = threshold[threshold["Diagnosis1"] == d].groupby("threshold")["expanded_cell_fraction"].median()
        ax.plot(ss.index, ss.values, "o-", color=COL[d], lw=0.9, ms=3.5, label=d)
    ax.set_xlabel("Expansion threshold (cells)"); ax.set_ylabel("Median expanded-cell fraction"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False); style_axis(ax, "y"); panel_title(ax, "Expansion-threshold sensitivity"); panel_label(ax, "D")
    ax = fig.add_subplot(gs[1, 1])
    vv = vj[vj["Family"].isin(["TRBV", "TRBJ"])].copy()
    top = vv.groupby("Gene")["Proportion"].mean().nlargest(18).index
    mat = vv[vv["Gene"].isin(top)].pivot_table(index="Gene", columns="Diagnosis", values="Proportion", aggfunc="median").reindex(columns=DIAG)
    heatmap(ax, mat, cmap="YlGnBu", center=None, cbar_label="Median clone proportion")
    panel_title(ax, "TCR beta-chain V/J usage"); panel_label(ax, "E")
    ax = fig.add_subplot(gs[2, :])
    top_states = state.groupby("cell_type")["expanded_cells"].sum().nlargest(18).index
    mat = state[state["cell_type"].isin(top_states)].pivot_table(index="cell_type", columns=["lineage", "diagnosis"], values="prop_within_expanded", aggfunc="mean").fillna(0)
    heatmap(ax, mat, cmap="YlGnBu", center=None, cbar_label="Fraction of expanded cells")
    panel_title(ax, "Expanded CD4 and CD8 clonotype state occupancy"); panel_label(ax, "F", x=-0.07)
    save_figure(fig, "Figure_S2", SUPP)


def build_figure_s3():
    models = read_csv(ROOT / "High Impact Additional Analyses/Clone State Interactions/Table_CSI3_formal_interaction_models.csv")
    meta = read_csv(ROOT / "High Impact Additional Analyses/Clone State Interactions/Table_CSI5_random_effects_meta_analysis.csv")
    series = read_csv(ROOT / "High Impact Additional Analyses/Clone State Interactions/Table_CSI4_acquisition_series_effects.csv")
    fig = plt.figure(figsize=(7.48, 8.3))
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.55, left=0.12, right=0.98, top=0.97, bottom=0.07)
    for j, mod in enumerate(["TCR", "BCR"]):
        ax = fig.add_subplot(gs[0, j])
        mm = models[models["family"].str.startswith(mod)].copy()
        mm = mm[mm["estimable"].astype(str).str.lower() == "true"]
        mm = mm.reindex(mm["effect"].abs().sort_values(ascending=False).head(14).index)
        lo, hi = mm["effect"] - 1.96 * mm["SE"], mm["effect"] + 1.96 * mm["SE"]
        labels = [f"{clean_label(m)} | {clean_label(c)}" for m, c in zip(mm["module"], mm["contrast"])]
        forest(ax, labels, mm["effect"], lo, hi, xlabel="Interaction effect (95% CI)")
        panel_title(ax, f"{mod} expansion-by-disease interactions"); panel_label(ax, "A" if j == 0 else "B")
    ax = fig.add_subplot(gs[1, 0])
    mm = meta.copy(); mm["label"] = mm["Diagnosis1"] + " / " + mm["Inflammation1"] + " / " + mm["module"].map(clean_label)
    mm = mm.reindex(mm["effect"].abs().sort_values(ascending=False).head(16).index)
    forest(ax, mm["label"], mm["effect"], mm["effect"] - 1.96 * mm["se"], mm["effect"] + 1.96 * mm["se"], xlabel="Random-effects estimate")
    panel_title(ax, "Acquisition-series meta-analysis"); panel_label(ax, "C")
    ax = fig.add_subplot(gs[1, 1])
    ss = series.groupby(["acquisition_series", "modality"])["effect"].median().unstack()
    for mod in ss.columns:
        ax.plot(range(len(ss)), ss[mod], "o-", color=COL[mod], lw=0.9, ms=3.5, label=mod)
    ax.axhline(0, color=COL["muted"], lw=0.6)
    ax.set_xticks(range(len(ss)), ss.index, rotation=45); ax.set_ylabel("Median series-specific effect")
    ax.legend(frameon=False); style_axis(ax, "y"); panel_title(ax, "Replication across acquisition series"); panel_label(ax, "D")
    save_figure(fig, "Figure_S3", SUPP)


def build_figure_s4():
    perm = read_csv(ROOT / "High Impact Additional Analyses/Paired TCR Sequence State/Table_TSS4_sequence_state_permutation_tests_by_series.csv")
    meta = read_csv(ROOT / "High Impact Additional Analyses/Paired TCR Sequence State/Table_TSS5_sequence_state_replication_meta_analysis.csv")
    fig = plt.figure(figsize=(7.48, 8.35))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.82, 1.05, 0.88],
        hspace=0.68,
        wspace=0.72,
        left=0.10,
        right=0.98,
        top=0.97,
        bottom=0.07,
    )
    module_order = [
        "EOMES_ZEB2_inflammatory_CD8_TRM_like",
        "Effector_cytotoxicity",
        "Th1_Tc1_inflammatory",
        "Tissue_resident_mucosal_retention",
        "Gut_homing_intestinal_trafficking",
    ]
    primary = module_order[:3]
    series_order = sorted(perm.acquisition_series.unique(), key=lambda value: int(str(value).replace("S", "")))

    ax = fig.add_subplot(gs[0, 0])
    dom = perm[(perm["module"] == "Dominant_state_concordance") & (perm["stratum"] == "All")].copy()
    dom["series_order"] = dom["acquisition_series"].str.replace("S", "", regex=False).astype(int)
    dom = dom.sort_values("series_order")
    delta = dom["observed_edge_correlation"] - dom["null_mean"]
    err = 1.96 * dom["null_sd"]
    x = np.arange(len(dom))
    ax.errorbar(x, delta, yerr=err, fmt="o", ms=3.8, color=COL["ink"], ecolor="#A7A7A7",
                elinewidth=0.9, capsize=1.8)
    ax.axhline(0, color=COL["muted"], lw=0.7, ls="--")
    ax.set_xticks(x, dom["acquisition_series"], rotation=45, ha="right")
    ax.set_ylabel("Observed - permutation mean")
    ax.set_xlabel("Acquisition series")
    ax.text(0.98, 0.96, "Meta-analysis FDR > 0.05", transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
    style_axis(ax, "y"); panel_title(ax, "Exact state labels do not replicate"); panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    mm = meta.reindex(meta["meta_z"].abs().sort_values(ascending=False).head(12).index).sort_values("meta_z")
    ax.barh(range(len(mm)), mm["meta_z"], color=np.where(mm["meta_FDR"] < 0.05, COL["TCR"], "#B8B8B8"))
    ax.set_yticks(range(len(mm)), [f"{concise_module_label(module)} ({stratum})" for module, stratum in zip(mm["module"], mm["stratum"])]); ax.axvline(0, color=COL["ink"], lw=0.6)
    ax.tick_params(axis="y", labelsize=6.0)
    ax.set_xlabel("Weighted Stouffer z"); style_axis(ax, "x"); panel_title(ax, "Meta-analysis and replication"); panel_label(ax, "B")

    # C: the acquisition-series heatmap moved from the main figure.
    ax = fig.add_subplot(gs[1, :])
    all_perm = perm[perm.stratum.eq("All") & perm.module.isin(module_order)]
    matrix = all_perm.pivot(index="module", columns="acquisition_series", values="observed_edge_correlation").reindex(
        index=module_order, columns=series_order
    )
    heatmap(
        ax,
        matrix.rename(index={module: concise_module_label(module) for module in module_order}),
        cbar_label="Edge correlation",
        vlim=0.18,
    )
    qmatrix = all_perm.pivot(index="module", columns="acquisition_series", values="FDR_within_series_stratum").reindex(
        index=module_order, columns=series_order
    )
    for row in range(qmatrix.shape[0]):
        for column in range(qmatrix.shape[1]):
            if pd.notna(qmatrix.iloc[row, column]) and qmatrix.iloc[row, column] < 0.05:
                ax.text(
                    column,
                    row,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white" if abs(matrix.iloc[row, column]) > 0.10 else COL["ink"],
                )
    ax.tick_params(axis="y", labelsize=6.0)
    ax.text(1.0, -0.20, "* within-series FDR < 0.05", transform=ax.transAxes, ha="right", fontsize=5.7)
    panel_title(ax, "Complete program concordance across acquisition series")
    panel_label(ax, "C", x=-0.11)

    # D-E: diagnosis-restricted replication supporting the main Figure 3 forest plot.
    for column, diagnosis in enumerate(("CD", "UC")):
        ax = fig.add_subplot(gs[2, column])
        subset = perm[perm.stratum.eq(diagnosis) & perm.module.isin(primary)]
        available_series = [series for series in series_order if series in set(subset.acquisition_series)]
        matrix = subset.pivot(index="module", columns="acquisition_series", values="observed_edge_correlation").reindex(
            index=primary, columns=available_series
        )
        heatmap(
            ax,
            matrix.rename(index={module: concise_module_label(module) for module in primary}),
            cbar_label="Edge correlation",
            vlim=0.40,
        )
        qmatrix = subset.pivot(index="module", columns="acquisition_series", values="FDR_within_series_stratum").reindex(
            index=primary, columns=available_series
        )
        for row in range(qmatrix.shape[0]):
            for series_column in range(qmatrix.shape[1]):
                if pd.notna(qmatrix.iloc[row, series_column]) and qmatrix.iloc[row, series_column] < 0.05:
                    ax.text(
                        series_column,
                        row,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color="white" if abs(matrix.iloc[row, series_column]) > 0.20 else COL["ink"],
                    )
        ax.tick_params(axis="y", labelsize=5.8)
        panel_title(ax, f"{diagnosis} diagnosis-restricted replication")
        panel_label(ax, "D" if diagnosis == "CD" else "E", x=-0.22)
    save_figure(fig, "Figure_S4", SUPP)


def build_figure_s5():
    metrics = read_csv(BCR_DIR / "clonality_metrics_IBDBCR_Immunarch.csv").rename(columns={"Diagnosis1": "Diagnosis"})
    iso = read_csv(BCR_DIR / "ig_isotype_proportions_control_uc_cd_values.csv")
    shm = read_csv(BCR_DIR / "bcr_shm_immunarch_sample_metrics.csv")
    vj = read_csv(BCR_DIR / "bcr_top_gene_usage_IBDBCR_per_sample_values.csv")
    imm = read_csv(PBMC_DIR / "Figure 5/paired_bcr_immunomatch_nominal_p_lt_0p05_pairs.csv")
    burden = read_csv(PBMC_DIR / "Figure 5/immunomatch_p_lt_0p05_sample_burden_by_calprotectin.csv")
    fig = plt.figure(figsize=(7.48, 9.0))
    gs = GridSpec(3, 2, figure=fig, hspace=0.62, wspace=0.72, left=0.10, right=0.98, top=0.97, bottom=0.06)
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.42)
    for j, (m, label) in enumerate([("Clonality", "Clonality"), ("Gini", "Gini coefficient"), ("D50_Clones", "D50 clones")]):
        ax = fig.add_subplot(sub[0, j]); boxstrip(ax, metrics, m, ylabel=label); ax.tick_params(axis="x", rotation=30)
        panel_label(ax, chr(ord("A") + j), x=-0.35)
    ax = fig.add_subplot(gs[1, 0])
    summary = iso.groupby(["Diagnosis", "Isotype"])["Percent"].median().unstack(fill_value=0).reindex(DIAG)
    bottom = np.zeros(len(summary)); colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(summary.columns)))
    for c, color in zip(summary.columns, colors):
        ax.bar(range(len(summary)), summary[c], bottom=bottom, color=color, width=0.65, label=c); bottom += summary[c].to_numpy()
    ax.set_xticks(range(3), DIAG); ax.set_ylabel("Median isotype composition (%)"); ax.legend(frameon=False, ncol=3, fontsize=6)
    style_axis(ax, "y"); panel_title(ax, "Heavy-chain isotype composition"); panel_label(ax, "D")
    ax = fig.add_subplot(gs[1, 1]); boxstrip(ax, shm, "SHM_Frequency", ylabel="Somatic hypermutation frequency (%)")
    ax.tick_params(axis="x", rotation=25); panel_title(ax, "Participant-level SHM"); panel_label(ax, "E")
    ax = fig.add_subplot(gs[2, 0])
    top = vj.groupby(["Family", "Gene"])["Usage"].mean().nlargest(18).reset_index()[["Family", "Gene"]]
    vv = vj.merge(top, on=["Family", "Gene"])
    vv["label"] = vv["Family"].astype(str) + vv["Gene"].astype(str)
    mat = vv.pivot_table(index="label", columns="Diagnosis", values="Usage", aggfunc="median").reindex(columns=DIAG)
    heatmap(ax, mat, cmap="YlGnBu", center=None, cbar_label=""); panel_title(ax, "BCR V/J gene usage"); panel_label(ax, "F")
    ax = fig.add_subplot(gs[2, 1])
    top = imm.nsmallest(15, "p_value").sort_values("log2_enrichment")
    ax.barh(range(len(top)), top["log2_enrichment"], color=np.where(top["log2_enrichment"] >= 0, COL["positive"], COL["negative"]))
    ax.set_yticks(range(len(top)), [textwrap.shorten(x, width=28, placeholder="...") for x in top["pair_label"]])
    ax.axvline(0, color=COL["ink"], lw=0.6); ax.set_xlabel("log2 enrichment"); style_axis(ax, "x")
    panel_title(ax, "Nominal ImmunoMatch paired-BCR enrichments"); panel_label(ax, "G")
    save_figure(fig, "Figure_S5", SUPP)


def build_figure_s6():
    lineages = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL11_participant_lineage_metrics.csv")
    comparisons = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL12_diagnosis_comparisons.csv")
    threshold_effects = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL16_threshold_disease_effects.csv")
    edge_free = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL22_participant_germline_edge_free_divergence.csv")
    edge_free_effects = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL23_germline_edge_free_divergence_effects.csv")
    iso = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL5_lineage_isotype_composition.csv")
    states = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL6_lineage_transcriptional_state_composition.csv")
    lineage_metrics = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL4_germline_aware_lineage_metrics.csv")
    lineage_meta = lineage_metrics[["lineage_id", "Diagnosis"]].drop_duplicates()
    fig = plt.figure(figsize=(7.48, 8.45))
    gs = GridSpec(3, 2, figure=fig, hspace=0.64, wspace=0.52, left=0.10, right=0.98, top=0.97, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0])
    boxstrip(ax, lineages, "mean_MST_branch_length", ylabel="Mean germline-inclusive MST branch length")
    lineage_fdr = comparisons[
        (comparisons["analysis"] == "lineage")
        & (comparisons["metric"] == "mean_MST_branch_length")
        & comparisons["contrast"].isin(["CD vs Control", "UC vs Control"])
    ].set_index("contrast")["FDR"]
    ax.text(0.98, 0.98,
            f"CD vs Control FDR={lineage_fdr['CD vs Control']:.2g}\nUC vs Control FDR={lineage_fdr['UC vs Control']:.2g}",
            transform=ax.transAxes, va="top", ha="right", fontsize=5.8, color=COL["muted"])
    panel_title(ax, "Germline-inclusive branch length"); panel_label(ax, "A", x=-0.20, y=1.18)

    ax = fig.add_subplot(gs[0, 1])
    for disease, marker, xshift in [("CD", "o", -0.22), ("UC", "s", 0.22)]:
        dd = threshold_effects[threshold_effects["contrast"] == f"{disease} vs Control"].sort_values("threshold")
        ax.errorbar(dd["threshold"] * 100 + xshift, dd["median_difference"],
                    yerr=[dd["median_difference"] - dd["ci_low"], dd["ci_high"] - dd["median_difference"]],
                    fmt=marker + "-", color=COL[disease], lw=1.0, ms=4.2, capsize=2.2,
                    label=f"{disease} - Control")
    ax.axhline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_xticks([10, 15, 20], ["10%", "15%", "20%"])
    ax.set_xlabel("Normalized CDR3 distance threshold")
    ax.set_ylabel("Median branch-length difference\n(95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=6)
    ax.text(0.98, 0.98, f"Rank-test FDR={threshold_effects['FDR'].max():.2g}", transform=ax.transAxes,
            va="top", ha="right", fontsize=5.8, color=COL["muted"])
    style_axis(ax, "y")
    panel_title(ax, "Clustering-threshold sensitivity"); panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    boxstrip(ax, edge_free, "mean_within_lineage_divergence", ylabel="Mean observed-sequence divergence")
    ax.text(0.98, 0.98,
            "Germline edge excluded\nCD FDR=0.45; UC FDR=0.45",
            transform=ax.transAxes, va="top", ha="right", fontsize=5.8, color=COL["muted"])
    panel_title(ax, "Within-lineage diversification is not increased"); panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    occupancy = lineage_metrics.groupby("Diagnosis").agg(
        total=("lineage_id", "nunique"),
        cross_state=("n_cell_states", lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0) > 1).sum())),
        mixed=("mixed_unswitched_switched", lambda x: int(pd.Series(x).astype(str).str.lower().eq("true").sum())),
    ).reindex(DIAG)
    occupancy["cross_state_rate"] = 1000 * occupancy["cross_state"] / occupancy["total"]
    occupancy["mixed_rate"] = 1000 * occupancy["mixed"] / occupancy["total"]
    xpos = np.arange(2)
    for k, diagnosis in enumerate(DIAG):
        vals = occupancy.loc[diagnosis, ["cross_state_rate", "mixed_rate"]].to_numpy(float)
        xx = xpos + (k - 1) * 0.18
        ax.scatter(xx, vals, s=22, color=COL[diagnosis], edgecolor="white", lw=0.4, label=diagnosis, zorder=3)
        counts = occupancy.loc[diagnosis, ["cross_state", "mixed"]].astype(int).to_numpy()
        total = int(occupancy.loc[diagnosis, "total"])
        for metric_index, (px, val, count) in enumerate(zip(xx, vals, counts)):
            multiplier = 0.78 if metric_index == 0 else 1.45
            va = "top" if metric_index == 0 else "bottom"
            ax.text(px, val * multiplier, f"{count}/{total:,}", ha="center", va=va, fontsize=5.1,
                    color=COL[diagnosis])
    ax.set_yscale("log")
    ax.set_ylim(0.025, 14)
    ax.set_xticks(xpos, [">1 transcriptional\nstate", "Mixed unswitched/\nswitched isotype"])
    ax.set_ylabel("Lineages per 1,000 (log scale)")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    style_axis(ax, "y")
    panel_title(ax, "Cross-state and mixed-isotype occupancy is rare"); panel_label(ax, "D")

    ax = fig.add_subplot(gs[2, 0])
    iso_m = iso.merge(lineage_meta, on="lineage_id", how="left")
    iso_summary = iso_m.groupby(["Diagnosis", "isotype_class"])["n_cells"].sum().reset_index()
    iso_summary["fraction"] = iso_summary["n_cells"] / iso_summary.groupby("Diagnosis")["n_cells"].transform("sum")
    mat = iso_summary.pivot_table(index="isotype_class", columns="Diagnosis", values="fraction", aggfunc="sum").reindex(columns=DIAG)
    heatmap(ax, mat, cmap="YlGnBu", center=None, cbar_label="Lineage fraction"); panel_title(ax, "Lineage isotype composition"); panel_label(ax, "E")
    ax = fig.add_subplot(gs[2, 1])
    state_m = states.merge(lineage_meta, on="lineage_id", how="left")
    top = state_m.groupby("cell_state")["n_cells"].sum().nlargest(12).index
    state_summary = state_m[state_m["cell_state"].isin(top)].groupby(["Diagnosis", "cell_state"])["n_cells"].sum().reset_index()
    state_summary["fraction"] = state_summary["n_cells"] / state_summary.groupby("Diagnosis")["n_cells"].transform("sum")
    mat = state_summary.pivot_table(index="cell_state", columns="Diagnosis", values="fraction", aggfunc="sum").reindex(columns=DIAG)
    heatmap(ax, mat, cmap="YlGnBu", center=None, cbar_label="Cell fraction")
    panel_title(ax, "Lineage transcriptional-state composition"); panel_label(ax, "F")
    save_figure(fig, "Figure_S6", SUPP)


def build_figure_s12():
    inflammation = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL39_objective_inflammation_lineage_effects.csv")
    calprotectin = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL40_calprotectin_lineage_partial_correlations.csv")
    mucosal = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL42_expanded_lineage_mucosal_state_effects.csv")
    light = read_csv(ROOT / "High Impact Additional Analyses/BCR Germline Lineages/Table_BGL44_paired_light_feature_effects.csv")
    inflammation.to_csv(SRC / "Figure_S12_objective_inflammation_lineage_effects.csv", index=False)
    calprotectin.to_csv(SRC / "Figure_S12_calprotectin_lineage_partial_correlations.csv", index=False)
    mucosal.to_csv(SRC / "Figure_S12_expanded_lineage_mucosal_state_effects.csv", index=False)
    light.to_csv(SRC / "Figure_S12_paired_light_feature_effects.csv", index=False)

    fig = plt.figure(figsize=(7.48, 7.85))
    gs = GridSpec(2, 2, figure=fig, hspace=0.58, wspace=0.62,
                  left=0.14, right=0.98, top=0.96, bottom=0.09)
    offsets = {"CD": 0.12, "UC": -0.12}

    metric_order = [
        "expanded_lineages", "mean_MST_branch_length", "mean_within_lineage_divergence",
        "CDR_minus_FWR", "exact_paired_HL_shm", "fraction_class_switched_lineages",
    ]
    metric_labels = [
        "Expanded lineage\nburden", "Germline-inclusive\nbranch length", "Observed-only\ndivergence",
        "CDR targeting", "Paired H-L\nheavy-chain SHM", "Class-switched\nlineage fraction",
    ]

    ax = fig.add_subplot(gs[0, 0])
    ybase = np.arange(len(metric_order))[::-1]
    for diagnosis in ["CD", "UC"]:
        data = inflammation[inflammation["Diagnosis"].eq(diagnosis)].set_index("metric").reindex(metric_order)
        yy = ybase + offsets[diagnosis]
        ax.hlines(yy, data["ci_low"], data["ci_high"], color=COL[diagnosis], lw=0.95)
        for yvalue, (_, row) in zip(yy, data.iterrows()):
            significant = row["FDR"] < 0.05
            ax.scatter(row["adjusted_median_difference"], yvalue, s=22,
                       facecolor=COL[diagnosis] if significant else "white", edgecolor=COL[diagnosis], lw=1.0, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(ybase, metric_labels)
    ax.set_xlabel("Inflamed - noninflamed adjusted rank residual\nmedian difference (95% CI)")
    ax.text(0.98, 0.98, f"All FDR >= {inflammation['FDR'].min():.2g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=COL["muted"])
    handles = [Line2D([0], [0], marker="o", ls="", ms=4.5, markerfacecolor="white",
                      markeredgecolor=COL[d], label=d) for d in ["CD", "UC"]]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    style_axis(ax, "x")
    panel_title(ax, "Objective inflammation within IBD")
    panel_label(ax, "A", x=-0.36)

    ax = fig.add_subplot(gs[0, 1])
    for diagnosis in ["CD", "UC"]:
        data = calprotectin[calprotectin["Diagnosis"].eq(diagnosis)].set_index("metric").reindex(metric_order)
        yy = ybase + offsets[diagnosis]
        ax.hlines(yy, data["ci_low"], data["ci_high"], color=COL[diagnosis], lw=0.95)
        for yvalue, (_, row) in zip(yy, data.iterrows()):
            significant = row["FDR"] < 0.05
            ax.scatter(row["partial_spearman_rho"], yvalue, s=22,
                       facecolor=COL[diagnosis] if significant else "white", edgecolor=COL[diagnosis], lw=1.0, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(ybase, metric_labels)
    ax.set_xlabel("Partial Spearman rho with log calprotectin\n(95% bootstrap CI)")
    uc_cdr = calprotectin[(calprotectin["Diagnosis"].eq("UC")) & (calprotectin["metric"].eq("CDR_minus_FWR"))].iloc[0]
    ax.text(0.98, 0.98, f"UC CDR targeting: FDR={uc_cdr['FDR']:.2g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=COL["UC"])
    style_axis(ax, "x")
    panel_title(ax, "Continuous intestinal inflammatory burden")
    panel_label(ax, "B", x=-0.36)

    ax = fig.add_subplot(gs[1, 0])
    state_order = ["IgA Plasma B Cell", "IgG Plasma B Cell", "Switched memory B", "Atypical memory B"]
    state_labels = ["IgA plasma B", "IgG plasma B", "Switched memory B", "Atypical memory B"]
    ystate = np.arange(len(state_order))[::-1]
    for disease in ["CD", "UC"]:
        data = mucosal[mucosal["contrast"].eq(f"{disease} vs Control")].set_index("cell_state").reindex(state_order)
        yy = ystate + offsets[disease]
        ax.hlines(yy, data["ci_low"], data["ci_high"], color=COL[disease], lw=0.95)
        for yvalue, (_, row) in zip(yy, data.iterrows()):
            significant = row["FDR"] < 0.05
            ax.scatter(row["median_fraction_difference"], yvalue, s=22,
                       facecolor=COL[disease] if significant else "white", edgecolor=COL[disease], lw=1.0, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(ystate, state_labels)
    ax.set_xlabel("Median expanded-lineage cell-fraction\ndifference (95% CI)")
    ax.text(0.98, 0.98, "IgA plasma: CD and UC FDR < 0.025", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=COL["muted"])
    style_axis(ax, "x")
    panel_title(ax, "Expanded-lineage mucosal state context")
    panel_label(ax, "C", x=-0.36)

    ax = fig.add_subplot(gs[1, 1])
    feature_order = ["IGL fraction", "IGKV1", "IGKV3", "IGLV2", "IGLV3", "IGLV1"]
    feature_labels = ["IGL locus fraction", "IGKV1", "IGKV3", "IGLV2", "IGLV3", "IGLV1"]
    ylight = np.arange(len(feature_order))[::-1]
    for disease in ["CD", "UC"]:
        data = light[light["contrast"].eq(f"{disease} vs Control")].set_index("feature").reindex(feature_order)
        yy = ylight + offsets[disease]
        ax.hlines(yy, data["ci_low"], data["ci_high"], color=COL[disease], lw=0.95)
        for yvalue, (_, row) in zip(yy, data.iterrows()):
            significant = row["FDR"] < 0.05
            ax.scatter(row["median_fraction_difference"], yvalue, s=22,
                       facecolor=COL[disease] if significant else "white", edgecolor=COL[disease], lw=1.0, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.65, ls="--")
    ax.set_yticks(ylight, feature_labels)
    ax.set_xlabel("Median paired-light fraction difference\n(95% CI)")
    uc_igkv1 = light[(light["contrast"].eq("UC vs Control")) & (light["feature"].eq("IGKV1"))].iloc[0]
    ax.text(0.98, 0.98, f"UC IGKV1: FDR={uc_igkv1['FDR']:.2g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=COL["UC"])
    style_axis(ax, "x")
    panel_title(ax, "Paired light-chain context")
    panel_label(ax, "D", x=-0.36)
    save_figure(fig, "Figure_S12", SUPP)


def build_figure_s7():
    coord = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_partial_spearman_TCR_BCR_coordination.csv")
    features = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_integrated_participant_features.csv")
    coupling = read_csv(ROOT / "High Impact Additional Analyses/Table_HI_CD_specific_T_B_module_coupling.csv")
    fig = plt.figure(figsize=(7.48, 8.4))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.15, 1.0, 1.05], hspace=0.62, wspace=0.55,
                  left=0.09, right=0.98, top=0.97, bottom=0.07)
    ax = fig.add_subplot(gs[0, :]); mat = coord.pivot_table(index="tcr_metric", columns="bcr_metric", values="partial_rho")
    heatmap(ax, mat, cbar_label="Partial Spearman rho", annotate=True, vlim=1); panel_title(ax, "Complete TCR-BCR coordination matrix"); panel_label(ax, "A", x=-0.07)
    top = coord.nsmallest(3, "p_adj")
    for j, (_, row) in enumerate(top.iterrows()):
        ax = fig.add_subplot(gs[1, j]); scatter_fit(ax, features[row.tcr_metric], features[row.bcr_metric], COL["Joint"], clean_label(row.tcr_metric), clean_label(row.bcr_metric),
                                                    annotation=f"rho={row.partial_rho:.2f}; FDR={row.p_adj:.2g}; n={int(row.n)}")
        panel_label(ax, chr(ord("B") + j))
    outcomes = coupling["outcome_label"].drop_duplicates().tolist()[:3]
    for j, outcome in enumerate(outcomes):
        ax = fig.add_subplot(gs[2, j]); cc = coupling[coupling["outcome_label"] == outcome].set_index("diagnosis").reindex(DIAG).reset_index()
        forest(ax, cc["diagnosis"], cc["partial_rho"], cc["ci_low"], cc["ci_high"], xlabel="Partial rho", colors=[COL[d] for d in DIAG])
        panel_title(ax, clean_label(outcome)); panel_label(ax, chr(ord("E") + j))
    save_figure(fig, "Figure_S7", SUPP)


def build_figure_s8():
    val = read_csv(ROOT / "ML Sensitivity Validation/Table_S_primary_model_validation_summary.csv")
    null = read_csv(ROOT / "ML Sensitivity Validation/Table_S_permutation_null_distributions.csv")
    perm = read_csv(ROOT / "ML Sensitivity Validation/Table_S_permutation_test_summary.csv")
    optimism = read_csv(ROOT / "ML Sensitivity Validation/Table_S_model_selection_optimism.csv")
    pred = read_csv(ROOT / "ML Sensitivity Validation/Table_S_nested_outer_fold_predictions.csv")
    val = val[val["modality"].isin(["tcr", "bcr"])]
    fig = plt.figure(figsize=(7.48, 7.1))
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.45, left=0.10, right=0.98, top=0.97, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    labels = val["modality"].str.upper() + ": " + val["task"].str.replace("_", " ")
    forest(ax, labels, val["roc_auc_median"], val["roc_auc_ci_low"], val["roc_auc_ci_high"], xlabel="Median outer-fold ROC AUC")
    ax.axvline(0.5, color=COL["muted"], lw=0.6, ls="--"); ax.set_xlim(0.45, 1.02); panel_title(ax, "Fully nested validation"); panel_label(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    groups = list(dict.fromkeys(zip(null["modality"], null["task"])))
    data = [null[(null["modality"] == m) & (null["task"] == t)]["median_outer_auc"] for m, t in groups]
    vp = ax.violinplot(data, positions=range(len(groups)), showextrema=False, widths=0.75)
    for body in vp["bodies"]:
        body.set_facecolor("#C8C8C8"); body.set_edgecolor(COL["muted"]); body.set_alpha(0.8)
    obs = perm.set_index(["modality", "task"])["observed_median_outer_auc"]
    ax.scatter(range(len(groups)), [obs.get(g, np.nan) for g in groups], color=[COL.get(m.upper(), COL["ink"]) for m, _ in groups], s=20, zorder=3)
    ax.axhline(0.5, color=COL["muted"], lw=0.6, ls="--"); ax.set_xticks(range(len(groups)), [f"{m.upper()}\n{t.replace('_',' ')}" for m, t in groups], rotation=45, ha="right")
    ax.set_ylabel("ROC AUC"); style_axis(ax, "y"); panel_title(ax, "Permutation-derived null performance"); panel_label(ax, "B")
    ax = fig.add_subplot(gs[1, 0])
    pp = pred[(pred["task"] == "cd_vs_control") & (pred["modality"].isin(["tcr", "bcr"]))].copy()
    pp["bin"] = pp.groupby("modality")["probability"].transform(lambda x: pd.qcut(x.rank(method="first"), min(6, len(x)), labels=False, duplicates="drop"))
    cal = pp.groupby(["modality", "bin"]).agg(mean_pred=("probability", "mean"), obs=("truth", "mean")).reset_index()
    for mod in ["tcr", "bcr"]:
        cc = cal[cal["modality"] == mod]; ax.plot(cc["mean_pred"], cc["obs"], "o-", color=COL[mod.upper()], lw=0.9, ms=3.5, label=mod.upper())
    ax.plot([0, 1], [0, 1], color=COL["muted"], ls="--", lw=0.7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction"); ax.legend(frameon=False); style_axis(ax)
    panel_title(ax, "Calibration: CD versus control"); panel_label(ax, "C")
    ax = fig.add_subplot(gs[1, 1]); oo = optimism.copy(); oo["label"] = oo["modality"].str.upper() + ": " + oo["task"].str.replace("_", " ")
    colors = [COL.get(m.upper(), COL["ink"]) for m in oo["modality"]]
    ax.barh(range(len(oo)), oo["apparent_optimism"], color=colors); ax.set_yticks(range(len(oo)), oo["label"]); ax.axvline(0, color=COL["ink"], lw=0.6)
    ax.set_xlabel("Best historical AUC - nested primary-model AUC"); style_axis(ax, "x"); panel_title(ax, "Apparent model-selection optimism"); panel_label(ax, "D")
    save_figure(fig, "Figure_S8", SUPP)


LEGENDS = {
    "Figure_1": "Figure 1. Study design, participant-level receptor recovery, immune-cell annotation, and lineage-resolved differential abundance. (A) Cohort composition, objective intestinal inflammation, and biologic exposure across 249 participants with Crohn's disease (CD), ulcerative colitis (UC), or non-IBD control status. Bar length indicates the total number of participants; the saturated segment indicates participants with objective intestinal inflammation. (B) Plate-based BD Rhapsody single-cell transcriptomic and paired TCR/BCR workflow, barcode linkage, and participant-level analytical hierarchy across 10 acquisition series. (C) Participant-level receptor recovery among 426,758 quality-controlled T cells and 119,174 quality-controlled B cells. Points represent participants and are colored by diagnosis; diamonds and horizontal lines show the median and interquartile range, respectively. Of 249 transcriptomically profiled participants, 182 had at least one productive TCR or BCR call and were receptor-evaluable. (D) Full PBMC UMAP colored by 53 level 2 immune-cell annotations. The embedding was rigidly rotated for the landscape panel while retaining an equal coordinate aspect ratio, thereby preserving inter-point geometry. Ten prespecified anchor states spanning monocyte, dendritic-cell, CD4/Treg, CD8/innate-like T-cell, B-cell/plasma-cell, and platelet compartments are labeled directly using observed cluster-medoid coordinates, outward displacement, leader lines, collision resolution, and white text halos. The compact key identifies the seven lineage color families; the complete level 2 annotation key is provided in Figure S10. (E-G) Lineage-resolved UMAP representations of CD4/Treg, CD8 T-cell, and B-cell states using biologically organized color families; titles report complete-dataset cell and participant denominators. (H-J) MiloR differential-abundance summaries for CD4 T-cell (H), CD8 T-cell (I), and B-cell (J) states across UC versus control, CD versus control, and CD versus UC. PBMC-wide neighborhoods were assigned to level 2 states, and neighborhoods with an annotation fraction of at least 0.20 were retained. Color indicates the annotation-fraction-weighted median log2 fold change among neighborhoods with SpatialFDR < 0.05; positive values indicate enrichment in the first-named group. Gray cells had fewer than five significant neighborhoods and were not assigned a summary effect. UMAP coordinates were downsampled within cell-state-by-diagnosis strata for visualization only; all quantitative summaries used the complete quality-controlled datasets. Productive and paired receptor definitions are provided in STAR Methods. The explanatory workflow schematic in (B) was initially generated using the OpenAI ChatGPT image-generation tool (accessed August 24, 2026; model version was not exposed in the interface) and was subsequently labeled, edited, and verified for scientific accuracy by the authors.",
    "Figure_2": "Figure 2. TCR clonal expansion exhibits a graded association with cytotoxic and inflammatory T-cell states across IBD diagnoses. (A) Participant-level repertoire clonality and the percentage of T cells belonging to exact productive TCR-beta clonotypes represented by at least two cells. Boxes show medians and interquartile ranges; points represent participants. (B) Clone-size transcriptional dose response for EOMES-ZEB2 inflammatory CD8 TRM-like, effector-cytotoxicity, and Th1/Tc1-inflammatory programs. For each participant and cell state, the mean score in singleton cells was subtracted from the corresponding clone-size-bin score; matched states and CD4/CD8 compartments were then equally weighted. Points show diagnosis-specific participant medians and error bars show 95% participant-bootstrap confidence intervals. (C) Within-participant state enrichment among expanded versus singleton exact productive TCR-beta clonotypes. Values are median log2 odds ratios with 95% participant-bootstrap confidence intervals; positive values indicate enrichment among expanded clonotypes. States were selected as the 12 most abundant receptor-bearing states without reference to effect size. Orange and blue indicate positive and negative Benjamini-Hochberg FDR-significant effects, respectively. (D) Participant-paired singleton and expanded program scores after equal weighting of matched states and CD4/CD8 compartments. Thin gray lines represent participants; colored lines connect diagnosis-specific medians. (E) Conservation of diagnosis-specific expanded-minus-singleton effects from models adjusted for biologic class and acquisition series. Each point represents a predefined transcriptional program; the dashed line is the identity line. (F) Acquisition-series replication of the EOMES-ZEB2 and effector-cytotoxicity participant-level expansion effects. Points show series means with approximate 95% confidence intervals; pooled estimates use participant-bootstrap confidence intervals. Exact clonotype definitions, module construction, covariates, state matching, and multiplicity correction are described in STAR Methods.",
    "Figure_3": "Figure 3. Related TCR sequences converge on inflammatory programs and diagnosis-associated beta-chain motifs. (A) Integrated analysis framework and representative cross-participant sequence-neighborhood component. Participant-specific paired alpha-beta clonotypes were represented using alpha- and beta-chain CDR3 amino-acid 2- and 3-mers together with alpha/beta V- and J-gene tokens. Up to five neighbors were retained among 40 candidates at cosine distance <=0.50; exact paired receptors were excluded. The displayed network was prespecified as the S3 component containing the largest number of participants, and nodes are colored by participant-centered Th1/Tc1 score. (B) Series-specific edge correlations for five transcriptional programs. Module labels were permuted 5,000 times within participant and acquisition series; asterisks denote Benjamini-Hochberg FDR <0.05 across the six endpoints tested within each series. (C) Random-effects pooled edge correlations with 95% participant-jackknife confidence intervals. Annotations report the number of series with a positive estimate and I-squared heterogeneity. (D) Pooled effects for the three primary programs across ten graph definitions, including neighbor number, maximum sequence distance, CDR3-only, alpha-only, beta-only, one-neighbor, and inverse-degree-weighted analyses. Daggers identify participant-jackknife confidence intervals crossing zero. The graph analysis included 43,742 participant-specific paired alpha-beta clonotypes and 5,548 baseline cross-participant edges across eight acquisition series. (E) Enriched-group odds ratios and 95% confidence intervals for three distinct leading beta-chain GLIPH2 clusters per contrast; labels give enriched-group and comparator carrier counts. Odds ratios use a 0.5 continuity correction when a contingency-table cell is zero. GLIPH2 input retained valid TRBV CDR3 amino-acid sequences with one record per participant-sequence. Exact sequence-member sets were deduplicated before two-sided Fisher exact testing and Benjamini-Hochberg correction within each contrast. CD versus control included 91 and 24 participants, UC versus control included 67 and 24, and CD versus UC included 91 and 67. (F) Sequence logos for four distinct representative significant CDR3-beta motifs. (G) Number and direction of FDR-significant unique clusters and exact-sequence overlap audit. Nine CD-versus-control and eight UC-versus-control clusters were control enriched; eight CD-versus-UC clusters were UC enriched. None of the member sequences in the 25 significant GLIPH2 clusters exactly matched a beta-chain CDR3 in the paired alpha-beta sequence-state graph, so the two analyses are presented as complementary rather than directly linked. Participant HLA genotype was unavailable; the sequence-state analysis is V/J-aware but not HLA-stratified, and the GLIPH2 motifs should not be interpreted as validated antigen specificity.",
    "Figure_4": "Figure 4. BCR expansion is concentrated in class-switched memory and antibody-secreting states. (A) Participant-level repertoire structure. (B) Clonal expansion and class switching. (C) State occupancy. (D) Clone-aware module enrichment. (E) Diagnosis-stratified clonal programs. (F) Heavy-chain isotype composition.",
    "Figure_5": "Figure 5. IBD cohorts exhibit region-wide heavy-chain somatic hypermutation remodeling that is shaped by isotype composition, retained after paired heavy-light clonotype resolution, and preferentially CDR targeted. (A) Participant-level germline-relative somatic hypermutation (SHM) rates in heavy-chain framework regions (FWRs) and complementarity-determining regions (CDRs). Small points represent participants; large points and error bars show medians and interquartile ranges. The triangle marks one UC CDR2 value above the displayed range (0.402). Control, Crohn's disease (CD), and ulcerative colitis (UC) groups included 24, 91, and 67 participants, respectively. All 12 regional disease-control contrasts passed Benjamini-Hochberg FDR <= 0.0019 by two-sided Mann-Whitney testing. (B) Isotype-matched disease-control differences in participant-level SHM for IgM-, IgA-, and IgG-assigned heavy sequences. Points show disease-minus-control median differences and bars show 95% participant-bootstrap confidence intervals from 10,000 resamples. Filled symbols denote FDR < 0.05 after correction across the six contrasts. The regional cohort differences were not uniformly retained within isotype classes, indicating that isotype composition contributes to the aggregate pattern. (C) Disease-control differences in overall heavy-chain SHM under four participant-level estimands: all unique heavy sequences, heavy clonotypes with an exact paired light-chain assignment, exact paired heavy-light clonotypes defined by heavy V/J/CDR3 plus light locus/V/J/CDR3, and the same paired clonotypes weighted by linked cell abundance. Points show disease-minus-control median differences and bars show 95% participant-bootstrap confidence intervals. All eight displayed contrasts passed FDR correction (FDR <= 0.022). Heavy-chain SHM remained higher in both disease cohorts after restriction to the paired subset, paired-clonotype resolution, and cell-abundance weighting. Full light-chain variable-region alignments were unavailable, so this panel reports paired-clonotype-resolved heavy-chain SHM rather than joint heavy- and light-chain mutation. (D) Within-participant CDR-targeting contrast, calculated as mean CDR1/CDR2 SHM minus mean FWR1-FWR4 SHM. Boxes show medians and interquartile ranges, whiskers extend to 1.5 times the interquartile range, and points represent participants. Annotations report disease-minus-control median differences and Benjamini-Hochberg FDR values across the two contrasts. Regional SHM was calculated from IgBLAST/IMGT observed-germline alignments. Exact-pair mapping coverage, dominant-light sensitivity, and exploratory expanded-versus-singleton paired-clonotype contrasts are provided in the source-data tables. Acquisition series was largely confounded with diagnosis and could not be included as an independent covariate; findings are therefore described as cohort-associated. Full germline-lineage quality-control summaries are shown in Figure S6 and described in STAR Methods.",
    "Figure_6": "Figure 6. Diagnosis-specific coupling links cytotoxic T-cell expansion to B-cell activation programs in Crohn's disease. (A) Diagnosis-stratified partial Spearman correlations between the expanded-TCR cytotoxicity score and the IgA mucosal plasma-cell or plasmablast/plasma-cell differentiation program. Points show partial correlation coefficients and bars show participant-bootstrap 95% confidence intervals. Primary models adjusted for age, sex, log-transformed TCR depth, and log-transformed BCR depth. Diagnosis modified the association with the IgA mucosal program (interaction P=0.00251) and the plasma-cell-differentiation program (interaction P=0.00127). Six diagnosis-by-program estimates were multiplicity adjusted by the Benjamini-Hochberg method. (B) Matched-participant permutation validation among 56 participants with Crohn's disease. B-cell-program residuals were re-paired 10,000 times within acquisition series while preserving the observed cytotoxicity residuals and series structure. Gray violins show empirical null distributions, horizontal bars show their 95% intervals, and blue points show observed partial correlations; P values are two-sided empirical permutation probabilities. (C and D) Covariate-adjusted rank-residual plots among the same 56 participants. Lines show linear fits in adjusted rank-residual space and shading shows participant-bootstrap 95% confidence bands. (E) Sensitivity of the Crohn's disease estimates to additional acquisition-series, objective-inflammation-status, biologic-exposure, or cell-state-composition adjustment. The composition model additionally included participant-level logit-transformed fractions of CD8 Tem GZMB+ cells within the CD8/innate-like T-cell compartment and IgM plasma cells within the B/plasma-cell compartment. Error bars show 95% intervals from 2,000 participant bootstrap samples, except that primary-model intervals reproduce the prespecified estimates in (A). Biologic-exposure models included participants with complete treatment metadata. Text reports the estimate range after omitting each acquisition series in turn. (F) Formal participant-paired contrasts between correlations for the plasma-cell programs and the IgG-inflammatory or BAFF/APRIL programs. Points show observed differences in partial Spearman correlation and bars show 95% confidence intervals from 5,000 participant bootstrap samples; displayed FDR values are Benjamini-Hochberg adjusted across the four contrasts. All tests were two-sided. Cross-program correlations and all new validation summaries are provided in the accompanying source-data tables. These cross-sectional associations do not establish causal direction or program specificity.",
    "Figure_7": "Figure 7. Immune-receptor sequence architecture classifies diagnosis and is associated with contemporaneous clinical state. (A) Nested validation design. (B) Diagnosis ROC AUCs. (C) Held-out ROC and precision-recall curves. (D) Calibration. (E) Exploratory clinical-state and six-month response-status screens. (F) Incremental information from paired chains. Therapy analyses classify contemporaneous response status and are not prospective predictions.",
    "Figure_S1": "Figure S1. PBMC atlas and Milo differential-abundance summaries.",
    "Figure_S2": "Figure S2. Complete TCR repertoire, expansion-threshold, gene-usage, and state-occupancy analyses.",
    "Figure_S3": "Figure S3. Clone-state interaction models and acquisition-series replication analyses.",
    "Figure_S4": "Figure S4. Complete paired-TCR sequence-state replication analyses. (A) Exact dominant-state-label concordance relative to participant-stratified permutation expectations across acquisition series; error bars show 95% permutation intervals. The endpoint did not replicate after meta-analysis. (B) Weighted Stouffer meta-analysis across complete-cohort and diagnosis-restricted sequence-state tests. (C) Acquisition-series-specific edge correlations for all five prespecified Figure 2 programs. Module labels were permuted 5,000 times within participant and acquisition series; asterisks denote Benjamini-Hochberg FDR <0.05 within series. (D-E) Diagnosis-restricted acquisition-series correlations for the three primary programs in Crohn's disease (D) and ulcerative colitis (E), using the same participant-stratified permutation framework.",
    "Figure_S5": "Figure S5. Complete BCR repertoire, isotype, SHM, gene-usage, and ImmunoMatch analyses.",
    "Figure_S6": "Figure S6. BCR germline-lineage quality control, composition, and sensitivity analyses. (A) Participant-level mean minimum-spanning-tree (MST) branch length when the inferred germline-to-observed-sequence edge is included. (B) Disease-control differences in the germline-inclusive metric across normalized CDR3 distance thresholds of 10%, 15%, and 20%; error bars show 95% participant-bootstrap confidence intervals and annotations report FDR from rank tests. (C) Mean pairwise divergence among at least two distinct observed heavy-chain sequences within each expanded lineage, excluding the germline edge and averaging lineages equally within participants. No disease-control comparison passed FDR correction. Boxes show medians and interquartile ranges, whiskers extend to 1.5 times the interquartile range, and points represent evaluable participants. (D) Cross-state and mixed unswitched/switched isotype occupancy expressed per 1,000 reconstructed lineages; labels report exact numerator/denominator counts. The log scale is used because mixed-isotype lineages were exceptionally rare. (E) Isotype composition among mapped lineages. (F) Transcriptional-state composition among mapped lineages. Full heavy-variable sequences were not linked to individual cell barcodes, preventing assignment of transcriptional states or isotypes to terminal phylogenetic branches. The contrast between (A) and (C) indicates that the germline-inclusive branch-length association should not be interpreted as evidence of increased within-lineage diversification.",
    "Figure_S7": "Figure S7. Complete cross-compartment TCR-BCR coordination analyses.",
    "Figure_S8": "Figure S8. Nested machine-learning validation, permutation, calibration, and optimism analyses.",
    "Figure_S9": "Figure S9. Paired-receptor recovery across representative adaptive immune-cell states. Dot size indicates the number of cells with the indicated productive paired-chain configuration, and color indicates the percentage of cells within each state. Alpha-beta and gamma-delta pairing are shown for T-cell states, and heavy-light pairing is shown for B-cell states. Quantitative summaries use the complete quality-controlled dataset.",
    "Figure_S10": "Figure S10. Full PBMC UMAP with complete level 2 annotation key. Cells are colored by all 53 level 2 immune-cell annotations using lineage-structured color families. Twenty-five representative states are labeled directly using observed cluster-medoid coordinates, collision-aware displacement, leader lines, and white text halos. Rare populations were plotted after abundant populations with slightly larger, more opaque points to preserve visibility. UMAP coordinates were downsampled within cell-state-by-diagnosis strata for visualization only; the complete quality-controlled dataset was used for quantitative analyses.",
    "Figure_S11": "Figure S11. Canonical marker validation of adaptive immune-cell states. Dot size indicates the percentage of cells expressing each gene, and color indicates gene-wise scaled average expression across 15 representative CD4/Treg, CD8 T-cell, B-cell, and plasma-cell states. Marker-expression estimates were calculated from the complete quality-controlled datasets.",
}

LEGENDS["Figure_2"] = (
    "Figure 2. Graded TCR clonal expansion is linked to cytotoxic and inflammatory T-cell states across IBD diagnoses. "
    "(A) Participant-level repertoire clonality and the percentage of T cells belonging to exact productive TCR-beta "
    "clonotypes represented by at least two cells. Boxes show medians and interquartile ranges; points represent "
    "participants. (B) Clone-size transcriptional dose response for EOMES-ZEB2 inflammatory CD8 TRM-like, "
    "effector-cytotoxicity, and Th1/Tc1-inflammatory programs. For each participant and cell state, the mean score in "
    "singleton cells was subtracted from the corresponding clone-size-bin score; matched states and CD4/CD8 "
    "compartments were then equally weighted. Thin colored trajectories show diagnosis-specific participant medians; "
    "black diamonds and lines show pooled participant medians, and error bars show pooled 95% participant-bootstrap "
    "confidence intervals. The number of contributing participant-bin observations is printed beneath each clone-size "
    "category. Annotations report the pooled slope per ordered clone-size bin "
    "from participant-fixed-effect models with participant-clustered standard errors and Benjamini-Hochberg FDR; "
    "diagnosis-by-clone-size trend interactions were not significant. (C) Participant-level log2 odds ratios for state "
    "occupancy among expanded versus singleton exact productive TCR-beta clonotypes, summarized after adjustment for "
    "diagnosis, acquisition batch, and log-transformed receptor depth. Points and bars show standardized marginal "
    "estimates and heteroskedasticity-robust 95% confidence intervals. States were selected as the 12 most abundant "
    "receptor-bearing states without reference to effect size. Orange and blue indicate positive and negative "
    "Benjamini-Hochberg FDR-significant effects, respectively. (D) Distributions of participant-level "
    "expanded-minus-singleton program-score differences after equal weighting of matched states and CD4/CD8 "
    "compartments. Colored points represent participants by diagnosis, gray violins show the complete distribution, "
    "diamonds show medians, and bars show participant-bootstrap 95% confidence intervals. Displayed FDR values are "
    "from two-sided paired Wilcoxon tests across the three prespecified programs. (E) Conservation of "
    "diagnosis-specific expanded-minus-singleton effects from models adjusted for biologic class and acquisition "
    "series. Each point represents a predefined transcriptional program; horizontal and vertical bars show 95% "
    "confidence intervals for CD and UC effects, respectively, and the dashed line is the identity line. Across 36 "
    "formal diagnosis-interaction contrasts, none passed FDR <0.05. (F) Acquisition-series replication of the "
    "EOMES-ZEB2 and effector-cytotoxicity participant-level expansion effects. Points show series means and 95% "
    "t-distribution confidence intervals; diamonds show pooled participant means. Annotations report inverse-variance "
    "I-squared heterogeneity and the pooled-effect range after omitting each acquisition series. Series S6 and S9 "
    "lacked estimable state-matched effects and are not shown. Exact clonotype definitions, module construction, "
    "covariates, state matching, and multiplicity correction are described in STAR Methods."
)

LEGENDS["Figure_1"] = (
    "Figure 1. Study design, clone-aware immune-cell annotation, and lineage-resolved differential abundance. "
    "(A) Cohort composition, objective intestinal inflammation, and biologic exposure across 249 participants with "
    "Crohn's disease (CD), ulcerative colitis (UC), or non-IBD control status. Bar length indicates the total number "
    "of participants; the saturated segment indicates participants with objective intestinal inflammation. "
    "(B) Plate-based BD Rhapsody workflow linking PBMC isolation, single-cell capture, whole-transcriptome analysis, "
    "paired TCR/BCR sequencing, and cell-state-resolved repertoire analysis across 10 acquisition series. "
    "(C) Full PBMC UMAP colored by 53 level 2 immune-cell annotations using lineage-structured state colors. The "
    "embedding was rigidly rotated for the landscape panel while retaining equal coordinate scaling and therefore "
    "preserving inter-point geometry. Ten representative anchor states are labeled directly; the complete annotation "
    "key is provided in Figure S10. "
    "(D) Participant-level receptor recovery among quality-controlled T cells and B cells. Colored points represent "
    "receptor-evaluable participants by diagnosis, horizontal lines show the pooled interquartile range, and diamonds "
    "show the pooled median; exact median [IQR] values are printed beside each metric. Of 249 transcriptomically profiled "
    "participants, 182 had productive TCR and BCR calls and were receptor-evaluable (control, n=24; CD, n=91; UC, n=67). "
    "Participants without productive receptor recovery were excluded rather than represented as biological zeroes. "
    "Paired gamma-delta recovery is shown on an expanded 0-4% axis; all other metrics use a 0-100% axis. "
    "(E-G) Lineage-resolved CD4 T-cell (E), CD8 T-cell (F), and B-cell (G) UMAPs colored using the original level 3 "
    "annotations from the corresponding lineage Seurat objects. Complete level 3 annotation keys are positioned to "
    "the right of each UMAP; direct labels were omitted from the embeddings to preserve cluster geometry and density. "
    "Adaptive states are shown using biologically organized colors, whereas a saturated orange-magenta-purple overlay "
    "identifies cells belonging to expanded paired receptor clonotypes. Overlay color and point size "
    "encode clonotype size as small (2-5 cells), medium (6-20 cells), large (21-100 cells), or hyperexpanded (>100 cells). "
    "Paired TCR clonotypes were "
    "defined within participants by concordant productive alpha-beta or gamma-delta V-gene, J-gene, and CDR3 amino-acid "
    "identities, and paired BCR clonotypes by concordant productive heavy-light V-gene, J-gene, and CDR3 amino-acid "
    "identities. Expansion required detection in at least two cells. Panel subtitles report complete-data cell and "
    "participant denominators. Background point clouds were sampled to at most 60,000 cells per panel within level 3 "
    "state-by-"
    "diagnosis strata to harmonize visual density. Expanded cells were plotted independently above the background; the "
    "CD4/Treg display embedding contained 161 of 448 complete-data expanded cells, whereas the complete CD8 overlay was "
    "shown. To prevent overplotting in (G), expanded BCR-bearing cells were sampled within cell state and clone-size bin "
    "to a maximum of 180 cells per stratum, displaying 2,080 of 30,300 expanded cells. Sampling affected visualization "
    "only; all quantitative summaries use complete quality-controlled datasets. Panels D-G were redrawn from the source "
    "Seurat objects using the "
    "PBMC scvi50.umap, CD4 umap_SCVI_50, CD8 umap_SCVI_50E400, and B-cell umap reductions. "
    "(H) Canonical identity and clone engagement across 15 representative Level 3 adaptive immune-cell states, displayed "
    "as lineage-specific CD4 T-cell, CD8 T-cell, and B-cell facets matching (E-G). Within the marker matrices, dot size "
    "indicates the percentage of cells expressing each gene and color indicates gene-wise scaled average expression; "
    "estimates were calculated from the complete quality-controlled lineage objects. The aligned clone-engagement columns "
    "show participant-median paired-receptor recovery within each state and the participant-median fraction of paired-receptor "
    "cells belonging to expanded clonotypes. Expansion required at least two cells assigned to the same exact paired clonotype; "
    "participants without paired-receptor cells in a state were excluded from that state's expanded-fraction denominator. "
    "(I) Unified MiloR differential-abundance summaries for CD4 T-cell, CD8 T-cell, and B-cell "
    "states across UC versus control, CD versus control, and CD versus UC. PBMC-wide neighborhoods were assigned to "
    "level 2 states, and neighborhoods with an annotation fraction of at least 0.20 were retained. Color indicates "
    "the annotation-fraction-weighted median log2 fold change among neighborhoods with SpatialFDR < 0.05; positive "
    "values indicate enrichment in the first-named group. Gray cells had fewer than five significant neighborhoods. "
    "All quantitative summaries used the complete quality-controlled datasets. "
    "The explanatory artwork in (B) was generated using the OpenAI ChatGPT image-generation tool (accessed August 26, "
    "2026; model version was not exposed in the interface), scientifically reviewed by the authors, and combined with "
    "author-generated labels to ensure accurate platform terminology."
)

LEGENDS["Figure_3"] = (
    "Figure 3. Paired TCR sequence architecture links clonal expansion to inflammatory-state organization in Crohn's "
    "disease and ulcerative colitis. (A) Left-to-right bridge from the exact-clone expansion result in Figure 2 to a test of "
    "non-identical paired alpha-beta receptor neighborhoods in the same 182 receptor-evaluable participants. The three colored "
    "marks identify the expansion-linked EOMES-ZEB2, cytotoxicity, and Th1/Tc1 programs. The sequence-state analysis included "
    "43,742 participant-specific paired alpha-beta clonotypes. The displayed network is the S3 component containing the "
    "largest number of participants. Every displayed edge connects receptors from different participants; edge width encodes "
    "sequence similarity, exact receptor pairs were excluded, and node color represents participant-centered EOMES-ZEB2 "
    "activity. The annotation identifies the edge with the largest combined EOMES-ZEB2 activity. (B) Random-effects pooled "
    "correlations between neighboring receptors for the five Figure 2 programs, "
    "with 95% participant-jackknife confidence intervals. The baseline graph contained 5,548 cross-participant edges and "
    "181 participants with at least one edge. Annotations report acquisition series with positive estimates and I-squared "
    "heterogeneity. (C) Random-effects pooled correlations for the three primary programs within Crohn's disease (CD) and "
    "ulcerative colitis (UC), with 95% participant-jackknife confidence intervals. Diagnosis-series strata required at least "
    "100 edges and six participants, yielding three CD strata with 1,023 edges and four UC strata with 1,922 edges. All three "
    "program estimates were positive in every contributing CD and UC stratum. The omnibus CD-versus-UC interaction used "
    "2,000 participant-block bootstrap replicates across the three correlated programs and was not significant (P=0.87). "
    "The complete acquisition-series heatmap is shown in Figure S4. (D) Random-effects pooled sensitivity estimates for the "
    "three primary programs, displayed as coefficients and 95% participant-jackknife confidence intervals. Filled symbols "
    "denote intervals excluding zero; open symbols denote intervals including zero. Edge totals are summed across the eight "
    "acquisition series. Clone-size-adjusted scores removed a common within-participant linear association with log clonotype "
    "size before participant standardization; singleton-only analyses excluded all expanded clonotypes; dominant-state-adjusted "
    "scores removed dominant-state means before participant standardization. Additional rows show paired CDR3-only, alpha-chain-"
    "only, and beta-chain-only graphs. Formal attenuation contrasts resampled participants within acquisition series 2,000 times "
    "while retaining the fixed sequence graph and fixed random-effects weights. Adjustment for clone size did not jointly "
    "attenuate the three correlations (omnibus P=0.45), whereas dominant-state adjustment produced joint attenuation "
    "(omnibus P=1.85×10^-5). These results are consistent with broader cell-state context carrying much of the continuous "
    "program concordance but do not establish causal mediation. No batch term was included. (E) Inflammatory-status organization of within-diagnosis "
    "IBD sequence-neighbor edges. The left panel shows the observed minus participant-label-permutation expectation for the "
    "fraction of edges joining participants with the same inflammatory status, with 95% participant-delete-one jackknife "
    "confidence intervals. Inflammatory-status labels were permuted 10,000 times among participants within diagnosis while "
    "preserving graph topology and participant clonotype burden. The analysis included 3,246 CD-CD or UC-UC edges from 157 "
    "participants; excess same-status fractions were 0.168 in pooled IBD, 0.233 in CD, and 0.128 in UC (permutation P=0.0006, "
    "0.0002, and 0.019, respectively). The right panel compares observed and expected edge fractions for inflamed-inflamed, "
    "noninflamed-noninflamed, and mixed-status pairs; q values are Benjamini-Hochberg-adjusted within the three categories. "
    "No batch term was included. (F) Participant-normalized paired gamma-delta V-gene architecture across 135 participants. "
    "Bubble area represents participant prevalence, fill represents the log2 ratio of the observed mean within-participant "
    "pair fraction to its expectation under independent V-gene pairing, and dark outlines denote pairing FDR<0.05. Expected "
    "fractions and two-sided P values were obtained by shuffling delta-chain V-gene labels among cells within each participant "
    "10,000 times; FDR was controlled across all 24 observed or possible V-gene pairs. TRGV9-TRDV2 accounted for 403 of 600 "
    "cells, occurred in 113 of 135 participants, and had a mean participant fraction of 64% (log2 observed/expected=0.095, "
    "q=0.00060). The inset displays each participant's TRGV9-TRDV2 fraction by diagnosis with median and interquartile range; "
    "the descriptive diagnosis comparison was not significant (Kruskal-Wallis P=0.13). (G) Participant- and observed-state-"
    "matched standardized program contrasts for TRGV9-TRDV2 versus other paired gamma-delta receptors (38 participants) and "
    "expanded versus singleton paired gamma-delta clonotypes (18 participants). Points show mean participant contrasts and "
    "bars show 95% participant-bootstrap confidence intervals from 10,000 replicates; filled points have intervals excluding "
    "zero. Two-sided mean contrasts used participant sign-flip tests, exact for the 18-participant expansion analysis and "
    "200,000 draws for the 38-participant receptor-identity analysis, with FDR correction across the five programs within "
    "each contrast. Receptor-identity intervals all included zero. Expansion was associated with higher mucosal-retention "
    "activity (effect=0.591, 95% CI 0.237-0.987, q=0.026); the cytotoxicity estimate was also positive but did not meet the "
    "5% FDR threshold (effect=0.308, 95% CI 0.071-0.547, q=0.063). These exploratory expansion contrasts are limited by "
    "the 18 contributing participants. No batch variables or batch terms were included in the gamma-delta analyses."
)

LEGENDS["Figure_5"] = (
    "Figure 5. IBD is associated with expanded, germline-divergent BCR lineages and regionally targeted heavy-chain "
    "somatic hypermutation without increased observed-only lineage diversification. (A) Participant-level "
    "germline-relative somatic hypermutation (SHM) rates in heavy-chain framework regions (FWRs) and "
    "complementarity-determining regions (CDRs). Small points represent participants; large points and error bars show "
    "medians and interquartile ranges. The triangle marks one UC CDR2 value above the displayed range (0.402). Control, "
    "Crohn's disease (CD), and ulcerative colitis (UC) groups included 24, 91, and 67 participants, respectively. All "
    "12 regional disease-control contrasts passed Benjamini-Hochberg FDR <=0.0019. (B) Isotype-matched disease-control "
    "differences in participant-level SHM for IgM-, IgA-, and IgG-assigned heavy sequences. Points show median "
    "differences and bars show 95% participant-bootstrap confidence intervals. Filled symbols denote FDR <0.05 across "
    "the six contrasts. The regional differences were not uniformly retained within isotype classes, indicating that "
    "isotype composition contributes to the aggregate pattern. (C) Disease-control differences in overall heavy-chain "
    "SHM under four participant-level estimands: all unique heavy sequences, heavy clonotypes with an exact paired "
    "light-chain assignment, exact paired heavy-light clonotypes, and paired clonotypes weighted by linked cell "
    "abundance. All eight displayed contrasts passed FDR correction (FDR <=0.022). Full light-chain variable-region "
    "alignments were unavailable, so this panel reports paired-clonotype-resolved heavy-chain SHM rather than joint "
    "heavy- and light-chain mutation. (D) Within-participant CDR-targeting contrast, calculated as mean CDR1/CDR2 SHM "
    "minus mean FWR1-FWR4 SHM. Boxes show medians and interquartile ranges, whiskers extend to 1.5 times the "
    "interquartile range, and points represent participants. (E) Prespecified representative germline-rooted minimum-"
    "spanning lineage graphs for Control, CD, and UC. Eligible examples were class-switched, had a valid paired light "
    "chain, contained 4-10 distinct observed heavy alignments, and were not truncated. Within each diagnosis, the lineage "
    "closest to the pooled eligible medians for node count, log abundance, and germline-inclusive branch length was "
    "selected without reference to clinical outcome. Node size represents observed abundance; black diamonds mark the "
    "IMGT germline; dashed gray edges mark inferred germline connections; solid colored edges connect observed "
    "sequences. Isotype, dominant paired light chain, and dominant B-cell state are lineage-level annotations and cannot "
    "be assigned to individual terminal branches. (F) Control-IQR-standardized participant-level disease-control effects "
    "for expanded-lineage burden, germline-inclusive branch length, observed-only divergence, and class-switched lineage "
    "fraction. Filled symbols denote FDR <0.05. CD and UC did not differ for any displayed lineage metric after "
    "correction (all FDR >=0.65), supporting a shared IBD-associated architecture. Germline-inclusive branch-length "
    "differences should not be interpreted as accelerated lineage evolution because observed-only diversification was "
    "not increased. (G) Descriptive maturation landscape for expanded lineages with at least two distinct observed "
    "heavy-chain alignments. Each point is one lineage; the x axis shows germline-inclusive total MST branch length, "
    "the y axis shows observed-only mean pairwise divergence, point area scales with log-linked cell abundance, and "
    "color indicates the dominant lineage-level isotype. Stars identify the prespecified lineages displayed in (E), "
    "dotted lines mark diagnosis-specific medians, and marginal rugs show the univariate distributions. Lineage-level "
    "points are descriptive; all inferential comparisons remain participant-level. (H) Adjusted participant-level "
    "association between CDR targeting and log-transformed fecal calprotectin, shown separately for CD and UC. Both "
    "variables were rank transformed and residualized for log lineage depth, biologic exposure, and acquisition series. "
    "Lines show linear fits to the adjusted ranks and shading shows 95% participant-bootstrap confidence bands. UC "
    "CDR targeting correlated positively with calprotectin (partial rho=0.366, FDR=0.0275), whereas the CD association "
    "was null. Acquisition series was largely confounded with diagnosis; findings are cohort-associated."
)

LEGENDS["Figure_S12"] = (
    "Figure S12. Clinical, mucosal-state, and paired-light context of IBD-associated BCR lineage remodeling. "
    "(A) Within-diagnosis objective-inflammation contrasts for six participant-level lineage or SHM metrics. Outcomes "
    "were rank transformed and residualized for log lineage depth, biologic exposure, and acquisition series before "
    "inflamed-versus-noninflamed comparison. Points show adjusted median differences and bars show 95% participant-"
    "bootstrap confidence intervals; no contrast passed FDR correction. (B) Partial Spearman correlations with "
    "log-transformed fecal calprotectin after adjustment for log lineage depth, biologic exposure, and acquisition "
    "series. UC CDR targeting correlated positively with calprotectin (rho=0.366, FDR=0.0275); the remaining endpoints "
    "did not pass correction. (C) Disease-control differences in participant-level cell-state fractions among cells "
    "mapped to expanded heavy-chain lineages. IgA plasma-cell representation was higher in CD and UC (median differences "
    "0.222 and 0.200; FDR=0.00233 and 0.02395), whereas the other prespecified mucosal-relevant states did not pass "
    "correction. (D) Disease-control differences in paired light-chain locus or V-family fractions. The five most "
    "abundant light V-gene families were selected without reference to effect size. IGKV1 representation was higher in "
    "UC (median difference 0.0576, FDR=0.0264); other features did not pass correction. All analyses used participants "
    "as the inferential unit. Clinical and light-chain findings are cross-sectional and require independent replication; "
    "they do not establish antigen specificity, temporal class switching, or treatment response."
)


def write_supporting_files():
    all_text = []
    for stem, legend in LEGENDS.items():
        (LEG / f"{stem}_legend.txt").write_text(legend + "\n", encoding="utf-8")
        all_text.append(legend)
    (LEG / "All_figure_legends.txt").write_text("\n\n".join(all_text) + "\n", encoding="utf-8")
    compliance = """Cell Press artwork specification applied

- Full-width target: 190 mm (7.48 inches).
- Arial typography; 7-8 pt data labels and 12 pt bold panel letters at final size.
- Capital panel labels placed consistently at upper left.
- Figure titles are retained in the manuscript legends, not embedded in artwork.
- Okabe-Ito-derived, color-vision-accessible diagnostic palette reinforced by labels and position.
- Individual participant points accompany distribution summaries where applicable.
- Vector PDF, RGB 500-dpi LZW-compressed TIFF, and 300-dpi PNG supplied for every figure.
- Dense cell-level point clouds are rasterized within otherwise vector PDF files to control file size.
- Therapy analyses are labeled as contemporaneous six-month response-status classification.
"""
    (OUT / "Cell_Press_figure_specification.txt").write_text(compliance, encoding="utf-8")
    manifest_rows = []
    for stem in [f"Figure_{i}" for i in range(1, 8)] + [f"Figure_S{i}" for i in range(1, 13)]:
        folder = "Main Figures" if "_S" not in stem else "Supplementary Figures"
        for ext in ("pdf", "tif", "png"):
            manifest_rows.append({"figure": stem, "format": ext.upper(), "path": f"{folder}/{stem}.{ext}"})
    pd.DataFrame(manifest_rows).to_csv(SRC / "Cell_Press_figure_output_manifest.csv", index=False)


def main():
    builders = [
        build_figure_1, build_figure_2, build_figure_3, build_figure_4, build_figure_5,
        build_figure_6, build_figure_7, build_figure_s1, build_figure_s2, build_figure_s3,
        build_figure_s4, build_figure_s5, build_figure_s6, build_figure_s7, build_figure_s8,
        build_figure_s9, build_figure_s10, build_figure_s11, build_figure_s12,
    ]
    for builder in builders:
        print(f"Building {builder.__name__}...", flush=True)
        builder()
    write_supporting_files()
    merge_pdfs([MAIN / f"Figure_{i}.pdf" for i in range(1, 8)], OUT / "Main_Figures_1-7_Cell_Press.pdf")
    merge_pdfs([SUPP / f"Figure_S{i}.pdf" for i in range(1, 13)], OUT / "Supplementary_Figures_S1-S12_Cell_Press.pdf")
    print(OUT)


if __name__ == "__main__":
    import sys
    if "--figure1-only" in sys.argv:
        print("Building build_figure_1...", flush=True)
        build_figure_1()
        print("Building build_figure_s10...", flush=True)
        build_figure_s10()
        write_supporting_files()
        merge_pdfs([MAIN / f"Figure_{i}.pdf" for i in range(1, 8)], OUT / "Main_Figures_1-7_Cell_Press.pdf")
        merge_pdfs([SUPP / f"Figure_S{i}.pdf" for i in range(1, 13)], OUT / "Supplementary_Figures_S1-S12_Cell_Press.pdf")
        print(OUT)
    elif "--figure3-only" in sys.argv:
        print("Building build_figure_3...", flush=True)
        build_figure_3()
        print("Building build_figure_s4...", flush=True)
        build_figure_s4()
        write_supporting_files()
        merge_pdfs([MAIN / f"Figure_{i}.pdf" for i in range(1, 8)], OUT / "Main_Figures_1-7_Cell_Press.pdf")
        merge_pdfs([SUPP / f"Figure_S{i}.pdf" for i in range(1, 12)], OUT / "Supplementary_Figures_S1-S11_Cell_Press.pdf")
        print(OUT)
    elif "--figure2-only" in sys.argv:
        print("Building build_figure_2...", flush=True)
        build_figure_2()
        write_supporting_files()
        merge_pdfs([MAIN / f"Figure_{i}.pdf" for i in range(1, 8)], OUT / "Main_Figures_1-7_Cell_Press.pdf")
        print(OUT)
    elif "--figure5-only" in sys.argv:
        print("Building build_figure_5...", flush=True)
        build_figure_5()
        print("Building build_figure_s6...", flush=True)
        build_figure_s6()
        print("Building build_figure_s12...", flush=True)
        build_figure_s12()
        write_supporting_files()
        merge_pdfs([MAIN / f"Figure_{i}.pdf" for i in range(1, 8)], OUT / "Main_Figures_1-7_Cell_Press.pdf")
        merge_pdfs([SUPP / f"Figure_S{i}.pdf" for i in range(1, 13)], OUT / "Supplementary_Figures_S1-S12_Cell_Press.pdf")
        print(OUT)
    elif "--figure6-only" in sys.argv:
        print("Building build_figure_6...", flush=True)
        build_figure_6()
        write_supporting_files()
        print(OUT)
    else:
        main()

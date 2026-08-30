from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTICIPANT_DATA = ROOT / "High Impact Additional Analyses" / "Table_HI_clonotype_state_breadth_by_participant.csv"
DEFAULT_EFFECT_DATA = ROOT / "High Impact Additional Analyses" / "Table_HI_clonotype_state_breadth_pairwise_tests.csv"

INK = "#222222"
MUTED = "#6F777F"
GRID = "#D9D9D9"
DIAGNOSIS_ORDER = ["Control", "CD", "UC"]
DIAGNOSIS_COLORS = {"Control": "#737373", "CD": "#0072B2", "UC": "#D55E00"}
MODALITY_COLORS = {"BCR": "#D55E00", "TCR": "#7B3294"}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.titlesize": 7.5,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.7,
        "ytick.major.size": 2.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_axis(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, pad=1.6)
    ax.grid(axis=grid_axis, color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def add_panel_heading(fig: mpl.figure.Figure, letter: str, title: str, y: float) -> None:
    fig.text(0.050, y, letter, fontsize=12, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.082, y, title, fontsize=8.2, fontweight="bold", color=INK, va="top", ha="left")


def box_strip(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    metric: str,
    modality: str,
    threshold: int,
    seed: int,
) -> None:
    subset = data[(data["modality"] == modality) & (data["threshold"] == threshold)]
    arrays = [
        subset.loc[subset["Diagnosis1"] == diagnosis, metric].dropna().to_numpy(float)
        for diagnosis in DIAGNOSIS_ORDER
    ]
    positions = np.arange(len(DIAGNOSIS_ORDER))
    boxes = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.0},
        whiskerprops={"color": MUTED, "linewidth": 0.75},
        capprops={"color": MUTED, "linewidth": 0.75},
        boxprops={"color": MUTED, "linewidth": 0.75},
    )
    for patch, diagnosis in zip(boxes["boxes"], DIAGNOSIS_ORDER):
        patch.set_facecolor(DIAGNOSIS_COLORS[diagnosis])
        patch.set_alpha(0.20)

    rng = np.random.default_rng(seed)
    for index, (diagnosis, values) in enumerate(zip(DIAGNOSIS_ORDER, arrays)):
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(
            index + jitter,
            values,
            s=10,
            color=DIAGNOSIS_COLORS[diagnosis],
            alpha=0.58,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
    ax.set_xticks(positions, DIAGNOSIS_ORDER)
    style_axis(ax)


def draw_box_panel(
    fig: mpl.figure.Figure,
    data: pd.DataFrame,
    metric: str,
    letter: str,
    title: str,
    heading_y: float,
    top_row_y: float,
    bottom_row_y: float,
    row_height: float,
    percent: bool = False,
) -> None:
    add_panel_heading(fig, letter, title, heading_y)
    left = 0.115
    width = 0.378
    gap = 0.045
    y_limits: dict[str, tuple[float, float]] = {}
    for modality in ("BCR", "TCR"):
        values = data[(data["modality"] == modality) & data["threshold"].isin([2, 5])][metric].dropna()
        if percent:
            y_limits[modality] = (-0.05, 1.07)
        else:
            span = max(float(values.max() - values.min()), 0.4)
            y_limits[modality] = (float(values.min() - 0.07 * span), float(values.max() + 0.10 * span))

    for row_index, (modality, y_pos) in enumerate((("BCR", top_row_y), ("TCR", bottom_row_y))):
        for column_index, threshold in enumerate((2, 5)):
            ax = fig.add_axes([left + column_index * (width + gap), y_pos, width, row_height])
            box_strip(ax, data, metric, modality, threshold, seed=20260830 + row_index * 10 + column_index)
            ax.set_ylim(*y_limits[modality])
            if percent:
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            if column_index:
                ax.tick_params(axis="y", labelleft=False)
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_title(f"Clonotypes >={threshold} cells", fontweight="bold", pad=3.2)
            else:
                ax.set_xlabel("")
        fig.text(0.925, y_pos + row_height / 2, modality, fontsize=6.5, fontweight="bold",
                 color=MODALITY_COLORS[modality], rotation=-90, va="center", ha="center")
    shared_ylabel = (
        "Median excess states per clonotype"
        if metric == "median_breadth_excess"
        else "Multistate clonotype fraction"
    )
    block_center = (top_row_y + row_height + bottom_row_y) / 2
    fig.text(0.078, block_center, shared_ylabel, fontsize=6.8, color=INK,
             rotation=90, va="center", ha="center")


def draw_effect_panel(fig: mpl.figure.Figure, effects: pd.DataFrame) -> None:
    add_panel_heading(fig, "J", "Diagnosis effect sizes", 0.160)
    contrasts = ["CD vs UC", "UC vs Control", "CD vs Control"]
    metrics = [
        ("multistate_fraction", "Multistate fraction", 0.095),
        ("median_breadth_excess", "Null-adjusted breadth", 0.030),
    ]
    left = 0.115
    width = 0.378
    gap = 0.045
    height = 0.052
    offsets = {"BCR": -0.10, "TCR": 0.10}

    filtered = effects[
        effects["metric"].isin([metric for metric, _, _ in metrics])
        & effects["threshold"].isin([2, 5])
    ].copy()
    limits: dict[str, tuple[float, float]] = {}
    for metric, _, _ in metrics:
        values = filtered[filtered["metric"] == metric][["ci_low", "ci_high"]].to_numpy(float)
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        span = max(high - low, 0.25)
        limits[metric] = (low - 0.07 * span, high + 0.07 * span)

    for row_index, (metric, row_label, y_pos) in enumerate(metrics):
        for column_index, threshold in enumerate((2, 5)):
            ax = fig.add_axes([left + column_index * (width + gap), y_pos, width, height])
            subset = filtered[(filtered["metric"] == metric) & (filtered["threshold"] == threshold)]
            for modality in ("BCR", "TCR"):
                rows = subset[subset["modality"] == modality].set_index("contrast").reindex(contrasts)
                for base_y, (_, result) in zip(np.arange(3)[::-1], rows.iterrows()):
                    if not np.isfinite(result.get("rank_biserial", np.nan)):
                        continue
                    y_value = base_y + offsets[modality]
                    color = MODALITY_COLORS[modality]
                    ax.plot([result["ci_low"], result["ci_high"]], [y_value, y_value], color=color, lw=0.9)
                    significant = bool(np.isfinite(result.get("p_adj", np.nan)) and result["p_adj"] < 0.05)
                    ax.scatter(result["rank_biserial"], y_value, s=18,
                               facecolor=color if significant else "white", edgecolor=color, lw=0.8, zorder=3)
            ax.axvline(0, color=MUTED, lw=0.65, ls="--")
            ax.set_xlim(*limits[metric])
            ax.set_ylim(-0.45, 2.45)
            ax.set_yticks(np.arange(3)[::-1], contrasts)
            if column_index:
                ax.tick_params(axis="y", labelleft=False)
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_title(f"Threshold >={threshold} cells", fontweight="bold", pad=3.2)
            else:
                ax.set_xlabel("Rank-biserial effect (higher to the right)", labelpad=2)
            style_axis(ax, grid_axis="x")
        fig.text(0.925, y_pos + height / 2, row_label, fontsize=6.2, fontweight="bold",
                 color=INK, rotation=-90, va="center", ha="center")

    handles = [
        Line2D([0], [0], color=MODALITY_COLORS[modality], marker="o", markerfacecolor="white",
               markeredgecolor=MODALITY_COLORS[modality], lw=0.9, ms=4, label=modality)
        for modality in ("BCR", "TCR")
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.004), ncol=2,
               frameon=False, fontsize=6.1, handlelength=1.2, columnspacing=1.0)


def build_overlay(
    width_points: float,
    height_points: float,
    output: Path,
    participant_data: Path,
    effect_data: Path,
) -> None:
    participant = pd.read_csv(participant_data)
    effects = pd.read_csv(effect_data)
    fig = plt.figure(figsize=(width_points / 72, height_points / 72))
    fig.patch.set_alpha(0)
    fig.add_artist(Rectangle((0, 0), 1, 0.49, transform=fig.transFigure,
                             facecolor="white", edgecolor="none", zorder=-100))

    draw_box_panel(
        fig, participant, "median_breadth_excess", "H", "Clone-size-adjusted state breadth",
        heading_y=0.478, top_row_y=0.407, bottom_row_y=0.342, row_height=0.052,
    )
    draw_box_panel(
        fig, participant, "multistate_fraction", "I", "Multistate occupancy by expanded clonotypes",
        heading_y=0.322, top_row_y=0.251, bottom_row_y=0.186, row_height=0.052, percent=True,
    )
    draw_effect_panel(fig, effects)
    fig.savefig(output, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def rebuild(
    source: Path,
    output: Path,
    overlay_path: Path,
    participant_data: Path,
    effect_data: Path,
) -> None:
    reader = PdfReader(source)
    if len(reader.pages) != 4:
        raise ValueError(f"Expected four pages in Figure S2; found {len(reader.pages)}")
    page = reader.pages[1]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    build_overlay(width, height, overlay_path, participant_data, effect_data)
    overlay = PdfReader(overlay_path).pages[0]
    page.merge_page(overlay)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for index, source_page in enumerate(reader.pages):
        writer.add_page(page if index == 1 else source_page)
    with output.open("wb") as stream:
        writer.write(stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("--participant-data", type=Path, default=DEFAULT_PARTICIPANT_DATA)
    parser.add_argument("--effect-data", type=Path, default=DEFAULT_EFFECT_DATA)
    arguments = parser.parse_args()
    rebuild(
        arguments.source,
        arguments.output,
        arguments.overlay,
        arguments.participant_data,
        arguments.effect_data,
    )

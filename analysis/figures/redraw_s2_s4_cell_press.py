from __future__ import annotations

from pathlib import Path
import shutil
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from pypdf import PdfReader, PdfWriter
from scipy.stats import kruskal, mannwhitneyu


ROOT = Path("C:/path/to/private-manuscript-workspace")
OUT = ROOT / "output" / "pdf" / "Pruned Reordered Supplementary Figures"
WORK = ROOT / "output" / "pdf" / "cell_press_redraw_work"
ARCHIVE = OUT / "archive_before_cell_press_redraw_20260829"

TCR = ROOT / "Figure 2 TCR"
TCR_ARCH = TCR / "TCR Architecture Analyses"
BCR_SUB = ROOT / "Figure 3 BCR" / "BCR Architecture Analyses" / "Subfigures"
F5 = Path("C:/path/to/private-legacy-manuscript-assets/Figure 5")
BCR_MODULES = ROOT / "BCR Module Scores" / "BCR Clonotype Modules Cell Number Matched CD UC Only"
BCR_SWITCH = ROOT / "Figure 3 BCR" / "BCR Architecture Analyses" / "BCR isotype switching diagnosis comparisons" / "tables"

S2_OLD = OUT / "Figure_S2.pdf"
S4_OLD = OUT / "Figure_S4.pdf"
S2_NEW = OUT / "Figure_S2.pdf"
S4_NEW = OUT / "Figure_S4.pdf"

DIAG_ORDER = ["Control", "UC", "CD"]
DIAG_COLORS = {"Control": "#737373", "UC": "#D55E00", "CD": "#0072B2"}
INK = "#222222"
MID = "#666666"
GRID = "#D9D9D9"
TCR_BLUE = "#0072B2"
BCR_ORANGE = "#D55E00"
NEGATIVE = "#3E82B7"
POSITIVE = "#D8633B"

MM = 1 / 25.4
RNG = np.random.default_rng(20260829)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.7,
        "ytick.major.size": 2.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def bh_adjust(values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    good = np.isfinite(p)
    if not good.any():
        return out
    x = p[good]
    order = np.argsort(x)
    ranked = x[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.minimum(q, 1)
    out[good] = restored
    return out


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3f}"


def pairwise_stats(frame: pd.DataFrame, value: str, group: str) -> tuple[float, list[tuple[str, str, float]]]:
    arrays = [frame.loc[frame[group] == diagnosis, value].dropna().to_numpy() for diagnosis in DIAG_ORDER]
    valid = [x for x in arrays if len(x)]
    overall = kruskal(*valid).pvalue if len(valid) >= 2 else np.nan
    pairs = [("Control", "UC"), ("Control", "CD"), ("UC", "CD")]
    raw = []
    for a, b in pairs:
        xa = frame.loc[frame[group] == a, value].dropna()
        xb = frame.loc[frame[group] == b, value].dropna()
        raw.append(mannwhitneyu(xa, xb, alternative="two-sided").pvalue if len(xa) and len(xb) else np.nan)
    adjusted = bh_adjust(raw)
    return overall, [(a, b, q) for (a, b), q in zip(pairs, adjusted)]


def clean_axes(ax: mpl.axes.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, pad=1.7)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def add_panel_letter(fig: mpl.figure.Figure, x: float, y: float, letter: str) -> None:
    fig.text(x, y, letter, fontsize=15, fontweight="bold", color=INK, va="top", ha="left")


def add_page_header(fig: mpl.figure.Figure, text: str) -> None:
    fig.text(0.025, 0.986, text, fontsize=7.8, fontweight="bold", color=INK, ha="left", va="top")
    fig.lines.append(
        mpl.lines.Line2D([0.025, 0.975], [0.965, 0.965], transform=fig.transFigure, color=GRID, linewidth=0.7)
    )


def bracket(ax: mpl.axes.Axes, x1: float, x2: float, y: float, h: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=INK, lw=0.55, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=5.6, color=INK, clip_on=False)


def box_strip(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    value: str,
    group: str,
    title: str,
    ylabel: str | None = None,
    percent: bool = False,
    show_stats: bool = True,
) -> None:
    arrays = [frame.loc[frame[group] == diagnosis, value].dropna().to_numpy() for diagnosis in DIAG_ORDER]
    positions = np.arange(3)
    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.0},
        whiskerprops={"color": MID, "linewidth": 0.7},
        capprops={"color": MID, "linewidth": 0.7},
        boxprops={"color": MID, "linewidth": 0.7},
    )
    for patch, diagnosis in zip(bp["boxes"], DIAG_ORDER):
        patch.set_facecolor(DIAG_COLORS[diagnosis])
        patch.set_alpha(0.28)
    for i, (diagnosis, values) in enumerate(zip(DIAG_ORDER, arrays)):
        jitter = RNG.normal(0, 0.055, len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=8,
            facecolor=DIAG_COLORS[diagnosis],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.78,
            zorder=3,
        )
    ax.set_xticks(positions, DIAG_ORDER)
    ax.set_title(title, loc="left", fontweight="bold", pad=2.5)
    ax.set_ylabel(ylabel or "")
    clean_axes(ax)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if show_stats:
        overall, pairs = pairwise_stats(frame, value, group)
        finite = frame[value].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite):
            ymin, ymax = float(finite.min()), float(finite.max())
            yrange = max(ymax - ymin, abs(ymax) * 0.08, 1e-8)
            sig = [(a, b, q) for a, b, q in pairs if np.isfinite(q) and q < 0.05]
            for rank, (a, b, q) in enumerate(sig):
                y = ymax + yrange * (0.10 + 0.13 * rank)
                bracket(ax, DIAG_ORDER.index(a), DIAG_ORDER.index(b), y, yrange * 0.035, stars(q))
            upper = ymax + yrange * (0.22 + 0.13 * max(len(sig), 1))
            ax.set_ylim(ymin - yrange * 0.08, upper)
            ax.text(
                0.99,
                0.985,
                f"KW p={p_text(overall)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.3,
                color=MID,
            )


def horizontal_clone_bin(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    bin_col: str,
    bin_value: str,
    value: str,
    group: str,
    title: str,
) -> None:
    sub = frame[frame[bin_col] == bin_value].copy()
    arrays = [sub.loc[sub[group] == diagnosis, value].dropna().to_numpy() for diagnosis in DIAG_ORDER]
    positions = np.arange(3)
    bp = ax.boxplot(
        arrays,
        vert=False,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 0.9},
        whiskerprops={"color": MID, "linewidth": 0.65},
        capprops={"color": MID, "linewidth": 0.65},
        boxprops={"color": MID, "linewidth": 0.65},
    )
    for patch, diagnosis in zip(bp["boxes"], DIAG_ORDER):
        patch.set_facecolor(DIAG_COLORS[diagnosis])
        patch.set_alpha(0.26)
    for i, (diagnosis, values) in enumerate(zip(DIAG_ORDER, arrays)):
        jitter = RNG.normal(0, 0.055, len(values))
        ax.scatter(
            values,
            np.full(len(values), i) + jitter,
            s=6.5,
            color=DIAG_COLORS[diagnosis],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.2,
            zorder=3,
        )
    ax.set_yticks(positions, DIAG_ORDER)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=6.8, fontweight="bold", pad=1.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0, pad=1.5, labelsize=5.5)
    ax.tick_params(axis="x", pad=1.2, labelsize=5.2)
    overall, pairs = pairwise_stats(sub, value, group)
    sig = [f"{a[:1]}-{b[:1]} {stars(q)}" for a, b, q in pairs if np.isfinite(q) and q < 0.05]
    note = "  ".join(sig) if sig else f"KW p={p_text(overall)}"
    ax.text(0.995, 0.96, note, transform=ax.transAxes, ha="right", va="top", fontsize=4.7, color=MID)


def stacked_composition(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    state_col: str,
    class_col: str,
    value_col: str,
    state_order: list[str],
    title: str,
) -> None:
    palette = [
        "#A64B75", "#D081A7", "#E7A3A1", "#CFA0D7", "#9A83C9", "#7FA6D6",
        "#5EB1C5", "#45B8A0", "#79B87A", "#A7C46B", "#D1C56E", "#E5AD62",
        "#D8845F", "#B86A6A", "#7F8C8D", "#AFB7BA", "#6A8EAE", "#B49BC8",
        "#C98F65", "#8BB3A3", "#D6A0B5", "#8E7EAA", "#B7A36A", "#669999",
    ]
    colors = {state: palette[i % len(palette)] for i, state in enumerate(state_order)}
    classes = ["Non-expanded", "Expanded"]
    positions, tick_labels = [], []
    x = 0
    for diagnosis in DIAG_ORDER:
        for clone_class in classes:
            positions.append(x)
            tick_labels.append(f"{diagnosis}\n{clone_class}")
            x += 1
        x += 0.45
    for state in state_order:
        bottoms = []
        values = []
        for diagnosis in DIAG_ORDER:
            for clone_class in classes:
                mask = (frame["Diagnosis1"] == diagnosis) & (frame[class_col] == clone_class)
                row = frame.loc[mask & (frame[state_col] == state), value_col]
                values.append(float(row.iloc[0]) if len(row) else 0.0)
                earlier = state_order[: state_order.index(state)]
                prior = frame.loc[mask & frame[state_col].isin(earlier), value_col].sum()
                bottoms.append(float(prior))
        ax.bar(positions, values, bottom=bottoms, width=0.78, color=colors[state], edgecolor="white", linewidth=0.15, label=state)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks(positions, tick_labels, fontsize=5.6)
    ax.set_ylabel("Mean cellular composition")
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    clean_axes(ax)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=5.2,
        handlelength=0.9,
        handleheight=0.8,
        labelspacing=0.28,
        borderaxespad=0,
        ncol=2 if len(state_order) > 13 else 1,
        columnspacing=0.8,
    )


def metric_frame_tcr() -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    frame = pd.read_csv(TCR / "clonality_metrics_IBDTCR_Immunarch.csv")
    frame = frame.rename(columns={"Diagnosis": "Diagnosis1"})
    frame["Evenness"] = frame["NormShannon"]
    frame["Richness"] = np.exp(frame["Shannon"] / frame["NormShannon"])
    metrics = [
        ("Clonality", "Clonality"),
        ("Evenness", "Evenness"),
        ("Gini", "Gini coefficient"),
        ("InvSimp", "Inverse Simpson"),
        ("Richness", "Richness"),
        ("Shannon", "Shannon entropy"),
    ]
    return frame, metrics


def metric_frame_bcr() -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    frame = pd.read_csv(F5 / "Figure 5A clonality_metrics_IBDBCRonly_Publication.csv")
    frame["Evenness"] = frame["evenness"]
    metrics = [
        ("clonality", "Clonality"),
        ("Evenness", "Evenness"),
        ("gini", "Gini coefficient"),
        ("inv_simpson", "Inverse Simpson"),
        ("richness", "Richness"),
        ("shannon", "Shannon entropy"),
    ]
    return frame, metrics


def draw_metrics(gs, frame: pd.DataFrame, metrics: list[tuple[str, str]]) -> None:
    inner = gs.subgridspec(2, 3, wspace=0.44, hspace=0.55)
    for i, (column, title) in enumerate(metrics):
        ax = plt.gcf().add_subplot(inner[i // 3, i % 3])
        box_strip(ax, frame, column, "Diagnosis1", title, show_stats=True)
        if i % 3:
            ax.set_ylabel("")


def draw_clone_bins(gs, frame: pd.DataFrame, bin_col: str, value: str, group: str, bins: list[tuple[str, str]]) -> None:
    inner = gs.subgridspec(len(bins), 1, hspace=0.58)
    for i, (raw, label) in enumerate(bins):
        ax = plt.gcf().add_subplot(inner[i, 0])
        horizontal_clone_bin(ax, frame, bin_col, raw, value, group, label)
        if i == len(bins) - 1:
            ax.set_xlabel("Fraction of repertoire")


def build_s2_page1(path: Path) -> None:
    metric_df, metrics = metric_frame_tcr()
    clone_df = pd.read_csv(
        TCR_ARCH / "outputs" / "gliph2_fdr05_annotationL2_20260601_081428" / "manuscript_figures"
        / "clone_size_composition_comparison_violin_data.csv"
    )
    stack_df = pd.read_csv(
        TCR_ARCH / "outputs" / "gliph2_fdr05_annotationL2_20260601_081428"
        / "paired_alphaBeta_expanded_vs_nonexpanded_ComparisonBars_StackData.csv"
    )
    stack_df["expanded_state"] = stack_df["expanded_state"].replace({"Nonexpanded": "Non-expanded", "Non-expanded": "Non-expanded"})
    state_order = (
        stack_df.groupby("AnnotationLevel2")["mean_prop"].sum().sort_values(ascending=False).index.tolist()
    )
    fig = plt.figure(figsize=(190 * MM, 164 * MM))
    outer = fig.add_gridspec(
        2, 2, left=0.067, right=0.93, bottom=0.075, top=0.875,
        width_ratios=[1.75, 1.0], height_ratios=[1.45, 1.0], wspace=0.28, hspace=0.38,
    )
    add_page_header(fig, "Figure S2. TCR repertoire structure and clone-size context | page 1 of 4")
    add_panel_letter(fig, 0.018, 0.925, "A")
    add_panel_letter(fig, 0.625, 0.925, "B")
    add_panel_letter(fig, 0.018, 0.43, "C")
    fig.text(0.06, 0.915, "Participant-level repertoire diversity", fontsize=8.3, fontweight="bold", color=TCR_BLUE, va="top")
    fig.text(0.68, 0.915, "Clone-size architecture", fontsize=8.3, fontweight="bold", color=TCR_BLUE, va="top")
    draw_metrics(outer[0, 0], metric_df, metrics)
    bins = [("singles", "Singleton"), ("small", "Small (2–5)"), ("medium", "Medium (6–20)"), ("large", "Large (21–100)"), ("hyperexpanded", "Hyperexpanded (>100)")]
    draw_clone_bins(outer[0, 1], clone_df, "clone_bin", "prop", "Diagnosis1", bins)
    ax_c = fig.add_subplot(outer[1, :])
    cpos = ax_c.get_position()
    ax_c.set_position([cpos.x0, cpos.y0, cpos.width * 0.73, cpos.height])
    stacked_composition(
        ax_c, stack_df, "AnnotationLevel2", "expanded_state", "mean_prop", state_order,
        "Cell-state composition of exact paired αβ clonotypes",
    )
    fig.savefig(path, bbox_inches=None)
    plt.close(fig)


def build_s4_page1(path: Path) -> None:
    metric_df, metrics = metric_frame_bcr()
    clone_df = pd.read_csv(BCR_SUB / "clone_size_composition_comparison_violin_IBDTCR_Publication_sample_summary.csv")
    stack_df = pd.read_csv(TCR_ARCH / "Subfigures" / "paired_BCR heavylightchain_expanded_vs_nonexpanded_ComparisonBars_Control_UC_CD_StackData.csv")
    stack_df["expanded_state"] = stack_df["expanded_state"].replace({"Nonexpanded": "Non-expanded"})
    state_order = stack_df.groupby("Level2Annotation")["mean_prop"].sum().sort_values(ascending=False).index.tolist()
    fig = plt.figure(figsize=(190 * MM, 164 * MM))
    outer = fig.add_gridspec(
        2, 2, left=0.067, right=0.92, bottom=0.075, top=0.875,
        width_ratios=[1.75, 1.0], height_ratios=[1.45, 1.0], wspace=0.28, hspace=0.38,
    )
    add_page_header(fig, "Figure S4. BCR repertoire structure and expansion-linked programs | page 1 of 3")
    add_panel_letter(fig, 0.018, 0.925, "A")
    add_panel_letter(fig, 0.625, 0.925, "B")
    add_panel_letter(fig, 0.018, 0.43, "C")
    fig.text(0.06, 0.915, "Participant-level repertoire diversity", fontsize=8.3, fontweight="bold", color=BCR_ORANGE, va="top")
    fig.text(0.68, 0.915, "Clone-size architecture", fontsize=8.3, fontweight="bold", color=BCR_ORANGE, va="top")
    draw_metrics(outer[0, 0], metric_df, metrics)
    unique = set(clone_df["clone_size_bin"].astype(str))
    requested = [("singleton", "Singleton"), ("small", "Small (2–5)"), ("medium", "Medium (6–20)"), ("large", "Large (21–100)"), ("hyperexpanded", "Hyperexpanded (>100)")]
    bins = [(raw, label) for raw, label in requested if raw in unique]
    draw_clone_bins(outer[0, 1], clone_df, "clone_size_bin", "prop", "Diagnosis1", bins)
    ax_c = fig.add_subplot(outer[1, :])
    cpos = ax_c.get_position()
    ax_c.set_position([cpos.x0, cpos.y0, cpos.width * 0.75, cpos.height])
    stacked_composition(
        ax_c, stack_df, "Level2Annotation", "expanded_state", "mean_prop", state_order,
        "B-cell state composition of exact paired heavy–light clonotypes",
    )
    fig.savefig(path, bbox_inches=None)
    plt.close(fig)


def draw_module_summary(fig: mpl.figure.Figure, gs, diagnosis: str, letter: str) -> None:
    data = pd.read_csv(BCR_MODULES / f"bcr_clonotype_gene_module_summary_{diagnosis}_only_plotted_values.csv")
    data = data.sort_values("diff_means", ascending=True).reset_index(drop=True)
    labels = [str(x).replace("\n", " ") for x in data["module_label"]]
    labels = [textwrap.fill(x, width=27) for x in labels]
    inner = gs.subgridspec(1, 2, width_ratios=[1.18, 0.82], wspace=0.34)
    ax_d = fig.add_subplot(inner[0, 0])
    ax_m = fig.add_subplot(inner[0, 1], sharey=ax_d)
    y = np.arange(len(data))
    bar_colors = np.where(data["diff_means"] >= 0, POSITIVE, NEGATIVE)
    ax_d.barh(y, data["diff_means"], color=bar_colors, height=0.72, edgecolor="white", linewidth=0.2)
    ax_d.axvline(0, color=INK, linewidth=0.65)
    ax_d.set_yticks(y, labels, fontsize=5.2)
    ax_d.set_xlabel("Mean normalized score difference\n(expanded − non-expanded)")
    ax_d.set_title(f"{diagnosis}: direction and magnitude", loc="left", fontweight="bold", pad=3)
    ax_d.grid(axis="x", color=GRID, linewidth=0.4)
    ax_d.set_axisbelow(True)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    for yi, (x, p) in enumerate(zip(data["diff_means"], data["p_adj_wilcox"])):
        label = stars(float(p))
        if label and label != "ns":
            offset = max(abs(data["diff_means"]).max() * 0.025, 1e-6)
            ax_d.text(x + (offset if x >= 0 else -offset), yi, label, va="center", ha="left" if x >= 0 else "right", fontsize=4.6)
    ax_m.hlines(y, data["mean_nonexpanded"], data["mean_expanded"], color="#B7B7B7", linewidth=0.65)
    ax_m.scatter(data["mean_nonexpanded"], y, s=10, color="#6F7C80", label="Non-expanded", zorder=3)
    ax_m.scatter(data["mean_expanded"], y, s=10, color="#C23868", label="Expanded", zorder=3)
    ax_m.set_xlabel("Mean participant-normalized\nmodule score")
    ax_m.set_title("Group means", loc="left", fontweight="bold", pad=3)
    ax_m.grid(axis="x", color=GRID, linewidth=0.4)
    ax_m.set_axisbelow(True)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)
    ax_m.tick_params(axis="y", left=False, labelleft=False)
    ax_m.legend(loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.35, borderaxespad=0.2)
    add_panel_letter(fig, gs.get_position(fig).x0 - 0.042, gs.get_position(fig).y1 + 0.015, letter)


def build_s4_page2(path: Path) -> None:
    fig = plt.figure(figsize=(190 * MM, 230 * MM))
    outer = fig.add_gridspec(2, 1, left=0.205, right=0.975, bottom=0.06, top=0.93, hspace=0.30)
    add_page_header(fig, "Figure S4. BCR expansion-linked transcriptional programs | page 2 of 3")
    draw_module_summary(fig, outer[0, 0], "CD", "F")
    draw_module_summary(fig, outer[1, 0], "UC", "G")
    fig.savefig(path, bbox_inches=None)
    plt.close(fig)


def mean_isotype_stack(ax: mpl.axes.Axes, sample: pd.DataFrame) -> list[str]:
    iso_order = ["IGHM", "IGHD", "IGHA1", "IGHA2", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHE"]
    iso_order = [x for x in iso_order if x in set(sample["isotype"])]
    colors = {
        "IGHM": "#76B7B2", "IGHD": "#59A14F", "IGHA1": "#F28E2B", "IGHA2": "#EDC948",
        "IGHG1": "#4E79A7", "IGHG2": "#7DA7D9", "IGHG3": "#B07AA1", "IGHG4": "#D4A6C8", "IGHE": "#E15759",
    }
    mean = sample.groupby(["Diagnosis1", "isotype"])["prop"].mean().unstack(fill_value=0).reindex(DIAG_ORDER)
    left = np.zeros(len(DIAG_ORDER))
    for iso in iso_order:
        vals = mean[iso].to_numpy() if iso in mean else np.zeros(len(mean))
        ax.barh(np.arange(3), vals, left=left, color=colors[iso], edgecolor="white", linewidth=0.25, height=0.65, label=iso)
        left += vals
    ax.set_yticks(np.arange(3), DIAG_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Mean isotype composition")
    ax.set_title("Heavy-chain isotype composition", loc="left", fontweight="bold", pad=3)
    clean_axes(ax, grid=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False, fontsize=5.3, handlelength=0.9, columnspacing=0.7)
    return iso_order


def isotype_small_multiples(fig: mpl.figure.Figure, gs, sample: pd.DataFrame, iso_order: list[str]) -> None:
    shown = iso_order[:8]
    inner = gs.subgridspec(2, 4, wspace=0.42, hspace=0.65)
    for i, iso in enumerate(shown):
        ax = fig.add_subplot(inner[i // 4, i % 4])
        sub = sample[sample["isotype"] == iso]
        box_strip(ax, sub, "prop", "Diagnosis1", iso, percent=True, show_stats=False)
        if i % 4:
            ax.set_ylabel("")


def shm_small_multiples(fig: mpl.figure.Figure, gs, shm: pd.DataFrame) -> None:
    states = list(dict.fromkeys(shm["BcellState"].astype(str)))
    preferred = ["Total BCR", "Naive B", "CD5+ B Cell", "Non-switched memory B", "Switched memory B", "Atypical memory B", "Transitional B", "IgA Plasma B Cell", "IgG Plasma B Cell", "IgM Plasma B Cell"]
    states = [x for x in preferred if x in states] + [x for x in states if x not in preferred]
    states = states[:10]
    inner = gs.subgridspec(2, 5, wspace=0.48, hspace=0.62)
    for i, state in enumerate(states):
        ax = fig.add_subplot(inner[i // 5, i % 5])
        sub = shm[shm["BcellState"] == state]
        box_strip(ax, sub, "PlotMetric", "Diagnosis1", textwrap.fill(state, 18), show_stats=False)
        if i % 5:
            ax.set_ylabel("")
        if i // 5 == 1:
            ax.set_xlabel("")


def build_s4_page3(path: Path) -> None:
    switch = pd.read_csv(BCR_SWITCH / "bcr_isotype_switched_fraction_by_sample.csv")
    iso = pd.read_csv(F5 / "BCR_isotype_proportions_recommended_manuscript_sample_level_values.csv")
    shm = pd.read_csv(F5 / "fig6_shm_immunarch_by_group_by_bcelltype_recommended_600dpi_sample_level_values.csv")
    fig = plt.figure(figsize=(190 * MM, 230 * MM))
    outer = fig.add_gridspec(
        3, 1, left=0.07, right=0.975, bottom=0.06, top=0.92,
        height_ratios=[0.8, 1.25, 1.45], hspace=0.48,
    )
    add_page_header(fig, "Figure S4. BCR class switching, isotype composition, and SHM | page 3 of 3")
    top = outer[0, 0].subgridspec(1, 2, width_ratios=[0.36, 0.64], wspace=0.42)
    ax_j = fig.add_subplot(top[0, 0])
    box_strip(ax_j, switch, "switched_fraction", "Diagnosis1", "Class-switched B cells", ylabel="Switched fraction", percent=True, show_stats=True)
    ax_k = fig.add_subplot(top[0, 1])
    iso_order = mean_isotype_stack(ax_k, iso)
    add_panel_letter(fig, 0.018, 0.942, "F")
    add_panel_letter(fig, 0.395, 0.942, "G")
    isotype_small_multiples(fig, outer[1, 0], iso, iso_order)
    fig.text(0.07, outer[1, 0].get_position(fig).y1 + 0.028, "Participant-level isotype fractions", fontsize=8.0, fontweight="bold", color=INK)
    shm_small_multiples(fig, outer[2, 0], shm)
    add_panel_letter(fig, 0.018, outer[2, 0].get_position(fig).y1 + 0.034, "H")
    fig.text(0.07, outer[2, 0].get_position(fig).y1 + 0.028, "Somatic hypermutation across B-cell states", fontsize=8.0, fontweight="bold", color=INK)
    fig.text(0.975, 0.024, "Boxes show median and interquartile range; points denote participants. *FDR < 0.05; **FDR < 0.01; ***FDR < 0.001; ****FDR < 0.0001.", ha="right", fontsize=5.3, color=MID)
    fig.savefig(path, bbox_inches=None)
    plt.close(fig)


def merge_s2(new_page1: Path, old_pdf: Path, output: Path) -> None:
    old = PdfReader(str(old_pdf))
    first = PdfReader(str(new_page1))
    writer = PdfWriter()
    writer.add_page(first.pages[0])
    for page in old.pages[1:]:
        writer.add_page(page)
    with output.open("wb") as handle:
        writer.write(handle)


def merge_pages(paths: list[Path], output: Path) -> None:
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(str(path))
        for page in reader.pages:
            writer.add_page(page)
    with output.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    required = [
        S2_OLD,
        S4_OLD,
        TCR / "clonality_metrics_IBDTCR_Immunarch.csv",
        TCR_ARCH / "outputs" / "gliph2_fdr05_annotationL2_20260601_081428" / "manuscript_figures" / "clone_size_composition_comparison_violin_data.csv",
        TCR_ARCH / "outputs" / "gliph2_fdr05_annotationL2_20260601_081428" / "paired_alphaBeta_expanded_vs_nonexpanded_ComparisonBars_StackData.csv",
        F5 / "Figure 5A clonality_metrics_IBDBCRonly_Publication.csv",
        BCR_SUB / "clone_size_composition_comparison_violin_IBDTCR_Publication_sample_summary.csv",
        TCR_ARCH / "Subfigures" / "paired_BCR heavylightchain_expanded_vs_nonexpanded_ComparisonBars_Control_UC_CD_StackData.csv",
        BCR_MODULES / "bcr_clonotype_gene_module_summary_CD_only_plotted_values.csv",
        BCR_MODULES / "bcr_clonotype_gene_module_summary_UC_only_plotted_values.csv",
        BCR_SWITCH / "bcr_isotype_switched_fraction_by_sample.csv",
        F5 / "BCR_isotype_proportions_recommended_manuscript_sample_level_values.csv",
        F5 / "fig6_shm_immunarch_by_group_by_bcelltype_recommended_600dpi_sample_level_values.csv",
    ]
    require(required)
    WORK.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archived_s2 = ARCHIVE / "Figure_S2_before_redraw.pdf"
    archived_s4 = ARCHIVE / "Figure_S4_before_redraw.pdf"
    if not archived_s2.exists():
        shutil.copy2(S2_OLD, archived_s2)
    if not archived_s4.exists():
        shutil.copy2(S4_OLD, archived_s4)

    s2_page1 = WORK / "Figure_S2_page1_redrawn.pdf"
    s4_page1 = WORK / "Figure_S4_page1_redrawn.pdf"
    s4_page2 = WORK / "Figure_S4_page2_redrawn.pdf"
    s4_page3 = WORK / "Figure_S4_page3_redrawn.pdf"
    build_s2_page1(s2_page1)
    build_s4_page1(s4_page1)
    build_s4_page2(s4_page2)
    build_s4_page3(s4_page3)

    s2_tmp = WORK / "Figure_S2_complete_redrawn.pdf"
    s4_tmp = WORK / "Figure_S4_complete_redrawn.pdf"
    merge_s2(s2_page1, archived_s2, s2_tmp)
    merge_pages([s4_page1, s4_page2, s4_page3], s4_tmp)
    shutil.copy2(s2_tmp, S2_NEW)
    shutil.copy2(s4_tmp, S4_NEW)

    for path, expected in [(S2_NEW, 4), (S4_NEW, 3)]:
        reader = PdfReader(str(path))
        if len(reader.pages) != expected:
            raise RuntimeError(f"{path.name}: expected {expected} pages, found {len(reader.pages)}")
        if path.stat().st_size < 20_000:
            raise RuntimeError(f"{path.name}: unexpectedly small output")
    print(S2_NEW)
    print(S4_NEW)


if __name__ == "__main__":
    main()

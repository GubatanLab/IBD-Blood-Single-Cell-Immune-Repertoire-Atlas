from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Cell Press Redrawn Figure Set" / "Preview Alternatives"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"

INPUTS = {
    "CD vs control": Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD MiloR\CDvsControls\CDControlSCVI_milo_da_results.csv"),
    "UC vs control": Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD MiloR\UCvsControls\UCControlSCVI_milo_da_results.csv"),
    "CD vs UC": Path(r"C:/path/to/private-user-home\OneDrive\Desktop\IBD MiloR\CDvsUC\CDvsUCSCVI_milo_da_results.csv"),
}

COMPARISONS = ["CD vs control", "UC vs control", "CD vs UC"]
LINEAGES = {
    "CD4/Treg": ["CD4 Naive", "CD4 Th17", "CD4 Temra", "TReg Cytotoxic"],
    "CD8/innate-like T": ["CD8 Naive", "CD8 Tem GZMB+", "CD8 HLA-DR+", "MAIT"],
    "B/plasma": [
        "Naive B", "Switched memory B", "Atypical memory B",
        "IgM Plasma B Cell", "IgA Plasma B Cell", "IgG Plasma B Cell",
    ],
}
LINEAGE_COLORS = {"CD4/Treg": "#7656A8", "CD8/innate-like T": "#168C78", "B/plasma": "#C23B78"}
DISPLAY_LABELS = {
    "CD4 Naive": "CD4 naive",
    "CD4 Th17": "CD4 Th17",
    "CD4 Temra": "CD4 Temra",
    "TReg Cytotoxic": "Cytotoxic Treg",
    "CD8 Naive": "CD8 naive",
    "CD8 Tem GZMB+": "GZMB+ effector",
    "CD8 HLA-DR+": "HLA-DR+ CD8",
    "MAIT": "MAIT",
    "Naive B": "Naive B",
    "Switched memory B": "Switched memory",
    "Atypical memory B": "Atypical memory",
    "IgM Plasma B Cell": "IgM plasma",
    "IgA Plasma B Cell": "IgA plasma",
    "IgG Plasma B Cell": "IgG plasma",
}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.nan
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    index = np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")
    return float(values[index])


def summarize():
    participant = pd.read_csv(SOURCE / "Figure1_participant_state_effects.csv", low_memory=False)
    participant_lookup = {
        (row.cell_state, row.comparison): (float(row.effect), float(row.adjusted_p_value))
        for row in participant.itertuples(index=False)
    }

    grouped = {}
    for comparison, path in INPUTS.items():
        data = pd.read_csv(path, low_memory=False)
        data["Celltype_fraction"] = pd.to_numeric(data["Celltype_fraction"], errors="coerce")
        data["logFC"] = pd.to_numeric(data["logFC"], errors="coerce")
        data["SpatialFDR"] = pd.to_numeric(data["SpatialFDR"], errors="coerce")
        data = data[data["Celltype_fraction"].ge(0.20)].copy()
        for state, group in data.groupby("Celltype", sort=False):
            grouped[(str(state), comparison)] = group

    rows = []
    for lineage, states in LINEAGES.items():
        for state in states:
            for comparison in COMPARISONS:
                group = grouped.get((state, comparison))
                if group is None or group.empty:
                    n_total = 0
                    n_significant = 0
                    fraction_significant = np.nan
                    effect = np.nan
                else:
                    significant = group["SpatialFDR"].lt(0.05)
                    n_total = len(group)
                    n_significant = int(significant.sum())
                    fraction_significant = n_significant / n_total
                    effect = weighted_median(group["logFC"], group["Celltype_fraction"])

                participant_effect, participant_fdr = participant_lookup.get((state, comparison), (np.nan, np.nan))
                participant_tested = np.isfinite(participant_effect)
                sign_concordant = (
                    participant_tested and np.isfinite(effect)
                    and np.sign(participant_effect) == np.sign(effect)
                    and np.sign(effect) != 0
                )
                rows.append(
                    {
                        "lineage": lineage,
                        "cell_state": state,
                        "display_label": DISPLAY_LABELS[state],
                        "comparison": comparison,
                        "n_neighborhoods": n_total,
                        "n_significant_neighborhoods": n_significant,
                        "fraction_significant": fraction_significant,
                        "weighted_median_all_logFC": effect,
                        "coverage_adequate": n_total >= 10,
                        "participant_tested": participant_tested,
                        "participant_effect": participant_effect,
                        "participant_fdr": participant_fdr,
                        "participant_sign_concordant": sign_concordant,
                        "participant_confirmed": sign_concordant and np.isfinite(participant_fdr) and participant_fdr < 0.05,
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "Figure_1J_revised_preview_source.csv", index=False)
    summary.to_json(OUT / "Figure_1J_revised_preview_source.json", orient="records", indent=2)
    return summary


def build_plot(summary):
    fig = plt.figure(figsize=(7.48, 3.75), facecolor="white")
    fig.text(0.06, 0.955, "Proposed 1J", fontsize=11.5, fontweight="bold", ha="left", va="top")
    fig.text(
        0.235, 0.955, "Neighborhood-level abundance across validated adaptive states",
        fontsize=9.0, fontweight="bold", ha="left", va="top",
    )
    fig.text(
        0.50, 0.895,
        "E-G  clone-state localization   >   H  marker validation   >   I  participant confirmation   >   J  full state landscape",
        fontsize=5.6, color="#666666", ha="center", va="center",
    )
    fig.text(
        0.50, 0.855,
        "Color: annotation-weighted median log2 FC across all tested neighborhoods  |  Size: neighborhoods with SpatialFDR < 0.05",
        fontsize=5.4, color="#666666", ha="center", va="center",
    )

    positions = {
        "CD4/Treg": [0.145, 0.255, 0.205, 0.515],
        "CD8/innate-like T": [0.445, 0.255, 0.205, 0.515],
        "B/plasma": [0.745, 0.255, 0.205, 0.515],
    }
    norm = mpl.colors.Normalize(vmin=-4, vmax=4)
    cmap = mpl.colormaps["RdBu_r"]

    for lineage, states in LINEAGES.items():
        ax = fig.add_axes(positions[lineage])
        block = summary[summary["lineage"].eq(lineage)].copy()
        state_index = {state: index for index, state in enumerate(states)}
        comparison_index = {comparison: index for index, comparison in enumerate(COMPARISONS)}

        for row in block.itertuples(index=False):
            x = comparison_index[row.comparison]
            y = state_index[row.cell_state]
            if not row.coverage_adequate:
                ax.scatter(x, y, marker="x", s=30, color="#9A9A9A", linewidth=0.9, zorder=4)
                continue
            fraction = float(row.fraction_significant) if np.isfinite(row.fraction_significant) else 0.0
            size = 18 + 150 * np.sqrt(max(fraction, 0))
            facecolor = cmap(norm(float(row.weighted_median_all_logFC)))
            edgecolor = "#222222" if row.participant_tested else "white"
            linewidth = 0.90 if row.participant_tested else 0.45
            ax.scatter(x, y, s=size, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=3)
            if row.participant_confirmed:
                ax.scatter(x + 0.19, y - 0.20, marker="*", s=18, color="#222222", linewidth=0, zorder=5)

        labels = [DISPLAY_LABELS[state] for state in states]
        ax.set_xticks(range(3), ["CD vs\ncontrol", "UC vs\ncontrol", "CD vs\nUC"])
        ax.set_yticks(range(len(states)), labels)
        ax.set_xlim(-0.50, 2.50)
        ax.set_ylim(len(states) - 0.45, -0.55)
        ax.tick_params(axis="x", labelsize=5.8, length=0, pad=3)
        ax.tick_params(axis="y", labelsize=5.8, length=0, pad=3)
        for tick, state in zip(ax.get_yticklabels(), states):
            if state in set(block.loc[block["participant_tested"], "cell_state"]):
                tick.set_fontweight("bold")
        ax.grid(color="#E5E5E5", lw=0.45)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(lineage, fontsize=7.2, fontweight="bold", color=LINEAGE_COLORS[lineage], pad=12)
        ax.plot([0, 1], [1.04, 1.04], transform=ax.transAxes, color=LINEAGE_COLORS[lineage], lw=1.5, clip_on=False)

    cax = fig.add_axes([0.145, 0.115, 0.285, 0.027])
    colorbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    colorbar.set_ticks([-4, -2, 0, 2, 4])
    colorbar.ax.tick_params(labelsize=5.1, length=2, pad=1)
    colorbar.outline.set_linewidth(0.4)
    fig.text(0.287, 0.078, "negative: second-named group    |    positive: first-named group",
             fontsize=5.0, color="#555555", ha="center")

    size_handles = [
        Line2D([0], [0], marker="o", ls="", ms=2.8 + 4.8 * np.sqrt(value),
               markerfacecolor="#B8B8B8", markeredgecolor="white", label=f"{int(value * 100)}%")
        for value in (0.10, 0.25, 0.50, 0.75)
    ]
    fig.legend(
        handles=size_handles, title="Significant neighborhoods", frameon=False,
        loc="lower left", bbox_to_anchor=(0.455, 0.080), ncol=4,
        fontsize=5.0, title_fontsize=5.2, handletextpad=0.25, columnspacing=0.55,
    )
    meaning_handles = [
        Line2D([0], [0], marker="o", ls="", ms=5.2, markerfacecolor="white", markeredgecolor="#222222",
               markeredgewidth=0.9, label="Modeled in I"),
        Line2D([0], [0], marker="*", ls="", ms=5.2, color="#222222", label="I FDR < 0.05; same direction"),
        Line2D([0], [0], marker="x", ls="", ms=4.2, color="#9A9A9A", label="<10 neighborhoods"),
    ]
    fig.legend(
        handles=meaning_handles, frameon=False, loc="lower left", bbox_to_anchor=(0.455, 0.020),
        ncol=3, fontsize=4.8, handletextpad=0.25, columnspacing=0.65,
    )
    fig.text(
        0.945, 0.005,
        "Descriptive state summaries; neighborhood SpatialFDR remains the inferential unit.",
        fontsize=4.6, color="#666666", ha="right", va="bottom",
    )

    fig.savefig(OUT / "Figure_1J_revised_preview.pdf", facecolor="white")
    fig.savefig(OUT / "Figure_1J_revised_preview.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    data = summarize()
    build_plot(data)
    print(OUT)

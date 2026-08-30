from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MODULE_LABELS = {
    "module_EOMES_ZEB2_inflammatory_CD8_TRM_like": "EOMES/ZEB2 inflammatory\nCD8 TRM-like",
    "module_Effector_cytotoxicity": "Effector cytotoxicity",
    "module_Th1_Tc1_inflammatory": "Th1/Tc1 inflammatory",
    "module_GZMK_inflammatory_memory_T": "GZMK inflammatory\nmemory T",
    "module_Tissue_resident_memory_mucosal_retention": "Tissue-resident memory /\nmucosal retention",
    "module_Chronic_stimulation_exhaustion_like": "Chronic stimulation /\nexhaustion-like",
    "module_MAIT_like_unconventional_T_cell": "MAIT-like unconventional\nT cell",
    "module_Tph_Tfh_like_B_cell_help": "Tph/Tfh-like\nB-cell help",
    "module_Cell_cycle_clonal_proliferation": "Cell cycle /\nclonal proliferation",
    "module_Type_I_II_interferon_response": "Type I/II interferon\nresponse",
    "module_Activated_Treg_suppressive_T_cell": "Activated Treg /\nsuppressive T cell",
    "module_Gut_homing_intestinal_trafficking": "Gut homing /\nintestinal trafficking",
    "module_Th17_Tc17_IL23_axis": "Th17/Tc17\nIL-23 axis",
    "module_Stress_dissociation_response": "Stress /\ndissociation response",
    "module_Recent_TCR_stimulation_immediate_early": "Recent TCR stimulation /\nimmediate early",
    "module_TNF_NFkB_inflammatory_activation": "TNF/NFkB inflammatory\nactivation",
    "module_Naive_central_memory": "Naive / central memory",
}


def significance_label(value: float) -> str:
    if pd.isna(value):
        return "ns"
    if value < 0.0001:
        return "****"
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def make_figure(input_path: Path, output_prefix: Path) -> None:
    df = pd.read_csv(input_path)
    df["module_label"] = df["module"].map(MODULE_LABELS).fillna(
        df["module"].str.replace("module_", "", regex=False).str.replace("_", " ", regex=False)
    )
    df = df.sort_values("diff_means", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    positive_color = "#c95746"
    negative_color = "#3f79b5"
    expanded_color = "#b33f62"
    nonexpanded_color = "#5f6b73"

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, (ax_effect, ax_mean) = plt.subplots(
        1,
        2,
        figsize=(8.4, 6.4),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.46},
        sharey=True,
    )

    bar_colors = np.where(df["diff_means"] >= 0, positive_color, negative_color)
    ax_effect.barh(y, df["diff_means"], color=bar_colors, edgecolor="none", height=0.70)
    ax_effect.axvline(0, color="#333333", linewidth=0.8)
    ax_effect.set_yticks(y)
    ax_effect.set_yticklabels(df["module_label"])
    ax_effect.set_xlabel("Mean module score difference\n(expanded - non-expanded)")
    ax_effect.set_title("Direction and magnitude", loc="left", fontweight="bold", pad=8)
    ax_effect.grid(axis="x", color="#e5e5e5", linewidth=0.7)
    ax_effect.set_axisbelow(True)
    ax_effect.set_xlim(-0.48, 0.70)

    for yi, diff, q_value in zip(y, df["diff_means"], df["p_adj_wilcox"]):
        x = diff + (0.025 if diff >= 0 else -0.025)
        ha = "left" if diff >= 0 else "right"
        ax_effect.text(
            x,
            yi,
            significance_label(q_value),
            ha=ha,
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#202020",
        )

    for yi, nonexpanded, expanded in zip(y, df["mean_nonexpanded"], df["mean_expanded"]):
        ax_mean.plot([nonexpanded, expanded], [yi, yi], color="#b7b7b7", linewidth=1.1, zorder=1)
    ax_mean.scatter(df["mean_nonexpanded"], y, s=28, color=nonexpanded_color, label="Non-expanded", zorder=2)
    ax_mean.scatter(df["mean_expanded"], y, s=32, color=expanded_color, label="Expanded", zorder=3)
    ax_mean.set_xlabel("Mean module score")
    ax_mean.set_title("Group mean module scores", loc="left", fontweight="bold", pad=8)
    ax_mean.grid(axis="x", color="#e5e5e5", linewidth=0.7)
    ax_mean.set_axisbelow(True)
    ax_mean.set_xlim(0, 1.0)
    ax_mean.tick_params(axis="y", length=0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=expanded_color, markersize=6, label="Expanded"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=nonexpanded_color, markersize=6, label="Non-expanded"),
    ]
    ax_mean.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=8,
        handletextpad=0.5,
        borderaxespad=0.1,
    )

    fig.suptitle(
        "Gene Module Scores in Expanded vs Non-expanded Total TCR Clonotypes",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.285, right=0.985, top=0.925, bottom=0.10)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg", "tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(output_prefix.with_suffix(f".{ext}"), **kwargs)
    df.to_csv(output_prefix.with_name(output_prefix.name + "_plotted_values.csv"), index=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "C:/path/to/private-user-home/OneDrive/Desktop/TCR Module Scores/TCR Module Figures/"
            "CD4CD8_combined/clonotype_module_comparison_CD4CD8_combined_module_stats.csv"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/gene_modules_total_tcr_clonotypes/total_tcr_clonotype_gene_module_summary",
    )
    args = parser.parse_args()
    make_figure(Path(args.input), Path(args.output_prefix))


if __name__ == "__main__":
    main()

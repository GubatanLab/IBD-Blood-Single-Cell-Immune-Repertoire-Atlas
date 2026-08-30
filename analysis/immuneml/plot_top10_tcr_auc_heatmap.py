from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPARISON_LABELS = {
    "cd_vs_control": "CD vs Control",
    "uc_vs_control": "UC vs Control",
    "cd_vs_uc": "CD vs UC",
}

CHAIN_LABELS = {
    "tcr_alpha": "TCR alpha",
    "tcr_beta": "TCR beta",
    "tcr_alpha_beta": "TCR alpha+beta",
    "tcr_gamma": "TCR gamma",
    "tcr_delta": "TCR delta",
    "tcr_gamma_delta": "TCR gamma+delta",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="chain_tcr_immuneml/sklearn_results/final_metrics_table.csv",
    )
    parser.add_argument(
        "--output",
        default="chain_tcr_immuneml/sklearn_results/top10_model_auc_heatmap.png",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    metrics["model"] = metrics["chain_group"].map(CHAIN_LABELS) + " " + metrics["kmer"].str.upper()

    pivot = metrics.pivot_table(
        index="model",
        columns="comparison",
        values="Diagnosis1_roc_auc_mean",
        aggfunc="mean",
    )
    ordered_columns = ["cd_vs_control", "uc_vs_control", "cd_vs_uc"]
    pivot = pivot[ordered_columns]

    top_models = pivot.mean(axis=1).sort_values(ascending=False).head(10).index
    heatmap = pivot.loc[top_models].rename(columns=COMPARISON_LABELS)
    heatmap.insert(0, "Mean AUC", heatmap.mean(axis=1))
    heatmap = heatmap.sort_values("Mean AUC", ascending=True)

    csv_path = output_path.with_suffix(".csv")
    heatmap.sort_values("Mean AUC", ascending=False).to_csv(csv_path)

    plot_values = heatmap.drop(columns=["Mean AUC"])
    data = plot_values.to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    image = ax.imshow(data, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(plot_values.shape[1]))
    ax.set_xticklabels(plot_values.columns, fontsize=10)
    ax.set_yticks(np.arange(plot_values.shape[0]))
    ax.set_yticklabels(plot_values.index, fontsize=10)
    ax.set_title("Top 10 TCR Chain/k-mer Models by ROC AUC", fontsize=14, pad=14)
    ax.set_xlabel("Diagnosis Contrast", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            text_color = "white" if value < 0.75 else "black"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean ROC AUC across held-out assessment splits", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

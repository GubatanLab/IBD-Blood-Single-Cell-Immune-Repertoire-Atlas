from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPARISON_ORDER = ["cd_vs_control", "uc_vs_control", "cd_vs_uc"]
COMPARISON_LABELS = {
    "cd_vs_control": "CD vs Control",
    "uc_vs_control": "UC vs Control",
    "cd_vs_uc": "CD vs UC",
    "cd_inflamed_vs_noninflamed": "CD Inflamed vs Noninflamed",
    "uc_inflamed_vs_noninflamed": "UC Inflamed vs Noninflamed",
    "therapy_combined_response": "Combined Therapy Nonresponder vs Responder",
    "therapy_antitnf_response": "AntiTNF Nonresponder vs Responder",
    "therapy_ustekinumab_response": "Ustekinumab Responder vs Nonresponder",
    "therapy_vedolizumab_response": "Vedolizumab Nonresponder vs Responder",
}

CHAIN_LABELS = {
    "tcr_alpha": "alpha",
    "tcr_beta": "beta",
    "tcr_alpha_beta": "alpha+beta",
    "tcr_gamma": "gamma",
    "tcr_delta": "delta",
    "tcr_gamma_delta": "gamma+delta",
}

METRICS = [
    ("Diagnosis1_roc_auc_mean", "ROC AUC", "viridis", 0.5, 1.0),
    ("Diagnosis1_balanced_accuracy_mean", "Balanced Accuracy", "magma", 0.45, 0.8),
    ("Diagnosis1_mcc_mean", "Matthews Correlation", "coolwarm", -0.1, 0.65),
    ("Diagnosis1_f1_mean", "F1 Score", "cividis", 0.55, 0.95),
]


def model_label(row: pd.Series) -> str:
    if row.get("model_family", "cdr3_kmer") == "deeprc":
        chain = CHAIN_LABELS.get(row["chain_group"], row["chain_group"])
        return f"DeepRC {chain}"

    method = {"logistic_regression": "LR", "svm_linear": "SVM"}.get(row.get("ml_method", ""), "LR")
    if row.get("model_family", "cdr3_kmer") == "diversity_clonality":
        method = {
            "logistic_regression": "LR",
            "random_forest": "RF",
        }.get(row.get("ml_method", ""), row.get("ml_method", "ML"))
        chain = CHAIN_LABELS.get(row["chain_group"], row["chain_group"])
        return f"DIV {chain} {method}"

    seq = row["sequence_type"].upper()
    chain = CHAIN_LABELS.get(row["chain_group"], row["chain_group"])
    kmer = row["kmer"].upper()
    return f"{method} {seq} {chain} {kmer}"


def make_metric_matrix(data: pd.DataFrame, selected_models: list[str], metric: str) -> pd.DataFrame:
    matrix = data.pivot_table(index="model", columns="comparison", values=metric, aggfunc="mean")
    matrix = matrix.reindex(index=selected_models, columns=COMPARISON_ORDER)
    matrix = matrix.rename(columns=COMPARISON_LABELS)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="chain_tcr_immuneml/combined_results/combined_aa_nt_final_metrics_table.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="chain_tcr_immuneml/combined_results/tcr_manuscript_summary",
    )
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--metric-prefix", default="Diagnosis1")
    parser.add_argument("--comparison-order", nargs="+", default=None)
    parser.add_argument(
        "--comparison-labels",
        nargs="*",
        default=[],
        help="Optional key=value labels, e.g. therapy_antitnf_response='AntiTNF Nonresponder vs Responder'",
    )
    parser.add_argument(
        "--title",
        default="TCR Repertoire Diagnosis Models: CDR3 k-mers and Diversity/Clonality Features",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    metric_prefix = args.metric_prefix
    metric_map = {
        f"{metric_prefix}_roc_auc_mean": "roc_auc_mean",
        f"{metric_prefix}_balanced_accuracy_mean": "balanced_accuracy_mean",
        f"{metric_prefix}_mcc_mean": "mcc_mean",
        f"{metric_prefix}_f1_mean": "f1_mean",
    }
    data = data.rename(columns=metric_map)
    global METRICS
    METRICS = [
        ("roc_auc_mean", "ROC AUC", "viridis", 0.5, 1.0),
        ("balanced_accuracy_mean", "Balanced Accuracy", "magma", 0.45, 0.8),
        ("mcc_mean", "Matthews Correlation", "coolwarm", -0.1, 0.65),
        ("f1_mean", "F1 Score", "cividis", 0.3, 0.95),
    ]
    global COMPARISON_ORDER
    if args.comparison_order is not None:
        COMPARISON_ORDER = args.comparison_order
    for item in args.comparison_labels:
        key, value = item.split("=", 1)
        COMPARISON_LABELS[key] = value
    data["model"] = data.apply(model_label, axis=1)

    model_scores = (
        data.groupby("model", as_index=False)
        .agg(
            mean_auc=("roc_auc_mean", "mean"),
            mean_balanced_accuracy=("balanced_accuracy_mean", "mean"),
            mean_mcc=("mcc_mean", "mean"),
            mean_f1=("f1_mean", "mean"),
        )
        .sort_values("mean_auc", ascending=False)
    )
    selected_models = (
        model_scores.head(args.top_n)["model"].tolist()
        if args.top_n > 0
        else model_scores["model"].tolist()
    )
    selected_models = list(reversed(selected_models))

    selected_long = data[data["model"].isin(selected_models)].copy()
    selected_long["model"] = pd.Categorical(selected_long["model"], categories=selected_models, ordered=True)
    selected_long = selected_long.sort_values(["model", "comparison"])
    selected_long.to_csv(output_prefix.with_name(output_prefix.name + "_plotted_values.csv"), index=False)
    model_scores.to_csv(output_prefix.with_name(output_prefix.name + "_model_ranking.csv"), index=False)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig_width = max(12.8, 3.1 * len(METRICS))
    fig_height = max(5.8, 0.32 * len(selected_models) + 2.4)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(fig_width, fig_height), constrained_layout=True)
    for ax, (metric, title, cmap, vmin, vmax) in zip(axes, METRICS):
        matrix = make_metric_matrix(data, selected_models, metric)
        values = matrix.to_numpy()
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_title(title, pad=8)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(matrix.shape[0]))
        if ax is axes[0]:
            ax.set_yticklabels(matrix.index)
            ylabel = (
                "Models ranked by mean ROC AUC"
                if args.top_n <= 0
                else "Top models ranked by mean ROC AUC"
            )
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])

        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                text_color = "white" if (value < (vmin + vmax) / 2 and cmap != "coolwarm") else "black"
                if cmap == "coolwarm":
                    text_color = "black"
                fontsize = 6 if len(selected_models) > 24 else 7
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=fontsize, color=text_color)

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=7, length=2)

    fig.suptitle(
        args.title,
        fontsize=13,
        y=1.02,
    )

    for ext in ("png", "pdf", "svg"):
        out_path = output_prefix.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")
        print(f"Wrote {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()

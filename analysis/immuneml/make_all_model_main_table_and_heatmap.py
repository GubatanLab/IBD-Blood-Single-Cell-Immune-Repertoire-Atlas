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
}

CHAIN_LABELS = {
    "tcr_alpha": "TCR alpha",
    "tcr_beta": "TCR beta",
    "tcr_alpha_beta": "TCR alpha+beta",
    "tcr_gamma": "TCR gamma",
    "tcr_delta": "TCR delta",
    "tcr_gamma_delta": "TCR gamma+delta",
}

METHOD_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "svm_linear": "Linear SVM",
    "deeprc": "DeepRC",
}

METHOD_ABBREVIATIONS = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "svm_linear": "SVM",
    "deeprc": "DeepRC",
}


def model_label(row: pd.Series) -> str:
    chain = CHAIN_LABELS.get(row["chain_group"], row["chain_group"])
    method = METHOD_ABBREVIATIONS.get(row["ml_method"], row["ml_method"])
    if row["model_family"] == "diversity_clonality":
        return f"DIV {chain.replace('TCR ', '')} {method}"
    if row["model_family"] == "deeprc":
        return f"{method} {chain.replace('TCR ', '')}"
    return f"{method} {str(row['sequence_type']).upper()} {chain.replace('TCR ', '')} {str(row['kmer']).upper()}"


def feature_representation(row: pd.Series) -> str:
    if row["model_family"] == "diversity_clonality":
        return "Diversity/clonality repertoire features"
    if row["model_family"] == "deeprc":
        return "DeepRC amino-acid CDR3 repertoire sequence bags"
    return f"{str(row['sequence_type']).upper()} CDR3 {str(row['kmer']).upper()} normalized k-mer frequencies"


def model_used(row: pd.Series) -> str:
    return f"{row['feature_representation']} + {row['ml_method_label']}"


def prepare_main_table(data: pd.DataFrame) -> pd.DataFrame:
    table = data.copy()
    table["comparison_label"] = table["comparison"].map(COMPARISON_LABELS)
    table["chain_label"] = table["chain_group"].map(CHAIN_LABELS)
    table["ml_method_label"] = table["ml_method"].map(METHOD_LABELS).fillna(table["ml_method"])
    table["model_label"] = table.apply(model_label, axis=1)
    table["feature_label"] = np.where(
        table["model_family"].eq("diversity_clonality"),
        "Diversity/clonality",
        np.where(
            table["model_family"].eq("deeprc"),
            "DeepRC sequence bag",
            table["sequence_type"].astype(str).str.upper() + " CDR3 " + table["kmer"].astype(str).str.upper(),
        ),
    )
    table["feature_representation"] = table.apply(feature_representation, axis=1)
    table["classifier"] = table["ml_method_label"]
    table["model_used"] = table.apply(model_used, axis=1)

    keep = [
        "comparison",
        "comparison_label",
        "chain_group",
        "chain_label",
        "model_family",
        "feature_label",
        "sequence_type",
        "kmer",
        "ml_method",
        "ml_method_label",
        "model_label",
        "feature_representation",
        "classifier",
        "model_used",
        "balanced_accuracy_mean",
        "mcc_mean",
        "f1_mean",
        "roc_auc_mean",
    ]
    table = table[keep].sort_values(
        ["comparison", "model_family", "chain_group", "sequence_type", "kmer", "ml_method"]
    )
    return table


def make_auc_heatmap(
    main_table: pd.DataFrame,
    output_prefix: Path,
    top_n: int,
    x_label: str,
    title: str,
) -> None:
    pivot_source = main_table[["comparison", "model_label", "roc_auc_mean"]]
    matrix = pivot_source.pivot_table(
        index="model_label",
        columns="comparison",
        values="roc_auc_mean",
        aggfunc="max",
    ).reindex(columns=COMPARISON_ORDER)

    ranked_models = matrix.mean(axis=1).sort_values(ascending=False)
    top_models = ranked_models.head(top_n).index if top_n > 0 else ranked_models.index
    heatmap = matrix.loc[top_models].rename(columns=COMPARISON_LABELS)
    heatmap.insert(0, "Mean AUC", heatmap.mean(axis=1))
    heatmap.sort_values("Mean AUC", ascending=False).to_csv(
        output_prefix.with_name(output_prefix.name + "_top_models.csv")
    )

    plot_values = heatmap.sort_values("Mean AUC", ascending=True).drop(columns=["Mean AUC"])
    values = plot_values.to_numpy()

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig_width = max(8.8, 1.5 * plot_values.shape[1] + 3.8)
    fig_height = max(6.6, 0.34 * plot_values.shape[0] + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(plot_values.shape[1]))
    ax.set_xticklabels(plot_values.columns, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(plot_values.shape[0]))
    ax.set_yticklabels(plot_values.index)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, plot_values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, plot_values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.75 else "black",
                fontsize=8 if plot_values.shape[0] > 24 else 9,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean ROC AUC across held-out assessment splits")
    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        out_path = output_prefix.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")
        print(f"Wrote {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="chain_tcr_immuneml/combined_results/combined_all_model_metrics_table.csv",
    )
    parser.add_argument("--output-dir", default="chain_tcr_immuneml/combined_results")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--metric-prefix", default="Diagnosis1")
    parser.add_argument(
        "--comparison-labels",
        nargs="*",
        default=[],
        help="Optional key=value labels, e.g. cd_inflamed_vs_noninflamed='CD Inflamed vs Noninflamed'",
    )
    parser.add_argument("--comparison-order", nargs="+", default=None)
    parser.add_argument("--x-label", default="Diagnosis Contrast")
    parser.add_argument("--heatmap-prefix", default="top15_all_models_auc_heatmap_with_diversity")
    parser.add_argument("--heatmap-title", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    for item in args.comparison_labels:
        key, value = item.split("=", 1)
        COMPARISON_LABELS[key] = value
    global COMPARISON_ORDER
    if args.comparison_order is not None:
        COMPARISON_ORDER = args.comparison_order
    data = data.rename(
        columns={
            f"{args.metric_prefix}_balanced_accuracy_mean": "balanced_accuracy_mean",
            f"{args.metric_prefix}_mcc_mean": "mcc_mean",
            f"{args.metric_prefix}_f1_mean": "f1_mean",
            f"{args.metric_prefix}_roc_auc_mean": "roc_auc_mean",
        }
    )
    main_table = prepare_main_table(data)

    csv_path = output_dir / "main_model_metrics_table_with_diversity.csv"
    main_table.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    try:
        xlsx_path = output_dir / "main_model_metrics_table_with_diversity.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            main_table.to_excel(writer, index=False, sheet_name="all_models")
            top_by_auc = main_table.sort_values(["comparison", "roc_auc_mean"], ascending=[True, False])
            excel_top_n = args.top_n if args.top_n > 0 else len(main_table)
            top_by_auc.groupby("comparison").head(excel_top_n).to_excel(
                writer, index=False, sheet_name="top_by_auc"
            )
        print(f"Wrote {xlsx_path}")
    except Exception as exc:
        print(f"Skipped XLSX export: {exc}")

    heatmap_title = args.heatmap_title
    if heatmap_title is None:
        heatmap_title = (
            "All TCR Models by ROC AUC Including Diversity/Clonality"
            if args.top_n <= 0
            else "Top TCR Models by ROC AUC Including Diversity/Clonality"
        )
    make_auc_heatmap(
        main_table,
        output_dir / args.heatmap_prefix,
        args.top_n,
        args.x_label,
        heatmap_title,
    )


if __name__ == "__main__":
    main()

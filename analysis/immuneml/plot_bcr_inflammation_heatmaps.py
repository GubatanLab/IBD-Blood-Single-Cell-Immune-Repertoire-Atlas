from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS = {
    "roc_auc": ("Inflammation1_roc_auc_mean", "ROC AUC", 0.5, 1.0, "viridis"),
    "balanced_accuracy": ("Inflammation1_balanced_accuracy_mean", "Balanced Accuracy", 0.45, 0.80, "magma"),
    "mcc": ("Inflammation1_matthews_corrcoef_mean", "Matthews Correlation", -0.10, 0.65, "coolwarm"),
    "f1": ("Inflammation1_f1_mean", "F1 Score", 0.45, 0.90, "cividis"),
}

COMPARISON_LABELS = {
    "cd_inflamed_vs_noninflamed": "CD inflamed vs noninflamed",
    "uc_inflamed_vs_noninflamed": "UC inflamed vs noninflamed",
}

CHAIN_LABELS = {
    "bcr_light": "light",
    "bcr_heavy": "heavy",
    "bcr_heavy_light": "heavy+light",
    "bcr_isotype": "isotype",
    "bcr_shm": "SHM",
}

FEATURE_LABELS = {
    "aa_k3": "AA {chain} K3",
    "aa_k4": "AA {chain} K4",
    "nt_k3": "NT {chain} K3",
    "nt_k4": "NT {chain} K4",
    "repertoire_metrics": "DIV {chain}",
    "aa_k3_plus_repertoire_metrics": "AA {chain} K3 + DIV",
    "aa_k4_plus_repertoire_metrics": "AA {chain} K4 + DIV",
    "nt_k3_plus_repertoire_metrics": "NT {chain} K3 + DIV",
    "nt_k4_plus_repertoire_metrics": "NT {chain} K4 + DIV",
    "isotype_proportions": "ISO IgA/G/M/D",
    "shm_rates": "SHM rate",
}

FEATURE_ORDER = list(FEATURE_LABELS)
CHAIN_ORDER = list(CHAIN_LABELS)
COMPARISON_ORDER = list(COMPARISON_LABELS)
MODEL_LABELS = {
    "LogisticRegression": "LR",
    "immuneML SVM": "SVM",
    "DeepRC": "DeepRC",
}
def row_label(row: pd.Series) -> str:
    chain = CHAIN_LABELS[row["chain_group"]]
    model = MODEL_LABELS.get(str(row.get("ml_model", "")), str(row.get("ml_model", "")))
    suffix = f" {model}" if model and model != "nan" else ""
    return FEATURE_LABELS[row["feature_set"]].format(chain=chain) + suffix


def top_model_labels(df: pd.DataFrame, top_n: int) -> list[str]:
    work = df.copy()
    work["row_label"] = work.apply(row_label, axis=1)
    ranking = (
        work.groupby("row_label", sort=False)["Inflammation1_roc_auc_mean"]
        .mean()
        .sort_values(ascending=False)
    )
    if top_n > 0:
        ranking = ranking.head(top_n)
    return ranking.index.tolist()


def load_matrix(df: pd.DataFrame, metric_col: str, row_order: list[str] | None = None) -> pd.DataFrame:
    work = df.copy()
    work["row_label"] = work.apply(row_label, axis=1)
    work["chain_group"] = pd.Categorical(work["chain_group"], CHAIN_ORDER, ordered=True)
    work["feature_set"] = pd.Categorical(work["feature_set"], FEATURE_ORDER, ordered=True)
    work["comparison"] = pd.Categorical(work["comparison"], COMPARISON_ORDER, ordered=True)
    work = work.sort_values(["chain_group", "feature_set", "comparison"])
    matrix = work.pivot(index="row_label", columns="comparison", values=metric_col)
    matrix = matrix.rename(columns=COMPARISON_LABELS)
    if row_order is not None:
        matrix = matrix.reindex(row_order)
    return matrix


def draw_heatmap(matrix: pd.DataFrame, title: str, vmin: float, vmax: float, cmap: str, ax, show_ylabel: bool = False) -> None:
    y_fontsize = 5.2 if matrix.shape[0] > 40 else 8
    annot_fontsize = 5.4 if matrix.shape[0] > 40 else 8
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        annot_kws={"fontsize": annot_fontsize},
        fmt=".2f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"shrink": 0.42, "pad": 0.035, "aspect": 18},
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Top models ranked by mean ROC AUC" if show_ylabel else "", fontsize=10)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=y_fontsize)


def save_individual_heatmaps(df: pd.DataFrame, output_dir: Path, row_order: list[str]) -> None:
    for metric_name, (metric_col, title, vmin, vmax, cmap) in METRICS.items():
        matrix = load_matrix(df, metric_col, row_order=row_order)
        fig, ax = plt.subplots(figsize=(4.2, 7.2))
        draw_heatmap(matrix, title, vmin, vmax, cmap, ax, show_ylabel=True)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"bcr_inflammation_{metric_name}_heatmap.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)


def save_combined_heatmap(df: pd.DataFrame, output_dir: Path, row_order: list[str], filename_suffix: str = "") -> None:
    fig_height = max(8.4, 2.8 + 0.25 * len(row_order))
    fig, axes = plt.subplots(1, 4, figsize=(18.5, fig_height), gridspec_kw={"wspace": 0.60})
    for index, (ax, (_metric_name, (metric_col, title, vmin, vmax, cmap))) in enumerate(zip(axes, METRICS.items())):
        matrix = load_matrix(df, metric_col, row_order=row_order)
        draw_heatmap(matrix, title, vmin, vmax, cmap, ax, show_ylabel=index == 0)
    fig.suptitle("immuneML BCR Repertoire Inflammation Models", fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.12, wspace=0.60)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"bcr_inflammation_prediction_metric_heatmaps{filename_suffix}.{ext}", dpi=600, bbox_inches="tight")
        fig.savefig(output_dir / f"bcr_inflammation_prediction_metric_heatmaps_tcr_style{filename_suffix}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default="chain_bcr_immuneml/bcr_inflammation_lr_svm_feature_model_summary_aggregate.csv")
    parser.add_argument("--output-dir", default="chain_bcr_immuneml/figures")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--filename-suffix", default="")
    parser.add_argument("--combined-only", action="store_true")
    args = parser.parse_args()

    aggregate = Path(args.aggregate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white", context="paper")
    df = pd.read_csv(aggregate)
    row_order = top_model_labels(df, 0 if args.all_models else args.top_n)
    save_combined_heatmap(df, output_dir, row_order, args.filename_suffix)
    if not args.combined_only:
        save_individual_heatmaps(df, output_dir, row_order)
    print(f"Wrote inflammation heatmaps to {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from run_tcr_chain_kmer_models import POSITIVE_LABEL, counters_to_matrix, read_sample_sequences, sample_counter


LABEL_COLUMN = {
    "Diagnosis": "Diagnosis1",
    "Inflammation": "Inflammation1",
    "Therapy response": "TherapyResponse1",
}

MAIN_TABLES = {
    "Diagnosis": Path("chain_tcr_immuneml/combined_results/main_model_metrics_table_with_diversity.csv"),
    "Inflammation": Path("chain_tcr_immuneml/inflammation_results/combined_results/main_model_metrics_table_with_diversity.csv"),
    "Therapy response": Path("chain_tcr_immuneml/therapy_response_results/combined_results/main_model_metrics_table_with_diversity.csv"),
}

GROUP_FILE_STEM = {
    "Diagnosis": "diagnosis",
    "Inflammation": "inflammation",
    "Therapy response": "therapy_response",
}

PAIRED_FIGURE_ORDER = {
    "Diagnosis": ["CD vs Control", "UC vs Control", "CD vs UC"],
    "Therapy response": ["Combined NR vs R", "Anti-TNF NR vs R", "Ustekinumab R vs NR", "Vedolizumab NR vs R"],
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def sort_plot_items(group: str, plot_items: list[dict]) -> list[dict]:
    order = PAIRED_FIGURE_ORDER.get(group)
    if order is None:
        return plot_items
    order_lookup = {label: index for index, label in enumerate(order)}
    return sorted(plot_items, key=lambda item: order_lookup.get(item["comparison_label"], len(order_lookup)))


def load_best_rows(best_models_path: Path) -> pd.DataFrame:
    best = pd.read_csv(best_models_path)
    full_rows = []
    for group, main_table_path in MAIN_TABLES.items():
        main = pd.read_csv(main_table_path)
        group_best = best[best["analysis_group"].eq(group)].copy()
        merged = group_best[["analysis_group", "comparison", "model_label"]].merge(
            main,
            on=["comparison", "model_label"],
            how="left",
            validate="one_to_one",
        )
        missing = merged[merged["model_family"].isna()]
        if not missing.empty:
            raise ValueError(f"Could not match best model rows for {group}: {missing[['comparison', 'model_label']]}")
        full_rows.append(merged)
    return pd.concat(full_rows, ignore_index=True)


def fit_kmer_model(input_root: Path, model_row: pd.Series):
    group_root = input_root / model_row["chain_group"]
    label_column = LABEL_COLUMN[model_row["analysis_group"]]
    metadata = pd.read_csv(group_root / f"metadata_{model_row['comparison']}.csv")
    labels = metadata[label_column].astype(str).to_numpy()
    k = int(str(model_row["kmer"]).replace("k", ""))
    sequence_type = model_row["sequence_type"]

    counters = []
    for filename in metadata["filename"].astype(str):
        sequences = read_sample_sequences(group_root / "repertoires" / filename, sequence_type)
        counters.append(sample_counter(sequences, k))

    vocabulary = {
        kmer: index
        for index, kmer in enumerate(sorted({kmer for counter in counters for kmer in counter}))
    }
    x_raw = counters_to_matrix(counters, vocabulary)
    scaler = StandardScaler(with_mean=False, with_std=True)
    x_scaled = scaler.fit_transform(x_raw)

    if model_row["ml_method"] == "svm_linear":
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=8001, max_iter=50000)
    elif model_row["ml_method"] == "logistic_regression":
        model = LogisticRegression(max_iter=5000, solver="liblinear", random_state=2001)
    else:
        raise ValueError(f"Unsupported model for SHAP: {model_row['ml_method']}")
    model.fit(x_scaled, labels)

    feature_names = np.array(sorted(vocabulary, key=vocabulary.get))
    return model, x_scaled, x_raw, feature_names


def linear_shap_values(model, x_scaled: sparse.csr_matrix, positive_label: str) -> np.ndarray:
    x_dense = x_scaled.toarray()
    coef = model.coef_[0].copy()
    if list(model.classes_)[1] != positive_label:
        coef = -coef
    baseline = x_dense.mean(axis=0)
    return (x_dense - baseline) * coef


def write_importance(
    shap_values: np.ndarray,
    feature_values: sparse.csr_matrix,
    feature_names: np.ndarray,
    output_prefix: Path,
) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    feature_mean = np.asarray(feature_values.mean(axis=0)).ravel()
    order = np.argsort(mean_abs)[::-1]
    importance = pd.DataFrame(
        {
            "feature": feature_names[order],
            "mean_abs_shap": mean_abs[order],
            "mean_feature_value": feature_mean[order],
        }
    )
    importance.to_csv(output_prefix.with_name(output_prefix.name + "_importance.csv"), index=False)
    return importance


def save_shap_plots(
    shap_values: np.ndarray,
    feature_values: sparse.csr_matrix,
    feature_names: np.ndarray,
    title: str,
    output_prefix: Path,
    max_display: int,
) -> tuple[Path, Path]:
    values_for_plot = feature_values.toarray()

    bar_path = output_prefix.with_name(output_prefix.name + "_bar.png")
    beeswarm_path = output_prefix.with_name(output_prefix.name + "_beeswarm.png")

    plt.figure(figsize=(7.2, 5.4))
    shap.summary_plot(
        shap_values,
        values_for_plot,
        feature_names=feature_names.tolist(),
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(title + "\nMean absolute SHAP value", fontsize=11)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_bar.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.2, 5.4))
    shap.summary_plot(
        shap_values,
        values_for_plot,
        feature_names=feature_names.tolist(),
        max_display=max_display,
        show=False,
    )
    plt.title(title + "\nSHAP beeswarm", fontsize=11)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_beeswarm.pdf"), bbox_inches="tight")
    plt.close()

    return bar_path, beeswarm_path


def beeswarm_offsets(x_values: np.ndarray, bin_count: int = 45, step: float = 0.07) -> np.ndarray:
    if len(x_values) == 0 or np.nanmax(x_values) == np.nanmin(x_values):
        return np.zeros_like(x_values, dtype=float)
    bins = np.linspace(np.nanmin(x_values), np.nanmax(x_values), bin_count)
    bin_index = np.digitize(x_values, bins)
    offsets = np.zeros_like(x_values, dtype=float)
    for current_bin in np.unique(bin_index):
        members = np.where(bin_index == current_bin)[0]
        if len(members) <= 1:
            continue
        order = members[np.argsort(x_values[members])]
        centered = np.arange(len(order)) - (len(order) - 1) / 2
        offsets[order] = centered * step
    return np.clip(offsets, -0.34, 0.34)


def draw_compact_bar(ax, shap_values: np.ndarray, feature_names: np.ndarray, top_indices: np.ndarray) -> None:
    mean_abs = np.mean(np.abs(shap_values[:, top_indices]), axis=0)
    y_positions = np.arange(len(top_indices))
    ax.barh(y_positions, mean_abs, color="#2b8fd8", height=0.72)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names[top_indices], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|", fontsize=8)
    ax.tick_params(axis="x", labelsize=7, length=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dddddd", linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def draw_compact_beeswarm(
    ax,
    shap_values: np.ndarray,
    feature_values: sparse.csr_matrix,
    feature_names: np.ndarray,
    top_indices: np.ndarray,
) -> object:
    values = feature_values[:, top_indices].toarray()
    shap_subset = shap_values[:, top_indices]
    y_positions = np.arange(len(top_indices))
    all_feature_values = values.ravel()
    if np.nanmax(all_feature_values) > np.nanmin(all_feature_values):
        norm = plt.Normalize(np.nanmin(all_feature_values), np.nanmax(all_feature_values))
    else:
        norm = plt.Normalize(0, 1)

    rng = np.random.default_rng(2405)
    mappable = None
    for feature_rank, feature_index in enumerate(top_indices):
        x = shap_values[:, feature_index]
        jitter = beeswarm_offsets(x) + rng.normal(0, 0.008, size=len(x))
        mappable = ax.scatter(
            x,
            np.full(len(x), feature_rank) + jitter,
            c=feature_values[:, feature_index].toarray().ravel(),
            cmap=shap.plots.colors.red_blue,
            norm=norm,
            s=5,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names[top_indices], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value", fontsize=8)
    ax.tick_params(axis="x", labelsize=7, length=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return mappable


def make_paired_figure(group: str, plot_items: list[dict], output_root: Path, max_display: int) -> None:
    n_rows = len(plot_items)
    fig_height = max(3.8, 2.25 * n_rows + 0.90)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(9.2, fig_height),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.18], "wspace": 0.30, "hspace": 0.72},
    )
    for row_index, item in enumerate(plot_items):
        shap_values = item["shap_values"]
        feature_names = item["feature_names"]
        top_indices = np.argsort(np.mean(np.abs(shap_values), axis=0))[::-1][:max_display]
        row_label = f"{item['comparison_label']}: {item['model_label']}"

        bar_ax = axes[row_index, 0]
        beeswarm_ax = axes[row_index, 1]
        draw_compact_bar(bar_ax, shap_values, feature_names, top_indices)
        mappable = draw_compact_beeswarm(beeswarm_ax, shap_values, item["feature_values"], feature_names, top_indices)
        bar_ax.set_title(row_label + "\nMean |SHAP|", fontsize=9, pad=5)
        beeswarm_ax.set_title("SHAP beeswarm", fontsize=9, pad=5)
        if mappable is not None:
            colorbar = fig.colorbar(mappable, ax=beeswarm_ax, fraction=0.032, pad=0.012, aspect=12)
            colorbar.set_ticks([mappable.norm.vmin, mappable.norm.vmax])
            colorbar.set_ticklabels(["Low", "High"])
            colorbar.ax.tick_params(labelsize=6, length=0, pad=1)
            colorbar.set_label("Feature value", fontsize=7, labelpad=2)

    title_map = {
        "Diagnosis": "Diagnosis SHAP Feature Importance",
        "Inflammation": "Inflammation SHAP Feature Importance",
        "Therapy response": "Therapy Response SHAP Feature Importance",
    }
    fig.suptitle(title_map.get(group, f"{group} SHAP Feature Importance"), fontsize=13, y=0.988)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.875 if n_rows <= 2 else 0.915, bottom=0.060)

    combined_prefix = output_root / f"figure4_{GROUP_FILE_STEM[group]}_best_nondiversity_shap_paired"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(combined_prefix.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best-models",
        default=(
            "C:/path/to/private-legacy-manuscript-assets/"
            "Figure 4/figure4_best_nondiversity_models_by_balanced_accuracy.csv"
        ),
    )
    parser.add_argument("--input-root", default="chain_tcr_immuneml/airr_input")
    parser.add_argument("--output-dir", default="chain_tcr_immuneml/shap_figure4_best_nondiversity")
    parser.add_argument("--max-display", type=int, default=15)
    args = parser.parse_args()

    best_rows = load_best_rows(Path(args.best_models))
    input_root = Path(args.input_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    best_rows.to_csv(output_root / "selected_figure4_best_nondiversity_models.csv", index=False)

    importance_frames = []
    plot_items_by_group: dict[str, list[dict]] = {}
    for model_row in best_rows.itertuples(index=False):
        row = pd.Series(model_row._asdict())
        if row["model_family"] != "cdr3_kmer":
            raise ValueError(f"Expected only k-mer models after non-diversity filtering, got {row['model_family']}")

        group_dir = output_root / GROUP_FILE_STEM[row["analysis_group"]] / row["comparison"]
        group_dir.mkdir(parents=True, exist_ok=True)
        prefix = group_dir / safe_name(row["model_label"])
        title = f"{row['comparison_label']}: {row['model_label']}"
        print(f"Generating SHAP for {row['analysis_group']} - {title}")

        model, x_scaled, x_raw, feature_names = fit_kmer_model(input_root, row)
        positive_label = POSITIVE_LABEL[row["comparison"]]
        shap_values = linear_shap_values(model, x_scaled, positive_label)
        importance = write_importance(shap_values, x_raw, feature_names, prefix)
        importance.insert(0, "analysis_group", row["analysis_group"])
        importance.insert(1, "comparison", row["comparison"])
        importance.insert(2, "comparison_label", row["comparison_label"])
        importance.insert(3, "model_label", row["model_label"])
        importance_frames.append(importance.head(args.max_display))
        save_shap_plots(shap_values, x_raw, feature_names, title, prefix, args.max_display)
        plot_items_by_group.setdefault(row["analysis_group"], []).append(
            {
                "comparison": row["comparison"],
                "comparison_label": row["comparison_label"],
                "model_label": row["model_label"],
                "shap_values": shap_values,
                "feature_values": x_raw,
                "feature_names": feature_names,
            }
        )

    pd.concat(importance_frames, ignore_index=True).to_csv(
        output_root / "figure4_best_nondiversity_shap_top_features.csv", index=False
    )

    for group, plot_items in plot_items_by_group.items():
        make_paired_figure(group, sort_plot_items(group, plot_items), output_root, args.max_display)

    print(f"Wrote Figure 4 SHAP outputs to {output_root}")


if __name__ == "__main__":
    main()

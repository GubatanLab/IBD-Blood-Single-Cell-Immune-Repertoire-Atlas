from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from plot_bcr_top_model_shap import (
    ANALYSES as BASE_ANALYSES,
    CHAIN_LABELS,
    FEATURE_LABELS,
    display_feature_name,
    load_feature_sets,
    slug,
)


ANALYSES = {
    "diagnosis": {
        **BASE_ANALYSES["diagnosis"],
        "aggregate": "chain_bcr_immuneml/bcr_diagnosis_lr_svm_feature_model_summary_aggregate.csv",
        "balanced_col": "Diagnosis1_balanced_accuracy_mean",
        "roc_col": "Diagnosis1_roc_auc_mean",
        "comparison_labels": {
            "cd_vs_control": "CD vs Control",
            "uc_vs_control": "UC vs Control",
            "cd_vs_uc": "CD vs UC",
        },
        "title": "BCR Diagnosis SHAP Feature Importance",
        "order": ["cd_vs_control", "uc_vs_control", "cd_vs_uc"],
    },
    "inflammation": {
        **BASE_ANALYSES["inflammation"],
        "aggregate": "chain_bcr_immuneml/bcr_inflammation_lr_svm_feature_model_summary_aggregate.csv",
        "balanced_col": "Inflammation1_balanced_accuracy_mean",
        "roc_col": "Inflammation1_roc_auc_mean",
        "comparison_labels": {
            "cd_inflamed_vs_noninflamed": "CD Inflamed vs Noninflamed",
            "uc_inflamed_vs_noninflamed": "UC Inflamed vs Noninflamed",
        },
        "title": "BCR Inflammation SHAP Feature Importance",
        "order": ["cd_inflamed_vs_noninflamed", "uc_inflamed_vs_noninflamed"],
    },
    "therapy_response": {
        **BASE_ANALYSES["therapy_response"],
        "aggregate": "chain_bcr_immuneml/bcr_therapy_response_lr_svm_feature_model_summary_aggregate.csv",
        "balanced_col": "TherapyResponse_balanced_accuracy_mean",
        "roc_col": "TherapyResponse_roc_auc_mean",
        "comparison_labels": {
            "combined_biologic_nonresponder_vs_responder": "Combined NR vs R",
            "anti_tnf_nonresponder_vs_responder": "Anti-TNF NR vs R",
            "ustekinumab_nonresponder_vs_responder": "Ustekinumab NR vs R",
            "vedolizumab_nonresponder_vs_responder": "Vedolizumab NR vs R",
        },
        "title": "BCR Therapy Response SHAP Feature Importance",
        "order": [
            "combined_biologic_nonresponder_vs_responder",
            "anti_tnf_nonresponder_vs_responder",
            "ustekinumab_nonresponder_vs_responder",
            "vedolizumab_nonresponder_vs_responder",
        ],
    },
}

EXCLUDED_FEATURES_BY_ANALYSIS_COMPARISON = {
    ("diagnosis", "uc_vs_control"): {"isotype_proportions"},
}


def model_short_label(row: pd.Series) -> str:
    learner = "SVM" if "SVM" in str(row["ml_model"]) else "LR"
    if row["chain_group"] == "bcr_isotype":
        return f"{learner} BCR isotype proportions"
    chain = CHAIN_LABELS.get(row["chain_group"], row["chain_group"])
    feature = FEATURE_LABELS.get(row["feature_set"], row["feature_set"])
    return f"{learner} BCR {chain} {feature}"


def select_top_models(analysis: str, aggregate_path: Path) -> pd.DataFrame:
    config = ANALYSES[analysis]
    df = pd.read_csv(aggregate_path)
    feature_set = df["feature_set"].astype(str)
    chain_group = df["chain_group"].astype(str)
    filtered = df.loc[
        ~feature_set.str.contains("repertoire_metrics", case=False, na=False)
        & ~feature_set.str.contains("shm", case=False, na=False)
        & ~chain_group.str.contains("shm", case=False, na=False)
    ].copy()
    top_rows = []
    for comparison in config["order"]:
        subset = filtered.loc[filtered["comparison"].eq(comparison)].copy()
        excluded_features = EXCLUDED_FEATURES_BY_ANALYSIS_COMPARISON.get((analysis, comparison), set())
        if excluded_features:
            subset = subset.loc[~subset["feature_set"].isin(excluded_features)].copy()
        if subset.empty:
            raise ValueError(f"No non-diversity/no-SHM rows found for {analysis} {comparison}")
        top_rows.append(
            subset.sort_values(
                [config["balanced_col"], config["roc_col"]],
                ascending=[False, False],
            ).iloc[0]
        )
    return pd.DataFrame(top_rows)


def fitted_linear_contributions(
    features: list[dict[str, float]],
    labels: np.ndarray,
    positive_label: str,
    ml_model: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "SVM" in str(ml_model):
        classifier = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=seed)
    else:
        classifier = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)

    model = make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        classifier,
    )
    model.fit(features, labels)

    vectorizer: DictVectorizer = model[0]
    scaler: StandardScaler = model[1]
    fitted_classifier = model[2]
    x_vectorized = vectorizer.transform(features)
    x_scaled = scaler.transform(x_vectorized).toarray()
    raw_coefficients = fitted_classifier.coef_[0]
    if hasattr(raw_coefficients, "toarray"):
        coefficients = np.asarray(raw_coefficients.toarray()).ravel()
    else:
        coefficients = np.asarray(raw_coefficients).ravel().copy()
    if list(fitted_classifier.classes_)[1] != positive_label:
        coefficients *= -1.0
    shap_values = (x_scaled - x_scaled.mean(axis=0)) * coefficients
    feature_names = np.asarray([display_feature_name(name) for name in vectorizer.get_feature_names_out()])
    return shap_values, x_scaled, feature_names


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
    feature_values: np.ndarray,
    feature_names: np.ndarray,
    top_indices: np.ndarray,
) -> object:
    values = feature_values[:, top_indices]
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
            c=feature_values[:, feature_index],
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


def save_individual_plots(item: dict, output_base: Path, max_display: int) -> None:
    shap_values = item["shap_values"]
    feature_values = item["feature_values"]
    feature_names = item["feature_names"]
    top_indices = np.argsort(np.mean(np.abs(shap_values), axis=0))[::-1][:max_display]

    fig, ax = plt.subplots(figsize=(4.8, max(3.4, 0.22 * len(top_indices) + 1.0)))
    draw_compact_bar(ax, shap_values, feature_names, top_indices)
    ax.set_title(f"{item['comparison_label']}: {item['model_label']}\nMean |SHAP|", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_name(output_base.name + f"_bar.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, max(3.4, 0.22 * len(top_indices) + 1.0)))
    mappable = draw_compact_beeswarm(ax, shap_values, feature_values, feature_names, top_indices)
    ax.set_title(f"{item['comparison_label']}: {item['model_label']}\nSHAP beeswarm", fontsize=9)
    if mappable is not None:
        colorbar = fig.colorbar(mappable, ax=ax, fraction=0.032, pad=0.012, aspect=12)
        colorbar.set_ticks([mappable.norm.vmin, mappable.norm.vmax])
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.ax.tick_params(labelsize=6, length=0, pad=1)
        colorbar.set_label("Feature value", fontsize=7, labelpad=2)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_name(output_base.name + f"_beeswarm.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_paired_figure(analysis: str, plot_items: list[dict], output_dir: Path, max_display: int) -> None:
    n_rows = len(plot_items)
    fig_height = max(3.8, 2.25 * n_rows + 0.90)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(7.6, fig_height),
        squeeze=False,
        gridspec_kw={"width_ratios": [0.96, 1.08], "wspace": 0.24, "hspace": 0.72},
    )
    for row_index, item in enumerate(plot_items):
        shap_values = item["shap_values"]
        feature_names = item["feature_names"]
        top_indices = np.argsort(np.mean(np.abs(shap_values), axis=0))[::-1][:max_display]
        row_label = f"{item['comparison_label']}: {item['model_label']}"

        bar_ax = axes[row_index, 0]
        beeswarm_ax = axes[row_index, 1]
        draw_compact_bar(bar_ax, shap_values, feature_names, top_indices)
        mappable = draw_compact_beeswarm(
            beeswarm_ax,
            shap_values,
            item["feature_values"],
            feature_names,
            top_indices,
        )
        bar_ax.set_title(row_label + "\nMean |SHAP|", fontsize=9, pad=5)
        beeswarm_ax.set_title("SHAP beeswarm", fontsize=9, pad=5)
        if mappable is not None:
            colorbar = fig.colorbar(mappable, ax=beeswarm_ax, fraction=0.032, pad=0.012, aspect=12)
            colorbar.set_ticks([mappable.norm.vmin, mappable.norm.vmax])
            colorbar.set_ticklabels(["Low", "High"])
            colorbar.ax.tick_params(labelsize=6, length=0, pad=1)
            colorbar.set_label("Feature value", fontsize=7, labelpad=2)

    fig.suptitle(ANALYSES[analysis]["title"], fontsize=13, y=0.988)
    fig.subplots_adjust(left=0.100, right=0.985, top=0.875 if n_rows <= 2 else 0.915, bottom=0.060)

    combined_prefix = output_dir / f"bcr_{analysis}_top_nondiversity_no_shm_shap_paired"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(combined_prefix.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_bcr_immuneml/airr_input")
    parser.add_argument("--isotype-features", default="chain_bcr_immuneml/bcr_isotype_features.csv")
    parser.add_argument("--shm-features", default="chain_bcr_immuneml/bcr_shm_rates_by_patientID.csv")
    parser.add_argument("--output-dir", default="chain_bcr_immuneml/figures/shap_paired_balacc_nondiv_no_shm")
    parser.add_argument("--max-display", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_paths = {"isotype": Path(args.isotype_features), "shm": Path(args.shm_features)}

    selected_rows = []
    feature_rows = []
    for analysis, config in ANALYSES.items():
        module: Any = importlib.import_module(config["module"])
        top_models = select_top_models(analysis, Path(config["aggregate"]))
        plot_items = []
        for row in top_models.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            comparison = row_series["comparison"]
            comparison_label = config["comparison_labels"][comparison]
            model_label = model_short_label(row_series)
            print(f"Generating BCR SHAP for {analysis} - {comparison_label}: {model_label}")

            features, labels = load_feature_sets(module, input_root, row_series, extra_paths)
            shap_values, feature_values, feature_names = fitted_linear_contributions(
                features,
                labels,
                config["positive_labels"][comparison],
                row_series["ml_model"],
                args.seed,
            )
            item = {
                "analysis": analysis,
                "comparison": comparison,
                "comparison_label": comparison_label,
                "model_label": model_label,
                "shap_values": shap_values,
                "feature_values": feature_values,
                "feature_names": feature_names,
            }
            plot_items.append(item)
            save_individual_plots(
                item,
                output_dir / f"bcr_{analysis}_{slug(comparison)}_top_nondiversity_no_shm_shap",
                args.max_display,
            )

            top_indices = np.argsort(np.mean(np.abs(shap_values), axis=0))[::-1][: args.max_display]
            for rank, feature_index in enumerate(top_indices, start=1):
                feature_rows.append(
                    {
                        "analysis": analysis,
                        "comparison": comparison,
                        "comparison_label": comparison_label,
                        "model_label": model_label,
                        "rank": rank,
                        "feature": feature_names[feature_index],
                        "mean_abs_shap": float(np.mean(np.abs(shap_values[:, feature_index]))),
                    }
                )
            selected = row_series.to_dict()
            selected.update(
                {
                    "analysis": analysis,
                    "comparison_label": comparison_label,
                    "selected_model_label": model_label,
                    "selection_metric": "balanced_accuracy_mean",
                    "selection_filter": "non-diversity; no SHM",
                }
            )
            selected_rows.append(selected)

        make_paired_figure(analysis, plot_items, output_dir, args.max_display)

    pd.DataFrame(selected_rows).to_csv(output_dir / "bcr_selected_top_nondiversity_no_shm_models_by_balanced_accuracy.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(output_dir / "bcr_top_nondiversity_no_shm_shap_top_features.csv", index=False)
    print(f"Wrote BCR paired SHAP outputs to {output_dir}")


if __name__ == "__main__":
    main()

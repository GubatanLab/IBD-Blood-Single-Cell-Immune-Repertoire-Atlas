from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ANALYSES = {
    "diagnosis": {
        "aggregate": "chain_bcr_immuneml/bcr_chain_feature_model_summary_aggregate.csv",
        "roc_col": "Diagnosis1_roc_auc_mean",
        "module": "run_bcr_chain_feature_models",
        "comparison_labels": {
            "cd_vs_control": "CD vs control",
            "uc_vs_control": "UC vs control",
            "cd_vs_uc": "CD vs UC",
        },
        "positive_labels": {
            "cd_vs_control": "CD",
            "uc_vs_control": "UC",
            "cd_vs_uc": "CD",
        },
        "title": "Diagnosis",
    },
    "inflammation": {
        "aggregate": "chain_bcr_immuneml/bcr_inflammation_feature_model_summary_aggregate.csv",
        "roc_col": "Inflammation1_roc_auc_mean",
        "module": "run_bcr_inflammation_feature_models",
        "comparison_labels": {
            "cd_inflamed_vs_noninflamed": "CD inflamed vs noninflamed",
            "uc_inflamed_vs_noninflamed": "UC inflamed vs noninflamed",
        },
        "positive_labels": {
            "cd_inflamed_vs_noninflamed": "Inflamed",
            "uc_inflamed_vs_noninflamed": "Inflamed",
        },
        "title": "Inflammation",
    },
    "therapy_response": {
        "aggregate": "chain_bcr_immuneml/bcr_therapy_response_feature_model_summary_aggregate.csv",
        "roc_col": "TherapyResponse_roc_auc_mean",
        "module": "run_bcr_therapy_response_feature_models",
        "comparison_labels": {
            "combined_biologic_nonresponder_vs_responder": "Combined biologic NR vs R",
            "anti_tnf_nonresponder_vs_responder": "Anti-TNF NR vs R",
            "ustekinumab_nonresponder_vs_responder": "Ustekinumab NR vs R",
            "vedolizumab_nonresponder_vs_responder": "Vedolizumab NR vs R",
        },
        "positive_labels": {
            "combined_biologic_nonresponder_vs_responder": "NonResponder",
            "anti_tnf_nonresponder_vs_responder": "NonResponder",
            "ustekinumab_nonresponder_vs_responder": "NonResponder",
            "vedolizumab_nonresponder_vs_responder": "NonResponder",
        },
        "title": "Therapy response",
    },
}

FEATURE_LABELS = {
    "aa_k3": "AA K3",
    "aa_k4": "AA K4",
    "nt_k3": "NT K3",
    "nt_k4": "NT K4",
    "repertoire_metrics": "DIV",
    "aa_k3_plus_repertoire_metrics": "AA K3 + DIV",
    "aa_k4_plus_repertoire_metrics": "AA K4 + DIV",
    "nt_k3_plus_repertoire_metrics": "NT K3 + DIV",
    "nt_k4_plus_repertoire_metrics": "NT K4 + DIV",
    "isotype_proportions": "isotype",
    "shm_rates": "SHM",
}

CHAIN_LABELS = {
    "bcr_light": "light",
    "bcr_heavy": "heavy",
    "bcr_heavy_light": "heavy+light",
    "bcr_isotype": "isotype",
    "bcr_shm": "SHM",
}


def slug(value: str) -> str:
    return value.lower().replace("+", "plus").replace("/", "_").replace(" ", "_")


def display_feature_name(name: str) -> str:
    replacements = {
        "aa_k3_": "AA3 ",
        "aa_k4_": "AA4 ",
        "nt_k3_": "NT3 ",
        "nt_k4_": "NT4 ",
        "rep_": "DIV ",
        "iso_": "ISO ",
        "shm_": "SHM ",
    }
    for prefix, label in replacements.items():
        if name.startswith(prefix):
            return label + name[len(prefix) :]
    return name


def model_short_label(row: pd.Series) -> str:
    chain = CHAIN_LABELS.get(row["chain_group"], row["chain_group"])
    feature = FEATURE_LABELS.get(row["feature_set"], row["feature_set"])
    return f"{chain} {feature}"


def select_top_models(analysis: str, aggregate_path: Path) -> pd.DataFrame:
    config = ANALYSES[analysis]
    df = pd.read_csv(aggregate_path)
    top_rows = []
    for comparison in config["comparison_labels"]:
        subset = df.loc[df["comparison"].eq(comparison)].copy()
        if subset.empty:
            raise ValueError(f"No rows found for {analysis} {comparison}")
        top_rows.append(subset.sort_values(config["roc_col"], ascending=False).iloc[0])
    return pd.DataFrame(top_rows)


def load_feature_sets(
    module: Any,
    input_root: Path,
    row: pd.Series,
    extra_paths: dict[str, Path],
) -> tuple[list[dict[str, float]], np.ndarray]:
    chain_group = row["chain_group"]
    comparison = row["comparison"]
    feature_set = row["feature_set"]

    if chain_group == "bcr_isotype":
        feature_sets, labels, _samples = module.load_isotype_dataset(input_root, extra_paths["isotype"], comparison)
    elif chain_group == "bcr_shm":
        feature_sets, labels, _samples = module.load_shm_dataset(input_root, extra_paths["shm"], comparison)
    else:
        feature_sets, labels, _samples = module.load_dataset(input_root, chain_group, comparison)

    return feature_sets[feature_set], labels


def fitted_linear_shap(
    features: list[dict[str, float]],
    labels: np.ndarray,
    positive_label: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model = make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
    )
    model.fit(features, labels)

    vectorizer: DictVectorizer = model[0]
    scaler: StandardScaler = model[1]
    classifier: LogisticRegression = model[2]
    x_vectorized = vectorizer.transform(features)
    x_scaled = scaler.transform(x_vectorized).toarray()
    coefficients = classifier.coef_[0].copy()
    if list(classifier.classes_)[1] != positive_label:
        coefficients *= -1.0

    shap_values = (x_scaled - x_scaled.mean(axis=0)) * coefficients
    feature_names = [display_feature_name(name) for name in vectorizer.get_feature_names_out()]
    return shap_values, x_scaled, coefficients, feature_names


def top_feature_importance(shap_values: np.ndarray, feature_names: list[str], top_n: int) -> pd.DataFrame:
    importance = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "feature": [feature_names[idx] for idx in order],
            "mean_abs_shap": importance[order],
            "feature_index": order,
        }
    )


def draw_beeswarm_axis(
    ax,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    importance: pd.DataFrame,
    title: str,
    top_n: int,
) -> None:
    shown = importance.head(top_n)
    order = shown["feature_index"].to_numpy()[::-1]
    labels = shown["feature"].tolist()[::-1]
    rng = np.random.default_rng(20260704)

    for y_position, feature_idx in enumerate(order):
        values = shap_values[:, feature_idx]
        color_values = feature_values[:, feature_idx]
        if np.nanmax(color_values) > np.nanmin(color_values):
            color_values = (color_values - np.nanmin(color_values)) / (np.nanmax(color_values) - np.nanmin(color_values))
        else:
            color_values = np.full_like(color_values, 0.5)
        jitter = rng.normal(0, 0.055, size=values.shape[0])
        ax.scatter(
            values,
            np.full(values.shape[0], y_position) + jitter,
            c=color_values,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=18,
            alpha=0.82,
            edgecolors="none",
        )

    ax.axvline(0, color="#2f2f2f", linewidth=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("SHAP value on positive-class log-odds")
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)


def save_beeswarm(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    importance: pd.DataFrame,
    title: str,
    output_base: Path,
) -> None:
    fig_height = max(4.8, 0.32 * len(importance) + 1.6)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    draw_beeswarm_axis(ax, shap_values, feature_values, importance, title, top_n=len(importance))
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.025)
    cbar.set_label("Feature value", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_base.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_combined_beeswarm(
    analysis: str,
    panel_data: list[tuple[pd.Series, pd.DataFrame, np.ndarray, np.ndarray]],
    output_dir: Path,
    top_n: int,
) -> None:
    config = ANALYSES[analysis]
    panel_count = len(panel_data)
    fig, axes = plt.subplots(1, panel_count, figsize=(5.9 * panel_count, 7.2), squeeze=False)
    axes = axes[0]
    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, (ax, (row, importance, shap_values, feature_values)) in enumerate(zip(axes, panel_data)):
        draw_beeswarm_axis(
            ax,
            shap_values,
            feature_values,
            importance,
            (
                f"{panel_letters[idx]}  {config['comparison_labels'][row['comparison']]}\n"
                f"{model_short_label(row)}; AUC={float(row[config['roc_col']]):.2f}"
            ),
            top_n=top_n,
        )
        ax.tick_params(axis="x", labelsize=8)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    fig.subplots_adjust(left=0.055, right=0.925, top=0.83, bottom=0.14, wspace=0.42)
    cbar_ax = fig.add_axes([0.945, 0.31, 0.009, 0.38])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Feature value", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"BCR {config['title']} Top-Model SHAP Beeswarm Feature Importance", fontsize=15, y=0.995)
    fig.text(
        0.5,
        0.012,
        "SHAP values are exact linear contributions for the refit LogisticRegression model on DictVectorizer + StandardScaler features.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"bcr_{analysis}_top_model_shap_beeswarm.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_combined_bar(
    analysis: str,
    panel_data: list[tuple[pd.Series, pd.DataFrame]],
    output_dir: Path,
    top_n: int,
) -> None:
    config = ANALYSES[analysis]
    panel_count = len(panel_data)
    fig, axes = plt.subplots(1, panel_count, figsize=(5.2 * panel_count, 6.2), squeeze=False)
    axes = axes[0]
    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx, (ax, (row, importance)) in enumerate(zip(axes, panel_data)):
        data = importance.head(top_n).iloc[::-1]
        ax.barh(data["feature"], data["mean_abs_shap"], color="#3f7f93")
        ax.set_title(
            f"{panel_letters[idx]}  {config['comparison_labels'][row['comparison']]}\n"
            f"{model_short_label(row)}; AUC={float(row[config['roc_col']]):.2f}",
            fontsize=10,
            loc="left",
            pad=8,
        )
        ax.set_xlabel("mean |SHAP|")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(f"BCR {config['title']} Top-Model SHAP Feature Importance", fontsize=15, y=0.995)
    fig.text(
        0.5,
        0.012,
        "SHAP values are exact linear contributions for the refit LogisticRegression model on DictVectorizer + StandardScaler features.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"bcr_{analysis}_top_model_shap_bar.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="chain_bcr_immuneml/airr_input")
    parser.add_argument("--isotype-features", default="chain_bcr_immuneml/bcr_isotype_features.csv")
    parser.add_argument("--shm-features", default="chain_bcr_immuneml/bcr_shm_rates_by_patientID.csv")
    parser.add_argument("--output-dir", default="chain_bcr_immuneml/figures/shap")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--bar-top-n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_paths = {
        "isotype": Path(args.isotype_features),
        "shm": Path(args.shm_features),
    }

    summary_rows = []
    for analysis, config in ANALYSES.items():
        module = importlib.import_module(config["module"])
        top_models = select_top_models(analysis, Path(config["aggregate"]))
        panel_data = []
        beeswarm_panel_data = []

        for row in top_models.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            features, labels = load_feature_sets(module, input_root, row_series, extra_paths)
            shap_values, feature_values, _coefficients, feature_names = fitted_linear_shap(
                features,
                labels,
                config["positive_labels"][row_series["comparison"]],
                args.seed,
            )
            importance = top_feature_importance(shap_values, feature_names, args.top_n)
            comparison_label = config["comparison_labels"][row_series["comparison"]]
            title = (
                f"BCR {config['title']}: {comparison_label}\n"
                f"Top model: {model_short_label(row_series)}; AUC={float(row_series[config['roc_col']]):.2f}"
            )
            output_base = output_dir / f"bcr_{analysis}_{slug(row_series['comparison'])}_top_model_shap"
            save_beeswarm(shap_values, feature_values, importance, title, output_base)
            importance.assign(
                analysis=analysis,
                comparison=row_series["comparison"],
                comparison_label=comparison_label,
                chain_group=row_series["chain_group"],
                feature_set=row_series["feature_set"],
                model_label=row_series.get("model_label", model_short_label(row_series)),
                roc_auc_mean=float(row_series[config["roc_col"]]),
            ).to_csv(output_base.with_name(output_base.name + "_features.csv"), index=False)
            panel_data.append((row_series, importance))
            beeswarm_panel_data.append((row_series, importance, shap_values, feature_values))

            for feature_row in importance.head(args.top_n).itertuples(index=False):
                summary_rows.append(
                    {
                        "analysis": analysis,
                        "comparison": row_series["comparison"],
                        "comparison_label": comparison_label,
                        "chain_group": row_series["chain_group"],
                        "feature_set": row_series["feature_set"],
                        "model_label": row_series.get("model_label", model_short_label(row_series)),
                        "roc_auc_mean": float(row_series[config["roc_col"]]),
                        "rank": int(feature_row.rank),
                        "feature": feature_row.feature,
                        "mean_abs_shap": float(feature_row.mean_abs_shap),
                    }
                )

        save_combined_bar(analysis, panel_data, output_dir, args.bar_top_n)
        save_combined_beeswarm(analysis, beeswarm_panel_data, output_dir, args.bar_top_n)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "bcr_top_model_shap_feature_importance_summary.csv", index=False)
    print(f"Wrote SHAP plots and feature summaries to {output_dir}")


if __name__ == "__main__":
    main()

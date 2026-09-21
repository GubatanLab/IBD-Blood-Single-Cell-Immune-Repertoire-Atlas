from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer

import build_grant_combined_tcr_bcr_models as combined


RESULTS = (
    combined.OUT / "Grant_Combined_TCR_BCR_All_Model_CV_Results.csv"
)
OUT = combined.OUT
TOP_FEATURES = 20

RECEPTOR_COLORS = {
    "TCR": "#147D92",
    "BCR": "#B23A48",
}


def best_models():
    results = pd.read_csv(RESULTS)
    ordered = results.sort_values(
        ["domain", "comparison", "auc_mean", "balanced_accuracy_mean"],
        ascending=[True, True, False, False],
    )
    return ordered.groupby(
        ["domain", "comparison"], as_index=False, sort=False
    ).first()


def full_feature_matrix(domain, comparison, model_label):
    model_name, tcr_chain, bcr_chain, k = combined.parse_model_label(model_label)
    meta, label_col, positive_label = combined.comparison_metadata(
        domain, comparison
    )
    rows, y, patients = combined.paired_feature_table(
        meta,
        label_col,
        positive_label,
        tcr_chain,
        bcr_chain,
        k,
    )
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(rows)
    model = combined.make_model(model_name)
    model.fit(x, y)
    x_scaled = model.named_steps["scale"].transform(x).tocsc()
    coefficients = model.named_steps["classifier"].coef_.ravel()
    feature_names = vectorizer.get_feature_names_out()
    return (
        x_scaled,
        y,
        patients,
        coefficients,
        feature_names,
        positive_label,
        k,
    )


def exact_linear_shap_importance(x_scaled, coefficients):
    background_mean = np.asarray(x_scaled.mean(axis=0)).ravel()
    n_samples, n_features = x_scaled.shape
    mean_absolute_centered = np.empty(n_features, dtype=float)

    for feature_index in range(n_features):
        start = x_scaled.indptr[feature_index]
        stop = x_scaled.indptr[feature_index + 1]
        nonzero = x_scaled.data[start:stop]
        center = background_mean[feature_index]
        absolute_sum = (n_samples - len(nonzero)) * abs(center)
        if len(nonzero):
            absolute_sum += np.abs(nonzero - center).sum()
        mean_absolute_centered[feature_index] = absolute_sum / n_samples

    return np.abs(coefficients) * mean_absolute_centered, background_mean


def display_feature(feature):
    receptor, kmer = feature.split(":", maxsplit=1)
    return receptor, f"{receptor} {kmer}"


def shap_table(best_row):
    (
        x_scaled,
        y,
        patients,
        coefficients,
        feature_names,
        positive_label,
        k,
    ) = full_feature_matrix(
        best_row["domain"],
        best_row["comparison"],
        best_row["model_label"],
    )
    importance, background_mean = exact_linear_shap_importance(
        x_scaled, coefficients
    )
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "background_mean_standardized_feature": background_mean,
            "mean_absolute_shap": importance,
        }
    )
    parsed = table["feature"].map(display_feature)
    table["receptor"] = parsed.map(lambda value: value[0])
    table["display_feature"] = parsed.map(lambda value: value[1])
    table["direction"] = np.where(
        table["coefficient"].ge(0),
        f"+ {positive_label}",
        f"- {positive_label}",
    )
    table = table.sort_values(
        ["mean_absolute_shap", "feature"], ascending=[False, True]
    ).head(TOP_FEATURES)
    table["rank"] = np.arange(1, len(table) + 1)
    table["domain"] = best_row["domain"]
    table["comparison"] = best_row["comparison"]
    table["comparison_label"] = combined.COMPARISON_LABELS[
        best_row["comparison"]
    ]
    table["model_label"] = best_row["model_label"]
    table["selection_auc_mean"] = best_row["auc_mean"]
    table["selection_balanced_accuracy_mean"] = best_row[
        "balanced_accuracy_mean"
    ]
    table["positive_label"] = positive_label
    table["n_paired_samples"] = len(y)
    table["n_positive"] = int(y.sum())
    table["n_negative"] = int((1 - y).sum())
    table["kmer_length"] = k
    table["shap_method"] = (
        "Exact linear SHAP: coefficient * "
        "(standardized feature - cohort background mean)"
    )
    return table


def plot_barplot(table):
    domain = table["domain"].iloc[0]
    comparison = table["comparison"].iloc[0]
    comparison_label = table["comparison_label"].iloc[0]
    model_label = table["model_label"].iloc[0]
    auc_mean = table["selection_auc_mean"].iloc[0]
    positive_label = table["positive_label"].iloc[0]

    plot_data = table.sort_values("mean_absolute_shap", ascending=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 7.4), constrained_layout=True)
    colors = plot_data["receptor"].map(RECEPTOR_COLORS)
    bars = ax.barh(
        plot_data["display_feature"],
        plot_data["mean_absolute_shap"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    maximum = plot_data["mean_absolute_shap"].max()
    ax.set_xlim(0, maximum * 1.32)
    for bar, direction in zip(bars, plot_data["direction"]):
        ax.text(
            bar.get_width() + maximum * 0.018,
            bar.get_y() + bar.get_height() / 2,
            direction,
            va="center",
            ha="left",
            fontsize=7.4,
            color="#333333",
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=receptor)
        for receptor, color in RECEPTOR_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="Feature source",
        loc="lower right",
        frameon=True,
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_title(
        f"{comparison_label}: Top {TOP_FEATURES} SHAP Features\n"
        f"{model_label}; held-out mean AUC {auc_mean:.2f}; "
        f"positive class: {positive_label}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.7)
    ax.grid(axis="y", visible=False)

    stem = (
        f"Grant_Combined_TCR_BCR_{combined.DOMAIN_TITLES[domain].replace(' ', '_')}"
        f"_{comparison}_Best_Model_SHAP_Top{TOP_FEATURES}"
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=400 if extension == "png" else None,
        )
    plt.close(fig)
    return stem


def main():
    selected = best_models()
    feature_tables = []
    selection_records = []
    for _, best_row in selected.iterrows():
        print(
            "Calculating SHAP:",
            combined.COMPARISON_LABELS[best_row["comparison"]],
        )
        table = shap_table(best_row)
        stem = plot_barplot(table)
        feature_tables.append(table)
        selection_records.append(
            {
                "domain": best_row["domain"],
                "comparison": best_row["comparison"],
                "comparison_label":
                    combined.COMPARISON_LABELS[best_row["comparison"]],
                "model_label": best_row["model_label"],
                "selection_auc_mean": best_row["auc_mean"],
                "selection_balanced_accuracy_mean":
                    best_row["balanced_accuracy_mean"],
                "n_paired_samples": best_row["n_paired_samples"],
                "figure_stem": stem,
            }
        )

    pd.concat(feature_tables, ignore_index=True).to_csv(
        OUT / "Grant_Combined_TCR_BCR_Best_Model_SHAP_Top20_Features.csv",
        index=False,
    )
    pd.DataFrame(selection_records).to_csv(
        OUT / "Grant_Combined_TCR_BCR_Best_Model_SHAP_Model_Selection.csv",
        index=False,
    )
    print("Wrote combined TCR+BCR SHAP barplots to", OUT)


if __name__ == "__main__":
    main()

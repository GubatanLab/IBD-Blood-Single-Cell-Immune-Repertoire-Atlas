from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG4_ROOT = Path(
    r"C:/path/to/private-legacy-manuscript-assets\Figure 4"
)
TABLE_PATH = FIG4_ROOT / "Supplementary" / "tcr_all_models_auc_summary_with_svm_top_models.xlsx"
SHAP_PATH = FIG4_ROOT / "SHAP_best_nondiversity_models" / "selected_figure4_best_nondiversity_models.csv"
OUT_DIR = Path(__file__).resolve().parent / "figure4_heatmap_revisions"
OUT_DIR.mkdir(exist_ok=True)

METRICS = [
    ("roc_auc_mean", "ROC AUC", "viridis", 0.5, 1.0),
    ("balanced_accuracy_mean", "Balanced Accuracy", "magma", 0.45, 0.80),
    ("mcc_mean", "Matthews Correlation", "coolwarm", -0.10, 0.65),
    ("f1_mean", "F1 Score", "cividis", 0.30, 0.95),
]

PANELS = {
    "Diagnosis": {
        "title": "TCR Repertoire Diagnosis Models: CDR3 k-mers, DeepRC, and Diversity/Clonality Features",
        "summary_sheet": "Diagnosis AUC",
        "comparisons": ["CD vs Control", "UC vs Control", "CD vs UC"],
        "shap_group": "Diagnosis",
        "forced_extra": ["DIV alpha+beta RF"],
        "output": "Figure 4A tcr_diagnosis_summary_with_diversity.png",
    },
    "Inflammation": {
        "title": "TCR Repertoire Inflammation Models: CDR3 k-mers, DeepRC, and Diversity/Clonality Features",
        "summary_sheet": "Inflammation AUC",
        "comparisons": ["CD Inflamed vs Noninflamed", "UC Inflamed vs Noninflamed"],
        "shap_group": "Inflammation",
        "forced_extra": [],
        "output": "Figure 4C tcr_inflammation_manuscript_summary_with_diversity.png",
    },
    "Therapy Response": {
        "title": "TCR Repertoire Therapy Response Models: CDR3 k-mers, DeepRC, and Diversity/Clonality Features",
        "summary_sheet": "Therapy AUC",
        "comparisons": [
            "Combined NR vs R",
            "Anti-TNF NR vs R",
            "Ustekinumab R vs NR",
            "Vedolizumab NR vs R",
        ],
        "shap_group": "Therapy response",
        "forced_extra": [],
        "output": "Figure 4 E tcr_therapy_response_manuscript_summary_with_diversity.png",
    },
}


def select_models(summary_df, shap_models, forced_extra, n_models=15):
    ranked = summary_df.sort_values("Mean AUC", ascending=False).reset_index(drop=True)
    top = ranked.head(n_models)["model_label"].tolist()
    required = list(dict.fromkeys([*forced_extra, *shap_models]))

    selected = list(dict.fromkeys([*top, *required]))
    mean_auc = dict(zip(ranked["model_label"], ranked["Mean AUC"]))

    def removable(model):
        return model not in required

    while len(selected) > n_models:
        candidates = [m for m in selected if removable(m)]
        drop = min(candidates, key=lambda m: mean_auc.get(m, -np.inf))
        selected.remove(drop)

    selected.sort(key=lambda m: mean_auc.get(m, -np.inf))
    removed_from_top = [m for m in top if m not in selected]
    added = [m for m in required if m not in top]
    return selected, added, removed_from_top


def build_metric_frame(full, analysis_group, model_order, comparison_order):
    sub = full[
        (full["analysis_group"] == analysis_group)
        & (full["model_label"].isin(model_order))
        & (full["comparison_label_ordered"].isin(comparison_order))
    ].copy()
    sub["model_label"] = pd.Categorical(sub["model_label"], model_order, ordered=True)
    sub["comparison_label_ordered"] = pd.Categorical(
        sub["comparison_label_ordered"], comparison_order, ordered=True
    )
    return sub.sort_values(["model_label", "comparison_label_ordered"])


def annotate_heatmap(ax, values, im, fmt=".2f"):
    threshold = (im.norm.vmin + im.norm.vmax) / 2
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value < threshold else "black"
            ax.text(col, row, format(value, fmt), ha="center", va="center", color=color, fontsize=7.5)


def draw_panel(panel_name, panel, data, model_order):
    comparison_order = panel["comparisons"]
    fig = plt.figure(figsize=(12.913, 7.502), dpi=600)
    grid = fig.add_gridspec(
        1,
        11,
        width_ratios=[3.0, 0.14, 0.50, 3.0, 0.14, 0.50, 3.0, 0.14, 0.50, 3.0, 0.14],
        left=0.17,
        right=0.950,
        bottom=0.17,
        top=0.90,
        wspace=0.02,
    )
    fig.suptitle(panel["title"], fontsize=15, y=0.965)

    for idx, (metric, title, cmap, vmin, vmax) in enumerate(METRICS):
        ax = fig.add_subplot(grid[0, idx * 3])
        cax = fig.add_subplot(grid[0, idx * 3 + 1])
        matrix = (
            data.pivot(index="model_label", columns="comparison_label_ordered", values=metric)
            .reindex(index=model_order, columns=comparison_order)
        )
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        annotate_heatmap(ax, values, im)

        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks(range(len(comparison_order)))
        ax.set_xticklabels(comparison_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(model_order)))
        if idx == 0:
            ax.set_yticklabels(model_order, fontsize=8)
            ax.set_ylabel("Top models ranked by mean ROC AUC", fontsize=10, labelpad=4)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        ax.set_xticks(np.arange(-0.5, len(comparison_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(model_order), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=7)

    path = OUT_DIR / panel["output"]
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


def main():
    sheets = pd.read_excel(TABLE_PATH, sheet_name=None)
    full = sheets["Full Metrics"].copy()
    shap = pd.read_csv(SHAP_PATH)

    selection_records = []
    plotted_records = []
    outputs = []

    for analysis_group, panel in PANELS.items():
        summary = sheets[panel["summary_sheet"]]
        shap_models = (
            shap.loc[shap["analysis_group"] == panel["shap_group"], "model_label"]
            .drop_duplicates()
            .tolist()
        )
        model_order, added, removed = select_models(
            summary,
            shap_models=shap_models,
            forced_extra=panel["forced_extra"],
        )
        data = build_metric_frame(full, analysis_group, model_order, panel["comparisons"])
        outputs.append(draw_panel(analysis_group, panel, data, model_order))

        selection_records.extend(
            {
                "analysis_group": analysis_group,
                "model_label": model,
                "display_order": idx + 1,
                "mean_auc": float(summary.loc[summary["model_label"] == model, "Mean AUC"].iloc[0]),
                "forced_shap_model": model in shap_models,
                "forced_extra_model": model in panel["forced_extra"],
            }
            for idx, model in enumerate(model_order)
        )
        selection_records.extend(
            {
                "analysis_group": analysis_group,
                "model_label": model,
                "display_order": "",
                "mean_auc": float(summary.loc[summary["model_label"] == model, "Mean AUC"].iloc[0]),
                "forced_shap_model": False,
                "forced_extra_model": False,
                "displaced_from_original_top15": True,
            }
            for model in removed
        )

        plotted = data.copy()
        plotted["analysis_group"] = analysis_group
        plotted_records.append(plotted)
        print(f"{analysis_group}: added {added or 'none'}; displaced {removed or 'none'}")

    selection = pd.DataFrame(selection_records)
    if "displaced_from_original_top15" not in selection.columns:
        selection["displaced_from_original_top15"] = False
    selection["displaced_from_original_top15"] = np.where(
        selection["displaced_from_original_top15"].isna(),
        False,
        selection["displaced_from_original_top15"],
    ).astype(bool)
    selection.to_csv(OUT_DIR / "figure4_heatmap_model_selection_with_shap.csv", index=False)

    pd.concat(plotted_records, ignore_index=True).to_csv(
        OUT_DIR / "figure4_heatmap_plotted_values_with_shap.csv", index=False
    )
    print("\nOutputs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

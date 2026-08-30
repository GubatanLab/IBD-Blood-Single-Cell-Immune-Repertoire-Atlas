from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG6_DIR = Path(
    r"C:/path/to/private-legacy-manuscript-assets\Figure 6"
)
IMMUNEML_TABLES = Path(r"C:/path/to/private-immuneml-results\BCR Results\tables")
OUT = Path(r"C:/path/to/private-manuscript-workspace\updated_figure6_heatmaps")
OUT.mkdir(parents=True, exist_ok=True)
FORCE_SHAP_MODELS = False

METRICS = {
    "roc_auc": ("ROC AUC", "roc_auc_mean"),
    "balanced_accuracy": ("Balanced accuracy", "balanced_accuracy_mean"),
    "mcc": ("MCC", "matthews_corrcoef_mean"),
    "f1": ("F1 score", "f1_mean"),
}

METRIC_CMAPS = {
    "roc_auc": "viridis",
    "balanced_accuracy": "magma",
    "mcc": "coolwarm",
    "f1": "cividis",
}

METRIC_LIMITS = {
    "roc_auc": (0.5, 1.0),
    "balanced_accuracy": (0.45, 1.0),
    "mcc": (-0.1, 1.0),
    "f1": (0.4, 1.0),
}

COMPARISON_LABELS = {
    "cd_vs_control": "CD vs Control",
    "uc_vs_control": "UC vs Control",
    "cd_vs_uc": "CD vs UC",
    "cd_inflamed_vs_noninflamed": "CD Inflamed vs Noninflamed",
    "uc_inflamed_vs_noninflamed": "UC Inflamed vs Noninflamed",
    "combined_biologic_nonresponder_vs_responder": "Combined NR vs R",
    "anti_tnf_nonresponder_vs_responder": "Anti-TNF NR vs R",
    "ustekinumab_nonresponder_vs_responder": "Ustekinumab NR vs R",
    "vedolizumab_nonresponder_vs_responder": "Vedolizumab NR vs R",
}

ANALYSES = {
    "diagnosis": {
        "source": "bcr_diagnosis_lr_svm_feature_model_summary_aggregate.csv",
        "prefix": "Diagnosis1",
        "comparisons": ["cd_vs_control", "uc_vs_control", "cd_vs_uc"],
        "title": "BCR diagnosis prediction",
        "basename": "bcr_prediction_metric_heatmaps_tcr_style",
        "root_alias": "bcr_diagnosis_prediction_metric_heatmaps_tcr_style",
        "final_name": "Figure 6A bcr_diagnosis_prediction_metric_heatmaps_tcr_style",
    },
    "inflammation": {
        "source": "bcr_inflammation_lr_svm_feature_model_summary_aggregate.csv",
        "prefix": "Inflammation1",
        "comparisons": ["cd_inflamed_vs_noninflamed", "uc_inflamed_vs_noninflamed"],
        "title": "BCR inflammation prediction",
        "basename": "bcr_inflammation_prediction_metric_heatmaps",
        "root_alias": None,
        "final_name": "Figure 6C bcr_inflammation_prediction_metric_heatmaps",
    },
    "therapy_response": {
        "source": "bcr_therapy_response_lr_svm_feature_model_summary_aggregate.csv",
        "prefix": "TherapyResponse",
        "comparisons": [
            "combined_biologic_nonresponder_vs_responder",
            "anti_tnf_nonresponder_vs_responder",
            "ustekinumab_nonresponder_vs_responder",
            "vedolizumab_nonresponder_vs_responder",
        ],
        "title": "BCR therapy-response prediction",
        "basename": "bcr_therapy_response_prediction_metric_heatmaps",
        "root_alias": None,
        "final_name": "Figure 6E bcr_therapy_response_prediction_metric_heatmaps",
    },
}


def compact_model_label(row):
    model = "SVM" if "SVM" in str(row["ml_model"]) else "LR"
    chain = {
        "bcr_heavy": "BCR heavy",
        "bcr_light": "BCR light",
        "bcr_heavy_light": "BCR heavy+light",
    }.get(str(row["chain_group"]), str(row["chain_group"]).replace("_", " "))

    feature_set = str(row["feature_set"])
    sequence_type = str(row["sequence_type"])
    seq = {"amino_acid": "AA", "nucleotide": "NT"}.get(sequence_type, sequence_type)

    if feature_set == "repertoire_metrics":
        feature = "diversity/clonality"
    elif feature_set == "shm_features":
        feature = "SHM"
    elif feature_set == "isotype_features":
        feature = "isotype"
    else:
        k_match = re.search(r"k(\d)", feature_set)
        feature = f"{seq} K{k_match.group(1)}" if k_match else feature_set.replace("_", " ")
        if "plus_repertoire_metrics" in feature_set:
            feature += "+diversity"
    return f"{model} {chain} {feature}".strip()


def prepare_rows(analysis_name, config, selected):
    df = pd.read_csv(IMMUNEML_TABLES / config["source"])
    df["compact_model_label"] = df.apply(compact_model_label, axis=1)

    metric_cols = {}
    for metric_key, (_, suffix) in METRICS.items():
        source_col = f"{config['prefix']}_{suffix}"
        metric_cols[metric_key] = source_col

    # One row per model per comparison, then pivot for each metric. Ranking is by mean ROC AUC.
    auc = (
        df.pivot_table(
            index="compact_model_label",
            columns="comparison",
            values=metric_cols["roc_auc"],
            aggfunc="max",
        )
        .reindex(columns=config["comparisons"])
    )
    mean_auc = auc.mean(axis=1, skipna=True).sort_values(ascending=False)
    chosen = list(mean_auc.head(15).index)

    shap_labels = selected.loc[selected["analysis"].eq(analysis_name), "selected_model_label"].dropna().tolist()
    added = []
    removed = []
    if FORCE_SHAP_MODELS:
        for label in shap_labels:
            if label not in mean_auc.index:
                raise ValueError(f"Selected SHAP model not found in {analysis_name}: {label}")
            if label not in chosen:
                removable = [x for x in chosen if x not in shap_labels]
                drop = min(removable, key=lambda x: mean_auc.loc[x])
                chosen.remove(drop)
                removed.append(drop)
                chosen.append(label)
                added.append(label)

    chosen = sorted(chosen, key=lambda x: mean_auc.loc[x], reverse=True)
    matrices = {}
    for metric_key, source_col in metric_cols.items():
        matrices[metric_key] = (
            df.pivot_table(
                index="compact_model_label",
                columns="comparison",
                values=source_col,
                aggfunc="max",
            )
            .reindex(index=chosen, columns=config["comparisons"])
        )

    selection = pd.DataFrame(
        {
            "analysis": analysis_name,
            "rank": range(1, len(chosen) + 1),
            "model_label": chosen,
            "mean_roc_auc": [mean_auc.loc[x] for x in chosen],
            "force_included_from_shap": [FORCE_SHAP_MODELS and x in shap_labels for x in chosen],
        }
    )
    swaps = pd.DataFrame(
        {
            "analysis": analysis_name,
            "added_shap_model": pd.Series(added, dtype="object"),
            "removed_to_keep_15_rows": pd.Series(removed, dtype="object"),
        }
    )
    return matrices, selection, swaps


def plot_heatmap_panel(matrices, title, output_stem):
    sns.set_theme(style="white", font_scale=0.78)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    ncols = len(next(iter(matrices.values())).columns)
    width = 11.2 if ncols <= 2 else 11.6 if ncols == 3 else 13.6
    fig, axes = plt.subplots(1, 4, figsize=(width, 6.8), constrained_layout=True)
    for ax, (metric_key, matrix) in zip(axes, matrices.items()):
        labels = [COMPARISON_LABELS.get(c, c.replace("_", " ")) for c in matrix.columns]
        plot_df = matrix.copy()
        plot_df.columns = labels
        cbar_label = METRICS[metric_key][0]
        vmin, vmax = METRIC_LIMITS[metric_key]
        sns.heatmap(
            plot_df,
            ax=ax,
            cmap=METRIC_CMAPS[metric_key],
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            linewidths=0.35,
            linecolor="#FFFFFF",
            cbar_kws={"label": cbar_label, "shrink": 0.72},
            annot_kws={"fontsize": 6.8},
        )
        ax.set_title(cbar_label, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=55, labelsize=7.0)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.tick_params(axis="y", labelsize=7.2)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    for ext in ("png", "pdf", "svg"):
        dpi = 400 if ext == "png" else None
        fig.savefig(OUT / f"{output_stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main():
    selected = pd.read_csv(FIG6_DIR / "bcr_selected_top_nondiversity_no_shm_models_by_balanced_accuracy.csv")
    selections = []
    swaps = []
    outputs = []
    for analysis_name, config in ANALYSES.items():
        matrices, selection, swap = prepare_rows(analysis_name, config, selected)
        plot_heatmap_panel(matrices, config["title"], config["basename"])
        selections.append(selection)
        swaps.append(swap)
        outputs.append(config["basename"])
        if config["root_alias"]:
            for ext in ("png", "pdf", "svg"):
                src = OUT / f"{config['basename']}.{ext}"
                dst = OUT / f"{config['root_alias']}.{ext}"
                dst.write_bytes(src.read_bytes())
            outputs.append(config["root_alias"])
        for ext in ("png", "pdf", "svg"):
            src = OUT / f"{config['basename']}.{ext}"
            dst = OUT / f"{config['final_name']}.{ext}"
            dst.write_bytes(src.read_bytes())
        outputs.append(config["final_name"])

    pd.concat(selections, ignore_index=True).to_csv(OUT / "figure6_heatmap_rows_with_forced_shap_models.csv", index=False)
    pd.concat(swaps, ignore_index=True).to_csv(OUT / "figure6_heatmap_model_swaps.csv", index=False)
    print("Wrote updated heatmaps to", OUT)
    print("Output stems:")
    for output in outputs:
        print(" -", output)


if __name__ == "__main__":
    main()

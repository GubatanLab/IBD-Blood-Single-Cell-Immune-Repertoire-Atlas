from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer

import build_grant_combined_tcr_bcr_models as combined
import build_grant_combined_tcr_bcr_shap_barplots as combined_shap


OUT = combined.OUT
TOP_FEATURES = 20

TCR_SUMMARIES = {
    "diagnosis": (
        combined.IMMUNEML / "TCR Results" / "advanced_immuneml_models"
        / "diagnosis_with_svm"
        / "tcr_diagnosis_all_models_summary_with_svm_plotted_values.csv"
    ),
    "inflammation": (
        combined.IMMUNEML / "TCR Results" / "advanced_immuneml_models"
        / "inflammation_with_svm"
        / "tcr_inflammation_all_models_summary_with_svm_plotted_values.csv"
    ),
    "therapy_response": (
        combined.IMMUNEML / "TCR Results" / "advanced_immuneml_models"
        / "therapy_response_with_svm"
        / "tcr_therapy_response_all_models_summary_with_svm_plotted_values.csv"
    ),
}

BCR_SUMMARIES = {
    "diagnosis": (
        combined.IMMUNEML / "BCR Results" / "tables"
        / "bcr_diagnosis_lr_svm_feature_model_summary_aggregate.csv"
    ),
    "inflammation": (
        combined.IMMUNEML / "BCR Results" / "tables"
        / "bcr_inflammation_lr_svm_feature_model_summary_aggregate.csv"
    ),
    "therapy_response": (
        combined.IMMUNEML / "BCR Results" / "tables"
        / "bcr_therapy_response_lr_svm_feature_model_summary_aggregate.csv"
    ),
}

BCR_METRICS = {
    "diagnosis": (
        "Diagnosis1_roc_auc_mean",
        "Diagnosis1_balanced_accuracy_mean",
    ),
    "inflammation": (
        "Inflammation1_roc_auc_mean",
        "Inflammation1_balanced_accuracy_mean",
    ),
    "therapy_response": (
        "TherapyResponse_roc_auc_mean",
        "TherapyResponse_balanced_accuracy_mean",
    ),
}

TCR_COMPARISON_MAP = {
    "therapy_combined_response":
        "combined_biologic_nonresponder_vs_responder",
    "therapy_antitnf_response":
        "anti_tnf_nonresponder_vs_responder",
    "therapy_ustekinumab_response":
        "ustekinumab_nonresponder_vs_responder",
    "therapy_vedolizumab_response":
        "vedolizumab_nonresponder_vs_responder",
}

TCR_LOCI = {
    "tcr_alpha": {"TRA"},
    "tcr_beta": {"TRB"},
    "tcr_alpha_beta": {"TRA", "TRB"},
    "tcr_gamma": {"TRG"},
    "tcr_delta": {"TRD"},
    "tcr_gamma_delta": {"TRG", "TRD"},
}

BCR_LOCI = {
    "bcr_heavy": {"IGH"},
    "bcr_light": {"IGK", "IGL"},
    "bcr_heavy_light": {"IGH", "IGK", "IGL"},
}

CHAIN_LABELS = {
    "tcr_alpha": "alpha",
    "tcr_beta": "beta",
    "tcr_alpha_beta": "alpha+beta",
    "tcr_gamma": "gamma",
    "tcr_delta": "delta",
    "tcr_gamma_delta": "gamma+delta",
    "bcr_heavy": "heavy",
    "bcr_light": "light",
    "bcr_heavy_light": "heavy+light",
}


def select_best_tcr_models():
    records = []
    for domain, path in TCR_SUMMARIES.items():
        table = pd.read_csv(path)
        table["comparison"] = table["comparison"].replace(TCR_COMPARISON_MAP)
        table = table[
            table["model_family"].eq("cdr3_kmer")
            & table["sequence_type"].isin(["aa", "nt"])
            & table["kmer"].isin(["k3", "k4"])
            & table["ml_method"].isin(
                ["logistic_regression", "svm_linear"]
            )
            & table["chain_group"].isin(TCR_LOCI)
        ].copy()
        table = table.sort_values(
            ["comparison", "roc_auc_mean", "balanced_accuracy_mean"],
            ascending=[True, False, False],
        )
        for _, row in table.groupby(
            "comparison", as_index=False, sort=False
        ).first().iterrows():
            records.append(
                {
                    "receptor_model": "TCR",
                    "domain": domain,
                    "comparison": row["comparison"],
                    "chain_group": row["chain_group"],
                    "sequence_type": row["sequence_type"],
                    "k": int(str(row["kmer"]).replace("k", "")),
                    "model_name": (
                        "LR"
                        if row["ml_method"] == "logistic_regression"
                        else "SVM"
                    ),
                    "model_label": row["model"],
                    "selection_auc_mean": row["roc_auc_mean"],
                    "selection_balanced_accuracy_mean":
                        row["balanced_accuracy_mean"],
                    "selection_source": str(path),
                }
            )
    return pd.DataFrame(records)


def select_best_bcr_models():
    records = []
    for domain, path in BCR_SUMMARIES.items():
        auc_col, balanced_col = BCR_METRICS[domain]
        table = pd.read_csv(path)
        table = table[
            table["feature_set"].astype(str).str.fullmatch(
                r"(aa|nt)_k[34]", case=False, na=False
            )
            & table["chain_group"].isin(BCR_LOCI)
            & table["ml_model"].isin(
                ["LogisticRegression", "immuneML SVM"]
            )
        ].copy()
        table = table.sort_values(
            ["comparison", auc_col, balanced_col],
            ascending=[True, False, False],
        )
        for _, row in table.groupby(
            "comparison", as_index=False, sort=False
        ).first().iterrows():
            sequence_type, k_text = row["feature_set"].split("_")
            model_name = (
                "LR" if row["ml_model"] == "LogisticRegression" else "SVM"
            )
            model_label = (
                f"{model_name} "
                f"{'AA' if sequence_type == 'aa' else 'NT'} "
                f"{CHAIN_LABELS[row['chain_group']]} "
                f"K{k_text.replace('k', '')}"
            )
            records.append(
                {
                    "receptor_model": "BCR",
                    "domain": domain,
                    "comparison": row["comparison"],
                    "chain_group": row["chain_group"],
                    "sequence_type": sequence_type,
                    "k": int(k_text.replace("k", "")),
                    "model_name": model_name,
                    "model_label": model_label,
                    "selection_auc_mean": row[auc_col],
                    "selection_balanced_accuracy_mean": row[balanced_col],
                    "selection_source": str(path),
                }
            )
    return pd.DataFrame(records)


def productive_sequences(path, loci, sequence_type):
    sequence_column = "junction_aa" if sequence_type == "aa" else "junction"
    table = pd.read_csv(
        path,
        sep="\t",
        usecols=[sequence_column, "productive", "locus"],
    )
    table = table[
        table["productive"].astype(str).str.upper().isin(["T", "TRUE"])
        & table["locus"].isin(loci)
    ]
    return sorted(set(table[sequence_column].dropna().astype(str)))


def normalized_kmer_features(path, loci, sequence_type, k, receptor):
    counts = Counter()
    for sequence in productive_sequences(path, loci, sequence_type):
        counts.update(combined.kmer_counts(sequence, k))
    total = sum(counts.values())
    return (
        {f"{receptor}:{key}": value / total for key, value in counts.items()}
        if total
        else {}
    )


def single_receptor_matrix(selection):
    receptor = selection["receptor_model"]
    domain = selection["domain"]
    comparison = selection["comparison"]
    meta, label_col, positive_label = combined.comparison_metadata(
        domain, comparison
    )
    repertoire_dir = (
        combined.TCR_REPERTOIRES
        if receptor == "TCR"
        else combined.BCR_REPERTOIRES
    )
    suffix = "TCR" if receptor == "TCR" else "BCR"
    loci = (
        TCR_LOCI[selection["chain_group"]]
        if receptor == "TCR"
        else BCR_LOCI[selection["chain_group"]]
    )
    rows = []
    labels = []
    patients = []
    for _, row in meta.drop_duplicates("PatientID").iterrows():
        patient = str(row["PatientID"])
        path = repertoire_dir / f"{patient}_{suffix}_AIRR.tsv"
        if not path.exists():
            continue
        rows.append(
            normalized_kmer_features(
                path,
                loci,
                selection["sequence_type"],
                int(selection["k"]),
                receptor,
            )
        )
        labels.append(1 if row[label_col] == positive_label else 0)
        patients.append(patient)
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(rows)
    return (
        x,
        np.asarray(labels),
        patients,
        vectorizer.get_feature_names_out(),
        positive_label,
    )


def shap_table(selection):
    x, y, patients, feature_names, positive_label = (
        single_receptor_matrix(selection)
    )
    model = combined.make_model(selection["model_name"])
    model.fit(x, y)
    x_scaled = model.named_steps["scale"].transform(x).tocsc()
    coefficients = model.named_steps["classifier"].coef_.ravel()
    importance, background_mean = (
        combined_shap.exact_linear_shap_importance(
            x_scaled, coefficients
        )
    )
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "background_mean_standardized_feature": background_mean,
            "mean_absolute_shap": importance,
        }
    )
    parsed = table["feature"].map(combined_shap.display_feature)
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
    for column in [
        "receptor_model",
        "domain",
        "comparison",
        "chain_group",
        "sequence_type",
        "k",
        "model_name",
        "model_label",
        "selection_auc_mean",
        "selection_balanced_accuracy_mean",
        "selection_source",
    ]:
        table[column] = selection[column]
    table["comparison_label"] = combined.COMPARISON_LABELS[
        selection["comparison"]
    ]
    table["positive_label"] = positive_label
    table["n_samples_refit"] = len(y)
    table["n_positive_refit"] = int(y.sum())
    table["n_negative_refit"] = int((1 - y).sum())
    table["shap_method"] = (
        "Exact linear SHAP: coefficient * "
        "(standardized feature - cohort background mean)"
    )
    return table


def plot_shap(table):
    receptor_model = table["receptor_model"].iloc[0]
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
    colors = plot_data["receptor"].map(
        combined_shap.RECEPTOR_COLORS
    )
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
    source_color = combined_shap.RECEPTOR_COLORS[receptor_model]
    ax.legend(
        handles=[
            plt.Rectangle(
                (0, 0), 1, 1, color=source_color, label=receptor_model
            )
        ],
        title="Feature source",
        loc="lower right",
        frameon=True,
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_title(
        f"{receptor_model} | {comparison_label}: "
        f"Top {TOP_FEATURES} SHAP Features\n"
        f"{model_label}; held-out mean AUC {auc_mean:.2f}; "
        f"positive class: {positive_label}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.7)
    ax.grid(axis="y", visible=False)

    stem = (
        f"Grant_{receptor_model}_{combined.DOMAIN_TITLES[domain].replace(' ', '_')}"
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


def combined_selection_and_features():
    selection = pd.read_csv(
        OUT
        / "Grant_Combined_TCR_BCR_Best_Model_SHAP_Model_Selection.csv"
    )
    selection = selection.rename(
        columns={
            "n_paired_samples": "n_samples_refit",
            "figure_stem": "figure_stem",
        }
    )
    selection["receptor_model"] = "TCR+BCR"
    selection["selection_source"] = str(
        combined.OUT / "Grant_Combined_TCR_BCR_All_Model_CV_Results.csv"
    )
    features = pd.read_csv(
        OUT / "Grant_Combined_TCR_BCR_Best_Model_SHAP_Top20_Features.csv"
    )
    features["receptor_model"] = "TCR+BCR"
    return selection, features


def main():
    selections = pd.concat(
        [select_best_tcr_models(), select_best_bcr_models()],
        ignore_index=True,
    )
    feature_tables = []
    figure_stems = []
    for _, selection in selections.iterrows():
        print(
            "Calculating SHAP:",
            selection["receptor_model"],
            combined.COMPARISON_LABELS[selection["comparison"]],
        )
        table = shap_table(selection)
        figure_stems.append(plot_shap(table))
        feature_tables.append(table)

    selections = selections.copy()
    selections["figure_stem"] = figure_stems
    combined_selection, combined_features = combined_selection_and_features()
    all_selections = pd.concat(
        [selections, combined_selection], ignore_index=True, sort=False
    )
    all_features = pd.concat(
        feature_tables + [combined_features],
        ignore_index=True,
        sort=False,
    )
    all_selections.to_csv(
        OUT / "Grant_All_TCR_BCR_Combined_Best_Model_SHAP_Selection.csv",
        index=False,
    )
    all_features.to_csv(
        OUT / "Grant_All_TCR_BCR_Combined_Best_Model_SHAP_Top20_Features.csv",
        index=False,
    )
    print("Wrote all-receptor SHAP barplots to", OUT)


if __name__ == "__main__":
    main()

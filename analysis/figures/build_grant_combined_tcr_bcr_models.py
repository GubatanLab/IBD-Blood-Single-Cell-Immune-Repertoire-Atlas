from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


IMMUNEML = Path(r"C:/path/to/private-immuneml-workspace")
AIRR_ROOT = IMMUNEML / "immuneML_runs" / "airr_input"
TCR_AIRR = AIRR_ROOT / "tcr"
BCR_AIRR = AIRR_ROOT / "bcr"
TCR_REPERTOIRES = TCR_AIRR / "repertoires"
BCR_REPERTOIRES = BCR_AIRR / "repertoires"
TCR_THERAPY_META = IMMUNEML / "TCR Results" / "airr_input" / "tcr_beta"
OUT = Path(
    r"C:/path/to/private-manuscript-workspace"
    r"/grant_combined_tcr_bcr_figures"
)
OUT.mkdir(parents=True, exist_ok=True)

DOMAIN_CONFIGS = {
    "diagnosis": ["cd_vs_control", "uc_vs_control", "cd_vs_uc"],
    "inflammation": [
        "cd_inflamed_vs_noninflamed",
        "uc_inflamed_vs_noninflamed",
    ],
    "therapy_response": [
        "combined_biologic_nonresponder_vs_responder",
        "anti_tnf_nonresponder_vs_responder",
        "ustekinumab_nonresponder_vs_responder",
        "vedolizumab_nonresponder_vs_responder",
    ],
}

COMPARISON_LABELS = {
    "cd_vs_control": "CD vs Control",
    "uc_vs_control": "UC vs Control",
    "cd_vs_uc": "CD vs UC",
    "cd_inflamed_vs_noninflamed": "CD inflamed vs noninflamed",
    "uc_inflamed_vs_noninflamed": "UC inflamed vs noninflamed",
    "combined_biologic_nonresponder_vs_responder": "Combined NR vs R",
    "anti_tnf_nonresponder_vs_responder": "Anti-TNF NR vs R",
    "ustekinumab_nonresponder_vs_responder": "Ustekinumab NR vs R",
    "vedolizumab_nonresponder_vs_responder": "Vedolizumab NR vs R",
}

DOMAIN_TITLES = {
    "diagnosis": "Diagnosis",
    "inflammation": "Inflammation",
    "therapy_response": "Therapy Response",
}

TCR_CHAINS = {
    "alpha": {"TRA"},
    "beta": {"TRB"},
    "alpha+beta": {"TRA", "TRB"},
}

BCR_CHAINS = {
    "heavy": {"IGH"},
    "light": {"IGK", "IGL"},
    "heavy+light": {"IGH", "IGK", "IGL"},
}

FEATURE_CACHE = {}
MATRIX_CACHE = {}
EVALUATION_CACHE = {}


def normalize_response(value):
    text = str(value).strip().lower()
    if text == "nonresponder":
        return "NonResponder"
    if text == "responder":
        return "Responder"
    return value


def comparison_metadata(domain, comparison):
    if domain == "diagnosis":
        meta = pd.read_csv(TCR_AIRR / f"metadata_{comparison}.csv")
        positive = {
            "cd_vs_control": "CD",
            "uc_vs_control": "UC",
            "cd_vs_uc": "CD",
        }[comparison]
        return meta, "Diagnosis1", positive

    if domain == "inflammation":
        meta = pd.read_csv(TCR_AIRR / "metadata_all.csv")
        diagnosis = "CD" if comparison.startswith("cd_") else "UC"
        meta = meta[
            meta["Diagnosis1"].eq(diagnosis)
            & meta["Inflammation1"].isin(["Inflamed", "Noninflamed"])
        ].copy()
        return meta, "Inflammation1", "Inflamed"

    therapy_files = {
        "combined_biologic_nonresponder_vs_responder":
            "metadata_therapy_combined_response.csv",
        "anti_tnf_nonresponder_vs_responder":
            "metadata_therapy_antitnf_response.csv",
        "ustekinumab_nonresponder_vs_responder":
            "metadata_therapy_ustekinumab_response.csv",
        "vedolizumab_nonresponder_vs_responder":
            "metadata_therapy_vedolizumab_response.csv",
    }
    meta = pd.read_csv(TCR_THERAPY_META / therapy_files[comparison])
    meta["TherapyResponse1"] = meta["TherapyResponse1"].map(normalize_response)
    return meta, "TherapyResponse1", "NonResponder"


def kmer_counts(sequence, k):
    if not isinstance(sequence, str) or len(sequence) < k:
        return Counter()
    return Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def repertoire_features(receptor, filename, loci, k):
    cache_key = (receptor, filename, tuple(sorted(loci)), k)
    if cache_key in FEATURE_CACHE:
        return FEATURE_CACHE[cache_key]

    repertoire_dir = TCR_REPERTOIRES if receptor == "TCR" else BCR_REPERTOIRES
    path = repertoire_dir / filename
    df = pd.read_csv(path, sep="\t", usecols=["junction_aa", "productive", "locus"])
    df = df[df["productive"].astype(str).str.upper().isin(["T", "TRUE"])]
    df = df[df["locus"].isin(loci)]
    sequences = sorted(set(df["junction_aa"].dropna().astype(str)))
    counts = Counter()
    for sequence in sequences:
        counts.update(kmer_counts(sequence, k))
    total = sum(counts.values())
    prefix = "TCR:" if receptor == "TCR" else "BCR:"
    features = (
        {f"{prefix}{key}": value / total for key, value in counts.items()}
        if total
        else {}
    )
    FEATURE_CACHE[cache_key] = features
    return features


def paired_feature_table(meta, label_col, positive_label, tcr_chain, bcr_chain, k):
    rows = []
    labels = []
    patients = []
    for _, row in meta.drop_duplicates("PatientID").iterrows():
        patient = str(row["PatientID"])
        tcr_filename = f"{patient}_TCR_AIRR.tsv"
        bcr_filename = f"{patient}_BCR_AIRR.tsv"
        if not (TCR_REPERTOIRES / tcr_filename).exists():
            continue
        if not (BCR_REPERTOIRES / bcr_filename).exists():
            continue
        tcr = repertoire_features("TCR", tcr_filename, TCR_CHAINS[tcr_chain], k)
        bcr = repertoire_features("BCR", bcr_filename, BCR_CHAINS[bcr_chain], k)
        rows.append({**tcr, **bcr})
        labels.append(1 if row[label_col] == positive_label else 0)
        patients.append(patient)
    if len(set(labels)) < 2:
        raise ValueError("Only one class remains after TCR+BCR sample pairing")
    return rows, np.asarray(labels), patients


def model_label(model_name, tcr_chain, bcr_chain, k):
    return f"{model_name} | TCR {tcr_chain} + BCR {bcr_chain} | AA K{k}"


def parse_model_label(label):
    match = re.fullmatch(
        r"(LR|SVM) \| TCR (alpha|beta|alpha\+beta) "
        r"\+ BCR (heavy|light|heavy\+light) \| AA K([34])",
        label,
    )
    if not match:
        raise ValueError(f"Unrecognized model label: {label}")
    return match.group(1), match.group(2), match.group(3), int(match.group(4))


def make_model(model_name):
    if model_name == "LR":
        classifier = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    else:
        classifier = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=50000,
        )
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("classifier", classifier),
        ]
    )


def decision_scores(model, x_test):
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    return model.decision_function(x_test)


def comparison_feature_matrix(
    domain, comparison, tcr_chain, bcr_chain, k
):
    key = (domain, comparison, tcr_chain, bcr_chain, k)
    if key in MATRIX_CACHE:
        return MATRIX_CACHE[key]
    meta, label_col, positive_label = comparison_metadata(domain, comparison)
    rows, y, patients = paired_feature_table(
        meta, label_col, positive_label, tcr_chain, bcr_chain, k
    )
    matrix = DictVectorizer(sparse=True).fit_transform(rows)
    result = (matrix, y, patients, positive_label)
    MATRIX_CACHE[key] = result
    return result


def evaluate_model(domain, comparison, label):
    key = (domain, comparison, label)
    if key in EVALUATION_CACHE:
        return EVALUATION_CACHE[key]

    model_name, tcr_chain, bcr_chain, k = parse_model_label(label)
    x, y, patients, positive_label = comparison_feature_matrix(
        domain, comparison, tcr_chain, bcr_chain, k
    )
    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=0.3, random_state=42
    )
    split_records = []
    prediction_frames = []
    for split, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        fitted = make_model(model_name)
        fitted.fit(x_train, y_train)
        scores = decision_scores(fitted, x_test)
        predictions = fitted.predict(x_test)
        split_records.append(
            {
                "split": split,
                "auc": roc_auc_score(y_test, scores),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "PatientID": [patients[i] for i in test_idx],
                    "y_true": y_test,
                    "score": scores,
                }
            )
        )

    splits = pd.DataFrame(split_records)
    result = {
        "auc_mean": splits["auc"].mean(),
        "auc_sd": splits["auc"].std(ddof=1),
        "balanced_accuracy_mean": splits["balanced_accuracy"].mean(),
        "balanced_accuracy_sd": splits["balanced_accuracy"].std(ddof=1),
        "n_paired_samples": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "positive_label": positive_label,
        "predictions": pd.concat(prediction_frames, ignore_index=True),
    }
    EVALUATION_CACHE[key] = result
    return result


def all_model_labels():
    return [
        model_label(model, tcr_chain, bcr_chain, k)
        for model in ("LR", "SVM")
        for tcr_chain in TCR_CHAINS
        for bcr_chain in BCR_CHAINS
        for k in (3, 4)
    ]


def fit_all_models():
    records = []
    labels = all_model_labels()
    for domain, comparisons in DOMAIN_CONFIGS.items():
        for comparison in comparisons:
            print(f"Evaluating {DOMAIN_TITLES[domain]}: {COMPARISON_LABELS[comparison]}")
            for rank_index, label in enumerate(labels, start=1):
                result = evaluate_model(domain, comparison, label)
                records.append(
                    {
                        "receptor": "TCR+BCR",
                        "domain": domain,
                        "comparison": comparison,
                        "model_label": label,
                        "auc_mean": result["auc_mean"],
                        "auc_sd": result["auc_sd"],
                        "balanced_accuracy_mean":
                            result["balanced_accuracy_mean"],
                        "balanced_accuracy_sd":
                            result["balanced_accuracy_sd"],
                        "n_paired_samples": result["n_paired_samples"],
                        "n_positive": result["n_positive"],
                        "n_negative": result["n_negative"],
                        "positive_label": result["positive_label"],
                        "evaluation_order": rank_index,
                    }
                )
    results = pd.DataFrame(records)
    results.to_csv(
        OUT / "Grant_Combined_TCR_BCR_All_Model_CV_Results.csv", index=False
    )
    return results


def ranked_matrix(results, domain, metric, top_n):
    comparisons = DOMAIN_CONFIGS[domain]
    subset = results[results["domain"].eq(domain)]
    matrix = subset.pivot(
        index="model_label", columns="comparison", values=metric
    ).reindex(columns=comparisons)
    ranking = matrix.mean(axis=1, skipna=True).sort_values(ascending=False)
    return matrix.reindex(ranking.head(top_n).index), ranking


def plot_heatmap(results, domain, metric):
    matrix, ranking = ranked_matrix(results, domain, metric, 15)
    display = matrix.copy()
    display.columns = [COMPARISON_LABELS[column] for column in display.columns]
    is_auc = metric == "auc_mean"
    metric_title = "AUC" if is_auc else "Balanced Accuracy"
    metric_slug = "AUC" if is_auc else "BalancedAccuracy"

    sns.set_theme(style="white", font_scale=0.78)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    n_columns = len(display.columns)
    width = 8.8 if n_columns == 2 else 10.4 if n_columns == 3 else 12.2
    fig, ax = plt.subplots(figsize=(width, 8.3), constrained_layout=True)
    sns.heatmap(
        display,
        ax=ax,
        cmap="magma",
        vmin=0.45,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.45,
        linecolor="#FFFFFF",
        cbar_kws={"label": metric_title, "shrink": 0.78},
        annot_kws={"fontsize": 7.1},
    )
    ax.set_title(
        f"Top 15 Combined TCR+BCR {DOMAIN_TITLES[domain]} Models "
        f"by {metric_title}\n"
        "Paired amino-acid CDR3 k-mer models; diversity models excluded",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    stem = (
        f"Grant_Combined_TCR_BCR_{DOMAIN_TITLES[domain].replace(' ', '_')}"
        f"_Top15_{metric_slug}_NoDiversity"
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=400 if extension == "png" else None,
        )
    plt.close(fig)

    selected = matrix.reset_index().melt(
        id_vars="model_label",
        var_name="comparison",
        value_name=metric,
    )
    selected["domain"] = domain
    selected["metric"] = metric_title
    selected["mean_across_comparisons"] = selected["model_label"].map(ranking)
    selected["rank"] = selected["model_label"].map(
        {label: index + 1 for index, label in enumerate(matrix.index)}
    )
    return selected


def plot_roc_overlays(results, domain):
    matrix, ranking = ranked_matrix(results, domain, "auc_mean", 6)
    top_labels = matrix.index.tolist()
    comparisons = DOMAIN_CONFIGS[domain]
    n_panels = len(comparisons)
    ncols = 2 if n_panels <= 2 else 3
    nrows = 1 if n_panels <= 3 else 2

    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.3 * ncols, 4.8 * nrows),
        constrained_layout=True,
    )
    axes = list(axes.ravel() if hasattr(axes, "ravel") else [axes])
    palette = sns.color_palette("tab10", n_colors=6)
    colors = {label: palette[index] for index, label in enumerate(top_labels)}
    metadata = []

    for ax, comparison in zip(axes, comparisons):
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#9A9A9A",
            linewidth=1.1,
            label="Chance",
        )
        for label in top_labels:
            result = evaluate_model(domain, comparison, label)
            predictions = result["predictions"]
            fpr, tpr, _ = roc_curve(
                predictions["y_true"], predictions["score"]
            )
            pooled_auc = roc_auc_score(
                predictions["y_true"], predictions["score"]
            )
            ax.plot(
                fpr,
                tpr,
                color=colors[label],
                linewidth=2.0,
                alpha=0.95,
                label=f"{label} (AUC {pooled_auc:.2f})",
            )
            metadata.append(
                {
                    "receptor": "TCR+BCR",
                    "domain": domain,
                    "comparison": comparison,
                    "model_label": label,
                    "rank": top_labels.index(label) + 1,
                    "mean_auc_for_ranking": ranking[label],
                    "comparison_auc_mean": matrix.loc[label, comparison],
                    "pooled_roc_auc": pooled_auc,
                    "n_predictions": len(predictions),
                    "n_paired_samples": result["n_paired_samples"],
                    "positive_label": result["positive_label"],
                    "split_design":
                        "5 StratifiedShuffleSplit 70/30 splits, random_state=42",
                    "feature_source":
                        f"{TCR_REPERTOIRES}; {BCR_REPERTOIRES}",
                    "status": "regenerated_and_plotted",
                }
            )
        ax.set_title(COMPARISON_LABELS[comparison], fontsize=12, fontweight="bold")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", fontsize=6.3, frameon=True, framealpha=0.9)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Top 6 Combined TCR+BCR {DOMAIN_TITLES[domain]} Models "
        "by AUC: ROC Overlays\n"
        "Regenerated held-out probabilities from paired AIRR CDR3 k-mer features",
        fontsize=15,
        fontweight="bold",
    )
    stem = (
        f"Grant_Combined_TCR_BCR_{DOMAIN_TITLES[domain].replace(' ', '_')}"
        "_Top6_ROC_AUC_Overlays_NoDiversity"
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=400 if extension == "png" else None,
        )
    plt.close(fig)
    return pd.DataFrame(metadata)


def main():
    results = fit_all_models()
    selected_rows = []
    roc_rows = []
    for domain in DOMAIN_CONFIGS:
        selected_rows.append(plot_heatmap(results, domain, "auc_mean"))
        selected_rows.append(
            plot_heatmap(results, domain, "balanced_accuracy_mean")
        )
        roc_rows.append(plot_roc_overlays(results, domain))

    pd.concat(selected_rows, ignore_index=True).to_csv(
        OUT / "Grant_Combined_TCR_BCR_Top15_Selected_Models.csv", index=False
    )
    pd.concat(roc_rows, ignore_index=True).to_csv(
        OUT / "Grant_Combined_TCR_BCR_Top6_ROC_Curve_Metadata.csv", index=False
    )
    print("Wrote combined TCR+BCR grant figures to", OUT)


if __name__ == "__main__":
    main()

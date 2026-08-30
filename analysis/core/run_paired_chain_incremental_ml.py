from pathlib import Path
from collections import Counter, defaultdict
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 20260824
RNG = np.random.default_rng(SEED)
OUT = Path(r"C:/path/to/private-manuscript-workspace\High Impact Additional Analyses")
TASKS = [
    ("CD vs Control", "CD", "Control"),
    ("UC vs Control", "UC", "Control"),
    ("CD vs UC", "CD", "UC"),
]
REPRESENTATIONS = ["single_chain", "dual_additive", "paired_interaction"]


def kmers(seq, k):
    seq = str(seq).strip().upper()
    if not seq or seq == "NAN":
        return []
    return [seq[i : i + k] for i in range(len(seq) - k + 1) if "*" not in seq[i : i + k] and "X" not in seq[i : i + k]]


def normalize_namespace(counter, prefix):
    keys = [k for k in counter if k.startswith(prefix)]
    total = sum(counter[k] for k in keys)
    if total > 0:
        for key in keys:
            counter[key] /= total


def build_feature_dicts(df, modality):
    """Build matched-cell single, additive, and pairing-aware feature dictionaries.

    TCR chain1 is alpha and chain2 is beta; the single-chain reference is beta.
    BCR chain1 is heavy and chain2 is light; the single-chain reference is heavy.
    Pairing-aware features are abundance-weighted cross-chain amino-acid composition
    interactions computed within each exact paired clonotype.
    """
    sample_info = df[["SampleID", "Diagnosis1"]].drop_duplicates().sort_values("SampleID")
    features = {rep: {} for rep in REPRESENTATIONS}
    for sample_id, group in df.groupby("SampleID", sort=False):
        c_single = Counter()
        c_add = Counter()
        c_pair = Counter()
        for row in group.itertuples(index=False):
            c1 = str(row.chain1_cdr3)
            c2 = str(row.chain2_cdr3)
            weight = float(row.n_cells)
            k1 = kmers(c1, 4)
            k2 = kmers(c2, 4)
            # Matched single-chain reference.
            ref = k2 if modality == "TCR" else k1
            ref_prefix = "B4:" if modality == "TCR" else "H4:"
            for token in ref:
                c_single[ref_prefix + token] += weight
            # Chain-aware additive representation.
            p1, p2 = (("A4:", "B4:") if modality == "TCR" else ("H4:", "L4:"))
            for token in k1:
                c_add[p1 + token] += weight
                c_pair[p1 + token] += weight
            for token in k2:
                c_add[p2 + token] += weight
                c_pair[p2 + token] += weight
            # Explicit pairing interaction; unique residues prevent long CDR3s from dominating.
            aa1 = sorted(set(c1))
            aa2 = sorted(set(c2))
            for a in aa1:
                for b in aa2:
                    if a not in "*X" and b not in "*X":
                        c_pair[f"PAIR1:{a}>{b}"] += weight
        normalize_namespace(c_single, "B4:" if modality == "TCR" else "H4:")
        for prefix in (("A4:", "B4:") if modality == "TCR" else ("H4:", "L4:")):
            normalize_namespace(c_add, prefix)
            normalize_namespace(c_pair, prefix)
        normalize_namespace(c_pair, "PAIR1:")
        features["single_chain"][sample_id] = dict(c_single)
        features["dual_additive"][sample_id] = dict(c_add)
        features["paired_interaction"][sample_id] = dict(c_pair)
    return sample_info, features


def make_model():
    pipeline = Pipeline(
        [
            ("vectorize", DictVectorizer(sparse=True)),
            ("select", SelectPercentile(chi2)),
            ("scale", StandardScaler(with_mean=False)),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )
    grid = {"select__percentile": [25, 100], "model__C": [0.1, 1.0, 10.0]}
    return pipeline, grid


def stratified_bootstrap_auc(y, predictions, n_boot=3000):
    y = np.asarray(y, int)
    predictions = {k: np.asarray(v, float) for k, v in predictions.items()}
    classes = [np.flatnonzero(y == c) for c in np.unique(y)]
    aucs = defaultdict(list)
    for _ in range(n_boot):
        idx = np.concatenate([RNG.choice(ii, len(ii), replace=True) for ii in classes])
        for name, values in predictions.items():
            aucs[name].append(roc_auc_score(y[idx], values[idx]))
    return {name: np.asarray(vals) for name, vals in aucs.items()}


def run_modality(modality, csv_path):
    df = pd.read_csv(csv_path)
    sample_info, all_features = build_feature_dicts(df, modality)
    all_folds, all_predictions, all_summaries, all_deltas = [], [], [], []
    for task, positive, negative in TASKS:
        sub = sample_info[sample_info.Diagnosis1.isin([positive, negative])].copy().reset_index(drop=True)
        ids = sub.SampleID.astype(str).to_numpy()
        y = (sub.Diagnosis1 == positive).astype(int).to_numpy()
        outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(np.zeros(len(y)), y))
        task_probabilities = {}
        fold_auc_by_rep = {}
        for representation in REPRESENTATIONS:
            print(f"{modality}: {task}: {representation}", flush=True)
            X = [all_features[representation][sid] for sid in ids]
            fold_rows, pred_rows = [], []
            for fold_id, (train_idx, test_idx) in enumerate(outer, 1):
                pipeline, grid = make_model()
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold_id)
                search = GridSearchCV(pipeline, grid, cv=inner, scoring="roc_auc", refit=True, n_jobs=-1)
                search.fit([X[i] for i in train_idx], y[train_idx])
                prob = search.predict_proba([X[i] for i in test_idx])[:, 1]
                auc = roc_auc_score(y[test_idx], prob)
                fold_rows.append(
                    {
                        "modality": modality,
                        "task": task,
                        "representation": representation,
                        "outer_fold": fold_id,
                        "roc_auc": auc,
                        "n_test": len(test_idx),
                        "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                    }
                )
                pred_rows.extend(
                    {
                        "modality": modality,
                        "task": task,
                        "representation": representation,
                        "outer_fold": fold_id,
                        "SampleID": ids[i],
                        "truth": int(y[i]),
                        "probability": float(p),
                    }
                    for i, p in zip(test_idx, prob)
                )
            fold_df = pd.DataFrame(fold_rows)
            pred_df = pd.DataFrame(pred_rows)
            pooled = pred_df.groupby(["SampleID", "truth"], as_index=False).probability.mean()
            pooled = sub[["SampleID"]].merge(pooled, on="SampleID", how="left")
            pooled_auc = roc_auc_score(pooled.truth, pooled.probability)
            task_probabilities[representation] = pooled.probability.to_numpy()
            fold_auc_by_rep[representation] = fold_df.roc_auc.to_numpy()
            all_folds.append(fold_df)
            all_predictions.append(pred_df)
            all_summaries.append(
                {
                    "modality": modality,
                    "task": task,
                    "positive_class": positive,
                    "negative_class": negative,
                    "representation": representation,
                    "n": len(y),
                    "n_positive": int(y.sum()),
                    "n_negative": int((1 - y).sum()),
                    "pooled_outer_fold_auc": pooled_auc,
                    "median_outer_fold_auc": float(np.median(fold_df.roc_auc)),
                    "outer_fold_auc_q025": float(np.quantile(fold_df.roc_auc, 0.025)),
                    "outer_fold_auc_q975": float(np.quantile(fold_df.roc_auc, 0.975)),
                }
            )
        boot = stratified_bootstrap_auc(y, task_probabilities)
        for representation in REPRESENTATIONS:
            row = next(r for r in reversed(all_summaries) if r["modality"] == modality and r["task"] == task and r["representation"] == representation)
            row["pooled_auc_bootstrap_ci_low"] = float(np.quantile(boot[representation], 0.025))
            row["pooled_auc_bootstrap_ci_high"] = float(np.quantile(boot[representation], 0.975))
        for representation in ["dual_additive", "paired_interaction"]:
            delta_boot = boot[representation] - boot["single_chain"]
            fold_delta = fold_auc_by_rep[representation] - fold_auc_by_rep["single_chain"]
            all_deltas.append(
                {
                    "modality": modality,
                    "task": task,
                    "comparison": f"{representation} vs single_chain",
                    "delta_pooled_auc": float(
                        roc_auc_score(y, task_probabilities[representation])
                        - roc_auc_score(y, task_probabilities["single_chain"])
                    ),
                    "delta_auc_ci_low": float(np.quantile(delta_boot, 0.025)),
                    "delta_auc_ci_high": float(np.quantile(delta_boot, 0.975)),
                    "median_outer_fold_delta_auc": float(np.median(fold_delta)),
                    "outer_fold_wilcoxon_p": float(wilcoxon(fold_delta).pvalue) if np.any(fold_delta != 0) else 1.0,
                }
            )
    return (
        pd.concat(all_folds, ignore_index=True),
        pd.concat(all_predictions, ignore_index=True),
        pd.DataFrame(all_summaries),
        pd.DataFrame(all_deltas),
    )


def plot_results(summary, delta):
    labels = {
        ("TCR", "single_chain"): "beta chain",
        ("TCR", "dual_additive"): "alpha + beta",
        ("TCR", "paired_interaction"): "paired alpha-beta interaction",
        ("BCR", "single_chain"): "heavy chain",
        ("BCR", "dual_additive"): "heavy + light",
        ("BCR", "paired_interaction"): "paired heavy-light interaction",
    }
    colors = {"single_chain": "#666666", "dual_additive": "#3568A8", "paired_interaction": "#B2495B"}
    tasks = [x[0] for x in TASKS]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    for ax, modality, panel in zip(axes[:2], ["TCR", "BCR"], ["A", "B"]):
        d = summary[summary.modality == modality].copy()
        for j, representation in enumerate(REPRESENTATIONS):
            z = d[d.representation == representation].set_index("task").loc[tasks]
            x = np.arange(len(tasks)) + (j - 1) * 0.18
            y = z.pooled_outer_fold_auc.to_numpy()
            lo = z.pooled_auc_bootstrap_ci_low.to_numpy()
            hi = z.pooled_auc_bootstrap_ci_high.to_numpy()
            ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="o", color=colors[representation],
                        capsize=3, linewidth=1.2, markersize=5, label=labels[(modality, representation)])
        ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xticks(np.arange(len(tasks)), tasks, rotation=28, ha="right")
        ax.set_ylim(0.45, 1.03)
        ax.set_ylabel("Pooled outer-fold ROC AUC")
        ax.set_title(f"{panel}  {modality} matched-cell representations", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    d = delta.copy()
    d["representation"] = d.comparison.str.replace(" vs single_chain", "", regex=False)
    ypos, ylabels = [], []
    pos = 0
    for modality in ["TCR", "BCR"]:
        for task in tasks:
            for representation in ["dual_additive", "paired_interaction"]:
                z = d[(d.modality == modality) & (d.task == task) & (d.representation == representation)].iloc[0]
                ypos.append(pos)
                ylabels.append(f"{modality}: {task}\n{labels[(modality, representation)]}")
                ax.errorbar(z.delta_pooled_auc, pos,
                            xerr=[[z.delta_pooled_auc - z.delta_auc_ci_low], [z.delta_auc_ci_high - z.delta_pooled_auc]],
                            fmt="o", color=colors[representation], capsize=3, linewidth=1.1, markersize=4.5)
                pos += 1
            pos += 0.4
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_yticks(ypos, ylabels, fontsize=7)
    ax.set_xlabel("Delta pooled ROC AUC vs matched single chain")
    ax.set_title("C  Incremental value of two-chain features", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Nested participant-level evaluation of paired-chain repertoire information", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "Figure_HI4_paired_chain_incremental_value.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_HI4_paired_chain_incremental_value.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    outputs = []
    for modality in ["TCR", "BCR"]:
        path = OUT / f"paired_{modality}_clonotypes_for_nested_ML.csv"
        outputs.append(run_modality(modality, path))
    folds = pd.concat([o[0] for o in outputs], ignore_index=True)
    predictions = pd.concat([o[1] for o in outputs], ignore_index=True)
    summary = pd.concat([o[2] for o in outputs], ignore_index=True)
    delta = pd.concat([o[3] for o in outputs], ignore_index=True)
    delta["p_adj_outer_fold"] = delta.groupby("modality").outer_fold_wilcoxon_p.transform(
        lambda x: np.minimum(1, x.rank(method="first") * 0 + np.array([min(1, p * len(x) / r) for p, r in zip(np.sort(x), np.arange(1, len(x) + 1))]))
    ) if False else np.nan
    # Standard BH adjustment across the 12 incremental comparisons.
    order = np.argsort(delta.outer_fold_wilcoxon_p.to_numpy())
    adjusted = np.empty(len(delta))
    ranked = delta.outer_fold_wilcoxon_p.to_numpy()[order] * len(delta) / np.arange(1, len(delta) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    delta["p_adj_outer_fold"] = adjusted
    folds.to_csv(OUT / "Table_HI_paired_chain_nested_outer_fold_metrics.csv", index=False)
    predictions.to_csv(OUT / "Table_HI_paired_chain_nested_outer_fold_predictions.csv", index=False)
    summary.to_csv(OUT / "Table_HI_paired_chain_nested_validation_summary.csv", index=False)
    delta.to_csv(OUT / "Table_HI_paired_chain_incremental_value.csv", index=False)
    plot_results(summary, delta)
    manifest = {
        "seed": SEED,
        "statistical_unit": "participant",
        "outer_resampling": "5-fold stratified cross-validation repeated 3 times",
        "inner_resampling": "3-fold stratified cross-validation",
        "pipeline": "DictVectorizer; within-training-fold chi-squared feature selection; sparse scaling; class-balanced logistic regression",
        "representations": {
            "single_chain": "TCR beta or BCR heavy CDR3 amino-acid 4-mers",
            "dual_additive": "chain-prefixed alpha+beta or heavy+light CDR3 amino-acid 4-mers",
            "paired_interaction": "dual additive features plus within-clonotype cross-chain amino-acid interaction features",
        },
        "comparison_design": "All representations use the same exact paired clonotypes, participants, and outer splits.",
    }
    (OUT / "paired_chain_ML_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_grant_all_receptor_shap_barplots as receptor_models
import build_grant_combined_tcr_bcr_models as combined_models
import build_figure7_immunity_revision as previous


OUT = ROOT / "Figure 7 Clinical Translation"
OUT.mkdir(parents=True, exist_ok=True)

COL = previous.COL
COL.update({
    "diagnosis": "#5B6573",
    "inflammation": "#A33A3A",
    "therapy": "#6F4C9B",
})

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.1,
    "axes.titlesize": 8.1,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


SELECTION_FILE = (
    ROOT / "grant_combined_tcr_bcr_figures"
    / "Grant_All_TCR_BCR_Combined_Best_Model_SHAP_Selection.csv"
)
SHAP_FILE = (
    ROOT / "grant_combined_tcr_bcr_figures"
    / "Grant_All_TCR_BCR_Combined_Best_Model_SHAP_Top20_Features.csv"
)
NESTED_FILE = (
    ROOT / "Recreated Figure 6"
    / "Figure_6_panel_B_nested_diagnosis_source_data.csv"
)
NULL_FILE = ROOT / "ML Sensitivity Validation" / "Table_S_permutation_test_summary.csv"
PERFORMANCE_FILE = OUT / "Figure_7BDF_performance_source_data.csv"
PREDICTION_FILE = OUT / "Figure_7DF_clinical_state_oof_predictions.csv"
SCHEMATIC_FILE = OUT / "Figure_7A_ChatGPT_ImageGen_immuneML_redraw_transparent.png"


TASKS = {
    "diagnosis": [
        ("cd_vs_control", "CD vs control"),
        ("uc_vs_control", "UC vs control"),
        ("cd_vs_uc", "CD vs UC"),
    ],
    "inflammation": [
        ("cd_inflamed_vs_noninflamed", "CD: inflamed vs noninflamed"),
        ("uc_inflamed_vs_noninflamed", "UC: inflamed vs noninflamed"),
    ],
    "therapy_response": [
        ("combined_biologic_nonresponder_vs_responder", "All biologics: NR vs R"),
        ("anti_tnf_nonresponder_vs_responder", "Anti-TNF: NR vs R"),
        ("ustekinumab_nonresponder_vs_responder", "Ustekinumab: NR vs R"),
        ("vedolizumab_nonresponder_vs_responder", "Vedolizumab: NR vs R"),
    ],
}


def style_axis(ax, grid_axis=None):
    previous.style_axis(ax, grid_axis)


def panel_label(ax, letter, x=-0.14, y=1.06):
    previous.panel_label(ax, letter, x=x, y=y)


def panel_title(ax, title):
    ax.set_title(title, loc="left", pad=3, fontweight="bold", color=COL["ink"])


def rounded_box(ax, xy, width, height, text, edge, face="#FFFFFF", fontsize=6.0):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.014,rounding_size=0.024",
        linewidth=0.9, edgecolor=edge, facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=fontsize, color=COL["ink"], linespacing=1.08,
    )


def bridge_panel(ax):
    ax.set_axis_off()
    panel_label(ax, "A", x=-0.055, y=1.01)
    ax.text(
        0.015, 1.01,
        "Adaptive immune remodeling supports repertoire-based clinical classification in IBD",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.4, fontweight="bold", color=COL["ink"],
    )

    if not SCHEMATIC_FILE.exists():
        raise FileNotFoundError(f"Missing AI schematic: {SCHEMATIC_FILE}")
    artwork = plt.imread(SCHEMATIC_FILE)
    # Shift the complete schematic unit left to align its visual edge with the
    # quantitative panels while preserving the native 2:1 footprint.
    artwork_x_shift = -0.05
    ax.imshow(
        artwork,
        extent=(0.02 + artwork_x_shift, 0.75 + artwork_x_shift, 0.03, 0.975),
        aspect="auto", zorder=0,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    labels = [
        (0.118 + artwork_x_shift, 0.656, "TCR repertoire  |  Figs. 2-3", COL["ink"], 3.15, "bold"),
        (0.118 + artwork_x_shift, 0.256, "BCR repertoire  |  Figs. 4-5", COL["ink"], 3.15, "bold"),
        (0.385 + artwork_x_shift, 0.239, "immuneML: k-mers | LR/SVM", COL["ink"], 3.65, "bold"),
        (0.634 + artwork_x_shift, 0.706, "Diagnosis\nCD | UC | control [B-C]", COL["ink"], 3.45, "bold"),
        (0.634 + artwork_x_shift, 0.442, "Inflammation\nInflamed | noninfl. [D-E]", COL["ink"], 3.35, "bold"),
        (0.634 + artwork_x_shift, 0.154, "Biologic response\nNR | R [F-G]", COL["ink"], 3.05, "bold"),
    ]
    for x, y, text, color, fontsize, weight in labels:
        ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=color,
                linespacing=0.92, zorder=3)

    motif_x = [x + artwork_x_shift for x in [0.0925, 0.1175, 0.1430]]
    for x, token in zip(motif_x, ["CASS", "KLF", "SFSG"]):
        ax.text(x, 0.597, token, transform=ax.transAxes,
                ha="center", va="center", fontsize=3.25, family="monospace",
                color=COL["TCR"], zorder=4)
    for x, token in zip(motif_x, ["CQQY", "AKR", "TYYF"]):
        ax.text(x, 0.196, token, transform=ax.transAxes,
                ha="center", va="center", fontsize=3.25, family="monospace",
                color=COL["BCR"], zorder=4)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COL["TCR"], label="TCR", markersize=5.4,
               markeredgewidth=1.25),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=COL["BCR"], label="BCR", markersize=5.4,
               markeredgewidth=1.25),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor=COL["Joint"], label="TCR+BCR", markersize=5.4,
               markeredgewidth=1.25),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["muted"],
               markeredgecolor=COL["muted"], label="nested", markersize=5.0),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COL["muted"], label="post-selection", markersize=5.0,
               markeredgewidth=1.25),
    ]
    legend_box = FancyBboxPatch(
        (0.765, 0.305), 0.222, 0.545,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        transform=ax.transAxes, facecolor="white", edgecolor="#D4E1ED",
        linewidth=0.75, alpha=0.96, zorder=2,
    )
    ax.add_patch(legend_box)
    ax.text(0.785, 0.805, "Model key", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.2, fontweight="bold",
            color=COL["ink"])
    ax.text(0.785, 0.742, "Repertoire", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=4.9, color=COL["muted"])
    receptor_legend = ax.legend(
        handles=legend_handles[:3], frameon=False, ncol=1, loc="upper left",
        bbox_to_anchor=(0.785, 0.724), borderaxespad=0, fontsize=5.6,
        handlelength=0.9, handletextpad=0.48, labelspacing=0.42,
    )
    ax.add_artist(receptor_legend)
    ax.text(0.785, 0.445, "Validation", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=4.9, color=COL["muted"])
    ax.legend(
        handles=legend_handles[3:], frameon=False, ncol=1, loc="upper left",
        bbox_to_anchor=(0.785, 0.427), borderaxespad=0, fontsize=5.6,
        handlelength=0.9, handletextpad=0.48, labelspacing=0.42,
    )


def bootstrap_metrics(y, score, prediction, seed, n_boot=3000):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, aps, bas = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
        aps.append(average_precision_score(y[idx], score[idx]))
        bas.append(balanced_accuracy_score(y[idx], prediction[idx]))
    return {
        "roc_auc_ci_low": np.quantile(aucs, 0.025),
        "roc_auc_ci_high": np.quantile(aucs, 0.975),
        "pr_auc_ci_low": np.quantile(aps, 0.025),
        "pr_auc_ci_high": np.quantile(aps, 0.975),
        "balanced_accuracy_ci_low": np.quantile(bas, 0.025),
        "balanced_accuracy_ci_high": np.quantile(bas, 0.975),
        "bootstrap_replicates": len(aucs),
    }


def selected_matrix(row):
    receptor = row["receptor_model"]
    if receptor in {"TCR", "BCR"}:
        x, y, patients, _, positive_label = receptor_models.single_receptor_matrix(row)
        model_name = row["model_name"]
    else:
        model_name, tcr_chain, bcr_chain, k = combined_models.parse_model_label(row["model_label"])
        x, y, patients, positive_label = combined_models.comparison_feature_matrix(
            row["domain"], row["comparison"], tcr_chain, bcr_chain, k
        )
    return x, np.asarray(y), np.asarray(patients), positive_label, model_name


def clinical_cross_validation(selection):
    performance_rows = []
    prediction_rows = []
    clinical = selection[selection.domain.isin(["inflammation", "therapy_response"])].copy()
    clinical = clinical.sort_values(["domain", "comparison", "receptor_model"])
    for model_index, (_, row) in enumerate(clinical.iterrows(), start=1):
        print(
            f"CV {model_index}/{len(clinical)}: {row.domain} | "
            f"{row.comparison} | {row.receptor_model}", flush=True
        )
        x, y, patients, positive_label, model_name = selected_matrix(row)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260828)
        score = np.full(len(y), np.nan)
        prediction = np.full(len(y), -1, dtype=int)
        fold_id = np.full(len(y), -1, dtype=int)
        fold_iterations = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
            model = combined_models.make_model(model_name)
            if hasattr(model.named_steps["classifier"], "max_iter"):
                model.named_steps["classifier"].set_params(max_iter=500000)
            model.fit(x[train_idx], y[train_idx])
            n_iter = getattr(model.named_steps["classifier"], "n_iter_", np.nan)
            fold_iterations.append(float(np.max(np.atleast_1d(n_iter))))
            score[test_idx] = combined_models.decision_scores(model, x[test_idx])
            prediction[test_idx] = model.predict(x[test_idx]).astype(int)
            fold_id[test_idx] = fold
        if np.isnan(score).any() or (fold_id < 1).any():
            raise RuntimeError("Incomplete out-of-fold predictions")

        seed = 1000 + model_index
        intervals = bootstrap_metrics(y, score, prediction, seed=seed)
        performance_rows.append({
            "domain": row.domain,
            "comparison": row.comparison,
            "receptor_model": row.receptor_model,
            "model_label": row.model_label,
            "model_name": row.get("model_name", np.nan),
            "chain_group": row.get("chain_group", np.nan),
            "sequence_type": row.get("sequence_type", np.nan),
            "k": row.get("k", np.nan),
            "selection_auc_mean": row.selection_auc_mean,
            "selection_balanced_accuracy_mean": row.selection_balanced_accuracy_mean,
            "roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            **intervals,
            "n": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "positive_label": positive_label,
            "evaluation": "fixed selected model; participant-level 5-fold CV",
            "interval_method": "participant bootstrap of out-of-fold predictions",
            "maximum_fitting_iterations": max(fold_iterations),
            "configured_max_iterations": 500000,
        })
        prediction_rows.append(pd.DataFrame({
            "domain": row.domain,
            "comparison": row.comparison,
            "receptor_model": row.receptor_model,
            "model_label": row.model_label,
            "participant": patients,
            "truth": y,
            "fold": fold_id,
            "score": score,
            "predicted_class": prediction,
            "positive_label": positive_label,
        }))
    return pd.DataFrame(performance_rows), pd.concat(prediction_rows, ignore_index=True)


def performance_data(selection, nested, null):
    if PERFORMANCE_FILE.exists() and PREDICTION_FILE.exists():
        cached = pd.read_csv(PERFORMANCE_FILE)
        clinical_cached = cached[cached.domain.isin(["inflammation", "therapy_response"])]
        if len(clinical_cached) == 18:
            return cached, pd.read_csv(PREDICTION_FILE)

    clinical_perf, predictions = clinical_cross_validation(selection)
    nested_perf = nested.merge(
        null[["modality", "task", "null_95th_percentile"]],
        on=["modality", "task"], how="left",
    ).rename(columns={
        "task": "comparison",
        "roc_auc_median": "roc_auc",
        "modality": "receptor_model",
    })
    nested_perf["receptor_model"] = nested_perf.receptor_model.str.upper()
    nested_perf["domain"] = "diagnosis"
    nested_perf["evaluation"] = "fully nested outer-fold validation"
    nested_perf["interval_method"] = "outer-fold distribution"
    keep = [
        "domain", "comparison", "receptor_model", "roc_auc",
        "roc_auc_ci_low", "roc_auc_ci_high", "n", "n_positive", "n_negative",
        "null_95th_percentile", "evaluation", "interval_method",
    ]
    performance = pd.concat([nested_perf[keep], clinical_perf], ignore_index=True, sort=False)
    performance.to_csv(PERFORMANCE_FILE, index=False)
    predictions.to_csv(PREDICTION_FILE, index=False)
    return performance, predictions


def forest_panel(ax, performance, domain, letter, title, tier, task_rows,
                 show_legend=False, xlim=(0.35, 1.01)):
    panel_title(ax, title)
    panel_label(ax, letter, x=-0.16, y=1.07)
    ax.text(1.0, 1.015, tier, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.1, fontweight="bold", color=COL["muted"])
    d = performance[performance.domain.eq(domain)].copy()
    ybase = np.arange(len(task_rows))[::-1]
    offsets = {"TCR": 0.18, "TCR+BCR": 0.0, "BCR": -0.18}
    markers = {"TCR": "o", "TCR+BCR": "D", "BCR": "s"}
    colors = {"TCR": COL["TCR"], "TCR+BCR": COL["Joint"], "BCR": COL["BCR"]}
    models = ["TCR", "BCR"] if domain == "diagnosis" else ["TCR", "TCR+BCR", "BCR"]
    nested = domain == "diagnosis"

    labels = []
    for yy, (comparison, label) in zip(ybase, task_rows):
        sub = d[d.comparison.eq(comparison)]
        nrow = sub[sub.receptor_model.eq("TCR+BCR")]
        if nrow.empty:
            nrow = sub.iloc[[0]]
        nr = nrow.iloc[0]
        labels.append(f"{label}\nn={int(nr.n)} ({int(nr.n_positive)}/{int(nr.n_negative)})")
        for model_name in models:
            row = sub[sub.receptor_model.eq(model_name)]
            if row.empty:
                continue
            row = row.iloc[0]
            mid = float(row.roc_auc)
            lo = float(row.roc_auc_ci_low)
            hi = float(row.roc_auc_ci_high)
            ax.errorbar(
                mid, yy + offsets[model_name], xerr=[[mid - lo], [hi - mid]],
                fmt=markers[model_name], ms=4.1, capsize=1.8, elinewidth=0.95,
                markerfacecolor=colors[model_name] if nested else "white",
                markeredgecolor=colors[model_name], markeredgewidth=1.0,
                color=colors[model_name], zorder=3,
            )
            if nested and pd.notna(row.get("null_95th_percentile", np.nan)):
                xn = float(row.null_95th_percentile)
                ax.plot([xn, xn], [yy + offsets[model_name] - 0.055,
                                  yy + offsets[model_name] + 0.055],
                        color="#999999", lw=1.0, zorder=2)

    ax.axvline(0.5, color=COL["muted"], ls="--", lw=0.65)
    ax.set_yticks(ybase, labels)
    ax.set_ylim(-0.5, len(task_rows) - 0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Out-of-fold ROC AUC (95% CI)", labelpad=1.0)
    if show_legend:
        handles = [
            Line2D([0], [0], marker=markers[x], color="none",
                   markerfacecolor=colors[x] if nested else "white",
                   markeredgecolor=colors[x], label=x, markersize=4.5)
            for x in models
        ]
        ax.legend(handles=handles, frameon=False, loc="lower right", ncol=len(models),
                  bbox_to_anchor=(1.0, 1.01), borderaxespad=0, handletextpad=0.25,
                  columnspacing=0.8, fontsize=5.5)
    style_axis(ax, "x")


def receptor_shap_axis(ax, features, receptor, negative_label, positive_label,
                       show_xlabel=False):
    d = features[features.receptor.eq(receptor)].copy()
    d = d.sort_values("relative_mean_absolute_shap", ascending=True)
    y = np.arange(len(d), dtype=float)
    for yy, (_, row) in zip(y, d.iterrows()):
        color = COL[receptor]
        positive = row.direction_sign > 0
        ax.barh(yy, row.relative_mean_absolute_shap, height=0.66,
                color=color, edgecolor=color, alpha=0.78, lw=0.7, zorder=3)
        ax.scatter(row.relative_mean_absolute_shap, yy,
                   marker=">" if positive else "<", s=14,
                   facecolor=color if positive else "white", edgecolor=color,
                   lw=0.75, zorder=4)
    ax.set_yticks(y, d.feature_token, family="monospace", fontsize=5.0)
    for tick in ax.get_yticklabels():
        tick.set_color(COL[receptor])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.5, 1.0])
    if show_xlabel:
        ax.set_xlabel(f"> {positive_label}   < {negative_label}", fontsize=4.8, labelpad=2)
    else:
        ax.set_xticklabels([])
    ax.set_title(receptor, loc="left", fontsize=5.8, fontweight="bold",
                 color=COL[receptor], pad=2)
    model_text = d.model_display.iloc[0].replace(f"{receptor}: ", "")
    ax.text(1.0, 1.04, model_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=4.0, color=COL[receptor])
    style_axis(ax, "x")


def shap_panel(fig, spec, shap, domain, letter, title, task_rows):
    outer = fig.add_subplot(spec)
    outer.set_axis_off()
    panel_title(outer, title)
    panel_label(outer, letter, x=-0.12, y=1.07)
    inner = GridSpecFromSubplotSpec(
        len(task_rows) + 1, 1, subplot_spec=spec,
        height_ratios=[0.10] + [1.0] * len(task_rows),
        hspace=0.62 if len(task_rows) == 3 else 0.78,
    )
    source_rows = []
    label_map = {
        "cd_vs_control": ("Control", "CD"),
        "uc_vs_control": ("Control", "UC"),
        "cd_vs_uc": ("UC", "CD"),
        "cd_inflamed_vs_noninflamed": ("Noninfl.", "Inflamed"),
        "uc_inflamed_vs_noninflamed": ("Noninfl.", "Inflamed"),
        "combined_biologic_nonresponder_vs_responder": ("R", "NR"),
        "anti_tnf_nonresponder_vs_responder": ("R", "NR"),
        "ustekinumab_nonresponder_vs_responder": ("R", "NR"),
    }
    for i, (comparison, task_title) in enumerate(task_rows):
        features = pd.concat([
            previous.best_model_shap_features(shap, domain, comparison, "TCR", n_features=6),
            previous.best_model_shap_features(shap, domain, comparison, "BCR", n_features=6),
        ], ignore_index=True)
        source_rows.append(features)
        neg, pos = label_map[comparison]
        task_outer = fig.add_subplot(inner[i + 1, 0])
        task_outer.set_axis_off()
        task_outer.text(0.0, 1.24, task_title, transform=task_outer.transAxes,
                        ha="left", va="bottom", fontsize=6.2, fontweight="bold")
        receptor_grid = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=inner[i + 1, 0], wspace=0.36
        )
        for j, receptor in enumerate(["TCR", "BCR"]):
            receptor_shap_axis(
                fig.add_subplot(receptor_grid[0, j]), features, receptor,
                neg, pos, show_xlabel=i == len(task_rows) - 1,
            )
    return pd.concat(source_rows, ignore_index=True)


def build():
    selection = pd.read_csv(SELECTION_FILE)
    shap = pd.read_csv(SHAP_FILE)
    nested = pd.read_csv(NESTED_FILE)
    null = pd.read_csv(NULL_FILE)
    performance, predictions = performance_data(selection, nested, null)

    # Final production canvas: 171.45 x 224.79 mm, within the commonly used
    # Cell Press full-width (172 mm) and full-page height (225 mm) envelope.
    fig = plt.figure(figsize=(6.75, 8.85), facecolor="white")
    gs = GridSpec(
        7, 2, figure=fig,
        height_ratios=[2.50, 0.12, 2.20, 0.48, 1.65, 0.48, 2.10],
        width_ratios=[1.02, 0.98],
        hspace=0.0, wspace=0.32,
        left=0.18, right=0.99, top=0.982, bottom=0.037,
    )
    bridge_panel(fig.add_subplot(gs[0, :]))

    forest_panel(
        fig.add_subplot(gs[2, 0]), performance, "diagnosis", "B",
        "Diagnosis models", "FULLY NESTED VALIDATION",
        TASKS["diagnosis"], show_legend=False, xlim=(0.45, 1.01),
    )
    shap_rows = [shap_panel(
        fig, gs[2, 1], shap, "diagnosis", "C",
        "Best-model diagnosis features", TASKS["diagnosis"],
    )]

    forest_panel(
        fig.add_subplot(gs[4, 0]), performance, "inflammation", "D",
        "Inflammation models", "POST-SELECTION 5-FOLD CV",
        TASKS["inflammation"], show_legend=False, xlim=(0.25, 1.01),
    )
    shap_rows.append(shap_panel(
        fig, gs[4, 1], shap, "inflammation", "E",
        "Best-model inflammation features", TASKS["inflammation"],
    ))

    forest_panel(
        fig.add_subplot(gs[6, 0]), performance, "therapy_response", "F",
        "Biologic-response models", "POST-SELECTION 5-FOLD CV",
        TASKS["therapy_response"], show_legend=False, xlim=(0.25, 1.01),
    )
    therapy_features = [
        TASKS["therapy_response"][0],
        TASKS["therapy_response"][1],
        TASKS["therapy_response"][2],
    ]
    shap_rows.append(shap_panel(
        fig, gs[6, 1], shap, "therapy_response", "G",
        "Focused biologic-response features", therapy_features,
    ))

    shap_source = pd.concat(shap_rows, ignore_index=True)
    shap_source.to_csv(OUT / "Figure_7CEG_SHAP_top6_source_data.csv", index=False)
    complete_therapy = []
    for comparison, _ in TASKS["therapy_response"]:
        for receptor in ["TCR", "BCR"]:
            complete_therapy.append(previous.best_model_shap_features(
                shap, "therapy_response", comparison, receptor, n_features=20
            ))
    pd.concat(complete_therapy, ignore_index=True).to_csv(
        OUT / "Figure_7G_complete_therapy_SHAP_source_data.csv", index=False
    )

    stem = OUT / "Figure_7_Clinical_Translation_CellPress_7A_Labels_CompactGap_revised"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=400, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    legend = """Figure 7. Adaptive immune remodeling supports repertoire-based clinical classification in IBD.

(A) ChatGPT ImageGen redraw of the biological-to-clinical workflow, styled to match the rounded-box language of Figure 1B. Dedicated sequence-feature chips flank the TCR and BCR inputs, an integrated blue-orange feature grid represents immuneML repertoire learning, and a green shield denotes participant-level validation before classification of diagnosis, tissue inflammation, and biologic response status. The schematic links the TCR expansion and sequence-neighborhood results in Figures 2-3, the BCR class-switching, somatic-hypermutation, and lineage results in Figures 4-5, and the coordinated adaptive immune axis in Figure 6 to the clinical applications quantified in panels B-G. Representative TCR (CASS, KLF, and SFSG) and BCR (CQQY, AKR, and TYYF) tokens are endpoint-spanning examples selected from the SHAP-ranked features displayed in panels C, E, and G; they are descriptive model features and should not be interpreted as stable antigen-specificity assignments. Productive paired TCR and BCR CDR3 repertoires were represented as amino-acid (AA) or nucleotide (NT) k-mer frequencies across prespecified chain compartments. The central immuneML model block summarizes the selected logistic-regression and support-vector-machine pipelines and participant-level cross-validation. The artwork is conceptual and does not depict quantitative data. All scientific text labels and quantitative panels were generated programmatically. Model fitting and preprocessing were performed within training folds; SHAP values were calculated after refitting the selected model to the full analysis cohort.

(B) Fully nested diagnosis classification for CD versus control, UC versus control, and CD versus UC. Points show median outer-fold ROC AUC and horizontal intervals show 95% intervals; gray ticks indicate the 95th percentile of the corresponding fixed-pipeline permutation null. Sample sizes are total participants, with positive/negative class counts in parentheses.

(C) Descriptive SHAP attribution for the selected TCR and BCR diagnosis models. (D) Tissue-inflammation classification within CD and UC using the frozen best immuneML TCR, BCR, and joint TCR+BCR models. Points show participant-level five-fold out-of-fold ROC AUC and intervals show participant-bootstrap 95% confidence intervals. Because the models were selected from the same endpoint-specific screen, these are conditional post-selection estimates rather than fully nested validation results. (E) Descriptive SHAP attribution for the selected TCR and BCR inflammation models.

(F) Post-treatment biologic response-status classification for pooled biologic exposure, anti-TNF therapy, ustekinumab, and vedolizumab. Points and intervals are defined as in panel D. These analyses classify contemporaneous six-month response status and are not prospective treatment-response predictions. Small treatment-specific cohorts, especially ustekinumab and vedolizumab, should be interpreted cautiously. Participant-level precision-recall AUC and balanced accuracy, with bootstrap intervals, are provided in the source data. NR, non-responder; R, responder. (G) Focused descriptive SHAP attribution for the pooled biologic, anti-TNF, and ustekinumab TCR and BCR models; complete vedolizumab feature tables are retained in the source data and supplemental immuneML figures.

Panels C, E, and G show the six highest-ranked features from each selected receptor-specific model in paired TCR and BCR mini-panels. Bar length is normalized to the largest mean absolute SHAP value within each receptor model; raw mean absolute SHAP values are retained in the source data. Blue and orange denote TCR and BCR features. Right-pointing filled markers indicate association with the modeled positive class and left-pointing open markers indicate association with the negative class. Model annotations report classifier, chain compartment, sequence representation, k-mer length, and model-selection ROC AUC. SHAP values are full-cohort refit-model attributions. Fold-specific selected-feature records were not available, so outer-fold feature stability is not displayed and the k-mers should not be interpreted as stable antigen-specificity assignments.
"""
    (OUT / "Figure_7_Clinical_Translation_CellPress_7A_Labels_CompactGap_revised_legend.txt").write_text(
        legend, encoding="utf-8"
    )


if __name__ == "__main__":
    build()

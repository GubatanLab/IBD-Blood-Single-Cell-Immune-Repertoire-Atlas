from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Recreated Figure 6"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "TCR": "#2F6FB0",
    "BCR": "#D47A32",
    "TCR+BCR": "#7A4EAB",
    "single": "#7A7A7A",
    "paired": "#248B73",
    "grid": "#D8D8D8",
    "text": "#1E1E1E",
    "muted": "#626262",
    "null": "#A8A8A8",
}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def clean_axis(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(length=3, color="#777777")
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter, x=-0.12, y=1.07):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["text"],
    )


def task_label(task):
    return {
        "cd_vs_control": "CD vs control",
        "uc_vs_control": "UC vs control",
        "cd_vs_uc": "CD vs UC",
    }[task]


summary = pd.read_csv(OUT.parent / "ML Sensitivity Validation" / "Table_S_primary_model_validation_summary.csv")
summary = summary[summary["modality"].isin(["tcr", "bcr"])].copy()
summary["modality_label"] = summary["modality"].str.upper()

perm = pd.read_csv(OUT.parent / "ML Sensitivity Validation" / "Table_S_permutation_test_summary.csv")
perm = perm[perm["modality"].isin(["tcr", "bcr"])].copy()

pred = pd.read_csv(OUT.parent / "ML Sensitivity Validation" / "Table_S_nested_outer_fold_predictions.csv")
pred = pred[pred["modality"].isin(["tcr", "bcr"])].copy()
pred_avg = (
    pred.groupby(["modality", "task", "participant"], as_index=False)
    .agg(truth=("truth", "first"), probability=("probability", "mean"))
)

candidates = pd.read_csv(OUT.parent / "Best Models" / "Grant_BalancedAccuracy_TCR_BCR_Combined_Candidates.csv")
clinical = candidates[candidates["domain"].isin(["inflammation", "therapy_response"])].copy()
clinical["auc"] = pd.to_numeric(clinical["selection_auc_mean"], errors="coerce")
clinical["n"] = pd.to_numeric(clinical["n_paired_samples"], errors="coerce")

paired = pd.read_csv(OUT.parent / "High Impact Additional Analyses" / "Table_HI_paired_chain_nested_validation_summary.csv")
paired = paired[paired["representation"].isin(["single_chain", "paired_interaction"])].copy()


fig = plt.figure(figsize=(15.5, 18.3), facecolor="white")
outer = GridSpec(
    3,
    2,
    figure=fig,
    height_ratios=[0.82, 1.0, 1.13],
    width_ratios=[0.93, 1.07],
    hspace=0.43,
    wspace=0.30,
)
fig.suptitle(
    "Immune-receptor sequence architecture classifies IBD diagnosis and is associated with clinical state",
    fontsize=17,
    fontweight="bold",
    y=0.992,
    color=COLORS["text"],
)


# Panel A: analysis design
ax_a = fig.add_subplot(outer[0, 0])
ax_a.set_axis_off()
panel_label(ax_a, "A", x=-0.02, y=1.03)
ax_a.set_title("Participant-level immuneML analysis and validation", loc="left", pad=12)

boxes = [
    (0.02, 0.58, 0.19, 0.22, "Participant-level\nTCR and BCR\nrepertoires"),
    (0.27, 0.58, 0.19, 0.22, "CDR3 k-mers,\nV(D)J usage and\nrepertoire metrics"),
    (0.52, 0.58, 0.19, 0.22, "Inner folds:\nscaling, selection,\ntuning and model choice"),
    (0.77, 0.58, 0.20, 0.22, "Outer folds:\nparticipant-level\nheld-out predictions"),
]
for x, y, w, h, txt in boxes:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.25,
        edgecolor="#4B6F8F",
        facecolor="#EEF4F8",
        transform=ax_a.transAxes,
    )
    ax_a.add_patch(p)
    ax_a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", transform=ax_a.transAxes, color=COLORS["text"])
for x1, x2 in [(0.21, 0.27), (0.46, 0.52), (0.71, 0.77)]:
    ax_a.add_patch(
        FancyArrowPatch(
            (x1 + 0.006, 0.69),
            (x2 - 0.006, 0.69),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#4B6F8F",
            transform=ax_a.transAxes,
        )
    )

lower_boxes = [
    (0.10, 0.20, 0.23, 0.17, "Nested 3 × 5-fold\nvalidation"),
    (0.385, 0.20, 0.23, 0.17, "Permutation-derived\nnull performance"),
    (0.67, 0.20, 0.23, 0.17, "Discrimination,\ncalibration and error"),
]
for x, y, w, h, txt in lower_boxes:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor="#777777",
        facecolor="#F5F5F5",
        transform=ax_a.transAxes,
    )
    ax_a.add_patch(p)
    ax_a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", transform=ax_a.transAxes, color=COLORS["text"])
ax_a.text(
    0.5,
    0.04,
    "No cells from the same participant crossed training and test partitions",
    ha="center",
    va="bottom",
    transform=ax_a.transAxes,
    fontsize=8.5,
    color=COLORS["muted"],
)


# Panel B: diagnosis nested validation forest plot
ax_b = fig.add_subplot(outer[0, 1])
panel_label(ax_b, "B")
ax_b.set_title("Nested validation of diagnosis classifiers", loc="left", pad=12)

task_order = ["cd_vs_control", "uc_vs_control", "cd_vs_uc"]
base_y = {"cd_vs_control": 5.0, "uc_vs_control": 2.9, "cd_vs_uc": 0.8}
offset = {"TCR": 0.30, "BCR": -0.30}
for task in task_order:
    s_task = summary[summary["task"] == task]
    p_task = perm[perm["task"] == task]
    for _, row in s_task.iterrows():
        mod = row["modality_label"]
        y = base_y[task] + offset[mod]
        x = row["roc_auc_median"]
        lo = row["roc_auc_ci_low"]
        hi = row["roc_auc_ci_high"]
        ax_b.errorbar(
            x,
            y,
            xerr=np.array([[x - lo], [hi - x]]),
            fmt="o" if mod == "TCR" else "s",
            markersize=7,
            color=COLORS[mod],
            ecolor=COLORS[mod],
            elinewidth=1.5,
            capsize=3,
            zorder=3,
        )
        n95 = p_task[p_task["modality"] == row["modality"]]["null_95th_percentile"]
        if len(n95):
            ax_b.scatter(float(n95.iloc[0]), y, marker="|", s=95, color=COLORS["null"], linewidths=2.0, zorder=2)
        ax_b.text(min(hi + 0.018, 1.005), y, f"{x:.2f}", va="center", ha="left", fontsize=7.6, color=COLORS["text"])
    n = int(s_task["n"].iloc[0])
    n_pos = int(s_task["n_positive"].iloc[0])
    n_neg = int(s_task["n_negative"].iloc[0])
    ax_b.text(0.472, base_y[task], f"{task_label(task)}\n$n$={n} ({n_pos}/{n_neg})", va="center", ha="right", fontsize=8.6)

ax_b.axvline(0.5, color="#777777", linestyle="--", linewidth=1.0)
ax_b.set_xlim(0.45, 1.035)
ax_b.set_ylim(-0.1, 5.9)
ax_b.set_yticks([])
ax_b.set_xlabel("Median outer-fold ROC AUC (95% CI)")
clean_axis(ax_b, "x")
ax_b.legend(
    handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["TCR"], markeredgecolor=COLORS["TCR"], label="TCR", markersize=7),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["BCR"], markeredgecolor=COLORS["BCR"], label="BCR", markersize=7),
        Line2D([0], [0], marker="|", color=COLORS["null"], linestyle="none", label="Permutation null 95th percentile", markersize=10, markeredgewidth=2),
    ],
    loc="lower right",
    frameon=False,
    fontsize=8,
)


# Panel C: participant-averaged OOF ROC and precision-recall, CD vs UC
sub_c = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], wspace=0.35)
ax_c1 = fig.add_subplot(sub_c[0, 0])
ax_c2 = fig.add_subplot(sub_c[0, 1])
panel_label(ax_c1, "C", x=-0.28, y=1.10)
ax_c1.set_title("ROC: CD vs UC", loc="left", pad=11)
ax_c2.set_title("Precision–recall: CD vs UC", loc="left", pad=11)
curve_rows = []
for mod_key, mod_label in [("tcr", "TCR"), ("bcr", "BCR")]:
    d = pred_avg[(pred_avg["modality"] == mod_key) & (pred_avg["task"] == "cd_vs_uc")]
    y = d["truth"].to_numpy()
    prob = d["probability"].to_numpy()
    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    auc = roc_auc_score(y, prob)
    ap = average_precision_score(y, prob)
    ax_c1.plot(fpr, tpr, color=COLORS[mod_label], linewidth=2.0, label=f"{mod_label}, AUC={auc:.2f}")
    ax_c2.plot(recall, precision, color=COLORS[mod_label], linewidth=2.0, label=f"{mod_label}, AP={ap:.2f}")
    curve_rows.extend(
        [{"curve": "ROC", "modality": mod_label, "x": a, "y": b} for a, b in zip(fpr, tpr)]
        + [{"curve": "PR", "modality": mod_label, "x": a, "y": b} for a, b in zip(recall, precision)]
    )

prevalence = pred_avg[(pred_avg["modality"] == "tcr") & (pred_avg["task"] == "cd_vs_uc")]["truth"].mean()
ax_c1.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0)
ax_c2.axhline(prevalence, linestyle="--", color="#888888", linewidth=1.0, label=f"Prevalence={prevalence:.2f}")
for ax, xlabel, ylabel in [
    (ax_c1, "False-positive rate", "True-positive rate"),
    (ax_c2, "Recall", "Precision"),
]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax, "both")
    ax.legend(frameon=False, loc="lower right", fontsize=7.8)


# Panel D: calibration across diagnosis tasks
sub_d = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, 1], wspace=0.35)
cal_rows = []
for j, task in enumerate(task_order):
    ax = fig.add_subplot(sub_d[0, j])
    if j == 0:
        panel_label(ax, "D", x=-0.31, y=1.10)
    ax.set_title(task_label(task), pad=11, fontsize=9.4)
    for mod_key, mod_label, marker in [("tcr", "TCR", "o"), ("bcr", "BCR", "s")]:
        d = pred_avg[(pred_avg["modality"] == mod_key) & (pred_avg["task"] == task)]
        frac_pos, mean_pred = calibration_curve(d["truth"], d["probability"], n_bins=5, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker=marker, color=COLORS[mod_label], linewidth=1.7, markersize=4.8, label=mod_label)
        cal_rows.extend(
            [{"task": task, "modality": mod_label, "mean_predicted": a, "observed_fraction": b} for a, b in zip(mean_pred, frac_pos)]
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    if j == 0:
        ax.set_ylabel("Observed fraction")
    else:
        ax.set_yticklabels([])
    clean_axis(ax, "both")
    if j == 2:
        ax.legend(frameon=False, loc="lower right", fontsize=7.7)


# Panel E: exploratory clinical-state screening estimates
ax_e = fig.add_subplot(outer[2, 0])
panel_label(ax_e, "E")
ax_e.set_title("Clinical-state repertoire classifiers", loc="left", pad=13)
clinical_task_order = [
    "cd_inflamed_vs_noninflamed",
    "uc_inflamed_vs_noninflamed",
    "combined_biologic_nonresponder_vs_responder",
    "anti_tnf_nonresponder_vs_responder",
    "ustekinumab_nonresponder_vs_responder",
    "vedolizumab_nonresponder_vs_responder",
]
clinical_labels = {
    "cd_inflamed_vs_noninflamed": "CD inflammation",
    "uc_inflamed_vs_noninflamed": "UC inflammation",
    "combined_biologic_nonresponder_vs_responder": "All biologics: 6-month status",
    "anti_tnf_nonresponder_vs_responder": "Anti-TNF: 6-month status",
    "ustekinumab_nonresponder_vs_responder": "Ustekinumab: 6-month status†",
    "vedolizumab_nonresponder_vs_responder": "Vedolizumab: 6-month status†",
}
mod_offsets = {"TCR": 0.22, "BCR": 0.0, "TCR+BCR": -0.22}
markers = {"TCR": "o", "BCR": "s", "TCR+BCR": "D"}
base = {task: len(clinical_task_order) - 1 - i for i, task in enumerate(clinical_task_order)}
for task in clinical_task_order:
    g = clinical[clinical["comparison"] == task]
    joint_n = g[g["receptor_model"] == "TCR+BCR"]["n"].dropna()
    n_txt = f"  n={int(joint_n.iloc[0])}" if len(joint_n) else ""
    ax_e.text(0.485, base[task], clinical_labels[task] + n_txt, ha="right", va="center", fontsize=8.2)
    for _, row in g.iterrows():
        mod = row["receptor_model"]
        ax_e.scatter(
            row["auc"],
            base[task] + mod_offsets[mod],
            marker=markers[mod],
            s=45,
            facecolors="white",
            edgecolors=COLORS[mod],
            linewidths=1.7,
            zorder=3,
        )
ax_e.axvline(0.5, color="#777777", linestyle="--", linewidth=1.0)
ax_e.set_xlim(0.45, 1.02)
ax_e.set_ylim(-0.65, 5.65)
ax_e.set_yticks([])
ax_e.set_xlabel("Best mean cross-validated ROC AUC")
clean_axis(ax_e, "x")
ax_e.legend(
    handles=[
        Line2D([0], [0], marker=markers[m], color="none", markerfacecolor="white", markeredgecolor=COLORS[m], markeredgewidth=1.5, label=m, markersize=6.5)
        for m in ["TCR", "BCR", "TCR+BCR"]
    ],
    frameon=False,
    loc="lower right",
    fontsize=8,
)
ax_e.text(
    0.00,
    -0.18,
    "Open symbols: exploratory best-of-screen estimates; not fully nested. †n<40. Therapy analyses classify contemporaneous six-month response status.",
    transform=ax_e.transAxes,
    ha="left",
    va="top",
    fontsize=7.9,
    color=COLORS["muted"],
    wrap=True,
)


# Panel F: paired-chain incremental value under nested validation
ax_f = fig.add_subplot(outer[2, 1])
panel_label(ax_f, "F")
ax_f.set_title("Incremental information from paired receptor chains", loc="left", pad=13)
f_rows = []
y = 5.4
for mod in ["TCR", "BCR"]:
    for task in ["CD vs Control", "UC vs Control", "CD vs UC"]:
        g = paired[(paired["modality"] == mod) & (paired["task"] == task)]
        single = g[g["representation"] == "single_chain"].iloc[0]
        pair = g[g["representation"] == "paired_interaction"].iloc[0]
        ax_f.text(0.49, y, f"{mod}: {task}", ha="right", va="center", fontsize=8.2)
        for row, rep, marker, color in [
            (single, "Single-chain", "o", COLORS["single"]),
            (pair, "Paired interaction", "D", COLORS["paired"]),
        ]:
            x = row["pooled_outer_fold_auc"]
            lo = row["pooled_auc_bootstrap_ci_low"]
            hi = row["pooled_auc_bootstrap_ci_high"]
            yy = y + (0.14 if rep == "Paired interaction" else -0.14)
            ax_f.errorbar(
                x,
                yy,
                xerr=np.array([[x - lo], [hi - x]]),
                fmt=marker,
                markersize=5.6,
                color=color,
                ecolor=color,
                elinewidth=1.25,
                capsize=2.5,
                zorder=3,
            )
            f_rows.append({"modality": mod, "task": task, "representation": rep, "auc": x, "ci_low": lo, "ci_high": hi})
        delta = pair["pooled_outer_fold_auc"] - single["pooled_outer_fold_auc"]
        ax_f.text(1.012, y, f"Δ={delta:+.02f}", ha="right", va="center", fontsize=7.7, color=COLORS["text"])
        y -= 1.0
    y -= 0.35
ax_f.axvline(0.5, color="#777777", linestyle="--", linewidth=1.0)
ax_f.set_xlim(0.45, 1.025)
ax_f.set_ylim(-0.5, 5.95)
ax_f.set_yticks([])
ax_f.set_xlabel("Pooled outer-fold ROC AUC (bootstrap 95% CI)")
clean_axis(ax_f, "x")
ax_f.legend(
    handles=[
        Line2D([0], [0], marker="o", color=COLORS["single"], linestyle="none", label="Single-chain", markersize=6),
        Line2D([0], [0], marker="D", color=COLORS["paired"], linestyle="none", label="Paired-chain interaction", markersize=6),
    ],
    frameon=False,
    loc="lower right",
    fontsize=8,
)


fig.text(
    0.5,
    0.012,
    "Panels B–D and F use participant-level held-out predictions. Panel E summarizes exploratory immuneML screens and should not be interpreted as prospective treatment-response prediction.",
    ha="center",
    va="bottom",
    fontsize=8.2,
    color=COLORS["muted"],
)

fig.subplots_adjust(left=0.12, right=0.985, top=0.965, bottom=0.06)

png = OUT / "Figure_6_recreated_immuneML.png"
pdf = OUT / "Figure_6_recreated_immuneML.pdf"
svg = OUT / "Figure_6_recreated_immuneML.svg"
fig.savefig(png, dpi=350, facecolor="white")
fig.savefig(pdf, dpi=350, facecolor="white")
fig.savefig(svg, facecolor="white")
plt.close(fig)


# Source-data exports
summary.to_csv(OUT / "Figure_6_panel_B_nested_diagnosis_source_data.csv", index=False)
pd.DataFrame(curve_rows).to_csv(OUT / "Figure_6_panel_C_curves_source_data.csv", index=False)
pd.DataFrame(cal_rows).to_csv(OUT / "Figure_6_panel_D_calibration_source_data.csv", index=False)
clinical[
    [
        "domain",
        "comparison",
        "receptor_model",
        "model_label",
        "auc",
        "selection_balanced_accuracy_mean",
        "n",
        "n_positive",
        "n_negative",
    ]
].to_csv(OUT / "Figure_6_panel_E_clinical_state_screen_source_data.csv", index=False)
pd.DataFrame(f_rows).to_csv(OUT / "Figure_6_panel_F_paired_chain_source_data.csv", index=False)

legend = """Figure 6. Immune-receptor sequence architecture classifies IBD diagnosis and is associated with clinical state.

(A) Participant-level immuneML workflow. TCR and BCR repertoire features were analyzed using nested cross-validation in which feature preprocessing, model selection, and tuning were confined to training data; held-out outer-fold predictions were used for performance assessment. (B) Median outer-fold ROC AUC and 95% intervals for TCR and BCR diagnosis classifiers. Gray ticks indicate the 95th percentile of permutation-derived null performance. Sample sizes are shown as total participants (positive/negative class). (C) Participant-averaged out-of-fold receiver-operating-characteristic and precision-recall curves for CD versus UC classification. The dashed precision-recall reference denotes outcome prevalence. (D) Calibration curves generated from participant-averaged outer-fold probabilities for diagnosis classifiers. (E) Cross-validated ROC AUCs for objective-inflammation and six-month post-treatment response-status classifiers. Open symbols denote exploratory best-of-screen estimates that have not undergone the fully nested validation used in panels B–D; paired TCR+BCR sample sizes are shown, and analyses with fewer than 40 participants are marked with a dagger. These therapy analyses classify contemporaneous six-month response status and do not predict future response. (F) Nested sensitivity analysis comparing single-chain and paired-chain-interaction representations. Points show pooled outer-fold ROC AUC with participant-bootstrap 95% confidence intervals; delta values denote the paired-interaction minus single-chain AUC.
"""
(OUT / "Figure_6_legend.txt").write_text(legend, encoding="utf-8")

print(png)
print(pdf)
print(svg)

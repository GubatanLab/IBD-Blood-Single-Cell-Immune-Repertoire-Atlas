from pathlib import Path
from collections import defaultdict
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path("C:/path/to/private-manuscript-workspace")
OUT = ROOT / "Figure 7 Revised Immunity"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "TCR": "#0072B2",
    "BCR": "#D55E00",
    "Joint": "#009E73",
    "CD": "#8B3A3A",
    "UC": "#4C78A8",
    "Control": "#7A7A7A",
    "ink": "#202124",
    "muted": "#747474",
    "grid": "#D9D9D9",
    "light": "#F4F5F6",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.2,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def read_csv(rel):
    return pd.read_csv(ROOT / rel)


def style_axis(ax, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["ink"])
    ax.spines["bottom"].set_color(COL["ink"])
    ax.tick_params(length=3, width=0.65, color=COL["ink"], pad=2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COL["grid"], lw=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter, x=-0.14, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", va="top", ha="left", color=COL["ink"])


def panel_title(ax, title):
    ax.set_title(title, loc="left", pad=5, fontweight="bold", color=COL["ink"])


def rounded_box(ax, xy, width, height, text, edge, face="#FFFFFF", fontsize=5.8):
    box = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.85, edgecolor=edge, facecolor=face, transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=fontsize, color=COL["ink"], linespacing=1.1)


def workflow_panel(ax):
    ax.set_axis_off()
    panel_title(ax, "immuneML participant-level modeling and interpretation")
    panel_label(ax, "A", x=-0.09, y=1.08)

    y = 0.48
    xs = [0.02, 0.245, 0.49, 0.74]
    widths = [0.18, 0.19, 0.20, 0.23]
    labels = [
        "Paired CDR3\nrepertoires",
        "immuneML\nAA/NT k-mers",
        "LR/SVM screen\nby chain set",
        "Held-out validation\n+ refit SHAP",
    ]
    edges = [COL["Joint"], COL["muted"], COL["muted"], COL["Joint"]]
    for x, w, label, edge in zip(xs, widths, labels, edges):
        rounded_box(ax, (x, y), w, 0.27, label, edge=edge, face="#FAFAFA")
    for i in range(3):
        ax.add_patch(FancyArrowPatch(
            (xs[i] + widths[i] + 0.008, y + 0.135), (xs[i + 1] - 0.008, y + 0.135),
            arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
            color=COL["muted"], transform=ax.transAxes,
        ))

    ax.text(0.035, 0.33, "TCR", color=COL["TCR"], fontweight="bold", transform=ax.transAxes)
    ax.text(0.105, 0.33, "CASSLGQGYEQYF", family="monospace", fontsize=6.4, transform=ax.transAxes)
    ax.add_patch(Rectangle((0.105, 0.305), 0.052, 0.015, transform=ax.transAxes,
                           color=COL["TCR"], alpha=0.25, lw=0))
    ax.text(0.035, 0.20, "BCR", color=COL["BCR"], fontweight="bold", transform=ax.transAxes)
    ax.text(0.105, 0.20, "CARGMDVWGQGT", family="monospace", fontsize=6.4, transform=ax.transAxes)
    ax.add_patch(Rectangle((0.145, 0.175), 0.052, 0.015, transform=ax.transAxes,
                           color=COL["BCR"], alpha=0.25, lw=0))
    ax.text(0.49, 0.20, "Training-fold preprocessing; full-cohort refit SHAP",
            fontsize=5.3, color=COL["muted"], transform=ax.transAxes)


def diagnosis_forest(ax, nested, null):
    panel_title(ax, "Nested diagnosis classification")
    panel_label(ax, "B", x=-0.12, y=1.08)
    tasks = ["cd_vs_control", "uc_vs_control", "cd_vs_uc"]
    labels = ["CD vs control", "UC vs control", "CD vs UC"]
    ybase = np.arange(3)[::-1]
    offsets = {"tcr": 0.11, "bcr": -0.11}
    for mod in ["tcr", "bcr"]:
        sub = nested[(nested.modality == mod)].set_index("task").reindex(tasks)
        y = ybase + offsets[mod]
        lo = sub.roc_auc_ci_low.to_numpy(float)
        mid = sub.roc_auc_median.to_numpy(float)
        hi = sub.roc_auc_ci_high.to_numpy(float)
        ax.errorbar(mid, y, xerr=[mid - lo, hi - mid], fmt="o", ms=4.2,
                    capsize=2, elinewidth=1.0, markeredgewidth=0,
                    color=COL[mod.upper()], label=mod.upper(), zorder=3)
        nsub = null[(null.modality == mod)].set_index("task").reindex(tasks)
        for yy, xnull in zip(y, nsub.null_95th_percentile.astype(float)):
            ax.plot([xnull, xnull], [yy - 0.055, yy + 0.055], color="#9B9B9B", lw=1.2, zorder=2)
    ax.axvline(0.5, color=COL["muted"], ls="--", lw=0.7)
    ax.set_xlim(0.45, 1.015)
    ax.set_ylim(-0.45, 2.45)
    ax.set_yticks(ybase, labels)
    ax.set_xlabel("Median outer-fold ROC AUC (95% CI)")
    for yy, task in zip(ybase, tasks):
        row = nested[(nested.modality == "tcr") & (nested.task == task)].iloc[0]
        ax.text(0.455, yy - 0.27, f"n={int(row.n)} ({int(row.n_positive)}/{int(row.n_negative)})",
                fontsize=5.8, color=COL["muted"], va="center")
    handles = [
        Line2D([0], [0], marker="o", color=COL["TCR"], lw=0, label="TCR"),
        Line2D([0], [0], marker="o", color=COL["BCR"], lw=0, label="BCR"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(0.01, 0.96), ncol=1, handletextpad=0.25,
              borderaxespad=0.0, fontsize=5.9)
    style_axis(ax, "x")


def integrated_performance_panel(ax, nested, null, clinical):
    """Place confirmatory diagnosis and exploratory clinical screens on one scale."""
    panel_title(ax, "Model performance and validation hierarchy")
    panel_label(ax, "B", x=-0.055, y=1.055)

    rows = [
        ("header", "DIAGNOSIS — FULLY NESTED", None, None),
        ("nested", "CD vs control", "cd_vs_control", None),
        ("nested", "UC vs control", "uc_vs_control", None),
        ("nested", "CD vs UC", "cd_vs_uc", None),
        ("header", "TISSUE INFLAMMATION — EXPLORATORY", None, None),
        ("clinical", "CD: inflamed vs noninflamed", "cd_inflamed_vs_noninflamed", "inflammation"),
        ("clinical", "UC: inflamed vs noninflamed", "uc_inflamed_vs_noninflamed", "inflammation"),
        ("header", "THERAPY RESPONSE STATUS — EXPLORATORY", None, None),
        ("clinical", "All biologics: NR vs R", "combined_biologic_nonresponder_vs_responder", "therapy_response"),
        ("clinical", "Anti-TNF: NR vs R", "anti_tnf_nonresponder_vs_responder", "therapy_response"),
        ("clinical", "Ustekinumab: NR vs R", "ustekinumab_nonresponder_vs_responder", "therapy_response"),
        ("clinical", "Vedolizumab: NR vs R", "vedolizumab_nonresponder_vs_responder", "therapy_response"),
    ]
    y_positions = np.arange(len(rows))[::-1]
    offsets = {"TCR": 0.16, "BCR": -0.16, "TCR+BCR": 0.0}
    markers = {"TCR": "o", "BCR": "s", "TCR+BCR": "D"}
    colors = {"TCR": COL["TCR"], "BCR": COL["BCR"], "TCR+BCR": COL["Joint"]}

    for yy, (kind, label, task, domain) in zip(y_positions, rows):
        if kind == "header":
            ax.axhspan(yy - 0.38, yy + 0.38, color=COL["light"], zorder=0)
            ax.text(0.455, yy, label, fontsize=5.8, fontweight="bold",
                    color=COL["muted"], va="center")
            continue
        if kind == "nested":
            for modality, display in [("tcr", "TCR"), ("bcr", "BCR")]:
                row = nested[(nested.modality == modality) & (nested.task == task)].iloc[0]
                mid, lo, hi = [float(row[x]) for x in
                               ["roc_auc_median", "roc_auc_ci_low", "roc_auc_ci_high"]]
                ax.errorbar(mid, yy + offsets[display], xerr=[[mid - lo], [hi - mid]],
                            fmt=markers[display], ms=4.0, capsize=1.8, elinewidth=0.9,
                            markeredgewidth=0, color=colors[display], zorder=3)
                nrow = null[(null.modality == modality) & (null.task == task)].iloc[0]
                xnull = float(nrow.null_95th_percentile)
                ax.plot([xnull, xnull], [yy + offsets[display] - 0.055,
                                        yy + offsets[display] + 0.055],
                        color="#9B9B9B", lw=1.0, zorder=2)
        else:
            sub = clinical[(clinical.domain == domain) & (clinical.comparison == task)]
            for display in ["TCR", "BCR", "TCR+BCR"]:
                row = sub[sub.receptor_model == display]
                if row.empty:
                    continue
                ax.scatter(float(row.iloc[0].auc), yy + offsets[display], s=19,
                           marker=markers[display], facecolor="white",
                           edgecolor=colors[display], linewidth=0.95, zorder=3)

    data_ticks = [(yy, label) for yy, (kind, label, _, _) in zip(y_positions, rows)
                  if kind != "header"]
    ax.set_yticks([x[0] for x in data_ticks], [x[1] for x in data_ticks])
    ax.axvline(0.5, color=COL["muted"], ls="--", lw=0.65)
    ax.set_xlim(0.45, 1.015)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("ROC AUC (95% CI for nested diagnosis models)")
    handles = [
        Line2D([0], [0], marker=markers[x], color="none", markerfacecolor="white",
               markeredgecolor=colors[x], label=x, markersize=4.5)
        for x in ["TCR", "BCR", "TCR+BCR"]
    ]
    handles.extend([
        Line2D([0], [0], marker="o", color=COL["muted"], markerfacecolor=COL["muted"],
               lw=0, label="nested estimate", markersize=4),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COL["muted"], lw=0, label="best-of-screen", markersize=4),
    ])
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=5,
              bbox_to_anchor=(1.0, 1.075), borderaxespad=0, columnspacing=0.85,
              handletextpad=0.25, fontsize=5.4)
    style_axis(ax, "x")


def merge_kmers(tokens):
    """Greedy shortest-superstring display for overlapping equal-length k-mers."""
    tokens = list(dict.fromkeys(tokens))
    if not tokens:
        return ""
    current = tokens.pop(0)
    while tokens:
        best = None
        for i, token in enumerate(tokens):
            for overlap in range(min(len(current), len(token)) - 1, 1, -1):
                if current.endswith(token[:overlap]):
                    candidate = current + token[overlap:]
                    score = overlap
                    if best is None or score > best[0]:
                        best = (score, i, candidate)
                if token.endswith(current[:overlap]):
                    candidate = token + current[overlap:]
                    score = overlap
                    if best is None or score > best[0]:
                        best = (score, i, candidate)
        if best is None:
            break
        _, idx, current = best
        tokens.pop(idx)
    return current


def feature_families(shap, domain, comparison, receptor, max_families=2):
    candidates = shap[
        (shap.receptor == receptor)
        & (shap.domain == domain)
        & (shap.comparison == comparison)
    ].copy()
    # Prefer the selected receptor-specific amino-acid model. If that model is
    # nucleotide based, use amino-acid features attributed to the receptor in
    # the corresponding joint TCR+BCR model.
    specific = candidates[
        candidates.chain_group.notna()
        & candidates.sequence_type.eq("aa")
    ].copy()
    if not specific.empty:
        d = specific
        provenance = "receptor-specific amino-acid model"
    else:
        d = candidates[
            candidates.chain_group.isna()
            & candidates.model_label.fillna("").str.contains(r"\bAA\b", regex=True)
        ].copy()
        provenance = "joint TCR+BCR amino-acid model"
    if d.empty:
        raise ValueError(f"No amino-acid attribution rows for {domain}/{comparison}/{receptor}")
    d["token"] = d.feature.str.split(":").str[-1]
    d["importance"] = pd.to_numeric(d.mean_absolute_shap)
    d["sign"] = np.where(d.direction.str.startswith("+"), 1, -1)

    # Connect same-direction k-mers sharing a 3-aa prefix/suffix.
    parent = list(range(len(d)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
    toks = d.token.tolist()
    signs = d.sign.tolist()
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            if signs[i] != signs[j]:
                continue
            if len(toks[i]) == len(toks[j]) and (
                toks[i][1:] == toks[j][:-1] or toks[j][1:] == toks[i][:-1]
            ):
                union(i, j)
    groups = defaultdict(list)
    for i in range(len(d)):
        groups[find(i)].append(i)
    rows = []
    for inds in groups.values():
        sub = d.iloc[inds]
        merged = merge_kmers(sub.sort_values("importance", ascending=False).token.tolist())
        rows.append({
            "domain": domain,
            "comparison": comparison,
            "receptor": receptor,
            "family": merged,
            "members": ";".join(sub.token.tolist()),
            "direction": "positive" if sub.sign.iloc[0] > 0 else "negative",
            "signed_direction": int(sub.sign.iloc[0]),
            "importance": sub.importance.sum(),
            "model_label": sub.model_label.iloc[0],
            "chain_group": sub.chain_group.iloc[0] if sub.chain_group.notna().any() else "joint model",
            "feature_provenance": provenance,
        })
    families = pd.DataFrame(rows).sort_values("importance", ascending=False)
    selected = []
    for sign in [1, -1]:
        q = families[families.signed_direction == sign]
        if not q.empty:
            selected.append(q.head(1))
    out = pd.concat(selected, ignore_index=True) if selected else families.head(max_families).copy()
    if len(out) < max_families:
        used = set(out.family)
        fill = families[~families.family.isin(used)].head(max_families - len(out))
        out = pd.concat([out, fill], ignore_index=True)
    out["relative_importance"] = out.importance / out.importance.max()
    return out


def motif_axis(ax, fam, receptor):
    fam = fam.sort_values("relative_importance", ascending=True)
    y = np.arange(len(fam))
    signed = fam.relative_importance * np.where(fam.direction.eq("CD"), 1, -1)
    colors = [COL["CD"] if v > 0 else COL["UC"] for v in signed]
    ax.hlines(y, 0, signed, color=colors, lw=2.6, alpha=0.75)
    ax.scatter(signed, y, color=colors, s=22, zorder=3, edgecolor="white", lw=0.5)
    ax.axvline(0, color=COL["ink"], lw=0.65)
    ax.set_yticks(y, fam.family, family="monospace")
    ax.set_xlim(-1.10, 1.10)
    ax.set_xticks([-1, 0, 1], ["UC", "0", "CD"])
    ax.set_xlabel("Relative model contribution and direction")
    ax.set_title(receptor, loc="left", color=COL[receptor], fontweight="bold", pad=3)
    style_axis(ax, "x")


def motif_comparison_axis(ax, fam, title, negative_label, positive_label):
    fam = fam.copy()
    fam["signed_importance"] = fam.relative_importance * fam.signed_direction
    fam = fam.sort_values(["receptor", "signed_direction", "importance"],
                          ascending=[False, False, True]).reset_index(drop=True)
    y = np.arange(len(fam))[::-1]
    colors = [COL[r] for r in fam.receptor]
    markers = ["o" if r == "TCR" else "s" for r in fam.receptor]
    for yy, val, color, marker in zip(y, fam.signed_importance, colors, markers):
        ax.hlines(yy, 0, val, color=color, lw=2.1, alpha=0.75)
        ax.scatter(val, yy, color=color, marker=marker, s=17,
                   edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(0, color=COL["ink"], lw=0.6)
    ax.set_yticks(y, fam.family, family="monospace", fontsize=5.5)
    for tick, receptor in zip(ax.get_yticklabels(), fam.receptor):
        tick.set_color(COL[receptor])
    ax.set_xlim(-1.08, 1.08)
    ax.set_xticks([-1, 0, 1], [negative_label, "", positive_label])
    ax.tick_params(axis="x", labelsize=5.3)
    ax.set_title(title, fontsize=6.6, fontweight="bold", pad=3)
    style_axis(ax, "x")


def motif_grid_panel(fig, spec, shap, letter, title, tasks):
    outer = fig.add_subplot(spec)
    outer.set_axis_off()
    panel_title(outer, title)
    panel_label(outer, letter, x=-0.07, y=1.07)
    inner = GridSpecFromSubplotSpec(2, len(tasks), subplot_spec=spec,
                                    height_ratios=[0.18, 0.82], hspace=0.0,
                                    wspace=0.55 if len(tasks) <= 3 else 0.72)
    rows = []
    for j, task in enumerate(tasks):
        fam = pd.concat([
            feature_families(shap, task["domain"], task["comparison"], "TCR"),
            feature_families(shap, task["domain"], task["comparison"], "BCR"),
        ], ignore_index=True)
        rows.append(fam)
        ax = fig.add_subplot(inner[1, j])
        motif_comparison_axis(
            ax, fam, task["title"], task["negative_label"], task["positive_label"]
        )
    return pd.concat(rows, ignore_index=True)


def best_model_shap_features(shap, domain, comparison, receptor, n_features=2):
    """Top SHAP features from the selected receptor-specific model."""
    d = shap[
        shap.domain.eq(domain)
        & shap.comparison.eq(comparison)
        & shap.receptor.eq(receptor)
        & shap.chain_group.notna()
    ].copy()
    if d.empty:
        raise ValueError(f"No selected best-model SHAP rows for {domain}/{comparison}/{receptor}")
    d["mean_absolute_shap"] = pd.to_numeric(d.mean_absolute_shap)
    d["rank"] = pd.to_numeric(d["rank"])
    d = d.sort_values(["rank", "mean_absolute_shap"], ascending=[True, False]).head(n_features)
    d["feature_token"] = d.feature.str.split(":").str[-1]
    d["direction_sign"] = np.where(d.direction.str.startswith("+"), 1, -1)
    d["relative_mean_absolute_shap"] = d.mean_absolute_shap / d.mean_absolute_shap.max()
    chain_labels = {
        "tcr_alpha_beta": "a+b", "tcr_gamma_delta": "g+d",
        "tcr_alpha": "a", "tcr_beta": "b", "tcr_gamma": "g", "tcr_delta": "d",
        "bcr_heavy_light": "H+L", "bcr_heavy": "H", "bcr_light": "L",
    }
    d["model_display"] = d.apply(
        lambda row: (
            f"{row.receptor}: {row.model_name} "
            f"{chain_labels.get(row.chain_group, row.chain_group)} "
            f"{str(row.sequence_type).upper()} K{int(row.k)} | sel. AUC {row.selection_auc_mean:.2f}"
        ),
        axis=1,
    )
    d["feature_stability_status"] = "Not estimated; full-cohort refit SHAP"
    keep = [
        "domain", "comparison", "receptor", "feature", "feature_token",
        "mean_absolute_shap", "relative_mean_absolute_shap", "direction", "direction_sign", "rank",
        "model_name", "model_label", "chain_group", "sequence_type", "k",
        "selection_auc_mean", "selection_balanced_accuracy_mean", "selection_source",
        "comparison_label", "positive_label", "n_samples_refit", "n_positive_refit",
        "n_negative_refit", "shap_method", "model_display", "feature_stability_status",
    ]
    return d[keep]


def shap_comparison_axis(ax, features, title, negative_label, positive_label):
    d = features.copy()
    d["receptor_order"] = pd.Categorical(d.receptor, ["BCR", "TCR"], ordered=True)
    d = d.sort_values(["receptor_order", "relative_mean_absolute_shap"], ascending=[True, True])
    y = np.arange(len(d), dtype=float)
    colors = [COL[r] for r in d.receptor]
    for yy, (_, row), color in zip(y, d.iterrows(), colors):
        positive = row.direction_sign > 0
        ax.barh(
            yy, row.relative_mean_absolute_shap, height=0.64,
            color=color, edgecolor=color, alpha=0.78,
            linewidth=0.8, zorder=3,
        )
        ax.scatter(
            row.relative_mean_absolute_shap, yy, marker=">" if positive else "<", s=15,
            facecolor=color if positive else "white", edgecolor=color,
            linewidth=0.8, zorder=4,
        )
    ax.set_yticks(y, d.feature_token, family="monospace", fontsize=5.4)
    for tick, receptor in zip(ax.get_yticklabels(), d.receptor):
        tick.set_color(COL[receptor])
    ax.set_xlabel(
        f"Relative mean |SHAP|; > {positive_label}, < {negative_label}",
        fontsize=5.1, labelpad=2,
    )
    ax.set_title(title, fontsize=6.6, fontweight="bold", pad=25)
    metadata = d.groupby("receptor", observed=True).model_display.first().to_dict()
    ax.text(0.5, 1.105, metadata.get("TCR", ""), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=4.65, color=COL["TCR"])
    ax.text(0.5, 1.045, metadata.get("BCR", ""), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=4.65, color=COL["BCR"])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.5, 1.0])
    ax.margins(y=0.08)
    style_axis(ax, "x")


def shap_grid_panel(fig, spec, shap, letter, title, tasks, show_legend=False):
    outer = fig.add_subplot(spec)
    outer.set_axis_off()
    panel_title(outer, title)
    panel_label(outer, letter, x=-0.07, y=1.07)
    inner = GridSpecFromSubplotSpec(
        2, len(tasks), subplot_spec=spec, height_ratios=[0.34, 0.66],
        hspace=0.0, wspace=0.52 if len(tasks) <= 3 else 0.68,
    )
    rows = []
    for j, task in enumerate(tasks):
        features = pd.concat([
            best_model_shap_features(shap, task["domain"], task["comparison"], "TCR"),
            best_model_shap_features(shap, task["domain"], task["comparison"], "BCR"),
        ], ignore_index=True)
        rows.append(features)
        shap_comparison_axis(
            fig.add_subplot(inner[1, j]), features, task["title"],
            task["negative_label"], task["positive_label"],
        )
    if show_legend:
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=COL["TCR"], edgecolor=COL["TCR"], label="TCR"),
            Rectangle((0, 0), 1, 1, facecolor=COL["BCR"], edgecolor=COL["BCR"], label="BCR"),
            Line2D([0], [0], marker=">", color="none", markerfacecolor=COL["muted"],
                   markeredgecolor=COL["muted"], label="positive class"),
            Line2D([0], [0], marker="<", color="none", markerfacecolor="white",
                   markeredgecolor=COL["muted"], label="negative class"),
        ]
        outer.legend(
            handles=legend_handles, frameon=False, ncol=4, loc="upper right",
            bbox_to_anchor=(1.0, 1.08), fontsize=5.3, handlelength=1.2,
            columnspacing=0.8, borderaxespad=0.0,
        )
    return pd.concat(rows, ignore_index=True)


def feature_panel(fig, spec, shap):
    outer = fig.add_subplot(spec)
    outer.set_axis_off()
    panel_title(outer, "Representative predictive CDR3 feature families: CD versus UC")
    panel_label(outer, "C", x=-0.07, y=1.07)
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, height_ratios=[0.13, 0.87],
                                    hspace=0.02, wspace=0.47)
    allfam = []
    for j, receptor in enumerate(["TCR", "BCR"]):
        fam = feature_families(shap, receptor)
        allfam.append(fam)
        ax = fig.add_subplot(inner[1, j])
        motif_axis(ax, fam, receptor)
    pd.concat(allfam, ignore_index=True).to_csv(OUT / "Figure_7C_predictive_kmer_families_source_data.csv", index=False)


def curve_panel(fig, spec, curves, predictions, letter="D"):
    outer = fig.add_subplot(spec)
    outer.set_axis_off()
    panel_title(outer, "Held-out discrimination: CD versus UC")
    panel_label(outer, letter, x=-0.13, y=1.08)
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.45)
    pooled = predictions[(predictions.modality.isin(["tcr", "bcr"])) &
                         (predictions.task == "cd_vs_uc")].groupby(
        ["modality", "participant", "truth"], as_index=False).probability.mean()
    prevalence = pooled.drop_duplicates(["participant", "truth"]).truth.mean()
    for j, curve_name in enumerate(["ROC", "PR"]):
        ax = fig.add_subplot(inner[0, j])
        for mod in ["TCR", "BCR"]:
            cc = curves[(curves.curve == curve_name) & (curves.modality == mod)]
            sub = pooled[pooled.modality == mod.lower()]
            metric = roc_auc_score(sub.truth, sub.probability) if curve_name == "ROC" else average_precision_score(sub.truth, sub.probability)
            ax.plot(cc.x, cc.y, color=COL[mod], lw=1.35, label=f"{mod} {metric:.2f}")
        if curve_name == "ROC":
            ax.plot([0, 1], [0, 1], color=COL["muted"], ls="--", lw=0.7)
            ax.set_xlabel("False-positive rate")
            ax.set_ylabel("True-positive rate")
        else:
            ax.axhline(prevalence, color=COL["muted"], ls="--", lw=0.7)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.02, 0.98, curve_name, transform=ax.transAxes, ha="left", va="top",
                fontweight="bold", fontsize=7.3)
        ax.legend(frameon=False, loc="lower right", title="AUC" if curve_name == "ROC" else "AP")
        style_axis(ax)


def delta_auc_panel(ax, delta, letter="E"):
    panel_title(ax, "Incremental information from paired chains")
    panel_label(ax, letter, x=-0.14, y=1.08)
    d = delta[delta.comparison == "paired_interaction vs single_chain"].copy()
    tasks = ["CD vs Control", "UC vs Control", "CD vs UC"]
    ybase = np.arange(3)[::-1]
    offsets = {"TCR": 0.11, "BCR": -0.11}
    for mod in ["TCR", "BCR"]:
        sub = d[d.modality == mod].set_index("task").reindex(tasks)
        mid = sub.delta_pooled_auc.astype(float).to_numpy()
        lo = sub.delta_auc_ci_low.astype(float).to_numpy()
        hi = sub.delta_auc_ci_high.astype(float).to_numpy()
        ax.errorbar(mid, ybase + offsets[mod], xerr=[mid - lo, hi - mid],
                    fmt="D" if mod == "TCR" else "o", ms=4.0, capsize=2,
                    lw=1.0, color=COL[mod], label=mod, zorder=3)
    ax.axvline(0, color=COL["ink"], lw=0.7)
    ax.set_yticks(ybase, tasks)
    ax.set_xlim(-0.12, 0.22)
    ax.set_xlabel("Paired-interaction minus single-chain AUC (95% CI)")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, "x")
    d.to_csv(OUT / "Figure_7G_paired_chain_delta_auc_source_data.csv", index=False)


def clinical_model_axis(ax, clinical, domain, letter, title, order, labels):
    panel_title(ax, title)
    panel_label(ax, letter, x=-0.14, y=1.08)
    d = clinical[clinical.domain == domain].copy()
    ybase = np.arange(len(order))[::-1]
    ymap = dict(zip(order, ybase))
    offsets = {"TCR": 0.16, "BCR": -0.16, "TCR+BCR": 0.0}
    markers = {"TCR": "o", "BCR": "s", "TCR+BCR": "D"}
    colors = {"TCR": COL["TCR"], "BCR": COL["BCR"], "TCR+BCR": COL["Joint"]}
    for model in ["TCR", "BCR", "TCR+BCR"]:
        sub = d[d.receptor_model == model].set_index("comparison").reindex(order)
        ys = np.array([ymap[x] for x in order], float) + offsets[model]
        ax.scatter(sub.auc.astype(float), ys, s=25, marker=markers[model],
                   facecolor="white", edgecolor=colors[model], lw=1.1,
                   label=model, zorder=3)
    ax.axvline(0.5, color=COL["muted"], ls="--", lw=0.7)
    ax.set_yticks(ybase, [labels[x] for x in order])
    ax.set_xlim(0.48, 1.015)
    ax.set_ylim(-0.48, len(order) - 0.52)
    ax.set_xlabel("Exploratory cross-validated ROC AUC")
    ax.legend(frameon=False, loc="lower right", ncol=1, handletextpad=0.3)
    style_axis(ax, "x")
    for comparison, yy in ymap.items():
        row = d[(d.comparison == comparison) & (d.receptor_model == "TCR+BCR")]
        if not row.empty and pd.notna(row.iloc[0].n):
            r = row.iloc[0]
            ax.text(0.49, yy - 0.31, f"n={int(r.n)} ({int(r.n_positive)}/{int(r.n_negative)})",
                    fontsize=5.5, color=COL["muted"], va="center")


def build():
    nested = read_csv("Recreated Figure 6/Figure_6_panel_B_nested_diagnosis_source_data.csv")
    null = read_csv("ML Sensitivity Validation/Table_S_permutation_test_summary.csv")
    paired_delta = read_csv("High Impact Additional Analyses/Table_HI_paired_chain_incremental_value.csv")
    clinical = read_csv("Recreated Figure 6/Figure_6_panel_E_clinical_state_screen_source_data.csv")
    shap = read_csv("grant_combined_tcr_bcr_figures/Grant_All_TCR_BCR_Combined_Best_Model_SHAP_Top20_Features.csv")

    fig = plt.figure(figsize=(7.48, 10.25), facecolor="white")
    gs = GridSpec(
        5, 2, figure=fig,
        height_ratios=[0.62, 2.2, 0.88, 0.88, 1.0],
        hspace=0.66, wspace=0.48,
        left=0.165, right=0.985, top=0.982, bottom=0.058,
    )
    workflow_panel(fig.add_subplot(gs[0, :]))
    integrated_performance_panel(fig.add_subplot(gs[1, :]), nested, null, clinical)
    shap_rows = []
    shap_rows.append(shap_grid_panel(fig, gs[2, :], shap, "C", "Diagnosis: selected-model CDR3 features", [
        {"domain": "diagnosis", "comparison": "cd_vs_control", "title": "CD vs control", "negative_label": "Control", "positive_label": "CD"},
        {"domain": "diagnosis", "comparison": "uc_vs_control", "title": "UC vs control", "negative_label": "Control", "positive_label": "UC"},
        {"domain": "diagnosis", "comparison": "cd_vs_uc", "title": "CD vs UC", "negative_label": "UC", "positive_label": "CD"},
    ]))
    shap_rows.append(shap_grid_panel(fig, gs[3, :], shap, "D", "Tissue inflammation: selected-model CDR3 features", [
        {"domain": "inflammation", "comparison": "cd_inflamed_vs_noninflamed", "title": "CD inflammation", "negative_label": "Noninfl.", "positive_label": "Inflamed"},
        {"domain": "inflammation", "comparison": "uc_inflamed_vs_noninflamed", "title": "UC inflammation", "negative_label": "Noninfl.", "positive_label": "Inflamed"},
    ]))
    shap_rows.append(shap_grid_panel(fig, gs[4, 0], shap, "E", "Therapy response: focused CDR3 features", [
        {"domain": "therapy_response", "comparison": "combined_biologic_nonresponder_vs_responder", "title": "All biologics", "negative_label": "R", "positive_label": "NR"},
        {"domain": "therapy_response", "comparison": "ustekinumab_nonresponder_vs_responder", "title": "Ustekinumab", "negative_label": "R", "positive_label": "NR"},
    ]))
    pd.concat(shap_rows, ignore_index=True).to_csv(
        OUT / "Figure_7CDE_CellPress_SHAP_source_data.csv", index=False
    )
    complete_therapy_rows = []
    for comparison in [
        "combined_biologic_nonresponder_vs_responder",
        "anti_tnf_nonresponder_vs_responder",
        "ustekinumab_nonresponder_vs_responder",
        "vedolizumab_nonresponder_vs_responder",
    ]:
        for receptor in ["TCR", "BCR"]:
            complete_therapy_rows.append(
                best_model_shap_features(
                    shap, "therapy_response", comparison, receptor, n_features=20
                )
            )
    pd.concat(complete_therapy_rows, ignore_index=True).to_csv(
        OUT / "Figure_7E_complete_therapy_SHAP_source_data.csv", index=False
    )
    delta_auc_panel(fig.add_subplot(gs[4, 1]), paired_delta, letter="F")
    nested_export = nested.merge(
        null[["modality", "task", "null_95th_percentile"]],
        on=["modality", "task"], how="left",
    ).assign(evidence_tier="fully nested diagnosis")
    clinical_export = clinical.assign(evidence_tier="exploratory best-of-screen")
    pd.concat([nested_export, clinical_export], ignore_index=True, sort=False).to_csv(
        OUT / "Figure_7B_integrated_performance_source_data.csv", index=False
    )

    stem = OUT / "Figure_7_CellPress_revised"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=400, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    legend = """Figure 7. Nested validation links distributed TCR and BCR sequence features to diagnosis and clinical state.

(A) Participant-level immuneML modeling workflow. Productive paired TCR and BCR CDR3 repertoires were encoded as amino-acid or nucleotide k-mer frequencies across prespecified receptor-chain compartments. Logistic-regression and support-vector-machine models were compared with preprocessing and selection confined to training folds. Held-out predictions were used for validation, whereas SHAP values were calculated after refitting the selected model to the full analysis cohort. Sequence examples are schematic and illustrate k-mer encoding.

(B) Integrated performance summary across diagnosis, tissue inflammation, and post-treatment response status. Filled symbols and horizontal intervals show median outer-fold ROC AUC and 95% confidence intervals for fully nested TCR and BCR diagnosis classifiers; gray ticks indicate the 95th percentile of the corresponding fixed-pipeline permutation null. Open symbols show exploratory best-of-screen cross-validated ROC AUCs for TCR, BCR, and joint TCR+BCR clinical-state classifiers. Therapy analyses classify contemporaneous six-month response status and are not prospective treatment-response predictions. NR, non-responder; R, responder.

(C) Selected immuneML receptor-model SHAP barplots for diagnosis classifiers comparing CD with control, UC with control, and CD with UC. (D) Corresponding barplots for inflamed versus noninflamed tissue within CD and UC. (E) Focused therapy-response-status barplots for the pooled biologic cohort and ustekinumab, the strongest treatment-specific screen; complete anti-TNF and vedolizumab displays are retained in the supplemental immuneML figures and source data. Panels C-E show the two highest-ranked features from each selected receptor-specific model. Colored subtitles report receptor, classifier, chain compartment, sequence representation, k-mer length, and model-selection ROC AUC. Bar length is normalized to the largest mean absolute SHAP value within each receptor model, preventing direct comparison of raw attribution scales across separately fitted models; raw mean absolute SHAP values are retained in the source data. Blue and orange denote TCR and BCR features. Right-pointing filled markers indicate association with the modeled positive class and left-pointing open markers indicate association with the negative class. AA and NT identify amino-acid and nucleotide k-mer models. SHAP values are descriptive full-cohort refit-model attributions. Fold-specific selected-feature records were not available, so outer-fold feature stability is not displayed; these features should not be interpreted as stable antigen-specificity assignments.

(F) Incremental value of paired-chain interaction representations relative to single-chain representations. Points show the difference in pooled outer-fold ROC AUC; intervals are paired participant-bootstrap 95% confidence intervals. Positive values favor the paired-interaction representation.
"""
    (OUT / "Figure_7_CellPress_revised_legend.txt").write_text(legend, encoding="utf-8")


if __name__ == "__main__":
    build()

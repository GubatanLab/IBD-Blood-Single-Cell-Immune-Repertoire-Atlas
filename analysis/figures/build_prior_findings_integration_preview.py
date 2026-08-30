from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "High Impact Prior Findings Integration Preview 20260829"
SRC = OUT / "source_data"
OUT.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

GLIPH_ROOT = ROOT / "High Impact Additional Analyses" / "Priority Analyses"
FIG3_ROOT = ROOT / "High Impact Additional Analyses" / "Figure 3 High Impact Revision"
TSS_ROOT = ROOT / "High Impact Additional Analyses" / "Paired TCR Sequence State"

BLUE = "#137CBD"
ORANGE = "#E66A00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GRAY = "#777777"
LIGHT = "#E9EEF3"
RED = "#C83E4D"

LABELS = {
    "EOMES_ZEB2_inflammatory_CD8_TRM_like": "EOMES–ZEB2 CD8/TRM-like",
    "Effector_cytotoxicity": "Effector cytotoxicity",
    "Th1_Tc1_inflammatory": "Th1/Tc1 inflammation",
    "GZMK_inflammatory_memory": "GZMK inflammatory memory",
    "Chronic_stimulation_exhaustion_like": "Chronic stimulation/exhaustion",
    "Tissue_resident_mucosal_retention": "Tissue-resident/mucosal retention",
    "Tph_Tfh_like_B_cell_help": "Tph/Tfh-like B-cell help",
    "Cell_cycle_clonal_proliferation": "Cell-cycle/proliferation",
    "MAIT_like": "MAIT-like",
    "Gut_homing_intestinal_trafficking": "Gut-homing/trafficking",
    "Recent_TCR_stimulation": "Recent TCR stimulation",
    "Naive_central_memory": "Naive/central memory",
}


def bh(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    x = p[ok].to_numpy(float)
    if not len(x):
        return out
    order = np.argsort(x)
    ranked = x[order] * len(x) / np.arange(1, len(x) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(ranked)
    adj[order] = np.clip(ranked, 0, 1)
    out.loc[ok] = adj
    return out


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def collapse_gliph_families() -> pd.DataFrame:
    sig = pd.read_csv(FIG3_ROOT / "Table_F3R3_GLIPH2_significant_cluster_effects.csv")
    seq = pd.read_csv(
        GLIPH_ROOT
        / "IBDTCR_BetaOnly_GLIPH2"
        / "IBDTCR_TRBV_participant_sequence_table.csv"
    )
    rows: list[dict] = []
    for comparison, d in sig.groupby("comparison", sort=False):
        clusters = d["unique_cluster_id"].tolist()
        uf = UnionFind(clusters)
        members = {
            r.unique_cluster_id: set(str(r.member_set).split())
            for r in d.itertuples(index=False)
        }
        seq_to_clusters: dict[str, list[str]] = defaultdict(list)
        for cluster, member_set in members.items():
            for s in member_set:
                seq_to_clusters[s].append(cluster)
        for linked in seq_to_clusters.values():
            for other in linked[1:]:
                uf.union(linked[0], other)
        groups: dict[str, list[str]] = defaultdict(list)
        for cluster in clusters:
            groups[uf.find(cluster)].append(cluster)

        first, second = comparison.split("_vs_")
        eligible = seq[seq["Diagnosis1"].isin([first, second])].copy()
        totals = eligible[["SampleID", "Diagnosis1"]].drop_duplicates().groupby("Diagnosis1").size()
        for family_index, family_clusters in enumerate(groups.values(), start=1):
            family_members = set().union(*(members[c] for c in family_clusters))
            carrier = (
                eligible[eligible["CDR3b"].isin(family_members)][["SampleID", "Diagnosis1"]]
                .drop_duplicates()
                .groupby("Diagnosis1")
                .size()
            )
            a = int(carrier.get(first, 0))
            c = int(carrier.get(second, 0))
            n1 = int(totals.get(first, 0))
            n2 = int(totals.get(second, 0))
            table = [[a, n1 - a], [c, n2 - c]]
            _, p = fisher_exact(table, alternative="two-sided")
            frac1 = a / n1 if n1 else np.nan
            frac2 = c / n2 if n2 else np.nan
            odds = ((a + 0.5) * (n2 - c + 0.5)) / ((n1 - a + 0.5) * (c + 0.5))
            rows.append(
                {
                    "comparison": comparison,
                    "family_id": f"{comparison}_F{family_index:02d}",
                    "n_source_clusters": len(family_clusters),
                    "source_clusters": ";".join(family_clusters),
                    "n_member_sequences": len(family_members),
                    "member_sequences": " ".join(sorted(family_members)),
                    "first_group": first,
                    "second_group": second,
                    "first_carriers": a,
                    "first_total": n1,
                    "second_carriers": c,
                    "second_total": n2,
                    "first_fraction": frac1,
                    "second_fraction": frac2,
                    "carrier_fraction_difference": frac1 - frac2,
                    "odds_ratio_first_vs_second": odds,
                    "p_value": p,
                    "enriched_group": first if frac1 > frac2 else second,
                }
            )
    result = pd.DataFrame(rows)
    result["FDR"] = result.groupby("comparison", group_keys=False)["p_value"].apply(bh)
    result.to_csv(SRC / "Table_PI7_GLIPH2_collapsed_family_participant_enrichment.csv", index=False)
    return result


def gliph_summary(families: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participant = pd.read_csv(GLIPH_ROOT / "Table_PA5_IBDTCR_beta_only_GLIPH2_summary.csv")
    batch = pd.read_csv(
        GLIPH_ROOT / "IBDTCR_BetaOnly_GLIPH2" / "IBDTCR_participants_by_diagnosis_and_batch.csv"
    )
    batch_series = (
        batch.groupby(["Diagnosis1", "BatchSeries"], as_index=False)["n_participants"].sum()
    )
    batch_series.to_csv(SRC / "Table_PI10_acquisition_series_design.csv", index=False)

    legacy = {"CD_vs_Control": 6, "UC_vs_Control": 27, "CD_vs_UC": 8}
    rows = []
    for comparison in ["CD_vs_Control", "UC_vs_Control", "CD_vs_UC"]:
        full = participant[participant["comparison"].eq(comparison)].iloc[0]
        fam = families[(families["comparison"].eq(comparison)) & (families["FDR"] < 0.05)]
        shared = participant[participant["comparison"].eq("CD_vs_UC_shared_S1-S5")]
        rows.append(
            {
                "comparison": comparison,
                "legacy_sequence_level_significant": legacy[comparison],
                "participant_cluster_significant_full": int(full["n_significant_unique_member_sets_fdr05"]),
                "participant_cluster_first_group_enriched": int(full["n_first_group_enriched_fdr05"]),
                "participant_cluster_second_group_enriched": int(full["n_second_group_enriched_fdr05"]),
                "collapsed_family_significant_full": int(len(fam)),
                "collapsed_family_first_group_enriched": int((fam["enriched_group"] == comparison.split("_vs_")[0]).sum()),
                "collapsed_family_second_group_enriched": int((fam["enriched_group"] == comparison.split("_vs_")[1]).sum()),
                "shared_series_available": comparison == "CD_vs_UC",
                "shared_series_significant": int(shared.iloc[0]["n_significant_unique_member_sets_fdr05"]) if comparison == "CD_vs_UC" else np.nan,
                "interpretation": (
                    "No diagnosis-compatible acquisition series; controls are almost entirely S8"
                    if comparison in {"CD_vs_Control", "UC_vs_Control"}
                    else "No significant clusters after rerunning GLIPH2 in shared S1-S5 series"
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(SRC / "Table_PI8_GLIPH2_robustness_summary.csv", index=False)

    paired = pd.read_csv(TSS_ROOT / "Table_TSS1_paired_alpha_beta_clone_sequence_state.csv")
    paired_beta = set(paired["beta_cdr3"].dropna().astype(str))
    overlap_rows = []
    for r in families.itertuples(index=False):
        member_set = set(str(r.member_sequences).split())
        matches = sorted(member_set & paired_beta)
        overlap_rows.append(
            {
                "comparison": r.comparison,
                "family_id": r.family_id,
                "n_member_sequences": r.n_member_sequences,
                "n_exact_beta_matches_in_paired_graph": len(matches),
                "matching_beta_cdr3": " ".join(matches),
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(SRC / "Table_PI9_GLIPH2_family_paired_graph_overlap.csv", index=False)
    return summary, batch_series, overlap


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, length=3, width=0.8)


def build_program_figure():
    tests = pd.read_csv(SRC / "Table_PI3_program_tests_by_diagnosis.csv")
    series = pd.read_csv(SRC / "Table_PI4_acquisition_series_program_effects.csv")
    meta = pd.read_csv(SRC / "Table_PI5_random_effects_program_meta_analysis.csv")
    pooled = tests[tests["stratum"].eq("Pooled IBD")].sort_values("median_delta", ascending=True).copy()
    order = pooled["module"].tolist()

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(15.5, 9.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1, 1], hspace=0.34, wspace=0.48)

    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(pooled))
    colors = np.where(pooled["FDR"] < 0.05, BLUE, GRAY)
    for yi, (_, row) in enumerate(pooled.iterrows()):
        ax.errorbar(
            row["median_delta"], yi,
            xerr=[[row["median_delta"] - row["ci_low"]], [row["ci_high"] - row["median_delta"]]],
            fmt="o", color=colors[yi], markeredgecolor="white", markeredgewidth=0.7,
            markersize=7.2, elinewidth=1.6, capsize=3, zorder=3,
        )
    ax.axvline(0, color="#333333", lw=0.9, ls="--")
    ax.set_yticks(y, [LABELS[m] for m in pooled["module"]])
    ax.set_xlabel("Expanded minus singleton module score\nparticipant median (95% bootstrap CI)")
    ax.set_title("A  Exact paired αβ clone-state programs in pooled IBD (n=108)", loc="left", fontsize=12, weight="bold", pad=12)
    style_axes(ax)
    for yi, (_, r) in enumerate(pooled.iterrows()):
        q = r["FDR"]
        txt = f"q={q:.2g}" if q >= 0.001 else f"q={q:.1e}"
        ax.text(max(pooled["ci_high"].max() + 0.012, 0.19), yi, txt, va="center", fontsize=7, color=colors[yi])
    ax.set_xlim(min(-0.19, pooled["ci_low"].min() - 0.02), max(0.24, pooled["ci_high"].max() + 0.08))

    ax = fig.add_subplot(gs[0, 1])
    dx = tests[tests["stratum"].isin(["CD", "UC"])].copy()
    xpos = {m: i for i, m in enumerate(order)}
    offsets = {"CD": -0.16, "UC": 0.16}
    colors_dx = {"CD": BLUE, "UC": ORANGE}
    for stratum in ["CD", "UC"]:
        d = dx[dx["stratum"].eq(stratum)].copy()
        yy = np.array([xpos[m] for m in d["module"]]) + offsets[stratum]
        ax.errorbar(
            d["median_delta"], yy,
            xerr=[d["median_delta"] - d["ci_low"], d["ci_high"] - d["median_delta"]],
            fmt="o", color=colors_dx[stratum], ms=4.5, lw=1.2, capsize=2, label=stratum,
        )
    ax.axvline(0, color="#333333", lw=0.8, ls="--")
    meta_q = meta.set_index("module")["FDR"]
    heat_labels = [LABELS[m] + (" *" if meta_q.get(m, 1) < 0.05 else "") for m in order]
    ax.set_yticks(np.arange(len(order)), [LABELS[m] for m in order])
    ax.set_xlabel("Participant median difference")
    ax.set_title("B  Direction is conserved across CD and UC", loc="left", fontsize=12, weight="bold")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    ax = fig.add_subplot(gs[1, 1])
    series_order = sorted(series["acquisition_series"].unique(), key=lambda x: int(str(x).replace("S", "")))
    matrix = series.pivot(index="module", columns="acquisition_series", values="mean_delta").reindex(index=order, columns=series_order)
    vmax = np.nanmax(np.abs(matrix.to_numpy()))
    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(series_order)), series_order)
    ax.set_yticks(np.arange(len(order)), heat_labels)
    ax.set_title("C  Acquisition-series replication", loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("Acquisition series")
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean expanded−singleton score", fontsize=8)
    ax.text(1.01, 1.02, "* random-effects q<0.05", transform=ax.transAxes, fontsize=7, color=BLUE, ha="right")

    fig.suptitle(
        "Non-canonical preview: expanded paired TCR clonotypes carry a focused four-program inflammatory-memory signature",
        x=0.02, y=0.985, ha="left", fontsize=14, weight="bold",
    )
    fig.text(
        0.02, 0.01,
        "Exact productive paired αβ clonotypes were defined within participant. Expanded and singleton cells were compared within the same participant, lineage compartment, and annotated state; states and compartments received equal weight.",
        fontsize=7.5, color="#4D4D4D",
    )
    fig.savefig(OUT / "Preview_Figure_S3_extended_paired_TCR_programs.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Preview_Figure_S3_extended_paired_TCR_programs.pdf", bbox_inches="tight")
    plt.close(fig)


def build_gliph_figure(summary: pd.DataFrame, batch: pd.DataFrame, families: pd.DataFrame, overlap: pd.DataFrame):
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(15.5, 9.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1], hspace=0.38, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    order = sorted(batch["BatchSeries"].unique(), key=lambda x: int(str(x).replace("S", "")))
    pivot = batch.pivot(index="BatchSeries", columns="Diagnosis1", values="n_participants").fillna(0).reindex(order)
    bottom = np.zeros(len(pivot))
    for diagnosis, color in [("Control", GRAY), ("CD", BLUE), ("UC", ORANGE)]:
        values = pivot.get(diagnosis, pd.Series(0, index=pivot.index)).to_numpy()
        ax.bar(np.arange(len(pivot)), values, bottom=bottom, color=color, label=diagnosis, width=0.72)
        bottom += values
    ax.set_xticks(np.arange(len(pivot)), pivot.index)
    ax.set_ylabel("Participants")
    ax.set_title("A  Diagnosis is confounded with acquisition series", loc="left", fontsize=12, weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    style_axes(ax)
    ax.text(order.index("S8"), pivot.loc["S8"].sum() / 2, "n=23", ha="center", va="center", color="white", fontsize=8, weight="bold")

    ax = fig.add_subplot(gs[0, 1])
    comp_order = ["CD_vs_Control", "UC_vs_Control", "CD_vs_UC"]
    x = np.arange(3)
    width = 0.24
    legacy = summary.set_index("comparison").loc[comp_order, "legacy_sequence_level_significant"].to_numpy(float)
    participant = summary.set_index("comparison").loc[comp_order, "participant_cluster_significant_full"].to_numpy(float)
    shared = summary.set_index("comparison").loc[comp_order, "shared_series_significant"].to_numpy(float)
    ax.bar(x - width, legacy, width, color=PURPLE, label="Legacy sequence-count")
    ax.bar(x, participant, width, color=GREEN, label="Participant carriers, full cohort")
    ax.bar(x + width, np.nan_to_num(shared, nan=0), width, color=GRAY, label="Shared-series rerun")
    for i, value in enumerate(shared):
        if np.isnan(value):
            ax.text(i + width, 0.4, "not estimable", rotation=90, ha="center", va="bottom", fontsize=7, color=GRAY)
    for i, direction in enumerate(["Control", "Control", "UC"]):
        ax.text(i, participant[i] + 0.5, direction, ha="center", va="bottom", fontsize=7, color=GREEN, weight="bold")
    ax.set_xticks(x, ["CD vs control", "UC vs control", "CD vs UC"])
    ax.set_ylabel("FDR-significant GLIPH2 clusters")
    ax.set_title("B  Legacy motif associations do not survive design-aware review", loc="left", fontsize=12, weight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    style_axes(ax)
    ax.text(0.02, 0.94, "Full-cohort participant hits reverse toward control or UC", transform=ax.transAxes, fontsize=8, color=RED, va="top")

    ax = fig.add_subplot(gs[1, 0])
    fam = families[families["FDR"] < 0.05].copy().sort_values("carrier_fraction_difference")
    y = np.arange(len(fam))
    colors = [GRAY if g == "Control" else ORANGE if g == "UC" else BLUE for g in fam["enriched_group"]]
    ax.scatter(fam["carrier_fraction_difference"], y, c=colors, s=48, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#333333", lw=0.8, ls="--")
    labels = []
    for r in fam.itertuples(index=False):
        tag = str(r.family_id).replace("_vs_", "–").replace("_", " ")
        labels.append(f"{tag} ({r.enriched_group})")
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.set_xlabel("Carrier fraction: first group minus second group")
    ax.set_title("C  Collapsing overlapping motifs yields batch-linked families", loc="left", fontsize=12, weight="bold")
    style_axes(ax)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    exact_matches = int(overlap["n_exact_beta_matches_in_paired_graph"].sum())
    total_members = int(families["n_member_sequences"].sum())
    audit_text = (
        "DESIGN-AWARE CONCLUSION\n\n"
        f"• {exact_matches} exact β-chain matches between significant GLIPH2 families "
        f"and the current paired αβ graph ({total_members} family-member entries audited).\n\n"
        "• Disease–control motif results cannot be separated from acquisition series: "
        "23 of 24 controls are in S8, a series without CD or UC participants.\n\n"
        "• CD-versus-UC GLIPH2 was rerun in shared series S1–S5 and produced 0 FDR-significant clusters.\n\n"
        "INTEGRATION DECISION\n"
        "Do not restore the legacy positive GLIPH2 claim. Retain this audit as internal rigor documentation "
        "or a brief limitation; prioritize the current cross-participant paired-chain graph."
    )
    ax.text(0.02, 0.96, audit_text, va="top", ha="left", fontsize=10, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F7F9", edgecolor="#C8D1DA"))
    ax.set_title("D  Orthogonal-validation decision", loc="left", fontsize=12, weight="bold")

    fig.suptitle(
        "Non-canonical preview: participant- and acquisition-aware GLIPH2 audit",
        x=0.02, y=0.985, ha="left", fontsize=14, weight="bold",
    )
    fig.text(
        0.02, 0.01,
        "Participant-carrier enrichment was corrected within comparison after collapsing overlapping significant member sets into connected motif families. Shared-series results use an independent GLIPH2 rerun restricted to CD and UC participants in S1–S5.",
        fontsize=7.5, color="#4D4D4D",
    )
    fig.savefig(OUT / "Preview_Figure_S4_GLIPH2_design_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Preview_Figure_S4_GLIPH2_design_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, families: pd.DataFrame, overlap: pd.DataFrame):
    tests = pd.read_csv(SRC / "Table_PI3_program_tests_by_diagnosis.csv")
    pooled = tests[tests["stratum"].eq("Pooled IBD")].set_index("module")
    meta = pd.read_csv(SRC / "Table_PI5_random_effects_program_meta_analysis.csv").set_index("module")

    def stat(module: str) -> str:
        r = pooled.loc[module]
        return f"median Δ={r.median_delta:.3f}, 95% CI {r.ci_low:.3f} to {r.ci_high:.3f}, FDR={r.FDR:.2g}"

    report = f"""# High-impact prior-findings integration preview

Status: non-canonical analysis preview. No main figure or manuscript file was replaced.

## Evidence-based integration decision

The broadened exact paired-alpha/beta analysis supports adding **GZMK inflammatory-memory activity** to the current three-program TCR expansion axis. It also reveals reproducible depletion of naive/central-memory, cell-cycle/proliferation, MAIT-like, and gut-homing/trafficking programs among expanded clonotypes. The former legacy claims of chronic-stimulation, tissue-retention, and Tph/Tfh-like enrichment are not supported after exact paired-chain, state-matched participant-level analysis.

The positive legacy GLIPH2 diagnosis claim should **not** be restored. Participant-carrier reanalysis reversed the disease-control signal toward controls, disease and control groups are confounded with acquisition series, CD-versus-UC clusters disappear in a shared-series rerun, and the significant motif families have no exact beta-chain overlap with the current paired-alpha/beta sequence graph.

## Definition QA note before any integration

The current Figure 2 clone-size program source-generation workflow defines TCR clonotypes using productive beta-chain V gene, J gene, and CDR3 amino-acid identity, whereas the current Figure 2 legend describes exact paired-alpha/beta clonotypes. No canonical file was changed here. The stricter analysis in this preview required productive alpha and beta chains and independently reproduced the three primary programs in 108 state-matched IBD participants. Before integration, the canonical Figure 2 source definition and legend should be reconciled: either regenerate its dose-response panel with the strict paired-chain definition or relabel the existing analysis as beta-chain-defined.

## Key exact paired-alpha/beta findings

- Effector cytotoxicity: {stat('Effector_cytotoxicity')}.
- EOMES-ZEB2 inflammatory CD8/TRM-like activity: {stat('EOMES_ZEB2_inflammatory_CD8_TRM_like')}.
- Th1/Tc1 inflammation: {stat('Th1_Tc1_inflammatory')}.
- GZMK inflammatory memory: {stat('GZMK_inflammatory_memory')}.
- Naive/central-memory activity: {stat('Naive_central_memory')}.
- Cell-cycle/proliferation: {stat('Cell_cycle_clonal_proliferation')}.
- Gut-homing/trafficking: {stat('Gut_homing_intestinal_trafficking')}.

All four positively enriched programs had positive mean effects in all seven evaluable acquisition series. Random-effects meta-analysis retained EOMES-ZEB2, cytotoxicity, Th1/Tc1, and GZMK effects at FDR < 0.005.

## Proposed Results revision

### Expanded paired TCR clonotypes carry a focused inflammatory-memory program

To determine whether the broader transcriptional pattern reported in the earlier analysis persisted under the current paired-chain framework, we compared expanded and singleton exact paired alpha-beta clonotypes within the same participant, lineage compartment, and annotated T-cell state. Across 108 participants with CD or UC contributing both expanded and singleton cells, expansion was associated with increased effector-cytotoxic, EOMES-ZEB2 inflammatory CD8/TRM-like, Th1/Tc1, and GZMK inflammatory-memory activity. The GZMK effect was smaller than the three primary programs but remained directionally positive in all seven evaluable acquisition series and significant in random-effects meta-analysis. Expanded clonotypes showed reciprocal depletion of naive/central-memory, cell-cycle/proliferation, MAIT-like, and gut-homing/trafficking activity. Chronic-stimulation/exhaustion-like, tissue-retention, and Tph/Tfh-like helper programs were not significantly increased. Thus, the expanded peripheral repertoire carries a focused cytotoxic and inflammatory-memory signature rather than generalized activation across all antigen-experience programs.

## Proposed GLIPH2 audit text

We re-evaluated the earlier beta-chain GLIPH2 findings using participants, rather than unique sequences, as the inferential unit. Full-cohort participant-carrier tests identified control-enriched or UC-enriched clusters rather than the previously reported disease-control pattern. These results were inseparable from acquisition series because 23 of 24 controls were processed in S8, which contained no CD or UC participants. When GLIPH2 was rerun for CD versus UC using only shared acquisition series S1-S5, no cluster passed FDR correction. Significant full-cohort motif families also showed no exact beta-chain overlap with the paired-alpha/beta sequence graph used in the current analysis. The legacy GLIPH2 disease-association claim was therefore not integrated.

## Proposed Discussion revision

The expanded-clone analysis refines the functional interpretation of peripheral TCR selection. Exact paired clonotypes were consistently enriched for cytotoxic, EOMES-ZEB2, Th1/Tc1, and GZMK inflammatory-memory programs while losing naive/central-memory and selected trafficking or proliferative programs. This pattern supports focused differentiation into inflammatory effector and memory states, but does not support uniform enrichment of chronic-stimulation, tissue-retention, or helper programs. An orthogonal beta-chain motif analysis did not withstand acquisition-series-aware evaluation, emphasizing the importance of participant-level inference and motivating external validation of the paired-chain sequence neighborhoods in cohorts where diagnosis is balanced across processing series.

## Figure integration strategy

1. Keep canonical Figure 2 unchanged for now.
2. If approved, add `Preview_Figure_S3_extended_paired_TCR_programs` to the existing Figure S3 rather than the main figure.
3. Add one sentence on GZMK inflammatory memory to the Figure 2 Results section; retain the three primary programs in the main panel.
4. Do not add GLIPH2 to Figure 3. The design audit can remain internal or be summarized in the limitations; if transparency warrants a figure, place it in a methods/QC supplement, not the mechanistic main story.
5. Keep the current paired-chain graph as the sequence-convergence result because it uses cross-participant edges, acquisition-series replication, and participant-label permutations.

## Preview figure legends

**Preview Figure S3. Extended exact paired-alpha/beta clone-state program analysis.** (A) Participant-level median expanded-minus-singleton differences for 12 prespecified T-cell programs in pooled IBD. Comparisons were state- and compartment-matched within participant. Error bars show 95% participant-bootstrap confidence intervals; blue symbols pass FDR correction across 12 programs. (B) Diagnosis-stratified median differences in CD and UC. (C) Acquisition-series-specific mean differences; asterisks denote programs passing random-effects meta-analysis FDR < 0.05.

**Preview Figure S4. Participant- and acquisition-aware GLIPH2 design audit.** (A) Diagnosis composition across acquisition series. (B) Numbers of FDR-significant clusters under the legacy sequence-count analysis, participant-carrier analysis in the full cohort, and shared-series rerun. Disease-control shared-series estimates were not possible because controls were almost entirely confined to S8. (C) Participant carrier-fraction effects after overlapping significant clusters were collapsed into connected motif families. (D) Audit conclusion and integration decision.
"""
    (OUT / "High_Impact_Integration_Report.md").write_text(report, encoding="utf-8")


def main():
    families = collapse_gliph_families()
    summary, batch, overlap = gliph_summary(families)
    build_program_figure()
    build_gliph_figure(summary, batch, families, overlap)
    write_report(summary, families, overlap)
    print(f"Wrote non-canonical preview outputs to {OUT}")


if __name__ == "__main__":
    main()

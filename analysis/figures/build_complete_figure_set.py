from pathlib import Path
from shutil import copy2

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Final 7-Figure Manuscript Set"
MAIN = OUT / "Main Figures"
SUPP = OUT / "Supplementary Figures"
LEGENDS = OUT / "Legends"
SOURCE = OUT / "Source Data"
for directory in [MAIN, SUPP, LEGENDS, SOURCE]:
    directory.mkdir(parents=True, exist_ok=True)

COL = {
    "control": "#767676",
    "cd": "#6B4FB3",
    "uc": "#159C9C",
    "tcr": "#2F6FB0",
    "bcr": "#D47A32",
    "joint": "#7A4EAB",
    "ink": "#202020",
    "line": "#D7D7D7",
    "soft": "#F4F5F7",
}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load(path):
    return Image.open(ROOT / path).convert("RGB")


def crop(path, box):
    return load(path).crop(box)


def trim_white(image, pad=18, threshold=246):
    """Remove inherited white margins from reusable composite panels."""
    gray = image.convert("L")
    mask = gray.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    return image.crop((left, top, right, bottom))


def show_img(ax, image, title=None, label=None):
    ax.imshow(trim_white(image))
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
    if label:
        ax.text(-0.035, 1.03, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")


def save(fig, stem, folder, title):
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.savefig(folder / f"{stem}.png", dpi=320, facecolor="white", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", dpi=320, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def title_only_composite(stem, title, images, folder, heights=None, labels=None, figsize=(15, 17)):
    if heights is None:
        heights = [1] * len(images)
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = GridSpec(len(images), 1, figure=fig, height_ratios=heights, hspace=0.12, top=0.955, bottom=0.03, left=0.03, right=0.985)
    for i, image in enumerate(images):
        ax = fig.add_subplot(gs[i, 0])
        show_img(ax, image, label=(labels[i] if labels else None))
    save(fig, stem, folder, title)


def pretty_module(value):
    return value.replace("_", " ").replace("NFkB", "NF-kB")


module_enrichment = pd.read_csv(ROOT / "High Impact Additional Analyses" / "Priority Analyses" / "Table_PA1_clone_aware_pseudobulk_module_enrichment.csv")
clone_deltas = pd.read_csv(ROOT / "High Impact Additional Analyses" / "Clone State Interactions" / "Table_CSI2_participant_expanded_minus_singleton_deltas.csv")


def plot_module_lollipop(ax, modality, title):
    data = module_enrichment[module_enrichment["modality"] == modality].copy()
    data["signed"] = pd.to_numeric(data["signed_log10_FDR"], errors="coerce")
    data = data.loc[data["signed"].abs().sort_values(ascending=False).index].head(11).sort_values("signed")
    y = np.arange(len(data))
    colors = np.where(data["signed"] >= 0, "#C83E5A", "#2F73B8")
    ax.hlines(y, 0, data["signed"], color=colors, linewidth=1.6)
    ax.scatter(data["signed"], y, color=colors, s=44, zorder=3)
    ax.axvline(0, color="#888888", linewidth=0.9)
    ax.set_yticks(y, [pretty_module(x) for x in data["module"]])
    ax.set_xlabel("Signed -log10 FDR (expanded vs singleton)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=COL["line"], linewidth=0.55)


def plot_diagnosis_heatmap(ax, modality, modules, title):
    data = clone_deltas[(clone_deltas["modality"] == modality) & (clone_deltas["module"].isin(modules))].copy()
    table = data.pivot_table(index="module", columns="Diagnosis1", values="delta", aggfunc="median").reindex(index=modules, columns=["Control", "CD", "UC"])
    lim = np.nanmax(np.abs(table.to_numpy()))
    lim = max(lim, 0.05)
    image = ax.imshow(table, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(3), ["Control", "CD", "UC"], rotation=30, ha="right")
    ax.set_yticks(range(len(modules)), [pretty_module(x) for x in modules])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.02f}", ha="center", va="center", fontsize=7.2, color="white" if abs(value) > lim * 0.55 else COL["ink"])
    ax.set_title(title, loc="left", fontweight="bold")
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Median expanded - singleton score")


f1 = "main_figures_rendered/Figure_1.png"
f3 = "main_figures_rendered/Figure_3.png"
f5 = "main_figures_rendered/Figure_5.png"
csi = "High Impact Additional Analyses/Requested Priority Analyses Figures/Figure_S_clone_state_interactions.png"
tss = "High Impact Additional Analyses/Requested Priority Analyses Figures/Figure_S_paired_TCR_sequence_state.png"
bgl = "High Impact Additional Analyses/Requested Priority Analyses Figures/Figure_S_BCR_germline_lineages.png"
hi1 = "High Impact Additional Analyses/Figure_HI1_TCR_BCR_coordination.png"
hi1b = "High Impact Additional Analyses/Figure_HI1b_CD_specific_T_B_module_coupling.png"
hi2 = "High Impact Additional Analyses/Figure_HI2_clonotype_state_breadth.png"
pa1b = "High Impact Additional Analyses/Priority Analyses/Figure_PA1b_clone_aware_module_enrichment.png"
gliph = "High Impact Additional Analyses/Priority Analyses/Figure_PA5_IBDTCR_beta_only_GLIPH2_enrichment.png"
gliph_batch = "High Impact Additional Analyses/Priority Analyses/Figure_PA5b_IBDTCR_beta_only_GLIPH2_shared_batch_sensitivity.png"
ml_nested = "ML Sensitivity Validation/Figure_S_ML_nested_and_permutation.png"
ml_cal = "ML Sensitivity Validation/Figure_S_ML_calibration_and_optimism.png"


# Main Figure 1
fig = plt.figure(figsize=(15, 14), facecolor="white")
gs = GridSpec(2, 3, figure=fig, height_ratios=[0.82, 1.18], wspace=0.28, hspace=0.24, top=0.955, bottom=0.055, left=0.055, right=0.98)

ax = fig.add_subplot(gs[0, 0])
ax.set_title("A  Cohort", loc="left", fontweight="bold")
groups = ["Control", "UC", "CD"]
vals = [35, 87, 127]
colors = [COL["control"], COL["uc"], COL["cd"]]
bars = ax.bar(groups, vals, color=colors, width=0.68)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha="center", fontweight="bold")
ax.set_ylabel("Participants")
ax.set_ylim(0, 145)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=COL["line"], linewidth=0.6)

ax = fig.add_subplot(gs[0, 1])
ax.set_axis_off()
ax.set_title("B  Single-cell and receptor workflow", loc="left", fontweight="bold")
workflow = [
    (0.02, "Peripheral blood\nPBMC isolation"),
    (0.28, "BD Rhapsody\nscRNA-seq"),
    (0.54, "Paired TCR/BCR\nsequencing"),
    (0.80, "Cell state +\nclonotype"),
]
for x, text in workflow:
    box = FancyBboxPatch((x, 0.40), 0.18, 0.24, boxstyle="round,pad=0.018", facecolor="#EEF4F8", edgecolor="#4B6F8F", transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(x + 0.09, 0.52, text, ha="center", va="center", transform=ax.transAxes)
for x in [0.20, 0.46, 0.72]:
    ax.add_patch(FancyArrowPatch((x, 0.52), (x + 0.07, 0.52), arrowstyle="-|>", mutation_scale=12, color="#4B6F8F", transform=ax.transAxes))
ax.text(0.5, 0.20, "Participant is the biological unit for all group comparisons", ha="center", transform=ax.transAxes, color="#5C5C5C")

ax = fig.add_subplot(gs[0, 2])
ax.set_axis_off()
ax.set_title("C  Analysis architecture", loc="left", fontweight="bold")
nodes = [
    (0.05, 0.64, "TCR clonal\nstates", COL["tcr"]),
    (0.55, 0.64, "BCR clonal\nlineages", COL["bcr"]),
    (0.05, 0.24, "T-B\ncoordination", COL["joint"]),
    (0.55, 0.24, "Nested clinical\nclassification", "#248B73"),
]
for x, y, text, color in nodes:
    box = FancyBboxPatch((x, y), 0.38, 0.22, boxstyle="round,pad=0.015", facecolor="white", edgecolor=color, linewidth=1.6, transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(x + 0.19, y + 0.11, text, ha="center", va="center", transform=ax.transAxes, color=COL["ink"])

t_atlas = crop(f1, (45, 1060, 875, 1405))
b_atlas = crop(f1, (905, 1060, 1485, 1405))
ax = fig.add_subplot(gs[1, :2])
show_img(ax, t_atlas, "D  Focused CD4 and CD8 T-cell states")
ax = fig.add_subplot(gs[1, 2])
show_img(ax, b_atlas, "E  Focused B-cell states")
save(fig, "Figure_1", MAIN, "Single-cell profiling resolves circulating T- and B-cell states and paired immune receptors in IBD")


# Main Figure 2
fig = plt.figure(figsize=(15, 17), facecolor="white")
gs = GridSpec(3, 3, figure=fig, height_ratios=[0.70, 1.0, 1.28], wspace=0.18, hspace=0.18, top=0.955, bottom=0.035, left=0.035, right=0.985)
show_img(fig.add_subplot(gs[0, 0]), crop(f3, (30, 105, 500, 360)), "A  Repertoire structure")
show_img(fig.add_subplot(gs[0, 1]), crop(f3, (535, 105, 1055, 365)), "B  Clone-size composition")
show_img(fig.add_subplot(gs[0, 2]), crop(f3, (1085, 100, 1485, 345)), "C  State occupancy")
plot_module_lollipop(fig.add_subplot(gs[1, :]), "TCR", "D  Clone-aware T-cell transcriptional programs")
plot_diagnosis_heatmap(
    fig.add_subplot(gs[2, 0]),
    "TCR",
    [
        "Naive_central_memory",
        "Effector_cytotoxicity",
        "Th1_Tc1_inflammatory",
        "EOMES_ZEB2_inflammatory_CD8_TRM_like",
        "Tissue_resident_mucosal_retention",
        "Gut_homing_intestinal_trafficking",
    ],
    "E  Diagnosis-stratified clonal programs",
)
show_img(fig.add_subplot(gs[2, 1:]), load(hi2), "F  Clone-size-adjusted state breadth")
save(fig, "Figure_2", MAIN, "TCR clonal expansion is coupled to inflammatory and cytotoxic T-cell states in IBD")


# Main Figure 3
fig = plt.figure(figsize=(15, 14.5), facecolor="white")
gs = GridSpec(2, 2, figure=fig, height_ratios=[0.93, 1.07], width_ratios=[1.35, 0.65], wspace=0.12, hspace=0.16, top=0.955, bottom=0.035, left=0.035, right=0.985)
show_img(fig.add_subplot(gs[0, :]), load(tss), "A-C  Paired alpha-beta sequence-state associations")
show_img(fig.add_subplot(gs[1, 0]), load(gliph), "D  Participant-carrier GLIPH2 enrichment")
show_img(fig.add_subplot(gs[1, 1]), load(gliph_batch), "E  Shared-series sensitivity analysis")
save(fig, "Figure_3", MAIN, "Sequence-similar TCRs converge on shared inflammatory T-cell programs")


# Main Figure 4
fig = plt.figure(figsize=(15, 17), facecolor="white")
gs = GridSpec(3, 3, figure=fig, height_ratios=[0.72, 1.03, 1.12], wspace=0.18, hspace=0.18, top=0.955, bottom=0.035, left=0.035, right=0.985)
show_img(fig.add_subplot(gs[0, 0]), crop(f5, (30, 115, 500, 365)), "A  Repertoire structure")
show_img(fig.add_subplot(gs[0, 1]), crop(f5, (525, 115, 1065, 370)), "B  Clone-size composition")
show_img(fig.add_subplot(gs[0, 2]), crop(f5, (1080, 145, 1485, 350)), "C  B-cell state occupancy")
plot_module_lollipop(fig.add_subplot(gs[1, :]), "BCR", "D  Clone-aware B-cell transcriptional programs")
plot_diagnosis_heatmap(
    fig.add_subplot(gs[2, 0]),
    "BCR",
    [
        "Resting_naive_B_cell",
        "Resting_memory_B_cell",
        "Plasmablast_plasma_differentiation",
        "IgA_mucosal_plasma_cell",
        "IgG_inflammatory_plasma_cell",
        "B_cell_antigen_presentation",
        "BAFF_APRIL_survival_response",
    ],
    "E  Diagnosis-stratified clonal programs",
)
show_img(fig.add_subplot(gs[2, 1:]), crop(f5, (25, 1380, 1480, 1635)), "F  Class switching and isotype composition")
save(fig, "Figure_4", MAIN, "BCR expansion is concentrated in class-switched memory and antibody-secreting states")


# Main Figure 5
title_only_composite(
    "Figure_5",
    "IBD reshapes somatic hypermutation and the evolution of class-switched BCR lineages",
    [load(bgl)],
    MAIN,
    figsize=(15, 12.5),
)


# Main Figure 6
title_only_composite(
    "Figure_6",
    "Coordinated T- and B-cell clonal programs define a systemic adaptive immune axis in IBD",
    [load(hi1), load(hi1b)],
    MAIN,
    heights=[1.0, 0.95],
    figsize=(15, 18),
)


# Main Figure 7: validated classifier figure generated previously
copy2(ROOT / "Recreated Figure 6" / "Figure_6_recreated_immuneML.png", MAIN / "Figure_7.png")
copy2(ROOT / "Recreated Figure 6" / "Figure_6_recreated_immuneML.pdf", MAIN / "Figure_7.pdf")


# Supplementary figures
title_only_composite(
    "Figure_S1",
    "Full PBMC and adaptive-cell atlas with differential-abundance summaries",
    [crop(f1, (0, 580, 1489, 2105))],
    SUPP,
    figsize=(15, 16),
)
title_only_composite("Figure_S2", "Complete TCR repertoire analyses", [load(f3)], SUPP, figsize=(15, 19))
title_only_composite(
    "Figure_S3",
    "Clone-state interaction models and clonotype-state breadth sensitivity analyses",
    [load(csi), load(hi2)],
    SUPP,
    heights=[1.0, 1.05],
    figsize=(15, 19),
)
title_only_composite(
    "Figure_S4",
    "Paired-TCR sequence-state replication and GLIPH2 robustness analyses",
    [load(tss), load(gliph), load(gliph_batch)],
    SUPP,
    heights=[0.85, 0.75, 0.85],
    figsize=(15, 18),
)
title_only_composite("Figure_S5", "Complete BCR repertoire, isotype, SHM, and ImmunoMatch analyses", [load(f5)], SUPP, figsize=(15, 19))
title_only_composite("Figure_S6", "BCR germline-lineage and lineage-definition sensitivity analyses", [load(bgl)], SUPP, figsize=(15, 13))
title_only_composite(
    "Figure_S7",
    "Complete cross-compartment TCR-BCR coordination analyses",
    [load(hi1), load(hi1b)],
    SUPP,
    heights=[1.0, 0.95],
    figsize=(15, 18),
)
title_only_composite(
    "Figure_S8",
    "Nested machine-learning validation, permutation, calibration, and optimism analyses",
    [load(ml_nested), load(ml_cal)],
    SUPP,
    heights=[0.85, 1.15],
    figsize=(15, 14),
)


main_legends = {
    "Figure_1": "Figure 1. Single-cell profiling resolves circulating T- and B-cell states and paired immune receptors in IBD. (A) Diagnostic composition of the 249-participant cohort. (B) BD Rhapsody single-cell transcriptomic and paired immune-receptor workflow. (C) Analysis architecture. (D) Focused CD4 and CD8 T-cell state atlas. (E) Focused B-cell state atlas.",
    "Figure_2": "Figure 2. TCR clonal expansion is coupled to inflammatory and cytotoxic T-cell states in IBD. Participant-level repertoire structure, clone-size composition, T-cell-state occupancy, clone-aware transcriptional programs, diagnosis-stratified clonal programs, and clone-size-adjusted state breadth are shown.",
    "Figure_3": "Figure 3. Sequence-similar TCRs converge on shared inflammatory T-cell programs. Paired alpha-beta sequence-state associations were tested within participant and acquisition series. GLIPH2 enrichment used participant-carrier frequencies. The shared-series CD-versus-UC sensitivity analysis is shown to distinguish reproducible from batch-sensitive convergence.",
    "Figure_4": "Figure 4. BCR expansion is concentrated in class-switched memory and antibody-secreting states. Participant-level repertoire structure, clone-size composition, state occupancy, clone-aware B-cell programs, diagnosis-stratified clonal programs, class switching, and isotype composition are shown.",
    "Figure_5": "Figure 5. IBD reshapes somatic hypermutation and the evolution of class-switched BCR lineages. Heavy-chain SHM is summarized by framework and complementarity-determining regions, together with participant-level lineage branch length, lineage-definition sensitivity, and cross-state lineage occupancy.",
    "Figure_6": "Figure 6. Coordinated T- and B-cell clonal programs define a systemic adaptive immune axis in IBD. Partial Spearman correlations were adjusted for diagnosis, age, sex, and receptor depth. Diagnosis-stratified analyses show selective coupling between expanded-TCR cytotoxicity and IgA-mucosal or plasma-cell-differentiation programs in Crohn's disease.",
    "Figure_7": (ROOT / "Recreated Figure 6" / "Figure_6_legend.txt").read_text(encoding="utf-8").replace("Figure 6.", "Figure 7.", 1),
}
supp_legends = {
    "Figure_S1": "Figure S1. Full PBMC and adaptive-cell atlas with differential-abundance summaries.",
    "Figure_S2": "Figure S2. Complete TCR repertoire analyses, including diversity, clone size, V-J usage, expanded-clone states, dominant clonotypes, GLIPH2 clustering, and putative antigen mapping.",
    "Figure_S3": "Figure S3. Clone-state interaction models and clonotype-state breadth sensitivity analyses.",
    "Figure_S4": "Figure S4. Paired-TCR sequence-state replication and GLIPH2 robustness analyses, including the shared-acquisition-series sensitivity analysis.",
    "Figure_S5": "Figure S5. Complete BCR repertoire, isotype, SHM, and ImmunoMatch analyses.",
    "Figure_S6": "Figure S6. BCR germline-lineage and lineage-definition sensitivity analyses.",
    "Figure_S7": "Figure S7. Complete cross-compartment TCR-BCR coordination analyses.",
    "Figure_S8": "Figure S8. Nested machine-learning validation, permutation-derived null performance, calibration, and model-selection optimism analyses.",
}
for name, text in {**main_legends, **supp_legends}.items():
    (LEGENDS / f"{name}_legend.txt").write_text(text.strip() + "\n", encoding="utf-8")
(LEGENDS / "All_figure_legends.txt").write_text("\n\n".join([main_legends[k] for k in main_legends] + [supp_legends[k] for k in supp_legends]) + "\n", encoding="utf-8")


manifest_rows = [
    ("Figure 1", f1, "Cohort and focused T/B cell atlas crops; cohort counts from manuscript"),
    ("Figure 2", f3, "TCR repertoire structure and clone-state panels"),
    ("Figure 2", pa1b, "Clone-aware module enrichment"),
    ("Figure 2", csi, "Diagnosis-stratified clone-state programs"),
    ("Figure 2", hi2, "Clonotype-state breadth"),
    ("Figure 3", tss, "Paired TCR sequence-state analysis"),
    ("Figure 3", gliph, "IBDTCR beta-chain GLIPH2 participant-level enrichment"),
    ("Figure 3", gliph_batch, "Shared-series GLIPH2 sensitivity"),
    ("Figure 4", f5, "BCR repertoire and isotype panels"),
    ("Figure 4", pa1b, "Clone-aware BCR module enrichment"),
    ("Figure 4", csi, "Diagnosis-stratified BCR clonal programs"),
    ("Figure 5", bgl, "Germline-aware BCR lineage analysis"),
    ("Figure 6", hi1, "Cross-compartment partial correlations"),
    ("Figure 6", hi1b, "CD-specific T-B module coupling"),
    ("Figure 7", "Recreated Figure 6/Figure_6_recreated_immuneML.pdf", "Nested immuneML validation and exploratory clinical-state screens"),
]
pd.DataFrame(manifest_rows, columns=["figure", "source_path", "content"]).to_csv(SOURCE / "figure_source_manifest.csv", index=False)


def merge_pdfs(paths, output):
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)
    with open(output, "wb") as stream:
        writer.write(stream)


merge_pdfs([MAIN / f"Figure_{i}.pdf" for i in range(1, 8)], OUT / "Main_Figures_1-7.pdf")
merge_pdfs([SUPP / f"Figure_S{i}.pdf" for i in range(1, 9)], OUT / "Supplementary_Figures_S1-S8.pdf")

print(OUT)

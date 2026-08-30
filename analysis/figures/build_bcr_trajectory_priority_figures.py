from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
from scipy.stats import norm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(20260828)

ROOT = Path(r"C:/path/to/private-manuscript-workspace")
OUT = ROOT / "High Impact Additional Analyses" / "BCR Trajectory Priority"
BAL = OUT / "Table_BT1_balanced_cell_pseudotime.csv.gz"
FULL = OUT / "Table_BT2_full_paired_clone_state_metadata.csv.gz"
LIN = OUT / "Table_BT1_slingshot_lineages.csv"

COL = {
    "teal": "#007C83", "gold": "#E3A018", "blue": "#3C6EAA",
    "magenta": "#B24C7C", "green": "#41976B", "orange": "#D66B2C",
    "grey": "#68727D", "light": "#D8DEE4", "ink": "#17232D"
}

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 400
})
sns.set_style("ticks")


def panel(ax, letter):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", ha="left", va="top", color=COL["ink"])


def bootstrap_summary(frame, value, group, participant="PatientID", n_boot=1000):
    wide = frame.pivot_table(index=participant, columns=group, values=value,
                             aggfunc="mean", observed=True)
    values = wide.to_numpy(float)
    point = np.nanmean(values, axis=0)
    sampled = RNG.integers(0, len(wide), size=(n_boot, len(wide)))
    boots = np.nanmean(values[sampled], axis=1)
    return pd.DataFrame({group: wide.columns, "mean": point,
                         "lo": np.nanquantile(boots, .025, axis=0),
                         "hi": np.nanquantile(boots, .975, axis=0)})


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


d = pd.read_csv(BAL, low_memory=False)
lineages = pd.read_csv(LIN)
d["PatientID"] = d["PatientID"].astype(str)
d["pt_bin"] = pd.cut(d.trajectory_pseudotime, [-.001, .2, .4, .6, .8, 1.001],
                     labels=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"])

# Participant-level clone-size dose response.
clone_order = ["Singleton", "2-5", "6-20", ">20"]
paired = d[d.paired_bcr.eq(True) & d.clone_bin.notna()].copy()
paired["clone_bin"] = pd.Categorical(paired.clone_bin, clone_order, ordered=True)
pt_part = paired.groupby(["PatientID", "clone_bin"], observed=True).trajectory_pseudotime.median().reset_index()
pt_part["clone_rank"] = pt_part.clone_bin.cat.codes
fit = smf.ols("trajectory_pseudotime ~ clone_rank + C(PatientID)", data=pt_part).fit(cov_type="HC1")
slope = fit.params["clone_rank"]
p_slope = fit.pvalues["clone_rank"]
pt_sum = bootstrap_summary(pt_part, "trajectory_pseudotime", "clone_bin")

# Participant-balanced program and isotype summaries across pseudotime bins.
programs = {
    "IgA mucosal": "IgA_mucosal_score",
    "Plasma differentiation": "plasma_differentiation_score",
    "Antibody secretion/UPR": "antibody_secretion_UPR_score"
}
prog_part = d.groupby(["PatientID", "pt_bin"], observed=True)[list(programs.values())].mean().reset_index()
prog_long = prog_part.melt(["PatientID", "pt_bin"], var_name="program", value_name="score")
prog_long["program"] = prog_long.program.map({v: k for k, v in programs.items()})

iso = d[d.paired_bcr.eq(True) & d.isotype.isin(["IgM", "IgD", "IgA", "IgG"])].copy()
iso_counts = iso.groupby(["PatientID", "pt_bin", "isotype"], observed=True).size().rename("n").reset_index()
tot = iso.groupby(["PatientID", "pt_bin"], observed=True).size().rename("total").reset_index()
iso_grid = (tot.assign(_key=1)
            .merge(pd.DataFrame({"isotype": ["IgM", "IgD", "IgA", "IgG"], "_key": 1}), on="_key")
            .drop(columns="_key"))
iso_part = iso_grid.merge(iso_counts, on=["PatientID", "pt_bin", "isotype"], how="left")
iso_part["n"] = iso_part.n.fillna(0)
iso_part["fraction"] = iso_part.n / iso_part.total

shm = d[d.paired_bcr.eq(True) & d.total_shm.notna() & d.isotype.isin(["IgM", "IgA", "IgG"])].copy()
shm_part = shm.groupby(["PatientID", "pt_bin", "isotype"], observed=True).total_shm.median().reset_index()

# Source tables.
pt_part.to_csv(OUT / "Table_BT3_participant_clone_size_pseudotime.csv", index=False)
prog_long.to_csv(OUT / "Table_BT4_participant_program_trends.csv", index=False)
iso_part.to_csv(OUT / "Table_BT5_participant_isotype_trends.csv", index=False)
shm_part.to_csv(OUT / "Table_BT6_participant_SHM_trends.csv", index=False)
pd.DataFrame({"term": ["ordered clone-size slope"], "estimate": [slope],
              "robust_se": [fit.bse["clone_rank"]], "p_value": [p_slope],
              "n_participant_bin_summaries": [len(pt_part)],
              "n_participants": [pt_part.PatientID.nunique()]}).to_csv(
                  OUT / "Table_BT7_primary_statistics.csv", index=False)

# Figure BT1.
fig = plt.figure(figsize=(13.2, 8.4), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1])
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[1, 2])

show = d.sample(min(22000, len(d)), random_state=20260828).sort_values("trajectory_pseudotime")
sc = axA.scatter(show.UMAP_1, show.UMAP_2, c=show.trajectory_pseudotime,
                 cmap="viridis", s=2.2, alpha=.62, linewidth=0, rasterized=True)
cent = d.groupby("state")[["UMAP_1", "UMAP_2"]].median()
for path in lineages.cluster_order:
    states = [x.strip() for x in path.split("->")]
    xy = cent.loc[states].values
    axA.plot(xy[:, 0], xy[:, 1], color="white", lw=4.2, alpha=.85, zorder=4)
    axA.plot(xy[:, 0], xy[:, 1], color=COL["ink"], lw=1.15, alpha=.75, zorder=5)
for state, row in cent.iterrows():
    axA.scatter(row.UMAP_1, row.UMAP_2, s=25, facecolor="white", edgecolor=COL["ink"], lw=.7, zorder=6)
axA.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Participant-balanced branching B-cell trajectory")
axA.set_xticks([]); axA.set_yticks([])
cb = fig.colorbar(sc, ax=axA, location="bottom", shrink=.68, pad=.035, aspect=35)
cb.set_label("Normalized pseudotime")
panel(axA, "A")

x = np.arange(len(clone_order))
axB.plot(x, pt_sum["mean"], color=COL["teal"], lw=1.8, marker="o", ms=4.5)
axB.fill_between(x, pt_sum.lo, pt_sum.hi, color=COL["teal"], alpha=.18, linewidth=0)
for i, b in enumerate(clone_order):
    vals = pt_part.loc[pt_part.clone_bin == b, "trajectory_pseudotime"].values
    xx = RNG.normal(i, .045, len(vals))
    axB.scatter(xx, vals, s=7, color=COL["grey"], alpha=.22, linewidth=0, rasterized=True)
axB.set_xticks(x, clone_order)
axB.set(xlabel="Exact paired-clone size (cells)", ylabel="Participant median pseudotime",
        title="Clone-size dose response is weak and sparse", ylim=(.30, .90))
n_by_bin = pt_part.groupby("clone_bin", observed=True).PatientID.nunique().reindex(clone_order)
axB.text(.03, .96, "participants per bin: " + " | ".join(str(int(v)) for v in n_by_bin),
         transform=axB.transAxes, va="top", color=COL["ink"])
panel(axB, "B")

iso_colors = {"IgM": COL["blue"], "IgD": COL["grey"], "IgA": COL["gold"], "IgG": COL["magenta"]}
for name in ["IgM", "IgD", "IgA", "IgG"]:
    z = iso_part[iso_part.isotype == name]
    s = bootstrap_summary(z, "fraction", "pt_bin")
    xx = np.arange(len(s))
    axC.plot(xx, s["mean"], marker="o", ms=3.5, lw=1.6, color=iso_colors[name], label=name)
    axC.fill_between(xx, s.lo, s.hi, color=iso_colors[name], alpha=.12, linewidth=0)
axC.set_xticks(range(5), ["0", ".2", ".4", ".6", ".8–1"])
axC.set(xlabel="Pseudotime interval", ylabel="Mean isotype fraction",
        title="Class-switch composition changes along trajectory", ylim=(0, 1))
axC.legend(frameon=False, ncol=2, fontsize=7, loc="upper left")
panel(axC, "C")

for name, color in zip(programs, [COL["green"], COL["orange"], COL["magenta"]]):
    z = prog_long[prog_long.program == name]
    s = bootstrap_summary(z, "score", "pt_bin")
    xx = np.arange(len(s))
    axD.plot(xx, s["mean"], marker="o", ms=3.5, lw=1.6, color=color, label=name)
    axD.fill_between(xx, s.lo, s.hi, color=color, alpha=.12, linewidth=0)
axD.axhline(0, color=COL["light"], lw=.8)
axD.set_xticks(range(5), ["0", ".2", ".4", ".6", ".8–1"])
axD.set(xlabel="Pseudotime interval", ylabel="Participant-mean module score",
        title="Effector programs emerge along pseudotime")
axD.legend(frameon=False, fontsize=6.8, loc="best")
panel(axD, "D")

for name in ["IgM", "IgA", "IgG"]:
    z = shm_part[shm_part.isotype == name]
    s = bootstrap_summary(z, "total_shm", "pt_bin")
    xx = np.arange(len(s))
    axE.plot(xx, 100*s["mean"], marker="o", ms=3.5, lw=1.6, color=iso_colors[name], label=name)
    axE.fill_between(xx, 100*s.lo, 100*s.hi, color=iso_colors[name], alpha=.12, linewidth=0)
axE.set_xticks(range(5), ["0", ".2", ".4", ".6", ".8–1"])
axE.set(xlabel="Pseudotime interval", ylabel="Median SHM (% informative nt)",
        title="Somatic hypermutation stratifies switched isotypes")
axE.legend(frameon=False, fontsize=7, loc="upper left")
panel(axE, "E")

for ax in [axB, axC, axD, axE]:
    sns.despine(ax=ax)
    ax.grid(axis="y", color="#E7EBEF", lw=.6)
fig.suptitle("BCR-linked B-cell trajectories resolve class switching and effector-state programs",
             fontsize=14, fontweight="bold", color=COL["ink"])
fig.savefig(OUT / "Figure_BT1_trajectory_repertoire_integration.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Figure_BT1_trajectory_repertoire_integration.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# Clone-state connectivity: each clone contributes at most once to a state pair.
f = pd.read_csv(FULL, low_memory=False)
f = f[f.full_clone_size > 1].dropna(subset=["exact_paired_clone", "PatientID", "state"]).copy()
states = sorted(f.state.unique())
pairs = list(itertools.combinations(states, 2))
clone_code, clone_names = pd.factorize(f.exact_paired_clone)
state_code = pd.Categorical(f.state, states).codes
part_groups = [idx for idx in f.groupby("PatientID").indices.values()]

def cooccurrence(sc):
    presence = np.zeros((len(clone_names), len(states)), dtype=np.int16)
    presence[clone_code, sc] = 1
    mat = presence.T @ presence
    return np.array([mat[i, j] for i, j in itertools.combinations(range(len(states)), 2)])

obs = cooccurrence(state_code)
perm = np.zeros((500, len(pairs)), dtype=int)
for b in range(500):
    shuffled = state_code.copy()
    for idx in part_groups:
        shuffled[idx] = RNG.permutation(shuffled[idx])
    perm[b] = cooccurrence(shuffled)
exp = perm.mean(0)
p_emp = (1 + (perm >= obs).sum(0)) / (perm.shape[0] + 1)
edge = pd.DataFrame(pairs, columns=["state_1", "state_2"])
edge["observed_clones"] = obs
edge["expected_clones"] = exp
edge["log2_observed_expected"] = np.log2((obs + .5) / (exp + .5))
edge["empirical_p"] = p_emp
edge["fdr"] = bh(p_emp)
edge.to_csv(OUT / "Table_BT8_clone_state_connectivity_permutation.csv", index=False)

sig = edge[(edge.observed_clones >= 4) & (edge.log2_observed_expected > 0)].copy()
sig = sig.sort_values(["fdr", "log2_observed_expected"], ascending=[True, False]).head(16)
G = nx.Graph()
node_n = f.groupby("state").size()
for s in states:
    G.add_node(s, n=int(node_n.get(s, 0)))
for _, r in sig.iterrows():
    G.add_edge(r.state_1, r.state_2, enrich=r.log2_observed_expected,
               observed=r.observed_clones, fdr=r.fdr)

short = {
    "Transitional B":"Transitional", "Naive B":"Naive", "Naive-IFN B":"Naive–IFN",
    "CD5+ B Cell":"CD5+", "Non-switched memory B":"Non-switched\nmemory",
    "Switched memory B":"Switched\nmemory", "Atypical memory B":"Atypical\nmemory",
    "IgM Plasma B Cell":"IgM plasma", "IgA Plasma B Cell":"IgA plasma",
    "IgG Plasma B Cell":"IgG plasma"
}
pos = {
    "Transitional B":(0,0), "Non-switched memory B":(1.15,0), "Naive B":(2.3,.75),
    "Naive-IFN B":(2.3,-.35), "CD5+ B Cell":(2.3,-1.35),
    "Switched memory B":(3.55,-1.35), "Atypical memory B":(3.55,.25),
    "IgM Plasma B Cell":(3.55,-.35), "IgA Plasma B Cell":(4.75,.5),
    "IgG Plasma B Cell":(4.75,-.55)
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.3), gridspec_kw={"width_ratios":[1.3, 1]}, constrained_layout=True)
for u, v, dat in G.edges(data=True):
    x1, y1 = pos[u]; x2, y2 = pos[v]
    ax1.plot([x1,x2], [y1,y2], color=COL["teal"], alpha=.25 + .12*min(dat["enrich"],4),
             lw=.8 + 1.25*min(dat["enrich"],4), zorder=1)
sizes = [85 + 480*np.sqrt(G.nodes[s]["n"] / max(node_n)) for s in states]
ax1.scatter([pos[s][0] for s in states], [pos[s][1] for s in states], s=sizes,
            color="#DCEDEF", edgecolor=COL["teal"], lw=1.2, zorder=3)
for s in states:
    ax1.text(pos[s][0], pos[s][1]-.16, short[s], ha="center", va="top", fontsize=7.2, zorder=4)
ax1.set(title="Observed cross-state sharing among expanded paired clones", xlim=(-.45,5.2), ylim=(-1.95,1.25))
ax1.axis("off"); panel(ax1,"A")
ax1.text(.01,.01,"Edges shown are descriptive; none survives FDR correction",
         transform=ax1.transAxes, fontsize=7, color=COL["grey"])

top = sig.sort_values("log2_observed_expected").tail(12)
labels = [f"{short[a].replace(chr(10),' ')}  ↔  {short[b].replace(chr(10),' ')}" for a,b in zip(top.state_1, top.state_2)]
y = np.arange(len(top))
ax2.barh(y, top.log2_observed_expected, color=COL["grey"], alpha=.82)
ax2.set_yticks(y, labels)
ax2.set(xlabel="log2(observed / permutation-expected clones)", title="No state pair exceeds the matched null after FDR")
for yy, (_, r) in zip(y, top.iterrows()):
    q = "<0.01" if r.fdr < .01 else f"={r.fdr:.2f}"
    ax2.text(r.log2_observed_expected+.04, yy, f"n={int(r.observed_clones)}, FDR {q}", va="center", fontsize=6.8)
ax2.axvline(0, color=COL["grey"], lw=.8)
ax2.set_xlim(0, max(1, top.log2_observed_expected.max()*1.38))
sns.despine(ax=ax2); ax2.grid(axis="x", color="#E7EBEF", lw=.6)
panel(ax2,"B")
fig.suptitle("Cross-state clone sharing is sparse and not enriched above a participant-matched null",
             fontsize=13.5, fontweight="bold", color=COL["ink"])
fig.savefig(OUT / "Figure_BT2_clone_state_connectivity.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Figure_BT2_clone_state_connectivity.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"clone slope={slope:.4f}, p={p_slope:.3g}")
print(f"paired trajectory cells={len(paired)}, SHM-mapped={d.total_shm.notna().sum()}")
print(f"connectivity clone cells={len(f)}, significant/top edges={len(sig)}")

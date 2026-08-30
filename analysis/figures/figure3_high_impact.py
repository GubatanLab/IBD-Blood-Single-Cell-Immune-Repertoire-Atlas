from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def build(root, main, col, concise_module_label, heatmap, style_axis, panel_title, panel_label, save_figure):
    read = lambda path: pd.read_csv(path, low_memory=False)
    revision_dir = root / "High Impact Additional Analyses/Figure 3 High Impact Revision"
    source_dir = root / "Cell Press Redrawn Figure Set/Source Data"
    source_dir.mkdir(parents=True, exist_ok=True)

    edges = read(root / "High Impact Additional Analyses/Paired TCR Sequence State/Table_TSS3_paired_TCR_distance_graph_edges.csv")
    nodes = read(root / "High Impact Additional Analyses/Paired TCR Sequence State/Table_TSS1_paired_alpha_beta_clone_sequence_state.csv")
    transition = read(revision_dir / "Table_F3R15_transition_robustness_meta_analysis.csv")
    transition_series = read(revision_dir / "Table_F3R14_transition_robustness_by_series.csv")
    attenuation = read(revision_dir / "Table_F3R23_primary_attenuation_bootstrap.csv")
    attenuation_omnibus = read(revision_dir / "Table_F3R24_primary_attenuation_omnibus.csv")
    diagnosis_meta = read(revision_dir / "Table_F3R19_diagnosis_convergence_meta_analysis.csv")
    diagnosis_interaction = read(revision_dir / "Table_F3R22_diagnosis_interaction_test.csv")
    inflammation = read(revision_dir / "Table_F3R20_inflammation_assortativity_summary.csv")
    inflammation_composition = read(revision_dir / "Table_F3R21_inflammation_edge_composition.csv")
    gamma_vpairs = read(revision_dir / "Table_F3R16_gamma_delta_participant_normalized_V_pairs.csv")
    gamma_states = read(revision_dir / "Table_F3R17_gamma_delta_participant_normalized_states.csv")
    gamma_programs = read(revision_dir / "Table_F3R12_gamma_delta_program_effects.csv")
    gamma_pairing = read(revision_dir / "Table_F3R25_gamma_delta_pairing_enrichment.csv")
    gamma_v9v2_participants = read(revision_dir / "Table_F3R26_gamma_delta_participant_V9V2_fraction.csv")
    gamma_qc = read(revision_dir / "Table_F3R11_gamma_delta_QC_summary.csv")

    gamma_color = "#7B3294"
    alpha_fill = "#E8F3F8"
    bridge_fill = "#FFF1E6"
    gamma_fill = "#F3EAF5"
    program_colors = {
        "EOMES_ZEB2_inflammatory_CD8_TRM_like": "#D55E00",
        "Effector_cytotoxicity": "#CC79A7",
        "Th1_Tc1_inflammatory": "#009E73",
        "Tissue_resident_mucosal_retention": col["muted"],
        "Gut_homing_intestinal_trafficking": col["muted"],
    }
    modules = [
        "EOMES_ZEB2_inflammatory_CD8_TRM_like",
        "Effector_cytotoxicity",
        "Th1_Tc1_inflammatory",
        "Tissue_resident_mucosal_retention",
        "Gut_homing_intestinal_trafficking",
    ]
    primary_modules = modules[:3]
    primary_labels = ["EOMES-ZEB2", "Cytotoxicity", "Th1/Tc1"]
    qc = gamma_qc.set_index("metric")["value"]

    fig = plt.figure(figsize=(7.48, 9.35))
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[0.92, 0.93, 0.84, 1.02],
        width_ratios=[1.0, 1.08],
        hspace=0.70,
        wspace=0.68,
        left=0.075,
        right=0.985,
        top=0.975,
        bottom=0.060,
    )

    # A: a single left-to-right Figure 2 -> sequence-neighborhood argument.
    ax = fig.add_subplot(gs[0, 0])
    # Extend panel A into its otherwise unused lower inter-row whitespace. This
    # enlarges the network without changing the Cell Press canvas or panel B.
    panel_a_position = ax.get_position()
    ax.set_position(
        [
            panel_a_position.x0,
            panel_a_position.y0 - 0.042,
            panel_a_position.width * 1.14,
            panel_a_position.height + 0.042,
        ]
    )
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.00, 0.08),
            0.315,
            0.82,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            fc="#F7F7F5",
            ec="#D8D8D3",
            lw=0.55,
        )
    )
    ax.text(0.025, 0.85, "FIGURE 2", fontsize=5.6, color=col["UC"], fontweight="bold", va="top")
    ax.text(0.025, 0.74, "Exact-clone expansion", fontsize=6.4, color=col["ink"], fontweight="bold", va="top")
    bridge_programs = [
        ("EOMES-ZEB2", program_colors["EOMES_ZEB2_inflammatory_CD8_TRM_like"]),
        ("Cytotoxicity", program_colors["Effector_cytotoxicity"]),
        ("Th1/Tc1", program_colors["Th1_Tc1_inflammatory"]),
    ]
    for y_value, (label, color) in zip([0.52, 0.40, 0.28], bridge_programs):
        ax.plot([0.03, 0.085], [y_value, y_value], color=color, lw=3.8, solid_capstyle="round", clip_on=False)
        ax.text(0.105, y_value, label, fontsize=5.6, color=col["ink"], va="center")

    ax.add_patch(
        FancyArrowPatch(
            (0.320, 0.50),
            (0.402, 0.50),
            arrowstyle="-|>",
            mutation_scale=10,
            color=col["TCR"],
            lw=1.0,
        )
    )
    ax.text(
        0.36,
        0.61,
        "Test paired αβ\nsequence neighbors",
        ha="center",
        va="center",
        fontsize=4.45,
        color=col["ink"],
        linespacing=0.95,
    )
    ax.text(0.36, 0.31, "same cohort\n182 participants", ha="center", va="center", fontsize=4.6, color=col["muted"], linespacing=0.95)

    example_edges = edges[edges.acquisition_series.eq("S3")]
    graph = nx.from_pandas_edgelist(example_edges, "node1_id", "node2_id", edge_attr="sequence_distance")
    component = max(
        nx.connected_components(graph),
        key=lambda component_nodes: (len({item.split("::", 1)[0] for item in component_nodes}), len(component_nodes)),
    )
    graph = graph.subgraph(component).copy()
    node_data = nodes.copy()
    node_data["node_id"] = node_data.SampleID.astype(str) + "::" + node_data.clone_id.astype(str)
    node_data["EOMES_z"] = node_data.groupby("SampleID")["EOMES_ZEB2_inflammatory_CD8_TRM_like"].transform(
        lambda values: ((values - values.mean()) / values.std()).fillna(0)
    )
    node_map = node_data.set_index("node_id")
    values = np.array([node_map.loc[node, "EOMES_z"] for node in graph.nodes()], float)
    positions = nx.spring_layout(graph, seed=20260828, weight=None, k=0.55)
    # Reserve a dedicated title band above and a statistics/colorbar band below
    # so no network nodes or annotations compete with labels.
    network_ax = ax.inset_axes([0.39, 0.195, 0.61, 0.650])
    edge_widths = []
    for node1, node2 in graph.edges():
        distance = float(graph.edges[node1, node2].get("sequence_distance", 0.50))
        similarity_strength = np.clip((0.50 - distance) / 0.50, 0, 1)
        edge_widths.append(0.35 + 0.95 * similarity_strength)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=network_ax,
        width=edge_widths,
        alpha=0.46,
        edge_color="#9E9E99",
    )
    vmax = max(1.5, np.nanpercentile(np.abs(values), 95))
    eomes_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "eomes_diverging", ["#2C7BB6", "#F2F2EF", program_colors["EOMES_ZEB2_inflammatory_CD8_TRM_like"]]
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=network_ax,
        node_color=values,
        cmap=eomes_cmap,
        vmin=-vmax,
        vmax=vmax,
        node_size=64,
        edgecolors="white",
        linewidths=0.62,
    )
    node_values = dict(zip(graph.nodes(), values))
    hottest_edge = max(graph.edges(), key=lambda pair: node_values[pair[0]] + node_values[pair[1]])
    hotspot = tuple((positions[hottest_edge[0]][axis] + positions[hottest_edge[1]][axis]) / 2 for axis in (0, 1))
    network_ax.annotate(
        "Concordant\ninflammatory activity",
        xy=hotspot,
        xycoords="data",
        xytext=(0.66, 0.055),
        textcoords="axes fraction",
        fontsize=4.8,
        color=program_colors["EOMES_ZEB2_inflammatory_CD8_TRM_like"],
        ha="left",
        va="bottom",
        linespacing=0.95,
        arrowprops={"arrowstyle": "-", "lw": 0.55, "color": program_colors["EOMES_ZEB2_inflammatory_CD8_TRM_like"]},
        bbox={"fc": "white", "ec": "none", "alpha": 0.84, "pad": 0.5},
    )
    network_ax.axis("off")
    cax = ax.inset_axes([0.50, 0.045, 0.43, 0.026])
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-vmax, vmax), cmap=eomes_cmap), cax=cax, orientation="horizontal"
    )
    colorbar.set_ticks([-2, 0, 2])
    colorbar.ax.tick_params(labelsize=4.9, length=1.3, pad=1)
    colorbar.set_label("Participant-centered EOMES-ZEB2 score", fontsize=5.1, labelpad=1)
    ax.text(0.39, 0.965, "Cross-participant sequence neighborhood", fontsize=6.1, color=col["ink"], fontweight="bold", va="top")
    ax.text(0.39, 0.895, "All edges connect different participants", fontsize=5.1, color=col["TCR"], va="top")
    ax.text(
        0.39,
        0.115,
        f"{len(graph)} clonotypes · {len({item.split('::', 1)[0] for item in graph})} participants",
        fontsize=5.0,
        color=col["muted"],
        va="bottom",
    )
    panel_title(ax, "Expansion-linked programs recur in αβ neighborhoods")
    panel_label(ax, "A")

    # B: lead with the primary pooled answer.
    ax = fig.add_subplot(gs[0, 1])
    baseline = transition[
        transition.configuration.eq("Paired alpha-beta baseline") & transition.module.isin(modules)
    ].set_index("module").reindex(modules)
    y = np.arange(len(baseline))[::-1]
    for yi, (module, row) in zip(y, baseline.iterrows()):
        color = program_colors[module]
        if row.ci_low <= 0:
            color = col["muted"]
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, lw=1.25)
        ax.plot(row.pooled_edge_correlation, yi, "o", color=color, ms=4.1)
        ax.text(
            1.02,
            yi,
            f"{int(row.n_series_positive)}/{int(row.n_series)}+ · I² {row.I2_percent:.0f}%",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=5.1,
            clip_on=False,
        )
    ax.axvline(0, color=col["muted"], lw=0.65, ls="--")
    ax.set_yticks(y, [concise_module_label(module) for module in modules])
    ax.tick_params(axis="y", labelsize=5.7)
    ax.set_xlabel("Pooled edge correlation (95% CI)", labelpad=2)
    ax.text(0.99, 0.03, "43,742 clonotypes · 5,548 edges · 181 participants", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8, color=col["muted"])
    style_axis(ax, "x")
    panel_title(ax, "Sequence-related paired TCRs converge on Figure 2 programs")
    panel_label(ax, "B")

    # C: diagnosis-restricted replication makes the sequence result explicitly IBD-facing.
    ax = fig.add_subplot(gs[1, 0])
    diagnosis_frame = diagnosis_meta[
        diagnosis_meta.diagnosis.isin(["CD", "UC"]) & diagnosis_meta.module.isin(primary_modules)
    ]
    y = np.arange(len(primary_modules))[::-1]
    offsets = {"CD": 0.10, "UC": -0.10}
    markers = {"CD": "o", "UC": "s"}
    for diagnosis in ("CD", "UC"):
        frame = diagnosis_frame[diagnosis_frame.diagnosis.eq(diagnosis)].set_index("module").reindex(primary_modules)
        yy = y + offsets[diagnosis]
        for yi, (_, row) in zip(yy, frame.iterrows()):
            ax.plot([row.ci_low, row.ci_high], [yi, yi], color=col[diagnosis], lw=1.15)
            ax.plot(row.pooled_edge_correlation, yi, marker=markers[diagnosis], color=col[diagnosis], ms=4.0, ls="")
    ax.axvline(0, color=col["muted"], lw=0.65, ls="--")
    ax.set_yticks(y, primary_labels)
    ax.tick_params(axis="y", labelsize=5.6)
    ax.set_xlabel("Diagnosis-specific edge correlation (95% CI)")
    ax.legend(
        [
            mpl.lines.Line2D([0], [0], marker="o", color=col["CD"], lw=1, ms=3.8),
            mpl.lines.Line2D([0], [0], marker="s", color=col["UC"], lw=1, ms=3.8),
        ],
        ["CD", "UC"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.01),
        ncol=2,
        fontsize=5.2,
        handlelength=1.4,
    )
    ax.set_ylim(-0.35, 2.55)
    omnibus_p = float(diagnosis_interaction.omnibus_p.iloc[0])
    ax.text(
        0.99,
        0.03,
        f"CD: 1,023 edges · UC: 1,922 edges\nCD-UC omnibus interaction P={omnibus_p:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.8,
        color=col["muted"],
    )
    style_axis(ax, "x")
    panel_title(ax, "Program convergence is conserved in CD and UC")
    panel_label(ax, "C")

    # D: uncertainty-forward grouped forest plot with paired attenuation tests.
    ax = fig.add_subplot(gs[1, 1])
    config_order = [
        "Paired alpha-beta baseline",
        "Clone-size adjusted",
        "Singleton clonotypes only",
        "Dominant-state adjusted",
        "Paired CDR3 only",
        "Alpha chain only",
        "Beta chain only",
    ]
    sensitivity = transition[transition.module.isin(primary_modules)]
    y_positions = {
        "Paired alpha-beta baseline": 8.00,
        "Clone-size adjusted": 6.55,
        "Singleton clonotypes only": 5.55,
        "Dominant-state adjusted": 4.05,
        "Paired CDR3 only": 2.65,
        "Alpha chain only": 1.65,
        "Beta chain only": 0.65,
    }
    edge_totals = (
        sensitivity.groupby("configuration", sort=False).total_edges.first().reindex(config_order).astype(int).to_dict()
    )
    n_transition_participants = int(
        transition_series[
            transition_series.configuration.eq("Paired alpha-beta baseline")
            & transition_series.module.eq(primary_modules[0])
        ].n_participants.sum()
    )

    row_labels = {
        "Paired alpha-beta baseline": "Paired αβ baseline",
        "Clone-size adjusted": "Clone-size adjusted",
        "Singleton clonotypes only": "Singleton only",
        "Dominant-state adjusted": "Dominant-state adjusted",
        "Paired CDR3 only": "Paired CDR3 only",
        "Alpha chain only": "α-chain only",
        "Beta chain only": "β-chain only",
    }
    clone_omnibus = float(
        attenuation_omnibus.loc[attenuation_omnibus.comparison.eq("Clone-size adjusted"), "omnibus_p"].iloc[0]
    )
    state_omnibus = float(
        attenuation_omnibus.loc[attenuation_omnibus.comparison.eq("Dominant-state adjusted"), "omnibus_p"].iloc[0]
    )
    state_exponent = int(np.floor(np.log10(state_omnibus)))
    state_coefficient = state_omnibus / (10**state_exponent)
    module_offsets = {
        "EOMES_ZEB2_inflammatory_CD8_TRM_like": 0.18,
        "Effector_cytotoxicity": 0.00,
        "Th1_Tc1_inflammatory": -0.18,
    }
    module_markers = {
        "EOMES_ZEB2_inflammatory_CD8_TRM_like": "o",
        "Effector_cytotoxicity": "s",
        "Th1_Tc1_inflammatory": "^",
    }
    ax.axhspan(7.62, 8.38, color="#F5F5F2", zorder=0)
    ax.axhspan(5.17, 6.92, color="#EFF8F2", zorder=0)
    ax.axhspan(3.62, 4.48, color="#EAF2F7", zorder=0)
    ax.axhspan(0.25, 3.08, color="#F8F8F6", zorder=0)
    ax.axvline(0, color=col["muted"], lw=0.65, ls="--", zorder=0)
    for module, label in zip(primary_modules, primary_labels):
        module_frame = sensitivity[sensitivity.module.eq(module)].set_index("configuration").reindex(config_order)
        color = program_colors[module]
        marker = module_markers[module]
        baseline_row = module_frame.loc["Paired alpha-beta baseline"]
        baseline_y = y_positions["Paired alpha-beta baseline"] + module_offsets[module]
        for adjusted_configuration, line_style, line_alpha in [
            ("Clone-size adjusted", "-", 0.26),
            ("Dominant-state adjusted", (0, (2.0, 2.0)), 0.30),
        ]:
            adjusted_row = module_frame.loc[adjusted_configuration]
            adjusted_y = y_positions[adjusted_configuration] + module_offsets[module]
            ax.add_patch(
                FancyArrowPatch(
                    (baseline_row.pooled_edge_correlation, baseline_y),
                    (adjusted_row.pooled_edge_correlation, adjusted_y),
                    arrowstyle="-|>" if adjusted_configuration == "Dominant-state adjusted" else "-",
                    mutation_scale=4.5,
                    connectionstyle="arc3,rad=0.035",
                    color=color,
                    lw=0.70,
                    linestyle=line_style,
                    alpha=line_alpha,
                    zorder=1,
                )
            )
        for configuration, row in module_frame.iterrows():
            y_value = y_positions[configuration] + module_offsets[module]
            sequence_content = configuration in {"Paired CDR3 only", "Alpha chain only", "Beta chain only"}
            interval_alpha = 0.62 if sequence_content else 0.96
            interval_width = 0.82 if sequence_content else 1.05
            ax.plot(
                [row.ci_low, row.ci_high],
                [y_value, y_value],
                color=color,
                lw=interval_width,
                alpha=interval_alpha,
                zorder=2,
            )
            interval_excludes_zero = row.ci_low > 0 or row.ci_high < 0
            marker_size = 4.45 if configuration == "Paired alpha-beta baseline" else (3.45 if sequence_content else 3.85)
            ax.plot(
                row.pooled_edge_correlation,
                y_value,
                marker=marker,
                ms=marker_size,
                mfc=color if interval_excludes_zero else "white",
                mec=color,
                mew=0.85,
                ls="",
                alpha=0.72 if sequence_content else 1.0,
                zorder=3,
            )
            if configuration in {
                "Paired alpha-beta baseline",
                "Clone-size adjusted",
                "Dominant-state adjusted",
            }:
                ax.text(
                    0.370,
                    y_value,
                    f"{row.pooled_edge_correlation:.2f}",
                    fontsize=4.90,
                    color=color,
                    ha="right",
                    va="center",
                )

    ax.set_yticks([y_positions[configuration] for configuration in config_order])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for configuration in config_order:
        ax.text(
            -0.13,
            y_positions[configuration],
            row_labels[configuration],
            transform=ax.get_yaxis_transform(),
            fontsize=4.85,
            color=col["muted"] if configuration in {"Paired CDR3 only", "Alpha chain only", "Beta chain only"} else col["ink"],
            ha="right",
            va="center",
            clip_on=False,
        )
    ax.text(-0.02, 8.90, "EDGES", transform=ax.get_yaxis_transform(), fontsize=4.25, color=col["muted"], fontweight="bold", ha="right", va="center", clip_on=False)
    ax.text(0.370, 8.90, "r", fontsize=4.80, color=col["muted"], fontweight="bold", ha="right", va="center")
    for configuration in config_order:
        ax.text(
            -0.02,
            y_positions[configuration],
            f"{edge_totals[configuration] / 1000:.1f}k",
            transform=ax.get_yaxis_transform(),
            fontsize=4.35,
            color=col["muted"],
            ha="right",
            va="center",
            clip_on=False,
        )
    ax.set_xlim(-0.04, 0.38)
    ax.set_ylim(0.20, 9.25)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Pooled edge correlation (95% CI)")
    style_axis(ax, "x")
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker=module_markers[module],
            color=program_colors[module],
            lw=1.0,
            ms=3.5,
            label=label,
        )
        for module, label in zip(primary_modules, primary_labels)
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.00),
        ncol=3,
        fontsize=4.6,
        handlelength=1.4,
        columnspacing=0.9,
    )
    group_headers = [
        (8.52, "REFERENCE", col["muted"]),
        (7.22, "EXPANSION INDEPENDENCE", "#247A4A"),
        (4.72, "STATE CONTEXT", "#365F91"),
        (3.32, "SEQUENCE CONTENT", col["muted"]),
    ]
    for y_value, label, header_color in group_headers:
        ax.text(-0.34, y_value, label, transform=ax.get_yaxis_transform(), fontsize=4.55, color=header_color, fontweight="bold", ha="left", va="center", clip_on=False)
    ribbon_y = -0.370
    ribbon_height = 0.145
    ax.add_patch(
        FancyBboxPatch(
            (0.00, ribbon_y),
            1.00,
            ribbon_height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="white",
            edgecolor="#D8D8D3",
            lw=0.45,
            clip_on=False,
            zorder=5,
        )
    )
    ax.add_patch(mpl.patches.Rectangle((0.004, ribbon_y + 0.004), 0.492, ribbon_height - 0.008, transform=ax.transAxes, facecolor="#EFF8F2", edgecolor="none", clip_on=False, zorder=5.1))
    ax.add_patch(mpl.patches.Rectangle((0.504, ribbon_y + 0.004), 0.492, ribbon_height - 0.008, transform=ax.transAxes, facecolor="#EAF2F7", edgecolor="none", clip_on=False, zorder=5.1))
    ax.plot([0.50, 0.50], [ribbon_y + 0.015, ribbon_y + ribbon_height - 0.015], transform=ax.transAxes, color="#D8D8D3", lw=0.45, clip_on=False, zorder=5.2)
    ax.text(
        0.25,
        ribbon_y + ribbon_height / 2,
        f"EXPANSION: PRESERVED\n6/6 CIs exclude 0 · omnibus P={clone_omnibus:.2f}",
        transform=ax.transAxes,
        fontsize=4.90,
        color="#247A4A",
        fontweight="bold",
        ha="center",
        va="center",
        linespacing=1.08,
        zorder=6,
    )
    ax.text(
        0.75,
        ribbon_y + ribbon_height / 2,
        f"STATE: ATTENUATED\n0/3 CIs exclude 0 · omnibus P={state_coefficient:.1f}×10$^{{{state_exponent}}}$",
        transform=ax.transAxes,
        fontsize=4.90,
        color="#365F91",
        fontweight="bold",
        ha="center",
        va="center",
        linespacing=1.08,
        zorder=6,
    )
    ax.text(
        0.50,
        -0.415,
        "Filled/open = CI excludes/includes 0 · arrows = baseline → adjusted · participant-jackknife CIs · 2,000 block bootstraps · no batch term",
        transform=ax.transAxes,
        fontsize=3.85,
        color=col["muted"],
        ha="center",
        va="top",
    )
    panel_title(ax, "Expansion-independent convergence is state-sensitive")
    panel_label(ax, "D")

    # E: IBD inflammatory-status organization.
    inflammation_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], wspace=0.62, width_ratios=[0.96, 1.04])
    ax = fig.add_subplot(inflammation_grid[0, 0])
    stratum_order = ["Pooled IBD", "CD", "UC"]
    inflammation_frame = inflammation.set_index("stratum").reindex(stratum_order)
    y = np.arange(len(stratum_order))[::-1]
    stratum_colors = {"Pooled IBD": col["ink"], "CD": col["CD"], "UC": col["UC"]}
    for yi, (stratum, row) in zip(y, inflammation_frame.iterrows()):
        color = stratum_colors[stratum]
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, lw=1.25)
        ax.plot(row.excess_same_inflammation_fraction, yi, "o", color=color, ms=4.2)
        p_label = "P<0.001" if row.permutation_p < 0.001 else f"P={row.permutation_p:.3f}"
        ax.text(
            1.02,
            yi,
            f"{p_label} · {int(row.n_participants)} P",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=4.9,
            clip_on=False,
        )
    ax.axvline(0, color=col["muted"], lw=0.65, ls="--")
    ax.set_yticks(y, stratum_order)
    ax.set_xlabel("Excess same-inflammation edge fraction (95% CI)")
    ax.tick_params(axis="y", labelsize=5.5)
    style_axis(ax, "x")
    panel_title(ax, "Sequence neighborhoods associate with inflammatory status")
    panel_label(ax, "E", x=-0.22, y=1.08)

    ax = fig.add_subplot(inflammation_grid[0, 1])
    category_order = ["Inflamed-Inflamed", "Noninflamed-Noninflamed", "Mixed"]
    category_labels = ["Inflamed-Inflamed", "Noninflamed-Noninflamed", "Mixed status"]
    category_colors = ["#B2182B", "#4D7EA8", col["muted"]]
    composition = inflammation_composition[
        inflammation_composition.stratum.eq("Pooled IBD")
    ].set_index("category").reindex(category_order)
    y = np.arange(len(category_order))[::-1]
    for yi, color, (_, row) in zip(y, category_colors, composition.iterrows()):
        ax.plot([row.expected_fraction, row.observed_fraction], [yi, yi], color=color, lw=1.2, alpha=0.75)
        ax.plot(row.expected_fraction, yi, "o", mfc="white", mec=color, mew=0.9, ms=4.0)
        ax.plot(row.observed_fraction, yi, "o", color=color, ms=4.2)
        q_value = row.permutation_FDR_within_stratum
        q_label = "q<0.01" if q_value < 0.01 else f"q={q_value:.2f}"
        ax.text(
            1.02,
            yi,
            f"O/E {row.observed_to_expected_ratio:.2f} · {q_label}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=4.9,
            clip_on=False,
        )
    ax.set_yticks(y, category_labels)
    ax.set_ylim(-0.45, 2.45)
    ax.set_xlabel("Within-diagnosis IBD edge fraction (open=expected; filled=observed)")
    ax.tick_params(axis="y", labelsize=5.4)
    style_axis(ax, "x")
    ax.text(0.98, 0.03, "3,246 CD-CD or UC-UC edges · 10,000 permutations", transform=ax.transAxes, ha="right", fontsize=4.8, color=col["muted"])
    panel_title(ax, "Mixed inflammatory-status edges are depleted")

    # F: participant-normalized pairing enrichment plus participant heterogeneity.
    gamma_grid = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[3, 0],
        width_ratios=[0.60, 0.18, 0.22],
        wspace=0.44,
    )
    ax = fig.add_subplot(gamma_grid[0, 0])
    gamma_order = (
        gamma_pairing.groupby("gamma_v").n_participant_carriers.max().sort_values(ascending=False).head(8).index.tolist()
    )
    delta_order = (
        gamma_pairing.groupby("delta_v").n_participant_carriers.max().sort_values(ascending=False).head(3).index.tolist()
    )
    displayed_pairing = gamma_pairing[
        gamma_pairing.gamma_v.isin(gamma_order) & gamma_pairing.delta_v.isin(delta_order)
    ].copy()
    displayed_pairing = displayed_pairing[displayed_pairing.n_cells > 0].copy()
    xmap = {name: index for index, name in enumerate(delta_order)}
    ymap = {name: index for index, name in enumerate(gamma_order)}
    n_gamma_participants = int(gamma_pairing.n_participants.max())
    total_gamma_cells = int(gamma_pairing.n_cells.sum())
    carrier_fraction = displayed_pairing.n_participant_carriers / n_gamma_participants
    marker_size = 11 + 128 * carrier_fraction
    pairing_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gamma_pairing", ["#4C78A8", "#F7F7F3", gamma_color]
    )
    pairing_norm = mpl.colors.TwoSlopeNorm(vmin=-1.6, vcenter=0.0, vmax=1.6)
    significant_pairing = displayed_pairing.fdr < 0.05
    scatter = ax.scatter(
        displayed_pairing.delta_v.map(xmap),
        displayed_pairing.gamma_v.map(ymap),
        s=marker_size,
        c=displayed_pairing.log2_observed_expected,
        cmap=pairing_cmap,
        norm=pairing_norm,
        edgecolor=np.where(significant_pairing, col["ink"], "white"),
        linewidth=np.where(significant_pairing, 0.85, 0.45),
        zorder=3,
    )
    ax.set_xticks(range(len(delta_order)), [name.replace("TRDV", "Vδ") for name in delta_order])
    ax.set_yticks(range(len(gamma_order)), [name.replace("TRGV", "Vγ") for name in gamma_order])
    ax.set_xlabel("Paired δ-chain V gene")
    ax.set_ylabel("γ-chain V gene")
    ax.set_xlim(-0.48, len(delta_order) - 0.52)
    ax.set_ylim(len(gamma_order) - 0.48, -0.78)
    ax.set_xticks(np.arange(-0.5, len(delta_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(gamma_order), 1), minor=True)
    ax.grid(which="minor", color="#E7E7E2", lw=0.42)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(labelsize=5.0, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    leading = displayed_pairing[
        displayed_pairing.gamma_v.eq("TRGV9") & displayed_pairing.delta_v.eq("TRDV2")
    ].iloc[0]
    ax.text(
        -0.44,
        -0.54,
        f"{int(leading.n_cells)}/{total_gamma_cells} cells · {int(leading.n_participant_carriers)}/{n_gamma_participants} P · mean {100 * leading.observed_mean_participant_fraction:.0f}%",
        fontsize=4.25,
        color=gamma_color,
        fontweight="bold",
        ha="left",
        va="center",
    )
    key_ax = fig.add_subplot(gamma_grid[0, 1])
    key_ax.set_axis_off()
    key_ax.text(
        0.50,
        0.97,
        "PAIRING KEYS",
        transform=key_ax.transAxes,
        fontsize=4.15,
        color=col["muted"],
        fontweight="bold",
        ha="center",
        va="top",
    )
    key_ax.text(0.50, 0.83, "Carriers", transform=key_ax.transAxes,
                fontsize=4.0, color=col["muted"], ha="center", va="center")
    for carrier_y, fraction in zip((0.70, 0.58, 0.46), (0.75, 0.50, 0.25)):
        key_ax.scatter(
            [0.32], [carrier_y], s=11 + 128 * fraction,
            facecolor="white", edgecolor=col["muted"], linewidth=0.55,
            transform=key_ax.transAxes, clip_on=False,
        )
        key_ax.text(
            0.64, carrier_y, f"{int(fraction * n_gamma_participants)} P",
            transform=key_ax.transAxes, fontsize=3.9, color=col["muted"],
            ha="left", va="center",
        )
    key_ax.scatter(
        [0.25],
        [0.35],
        s=20,
        facecolor="white",
        edgecolor=col["ink"],
        linewidth=0.85,
        transform=key_ax.transAxes,
        clip_on=False,
    )
    key_ax.text(
        0.50,
        0.35,
        "FDR<0.05",
        transform=key_ax.transAxes,
        fontsize=3.85,
        color=col["muted"],
        ha="left",
        va="center",
    )
    cbar_ax = key_ax.inset_axes([0.42, 0.05, 0.22, 0.23])
    colorbar = fig.colorbar(scatter, cax=cbar_ax, orientation="vertical")
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_label("$log_2$(O/E)", fontsize=4.15, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=4.2, length=1.5, pad=1)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    panel_title(ax, "Vγ9Vδ2 dominates paired γδ repertoires")
    panel_label(ax, "F")

    ax_heterogeneity = fig.add_subplot(gamma_grid[0, 2])
    diagnosis_order = ["Control", "CD", "UC"]
    diagnosis_labels = ["Ctl", "CD", "UC"]
    diagnosis_colors = {"Control": col["muted"], "CD": col["CD"], "UC": col["UC"]}
    jitter_rng = np.random.default_rng(20260828)
    for x_value, diagnosis in enumerate(diagnosis_order):
        values = gamma_v9v2_participants.loc[
            gamma_v9v2_participants.Diagnosis.eq(diagnosis), "v9v2_fraction"
        ].to_numpy(float)
        jitter = jitter_rng.uniform(-0.15, 0.15, len(values))
        ax_heterogeneity.scatter(
            x_value + jitter,
            values,
            s=6.0,
            color=diagnosis_colors[diagnosis],
            alpha=0.42,
            edgecolor="none",
            zorder=2,
        )
        quartile_low, median, quartile_high = np.quantile(values, [0.25, 0.50, 0.75])
        ax_heterogeneity.plot(
            [x_value, x_value], [quartile_low, quartile_high], color=diagnosis_colors[diagnosis], lw=2.4, zorder=3
        )
        ax_heterogeneity.plot(
            [x_value - 0.18, x_value + 0.18], [median, median], color=col["ink"], lw=1.0, zorder=4
        )
        ax_heterogeneity.text(
            x_value, 1.025, f"{len(values)} P", fontsize=4.0, color=col["muted"], ha="center", va="bottom"
        )
    diagnosis_p = float(gamma_v9v2_participants.diagnosis_kruskal_wallis_p.iloc[0])
    ax_heterogeneity.set_xticks(range(3), diagnosis_labels)
    ax_heterogeneity.set_ylim(-0.04, 1.10)
    ax_heterogeneity.set_yticks([0, 0.5, 1.0])
    ax_heterogeneity.set_ylabel("Vγ9Vδ2 fraction", fontsize=4.7, labelpad=2)
    ax_heterogeneity.tick_params(labelsize=4.5, length=2, pad=1)
    ax_heterogeneity.text(
        0.50,
        0.04,
        f"Kruskal-Wallis P={diagnosis_p:.2f}",
        transform=ax_heterogeneity.transAxes,
        fontsize=4.0,
        color=col["muted"],
        ha="center",
        va="bottom",
    )
    style_axis(ax_heterogeneity, "y")
    # Place the heterogeneity scale on the outer edge of panel F so its axis
    # label cannot collide with the central pairing legend or colorbar.
    ax_heterogeneity.yaxis.tick_right()
    ax_heterogeneity.yaxis.set_label_position("right")
    ax_heterogeneity.spines["left"].set_visible(False)
    ax_heterogeneity.spines["right"].set_visible(True)
    ax_heterogeneity.spines["right"].set_color(col["ink"])

    # G: participant- and state-matched gamma-delta program contrasts.
    ax = fig.add_subplot(gs[3, 1])
    gamma_program_order = modules
    gamma_program_labels = ["EOMES-ZEB2", "Cytotoxicity", "Th1/Tc1", "TRM/mucosal retention", "Gut homing"]
    block_specs = [
        ("V-pair architecture", 10.1, "RECEPTOR IDENTITY · Vγ9Vδ2 − other γδ · 38 participants"),
        ("Clonotype expansion", 4.55, "CLONOTYPE EXPANSION · expanded − singleton · 18 participants"),
    ]
    displayed_program_rows = []
    ax.axhspan(5.45, 10.45, color="#F6F6F3", zorder=0)
    ax.axhspan(-0.10, 4.90, color="#F3EAF5", alpha=0.72, zorder=0)
    ax.axvline(0, color=col["muted"], lw=0.65, ls="--", zorder=1)
    y_ticks = []
    y_labels = []
    for analysis, top_y, header in block_specs:
        block = gamma_programs[gamma_programs.analysis.eq(analysis)].set_index("module").reindex(gamma_program_order)
        ax.text(
            -0.71,
            top_y + 0.37,
            header,
            fontsize=4.25,
            color=gamma_color if analysis == "Clonotype expansion" else col["muted"],
            fontweight="bold",
            ha="left",
            va="center",
        )
        for row_index, (module, row) in enumerate(block.iterrows()):
            y_value = top_y - row_index * 0.92
            color = program_colors[module]
            ax.plot([row.ci_low, row.ci_high], [y_value, y_value], color=color, lw=1.15, zorder=2)
            interval_excludes_zero = row.ci_low > 0 or row.ci_high < 0
            ax.plot(
                row.effect,
                y_value,
                marker="o",
                ms=4.1,
                mfc=color if interval_excludes_zero else "white",
                mec=color,
                mew=0.85,
                ls="",
                zorder=3,
            )
            if row.fdr < 0.05:
                ax.text(row.effect, y_value + 0.28, "*", color=color, fontsize=6.0, fontweight="bold", ha="center", va="center")
            if analysis == "Clonotype expansion" and module in {
                "Effector_cytotoxicity", "Tissue_resident_mucosal_retention"
            }:
                ax.text(
                    1.04,
                    y_value + (0.26 if module == "Tissue_resident_mucosal_retention" else 0.0),
                    f"q={row.fdr:.3f}",
                    fontsize=4.1,
                    color=color,
                    ha="right",
                    va="center",
                )
            y_ticks.append(y_value)
            y_labels.append(gamma_program_labels[row_index])
            displayed_program_rows.append(row.to_dict() | {"analysis": analysis, "module": module})
    ax.set_yticks(y_ticks, y_labels)
    ax.tick_params(axis="y", labelsize=4.65, length=0, pad=2)
    ax.set_xlim(-0.72, 1.07)
    ax.set_ylim(-0.38, 10.78)
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("Participant-standardized program contrast (95% CI)")
    style_axis(ax, "x")
    ax.text(
        0.99,
        0.025,
        "Filled: CI excludes 0 · * FDR<0.05",
        transform=ax.transAxes,
        fontsize=4.15,
        color=col["muted"],
        ha="right",
        va="bottom",
    )
    panel_title(ax, "Expansion reveals γδ effector shifts beyond Vγ9Vδ2 identity")
    panel_label(ax, "G")

    transition.to_csv(source_dir / "Figure_3_displayed_transition_robustness.csv", index=False)
    attenuation.to_csv(source_dir / "Figure_3_displayed_primary_attenuation.csv", index=False)
    attenuation_omnibus.to_csv(source_dir / "Figure_3_displayed_primary_attenuation_omnibus.csv", index=False)
    diagnosis_frame.to_csv(source_dir / "Figure_3_displayed_diagnosis_convergence.csv", index=False)
    diagnosis_interaction.to_csv(source_dir / "Figure_3_displayed_diagnosis_interaction.csv", index=False)
    inflammation_frame.reset_index().to_csv(source_dir / "Figure_3_displayed_inflammation_assortativity.csv", index=False)
    composition.reset_index().to_csv(source_dir / "Figure_3_displayed_inflammation_edge_composition.csv", index=False)
    displayed_pairing.to_csv(source_dir / "Figure_3_displayed_gamma_delta_pairing_enrichment.csv", index=False)
    gamma_v9v2_participants.to_csv(
        source_dir / "Figure_3_displayed_gamma_delta_participant_V9V2_fraction.csv", index=False
    )
    pd.DataFrame(displayed_program_rows).to_csv(
        source_dir / "Figure_3_displayed_gamma_delta_program_effects.csv", index=False
    )
    save_figure(fig, "Figure_3", main)

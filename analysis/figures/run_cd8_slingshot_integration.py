from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "High Impact Additional Analyses" / "CD8 TCR Trajectory"
INPUT = OUT / "Table_TJ1_cd8_trajectory_input.csv.gz"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cd8_tcr_trajectory as trajectory_tools


def main() -> None:
    cells = pd.read_csv(INPUT, low_memory=False)
    reference = pd.read_csv(OUT / "Table_ST1_cd8_slingshot_reference.csv.gz", low_memory=False)
    scvi_cols = [f"scvi_{i}" for i in range(1, 21)]

    index_by_cell = pd.Series(cells.index, index=cells["cell"])
    reference_indices = index_by_cell.loc[reference["cell"]].to_numpy(int)
    ref_x = cells.loc[reference_indices, scvi_cols].to_numpy(np.float32)
    lineage_pt_cols = sorted(
        [c for c in reference.columns if c.startswith("lineage_") and c.endswith("_pseudotime")],
        key=lambda x: int(x.split("_")[1]),
    )
    lineage_weight_cols = sorted(
        [c for c in reference.columns if c.startswith("lineage_") and c.endswith("_weight")],
        key=lambda x: int(x.split("_")[1]),
    )
    lineage_pt = reference[lineage_pt_cols].to_numpy(float)
    lineage_weight = reference[lineage_weight_cols].to_numpy(float)
    normalized_lineage_pt = np.full_like(lineage_pt, np.nan, dtype=float)
    for column in range(lineage_pt.shape[1]):
        finite_column = np.isfinite(lineage_pt[:, column])
        low = np.nanmin(lineage_pt[finite_column, column])
        high = np.nanmax(lineage_pt[finite_column, column])
        normalized_lineage_pt[finite_column, column] = (
            lineage_pt[finite_column, column] - low
        ) / max(high - low, 1e-12)
    assigned_zero_based = np.nanargmax(np.where(np.isfinite(lineage_weight), lineage_weight, -np.inf), axis=1)
    ref_pt = normalized_lineage_pt[np.arange(len(reference)), assigned_zero_based]
    valid = np.isfinite(ref_pt)

    model = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
    model.fit(ref_x[valid], ref_pt[valid])
    all_x = cells[scvi_cols].to_numpy(np.float32)
    projected = np.clip(model.predict(all_x), 0, 1)
    projected[reference_indices[valid]] = ref_pt[valid]
    projected[cells["state"].eq("CD8 Trm").to_numpy()] = np.nan
    cells["pseudotime"] = projected
    cells["trajectory_reference"] = False
    cells.loc[reference_indices, "trajectory_reference"] = True

    assigned = pd.Series(reference["assigned_lineage"].to_numpy(), index=reference_indices)
    cells["assigned_lineage"] = np.nan
    cells.loc[assigned.index, "assigned_lineage"] = assigned.values

    graph = pd.read_csv(
        OUT / "Table_TJ2_cd8_cells_with_pseudotime.csv.gz",
        usecols=["cell", "pseudotime"],
        low_memory=False,
    ).rename(columns={"pseudotime": "graph_geodesic_pseudotime"})
    cells = cells.merge(graph, on="cell", how="left", validate="one_to_one")

    primary_cells = cells[cells["state"].ne("CD8 Trm")].copy()
    states = trajectory_tools.summarize_states(primary_cells)
    clones, clone_bins, clone_status, clone_tests = trajectory_tools.analyze_clones(primary_cells)
    participant_bins, kinetics_display, kinetics_tests = trajectory_tools.program_kinetics(primary_cells)

    paired_mask = cells["paired_alpha_beta"].astype(bool) & cells["state"].ne("CD8 Trm")
    state_compare = (
        primary_cells.groupby("state", as_index=False)
        .agg(
            slingshot_median=("pseudotime", "median"),
            graph_median=("graph_geodesic_pseudotime", "median"),
        )
    )
    graph_clone_tests = pd.read_csv(OUT / "Table_TJ9_clone_trajectory_tests.csv")
    clone_compare = clone_tests.merge(
        graph_clone_tests[["analysis", "estimate", "ci_low", "ci_high", "FDR"]],
        on="analysis",
        suffixes=("_slingshot", "_graph"),
    )

    qc = pd.DataFrame(
        [
            {
                "metric": "Slingshot versus graph-geodesic pseudotime Spearman rho, all conventional CD8 cells",
                "value": spearmanr(primary_cells["pseudotime"], primary_cells["graph_geodesic_pseudotime"], nan_policy="omit").statistic,
            },
            {
                "metric": "Slingshot versus graph-geodesic pseudotime Spearman rho, paired alpha-beta cells",
                "value": spearmanr(
                    cells.loc[paired_mask, "pseudotime"],
                    cells.loc[paired_mask, "graph_geodesic_pseudotime"],
                    nan_policy="omit",
                ).statistic,
            },
            {
                "metric": "State-median Slingshot versus graph-geodesic Spearman rho",
                "value": spearmanr(state_compare["slingshot_median"], state_compare["graph_median"]).statistic,
            },
            {"metric": "Cells with Slingshot pseudotime", "value": int(np.isfinite(cells["pseudotime"]).sum())},
            {"metric": "Slingshot reference cells", "value": int(cells["trajectory_reference"].sum())},
            {"metric": "Trajectory method used for integrated Figure 2", "value": "Lineage-assigned Slingshot pseudotime; Trm retained as a branch-specific sensitivity state"},
        ]
    )

    cells.to_csv(OUT / "Table_ST2_cd8_cells_with_slingshot_pseudotime.csv.gz", index=False)
    states.to_csv(OUT / "Table_ST3_slingshot_state_pseudotime_summary.csv", index=False)
    clones.to_csv(OUT / "Table_ST4_slingshot_clone_trajectory.csv", index=False)
    clone_bins.to_csv(OUT / "Table_ST5_slingshot_participant_clone_bins.csv", index=False)
    clone_status.to_csv(OUT / "Table_ST6_slingshot_expanded_singleton.csv", index=False)
    clone_tests.to_csv(OUT / "Table_ST7_slingshot_clone_tests.csv", index=False)
    participant_bins.to_csv(OUT / "Table_ST8_slingshot_participant_program_kinetics.csv", index=False)
    kinetics_display.to_csv(OUT / "Table_ST9_slingshot_program_kinetics_display.csv", index=False)
    kinetics_tests.to_csv(OUT / "Table_ST10_slingshot_program_kinetics_tests.csv", index=False)
    qc.to_csv(OUT / "Table_ST11_slingshot_concordance_qc.csv", index=False)
    state_compare.to_csv(OUT / "Table_ST12_method_state_concordance.csv", index=False)
    clone_compare.to_csv(OUT / "Table_ST13_method_clone_effect_concordance.csv", index=False)

    print(qc.to_string(index=False))
    print(states[["state", "median_pseudotime", "q1_pseudotime", "q3_pseudotime"]].to_string(index=False))
    print(clone_tests.to_string(index=False))
    print(kinetics_tests.to_string(index=False))


if __name__ == "__main__":
    main()

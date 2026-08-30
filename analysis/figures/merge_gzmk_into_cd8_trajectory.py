from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "High Impact Additional Analyses" / "CD8 TCR Trajectory"
INPUT = TRAJ / "Table_TJ1_cd8_trajectory_input.csv.gz"
PSEUDOTIME = TRAJ / "Table_TJ2_cd8_cells_with_pseudotime.csv.gz"
MODULE = "GZMK_inflammatory_memory"


def main() -> None:
    scores = pd.read_csv(INPUT, usecols=["cell", MODULE])
    cells = pd.read_csv(PSEUDOTIME, low_memory=False)
    if MODULE in cells.columns:
        cells = cells.drop(columns=[MODULE])
    cells = cells.merge(scores, on="cell", how="left", validate="one_to_one")
    if cells[MODULE].isna().any():
        raise RuntimeError(f"Missing {MODULE} values after cell-level merge")
    cells.to_csv(PSEUDOTIME, index=False)
    print(f"Merged {MODULE} for {len(cells):,} cells into {PSEUDOTIME}")


if __name__ == "__main__":
    main()

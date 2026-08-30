#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, wilcoxon


ROOT = Path(__file__).resolve().parent
REVISION = ROOT / "High Impact Additional Analyses" / "Figure 3 High Impact Revision"
SOURCE = ROOT / "Cell Press Redrawn Figure Set" / "Source Data"
CELLS = REVISION / "Table_F3R6_paired_gamma_delta_cells.csv"

MODULES = [
    "Th1_Tc1_inflammatory",
    "Effector_cytotoxicity",
    "EOMES_ZEB2_inflammatory_CD8_TRM_like",
    "Tissue_resident_mucosal_retention",
    "Gut_homing_intestinal_trafficking",
]

CONTRASTS = [
    {
        "analysis": "V-pair architecture",
        "group_column": "receptor_group",
        "focal": "TRGV9-TRDV2",
        "reference": "Other paired gamma-delta",
        "contrast": "TRGV9-TRDV2 minus other paired gamma-delta",
    },
    {
        "analysis": "Clonotype expansion",
        "group_column": "expansion_class",
        "focal": "Expanded",
        "reference": "Singleton",
        "contrast": "Expanded minus singleton paired gamma-delta",
    },
]
N_BOOTSTRAP = 10_000
N_SIGNFLIP = 200_000
N_PAIRING_PERMUTATIONS = 10_000
SEED = 20260827


def bh_fdr(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if not valid.any():
        return pd.Series(q, index=values.index)
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    q[valid] = restored
    return pd.Series(q, index=values.index)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, float)
    draws = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def signflip_mean_p(values: np.ndarray, rng: np.random.Generator) -> tuple[float, int, bool]:
    """Two-sided participant-level sign-flip test for the mean paired contrast."""
    values = np.asarray(values, float)
    observed = abs(float(values.mean()))
    n_values = len(values)
    tolerance = 1e-15
    if n_values <= 20:
        n_draws = 1 << n_values
        extreme = 0
        bit_positions = np.arange(n_values, dtype=np.uint64)
        for start in range(0, n_draws, 65_536):
            stop = min(start + 65_536, n_draws)
            integers = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where(((integers >> bit_positions) & 1) == 1, 1.0, -1.0)
            null_statistics = np.abs(signs @ values / n_values)
            extreme += int(np.count_nonzero(null_statistics >= observed - tolerance))
        return extreme / n_draws, n_draws, True

    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_SIGNFLIP, n_values), replace=True)
    null_statistics = np.abs(signs @ values / n_values)
    p_value = (1 + int(np.count_nonzero(null_statistics >= observed - tolerance))) / (N_SIGNFLIP + 1)
    return p_value, N_SIGNFLIP, False


def pairing_enrichment(
    cells: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant-normalized Vgamma-Vdelta enrichment with within-participant permutations."""
    working = cells[["SampleID", "Diagnosis", "gamma_v", "delta_v"]].dropna().copy()
    participant_order = working["SampleID"].drop_duplicates().tolist()
    gamma_order = sorted(working["gamma_v"].unique())
    delta_order = sorted(working["delta_v"].unique())
    participant_index = {participant: index for index, participant in enumerate(participant_order)}
    gamma_index = {gene: index for index, gene in enumerate(gamma_order)}
    delta_index = {gene: index for index, gene in enumerate(delta_order)}
    n_participants = len(participant_order)
    n_gamma = len(gamma_order)
    n_delta = len(delta_order)
    n_pairs = n_gamma * n_delta

    participant_codes = working["SampleID"].map(participant_index).to_numpy(int)
    gamma_codes = working["gamma_v"].map(gamma_index).to_numpy(int)
    delta_codes = working["delta_v"].map(delta_index).to_numpy(int)
    participant_groups = [np.flatnonzero(participant_codes == index) for index in range(n_participants)]
    participant_totals = np.bincount(participant_codes, minlength=n_participants)
    cell_weights = 1.0 / participant_totals[participant_codes] / n_participants

    observed_pair_codes = gamma_codes * n_delta + delta_codes
    observed_fraction = np.bincount(observed_pair_codes, weights=cell_weights, minlength=n_pairs)
    observed_cells = np.bincount(observed_pair_codes, minlength=n_pairs)

    gamma_counts = np.zeros((n_participants, n_gamma), dtype=float)
    delta_counts = np.zeros((n_participants, n_delta), dtype=float)
    np.add.at(gamma_counts, (participant_codes, gamma_codes), 1)
    np.add.at(delta_counts, (participant_codes, delta_codes), 1)
    gamma_fraction = gamma_counts / participant_totals[:, None]
    delta_fraction = delta_counts / participant_totals[:, None]
    expected_fraction = np.einsum("pg,pd->gd", gamma_fraction, delta_fraction).reshape(-1) / n_participants

    deviation = np.abs(observed_fraction - expected_fraction)
    extreme = np.zeros(n_pairs, dtype=int)
    permuted_delta = delta_codes.copy()
    for _ in range(N_PAIRING_PERMUTATIONS):
        for indices in participant_groups:
            permuted_delta[indices] = rng.permutation(delta_codes[indices])
        permuted_codes = gamma_codes * n_delta + permuted_delta
        permuted_fraction = np.bincount(permuted_codes, weights=cell_weights, minlength=n_pairs)
        extreme += np.abs(permuted_fraction - expected_fraction) >= deviation - 1e-15
    permutation_p = (extreme + 1) / (N_PAIRING_PERMUTATIONS + 1)

    carrier_counts = (
        working.drop_duplicates(["SampleID", "gamma_v", "delta_v"])
        .groupby(["gamma_v", "delta_v"], observed=True)
        .size()
        .to_dict()
    )
    rows = []
    for gamma_gene, gamma_code in gamma_index.items():
        for delta_gene, delta_code in delta_index.items():
            pair_code = gamma_code * n_delta + delta_code
            observed = float(observed_fraction[pair_code])
            expected = float(expected_fraction[pair_code])
            rows.append(
                {
                    "gamma_v": gamma_gene,
                    "delta_v": delta_gene,
                    "n_cells": int(observed_cells[pair_code]),
                    "n_participant_carriers": int(carrier_counts.get((gamma_gene, delta_gene), 0)),
                    "n_participants": n_participants,
                    "observed_mean_participant_fraction": observed,
                    "expected_mean_participant_fraction": expected,
                    "observed_to_expected_ratio": observed / expected if expected > 0 else np.nan,
                    "log2_observed_expected": np.log2(observed / expected) if observed > 0 and expected > 0 else np.nan,
                    "permutation_p": float(permutation_p[pair_code]),
                    "n_permutations": N_PAIRING_PERMUTATIONS,
                    "permutation_scheme": "TRD V labels shuffled among cells within participant",
                    "batch_used": False,
                }
            )
    enrichment = pd.DataFrame(rows)
    enrichment["fdr"] = bh_fdr(enrichment["permutation_p"])
    enrichment = enrichment.sort_values(
        ["n_participant_carriers", "n_cells", "gamma_v", "delta_v"], ascending=[False, False, True, True]
    )

    participant_fraction = (
        working.assign(is_v9v2=(working.gamma_v.eq("TRGV9") & working.delta_v.eq("TRDV2")).astype(int))
        .groupby(["SampleID", "Diagnosis"], observed=True)
        .agg(v9v2_fraction=("is_v9v2", "mean"), n_gamma_delta_cells=("is_v9v2", "size"))
        .reset_index()
    )
    diagnosis_groups = [group.v9v2_fraction.to_numpy(float) for _, group in participant_fraction.groupby("Diagnosis")]
    diagnosis_p = float(kruskal(*diagnosis_groups).pvalue) if len(diagnosis_groups) >= 2 else np.nan
    participant_fraction["diagnosis_kruskal_wallis_p"] = diagnosis_p
    participant_fraction["batch_used"] = False
    return enrichment, participant_fraction


def main() -> None:
    cells = pd.read_csv(CELLS, low_memory=False)
    rng = np.random.default_rng(SEED)
    effect_rows: list[dict] = []
    delta_rows: list[dict] = []

    for contrast_spec in CONTRASTS:
        group_column = contrast_spec["group_column"]
        focal = contrast_spec["focal"]
        reference = contrast_spec["reference"]
        for module in MODULES:
            values = pd.to_numeric(cells[module], errors="coerce")
            scale = float(values.std(ddof=1))
            if not np.isfinite(scale) or scale <= 0:
                continue

            working = cells[["SampleID", "Diagnosis", "state", group_column]].copy()
            working["standardized_score"] = values / scale
            state_means = (
                working.dropna(subset=["standardized_score"])
                .groupby(["SampleID", "Diagnosis", "state", group_column], observed=True)["standardized_score"]
                .mean()
                .unstack(group_column)
            )
            if focal not in state_means or reference not in state_means:
                continue
            matched = state_means.dropna(subset=[focal, reference]).copy()
            matched["delta"] = matched[focal] - matched[reference]
            participant = matched.groupby(level=["SampleID", "Diagnosis"], observed=True)["delta"].mean().reset_index()
            participant["analysis"] = contrast_spec["analysis"]
            participant["contrast"] = contrast_spec["contrast"]
            participant["module"] = module
            participant["n_matched_states"] = participant["SampleID"].map(
                matched.reset_index().groupby("SampleID", observed=True).size()
            )
            delta_rows.extend(participant.to_dict("records"))

            deltas = participant["delta"].to_numpy(float)
            ci_low, ci_high = bootstrap_mean_ci(deltas, rng)
            if len(deltas) >= 2 and np.any(np.abs(deltas) > 0):
                p_value = float(wilcoxon(deltas, alternative="two-sided", zero_method="wilcox").pvalue)
            else:
                p_value = np.nan
            signflip_p, signflip_draws, signflip_exact = signflip_mean_p(deltas, rng)
            effect_rows.append(
                {
                    "analysis": contrast_spec["analysis"],
                    "module": module,
                    "contrast": contrast_spec["contrast"],
                    "effect": float(np.mean(deltas)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "median_participant_delta": float(np.median(deltas)),
                    "p_value_wilcoxon": p_value,
                    "p_value_signflip_mean": signflip_p,
                    "signflip_draws": signflip_draws,
                    "signflip_exact": signflip_exact,
                    "n_participants": int(participant["SampleID"].nunique()),
                    "n_participant_state_strata": int(len(matched)),
                    "standardization": "Module-specific SD across all productive paired gamma-delta cells",
                    "matching": "Within participant and observed transcriptomic state; matched-state deltas equally weighted",
                    "batch_used": False,
                }
            )

    effects = pd.DataFrame(effect_rows)
    effects["fdr_wilcoxon"] = effects.groupby("analysis", group_keys=False)["p_value_wilcoxon"].apply(bh_fdr)
    effects["fdr"] = effects.groupby("analysis", group_keys=False)["p_value_signflip_mean"].apply(bh_fdr)
    effects["direction"] = np.where(effects["effect"] >= 0, "Focal group higher", "Reference group higher")
    deltas = pd.DataFrame(delta_rows)

    effects.to_csv(REVISION / "Table_F3R12_gamma_delta_program_effects.csv", index=False)
    deltas.to_csv(REVISION / "Table_F3R13_gamma_delta_participant_state_matched_deltas.csv", index=False)
    effects.to_csv(SOURCE / "Figure_3_R12_gamma_delta_program_effects.csv", index=False)
    deltas.to_csv(SOURCE / "Figure_3_R13_gamma_delta_participant_state_matched_deltas.csv", index=False)

    # Participant-normalized gamma-delta composition for the revised main figure.
    participants = cells[["SampleID"]].drop_duplicates().assign(_key=1)
    vpairs = cells[["gamma_v", "delta_v"]].drop_duplicates().assign(_key=1)
    participant_vpair = participants.merge(vpairs, on="_key").drop(columns="_key")
    vpair_counts = cells.groupby(["SampleID", "gamma_v", "delta_v"], observed=True).size().rename("n_cells").reset_index()
    participant_totals = cells.groupby("SampleID", observed=True).size().rename("participant_gamma_delta_cells").reset_index()
    participant_vpair = participant_vpair.merge(vpair_counts, how="left", on=["SampleID", "gamma_v", "delta_v"])
    participant_vpair = participant_vpair.merge(participant_totals, on="SampleID")
    participant_vpair["n_cells"] = participant_vpair["n_cells"].fillna(0).astype(int)
    participant_vpair["participant_fraction"] = participant_vpair["n_cells"] / participant_vpair["participant_gamma_delta_cells"]
    vpair_summary = (
        participant_vpair.groupby(["gamma_v", "delta_v"], observed=True)
        .agg(
            n_cells=("n_cells", "sum"),
            n_participant_carriers=("n_cells", lambda value: int((value > 0).sum())),
            mean_participant_fraction=("participant_fraction", "mean"),
            median_participant_fraction=("participant_fraction", "median"),
            median_fraction_among_carriers=("participant_fraction", lambda value: float(value[value > 0].median()) if (value > 0).any() else 0.0),
            n_participants=("SampleID", "nunique"),
        )
        .reset_index()
        .sort_values(["n_participant_carriers", "n_cells"], ascending=False)
    )

    group_carriers = cells[["SampleID", "receptor_group"]].drop_duplicates().assign(_key=1)
    states = cells[["state"]].drop_duplicates().assign(_key=1)
    participant_state = group_carriers.merge(states, on="_key").drop(columns="_key")
    state_counts = cells.groupby(["SampleID", "receptor_group", "state"], observed=True).size().rename("n_cells").reset_index()
    group_totals = cells.groupby(["SampleID", "receptor_group"], observed=True).size().rename("participant_group_cells").reset_index()
    participant_state = participant_state.merge(state_counts, how="left", on=["SampleID", "receptor_group", "state"])
    participant_state = participant_state.merge(group_totals, on=["SampleID", "receptor_group"])
    participant_state["n_cells"] = participant_state["n_cells"].fillna(0).astype(int)
    participant_state["participant_fraction"] = participant_state["n_cells"] / participant_state["participant_group_cells"]
    state_summary = (
        participant_state.groupby(["receptor_group", "state"], observed=True)
        .agg(
            n_cells=("n_cells", "sum"),
            n_participant_carriers=("n_cells", lambda value: int((value > 0).sum())),
            n_participants_with_receptor_group=("SampleID", "nunique"),
            mean_participant_fraction=("participant_fraction", "mean"),
            median_participant_fraction=("participant_fraction", "median"),
        )
        .reset_index()
        .sort_values(["receptor_group", "mean_participant_fraction"], ascending=[True, False])
    )

    enrichment, participant_v9v2 = pairing_enrichment(cells, rng)

    vpair_summary.to_csv(REVISION / "Table_F3R16_gamma_delta_participant_normalized_V_pairs.csv", index=False)
    state_summary.to_csv(REVISION / "Table_F3R17_gamma_delta_participant_normalized_states.csv", index=False)
    vpair_summary.to_csv(SOURCE / "Figure_3_R16_gamma_delta_participant_normalized_V_pairs.csv", index=False)
    state_summary.to_csv(SOURCE / "Figure_3_R17_gamma_delta_participant_normalized_states.csv", index=False)
    enrichment.to_csv(REVISION / "Table_F3R25_gamma_delta_pairing_enrichment.csv", index=False)
    participant_v9v2.to_csv(REVISION / "Table_F3R26_gamma_delta_participant_V9V2_fraction.csv", index=False)
    enrichment.to_csv(SOURCE / "Figure_3_R25_gamma_delta_pairing_enrichment.csv", index=False)
    participant_v9v2.to_csv(SOURCE / "Figure_3_R26_gamma_delta_participant_V9V2_fraction.csv", index=False)
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()

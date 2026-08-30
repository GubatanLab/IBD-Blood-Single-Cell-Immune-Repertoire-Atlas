#!/usr/bin/env python3
"""Calculate per-patient BCR SHM rates from Change-O aligned AIRR output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


VALID = set("ACGT")


def mutation_rate(sequence_alignment: str, germline_alignment: str) -> float | None:
    seq = (sequence_alignment or "").upper()
    germ = (germline_alignment or "").upper()
    denom = 0
    muts = 0
    for s, g in zip(seq, germ):
        if s in VALID and g in VALID:
            denom += 1
            if s != g:
                muts += 1
    if denom == 0:
        return None
    return muts / denom


def row_shm_rate(row: dict[str, str], germline_col: str) -> tuple[float | None, str]:
    try:
        v_identity = float(row.get("v_identity") or "")
    except ValueError:
        v_identity = None
    if v_identity is not None:
        if v_identity > 1:
            v_identity = v_identity / 100
        if 0 <= v_identity <= 1:
            return 1 - v_identity, "changeo_v_identity"
    return (
        mutation_rate(row.get("sequence_alignment", ""), row.get(germline_col, "")),
        f"{germline_col}_direct_mismatch_frequency",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aligned-airr",
        type=Path,
        default=Path("chain_bcr_immuneml/changeo_alignment/bcr_changeo_germline_airr_with_metadata.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chain_bcr_immuneml/bcr_shm_rates_by_patientID.csv"),
    )
    args = parser.parse_args()

    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "SampleID": set(),
            "Diagnosis1": set(),
            "n_sequences": 0,
            "n_weighted_sequences": 0.0,
            "n_IGH": 0,
            "n_IGK": 0,
            "n_IGL": 0,
            "rates": [],
            "weighted_sum": 0.0,
            "weight_total": 0.0,
        }
    )

    with args.aligned_airr.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        germline_col = (
            "germline_alignment_d_mask"
            if "germline_alignment_d_mask" in (reader.fieldnames or [])
            else "germline_alignment"
        )
        for row in reader:
            patient = row.get("PatientID", "")
            if not patient:
                continue
            rate, rate_source = row_shm_rate(row, germline_col)
            if rate is None:
                continue
            try:
                weight = float(row.get("duplicate_count") or 1)
            except ValueError:
                weight = 1.0
            if weight < 1:
                weight = 1.0
            locus = row.get("input_locus") or row.get("locus") or ""
            g = groups[patient]
            g["SampleID"].add(row.get("SampleID", ""))
            g["Diagnosis1"].add(row.get("Diagnosis1", ""))
            g["n_sequences"] += 1
            g["n_weighted_sequences"] += weight
            if locus in {"IGH", "IGK", "IGL"}:
                g[f"n_{locus}"] += 1
            g["rates"].append(rate)
            g["weighted_sum"] += rate * weight
            g["weight_total"] += weight

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "PatientID",
            "SampleID",
            "Diagnosis1",
            "n_sequences",
            "n_weighted_sequences",
            "n_IGH",
            "n_IGK",
            "n_IGL",
            "shm_rate",
            "shm_rate_weighted",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for patient in sorted(groups):
            g = groups[patient]
            rates = g["rates"]
            writer.writerow(
                {
                    "PatientID": patient,
                    "SampleID": ";".join(sorted(x for x in g["SampleID"] if x)),
                    "Diagnosis1": ";".join(sorted(x for x in g["Diagnosis1"] if x)),
                    "n_sequences": int(g["n_sequences"]),
                    "n_weighted_sequences": g["n_weighted_sequences"],
                    "n_IGH": int(g["n_IGH"]),
                    "n_IGK": int(g["n_IGK"]),
                    "n_IGL": int(g["n_IGL"]),
                    "shm_rate": sum(rates) / len(rates) if rates else "",
                    "shm_rate_weighted": (
                        g["weighted_sum"] / g["weight_total"] if g["weight_total"] else ""
                    ),
                    "status": (
                        "calculated_from_changeo_aligned_airr_"
                        f"primary_rate_source={rate_source}"
                    ),
                }
            )
    print(f"Wrote {args.output}")
    print(f"Patients: {len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

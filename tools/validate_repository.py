#!/usr/bin/env python3
"""Audit the shareable repository before commit or release."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".r", ".R", ".yml", ".yaml", ".json", ".md", ".txt", ".tsv", ".csv"}
FORBIDDEN_SUFFIXES = {".rds", ".h5ad", ".loom", ".bam", ".fastq", ".pickle", ".pkl", ".pt", ".pth"}
SECRET_RE = re.compile(r"(?:github_pat_|ghp_[A-Za-z0-9]{20,}|api[_-]?key\s*[:=]|password\s*[:=]|secret\s*[:=])", re.I)
USER_PATH_RE = re.compile(r"C:[/\\]Users[/\\]johng", re.I)
INTERNAL_ID_HEADER_RE = re.compile(r"(?:sampleid|sample_id|participant_id|patient_id|donor_id|subject_id)", re.I)


def main() -> int:
    problems: list[str] = []

    for index in range(1, 8):
        path = ROOT / "figures" / "main" / f"Figure_{index}.pdf"
        if not path.exists():
            problems.append(f"missing main figure: {path.relative_to(ROOT)}")
    for index in range(1, 11):
        path = ROOT / "figures" / "supplementary" / f"Figure_S{index}.pdf"
        if not path.exists():
            problems.append(f"missing supplementary figure: {path.relative_to(ROOT)}")

    tcr_specs = list((ROOT / "immuneml" / "tcr").rglob("*.yml")) + list((ROOT / "immuneml" / "tcr").rglob("*.yaml"))
    bcr_specs = list((ROOT / "immuneml" / "bcr").glob("*.yaml"))
    if len(tcr_specs) < 56:
        problems.append(f"expected at least 56 TCR immuneML specifications; found {len(tcr_specs)}")
    if len(bcr_specs) != 3:
        problems.append(f"expected 3 BCR immuneML specifications; found {len(bcr_specs)}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden raw/model artifact: {path.relative_to(ROOT)}")
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_RE.search(text):
            problems.append(f"possible credential in {path.relative_to(ROOT)}")
        if USER_PATH_RE.search(text):
            problems.append(f"machine-specific user path in {path.relative_to(ROOT)}")

    internal_source = ROOT / "source_data" / "main"
    for path in internal_source.glob("*.csv"):
        header = path.open(encoding="utf-8", errors="replace").readline()
        if INTERNAL_ID_HEADER_RE.search(header):
            problems.append(f"internal identifier column in {path.relative_to(ROOT)}")

    if problems:
        print("Repository audit failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Repository audit passed.")
    print(f"TCR immuneML specifications: {len(tcr_specs)}")
    print(f"BCR immuneML specifications: {len(bcr_specs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

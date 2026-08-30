#!/usr/bin/env python3
"""Create checksums and structural QA records for the ranked analyses."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "High Impact Additional Analyses" / "Literature Guided Ranked Analyses"
SCRIPTS = [
    ROOT / "scripts" / "export_ranked_helper_axis.R",
    ROOT / "scripts" / "export_ranked_bcr_annotations.R",
    ROOT / "scripts" / "run_ranked_helper_external.py",
    ROOT / "scripts" / "run_ranked_receptor_mapping.py",
    ROOT / "scripts" / "run_ranked_endotypes.py",
    Path(__file__),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


rows = []
for path in sorted([p for p in OUT.rglob("*") if p.is_file()] + SCRIPTS):
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    row = {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path),
           "extension": path.suffix.lower(), "rows": None, "columns": None,
           "pages": None, "width_px": None, "height_px": None, "qa_status": "PASS"}
    try:
        if path.suffix.lower() == ".csv":
            d = pd.read_csv(path)
            row["rows"], row["columns"] = d.shape
        elif path.suffix.lower() == ".pdf":
            row["pages"] = len(PdfReader(str(path)).pages)
        elif path.suffix.lower() == ".png":
            with Image.open(path) as im:
                row["width_px"], row["height_px"] = im.size
    except Exception as exc:
        row["qa_status"] = f"FAIL: {exc}"
    rows.append(row)

manifest = pd.DataFrame(rows)
manifest.to_csv(OUT / "analysis_manifest.csv", index=False)

required = [
    *(f"Figure_RA{i}" for i in range(1, 6)),
    "RANKED_ANALYSES_REPORT.md", "METHODS_AND_REPRODUCIBILITY.md", "FIGURE_LEGENDS.md",
]
checks = []
for token in required:
    matches = [p for p in OUT.iterdir() if token in p.name]
    checks.append((token, bool(matches), "; ".join(p.name for p in matches)))
all_csv_ok = all(x == "PASS" for x in manifest.loc[manifest.extension == ".csv", "qa_status"])
all_pdf_ok = all(x == "PASS" for x in manifest.loc[manifest.extension == ".pdf", "qa_status"])
all_png_ok = all(x == "PASS" for x in manifest.loc[manifest.extension == ".png", "qa_status"])

lines = [
    "# Structural QA report", "",
    f"- Python: {sys.version.split()[0]}",
    f"- Platform: {platform.platform()}",
    f"- CSV parse checks: {'PASS' if all_csv_ok else 'FAIL'}",
    f"- PDF parse checks: {'PASS' if all_pdf_ok else 'FAIL'}",
    f"- PNG dimension checks: {'PASS' if all_png_ok else 'FAIL'}",
    "", "## Required deliverables", "",
]
lines.extend(f"- {'PASS' if ok else 'FAIL'} — {token}: {files}" for token, ok, files in checks)
lines.extend(["", "## Scope", "",
              "This report verifies file presence and structural readability. Statistical interpretation and data limitations are documented in RANKED_ANALYSES_REPORT.md.", ""])
(OUT / "QA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

if not all(ok for _, ok, _ in checks) or not (all_csv_ok and all_pdf_ok and all_png_ok):
    raise SystemExit("QA failed; inspect analysis_manifest.csv and QA_REPORT.md")
print(f"QA passed for {len(manifest)} files")

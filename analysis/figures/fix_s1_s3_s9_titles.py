from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "S1 S3 S9 Title Corrections"
WORK = ROOT / "tmp" / "pdfs" / "s1-s3-s9-title-fixes"

sys.path.insert(0, str(ROOT / "scripts"))
import harmonize_supplementary_figure_set as harmonize  # noqa: E402
from fix_supplement_title_text_overlaps import LetterPatch, patch_vector_page  # noqa: E402


def merge(paths: list[Path], output: Path) -> None:
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(path)
        if len(reader.pages) != 1:
            raise ValueError(f"Expected one-page source: {path}")
        writer.add_page(reader.pages[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)


def build() -> None:
    harmonize.register_fonts()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    # S1 page 1 already has a complete A label and title. Page 2 has a source A
    # whose top extends beyond the original media box; redraw it as B lower down.
    s1_page1_source = ROOT / "Cell Press Redrawn Figure Set" / "Supplementary Figures" / "Figure_S9.pdf"
    s1_page2_source = ROOT / "Cell Press Redrawn Figure Set" / "Supplementary Figures" / "Figure_S11.pdf"
    s1_page2_patched = WORK / "S1_page2_marker_validation_B.pdf"
    patch_vector_page(
        s1_page2_source,
        s1_page2_patched,
        [LetterPatch(32.6, 0.0, "B", 12.0, width=13.0, height=14.0)],
    )
    s1_page1 = WORK / "S1_page1_ready.pdf"
    s1_page2 = WORK / "S1_page2_ready.pdf"
    harmonize.compose_vector_page(
        [s1_page1_source], s1_page1, 1, page_number=1, page_count=2
    )
    harmonize.compose_vector_page(
        [s1_page2_patched], s1_page2, 1, page_number=2, page_count=2
    )
    merge([s1_page1, s1_page2], OUTPUT / "Figure_S1.pdf")

    # S3 and S9 are retained as complete vector figures and inset below a new
    # header band so every A-D letter/title remains fully inside the media box.
    s3_source = (
        ROOT
        / "External Validation 20260829"
        / "results"
        / "ExternalValidation_Figure1_Longitudinal_PBMC.pdf"
    )
    s9_source = (
        ROOT
        / "External Validation 20260829"
        / "results"
        / "ExternalValidation_Figure2_Mucosal_Clonotypes.pdf"
    )
    harmonize.compose_vector_page([s3_source], OUTPUT / "Figure_S3.pdf", 3)
    harmonize.compose_vector_page([s9_source], OUTPUT / "Figure_S9.pdf", 9)

    expected = {"Figure_S1.pdf": 2, "Figure_S3.pdf": 1, "Figure_S9.pdf": 1}
    for name, pages in expected.items():
        path = OUTPUT / name
        reader = PdfReader(path)
        if len(reader.pages) != pages:
            raise RuntimeError(f"{name}: expected {pages} pages, found {len(reader.pages)}")
        if path.stat().st_size < 10_000:
            raise RuntimeError(f"{name}: unexpectedly small output")
        print(path)


if __name__ == "__main__":
    build()

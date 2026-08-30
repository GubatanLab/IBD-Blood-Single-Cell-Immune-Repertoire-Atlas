from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "output" / "pdf" / "Immunity Revised Supplementary Figures S1-S10"
OUTPUT = ROOT / "output" / "pdf" / "Supplementary Figures Title and Overlap Corrected"
WORK = ROOT / "tmp" / "pdfs" / "supplement-title-overlap-fixes"

sys.path.insert(0, str(ROOT / "scripts"))
import harmonize_supplementary_figure_set as harmonize  # noqa: E402


@dataclass(frozen=True)
class LetterPatch:
    x: float
    top: float
    letter: str
    size: float
    width: float = 14.0
    height: float = 15.0


def patch_vector_page(
    source: Path,
    output: Path,
    letters: list[LetterPatch],
    replace_top_title: str | None = None,
) -> None:
    reader = PdfReader(source)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected a one-page source: {source}")
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(width, height), pageCompression=1)
    if replace_top_title:
        pdf.setFillColorRGB(1, 1, 1)
        pdf.rect(0, height - 14.5, width, 14.5, fill=1, stroke=0)
        pdf.setFillColorRGB(0.10, 0.10, 0.10)
        pdf.setFont(harmonize.BOLD_FONT, 9.2)
        pdf.drawCentredString(width / 2.0, height - 10.9, replace_top_title)

    for patch in letters:
        pdf.setFillColorRGB(1, 1, 1)
        pdf.rect(
            patch.x - 2.0,
            height - patch.top - patch.height,
            patch.width,
            patch.height,
            fill=1,
            stroke=0,
        )
        pdf.setFillColorRGB(0.08, 0.08, 0.08)
        pdf.setFont(harmonize.BOLD_FONT, patch.size)
        pdf.drawString(patch.x, height - patch.top - patch.size * 0.86, patch.letter)
    pdf.save()
    stream.seek(0)
    page.merge_page(PdfReader(stream).pages[0])

    writer = PdfWriter()
    writer.add_page(page)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)


def merge_pages(paths: list[tuple[Path, list[int] | None]], output: Path) -> None:
    writer = PdfWriter()
    for path, requested_pages in paths:
        reader = PdfReader(path)
        indices = requested_pages if requested_pages is not None else list(range(len(reader.pages)))
        for index in indices:
            writer.add_page(reader.pages[index])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)


def refresh_shared_headers(path: Path, figure_number: int) -> None:
    """Rewrite the header after page assembly so page numbering is accurate."""
    reader = PdfReader(path)
    writer = PdfWriter()
    total = len(reader.pages)
    for page_number, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        page.merge_page(
            harmonize.header_overlay(width, height, figure_number, page_number, total).pages[0]
        )
        writer.add_page(page)
    with path.open("wb") as handle:
        writer.write(handle)


def build() -> None:
    harmonize.register_fonts()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    # Start from the visually audited canonical set and replace only affected pages.
    for figure_number in range(1, 11):
        shutil.copy2(CURRENT / f"Figure_S{figure_number}.pdf", OUTPUT / f"Figure_S{figure_number}.pdf")

    # S2 page 1: retain the complete panel content while adding enough left
    # margin for the vertical cell-composition label.
    s2_raw = WORK / "Figure_S2_page1_raw.pdf"
    s2_ready = WORK / "Figure_S2_page1_ready.pdf"
    if not s2_raw.exists():
        raise FileNotFoundError(f"Generate the corrected S2 page first: {s2_raw}")
    harmonize.harmonize(s2_raw, s2_ready, 2)
    merge_pages(
        [(s2_ready, [0]), (CURRENT / "Figure_S2.pdf", [1, 2, 3])],
        OUTPUT / "Figure_S2.pdf",
    )

    # S4 page 3: separate section headings from the first row of subplot titles.
    s4_page1_raw = WORK / "Figure_S4_page1_raw.pdf"
    s4_page1_ready = WORK / "Figure_S4_page1_ready.pdf"
    if not s4_page1_raw.exists():
        raise FileNotFoundError(f"Generate the corrected S4 page first: {s4_page1_raw}")
    harmonize.harmonize(s4_page1_raw, s4_page1_ready, 4)
    s4_raw = WORK / "Figure_S4_page3_raw.pdf"
    s4_ready = WORK / "Figure_S4_page3_ready.pdf"
    if not s4_raw.exists():
        raise FileNotFoundError(f"Generate the corrected S4 page first: {s4_raw}")
    harmonize.harmonize(s4_raw, s4_ready, 4)
    merge_pages(
        [(s4_page1_ready, [0]), (CURRENT / "Figure_S4.pdf", [1]), (s4_ready, [0])],
        OUTPUT / "Figure_S4.pdf",
    )

    # S5: rebuilt separately before this assembly; compose its vector source under
    # the shared header so the former bar-overlaid explanatory note is absent.
    s5_source = (
        ROOT
        / "High Impact Additional Analyses"
        / "Requested Priority Analyses Figures"
        / "Figure_S_BCR_germline_lineages.pdf"
    )
    harmonize.compose_vector_page([s5_source], OUTPUT / "Figure_S5.pdf", 5)

    # S6 page 1: reserve a dedicated right-side gutter for q-value annotations.
    s6_source = (
        ROOT
        / "High Impact Additional Analyses"
        / "BCR Expansion Maturation 20260829"
        / "Figure_BEM2_lineage_maturation_preview.pdf"
    )
    if not s6_source.exists():
        raise FileNotFoundError(f"Generate the corrected S6 source first: {s6_source}")
    s6_patched = WORK / "Figure_S6_page1_vector_patched.pdf"
    patch_vector_page(
        s6_source,
        s6_patched,
        [LetterPatch(3.0, 2.0, "A", 11.3, width=15.0, height=15.0)],
    )
    s6_page1 = WORK / "Figure_S6_page1_ready.pdf"
    harmonize.compose_vector_page([s6_patched], s6_page1, 6)
    merge_pages(
        [(s6_page1, [0]), (CURRENT / "Figure_S6.pdf", [1])],
        OUTPUT / "Figure_S6.pdf",
    )

    # S7 page 2: remove the obsolete Figure 6 subtitle and rebuild F-K as clean
    # vector panel letters instead of layered raster replacements.
    s7_source = ROOT / "Cell Press Redrawn Figure Set" / "Supplementary Figures" / "Figure_S13.pdf"
    s7_patched = WORK / "Figure_S7_page2_vector_patched.pdf"
    patch_vector_page(
        s7_source,
        s7_patched,
        [
            LetterPatch(47.1, 14.4, "F", 11.5),
            LetterPatch(330.7, 14.4, "G", 11.5),
            LetterPatch(47.1, 243.7, "H", 11.5),
            LetterPatch(330.7, 243.7, "I", 11.5),
            LetterPatch(47.1, 480.3, "J", 11.5),
            LetterPatch(330.7, 480.3, "K", 11.5),
        ],
        replace_top_title="Robustness, sensitivity, and specificity analyses",
    )
    s7_page2 = WORK / "Figure_S7_page2_ready.pdf"
    harmonize.compose_vector_page([s7_patched], s7_page2, 7)
    merge_pages(
        [(CURRENT / "Figure_S7.pdf", [0]), (s7_page2, [0])],
        OUTPUT / "Figure_S7.pdf",
    )

    # S8 page 2: rebuild E-H directly from the vector analysis panel so no
    # fragments from the earlier A-D letter erasure remain.
    s8_source = ROOT / "output" / "pdf" / "Figure_H2_Th17_Treg_B_helper_coupling.pdf"
    s8_patched = WORK / "Figure_S8_page2_vector_patched.pdf"
    patch_vector_page(
        s8_source,
        s8_patched,
        [
            LetterPatch(70.6, 1.4, "E", 8.64, width=12.0, height=13.0),
            LetterPatch(364.3, 1.4, "F", 8.64, width=12.0, height=13.0),
            LetterPatch(70.6, 318.7, "G", 8.64, width=12.0, height=13.0),
            LetterPatch(364.3, 328.8, "H", 8.64, width=12.0, height=13.0),
        ],
    )
    s8_page2 = WORK / "Figure_S8_page2_ready.pdf"
    harmonize.compose_vector_page([s8_patched], s8_page2, 8)
    merge_pages(
        [
            (CURRENT / "Figure_S8.pdf", [0]),
            (s8_page2, [0]),
            (CURRENT / "Figure_S8.pdf", [2]),
        ],
        OUTPUT / "Figure_S8.pdf",
    )

    for figure_number in range(1, 11):
        refresh_shared_headers(OUTPUT / f"Figure_S{figure_number}.pdf", figure_number)

    # Parse every final PDF and assert that page counts match the canonical set.
    for figure_number in range(1, 11):
        source = CURRENT / f"Figure_S{figure_number}.pdf"
        final = OUTPUT / source.name
        source_count = len(PdfReader(source).pages)
        final_count = len(PdfReader(final).pages)
        if final_count != source_count:
            raise RuntimeError(f"{final.name}: expected {source_count} pages, found {final_count}")
        if final.stat().st_size < 10_000:
            raise RuntimeError(f"{final.name}: unexpectedly small output")
        print(final)


if __name__ == "__main__":
    build()

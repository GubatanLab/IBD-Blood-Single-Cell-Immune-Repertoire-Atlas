from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(r"C:/path/to/private-manuscript-workspace")
SOURCE = ROOT / "output" / "pdf" / "Pruned Reordered Supplementary Figures"
STAGING = ROOT / "output" / "pdf" / "Supplementary Figures Panel Labels Standardized Preview"
ARCHIVE = SOURCE / f"archive_before_panel_label_standardization_{date.today():%Y%m%d}"

pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))


@dataclass(frozen=True)
class LabelEdit:
    x: float
    top: float
    label: str
    size: float = 11.3
    erase: bool = True
    erase_width: float = 18.0
    erase_height: float = 18.0
    erase_dx: float = -3.0
    erase_dtop: float = -2.0


# Coordinates use PDF points measured from the page's top-left corner.
# Only pages with missing, duplicated, or non-sequential labels are modified.
EDITS: dict[int, dict[int, list[LabelEdit]]] = {
    1: {
        # Page 1 is A; the canonical-marker page becomes B.
        2: [LabelEdit(53.0, 28.0, "B", erase=False)],
    },
    2: {
        # Page 1 is A-C. Page 2 contains two imported composites whose source
        # labels restarted at A; they become D-J.
        2: [
            LabelEdit(280.8, 126.6, "D", size=10.8, erase_width=8, erase_dx=-1),
            LabelEdit(549.6, 126.6, "E", size=10.8, erase_width=8, erase_dx=-1),
            LabelEdit(280.8, 391.2, "F", size=10.8, erase_width=8, erase_dx=-1),
            LabelEdit(549.6, 391.2, "G", size=10.8, erase_width=8, erase_dx=-1),
            LabelEdit(155.4, 741.6, "H", size=10.8, erase_width=8, erase_dx=-1),
            LabelEdit(155.4, 928.8, "I", size=10.8, erase_width=8, erase_dx=-1),
            LabelEdit(155.4, 1117.8, "J", size=10.8, erase_width=8, erase_dx=-1),
        ],
        # Page 3 continues K-O.
        3: [
            LabelEdit(70.7, 30.5, "K"),
            LabelEdit(352.4, 30.5, "L"),
            LabelEdit(51.0, 211.2, "M"),
            LabelEdit(63.2, 426.4, "N"),
            LabelEdit(344.9, 426.4, "O"),
        ],
        # The independent motif panel completes the sequence.
        4: [LabelEdit(12.0, 31.0, "P", size=11.3, erase=False)],
    },
    4: {
        # Page 1 is A-C; the transcriptional-program panels become D-E.
        2: [
            LabelEdit(87.8, 34.8, "D", size=15.0, erase_width=22, erase_height=22),
            LabelEdit(87.8, 355.4, "E", size=15.0, erase_width=22, erase_height=22),
        ],
        # Class switching, isotype composition, and SHM become F-H.
        3: [
            LabelEdit(9.7, 47.8, "F", size=15.0, erase_width=22, erase_height=22),
            LabelEdit(212.7, 47.8, "G", size=15.0, erase_width=22, erase_height=22),
            LabelEdit(9.7, 424.1, "H", size=15.0, erase_width=22, erase_height=22),
        ],
    },
    6: {
        # Page 1 is one four-facet maturation analysis (A).
        1: [LabelEdit(10.0, 33.0, "A", size=11.3, erase=False)],
        # Page 2 continues B-E.
        2: [
            LabelEdit(23.3, 29.3, "B"),
            LabelEdit(286.2, 29.3, "C"),
            LabelEdit(23.3, 312.4, "D"),
            LabelEdit(286.2, 312.4, "E"),
        ],
    },
    7: {
        # Page 1 already contains A-D; its diagnosis-stratified composite is E.
        1: [LabelEdit(31.0, 714.0, "E", size=11.3, erase=False)],
        # Page 2 continues F-K.
        2: [
            LabelEdit(60.7, 47.6, "F", size=10.8),
            LabelEdit(327.3, 47.6, "G", size=10.8),
            LabelEdit(60.7, 263.1, "H", size=10.8),
            LabelEdit(327.3, 263.1, "I", size=10.8),
            LabelEdit(60.7, 485.5, "J", size=10.8),
            LabelEdit(327.3, 485.5, "K", size=10.8),
        ],
    },
    8: {
        # Page 1 is A-D. Page 2 continues E-H.
        2: [
            LabelEdit(81.5, 29.5, "E", size=10.0, erase_width=13, erase_height=15),
            LabelEdit(357.5, 29.5, "F", size=10.0, erase_width=13, erase_height=15),
            LabelEdit(
                81.5,
                328.5,
                "G",
                size=10.0,
                erase_width=11,
                erase_height=11,
                erase_dx=-1,
                erase_dtop=0,
            ),
            LabelEdit(357.5, 340.5, "H", size=10.0, erase_width=13, erase_height=15),
        ],
        3: [LabelEdit(10.0, 31.0, "I", size=11.3, erase=False)],
    },
    10: {
        # The lower row formerly restarted at A-B; it becomes C-D.
        1: [
            LabelEdit(209.4, 531.0, "C", size=11.3, erase_width=16, erase_height=18),
            LabelEdit(687.6, 531.0, "D", size=11.3, erase_width=16, erase_height=18),
        ],
    },
}


def overlay_page(width: float, height: float, edits: list[LabelEdit]):
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=(width, height))
    for edit in edits:
        if edit.erase:
            left = edit.x + edit.erase_dx
            top = edit.top + edit.erase_dtop
            c.setFillColorRGB(1, 1, 1)
            c.rect(
                left,
                height - top - edit.erase_height,
                edit.erase_width,
                edit.erase_height,
                fill=1,
                stroke=0,
            )
        c.setFillColorRGB(0.08, 0.08, 0.08)
        c.setFont("Arial-Bold", edit.size)
        baseline = height - edit.top - edit.size * 0.86
        c.drawString(edit.x, baseline, edit.label)
    c.save()
    stream.seek(0)
    return PdfReader(stream).pages[0]


def rewrite_pdf(source: Path, destination: Path, figure_number: int) -> None:
    reader = PdfReader(source)
    writer = PdfWriter()
    page_edits = EDITS.get(figure_number, {})
    for page_number, page in enumerate(reader.pages, start=1):
        if page_number in page_edits:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            page.merge_page(overlay_page(width, height, page_edits[page_number]))
        writer.add_page(page)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        writer.write(handle)

    # Reopen immediately to catch malformed output and preserve page counts.
    check = PdfReader(destination)
    if len(check.pages) != len(reader.pages):
        raise RuntimeError(f"Page-count mismatch for {destination.name}")


def build_staging() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    for figure_number in range(1, 11):
        source = SOURCE / f"Figure_S{figure_number}.pdf"
        destination = STAGING / source.name
        rewrite_pdf(source, destination, figure_number)
        print(destination)


def commit() -> None:
    expected = [STAGING / f"Figure_S{n}.pdf" for n in range(1, 11)]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Build staging first; missing: {missing}")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for figure_number in range(1, 11):
        canonical = SOURCE / f"Figure_S{figure_number}.pdf"
        archived = ARCHIVE / canonical.name
        if not archived.exists():
            shutil.copy2(canonical, archived)
        shutil.copy2(STAGING / canonical.name, canonical)
        print(canonical)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    if args.commit:
        commit()
    else:
        build_staging()


if __name__ == "__main__":
    main()

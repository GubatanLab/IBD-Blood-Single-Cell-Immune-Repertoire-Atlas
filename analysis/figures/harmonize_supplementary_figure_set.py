from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


FIGURE_TITLES = {
    1: "Receptor recovery and atlas validation",
    2: "TCR repertoire structure and robustness",
    3: "Longitudinal external validation",
    4: "BCR repertoire structure and programs",
    5: "BCR germline lineage sensitivity",
    6: "BCR lineage maturation and clinical context",
    7: "Global TCR-BCR covariation",
    8: "Helper and regulatory T-cell analyses",
    9: "External mucosal validation",
    10: "Nested machine-learning validation",
}

ARIAL = Path(r"C:\Windows\Fonts\arial.ttf")
ARIAL_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
REGULAR_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"
HEADER_HEIGHT = 31.0
LEFT_MARGIN = 16.0
RIGHT_MARGIN = 16.0


def register_fonts() -> None:
    global REGULAR_FONT, BOLD_FONT
    if ARIAL.exists() and ARIAL_BOLD.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(ARIAL)))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(ARIAL_BOLD)))
        REGULAR_FONT = "Arial"
        BOLD_FONT = "Arial-Bold"


def fit_font_size(text: str, available_width: float, preferred: float = 7.8, minimum: float = 5.8) -> float:
    measured = pdfmetrics.stringWidth(text, BOLD_FONT, preferred)
    if measured <= available_width:
        return preferred
    return max(minimum, preferred * available_width / measured)


def header_overlay(
    width: float,
    height: float,
    figure_number: int,
    page_number: int,
    page_count: int,
) -> PdfReader:
    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(width, height), pageCompression=1)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, height - HEADER_HEIGHT, width, HEADER_HEIGHT, fill=1, stroke=0)

    page_label = f"page {page_number} of {page_count}"
    page_font_size = 6.5
    page_label_width = pdfmetrics.stringWidth(page_label, REGULAR_FONT, page_font_size)
    title = f"Figure S{figure_number}. {FIGURE_TITLES[figure_number]}"
    title_available = width - LEFT_MARGIN - RIGHT_MARGIN - page_label_width - 11
    title_font_size = fit_font_size(title, title_available)

    baseline = height - 15.8
    pdf.setFillColorRGB(0.13, 0.13, 0.13)
    pdf.setFont(BOLD_FONT, title_font_size)
    pdf.drawString(LEFT_MARGIN, baseline, title)
    pdf.setFillColorRGB(0.38, 0.38, 0.38)
    pdf.setFont(REGULAR_FONT, page_font_size)
    pdf.drawRightString(width - RIGHT_MARGIN, baseline, page_label)

    pdf.setStrokeColorRGB(0.84, 0.84, 0.84)
    pdf.setLineWidth(0.65)
    pdf.line(LEFT_MARGIN, height - 27.5, width - RIGHT_MARGIN, height - 27.5)
    pdf.save()
    stream.seek(0)
    return PdfReader(stream)


def harmonize(source: Path, output: Path, figure_number: int) -> None:
    reader = PdfReader(source)
    writer = PdfWriter()
    total = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        overlay = header_overlay(width, height, figure_number, index, total).pages[0]
        page.merge_page(overlay)
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": f"Figure S{figure_number}. {FIGURE_TITLES[figure_number]}",
            "/Subject": "Supplementary figure for the IBD single-cell immune repertoire manuscript",
            "/Creator": "Gubatan Lab manuscript figure workflow",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)


def compose_vector_page(
    sources: list[Path],
    output: Path,
    figure_number: int,
    gap: float = 8.0,
    page_number: int = 1,
    page_count: int = 1,
) -> None:
    """Stack tight vector source figures beneath the shared supplementary header."""
    source_pages = [PdfReader(source).pages[0] for source in sources]
    widths = [float(page.mediabox.width) for page in source_pages]
    heights = [float(page.mediabox.height) for page in source_pages]
    width = max(widths) + LEFT_MARGIN + RIGHT_MARGIN
    height = HEADER_HEIGHT + 4.0 + sum(heights) + gap * (len(sources) - 1) + 12.0

    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    top = height - HEADER_HEIGHT - 4.0
    for source_page, source_width, source_height in zip(source_pages, widths, heights):
        top -= source_height
        page.merge_transformed_page(
            source_page,
            Transformation().translate((width - source_width) / 2.0, top),
        )
        top -= gap
    page.merge_page(header_overlay(width, height, figure_number, page_number, page_count).pages[0])
    writer.add_metadata(
        {
            "/Title": f"Figure S{figure_number}. {FIGURE_TITLES[figure_number]}",
            "/Subject": "Supplementary figure for the IBD single-cell immune repertoire manuscript",
            "/Creator": "Gubatan Lab manuscript figure workflow",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)


def build_set(source_dir: Path, output_dir: Path) -> None:
    register_fonts()
    for figure_number in range(1, 11):
        source = source_dir / f"Figure_S{figure_number}.pdf"
        output = output_dir / f"Figure_S{figure_number}.pdf"
        if not source.exists():
            raise FileNotFoundError(source)
        harmonize(source, output, figure_number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    build_set(arguments.source_dir, arguments.output_dir)

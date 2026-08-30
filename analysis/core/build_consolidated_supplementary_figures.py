from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from reportlab.pdfgen import canvas


ROOT = Path(r"C:/path/to/private-manuscript-workspace")
SOURCE = ROOT / "Final 7-Figure Manuscript Set" / "Supplementary Figures"
OUTPUT = ROOT / "output" / "pdf" / "Consolidated Supplementary Figures"
OUTPUT.mkdir(parents=True, exist_ok=True)


FIGURES = {
    1: {
        "title": "Cohort atlas and differential abundance",
        "sources": [(1, None)],
    },
    2: {
        "title": "Receptor recovery and adaptive-state validation",
        "sources": [(9, None), (11, None)],
    },
    3: {
        "title": "TCR repertoire definitions and sensitivity analyses",
        "sources": [(2, None)],
    },
    4: {
        "title": "TCR clone-state, sequence-neighbor, and motif robustness",
        "sources": [(3, None), (4, None), (20, 0.53)],
    },
    5: {
        "title": "Independent longitudinal validation of expansion-linked states",
        "sources": [(18, None)],
    },
    6: {
        "title": "BCR repertoire, isotype, SHM, and reference matching",
        "sources": [(5, None)],
    },
    7: {
        "title": "BCR lineage reconstruction and definition sensitivity",
        "sources": [(6, None)],
    },
    8: {
        "title": "BCR lineage maturation and clinical context",
        "sources": [(17, None), (14, None)],
    },
    9: {
        "title": "Global TCR-BCR coordination and robustness",
        "sources": [(7, None), (13, None)],
    },
    10: {
        "title": "Helper, Th17, and Treg clone architecture and B-cell coordination",
        "sources": [(15, None), (16, None), (12, None)],
    },
    11: {
        "title": "External mucosal and cross-tissue validation",
        "sources": [(19, None)],
    },
    12: {
        "title": "Nested machine-learning validation",
        "sources": [(8, None)],
    },
}


def overlay_header(width: float, height: float, text: str) -> PageObject:
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=(width, height))
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, height - 28, width, 28, fill=1, stroke=0)
    c.setFillColorRGB(0.12, 0.12, 0.12)
    target_size = 10
    measured = c.stringWidth(text, "Helvetica-Bold", target_size)
    font_size = max(6.5, min(target_size, target_size * (width - 28) / measured))
    c.setFont("Helvetica-Bold", font_size)
    c.drawString(14, height - 18, text)
    c.save()
    stream.seek(0)
    return PdfReader(stream).pages[0]


def normalized_page(page: PageObject, header: str, crop_fraction: float | None) -> PageObject:
    if crop_fraction is not None:
        width = float(page.mediabox.width) * crop_fraction
        height = float(page.mediabox.height)
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (width, height)
        page.cropbox.lower_left = (0, 0)
        page.cropbox.upper_right = (width, height)
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    result = PageObject.create_blank_page(width=width, height=height)
    scale = 0.94
    tx = width * 0.03
    ty = height * 0.01
    result.merge_transformed_page(page, Transformation().scale(scale).translate(tx, ty))
    result.merge_page(overlay_header(width, height, header))
    return result


def main() -> None:
    for figure_number, specification in FIGURES.items():
        pages = []
        for source_number, crop_fraction in specification["sources"]:
            reader = PdfReader(SOURCE / f"Figure_S{source_number}.pdf")
            if len(reader.pages) != 1:
                raise ValueError(f"Expected one page in source Figure S{source_number}")
            pages.append((source_number, reader.pages[0], crop_fraction))

        writer = PdfWriter()
        total = len(pages)
        for page_index, (source_number, page, crop_fraction) in enumerate(pages, start=1):
            continuation = "" if total == 1 else f" | page {page_index} of {total}"
            if crop_fraction is not None:
                header = f"Figure S{figure_number}. Independent motif context{continuation} (from S{source_number})"
            else:
                header = (
                    f"Figure S{figure_number}. {specification['title']}"
                    f"{continuation} (from S{source_number})"
                )
            writer.add_page(normalized_page(page, header, crop_fraction))

        destination = OUTPUT / f"Figure_S{figure_number}.pdf"
        with destination.open("wb") as handle:
            writer.write(handle)
        print(destination)


if __name__ == "__main__":
    main()

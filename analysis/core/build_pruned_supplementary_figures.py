from copy import deepcopy
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(r"C:/path/to/private-manuscript-workspace")
SOURCE = ROOT / "Final 7-Figure Manuscript Set" / "Supplementary Figures"
OUTPUT = ROOT / "output" / "pdf" / "Pruned Reordered Supplementary Figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
BCR_RASTER = ROOT / "tmp" / "pdfs" / "prune_inspect" / "rawS5_300.png"


FIGURES = {
    1: {
        "title": "Receptor recovery and adaptive-state validation",
        "sources": [(9, None), (11, None)],
    },
    2: {
        "title": "TCR repertoire structure, clone-state, and motif robustness",
        "special": "tcr",
        "sources": [(3, None), (4, None), (20, 0.53)],
    },
    3: {
        "title": "Independent longitudinal validation of expansion-linked states",
        "sources": [(18, None)],
    },
    4: {
        "title": "BCR repertoire structure, expansion-linked programs, isotype, and SHM",
        "special": "bcr",
    },
    5: {
        "title": "BCR lineage reconstruction and definition sensitivity",
        "sources": [(6, None)],
    },
    6: {
        "title": "BCR lineage maturation and clinical context",
        "sources": [(17, None), (14, None)],
    },
    7: {
        "title": "Global TCR-BCR coordination and robustness",
        "sources": [(7, None), (13, None)],
    },
    8: {
        "title": "Helper, Th17, and Treg clone architecture and B-cell coordination",
        "sources": [(15, None), (16, None), (12, None)],
    },
    9: {
        "title": "External mucosal and cross-tissue validation",
        "sources": [(19, None)],
    },
    10: {
        "title": "Nested machine-learning validation",
        "sources": [(8, None)],
    },
}


def source_page(number: int) -> PageObject:
    reader = PdfReader(SOURCE / f"Figure_S{number}.pdf")
    if len(reader.pages) != 1:
        raise ValueError(f"Expected one page in source Figure S{number}")
    return reader.pages[0]


def header_overlay(width: float, height: float, text: str) -> PageObject:
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=(width, height))
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, height - 30, width, 30, fill=1, stroke=0)
    c.setFillColorRGB(0.12, 0.12, 0.12)
    target_size = 10
    measured = c.stringWidth(text, "Helvetica-Bold", target_size)
    font_size = max(6.5, min(target_size, target_size * (width - 28) / measured))
    c.setFont("Helvetica-Bold", font_size)
    c.drawString(14, height - 19, text)
    c.save()
    stream.seek(0)
    return PdfReader(stream).pages[0]


def add_header(page: PageObject, text: str) -> PageObject:
    page.merge_page(header_overlay(float(page.mediabox.width), float(page.mediabox.height), text))
    return page


def normalized_page(page: PageObject, header: str, crop_fraction: float | None) -> PageObject:
    page = deepcopy(page)
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
    result.merge_transformed_page(
        page,
        Transformation().scale(scale).translate(width * 0.03, height * 0.01),
    )
    return add_header(result, header)


def crop_from_top(page: PageObject, box: tuple[float, float, float, float]) -> PageObject:
    """Crop using (left, top, right, bottom) coordinates measured from page top-left."""
    left, top, right, bottom = box
    original_height = float(page.mediabox.height)
    pdf_bottom = original_height - bottom
    result = deepcopy(page)
    # Keep the source coordinate system intact. merge_transformed_page will
    # construct a clipping path from this crop box before applying placement.
    result.cropbox.lower_left = (left, pdf_bottom)
    result.cropbox.upper_right = (right, original_height - top)
    return result


def place_panel(
    destination: PageObject,
    panel: PageObject,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
) -> None:
    width = float(panel.cropbox.width)
    height = float(panel.cropbox.height)
    scale = min(max_width / width, max_height / height)
    left = float(panel.cropbox.left)
    bottom = float(panel.cropbox.bottom)
    destination.merge_transformed_page(
        panel,
        Transformation()
        .translate(-left, -bottom)
        .scale(scale)
        .translate(x, y),
    )


def pil_crop_from_pdf_coordinates(
    image: Image.Image,
    box: tuple[float, float, float, float],
    pdf_width: float = 931.5530443254,
    pdf_height: float = 1334.52,
) -> Image.Image:
    sx = image.width / pdf_width
    sy = image.height / pdf_height
    left, top, right, bottom = box
    return image.crop(
        (
            round(left * sx),
            round(top * sy),
            round(right * sx),
            round(bottom * sy),
        )
    )


def paste_fit(canvas_image: Image.Image, panel: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    max_width = right - left
    max_height = bottom - top
    scale = min(max_width / panel.width, max_height / panel.height)
    resized = panel.resize(
        (round(panel.width * scale), round(panel.height * scale)),
        Image.Resampling.LANCZOS,
    )
    x = left + (max_width - resized.width) // 2
    y = top + (max_height - resized.height) // 2
    canvas_image.paste(resized, (x, y))


def page_from_raster(image: Image.Image, width: float, height: float) -> PageObject:
    png = BytesIO()
    image.save(png, format="PNG", optimize=True)
    png.seek(0)
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=(width, height))
    c.drawImage(ImageReader(png), 0, 0, width=width, height=height, mask="auto")
    c.save()
    stream.seek(0)
    return PdfReader(stream).pages[0]


def tcr_core_page(figure_number: int, total_pages: int) -> PageObject:
    raw = source_page(2)
    # Retain only the three core participant-level repertoire panels (former A-C).
    core = crop_from_top(raw, (0, 43, 915.92, 286))
    width, height = 915.92, 300.0
    page = PageObject.create_blank_page(width=width, height=height)
    place_panel(page, core, 0, 8, width, 250)
    return add_header(
        page,
        f"Figure S{figure_number}. TCR repertoire structure and clone-size context | page 1 of {total_pages}",
    )


def bcr_pages(figure_number: int) -> list[PageObject]:
    if not BCR_RASTER.exists():
        raise FileNotFoundError(
            f"Render former Figure S5 at 300 dpi before building: {BCR_RASTER}"
        )
    with Image.open(BCR_RASTER) as opened:
        raw_image = opened.convert("RGB")

    # Page 1: participant-level repertoire/clone-size/state composition plus
    # diagnosis-stratified expansion-linked transcriptional programs.
    page1_image = Image.new("RGB", (3300, 2550), "white")
    panels = {
        "A": pil_crop_from_pdf_coordinates(raw_image, (12, 43, 328, 278)),
        "B": pil_crop_from_pdf_coordinates(raw_image, (330, 43, 646, 278)),
        "C": pil_crop_from_pdf_coordinates(raw_image, (654, 43, 930, 278)),
        "F": pil_crop_from_pdf_coordinates(raw_image, (650, 300, 930, 535)),
        "G": pil_crop_from_pdf_coordinates(raw_image, (8, 548, 276, 838)),
    }
    paste_fit(page1_image, panels["A"], (20, 150, 1085, 965))
    paste_fit(page1_image, panels["B"], (1100, 150, 2190, 965))
    paste_fit(page1_image, panels["C"], (2205, 150, 3280, 965))
    paste_fit(page1_image, panels["F"], (500, 1010, 1550, 2480))
    paste_fit(page1_image, panels["G"], (1750, 1010, 2800, 2480))
    label_font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 120)
    ImageDraw.Draw(page1_image).text((520, 1060), "F", fill="black", font=label_font)
    page1 = page_from_raster(page1_image, 931.55, 720)
    add_header(
        page1,
        f"Figure S{figure_number}. BCR repertoire structure and expansion-linked programs | page 1 of 2",
    )

    # Page 2: retain class switching, isotype composition, and state-matched SHM.
    page2_image = Image.new("RGB", (3300, 1170), "white")
    maturation = pil_crop_from_pdf_coordinates(raw_image, (7, 835, 930, 1092))
    paste_fit(page2_image, maturation, (15, 145, 3285, 1145))
    page2 = page_from_raster(page2_image, 931.55, 330)
    add_header(
        page2,
        f"Figure S{figure_number}. BCR class switching, isotype composition, and SHM | page 2 of 2",
    )
    return [page1, page2]


def build_standard_pages(figure_number: int, specification: dict) -> list[PageObject]:
    pages = []
    total = len(specification["sources"])
    for page_index, (source_number, crop_fraction) in enumerate(specification["sources"], start=1):
        continuation = "" if total == 1 else f" | page {page_index} of {total}"
        if crop_fraction is not None:
            title = "Independent motif context"
        else:
            title = specification["title"]
        pages.append(
            normalized_page(
                source_page(source_number),
                f"Figure S{figure_number}. {title}{continuation}",
                crop_fraction,
            )
        )
    return pages


def main() -> None:
    for old in OUTPUT.glob("Figure_S*.pdf"):
        old.unlink()

    for figure_number, specification in FIGURES.items():
        if specification.get("special") == "tcr":
            standard = build_standard_pages(figure_number, specification)
            pages = [tcr_core_page(figure_number, len(standard) + 1), *standard]
            # Re-header the standard pages because their page numbers start at 1.
            total = len(pages)
            for index in range(1, total):
                # The existing header is covered by add_header's white band.
                source_number, crop_fraction = specification["sources"][index - 1]
                title = "Independent motif context" if crop_fraction is not None else specification["title"]
                add_header(pages[index], f"Figure S{figure_number}. {title} | page {index + 1} of {total}")
        elif specification.get("special") == "bcr":
            pages = bcr_pages(figure_number)
        else:
            pages = build_standard_pages(figure_number, specification)

        writer = PdfWriter()
        for page in pages:
            writer.add_page(page)
        destination = OUTPUT / f"Figure_S{figure_number}.pdf"
        with destination.open("wb") as handle:
            writer.write(handle)
        print(destination)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply final label-fit and footnote-cleanup edits to main Figures 1-7."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import Color, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "Cell Press Redrawn Figure Set" / "Main Figures"
BOOK = (
    ROOT
    / "output"
    / "pdf"
    / "Publication Ready Combined Figures"
    / "IBD_Immune_Repertoire_Main_Figures_1-7_600dpi_Publication_Ready.pdf"
)


def erase_top_rect(
    layer: canvas.Canvas,
    page_height: float,
    x0: float,
    top0: float,
    x1: float,
    top1: float,
    fill=white,
) -> None:
    """Erase a rectangle specified in top-origin PDF coordinates."""
    layer.setFillColor(fill)
    layer.setStrokeColor(fill)
    layer.rect(x0, page_height - top1, x1 - x0, top1 - top0, stroke=0, fill=1)


def overlay_page(width: float, height: float, painter) -> PageObject:
    stream = BytesIO()
    layer = canvas.Canvas(stream, pagesize=(width, height))
    painter(layer, width, height)
    layer.save()
    stream.seek(0)
    return PdfReader(stream).pages[0]


def revise_vector_pdf(
    source: Path,
    *,
    painter=None,
    scale: float | None = None,
    translate_x: float = 0.0,
) -> None:
    reader = PdfReader(str(source))
    if len(reader.pages) != 1:
        raise RuntimeError(f"Expected a one-page figure: {source}")

    original = reader.pages[0]
    width = float(original.mediabox.width)
    height = float(original.mediabox.height)

    if scale is None:
        page = original
    else:
        translate_y = (1.0 - scale) * height / 2.0
        page = PageObject.create_blank_page(width=width, height=height)
        page.merge_transformed_page(
            original,
            Transformation().scale(scale, scale).translate(translate_x, translate_y),
        )

    if painter is not None:
        page.merge_page(overlay_page(width, height, painter))

    writer = PdfWriter()
    writer.add_page(page)
    with source.open("wb") as stream:
        writer.write(stream)


def paint_figure2(layer: canvas.Canvas, width: float, height: float) -> None:
    # Remove the panel A methods subtitle and the panel H cohort/bin footnote.
    erase_top_rect(layer, height, 25.5, 14.0, 156.5, 21.5)
    erase_top_rect(layer, height, 25.5, 579.6, 207.5, 587.2)


def paint_figure3(layer: canvas.Canvas, width: float, height: float) -> None:
    # Remove duplicated sample-size and methods footnotes while preserving q/I2 and P values.
    erase_top_rect(layer, height, 420.0, 107.2, 529.3, 114.4)
    erase_top_rect(layer, height, 319.0, 326.6, 549.0, 332.8)
    erase_top_rect(layer, height, 412.0, 439.0, 527.7, 446.2)
    # This note sits on the lavender expansion block, so preserve that panel fill.
    expansion_block = Color(246 / 255, 239 / 255, 247 / 255)
    erase_top_rect(layer, height, 464.0, 618.6, 528.5, 625.5, expansion_block)


def paint_figure5(layer: canvas.Canvas, width: float, height: float) -> None:
    # Panel C FDR-range note duplicates the plotted q/FDR results.
    erase_top_rect(layer, height, 154.0, 159.8, 248.5, 168.3)

    # Remove the panel G symbol-method note that collides with the final legend item.
    # Redraw the covered legend text after clearing the shared area.
    erase_top_rect(layer, height, 416.0, 445.5, 546.7, 458.2)
    layer.setFillColor(Color(0.12, 0.12, 0.12))
    layer.setFont("Helvetica", 7.0)
    layer.drawString(417.87, 271.0, "Unmapped")


def revise_figure7_raster() -> None:
    png_path = MAIN / "Figure_7.png"
    tif_path = MAIN / "Figure_7.tif"
    pdf_path = MAIN / "Figure_7.pdf"

    image = Image.open(png_path).convert("RGB")
    if image.size != (2700, 3540):
        raise RuntimeError(f"Unexpected Figure 7 raster size: {image.size}")

    draw = ImageDraw.Draw(image)
    # Bring the model key into the main visual field while preserving its
    # complete contents. Detect the original border so repeated finalization
    # remains safe and does not shift an already-moved key a second time.
    legend_box = (2128, 177, 2674, 690)
    legend_shift = 160
    border_color = (213, 226, 237)
    border_count = sum(
        1 for y in range(185, 684)
        if image.getpixel((2137, y)) == border_color
    )
    if border_count > 300:
        legend = image.crop(legend_box)
        draw.rectangle(legend_box, fill="white")
        image.paste(legend, (legend_box[0] - legend_shift, legend_box[1]))
        draw = ImageDraw.Draw(image)
    # Panel F already shares the post-selection validation scheme established in panel D.
    # Clear the duplicated tag and redraw the title so the words no longer collide.
    draw.rectangle((475, 2615, 1730, 2685), fill="white")
    title_font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 45)
    panel_font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 56)
    draw.text((484, 2615), "Biologic-response models", font=title_font, fill=(32, 33, 36))
    # The shared cleanup band crosses the original G marker; restore it explicitly.
    draw.text((1641, 2611), "G", font=panel_font, fill=(32, 33, 36))

    image.save(png_path, optimize=True)
    image.save(tif_path, compression="tiff_lzw", dpi=(300, 300))

    current = PdfReader(str(pdf_path))
    width = float(current.pages[0].mediabox.width)
    height = float(current.pages[0].mediabox.height)
    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(width, height))
    pdf.drawImage(ImageReader(image), 0, 0, width=width, height=height, mask="auto")
    pdf.showPage()
    pdf.save()
    stream.seek(0)
    rebuilt = PdfReader(stream)
    writer = PdfWriter()
    writer.add_page(rebuilt.pages[0])
    with pdf_path.open("wb") as output:
        writer.write(output)


def rebuild_book() -> None:
    writer = PdfWriter()
    starts: list[tuple[str, int]] = []
    for number in range(1, 8):
        source = MAIN / f"Figure_{number}.pdf"
        reader = PdfReader(str(source))
        if len(reader.pages) != 1:
            raise RuntimeError(f"Expected one page in {source}")
        starts.append((f"Figure {number}", len(writer.pages)))
        writer.append_pages_from_reader(reader)

    for label, page_index in starts:
        writer.add_outline_item(label, page_index)
    writer.add_metadata(
        {
            "/Title": "IBD Blood Single-Cell Immune Repertoire Atlas - Main Figures 1-7",
            "/Subject": "Publication-ready main figures for Immunity submission",
            "/Author": "Gubatan et al.",
            "/Creator": "Vector-preserving final figure assembly",
        }
    )
    writer.page_mode = "/UseOutlines"
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    with BOOK.open("wb") as output:
        writer.write(output)


def main() -> None:
    # Modest uniform scaling brings existing right-edge labels inside the fixed Cell Press canvas.
    revise_vector_pdf(MAIN / "Figure_1.pdf", scale=0.988, translate_x=1.0)
    revise_vector_pdf(MAIN / "Figure_2.pdf", painter=paint_figure2)
    revise_vector_pdf(MAIN / "Figure_3.pdf", painter=paint_figure3)
    revise_vector_pdf(MAIN / "Figure_5.pdf", painter=paint_figure5)
    revise_vector_pdf(MAIN / "Figure_6.pdf", scale=0.972, translate_x=2.0)
    revise_figure7_raster()
    rebuild_book()

    check = PdfReader(str(BOOK))
    if len(check.pages) != 7:
        raise RuntimeError(f"Expected seven pages, found {len(check.pages)}")
    print(f"Updated {BOOK}")
    print("Figures revised: 1, 2, 3, 5, 6, 7; Figure 4 unchanged")


if __name__ == "__main__":
    main()

from pathlib import Path
from copy import copy
from io import BytesIO
import json
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, NameObject, TextStringObject, ArrayObject, RectangleObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path("C:/path/to/private-manuscript-workspace")
OUT=ROOT/'output/pdf/S2 Single Page Revision 2026-09-20'
OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'output/archive/Supplementary_before_size_standardization_20260920/Figure_S2.pdf'
reader=PdfReader(SRC)
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold','C:/Windows/Fonts/arialbd.ttf'))
W,H=612,792

def overlay(draw,w=W,h=H):
    b=BytesIO();c=canvas.Canvas(b,pagesize=(w,h));draw(c);c.save()
    return PdfReader(b).pages[0]

def crop(page_index,box,remove=()):
    page=copy(reader.pages[page_index]);ph=float(page.mediabox.height)
    if remove:
        content=ContentStream(page.get_contents(),page.pdf)
        for args,op in content.operations:
            if op in (b'Tj',b'TJ'):
                s=''.join(str(t)for t in (args[0]if op==b'TJ'else args)if isinstance(t,str))
                if s in remove:args[0]=ArrayObject([])if op==b'TJ'else TextStringObject('')
        page[NameObject('/Contents')]=content
    x0,y0,x1,y1=box
    page.cropbox=RectangleObject((x0,ph-y1,x1,ph-y0))
    writer=PdfWriter();result=writer.add_blank_page(width=x1-x0,height=y1-y0)
    result.merge_transformed_page(page,Transformation().translate(-x0,-(ph-y1)))
    return result

panels=[
 ('A','Participant-level repertoire diversity',0,(14,49,312,255),(24,55,335,292),()),
 ('B','Clone-size architecture',0,(331,49,525,266),(377,55,211,292),()),
 ('C','Cell-state composition of exact paired alpha-beta clonotypes',0,(0,300,516,453),(24,333,564,195),('Cell-state composition of exact paired αβ clonotypes',)),
 ('D','Independent TCR motif context',1,(12,30,275,247),(170,538,273,230),('D',)),
]
writer=PdfWriter();page=writer.add_blank_page(width=W,height=H)
def header(c):
    c.setFillColorRGB(.08,.08,.08);c.setFont('Arial-Bold',10)
    c.drawString(24,H-25,'Figure S2. TCR repertoire structure and external motif context')
    c.setFillColorRGB(.38,.38,.38);c.setFont('Arial',8);c.drawRightString(588,H-25,'1/1')
    c.setStrokeColorRGB(.8,.8,.8);c.setLineWidth(.5);c.line(24,H-38,588,H-38)
page.merge_page(overlay(header))
layout=[]
for letter,title,index,box,slot,remove in panels:
    art=crop(index,box,remove);x,top,w,h=slot
    pw,ph=float(art.mediabox.width),float(art.mediabox.height)
    scale=min(w/pw,(h-26)/ph)
    px=x+(w-pw*scale)/2;py=H-top-26-ph*scale
    assert py>=24
    page.merge_transformed_page(art,Transformation().scale(scale).translate(px,py))
    def label(c):
        c.setFillColorRGB(.07,.07,.07);c.setFont('Arial-Bold',12);c.drawString(x,H-top-10,letter)
        c.setFont('Arial-Bold',8.5);c.drawString(x+20,H-top-9,title)
    page.merge_page(overlay(label))
    layout.append({'figure':2,'page':1,'panel':letter,'title':title,'slot_points':slot,'art_scale':scale,'origin':f'S2 page {index+1}','source_crop':box})
page.compress_content_streams(level=6)
writer.add_metadata({'/Title':'Figure S2. TCR repertoire structure and external motif context','/Subject':'Panels A-D on one page; plot data preserved; 20 September 2026'})
with (OUT/'Figure_S2_Single_Page_09-20-2026.pdf').open('wb')as f:writer.write(f)
(OUT/'S2 layout.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
assert len(PdfReader(OUT/'Figure_S2_Single_Page_09-20-2026.pdf').pages)==1
print(OUT/'Figure_S2_Single_Page_09-20-2026.pdf')

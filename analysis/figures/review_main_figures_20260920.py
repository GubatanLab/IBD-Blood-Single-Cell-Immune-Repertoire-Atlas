"""Targeted, data-preserving PDF layout edits after visual review."""
from pathlib import Path
from copy import copy
from io import BytesIO
import shutil, json
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import ContentStream, NameObject, ArrayObject, TextStringObject, RectangleObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path("C:/path/to/private-manuscript-workspace")
SRC=ROOT/'Final 7-Figure Manuscript Set/Main Figures'
SUP=ROOT/'Final 7-Figure Manuscript Set/Pruned Reordered Supplementary Figures S1-S10'
OUT=ROOT/'output/pdf/Figures Reviewed 2026-09-20'
ARCH=ROOT/'output/archive/Before_figure_consistency_review_20260920'
OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold','C:/Windows/Fonts/arialbd.ttf'))

def backup(p):
    d=ARCH/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True)
    if not d.exists():shutil.copy2(p,d)
    return d

def decoded(v):
    if isinstance(v,str):return v
    if isinstance(v,bytes):
        try:return v.decode('utf-16-be')
        except UnicodeDecodeError:return ''
    return ''

def edit_text(p,remove=(),replace=None):
    cs=ContentStream(p.get_contents(),p.pdf);found=[]
    for a,op in cs.operations:
        if op not in (b'Tj',b'TJ'):continue
        vals=a[0] if op==b'TJ' else a
        s=''.join(decoded(v) for v in vals)
        if s in remove:
            found.append(s);a[0]=ArrayObject([]) if op==b'TJ' else TextStringObject('')
        elif replace and s in replace:
            found.append(s);a[0]=ArrayObject([TextStringObject(replace[s])]) if op==b'TJ' else TextStringObject(replace[s])
    p[NameObject('/Contents')]=cs
    return found

def overlay(p,paint):
    w,h=float(p.mediabox.width),float(p.mediabox.height)
    b=BytesIO();c=canvas.Canvas(b,pagesize=(w,h));paint(c,h);c.save();p.merge_page(PdfReader(b).pages[0])

def label(c,h,x,y,s,size=8,bold=True,color=(.08,.08,.08)):
    c.setFillColorRGB(*color);c.setFont('Arial-Bold' if bold else 'Arial',size);c.drawString(x,h-y,s)

def write(p,path,pad=True):
    if pad:
        w,h=float(p.mediabox.width),float(p.mediabox.height)
        out=PageObject.create_blank_page(width=w+16,height=h+16)
        out.merge_transformed_page(p,Transformation().translate(8,8));p=out
    wr=PdfWriter();wr.add_page(p);wr.add_metadata({'/Title':path.stem.replace('_',' '),'/Subject':'Layout and manuscript consistency review, 20 September 2026'})
    with path.open('wb')as f:wr.write(f)

audit=[]
for n in range(1,7):
    p=PdfReader(backup(SRC/f'Figure_{n}.pdf')).pages[0]
    found=[]
    if n==1:
        found=edit_text(p,{'A','H','I','MiloR differential abundance','Original Level 3'})
        def paint(c,h):
            label(c,h,58,23,'A',12)
            label(c,h,60,471,'H',12)
            label(c,h,60,672,'I',12)
            label(c,h,75.5,671.5,'Milo differential abundance',8)
            for x in [166.879,327.489,491.75]:label(c,h,x,334.4,'Level 3 states',5.3)
        overlay(p,paint)
    if n==3:
        old={'Sequence-related paired TCRs converge on Figure 2 programs','Expansion reveals γδ effector shifts beyond Vγ9Vδ2 identity'}
        found=edit_text(p,old)
        def paint(c,h):
            label(c,h,341,13.5,'Paired-TCR relatedness and program concordance',7.8)
            label(c,h,341,513,'γδ programs by receptor identity and expansion',7.8)
        overlay(p,paint)
    if n==4:
        found=edit_text(p,{'A','Participant-balanced B-cell Slingshot trajectory','Effector programs across pseudotime'}, {'q=0.298':'P=0.298','q=0.176':'P=0.176'})
        def paint(c,h):
            label(c,h,25,10,'A',12)
            label(c,h,190.2,223,'Balanced B-cell Slingshot trajectory',8)
            label(c,h,411.7,223,'Programs across pseudotime',8)
        overlay(p,paint)
    if n==5:
        rem={'C','D','Paired-clonotype-resolved heavy-chain SHM','Preferential CDR-targeted mutation','CD vs UC: all FDR >= 0.65',
            'P200 | IGHV7-4-1/IGHJ5','5 seq; 15 cells | IgA;IgM','IGKV1-33*01 | Naive B',
            'P96 | IGHV4-59/IGHJ4','4 seq; 5 cells | IgA','IGKV3-11*01 | Naive B',
            'P92 | IGHV3-33/IGHJ6','5 seq; 16 cells | IgA','IGLV6-57*01 | CD5+ B Cell'}
        found=edit_text(p,rem)
        def paint(c,h):
            label(c,h,61.84,150,'C',12);label(c,h,89.5,150,'Paired-clonotype-resolved heavy-chain SHM',8)
            label(c,h,359.28,150,'D',12);label(c,h,387,150,'Preferential CDR-targeted mutation',8)
            for x,lines in [(79.88,['P200 | IGHV7-4-1/IGHJ5','5 seq; 15 cells | IgA;IgM','IGKV1-33*01 | Naive B']),
                            (149.5,['P96 | IGHV4-59/IGHJ4','4 seq; 5 cells | IgA','IGKV3-11*01 | Naive B']),
                            (214.5,['P92 | IGHV3-33/IGHJ6','5 seq; 16 cells | IgA','IGLV6-57*01 | CD5+ B Cell'])]:
                for j,s in enumerate(lines):label(c,h,x,390+j*5,s,4.35,False)
            label(c,h,413,419,'CD vs UC: all FDR ≥0.65',5.5,False,(.40,.40,.40))
        overlay(p,paint)
    if n==6:
        # Recover the full color-bar label, which exists outside the old page boundary.
        p.mediabox=RectangleObject((0,0,526,float(p.mediabox.height)));p.cropbox=p.mediabox
        found=edit_text(p,{'Helper only','Cytotoxic only','Joint','Bidirectional helper–B-cell signaling'})
        def paint(c,h):
            c.setFillColorRGB(1,1,1);c.rect(84,h-644,93,14,fill=1,stroke=0)
            items=[(83,'Helper only',(.48,.20,.60)),(134,'Cytotoxic only',(0,.45,.70)),(195,'Joint',(.12,.12,.12))]
            for x,s,col in items:
                c.setFillColorRGB(*col)
                if s=='Joint':
                    pa=c.beginPath();pa.moveTo(x,h-636);pa.lineTo(x+3,h-639);pa.lineTo(x,h-642);pa.lineTo(x-3,h-639);pa.close();c.drawPath(pa,fill=1,stroke=0)
                else:c.circle(x,h-639,2.7,fill=1,stroke=0)
                label(c,h,x+5,641,s,5.3,False)
            label(c,h,305,523.5,'Inferred helper–B-cell interactions',7.6)
        overlay(p,paint)
    write(p,OUT/f'Figure_{n}.pdf')
    audit.append({'figure':n,'removed_or_replaced':found})

# Figure 7 quantitative panels are redrawn from their unchanged source CSVs,
# while its original schematic is preserved as a PDF region.
src=PdfReader(backup(SRC/'Figure_7.pdf')).pages[0]
w,h=float(src.mediabox.width),float(src.mediabox.height)
page=PdfReader(ROOT/'tmp/main_figure_consistency_review/Figure_7_numeric_layout.pdf').pages[0]
src.cropbox=RectangleObject((0,h-159,w,h))
scale=560/w
page.merge_transformed_page(src,Transformation().scale(scale).translate(0,float(page.mediabox.height)-h*scale))
write(page,OUT/'Figure_7.pdf')

# Wording on these supplementary panel titles must match the actual analyses.
supp_changes={7:[(0,'Within-participant TCR-BCR covariation','Participant-level TCR-BCR covariation',44,64)],
              8:[(1,'Strongest prespecified association:','Representative adjusted association:',335,406)],
              9:[(0,'BCR expansion-program kinetics','BCR clone-size/program associations',44,424)]}
for n,changes in supp_changes.items():
    r=PdfReader(backup(SUP/f'Figure_S{n}.pdf'));wr=PdfWriter()
    for i,p in enumerate(r.pages):
        for pi,old,new,x,y in changes:
            if pi!=i:continue
            found=edit_text(p,{old});assert found,(n,old)
            overlay(p,lambda c,h:label(c,h,x,y,new,8.5))
        wr.add_page(p)
    with (OUT/f'Figure_S{n}.pdf').open('wb') as f:wr.write(f)
(OUT/'layout_edit_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
print('Prepared seven main figures and three supplementary title corrections:',OUT)

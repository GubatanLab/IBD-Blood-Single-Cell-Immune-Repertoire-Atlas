from pathlib import Path
from io import BytesIO
from copy import copy
import csv, json, math, shutil
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject, ContentStream, NameObject, TextStringObject, ArrayObject, FloatObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image

ROOT = Path("C:/path/to/private-manuscript-workspace")
SRC=ROOT/'Final 7-Figure Manuscript Set/Pruned Reordered Supplementary Figures S1-S10'
OUT=ROOT/'output/pdf/Supplementary Figures Standardized 2026-09-20'
WORK=ROOT/'tmp/pdfs/supplement_standardization_review'
ARCH=ROOT/'output/archive/Supplementary_before_size_standardization_20260920'
OUT.mkdir(parents=True,exist_ok=True);ARCH.mkdir(parents=True,exist_ok=True)
for p in SRC.iterdir():
    if p.is_file() and not (ARCH/p.name).exists():shutil.copy2(p,ARCH/p.name)
pdfmetrics.registerFont(TTFont('Arial',r'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold',r'C:/Windows/Fonts/arialbd.ttf'))
W,H=612,792
M,GAP=24,18
FULL,HALF=564,273
TITLES={1:'Receptor recovery and adaptive-state validation',2:'TCR repertoire structure and external motif context',3:'Independent longitudinal PBMC validation',4:'BCR repertoire structure, programs, and maturation',5:'BCR germline-lineage definition sensitivity',6:'BCR lineage maturation and clinical context',7:'Global TCR-BCR coordination and robustness',8:'Helper, Th17, and Treg analyses',9:'External mucosal and cross-tissue validation',10:'Nested machine-learning validation'}
readers={n:PdfReader(ARCH/f'Figure_S{n}.pdf') for n in range(1,11)}
extra={
 'hi1':PdfReader(ROOT/'High Impact Additional Analyses/Figure_HI1_TCR_BCR_coordination.pdf'),
 'hi1b':PdfReader(ROOT/'High Impact Additional Analyses/Figure_HI1b_CD_specific_T_B_module_coupling.pdf'),
}
# Give the dense, two-line program labels breathing room at the same panel width.
label_page=readers[4].pages[1]
label_contents=ContentStream(label_page.get_contents(),label_page.pdf)
for args,op in label_contents.operations:
    if op==b'Tf' and abs(float(args[1])-5.2)<.001:args[1]=FloatObject(4.6)
label_page[NameObject('/Contents')]=label_contents
audit=[]

def wrap(s,width,font='Arial-Bold',size=8.5):
    lines=[]
    for part in s.split('\n'):
        line=''
        for word in part.split():
            nxt=(line+' '+word).strip()
            if line and pdfmetrics.stringWidth(nxt,font,size)>width:lines.append(line);line=word
            else:line=nxt
        lines.append(line)
    return lines

def overlay(fn,w=W,h=H):
    b=BytesIO();c=canvas.Canvas(b,pagesize=(w,h));fn(c);c.save();return PdfReader(b).pages[0]

def crop(source,box,masks=(),remove=()):
    x0,y0,x1,y1=box
    p=copy(source)
    if remove:
        contents=ContentStream(source.get_contents(),source.pdf)
        for args,op in contents.operations:
            if op in (b'Tj',b'TJ'):
                text=''.join(str(t) for t in (args[0] if op==b'TJ' else args) if isinstance(t,str))
                if text in remove:args[0]=ArrayObject([]) if op==b'TJ' else TextStringObject('')
        p[NameObject('/Contents')]=contents
    ph=float(p.mediabox.height)
    if masks:
        def erase(c):
            c.setFillColorRGB(1,1,1)
            for a,b,z,d in masks:c.rect(a,ph-d,z-a,d-b,fill=1,stroke=0)
        p.merge_page(overlay(erase,float(p.mediabox.width),ph))
    p.cropbox=RectangleObject((x0,ph-y1,x1,ph-y0))
    wr=PdfWriter();out=wr.add_blank_page(width=x1-x0,height=y1-y0)
    out.merge_transformed_page(p,Transformation().translate(-x0,-(ph-y1)))
    return out

def source(n,pi,box,pixels=False,masks=(),remove=()):
    p=readers[n].pages[pi-1]
    if pixels:
        iw,ih=Image.open(WORK/f'S{n}-{pi}.png').size
        sx,sy=float(p.mediabox.width)/iw,float(p.mediabox.height)/ih
        box=(box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy)
        masks=[(b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy)for b in masks]
    return crop(p,box,masks,remove)

def panel(letter,title,n,pi,box,slot,pixels=False,masks=(),remove=()):
    return {'letter':letter,'title':title,'src':source(n,pi,box,pixels,masks,remove),'slot':slot,'origin':f'S{n} page {pi}','crop':box}

def extpanel(letter,title,key,box,slot):
    return {'letter':letter,'title':title,'src':crop(extra[key].pages[0],box),'slot':slot,'origin':key,'crop':box}

def rgb(hex):return tuple(int(hex[i:i+2],16)/255 for i in (1,3,5))
def heat_color(v,lim):
    lo,mid,hi=rgb('#5E7DA8'),rgb('#FAF8F5'),rgb('#B64A50')
    t=min(abs(v)/lim,1);end=hi if v>=0 else lo
    return tuple(a+(b-a)*t for a,b in zip(mid,end))

def heatmap_panel(kind,slot):
    if kind=='S7A':
        rows=list(csv.DictReader((ROOT/'High Impact Additional Analyses/Table_HI_partial_spearman_TCR_BCR_coordination.csv').open()))
        ry=['tcr_clonality','tcr_gini','tcr_expanded_cell_fraction','tcr_multistate_clone_fraction','tcr_cytotoxic_expanded']
        cx=['bcr_gini','bcr_expanded_cell_fraction','bcr_multistate_clone_fraction','bcr_switched_fraction','bcr_SHM_rate','bcr_IgA_mucosal_module','bcr_plasma_differentiation_module','bcr_BAFF_APRIL_module']
        yl=['TCR clonality','TCR Gini','Expanded TCR-cell fraction','Multistate TCR-clone fraction','Expanded-TCR cytotoxic score']
        xl=['BCR Gini','Expanded BCR-cell fraction','Multistate BCR-clone fraction','Class-switched BCR fraction','BCR SHM rate','IgA mucosal module','Plasma-cell differentiation','BAFF/APRIL module']
        rkey,ckey,qkey='tcr_metric','bcr_metric','p_adj';lim=1.;height=260;top=22
        letter,title='A','Within-participant TCR-BCR covariation'
    else:
        rows=[r for r in csv.DictReader((ROOT/'High Impact Additional Analyses/Th17 Treg B Helper Analyses/Table_TB10_targeted_T_B_correlations.csv').open())if r['group']=='IBD']
        ry=yl=['Pathogenic Th17','Conventional Th17','Suppressive Treg','Reprogrammed Treg','Tph/Tfh help']
        cx=xl=['Plasma differentiation','IgA mucosal plasma','IgG inflammatory plasma','Atypical memory','Antigen presentation','Class-switched BCR','BCR SHM']
        rkey,ckey,qkey='T_label','B_label','FDR_within_group';lim=.55;height=280;top=4
        letter,title='E','Targeted T-cell-B-cell coupling in IBD'
    values={(r[rkey],r[ckey]):r for r in rows}
    assert all((y,x)in values for y in ry for x in cx)
    def draw(c):
        width=564;left=124;right=37;bottom=87
        cw=(width-left-right)/len(cx);ch=(height-top-bottom)/len(ry)
        if kind=='S7A':
            c.setFont('Arial',7.2);c.drawString(left,height-10,'Partial Spearman correlations adjusted for diagnosis, age, sex, and receptor depth')
        for i,y in enumerate(ry):
            cy=height-top-(i+1)*ch
            c.setFillColorRGB(.12,.12,.12);c.setFont('Arial',7.5);c.drawRightString(left-7,cy+ch/2-2.5,yl[i])
            for j,x in enumerate(cx):
                r=values[y,x];v=float(r['partial_rho']);q=float(r[qkey]);stars=('**'if q<.01 else '*'if q<.05 else '')if kind=='S7A'else('***'if q<.001 else '**'if q<.01 else '*'if q<.05 else '')
                c.setFillColorRGB(*heat_color(v,lim));c.setStrokeColorRGB(1,1,1);c.setLineWidth(.45)
                c.rect(left+j*cw,cy,cw,ch,fill=1,stroke=1)
                c.setFillColorRGB(*((1,1,1)if abs(v)>.55*lim else(.1,.1,.1)))
                c.setFont('Arial',7.5);c.drawCentredString(left+(j+.5)*cw,cy+ch/2-2.5,f'{v:.2f}{stars}')
        c.setFillColorRGB(.1,.1,.1);c.setFont('Arial',7.1)
        for j,label in enumerate(xl):
            c.saveState();c.translate(left+(j+.5)*cw,bottom-8);c.rotate(48);c.drawRightString(0,0,label);c.restoreState()
        barx=width-right+9;bary=bottom+15;barh=height-top-bottom-30
        for i in range(100):
            c.setFillColorRGB(*heat_color(-lim+2*lim*(i+.5)/100,lim));c.rect(barx,bary+i*barh/100,8,barh/100+.1,fill=1,stroke=0)
        c.setFillColorRGB(.2,.2,.2);c.setFont('Arial',6.5)
        for v in [-lim,0,lim]:c.drawString(barx+10,bary+(v+lim)/(2*lim)*barh-2,f'{v:g}')
        c.setFont('Arial',6);c.drawCentredString(barx+9,bary+barh+11,'Partial rho')
    p=overlay(draw,564,height)
    return {'letter':letter,'title':title,'src':p,'slot':slot,'origin':'existing published-panel source table; values and significance unchanged','crop':None}

def external_heatmap(slot):
    rows=list(csv.DictReader((ROOT/'External Validation 20260829/results/GSE261334_T_B_coordination.csv').open()))
    ys=['Activated Treg','GZMK memory','Th17/IL23','Tph/Tfh help']
    xs=['Antibody secretion/UPR','IgA mucosal','IgG inflammatory','Plasma differentiation']
    values={(r['T_module'],r['B_module']):float(r['rho']) for r in rows}
    def draw(c):
        left=80;bottom=89;cw=40;ch=38;height=245
        for i,y in enumerate(ys):
            yy=bottom+(3-i)*ch
            c.setFont('Arial',7);c.setFillColorRGB(.1,.1,.1);c.drawRightString(left-5,yy+ch/2-2.3,y)
            for j,x in enumerate(xs):
                v=values[y,x];c.setFillColorRGB(*heat_color(v,1));c.setStrokeColorRGB(1,1,1);c.setLineWidth(.5)
                c.rect(left+j*cw,yy,cw,ch,fill=1,stroke=1)
                c.setFillColorRGB(.1,.1,.1);c.setFont('Arial',8);c.drawCentredString(left+(j+.5)*cw,yy+ch/2-2.6,f'{v:.2f}')
        c.setFont('Arial',7)
        for j,x in enumerate(xs):
            c.saveState();c.translate(left+(j+.5)*cw,bottom-7);c.rotate(50);c.drawRightString(0,0,x);c.restoreState()
        for i in range(100):
            c.setFillColorRGB(*heat_color(-1+2*(i+.5)/100,1));c.rect(246,bottom+i*1.52,6,1.6,fill=1,stroke=0)
        c.setFillColorRGB(.1,.1,.1);c.setFont('Arial',6)
        for v in [-1,0,1]:c.drawString(254,bottom+(v+1)*76-2,str(v))
        c.saveState();c.translate(272,bottom+76);c.rotate(90);c.drawCentredString(0,0,'Spearman rho');c.restoreState()
    return {'letter':'D','title':'Coordinated T-B programs','src':overlay(draw,273,245),'slot':slot,'origin':'GSE261334_T_B_coordination.csv; existing correlations','crop':None}

P=lambda l,t,n,pg,b,s,**kw:panel(l,t,n,pg,b,s,True,**kw)
L,R=24,315
TOP,LOW=55,414
REG=lambda x,y:(x,y,273,350)

pages={}
pages[1]=[
 [panel('A','Paired-receptor recovery across representative adaptive immune-cell states',1,1,(50,61,554,418),(24,55,564,655),remove=['Paired-receptor recovery across representative adaptive immune-cell states'])],
 [panel('B','Canonical marker validation of adaptive immune-cell states',1,2,(29,58,551,363),(24,55,552,655))],
]
# Restore the marker-size legend in a reserved band; the clipped source colorbar label is added below.
pages[1][1][0]['marker_legend']=True
pages[2]=[
 [panel('A','Participant-level repertoire diversity',2,1,(14,49,312,255),(24,55,335,292)),
  panel('B','Clone-size architecture',2,1,(331,49,525,266),(377,55,211,292)),
  panel('C','Cell-state composition of exact paired alpha-beta clonotypes',2,1,(0,300,516,453),(24,333,564,195),remove=['Cell-state composition of exact paired αβ clonotypes']),
  panel('D','Independent TCR motif context',2,2,(12,30,275,247),(170,538,273,230),remove=['D'])],
]
pages[3]=[[
 P('A','Baseline UC versus healthy',3,1,(55,145,678,640),REG(L,TOP)),
 P('B','Week 6 change from baseline',3,1,(686,145,1305,640),REG(R,TOP)),
 P('C','Clone-size dose response',3,1,(55,687,678,1192),REG(L,LOW)),
 external_heatmap(REG(R,LOW)),
]]
pages[4]=[
 [P('A','Participant-level repertoire diversity',4,1,(47,135,825,715),(24,55,335,292)),
  P('B','Clone-size architecture',4,1,(912,139,1386,731),(377,55,211,292)),
  P('C','B-cell-state composition of exact paired heavy-light clonotypes',4,1,(4,839,1330,1248),(24,376,564,345),remove=['B-cell state composition of exact paired heavy–light clonotypes'])],
 [P('D',"Crohn's disease: expansion-linked B-cell programs",4,2,(40,105,1228,746),(24,55,564,350),remove=['D','E','F','G']),
  P('E','Ulcerative colitis: expansion-linked B-cell programs',4,2,(40,842,1228,1482),(24,414,564,350),remove=['D','E','F','G'])],
 [P('F','Class-switched B cells',4,3,(12,110,427,372),(24,55,273,190),remove=['F','Class-switched B cells']),
  P('G','Heavy-chain isotype composition',4,3,(551,119,1224,466),(315,55,273,190)),
  P('','Participant-level isotype fractions (G)',4,3,(43,477,1224,876),(24,259,564,229)),
  P('H','Somatic hypermutation across B-cell states',4,3,(15,966,1230,1478),(24,506,564,264),remove=['H','Somatic hypermutation across B-cell states'])],
]
pages[5]=[[
 P('A','Regional heavy-chain somatic hypermutation',5,1,(102,130,719,629),REG(L,TOP)),
 P('B','Participant-level lineage diversification',5,1,(850,130,1459,629),REG(R,TOP)),
 P('C','Lineage-definition sensitivity',5,1,(95,670,785,1202),REG(L,LOW),remove=['Lineage-definition sensitivity']),
 P('D','Cross-state lineage occupancy',5,1,(860,681,1461,1180),REG(R,LOW)),
]]
pages[6]=[
 [panel('A','State-residualized B-cell programs and lineage maturation',6,1,(20,55,562,479),(24,55,564,650))],
 [P('B','Objective inflammation within IBD',6,2,(55,130,677,713),REG(L,TOP),remove=['All FDR >= 0.79']),
  P('C','Continuous intestinal inflammatory burden',6,2,(780,130,1391,714),REG(R,TOP),remove=['UC CDR targeting: FDR=0.028']),
  P('D','Expanded-lineage mucosal-state context',6,2,(30,905,677,1486),REG(L,LOW),remove=['IgA plasma: CD and UC FDR < 0.025']),
  P('E','Paired light-chain context',6,2,(780,905,1393,1486),REG(R,LOW),remove=['UC IGKV1: FDR=0.026'])],
]
for spec,note in zip(pages[6][1],['All FDR >= 0.79','UC CDR targeting: FDR=0.028','IgA plasma: CD and UC FDR < 0.025','UC IGKV1: FDR=0.026']):spec['note']=note
pages[7]=[
 [heatmap_panel('S7A',(24,55,564,304)),
  extpanel('B','TCR clonality versus BCR Gini','hi1',(99,365,309,599),(24,410,273,330)),
  extpanel('C','TCR Gini versus BCR Gini','hi1',(312.24,365,522.24,599),(315,410,273,330))],
 [extpanel('D','TCR Gini versus class-switched BCR fraction','hi1',(525.48,365,735.48,599),(24,55,273,330)),
  extpanel('E',"Cytotoxic T-cell and plasma-cell programs covary selectively in Crohn's disease",'hi1b',(0,25,764.39952,461.99952),(24,408,564,360))],
 [P('F','Model variance and incremental information',7,2,(89,132,580,454),(24,55,273,222)),
  P('G','Acquisition-series influence',7,2,(681,132,1168,454),(315,55,273,222)),
  P('H','Orthogonal cell-composition validation',7,2,(34,607,593,946),(24,298,273,222)),
  P('I','Contemporaneous clinical association',7,2,(751,607,1168,946),(315,298,273,222)),
  P('J',"Cross-program specificity in Crohn's disease",7,2,(99,1093,581,1477),(24,541,273,227)),
  P('K','Formal between-program contrasts',7,2,(674,1093,1168,1433),(315,541,273,227))],
]
pages[8]=[
 [P('A','Th17/Treg clone-state composition',8,1,(143,113,746,757),REG(L,TOP)),
  P('B','Exact-clone sharing null',8,1,(907,113,1321,757),REG(R,TOP)),
  P('C','Clone-size dose response in IBD',8,1,(41,804,590,1484),REG(L,LOW)),
  P('D','State-matched expansion',8,1,(744,804,1322,1484),REG(R,LOW))],
 [heatmap_panel('S8E',(24,55,564,313)),
  P('F','Independent helper model',8,2,(720,121,1300,799),(24,398,273,369)),
  P('G','Strongest prespecified association:\nPathogenic Th17 versus plasma differentiation',8,2,(115,905,483,1471),(315,398,273,369))],
 [P('H','Acquisition-series robustness',8,2,(611,905,1265,1471),(24,55,564,292)),
  panel('I','Helper-B-cell interaction estimates across shared acquisition series',8,3,(14,35,450,327),(24,382,564,386))],
]
pages[9]=[[
 P('A','Rectal Th17/Treg clone expansion',9,1,(187,145,778,733),REG(L,TOP)),
 P('B','Exact paired clones shared with blood',9,1,(833,145,1433,656),REG(R,TOP)),
 P('C','BCR expansion-program kinetics',9,1,(51,778,778,1325),REG(L,LOW)),
 P('D','Blood-shared versus colon-private clones',9,1,(780,778,1433,1325),REG(R,LOW)),
]]
pages[10]=[[
 P('A','Fully nested participant-level validation',10,1,(28,89,758,650),REG(L,TOP)),
 P('B','Permutation-derived null distributions',10,1,(749,89,1469,650),REG(R,TOP),masks=[(748,88,762,540)]),
 P('C','Calibration: CD versus control',10,1,(74,712,747,1232),REG(L,LOW)),
 P('D','Apparent model-selection optimism',10,1,(745,712,1438,1265),REG(R,LOW)),
]]

# Reposition source labels that were outside their original plot bounds.
def b_gini_label(c):
    c.setFont('Arial',9);c.saveState();c.translate(7,148);c.rotate(90);c.drawCentredString(0,0,'BCR Gini');c.restoreState()
pages[7][0][1]['src'].merge_page(overlay(b_gini_label,210,234))
esrc=crop(extra['hi1b'].pages[0],(0,25,764.39952,461.99952),remove=['IgA mucosal plasma-cell module','Plasmablast/plasma-cell differentiation module'])
def e_labels(c):
    c.setFont('Arial',9)
    for text,center in [('IgA mucosal plasma-cell module',132),('Plasmablast/plasma-cell differentiation module',345)]:
        c.saveState();c.translate(17,461.99952-center);c.rotate(90);c.drawCentredString(0,0,text);c.restoreState()
esrc.merge_page(overlay(e_labels,764.39952,436.99952))
pages[7][1][1]['src']=esrc

def compose(n,groups):
    writer=PdfWriter()
    for pi,panels in enumerate(groups,1):
        page=writer.add_blank_page(width=W,height=H)
        def header(c):
            c.setFillColorRGB(.08,.08,.08);c.setFont('Arial-Bold',10)
            c.drawString(M,H-25,f'Figure S{n}. {TITLES[n]}')
            c.setFillColorRGB(.38,.38,.38);c.setFont('Arial',8)
            c.drawRightString(W-M,H-25,f'{pi}/{len(groups)}')
            c.setStrokeColorRGB(.8,.8,.8);c.setLineWidth(.5);c.line(M,H-38,W-M,H-38)
        page.merge_page(overlay(header))
        for spec in panels:
            x,top,w,h=spec['slot'];lines=wrap(spec['title'],w-22)
            title_h=max(26,12*len(lines)+9)
            if spec.get('marker_legend'):title_h=45
            if spec.get('note'):title_h=42
            pw,ph=float(spec['src'].mediabox.width),float(spec['src'].mediabox.height)
            scale=min(w/pw,(h-title_h)/ph)
            px=x+(w-pw*scale)/2;py=H-top-title_h-ph*scale
            page.merge_transformed_page(spec['src'],Transformation().scale(scale).translate(px,py))
            if spec.get('marker_legend'):
                legend=source(1,2,(444,35,516,49))
                page.merge_transformed_page(legend,Transformation().scale(scale).translate(x+w-90,H-top-title_h+7))
            def label(c):
                c.setFillColorRGB(.07,.07,.07)
                if spec['letter']:
                    c.setFont('Arial-Bold',12);c.drawString(x,H-top-10,spec['letter'])
                c.setFont('Arial-Bold',8.5)
                for j,line in enumerate(lines):c.drawString(x+20,H-top-9-12*j,line)
                if spec.get('note'):
                    c.setFillColorRGB(.4,.4,.4);c.setFont('Arial',7);c.drawString(x+20,H-top-27,spec['note'])
                if spec.get('marker_legend'):
                    c.saveState();c.setFont('Arial',7);c.translate(x+w+10,py+ph*scale/2);c.rotate(90);c.drawCentredString(0,0,'Scaled average expression');c.restoreState()
            page.merge_page(overlay(label))
            audit.append({'figure':n,'page':pi,'panel':spec['letter'],'title':spec['title'],'slot_points':spec['slot'],'art_scale':scale,'origin':spec['origin'],'source_crop':spec['crop']})
    writer.add_metadata({'/Title':f'Figure S{n}. {TITLES[n]}','/Subject':'Standardized supplementary panel geometry; data unchanged; 20 September 2026'})
    for p in writer.pages:p.compress_content_streams(level=6)
    with (OUT/f'Figure_S{n}.pdf').open('wb')as f:writer.write(f)
    print(f'Figure S{n}: {len(groups)} pages',flush=True)

for n,groups in pages.items():compose(n,groups)
book=PdfWriter()
for n in range(1,11):
    for p in PdfReader(OUT/f'Figure_S{n}.pdf').pages:book.add_page(p)
book.add_metadata({'/Title':'Supplementary Figures S1-S10 - standardized panel sizes','/Subject':'Includes revised S2A-D; scientific values and panel identities preserved'})
with (OUT/'Supplementary_Figures_S1-S10_Standardized.pdf').open('wb')as f:book.write(f)
shutil.copy2(SRC/'All_supplementary_figure_legends_S1-S10.txt',OUT/'All_supplementary_figure_legends_S1-S10.txt')
(OUT/'Panel layout specifications.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(OUT/'README.txt').write_text('Standardized supplementary figure set - 20 September 2026\n\nPage size: 612 x 792 pt (US Letter).\nMargins: 24 pt. Regular panel width: 273 pt; wide panel width: 564 pt; column gutter: 18 pt.\nRepertoire composite panels retain matched 335/211-pt columns in both S2 and S4.\nPanel letters: Arial Bold 12 pt. Panel titles: Arial Bold 8.5 pt. Figure headers: Arial Bold 10 pt.\nPlots are scaled proportionally; axes, data, uncertainty intervals, and statistical values are preserved.\nS3D, S7A, and S8E heatmaps are redrawn from their existing source tables to give numbers and labels adequate space; no analyses are recomputed.\nS7 uses three pages to avoid shrinking its dense panels. Panel identities and figure legends are unchanged.\nS2 remains A-D; removed panels are not restored.\nSource versions are archived separately.\n',encoding='utf-8')
expected={1:'AB',2:'ABCD',3:'ABCD',4:'ABCDEFGH',5:'ABCD',6:'ABCDE',7:'ABCDEFGHIJK',8:'ABCDEFGHI',9:'ABCD',10:'ABCD'}
for n,letters in expected.items():
    assert ''.join(s['letter']for group in pages[n]for s in group)==letters
print('Complete:',OUT,flush=True)

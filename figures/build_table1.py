from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Table,TableStyle,Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from pypdf import PdfReader
import pypdfium2
root=Path(__file__).resolve().parents[1];out=root/'tables/table1_pretraining_adaptation_standalone.pdf';W,H=210*mm,193*mm;L=13.5*mm;TW=183*mm
c=canvas.Canvas(str(out),pagesize=(W,H));c.setTitle('Table 1 | Pretraining, source replay and target-support efficiency')
head=colors.HexColor('#D9D8C8');tint=colors.HexColor('#EEEDE4');rule=colors.HexColor('#303030')
styles={k:ParagraphStyle(k,fontName=f,fontSize=sz,leading=lead) for k,f,sz,lead in [('title','Helvetica-Bold',10,12),('cell','Helvetica',7.5,9),('group','Helvetica-Bold',7.5,10),('note','Helvetica',7,9)]}
def P(s,k='cell'):return Paragraph(s,styles[k])
def drawpara(s,y,k):
 p=P(s,k);_,h=p.wrap(TW,999);p.drawOn(c,L,y-h);return y-h
Y=H-13*mm;Y=drawpara('Table 1 | Pretraining, source replay and target-support efficiency',Y,'title')-8
A=[['Plain gated TCN','Random',13.12,12.16,18.61,6.35],['Plain gated TCN','Four-source pretrained',7.02,11.61,14.11,6.12],['SpectroPhys','Random',9.39,12.12,10.56,11.77],['SpectroPhys','Four-source pretrained',2.05,3.59,6.32,4.45]]
B=[['Target-only adaptation','',7.57,9.21,10.96,6.13],['Target adaptation + source replay','',2.08,5.87,10.98,5.80],['Target adaptation + source replay + modulation','',2.14,5.81,10.84,12.58]]
C=[['25% of permitted target support','',2.08,6.50,10.73,9.85],['50% of permitted target support','',2.07,6.60,11.93,12.83],['75% of permitted target support','',2.11,5.76,11.92,16.42],['100% of permitted target support','',2.10,5.83,10.84,6.03]]
D=[]
for label,reference,result in [('Plain gated TCN: pretraining vs random',A[0],A[1]),('SpectroPhys: pretraining vs random',A[2],A[3]),('Pretrained SpectroPhys vs pretrained TCN',A[1],A[3])]:
 D.append([label,'']+[(x-y)/x*100 for x,y in zip(reference[2:],result[2:])])
E=[]
for label,reference,result in [('Adding replay vs target-only adaptation',B[0],B[1]),('Adding modulation vs replay alone',B[1],B[2]),('Replay + modulation vs target-only adaptation',B[0],B[2])]:
 E.append([label,'']+[(x-y)/x*100 for x,y in zip(reference[2:],result[2:])])
rows=[['Experiment','Model / condition','Initialization','Adult ICU (ZPU)','','Neonatal ICU (KWH)',''],['','','','HR','RR','HR','RR']]
for group,data in [('a  Pretraining<br/>attribution',A),('b  Adaptation<br/>components',B),('c  Target-support<br/>efficiency',C),('d  Pretraining gain<br/>(% reduction)',D),('e  Adaptation gain<br/>(% reduction)',E)]:
 for i,row in enumerate(data):rows.append([P(group,'group') if i==0 else '',P(row[0]),P(row[1])]+[(f'{v:.1f}%' if data is D or data is E else f'{v:.2f}') for v in row[2:]])
widths=[26*mm,33*mm,48*mm]+[19*mm]*4
T=Table(rows,colWidths=widths,rowHeights=[17,17]+[16]*17)
cmd=[('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),7.5),('LEADING',(0,0),(-1,-1),9),('FONTNAME',(0,0),(-1,1),'Helvetica-Bold'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(3,0),(-1,1),'CENTER'),('ALIGN',(3,2),(-1,-1),'RIGHT'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),5),('BACKGROUND',(0,0),(-1,1),head),('BACKGROUND',(0,6),(-1,8),tint),('LINEABOVE',(0,0),(-1,0),.6,rule),('LINEBELOW',(0,1),(-1,1),.6,rule),('LINEBELOW',(0,-1),(-1,-1),.6,rule)]
pad=(19*mm-stringWidth('00.00','Helvetica',7.5))/2
cmd += [('RIGHTPADDING',(3,2),(-1,-1),pad),('LEFTPADDING',(3,2),(-1,-1),pad),('BACKGROUND',(0,13),(-1,15),tint)]
for j in [0,1,2]:cmd.append(('SPAN',(j,0),(j,1)))
for j in [3,5]:cmd.extend([('SPAN',(j,0),(j+1,0)),('LINEBELOW',(j,0),(j+1,0),.3,rule)])
start=2
for group,data in [('a',A),('b',B),('c',C),('d',D),('e',E)]:
 end=start+len(data)-1;cmd.append(('SPAN',(0,start),(0,end)))
 for i in range(start,end):cmd.append(('LINEBELOW',(1,i),(-1,i),.25,rule))
 if end<18:cmd.append(('LINEBELOW',(0,end),(-1,end),.5,rule))
 if group!='a':
  for i in range(start,end+1):cmd.append(('SPAN',(1,i),(2,i)))
 for j in range(2,6):
  m=min(x[j] for x in data)
  for i,row in enumerate(data):
   if group not in ['d','e'] and row[j]==m:cmd.append(('FONTNAME',(j+1,start+i),(j+1,start+i),'Helvetica-Bold'))
 start=end+1
T.setStyle(TableStyle(cmd));_,h=T.wrap(TW,999);T.drawOn(c,L,Y-h);Y-=h+7
notes=[
'Blocks a-c report MAE (HR, beats min<super>-1</super>; RR, breaths min<super>-1</super>); bold marks each block\'s minimum. Blocks d-e show 100 &times; (reference - comparison) / reference, calculated from the reported MAEs; negative values indicate higher error.',
'Block a uses ten training clips, with unified adaptation on ZPU and target-specialized adaptation on KWH. Blocks b-c are separate single-seed analyses; c uses fractions of the full permitted training manifest. Validation and test sets are fixed within each analysis. Modulation is dataset-conditioned.'
]
for s in notes:Y=drawpara(s,Y,'note')-3
assert Y>9*mm,Y
c.save();reader=PdfReader(out);assert len(reader.pages)==1
text=reader.pages[0].extract_text()
for data in [A,B,C]:
 for row in data:
  for v in row[2:]:assert f'{v:.2f}' in text
                                                                               
source=(root/'tables/table_mechanistic_integrated.tex').read_text()
for data in [A,B,C]:
 for row in data:assert ' & '.join(f'{v:.2f}' for v in row[2:]) in source
pdf=pypdfium2.PdfDocument(str(out));pdf[0].render(scale=2.5).to_pil().save(root/'tables/table1_pretraining_adaptation_preview.png')
print('17 rows; 44 source values plus 24 calculated reductions. Bottom margin',round(Y/mm,1),'mm.');print(E)

                                                                
from pypdf import PdfWriter
page=PdfReader(out).pages[0]
page.cropbox.lower_left=(13.2*mm,Y-2)
page.cropbox.upper_right=(196.8*mm,181*mm)
page.mediabox.lower_left=page.cropbox.lower_left
page.mediabox.upper_right=page.cropbox.upper_right
writer=PdfWriter();writer.add_page(page)
with (root/'tables/table1_pretraining_adaptation.pdf').open('wb') as handle:writer.write(handle)

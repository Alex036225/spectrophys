import pymupdf as f,numpy as np,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def setup(name):
 p=f.open(root/name)[0]; ds=p.get_drawings()
 rs=[d['rect'] for d in ds if d['fill']==(1,1,1) and d['rect'].width>80 and d['rect'].height>60][1:]
 return p,ds,rs

def calibration(p,ds,r,axis):
 ticks=[]
 for d in ds:
  for it in d['items']:
   if it[0]!='l': continue
   a,b=it[1:]
   if axis=='y' and abs(a.y-b.y)<.01 and abs(a.x-r.x0)<.1 and 2<abs(a.x-b.x)<5 and r.y0-1<a.y<r.y1+1:ticks.append(a.y)
   if axis=='x' and abs(a.x-b.x)<.01 and abs(a.y-r.y1)<.1 and 2<abs(a.y-b.y)<5 and r.x0-1<a.x<r.x1+1:ticks.append(a.x)
 vals=[]
 for w in p.get_text('words'):
  try:v=float(w[4])
  except:continue
  if axis=='y' and r.x0-30<w[2]<r.x0 and r.y0-8<(w[1]+w[3])/2<r.y1+8: pos=(w[1]+w[3])/2
  elif axis=='x' and r.y1<w[1]<r.y1+10 and r.x0-8<(w[0]+w[2])/2<r.x1+8: pos=(w[0]+w[2])/2
  else:continue
  if ticks:
   t=min(ticks,key=lambda x:abs(x-pos))
   if abs(t-pos)<5: vals.append((t,v))
 assert len(vals)>=2,(r,axis,ticks,vals)
 a,b=np.polyfit(*np.array(vals).T,1)
 assert max(abs(a*x+b-y) for x,y in vals)<1e-4
 return lambda x:float(a*x+b)

p,ds,rs=setup('fig13_robustness_stability_transfer.pdf');adapt=[]
methods=['PhysNet','DeepPhys','PhysFormer','RhythmFormer','BigSmall','EfficientPhys','MultiPhysNet','SpectroPhys']
for row in range(3):
 r=rs[row*3+1]; conv=calibration(p,ds,r,'x'); bars=[]
 for d in ds:
  dr=d['rect']; c=d['color']
  if r.contains(dr) and c and c!=(0,0,0) and len(d['items'])==1 and d['items'][0][0]=='l' and dr.height<.01 and 0<dr.width<100:
   bars.append((dr.y0,(conv(dr.x0)+conv(dr.x1))/2,abs(conv(dr.x1)-conv(dr.x0))/2))
 bars.sort();assert len(bars)==8,len(bars)
 r=rs[row*3+2];conv=calibration(p,ds,r,'y');pairs=[]
 for d in ds:
  dr=d['rect'];c=d['color']
  if r.contains(dr) and c and len(d['items'])==1 and d['items'][0][0]=='l' and 99<dr.width<102:
   it=d['items'][0];pairs.append([conv(it[1].y),conv(it[2].y)])
 assert len(pairs)==8
 adapt.append({'cohort':['ZPU','KWH','SZC'][row],'mean':[x[1] for x in bars],'sd':[x[2] for x in bars],'pairs':pairs})

p,ds,rs=setup('fig15_apnea_low_label.pdf');apnea=[]
for i,r in enumerate(rs):
 cv=calibration(p,ds,r,'y');curves=[];bands=[]
 for d in ds:
  dr=d['rect'];items=d['items']
  if r.contains(dr) and len(items)==3 and all(it[0]=='l' for it in items) and dr.width>200:
   pts=[items[0][1]]+[it[2] for it in items]
   curves.append({'color':d['color'],'y':[cv(v.y) for v in pts]})
  if r.contains(dr) and d['fill'] and d.get('fill_opacity',1)<.5 and dr.width>200:
   pts=[]
   for it in items:
    if it[0]=='l':pts.extend(it[1:])
   xs=sorted(set(round(pt.x,2) for pt in pts));bounds=[]
   for x in xs:
    ys=[cv(pt.y) for pt in pts if abs(pt.x-x)<.02];bounds.append([min(ys),max(ys)])
   bands.append({'color':d['fill'],'bounds':bounds})
 assert len(curves)==12,(i,len(curves))
 assert len(bands)==3,(i,len(bands))
 apnea.append({'metric':['AUC','AUPR','F1','Precision','Sensitivity','Specificity'][i],'curves':curves,'bands':bands})

p,ds,rs=setup('fig18_corruption_severity.pdf');corruption=[]
for r in rs[:3]:
 cv=calibration(p,ds,r,'y'); curves=[]
 for d in ds:
  items=d['items'];dr=d['rect']
  if r.contains(dr) and len(items)==4 and all(it[0]=='l' for it in items) and dr.width>180:
   curves.append({'color':d['color'],'y':[cv(items[0][1].y)]+[cv(it[2].y) for it in items]})
 assert len(curves)==8
 corruption.append(curves)
                                                           
r=rs[3]; cv=calibration(p,ds,r,'x'); overall=[]
for d in ds:
 dr=d['rect']
 if r.contains(dr) and d['fill'] and 10<dr.height<20 and dr.width>20: overall.append((dr.y0,cv(dr.x1)))
overall.sort();assert len(overall)==8
r=rs[4];heat=[]
for w in p.get_text('words'):
 if r.contains(f.Rect(w[:4])):
  try:v=float(w[4])
  except:continue
  heat.append((w[1],w[0],v))
heat.sort();assert len(heat)==30
result={'provenance':'Plotting values recovered from vector coordinates in the original project figures; approximate numerical reconstruction, not participant-level source data. Unmodified original PDFs retained. No new observations or uncertainty estimated.', 'methods':methods,'adaptation':adapt,'apnea':apnea,'corruption':corruption,'overall':[v for y,v in overall],'overall_methods':['SpectroPhys','PhysFormer','MultiPhysNet','BigSmall','RhythmFormer','PhysNet','EfficientPhys','DeepPhys'],'heatmap':np.array([x[2] for x in heat]).reshape(6,5).tolist()}
(root/'data/extended_data_plot_values.json').write_text(json.dumps(result,indent=2))
print('Extracted',len(adapt),'cohorts,',len(apnea),'apnea metrics,',len(corruption),'corruption cohorts')
print('SpectroPhys support differences',np.mean([a['pairs'][-1][0]-a['pairs'][-1][1] for a in adapt]))
print('Overall first two',result['overall'][:2])

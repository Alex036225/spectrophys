from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
root=Path(__file__).resolve().parents[1];out=root/'figure4_exports';out.mkdir(exist_ok=True)
data=json.loads((root/'data/figure4_plot_values.json').read_text());v={(r['method'],r['metric']):r['value'] for r in data['records']}
methods=sorted({r['method'] for r in data['records']},key=lambda m:v[m,'KWH HR MAE'])
tasks=['KWH HR','KWH RR','SZC HR'];orders=[sorted([m for m in methods if (m,t+' MAE') in v],key=lambda m:v[m,t+' MAE']) for t in tasks]
rank=[{m:i+1 for i,m in enumerate(order)} for order in orders]
ink='#283541';orange='#C96045';colours={'SpectroPhys':orange,'POS':'#397F9C','DeepPhys':'#429A8B','BigSmall':'#8C6BA6'}
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(183/25.4,220/25.4),facecolor='white')
fig.text(.025,.983,'Neonatal transfer across tasks and cohorts',fontsize=11,fontweight='bold',color=ink,va='top')
fig.text(.025,.929,'a',fontsize=10,fontweight='bold',color=ink)
ax=fig.add_axes([.205,.635,.580,.245]);ax.set_xlim(-.04,2.04);ax.set_ylim(14.7,.3);ax.axis('off')
for j,title in enumerate(['KWH · heart rate','KWH · respiratory rate','SZC · heart rate']):
 ax.text(j,-.35,title,ha='center',va='bottom',fontsize=8.5,fontweight='bold',color=ink)
 ax.plot([j,j],[1,14],color='#E5E9EF',lw=1,zorder=0)
for m in sorted(methods,key=lambda m:m in colours):
 col=colours.get(m,'#B7C1CF');highlight=m in colours
 for j in range(2):
  if m not in rank[j] or m not in rank[j+1]:continue
  y0,y1=rank[j][m],rank[j+1][m]
  path=MPath([(j+(.13 if j==1 else 0),y0),(j+.38,y0),(j+.62,y1),(j+1,y1)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
  ax.add_patch(PathPatch(path,facecolor='none',edgecolor=col,lw=1.7 if highlight else .9,alpha=.95 if highlight else .55,zorder=2 if highlight else 1))
 for j in range(3):
  if m not in rank[j]:continue
  ax.scatter(j,rank[j][m],s=27 if highlight else 16,color=col,edgecolor='white',lw=.6,zorder=4)
 for j,ha,dx in [(0,'right',-.085),(2,'left',.085)]:
  ax.text(j+dx,rank[j][m],f'{rank[j][m]:02d}  {m}',ha=ha,va='center',fontsize=7.5,color=col if highlight else ink,fontweight='bold' if m=='SpectroPhys' else 'normal')
for m,r in rank[1].items():
 ax.text(1+.045,r,f'{r:02d}',ha='left',va='center',fontsize=7,color=colours.get(m,'#6D7887'),bbox=dict(facecolor='white',edgecolor='none',pad=.4),zorder=5)
                                                             
summary=[];axes=[]
from matplotlib.ticker import MaxNLocator
for i,(task,title,col) in enumerate(zip(tasks,['KWH · heart rate','KWH · respiratory rate','SZC · heart rate'],['#578AA5','#60A593','#947DB1'])):
 x=.080+i*.320
 fig.text(x,.590,title,fontweight='bold',fontsize=8.5,color=ink)
 for j,k in enumerate(['MAE','RMSE','MAPE']):
  metric=task+' '+k;y=.425-j*.175;ac=fig.add_axes([x,y,.270,.120]);axes.append(ac)
  maximum=max(value for (m,key),value in v.items() if key==metric)
  best=min((m for m in methods if m!='SpectroPhys' and (m,metric) in v),key=lambda m:v[m,metric]);base=v[best,metric];ours=v['SpectroPhys',metric]
  summary.append(dict(metric=metric,baseline=best,baseline_error=base,spectrophys_error=ours,reduction_percent=100*(base-ours)/base))
  for row,m in enumerate(methods):
   if (m,metric) not in v:continue
   value=v[m,metric]
   ac.bar(row,value,width=.73,color=orange if m=='SpectroPhys' else col,alpha=1 if m in ['SpectroPhys',best] else .55,edgecolor=ink if m==best else 'none',linewidth=.6)
   if m=='SpectroPhys':ac.text(.03,1.025,f'{value:.2f}',transform=ac.transAxes,ha='left',va='bottom',fontsize=7.5,color=orange,fontweight='bold')
  ac.set_xlim(-.8,13.8);ac.set_ylim(0,maximum*1.18)
  ac.yaxis.set_major_locator(MaxNLocator(3));ac.tick_params(length=2,width=.5)
  ac.set_xticks(range(14),[str(n+1) for n in range(14)] if j==2 else ['']*14);ac.tick_params(axis='x',labelsize=7,length=0)
  if j==2:ac.set_xlabel('Method',labelpad=4)
  unit='%' if k=='MAPE' else 'bpm' if 'HR' in task else 'breaths/min'
  ac.set_ylabel(f'{k} ({unit})',fontsize=7.5,labelpad=3)
  ac.text(-.13,1.10,chr(98+j*3+i),transform=ac.transAxes,fontweight='bold',fontsize=10,color=ink)
fig.savefig(out/'figure4-redesigned.pdf');fig.savefig(out/'figure4-redesigned.svg');fig.savefig(out/'figure4-redesigned.png',dpi=300)
(root/'fig4_kwh_szc_benchmark.pdf').write_bytes((out/'figure4-redesigned.pdf').read_bytes())
(out/'derived-summary.json').write_text(json.dumps(summary,indent=2))

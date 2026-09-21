"""Complete model resource comparison under the manuscript's fixed protocol.
Contract: separate measured inference latency from parameter and FLOP budgets.
All eight models and 24 original measurements are retained. No uncertainty,
throughput, or adaptation-cost measurements are inferred. Quantitative grid,
156 mm nominal width, editable PDF/SVG, zero-origin linear axes.
"""
from pathlib import Path
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'data/extended_data_qa'
rows=[('PhysNet',.81,22.13,2.31),('DeepPhys',2.27,36.35,3.23),('PhysFormer',7.42,15.70,9.50),('RhythmFormer',3.37,11.50,6.28),('BigSmall',2.18,24.56,2.68),('EfficientPhys',2.20,18.37,2.08),('MultiPhysNet',.92,22.14,2.35),('SpectroPhys',5.25,11.75,1.71)]
with (ROOT/'data/model_efficiency.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['Method','Parameters_M','GFLOPs','Latency_ms']);w.writerows(rows)
rows=sorted(rows,key=lambda x:x[3])
red='#C75D63';blue='#82A9BD';ink='#293641'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.55,'text.color':ink,'axes.labelcolor':ink,'xtick.color':ink,'ytick.color':ink})
fig,axs=plt.subplots(1,3,figsize=(6.14,3.7),sharey=True)
fig.subplots_adjust(left=.185,right=.985,bottom=.20,top=.85,wspace=.14)
                                                                               
for j,(ax,col,title,label,xmax,ticks) in enumerate(zip(axs,[3,1,2],['Inference latency','Model size','Computation'],['Latency (ms)','Parameters (M)','GFLOPs'],[12.5,10,48],[[0,3,6,9],[0,2,4,6,8],[0,10,20,30]])):
 ax.set_xlim(0,xmax);ax.set_ylim(7.65,-.8)
 for i,row in enumerate(rows):
  c=red if row[0]=='SpectroPhys' else blue
  if i%2==0:ax.axhspan(i-.46,i+.46,color='#F5F7F9',lw=0,zorder=0)
  if i==0:ax.axhspan(-.46,.46,color='#F9ECEB',lw=0,zorder=1)
  v=row[col]
  ax.barh(i,v,height=.23,color=c,alpha=.8,zorder=3)
  ax.scatter(v,i,s=25 if i==0 else 18,color=c,edgecolor='white',linewidth=.6,zorder=4)
  ax.text(xmax*.985,i,f'{v:.2f}',ha='right',va='center',fontsize=6.8,color=red if i==0 else ink,fontweight='bold' if i==0 else 'normal',zorder=5)
 ax.set_xticks(ticks);ax.set_xlabel(label,labelpad=7)
 ax.tick_params(axis='both',length=2.5,width=.5);ax.tick_params(axis='y',length=0,pad=7)
 for name in ['top','right','left']:ax.spines[name].set_visible(False)
 ax.spines['bottom'].set_color('#ACB7C0');ax.set_yticks(range(8),[r[0] for r in rows])
 ax.text(0,1.07,chr(97+j),transform=ax.transAxes,fontsize=9,fontweight='bold',ha='left')
 ax.text(.10,1.07,title,transform=ax.transAxes,fontsize=7.2,ha='left')
 ax.set_axisbelow(True);ax.grid(axis='x',color='#E4E9ED',lw=.4,zorder=-1)
for t in axs[0].get_yticklabels():
 if t.get_text()=='SpectroPhys':t.set_color(red);t.set_fontweight('bold')
fig.legend([Line2D([],[],marker='o',lw=0,color=c,markersize=4) for c in [red,blue]],['SpectroPhys','Neural baselines'],loc='lower center',bbox_to_anchor=(.56,.018),ncol=2,frameon=False,columnspacing=2,handletextpad=.35,fontsize=6.5)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=list(axs),panel_ids=list('abc'),json_out=QA/'model_efficiency.alignment.json',strict=True)
fig.savefig(ROOT/'extended_data_model_efficiency.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(ROOT/'extended_data_model_efficiency.svg',bbox_inches='tight',pad_inches=.04)
fig.savefig(QA/'model_efficiency.tiff',dpi=600,bbox_inches='tight',pad_inches=.04)
fig.savefig(QA/'model_efficiency.png',dpi=400,bbox_inches='tight',pad_inches=.04)

"""Complete out-of-fold apnea benchmark, transcribed from the manuscript table.
Contract: compare discrimination (AUC), precision–recall performance (AUPR),
and thresholded classification (F1). Keep all 12 methods and 36 point estimates.
Quantitative grid, 156 mm width, editable PDF/SVG. No fold-level uncertainty
is available in the source table; do not synthesize error bars or ROC curves.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'data/extended_data_qa'
rows=[('GREEN',.620,.639,.595),('ICA',.525,.608,.513),('CHROM',.405,.435,.400),('POS',.530,.563,.591),('PhysNet',.728,.769,.683),('DeepPhys',.640,.690,.619),('PhysFormer',.723,.695,.588),('RhythmFormer',.690,.710,.611),('BigSmall',.762,.719,.732),('EfficientPhys',.635,.626,.564),('MultiPhysNet',.578,.556,.619),('SpectroPhys',.800,.784,.734)]
with (ROOT/'data/apnea_benchmark.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['Method','AUC','AUPR','F1']);w.writerows(rows)
rows=sorted(rows,key=lambda x:x[1],reverse=True)
red='#C75D63';blue='#82A9BD';grey='#B6BDC4';ink='#293641'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.55,'text.color':ink,'axes.labelcolor':ink,'xtick.color':ink,'ytick.color':ink})
fig,axs=plt.subplots(1,3,figsize=(6.14,4.6),sharey=True)
fig.subplots_adjust(left=.185,right=.985,bottom=.18,top=.87,wspace=.14)
y=np.arange(12)
for j,(ax,metric,title) in enumerate(zip(axs,['AUC','AUPR','F1'],['Discrimination','Precision–recall','Classification'])):
 ax.set_xlim(0,1);ax.set_ylim(11.7,-.9)
 for i,row in enumerate(rows):
  c=red if row[0]=='SpectroPhys' else grey if row[0] in ['GREEN','ICA','CHROM','POS'] else blue
  if i%2==0:ax.axhspan(i-.47,i+.47,color='#F5F7F9',lw=0,zorder=0)
  if i==0:ax.axhspan(-.47,.47,color='#F9ECEB',lw=0,zorder=1)
  v=row[j+1]
  ax.barh(i,v,height=.25,color=c,alpha=.8,zorder=3)
  ax.scatter(v,i,s=24 if i==0 else 17,color=c,edgecolor='white',linewidth=.6,zorder=4)
  ax.text(.985,i,f'{v:.3f}',ha='right',va='center',fontsize=6.8,color=red if i==0 else ink,fontweight='bold' if i==0 else 'normal',zorder=5)
 ax.set_xticks([0,.25,.5,.75]);ax.set_xticklabels(['0','0.25','0.50','0.75']);ax.set_xlabel(metric,labelpad=7)
 ax.tick_params(axis='both',length=2.5,width=.5);ax.tick_params(axis='y',length=0,pad=7)
 for name in ['top','right','left']:ax.spines[name].set_visible(False)
 ax.spines['bottom'].set_color('#ACB7C0')
 ax.set_yticks(y,[r[0] for r in rows])
 ax.text(0,1.055,chr(97+j),transform=ax.transAxes,fontsize=9,fontweight='bold',ha='left')
 ax.text(.10,1.055,title,transform=ax.transAxes,fontsize=7.2,ha='left')
 ax.set_axisbelow(True);ax.grid(axis='x',color='#E4E9ED',lw=.4,zorder=-1)
for t in axs[0].get_yticklabels():
 if t.get_text()=='SpectroPhys':t.set_color(red);t.set_fontweight('bold')
fig.legend([Line2D([],[],marker='o',lw=0,color=c,markersize=4) for c in [red,blue,grey]],['SpectroPhys','Neural baselines','Classical methods'],loc='lower center',bbox_to_anchor=(.56,.025),ncol=3,frameon=False,columnspacing=1.5,handletextpad=.35,fontsize=6.5)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=list(axs),panel_ids=list('abc'),json_out=QA/'apnea_benchmark.alignment.json',strict=True)
fig.savefig(ROOT/'extended_data_apnea_benchmark.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(ROOT/'extended_data_apnea_benchmark.svg',bbox_inches='tight',pad_inches=.04)
fig.savefig(QA/'apnea_benchmark.tiff',dpi=600,bbox_inches='tight',pad_inches=.04)
fig.savefig(QA/'apnea_benchmark.png',dpi=400,bbox_inches='tight',pad_inches=.04)

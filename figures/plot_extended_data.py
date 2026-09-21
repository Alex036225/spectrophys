"""Nature-style Extended Data: explicit figure-level evidence and restrained colour.
Read reconstructed plotting values; never synthesize observations or uncertainty.
Run with the nature-figure QA scripts directory on PYTHONPATH.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'data'/'extended_data_qa';QA.mkdir(exist_ok=True)
D=json.loads((ROOT/'data/extended_data_plot_values.json').read_text())
ORANGE='#C75D63';TEAL='#5799C6';PURPLE='#A28AB8';GREY='#B9C5CF';INK='#222222'
METHOD_COLORS=dict(zip(D['methods'],['#76A9CB','#82B6A0','#AC9CC1','#C5B878','#E2AA80','#96BFC4','#A7ACB5',ORANGE]))
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
 'axes.titlesize':7.5,'legend.fontsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55,
 'axes.edgecolor':'#88929A','text.color':INK,'axes.labelcolor':INK,
 'xtick.color':INK,'ytick.color':INK,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'savefig.dpi':400})

def head(ax,letter,title):
 ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-7,8),textcoords='offset points',fontsize=8,fontweight='bold',ha='right',va='bottom')
 ax.annotate(title,(0,1),xycoords='axes fraction',xytext=(0,8),textcoords='offset points',fontsize=7,fontweight='normal',ha='left',va='bottom')
def grid(ax,axis='y'):
 ax.set_axisbelow(True);ax.grid(axis=axis,color='#E6EAED',lw=.5)
def export(fig,name,**kw):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=QA/(name+'.alignment.json'),strict=True,**kw)
 fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
 fig.savefig(ROOT/(name+'.svg'),bbox_inches='tight',pad_inches=.035)
 fig.savefig(QA/(name+'.png'),dpi=400,bbox_inches='tight',pad_inches=.035)
 plt.close(fig)

                                                                              
fig=plt.figure(figsize=(6.14,3.85))
outer=fig.add_gridspec(2,1,left=.16,right=.965,bottom=.22,top=.91,hspace=.60,height_ratios=[1.13,1])
top=outer[0].subgridspec(1,2,wspace=.66)
bot=outer[1].subgridspec(1,3,wspace=.42)
a=fig.add_subplot(top[0]); b=fig.add_subplot(top[1]); axes=[a,b]
y=np.arange(8);values=np.array(D['overall'])
a.barh(y,values,height=.55,color=[METHOD_COLORS[n] for n in D['overall_methods']],zorder=2)
a.set_yticks(y,D['overall_methods']);a.invert_yaxis();a.set_xlim(0,11.8)
a.set_xticks([0,5,10]);a.set_xlabel('Mean absolute HR error change (bpm)',fontsize=6.5)
a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False)
for j,v in enumerate(values):a.text(v+.18,j,f'{v:.2f}',va='center',fontsize=6.1,color=ORANGE if j==0 else INK)
head(a,'a','Mean error change')
hm=np.array(D['heatmap']);cm=LinearSegmentedColormap.from_list('error',['#F7FBFD','#D9EAF3','#8CBBD3','#376E9A'])
b.imshow(hm,aspect='auto',cmap=cm,vmin=0,vmax=6.5)
b.set_xticks(range(5),range(1,6));b.set_yticks(range(6),['Brightness','Noise','Blur','JPEG','Frame drop','ROI shift'])
b.tick_params(length=0);b.set_xlabel('Corruption severity')
for spine in b.spines.values():spine.set_visible(False)
for row in range(6):
 for col in range(5):b.text(col,row,f'{hm[row,col]:.1f}',ha='center',va='center',fontsize=6.4,color='white' if hm[row,col]>4.3 else INK)
head(b,'b','SpectroPhys: corruption profile (bpm)')
for i,cohort in enumerate(['ZPU','KWH','SZC']):
 ax=fig.add_subplot(bot[i]);axes.append(ax)
 for j,q in enumerate(D['corruption'][i]):
  ax.plot(range(1,6),q['y'],color=METHOD_COLORS[['SpectroPhys','PhysNet','DeepPhys','PhysFormer','RhythmFormer','BigSmall','EfficientPhys','MultiPhysNet'][j]],lw=1.4 if j==0 else .7,marker=['o','s','^','D','v','p','h','X'][j],ms=2.7 if j==0 else 2,zorder=4 if j==0 else 2)
 ax.set_xlim(.8,5.2);ax.set_ylim(0,15);ax.set_xticks(range(1,6));ax.set_yticks([0,5,10,15]);grid(ax)
 ax.set_xlabel('Corruption severity');head(ax,chr(99+i),cohort)
 if i==0:ax.set_ylabel('Absolute HR error change (bpm)')
fig.legend([Line2D([],[],color=METHOD_COLORS[n],lw=1.2) for n in D['methods']],D['methods'],loc='lower center',bbox_to_anchor=(.53,.01),ncol=4,frameon=False,columnspacing=1.5,handlelength=1.6)
export(fig,'extended_data_corruption',axes=axes,panel_ids=list('abcde'))

                                                                            
                                                                             
oldcolors={'SpectroPhys':(.03137,.49804,.54902),'PhysNet':(.85098,.37255,.00784),'BigSmall':(.36863,.23529,.6)}
colors={n:METHOD_COLORS[n] for n in ['SpectroPhys','PhysNet','BigSmall']}
def identity(rgb):
 for name,col in oldcolors.items():
  if np.max(np.abs(np.array(rgb)-col))<.002:return name
 return None

def apnea_panel(ax,metric,letter):
 data=next(q for q in D['apnea'] if q['metric']==metric);x=np.array([25,50,75,100])
 for q in data['bands']:
  name=identity(q['color']);bounds=np.array(q['bounds']);assert bounds.shape==(4,2)
  ax.fill_between(x,bounds[:,0],bounds[:,1],color=colors[name],alpha=.10,lw=0,zorder=1)
 for q in data['curves']:
  name=identity(q['color']);ax.plot(x,q['y'],color=colors[name] if name else GREY,lw=1.6 if name=='SpectroPhys' else 1 if name else .6,marker='o' if name else None,ms=3 if name else 0,zorder=4 if name else 2)
 ax.set_xlim(21,104);ax.set_ylim(.35,.90);ax.set_xticks(x);ax.set_yticks([.4,.5,.6,.7,.8,.9]);grid(ax)
 ax.set_xlabel('Labelled training subjects (%)');ax.set_ylabel(metric);head(ax,letter,metric)

fig,axs=plt.subplots(1,3,figsize=(6.14,2.35));fig.subplots_adjust(left=.08,right=.98,bottom=.26,top=.78,wspace=.40)
for ax,metric,letter in zip(axs,['AUC','AUPR','F1'],'abc'):apnea_panel(ax,metric,letter)
handles=[Line2D([],[],color=colors[n],lw=1.5) for n in colors]+[Line2D([],[],color=GREY,lw=1)]
fig.legend(handles,list(colors)+['Other methods'],loc='lower center',bbox_to_anchor=(.53,-.005),ncol=4,frameon=False)
export(fig,'extended_data_apnea',axes=axs,panel_ids=list('abc'))
fig,axs=plt.subplots(1,3,figsize=(6.14,2.35));fig.subplots_adjust(left=.08,right=.98,bottom=.26,top=.78,wspace=.39)
for ax,metric,letter in zip(axs,['Precision','Sensitivity','Specificity'],'abc'):apnea_panel(ax,metric,letter)
fig.legend(handles,list(colors)+['Other methods'],loc='lower center',bbox_to_anchor=(.53,-.005),ncol=4,frameon=False)
export(fig,'extended_data_apnea_secondary',axes=list(axs),panel_ids=list('abc'))

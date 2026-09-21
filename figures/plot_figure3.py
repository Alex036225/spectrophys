from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
root=Path(__file__).resolve().parents[1];out=root/'figure3_exports';out.mkdir(parents=True,exist_ok=True);data=json.loads((root/'data/figure3_plot_values.json').read_text());vals={(r['method'],r['metric']):r['value'] for r in data['records']}
methods0=list(dict.fromkeys(r['method'] for r in data['records']))
metrics=['HR MAE','HR RMSE','RR MAE','RR RMSE','SpO2 MAE','SpO2 RMSE','SBP MAE','DBP MAE','MAP MAE']
raw0=np.array([[vals[(m,k)] for k in metrics] for m in methods0])
ranks=np.empty_like(raw0)
for j in range(9):
 for i in range(10):ranks[i,j]=1+sum(raw0[:,j]<raw0[i,j])+(sum(raw0[:,j]==raw0[i,j])-1)/2
meanrank=ranks.mean(axis=1);order=np.argsort(meanrank,kind='stable');methods=[methods0[i] for i in order];raw=raw0[order];mr=meanrank[order]
norm=(raw-raw.min(axis=0))/(raw.max(axis=0)-raw.min(axis=0))
accent='#C55D43';ink='#263743';muted='#647784';baseline='#7696AA'
cmap=LinearSegmentedColormap.from_list('within_metric_error',['#246578','#7EB3BC','#C8DEDF','#F1F4F3'])
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.titlesize':8.5,'axes.labelsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(183/25.4,220/25.4),facecolor='white')
fig.text(.025,.987,'ZPU clinical benchmark',fontsize=11,fontweight='bold',color=ink,va='top')
ax=fig.add_axes([.205,.646,.62,.241]);ax.set_label('heatmap')
ax.imshow(norm,aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='none')
ax.set_yticks(range(10),methods);ax.tick_params(axis='y',length=0,pad=5)
ax.set_xticks(range(9),['MAE','RMSE','MAE','RMSE','MAE','RMSE','SBP','DBP','MAP']);ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=5)
for t in ax.get_yticklabels():
 if t.get_text()=='SpectroPhys':t.set_fontweight('bold');t.set_color(accent)
for i in range(10):
 for j in range(9):
  ax.text(j,i,f'{raw[i,j]:.2f}',ha='center',va='center',fontsize=7.5,color='white' if norm[i,j]<.23 else ink,fontweight='bold' if i==0 else 'normal')
ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True)
ax.grid(which='minor',color='white',lw=.75);ax.tick_params(which='minor',length=0)
for spine in ax.spines.values():spine.set_visible(False)
ax.add_patch(Rectangle((-.5,-.5),9,1,fill=False,edgecolor=accent,linewidth=1.4,clip_on=False))
for boundary in [1.5,3.5,5.5]:ax.axvline(boundary,color='white',lw=3)
for center,label,col,span in [(.5,'Heart rate','#DCE8F2',2),(2.5,'Respiration','#DCEEE9',2),(4.5,'Oxygenation','#E9E2F1',2),(7,'Blood pressure (MAE)','#F1E6D6',3)]:
 x=(center+.5)/9;ax.text(x,1.125,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=7.5,fontweight='bold',color=ink)
 ax.add_patch(Rectangle(((center+.5-span/2)/9,1.093),span/9,.012,transform=ax.transAxes,facecolor=col,edgecolor='none',clip_on=False))
fig.text(.025,.932,'a',fontsize=10,fontweight='bold',color=ink)
                                                                                     
ar=fig.add_axes([.858,.646,.115,.241]);ar.set_label('mean_rank');ar.set_xlim(.5,11.5);ar.set_ylim(9.5,-.5)
ar.axhspan(-.5,.5,color='#F8EDE7',lw=0)
ar.hlines(range(10),.5,mr,color='#DAE3E7',lw=1)
ar.scatter(mr,np.arange(10),c=[accent]+[baseline]*9,s=[24]+[16]*9,zorder=3,edgecolors='white',linewidths=.3)
for i,v in enumerate(mr):ar.annotate(f'{v:.2f}',(v,i),xytext=(4,0),textcoords='offset points',va='center',fontsize=7.5,color=accent if i==0 else ink)
ar.set_yticks([]);ar.set_xticks([1,5,9]);ar.tick_params(length=2);ar.spines['left'].set_visible(False)
fig.text(.835,.932,'b',fontsize=10,fontweight='bold',color=ink)
fig.text(.9155,.917125,'Mean rank',fontsize=7.5,fontweight='bold',ha='center',va='bottom',color=ink)
cbax=fig.add_axes([.205,.628,.24,.007]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cbax,orientation='horizontal');cb.outline.set_visible(False);cb.set_ticks([0,1],labels=['Best','Worst']);cb.ax.tick_params(length=0,labelsize=7,pad=2)
                                                                                    
contrast=[('HR MAE','Heart rate','bpm'),('RR MAE','Respiratory rate','breaths/min'),('SpO2 MAE','Oxygen saturation','percentage points'),('SBP MAE','Systolic pressure','mmHg'),('DBP MAE','Diastolic pressure','mmHg'),('MAP MAE','Mean arterial pressure','mmHg')]
summary=[];compaxes=[]
for i,(metric,title,unit) in enumerate(contrast):
 best=min((m for m in methods if m!='SpectroPhys'),key=lambda m:vals[(m,metric)])
 b=vals[(best,metric)];v=vals[('SpectroPhys',metric)];gain=(b-v)/b*100;summary.append(dict(metric=metric,baseline=best,baseline_error=b,spectrophys_error=v,reduction_percent=gain))
 x=.182+(i%3)*.322;y=.360 if i<3 else .055
 ac=fig.add_axes([x,y,.149,.195]);ac.set_label(metric);compaxes.append(ac)
 ranked=sorted(methods,key=lambda m:vals[(m,metric)])
 errors=np.array([vals[(m,metric)] for m in ranked]);yy=np.arange(10)
 colours=[accent if m=='SpectroPhys' else '#246578' if m==best else baseline for m in ranked]
 ac.set_xlim(0,max(errors)*1.50);ac.set_ylim(9.6,-.6)
 ac.axhspan(-.45,.45,color='#F8EDE7',lw=0)
 ac.hlines(yy,0,errors,color='#D5E0E4',lw=.8)
 ac.scatter(errors,yy,c=colours,s=[23 if m=='SpectroPhys' else 16 for m in ranked],edgecolors='white',linewidths=.35,zorder=3)
 for j,(m,value) in enumerate(zip(ranked,errors)):
  ac.annotate(f'{value:.2f}',(value,j),xytext=(4,0),textcoords='offset points',va='center',fontsize=7.5,color=colours[j] if j<2 else ink,fontweight='bold' if j<2 else 'normal')
 ac.set_yticks(yy,ranked);ac.tick_params(axis='y',length=0,pad=0)
 for tick,m in zip(ac.get_yticklabels(),ranked):
  tick.set_horizontalalignment('left');tick.set_verticalalignment('center');tick.set_x((.049+(i%3)*.322-x)/.149)
  tick.set_color(accent if m=='SpectroPhys' else '#246578' if m==best else ink)
  if m=='SpectroPhys' or m==best:tick.set_fontweight('bold')
 title_x=.025+(i%3)*.322;title_y=.598 if i<3 else .295
 fig.text(title_x,title_y,chr(99+i),fontsize=10,fontweight='bold',color=ink)
 fig.text(title_x+.024,title_y,title,fontsize=8.5,color=ink)
 fig.text(title_x+.024,title_y-.022,f'{gain:.1f}% reduction  |  Δ = {b-v:.2f}',fontsize=7.5,fontweight='bold',color=accent)
 ac.spines['left'].set_visible(False);ac.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3));ac.tick_params(axis='x',length=2,width=.5)
 ac.set_xlabel(f'MAE ({unit})',labelpad=4)
fig.canvas.draw()
fig.savefig(out/'figure3-redesigned.pdf');fig.savefig(out/'figure3-redesigned.svg');fig.savefig(out/'figure3-redesigned.png',dpi=300)
(root/'fig1_zpu_benchmark.pdf').write_bytes((out/'figure3-redesigned.pdf').read_bytes())
(out/'derived-summary.json').write_text(json.dumps(summary,indent=2))

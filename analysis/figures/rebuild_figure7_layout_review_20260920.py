"""Replot existing Figure 7 source values only; no fitting or analysis."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path("C:/path/to/private-manuscript-workspace")
OUT=ROOT/'tmp/main_figure_consistency_review/Figure_7_numeric_layout.pdf'
perf=pd.read_csv(ROOT/'Figure 7 Clinical Translation/Figure_7BDF_performance_source_data.csv')
shap=pd.read_csv(ROOT/'Figure 7 Clinical Translation/Figure_7CEG_SHAP_top6_source_data.csv')
W,H=560,842
plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'axes.linewidth':.6,'xtick.labelsize':6.5,'ytick.labelsize':7})
fig=plt.figure(figsize=(W/72,H/72))
COL={'TCR':'#0072B2','BCR':'#D55E00','TCR+BCR':'#009E73'}
def text(x,y,s,size=8,bold=False,color='#202124'):
    fig.text(x/W,1-y/H,s,ha='left',va='baseline',fontsize=size,weight='bold' if bold else 'normal',color=color)
def axis(x,t,w,h):return fig.add_axes([x/W,1-(t+h)/H,w/W,h/H])
def style(ax):
    ax.spines[['right','top']].set_visible(False);ax.grid(axis='x',color='#dddddd',lw=.5);ax.set_axisbelow(True)
def heading(x,y,letter,s):text(x-19,y,letter,12,True);text(x,y,s,8.4,True)
TASKS={
 'diagnosis':[('cd_vs_control','CD vs control'),('uc_vs_control','UC vs control'),('cd_vs_uc','CD vs UC')],
 'inflammation':[('cd_inflamed_vs_noninflamed','CD: inflamed vs noninflamed'),('uc_inflamed_vs_noninflamed','UC: inflamed vs noninflamed')],
 'therapy_response':[('combined_biologic_nonresponder_vs_responder','All biologics: NR vs R'),('anti_tnf_nonresponder_vs_responder','Anti-TNF: NR vs R'),('ustekinumab_nonresponder_vs_responder','Ustekinumab: NR vs R'),('vedolizumab_nonresponder_vs_responder','Vedolizumab: NR vs R')]}
def forest(domain,t,h):
    ax=axis(106,t,181,h);tasks=TASKS[domain];nested=domain=='diagnosis';ys=np.arange(len(tasks))[::-1];labels=[]
    for yy,(task,lab)in zip(ys,tasks):
        sub=perf[(perf.domain==domain)&(perf.comparison==task)]
        nr=sub[sub.receptor_model=='TCR+BCR'];nr=(nr if len(nr) else sub).iloc[0]
        labels.append(f'{lab}\nn={int(nr.n)} ({int(nr.n_positive)}/{int(nr.n_negative)})')
        for model,off,marker in [('TCR',.18,'o'),('TCR+BCR',0,'D'),('BCR',-.18,'s')]:
            row=sub[sub.receptor_model==model]
            if row.empty:continue
            r=row.iloc[0];c=COL[model]
            ax.errorbar(r.roc_auc,yy+off,xerr=[[r.roc_auc-r.roc_auc_ci_low],[r.roc_auc_ci_high-r.roc_auc]],fmt=marker,ms=4.2,capsize=2,lw=.9,color=c,mfc=c if nested else 'white',mec=c)
            if nested and pd.notna(r.get('null_95th_percentile')):ax.plot([r.null_95th_percentile]*2,[yy+off-.055,yy+off+.055],color='#999999',lw=1)
    ax.axvline(.5,ls='--',color='#777777',lw=.6);ax.set_yticks(ys,labels);ax.set_ylim(-.5,len(tasks)-.5);ax.set_xlim(.45 if nested else .25,1.01)
    ax.set_xlabel('Median outer-fold ROC AUC (95% CI)' if nested else 'Out-of-fold ROC AUC (95% CI)',fontsize=7,labelpad=3);style(ax)
def features(domain,rows,tops):
    for (task,lab),top in zip(rows,tops):
        text(355,top,lab,6.8,True)
        for rec,x in [('TCR',355),('BCR',468)]:
            d=shap[(shap.domain==domain)&(shap.comparison==task)&(shap.receptor==rec)].sort_values('relative_mean_absolute_shap')
            assert len(d)==6,(domain,task,rec,len(d))
            text(x,top+9,rec,6.6,True,COL[rec])
            model=d.model_display.iloc[0].replace(rec+': ','')
            # Model metadata gets a dedicated line, separate from the receptor label.
            text(x,top+16,model,4.6,False,COL[rec])
            ax=axis(x,top+21,76,30)
            y=np.arange(6)
            for yy,(_,r) in zip(y,d.iterrows()):
                ax.barh(yy,r.relative_mean_absolute_shap,height=.66,color=COL[rec],alpha=.78,lw=0)
                positive=r.direction_sign>0
                ax.scatter(r.relative_mean_absolute_shap,yy,marker='>' if positive else '<',s=12,facecolor=COL[rec] if positive else 'white',edgecolor=COL[rec],lw=.65,zorder=4)
            ax.set_yticks(y,d.feature_token,fontfamily='DejaVu Sans Mono',fontsize=5.5,color=COL[rec]);ax.tick_params(axis='y',pad=2,length=2)
            ax.set_xlim(0,1.08);ax.set_xticks([0,.5,1]);ax.tick_params(axis='x',labelsize=5.5,pad=2,length=2)
            if top!=tops[-1]:ax.set_xticklabels([])
            else:ax.set_xlabel('Relative mean |SHAP|',fontsize=5.3,labelpad=2)
            style(ax)

heading(106,208,'B','Diagnosis classification');text(106,219,'FULLY NESTED VALIDATION',5.6,True,'#555555')
heading(355,208,'C','Selected diagnosis features')
forest('diagnosis',234,159);features('diagnosis',TASKS['diagnosis'],[231,294,357])
heading(106,441,'D','Inflammation models');text(106,452,'POST-SELECTION 5-FOLD CV',5.6,True,'#555555')
heading(355,441,'E','Selected inflammation features')
forest('inflammation',465,107);features('inflammation',TASKS['inflammation'],[464,530])
heading(106,619,'F','Biologic-response models');text(106,630,'POST-SELECTION 5-FOLD CV',5.6,True,'#555555')
heading(355,619,'G','Selected response-status features')
forest('therapy_response',643,155);features('therapy_response',TASKS['therapy_response'][:3],[642,705,768])
fig.savefig(OUT,dpi=300);plt.close(fig)
print(OUT)

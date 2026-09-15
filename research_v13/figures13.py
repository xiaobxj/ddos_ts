"""Publication-style standalone figures from verified, frozen measurements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none'})
COLORS={'archived20':'#5a6b80','rolling_shrink':'#087f8c','half_shrink':'#d28a26'}
NAMES={'archived20':'Original MSE20','rolling_shrink':'Rolling shrinkage','half_shrink':'Fixed half'}

def export(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    coefficients=pd.read_csv(OUT/'coefficients.csv');jack=pd.read_csv(OUT/'calibration_leave_year_out.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    x=np.arange(3)
    for i,r in enumerate(coefficients.itertuples()):
        axes[0].plot([i,i],[r.raw_alpha,r.alpha],color='#a7b5bf',lw=2)
        axes[0].scatter(i,r.raw_alpha,s=70,facecolors='white',edgecolors='#5a6b80',zorder=3,
                        label='Unconstrained estimate' if i==0 else None)
        axes[0].scatter(i,r.alpha,s=60,color='#087f8c',zorder=4,label='Frozen coefficient' if i==0 else None)
        axes[0].annotate(f'{r.alpha:.3f}',(i,r.alpha),xytext=(10,7),textcoords='offset points',color='#087f8c')
        h=jack[jack.outer_cutoff.eq(r.outer_cutoff)]
        for j,z in enumerate(h.itertuples()):
            axes[1].scatter(i+(j-1)*.12,z.alpha,color='#b3bec7',marker='o',s=42,zorder=2)
            offset=[(-12,-16),(18,5),(10,-18)][j]
            axes[1].annotate(str(z.omitted_year),(i+(j-1)*.12,z.alpha),xytext=offset,
                textcoords='offset points',ha='center',fontsize=8,color='#677783')
        axes[1].scatter(i,r.alpha,color='#087f8c',marker='D',s=65,zorder=4)
    labels=[str(int(c[:4])+1) for c in coefficients.outer_cutoff]
    for ax in axes:
        ax.set_xticks(x,labels);ax.set_xlabel('Outer prediction year');ax.set_xlim(-.45,2.55)
        ax.axhline(0,color='#aab5be',ls='--',lw=.8);ax.axhline(1,color='#aab5be',ls='--',lw=.8)
        ax.grid(axis='y',alpha=.15);ax.set_ylabel('Retained forecast amplitude (alpha)')
    axes[0].set_title('Three-year rolling fit, then clip to [0, 1]')
    axes[0].legend(frameon=False,fontsize=9,loc='best')
    axes[1].set_ylim(-.22,1.17);axes[1].set_title('Leave-one-calibration-year-out sensitivity')
    axes[1].plot([],[],marker='D',color='#087f8c',ls='',label='Frozen full-window coefficient')
    axes[1].plot([],[],marker='o',color='#b3bec7',ls='',label='Omit labeled year; diagnostic only')
    axes[1].legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Round 13 | Coefficients use only matured, pre-cutoff rolling labels',fontsize=15,fontweight='bold')
    fig.text(.02,-.085,'Calibration counts: 118 / 145 / 144 weeks. All three coefficients are frozen before outer evaluation.\n'
        'Alpha = 0 uses the causal training mean; alpha = 1 keeps the original forecast. Leave-year variants are never deployed.',
        color='#425563',fontsize=9)
    export(fig,'rolling_calibration')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('method')
    seeds=pd.read_csv(OUT/'seed_metrics.csv');years=pd.read_csv(OUT/'yearly_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.9),layout='constrained')
    for i,method in enumerate(NAMES):
        g=seeds[seeds.method.eq(method)].sort_values('seed')
        axes[0].scatter(i+np.array([-.14,0,.14]),g.rmse*1e4,color=COLORS[method],s=27,alpha=.65)
        axes[0].scatter(i,metrics.loc[method,'rmse']*1e4,color=COLORS[method],marker='D',s=75,zorder=4)
        axes[0].annotate(f"{metrics.loc[method,'rmse']*1e4:.2f}",(i,metrics.loc[method,'rmse']*1e4),
            xytext=(16,0) if method=='rolling_shrink' else (0,-17),textcoords='offset points',
            ha='left' if method=='rolling_shrink' else 'center',fontsize=9,color=COLORS[method])
        for j,value in enumerate(g.rmse):
            axes[0].annotate(str(j+1),(i+[-.14,0,.14][j],value*1e4),xytext=(0,6),
                textcoords='offset points',ha='center',fontsize=8)
        annual=years[years.method.eq(method)].sort_values('year')
        axes[1].bar(x+(i-1)*.24,annual.mse_skill_vs_training_mean*100,width=.23,color=COLORS[method],label=NAMES[method])
    axes[0].axhline(metrics.loc['training_mean','rmse']*1e4,color='#8f485d',ls='--',lw=1,label='Training mean')
    axes[0].set_xticks(x,list(NAMES.values()));axes[0].set_ylabel('Executable-return RMSE (bp; zoomed axis)')
    axes[0].set_xlim(-.45,2.45);axes[0].margins(y=.2);axes[0].grid(axis='y',alpha=.15)
    axes[0].set_title('Diamonds: ensemble; small dots: seeds 1/2/3');axes[0].legend(frameon=False,fontsize=9)
    axes[1].set_xticks(x,['2018','2019','2020']);axes[1].axhline(0,color='#5a6b80',lw=.8)
    axes[1].set_ylabel('MSE improvement vs training mean (%)');axes[1].grid(axis='y',alpha=.15)
    axes[1].set_title('Annual ensemble results (higher is better)');axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Round 13 | Same 141 historical validation weeks',fontsize=15,fontweight='bold')
    fig.text(.02,-.09,'Calibration target: m_outer + alpha * (prediction - m_outer). Architecture, neural weights and seeds stay fixed.\n'
        'These dates have been repeatedly inspected. Causal calibration and Holm correction do not create independent confirmation.',
        color='#425563',fontsize=9)
    export(fig,'shrinkage_validation')
    print('Created two PNG and two SVG figures.')

if __name__=='__main__':main()

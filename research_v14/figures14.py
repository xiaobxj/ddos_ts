"""Standalone figures from verified classification measurements."""
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
COLORS={'archived20':'#5a6b80','direction_bce':'#087f8c','training_frequency':'#b16a24','neutral_50':'#9a96a5'}
NAMES={'archived20':'Original MSE20','direction_bce':'Direction BCE','training_frequency':'Training frequency','neutral_50':'Neutral 0.5'}

def export(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('method');seeds=pd.read_csv(OUT/'seed_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    for i,method in enumerate(['archived20','direction_bce']):
        g=seeds[seeds.method.eq(method)].sort_values('seed');a=metrics.loc[method,'accuracy']*100
        axes[0].scatter(i+np.array([-.14,0,.14]),g.accuracy*100,color=COLORS[method],alpha=.65,s=30)
        axes[0].scatter(i,a,color=COLORS[method],marker='D',s=75,zorder=4)
        axes[0].annotate(f'{a:.2f}%',(i,a),xytext=(17,7),textcoords='offset points',color=COLORS[method],fontsize=10)
        for j,value in enumerate(g.accuracy):
            axes[0].annotate(str(j+1),(i+[-.14,0,.14][j],value*100),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set_xticks([0,1],[NAMES['archived20'],NAMES['direction_bce']]);axes[0].set_xlim(-.4,1.65)
    axes[0].axhline(metrics.loc['training_frequency','accuracy']*100,color=COLORS['training_frequency'],ls='--',lw=1,label='Training-frequency decision')
    axes[0].set_ylabel('Direction accuracy (%; zoomed axis)');axes[0].set_title('Diamonds: ensemble; dots: fixed seeds 1/2/3')
    axes[0].margins(y=.23);axes[0].grid(axis='y',alpha=.15);axes[0].legend(frameon=False,fontsize=9,loc='best')
    methods=['direction_bce','training_frequency','neutral_50']
    for i,method in enumerate(methods):
        value=metrics.loc[method,'brier']
        axes[1].scatter(i,value,color=COLORS[method],marker='D',s=75,zorder=4)
        axes[1].annotate(f'{value:.4f}',(i,value),xytext=(0,-18),textcoords='offset points',ha='center',color=COLORS[method])
        if method=='direction_bce':
            g=seeds[seeds.method.eq(method)].sort_values('seed')
            offsets=[-.18,.18,.32]
            axes[1].scatter(i+np.array(offsets),g.brier,color=COLORS[method],alpha=.65,s=30)
            for j,value in enumerate(g.brier):axes[1].annotate(str(j+1),(i+offsets[j],value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    axes[1].set_xticks(range(3),[NAMES[n] for n in methods]);axes[1].set_xlim(-.4,2.4)
    axes[1].set_ylabel('Binary Brier score (lower is better; zoomed axis)')
    axes[1].set_title('Probability error has its own constant references');axes[1].margins(y=.28);axes[1].grid(axis='y',alpha=.15)
    fig.suptitle('Round 14 | Direct direction learning on the same 141 historical weeks',fontsize=15,fontweight='bold')
    fig.text(.02,-.085,'Three seed probabilities are averaged; the direction threshold is fixed at 0.5. All models train for 20 epochs.\n'
        'Original return predictions have no probability score. These repeatedly inspected dates provide exploratory evidence only.',fontsize=9,color='#425563')
    export(fig,'direction_and_probability')
    years=pd.read_csv(OUT/'yearly_metrics.csv');bins=pd.read_csv(OUT/'reliability_bins.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    for method in ['archived20','direction_bce','training_frequency']:
        g=years[years.method.eq(method)].sort_values('year')
        axes[0].plot(g.year,g.accuracy*100,color=COLORS[method],marker='o',lw=1.7,label=NAMES[method])
    axes[0].set_xticks([2018,2019,2020]);axes[0].set_ylabel('Direction accuracy (%)');axes[0].grid(axis='y',alpha=.15)
    axes[0].set_title('Annual results, all fixed comparators');axes[0].legend(frameon=False,fontsize=9)
    axes[1].plot([0,1],[0,1],ls='--',color='#a5adb6',lw=1,label='Identity line')
    g=bins[bins.method.eq('direction_bce')&bins.n.gt(0)]
    axes[1].scatter(g.mean_probability,g.observed_frequency,s=30+g.n*2,color=COLORS['direction_bce'],alpha=.8)
    for i,r in enumerate(g.itertuples()):
        axes[1].annotate(f'n={r.n}',(r.mean_probability,r.observed_frequency),xytext=(-12,-20) if i==0 else (9,9),
            textcoords='offset points',ha='right' if i==0 else 'left',fontsize=9,color=COLORS['direction_bce'])
    axes[1].set_xlim(0,1);axes[1].set_ylim(0,1);axes[1].set_aspect('equal');axes[1].grid(alpha=.15)
    axes[1].set_xlabel('Mean predicted up probability');axes[1].set_ylabel('Observed up frequency')
    axes[1].set_title('Classifier reliability: 5 fixed probability bins');axes[1].legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Round 14 | Annual variation and probability diagnostics',fontsize=15,fontweight='bold')
    fig.text(.02,-.085,'Reliability uses only populated bins; all empty-bin records are retained. Marker area increases with the bin count.\n'
        'No probability calibration, temperature fitting, confidence filtering or threshold tuning is performed.',fontsize=9,color='#425563')
    export(fig,'annual_and_calibration')
    print('Created two PNG and two SVG figures.')

if __name__=='__main__':main()

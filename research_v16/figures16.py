"""Standalone plots from frozen and verified measurements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
NAMES={'native_mse':'Original MSE20','learned_probe':'Learned features + logistic','raw25_probe':'Simple OHLCV + logistic','training_frequency':'Training frequency'}
COLORS={'native_mse':'#697889','learned_probe':'#196caf','raw25_probe':'#ba526c','training_frequency':'#a46b29'}


def export(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    assert json.loads((OUT/'verification.json').read_text(encoding='utf-8'))['status']=='PASS'
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv');years=pd.read_csv(OUT/'yearly_metrics.csv')
    methods=['native_mse','learned_probe','raw25_probe','training_frequency']
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.3),sharey=True,layout='constrained')
    periods=[('early_2015_2017','2015-2017 | 120 weeks','#8559a0',.14),('late_2018_2020','2018-2020 | 141 weeks','#237ca3',-.14)]
    for window,label,color,offset in periods:
        t=metrics[metrics.window.eq(window)].set_index('method')
        for i,method in enumerate(methods):
            y=3-i+offset;a=t.loc[method,'accuracy']*100;b=t.loc[method,'brier']
            axes[0].scatter(a,y,color=color,s=55,label=label if i==0 else None,zorder=3)
            axes[0].text(a+.55,y,f'{a:.2f}%',color=color,va='center',fontsize=9.5)
            if pd.notna(b):
                axes[1].scatter(b,y,color=color,s=55,zorder=3)
                axes[1].text(b+.001,y,f'{b:.4f}',color=color,va='center',fontsize=9.5)
    axes[1].text(.268,3,'No probability score for raw returns',ha='center',va='center',color='#6f7c8a',fontsize=9,
        bbox=dict(facecolor='white',edgecolor='none',pad=3))
    axes[0].set_yticks(range(3,-1,-1),[NAMES[m] for m in methods]);axes[0].tick_params(axis='y',length=0);axes[1].tick_params(axis='y',length=0)
    axes[0].set_xlim(40,67);axes[0].set_xticks([40,45,50,55,60,65]);axes[0].set_ylim(-.6,3.6)
    axes[1].set_xlim(.233,.310);axes[1].set_xticks([.24,.25,.26,.27,.28,.29,.30]);axes[1].axvline(.25,ls=':',lw=1,color='#a9adb4',zorder=0)
    axes[0].set_title('Direction accuracy');axes[1].set_title('Brier probability error')
    axes[0].set_xlabel('Accuracy (%; zoomed axis)');axes[1].set_xlabel('Brier (lower is better; zoomed axis)')
    for ax in axes:ax.grid(axis='x',alpha=.13)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.57,-.04),ncol=2,frameon=False)
    fig.suptitle('Round 16 | Same algorithms across two historical periods',fontsize=15,fontweight='bold')
    fig.text(.02,-.105,'Both feature families use 25 coordinates, the same fixed L2 penalty and a 0.5 probability threshold. Learned probabilities average three seeds.\n'
        'The simple-feature model is deterministic. Dotted line: neutral 0.5 Brier. All dates were previously inspected; this is exploratory evidence.',fontsize=9,color='#425563')
    export(fig,'period_comparison')
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.3),layout='constrained')
    for method in methods:
        g=years[years.method.eq(method)].sort_values('year')
        axes[0].plot(g.year,g.accuracy*100,marker='o',color=COLORS[method],lw=1.8,label=NAMES[method])
    axes[0].set_ylabel('Direction accuracy (%)');axes[0].set_title('Annual variation, all fixed comparators');axes[0].set_ylim(35,77)
    axes[0].legend(frameon=False,fontsize=8.5,loc='upper center',ncol=2)
    raw=years[years.method.eq('raw25_probe')].sort_values('year');freq=years[years.method.eq('training_frequency')].sort_values('year')
    axes[1].plot(raw.year,100*raw.mean_probability,marker='o',color=COLORS['raw25_probe'],lw=1.8,label='Mean simple-model probability')
    axes[1].plot(raw.year,100*raw.observed_up_fraction,marker='o',color='#52606f',lw=1.8,label='Observed up frequency')
    axes[1].plot(freq.year,100*freq.mean_probability,marker='o',color=COLORS['training_frequency'],ls='--',lw=1.5,label='Training up frequency')
    axes[1].set_ylabel('Up probability / frequency (%)');axes[1].set_title('Simple features: annual probability bias');axes[1].set_ylim(25,82)
    axes[1].legend(frameon=False,fontsize=9,loc='upper center')
    for ax in axes:ax.set_xticks(range(2015,2021));ax.set_xlabel('Signal year');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Round 16 | Annual results and probability diagnostics',fontsize=15,fontweight='bold')
    fig.text(.02,-.105,'Retained weeks by year: 22, 48, 50, 49, 46, 46. The 2015 sample is incomplete under the unchanged OHLC validity mask.\n'
        'Probabilities are evaluated after fitting; observed frequencies and these diagnostics never alter the frozen models or decisions.',fontsize=9,color='#425563')
    export(fig,'annual_and_probability_bias')
    print('Created two PNG and two SVG figures.')


if __name__=='__main__':main()

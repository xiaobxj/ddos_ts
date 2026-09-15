"""Standalone scientific plots from the completed, verified experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none'})
COLORS={'archived20':'#68778b','probe_mse':'#1f65ab','direction_bce':'#43969c','probe_bce':'#bd526c','training_frequency':'#a36b24'}
NAMES={'archived20':'MSE20 / original head','probe_mse':'MSE20 / logistic probe','direction_bce':'BCE20 / original head',
       'probe_bce':'BCE20 / logistic probe','training_frequency':'Training frequency'}


def export(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('method');seeds=pd.read_csv(OUT/'seed_metrics.csv')
    methods=['archived20','probe_mse','direction_bce','probe_bce','training_frequency']
    fig,axes=plt.subplots(1,2,figsize=(14,5.4),sharey=True,layout='constrained')
    for i,method in enumerate(methods):
        y=4-i;g=seeds[seeds.method.eq(method)].sort_values('seed');color=COLORS[method]
        a=100*metrics.loc[method,'accuracy'];b=metrics.loc[method,'brier']
        axes[0].scatter(a,y,marker='D',color=color,s=65,zorder=4)
        axes[0].text(68.5,y,f'{a:.2f}%',ha='right',va='center',color=color,fontweight='bold')
        if len(g):axes[0].scatter(g.accuracy*100,y+np.array([-.15,0,.15]),facecolors='none',edgecolors=color,s=38,zorder=3)
        if pd.notna(b):
            axes[1].scatter(b,y,marker='D',color=color,s=65,zorder=4)
            axes[1].text(.270,y,f'{b:.6f}',ha='right',va='center',color=color,fontweight='bold')
            if len(g):axes[1].scatter(g.brier,y+np.array([-.15,0,.15]),facecolors='none',edgecolors=color,s=38,zorder=3)
        else:axes[1].text(.250,y,'No probability score for raw returns',ha='center',va='center',color='#697583',fontsize=9)
    axes[0].set_yticks(range(4,-1,-1),[NAMES[m] for m in methods]);axes[0].tick_params(axis='y',length=0)
    axes[1].tick_params(axis='y',length=0)
    axes[0].axvline(100*metrics.loc['training_frequency','accuracy'],ls='--',lw=1,color=COLORS['training_frequency'],zorder=0)
    axes[1].axvline(metrics.loc['training_frequency','brier'],ls='--',lw=1,color=COLORS['training_frequency'],zorder=0)
    axes[1].axvline(.25,ls=':',lw=1,color='#9496a0',zorder=0)
    axes[0].set_xlim(48,69);axes[0].set_xticks([50,55,60,65]);axes[0].set_ylim(-.6,4.7)
    axes[1].set_xlim(.238,.271);axes[1].set_xticks([.240,.245,.250,.255,.260])
    axes[0].set_xlabel('Direction accuracy (%; zoomed axis)')
    axes[1].set_xlabel('Brier score (lower is better; zoomed axis)')
    axes[0].set_title('Fixed decision rules; all three inherited seeds',pad=14)
    axes[1].set_title('Probabilities averaged before scoring',pad=14)
    for ax in axes:
        ax.grid(axis='x',alpha=.12);ax.axhline(2.5,color='#edf0f4',lw=1);ax.axhline(.5,color='#edf0f4',lw=1)
    handles=[Line2D([],[],marker='D',ls='none',color='#35465b',label='Ensemble'),
             Line2D([],[],marker='o',ls='none',markerfacecolor='none',color='#35465b',label='Individual seed'),
             Line2D([],[],ls='--',color=COLORS['training_frequency'],label='Training-frequency reference')]
    axes[0].legend(handles=handles,frameon=False,ncol=1,fontsize=8.5,loc='lower right',bbox_to_anchor=(1,.01))
    # Place the legend below the panels, away from numerical labels.
    axes[0].get_legend().remove();fig.legend(handles=handles,frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.53,-.055),fontsize=9)
    fig.suptitle('Round 15 | Identical logistic probes on two frozen representations',fontsize=15,fontweight='bold')
    fig.text(.02,-.115,'Same 141 historical weeks, 2018-2020. Diamonds are ensemble results; open circles are fixed seeds (not confidence intervals).\n'
        'Probe threshold: p > 0.5. Dotted line: neutral p = 0.5 Brier reference. These repeatedly inspected dates provide exploratory evidence.',fontsize=9,color='#425563')
    export(fig,'frozen_probe_comparison')
    train=pd.read_csv(OUT/'training_fold_summary.csv');years=pd.read_csv(OUT/'yearly_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(13.2,5),layout='constrained')
    for family in ['probe_mse','probe_bce']:
        g=train[train.family.eq(family)].sort_values('cutoff')
        axes[0].plot(g.cutoff.str[:4].astype(int),100*g.log_loss_skill,marker='o',lw=1.8,color=COLORS[family],label=NAMES[family])
    g=train[train.family.eq('probe_bce')].sort_values('cutoff')
    axes[0].plot(g.cutoff.str[:4].astype(int),100*g.native_log_loss_skill,marker='o',lw=1.8,color=COLORS['direction_bce'],label=NAMES['direction_bce'])
    axes[0].set_xticks([2017,2018,2019]);axes[0].set_xlabel('Training cutoff year (year end)')
    axes[0].set_ylabel('Log-loss reduction vs training frequency (%)');axes[0].set_title('Training fit: mean of three seeds')
    axes[0].legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.02),ncol=1)
    axes[0].set_ylim(-.35,10.5)
    base=years[years.method.eq('training_frequency')].set_index('year').brier
    for method in ['probe_mse','probe_bce','direction_bce']:
        g=years[years.method.eq(method)].sort_values('year');skill=100*(1-g.brier.to_numpy()/base.loc[g.year].to_numpy())
        axes[1].plot(g.year,skill,marker='o',lw=1.8,color=COLORS[method],label=NAMES[method])
    axes[1].set_xticks([2018,2019,2020]);axes[1].set_xlabel('Validation year')
    axes[1].set_ylabel('Brier reduction vs training frequency (%)');axes[1].set_title('Validation: three-seed probability ensemble')
    for ax in axes:ax.axhline(0,ls='--',color='#a36b24',lw=1);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Round 15 | Better training readouts do not give stable future gains',fontsize=15,fontweight='bold')
    fig.text(.02,-.105,'Positive values indicate lower loss than the causal training-frequency reference. All probe heads use the same fixed L2 penalty.\n'
        'Training features are supervised in-sample representations. Training means and validation ensembles are different summaries.',fontsize=9,color='#425563')
    export(fig,'training_and_future_skill')
    print('Created two PNG and two SVG figures from verified tables.')


if __name__=='__main__':main()

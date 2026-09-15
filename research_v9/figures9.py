"""Export scientific figures for mechanism and prediction comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
COLORS={'legacy':'#687e91','balanced':'#209e91'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})


def export(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    definitions=pd.read_csv(OUT/'batch_definitions.csv');probe=pd.read_csv(OUT/'fixed_state_gradient_summary.csv')
    x=np.arange(3);fig,ax=plt.subplots(1,2,figsize=(12,4.9))
    for offset,rule in [(-.18,'legacy'),(.18,'balanced')]:
        d=definitions[definitions.rule==rule].sort_values('cutoff');p=probe[probe.rule==rule].sort_values('cutoff')
        ax[0].bar(x+offset,d.final_batch,.36,color=COLORS[rule],label=rule.title())
        ax[1].bar(x+offset,p.gradient_norm_max,.36,color=COLORS[rule],label=rule.title())
    for a in ax:a.set_xticks(x,['2018 fold','2019 fold','2020 fold']);a.legend(frameon=False)
    ax[0].set_ylabel('Rows in the last batch');ax[0].set_title('Same rows and number of updates')
    ax[1].set_ylabel('Maximum raw gradient norm');ax[1].set_title('Frozen-state probe, dropout disabled')
    fig.suptitle('Round 9 | The fixed batching intervention',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.02)
    fig.text(.06,-.03,'Each probe uses the archived seed20260910 model for its fold and the same first-epoch row permutation.\nNo optimizer steps are applied. Gradient statistics here do not measure variance during stochastic training.',fontsize=9,color='#425563')
    fig.tight_layout();export(fig,'batch_mechanism')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('rule')
    seeds=pd.read_csv(OUT/'seed_metrics.csv');years=pd.read_csv(OUT/'yearly_metrics.csv')
    fig,ax=plt.subplots(1,2,figsize=(12,5.2))
    for i,rule in enumerate(['legacy','balanced']):
        g=seeds[seeds.rule==rule].sort_values('seed')
        ax[0].scatter(i,metrics.loc[rule,'rmse']*1e4,color=COLORS[rule],marker='D',s=90,zorder=3)
        ax[0].scatter(i+np.array([-.18,0,.18]),g.rmse*1e4,c='#172b38',s=25,zorder=4)
        for j,v in enumerate(g.rmse):ax[0].annotate(str(j+1),(i+[-.18,0,.18][j],v*1e4),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        h=years[years.rule==rule].sort_values('year')
        ax[1].bar(x+[-.18,.18][i],h.mse_skill_vs_training_mean*100,.36,color=COLORS[rule],label=rule.title())
    p=pd.read_csv(OUT/'ensemble_predictions.csv');g=p[p.rule=='legacy']
    mean_rmse=float(np.sqrt(np.mean((g.training_mean-g.actual)**2)))
    ax[0].axhline(mean_rmse*1e4,color='#bd5947',ls='--',label='Training mean')
    ax[0].set_xticks([0,1],['Legacy batching','Balanced batching']);ax[0].set_ylabel('Return RMSE (bp; lower is better)')
    ax[0].set_title('Diamonds: ensemble; dots: fixed seeds1/2/3');ax[0].legend(frameon=False)
    bottom=min(mean_rmse*1e4,metrics.rmse.min()*1e4,seeds.rmse.min()*1e4)
    ax[0].set_ylim(max(0,bottom-8),max(mean_rmse*1e4,seeds.rmse.max()*1e4)+10)
    ax[1].axhline(0,color='#65737d',lw=1);ax[1].set_xticks(x,['2018','2019','2020'])
    ax[1].set_ylabel('MSE improvement over training mean (%)');ax[1].set_title('Annual three-seed ensemble results')
    ax[1].legend(frameon=False)
    fig.suptitle('Round 9 | Same 141 historical validation weeks',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.02)
    fig.text(.06,-.03,'One prespecified candidate; no epoch or seed selection. Left axis is zoomed to compare errors.\nThe 2018-2020 validation period has been inspected repeatedly; this is exploratory evidence.',fontsize=9,color='#425563')
    fig.tight_layout();export(fig,'validation_results')


if __name__=='__main__':main()

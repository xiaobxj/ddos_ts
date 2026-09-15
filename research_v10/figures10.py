"""Standalone scientific plots for the fixed robust-loss comparison."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
COLORS={'mse':'#687e91','huber':'#229c91'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})


def export(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    audit=pd.read_csv(OUT/'training_loss_audit.csv');audit=audit[audit.state=='frozen_mse_model'].groupby('cutoff').mean(numeric_only=True)
    fig,ax=plt.subplots(1,2,figsize=(12,4.8));e=np.linspace(-4,4,601)
    ax[0].plot(e,e**2,color=COLORS['mse'],label='Squared error',lw=2)
    ax[0].plot(e,np.where(np.abs(e)<=1,e**2,2*np.abs(e)-1),color=COLORS['huber'],label='2 x Huber, delta=1',lw=2)
    ax[0].axvline(-1,color='#a0abb2',ls=':',lw=1);ax[0].axvline(1,color='#a0abb2',ls=':',lw=1)
    ax[0].set_xlabel('Return residual / training return SD');ax[0].set_ylabel('Per-observation return loss')
    ax[0].set_title('Same quadratic region, linear tails');ax[0].legend(frameon=False)
    x=np.arange(3)
    ax[1].bar(x-.18,audit.top_one_percent_mse_share*100,.36,color=COLORS['mse'],label='Squared error')
    ax[1].bar(x+.18,audit.top_one_percent_huber_share*100,.36,color=COLORS['huber'],label='Scaled Huber')
    ax[1].set_xticks(x,['2018 fold','2019 fold','2020 fold']);ax[1].set_ylabel('Loss share from top 1% residuals (%)')
    ax[1].set_title('Same frozen training residuals');ax[1].legend(frameon=False)
    fig.suptitle('Round 10 | One fixed robust return loss',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.02)
    fig.text(.06,-.03,'Right: average of the three archived models per fold; no model is refitted for this concentration audit.\nOnly return loss changes. Auxiliary OHLCV MSE, original batches, target scales and all observations are retained.',fontsize=9,color='#425563')
    fig.tight_layout();export(fig,'robust_loss_mechanism')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('rule');seeds=pd.read_csv(OUT/'seed_metrics.csv')
    years=pd.read_csv(OUT/'yearly_metrics.csv');bases=pd.read_csv(OUT/'baseline_metrics.csv').set_index('reference')
    fig,ax=plt.subplots(1,2,figsize=(12,5.2))
    for i,rule in enumerate(['mse','huber']):
        g=seeds[seeds.rule==rule].sort_values('seed')
        ax[0].scatter(i,metrics.loc[rule,'rmse']*1e4,color=COLORS[rule],marker='D',s=90,zorder=3)
        ax[0].scatter(i+np.array([-.18,0,.18]),g.rmse*1e4,c='#172b38',s=25,zorder=4)
        for j,v in enumerate(g.rmse):ax[0].annotate(str(j+1),(i+[-.18,0,.18][j],v*1e4),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        y=years[years.rule==rule].sort_values('year');ax[1].bar(x+[-.18,.18][i],y.mse_skill_vs_training_mean*100,.36,color=COLORS[rule],label=rule.upper() if rule=='mse' else 'Scaled Huber')
    ax[0].axhline(bases.loc['training_mean','rmse']*1e4,color='#bd5947',ls='--',label='Training mean')
    ax[0].axhline(bases.loc['huber_intercept','rmse']*1e4,color='#8b75ab',ls=':',label='Training Huber intercept')
    ax[0].set_xticks([0,1],['MSE training','Huber training']);ax[0].set_ylabel('Return RMSE (bp; lower is better)')
    ax[0].set_title('Diamonds: ensemble; dots: fixed seeds 1/2/3');ax[0].legend(frameon=False,fontsize=9)
    lo=min(seeds.rmse.min(),metrics.rmse.min(),bases.rmse.min())*1e4
    hi=max(seeds.rmse.max(),metrics.rmse.max(),bases.rmse.max())*1e4
    ax[0].set_ylim(max(0,lo-7),hi+12)
    ax[1].axhline(0,color='#65737d',lw=1);ax[1].set_xticks(x,['2018','2019','2020'])
    ax[1].set_ylabel('MSE improvement over training mean (%)');ax[1].set_title('Annual three-seed ensemble results');ax[1].legend(frameon=False)
    fig.suptitle('Round 10 | Same 141 historical validation weeks',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.02)
    fig.text(.06,-.03,'MSE remains the primary endpoint. The Huber intercept is fitted only on training labels. No delta or seed selection.\nThe same 2018-2020 dates have been used repeatedly; this is exploratory evidence. Left axis is zoomed.',fontsize=9,color='#425563')
    fig.tight_layout();export(fig,'robust_validation_results')


if __name__=='__main__':main()

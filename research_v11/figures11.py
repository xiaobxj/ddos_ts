"""Standalone training-only figures; no validation series are loaded."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'results'
COLORS = {'archived_mse':'#59788e', 'archived_huber':'#a27136', 'joint':'#167d9a', 'return_only':'#bc5267'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'svg.fonttype':'none'})


def export(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=165, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / (name+'.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    pairs = pd.read_csv(OUT/'paired_training_comparisons.csv')
    gradient = pd.read_csv(OUT/'gradient_diagnostics.csv')
    fig, ax = plt.subplots(1, 2, figsize=(13,5.5))
    for mode, marker, color, label in [('eval','o','#59788e','Dropout off'), ('dropout','D','#bc5267','8-draw dropout mean')]:
        g = pairs[pairs.comparison.eq('archived_huber_minus_mse') & pairs['mode'].eq(mode)].sort_values(['cutoff','seed'])
        y = np.arange(9) + (-.10 if mode=='eval' else .10)
        ax[0].scatter(g.common_joint_difference, y, marker=marker, color=color, label=label, s=40, zorder=3)
    labels = [f"{r.cutoff[:4]} / s{int(r.seed)-20260909}" for r in g.itertuples()]
    ax[0].set_yticks(np.arange(9),labels)
    ax[0].invert_yaxis()
    ax[0].axvline(0,color='#72828d',lw=1)
    ax[0].set_xlabel('Huber-trained minus MSE-trained common loss')
    ax[0].set_title('Frozen states: positive favors MSE-trained weights')
    ax[0].legend(frameon=False,loc='best',fontsize=9)
    for state, label in [('archived_mse','MSE-trained'),('archived_huber','Huber-trained')]:
        a = gradient[gradient.state.eq(state) & gradient['mode'].eq('eval') & gradient.parameter_group.eq('shared')].set_index(['cutoff','seed'])
        b = gradient[gradient.state.eq(state) & gradient['mode'].eq('dropout_mean4') & gradient.parameter_group.eq('shared')].set_index(['cutoff','seed']).loc[a.index]
        ax[1].scatter(a.cosine,b.cosine,color=COLORS[state],label=label,s=48,alpha=.9)
    ax[1].axhline(0,color='#72828d',lw=1)
    ax[1].axvline(0,color='#72828d',lw=1)
    ax[1].set_xlim(-1,1); ax[1].set_ylim(-1,1)
    ax[1].set_xlabel('Return/auxiliary gradient cosine: dropout off')
    ax[1].set_ylabel('Cosine of mean gradients: 4 dropout draws')
    ax[1].set_title('Shared parameters: local gradient alignment')
    ax[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Round 11 | Frozen training-state diagnosis',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.01)
    fig.text(.06,-.04,'Left: common loss = scaled return Huber + 0.1 x auxiliary MSE; nine matched fold/seed pairs.\nRight: full-cohort gradients before clipping or AdamW. Negative cosine is local conflict, not proof of harmful prediction effects.',fontsize=9,color='#425563')
    fig.tight_layout(); export(fig,'frozen_training_diagnosis')
    monitors = pd.read_csv(OUT/'restart_training_monitors.csv')
    fig, ax = plt.subplots(2,2,figsize=(13,9))
    for state,label in [('joint','Joint loss'),('return_only','Return only')]:
        g = monitors[monitors.state.eq(state)]
        for col,key in enumerate(['return_huber','common_joint']):
            for _,case in g.groupby(['cutoff','seed']):
                ax[0,col].plot(case.epoch,case[key],color=COLORS[state],alpha=.16,lw=.9)
            mean = g.groupby('epoch')[key].mean()
            ax[0,col].plot(mean.index,mean.values,color=COLORS[state],lw=2.8,marker='o',label=label)
            ax[0,col].set_xticks([0,5,10,15,20])
            ax[0,col].set_xlabel('Restart epoch (fresh AdamW, fixed lr=0.0001)')
            ax[0,col].legend(frameon=False,fontsize=9)
    ax[0,0].set_title('Return Huber training loss; dropout off')
    ax[0,1].set_title('Common joint training loss; dropout off')
    for col,key,title in [(0,'return_huber','Return Huber loss'),(1,'auxiliary_mse','Auxiliary MSE')]:
        g = pairs[pairs.comparison.eq('return_only_minus_joint') & pairs['mode'].eq('dropout')].sort_values(['cutoff','seed'])
        x = g[key+'_difference'].to_numpy()
        ax[1,col].barh(np.arange(9),x,color=np.where(x<0,'#167d9a','#bc5267'),height=.62)
        ax[1,col].set_yticks(np.arange(9),labels)
        ax[1,col].invert_yaxis()
        ax[1,col].axvline(0,color='#72828d',lw=1)
        ax[1,col].set_xlabel('Return-only minus joint; negative is lower')
        ax[1,col].set_title(f'{title}: fixed epoch 20, 8-draw dropout mean')
    fig.suptitle('Round 11 | Paired training restarts from Huber20 weights',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.01)
    fig.text(.06,-.03,'Top: thin lines show all nine starts; thick lines are descriptive means. Bottom: all nine paired endpoints, no best-seed selection.\nBoth arms retain dropout 0.3, weight decay 0.1, original batches and 20 epochs. These are training losses; no validation was scored.',fontsize=9,color='#425563')
    fig.tight_layout(); export(fig,'paired_training_restarts')


if __name__=='__main__':
    main()

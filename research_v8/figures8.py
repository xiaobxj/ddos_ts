"""Standalone scientific plots; all six policies are shown without ranking."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
POLICIES=['full','disjoint0','disjoint1','disjoint2','quarter_equal','recent3y']
NAMES=['Full history','Disjoint phase 0','Disjoint phase 1','Disjoint phase 2','Equal quarter mass','Recent 3 years']
COLORS=['#435b70','#2a788e','#22a884','#7ad151','#e79a33','#bd5b79']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})


def export(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)


def audit():
    a=pd.read_csv(OUT/'training_loss_attribution.csv')
    g=a[(a.cutoff=='2019-12-31')&(a.grouping=='year')].copy()
    fig,ax=plt.subplots(1,2,figsize=(13,4.9),gridspec_kw={'width_ratios':[1.35,1]})
    x=np.arange(len(g))
    ax[0].bar(x-.19,g.sample_share*100,.38,color='#829db3',label='Share of training rows')
    ax[0].bar(x+.19,g.fitted_loss_share*100,.38,color='#ba6447',label='Share of fitted squared error')
    ax[0].set_xticks(x,g.period);ax[0].set_ylabel('Share (%)');ax[0].legend(frameon=False,fontsize=9)
    ax[0].set_title('Loss contribution differs by historical year')
    d=pd.read_csv(OUT/'policy_definitions.csv');x=np.arange(3)
    for offset,policy,color in [(-.25,'full',COLORS[0]),(0,'disjoint0',COLORS[1]),(.25,'recent3y',COLORS[5])]:
        h=d[d.policy==policy];ax[1].bar(x+offset,h.train_n,.25,label=NAMES[POLICIES.index(policy)],color=color)
    ax[1].set_xticks(x,['2018 fold','2019 fold','2020 fold']);ax[1].set_ylabel('Unique training anchors')
    ax[1].set_title('Disjoint labels leave roughly one sixth of rows');ax[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Round 8 | Training-only audit',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.01)
    fig.text(.06,-.02,'Left: cutoff 2019-12-31; fitted loss is averaged across three frozen models, before summing by year.\nRight: phases 1 and 2 have similar row counts. Repeated presentations preserve the full-history optimizer budget.',fontsize=9,color='#44515c')
    fig.tight_layout();export(fig,'training_audit')


def results():
    c=pd.read_csv(OUT/'policy_comparisons.csv').set_index('policy').loc[POLICIES]
    seeds=pd.read_csv(OUT/'seed_metrics.csv');m=pd.read_csv(OUT/'ensemble_metrics.csv')
    fig,ax=plt.subplots(1,3,figsize=(17,6),gridspec_kw={'width_ratios':[1.35,1,1]})
    y=np.arange(6)
    ax[0].barh(y-.15,c.rmse*1e4,.28,color=COLORS,label='Crossformer, 3-seed mean')
    ax[0].barh(y+.15,c.ridge_rmse*1e4,.28,color='#bcc7ce',label='Fixed Ridge control')
    ax[0].axvline(c.full_mean_rmse.iloc[0]*1e4,color='#aa3a31',ls='--',lw=1.4,label='Common full-history mean')
    ax[0].set_yticks(y,NAMES);ax[0].invert_yaxis();ax[0].set_xlabel('Return RMSE (basis points; lower is better)')
    ax[0].set_title('Forecast error');ax[0].legend(frameon=False,fontsize=8,loc='lower right')
    ax[1].barh(y,c.mse_skill_vs_common_mean*100,color=COLORS)
    ax[1].axvline(0,color='#5c6269',lw=1);ax[1].set_yticks(y,[]);ax[1].invert_yaxis()
    ax[1].set_xlabel('MSE improvement over common mean (%)');ax[1].set_title('Neural ensemble vs a shared baseline')
    for i,policy in enumerate(POLICIES):
        h=seeds[seeds.policy==policy].sort_values('seed')
        ax[2].scatter(h.mse_skill_vs_training_mean*100,i+np.array([-.12,0,.12]),c=COLORS[i],s=37,alpha=.85)
        ax[2].scatter(c.loc[policy,'mse_skill_vs_own_mean']*100,i,c='black',marker='D',s=24,zorder=4)
    ax[2].axvline(0,color='#5c6269',lw=1);ax[2].set_yticks(y,[]);ax[2].invert_yaxis()
    ax[2].set_xlabel('MSE improvement over own training mean (%)');ax[2].set_title('Seed sensitivity')
    ax[2].scatter([],[],c='gray',label='Each seed');ax[2].scatter([],[],c='black',marker='D',label='3-seed ensemble')
    ax[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Round 8 | Fixed sampling policies, 141 historical validation weeks',fontsize=17,fontweight='bold',x=.03,ha='left',y=1.01)
    fig.text(.03,-.02,'All policies are prespecified; no best phase or seed is selected. These 2018-2020 dates have been viewed repeatedly.\nThree disjoint phases use different training anchors but are neither independent datasets nor an exhaustive phase sweep.',fontsize=9,color='#44515c')
    fig.tight_layout();export(fig,'policy_results')
    years=pd.read_csv(OUT/'yearly_metrics.csv');a=years[years.model=='neural']
    matrix=np.array([[a[(a.policy==p)&(a.year==year)].mse_skill_vs_full_mean.iloc[0]*100 for year in [2018,2019,2020]] for p in POLICIES])
    fig,ax=plt.subplots(figsize=(8.5,5.5));limit=max(abs(matrix.min()),abs(matrix.max()),1)
    im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(np.arange(3),['2018','2019','2020']);ax.set_yticks(np.arange(6),NAMES)
    for i in range(6):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:+.1f}%',ha='center',va='center',color='white' if abs(matrix[i,j])>.55*limit else '#17222d')
    fig.colorbar(im,ax=ax,label='MSE improvement over common full-history mean (%)',shrink=.8)
    ax.set_title('Year-by-year neural ensemble performance',pad=14)
    fig.tight_layout();export(fig,'yearly_performance')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    audit()
    if not args.audit_only:results()

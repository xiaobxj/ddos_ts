"""Standalone figures for fixed learning-rate schedules and their complete results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'results'
COLORS={'archived20':'#798a99','prefix20':'#798a99','constant40':'#bd6050','decay40':'#157e97'}
NAMES={'archived20':'Archived MSE20','prefix20':'Shared MSE20','constant40':'Constant LR, 40 epochs','decay40':'Step decay, 40 epochs'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})


def export(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=165,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    prefix=pd.read_csv(OUT/'prefix_training_monitors.csv')
    branches=pd.read_csv(OUT/'continuation_training_monitors.csv')
    fig,ax=plt.subplots(1,3,figsize=(15,4.8))
    x=np.arange(1,41)
    ax[0].plot(x,np.full(40,.001),lw=2.7,color=COLORS['constant40'],label='Constant')
    ax[0].step(x,np.r_[np.full(20,.001),np.full(20,.0001)],where='post',lw=2.7,color=COLORS['decay40'],label='Step decay')
    ax[0].axvspan(1,20,color='#dce2e6',alpha=.45)
    ax[0].set_yscale('log');ax[0].set_yticks([.0001,.001],['0.0001','0.001'])
    ax[0].set_ylim(.00007,.00145);ax[0].set_xlabel('Training epoch');ax[0].set_ylabel('AdamW learning rate')
    ax[0].set_title('Identical first 20 epochs');ax[0].legend(frameon=False,loc='center left')
    for col,key,title in [(1,'return_mse','Return MSE on training rows'),(2,'joint_mse','Joint MSE on training rows')]:
        start=prefix[prefix.epoch.eq(20)]
        for name in ['constant40','decay40']:
            g=pd.concat([start.assign(schedule=name),branches[branches.schedule.eq(name)]],ignore_index=True)
            for _,case in g.groupby(['cutoff','seed']):
                case=case.sort_values('epoch');ax[col].plot(case.epoch,case[key],color=COLORS[name],alpha=.16,lw=.9)
            m=g.groupby('epoch')[key].mean()
            ax[col].plot(m.index,m.values,color=COLORS[name],lw=2.6,marker='o',label='Constant' if name=='constant40' else 'Step decay')
        ax[col].set_title(title);ax[col].set_xticks([20,25,30,35,40]);ax[col].set_xlabel('Training epoch')
        ax[col].legend(frameon=False,fontsize=9)
    fig.suptitle('Round 12 | Equal-budget learning-rate schedules',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.03)
    fig.text(.06,-.035,'All nine shared prefixes exactly match archived weights; both branches preserve optimizer moments and random streams.\nTraining plots: dropout off, thin lines = all nine cases, thick lines = descriptive means. AdamW decay includes learning rate.',fontsize=9,color='#425563')
    fig.tight_layout();export(fig,'learning_rate_and_training')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('schedule')
    seeds=pd.read_csv(OUT/'seed_metrics.csv');years=pd.read_csv(OUT/'yearly_metrics.csv')
    baselines=pd.read_csv(OUT/'baseline_metrics.csv').set_index('reference')
    fig,ax=plt.subplots(1,2,figsize=(13,5.3));x=np.arange(3)
    for i,name in enumerate(['archived20','constant40','decay40']):
        g=seeds[seeds.schedule.eq(name)].sort_values('seed')
        ax[0].scatter(i,metrics.loc[name,'rmse']*1e4,color=COLORS[name],marker='D',s=90,zorder=3)
        ax[0].scatter(i+np.array([-.16,0,.16]),g.rmse*1e4,color=COLORS[name],s=32,zorder=4)
        for j,value in enumerate(g.rmse):
            ax[0].annotate(str(j+1),(i+[-.16,0,.16][j],value*1e4),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        annual=years[years.schedule.eq(name)].sort_values('year')
        ax[1].bar(x+(i-1)*.24,annual.mse_skill_vs_training_mean*100,.24,color=COLORS[name],label=NAMES[name])
    ax[0].axhline(baselines.loc['training_mean','rmse']*1e4,color='#2e596d',ls='--',label='Training mean')
    ax[0].axhline(baselines.loc['zero_return','rmse']*1e4,color='#8c6b9c',ls=':',label='Zero return')
    ax[0].set_xticks(x,['Archived20','Constant40','Decay40']);ax[0].set_ylabel('Return RMSE (bp; lower is better)')
    ax[0].set_title('Diamonds: ensemble; small dots: seeds 1/2/3');ax[0].legend(frameon=False,fontsize=9)
    low=min(seeds.rmse.min(),metrics.rmse.min(),baselines.rmse.min())*1e4
    high=max(seeds.rmse.max(),metrics.rmse.max(),baselines.rmse.max())*1e4
    ax[0].set_ylim(max(0,low-7),high+12)
    ax[1].axhline(0,color='#72818b',lw=1);ax[1].set_xticks(x,['2018','2019','2020'])
    ax[1].set_ylabel('Ensemble MSE improvement over training mean (%)')
    ax[1].set_title('Annual results, all fixed schedules');ax[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Round 12 | Same 141 historical validation weeks',fontsize=16,fontweight='bold',x=.06,ha='left',y=1.03)
    fig.text(.06,-.035,'Epoch40 is the predetermined endpoint. No best seed, intermediate validation checkpoint or replacement candidate is selected.\nLeft axis is zoomed; all three seeds are shown. Repeated historical reuse means these results remain exploratory.',fontsize=9,color='#425563')
    fig.tight_layout();export(fig,'learning_rate_validation')


if __name__=='__main__':main()

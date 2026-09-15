"""Standalone plots of the fixed candidate and multiseed historical sensitivity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'
HISTORIES=['full','drop_early','drop_middle','drop_recent']
LABELS=['Full history\n(reused)','Omit earliest\n20%','Omit middle\n20%','Omit latest\n20%']
COLORS={'candidate20':'#267da2','baseline10':'#c4772b','combined10':'#7d8095','baseline20':'#bb525d'}
NAMES={'candidate20':'Fixed candidate, 20 epochs','baseline10':'Frozen baseline, 10 epochs',
       'combined10':'Combined, 10 epochs (diagnostic)','baseline20':'Baseline, 20 epochs (diagnostic)'}


def main():
    audit=json.loads((OUT/'verification.json').read_text(encoding='utf-8'));assert audit['status']=='PASS'
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv'); seeds=pd.read_csv(OUT/'seed_metrics.csv')
    shifts=pd.read_csv(OUT/'prediction_sensitivity.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.3),layout='constrained')
    x=np.arange(4)
    for offset,name in [(-.19,'candidate20'),(.19,'baseline10')]:
        g=metrics[metrics.role==name].set_index('history').loc[HISTORIES]
        axes[0].bar(x+offset,g.mse_skill_vs_training_mean*100,width=.36,color=COLORS[name],label=NAMES[name])
    for offset,(number,g) in zip([-.13,0,.13],seeds[seeds.role=='candidate20'].groupby('seed')):
        g=g.set_index('history').loc[HISTORIES]
        axes[1].scatter(x+offset,g.mse_skill_vs_training_mean*100,label=f'Seed {number}',s=46,alpha=.75)
    g=metrics[metrics.role=='candidate20'].set_index('history').loc[HISTORIES]
    axes[1].plot(x,g.mse_skill_vs_training_mean*100,color=COLORS['candidate20'],marker='D',linewidth=2,label='Three-seed mean forecasts')
    for ax in axes:
        ax.axhline(0,color='#555555',linestyle='--',linewidth=1)
        ax.set_xticks(x,LABELS)
        ax.set_ylabel('MSE improvement over retained training mean (%)')
        ax.axvspan(-.45,.45,color='#999999',alpha=.07)
        ax.legend(fontsize=9,frameon=False)
    axes[0].set_title('Candidate vs the baseline at its selected training length')
    axes[1].set_title('Candidate seed consistency')
    fig.suptitle('Fixed 20-epoch candidate | 141 reused validation weeks, 2018-2020\n'
                 'Positive is better; history deletion uses equal update budgets',fontsize=13)
    fig.savefig(OUT/'candidate_stability.png',dpi=180);fig.savefig(OUT/'candidate_stability.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.2),layout='constrained')
    for name in ['candidate20','baseline10','combined10']:
        g=shifts[(shifts.role==name)&(shifts.level=='ensemble')].set_index('history').loc[HISTORIES[1:]]
        axes[0].plot(range(3),g.prediction_rms_shift*10000,marker='o',color=COLORS[name],label=NAMES[name])
    axes[0].set_xticks(range(3),['Omit earliest','Omit middle','Omit latest'])
    axes[0].set(ylabel='RMS forecast change from full history (bp)',title='Sensitivity of the three-seed ensemble')
    axes[0].legend(fontsize=9,frameon=False)
    for offset,name in [(-.19,'combined10'),(.19,'candidate20')]:
        g=metrics[metrics.role==name].set_index('history').loc[HISTORIES]
        axes[1].bar(x+offset,g.within_fold_forecast_std*10000,width=.36,color=COLORS[name],label=NAMES[name])
    axes[1].set_xticks(x,LABELS)
    axes[1].set(ylabel='Within-year forecast standard deviation (bp)',title='Forecast variation after removing annual means')
    axes[1].legend(fontsize=9,frameon=False)
    fig.suptitle('Forecast stability and near-constant-output diagnostics\n'
                 'Smaller forecast changes alone do not establish predictive skill',fontsize=13)
    fig.savefig(OUT/'forecast_diagnostics.png',dpi=180);fig.savefig(OUT/'forecast_diagnostics.svg');plt.close(fig)


if __name__=='__main__':
    main()

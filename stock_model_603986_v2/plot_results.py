"""Standalone publication-style comparison chart; run using desktop Python."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})
y=pd.read_csv(OUT/'year_metrics.csv');r=pd.read_csv(OUT/'reliability.csv')
fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
names=[('mse_5y_e20.vol.U','预先固定的主模型','#4169a1'),('frequency_5y','训练上涨频率','#e0993b'),('market4_5y','四项行情特征','#3b9b88')]
for col,h in enumerate([1,5]):
    ax=axes[0,col];years=sorted(y.year.unique());xx=np.arange(len(years))
    for j,(method,name,color) in enumerate(names):
        v=y[(y.h==h)&y.method.eq(method)].sort_values('year')
        bars=ax.bar(xx+(j-1)*.25,v.brier,width=.23,label=name,color=color)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.axhline(.25,color='#606060',linestyle='--',lw=1,label='固定50%概率')
    ax.set_xticks(xx,[str(year)+('（部分年）' if year==2026 else '') for year in years]);ax.set_ylabel('Brier：越低越好')
    ax.set_title(('快：下一交易日' if h==1 else '慢：未来5个交易日')+'｜年度概率误差')
    ax.set_ylim(0,max(.36,float(y[(y.h==h)&y.method.isin([n[0] for n in names])].brier.max())*1.28));ax.grid(axis='y',alpha=.15);ax.legend(fontsize=9,ncol=2,loc='upper center')
    ax=axes[1,col];ax.plot([0,1],[0,1],color='#888888',ls='--',lw=1)
    for method,name,color in names:
        v=r[(r.h==h)&r.method.eq(method)&r.n.gt(0)]
        ax.scatter(v.mean_probability,v.actual_up_frequency,s=35+np.sqrt(v.n)*5,color=color,label=name,alpha=.8)
        if method==names[0][0]:
            for row in v.itertuples():ax.annotate(str(row.n),(row.mean_probability,row.actual_up_frequency),xytext=(5,7),textcoords='offset points',fontsize=9,color=color)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='分箱内平均模型概率',ylabel='分箱内实际上涨比例',title='固定0.1宽度概率箱｜蓝点旁为主模型样本数')
    ax.grid(alpha=.15);ax.legend(fontsize=9,loc='upper left')
fig.suptitle('603986 兆易创新｜快慢模型的固定历史测试',fontsize=20,weight='bold')
fig.supxlabel('2023年起至2026年9月15日已成熟标签；H5日信号互有重叠，样本数不等于独立周数。历史探索结果。',fontsize=10)
fig.savefig(ROOT/'快慢模型测试.png',dpi=160,facecolor='white')

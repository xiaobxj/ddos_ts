"""Standalone research figure from sealed numerical outputs."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def rows(n):
    with (OUT/(n+'.csv')).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    yearly=rows('slow_years');fast=rows('fast_metrics');reliability=rows('slow_reliability')
    fig,axes=plt.subplots(2,2,figsize=(12.5,9.3));fig.patch.set_facecolor('#f6f8fc')
    styles=[('annual.vol.raw','年度原始','#60778b'),('quarterly.vol.raw','季度原始','#206ca8'),('quarterly.vol.shrink','季度收缩','#b85e28')]
    for name,label,color in styles:
        r=[v for v in yearly if v['name']==name];years=[int(v['year']) for v in r]
        axes[0,0].plot(years,[float(v['brier']) for v in r],marker='o',label=label,color=color)
        axes[0,1].plot(years,[100*float(v['accuracy']) for v in r],marker='o',label=label,color=color)
    axes[0,0].axhline(.25,ls='--',color='#90969c',lw=1,label='恒定 50% 的 Brier')
    axes[0,0].set(title='慢周期：年度概率误差',ylabel='Brier，越低越好',xticks=years,ylim=(.242,.29));axes[0,0].legend(fontsize=9)
    axes[0,1].axhline(50,ls='--',color='#90969c',lw=1)
    axes[0,1].set(title='慢周期：近期改善没有延伸到所有年份',ylabel='方向正确率（%）',xticks=years,ylim=(32,73));axes[0,1].legend(fontsize=9)
    order=['frequency','own','market','sector','both'];labels=['历史频率基准','个股短期特征','加沪深300','加国证芯片','同时加入两者']
    values=[float(next(r['brier'] for r in fast if r['name']==n)) for n in order]
    axes[1,0].scatter(values,range(len(labels)),color=['#798e9e']+['#3d86b2']*4,s=65)
    axes[1,0].set_yticks(range(len(labels)),labels)
    axes[1,0].axvline(.25,ls='--',color='#90969c',lw=1)
    for y,v in enumerate(values):axes[1,0].text(v+.00015,y,f'{v:.4f}',va='center',fontsize=9)
    axes[1,0].invert_yaxis();axes[1,0].set(title='快周期：外部特征尚未改善概率质量',xlabel='Brier，越低越好',xlim=(.2475,.255))
    for name,label,color in styles[1:]:
        r=[v for v in reliability if v['name']==name and int(v['n'])>=10]
        axes[1,1].scatter([100*float(v['p_mean']) for v in r],[100*float(v['up_rate']) for v in r],
            s=[int(v['n'])/1.8 for v in r],alpha=.75,color=color,label=label)
    axes[1,1].plot([0,100],[0,100],ls='--',color='#90969c',lw=1)
    axes[1,1].set(title='慢周期：高概率仍存在明显过度自信',xlabel='区间内平均预测概率（%）',ylabel='区间内实际上涨比例（%）',xlim=(0,100),ylim=(0,100));axes[1,1].legend(fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('603986 快慢周期验证｜2023—2026 年已成熟样本',fontsize=17,y=.99)
    fig.text(.5,.017,'H1：897 个日度样本；H5：893 个重叠日度样本。2026 年截至 9 月 15 日可结算数据。\n圆点面积代表样本量，只显示至少 10 条的概率区间；历史已多轮研究，所有结果仍属探索。',ha='center',fontsize=9,color='#566875')
    fig.tight_layout(rect=(0,.06,1,.965));fig.savefig(OUT/'comparison.png',dpi=180,facecolor=fig.get_facecolor());fig.savefig(OUT/'comparison.svg',facecolor=fig.get_facecolor())

if __name__=='__main__':main()

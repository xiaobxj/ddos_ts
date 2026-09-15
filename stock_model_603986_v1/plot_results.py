"""Standalone retrospective research figure; run with desktop Python/matplotlib."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
U='rolling5_annual20';Q='weekly_state_validated';W='annual_head_timeweight2y'
I='annual_head_weighted_intercept';S='annual_head_weighted_slopes'
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset']
HISTORIES=[U,Q,W,I,S]

def main():
    with (ROOT/'results/metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    def get(period,history,method):return next(r for r in rows if r['period']==period and r['history']==history and r['method']==method)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,(ax,heat)=plt.subplots(2,1,figsize=(11.6,8.1),gridspec_kw={'height_ratios':[1,1.25]})
    fig.patch.set_facecolor('#f8fafc')
    x=np.arange(4);years=['2023','2024','2025','2026']
    for off,history,method,name,color in [(-.25,U,'learned_vol_interaction','主模型：波动交互','#2563eb'),
          (0,'control','training_frequency','训练期上涨频率','#94a3b8'),(.25,'control','market4','简单行情模型','#0d9488')]:
        values=[float(get(y,history,method)['accuracy'])*100 for y in years]
        bars=ax.bar(x+off,values,width=.23,label=name,color=color)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=3,fontsize=9)
    labels=[f'{y}\n{get(y,U,"learned_vol_interaction")["weeks"]}周' for y in years]
    labels[-1]+='，截至9月4日'
    ax.set_xticks(x,labels);ax.set_ylim(0,100);ax.set_ylabel('方向准确率（%）')
    ax.axhline(50,color='#cbd5e1',ls='--',lw=1);ax.legend(loc='upper left',ncol=3,frameon=False)
    ax.set_title('603986 个股周度模型：逐年滚动检验',loc='left',fontsize=16,pad=17,fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    matrix=np.array([[float(get('pooled',h,m)['brier']) for m in METHODS] for h in HISTORIES])
    im=heat.imshow(matrix,cmap='YlOrRd',aspect='auto')
    heat.set_xticks(range(4),['神经特征＋行情','波动交互','涨跌顺序交互','涨跌顺序偏移'])
    heat.set_yticks(range(5),['年度重训','季度状态校准','近期样本加权','仅加权截距','仅加权斜率'])
    for i in range(5):
        for j in range(4):
            a=float(get('pooled',HISTORIES[i],METHODS[j])['accuracy'])*100
            heat.text(j,i,f'{matrix[i,j]:.4f}  /  {a:.1f}%',ha='center',va='center',fontsize=10,
                color='white' if matrix[i,j]>(matrix.min()+matrix.max())*.5 else '#172033')
    heat.set_title('全部固定组合：Brier（越低越好）／方向准确率',loc='left',pad=13,fontsize=12)
    cbar=fig.colorbar(im,ax=heat,pad=.02,fraction=.03);cbar.set_label('Brier')
    n=get('pooled',U,'learned_vol_interaction')['weeks']
    fig.text(.07,.02,f'同一批{n}个成熟周样本；每个输出先平均3个种子。历史重算，未扣费税，不代表可交易收益或未来准确率。',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.16,right=.94,top=.90,bottom=.10,hspace=.48)
    fig.savefig(ROOT/'603986_滚动检验.png',dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=='__main__':main()

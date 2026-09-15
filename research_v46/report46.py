from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
METHODS=['learned_vol_interaction','learned_order_extension','learned_order_offset'];MN=dict(zip(METHODS,['R19','R23','R25']))
ANNUAL='rolling5_annual20';QUARTER='weekly_state_validated'
PARENTS={'trend':'quarter_trend2_validated','volatility':'quarter_volatility2_validated'}
POLICIES={'trend':'quarter_trend2_state4_validated','volatility':'quarter_volatility2_state4_validated'}
HIST=[ANNUAL,QUARTER]+list(PARENTS.values())+list(POLICIES.values())
HN=dict(zip(HIST,['年度基线','原四状态季度','趋势两组验证（R45）','波动两组验证（R45）','趋势共享／四状态验证','波动共享／四状态验证']))
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'report_manifest.json').exists();v=read('verification.json');assert v['status']=='PASS'
    met=csv('ensemble_metrics');seed=csv('seed_metrics');support=csv('support_summary');gates=csv('gate_decisions');outcomes=csv('decision_outcomes');e=csv('ensemble_predictions');pairs=read('primary_comparisons.json')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
    colors=['#555555','#007DB5','#D0657C','#009E8E'];styles=[':', '--','-.','-']
    fig,axes=plt.subplots(2,3,figsize=(16,8.5),sharey=True)
    for row,family in enumerate(POLICIES):
        histories=[ANNUAL,QUARTER,PARENTS[family],POLICIES[family]]
        for ax,m in zip(axes[row],METHODS):
            for h,col,style in zip(histories,colors,styles):
                g=met[met.history.eq(h)&met.method.eq(m)&met.period.str.startswith('year_')].sort_values('period')
                ax.plot(range(2021,2027),g.accuracy,color=col,linestyle=style,marker='o',markersize=4,label=HN[h])
            ax.set_title(MN[m]+(' · 趋势共享' if family=='trend' else ' · 波动共享'));ax.set_xticks(range(2021,2027),['2021','2022','2023','2024','2025','2026*']);ax.set_ylim(.15,.85);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
        axes[row,0].set_ylabel('三种子集成方向准确率')
        axes[row,1].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=10)
    fig.suptitle('共享拟合参数后，按四状态分别验证是否启用',fontsize=19)
    fig.text(.5,.016,'每年周数 50、49、48、47、48、30；* 2026 年截至 8 月。全部历史已查看；曲线重合时可查完整数值表。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.07,right=.98,top=.9,bottom=.16,hspace=.6,wspace=.16)
    for ext in ['png','svg']:fig.savefig(OUT/f'yearly_accuracy.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5.6));schemes=['original4','parent_trend','child_trend','parent_volatility','child_volatility'];labels=['原四状态','趋势两组\nR45','趋势共享\n四状态验证','波动两组\nR45','波动共享\n四状态验证'];s=support.set_index('scheme').loc[schemes]
    vals=s.validation_support_fraction.to_numpy()*100;axes[0].bar(range(5),vals,color=['#007DB5','#D0657C','#009E8E','#D0657C','#009E8E'])
    for i,r in enumerate(s.itertuples()):axes[0].text(i,vals[i]+1,f'{r.validation_supported_cells}/{r.ready_cells}',ha='center')
    axes[0].set_xticks(range(5),labels,fontsize=9);axes[0].set_ylim(0,65);axes[0].set_ylabel('同时满足训练／验证数量下限（%）');axes[0].set_title('四状态验证仍受原样本支持限制')
    ei=e.set_index(['history','method','row_index']);coverhist=[QUARTER,PARENTS['trend'],POLICIES['trend'],PARENTS['volatility'],POLICIES['volatility']]
    for j,(h,col) in enumerate(zip(coverhist,['#007DB5','#DCA2AB','#009E8E','#D0657C','#67BCA7'])):
        counts=[]
        for m in METHODS:
            g=e[e.history.eq(h)&e.method.eq(m)&e.date.ge('2024-01-01')];counts.append(sum(r.probability!=ei.loc[(ANNUAL,m,r.row_index),'probability'] for r in g.itertuples()))
        x=np.arange(3)+(j-2)*.16;axes[1].bar(x,counts,width=.15,color=col,label=HN[h])
        for xx,n in zip(x,counts):axes[1].text(xx,n+.7,str(n),ha='center',fontsize=9)
    axes[1].set_xticks(range(3),[MN[m] for m in METHODS]);axes[1].set_ylim(0,55);axes[1].set_ylabel('近期125周中相对年度改变概率的周数');axes[1].set_title('验证通过后实际影响的预测');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('数量支持与实际覆盖分别报告',fontsize=18);fig.text(.5,.012,'左侧分母随分组数变化；共享的是拟合参数，没有放宽每个子状态训练≥10、验证≥5的要求。',ha='center',fontsize=10);fig.subplots_adjust(left=.07,right=.98,top=.81,bottom=.29,wspace=.24)
    for ext in ['png','svg']:fig.savefig(OUT/f'support_and_coverage.{ext}',dpi=170)
    plt.close(fig)
    lines=['# 第四十六轮：共享拟合参数，按原四状态分别验证启用','',
    '本轮预先固定两个候选：共享第45轮趋势两组参数、共享第45轮波动两组参数，再按原四状态分别决定是否启用。588个已拟合的标量参数以及全部1,104条头记录逐字节复用，没有新增神经网络或校准拟合。独立复核重新求解原训练目标仅用于确认参数来源与最优性。',
    '', '原年度模型、滚动5年自然20遍训练、市场和订单交互保持。沿用23个季度截止、原52／13成熟周训练验证划分，训练标签严格早于首个验证信号成熟。每个子状态仍需训练至少10周、验证至少5周；三种子平均概率在验证上的Brier改善超过1e-12，正确数不减少，才使用对应共享参数。共享参数的合并组在R45是否通过，不参与本轮门控。未通过时精确回退年度预测；没有叠加修正或验证后再拟合。',
    '', '这项干预改变了原四状态拟合所用的参数来源，但没有解决子状态验证样本不足。与R45相比，它改变了验证分组及子状态数量支持要求；本轮不另行尝试放宽门槛。四状态边界、年末重置、第一季度回退、季度末当日使用前一次决定和原逐组Series.mean集成算法均保留。','']
    for period,title in [('extension_2021_2023','早期：2021—2023，147周'),('recent_2024_2026','近期：2024—2026年8月，125周'),('pooled_2021_2026','全段：272周'),('year_2026','2026年截至8月：30周')]:
        lines += [f'## {title}','','| 方法 | 方案 | 正确／周数 | 准确率 | Brier↓ | 对数损失↓ | AUROC |','|---|---|---:|---:|---:|---:|---:|']
        for m in METHODS:
            for h in HIST:
                r=met[met.period.eq(period)&met.method.eq(m)&met.history.eq(h)].iloc[0];lines.append(f'| {MN[m]} | {HN[h]} | {r.correct_directions}/{r.n} | {r.accuracy:.2%} | {r.brier:.6f} | {r.log_loss:.6f} | {r.auroc:.6f} |')
        lines+=['']
    lines+=['## 数量支持与门控','','| 方案 | 历史具备的分组单元 | 训练≥10 | 同时验证≥5 | 同时满足比例 |','|---|---:|---:|---:|---:|']
    for scheme,label in zip(schemes,labels):
        r=s.loc[scheme];lines.append(f'| {label.replace(chr(10)," ")} | {int(r.ready_cells)} | {int(r.training_supported_cells)} | {int(r.validation_supported_cells)} | {r.validation_support_fraction:.1%} |')
    lines+=['','数量支持并不等于验证通过，且分母是13个可拟合季度乘以分组数，不是独立市场事件。以下门控转移按原四状态单元比较，合并组决定展开到其两个子状态，因此不能把分母或重复方法当成独立试验。','','| 方法 | 共享来源 | 新通过 | 两者通过 | 合并组拒绝／子状态通过 | 合并组通过／子状态拒绝 |','|---|---|---:|---:|---:|---:|']
    for m in ['learned_market']+METHODS:
        for f in POLICIES:
            g=outcomes[outcomes.method.eq(m)&outcomes.family.eq(f)];lines.append(f'| {MN.get(m,"R18（诊断）")} | {f} | {int(g.accepted.sum())}/{len(g)} | {sum(g.accepted & g.parent_accepted)} | {sum(g.accepted & ~g.parent_accepted)} | {sum(~g.accepted & g.parent_accepted)} |')
    lines+=['','## 近期三个种子诊断','','| 方法 | 候选 | 种子 | 正确／125 | Brier↓ |','|---|---|---|---:|---:|']
    for r in seed[seed.period.eq('recent_2024_2026')&seed.method.isin(METHODS)&seed.history.isin(POLICIES.values())].sort_values(['method','history','seed']).itertuples():lines.append(f'| {MN[r.method]} | {HN[r.history]} | {r.seed} | {r.correct_directions} | {r.brier:.6f} |')
    lines+=['','单种子仅作诊断，不用于选择候选。原四状态分组结果、736个门控决定对应的未来结果（含零周）、逐周变化及全部年份均已保存。','',
    '## 统计与审计','',
    f"预先固定72项探索性比较：两个候选各对照对应R45合并验证、原四状态季度和年度基线，三个主要方法、两种损失、两个时期。循环8周区块重采样10,000次，固定种子20260910，中心化双侧p值；全部72项统一Holm校正。最小原始p={min(r['p'] for r in pairs):.6f}；最小校正p={min(r['holm_adjusted_p'] for r in pairs):.6f}；校正后p<0.05共{sum(r['holm_adjusted_p']<.05 for r in pairs)}项。",
    '',f"独立复核PASS：{v['independent_scalar_solutions']}个原标量最优解，最大系数差{v['maximum_coefficient_gap']:.3g}；{v['independent_gate_cells']}个子状态门控；{v['independent_validation_seed_rows']}条验证种子概率；{v['independent_learned_seed_predictions']}条学习方法种子预测；{v['independent_metric_cells']}个指标单元；{v['independent_weekly_effects']}条逐周效果及{v['independent_decision_outcomes']}个决定后续单元。{v['exact_ensemble_fallback_predictions']}条零修正集成概率与年度精确相同。23个训练截止的非训练输入污染及23个验证截止的未来输入污染均未改变相应结果。",
    '', '两条新预测历史连同控制新增8,704条模型记录。原25条预测历史保持不变，5,986个旧文件及继承源代码、协议、输入经过哈希核验。原第45轮浮点归约问题的无效尝试继续留档；本轮使用修复后的R45成果并保留精确回退检查。',
    '', '全部历史此前已查看，没有新增盲测，多重比较校正不覆盖此前多轮自适应研究。本轮不能证明独立泛化能力，不作交易收益或部署结论。原旧预算R25近期73/125、58.4%，自然20遍年度R25为71/125、56.8%，原四状态季度R23为73/125、58.4%，分别保留。',
    '', '![各年方向准确率](yearly_accuracy.png)','','![数量支持与实际覆盖](support_and_coverage.png)','',
    '[完整指标](ensemble_metrics.csv) · [逐周变化](weekly_policy_effects.csv) · [门控及后续结果](decision_outcomes.csv) · [原四状态指标](state_metrics.csv) · [验证种子预测](validation_predictions.csv) · [72项比较](primary_comparisons.json) · [独立复核](verification.json)']
    p=OUT/'第四十六轮实验报告.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');files=[p]+[OUT/f'{n}.{ext}' for n in ['yearly_accuracy','support_and_coverage'] for ext in ['png','svg']]
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R46 report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

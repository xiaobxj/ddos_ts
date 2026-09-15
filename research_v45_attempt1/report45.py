from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def csv(n):return pd.read_csv(OUT/f'{n}.csv')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'report_manifest.json').exists();v=read('verification.json');assert v['status']=='PASS';met=csv('ensemble_metrics');seed=csv('seed_metrics');support=csv('support_summary');e=csv('ensemble_predictions');gates=csv('gate_decisions');original=csv('original_quarter_gates');pairs=read('primary_comparisons.json')
    methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];mn=dict(zip(methods,['R19','R23','R25']));hist=['rolling5_annual20','weekly_state_validated','quarter_trend2_validated','quarter_volatility2_validated'];hn=dict(zip(hist,['年度基线','四状态季度验证','趋势两组季度验证','波动两组季度验证']));colors=['#333333','#007DB5','#009E8E','#D0657C']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,m in zip(axes,methods):
        for h,color in zip(hist,colors):
            x=met[met.method.eq(m)&met.history.eq(h)&met.period.str.startswith('year_')].sort_values('period');ax.plot(range(2021,2027),x.accuracy,marker='o',color=color,label=hn[h])
        ax.set_xticks(range(2021,2027),['2021','2022','2023','2024','2025','2026*'],rotation=15);ax.set_title(mn[m]);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_ylim(.15,.85);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0].set_ylabel('三种子集成方向准确率');fig.suptitle('减少校准分组，是否改善预测？',fontsize=18);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,.04));fig.text(.5,.014,'每年周数 50、49、48、47、48、30；* 2026 年截至 8 月。原年度模型、交互项和季度验证流程保留。',ha='center',fontsize=10);fig.subplots_adjust(left=.07,right=.98,top=.81,bottom=.27,wspace=.15)
    for ext in ['png','svg']:fig.savefig(OUT/f'yearly_accuracy.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5));families=['four_state','trend','volatility'];labels=['四状态','趋势两组','波动两组'];sup=support.set_index('family').loc[families];bottom=np.zeros(3)
    values=[sup.validation_supported_cells.to_numpy(),(sup.training_supported_cells-sup.validation_supported_cells).to_numpy(),(sup.ready_cells-sup.training_supported_cells).to_numpy()]
    for vals,color,label in zip(values,['#009E8E','#E5A03B','#B6BCC4'],['训练与验证数量均足够','训练足够，验证不足','训练数量不足']):
        pct=vals/sup.ready_cells.to_numpy()*100;axes[0].bar(range(3),pct,bottom=bottom,color=color,label=label)
        for i,z in enumerate(vals):
            if z:axes[0].text(i,bottom[i]+pct[i]/2,f'{int(z)}/{int(sup.ready_cells.iloc[i])}',ha='center',va='center',fontsize=10)
        bottom+=pct
    axes[0].set_xticks(range(3),labels);axes[0].set_ylim(0,100);axes[0].set_ylabel('具备整体历史的季度：分组比例（%）');axes[0].set_title('合并后样本支持是否增加');axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=9)
    ei=e.set_index(['history','method','row_index'])
    for j,(h,color) in enumerate(zip(hist[1:],colors[1:])):
        vals=[]
        for m in methods:
            z=e[e.history.eq(h)&e.method.eq(m)&e.date.ge('2024-01-01')];vals.append(sum(r.probability!=ei.loc[(hist[0],m,r.row_index),'probability'] for r in z.itertuples()))
        axes[1].bar(np.arange(3)+(j-1)*.24,vals,width=.24,color=color,label=hn[h])
        for i,z in enumerate(vals):axes[1].text(i+(j-1)*.24,z+.8,str(z),ha='center',fontsize=10)
    axes[1].set_xticks(range(3),[mn[m] for m in methods]);axes[1].set_ylim(0,125);axes[1].set_ylabel('近期 125 周中相对年度改变概率的周数');axes[1].set_title('通过验证后实际影响的预测覆盖');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('样本支持与实际预测覆盖分开统计',fontsize=17);fig.text(.5,.012,'左图分母随分组数变化；覆盖增加本身不等于预测改善。',ha='center',fontsize=10);fig.subplots_adjust(left=.07,right=.98,top=.8,bottom=.27,wspace=.24)
    for ext in ['png','svg']:fig.savefig(OUT/f'support_and_coverage.{ext}',dpi=170)
    plt.close(fig)
    lines=['# 第四十五轮：趋势两组与波动两组的季度校准比较','',
    '本轮预先固定两个候选：趋势负／非负两组、波动低／高两组。均由原四状态作确定性合并，没有重新估计状态边界。原年度神经网络、滚动 5 年自然 20 遍训练、市场和订单交互均保留；仅重新拟合季度校准截距。',
    '', '沿用原 23 个季度截止和 52／13 个成熟周训练验证划分，训练标签须早于首个验证信号成熟；每组训练至少 10 周、验证至少 5 周。拟合目标为总对数损失加 20δ²/2，|δ|≤0.5，60 次二分求解。三种子平均概率在留出验证上必须 Brier 改善超过 1e-12，且正确数不减少，方可用于后续预测。验证后不再拟合；年末重置、第一季度回退和季度末当天信号使用前一次决定均不变。',
    '', f"本轮独立复核 {v['independent_scalar_solutions']} 个标量最优解。所有训练参数先冻结，368 个门控随后冻结，再产生 6,528 条学习方法种子预测。两条新历史连同对照共新增 8,704 行；原 23 条历史保持不变。分组变大后总损失中的每样本正则强度会相应变化；本轮保持原 lambda=20，结论限定在这套既定训练配方下。",'']
    for period,title in [('extension_2021_2023','2021—2023：147 周'),('recent_2024_2026','2024—2026 年 8 月：125 周'),('pooled_2021_2026','全段：272 周'),('year_2026','2026 年截至 8 月：30 周')]:
        lines += [f'## {title}','','| 方法 | 方案 | 正确／周数 | 准确率 | Brier↓ | 对数损失↓ | AUROC |','|---|---|---:|---:|---:|---:|---:|']
        for m in methods:
            for h in hist:
                r=met[met.period.eq(period)&met.method.eq(m)&met.history.eq(h)].iloc[0];lines.append(f'| {mn[m]} | {hn[h]} | {r.correct_directions}/{r.n} | {r.accuracy:.2%} | {r.brier:.6f} | {r.log_loss:.6f} | {r.auroc:.6f} |')
        lines+=['']
    lines+=['## 训练和验证数量支持','','| 分组 | 具备整体历史的分组单元 | 训练≥10 | 同时验证≥5 | 两者均满足比例 |','|---|---:|---:|---:|---:|']
    for family,label in zip(families,labels):
        r=sup.loc[family];lines.append(f'| {label} | {int(r.ready_cells)} | {int(r.training_supported_cells)} | {int(r.validation_supported_cells)} | {r.validation_support_fraction:.1%} |')
    lines+=['','上述分母是 13 个具备整体历史的季度乘以分组数；数量支持不是模型通过验证。各方法的正式通过数另列：','','| 方法 | 四状态通过／单元 | 趋势两组通过／单元 | 波动两组通过／单元 |','|---|---:|---:|---:|']
    for m in methods:
        vals=[]
        for fam,source in [('state',original),('trend',gates),('volatility',gates)]:
            g=source[source.method.eq(m)&source.family.eq(fam)&source['mode'].eq('ready')];vals.append(f'{int(g.accepted.sum())}/{len(g)}')
        lines.append(f'| {mn[m]} | '+' | '.join(vals)+' |')
    lines+=['','## 近期种子诊断','','| 方法 | 候选 | 种子 | 正确／125 | Brier↓ |','|---|---|---|---:|---:|']
    for r in seed[seed.period.eq('recent_2024_2026')&seed.method.isin(methods)&seed.history.isin(hist[2:])].sort_values(['method','history','seed']).itertuples():lines.append(f'| {mn[r.method]} | {hn[r.history]} | {r.seed} | {r.correct_directions} | {r.brier:.6f} |')
    lines+=['','单种子仅作诊断；门控和主结果仍使用全部原三种子等权平均概率。原四状态的子组表现、空后续单元和所有逐周改善／退步已保存，避免合并后遮蔽子状态差异。','',
    '## 统计与审计','',
    f"48 项预定比较涵盖两个候选、四状态季度与年度两种参照、三个主要方法、方向误差与 Brier、两个固定时期。循环 8 周区块重采样 10,000 次，固定种子，中心化双侧 p 值；48 项统一 Holm 校正。最小原始 p={min(r['p'] for r in pairs):.6f}，最小校正 p={min(r['holm_adjusted_p'] for r in pairs):.6f}，校正后 p<0.05 的比较为 {sum(r['holm_adjusted_p']<.05 for r in pairs)} 项。",
    '', f"独立复核最大系数差 {v['maximum_coefficient_gap']:.3g}，最大预测概率差 {v['maximum_prediction_gap']:.3g}；{v['independent_metric_cells']} 个指标单元、{v['independent_weekly_effects']} 条逐周比较、{v['independent_decision_outcomes']} 个决定后续单元及 {v['independent_substate_outcomes']} 个原四状态子组单元均通过核对。23 个训练截止的非训练标签污染和 23 个验证截止的未来标签污染均不影响对应结果。",
    '', '此前所有历史均已查看，本轮没有新增盲测；多重比较校正不覆盖此前自适应研究。分组变粗可能增加支持，也可能丢失市场状态差异；不得仅按覆盖或单个年份选择候选。没有部署或交易收益测试。5,848 个旧文件、原预测历史及神经模型保持不变。',
    '', '原旧训练预算 R25 近期 73/125、58.4%；自然 20 遍年度 R25 71/125、56.8%；原四状态季度 R23 73/125、58.4% 分别保留。R42 的局部改善与此前失败案例均未覆盖。','',
    '![年度准确率](yearly_accuracy.png)','','![支持与覆盖](support_and_coverage.png)','',
    '[全部指标](ensemble_metrics.csv) · [逐周比较](weekly_policy_effects.csv) · [各决定后续（含空样本）](decision_outcomes.csv) · [原四状态子组表现](substate_outcomes.csv) · [48 项比较](primary_comparisons.json) · [独立复核](verification.json)']
    p=OUT/'第四十五轮实验报告.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');files=[p]+[OUT/f'{n}.{ext}' for n in ['yearly_accuracy','support_and_coverage'] for ext in ['png','svg']]
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R45 report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

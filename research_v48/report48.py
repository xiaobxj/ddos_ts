from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
METHODS=['learned_vol_interaction','learned_order_extension','learned_order_offset'];MN=dict(zip(METHODS,['R19','R23','R25']))
HIST=['rolling5_annual20','annual_head_timeweight2y','annual_head_weighted_intercept','annual_head_weighted_slopes','weekly_state_validated'];HN=dict(zip(HIST,['原年度等权','完整时间加权（R47）','只替换截距','只替换斜率','原四状态季度']))
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'report_manifest.json').exists();v=read('verification.json');assert v['status']=='PASS';met=csv('ensemble_metrics');seed=csv('seed_metrics');factor=csv('factorial_effects');pairs=read('primary_comparisons.json')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
    colors=['#555555','#C05A72','#DA9A27','#009E8E','#007DB5'];fig,axes=plt.subplots(1,3,figsize=(15,5.8),sharey=True)
    for ax,m in zip(axes,METHODS):
        for h,col,style in zip(HIST,colors,[':','--','-.','-','--']):
            g=met[met.history.eq(h)&met.method.eq(m)&met.period.str.startswith('year_')].sort_values('period');ax.plot(range(2021,2027),g.accuracy,color=col,linestyle=style,marker='o',markersize=4,label=HN[h])
        ax.set_title(MN[m]);ax.set_xticks(range(2021,2027),['2021','2022','2023','2024','2025','2026*']);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('三种子集成方向准确率');fig.suptitle('拆开替换截距与斜率，观察方向预测变化',fontsize=18);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.07),ncol=3,fontsize=10);fig.text(.5,.021,'每年周数 50、49、48、47、48、30；* 2026 年截至8月。两种组合仅复用既有系数，没有重新拟合。',ha='center',fontsize=10);fig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.31,wspace=.15)
    for ext in ['png','svg']:fig.savefig(OUT/f'yearly_accuracy.{ext}',dpi=170)
    plt.close(fig)
    periods=['extension_2021_2023','recent_2024_2026','year_2026'];labels=['2021—2023','2024—2026*','2026*'];fig,axes=plt.subplots(2,3,figsize=(15,8.5));x=np.arange(3)
    for row,loss in enumerate(['direction_error','brier']):
        for col,m in enumerate(METHODS):
            ax=axes[row,col];g=factor[factor.method.eq(m)&factor.metric.eq(loss)].set_index('period').loc[periods];scale=g.n.to_numpy() if loss=='direction_error' else np.ones(3)
            ax.bar(x-.17,g.symmetric_intercept*scale,width=.34,color='#DA9A27',label='截距：对称代数分摊');ax.bar(x+.17,g.symmetric_slopes*scale,width=.34,color='#009E8E',label='斜率：对称代数分摊');ax.plot(x,g.total*scale,marker='D',linestyle='none',color='#333333',label='完整加权 − 原年度')
            ax.axhline(0,color='#555555',linewidth=.8);ax.set_xticks(x,labels,fontsize=10);ax.set_title(MN[m]+(' · 方向错误' if row==0 else ' · 概率误差'));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
            if col==0:ax.set_ylabel('增加的错误周数（允许半周分摊）' if row==0 else '平均Brier差')
    fig.suptitle('完整加权相对年度的变化：用四个组合对称分摊',fontsize=17);fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.064),ncol=3,fontsize=10);fig.text(.5,.018,'两项分摊相加等于完整变化；负数表示损失减少。2026是近期窗口的子集，非独立重复。* 截至8月；不作经济因果归因。',ha='center',fontsize=10);fig.subplots_adjust(left=.08,right=.98,top=.86,bottom=.19,wspace=.26,hspace=.4)
    for ext in ['png','svg']:fig.savefig(OUT/f'factorial_attribution.{ext}',dpi=170)
    plt.close(fig)
    lines=['# 第四十八轮：截距与斜率的固定组合对照','',
    '本轮固定两个机械组合：原等权斜率＋第47轮加权截距，以及第47轮加权斜率＋原等权截距。原年度等权、完整时间加权和原四状态季度同时保留作参照。覆盖6个年度截止、3个种子、R18诊断及R19／R23／R25主要方法；共144条组合参数，所有元素直接从已验证的两套参数复制，没有新增拟合、神经训练、特征推理或校准。',
    '', '可交换的前提是特征坐标相同。两套来源均使用原R28年度滚动5年自然20遍神经网络、相同特征标准化、原等权R18监督射线以及相同波动／订单交互残差。所有对应训练设计矩阵、行号和标签逐元素相同。R25所有斜率包含订单gamma一起复制；其混合组合不再声称是原约束目标的最优解。',
    '', '组合参数先全部冻结，再按前一年末截止用于下一年的原周信号。沿用272个成熟周、原三种子概率等权平均和严格p>0.5上涨规则。原native_mse和等权training_frequency精确保留，不新增标签频率候选或选种子。全部28条旧历史继续保留。',
    '', '这项实验拆分的是固定坐标下的模型系数。截距表示该坐标原点的logit，不等于纯粹的市场上涨概率；其变化可能与斜率共同适配。两个组合未经重新优化，不能直接当作训练所得的替代模型。','']
    for period,title in [('extension_2021_2023','早期：2021—2023，147周'),('recent_2024_2026','近期：2024—2026年8月，125周'),('pooled_2021_2026','全段：272周'),('year_2026','2026年截至8月：30周')]:
        lines += [f'## {title}','','| 方法 | 方案 | 正确／周数 | 准确率 | Brier↓ | 对数损失↓ | AUROC |','|---|---|---:|---:|---:|---:|---:|']
        for m in METHODS:
            for h in HIST:
                r=met[met.period.eq(period)&met.method.eq(m)&met.history.eq(h)].iloc[0];lines.append(f'| {MN[m]} | {HN[h]} | {r.correct_directions}/{r.n} | {r.accuracy:.2%} | {r.brier:.6f} | {r.log_loss:.6f} | {r.auroc:.6f} |')
        lines+=['']
    lines+=['## 四组合的代数拆分','',
    '记原年度为U、只替换截距为I、只替换斜率为S、完整加权为W。每个种子每周的logit满足 zW−zU = 截距差 + x·斜率差。经过sigmoid、种子平均和方向阈值之后，概率与损失不再一般可加。每个固定年度和种子内，仅改变截距应保留排序；跨年度或三种子集成的排序不保证不变。',
    '', '对每周方向误差与Brier，先计算四个组合的损失。截距条件影响分别为I−U、W−S；斜率条件影响分别为S−U、W−I；交互项为W−I−S+U。再对两条替换路径平均：截距分摊φI=[(I−U)+(W−S)]/2，斜率分摊φS=[(S−U)+(W−I)]/2，两者之和恰为W−U。这是描述性代数分摊，不是新的显著性检验或经济因果解释。方向错误的分摊允许半周数值。',
    '', '| 时期 | 方法 | 完整加权增加错误周数 | 截距对称分摊 | 斜率对称分摊 | 非加性交互 |','|---|---|---:|---:|---:|---:|']
    for period,label in zip(periods,labels):
        for m in METHODS:
            r=factor[factor.period.eq(period)&factor.method.eq(m)&factor.metric.eq('direction_error')].iloc[0];lines.append(f'| {label} | {MN[m]} | {r.total*r.n:.1f} | {r.symmetric_intercept*r.n:.1f} | {r.symmetric_slopes*r.n:.1f} | {r.interaction*r.n:.1f} |')
    lines+=['','负值表示错误减少。各个年份、两个时期及全段的Brier分摊与条件效应均保存在完整表格；2026属于近期的一部分，不是独立重复。','',
    '## 近期种子诊断','','| 方法 | 组合 | 种子 | 正确／125 | Brier↓ |','|---|---|---|---:|---:|']
    for r in seed[seed.period.eq('recent_2024_2026')&seed.method.isin(METHODS)&seed.history.isin(HIST[2:4])].sort_values(['method','history','seed']).itertuples():lines.append(f'| {MN[r.method]} | {HN[r.history]} | {r.seed} | {r.correct_directions} | {r.brier:.6f} |')
    lines+=['','所有种子均保留，主结果仍是三个原种子的等权概率集成。原四状态的完整指标、空样本单元和逐周错转对／对转错均保存。','',
    '## 统计与复核','',
    f"预定72项探索性比较：两个组合分别对照原年度、完整时间加权、原四状态季度，三个主要方法、两种损失、两个时期。循环8周区块重采样10,000次，固定种子20260910，中心化双侧p，统一Holm校正。最小原始p={min(r['p'] for r in pairs):.6f}，最小校正p={min(r['holm_adjusted_p'] for r in pairs):.6f}；校正后p<0.05共{sum(r['holm_adjusted_p']<.05 for r in pairs)}项。",
    '',f"独立复核PASS：18个来源任务、144个来源头预测重放、144个组合及4,500个系数坐标来源通过核对；{v['independent_learned_seed_predictions']}条新学习方法种子预测、{v['independent_metric_cells']}个指标单元、2,176条逐周比较、2,176个逐周四组合损失单元和72个时期分摊单元均已独立复核。144项年度／种子排序约束通过。最大logit恒等式残差{v['maximum_logit_identity_residual']:.3g}，最大预测概率差{v['maximum_probability_gap']:.3g}。",
    '', '6,143个旧文件、全部28条旧历史保持；新增两条机械组合历史。协议、10个源文件、来源参数和输入在评分前冻结。旧神经网络及缓存采用此前已核验成果的哈希证明，本轮不声称重新执行神经推理。',
    '', '所有历史此前已查看，没有新增盲测；校正不覆盖此前多轮自适应研究。截距／斜率划分依赖固定特征坐标。本轮只解释这些具体模型的历史输出，不能识别市场机制、宣称独立泛化、策略收益或部署效果。',
    '', '旧训练预算R25近期73/125、58.4%，自然20遍年度R25为71/125、56.8%，原四状态季度R23为73/125、58.4%，继续分别保留。',
    '', '![年度准确率](yearly_accuracy.png)','','![四组合对称分摊](factorial_attribution.png)','',
    '[完整指标](ensemble_metrics.csv) · [系数来源](coefficient_sources.csv) · [逐种子logit拆分](seed_logit_decomposition.csv) · [逐周比较](weekly_policy_effects.csv) · [四组合条件效应与分摊](factorial_effects.csv) · [原四状态指标](state_metrics.csv) · [72项比较](primary_comparisons.json) · [独立复核](verification.json)']
    p=OUT/'第四十八轮实验报告.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');files=[p]+[OUT/f'{n}.{ext}' for n in ['yearly_accuracy','factorial_attribution'] for ext in ['png','svg']]
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R48 report andtwofigures generated;visual review pending.',flush=True)
if __name__=='__main__':main()

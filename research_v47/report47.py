from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
METHODS=['learned_vol_interaction','learned_order_extension','learned_order_offset'];MN=dict(zip(METHODS,['R19','R23','R25']))
HIST=['rolling5_annual20','weekly_state_validated','annual_head_timeweight2y'];HN=dict(zip(HIST,['原年度等权','原四状态季度','年度时间加权（2年半衰期）']))
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'report_manifest.json').exists();v=read('verification.json');assert v['status']=='PASS';met=csv('ensemble_metrics');seed=csv('seed_metrics');summary=csv('weight_summary');weights=csv('training_weights');pairs=read('primary_comparisons.json');freq=csv('weighted_frequency_metrics')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
    colors=['#555555','#007DB5','#009E8E'];fig,axes=plt.subplots(1,3,figsize=(15,5.4),sharey=True)
    for ax,m in zip(axes,METHODS):
        for h,col,style in zip(HIST,colors,[':','--','-']):
            q=met[met.history.eq(h)&met.method.eq(m)&met.period.str.startswith('year_')].sort_values('period');ax.plot(range(2021,2027),q.accuracy,marker='o',linestyle=style,color=col,label=HN[h])
        ax.set_title(MN[m]);ax.set_xticks(range(2021,2027),['2021','2022','2023','2024','2025','2026*']);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('三种子集成方向准确率');fig.suptitle('固定5年窗口，只改变分类训练样本的时间权重',fontsize=18);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.07),fontsize=10);fig.text(.5,.022,'每年周数 50、49、48、47、48、30；* 2026 年截至8月。原神经网络、特征坐标和交互项固定。',ha='center',fontsize=10);fig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.25,wspace=.15)
    for ext in ['png','svg']:fig.savefig(OUT/f'yearly_accuracy.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5.5));years=summary.cutoff.str[:4].astype(int).to_numpy()+1;x=np.arange(6)
    axes[0].bar(x-.18,summary.n,width=.36,color='#007DB5',label='保留的日样本数');axes[0].bar(x+.18,summary.kish_effective_n,width=.36,color='#009E8E',label='Kish权重集中度折算数');axes[0].set_ylim(0,1400);axes[0].set_title('成员不变，权重更集中');axes[0].set_ylabel('日样本数／权重折算数');axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=9)
    uniform=[float(weights[weights.cutoff.eq(c)].age_days.le(730.5).mean()) for c in summary.cutoff]
    axes[1].plot(x,uniform,marker='o',color='#007DB5',label='等权');axes[1].plot(x,summary.recent_two_year_mass,marker='o',color='#009E8E',label='时间加权');axes[1].set_ylim(0,1);axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));axes[1].set_title('最近2年日样本的权重占比');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=9)
    axes[2].plot(x,summary.uniform_up_frequency,marker='o',color='#007DB5',label='等权标签频率');axes[2].plot(x,summary.weighted_up_frequency,marker='o',color='#009E8E',label='时间加权标签频率');axes[2].axhline(.5,color='#888888',linestyle=':',linewidth=1);axes[2].set_ylim(.35,.65);axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));axes[2].set_title('训练上涨比例的变化');axes[2].legend(loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=9)
    for ax in axes:ax.set_xticks(x,years);ax.set_xlabel('后续预测年份');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
    fig.suptitle('时间加权改变了什么',fontsize=18);fig.text(.5,.018,'权重在前一年末固定，仅来自已成熟样本。Kish折算反映权重集中度；日标签重叠，不能视为独立样本量。',ha='center',fontsize=10);fig.subplots_adjust(left=.07,right=.98,top=.8,bottom=.29,wspace=.26)
    for ext in ['png','svg']:fig.savefig(OUT/f'training_weights.{ext}',dpi=170)
    plt.close(fig)
    lines=['# 第四十七轮：固定5年窗口的年度时间加权分类头','',
    '本轮固定一个候选：保留5年成熟日样本成员、年度更新和现有年度神经网络，按2年半衰期加权分类头损失。原始权重为2^(−age_days/730.5)，age为决定截止日期减原信号日期的日数；归一化后总和为1。训练成员要求信号日期处于原5年窗口内，全部目标联合成熟时间不晚于截止。没有删样本、重采样或搜索半衰期。',
    '', '原R28年度滚动5年、自然20遍网络，以及神经特征、1／99分位裁剪、均值与标准差、原等权R18监督射线、波动与订单交互残差投影全部固定。新的加权R18仅为诊断，不用于重新生成交互项。本轮仅改变固定特征坐标下分类损失的样本权重，不是整套特征工程的加权重训。',
    '', '目标为归一化加权二元对数损失加0.01/2乘斜率平方和，普通分类头截距不惩罚，初始化为加权训练上涨比例的logit。R19、R23保留原交互结构；R25固定本轮新加权R19的全部系数与截距，仅拟合订单残差的一个gamma。沿用原Newton容差、回溯和迭代预算。共6个年度截止×3个种子×4个方法，54次联合拟合和18次条件标量拟合；全部72次拟合完成并冻结后才计算新周预测。没有神经训练或新特征推理。',
    '', '直接对照为原年度等权分类头，额外参照为原四状态季度校准。候选没有叠加季度校准。三种子等权平均概率，严格大于0.5才判断上涨；原272周目标和日期不变。native_mse与原等权training_frequency完全保留；加权标签频率另作诊断，不替换学习模型或增加候选比较。','']
    for period,title in [('extension_2021_2023','早期：2021—2023，147周'),('recent_2024_2026','近期：2024—2026年8月，125周'),('pooled_2021_2026','全段：272周'),('year_2026','2026年截至8月：30周')]:
        lines += [f'## {title}','','| 方法 | 方案 | 正确／周数 | 准确率 | Brier↓ | 对数损失↓ | AUROC |','|---|---|---:|---:|---:|---:|---:|']
        for m in METHODS:
            for h in HIST:
                r=met[met.period.eq(period)&met.method.eq(m)&met.history.eq(h)].iloc[0];lines.append(f'| {MN[m]} | {HN[h]} | {r.correct_directions}/{r.n} | {r.accuracy:.2%} | {r.brier:.6f} | {r.log_loss:.6f} | {r.auroc:.6f} |')
        lines+=['']
    lines+=['## 训练权重与标签比例','','| 年末截止 | 保留日样本 | Kish折算数 | 最近2年权重占比 | 等权上涨比例 | 加权上涨比例 |','|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.cutoff} | {r.n} | {r.kish_effective_n:.2f} | {r.recent_two_year_mass:.2%} | {r.uniform_up_frequency:.2%} | {r.weighted_up_frequency:.2%} |')
    lines+=['','Kish=1/Σw²，只描述权重集中程度。每日预测目标重叠，样本并非相互独立，不能把折算数当成独立观测量。加权标签频率不是学习模型，其独立诊断结果如下：','','| 时期 | 加权频率正确／周数 | 准确率 | Brier↓ |','|---|---:|---:|---:|']
    for r in freq[freq.period.isin(['extension_2021_2023','recent_2024_2026','pooled_2021_2026','year_2026'])].itertuples():lines.append(f'| {r.period} | {r.correct_directions}/{r.n} | {r.accuracy:.2%} | {r.brier:.6f} |')
    lines+=['','## 近期三个种子','','| 方法 | 种子 | 正确／125 | Brier↓ |','|---|---|---:|---:|']
    for r in seed[seed.period.eq('recent_2024_2026')&seed.method.isin(METHODS)&seed.history.eq(HIST[-1])].sort_values(['method','seed']).itertuples():lines.append(f'| {MN[r.method]} | {r.seed} | {r.correct_directions} | {r.brier:.6f} |')
    lines+=['','单种子及原四状态仅作诊断，不用于选择候选。所有年份、R18诊断、逐周恢复与退步、原四状态完整指标均保存。','',
    '## 统计与复核','',
    f"24项预定探索性比较涵盖年度和原四状态季度两种参照、三个主要方法、方向误差与Brier、早期和近期。循环8周区块重采样10,000次，固定种子20260910，中心化双侧p值，统一Holm校正。最小原始p={min(r['p'] for r in pairs):.6f}，最小校正p={min(r['holm_adjusted_p'] for r in pairs):.6f}，校正后p<0.05为{sum(r['holm_adjusted_p']<.05 for r in pairs)}项。",
    '',f"独立复核PASS：54个联合头由L-BFGS-B独立求解，18个条件标量由Brent独立求解；72个原等权头及其预测仍满足原目标和输出。18个年度／种子接口的未来特征与标签污染不影响训练输入。最大固定特征代数差{v['maximum_frozen_design_gap']:.3g}，最大预测概率差{v['maximum_probability_gap']:.3g}；{v['independent_metric_cells']}个指标单元、1,088条逐周效果和24项比较通过复核。",
    '', '6,056个旧文件和27条旧预测历史保持，新增一条年度候选历史。所有训练权重、协议、11个源文件和输入在拟合前冻结。对原神经与缓存来源采用此前已独立验证的固定成果及哈希核验，本轮没有重新执行神经推理，不作新神经重放声明。',
    '', '全部历史此前已查看，没有新增盲测；多重比较校正不覆盖此前自适应研究。日期加权不等于识别了隐藏市场状态，本轮只检验这一个固定配方。保留监督特征工程同样本拟合的原限制，不作全流程OOF、交易收益或部署结论。',
    '', '原旧训练预算R25近期73/125、58.4%，自然20遍年度R25为71/125、56.8%，原四状态季度R23为73/125、58.4%，分别保留。',
    '', '![逐年准确率](yearly_accuracy.png)','','![训练权重诊断](training_weights.png)','',
    '[完整指标](ensemble_metrics.csv) · [训练权重](training_weights.csv) · [加权频率诊断](weighted_frequency_metrics.csv) · [逐周比较](weekly_policy_effects.csv) · [原四状态指标](state_metrics.csv) · [24项比较](primary_comparisons.json) · [独立复核](verification.json)']
    p=OUT/'第四十七轮实验报告.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');files=[p]+[OUT/f'{n}.{ext}' for n in ['yearly_accuracy','training_weights'] for ext in ['png','svg']]
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R47 report and twofigures generated;visual review pending.',flush=True)
if __name__=='__main__':main()

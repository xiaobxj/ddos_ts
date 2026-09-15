from pathlib import Path
import sys,json,re
import numpy as np
import pandas as pd
ROOT=Path('D:/ddos_v3/research_v46');OUT=ROOT/'results';sys.path.insert(0,str(ROOT))
from common46 import read,sha,save,now,check_frozen,check_phase,old_evidence,PRIMARY,POLICIES,PARENTS,ANNUAL,QUARTER,REPORT_HIST
prep=check_frozen();v=read(OUT/'verification.json');assert v['status']=='PASS'
for phase in ['fitting','validation','scoring','evaluation','report']:check_phase(phase)
csv=lambda n:pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
m=csv('ensemble_metrics');e=csv('ensemble_predictions');w=csv('weekly_policy_effects');g=csv('gate_decisions');d=csv('decision_outcomes');c=csv('policy_coverage');support=csv('support_summary');parent=csv('parent_gate_decisions');pairs=read(OUT/'primary_comparisons.json')
mi=m.set_index(['period','history','method']);ei=e.set_index(['history','method','row_index']);gi=g.set_index(['cutoff','method','family','state']);pi=parent.set_index(['cutoff','method','family','component'])
def numbers(period,history,col):return [mi.loc[(period,history,x),col] for x in PRIMARY]
recent='recent_2024_2026';early='extension_2021_2023';pooled='pooled_2021_2026'
assert numbers(recent,QUARTER,'correct_directions')==[69,73,72]
assert numbers(early,QUARTER,'correct_directions')==[72,69,69]
assert numbers(pooled,QUARTER,'correct_directions')==[141,142,141]
assert numbers('year_2026',QUARTER,'correct_directions')==[12,16,16]
for h in POLICIES.values():
    for period in [early,recent,pooled,'year_2026']:
        assert numbers(period,h,'correct_directions')==numbers(period,QUARTER,'correct_directions')
    for method in PRIMARY:
        a=e[e.history.eq(h)&e.method.eq(method)].sort_values('row_index');b=e[e.history.eq(QUARTER)&e.method.eq(method)].sort_values('row_index')
        assert a.row_index.tolist()==b.row_index.tolist() and a.direction_up.tolist()==b.direction_up.tolist()
    for period in [early,recent,pooled]:
        assert all(a>b for a,b in zip(numbers(period,h,'brier'),numbers(period,QUARTER,'brier')))
assert numbers(recent,PARENTS['trend'],'correct_directions')==numbers(recent,PARENTS['volatility'],'correct_directions')==[68,72,71]
assert numbers(early,PARENTS['trend'],'correct_directions')==[71,69,69]
assert numbers(early,PARENTS['volatility'],'correct_directions')==[72,70,69]
assert numbers('year_2026',PARENTS['volatility'],'correct_directions')==[11,16,16]
brier_expected={QUARTER:[.246156,.248903,.248933],POLICIES['trend']:[.246841,.249189,.249220],POLICIES['volatility']:[.246502,.249007,.249036],ANNUAL:[.246843,.249790,.249824]}
for h,vals in brier_expected.items():np.testing.assert_allclose(numbers(recent,h,'brier'),vals,rtol=0,atol=.0000005)
np.testing.assert_allclose([mi.loc[('year_2026',h,PRIMARY[0]),'brier'] for h in [ANNUAL,QUARTER,POLICIES['trend'],POLICIES['volatility']]],[.257501,.258700,.260401,.259241],rtol=0,atol=.0000005)
for family in POLICIES:
    z=support[support.scheme.eq('child_'+family)].iloc[0];assert [z.ready_cells,z.training_supported_cells,z.validation_supported_cells]==[52,32,11]
    for period,expected in [(early,[12,12,12]),(recent,[6,3,3] if family=='trend' else [5,3,3]),('year_2026',[2,0,0])]:
        q=c[c.period.eq(period)&c.history.eq(POLICIES[family])].set_index('method').loc[PRIMARY]
        assert q.changed_probability_weeks.tolist()==expected and q.changed_probability_weeks.equals(q.validation_accepted_weeks)
assert len(g)==736 and g.accepted.sum()==32
assert g.groupby('family').accepted.sum().to_dict()=={'trend':14,'volatility':18}
for h in POLICIES.values():
    for method in PRIMARY:
        z=w[w.history.eq(h)&w.method.eq(method)&w.case_vs_annual.isin(['recovery','regression'])]
        assert z.date.tolist()==['2023-12-01','2024-06-28'] and z.case_vs_annual.eq('recovery').all()
# Parent transition cases and qualified counts.
for method in PRIMARY:
    z=gi.loc[('2024-03-31',method,'trend','negative_low')];assert z.accepted and [z.training_n,z.validation_n]==[25,7]
    assert not pi.loc[('2024-03-31',method,'trend','negative')].accepted
z=gi.loc[('2024-06-30',PRIMARY[1],'volatility','negative_low')];assert not z.accepted and z.validation_n==1 and z.reason=='insufficient_validation'
z=gi.loc[('2024-06-30',PRIMARY[1],'volatility','nonnegative_low')];assert not z.accepted and z.validation_n==11 and z.brier_difference>0
z=gi.loc[('2026-06-30',PRIMARY[0],'volatility','negative_high')];assert not z.accepted and [z.training_n,z.validation_n]==[4,4] and z.reason=='insufficient_state_train'
z=gi.loc[('2022-06-30',PRIMARY[1],'volatility','nonnegative_high')];assert not z.accepted and [z.training_n,z.validation_n]==[8,1]
def effect(family,method,date):return w[w.history.eq(POLICIES[family])&w.method.eq(method)&w.date.eq(date)].iloc[0]
assert effect('volatility',PRIMARY[0],'2026-08-14').case_vs_parent2=='recovery'
assert effect('volatility',PRIMARY[1],'2022-07-08').case_vs_parent2=='regression'
assert len(pairs)==72 and min(p['p'] for p in pairs)==.1060893910608939 and all(p['holm_adjusted_p']==1. for p in pairs)
assert v['independent_scalar_solutions']==588 and v['independent_gate_cells']==736 and v['independent_validation_seed_rows']==6792 and v['independent_metric_cells']==5967 and v['exact_ensemble_fallback_predictions']==2046
for n in ['correction_heads.json','correction_parameters.csv']:assert sha(OUT/n)==sha(ROOT.parent/'research_v45/results'/n)
assert old_evidence()==prep['old_evidence'] and len(prep['old_evidence'])==5986
text='''第46轮已完成。共享第45轮合并分组参数、再按原四状态分别验证，追回了上一轮近期少对的一周，但未超过原四状态季度方案。本轮继续保留原方案作为研究参照。

本轮没有新增神经网络或标量校准训练：复用588个已拟合参数及全部1,104条头记录，文件与第45轮逐字节相同。沿用原季度52／13成熟周划分、固定5年窗口、自然20遍年度模型和市场／订单交互。两个候选仅改变共享参数的启用方式：子状态训练至少10周、验证至少5周，三种子平均概率的Brier改善超过1e-12且正确数不减少才启用。合并组原来是否通过，不决定子状态是否通过；验证后不再拟合。

近期为2024年至2026年8月，共125周，准确率如下：

| 方案 | R19 | R23 | R25 |
|---|---:|---:|---:|
| 原四状态季度验证 | 69/125，55.2% | 73/125，58.4% | 72/125，57.6% |
| R45趋势两组验证 | 68/125，54.4% | 72/125，57.6% | 71/125，56.8% |
| 本轮趋势共享／四状态验证 | 69/125，55.2% | 73/125，58.4% | 72/125，57.6% |
| R45波动两组验证 | 68/125，54.4% | 72/125，57.6% | 71/125，56.8% |
| 本轮波动共享／四状态验证 | 69/125，55.2% | 73/125，58.4% | 72/125，57.6% |

两个新候选、三个主要方法在全部272周上的方向预测，与原四状态季度方案逐周完全相同，不只是汇总正确数相同。早期147周均为72／69／69周正确，全段均为141／142／141周正确。相对年度基线的方向改善，仍只来自2023-12-01和2024-06-28这两个已知日期；不能把跨方法、跨候选重复的同一周当成独立证据。

概率质量没有超过原方案。近期Brier（越低越好）为：

| 方案 | R19 | R23 | R25 |
|---|---:|---:|---:|
| 原四状态季度 | 0.246156 | 0.248903 | 0.248933 |
| 趋势共享／四状态验证 | 0.246841 | 0.249189 | 0.249220 |
| 波动共享／四状态验证 | 0.246502 | 0.249007 | 0.249036 |

两个候选在三个主要方法的早期、近期和全段Brier都略高于原四状态季度。新方案相对第45轮的近期改善值得保留，但目前没有依据替换原研究参照。

子状态验证确实改变了哪些修正会被使用。以R23为例，2024-03-31的趋势负组在R45整组验证未通过；本轮负趋势／低波动子状态有25个训练周、7个验证周，独立验证通过，恢复了6月28日的正确预测。波动共享则保留同一季度负趋势／低波动的修正，同时阻止只有1个验证周的非负趋势／低波动子状态继承整组决定。

2024-06-30，R23波动低组虽在R45整体通过，本轮负趋势／低波动只有1个验证周，非负趋势／低波动虽有11周但Brier变差，因此两者均不启用。这样避免了7月12日、8月9日的两次错误修正，也放弃了8月2日的一次正确修正。净改善存在，但不是保留了所有好修正。

2026年截至8月共30周：R19波动共享恢复为12/30、40.0%，好于R45波动两组的11/30、36.7%，与原四状态季度及年度相同。8月14日对应负趋势／高波动，6月截止时该子状态只有4个训练周和4个验证周，因此回退年度，避免了R45新增的错误。R23、R25仍各16/30、53.3%。但R19的2026年Brier，趋势共享为0.260401、波动共享为0.259241，仍高于原四状态季度0.258700及年度0.257501。

早期也有代价。R23波动共享没有保留R45在2022-07-08的一次正确修正：该信号属于非负趋势／高波动，决定时只有8个训练周和1个验证周，未达下限，故回退年度。其早期正确数由R45的70降回69，与原四状态季度相同。R19趋势共享则避免了R45在2023-08-18的错误，早期由71恢复72。旧结果及这些局部改善全部保留，不按近期表现抹去早期收益。

样本不足仍是约束。在13个具备整体历史的季度中，两种新方案均只有11/52个原四状态单元同时满足训练与验证数量下限，与原四状态季度完全相同。R45趋势两组为14/26、波动两组13/26，但分母不同。本轮共32/736个子状态门控通过，其中趋势14、波动18，含R18诊断方法；门控数量不等于独立事件数量。实际改变的近期预测周数，趋势共享R19／R23／R25为6／3／3，波动共享为5／3／3；R45波动两组为42／34／34。共享拟合参数并没有补足子状态的验证证据。

72项预定探索性比较的最小原始p值为0.106089，Holm校正后全部为1。全部历史此前已经查看，没有新增盲测，校正不覆盖此前多轮自适应研究。本轮结果不能作为独立泛化验证。

下一步建议暂时结束这条合并分组／共享截距校准分支，转向固定滚动窗口内的时间加权分类头。可预先固定一个候选：保留5年成员、年度更新、已有年度神经特征和R19／R23／R25交互，用2年半衰期对成熟训练样本按时间加权；以均匀权重的原年度分类头为直接对照，原四状态季度作为额外参照。2年是待检验的固定设计，不是本轮估出的最优值。下一轮需在新评分前冻结权重归一化、正则尺度、拟合与比较规则；不搜索多种半衰期，不同时增加更新频率或叠加新校准。本轮尚未拟合该候选，也不能预设它会提高准确率。

独立复核通过：588个原标量最优解、736个门控、6,792条验证种子概率、6,528条学习方法种子预测、5,967个指标单元、2,176条逐周比较和736个含空样本的决定后续结果均已核对。23个训练截止的非训练输入污染、23个验证截止的未来输入污染均不改变对应结果；2,046条零修正集成预测与年度精确相同。原25条历史、5,986个旧文件保留。两张图表已目视检查。

原旧训练预算R25近期73/125、58.4%，自然20遍年度R25为71/125、56.8%，原四状态季度R23为73/125、58.4%，继续分别保留。

[完整实验报告与图](第四十六轮实验报告.md) · [逐周比较](weekly_policy_effects.csv) · [门控与后续结果](decision_outcomes.csv) · [原四状态指标](state_metrics.csv) · [独立复核](verification.json)
'''
human=OUT/'结果解读与下一步.md';assert not human.exists();human.write_text(text,encoding='utf-8')
# Check table shape and local report links, not just the existence of a report file.
reports=[human,OUT/'第四十六轮实验报告.md']
for path in reports:
    body=path.read_text(encoding='utf-8');assert '\ufffd' not in body and '????' not in body
    width=None
    for line in body.splitlines():
        if line.startswith('|'):
            n=len(line.split('|'));width=width or n;assert n==width,(path,line)
        else:width=None
    for target in re.findall(r'\]\(([^)]+)\)',body):assert (path.parent/target).exists(),target
assert '本轮尚未拟合该候选' in text and '不是本轮估出的最优值' in text
assert not (ROOT.parent/'research_v47').exists()
images=[OUT/f'{n}.{ext}' for n in ['yearly_accuracy','support_and_coverage'] for ext in ['png','svg']]
assert not (OUT/'visual_review.json').exists() and not (OUT/'interpretation_checks.json').exists()
save(OUT/'visual_review.json',dict(status='PASS',completed_utc=now(),review='Both PNGs inspected via view_image. Chinese text legible; six annual panels, both legends and partial2026 footnote visible; support denominators11/52,14/26,11/52,13/26,11/52 and coverage values including zeros correct; no clipping or overlap. Original and new directions overlap intentionally and are disclosed.',artifacts={str(p.relative_to(ROOT)):sha(p) for p in images}))
save(OUT/'interpretation_checks.json',dict(status='PASS',completed_utc=now(),checks=['primary_directions_identical_to_quarter4_all272weeks','early_recent_pooled_2026_counts','all_six_candidates_Brier_worse_than_quarter4_early_recent_pooled','recent_Brier_and_coverage_values','old_local_gains_and_losses_retained','parent_rejected_child_accepted_March2024','child_insufficient_support_June2024_and_June2026','original_training_validation_support_identical','known_two_recovery_dates_only_vs_annual','72_comparisons_all_Holm1','588_parameters_reused_byte_identity','25oldhistories_5986oldfiles_unchanged','future_timeweight_candidate_not_started','markdown_tables_and_links','old_budget_R25_distinct_from_natural20_and_R23'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in reports+[OUT/'visual_review.json',OUT/'verification.json']}))
print('R46 human results, numerical claims, provenance, links and visual review PASS.')

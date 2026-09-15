from pathlib import Path
import sys,re
import numpy as np
import pandas as pd

ROOT=Path('D:/ddos_v3/research_v48');OUT=ROOT/'results';sys.path.insert(0,str(ROOT))
from common48 import read,sha,save,now,check_frozen,check_phase,old_evidence,PRIMARY,ANNUAL,QUARTER,WEIGHTED,INTERCEPT,SLOPES
prep=check_frozen();v=read(OUT/'verification.json');assert v['status']=='PASS'
for phase in ['assembly','scoring','evaluation','report']:check_phase(phase)
csv=lambda n:pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
m=csv('ensemble_metrics').set_index(['period','history','method'])
c=csv('reference_comparisons').set_index(['period','history','method','reference_history'])
f=csv('factorial_effects').set_index(['period','method','metric'])
pairs=read(OUT/'primary_comparisons.json')
recent='recent_2024_2026';early='extension_2021_2023';pooled='pooled_2021_2026';y26='year_2026'
U,W,I,S,Q=ANNUAL,WEIGHTED,INTERCEPT,SLOPES,QUARTER
def vals(period,h,col):return [m.loc[(period,h,x),col] for x in PRIMARY]
expected={
    recent:{U:[68,72,71],W:[66,64,65],I:[72,68,68],S:[71,68,69],Q:[69,73,72]},
    early:{U:[71,68,68],W:[70,71,70],I:[72,71,70],S:[69,70,70],Q:[72,69,69]},
    pooled:{U:[139,140,139],W:[136,135,135],I:[144,139,138],S:[140,138,139],Q:[141,142,141]},
    y26:{U:[12,16,16],W:[13,12,12],I:[14,12,12],S:[14,13,14],Q:[12,16,16]},
}
for period,schemes in expected.items():
    for h,counts in schemes.items():assert vals(period,h,'correct_directions')==counts
for h,expected_brier in {U:[.246843,.249790,.249824],W:[.252067,.255758,.255844],I:[.249373,.252181,.252272],S:[.249262,.253036,.253074],Q:[.246156,.248903,.248933]}.items():
    np.testing.assert_allclose(vals(recent,h,'brier'),expected_brier,rtol=0,atol=.0000005)
for h in [I,S]:
    for ref in [U,Q]:
        assert all(a>b for a,b in zip(vals(recent,h,'brier'),vals(recent,ref,'brier')))
        assert all(a<b for a,b in zip(vals(early,h,'brier'),vals(early,ref,'brier')))
    assert all(a<b for a,b in zip(vals(recent,h,'brier'),vals(recent,W,'brier')))
    assert all(a>b for a,b in zip(vals(early,h,'brier'),vals(early,W,'brier')))
for period in [pooled,y26]:
    for ref in [U,Q,W]:assert all(a<b for a,b in zip(vals(period,S,'brier'),vals(period,ref,'brier')))
assert all(a>b for a,b in zip(vals(pooled,I,'brier'),vals(pooled,Q,'brier')))
for h,rr in {I:[[7,3],[5,9],[5,8]],S:[[8,5],[4,8],[5,7]]}.items():
    for method,counts in zip(PRIMARY,rr):
        row=c.loc[(recent,h,method,U)];assert [row.recoveries,row.regressions]==counts
        for ref in [U,W,Q]:assert c.loc[(recent,h,method,ref),'changed_probability_weeks']==125
for h,counts in {U:[12,8,8],W:[17,18,18],I:[16,16,16],S:[14,11,10]}.items():
    np.testing.assert_allclose(np.array(vals(y26,h,'predicted_up_fraction'))*30,counts,rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.array(vals(y26,h,'observed_up_fraction'))*30,[14,14,14],rtol=0,atol=1e-12)
columns=['total','intercept_at_uniform_slopes','slopes_at_uniform_intercept','interaction','symmetric_intercept','symmetric_slopes']
for period,rows in {
    recent:[[2,-4,-3,9,.5,1.5],[8,4,4,0,4,4],[6,3,2,1,3.5,2.5]],
    y26:[[-1,-2,-2,3,-.5,-.5],[4,4,3,-3,2.5,1.5],[4,4,2,-2,3,1]],
    early:[[1,-1,2,0,-1,2],[-3,-3,-2,2,-2,-1],[-2,-2,-2,2,-1,-1]],
}.items():
    for method,expected_values in zip(PRIMARY,rows):
        row=f.loc[(period,method,'direction_error')];np.testing.assert_allclose(row[columns].to_numpy(float)*row.n,expected_values,rtol=0,atol=1e-12)
for method in PRIMARY:
    row=f.loc[(recent,method,'brier')];assert row.symmetric_intercept>0 and row.symmetric_slopes>0
    row=f.loc[(y26,method,'brier')];assert row.symmetric_intercept>0 and row.symmetric_slopes<0
best=min(pairs,key=lambda x:x['p'])
assert len(pairs)==72 and best['p']==.018598140185981403 and all(x['holm_adjusted_p']==1 for x in pairs)
assert [best[k] for k in ['window','history','candidate','reference_history','metric']]==[early,S,PRIMARY[1],U,'brier']
for k,n in {'old_files_preserved':6143,'new_head_fits':0,'new_neural_fits':0,'new_feature_inference':0,'annual_source_jobs':18,'endpoint_head_replays':144,'independent_comparisons':72,'assembled_heads_verified':144,'coefficient_coordinates_verified':4500,'independent_learned_seed_predictions':6528,'annual_seed_rank_checks':144,'independent_metric_cells':6390,'old_histories_preserved':28,'independent_weekly_effects':2176,'independent_factorial_weekly_cells':2176,'independent_factorial_period_cells':72}.items():assert v[k]==n,(k,v[k])
assert v['exact_native_frequency_controls'] and v['maximum_probability_gap']<1e-14 and v['maximum_logit_identity_residual']<1e-14
assert old_evidence()==prep['old_evidence'] and len(prep['old_evidence'])==6143
assert not (ROOT.parent/'research_v49').exists()

body='''第48轮完成，独立复核通过。结论是：第47轮近期退步不能只归咎于截距；截距和斜率的影响随方法与时期变化。R19的局部方向改善，以及只替换斜率在全段和2026年的概率改善，均保留记录；目前不足以整体替换原年度与原四状态季度研究参照。

本轮完成既定的两个组合对照：I为原等权斜率＋第47轮加权截距，S为第47轮加权斜率＋原等权截距。原年度等权记U，完整时间加权记W。覆盖6个年度截止、3个种子、R18诊断与R19／R23／R25主要方法，共144组参数，全部直接复制已验证的系数，没有新增拟合、神经训练或特征推理。全部组合冻结后才评分。

两套参数共享原年度滚动5年、自然20遍训练的神经特征、标准化、原等权监督射线及市场／订单交互坐标。R25订单残差系数随斜率一起替换。两种组合未经重新优化，是固定模型的拆分对照。截距依赖特征坐标原点，不能简单视为市场上涨概率。

近期为2024年至2026年8月，共125周。主结果仍使用原三个种子的平均概率，严格大于0.5判上涨：

| 方案 | R19 | R23 | R25 |
|---|---:|---:|---:|
| 原年度等权 | 68/125，54.4% | 72/125，57.6% | 71/125，56.8% |
| 原四状态季度 | 69/125，55.2% | 73/125，58.4% | 72/125，57.6% |
| 完整时间加权（R47） | 66/125，52.8% | 64/125，51.2% | 65/125，52.0% |
| I：只替换截距 | 72/125，57.6% | 68/125，54.4% | 68/125，54.4% |
| S：只替换斜率 | 71/125，56.8% | 68/125，54.4% | 69/125，55.2% |

R19只替换截距比原年度多对4周、比原季度多对3周；只替换斜率分别多对3周、2周。这些改善保留。R23两个单独替换都比原年度少对4周；R25分别少对3周、2周。相对年度，I的错转对／对转错分别为R19的7／3、R23的5／9、R25的5／8；S分别为8／5、4／8、5／7，逐周明细已保存。

近期概率质量没有同步改善。Brier为预测上涨概率与实际0／1标签的均方误差，越低越好：

| 方案 | R19 | R23 | R25 |
|---|---:|---:|---:|
| 原年度等权 | 0.246843 | 0.249790 | 0.249824 |
| 原四状态季度 | 0.246156 | 0.248903 | 0.248933 |
| 完整时间加权（R47） | 0.252067 | 0.255758 | 0.255844 |
| I：只替换截距 | 0.249373 | 0.252181 | 0.252272 |
| S：只替换斜率 | 0.249262 | 0.253036 | 0.253074 |

两个组合的三个方法近期Brier都优于完整加权，却都差于原年度和原四状态季度。R19方向命中改善，不能直接推出概率预测更可靠。

拆分揭示了不同的变化模式。近期R19单独替换截距减少4个错误周，单独替换斜率减少3个错误周，但共同替换反而增加2个错误周。R23两项单独替换各增加4个错误周，共同替换增加8周；R25分别增加3周、2周，共同替换增加6周。因此，简单地只保留或只删除截距变化，并不能同时解决三个方法的问题。

每个种子的logit变化严格满足“截距变化＋特征乘斜率变化”；经过sigmoid、三种子平均与0.5方向阈值后，损失通常不再可加。按四组合损失计算，交互项为W−I−S+U，近期R19／R23／R25分别相当于增加9／0／1个错误周。这里是模型输出的非加性，不是市场变量间的经济因果关系。

将两条替换路径平均，截距分摊为[(I−U)+(W−S)]/2，斜率分摊为[(S−U)+(W−I)]/2；两者之和等于完整变化。这种对称分摊下，近期R19的2个新增错误周分为0.5和1.5，R23的8周分为4和4，R25的6周分为3.5和2.5。分摊允许半周，不是实际出现了半个预测，也不能把分摊值误读为单独替换的效应。三个方法近期Brier的两项对称分摊均为正，即概率误差增加。

较早时期和2026年的局部收益也保留。2021—2023年147周，原年度正确数R19／R23／R25为71／68／68，I为72／71／70，S为69／70／70；两个组合的三个方法早期Brier均好于原年度与原季度，但仍差于完整加权。全段272周，原年度正确数139／140／139，原季度141／142／141，I为144／139／138，S为140／138／139。S全段Brier优于原年度、原季度和完整加权的全部三个对应方法，方向正确数却都低于原季度；I的R19全段方向正确数改善，但三个方法全段Brier均差于原季度。

2026年截至8月仅30周，原年度／季度正确数为12／16／16，完整加权为13／12／12，I为14／12／12，S为14／13／14。两种组合的R19都改善；R23、R25仍低于原年度方向准确率。S的三个方法在2026年的Brier均优于原年度、原季度和完整加权；对称分摊也显示该年斜率改善Brier、截距恶化Brier。这个结论不能推广到整个近期125周。

2026年R23／R25原年度判断上涨均为8/30，完整加权均为18/30；I均为16/30，S分别为11/30、10/30，实际上涨为14/30。截距与斜率都会影响阈值两侧的分布，而上涨次数接近实际次数本身不保证判对了相同的周。2026是近期窗口的子集，不是额外独立证据。

72项预先固定的探索性比较涵盖两个组合、三个参照、三个主要方法、方向误差与Brier、早期与近期两个窗口。采用8周循环区块重采样10,000次，原固定种子20260910，中心化双侧p值及统一Holm校正。最小原始p为0.018598，来自早期S的R23相对原年度Brier改善；全部校正p均为1.0，没有校正后显著的比较。这不证明效果为零。全部历史此前已查看，校正也不覆盖此前多轮自适应研究，本轮没有新增盲测。

下一步建议冻结候选和评价规则，建立前瞻性的逐周预测记录。沿用滚动5年和既定市场／订单交互，将原年度、原四状态季度、完整加权和两个固定组合的对应方法全部保留，统一保存预测时间、训练截止、当时可用数据和参数哈希；在标签成熟前写入预测，成熟后按预定频率评估方向误差与Brier。先固定最小观察长度、汇总时间和比较清单，再积累数据，不依据每次新结果临时挑方法、种子或阈值。这样可以检验本轮保留的局部改善能否延续。本轮尚未建立或启动下一轮前瞻预测，也没有读取新增年份数据。

独立复核通过：18个来源任务、144个来源头预测重放、144组组合及4,500个系数坐标来源；6,528条新学习方法种子预测、6,390个指标单元、2,176条逐周比较、2,176个四组合逐周损失单元、72个时期分摊单元及72项统计比较均核对。144项年度／种子排序约束通过，最大logit恒等式残差约8.05×10⁻¹⁶，最大概率差约2.22×10⁻¹⁶。6,143个旧文件、全部28条旧历史与原native_mse／等权training_frequency对照保持；两张图表目视检查通过。研究协议、10个源码文件和输入哈希均核验。

原旧训练预算R25近期73/125、58.4%，自然20遍年度R25为71/125、56.8%，原四状态季度R23为73/125、58.4%，继续分别保存，不能混为同一个结果。

[完整报告与图](第四十八轮实验报告.md) · [完整指标](ensemble_metrics.csv) · [系数来源](coefficient_sources.csv) · [逐种子logit拆分](seed_logit_decomposition.csv) · [逐周比较](weekly_policy_effects.csv) · [四组合分摊](factorial_effects.csv) · [72项比较](primary_comparisons.json) · [独立复核](verification.json)
'''
human=OUT/'结果解读与下一步.md';assert not human.exists();human.write_text(body,encoding='utf-8')
reports=[human,OUT/'第四十八轮实验报告.md']
for path in reports:
    text=path.read_text(encoding='utf-8');assert '\ufffd' not in text and '????' not in text
    width=None
    for line in text.splitlines():
        if line.startswith('|'):
            n=len(line.split('|'));width=width or n;assert n==width,(path,line)
        else:width=None
    for target in re.findall(r'\]\(([^)]+)\)',text):assert (path.parent/target).exists(),target
images=[OUT/f'{name}.{ext}' for name in ['yearly_accuracy','factorial_attribution'] for ext in ['png','svg']]
assert not (OUT/'visual_review.json').exists() and not (OUT/'interpretation_checks.json').exists()
save(OUT/'visual_review.json',dict(status='PASS',completed_utc=now(),review='Both final PNGs inspected using view_image. Annual accuracy: all three panels, five-series legend, yearly sample counts and partial-2026 footnote readable. Factorial figure: all six panels, zero lines, negative shares, total-change diamonds, symmetric-share legend and non-causal footnote readable. No clipped labels or overlapping text. Figure quantities agree with metric and factorial tables.',artifacts={str(p.relative_to(ROOT)):sha(p) for p in images}))
save(OUT/'interpretation_checks.json',dict(status='PASS',completed_utc=now(),checks=['early_recent_pooled_2026_correctcounts_all5schemes','recent_Brier_all5schemes','recent_R19_direction_gain_preserved','early_and_pooled_probability_gains_preserved','2026_slope_probability_gains_preserved','recent_recovery_regression_counts','2026_up_predictions_and_observed_counts','conditional_effects_vs_symmetric_decomposition','early_recent_2026_factorial_direction_values','recent_and2026_Brier_share_signs','72_raw_and_Holm_pvalues','zero_new_fits_and_inference','independent_verification_counts','28_old_histories_and6143_old_files_unchanged','prospective_work_recommended_not_started','old_budget_distinctions','markdown_encoding_links_and_table_shapes'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in reports+[OUT/'visual_review.json',OUT/'verification.json']}))
print('R48 human interpretation, numerical claims, links and visual review PASS.')

from pathlib import Path
import sys,json,re,math
import numpy as np
import pandas as pd
ROOT=Path('D:/ddos_v3/research_v47');OUT=ROOT/'results';sys.path.insert(0,str(ROOT))
from common47 import read,sha,save,now,check_frozen,check_phase,old_evidence,PRIMARY,NEW,ANNUAL,QUARTER
prep=check_frozen();v=read(OUT/'verification.json');assert v['status']=='PASS'
for phase in ['fitting','scoring','evaluation','report']:check_phase(phase)
csv=lambda n:pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
m=csv('ensemble_metrics');s=csv('seed_metrics');e=csv('ensemble_predictions');w=csv('weekly_policy_effects');c=csv('reference_comparisons');weights=csv('training_weights');summary=csv('weight_summary');freq=csv('weighted_frequency_metrics');pairs=read(OUT/'primary_comparisons.json');heads=read(OUT/'heads.json');original=read(OUT/'source_heads.json')
mi=m.set_index(['period','history','method']);ci=c.set_index(['period','history','method','reference_history']);fi=freq.set_index('period');H=NEW[0]
def vals(period,h,col):return [mi.loc[(period,h,x),col] for x in PRIMARY]
recent='recent_2024_2026';early='extension_2021_2023';pooled='pooled_2021_2026'
expected={early:{H:[70,71,70],ANNUAL:[71,68,68],QUARTER:[72,69,69]},recent:{H:[66,64,65],ANNUAL:[68,72,71],QUARTER:[69,73,72]},pooled:{H:[136,135,135],ANNUAL:[139,140,139],QUARTER:[141,142,141]},'year_2024':{H:[31,29,30],ANNUAL:[34,33,32]},'year_2025':{H:[22,23,23],ANNUAL:[22,23,23]},'year_2026':{H:[13,12,12],ANNUAL:[12,16,16],QUARTER:[12,16,16]}}
for period,schemes in expected.items():
    for h,x in schemes.items():assert vals(period,h,'correct_directions')==x
np.testing.assert_allclose(vals(recent,H,'brier'),[.252067,.255758,.255844],rtol=0,atol=.0000005)
np.testing.assert_allclose(vals(recent,ANNUAL,'brier'),[.246843,.249790,.249824],rtol=0,atol=.0000005)
np.testing.assert_allclose(vals(recent,QUARTER,'brier'),[.246156,.248903,.248933],rtol=0,atol=.0000005)
for period in [early,pooled]:assert all(a<b for a,b in zip(vals(period,H,'brier'),vals(period,ANNUAL,'brier')))
assert all(a>b for a,b in zip(vals(recent,H,'brier'),vals(recent,ANNUAL,'brier')))
assert all(a>b for a,b in zip(vals(pooled,H,'brier'),vals(pooled,QUARTER,'brier')))
for method,recovered,regressed in zip(PRIMARY,[7,7,8],[9,15,14]):
    r=ci.loc[(recent,H,method,ANNUAL)];assert [r.recoveries,r.regressions,r.changed_probability_weeks]==[recovered,regressed,125]
for method in PRIMARY[1:]:
    assert mi.loc[('year_2026',ANNUAL,method),'predicted_up_fraction']==8/30 and mi.loc[('year_2026',H,method),'predicted_up_fraction']==18/30 and mi.loc[('year_2026',H,method),'observed_up_fraction']==14/30
    q=ci.loc[('year_2026',H,method,ANNUAL)];assert [q.recoveries,q.regressions]==[4,8]
z=summary.set_index('cutoff').loc['2025-12-31'];np.testing.assert_allclose([z.uniform_up_frequency,z.weighted_up_frequency],[.4792703150912106,.5100268345623041],rtol=0,atol=1e-15)
assert summary.n.tolist()==[1212,1211,1209,1208,1206,1206] and summary.kish_effective_n.between(975,981).all() and summary.recent_two_year_mass.between(.602,.604).all()
mass=[]
for cutoff in summary.cutoff:mass.append(weights[weights.cutoff.eq(cutoff)].age_days.le(730.5).mean())
assert all(.395<x<.405 for x in mass)
assert fi.loc[recent,'correct_directions']==57 and mi.loc[(recent,ANNUAL,'training_frequency'),'correct_directions']==56
assert fi.loc['year_2026','correct_directions']==14 and mi.loc[('year_2026',ANNUAL,'training_frequency'),'correct_directions']==16
assert len(heads)==72 and v['independent_joint_solutions']==54 and v['independent_scalar_solutions']==18 and v['uniform_head_replays']==72 and v['future_poison_interfaces']==18
assert v['independent_metric_cells']==5733 and v['independent_weekly_effects']==1088 and v['independent_learned_seed_predictions']==3264
assert len(pairs)==24 and min(p['p'] for p in pairs)==.008599140085991401 and min(p['holm_adjusted_p'] for p in pairs)==.20637936206379365 and not any(p['holm_adjusted_p']<.05 for p in pairs)
assert old_evidence()==prep['old_evidence'] and len(prep['old_evidence'])==6056
text='''第47轮已完成：固定5年窗口、2年半衰期的年度时间加权分类头，未改善近期整体表现。继续保留原年度方案、原四状态季度方案及已有交互成果，不替换研究参照。

本轮完成72次新的分类头拟合：54次联合逻辑回归和18次R25条件标量拟合，覆盖6个年度截止、3个种子。没有重训神经网络或重新提取特征。神经特征、标准化、原监督射线和市场／订单交互项数值全部固定，只改变分类损失中的样本权重。R25固定本轮加权R19的系数与截距后拟合订单残差系数。全部拟合冻结后才计算新周预测。

近期为2024年至2026年8月，共125周，三个原种子平均概率的方向准确率如下：

| 方案 | R19 | R23 | R25 |
|---|---:|---:|---:|
| 原年度等权 | 68/125，54.4% | 72/125，57.6% | 71/125，56.8% |
| 原四状态季度 | 69/125，55.2% | 73/125，58.4% | 72/125，57.6% |
| 本轮时间加权 | 66/125，52.8% | 64/125，51.2% | 65/125，52.0% |

相对年度，三个方法分别少对2、8、6周；相对原四状态季度，分别少对3、9、7周。不是只改变了不影响方向的概率：相对年度，R19为7周错转对、9周对转错，R23为7周错转对、15周对转错，R25为8周错转对、14周对转错。近期全部125周的学习方法概率都发生了变化。

近期Brier同样变差，越低越好：

| 方案 | R19 | R23 | R25 |
|---|---:|---:|---:|
| 原年度等权 | 0.246843 | 0.249790 | 0.249824 |
| 原四状态季度 | 0.246156 | 0.248903 | 0.248933 |
| 本轮时间加权 | 0.252067 | 0.255758 | 0.255844 |

逐年看，退步集中在两个阶段。2024年47周，R19／R23／R25由年度的34／33／32周正确降至31／29／30。2025年48周仍为22／23／23周正确，但三个方法的Brier均变差。2026年截至8月30周，R19从12升至13周正确（40.0%→43.3%），R23和R25却都从16降至12周（53.3%→40.0%）。R19的这项局部改善保留，但不代表三个方法或整个近期窗口改善。

早期也有真实的局部收益。2021—2023年147周，本轮R19／R23／R25为70／71／70周正确，年度为71／68／68：R23多3周、R25多2周，R19少1周。三个方法早期Brier均改善。全段272周，本轮正确数136／135／135，低于年度139／140／139；全段Brier虽略好于年度，却仍差于原四状态季度。方向正确数与概率质量并不总是同步，因此早期收益和近期退步同时报告。

权重干预按预定规则执行。所有原5年成员均保留，每个信号必须在截止时已联合成熟；原始权重为2^(−信号年龄天数/730.5)，归一化后总和为1，原正则系数0.01不变。每年有1,206—1,212个成熟日样本，最近2年的权重占比从等权约40%升至约60.3%；Kish折算约975—980，仅反映权重集中度。日标签存在重叠，不能把它看作独立样本数。

一个值得继续拆解的变化是整体上涨倾向。2025年末供2026年使用的训练上涨比例，从等权47.93%变为加权51.00%；2026年R23、R25判断上涨的周数从8/30升至18/30，实际上涨为14/30，两者各出现4周恢复、8周退步。与此同时，斜率也重新拟合了，因此这些现象不能证明损失仅由截距或上涨比例变化造成。

加权标签频率已单独保存作为诊断：近期正确57/125，原等权频率为56/125；2026年加权频率14/30，原等权频率16/30。它不等于学习模型，也没有为本轮提供足以支持替换方案的近期优势。原native_mse和等权training_frequency仍在候选历史中精确保留。

24项预先固定的探索性比较中，最小原始p值为0.008599，来自早期R23相对年度的Brier改善；统一Holm校正后最小p为0.206379，没有校正后显著的比较。全部历史此前已经查看，没有新增盲测，多重比较校正不覆盖此前多轮自适应研究。不能用一个早期未校正p值认定时间加权有效，也不能把本轮结果扩大为所有加权方法都无效。

下一步建议做固定的“截距／斜率拆分”对照，定位退步来源。由于本轮两套系数使用完全相同的特征坐标，可以预先固定两个组合：原等权斜率加本轮加权截距，以及本轮加权斜率加原等权截距。分别检验整体涨跌倾向变化和特征系数变化的影响；它们是机械组合的研究对照，不是重新优化所得的模型。保留两套完整模型和原四状态季度作为参照，固定所有年份、种子与比较规则，不搜索半衰期或阈值。本轮尚未计算这两个组合的预测，也没有开始下一轮。

独立复核通过：72个新最优解由不同求解器核验，72个原等权头与其预测复现；18个年度／种子接口的未来特征与标签扰动不改变训练输入。3,264条学习方法种子预测、5,733个指标单元、1,088条逐周比较及24项统计比较均核对。原27条预测历史和6,056个旧文件保留；两张图表已目视检查。原神经缓存采用此前已核验的固定成果与哈希核验，本轮没有执行新的神经推理。

原旧训练预算R25近期73/125、58.4%，自然20遍年度R25为71/125、56.8%，原四状态季度R23为73/125、58.4%，继续分别保留。

[完整实验报告与图](第四十七轮实验报告.md) · [训练权重](training_weights.csv) · [权重与标签比例](weight_summary.csv) · [逐周比较](weekly_policy_effects.csv) · [24项比较](primary_comparisons.json) · [独立复核](verification.json)
'''
revision=read(OUT/'visual_revision.json');assert revision['frozen_report_source_sha256']==sha(ROOT/'report47.py')
for name,h in revision['original_artifacts'].items():assert sha(ROOT/name)==h
for name,h in revision['updated_artifacts'].items():assert sha(ROOT/name)==h
text=text.replace('两张图表已目视检查。','两张图表已目视检查。权重图初版的图例与横轴标题相挤，已仅调整排版；初版图像与清单保存在visual_attempt1，修订来源和记录另行留档，模型、指标及冻结源码未改动。')
human=OUT/'结果解读与下一步.md';assert not human.exists();human.write_text(text,encoding='utf-8');reports=[human,OUT/'第四十七轮实验报告.md']
for path in reports:
    body=path.read_text(encoding='utf-8');assert '\ufffd' not in body and '????' not in body
    width=None
    for line in body.splitlines():
        if line.startswith('|'):n=len(line.split('|'));width=width or n;assert n==width,(path,line)
        else:width=None
    for target in re.findall(r'\]\(([^)]+)\)',body):assert (path.parent/target).exists(),target
assert not (ROOT.parent/'research_v48').exists() and '本轮尚未计算这两个组合的预测' in text
images=[OUT/f'{n}.{ext}' for n in ['yearly_accuracy','training_weights'] for ext in ['png','svg']]
assert not (OUT/'visual_review.json').exists() and not (OUT/'interpretation_checks.json').exists()
save(OUT/'visual_review.json',dict(status='PASS',completed_utc=now(),review='Both final PNGs inspected with view_image:annual accuracy legends/partial2026footnote and all3panels readable;trainingn,Kish concentration,recent2yearmass andclassfrequency axes/legends legible;no clipping or overlapping text. Data labels and metric tables agree.',artifacts={str(p.relative_to(ROOT)):sha(p) for p in images}))
save(OUT/'interpretation_checks.json',dict(status='PASS',completed_utc=now(),checks=['early_recent_pooled_2024_2025_2026_correctcounts','recent_Brier_all3references','local_early_probability_improvement_preserved','2026R19_gain_notgeneralized','recent_recovery_regression_counts','2026up_prediction_shift_vsobserved','weight_concentration_and_no_independentN_claim','weightedfrequency_diagnostic_separate','24raw_and_Holmpvalues','72newsolutions_and72uniformreplays','27oldhistories_6056oldfiles_unchanged','new_neural_fit_and_inference_zero','next_intercept_slope_combinations_not_started','old_budget_distinctions','markdown_links_and_table_shapes'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in reports+[OUT/'visual_review.json',OUT/'verification.json']}))
print('R47 human results,numerical claims,provenance,links andvisualreview PASS.')

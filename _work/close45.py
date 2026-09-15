from pathlib import Path
import json,hashlib,math
import pandas as pd
root=Path('D:/ddos_v3/research_v45');out=root/'results';attempt=root.parent/'research_v45_attempt1/results'
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(out/f'{n}.csv',float_precision='round_trip')
v=read('verification.json');assert v['status']=='PASS'
expected=dict(old_files_preserved=5916,old_histories_preserved=23,new_neural_fits=0,independent_comparisons=48,invalid_attempt_files_preserved=68,independent_scalar_solutions=588,independent_gate_cells=368,independent_validation_seed_rows=6792,outside_training_poison_cutoffs=23,postcutoff_validation_poison_cutoffs=23,independent_learned_seed_predictions=6528,exact_zero_offset_predictions=5586,exact_ensemble_fallback_predictions=1862,independent_metric_cells=5301,independent_weekly_effects=2176,independent_decision_outcomes=368,independent_substate_outcomes=736)
assert all(v[k]==value for k,value in expected.items())
methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];trend='quarter_trend2_validated';vol='quarter_volatility2_validated';quarter='weekly_state_validated';annual='rolling5_annual20';m=csv('ensemble_metrics');s=csv('support_summary');coverage=csv('policy_coverage');refs=csv('reference_comparisons');w=csv('weekly_policy_effects');subs=csv('substate_outcomes');vp=csv('validation_predictions');facts=[]
cases=[
 ('recent_2024_2026',125,quarter,[69,73,72],[.246156,.248903,.248933]),
 ('recent_2024_2026',125,trend,[68,72,71],[.247539,.249790,.249824]),
 ('recent_2024_2026',125,vol,[68,72,71],[.247831,.249816,.249840]),
 ('recent_2024_2026',125,annual,[68,72,71],[.246843,.249790,.249824]),
 ('extension_2021_2023',147,quarter,[72,69,69],[.277416,.275610,.275517]),
 ('extension_2021_2023',147,trend,[71,69,69],[.277157,.275379,.275287]),
 ('extension_2021_2023',147,vol,[72,70,69],[.277453,.274654,.275558]),
 ('year_2026',30,quarter,[12,16,16],[.258700,.257140,.257098]),
 ('year_2026',30,trend,[12,16,16],[.260401,.257140,.257098]),
 ('year_2026',30,vol,[11,16,16],[.260833,.257140,.257098]),
 ('pooled_2021_2026',272,quarter,[141,142,141],[.263050,.263337,.263300]),
 ('pooled_2021_2026',272,vol,[140,142,140],[.263840,.263239,.263739]),
]
for period,n,history,correct,brier in cases:
    g=m[m.period.eq(period)&m.history.eq(history)].set_index('method').loc[methods];assert g.n.eq(n).all() and g.correct_directions.tolist()==correct and g.brier.round(6).tolist()==brier
    assert all(abs(a-c/n)<1e-14 for a,c in zip(g.accuracy,correct));facts.append(dict(period=period,history=history,n=n,correct=correct,brier=brier))
for family,counts,pct in [('four_state',[52,32,11],21.2),('trend',[26,26,14],53.8),('volatility',[26,23,13],50.0)]:
    r=s[s.family.eq(family)].iloc[0];assert [r.ready_cells,r.training_supported_cells,r.validation_supported_cells]==counts and round(100*r.validation_support_fraction,1)==pct
for history,counts in [(trend,[2,0,0]),(vol,[42,34,34])]:
    g=coverage[coverage.period.eq('recent_2024_2026')&coverage.history.eq(history)].set_index('method').loc[methods];assert g.changed_probability_weeks.tolist()==counts and g.validation_accepted_weeks.tolist()==counts
for method,nums in zip(methods,[[3,3],[2,2],[2,2]]):
    r=refs[refs.period.eq('recent_2024_2026')&refs.history.eq(vol)&refs.method.eq(method)&refs.reference_history.eq(annual)].iloc[0];assert [r.recoveries,r.regressions]==nums
for method in methods[1:]:
    g=w[w.history.eq(vol)&w.method.eq(method)&w.date.ge('2024-01-01')]
    assert g[g.case_vs_annual.eq('regression')].date.tolist()==['2024-07-12','2024-08-09']
    assert g[g.case_vs_annual.eq('recovery')].date.tolist()==['2024-06-28','2024-08-02']
r=w[w.history.eq(vol)&w.method.eq(methods[0])&w.date.eq('2026-08-14')].iloc[0];assert r.case_vs_annual=='regression' and round(r.probability,6)==.511458 and round(r.annual_probability,6)==.486399 and r.actual_up==0
assert w[w.history.eq(vol)&w.method.eq(methods[1])&w.date.eq('2022-07-08')].iloc[0].case_vs_annual=='recovery'
for cutoff,state,n,bd,recovery,regression in [('2024-03-31','negative_low',3,-.032642,1,0),('2024-03-31','nonnegative_low',8,.005596,0,0),('2024-06-30','negative_low',11,.002772,1,2)]:
    r=subs[subs.cutoff.eq(cutoff)&subs.history.eq(vol)&subs.method.eq(methods[1])&subs.state.eq(state)].iloc[0];assert [r.n,r.recoveries,r.regressions]==[n,recovery,regression] and round(r.brier_difference,6)==bd
g=vp[vp.cutoff.eq('2024-06-30')&vp.method.eq(methods[1])&vp.family.eq('volatility')&vp.component.eq('low')];rows=[]
for idx,z in g.groupby('row_index'):
    p=math.fsum(z.candidate_probability)/3;b=math.fsum(z.annual_probability)/3;y=int(z.actual_up.iloc[0]);rows.append(dict(state=z.state.iloc[0],bd=(p-y)**2-(b-y)**2))
d=pd.DataFrame(rows)
for state,n,bd in [('negative_low',1,-.050448),('nonnegative_low',11,.003765)]:
    g=d[d.state.eq(state)];assert len(g)==n and round(g.bd.mean(),6)==bd
pp=read('primary_comparisons.json');assert len(pp)==48 and round(min(x['p'] for x in pp),6)==.079492 and all(x['holm_adjusted_p']==1. for x in pp)
ensemble=csv('ensemble_predictions');a=pd.read_csv(attempt/'ensemble_predictions.csv',float_precision='round_trip').set_index(['history','method','row_index']).sort_index();b=ensemble.set_index(['history','method','row_index']).sort_index();assert a.direction_up.equals(b.direction_up) and (a.probability-b.probability).abs().max()<=2.23e-16
ei=ensemble.set_index(['history','method','row_index'])
for method,num in zip(methods,[6,3,3]):
    g=ensemble[ensemble.history.eq(quarter)&ensemble.method.eq(method)&ensemble.date.ge('2024-01-01')];assert sum(r.probability!=ei.loc[(annual,method,r.row_index),'probability'] for r in g.itertuples())==num
for name in v['numerical_repair_unchanged_fit_gate_seed_files']:assert sha(out/name)==sha(attempt/name)
report=read('report_manifest.json');assert all(sha(root/n)==h for n,h in report['artifacts'].items())
human=(out/'结果解读与下一步.md').read_text(encoding='utf-8')
for phrase in ['没有新增盲测','73/125、58.4%','71/125、56.8%','本轮尚未开始该候选实验','1,862 条零偏移集成预测与年度精确相同','没有调整分组、训练参数、验证门槛或比较方案','整体合并掩盖了部分子状态损失']:
    assert phrase in human
assert not (out/'visual_review.json').exists() and not (out/'interpretation_checks.json').exists()
review=dict(status='PASS',reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),observations='Both corrected-run PNGs visually inspected. All yearly lines and partial-2026 caption visible. Support panel explicitly shows distinct denominators52/26/26 and mutually exclusive support conditions. Actual recent coverage panel shows6/3/3,2/0/0,42/34/34 with zero labels, matching repaired ensemble-exact fallback. Legends/captions are readable and not clipped. Artifact hashes match report manifest.',artifacts={f'results/{n}.{ext}':sha(out/f'{n}.{ext}') for n in ['yearly_accuracy','support_and_coverage'] for ext in ['png','svg']})
(out/'visual_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
checks=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),numeric_facts=facts,checks=['All cited period counts and Brier','Support denominator distinctions','Corrected accepted coverage and quarterly reference counts','All recent recovery/regression dates and 2026 deterioration','Preserved early and pooled local R23 improvement','Merged-group validation contributions and original-state future effects','48 fixed comparisons nonsignificant','Numerical repair: identical fit/gate/seed files and unchanged direction forecasts; invalid attempt retained','Original protocol58.4distinctions and no new blindholdout','Next shared-fit four-state-gate experiment not started','Independent verification counts and both corrected figures actually inspected'],artifacts={f'results/{n}':sha(out/n) for n in ['结果解读与下一步.md','visual_review.json']})
(out/'interpretation_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8');print('R45 human results, repair provenance, numerical claims and visual review PASS.')

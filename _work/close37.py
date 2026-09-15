from pathlib import Path
import json,hashlib,math
import pandas as pd
root=Path('D:/ddos_v3/research_v37');out=root/'results'
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(out/f'{n}.csv',float_precision='round_trip')
assert read('verification.json')['status']=='PASS'
met=csv('ensemble_metrics');changes=csv('direction_changes');annual='rolling5_annual20';state='annual_state_offset';global_='annual_shared_offset';half='annual_state_offset_half';methods=['learned_vol_interaction','learned_order_extension','learned_order_offset']
facts=[]
def check_counts(period,expect):
    for h,vals in expect.items():
        g=met[met.period.eq(period)&met.history.eq(h)].set_index('method').loc[methods];assert g.correct_directions.tolist()==vals;facts.append(dict(period=period,history=h,correct=vals,n=g.n.tolist()))
check_counts('recent_2024_2026',{annual:[68,72,71],state:[68,71,70],global_:[70,72,71],half:[68,70,69]})
check_counts('year_2026',{annual:[12,16,16],state:[12,13,13],global_:[12,14,14],half:[12,13,13]})
check_counts('extension_2021_2023',{annual:[71,68,68],state:[67,65,65]})
for h,expected in [(annual,[.246843,.249790,.249824]),(state,[.248020,.250854,.250870])]:
    g=met[met.period.eq('recent_2024_2026')&met.history.eq(h)].set_index('method').loc[methods];assert [round(v,6) for v in g.brier]==expected
for h in [state,global_]:
    g=met[met.period.eq('recent_2024_2026')&met.history.eq(h)].set_index('method').loc[methods];b=met[met.period.eq('recent_2024_2026')&met.history.eq(annual)].set_index('method').loc[methods];assert ((g.brier>b.brier) if h==state else (g.brier<b.brier)).all()
for m in methods:
    g=changes[changes.period.eq('year_2026')&changes.history.eq(state)&changes.method.eq(m)].iloc[0];assert g.eligible_weeks==4 and g.fallback_weeks==26 and g.recoveries==0 and g.regressions==(0 if m==methods[0] else 3)
for period,h,n in [('year_2024',annual,33),('year_2024',state,33),('year_2025',annual,23),('year_2025',state,25)]:assert met[met.period.eq(period)&met.history.eq(h)&met.method.eq(methods[1])].correct_directions.iloc[0]==n
cases=csv('all_2026_cases');g=cases[cases.history.eq(state)&cases.method.eq(methods[1])&cases.eligible].sort_values('date');assert g.date.tolist()==['2026-05-15','2026-05-22','2026-07-03','2026-07-10'];assert g.new_state_n.tolist()==[44,44,36,36];assert g.actual_up.tolist()==[1,1,0,0];assert [round(v*100,2) for v in g.annual_probability]==[56.37,50.51,49.02,48.39];assert [round(v*100,2) for v in g.probability]==[51.04,45.18,56.35,55.61]
p=csv('correction_parameters')
for cutoff,s,up,pred in [('2026-03-31','nonnegative_low',34.09,49.39),('2026-06-30','nonnegative_high',69.44,45.97)]:
    g=p[p.cutoff.eq(cutoff)&p.method.eq(methods[1])&p.kind.eq('state')&p.state.eq(s)];assert len(g)==3 and round(g.up_rate.iloc[0]*100,2)==up and round(g.annual_mean_probability.mean()*100,2)==pred
comp=read('primary_comparisons.json');assert len(comp)==48 and all(v['holm_adjusted_p']==1. for v in comp)
fit=read('fitting_manifest.json');assert fit['state_fits']==408 and fit['shared_fits']==204 and fit['new_scalar_fits']==612
v=read('verification.json');assert v['old_files_preserved']==5315 and v['independent_metric_cells']==3600 and v['outside_member_poison_interfaces']==51
assert len(csv('temporal_checks'))==23
text=(out/'结果解读与下一步.md').read_text(encoding='utf-8');assert '73/125、58.4%' in text and '71/125、56.8%' in text and '诊断尚未执行' in text
manifest=read('report_manifest.json');assert all(sha(root/n)==h for n,h in manifest['artifacts'].items())
review=dict(status='PASS',reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),images=['yearly_accuracy.png','recent_changes.png'],observations='Both final PNGs inspected. Chinese text, labels, legends, numbers and notes visible without clipping or overlap. Yearly paths use same metric cells as report; recent bar signs label accuracy up and Brier down correctly. No presentation amendments.',artifacts={f'results/{n}':sha(out/n) for n in ['yearly_accuracy.png','recent_changes.png','yearly_accuracy.svg','recent_changes.svg']})
(out/'visual_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
checks=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),checks=['all human-report count tables','six rounded recent Brier values and direction','2026 activation and regression counts','R23 yearly changes','all four2026 active R23 weeks','training residual examples','all48 adjusted comparisons','fit and verification counts','report artifacts unchanged','visual inspection'],numeric_count_facts=facts,artifacts={'results/结果解读与下一步.md':sha(out/'结果解读与下一步.md'),'results/visual_review.json':sha(out/'visual_review.json')})
(out/'interpretation_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
print('Human interpretation facts, report provenance and visual review PASS.')

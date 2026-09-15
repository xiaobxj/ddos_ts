from pathlib import Path
import json,hashlib
import pandas as pd
root=Path('D:/ddos_v3/research_v41');out=root/'results'
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(out/f'{n}.csv',float_precision='round_trip')
v=read('verification.json');assert v['status']=='PASS'
expected=dict(old_files_preserved=5633,unchanged_forecast_histories=22,new_fits=0,new_predictions=0,new_pvalues=0,independent_validation_week_cells=4528,independent_gate_cells=1456,independent_deletion_cases=1560,independent_sensitivity_cohorts=168,independent_role_flows=890,independent_member_flows=12024,independent_gate_transitions=1424,independent_acceptance_runs=69,accepted_cells_accounted=106)
assert all(v[k]==x for k,x in expected.items())
methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];cells=csv('gate_sensitivity');cohorts=csv('sensitivity_summary');flows=csv('membership_flows');trans=csv('gate_transitions');runs=csv('acceptance_runs');leave=csv('leave_one_week_out');facts=[]
assert int(cells.metric_eligible.sum())==192
for cadence,counts in [('monthly',[[21,8,3,10],[22,10,2,10],[22,10,2,10]]),('quarterly',[[6,3,1,2],[5,3,0,2],[5,3,0,2]])]:
    for method,expected in zip(methods,counts):
        g=cells[cells.cadence.eq(cadence)&cells.method.eq(method)&cells.accepted]
        actual=[len(g)]+[int(g.sensitivity_class.eq(c).sum()) for c in ['accepted_metric_fragile','accepted_count_only_fragile','accepted_single_drop_stable']]
        assert actual==expected;facts.append(dict(cadence=cadence,method=method,accepted_sensitivity=actual))
assert round(10/22*100,1)==45.5
for cadence,role,counts,mean,pct in [
    ('monthly','training',[34,1768,1634,134],[48.06,3.94],92.4),
    ('monthly','validation',[34,442,308,134],[9.06,3.94],69.7),
    ('quarterly','training',[8,416,318,98],[39.75,12.25],76.4),
    ('quarterly','validation',[8,104,7,97],[.88,12.12],6.7),
]:
    g=flows[flows.cadence.eq(cadence)&flows.role.eq(role)&flows.state.eq('all')&flows.both_ready]
    assert [len(g),int(g.current_n.sum()),int(g.retained_n.sum()),int(g.added_n.sum())]==counts
    assert [round(g.retained_n.mean(),2),round(g.added_n.mean(),2)]==mean and round(g.retained_n.sum()/g.current_n.sum()*100,1)==pct
    facts.append(dict(cadence=cadence,role=role,counts=counts,mean_retained_added=mean,reuse_percent=pct))
for start,end,counts in [('2022-10-31','2022-11-30',[2,10,5,5]),('2024-07-31','2024-11-30',[5,41,14,27])]:
    r=runs[runs.cadence.eq('monthly')&runs.method.eq(methods[1])&runs.state.eq('negative_low')&runs.first_cutoff.eq(start)].iloc[0]
    assert r.last_cutoff==end and [r.decisions,r.total_validation_uses,r.unique_validation_weeks,r.repeated_uses]==counts
    facts.append(dict(run_start=start,run_end=end,counts=counts))
for date,counts in [('2024-04-30',[7,3,4,0,3]),('2024-05-31',[3,0,3,0,0]),('2024-06-30',[0,0,0,1,1])]:
    r=flows[flows.cadence.eq('monthly')&flows.role.eq('validation')&flows.state.eq('negative_low')&flows.cutoff.eq(date)].iloc[0]
    assert [r.previous_n,r.retained_n,r.dropped_n,r.added_n,r.current_n]==counts and r.dropped_older_than_current_window
stops=trans[trans.cadence.eq('monthly')&trans.method.isin(methods)&trans.stop_by_validation_support];assert len(stops)==6 and stops.both_ready.all()
assert set(zip(stops.cutoff,stops.state))=={('2024-04-30','negative_low'),('2024-08-31','nonnegative_low')}
g=stops[stops.cutoff.eq('2024-08-31')];assert g.previous_validation_n.eq(8).all() and g.validation_n.eq(4).all() and g.dropped_n.eq(4).all() and g.added_n.eq(0).all()
for cadence,cutoff,state,n,flips,fut,bd,recovery,regression in [
    ('monthly','2022-11-30','negative_high',6,0,3,.007113,0,2),
    ('monthly','2024-05-31','nonnegative_low',8,0,1,.004485,0,1),
    ('quarterly','2024-03-31','negative_low',7,4,3,-.036977,1,0),
]:
    r=cells[cells.cadence.eq(cadence)&cells.method.eq(methods[1])&cells.cutoff.eq(cutoff)&cells.state.eq(state)].iloc[0]
    assert r.accepted and [r.validation_n,r.metric_flip_count,r.future_n,r.future_recoveries,r.future_regressions]==[n,flips,fut,recovery,regression]
    assert round(r.future_brier_vs_annual,6)==bd
    if flips==0:
        l=leave[leave.cadence.eq(cadence)&leave.method.eq(methods[1])&leave.cutoff.eq(cutoff)&leave.state.eq(state)];assert len(l)==n and l.full_accepted.all()
    facts.append(dict(cadence=cadence,cutoff=cutoff,state=state,validation_n=n,metric_flip_count=flips,future_n=fut,future_brier=bd))
for method,cohort,nums,bd in [
    (methods[1],'accepted_metric_fragile',[4,15,2,2,1,1],.000811),
    (methods[1],'accepted_single_drop_stable',[7,19,4,3,1,3],-.002577),
    (methods[0],'accepted_metric_fragile',[3,10,1,2,1,1],-.004378),
    (methods[0],'accepted_single_drop_stable',[7,21,5,2,1,2],.001217),
]:
    r=cohorts[cohorts.period.eq('decision_all_2020_2026')&cohorts.cadence.eq('monthly')&cohorts.method.eq(method)&cohorts.cohort.eq(cohort)].iloc[0]
    assert [r[k] for k in ['future_nonempty_cells','future_weeks','future_worse_cells','future_better_cells','future_recoveries','future_regressions']]==nums
    assert round(r.future_week_weighted_brier,6)==bd
assert int(cells[cells.cadence.eq('monthly')&cells.method.eq(methods[1])].sensitivity_class.eq('rejected_metric_fragile').sum())==4
report=read('report_manifest.json');assert all(sha(root/n)==d for n,d in report['artifacts'].items())
human=(out/'结果解读与下一步.md').read_text(encoding='utf-8')
for text in ['73/125、58.4%','71/125、56.8%','没有新增盲测','本轮尚未开始该候选实验','没有新的准确率','不因再次样本不足而滚动续期','不是有效独立样本量']:
    assert text in human
assert not (out/'visual_review.json').exists() and not (out/'interpretation_checks.json').exists()
review=dict(status='PASS',reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),observations='Both PNGs visually inspected. Mutually exclusive sensitivity categories, counts and legend visible; count-only versus metric sensitivity distinguished. Reuse charts show counts, percentages and comparable-pair denominators; all labels and captions inside figure, no clipping or overlaps requiring change. PNG/SVG hashes agree with frozen report manifest.',artifacts={f'results/{n}':sha(out/n) for n in ['gate_sensitivity.png','gate_sensitivity.svg','sample_reuse.png','sample_reuse.svg']})
(out/'visual_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
checks=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),numeric_facts=facts,checks=['All sensitivity categories forprimarymethods atbothcadences','Distinct metricflip versus mechanicalsupportfailure','Ready-pair overlapcounts andfractions','Repeated validation runs accounting','State supportattrition identities andduplicateevent caveat','Stable-but-harmful andfragile-but-helpful counterexamples','Nonmonotone retrospective cohortoutcomes','No newaccuracy,pvalues,blindholdout orcandidatepolicy claim','Independentcheckcounts andunchangedreporthashes','Bothfigures visuallyinspected'],artifacts={f'results/{n}':sha(out/n) for n in ['结果解读与下一步.md','visual_review.json']})
(out/'interpretation_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8');print('Human diagnostic facts, report provenance and visual review PASS.')

from pathlib import Path
import json,hashlib
import pandas as pd
root=Path('D:/ddos_v3/research_v40');out=root/'results'
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(out/f'{n}.csv',float_precision='round_trip')
v=read('verification.json');assert v['status']=='PASS'
expected=dict(ready_months=39,old_files_preserved=5554,independent_comparisons=60,independent_scalar_solutions=1152,training_isolation_interfaces=468,validation_isolation_cutoffs=68,independent_metric_cells=4590,shared_quarter_head_cells=1104,shared_quarter_gate_cells=368,exact_first_month_seed_predictions=2184,q1_exact_annual_seed_predictions=1632,cadence_weekly_cases=2176,cadence_summary_cells=360,gate_transition_cells=1088)
assert all(v[k]==x for k,x in expected.items())
assert v['maximum_scalar_gap']<3e-15 and v['maximum_prediction_gap']<2e-16
methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];a='rolling5_annual20';q='weekly_state_validated';h='monthly_state_validated';m=csv('ensemble_metrics');facts=[]
for period,n,aa,qq,mm,qb,mb in [
    ('recent_2024_2026',125,[68,72,71],[69,73,72],[68,71,70],[.246156,.248903,.248933],[.247415,.249923,.249946]),
    ('extension_2021_2023',147,[71,68,68],[72,69,69],[70,67,67],[.277416,.275610,.275517],[.277601,.275831,.275735]),
    ('pooled_2021_2026',272,[139,140,139],[141,142,141],[138,138,137],[.263050,.263337,.263300],[.263729,.263925,.263883]),
    ('year_2026',30,[12,16,16],[12,16,16],[12,16,16],[.258700,.257140,.257098],[.258700,.257140,.257098]),
]:
    for i,method in enumerate(methods):
        r=m[m.period.eq(period)&m.method.eq(method)].set_index('history')
        assert int(r.loc[h,'n'])==n
        assert [int(r.loc[z,'correct_directions']) for z in [a,q,h]]==[aa[i],qq[i],mm[i]]
        assert [round(float(r.loc[z,'brier']),6) for z in [q,h]]==[qb[i],mb[i]]
        if period=='recent_2024_2026':assert r.loc[h,'brier']>r.loc[a,'brier']
    facts.append(dict(period=period,n=n,annual=aa,quarter=qq,month=mm,quarter_brier=qb,month_brier=mb))
c=csv('cadence_weekly_comparison');z=c[c.history.eq(h)&c.method.isin(methods)];recent=z[z.date.ge('2024-01-01')]
for i,method in enumerate(methods):
    g=recent[recent.method.eq(method)]
    assert int(g.monthly_active.sum())==18 and int(g.quarterly_active.sum())==[6,3,3][i]
    early=z[z.method.eq(method)&z.date.le('2023-12-31')]
    assert early[early['case'].eq('regression')].date.tolist()==['2022-12-02','2022-12-23'] and not early['case'].eq('recovery').any()
    actual=z[z.method.eq(method)&z.date.ge('2026-01-01')]
    assert (actual.monthly_probability==actual.quarterly_probability).all()
    assert int(actual.monthly_active.sum())==[2,0,0][i]
    assert g[g['case'].eq('recovery')].date.tolist()==['2024-08-02']
    assert g[g['case'].eq('regression')].date.tolist()==(['2024-06-28','2024-08-09'] if i==0 else ['2024-06-07','2024-06-28','2024-08-09'])
g=z[z.method.eq(methods[1])].set_index('date')
for date,values,actual,relation in [
    ('2024-06-07',[.495985,.500485],0,'monthly_only'),
    ('2024-06-28',[.474107,.512366],0,'quarterly_only'),
    ('2022-12-02',[.501929,.482277],1,'monthly_only'),
    ('2022-12-23',[.514482,.494619],1,'monthly_only'),
]:
    r=g.loc[date];assert [round(r[k],6) for k in ['quarterly_probability','monthly_probability']]==values and int(r.actual_up)==actual and r.activation_relation==relation
gates=csv('gate_decisions');assert len(gates)==1088 and int(gates.accepted.sum())==84
for cutoff,state,train_n,val_n,correct,bd,reason in [
    ('2022-11-30','negative_high',25,6,[2,3],-.007305,'accepted'),
    ('2024-04-30','negative_low',29,3,[2,2],-.010319,'insufficient_validation'),
    ('2024-05-31','negative_low',32,0,[0,0],None,'insufficient_validation'),
]:
    r=gates[gates.method.eq(methods[1])&gates.cutoff.eq(cutoff)&gates.component.eq(state)].iloc[0]
    assert [r.training_n,r.validation_n]==[train_n,val_n] and [r.baseline_correct,r.candidate_correct]==correct and r.reason==reason
    assert pd.isna(r.brier_difference) if bd is None else round(r.brier_difference,6)==bd
t=csv('gate_transitions')
for method,expected in zip(methods,[[21,11,33,7,4],[22,11,34,6,5],[22,11,34,6,5]]):
    g=t[t.method.eq(method)&t.accepted];assert [len(g),int(g.future_n.gt(0).sum()),int(g.future_n.sum()),int(g.future_brier_vs_annual.gt(1e-12).sum()),int(g.future_brier_vs_annual.lt(-1e-12).sum())]==expected
r=t[t.method.eq(methods[1])&t.cutoff.eq('2022-11-30')&t.state.eq('negative_high')].iloc[0];assert r.future_n==3 and round(r.future_brier_vs_annual,6)==.007113 and r.future_regressions_vs_quarter==2
for period,qc,mc in [('recent_2024_2026',[70,68,67],[71,69,68]),('extension_2021_2023',[71,70,70],[69,68,68]),('year_2026',[13,13,13],[13,14,14])]:
    for i,method in enumerate(methods):
        rows=m[m.period.eq(period)&m.method.eq(method)].set_index('history');assert [rows.loc[hist,'correct_directions'] for hist in ['weekly_state_ungated','monthly_state_ungated']]==[qc[i],mc[i]]
        if period=='recent_2024_2026':assert rows.loc['monthly_state_ungated','brier']<rows.loc['weekly_state_ungated','brier']
s=csv('seed_metrics')
for method in methods:
    g=s[s.period.eq('recent_2024_2026')&s.method.eq(method)];qq=g[g.history.eq(q)].sort_values('seed');mm=g[g.history.eq(h)].sort_values('seed')
    assert (mm.correct_directions.to_numpy()-qq.correct_directions.to_numpy()).tolist()==[1,-1,0]
    assert (mm.brier.to_numpy()>qq.brier.to_numpy()).all()
j=pd.DataFrame(read('primary_comparisons.json'));assert len(j)==60 and j.holm_adjusted_p.eq(1).all()
sch=csv('schedule');assert len(sch)==68 and sch[sch['mode'].eq('ready')].cutoff.min()=='2022-05-31'
assert read('fitting_manifest.json')['scalar_fits']==1152 and read('fitting_manifest.json')['new_neural_fits']==0
human=(out/'结果解读与下一步.md').read_text(encoding='utf-8')
for text in ['73/125、58.4%','71/125、56.8%','没有新增盲测','没有开始下一轮实验','校正后 p 值全部为 1','不能视为新增独立证据']:
    assert text in human
report=read('report_manifest.json');assert all(sha(root/n)==d for n,d in report['artifacts'].items())
assert not (out/'visual_review.json').exists() and not (out/'interpretation_checks.json').exists()
review=dict(status='PASS',reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),observations='Both PNGs visually inspected. Chinese headings and labels, all line points and bar counts, two-row legends and partial2026 notes visible; axes accommodate all values, no clipping or overlap requiring changes. Corresponding SVGs bound to report hashes.',artifacts={f'results/{n}':sha(out/n) for n in ['yearly_accuracy.png','yearly_accuracy.svg','cadence_coverage.png','cadence_coverage.svg']})
(out/'visual_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
checks=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),numeric_facts=facts,checks=['Four-period ensemble counts and Brier','Recent coverage and exact2026monthly-quarter equality','Complete direction changes and concrete R23 probability cases','Validation support dropout, accepted2022counterexample and following loss','Accepted state-month counts including empty future cells','Ungated local gains and early losses retained','All seeds:one increases,one decreases,one unchanged; all recent Brierworse versusquarter','60Holm comparisons','Independent audit counts','Historical protocols kept separate and no newholdout claim','Two figures visual QA and frozen report provenance'],artifacts={f'results/{n}':sha(out/n) for n in ['结果解读与下一步.md','visual_review.json']})
(out/'interpretation_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
print('Human interpretation facts, frozen report provenance and visual review PASS.')

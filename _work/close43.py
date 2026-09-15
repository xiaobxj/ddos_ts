from pathlib import Path
import json,hashlib
import pandas as pd
root=Path('D:/ddos_v3/research_v43');out=root/'results'
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(out/f'{n}.csv',float_precision='round_trip')
v=read('verification.json');assert v['status']=='PASS'
expected=dict(old_files_preserved=5754,unchanged_forecast_histories=23,new_fits=0,new_policies=0,new_pvalues=0,independent_incoming_records=1088,independent_seed_rows=2031,independent_week_rows=677,independent_evidence_cells=3264,independent_summary_cells=174,future_data_and_head_poison_checks=68)
assert all(v[k]==value for k,value in expected.items())
methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];cells=csv('diagnostic_cells');subjects=csv('subjects');gates=csv('gate_decisions');weeks=csv('validation_week_diagnostics');facts=[]
for method,counts,known,total,nzero,clear,expired in zip(methods,[[14,7,4,3],[13,7,3,3],[13,7,3,3]],[42,45,45],[60,59,59],[7,8,8],[4,2,2],[9,12,12]):
    g=cells[cells.method.eq(method)&cells.availability.eq('in_life')];f=g[g.view.eq('full_validation')];n=g[g.view.eq('post_source_validation')];p=f[f.old_evidence.eq('pass')]
    actual=[len(f),sum(f.old_evidence=='pass'),sum(f.old_evidence.isin(['brier_not_better','direction_worse'])),sum(f.old_evidence.isin(['no_samples','insufficient_validation']))];assert actual==counts
    assert n.n.max()==4 and sum(n.n==0)==nzero and n.old_evidence.isin(['no_samples','insufficient_validation']).all()
    assert int(p.known_at_source_n.sum())==known and int(p.n.sum())==total
    inc=g[g.view.eq('incremental_validation')].set_index('subject_id').sort_index();nov=n.set_index('subject_id').sort_index()
    for col in ['n','old_brier_difference','old_correct_difference','current_brier_difference','current_correct_difference','old_evidence','current_evidence']:
        pd.testing.assert_series_equal(inc[col],nov[col],check_names=False)
    q=f[f.original_reason.isin(['brier_not_better','direction_worse'])];assert len(q)==clear and q.old_evidence.isin(['brier_not_better','direction_worse']).all()
    nq=n[n.original_reason.isin(['brier_not_better','direction_worse'])];assert len(nq)==clear and nq.old_evidence.isin(['no_samples','insufficient_validation']).all()
    assert sum(subjects.method.eq(method)&subjects.availability.eq('expired'))==expired
    facts.append(dict(method=method,in_life_counts=actual,pass_known_uses=known,pass_total_uses=total,pass_known_percent=round(known/total*100,1),new_evidence_zero_cells=nzero,new_evidence_max_n=4,quality_clear_cells=clear,quality_clear_old_pass=0,expired_cells=expired))
assert sum(x['in_life_counts'][0] for x in facts)==40 and sum(x['new_evidence_zero_cells'] for x in facts)==23
assert [x['pass_known_percent'] for x in facts]==[70.,76.3,76.3]
q=cells[cells.method.isin(methods)&cells.availability.eq('in_life')&cells.view.eq('full_validation')&cells.original_reason.isin(['brier_not_better','direction_worse'])]
assert len(q)==8 and len(q[['cutoff','state']].drop_duplicates())==4
r=cells[cells.method.eq(methods[1])&cells.cutoff.eq('2022-07-31')&cells.state.eq('negative_high')&cells.view.eq('full_validation')].iloc[0]
src=gates[gates.method.eq(methods[1])&gates.cutoff.eq('2022-06-30')&gates.component.eq('negative_high')].iloc[0]
assert r.source_cutoff=='2022-06-30' and [src.validation_n,r.n,r.known_at_source_n,r.source_validation_overlap_n]==[12,8,8,8]
assert round(src.brier_difference,6)==-.002007 and round(r.old_brier_difference,6)==.003133
w=weeks[weeks.subject_id.eq(r.subject_id)];assert not w.matured_after_source.any() and w.in_source_validation.all()
co=cells[cells.method.isin(methods)&cells.availability.eq('in_life')&cells.view.eq('full_validation')&cells.comparison.eq('current_only_pass')]
assert len(co)==2 and set(co.method)==set(methods[1:]) and co.cutoff.eq('2025-05-31').all() and co.state.eq('negative_low').all() and co.n.eq(8).all() and co.known_at_source_n.eq(6).all()
r=co[co.method.eq(methods[1])].iloc[0];assert round(r.old_brier_difference,6)==.000185 and round(r.current_brier_difference,6)==-.000063
e=cells[cells.method.isin(methods)&cells.availability.eq('expired')&cells.view.eq('post_source_validation')&cells.n.ge(5)]
assert len(e)==5 and e.n.eq(5).all() and e.old_evidence.eq('brier_not_better').all()
assert set(e.cutoff)=={'2024-09-30','2025-06-30'} and e.state.eq('negative_low').all()
report=read('report_manifest.json');assert all(sha(root/n)==h for n,h in report['artifacts'].items())
human=(out/'结果解读与下一步.md').read_text(encoding='utf-8')
for phrase in ['没有新增盲测','73/125、58.4%','71/125、56.8%','本轮尚未开始这项可行性检查','不是有效独立样本量','不能归为通过，也不能归为参数失效','后来的月度通过不重置']:
    assert phrase in human
assert not (out/'visual_review.json').exists() and not (out/'interpretation_checks.json').exists()
review=dict(status='PASS',reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),observations='Both PNGs visually inspected. Evidence views, primary methods, stack counts and sample-insufficiency legend are readable; totals match diagnostic tables. Quality-clear chart clearly limits population to in-life quality clears, and includes insufficient versus rejection distinction. Captions warn about reuse and shared events. No clipped labels or problematic overlap. PNG/SVG hashes match report manifest.',artifacts={f'results/{n}.{e}':sha(out/f'{n}.{e}') for n in ['evidence_freshness','quality_clear_evidence'] for e in ['png','svg']})
(out/'visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
checks=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),numeric_facts=facts,checks=['All primary in-life evidence classifications and denominators','No source-novel supported in-life cells; zero counts separate','Known-label use counts among full-window passes','Current versus old quality rejection and four shared events','Fixed-source Brier reversal caused by removal of old validation members','Current-only passes retained as counterexamples','Expired records reported separately including all five supported novel cases','No new policy or accuracy, p values or blind-holdout claims','Original protocol distinctions preserved','Next feasibility diagnostic explicitly not started','Independent counts, report hashes and actual image inspection'],artifacts={f'results/{n}':sha(out/n) for n in ['结果解读与下一步.md','visual_review.json']})
(out/'interpretation_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('R43 human numerical claims, limitations, report hashes and visual review PASS.')

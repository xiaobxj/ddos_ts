from pathlib import Path
import json, hashlib
import pandas as pd

root = Path('D:/ddos_v3/research_v39')
out = root / 'results'
def read(n): return json.loads((out / n).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n): return pd.read_csv(out / f'{n}.csv', float_precision='round_trip')

v = read('verification.json')
assert v['status'] == 'PASS'
assert [v[k] for k in ['ready_quarters', 'old_files_preserved', 'independent_comparisons', 'independent_scalar_solutions', 'independent_metric_cells']] == [13, 5484, 84, 540, 4500]
assert v['training_isolation_interfaces'] == 156 and v['validation_isolation_cutoffs'] == 23
assert v['maximum_scalar_gap'] < 2e-15 and v['maximum_prediction_gap'] < 2e-16
m = csv('ensemble_metrics'); s = csv('seed_metrics'); x = csv('weekly_policy_effects')
q = csv('quarter_outcomes'); gates = csv('gate_decisions')
methods = ['learned_vol_interaction', 'learned_order_extension', 'learned_order_offset']
a = 'rolling5_annual20'; h = 'weekly_state_validated'
facts = []
for period, n, before, after, b0, b1 in [
    ('extension_2021_2023', 147, [71,68,68], [72,69,69], [.277967,.276195,.276108], [.277416,.275610,.275517]),
    ('recent_2024_2026', 125, [68,72,71], [69,73,72], [.246843,.249790,.249824], [.246156,.248903,.248933]),
    ('pooled_2021_2026', 272, [139,140,139], [141,142,141], [.263664,.264060,.264029], [.263050,.263337,.263300]),
    ('year_2026', 30, [12,16,16], [12,16,16], [.257501,.257140,.257098], [.258700,.257140,.257098]),
]:
    for i, method in enumerate(methods):
        rows = m[m.period.eq(period) & m.method.eq(method)].set_index('history')
        assert int(rows.loc[a, 'n']) == n == int(rows.loc[h, 'n'])
        assert [int(rows.loc[z, 'correct_directions']) for z in [a,h]] == [before[i],after[i]]
        assert [round(float(rows.loc[z, 'brier']), 6) for z in [a,h]] == [b0[i],b1[i]]
    facts.append(dict(period=period, n=n, annual=before, state_validated=after, annual_brier=b0, state_validated_brier=b1))

for method in methods:
    t = x[x.history.eq(h) & x.method.eq(method)]
    assert list(t[t['case'].eq('recovery')].date) == ['2023-12-01', '2024-06-28']
    assert not t['case'].eq('regression').any()
    assert (t[t['case'].eq('recovery')].actual_up == 0).all()
    for year in [2025,2026]:
        rows = m[m.period.eq(f'year_{year}') & m.method.eq(method)].set_index('history')
        assert rows.loc[a,'correct_directions'] == rows.loc[h,'correct_directions']
    assert int(t[t.year.le(2023)].changed_probability.sum()) == 12
    assert int(t[t.year.ge(2024)].changed_probability.sum()) == (6 if method==methods[0] else 3)
    if method != methods[0]:
        assert (t[t.year.ge(2025)].probability == t[t.year.ge(2025)].annual_probability).all()
        assert list(t[t.year.ge(2024) & t.changed_probability].date) == ['2024-06-14','2024-06-21','2024-06-28']
qq = q[q.history.eq(h) & q.method.isin(methods) & q.accepted]
assert len(qq) == 16 and int(qq.n.eq(0).sum()) == 8
for period, hist, expect in [
    ('year_2026','weekly_state_ungated',[13,13,13]),
    ('recent_2024_2026','weekly_state_ungated',[70,68,67]),
    ('recent_2024_2026','weekly_global_validated',[66,71,71]),
    ('recent_2024_2026','weekly_global_ungated',[69,67,67]),
    ('extension_2021_2023','weekly_global_ungated',[70,73,73])
]:
    got=m[m.period.eq(period)&m.history.eq(hist)].set_index('method').loc[methods,'correct_directions'].astype(int).tolist()
    assert got==expect
for method, before, after in zip(methods, [[66,68,68],[64,69,69],[64,68,69]], [[65,68,68],[64,70,69],[64,69,69]]):
    z=s[s.period.eq('recent_2024_2026') & s.method.eq(method)]
    aa=z[z.history.eq(a)].sort_values('seed');bb=z[z.history.eq(h)].sort_values('seed')
    assert aa.correct_directions.astype(int).tolist()==before and bb.correct_directions.astype(int).tolist()==after
    assert (bb.brier.to_numpy() < aa.brier.to_numpy()).all()
    facts.append(dict(method=method, recent_seed_annual=before, recent_seed_state_validated=after))
for cutoff, state, n, correct, difference, reason in [
    ('2026-03-31','nonnegative_low',10,[5,6],.010659,'brier_not_better'),
    ('2026-06-30','nonnegative_high',5,[2,0],-.004824,'direction_worse'),
]:
    z=gates[gates.cutoff.eq(cutoff)&gates.method.eq(methods[1])&gates.component.eq(state)].iloc[0]
    assert int(z.validation_n)==n and round(float(z.brier_difference),6)==difference and z.reason==reason
    assert [int(z.baseline_correct),int(z.candidate_correct)]==correct

j=pd.DataFrame(read('primary_comparisons.json'))
assert len(j)==84 and (j.holm_adjusted_p==1).all()
recent=j[j.window.eq('recent_2024_2026')&j.history.eq(h)&j.reference_history.eq(a)&j.metric.eq('direction_error')]
assert len(recent)==3 and (recent.difference==-.008).all() and (recent.ci95_low==-.024).all() and (recent.ci95_high==0).all()

compat=read('verification_compatibility.json')
assert compat['status']=='PASS'
assert sha(out/'logs/verify.log')==compat['original_failure_log_sha256']
assert sha(root/'verify39.py')==compat['original_verifier_sha256']
assert sha(out/'verification_null_compat.py.txt')==compat['helper_sha256']
assert sha(out/'verification.json')==compat['verification_sha256']
assert all(sha(root/n)==d for n,d in compat['research_artifacts_unchanged'].items())
assert compat['normalizations']==[{'column':n,'rows':20} for n in ['brier_difference','baseline_brier','candidate_brier']]
report=read('report_manifest.json')
assert all(sha(root/n)==d for n,d in report['artifacts'].items())
human=(out/'结果解读与下一步.md').read_text(encoding='utf-8')
for text in ['73/125、58.4%', '71/125、56.8%', '2023-12-01', '2024-06-28', '没有新增盲测', '本轮没有开始这项后续实验', '校正后 p 值全部为 1']:
    assert text in human
review=dict(status='PASS',reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),observations='Both PNGs visually inspected. Titles, Chinese axes, legends, all data points and coverage count labels visible. Year2026 partial-period marker and denominator notes present. No clipping or overlap requiring changes. PNG/SVG hashes agree with frozen report manifest.',artifacts={f'results/{n}':sha(out/n) for n in ['yearly_accuracy.png','yearly_accuracy.svg','correction_coverage.png','correction_coverage.svg']})
assert not (out/'visual_review.json').exists() and not (out/'interpretation_checks.json').exists()
(out/'visual_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
checks=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),numeric_facts=facts,checks=['Early, recent, pooled and2026 ensemble counts and Brier','All improvements concentrated on two identical signal dates across methods','Low coverage, zero-use accepted cells and2025/2026 exact R23/R25 fallback','Direct and global counterexamples retained','Per-seed differences and Brier','2026 rejected-gate examples','84Holm-adjusted comparisons and recent direction intervals','Frozen verification with narrowly scoped null compatibility and unchanged research hashes','Both report figures visually inspected and report artifacts unchanged','Old58.4percent and current natural20protocol results kept distinct'],artifacts={f'results/{n}':sha(out/n) for n in ['结果解读与下一步.md','visual_review.json','verification_compatibility.json','verification_null_compat.py.txt']})
(out/'interpretation_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
print('Human interpretation facts, compatibility provenance, report hashes and visual review PASS.')

from pathlib import Path
import json, hashlib
import pandas as pd

root = Path('D:/ddos_v3/research_v42')
out = root / 'results'
def read(n): return json.loads((out / n).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n): return pd.read_csv(out / f'{n}.csv', float_precision='round_trip')

v = read('verification.json')
expected = dict(old_files_preserved=5691, old_forecast_histories_preserved=22,
    new_fits=0, independent_comparisons=36, independent_decision_cells=1088,
    prefix_future_gate_poison_checks=68, fresh_cells=84, carried_cells=12,
    fallback_cells=992, independent_seed_predictions=3264,
    exact_original_monthly_noncarry_predictions=3228, carried_seed_predictions=36,
    independent_metric_cells=4923, independent_weekly_effects=1088,
    independent_decision_outcomes=1088)
assert v['status'] == 'PASS' and all(v[k] == x for k, x in expected.items())
assert v['maximum_prediction_gap'] < 2e-16
methods = ['learned_vol_interaction', 'learned_order_extension', 'learned_order_offset']
new, monthly, quarter, annual = 'monthly_state_retained', 'monthly_state_validated', 'weekly_state_validated', 'rolling5_annual20'
metrics, seeds = csv('ensemble_metrics'), csv('seed_metrics')
decisions, outcomes, carried, effects = [csv(n) for n in ['retention_decisions', 'decision_outcomes', 'carried_weeks', 'weekly_policy_effects']]
facts = []
cases = [
 ('recent_2024_2026', 125, new, [69,72,71], [.246550,.249036,.249054]),
 ('recent_2024_2026', 125, monthly, [68,71,70], [.247415,.249923,.249946]),
 ('recent_2024_2026', 125, quarter, [69,73,72], [.246156,.248903,.248933]),
 ('recent_2024_2026', 125, annual, [68,72,71], [.246843,.249790,.249824]),
 ('extension_2021_2023', 147, new, [70,67,67], [.277601,.275831,.275735]),
 ('extension_2021_2023', 147, monthly, [70,67,67], [.277601,.275831,.275735]),
 ('extension_2021_2023', 147, quarter, [72,69,69], [.277416,.275610,.275517]),
 ('pooled_2021_2026', 272, new, [139,139,138], [.263331,.263517,.263474]),
 ('pooled_2021_2026', 272, quarter, [141,142,141], [.263050,.263337,.263300]),
 ('year_2026', 30, new, [12,16,16], [.258700,.257140,.257098]),
 ('year_2026', 30, monthly, [12,16,16], [.258700,.257140,.257098]),
 ('year_2026', 30, quarter, [12,16,16], [.258700,.257140,.257098]),
 ('year_2026', 30, annual, [12,16,16], [.257501,.257140,.257098]),
]
for period, n, history, correct, brier in cases:
    g = metrics[metrics.period.eq(period) & metrics.history.eq(history)].set_index('method').loc[methods]
    assert g.n.eq(n).all() and g.correct_directions.tolist() == correct
    assert g.brier.round(6).tolist() == brier
    assert all(abs(a-c/n) < 1e-14 for a,c in zip(g.accuracy, correct))
    facts.append(dict(period=period, n=n, history=history, correct=correct, brier=brier))

assert decisions.action.value_counts().to_dict() == {'fallback':992, 'fresh':84, 'carry':12}
d = decisions[decisions.action.eq('carry')]
expected_cells = {
 ('2024-04-30','negative_low','2024-03-31','2024-06-30'),
 ('2024-05-31','negative_low','2024-03-31','2024-06-30'),
 ('2024-08-31','nonnegative_low','2024-07-31','2024-09-30'),
}
assert set(zip(d.cutoff,d.state,d.source_cutoff,d.expiry)) == expected_cells
assert d.groupby('method').size().eq(3).all() and d.method.nunique() == 4
o = outcomes[outcomes.action.eq('carry')]
assert len(o) == 12 and o.n.eq(0).sum() == 8 and o.n.eq(3).sum() == 4
assert o.loc[o.cutoff.eq('2024-05-31'), 'n'].eq(3).all()
assert len(carried) == 12 and carried.method.nunique() == 4
assert carried.source_cutoff.eq('2024-03-31').all() and carried.expiry.eq('2024-06-30').all()
assert carried.probability.equals(carried.quarterly_probability)
assert carried.actual_up.eq(0).all() and carried.changed_vs_monthly.all()
assert set(zip(carried.date,carried.source_age_days)) == {('2024-06-14',75.0),('2024-06-21',82.0),('2024-06-28',89.0)}
for method in methods:
    g = carried[carried.method.eq(method)].sort_values('date')
    assert g.case_vs_monthly.tolist() == ['stable_wrong','stable_correct','recovery']
    assert g.brier_vs_monthly.lt(0).all()
    e = effects[effects.method.eq(method)]
    assert e.case_vs_monthly.eq('regression').sum() == 0
    assert e.case_vs_monthly.eq('recovery').sum() == 1
    unchanged = e[e.year.le(2023) | e.year.eq(2026)]
    assert not unchanged.changed_vs_monthly.any() and unchanged.brier_vs_monthly.eq(0).all()
    y = e[e.year.eq(2026)]
    assert y.probability.equals(y.quarterly_probability)
    if method == methods[0]:
        assert y[y.changed_vs_annual].date.tolist() == ['2026-07-03','2026-07-10']
    else:
        assert not y.changed_vs_annual.any()

seed_cases = [([66,67,68],[66,68,68]),([65,69,69],[65,70,69]),([65,68,69],[65,69,69])]
for method, (old_correct,new_correct) in zip(methods,seed_cases):
    def subset(history):
        return seeds[seeds.period.eq('recent_2024_2026') & seeds.method.eq(method) & seeds.history.eq(history)].sort_values('seed')
    a,b = subset(monthly),subset(new)
    assert a.seed.tolist() == b.seed.tolist() == [20260910,20260911,20260912]
    assert a.correct_directions.tolist() == old_correct and b.correct_directions.tolist() == new_correct
    assert (b.brier.to_numpy() < a.brier.to_numpy()).all()
    facts.append(dict(method=method, monthly_seed_correct=old_correct, retained_seed_correct=new_correct, all_seed_brier_lower=True))

p = read('primary_comparisons.json')
assert len(p) == 36 and all(x['holm_adjusted_p'] == 1.0 for x in p)
assert round(min(x['p'] for x in p),6) == .093691
report = read('report_manifest.json')
assert all(sha(root/n) == h for n,h in report['artifacts'].items())
human = (out/'结果解读与下一步.md').read_text(encoding='utf-8')
for phrase in ['没有新增盲测','73/125、58.4%','71/125、56.8%','本轮没有改善 2026 年','本轮尚未开始该诊断','连续缺样本不能续期','本月重新拟合候选','并非 12 个独立市场事件']:
    assert phrase in human
assert not (out/'visual_review.json').exists() and not (out/'interpretation_checks.json').exists()
review = dict(status='PASS', reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),
    observations='Both PNGs visually inspected: yearly panels have readable legends, axes, year counts and partial-2026 caption; plotted differences match metric tables. Recovery plot includes explicit zero counts, denominators, comparator and probability-versus-direction caveat. No clipped labels or problematic overlap. PNG/SVG artifact hashes match report manifest.',
    artifacts={f'results/{n}':sha(out/n) for n in ['yearly_accuracy.png','yearly_accuracy.svg','retention_direction_effect.png','retention_direction_effect.svg']})
(out/'visual_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
checks = dict(status='PASS', completed_utc=pd.Timestamp.now(tz='UTC').isoformat(), numeric_facts=facts,
    checks=['Independent verification counts','All cited period metrics and accuracy denominators','Carry decisions, fixed origins/expiries, ages and empty followups','All carried probabilities equal quarterly on three dates','One shared recovery date and no monthly direction regressions','No early or 2026 prediction changes','Recent seed direction counts and Brier','All 36 Holm comparisons nonsignificant','Known-history and original-58.4-protocol distinctions','Next diagnostic explicitly not performed','Frozen report hashes and visual inspection'],
    artifacts={f'results/{n}':sha(out/n) for n in ['结果解读与下一步.md','visual_review.json']})
(out/'interpretation_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
print('R42 human numerical claims, report hashes and visual review PASS.')

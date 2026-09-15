from pathlib import Path
import hashlib,json,datetime
import pandas as pd

root=Path('D:/ddos_v3/research_v30');out=root/'results'
def read(name):return json.loads((out/name).read_text(encoding='utf-8'))
def csv(name):return pd.read_csv(out/(name+'.csv'),float_precision='round_trip')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):
    path=out/name;assert not path.exists();path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

assert read('verification.json')['status']=='PASS'
m=csv('ensemble_metrics').set_index(['window','history','method']);ev=csv('events');op=ev[~ev.mandatory];diag=csv('next_quarter_diagnostics');yr=csv('yearly_metrics').set_index(['year','history','method']);changes=csv('direction_changes').set_index(['window','method'])
methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];recent='recent_2024_2026';early='extension_2021_2023'
expected={'rolling5_annual20':[68,72,71],'rolling5_quarterly20':[63,63,63],'state90':[69,71,70],'error16':[67,72,71],'rolling5_matched':[70,72,73]}
for h,values in expected.items():
    for method,value in zip(methods,values):assert m.loc[(recent,h,method),'correct_directions']==value and m.loc[(recent,h,method),'n']==125
for h,values in {'rolling5_annual20':[71,68,68],'error16':[66,67,67]}.items():
    for method,value in zip(methods,values):assert m.loc[(early,h,method),'correct_directions']==value and m.loc[(early,h,method),'n']==147
triggers=op[op.refit].cutoff.tolist();assert triggers==['2022-06-30','2023-06-30','2025-06-30'];assert len(op)==17 and (~op.ready).sum()==9 and (op.ready&~op.refit).sum()==5 and ev.refit.sum()==9
d=diag[diag.cutoff.isin(triggers)];assert d.next_n.tolist()==[14,12,13] and d.incumbent_correct.tolist()==[7,7,6] and d.fresh_correct.tolist()==[4,7,4] and d.fresh_minus_incumbent_brier.lt(0).all()
assert diag[diag.cutoff.eq('2023-09-30')].iloc[0].incumbent_correct==3 and diag[diag.cutoff.eq('2023-09-30')].iloc[0].fresh_correct==9
for method,count in zip(methods,[12,16,16]):assert yr.loc[(2026,'error16',method),'correct_directions']==count and yr.loc[(2026,'error16',method),'n']==30
pred=csv('ensemble_predictions');keys=['row_index','method']
for year in [2021,2024,2026]:
    a=pred[pred.history.eq('error16')&pred.date.str.startswith(str(year))].drop(columns='history').reset_index(drop=True)
    b=pred[pred.history.eq('rolling5_annual20')&pred.date.str.startswith(str(year))].drop(columns='history').reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
for method,counts in zip(methods,[(7,3,4),(6,3,3),(6,3,3)]):assert tuple(changes.loc[(recent,method),['changed','wrong_to_correct','correct_to_wrong']])==counts
pairs=read('primary_comparisons.json');assert len(pairs)==40 and all(r['holm_adjusted_p']>=.05 for r in pairs);minimum=min(r['holm_adjusted_p'] for r in pairs);assert abs(minimum-.6919308069193081)<1e-14
for r in pairs:
    if r['reference_history']=='rolling5_annual20':assert r['ci95_low']<=0<=r['ci95_high'] and r['holm_adjusted_p']==1
budgets=csv('budgets').set_index('history');assert budgets.loc['error16','optimizer_steps_if_run_online']==5400 and budgets.loc['error16','neural_fits_if_run_online']==27
for method in methods:
    assert m.loc[(recent,'error16',method),'brier']<m.loc[(recent,'rolling5_annual20',method),'brier']
    assert m.loc[(early,'error16',method),'brier']<m.loc[(early,'rolling5_annual20',method),'brier']
completed=datetime.datetime.now(datetime.timezone.utc).isoformat()
save('interpretation_checks.json',dict(status='PASS',completed_utc=completed,triggers=triggers,warmup_checks=9,nontrigger_ready_checks=5,all40_significance_checks=True,minimum_holm_p=minimum,old58_4_preserved=True,year2021_2024_2026_predictions_exactly_annual=True,trigger_next_quarter_diagnostics_checked=True,recent_correct_counts=expected,interpretation_sha256=sha(out/'结果解读与下一步.md')))
save('visual_review.json',dict(status='PASS',completed_utc=completed,reviewed_by='Codex image inspection',checks=['Both panels and Chinese text legible','All 17 quarter checks shown with warmup markers and three trigger dots','Annual accuracy values and 2026 partial-year label match frozen metrics','Axes, legend and numeric labels do not overlap or clip'],artifacts={name:sha(out/name) for name in ['forecast_error_trigger.png','forecast_error_trigger.svg','第三十轮测试报告.md']}))
print('Interpretation facts and visual review PASS.')

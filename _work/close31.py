from pathlib import Path
import json,hashlib,datetime
import pandas as pd
root=Path('D:/ddos_v3/research_v31');out=root/'results'
def csv(n):return pd.read_csv(out/(n+'.csv'),float_precision='round_trip')
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):
    p=out/n;assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
assert read('verification.json')['status']=='PASS'
m=csv('ensemble_metrics').set_index(['window','history','method']);events=csv('events');methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];recent='recent_2024_2026';early='extension_2021_2023'
recent_counts={'rolling5_annual20':[68,72,71],'shadow_trial':[68,72,71],'annual_quarter_half':[65,65,65],'rolling5_matched':[70,72,73]};early_counts={'rolling5_annual20':[71,68,68],'shadow_trial':[68,66,66],'annual_quarter_half':[72,72,72]}
for window,counts,n in [(recent,recent_counts,125),(early,early_counts,147)]:
    for h,values in counts.items():
        for method,value in zip(methods,values):assert m.loc[(window,h,method),'correct_directions']==value and m.loc[(window,h,method),'n']==n
for method in methods:
    assert m.loc[(recent,'annual_quarter_half',method),'brier']>m.loc[(recent,'rolling5_annual20',method),'brier'];assert m.loc[(early,'annual_quarter_half',method),'brier']<m.loc[(early,'rolling5_annual20',method),'brier']
assert m.loc[(recent,'annual_quarter_half','native_mse'),'correct_directions']==70 and m.loc[(recent,'rolling5_annual20','native_mse'),'correct_directions']==66
promoted=events[events.promote];assert promoted.cutoff.tolist()==['2021-09-30','2023-06-30','2023-09-30'] and promoted.selected_model_cutoff.tolist()==['2021-06-30','2023-03-31','2023-06-30'];assert events.ready.sum()==11 and events.reason.eq('same_model').sum()==6 and events.selected_model_cutoff.nunique()==9
pred=csv('ensemble_predictions');a=pred[pred.history.eq('shadow_trial')&pred.date.ge('2024-01-01')].drop(columns='history').reset_index(drop=True);b=pred[pred.history.eq('rolling5_annual20')&pred.date.ge('2024-01-01')].drop(columns='history').reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_exact=True)
y=csv('yearly_metrics').set_index(['year','history','method'])
for h,values in {'rolling5_annual20':[12,16,16],'shadow_trial':[12,16,16],'annual_quarter_half':[10,12,12]}.items():
    for method,value in zip(methods,values):assert y.loc[(2026,h,method),'correct_directions']==value and y.loc[(2026,h,method),'n']==30
d=csv('next_quarter_diagnostics');d=d[d.promoted];assert d.next_n.tolist()==[13,12,12] and d.incumbent_correct.tolist()==[6,7,7] and d.challenger_correct.tolist()==[8,4,3];assert (d.challenger_minus_incumbent_brier<0).tolist()==[True,False,False]
changes=csv('direction_changes').set_index(['window','history','method'])
for method,values in zip(methods,[(15,6,9),(19,6,13),(20,7,13)]):assert tuple(changes.loc[(recent,'annual_quarter_half',method),['changed','wrong_to_correct','correct_to_wrong']])==values
pairs=read('primary_comparisons.json');assert len(pairs)==68 and all(r['holm_adjusted_p']>=.05 for r in pairs);minimum=min(r['holm_adjusted_p'] for r in pairs);assert abs(minimum-.9723027697230276)<1e-14
budgets=csv('budgets').set_index('history')
for h in ['shadow_trial','annual_quarter_half']:assert budgets.loc[h,'optimizer_steps_if_run_online']==13800 and budgets.loc[h,'neural_fits_if_run_online']==69 and budgets.loc[h,'head_fits_if_run_online']==276
completed=datetime.datetime.now(datetime.timezone.utc).isoformat();save('interpretation_checks.json',dict(status='PASS',completed_utc=completed,recent_correct_counts=recent_counts,early_correct_counts=early_counts,promotion_dates=promoted.cutoff.tolist(),recent_shadow_exact_annual=True,year2026_counts_checked=True,following_quarter_diagnostics_checked=True,all68_significance_checks=True,minimum_holm_p=minimum,old58_4_preserved=True,interpretation_sha256=sha(out/'结果解读与下一步.md')))
save('visual_review.json',dict(status='PASS',completed_utc=completed,reviewed_by='Codex image inspection after layout revision',initial_review='Legend and footnote overlapped; original artifacts preserved in results/visual_attempt1.',final_checks=['Chinese text and all three method panels legible','Axes include all observed accuracy values','Annual and shadow overlaps reflect identical observations','Legend and footnote now occupy separate rows without overlap','Partial 2026 sample and all6years labelled'],artifacts={n:sha(out/n) for n in ['shadow_and_blend_yearly.png','shadow_and_blend_yearly.svg','第三十一轮测试报告.md','visual_revision.json']}))
print('Interpretation facts and final visual review PASS.')

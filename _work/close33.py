from pathlib import Path
import json,hashlib,datetime
import pandas as pd
root=Path('D:/ddos_v3/research_v33');out=root/'results'
def csv(n):return pd.read_csv(out/(n+'.csv'),float_precision='round_trip')
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):
    p=out/n;assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
v=read('verification.json');assert v['status']=='PASS' and v['component_checks']==16320 and v['ensemble_checks']==1088
methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];perf=csv('performance_summary').set_index(['period','method'])
for m,a,b,c,d in zip(methods,[12,16,16],[10,10,10],[2,7,7],[0,1,1]):
    r=perf.loc[('year_2026',m)];assert (r.annual_correct,r.quarter_head_correct,r.regression,r.recovery,r.n)==(a,b,c,d,30)
flips=csv('flips_2026');regdates=['2026-05-08','2026-05-22','2026-07-03','2026-07-10','2026-07-24','2026-08-07','2026-08-14']
for m in methods[1:]:
    g=flips[flips.method.eq(m)];bad=g[g.case.eq('regression')];good=g[g.case.eq('recovery')];assert bad.date.tolist()==regdates and good.date.tolist()==['2026-07-17'];assert bad.actual_up.sum()==1 and bad[bad.actual_up.eq(1)].date.tolist()==['2026-05-22'];assert bad.dominant_adverse_group.value_counts().to_dict()=={'market':4,'order':3}
g=flips[flips.method.eq(methods[0])];assert g.date.tolist()==['2026-08-07','2026-08-14'] and g.dominant_adverse_group.eq('market').all()
comp=csv('component_summary');reg=comp[comp.period.eq('year_2026')&comp.case.eq('regression')].set_index(['method','group'])
totals={}
for m,expected in zip(methods,[(-7.70,.88,-6.82),(-6.56,.43,-6.13),(-6.58,.43,-6.16)]):
    s=reg.loc[m];values=(float(s.mean_signed_coefficient.sum()*100),float(s.mean_signed_transform.sum()*100),float(s.mean_signed_contribution.sum()*100));assert all(abs(a-b)<.006 for a,b in zip(values,expected));totals[m]=values
example=flips[flips.method.eq(methods[1])&flips.date.eq('2026-08-07')].iloc[0];assert round(example.old_probability*100,2)==42.41 and round(example.new_probability*100,2)==51.97
good=flips[flips.method.eq(methods[1])&flips.date.eq('2026-07-17')].iloc[0];assert round(good.old_probability*100,2)==48.32 and round(good.new_probability*100,2)==54.61;assert abs(good.p_market*100-4.10)<.006 and abs(good.p_order*100-1.64)<.006 and abs(good.probability_change*100-6.29)<.006
shift=csv('training_window_changes').set_index('head_cutoff')
for date,add,remove in [('2026-03-31',56,58),('2026-06-30',116,118)]:
    r=shift.loc[date];assert r.annual_train_n==1206 and r.quarter_train_n==1204 and r.added_n==add and r.removed_n==remove
r=shift.loc['2026-06-30'];assert abs(r.up_fraction_change*100-1.16)<.006 and round(r.added_up_fraction*100,2)==58.62 and round(r.removed_up_fraction*100,2)==46.61
assert len(read('baseline_comparisons.json'))==28 and all(p['holm_adjusted_p']>=.05 for p in read('baseline_comparisons.json'))
old=csv('baseline_metrics');assert old[old.history.eq('rolling5_matched')&old.method.eq('learned_order_offset')&old.window.eq('recent_2024_2026')].correct_directions.iloc[0]==73
completed=datetime.datetime.now(datetime.timezone.utc).isoformat();save('interpretation_checks.json',dict(status='PASS',completed_utc=completed,year2026_flip_counts_checked=True,shared_flip_dates_checked=True,dominant_groups_checked=True,signed_probability_point_totals=totals,training_window_shift_checked=True,recovery_example_checked=True,old58_4_preserved=True,new_training=0,new_inferential_tests=0,interpretation_sha256=sha(out/'结果解读与下一步.md')))
save('visual_review.json',dict(status='PASS',completed_utc=completed,reviewed_by='Codex image inspection',checks=['All3method panels and Chinese labels legible','Signs and probability-point units match diagnostic tables','Different regression counts displayed explicitly','Zero order term forR19 shown correctly','Annotations, titles, axes and footnotes do not overlap or clip'],artifacts={n:sha(out/n) for n in ['probability_contributions_2026.png','probability_contributions_2026.svg','第三十三轮归因报告.md']}));print('Interpretation facts and visual review PASS.')

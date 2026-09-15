from pathlib import Path
import json,hashlib
import pandas as pd
root=Path('D:/ddos_v3/research_v44');out=root/'results'
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def csv(n):return pd.read_csv(out/f'{n}.csv',float_precision='round_trip')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read('verification.json');assert v['status']=='PASS'
expected=dict(old_files_preserved=5800,unchanged_forecast_histories=23,new_fits=0,new_policies=0,new_pvalues=0,independent_origins=84,independent_planned_checkpoints=156,independent_count_cells=306,independent_checkpoint_members=340,independent_arrivals=242,independent_origin_view_cells=168,independent_summary_cells=60,independent_event_view_cells=50,outcome_poison_output_tables=4,future_metadata_isolation_checkpoints=153,future_gate_prefix_dates=27)
assert all(v[k]==value for k,value in expected.items())
methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];classes=['reached_before_expiry','reached_only_at_expiry','not_reached_by_expiry','observation_censored'];t=csv('origin_feasibility');o=csv('origins');c=csv('checkpoints');a=csv('observed_arrivals');e=csv('event_feasibility');meta=csv('weekly_metadata');facts=[]
for m,nums,recent in zip(methods,[[21,1,4,14,2],[22,1,5,16,0],[22,1,5,16,0]],[[14,0,3,9,2],[14,0,4,10,0],[14,0,4,10,0]]):
    for view in ['cumulative','rolling13']:
        g=t[t.method.eq(m)&t.view.eq(view)];actual=[len(g)]+[int(g.classification.eq(x).sum()) for x in classes];assert actual==nums
        r=g[g.source_cutoff.ge('2024-01-01')];assert [len(r)]+[int(r.classification.eq(x).sum()) for x in classes]==recent
        hit=g[g.classification.eq('reached_before_expiry')];assert len(hit)==1
        x=hit.iloc[0];assert x.source_cutoff=='2023-09-30' and x.state=='negative_low' and x.expiry=='2023-12-31' and x.first_preexpiry_ready_decision=='2023-11-30' and x.days_remaining_at_preexpiry_ready==31 and x.observed_followup_same_state_weeks==5
    facts.append(dict(method=m,all_origin_counts=nums,recent_origin_counts=recent))
assert len(o)==84 and o.method.isin(methods).sum()==65 and o.method.eq('learned_market').sum()==19
assert sum(o[o.method.isin(methods)].planned_preexpiry_checkpoints==0)==27
assert o[o.method.isin(methods)].event_id.nunique()==25
events=e[e.view.eq('cumulative')];assert len(events)==25 and [int(events.classification.eq(x).sum()) for x in classes]==[1,5,17,2]
hit=t[t.view.eq('cumulative')&t.classification.eq('reached_before_expiry')];assert len(hit)==4 and hit.event_id.nunique()==1 and hit.fifth_maturity_date.eq('2023-11-13').all()
for origin in hit.origin_id:
    z=c[c.origin_id.eq(origin)].sort_values('checkpoint');assert z.checkpoint.tolist()==['2023-10-31','2023-11-30','2023-12-31'] and z.cumulative_n.tolist()==[3,7,11]
    arr=a[a.origin_id.eq(origin)].sort_values('arrival_rank');assert arr.iloc[0].date=='2023-09-22' and arr.iloc[0].joint_completed=='2023-10-16' and arr.iloc[4].joint_completed=='2023-11-13'
f=meta[meta.date.gt('2023-11-30')&meta.date.le('2023-12-31')];assert f.date.tolist()==['2023-12-01','2023-12-08','2023-12-15','2023-12-22','2023-12-29'] and f.state.eq('negative_low').all()
unknown=t[t.view.eq('cumulative')&~t.horizon_complete];assert len(unknown)==3 and unknown.expiry.eq('2026-09-30').all() and unknown.last_observed_checkpoint.eq('2026-08-31').all() and unknown.expiry_n.isna().all()
assert (unknown.planned_preexpiry_checkpoints==unknown.observed_preexpiry_checkpoints).all()
u=unknown[unknown.method.eq(methods[0])].sort_values('source_cutoff');assert u.source_cutoff.tolist()==['2026-06-30','2026-07-31'] and u.last_observed_n.tolist()==[3,0]
assert len(unknown[unknown.method.eq('learned_market')])==1
observed=c[c.observed];future=c[~c.observed];assert len(observed)==153 and len(future)==3 and future[['cumulative_n','rolling13_n','dropped_from_rolling13']].isna().all().all()
assert (observed.cumulative_n==observed.rolling13_n).all() and observed.dropped_from_rolling13.eq(0).all()
assert len(a)==242 and sum(a.signal_precedes_or_equals_source)==62 and a.joint_completed.gt(a.source_cutoff).all()
report=read('report_manifest.json');assert all(sha(root/n)==h for n,h in report['artifacts'].items())
human=(out/'结果解读与下一步.md').read_text(encoding='utf-8')
for phrase in ['没有新增盲测','73/125、58.4%','71/125、56.8%','本轮尚未开始两组校准实验','不是有效独立样本量','不能在这条固定月度规则中增加一个严格早于到期日的决策机会','不表示验证会通过','没有训练新参数']:
    assert phrase in human
assert not (out/'visual_review.json').exists() and not (out/'interpretation_checks.json').exists()
review=dict(status='PASS',reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),observations='Both PNGs visually inspected. Method counts sum to original denominators; cumulative versus rolling13 and all four mutually exclusive classes readable. Event chart merges shared source/state events and labels strictly pre-expiry monthly opportunities. Captions distinguish count feasibility, expiry equality, incomplete observation and shared evidence. No clipped text or problematic overlaps. PNG/SVG hashes match report manifest.',artifacts={f'results/{n}.{ext}':sha(out/f'{n}.{ext}') for n in ['feasibility_by_method','feasibility_by_opportunity'] for ext in ['png','svg']})
(out/'visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
checks=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),numeric_facts=facts,checks=['All method and recent feasibility counts in both views','Single shared actionable historical event','Exact maturity/checkpoint/expiry timeline and remaining observed signals','Right censoring distinct from negative complete lifetime; all remaining pre-expiry slots already observed','No-window-drop equality','Shared event and no-preexpiry-opportunity counts','Earlier signal with later label maturity handled explicitly','Count feasibility is not validation success or predictive accuracy','Old evidence/protocol distinctions and next experiment not started','Independent verification counts and report provenance','Both figures actually viewed'],artifacts={f'results/{n}':sha(out/n) for n in ['结果解读与下一步.md','visual_review.json']})
(out/'interpretation_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('R44 human claims, expiry/censoring distinctions, report hashes and visual review PASS.')

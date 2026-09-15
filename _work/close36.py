from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];R=P/'research_v36';O=R/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(O/f'{n}.csv',float_precision='round_trip')
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
c=csv('state_comparisons');d=csv('label_mix_decomposition');s=csv('training_state_summary');m=csv('state_prediction_metrics');w=csv('weekly_state_effects');v=read(O/'verification.json');assert v['status']=='PASS'
states=['negative_low','negative_high','nonnegative_low','nonnegative_high']
for state,an,nn,ar,nr,delta in zip(states,[353,334,250,269],[6,19,55,36],[40.79,54.19,51.60,46.10],[66.67,89.47,40.,69.44],[25.87,35.28,-11.60,23.35]):
    r=c[c.cutoff.eq('2026-06-30')&c.target_role.eq('added')&c.state.eq(state)].iloc[0];assert r.annual_n==an and r.target_n==nn
    for actual,expected in [(r.annual_up_rate*100,ar),(r.target_up_rate*100,nr),(r.up_rate_delta*100,delta)]:assert abs(actual-expected)<.005
    assert r.supported==(nn>=20)
r=c[c.cutoff.eq('2026-03-31')&c.target_role.eq('added')&c.state.eq('nonnegative_low')].iloc[0];assert r.target_n==44
for actual,expected in [(r.target_share*100,78.57),(r.annual_share*100,20.73),(r.target_up_rate*100,34.09),(r.annual_up_rate*100,51.60)]:assert abs(actual-expected)<.005
r=d[d.cutoff.eq('2026-06-30')&d.target_role.eq('added')].iloc[0];assert r.target_n==116 and r.annual_n==1206 and r.supported_states==2
for actual,expected in [(r.target_up_rate*100,58.62),(r.annual_up_rate*100,47.93),(r.overall_delta*100,10.69),(r.composition*100,1.83),(r.within_state*100,8.86)]:assert abs(actual-expected)<.005
for cutoff,expected in [('2024-09-30',-11.65),('2025-09-30',9.57),('2026-06-30',-11.60)]:
    r=c[c.cutoff.eq(cutoff)&c.target_role.eq('added')&c.state.eq('nonnegative_low')].iloc[0];assert abs(r.up_rate_delta*100-expected)<.005
coverage={}
for role,expected in [('annual',[8.96,26.06]),('added',[.86,25.29])]:
    g=s[s.cutoff.eq('2026-06-30')&s.role.eq(role)];vals=[float((g.n*g[k].fillna(0)).sum()/g.n.sum()) for k in ['mean_market5_outside_any','mean_decoded25_outside_any']];coverage[role]=vals
    for value,target in zip(vals,expected):assert abs(value*100-target)<.005
assert abs(coverage['added'][0]*116-1)<1e-12
g=m[m.period.eq('year_2026')&m.method.eq('learned_order_extension')&m.arm.eq('annual')].set_index('state');assert g.loc[states,'n'].tolist()==[2,10,9,9] and int(g.supported.sum())==1
g=w[w.year.eq(2026)&w.method.eq('learned_order_extension')&w['case'].eq('regression')];assert len(g)==7 and g.groupby('state').size().to_dict()=={'negative_high':3,'nonnegative_high':3,'nonnegative_low':1}
g=g[g.state.eq('negative_high')];assert g.date.tolist()==['2026-07-24','2026-08-07','2026-08-14'] and g.added_state_n.eq(19).all()
old=csv('baseline_metrics');g=old[old.window.eq('recent_2024_2026')&old.history.eq('rolling5_matched')&old.method.eq('learned_order_offset')].iloc[0];assert g.correct_directions==73 and g.n==125
assert v['new_predictions']==0 and v['new_head_fits']==0 and v['new_neural_fits']==0 and v['new_inferential_tests']==0 and v['old_files_preserved']==5250
# Keep the exact layout-only helper with the delivered presentation provenance.
p=O/'layout_render_helper.py.txt';p.write_bytes((P/'_work/review36layout.py').read_bytes());a=read(O/'layout_adjustment.json');assert sha(p)==a['render_helper_sha256']
manifest=read(O/'report_manifest.json');manifest['artifacts'][str(p.relative_to(R))]=sha(p);save('report_manifest.json',manifest)
initial=read(O/'report_manifest_initial.json')
for path,record in a['initial_artifacts'].items():assert sha(R/record['archive'])==record['sha256']==initial['artifacts'][path]
for phase in ['preparation','calibration','diagnosis','report']:
    for n,digest in read(O/f'{phase}_manifest.json')['artifacts'].items():assert sha(R/n)==digest
save('interpretation_checks.json',dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),interpretation_sha256=sha(O/'结果解读与下一步.md'),report_sha256=sha(O/'第三十六轮诊断报告.md'),annual_counts=[353,334,250,269],added_counts=[6,19,55,36],weekly_state_counts=[2,10,9,9],label_mix_and_coverage_claims_verified=True,coverage=coverage,no_new_predictions_or_training=True,old584_preserved=True))
save('visual_review.json',dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),review='Viewed heatmap and final forecast chart. Chinese labels, per-cell counts, sparse markers, legends and colorbar readable. Corrected right margin of forecast chart from1% to5% after original rightmost label touched boundary, then viewed again; final labels fully visible. Initial plot and manifest plus exact render helper archived. Values and source metrics unchanged.',files={n:sha(O/n) for n in ['state_label_shift.png','state_label_shift.svg','state_forecast_2026.png','state_forecast_2026.svg']},layout_adjustment_sha256=sha(O/'layout_adjustment.json')))
print('Interpretation numeric facts, presentation provenance and visual review PASS.')

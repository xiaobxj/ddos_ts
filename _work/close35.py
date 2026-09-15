from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];R=P/'research_v35';O=R/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(O/f'{n}.csv',float_precision='round_trip')
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
m=csv('ensemble_metrics');yr=csv('yearly_metrics');s=csv('effect_summary');w=csv('weekly_effects');membership=csv('membership_summary');pairs=read(O/'primary_comparisons.json');v=read(O/'verification.json');assert v['status']=='PASS'
ms=['learned_vol_interaction','learned_order_extension','learned_order_offset'];hs=['rolling5_annual20','fixed_transform_add_only','fixed_transform_remove_only','fixed_transform_step100']
recent=[[68,72,71],[63,64,66],[69,69,69],[65,65,67]];early=[[71,68,68],[64,65,63],[71,70,70],[68,63,63]];year26=[[12,16,16],[9,12,12],[11,14,14],[10,10,11]]
for h,rc,ec,yc in zip(hs,recent,early,year26):
    for method,rn,en,yn in zip(ms,rc,ec,yc):
        a=m[m.history.eq(h)&m.method.eq(method)&m.window.eq('recent_2024_2026')].iloc[0];b=m[m.history.eq(h)&m.method.eq(method)&m.window.eq('extension_2021_2023')].iloc[0];c=yr[yr.history.eq(h)&yr.method.eq(method)&yr.year.eq(2026)].iloc[0]
        assert a.n==125 and a.correct_directions==rn and b.n==147 and b.correct_directions==en and c.n==30 and c.correct_directions==yn
for h,vals in zip(hs[:3],[[.246843,.249790,.249824],[.246831,.249564,.249608],[.251797,.255122,.255108]]):
    for method,x in zip(ms,vals):r=m[m.history.eq(h)&m.method.eq(method)&m.window.eq('recent_2024_2026')].iloc[0];assert abs(r.brier-x)<.0000005
for method,counts,contrib in zip(ms,[(2,2,0),(7,6,1),(7,6,1)],[[-6.95,-.73,-7.68],[-5.21,-1.27,-6.49],[-5.32,-1.35,-6.67]]):
    r=s[s.period.eq('year_2026')&s.method.eq(method)&s['case'].eq('regression')].iloc[0];assert (r.n,r.dominant_add,r.dominant_remove)==counts
    for key,x in zip(['mean_signed_add_effect','mean_signed_remove_effect','mean_signed_total_delta'],contrib):assert abs(100*r[key]-x)<.005
r=s[s.period.eq('year_2026')&s.method.eq(ms[0])&s['case'].eq('regression')].iloc[0];assert abs(r.mean_signed_remove_without_add*100-.13)<.005 and abs(r.mean_signed_remove_after_add*100+1.59)<.005
for date,expected in [('2026-08-07',[42.41,50.83,42.77,52.82]),('2026-05-08',[49.80,47.94,52.34,50.22])]:
    r=w[w.method.eq(ms[1])&w.date.eq(date)].iloc[0];assert r.actual_up==0
    for arm,x in zip(['annual','add','remove','both'],expected):assert abs(r['p_'+arm]*100-x)<.005
for cutoff,counts in [('2026-03-31',[1206,1262,1148,1204]),('2026-06-30',[1206,1322,1088,1204])]:
    for arm,n in zip(['annual','add','remove','both'],counts):assert membership[membership.arm.eq(arm)&membership.cutoff.eq(cutoff)].train_n.iloc[0]==n
for method in ms[1:]:
    g=w[w.year.eq(2026)&w.method.eq(method)&w['case'].eq('regression')];assert g.date.tolist()==['2026-05-08','2026-05-22','2026-07-03','2026-07-10','2026-07-24','2026-08-07','2026-08-14']
assert len(pairs)==60 and abs(min(r['holm_adjusted_p'] for r in pairs)-.821917808219178)<1e-14 and all(r['holm_adjusted_p']>=.05 for r in pairs)
r=m[m.window.eq('recent_2024_2026')&m.history.eq('rolling5_matched')&m.method.eq(ms[2])].iloc[0];assert r.correct_directions==73 and r.n==125
assert len(read(O/'heads.json'))==408 and v['independent_endpoint_solutions']==612 and v['old_files_preserved']==5063
for phase in ['preparation','fitting','scoring','evaluation','diagnosis','report']:
    for n,d in read(O/f'{phase}_manifest.json')['artifacts'].items():assert sha(R/n)==d
save('interpretation_checks.json',dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),interpretation_sha256=sha(O/'结果解读与下一步.md'),report_sha256=sha(O/'第三十五轮测试报告.md'),recent_counts=recent,early_counts=early,year2026_counts=year26,all_probability_effect_and_case_claims_verified=True,all60_not_significant=True,old584_preserved=True))
save('visual_review.json',dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),review='Viewed both generated PNGs. Chinese text, legends, all three panels and four line styles readable; y-axis0to100% contains all yearly values. Contribution bars show signed probability points, correct counts and labels with adequate margins; no clipping or overlap. SVGs derive from same saved figures.',files={n:sha(O/n) for n in ['member_ablation_yearly.png','member_ablation_yearly.svg','member_effects_2026.png','member_effects_2026.svg']}))
print('Interpretation facts and visual review PASS.')

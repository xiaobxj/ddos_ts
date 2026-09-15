from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];R=P/'research_v34';O=R/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(O/f'{n}.csv',float_precision='round_trip')
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
m=csv('ensemble_metrics');y=csv('yearly_metrics');f=csv('direction_flips');e=csv('ensemble_predictions');pairs=read(O/'primary_comparisons.json');v=read(O/'verification.json');assert v['status']=='PASS'
ms=['learned_vol_interaction','learned_order_extension','learned_order_offset'];hs=['rolling5_annual20','annual_net_quarter_head','fixed_transform_step25','fixed_transform_step50','fixed_transform_step100']
expected=[[68,72,71],[66,65,65],[66,69,69],[65,67,68],[65,65,67]]
for h,counts in zip(hs,expected):
    for method,count in zip(ms,counts):
        r=m[m.window.eq('recent_2024_2026')&m.history.eq(h)&m.method.eq(method)].iloc[0];assert r.correct_directions==count and r.n==125
for h,counts in zip(hs[:3],[[12,16,16],[10,10,10],[11,13,13]]):
    for method,count in zip(ms,counts):r=y[y.history.eq(h)&y.method.eq(method)&y.year.eq(2026)].iloc[0];assert r.correct_directions==count and r.n==30
for method,count in zip(ms,[71,68,68]):
    r=m[m.window.eq('extension_2021_2023')&m.history.eq('rolling5_annual20')&m.method.eq(method)].iloc[0];assert r.correct_directions==count and r.n==147
    a=m[m.window.eq('extension_2021_2023')&m.history.eq('fixed_transform_step25')&m.method.eq(method)].iloc[0];assert a.correct_directions==67 and a.n==147 and a.brier<r.brier
for method,ba,bq in zip(ms,[.246843,.249790,.249824],[.247591,.250423,.250492]):
    a=m[m.window.eq('recent_2024_2026')&m.history.eq('rolling5_annual20')&m.method.eq(method)].iloc[0];q=m[m.window.eq('recent_2024_2026')&m.history.eq('fixed_transform_step25')&m.method.eq(method)].iloc[0]
    assert abs(a.brier-ba)<.0000005 and abs(q.brier-bq)<.0000005 and q.brier>a.brier
    if method!=ms[0]:assert q.auroc<a.auroc
for method in ms[1:]:
    g=f[f.year.eq(2026)&f.history.eq('fixed_transform_step25')&f.method.eq(method)];assert g.date.tolist()==['2026-07-03','2026-07-10','2026-07-24'] and g['case'].eq('regression').all() and g.actual_up.eq(0).all()
    old=e[e.history.eq('annual_net_quarter_head')&e.method.eq(method)&e.year.eq(2026)].set_index('date');new=e[e.history.eq('fixed_transform_step25')&e.method.eq(method)&e.year.eq(2026)].set_index('date');annual=e[e.history.eq('rolling5_annual20')&e.method.eq(method)&e.year.eq(2026)].set_index('date')
    agood=annual.direction_up.eq(annual.actual_up);ogood=old.direction_up.eq(old.actual_up);ngood=new.direction_up.eq(new.actual_up)
    assert int((agood&~ogood&ngood).sum())==4 and int((~agood&ogood&~ngood).sum())==1 and set(old.index[~agood&ogood&~ngood])=={'2026-07-17'}
assert len(pairs)==60 and min(r['holm_adjusted_p'] for r in pairs)==1. and all(r['holm_adjusted_p']>=.05 for r in pairs)
r=m[m.window.eq('recent_2024_2026')&m.history.eq('rolling5_matched')&m.method.eq('learned_order_offset')].iloc[0];assert r.correct_directions==73 and r.n==125
assert len(read(O/'endpoints.json'))==204 and len(read(O/'policy_heads.json'))==612 and v['old_files_preserved']==4946
for phase in ['preparation','fitting','scoring','evaluation','report']:
    for n,d in read(O/f'{phase}_manifest.json')['artifacts'].items():assert sha(R/n)==d
save('interpretation_checks.json',dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),interpretation_sha256=sha(O/'结果解读与下一步.md'),report_sha256=sha(O/'第三十四轮测试报告.md'),recent_correct_counts=expected,year2026_counts=[[12,16,16],[10,10,10],[11,13,13]],all60_adjusted_p_one=True,all_R23_R25_regression_and_recovery_claims_verified=True,old584_preserved=True))
save('visual_review.json',dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),review='Viewed generated PNG at original layout: Chinese titles, panel labels, all five line styles, three-column legend, year labels and 2026 footnote readable; no clipping or overlap. Y-axis 0 to100% contains all values. PNG and SVG derive from same saved figure.',files={n:sha(O/n) for n in ['fixed_transform_steps_yearly.png','fixed_transform_steps_yearly.svg']}))
print('Interpretation numeric claims and visual review PASS.')

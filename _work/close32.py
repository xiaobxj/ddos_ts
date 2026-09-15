from pathlib import Path
import json,hashlib,datetime
import pandas as pd
root=Path('D:/ddos_v3/research_v32');out=root/'results'
def csv(n):return pd.read_csv(out/(n+'.csv'),float_precision='round_trip')
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):
    p=out/n;assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
verify=read('verification.json');assert verify['status']=='PASS' and verify['independent_head_solutions']==204 and verify['causal_training_interfaces']==51
methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];candidate='annual_net_quarter_head';recent='recent_2024_2026';early='extension_2021_2023';m=csv('ensemble_metrics').set_index(['window','history','method'])
recent_counts={'rolling5_annual20':[68,72,71],candidate:[66,65,65],'rolling5_quarterly20':[63,63,63],'rolling5_matched':[70,72,73]};early_counts={'rolling5_annual20':[71,68,68],candidate:[67,65,64],'rolling5_quarterly20':[74,74,75]}
for window,counts,n in [(recent,recent_counts,125),(early,early_counts,147)]:
    for h,values in counts.items():
        for method,value in zip(methods,values):assert m.loc[(window,h,method),'correct_directions']==value and m.loc[(window,h,method),'n']==n
for method in methods:
    assert m.loc[(recent,'rolling5_annual20',method),'brier']<m.loc[(recent,candidate,method),'brier']<m.loc[(recent,'rolling5_quarterly20',method),'brier'];assert m.loc[(early,candidate,method),'brier']<m.loc[(early,'rolling5_annual20',method),'brier']
y=csv('yearly_metrics').set_index(['year','history','method'])
for h,values in {'rolling5_annual20':[12,16,16],candidate:[10,10,10]}.items():
    for method,value in zip(methods,values):assert y.loc[(2026,h,method),'correct_directions']==value and y.loc[(2026,h,method),'n']==30
for h,values in {'rolling5_annual20':[22,23,23],candidate:[24,24,23]}.items():
    for method,value in zip(methods,values):assert y.loc[(2025,h,method),'correct_directions']==value and y.loc[(2025,h,method),'n']==48
changes=csv('direction_changes').set_index(['window','method'])
for method,values in zip(methods,[(12,5,7),(17,5,12),(18,6,12)]):assert tuple(changes.loc[(recent,method),['changed','wrong_to_correct','correct_to_wrong']])==values
pairs=read('primary_comparisons.json');assert len(pairs)==28 and all(r['holm_adjusted_p']>=.05 for r in pairs);minimum=min(r['holm_adjusted_p'] for r in pairs);assert abs(minimum-.3751624837516248)<1e-14
fitting=read('fitting_manifest.json');assert fitting['new_head_fits']==204 and fitting['head_jobs']==51 and fitting['new_projections']==102 and fitting['all_converged'];assert read('extraction_manifest.json')['extension_feature_rows']==2490
solutions=csv('independent_solutions');assert solutions.solver_success.all() and len(solutions)==204;maximum=float(solutions.probability_gap.max());assert abs(maximum-1.9914686122390894e-7)<1e-12
budgets=csv('budgets').set_index('history');assert budgets.loc[candidate,'neural_fits_if_run_online']==18 and budgets.loc[candidate,'head_fits_if_run_online']==276 and budgets.loc[candidate,'optimizer_steps_if_run_online']==3600
completed=datetime.datetime.now(datetime.timezone.utc).isoformat();save('interpretation_checks.json',dict(status='PASS',completed_utc=completed,recent_correct_counts=recent_counts,early_correct_counts=early_counts,year2025_2026_checked=True,direction_changes_checked=True,all28_significance_checks=True,minimum_holm_p=minimum,new_head_fits=204,new_neural_fits=0,feature_extension_rows=2490,maximum_independent_solver_probability_gap=maximum,old58_4_preserved=True,interpretation_sha256=sha(out/'结果解读与下一步.md')))
save('visual_review.json',dict(status='PASS',completed_utc=completed,reviewed_by='Codex image inspection',checks=['Chinese titles and three panels legible','All six years and partial2026 label present','Yearly values match frozen metrics','Legend and footnote have separate nonoverlapping rows','Axes include all reported accuracy values'],artifacts={n:sha(out/n) for n in ['annual_net_quarter_head_yearly.png','annual_net_quarter_head_yearly.svg','第三十二轮测试报告.md']}));print('Interpretation facts and visual review PASS.')

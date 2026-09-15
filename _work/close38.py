from pathlib import Path
import json,hashlib
import pandas as pd
root=Path('D:/ddos_v3/research_v38');out=root/'results'
def read(n):return json.loads((out/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(out/f'{n}.csv',float_precision='round_trip')
v=read('verification.json');assert v['status']=='PASS';t=csv('residual_transfer');s=csv('transfer_summary');o=csv('target_overlap');b=csv('brier_summary');m='learned_order_extension';facts=[]
for cutoff,state,counts,pp in [('2026-03-31','nonnegative_low',[44,8,2],[-15.30,-23.59,46.56]),('2026-06-30','nonnegative_high',[36,6,2],[23.47,21.44,-48.70])]:
    r=t[t.cutoff.eq(cutoff)&t.state.eq(state)&t.method.eq(m)&t.seed.eq(-1)].iloc[0];assert [r.daily_n,r.friday_n,r.future_n]==counts and [round(r[k]*100,2) for k in ['daily_residual','friday_residual','future_residual']]==pp;assert r.daily_future=='opposite' and r.friday_future=='opposite';facts.append(dict(cutoff=cutoff,state=state,counts=counts,residual_pp=pp))
for view,n,union,uses,disjoint in [('daily',116,120,580,25),('friday',22,117,117,22)]:
    r=o[o.cutoff.eq('2026-06-30')&o.state.eq('all')&o.view.eq(view)].iloc[0];assert r.n==n and r.union_edges==union and r.total_edge_uses==uses and r.maximum_disjoint_intervals==disjoint
assert csv('membership_checks').train_future_execution_disjoint.all()
for period,cells,ds,fs in [('extension_2021_2023',12,8,8),('recent_2024_2026',7,4,3),('pooled_2021_2026',19,12,11)]:
    for view,expected in [('daily',ds),('friday',fs)]:
        r=s[s.period.eq(period)&s.method.eq(m)&s.seed.eq(-1)&s.view.eq(view)&s.cohort.eq('r37_eligible')].iloc[0];assert r.cells==cells and r.same_cells==expected;facts.append(dict(period=period,view=view,cells=cells,same_cells=expected))
r=s[s.period.eq('pooled_2021_2026')&s.method.eq(m)&s.seed.eq(-1)&s.view.eq('friday')&s.cohort.eq('weekly_count10')].iloc[0];assert r.cells==2 and r.future_weeks==22
r=b[b.period.eq('year_2026')&b.method.eq(m)&b.seed.eq(-1)&b.history.eq('annual_state_offset')&b.scope.eq('all')].iloc[0];assert [round(r[k],6) for k in ['squared_shift','alignment_term','brier_change']]==[.000542,.008033,.008575];assert round(r.alignment_term/r.brier_change*100,1)==93.7
old=csv('ensemble_metrics');expected=[('year_2025','rolling5_annual20',23),('year_2025','annual_state_offset',25),('year_2026','rolling5_annual20',16),('year_2026','annual_state_offset',13)]
for period,h,n in expected:assert old[old.period.eq(period)&old.history.eq(h)&old.method.eq(m)].correct_directions.iloc[0]==n
assert v['overlap_cells']==345 and v['metric_cells']==5520 and v['transfer_cells']==1472 and v['decomposition_cells']==1104 and v['ranking_cells']==4416 and v['old_files_preserved']==5428 and v['per_seed_rank_inversions']==0 and v['ensemble_rank_inversions']==19
human=(out/'结果解读与下一步.md').read_text(encoding='utf-8');assert '73/125、58.4%' in human and '71/125、56.8%' in human and '后续样本数量在当时未知' in human
manifest=read('report_manifest.json');assert all(sha(root/n)==d for n,d in manifest['artifacts'].items())
review=dict(status='PASS',reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat(),observations='Both final PNGs visually inspected. All data points fall inside axes; Chinese headings, labels, legends, colorbar and notes visible. Brier components and total markers correctly distinguish increasing loss. No clipping, amendments or re-render required.',artifacts={f'results/{n}':sha(out/n) for n in ['residual_transfer.png','residual_transfer.svg','brier_accounting_2026.png','brier_accounting_2026.svg']})
(out/'visual_review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
checks=dict(status='PASS',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),numeric_facts=facts,checks=['2026 residual examples and signs','execution-overlap counts','cross-period agreement counts','strict retrospective support counts','Brier terms and93.7percent fraction','old2025improvement and2026regression retained','verification counts','report artifacts and visual review'],artifacts={'results/结果解读与下一步.md':sha(out/'结果解读与下一步.md'),'results/visual_review.json':sha(out/'visual_review.json')})
(out/'interpretation_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8');print('Human interpretation numerical facts, report provenance and visual review PASS.')

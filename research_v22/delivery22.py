"""Freeze the checked research delivery, preserving every previous artifact."""
from pathlib import Path
import json,hashlib,re,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';PROJECT=ROOT.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def main(freeze):
    prep=read(OUT/'preparation_manifest.json');p=read(ROOT/'protocol.json');v=read(OUT/'verification.json');assert v['status']=='PASS'
    expected=dict(previous_files_preserved=3166,new_primary_fits=60,independent_solutions=60,independent_qr_projections=48,state_split_checks=12,future_prefix_checks=18,new_hmm_fits=0,new_neural_forward_passes=0,neural_training_steps=0,inherited_neural_replay_states=18,
        training_feature_rows=94650,new_probability_forecasts=2610,reused_model_records=8613,model_records=11223,ensemble_records=6786,metric_groups=1520,state_metric_rows=1196,reliability_bins=250,primary_contrasts=52,independent_holdout_dates=0)
    assert all(v[k]==value for k,value in expected.items());assert prep['protocol_sha256']==v['protocol_sha256']==sha(ROOT/'protocol.json') and prep['source_sha256']==v['source_sha256']
    for key in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[key].items():assert sha(PROJECT/name)==digest,name
    assert len(prep['old_evidence'])==3166
    for r in [prep,v]:
        for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and len(c['tests'])==5 and c['source_sha256']==prep['source_sha256'] and c['protocol_sha256']==prep['protocol_sha256']
    for name,digest in c['artifacts'].items():assert sha(ROOT/name)==digest,name
    last=c['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        r=read(OUT/f'{phase}_manifest.json');assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert r[key]==prep[key]
        for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert last<=v['completed_utc'];fit=read(OUT/'training_manifest.json');assert fit['all_converged'] and not fit['validation_scoring_during_fit'] and fit['new_projection_fits']==48
    heads=read(OUT/'heads.json');assert len(heads)==60 and sum(h['iterations'] for h in heads)==fit['new_newton_iterations'] and sum(len(h['coefficients']) for h in heads)==1476
    for h in heads:
        variant=next(x for x in p['variants'] if x['method']==h['method']);assert h['dimensions']==variant['dimensions'] and h['gate']==variant['gate'] and sha(PROJECT/h['cache_file'])==h['cache_sha256'] and h['gradient_inf']<=1e-9 and h['hessian_min_eigenvalue']>0
        if h['source_job']:assert h['objective']<=h['parent_objective']+1e-12
    alt=csv('independent_solver_verification');settings=p['independent_solver'];assert len(alt)==60 and alt.objective_absolute_gap.max()<=settings['objective_absolute_tolerance'] and alt.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance'] and alt.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    checks=csv('state_verification');assert len(checks)==12 and checks.maximum_scalar_gap.max()<=2e-12
    for split,n in [('training',9465),('validation',261)]:assert len(csv(f'{split}_signal_states'))==n and len(csv(f'{split}_stability'))==12
    forecasts=csv('forecast_verification');assert len(forecasts)==60 and forecasts.training_rows.sum()==94650 and forecasts.heldout_rows.sum()==2610
    parts=csv('interaction_components');assert len(parts)==2088;np.testing.assert_allclose(parts.logit,parts.refitted_additive_logit+parts.interaction_logit,rtol=0,atol=1e-12)
    models=csv('model_predictions');ensemble=csv('ensemble_predictions');metrics=csv('ensemble_metrics');years=csv('yearly_metrics');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json');facts=read(OUT/'report_facts.json')
    assert len(models)==11223 and len(ensemble)==6786 and len(metrics)==78 and ensemble.groupby('method').size().eq(261).all()
    for r in metrics.itertuples():
        g=ensemble[ensemble.method.eq(r.method)]
        if r.window=='early_2015_2017':g=g[g.year.le(2017)]
        if r.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==r.n and g.date.is_unique and g.direction_up.eq(g.actual_up).sum()==r.correct_directions
        if r.method=='native_mse':assert g.probability.isna().all()
        else:assert abs(np.square(g.probability-g.actual_up).mean()-r.brier)<1e-14 and r.clipped_probabilities==0
    assert len(pairs)==52
    for r,q in zip(pairs,[dict(window=w['name'],**q) for w in p['windows'] for q in p['primary_comparisons_per_window']]):
        for key,value in q.items():assert r[key]==value
    improvements=[r for r in pairs if r['difference']<0 and r['holm_adjusted_p']<.05];worsenings=[r for r in pairs if r['difference']>0 and r['holm_adjusted_p']<.05];assert len(improvements)==facts['significant_improvements']==0 and len(worsenings)==facts['significant_deteriorations']==1
    assert len(assess)==8
    for a in assess:
        t=metrics[metrics.window.eq(a['window'])].set_index('method');r=t.loc[a['method']];w=next(w for w in p['windows'] if w['name']==a['window']);yr=years[years.year.between(int(w['start'][:4]),int(w['end'][:4]))];acc=yr.pivot(index='year',columns='method',values='accuracy');b=yr.pivot(index='year',columns='method',values='brier');cand=a['method']
        flags={f'accuracy_beats_{ref}':bool(r.accuracy>t.loc[ref,'accuracy']) for ref in [a['parent'],a['old_interaction'],'native_mse','training_frequency']};flags.update({f'brier_beats_{ref}':bool(r.brier<t.loc[ref,'brier']) for ref in [a['parent'],a['old_interaction'],'training_frequency',a['control']]})
        flags.update(log_loss_beats_frequency=bool(r.log_loss<t.loc['training_frequency','log_loss']),two_years_beat_native_accuracy=bool((acc[cand]>acc.native_mse).sum()>=2),two_years_beat_frequency_brier=bool((b[cand]<b.training_frequency).sum()>=2));assert flags==a['flags'] and len(flags)==11 and a['window_descriptive_pass']==all(flags.values()) and not a['cross_period_descriptive_pass'] and not a['strategy_promotion'] and a['interaction_hypothesis_retained']
    a=ensemble[ensemble.method.eq('learned_persistence_interaction')].sort_values('date');b=ensemble[ensemble.method.eq('learned_market')].sort_values('date');np.testing.assert_array_equal(a.direction_up,b.direction_up)
    cor=csv('state_correlations');q=cor[cor.split.eq('validation')&cor.gate.eq('persistence')].correlation_abs_trend60;assert facts['heldout_efficiency_abs_trend_correlation_min']==q.min() and facts['heldout_efficiency_abs_trend_correlation_max']==q.max()
    example=facts['permutation_example'];assert example['n']==60 and example['same_sign_neighbors_a']==58 and example['same_sign_neighbors_b']==19 and example['adjacent_pairs']==59 and abs(example['efficiency_a']-1/3)<1e-14 and abs(example['efficiency_b']-1/3)<1e-14
    usage=csv('historical_date_usage');assert len(usage)==261 and usage.previously_evaluated.all() and set(usage.date)==set(ensemble.date)
    from report22 import LABELS,MAIN,WINDOWS,FAMILIES,GATES,metric_row,pair_row,line,pct,num
    report_path=OUT/'第二十二轮测试报告.md';report=report_path.read_text(encoding='utf-8');note=(ROOT/'README.md').read_text(encoding='utf-8');assert report.startswith('# 第二十二轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report and v['protocol_sha256'] in report and '保留全历史交互' in report and '58.16%' in note
    for text,folder in [(report,OUT),(note,ROOT)]:
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if not link.startswith(('http://','https://')):assert (folder/link).resolve().is_file(),link
    for r in metrics[metrics.window.ne('pooled_2015_2020')&metrics.method.isin(MAIN)].itertuples():assert line(metric_row(r)) in report
    for r in pairs:assert line(pair_row(r)) in report
    for year,g in years.groupby('year'):
        t=g.set_index('method')
        for fam in FAMILIES:assert line([year,int(g.iloc[0]['n'])]+[int(t.loc[m,'correct_directions']) for m in fam+['native_mse','training_frequency']]) in report
    for r in csv('validation_stability').itertuples():assert line([int(r.cutoff[:4])+1,GATES[r.gate],r.n,r.missing,pct(r.high_state_share),f'{r.lag1_correlation:.4f}',f'{r.switches}/{r.adjacent_pairs}',f'{r.median_observed_run:g}',r.max_observed_run,f'{r.censored_runs}/{r.runs}']) in report
    for r in cor[cor.split.eq('validation')].itertuples():assert line([int(r.cutoff[:4])+1,GATES[r.gate],f'{r.correlation_volatility20:.6f}',f'{r.correlation_abs_trend60:.6f}']) in report
    for m,g in csv('gate_novelty').groupby('method'):assert line([LABELS[m],num(g.gate_linear_r2.min()),num(g.gate_linear_r2.mean()),num(g.gate_linear_r2.max())]) in report
    for (m,cutoff),g in csv('training_metrics').groupby(['method','cutoff']):assert line([LABELS[m],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())]) in report
    for a in assess:assert line([WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",'否','是']) in report
    visual=read(OUT/'visual_review.json');assert visual['status']=='PASS' and len(visual['files'])==2
    for name,digest in visual['files'].items():assert sha(ROOT/name)==digest and (ROOT/name).stat().st_size>50000
    for name in ['fixed_state_comparison','state_stability_and_overlap']:assert (OUT/f'{name}.svg').stat().st_size>10000
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve frozen delivery';files={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_manifest.json'}
        result=dict(experiment=22,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),protocol_sha256=v['protocol_sha256'],previous_files_preserved=3166,new_primary_fits=60,new_newton_updates=fit['new_newton_iterations'],new_projection_fits=48,
            independent_solutions=60,new_neural_forward_passes=0,neural_training_steps=0,new_hmm_fits=0,independent_holdout_dates=0,new_probability_forecasts=2610,model_records=11223,ensemble_records=6786,figures_visually_reviewed=2,
            cross_period_descriptive_pass={a['method']:False for a in assess},interaction_hypothesis_retained=True,new_state_interaction_adopted=False,path_efficiency_probability_improvement_retained=True,significant_improvements=0,significant_deteriorations=1,files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if target.exists():
        for name,digest in read(target)['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',frozen=target.exists(),hashed_files=len(read(target)['files']) if target.exists() else None,old_files_preserved=3166,report_characters=len(report),new_fits=60,interaction_hypothesis_retained=True,new_state_interaction_adopted=False),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

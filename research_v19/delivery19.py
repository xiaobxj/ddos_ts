"""Check the report against immutable evidence, then optionally freeze delivery."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(freeze):
    prep=read(OUT/'preparation_manifest.json');v=read(OUT/'verification.json');p=read(ROOT/'protocol.json')
    expected=dict(previous_files_preserved=2882,new_primary_fits=24,independent_solutions=24,independent_qr_projections=24,
        neural_states_with_inherited_round17_replay=18,new_neural_forward_passes=0,neural_training_steps=0,training_feature_rows=37860,
        new_probability_forecasts=1044,reused_model_records=4176,model_records=5220,ensemble_records=3132,metric_groups=441,state_metric_rows=288,
        reliability_bins=110,primary_contrasts=16,independent_holdout_dates=0)
    assert v['status']=='PASS' and all(v[key]==val for key,val in expected.items())
    assert prep['protocol_sha256']==v['protocol_sha256']==sha(ROOT/'protocol.json') and prep['source_sha256']==v['source_sha256']
    assert len(prep['old_evidence'])==2882
    for key in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[key].items():assert sha(PROJECT/name)==digest,name
    for record in [prep,v]:
        for name,digest in record['artifacts'].items():assert sha(ROOT/name)==digest,name
    contract=read(OUT/'contract_verification.json');assert contract['status']=='PASS' and len(contract['tests'])==3
    assert contract['source_sha256']==prep['source_sha256'] and contract['protocol_sha256']==prep['protocol_sha256']
    for name,digest in contract['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert prep['finished_utc']<=contract['completed_utc'];last=contract['completed_utc']
    for phase in ['training','scoring','evaluation']:
        r=read(OUT/f'{phase}_manifest.json');assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert r[key]==prep[key]
        for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert last<=v['completed_utc'];fitting=read(OUT/'training_manifest.json')
    assert fitting['new_primary_fits']==24 and fitting['new_newton_iterations']==93 and fitting['input_projections']==24
    assert fitting['all_converged'] and fitting['all_nested_training_objectives_pass'] and not fitting['validation_scoring_during_fit']
    assert fitting['neural_forward_passes']==fitting['neural_training_steps']==0
    heads=read(OUT/'heads.json');assert len(heads)==24 and sum(h['iterations'] for h in heads)==93
    for h in heads:
        variant=next(r for r in p['variants'] if r['method']==h['method']);assert h['dimensions']==variant['dimensions'] and h['base_dimensions']==variant['base_dimensions']
        assert len(h['coefficients'])==h['dimensions']+1 and h['gradient_inf']<=1e-9 and h['l2_lambda']==.01
        assert h['objective']<=h['parent_objective']+1e-12 and sha(PROJECT/h['cache_file'])==h['cache_sha256']
    alt=pd.read_csv(OUT/'independent_solver_verification.csv');settings=p['independent_solver'];assert len(alt)==24
    assert alt.objective_absolute_gap.max()<=settings['objective_absolute_tolerance']
    assert alt.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance'] and alt.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    projection=pd.read_csv(OUT/'projection_verification.csv');settings=p['projection_verification'];assert len(projection)==24
    assert projection.training_fitted_value_gap.max()<=settings['training_fitted_value_absolute_tolerance']
    assert projection.training_standardized_residual_gap.max()<=settings['training_standardized_residual_absolute_tolerance']
    assert projection.validation_standardized_residual_gap.max()<=settings['validation_standardized_residual_absolute_tolerance']
    assert projection.training_orthogonality_inf.max()<1e-8 and projection.qr_rank.eq(projection.svd_rank).all()
    coefficients=pd.read_csv(OUT/'coefficients.csv');assert len(coefficients)==726
    parts=pd.read_csv(OUT/'interaction_components.csv');assert len(parts)==1044
    np.testing.assert_allclose(parts.logit,parts.refitted_additive_logit+parts.interaction_logit,rtol=0,atol=1e-12)
    diag=pd.read_csv(OUT/'interaction_diagnostics.csv');assert len(diag)==24 and not diag.sd_floored.any()
    assert diag.augmented_design_rank.eq(diag.base_design_rank+1).all()
    states=pd.read_csv(OUT/'validation_states.csv');usage=pd.read_csv(OUT/'historical_date_usage.csv')
    assert len(states)==len(usage)==261 and states.date.is_unique and usage.previously_evaluated.all() and set(states.date)==set(usage.date)
    models=pd.read_csv(OUT/'model_predictions.csv');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv');metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    assert len(models)==5220 and len(ensemble)==3132 and len(metrics)==36 and ensemble.groupby('method').size().eq(261).all()
    for r in metrics.itertuples():
        g=ensemble[ensemble.method.eq(r.method)]
        if r.window=='early_2015_2017':g=g[g.year.le(2017)]
        if r.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==r.n and g.date.is_unique and int(g.direction_up.eq(g.actual_up).sum())==r.correct_directions
        if r.method=='native_mse':assert g.probability.isna().all() and pd.isna(r.brier)
        else:assert abs(np.square(g.probability-g.actual_up).mean()-r.brier)<1e-14 and r.clipped_probabilities==0
    pairs=read(OUT/'primary_comparisons.json');assert len(pairs)==16
    expected_pairs=[dict(window=w['name'],**q) for w in p['windows'] for q in p['primary_comparisons_per_window']]
    for pair,expected_pair in zip(pairs,expected_pairs):
        for key,val in expected_pair.items():assert pair[key]==val
    significant=[r for r in pairs if r['holm_adjusted_p']<.05];assert len(significant)==1
    assert significant[0]['candidate']=='raw_vol_interaction' and significant[0]['reference']=='training_frequency'
    assert significant[0]['window']=='early_2015_2017' and significant[0]['metric']=='brier' and significant[0]['difference']>0
    from report19 import LABELS,WINDOWS,MAIN,ANNUAL,CANDIDATES,pct,num
    years=pd.read_csv(OUT/'yearly_metrics.csv');assessments=read(OUT/'assessments.json');assert len(assessments)==4
    for a in assessments:
        t=metrics[metrics.window.eq(a['window'])].set_index('method');c=t.loc[a['method']];candidate=a['method'];parent=a['parent'];anchor=a['round17_anchor']
        w=next(w for w in p['windows'] if w['name']==a['window']);yr=years[years.year.between(int(w['start'][:4]),int(w['end'][:4]))]
        acc=yr.pivot(index='year',columns='method',values='accuracy');brier=yr.pivot(index='year',columns='method',values='brier')
        counts=dict(years_beating_native_accuracy=int((acc[candidate]>acc.native_mse).sum()),years_beating_frequency_brier=int((brier[candidate]<brier.training_frequency).sum()))
        flags=dict(accuracy_beats_parent=bool(c.accuracy>t.loc[parent,'accuracy']),accuracy_beats_round17=bool(c.accuracy>t.loc[anchor,'accuracy']),
            accuracy_beats_native=bool(c.accuracy>t.loc['native_mse','accuracy']),accuracy_beats_frequency=bool(c.accuracy>t.loc['training_frequency','accuracy']),
            brier_beats_parent=bool(c.brier<t.loc[parent,'brier']),brier_beats_round17=bool(c.brier<t.loc[anchor,'brier']),brier_beats_frequency=bool(c.brier<t.loc['training_frequency','brier']),
            log_loss_beats_frequency=bool(c.log_loss<t.loc['training_frequency','log_loss']),two_years_beat_native_accuracy=counts['years_beating_native_accuracy']>=2,
            two_years_beat_frequency_brier=counts['years_beating_frequency_brier']>=2)
        assert a['counts']==counts and a['flags']==flags and len(flags)==10
        assert a['window_descriptive_pass']==all(flags.values()) and not a['cross_period_descriptive_pass'] and not a['strategy_promotion'] and a['significant_improvements']==0
    report_path=OUT/'第十九轮测试报告.md';report=report_path.read_text(encoding='utf-8')
    assert report.startswith('# 第十九轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report and v['protocol_sha256'] in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().is_file(),link
    for r in metrics[metrics.window.ne('pooled_2015_2020')&metrics.method.isin(MAIN)].itertuples():
        line='| '+' | '.join([LABELS[r.method],str(r.correct_directions),pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)])+' |'
        assert line in report,line
    for r in pairs:
        line='| '+' | '.join([WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',
            f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"])+' |'
        assert line in report,line
    for year,g in years.groupby('year'):
        t=g.set_index('method');line='| '+' | '.join([str(year),str(int(g.iloc[0].n))]+[pct(t.loc[m,'accuracy']) for m in ANNUAL])+' |';assert line in report
    training=pd.read_csv(OUT/'training_metrics.csv')
    for (method,cutoff),g in training.groupby(['method','cutoff']):
        line='| '+' | '.join([LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.parent_objective.mean()),num(g.objective.mean())])+' |';assert line in report
    summary=pd.read_csv(OUT/'component_summary.csv');assert len(summary)==48
    for (method,cutoff),g in diag.groupby(['method','cutoff']):
        s=summary[summary.method.eq(method)&summary.cutoff.eq(cutoff)&summary.split.eq('validation')]
        line='| '+' | '.join([LABELS[method],str(int(cutoff[:4])+1),num(g.interaction_coefficient.mean()),num(s.interaction_feature_mean.mean()),num(s.interaction_feature_std.mean()),
            num(s.interaction_logit_mean.mean()),num(s.interaction_logit_std.mean())])+' |';assert line in report
    # Verify the main textual diagnoses against annual probabilities and components.
    y=years.pivot(index='year',columns='method',values='brier');delta=y.learned_vol_interaction-y.learned_market
    for year in [2018,2019,2020]:assert delta.loc[year]>0 and f'{delta.loc[year]:.6f}' in report
    m=years[years.year.eq(2015)].set_index('method')
    for key in [pct(m.loc['raw_trend','mean_probability']),pct(m.loc['raw_vol_interaction','mean_probability']),pct(m.loc['raw_vol_interaction','observed_up_fraction'])]:assert key in report
    g=summary[summary.method.eq('raw_vol_interaction')&summary.cutoff.eq('2014-12-31')&summary.split.eq('validation')].iloc[0]
    for key in ['interaction_feature_mean','interaction_feature_std','interaction_logit_mean']:assert f'{abs(g[key]):.6f}' in report
    state_metrics=pd.read_csv(OUT/'state_metrics.csv');cells=state_metrics[state_metrics.partition.eq('trend_volatility')&state_metrics.method.eq('learned_vol_interaction')]
    assert len(cells)==12 and cells['sparse'].sum()==5
    for a in assessments:
        line='| '+' | '.join([WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/10",'否'])+' |';assert line in report
    for name in ['interaction_comparison','annual_interaction_diagnostics']:
        assert (OUT/f'{name}.png').stat().st_size>50000 and (OUT/f'{name}.svg').stat().st_size>10000
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve delivery'
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_manifest.json'}
        result=dict(experiment=19,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),protocol_sha256=v['protocol_sha256'],
            previous_files_preserved=2882,new_primary_fits=24,new_newton_updates=93,independent_solutions=24,independent_qr_projections=24,new_neural_forward_passes=0,neural_training_steps=0,
            inherited_neural_replay_states=18,independent_holdout_dates=0,new_probability_forecasts=1044,model_records=5220,ensemble_records=3132,
            figures_visually_reviewed=2,cross_period_descriptive_pass={'learned_vol_interaction':False,'raw_vol_interaction':False},significant_improvements=0,
            significant_deteriorations=1,files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    delivery=read(target)
    for name,digest in delivery['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivery['files']),report_characters=len(report),previous_files_preserved=2882,
        independent_solutions=24,independent_qr_projections=24,new_probability_forecasts=1044,cross_period_descriptive_pass=delivery['cross_period_descriptive_pass']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

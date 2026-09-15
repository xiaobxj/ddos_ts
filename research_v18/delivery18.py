"""Validate delivered research evidence without refitting; freeze once."""
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
    expected=dict(previous_files_preserved=2795,new_primary_fits=30,independent_solutions=30,neural_states_with_inherited_round17_replay=18,
        new_neural_forward_passes=0,neural_training_steps=0,training_feature_rows=47325,market_training_rows=9465,market_validation_rows=261,
        omitted_duplicate_checks=18,new_probability_forecasts=1305,reused_model_records=2871,model_records=4176,ensemble_records=2610,
        metric_groups=366,state_metric_rows=240,reliability_bins=90,primary_contrasts=14,independent_holdout_dates=0)
    assert v['status']=='PASS' and all(v[k]==value for k,value in expected.items())
    assert prep['protocol_sha256']==v['protocol_sha256']==sha(ROOT/'protocol.json') and prep['source_sha256']==v['source_sha256']
    assert len(prep['old_evidence'])==2795
    for key in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[key].items():assert sha(PROJECT/name)==digest,name
    for record in [prep,v]:
        for name,digest in record['artifacts'].items():assert sha(ROOT/name)==digest,name
    contract=read(OUT/'contract_verification.json');assert contract['status']=='PASS' and len(contract['tests'])==3
    assert contract['source_sha256']==prep['source_sha256'] and contract['protocol_sha256']==prep['protocol_sha256']
    assert prep['finished_utc']<=contract['completed_utc'];last=contract['completed_utc']
    for phase in ['training','scoring','evaluation']:
        r=read(OUT/f'{phase}_manifest.json');assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert r[key]==prep[key]
        for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert last<=v['completed_utc'];fitting=read(OUT/'training_manifest.json')
    assert fitting['new_primary_fits']==30 and fitting['new_newton_iterations']==109 and fitting['all_converged'] and fitting['all_nested_training_objectives_pass']
    assert not fitting['validation_scoring_during_fit'] and fitting['neural_forward_passes']==fitting['neural_training_steps']==0
    heads=read(OUT/'heads.json');assert len(heads)==30 and sum(h['iterations'] for h in heads)==109
    for h in heads:
        variant=next(r for r in p['variants'] if r['method']==h['method']);assert h['dimensions']==variant['dimensions'] and h['market_indices']==variant['market_indices']
        assert len(h['coefficients'])==h['dimensions']+1 and h['gradient_inf']<=1e-9 and h['l2_lambda']==.01
        assert h['objective']<=h['parent_or_constant_objective']+1e-12 and sha(PROJECT/h['cache_file'])==h['cache_sha256']
    alt=pd.read_csv(OUT/'independent_solver_verification.csv');settings=p['independent_solver'];assert len(alt)==30
    assert alt.objective_absolute_gap.max()<=settings['objective_absolute_tolerance']
    assert alt.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance'] and alt.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    duplicate=pd.read_csv(OUT/'duplicate_column_verification.csv');assert len(duplicate)==18 and duplicate.omitted_from_raw_append.all()
    assert duplicate[['training_raw_error','validation_raw_error']].to_numpy().max()<1e-14
    assert duplicate[['training_standardized_error','validation_standardized_error']].to_numpy().max()<1e-12
    coefficients=pd.read_csv(OUT/'coefficients.csv');assert len(coefficients)==732
    parts=pd.read_csv(OUT/'logit_components.csv');assert len(parts)==1305
    np.testing.assert_allclose(parts.logit,parts.representation_logit+parts.market_logit+parts.intercept,rtol=0,atol=1e-12)
    states=pd.read_csv(OUT/'validation_states.csv');usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(states)==len(usage)==261 and states.date.is_unique and usage.previously_evaluated.all()
    assert set(states.date)==set(usage.date)
    models=pd.read_csv(OUT/'model_predictions.csv');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv');metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    assert len(models)==4176 and len(ensemble)==2610 and len(metrics)==30 and ensemble.groupby('method').size().eq(261).all()
    for r in metrics.itertuples():
        g=ensemble[ensemble.method.eq(r.method)]
        if r.window=='early_2015_2017':g=g[g.year.le(2017)]
        if r.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==r.n and g.date.is_unique and int(g.direction_up.eq(g.actual_up).sum())==r.correct_directions
        if r.method=='native_mse':assert g.probability.isna().all() and pd.isna(r.brier)
        else:assert abs(np.square(g.probability-g.actual_up).mean()-r.brier)<1e-14 and r.clipped_probabilities==0
    pairs=read(OUT/'primary_comparisons.json');assert len(pairs)==14 and all(r['holm_adjusted_p']>=.05 for r in pairs)
    expected_pairs=[dict(window=w['name'],**q) for w in p['windows'] for q in p['primary_comparisons_per_window']]
    for pair,expected_pair in zip(pairs,expected_pairs):
        for k,value in expected_pair.items():assert pair[k]==value
    assessments=read(OUT/'assessments.json');assert len(assessments)==4
    for a in assessments:assert len(a['flags'])==9 and a['window_descriptive_pass']==all(a['flags'].values()) and not a['cross_period_descriptive_pass']
    from report18 import LABELS,WINDOWS,MAIN,pct,num
    report_path=OUT/'第十八轮测试报告.md';report=report_path.read_text(encoding='utf-8')
    assert report.startswith('# 第十八轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report and v['protocol_sha256'] in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().is_file(),link
    for r in metrics[metrics.window.ne('pooled_2015_2020')&metrics.method.isin(MAIN)].itertuples():
        line='| '+' | '.join([LABELS[r.method],str(r.correct_directions),pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)])+' |'
        assert line in report,line
    for r in pairs:
        line='| '+' | '.join([WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',
            f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"])+' |'
        assert line in report,line
    years=pd.read_csv(OUT/'yearly_metrics.csv')
    for year,g in years.groupby('year'):
        t=g.set_index('method');line='| '+' | '.join([str(year),str(int(g.iloc[0].n))]+[pct(t.loc[m,'accuracy']) for m in MAIN])+' |';assert line in report
    training=pd.read_csv(OUT/'training_metrics.csv')
    for (method,cutoff),g in training.groupby(['method','cutoff']):
        line='| '+' | '.join([LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.log_loss.mean()),num(g.objective.mean()),num(g.parent_or_constant_objective.mean())])+' |';assert line in report
    # Validate the specific2017 diagnosis against original forecast and component rows.
    yr=years[years.year.eq(2017)].set_index('method');assert yr.loc['market4','predicted_up_fraction']==.02
    for m in ['learned_clip','learned_market','market4']:assert pct(yr.loc[m,'mean_probability']) in report
    summary=pd.read_csv(OUT/'component_summary.csv');g=summary[summary.method.eq('learned_market')&summary.cutoff.eq('2016-12-31')&summary.split.eq('validation')]
    for key in ['market_mean','representation_mean','intercept','total_logit_mean']:assert f'{abs(g[key].mean()):.6f}' in report
    for name in ['market_context_comparison','annual_probability_diagnostics']:
        assert (OUT/f'{name}.png').stat().st_size>50000 and (OUT/f'{name}.svg').stat().st_size>10000
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve delivery'
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_manifest.json'}
        result=dict(experiment=18,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),protocol_sha256=v['protocol_sha256'],
            previous_files_preserved=2795,new_primary_fits=30,new_newton_updates=109,independent_solutions=30,new_neural_forward_passes=0,neural_training_steps=0,
            inherited_neural_replay_states=18,market_state_dates=261,independent_holdout_dates=0,new_probability_forecasts=1305,model_records=4176,ensemble_records=2610,
            figures_visually_reviewed=2,cross_period_descriptive_pass={'learned_market':False,'raw_trend':False},files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    delivery=read(target)
    for name,digest in delivery['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivery['files']),report_characters=len(report),previous_files_preserved=2795,
        independent_solutions=30,new_probability_forecasts=1305,cross_period_descriptive_pass=delivery['cross_period_descriptive_pass']),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

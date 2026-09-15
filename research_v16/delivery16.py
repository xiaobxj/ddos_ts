"""Final report/evidence validation and one-time delivery freeze."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OUT=ROOT/'results'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(freeze):
    prep=read(OUT/'preparation_manifest.json');v=read(OUT/'verification.json');protocol=read(ROOT/'protocol.json')
    expected=dict(neural_states_replayed=18,learned_training_rows=28395,raw_training_rows=9465,learned_validation_rows=783,
        raw_validation_rows=261,native_forecasts=783,model_forecasts=1827,independent_solutions=24,retained_heads=9,retained_late_probabilities=423,
        metric_groups=63,reliability_bins=40,paired_contrasts=12,historical_dates_audited=261,independent_holdout_dates=0,previous_files_preserved=2618)
    assert v['status']=='PASS' and all(v[k]==value for k,value in expected.items())
    assert sha(ROOT/'protocol.json')==prep['protocol_sha256']==v['protocol_sha256']
    assert prep['source_sha256']==v['source_sha256'] and len(prep['old_evidence'])==2618
    for key in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[key].items():assert sha(PROJECT/name)==digest,name
    for name,digest in prep['artifacts'].items():assert sha(ROOT/name)==digest,name
    for name,digest in v['artifacts'].items():assert sha(ROOT/name)==digest,name
    contract=read(OUT/'contract_verification.json');assert contract['status']=='PASS' and len(contract['tests'])==4
    assert contract['source_sha256']==prep['source_sha256'] and contract['protocol_sha256']==prep['protocol_sha256']
    assert prep['finished_utc']<=contract['completed_utc'];finish=contract['completed_utc']
    for phase in ['training','scoring','evaluation']:
        run=read(OUT/f'{phase}_manifest.json');assert finish<=run['started_utc']<run['finished_utc'];finish=run['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert run[key]==prep[key]
        for name,digest in run['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert finish<=v['completed_utc']
    training=read(OUT/'training_manifest.json');assert training['new_primary_fits']==15 and training['reused_heads']==9
    assert training['all_converged'] and not training['validation_scoring_during_fit'] and training['neural_training_steps']==0
    assert training['new_newton_iterations']==55 and training['new_training_feature_rows']==20568
    heads=read(OUT/'heads.json');assert len(heads)==24 and sum(h['reused'] for h in heads)==9
    assert sum(h['iterations'] for h in heads if not h['reused'])==55
    old={(h['cutoff'],h['seed']):h for h in read(PROJECT/'research_v15/results/heads.json') if h['family']=='probe_mse'}
    for h in heads:
        assert len(h['coefficients'])==26 and h['l2_lambda']==.01 and h['gradient_inf']<=1e-9
        assert sha(PROJECT/h['cache_file'])==h['cache_sha256']
        if h['kind']=='learned':assert sha(PROJECT/h['project_file'])==h['sha256']
        if h['reused']:np.testing.assert_array_equal(h['coefficients'],old[(h['cutoff'],h['seed'])]['coefficients'])
    independent=pd.read_csv(OUT/'independent_solver_verification.csv');assert len(independent)==24
    settings=protocol['independent_solver']
    assert independent.objective_absolute_gap.max()<=settings['objective_absolute_tolerance']
    assert independent.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance']
    assert independent.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(usage)==261 and usage.previously_evaluated.all()
    assert usage[usage.year.le(2017)].seen_round13_rolling.all() and usage[usage.year.ge(2018)].seen_round15_validation.all()
    meta=read(OUT/'training_metadata.json');tr=pd.read_csv(OUT/'training_rows.csv')
    for cutoff,g in tr.groupby('cutoff'):
        assert g.joint_completed.le(cutoff).all() and len(g)==meta[cutoff]['train_n']
        assert abs(g.direction.mean()-meta[cutoff]['frequency'])<1e-15
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv');models=pd.read_csv(OUT/'model_predictions.csv')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv');assert len(ensemble)==1305 and len(models)==1827 and len(metrics)==15
    for r in metrics.itertuples():
        g=ensemble[ensemble.method.eq(r.method)]
        if r.window=='early_2015_2017':g=g[g.year.le(2017)]
        if r.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==r.n and g.date.is_unique;y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool)
        assert int((up==y).sum())==r.correct_directions and abs((up==y).mean()-r.accuracy)<1e-14
        if r.method=='native_mse':assert g.probability.isna().all() and pd.isna(r.brier)
        else:assert abs(np.square(g.probability.to_numpy()-y.astype(float)).mean()-r.brier)<1e-14
    pairs=read(OUT/'primary_comparisons.json');assert len(pairs)==12
    expected_pairs=[dict(window=w['name'],**p) for w in protocol['windows'] for p in protocol['primary_comparisons_per_window']]
    for p,e in zip(pairs,expected_pairs):
        for key,value in e.items():assert p[key]==value
        table=metrics[metrics.window.eq(p['window'])].set_index('method')
        assert abs(p['difference']-(table.loc[p['candidate'],p['metric']]-table.loc[p['reference'],p['metric']]))<1e-14
    assessments=read(OUT/'assessments.json');assert len(assessments)==4
    for r in assessments:
        assert len(r['flags'])==6 and r['window_descriptive_pass']==all(r['flags'].values())
        assert not r['cross_period_descriptive_pass'] and not r['independent_confirmation'] and not r['strategy_promotion']
        assert r['cross_period_descriptive_pass']==all(a['window_descriptive_pass'] for a in assessments if a['method']==r['method'])
    # Validate the post-evaluation descriptions against immutable feature caches.
    posthoc=read(OUT/'posthoc_diagnostic_provenance.json');assert posthoc['completed_utc']>=v['completed_utc']
    assert posthoc['new_fits']==posthoc['new_predictions']==0 and not posthoc['causal_explanation_claim']
    accounting=pd.read_csv(OUT/'posthoc_logit_accounting.csv').set_index('job');contributions=pd.read_csv(OUT/'posthoc_feature_shift.csv')
    assert len(accounting)==24 and len(contributions)==600
    validation={r['job']:r for r in read(OUT/'validation_features.json')}
    for head in heads:
        ref=validation[head['job']];assert sha(PROJECT/ref['cache_file'])==ref['cache_sha256']
        with np.load(PROJECT/head['cache_file']) as d,np.load(PROJECT/ref['cache_file']) as f:
            x=(f['features'].astype(float)-d['mean'])/d['sd'];theta=np.asarray(head['coefficients'])
            z=x@theta[:-1]+theta[-1];r=accounting.loc[head['job']]
            assert abs(z.mean()-r.mean_validation_logit)<1e-12 and abs(theta[-1]-r.training_intercept)<1e-14
            c=contributions[contributions.job.eq(head['job'])].sort_values('feature')
            np.testing.assert_allclose(c.mean_logit_contribution,x.mean(axis=0)*theta[:-1],rtol=0,atol=1e-12)
            assert abs(c.mean_logit_contribution.sum()+r.training_intercept-r.mean_validation_logit)<1e-12
    training_table=pd.read_csv(OUT/'training_metrics.csv');folds=pd.read_csv(OUT/'training_fold_summary.csv')
    for r in folds.itertuples():
        g=training_table[training_table.method.eq(r.method)&training_table.cutoff.eq(r.cutoff)]
        assert len(g)==(3 if r.method=='learned_probe' else 1)
        for key in ['accuracy','auroc','log_loss','constant_log_loss','brier','constant_brier']:assert abs(g[key].mean()-getattr(r,key))<1e-12
    report_path=OUT/'第十六轮测试报告.md';report=report_path.read_text(encoding='utf-8')
    assert report.startswith('# 第十六轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report
    assert prep['protocol_sha256'] in report and '评分后描述性诊断' in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().is_file(),link
    for r in metrics[metrics.window.ne('pooled_2015_2020')].itertuples():
        assert f'{r.accuracy*100:.2f}%' in report and f'{r.balanced_accuracy*100:.2f}%' in report and f'{r.auroc:.6f}' in report
        for value in [r.brier,r.log_loss]:
            if pd.notna(value):assert f'{value:.6f}' in report
    for r in folds.itertuples():assert f'{r.accuracy*100:.2f}%' in report and f'{r.log_loss:.6f}' in report and f'{r.constant_log_loss:.6f}' in report
    for r in pairs:assert f"{r['p']:.4f}" in report and f"{r['holm_adjusted_p']:.4f}" in report
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve delivery'
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
        result=dict(experiment=16,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),new_heads=15,
            reused_heads=9,new_newton_updates=55,neural_training_steps=0,independent_solutions=24,neural_states_replayed=18,
            historical_weeks=261,early_weeks=120,late_weeks=141,independent_holdout_dates=0,previous_files_preserved=2618,
            cross_period_descriptive_pass={m:all(r['window_descriptive_pass'] for r in assessments if r['method']==m) for m in ['learned_probe','raw25_probe']},files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    delivered=read(target)
    for name,digest in delivered['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivered['files']),report_characters=len(report),previous_files_preserved=2618,
        independent_solutions=24,cross_period_descriptive_pass=delivered['cross_period_descriptive_pass']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

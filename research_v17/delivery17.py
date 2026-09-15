"""Read-only delivery validation; --freeze records the completed evidence once."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V16=PROJECT/'research_v16'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(ref):
    path=PROJECT/ref['cache_file'];assert sha(path)==ref['cache_sha256']
    with np.load(path) as d:return {k:d[k].copy() for k in d.files}

def main(freeze):
    prep=read(OUT/'preparation_manifest.json');v=read(OUT/'verification.json');p=read(ROOT/'protocol.json')
    expected=dict(previous_files_preserved=2716,new_primary_fits=24,independent_solutions=24,neural_states_replayed=18,neural_training_steps=0,
        learned_training_rows=28395,raw_training_rows=9465,learned_validation_rows=783,raw_validation_rows=261,new_probability_forecasts=1044,
        reused_model_forecasts=1827,model_records=2871,ensemble_records=1827,market_state_training_rows=9465,market_state_validation_rows=261,
        metric_groups=258,state_metric_rows=168,reliability_bins=60,primary_contrasts=12,independent_holdout_dates=0)
    assert v['status']=='PASS' and all(v[k]==value for k,value in expected.items())
    assert sha(ROOT/'protocol.json')==prep['protocol_sha256']==v['protocol_sha256'] and prep['source_sha256']==v['source_sha256']
    assert len(prep['old_evidence'])==2716
    for key in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[key].items():assert sha(PROJECT/name)==digest,name
    for record in [prep,v]:
        for name,digest in record['artifacts'].items():assert sha(ROOT/name)==digest,name
    c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and len(c['tests'])==4
    assert c['source_sha256']==prep['source_sha256'] and c['protocol_sha256']==prep['protocol_sha256']
    assert prep['finished_utc']<=c['completed_utc'];last=c['completed_utc']
    for phase in ['training','scoring','evaluation']:
        r=read(OUT/f'{phase}_manifest.json');assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert r[key]==prep[key]
        for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert last<=v['completed_utc']
    fitting=read(OUT/'training_manifest.json');assert fitting['new_primary_fits']==24 and fitting['new_newton_iterations']==90 and fitting['all_converged']
    assert not fitting['validation_scoring_during_fit'] and not fitting['labels_used_to_fit_state_thresholds'] and fitting['neural_training_steps']==0
    heads=read(OUT/'heads.json');assert len(heads)==24 and sum(h['iterations'] for h in heads)==90
    assert all(h['gradient_inf']<=1e-9 and h['l2_lambda']==.01 for h in heads)
    alt=pd.read_csv(OUT/'independent_solver_verification.csv');assert len(alt)==24;settings=p['independent_solver']
    assert alt.objective_absolute_gap.max()<=settings['objective_absolute_tolerance']
    assert alt.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance']
    assert alt.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    # Independently validate the extra, explicitly post-evaluation tail accounting.
    post=read(OUT/'posthoc_tail_provenance.json');assert post['completed_utc']>=v['completed_utc']==post['after_verification_utc']
    assert post['new_fits']==post['new_predictions']==0
    for name,digest in post['artifacts'].items():assert sha(ROOT/name)==digest
    tails=pd.read_csv(OUT/'posthoc_tail_details.csv');assert len(tails)==600
    sources={h['job']:h for h in read(OUT/'source_heads.json')};refs={r['job']:r for r in read(V16/'results/validation_features.json')}
    clipping=pd.read_csv(OUT/'clipping_summary.csv').set_index('job');clip_rows=pd.read_csv(OUT/'clipping_rows.csv');assert len(clipping)==24 and len(clip_rows)==1044
    for h in heads:
        source=sources[h['source_job']];tr=load(source);te=load(refs[source['job']]);d=load(h);a=tr['features'].astype(float);b=te['features'].astype(float)
        g=tails[tails.job.eq(h['job'])].sort_values('feature');assert len(g)==25 and g.train_n.eq(len(a)).all() and g.validation_n.eq(len(b)).all()
        lower=b<d['lower'];upper=b>d['upper'];changed=lower|upper
        checks=dict(lower=d['lower'],upper=d['upper'],training_lower_fraction=(a<d['lower']).mean(axis=0),training_upper_fraction=(a>d['upper']).mean(axis=0),
            validation_lower_fraction=lower.mean(axis=0),validation_upper_fraction=upper.mean(axis=0),validation_raw_mean=b.mean(axis=0),
            training_clipped_mean=d['mean'],training_clipped_sd=d['sd'],standardized_validation_mean_before=((b-d['mean'])/d['sd']).mean(axis=0),
            standardized_validation_mean_after=((np.maximum(d['lower'],np.minimum(d['upper'],b))-d['mean'])/d['sd']).mean(axis=0))
        for k,value in checks.items():np.testing.assert_allclose(g[k],value,rtol=0,atol=1e-12)
        c=clipping.loc[h['job']];assert abs(c.validation_coordinate_clip_fraction-changed.mean())<1e-14
        assert abs(c.validation_row_clip_fraction-changed.any(axis=1).mean())<1e-14
        rr=clip_rows[clip_rows.job.eq(h['job'])].sort_values('row_index');np.testing.assert_array_equal(rr.row_index,te['row_index']);np.testing.assert_array_equal(rr.clipped_coordinates,changed.sum(axis=1))
    usage=pd.read_csv(OUT/'historical_date_usage.csv');states=pd.read_csv(OUT/'validation_states.csv');state_metrics=pd.read_csv(OUT/'state_metrics.csv')
    assert len(usage)==len(states)==261 and usage.previously_evaluated.all() and set(usage.date)==set(states.date)
    main_cells=state_metrics[state_metrics.partition.eq('trend_volatility')&state_metrics.method.eq('native_mse')];assert len(main_cells)==12 and main_cells['sparse'].sum()==5
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv');models=pd.read_csv(OUT/'model_predictions.csv');metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    assert len(ensemble)==1827 and len(models)==2871 and len(metrics)==21
    for r in metrics.itertuples():
        g=ensemble[ensemble.method.eq(r.method)]
        if r.window=='early_2015_2017':g=g[g.year.le(2017)]
        if r.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==r.n and g.date.is_unique and int(g.direction_up.eq(g.actual_up).sum())==r.correct_directions
        if r.method=='native_mse':assert g.probability.isna().all() and pd.isna(r.brier)
        else:assert abs(np.square(g.probability-g.actual_up).mean()-r.brier)<1e-14 and r.clipped_probabilities==0
    pairs=read(OUT/'primary_comparisons.json');assert len(pairs)==12
    assert sum(r['holm_adjusted_p']<.05 and r['difference']<0 for r in pairs)==0
    assert sum(r['holm_adjusted_p']<.05 and r['difference']>0 for r in pairs)==1
    assessments=read(OUT/'assessments.json');assert len(assessments)==4
    for a in assessments:assert len(a['flags'])==8 and a['window_descriptive_pass']==all(a['flags'].values()) and not a['cross_period_descriptive_pass']
    # Check the actual report tables and local asset links against verified values.
    from report17 import LABELS,WINDOWS,STATES,pct,num
    report_path=OUT/'第十七轮测试报告.md';report=report_path.read_text(encoding='utf-8')
    assert report.startswith('# 第十七轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report
    assert v['protocol_sha256'] in report and '评分后' in report and '5 个稀疏单元' in report
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if not link.startswith(('http://','https://')):assert (OUT/link).resolve().is_file(),link
    for r in metrics[metrics.window.ne('pooled_2015_2020')].itertuples():
        line='| '+' | '.join([LABELS[r.method],str(r.correct_directions),pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)])+' |'
        assert line in report,line
    for r in pairs:
        line='| '+' | '.join([WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',
            f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"])+' |'
        assert line in report,line
    for w in p['windows']:
        g=state_metrics[state_metrics.window.eq(w['name'])&state_metrics.partition.eq('trend_volatility')]
        for state,label in STATES.items():
            t=g[g.state.eq(state)].set_index('method');n=int(t.loc['native_mse','n'])
            line='| '+' | '.join([label+(' *' if n<20 else ''),str(n),pct(t.loc['native_mse','observed_up_fraction'])]+[pct(t.loc[m,'accuracy']) for m in ['native_mse','learned_clip','raw25_clip','training_frequency']])+' |'
            assert line in report,line
    for name in ['clipping_comparison','state_diagnostics','market_distribution_shift']:
        assert (OUT/f'{name}.png').stat().st_size>50000 and (OUT/f'{name}.svg').stat().st_size>10000
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve delivery'
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_manifest.json'}
        result=dict(experiment=17,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),
            protocol_sha256=v['protocol_sha256'],previous_files_preserved=2716,new_primary_fits=24,new_newton_updates=90,neural_training_steps=0,
            independent_solutions=24,market_state_dates=261,sparse_primary_cells=5,primary_cells=12,independent_holdout_dates=0,
            new_probability_forecasts=1044,model_records=2871,ensemble_records=1827,figures_visually_reviewed=3,
            cross_period_descriptive_pass={'learned_clip':False,'raw25_clip':False},files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    delivery=read(target)
    for name,digest in delivery['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivery['files']),report_characters=len(report),previous_files_preserved=2716,
        independent_solutions=24,market_state_dates=261,cross_period_descriptive_pass=delivery['cross_period_descriptive_pass']),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

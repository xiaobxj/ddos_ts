"""Final immutable delivery checks; no training, filtering, or prediction."""
from pathlib import Path
import json,hashlib,re,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')

def main(freeze):
    prep=read(OUT/'preparation_manifest.json');p=read(ROOT/'protocol.json');v=read(OUT/'verification.json');assert v['status']=='PASS'
    expected=dict(previous_files_preserved=3061,new_hmm_fits=6,new_primary_fits=30,independent_hmm_checks=6,independent_solutions=30,independent_qr_projections=24,
        neural_states_with_inherited_round17_replay=18,new_neural_forward_passes=0,neural_training_steps=0,training_feature_rows=47325,new_probability_forecasts=1305,
        reused_model_records=7308,model_records=8613,ensemble_records=5220,metric_groups=852,state_metric_rows=480,hmm_state_metric_rows=120,reliability_bins=190,primary_contrasts=26,independent_holdout_dates=0)
    assert all(v[k]==value for k,value in expected.items());assert prep['protocol_sha256']==v['protocol_sha256']==sha(ROOT/'protocol.json') and prep['source_sha256']==v['source_sha256']
    for key in ['old_evidence','source_sha256','input_sha256']:
        for name,digest in prep[key].items():assert sha(PROJECT/name)==digest,name
    assert len(prep['old_evidence'])==3061
    for record in [prep,v]:
        for name,digest in record['artifacts'].items():assert sha(ROOT/name)==digest,name
    c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and len(c['tests'])==5 and c['source_sha256']==prep['source_sha256'] and c['protocol_sha256']==prep['protocol_sha256']
    for name,digest in c['artifacts'].items():assert sha(ROOT/name)==digest,name
    last=c['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        r=read(OUT/f'{phase}_manifest.json');assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['source_sha256','input_sha256','protocol_sha256']:assert r[key]==prep[key]
        for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert last<=v['completed_utc'];fit=read(OUT/'training_manifest.json');assert fit['all_converged'] and not fit['heldout_filtering_during_fit'] and not fit['validation_scoring_during_fit']
    heads=read(OUT/'heads.json');contexts=read(OUT/'hmm_contexts.json');assert len(heads)==30 and len(contexts)==6 and sum(h['iterations'] for h in heads)==fit['new_newton_iterations'] and sum(c['iterations'] for c in contexts)==fit['new_em_iterations']
    for h in heads:
        variant=next(x for x in p['variants'] if x['method']==h['method']);assert h['dimensions']==variant['dimensions'] and len(h['coefficients'])==h['dimensions']+1
        assert sha(PROJECT/h['cache_file'])==h['cache_sha256'] and h['gradient_inf']<=1e-9 and h['hessian_min_eigenvalue']>0
        if h['source_job']:assert h['objective']<=h['parent_objective']+1e-12
    for c in contexts:assert sha(PROJECT/c['cache_file'])==c['cache_sha256'] and c['gradient_inf']<=1e-7
    assert sum(len(h['coefficients']) for h in heads)==len(csv('coefficients'))==738
    alt=csv('independent_solver_verification');settings=p['independent_solver'];assert len(alt)==30
    assert alt.objective_absolute_gap.max()<=settings['objective_absolute_tolerance'] and alt.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance'] and alt.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    ha=csv('hmm_independent_verification');hs=p['independent_hmm'];assert len(ha)==6 and ha.prefix_checks.eq(3).all()
    assert ha.alternate_probability_gap.max()<=hs['filtered_probability_absolute_tolerance'] and ha.alternate_objective_gap.max()<=hs['objective_absolute_tolerance'] and ha.alternate_gradient_inf.max()<=hs['gradient_infinity_tolerance']
    assert ha.training_steps.sum()==10924 and ha.heldout_steps.sum()==1462
    forecasts=csv('forecast_verification');assert len(forecasts)==30 and forecasts.heldout_rows.sum()==1305 and forecasts.training_rows.sum()==47325
    parts=csv('interaction_components');assert len(parts)==1044;np.testing.assert_allclose(parts.logit,parts.refitted_additive_logit+parts.interaction_logit,rtol=0,atol=1e-12)
    models=csv('model_predictions');ensemble=csv('ensemble_predictions');metrics=csv('ensemble_metrics');years=csv('yearly_metrics');states=csv('hmm_state_metrics');signal=csv('hmm_signal_states');hmm=csv('hmm_fit_summary');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json')
    assert len(models)==8613 and len(ensemble)==5220 and len(metrics)==60 and ensemble.groupby('method').size().eq(261).all()
    for r in metrics.itertuples():
        g=ensemble[ensemble.method.eq(r.method)]
        if r.window=='early_2015_2017':g=g[g.year.le(2017)]
        if r.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==r.n and g.date.is_unique and g.direction_up.eq(g.actual_up).sum()==r.correct_directions
        if r.method=='native_mse':assert g.probability.isna().all()
        else:assert abs(np.square(g.probability-g.actual_up).mean()-r.brier)<1e-14 and r.clipped_probabilities==0
    assert len(pairs)==26
    for r,q in zip(pairs,[dict(window=w['name'],**q) for w in p['windows'] for q in p['primary_comparisons_per_window']]):
        for key,value in q.items():assert r[key]==value
    improvements=[r for r in pairs if r['difference']<0 and r['holm_adjusted_p']<.05];worsenings=[r for r in pairs if r['difference']>0 and r['holm_adjusted_p']<.05]
    assert len(improvements)==0 and len(worsenings)==2
    for a in assess:
        t=metrics[metrics.window.eq(a['window'])].set_index('method');cand=a['method'];r=t.loc[cand];w=next(w for w in p['windows'] if w['name']==a['window']);yr=years[years.year.between(int(w['start'][:4]),int(w['end'][:4]))]
        accuracy=yr.pivot(index='year',columns='method',values='accuracy');brier=yr.pivot(index='year',columns='method',values='brier')
        flags={f'accuracy_beats_{ref}':bool(r.accuracy>t.loc[ref,'accuracy']) for ref in [a['parent'],a['old_interaction'],'native_mse','training_frequency']}
        flags.update({f'brier_beats_{ref}':bool(r.brier<t.loc[ref,'brier']) for ref in [a['parent'],a['old_interaction'],'training_frequency','hmm_only']})
        flags.update(log_loss_beats_frequency=bool(r.log_loss<t.loc['training_frequency','log_loss']),two_years_beat_native_accuracy=bool((accuracy[cand]>accuracy.native_mse).sum()>=2),two_years_beat_frequency_brier=bool((brier[cand]<brier.training_frequency).sum()>=2))
        assert len(flags)==11 and a['flags']==flags and a['window_descriptive_pass']==all(flags.values()) and not a['cross_period_descriptive_pass'] and not a['strategy_promotion'] and a['interaction_hypothesis_retained']
    late=ensemble[ensemble.year.ge(2018)];a=late[late.method.eq('learned_hmm_interaction')].sort_values('date');b=late[late.method.eq('learned_market')].sort_values('date');np.testing.assert_array_equal(a.direction_up,b.direction_up)
    assert late[late.method.eq('hmm_only')].direction_up.all()
    assert (signal[signal.date.ge('2018-01-01')].high_probability>.8).sum()==12 and (signal[signal.date.lt('2018-01-01')].high_probability>.8).sum()==13
    assert signal[signal.date.between('2017-01-01','2017-12-31')].hmm_bucket.eq('low_probability').all()
    from report21 import LABELS,MAIN,WINDOWS,FAMILIES,metric_line,pair_line,line,pct,num
    report_path=OUT/'第二十一轮测试报告.md';report=report_path.read_text(encoding='utf-8');note=(ROOT/'README.md').read_text(encoding='utf-8')
    assert report.startswith('# 第二十一轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report and v['protocol_sha256'] in report and '保留全历史交互' in report and '58.16%' in note
    for text,folder in [(report,OUT),(note,ROOT)]:
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if not link.startswith(('http://','https://')):assert (folder/link).resolve().is_file(),link
    for r in metrics[metrics.window.ne('pooled_2015_2020')&metrics.method.isin(MAIN)].itertuples():assert line(metric_line(r)) in report
    for r in pairs:assert line(pair_line(r)) in report
    for year,g in years.groupby('year'):
        t=g.set_index('method')
        for family in FAMILIES:assert line([year,int(g.iloc[0]['n'])]+[int(t.loc[m,'correct_directions']) for m in family+['hmm_only','native_mse','training_frequency']]) in report
    for r in hmm.itertuples():
        assert line([int(r.cutoff[:4])+1,pct(r.low_daily_sd),pct(r.high_daily_sd),f'{r.stay_low:.4f}',f'{r.stay_high:.4f}',f'{r.implied_low_duration:.2f}',f'{r.implied_high_duration:.2f}',r.iterations]) in report
        g=signal[signal.cutoff.eq(r.cutoff)];assert line([int(r.cutoff[:4])+1,len(g),pct(g.high_probability.mean())]+[int(g.hmm_bucket.eq(s).sum()) for s in ['low_probability','uncertain','high_probability']]) in report
    for r in states[states.method.isin(['learned_market','learned_vol_interaction','learned_hmm_interaction','raw_hmm_interaction','hmm_only','training_frequency'])].itertuples():assert line([r.state,LABELS[r.method],r.n,pct(r.accuracy),num(r.brier),'是' if r.sparse else '否']) in report
    for (method,cutoff),g in csv('training_metrics').groupby(['method','cutoff']):assert line([LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())]) in report
    for a in assess:assert line([WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",'否','是']) in report
    visual=read(OUT/'visual_review.json');assert visual['status']=='PASS' and len(visual['files'])==2
    for name,digest in visual['files'].items():assert sha(ROOT/name)==digest and (ROOT/name).stat().st_size>50000
    for name in ['state_interaction_comparison','annual_filtered_states']:assert (OUT/f'{name}.svg').stat().st_size>10000
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve delivery';files={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_manifest.json'}
        result=dict(experiment=21,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),protocol_sha256=v['protocol_sha256'],previous_files_preserved=3061,
            new_hmm_fits=6,new_primary_fits=30,new_em_updates=fit['new_em_iterations'],new_newton_updates=fit['new_newton_iterations'],new_neural_forward_passes=0,neural_training_steps=0,new_projection_fits=24,
            independent_solutions=30,independent_hmm_checks=6,independent_holdout_dates=0,new_probability_forecasts=1305,model_records=8613,ensemble_records=5220,figures_visually_reviewed=2,
            cross_period_descriptive_pass={'learned_hmm_interaction':False,'raw_hmm_interaction':False},interaction_hypothesis_retained=True,hmm_interaction_adopted=False,significant_improvements=0,significant_deteriorations=2,files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if target.exists():
        delivery=read(target)
        for name,digest in delivery['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',frozen=target.exists(),hashed_files=len(read(target)['files']) if target.exists() else None,report_characters=len(report),old_files_preserved=3061,new_hmm_fits=6,new_classifier_fits=30,
        hmm_interaction_adopted=False,interaction_hypothesis_retained=True),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

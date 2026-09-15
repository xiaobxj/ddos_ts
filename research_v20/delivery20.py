"""Audit the complete delivery without training or changing old evidence."""
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
    expected=dict(previous_files_preserved=2958,new_primary_fits=48,independent_solutions=48,neural_states_with_inherited_round17_replay=18,
        interaction_transforms_with_inherited_round19_qr_verification=24,new_neural_forward_passes=0,neural_training_steps=0,new_projection_fits=0,
        training_feature_rows=31608,full_fold_membership_rows=9465,recent_fold_membership_rows=3951,nested_recent_interaction_checks=24,
        new_probability_forecasts=2088,new_deterministic_baseline_forecasts=261,reused_model_records=5220,model_records=7308,ensemble_records=4437,
        metric_groups=624,state_metric_rows=408,reliability_bins=160,primary_contrasts=32,factorial_diagnostics=8,independent_holdout_dates=0)
    assert v['status']=='PASS' and all(v[key]==val for key,val in expected.items())
    assert prep['protocol_sha256']==v['protocol_sha256']==sha(ROOT/'protocol.json') and prep['source_sha256']==v['source_sha256']
    assert len(prep['old_evidence'])==2958
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
    assert fitting['new_primary_fits']==48 and fitting['new_newton_iterations']==191 and fitting['all_converged'] and not fitting['validation_scoring_during_fit']
    assert fitting['new_projection_fits']==fitting['neural_forward_passes']==fitting['neural_training_steps']==0
    heads=read(OUT/'heads.json');assert len(heads)==48 and sum(h['iterations'] for h in heads)==191
    for h in heads:
        variant=next(r for r in p['variants'] if r['method']==h['method']);assert h['dimensions']==variant['dimensions'] and h['interaction']==variant['interaction']
        assert len(h['coefficients'])==h['dimensions']+1 and h['gradient_inf']<=1e-9 and h['l2_lambda']==.01
        assert h['objective']<=h['source_on_recent_objective']+1e-12 and sha(PROJECT/h['cache_file'])==h['cache_sha256']
        if h['interaction']:assert h['objective']<=h['nested_recent_additive_objective']+1e-12
    alt=pd.read_csv(OUT/'independent_solver_verification.csv');settings=p['independent_solver'];assert len(alt)==48
    assert alt.objective_absolute_gap.max()<=settings['objective_absolute_tolerance']
    assert alt.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance'] and alt.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    members=pd.read_csv(OUT/'window_membership.csv');folds=pd.read_csv(OUT/'fold_summaries.csv')
    assert len(members)==9465 and members.included_recent.sum()==3951 and folds.recent_train_n.tolist()==[720,590,596,595,725,725]
    for r in folds.itertuples():
        g=members[members.cutoff.eq(r.cutoff)];assert len(g)==r.full_train_n and g.included_recent.sum()==r.recent_train_n
        assert g[g.included_recent].date.ge(r.recent_start).all() and g.joint_completed.le(r.cutoff).all()
    coefficients=pd.read_csv(OUT/'coefficients.csv');assert len(coefficients)==1428
    parts=pd.read_csv(OUT/'interaction_components.csv');assert len(parts)==1044
    np.testing.assert_allclose(parts.logit,parts.refitted_additive_logit+parts.interaction_logit,rtol=0,atol=1e-12)
    dist=pd.read_csv(OUT/'interaction_input_distribution.csv');assert len(dist)==72
    states=pd.read_csv(OUT/'validation_states.csv');usage=pd.read_csv(OUT/'historical_date_usage.csv')
    assert len(states)==len(usage)==261 and states.date.is_unique and usage.previously_evaluated.all() and set(states.date)==set(usage.date)
    models=pd.read_csv(OUT/'model_predictions.csv');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv');metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    assert len(models)==7308 and len(ensemble)==4437 and len(metrics)==51 and ensemble.groupby('method').size().eq(261).all()
    for r in metrics.itertuples():
        g=ensemble[ensemble.method.eq(r.method)]
        if r.window=='early_2015_2017':g=g[g.year.le(2017)]
        if r.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==r.n and g.date.is_unique and int(g.direction_up.eq(g.actual_up).sum())==r.correct_directions
        if r.method=='native_mse':assert g.probability.isna().all() and pd.isna(r.brier)
        else:assert abs(np.square(g.probability-g.actual_up).mean()-r.brier)<1e-14 and r.clipped_probabilities==0
    frequency=ensemble[ensemble.method.eq('recent_frequency')];assert frequency.direction_up.all()
    pairs=read(OUT/'primary_comparisons.json');assert len(pairs)==32 and all(r['holm_adjusted_p']>=.05 for r in pairs)
    expected_pairs=[dict(window=w['name'],**q) for w in p['windows'] for q in p['primary_comparisons_per_window']]
    for pair,expected_pair in zip(pairs,expected_pairs):
        for key,val in expected_pair.items():assert pair[key]==val
    from report20 import LABELS,WINDOWS,MAIN,LEARNED,RAW,FULL,INTERACTIONS,pct,num
    years=pd.read_csv(OUT/'yearly_metrics.csv');assessments=read(OUT/'assessments.json');assert len(assessments)==4
    for a in assessments:
        t=metrics[metrics.window.eq(a['window'])].set_index('method');c=t.loc[a['method']];candidate=a['method'];old=FULL[candidate];parent=INTERACTIONS[candidate]
        w=next(w for w in p['windows'] if w['name']==a['window']);yr=years[years.year.between(int(w['start'][:4]),int(w['end'][:4]))]
        acc=yr.pivot(index='year',columns='method',values='accuracy');brier=yr.pivot(index='year',columns='method',values='brier')
        counts=dict(years_beating_native_accuracy=int((acc[candidate]>acc.native_mse).sum()),years_beating_recent_frequency_brier=int((brier[candidate]<brier.recent_frequency).sum()))
        flags=dict(accuracy_beats_full_interaction=bool(c.accuracy>t.loc[old,'accuracy']),accuracy_beats_recent_additive=bool(c.accuracy>t.loc[parent,'accuracy']),
            accuracy_beats_native=bool(c.accuracy>t.loc['native_mse','accuracy']),accuracy_beats_full_frequency=bool(c.accuracy>t.loc['training_frequency','accuracy']),
            accuracy_beats_recent_frequency=bool(c.accuracy>t.loc['recent_frequency','accuracy']),brier_beats_full_interaction=bool(c.brier<t.loc[old,'brier']),
            brier_beats_recent_additive=bool(c.brier<t.loc[parent,'brier']),brier_beats_full_frequency=bool(c.brier<t.loc['training_frequency','brier']),
            brier_beats_recent_frequency=bool(c.brier<t.loc['recent_frequency','brier']),log_loss_beats_recent_frequency=bool(c.log_loss<t.loc['recent_frequency','log_loss']),
            two_years_beat_native_accuracy=counts['years_beating_native_accuracy']>=2,two_years_beat_recent_frequency_brier=counts['years_beating_recent_frequency_brier']>=2)
        assert a['counts']==counts and a['flags']==flags and len(flags)==12 and a['window_descriptive_pass']==all(flags.values())
        assert not a['cross_period_descriptive_pass'] and not a['strategy_promotion'] and a['significant_improvements']==0 and a['interaction_hypothesis_retained']
    report_path=OUT/'第二十轮测试报告.md';report=report_path.read_text(encoding='utf-8');note=(ROOT/'README.md').read_text(encoding='utf-8')
    assert report.startswith('# 第二十轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report and v['protocol_sha256'] in report
    assert '保留全历史交互' in report and '58.16%' in note and '2026-09-12' in note
    for text,folder in [(report,OUT),(note,ROOT)]:
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if not link.startswith(('http://','https://')):assert (folder/link).resolve().is_file(),link
    for r in metrics[metrics.window.ne('pooled_2015_2020')&metrics.method.isin(MAIN)].itertuples():
        line='| '+' | '.join([LABELS[r.method],str(r.correct_directions),pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)])+' |';assert line in report,line
    for r in pairs:
        line='| '+' | '.join([WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',
            f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"])+' |';assert line in report,line
    for year,g in years.groupby('year'):
        for methods in [LEARNED,RAW]:
            t=g.set_index('method');line='| '+' | '.join([str(year),str(int(g.iloc[0].n))]+[pct(t.loc[m,'accuracy']) for m in methods+['native_mse','recent_frequency']])+' |';assert line in report
    training=pd.read_csv(OUT/'training_metrics.csv')
    for (method,cutoff),g in training.groupby(['method','cutoff']):
        line='| '+' | '.join([LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean()),num(g.source_on_recent_objective.mean())])+' |';assert line in report
    factorial=pd.read_csv(OUT/'factorial_differences.csv');assert len(factorial)==8
    for r in factorial.itertuples():
        line='| '+' | '.join([WINDOWS[r.window],'学习' if r.family=='learned' else '简单','Brier' if r.metric=='brier' else '方向错误率',
            f'{r.full_interaction_increment:+.6f}',f'{r.recent_interaction_increment:+.6f}',f'{r.difference_in_differences:+.6f}'])+' |';assert line in report
    for (method,cutoff),g in coefficients[coefficients.block.eq('interaction')].groupby(['method','cutoff']):
        recent=dist[dist.method.eq(method)&dist.cutoff.eq(cutoff)&dist.split.eq('recent_training')];test=dist[dist.method.eq(method)&dist.cutoff.eq(cutoff)&dist.split.eq('validation')]
        line='| '+' | '.join([LABELS[method],str(int(cutoff[:4])+1),num(g.full_window_coefficient.mean()),num(g.coefficient.mean()),num(recent.interaction_feature_mean.mean()),num(recent.interaction_feature_std.mean()),num(test.interaction_logit_mean.mean())])+' |';assert line in report
    for a in assessments:
        line='| '+' | '.join([WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/12",'否','是'])+' |';assert line in report
    y=years[years.year.eq(2018)].set_index('method')
    for method in ['learned_recent_additive','learned_recent_interaction']:
        assert round(y.loc[method,'predicted_up_fraction']*49)==47 and pct(y.loc[method,'mean_probability']) in report
    assert pct(y.loc['learned_recent_interaction','observed_up_fraction']) in report
    s=dist[dist.method.eq('learned_recent_interaction')&dist.cutoff.eq('2017-12-31')&dist.split.eq('validation')]
    for key in ['total_logit_mean','refitted_additive_mean','interaction_logit_mean']:assert f'{s[key].mean():.6f}' in report
    for name in ['window_interaction_comparison','annual_window_diagnostics']:
        assert (OUT/f'{name}.png').stat().st_size>50000 and (OUT/f'{name}.svg').stat().st_size>10000
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve delivery'
        files={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_manifest.json'}
        result=dict(experiment=20,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),protocol_sha256=v['protocol_sha256'],
            previous_files_preserved=2958,new_primary_fits=48,new_newton_updates=191,independent_solutions=48,new_neural_forward_passes=0,neural_training_steps=0,new_projection_fits=0,
            inherited_neural_replay_states=18,inherited_interaction_qr_verifications=24,independent_holdout_dates=0,new_probability_forecasts=2088,
            new_deterministic_baseline_forecasts=261,model_records=7308,ensemble_records=4437,figures_visually_reviewed=2,
            cross_period_descriptive_pass={'learned_recent_interaction':False,'raw_recent_interaction':False},interaction_hypothesis_retained=True,
            recent_hard_window_adopted=False,significant_improvements=0,files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    delivery=read(target)
    for name,digest in delivery['files'].items():assert sha(ROOT/name)==digest,name
    print(json.dumps(dict(status='PASS',hashed_files=len(delivery['files']),report_characters=len(report),previous_files_preserved=2958,
        independent_solutions=48,new_probability_forecasts=2088,interaction_hypothesis_retained=delivery['interaction_hypothesis_retained'],recent_hard_window_adopted=delivery['recent_hard_window_adopted']),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

"""Final report, phase and archive integrity checks; no model execution."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';PROJECT=ROOT.parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def hashes(mapping,root):
    for name,digest in mapping.items():assert sha(root/name)==digest,name

def main(freeze):
    prep=read(OUT/'preparation_manifest.json');p=read(ROOT/'protocol.json');v=read(OUT/'verification.json');assert v['status']=='PASS'
    expected=dict(previous_files_preserved=3475,previous_round_files_preserved=3389,failed_attempt_files_preserved=86,archived_primary_fits=48,total_round_primary_fit_executions=96,new_primary_fits=48,independent_solutions=48,new_projection_fits=0,reused_projection_fits=24,inherited_feature_split_checks=48,temporal_provenance_chains=24,new_hmm_fits=0,new_neural_forward_passes=0,neural_training_steps=0,inherited_neural_replay_states=18,training_feature_rows=75720,new_probability_forecasts=2088,reused_model_records=12528,model_records=14616,ensemble_records=8613,metric_groups=2130,state_metric_rows=1716,reliability_bins=320,primary_contrasts=64,independent_holdout_dates=0)
    for key,value in expected.items():assert v[key]==value,key
    assert v['attempt_forecasts_identical'] and prep['protocol_sha256']==v['protocol_sha256']==sha(ROOT/'protocol.json') and prep['source_sha256']==v['source_sha256']
    for key in ['old_evidence','source_sha256','input_sha256']:hashes(prep[key],PROJECT)
    assert len(prep['old_evidence'])==3475
    for record in [prep,v]:hashes(record['artifacts'],ROOT)
    c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and len(c['tests'])==5 and c['protocol_sha256']==prep['protocol_sha256'] and c['source_sha256']==prep['source_sha256'];hashes(c['artifacts'],ROOT)
    last=c['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        record=read(OUT/f'{phase}_manifest.json');assert last<=record['started_utc']<record['finished_utc'];last=record['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert record[key]==prep[key]
        hashes(record['artifacts'],ROOT)
    assert last<=v['completed_utc'];fit=read(OUT/'training_manifest.json');heads=read(OUT/'heads.json');assert fit['all_converged'] and not fit['validation_scoring_during_fit'] and len(heads)==48
    assert sum(h['iterations'] for h in heads)==fit['new_newton_iterations']==186 and sum(len(h['coefficients']) for h in heads)==1500
    for h in heads:
        variant=next(q for q in p['variants'] if q['method']==h['method']);assert h['multiplier']==variant['multiplier'] and h['order_lambda']==variant['order_lambda'] and h['dimensions']==variant['dimensions'];assert sha(PROJECT/h['cache_file'])==h['cache_sha256']
        assert h['gradient_inf']<=1e-9 and h['hessian_min_eigenvalue']>0 and h['objective']<=h['parent_objective']+1e-12
        theta=np.asarray(h['coefficients']);beta=np.asarray(h['original_coefficients']);theta[-2]/=np.sqrt(h['multiplier']);np.testing.assert_array_equal(theta,beta)
    settings=p['independent_solver'];alternate=csv('independent_solver_verification');assert len(alternate)==48
    assert alternate.objective_absolute_gap.max()<=settings['objective_absolute_tolerance'] and alternate.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance'] and alternate.alternate_gradient_inf.max()<=settings['gradient_infinity_tolerance']
    source=csv('temporal_provenance');assert len(source)==24 and source.supervised_features_training_in_sample.all() and not source.full_pipeline_oof.any() and not source.new_neural_execution.any()
    assert source.last_training_maturity.le(source.cutoff).all() and source.first_nextyear_signal.gt(source.cutoff).all()
    forecasts=csv('forecast_verification');assert len(forecasts)==48 and forecasts.training_rows.sum()==75720 and forecasts.heldout_rows.sum()==2088
    components=csv('interaction_components');assert len(components)==2088;np.testing.assert_allclose(components.logit,components.refitted_base_logit+components.original_interaction_logit+components.order_interaction_logit,rtol=0,atol=1e-12)
    coefs=csv('coefficients');assert len(coefs)==1500
    for block in ['intercept','order_interaction','original_interaction']:assert coefs.block.eq(block).sum()==48
    assert coefs[coefs.block.eq('original_interaction')].penalty.eq(.01).all() and coefs[coefs.block.eq('intercept')].penalty.eq(0).all()
    assert set(coefs[coefs.block.eq('order_interaction')].penalty)=={.04,.16}
    models=csv('model_predictions');ensemble=csv('ensemble_predictions');metrics=csv('ensemble_metrics');years=csv('yearly_metrics');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json');facts=read(OUT/'report_facts.json');path=csv('shrinkage_path');targets=csv('retention_targets')
    assert len(models)==14616 and len(ensemble)==8613 and len(metrics)==99 and len(years)==198 and ensemble.method.nunique()==33 and ensemble.groupby('method').size().eq(261).all()
    assert len(csv('state_metrics'))==1716 and len(csv('seed_metrics'))==117 and len(csv('reliability_bins'))==320
    for row in metrics.itertuples():
        g=ensemble[ensemble.method.eq(row.method)]
        if row.window=='early_2015_2017':g=g[g.year.le(2017)]
        if row.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==row.n and g.date.is_unique and g.direction_up.eq(g.actual_up).sum()==row.correct_directions
        if row.method=='native_mse':assert g.probability.isna().all()
        else:assert abs(np.square(g.probability-g.actual_up).mean()-row.brier)<1e-14 and row.clipped_probabilities==0
    archive=PROJECT/'research_v24_attempt1';preserved=read(archive/'attempt_preservation.json');hashes(preserved['files'],archive);assert len(preserved['files'])==85 and preserved['new_primary_fits']==48
    assert read(archive/'results/heads.json')==heads and read(archive/'results/training_manifest.json')['new_newton_iterations']+fit['new_newton_iterations']==372
    for name in ['model_predictions','new_model_predictions','ensemble_predictions']:
        old=pd.read_csv(archive/'results'/f'{name}.csv',float_precision='round_trip');pd.testing.assert_frame_equal(old,csv(name),check_dtype=False,rtol=0,atol=0)
    assert len(pairs)==64
    declared=[dict(window=w['name'],**q) for w in p['windows'] for q in p['primary_comparisons_per_window']]
    for row,definition in zip(pairs,declared):
        for key,value in definition.items():assert row[key]==value
    assert sum(r['difference']<0 and r['holm_adjusted_p']<.05 for r in pairs)==facts['significant_improvements']==0
    assert sum(r['difference']>0 and r['holm_adjusted_p']<.05 for r in pairs)==facts['significant_deteriorations']==0
    assert len(assess)==8
    for a in assess:
        t=metrics[metrics.window.eq(a['window'])].set_index('method');r=t.loc[a['method']];w=next(w for w in p['windows'] if w['name']==a['window']);yr=years[years.year.between(int(w['start'][:4]),int(w['end'][:4]))];acc=yr.pivot(index='year',columns='method',values='accuracy');b=yr.pivot(index='year',columns='method',values='brier');cand=a['method']
        flags={f'accuracy_beats_{ref}':bool(r.accuracy>t.loc[ref,'accuracy']) for ref in [a['parent'],a['additive_anchor'],'native_mse','training_frequency']};flags.update({f'brier_beats_{ref}':bool(r.brier<t.loc[ref,'brier']) for ref in [a['parent'],a['additive_anchor'],'training_frequency',a['control']]})
        flags.update(log_loss_beats_frequency=bool(r.log_loss<t.loc['training_frequency','log_loss']),two_years_beat_native_accuracy=bool((acc[cand]>acc.native_mse).sum()>=2),two_years_beat_frequency_brier=bool((b[cand]<b.training_frequency).sum()>=2))
        assert flags==a['flags'] and len(flags)==11 and a['window_descriptive_pass']==all(flags.values()) and not a['cross_period_descriptive_pass'] and not a['strategy_promotion'] and not a['independent_confirmation'] and a['interaction_hypothesis_retained']
    assert len(targets)==4 and not targets.all_targets_met.any() and facts['retention_target_passes']==0 and not targets.independent_confirmation.any()
    assert facts['newton_updates']==186 and facts['total_primary_fit_executions']==96 and facts['attempt_forecasts_identical']
    assert len(path)==24 and path.absolute_coefficient_monotone.all()
    for method,g in path.groupby('source_method'):
        for factor in [4,16]:
            ratios=abs(g[f'coefficient_{factor}x']/g.coefficient_1x);expected_ratio=dict(min=float(ratios.min()),mean=float(ratios.mean()),max=float(ratios.max()));assert facts['coefficient_ratios'][method][str(factor)]==expected_ratio
        assert (abs(g.coefficient_16x)<=abs(g.coefficient_4x)+1e-7).all() and (abs(g.coefficient_4x)<=abs(g.coefficient_1x)+1e-7).all()
    from report24 import LABELS,MAIN,WINDOWS,FAMILIES,metric_row,pair_row,target_row,line,pct,num,yes
    report_path=OUT/'第二十四轮测试报告.md';report=report_path.read_text(encoding='utf-8');note=(ROOT/'README.md').read_text(encoding='utf-8')
    assert report.startswith('# 第二十四轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report and v['protocol_sha256'] in report and '96 次主要拟合' in report and '58.16%' in note
    for text,folder in [(report,OUT),(note,ROOT)]:
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if not link.startswith(('http://','https://')):assert (folder/link).resolve().is_file(),link
    for row in metrics[metrics.window.ne('pooled_2015_2020')&metrics.method.isin(MAIN)].itertuples():assert line(metric_row(row)) in report
    for row in pairs:assert line(pair_row(row)) in report
    for row in targets.itertuples():assert line(target_row(row)) in report
    for year,g in years.groupby('year'):
        t=g.set_index('method')
        for family in FAMILIES:assert line([year,int(g.iloc[0]['n'])]+[int(t.loc[m,'correct_directions']) for m in family+['native_mse','training_frequency']]) in report
    changes=csv('direction_changes')
    for row in changes[changes.reference.isin([m for family in FAMILIES for m in family[:2]])].itertuples():assert line([WINDOWS[row.window],LABELS[row.method],LABELS[row.reference],row.changed,row.correct_to_wrong,row.wrong_to_correct]) in report
    for (method,cutoff),g in path.groupby(['source_method','cutoff']):assert line([LABELS[method],int(cutoff[:4])+1,num(g.coefficient_1x.mean()),num(g.coefficient_4x.mean()),num(g.coefficient_16x.mean())]) in report
    for method,values in facts['coefficient_ratios'].items():
        for factor,stats in values.items():assert line([LABELS[method],factor,pct(stats['min']),pct(stats['mean']),pct(stats['max'])]) in report
    summary=csv('component_summary')
    for method,g in summary[summary.split.eq('validation')].groupby('method'):assert line([LABELS[method],num(g.order_logit_sd.mean()),num(g.rms_probability_change_vs_parent.mean()),num(g.rms_probability_change_vs_weak.mean())]) in report
    for (method,cutoff),g in csv('training_metrics').groupby(['method','cutoff']):assert line([LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())]) in report
    for a in assess:assert line([WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",yes(a['cross_period_descriptive_pass'])]) in report
    visual=read(OUT/'visual_review.json');assert visual['status']=='PASS' and len(visual['files'])==2 and visual['completed_utc']>=v['completed_utc'];hashes(visual['files'],ROOT)
    for name in ['shrinkage_comparison','shrinkage_mechanism']:assert (OUT/f'{name}.png').stat().st_size>50000 and (OUT/f'{name}.svg').stat().st_size>10000
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve frozen delivery';files={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_manifest.json'}
        result=dict(experiment=24,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),protocol_sha256=v['protocol_sha256'],previous_files_preserved=3475,previous_round_files_preserved=3389,failed_attempt_files_preserved=86,new_primary_fits=48,total_round_primary_fit_executions=96,new_newton_updates=186,total_round_primary_newton_updates=372,attempt_forecasts_identical=True,new_projection_fits=0,reused_projection_fits=24,independent_solutions=48,new_neural_forward_passes=0,neural_training_steps=0,new_hmm_fits=0,independent_holdout_dates=0,new_probability_forecasts=2088,model_records=14616,ensemble_records=8613,figures_visually_reviewed=2,retention_target_passes=0,cross_period_descriptive_pass={a['method']:False for a in assess},interaction_hypothesis_retained=True,previous_order_partial_improvement_retained=True,new_shrinkage_variant_adopted=False,significant_improvements=0,significant_deteriorations=0,files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if target.exists():hashes(read(target)['files'],ROOT)
    print(json.dumps(dict(status='PASS',frozen=target.exists(),hashed_files=len(read(target)['files']) if target.exists() else None,preserved_evidence_files=3475,report_characters=len(report),final_heads=48,total_primary_fit_executions=96,interaction_hypothesis_retained=True,new_shrinkage_variant_adopted=False),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

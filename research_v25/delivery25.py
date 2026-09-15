"""Final artifact and report checks; no model or neural execution."""
from pathlib import Path
import sys,json,argparse,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
sys.path.insert(0,str(PROJECT/'research_v24'));from delivery24 import sha,read,hashes
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')

def main(freeze):
    prep=read(OUT/'preparation_manifest.json');p=read(ROOT/'protocol.json');v=read(OUT/'verification.json');assert v['status']=='PASS'
    expected=dict(previous_files_preserved=3585,new_primary_fits=24,independent_roots=24,new_trainable_coefficients=24,frozen_parent_coefficients=726,new_projection_fits=0,reused_projection_fits=24,inherited_feature_split_checks=48,temporal_provenance_chains=24,new_hmm_fits=0,new_neural_forward_passes=0,neural_training_steps=0,inherited_neural_replay_states=18,training_feature_rows=37860,new_probability_forecasts=1044,reused_model_records=14616,model_records=15660,ensemble_records=9135,metric_groups=2261,state_metric_rows=1820,reliability_bins=340,primary_contrasts=32,independent_holdout_dates=0)
    for key,value in expected.items():assert v[key]==value,key
    assert v['parent_coefficients_bitwise_unchanged'] and v['parent_intercepts_bitwise_unchanged']
    assert prep['protocol_sha256']==v['protocol_sha256']==sha(ROOT/'protocol.json') and prep['source_sha256']==v['source_sha256']
    for key in ['old_evidence','source_sha256','input_sha256']:hashes(prep[key],PROJECT)
    assert len(prep['old_evidence'])==3585
    for record in [prep,v]:hashes(record['artifacts'],ROOT)
    c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and len(c['tests'])==5 and c['protocol_sha256']==prep['protocol_sha256'] and c['source_sha256']==prep['source_sha256'];hashes(c['artifacts'],ROOT)
    last=c['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        record=read(OUT/f'{phase}_manifest.json');assert last<=record['started_utc']<record['finished_utc'];last=record['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert record[key]==prep[key]
        hashes(record['artifacts'],ROOT)
    assert last<=v['completed_utc'];fit=read(OUT/'training_manifest.json');heads=read(OUT/'heads.json');assert fit['all_converged'] and not fit['validation_scoring_during_fit'] and len(heads)==24
    assert sum(h['iterations'] for h in heads)==fit['new_newton_iterations']==66 and sum(len(h['coefficients']) for h in heads)==750
    parents={h['job']:h for h in read(PROJECT/'research_v19/results/heads.json')};sources={h['job']:h for h in read(OUT/'source_heads.json')}
    for h in heads:
        parent=parents[h['parent_job']];source=sources[h['source_job']];theta=np.asarray(h['coefficients']);pt=np.asarray(parent['coefficients'])
        assert sha(PROJECT/h['cache_file'])==h['cache_sha256'] and h['free_parameters']==1 and h['free_intercepts']==0 and h['order_lambda']==.01
        np.testing.assert_array_equal(h['frozen_parent_coefficients'],pt);np.testing.assert_array_equal(theta,np.r_[pt[:-1],h['gamma'],pt[-1]])
        assert h['gradient_absolute']<=1e-10 and h['hessian']>=.01 and h['parent_coefficient_l2_drift']==h['parent_intercept_drift']==0
        assert source['objective']<=h['objective']+1e-12 and h['objective']<=parent['objective']+1e-12
        assert abs(h['objective']-h['reduced_objective']-h['parent_penalty_constant'])<1e-12
        assert abs(h['parent_penalty_constant']-.005*np.square(pt[:-1]).sum())<1e-14
    roots=csv('independent_solver_verification');settings=p['independent_solver'];assert len(roots)==24 and roots.root_converged.all()
    assert roots.coefficient_absolute_gap.max()<=settings['coefficient_absolute_tolerance'] and roots.objective_absolute_gap.max()<=settings['objective_absolute_tolerance'] and roots.maximum_training_probability_gap.max()<=settings['training_probability_absolute_tolerance'] and roots.alternate_gradient_absolute.max()<=settings['gradient_absolute_tolerance']
    provenance=csv('temporal_provenance');assert len(provenance)==24 and provenance.last_training_maturity.le(provenance.cutoff).all() and provenance.first_nextyear_signal.gt(provenance.cutoff).all() and not provenance.full_pipeline_oof.any() and not provenance.new_neural_execution.any()
    forecasts=csv('forecast_verification');assert len(forecasts)==24 and forecasts.training_rows.sum()==37860 and forecasts.heldout_rows.sum()==1044
    coefs=csv('coefficients');assert len(coefs)==750 and coefs.trainable.sum()==24 and (~coefs.trainable).sum()==726
    assert coefs.trainable.eq(coefs.block.eq('order_interaction')).all()
    for block in ['intercept','order_interaction','original_interaction']:assert coefs.block.eq(block).sum()==24
    parts=csv('interaction_components');assert len(parts)==1044
    np.testing.assert_allclose(parts.frozen_parent_logit,parts.frozen_base_logit+parts.frozen_original_interaction_logit,rtol=0,atol=1e-12)
    np.testing.assert_allclose(parts.logit,parts.frozen_parent_logit+parts.order_interaction_logit,rtol=0,atol=1e-12)
    np.testing.assert_allclose(parts.total_logit_difference,parts.parent_constraint_component+parts.order_coefficient_component,rtol=0,atol=1e-12)
    np.testing.assert_allclose(parts.total_logit_difference,parts.logit-parts.joint_logit,rtol=0,atol=1e-12)
    models=csv('model_predictions');ensemble=csv('ensemble_predictions');metrics=csv('ensemble_metrics');years=csv('yearly_metrics');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json');facts=read(OUT/'report_facts.json');targets=csv('retention_targets');summary=csv('component_summary')
    assert len(models)==15660 and len(ensemble)==9135 and len(metrics)==105 and len(years)==210 and ensemble.method.nunique()==35 and ensemble.groupby('method').size().eq(261).all()
    assert len(csv('seed_metrics'))==126 and len(csv('state_metrics'))==1820 and len(csv('reliability_bins'))==340 and len(summary)==48
    assert summary.parent_coefficient_l2_drift.eq(0).all() and summary.parent_intercept_drift.eq(0).all()
    for row in metrics.itertuples():
        g=ensemble[ensemble.method.eq(row.method)]
        if row.window=='early_2015_2017':g=g[g.year.le(2017)]
        if row.window=='late_2018_2020':g=g[g.year.ge(2018)]
        assert len(g)==row.n and g.date.is_unique and g.direction_up.eq(g.actual_up).sum()==row.correct_directions
        if row.method=='native_mse':assert g.probability.isna().all()
        else:assert abs(np.square(g.probability-g.actual_up).mean()-row.brier)<1e-14 and row.clipped_probabilities==0
    assert len(pairs)==32
    declared=[dict(window=w['name'],**q) for w in p['windows'] for q in p['primary_comparisons_per_window']]
    for row,definition in zip(pairs,declared):
        for key,value in definition.items():assert row[key]==value
    assert sum(r['difference']<0 and r['holm_adjusted_p']<.05 for r in pairs)==facts['significant_improvements']==0
    assert sum(r['difference']>0 and r['holm_adjusted_p']<.05 for r in pairs)==facts['significant_deteriorations']==0
    assert len(assess)==4
    for a in assess:
        t=metrics[metrics.window.eq(a['window'])].set_index('method');r=t.loc[a['method']];w=next(w for w in p['windows'] if w['name']==a['window']);yr=years[years.year.between(int(w['start'][:4]),int(w['end'][:4]))];acc=yr.pivot(index='year',columns='method',values='accuracy');b=yr.pivot(index='year',columns='method',values='brier');candidate=a['method']
        flags={f'accuracy_beats_{ref}':bool(r.accuracy>t.loc[ref,'accuracy']) for ref in [a['parent'],a['additive_anchor'],'native_mse','training_frequency']};flags.update({f'brier_beats_{ref}':bool(r.brier<t.loc[ref,'brier']) for ref in [a['parent'],a['additive_anchor'],'training_frequency',a['control']]})
        flags.update(log_loss_beats_frequency=bool(r.log_loss<t.loc['training_frequency','log_loss']),two_years_beat_native_accuracy=bool((acc[candidate]>acc.native_mse).sum()>=2),two_years_beat_frequency_brier=bool((b[candidate]<b.training_frequency).sum()>=2))
        assert flags==a['flags'] and len(flags)==11 and a['window_descriptive_pass']==all(flags.values()) and not a['cross_period_descriptive_pass'] and not a['strategy_promotion'] and not a['independent_confirmation'] and a['interaction_hypothesis_retained']
    assert len(targets)==2 and not targets.all_targets_met.any() and facts['retention_target_passes']==0 and not targets.independent_confirmation.any()
    assert facts['newton_updates']==66 and facts['parent_coefficients_unchanged'] and facts['parent_intercepts_unchanged']
    assert facts['minimum_joint_fit_advantage']==min(h['objective']-h['joint_objective'] for h in heads) and facts['maximum_joint_fit_advantage']==max(h['objective']-h['joint_objective'] for h in heads)
    from report25 import LABELS,MAIN,WINDOWS,FAMILIES,metric_row,pair_row,target_row,line,pct,num,yes
    details=[]
    for parent,joint,fixed in FAMILIES:
        a=ensemble[ensemble.method.eq(fixed)].set_index('date');b=ensemble[ensemble.method.eq(joint)].set_index('date');c=ensemble[ensemble.method.eq(parent)].set_index('date')
        np.testing.assert_array_equal(a[a.year.ge(2018)].direction_up,b[b.year.ge(2018)].direction_up)
        for date in a.index[a.direction_up.ne(b.direction_up)]:details.append(dict(method=fixed,date=date,actual_up=int(a.loc[date,'actual_up']),parent_probability=float(c.loc[date,'probability']),joint_probability=float(b.loc[date,'probability']),fixed_probability=float(a.loc[date,'probability']),joint_correct=bool(b.loc[date,'direction_up']==a.loc[date,'actual_up']),fixed_correct=bool(a.loc[date,'direction_up']==a.loc[date,'actual_up'])))
    pd.testing.assert_frame_equal(pd.DataFrame(details),csv('changed_signal_details'),check_dtype=False,rtol=0,atol=0);assert len(details)==facts['changed_signals_vs_joint']==2
    report_path=OUT/'第二十五轮测试报告.md';report=report_path.read_text(encoding='utf-8');note=(ROOT/'README.md').read_text(encoding='utf-8')
    assert report.startswith('# 第二十五轮方法测试：') and '\ufffd' not in report and '不能作为独立确认' in report and v['protocol_sha256'] in report and '726' in report and '58.16%' in note
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
    for row in details:assert line([LABELS[row['method']],row['date'],'涨' if row['actual_up'] else '非涨',num(row['parent_probability']),num(row['joint_probability']),num(row['fixed_probability']),yes(row['joint_correct']),yes(row['fixed_correct'])]) in report
    for (method,cutoff),g in summary[summary.split.eq('training')].groupby(['method','cutoff']):assert line([LABELS[method],int(cutoff[:4])+1,num(g.joint_gamma.mean()),num(g.offset_gamma.mean()),f'{g.gamma_difference.mean():+.6f}']) in report
    for method,g in summary[summary.split.eq('validation')].groupby('method'):assert line([LABELS[method],num(g.parent_constraint_rms.mean()),num(g.order_coefficient_change_rms.mean()),num(g.total_logit_difference_rms.mean()),num(g.rms_probability_change_vs_parent.mean()),num(g.rms_probability_change_vs_joint.mean())]) in report
    for (method,cutoff),g in csv('training_metrics').groupby(['method','cutoff']):assert line([LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())]) in report
    for a in assess:assert line([WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",yes(a['cross_period_descriptive_pass'])]) in report
    visual=read(OUT/'visual_review.json');assert visual['status']=='PASS' and len(visual['files'])==2 and visual['completed_utc']>=v['completed_utc'];hashes(visual['files'],ROOT)
    for name in ['fixed_parent_comparison','fixed_parent_diagnostics']:assert (OUT/f'{name}.png').stat().st_size>50000 and (OUT/f'{name}.svg').stat().st_size>10000
    target=OUT/'delivery_manifest.json'
    if freeze:
        assert not target.exists(),'Preserve frozen delivery';files={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_manifest.json'}
        result=dict(experiment=25,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),report=str(report_path.relative_to(ROOT)),protocol_sha256=v['protocol_sha256'],previous_files_preserved=3585,new_primary_fits=24,new_trainable_coefficients=24,frozen_parent_coefficients=726,new_newton_updates=66,independent_roots=24,parent_coefficients_unchanged=True,parent_intercepts_unchanged=True,new_projection_fits=0,reused_projection_fits=24,new_neural_forward_passes=0,neural_training_steps=0,new_hmm_fits=0,independent_holdout_dates=0,new_probability_forecasts=1044,model_records=15660,ensemble_records=9135,figures_visually_reviewed=2,retention_target_passes=0,cross_period_descriptive_pass={a['method']:False for a in assess},interaction_hypothesis_retained=True,previous_order_partial_improvement_retained=True,current_early_partial_improvement_retained=True,new_offset_variant_adopted=False,significant_improvements=0,significant_deteriorations=0,files=files)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if target.exists():hashes(read(target)['files'],ROOT)
    print(json.dumps(dict(status='PASS',frozen=target.exists(),hashed_files=len(read(target)['files']) if target.exists() else None,preserved_evidence_files=3585,report_characters=len(report),new_scalar_fits=24,parent_coefficients_unchanged=True,parent_intercepts_unchanged=True,new_offset_variant_adopted=False),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');main(parser.parse_args().freeze)

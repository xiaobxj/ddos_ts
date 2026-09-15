from common16 import *


def main():
    assert not (OUT/'preparation_manifest.json').exists(),'Preserve preparation'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence()
    prior_protocol=read(V15/'protocol.json');assert cfg()['probe']==prior_protocol['probe']
    for key in ['method','options','objective_absolute_tolerance','training_probability_absolute_tolerance','gradient_infinity_tolerance']:
        assert cfg()['independent_solver'][key]==prior_protocol['independent_solver'][key]
    assert cfg()['standardization']['sd_floor']==1e-6
    assert list(HORIZONS)==cfg()['simple_features']['horizons'] and list(KINDS)==cfg()['simple_features']['columns_per_horizon']
    shutil.copyfile(V15/'results/observation_table.csv',OUT/'observation_table.csv');obs,price,returns=data()
    records=[];metadata={};training=[]
    for fold in cfg()['folds']:
        tr,te=indices(obs,fold);rate=float((returns[tr]>0).mean());metadata[fold['cutoff']]=dict(train_n=len(tr),frequency=rate,up_count=int((returns[tr]>0).sum()))
        for i in tr:training.append(dict(cutoff=fold['cutoff'],row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i],direction=int(returns[i]>0)))
        for i in te:records.append(dict(cutoff=fold['cutoff'],row_index=int(i),date=obs.date.iloc[i],year=int(obs.date.iloc[i][:4]),
            joint_completed=obs.joint_completed.iloc[i],actual_return=float(returns[i]),actual_up=int(returns[i]>0),training_frequency=rate))
    base=pd.DataFrame(records);assert len(base)==261 and base.date.is_unique
    base.to_csv(OUT/'classification_baselines.csv',index=False);pd.DataFrame(training).to_csv(OUT/'training_rows.csv',index=False)
    save(OUT/'training_metadata.json',metadata);save(OUT/'feature_names.json',NAMES)
    earlier=read(V13/'results/inner_models.json');late=[h for h in read(V15/'results/heads.json') if h['family']=='probe_mse']
    assert len(earlier)==len(late)==9
    backbones=[];jobs=[];reused=[]
    for ref in earlier+late:
        r=dict(ref,kind='learned',method='learned_probe',job=job_id('learned',ref['cutoff'],ref['seed']),reused=ref['cutoff']>='2017-12-31')
        if r['reused']:
            r.update(cache_file=str((V15/ref['feature_file']).relative_to(PROJECT)),cache_sha256=ref['feature_sha256']);reused.append(r)
        else:jobs.append(r)
        backbones.append(r)
    for fold in cfg()['folds']:
        jobs.append(dict(kind='raw',method='raw25_probe',job=job_id('raw',fold['cutoff']),cutoff=fold['cutoff'],seed=-1,reused=False))
    assert len(jobs)==15 and len(reused)==9 and len(backbones)==18
    save(OUT/'backbones.json',backbones);save(OUT/'new_jobs.json',jobs);save(OUT/'reused_heads.json',reused)
    rolling=pd.read_csv(V13/'results/rolling_seed_predictions.csv');old15=pd.read_csv(V15/'results/all_seed_predictions.csv')
    old=[];old_probe=[]
    for ref in backbones:
        g=base[base.cutoff.eq(ref['cutoff'])].copy()
        if ref['reused']:
            reference=old15[old15.method.eq('archived20')&old15.cutoff.eq(ref['cutoff'])&old15.seed.eq(ref['seed'])].sort_values('row_index')
            p=reference.score.to_numpy()
        else:
            reference=rolling[rolling.inner_cutoff.eq(ref['cutoff'])&rolling.seed.eq(ref['seed'])].sort_values('row_index')
            p=reference.predicted_return.to_numpy()
        np.testing.assert_array_equal(reference.row_index,g.row_index)
        g['method']='native_mse';g['seed']=ref['seed'];g['score']=p;g['probability']=np.nan;g['logit']=np.nan;g['direction_up']=p>0;old.append(g)
        if ref['reused']:
            reference=old15[old15.method.eq('probe_mse')&old15.cutoff.eq(ref['cutoff'])&old15.seed.eq(ref['seed'])].sort_values('row_index')
            np.testing.assert_array_equal(reference.row_index,g.row_index)
            h=base[base.cutoff.eq(ref['cutoff'])].copy();h['method']='learned_probe';h['seed']=ref['seed'];h['score']=reference.score.to_numpy()
            h['probability']=reference.probability.to_numpy();h['logit']=reference.probe_logit.to_numpy();h['direction_up']=reference.direction_up.to_numpy();old_probe.append(h)
    native=pd.concat(old,ignore_index=True);cached=pd.concat(old_probe,ignore_index=True)
    assert len(native)==783 and len(cached)==423
    native.to_csv(OUT/'native_seed_reference.csv',index=False);cached.to_csv(OUT/'reused_probe_seed_reference.csv',index=False)
    sources={'round13_rolling':rolling,'round15_validation':old15,
        'round2_validation':pd.read_csv(PROJECT/'research_v2/results/validation_predictions.csv'),
        'round3_validation':pd.read_csv(PROJECT/'research_v3/results/validation_grid_predictions.csv'),
        'round1_later_development':pd.read_csv(PROJECT/'research/results/predictions.csv')}
    audit=base[['cutoff','row_index','date','year','joint_completed']].copy();inventory=[]
    for name,g in sources.items():
        dates=set(g.date);audit['seen_'+name]=audit.date.isin(dates)
        inventory.append(dict(source=name,unique_dates=len(dates),first_date=min(dates),last_date=max(dates),overlap_with_this_round=int(audit.date.isin(dates).sum())))
    audit['previously_evaluated']=audit[[c for c in audit if c.startswith('seen_')]].any(axis=1)
    assert audit.previously_evaluated.all();assert audit[audit.year.le(2017)].seen_round13_rolling.all()
    assert audit[audit.year.ge(2018)].seen_round15_validation.all()
    audit.to_csv(OUT/'historical_date_usage.csv',index=False)
    save(OUT/'historical_usage_audit.json',dict(status='ALL_DATES_PREVIOUSLY_INSPECTED',sources=inventory,early_dates=120,late_dates=141,
        independent_holdout_dates=0,notes=['Earlier120weeks already supplied causal forecasts/calibration in round13.',
        'Later141weeks already informed previous model and research choices.','2021-2026development outcomes were also inspected previously; no new later forecasts here.',
        'Year-end models satisfy chronological label maturity, but research design is retrospective.']))
    files=[OUT/n for n in ['observation_table.csv','classification_baselines.csv','training_rows.csv','training_metadata.json','feature_names.json',
        'backbones.json','new_jobs.json','reused_heads.json','native_seed_reference.csv','reused_probe_seed_reference.csv','historical_date_usage.csv','historical_usage_audit.json']]
    finish(run,files,new_primary_fits=15,reused_heads=9,neural_training_steps=0,previous_files_preserved=2618,validation_weeks=261)
    print(json.dumps(dict(status='FROZEN',new_heads=15,reused_heads=9,previous_files=2618,previously_inspected_weeks=261,protocol_sha256=run['protocol_sha256'])),flush=True)


if __name__=='__main__':main()

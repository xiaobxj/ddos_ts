from common13 import *

def main():
    legacy.initialize();check_frozen();training=read(OUT/'training_manifest.json')
    assert training.get('finished_utc') and training['fits']==9
    path=OUT/'rolling_manifest.json';assert not path.exists(),'Preserve rolling forecasts'
    run=manifest('rolling_forecasts');save(path,run)
    plan=pd.read_csv(OUT/'rolling_plan.csv');obs=pd.read_csv(OUT/'observation_table.csv')
    archived=pd.read_csv(OUT/'archived_predictions.csv');rows=[]
    refs=read(OUT/'inner_models.json')+[r for r in read(OUT/'archived_models.json') if r['cutoff'] in cfg()['inner_cutoffs']]
    for ref in refs:
        cutoff=ref['cutoff'];seed=ref['seed'];g=plan[plan.inner_cutoff.eq(cutoff)].copy()
        if cutoff in cfg()['new_training_cutoffs']:
            model,state=load_model(ref);est=forecast(model,g.row_index.to_numpy(),state['scales']);del model
        else:
            old=archived[archived.cutoff.eq(cutoff)&archived.seed.eq(seed)].sort_values('row_index')
            np.testing.assert_array_equal(old.row_index,g.row_index);est=old.predicted_return.to_numpy(float)
        g['seed']=seed;g['predicted_return']=est;g['actual']=obs.exec_return.iloc[g.row_index].to_numpy()
        g['model_project_file']=ref['project_file'];g['model_sha256']=ref['model_sha256'];rows.append(g)
    result=pd.concat(rows,ignore_index=True).sort_values(['date','seed'])
    assert len(result)==645 and result.groupby('date').size().eq(3).all()
    result.to_csv(OUT/'rolling_seed_predictions.csv',index=False)
    ensemble=result.groupby('date',sort=True).agg(inner_cutoff=('inner_cutoff','first'),row_index=('row_index','first'),
        joint_completed=('joint_completed','first'),training_mean=('training_mean','first'),source=('source','first'),
        predicted_return=('predicted_return','mean'),actual=('actual','first')).reset_index()
    ensemble.to_csv(OUT/'rolling_ensemble_predictions.csv',index=False)
    run.update(finished_utc=now(),models=15,rolling_dates=215,seed_forecasts=645,new_seed_forecasts=360,reused_seed_forecasts=285,
        artifacts={n:sha(OUT/n) for n in ['rolling_seed_predictions.csv','rolling_ensemble_predictions.csv']})
    save(path,run);print(json.dumps(dict(status='PASS',new_forecasts=360,reused_forecasts=285,dates=215)),flush=True)

if __name__=='__main__':main()

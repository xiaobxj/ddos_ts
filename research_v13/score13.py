from common13 import *

def main():
    check_frozen();cal=read(OUT/'calibration_manifest.json');assert cal.get('finished_utc') and cal['coefficients']==3
    for n,d in cal['artifacts'].items():assert sha(ROOT/n)==d,n
    path=OUT/'scoring_manifest.json';assert not path.exists(),'Preserve outer transformations'
    run=manifest('outer_transform');save(path,run)
    coefficients={r['outer_cutoff']:r['alpha'] for r in read(OUT/'coefficients.json')}
    old=pd.read_csv(OUT/'archived_predictions.csv');rows=[]
    assert len(old)==423 and old.groupby('date').size().eq(3).all()
    for method in cfg()['methods']:
        g=old[['cutoff','seed','row_index','date','actual','predicted_return','training_mean']].copy()
        g['method']=method['name'];g['original_prediction']=g.predicted_return
        g['alpha']=g.cutoff.map(coefficients) if method['name']=='rolling_shrink' else method['alpha']
        g['predicted_return']=transform(g.original_prediction,g.training_mean,g.alpha.to_numpy())
        rows.append(g)
    result=pd.concat(rows,ignore_index=True);assert len(result)==1269
    result.to_csv(OUT/'outer_seed_predictions.csv',index=False)
    ensemble=result.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),
        actual=('actual','first'),predicted_return=('predicted_return','mean'),training_mean=('training_mean','first'),
        original_prediction=('original_prediction','mean'),alpha=('alpha','first')).reset_index()
    ensemble.to_csv(OUT/'outer_ensemble_predictions.csv',index=False)
    run.update(finished_utc=now(),seed_predictions=1269,outer_weeks=141,coefficient_manifest_sha256=sha(OUT/'calibration_manifest.json'),
        artifacts={n:sha(OUT/n) for n in ['outer_seed_predictions.csv','outer_ensemble_predictions.csv']})
    save(path,run);print('Frozen outer transforms: 3 methods x 423 seed predictions.',flush=True)

if __name__=='__main__':main()

"""Predetermined epoch40 forward passes, only after every training path is finished."""
from common12 import *


def main():
    legacy.initialize();check_frozen()
    training=read(OUT/'continuation_manifest.json');prefix=read(OUT/'reconstruction_manifest.json')
    assert training.get('finished_utc') and training['continuations']==18
    assert prefix.get('finished_utc') and prefix['exact_matches']==9
    assert not (OUT/'scoring_manifest.json').exists(),'Preserve previous scoring'
    start=time.time();run=manifest('fixed_endpoint_forward_passes');save(OUT/'scoring_manifest.json',run)
    rows=[];truth=pd.read_csv(OUT/'validation_rows.csv')
    refs=read(OUT/'final_models.json');assert len(refs)==18
    for ref in refs:
        model,state=load_model(ref);values,te=load_validation(ref['cutoff']);model.eval()
        with torch.inference_mode():
            p,_=legacy.predict(model,values,False)
            altered,_=legacy.predict(model,{k:torch.roll(t,1,dims=0) for k,t in values.items()},False)
        scales=state['scales']
        p=p.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
        altered=altered.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
        g=truth[truth.cutoff.eq(ref['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,te)
        g['seed']=ref['seed'];g['schedule']=ref['schedule'];g['epoch']=40
        g['predicted_return']=p;g['reassigned_prediction']=altered;rows.extend(g.to_dict('records'))
        del model,values;torch.cuda.empty_cache()
    new=pd.DataFrame(rows);assert len(new)==846
    new.to_csv(OUT/'new_predictions.csv',index=False)
    all_=pd.concat([pd.read_csv(OUT/'archived_predictions.csv'),new],ignore_index=True)
    assert len(all_)==1269 and all_.groupby(['schedule','date']).size().eq(3).all()
    all_.to_csv(OUT/'all_seed_predictions.csv',index=False)
    check_frozen()
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),new_models=18,new_predictions=846,
        reused_predictions=423,total_predictions=1269,elapsed_seconds=time.time()-start)
    save(OUT/'scoring_manifest.json',run)
    print(json.dumps(dict(status='SCORED',new_models=18,new_predictions=846,historical_weeks=141)))


if __name__=='__main__':main()

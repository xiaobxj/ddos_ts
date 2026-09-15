from common14 import *

def main():
    legacy.initialize();check_frozen();training=read(OUT/'training_manifest.json')
    assert training.get('finished_utc') and training['fits']==9
    for name,digest in training['artifacts'].items():assert sha(OUT/name)==digest,name
    path=OUT/'scoring_manifest.json';assert not path.exists(),'Preserve scores'
    run=manifest('scoring');save(path,run)
    baselines=pd.read_csv(OUT/'classification_baselines.csv');rows=[]
    for ref in read(OUT/'models.json'):
        model,state=load_model(ref);values,te=load_validation(ref['cutoff'])
        with torch.inference_mode():z,_=legacy.predict(model,values,False)
        logits=z.cpu().numpy().astype(float);p=probability(logits)
        g=baselines[baselines.cutoff.eq(ref['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,te)
        g['method']='direction_bce';g['seed']=ref['seed'];g['logit']=logits;g['probability']=p;g['score']=p;g['direction_up']=p>.5
        rows.append(g);del model,values;torch.cuda.empty_cache()
    new=pd.concat(rows,ignore_index=True);assert len(new)==423
    new.to_csv(OUT/'classification_seed_predictions.csv',index=False)
    old=pd.read_csv(OUT/'archived_predictions.csv')
    old=old[['cutoff','seed','row_index','date','predicted_return']].merge(baselines,on=['cutoff','row_index','date'],validate='many_to_one')
    old['method']='archived20';old['probability']=np.nan;old['score']=old.predicted_return;old['direction_up']=old.score>0
    seed=pd.concat([new,old.drop(columns='predicted_return')],ignore_index=True)
    seed.to_csv(OUT/'all_seed_predictions.csv',index=False)
    frames=[]
    for method,g in seed.groupby('method'):
        h=g.groupby('date',sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),actual_return=('actual_return','first'),
            actual_up=('actual_up','first'),training_frequency=('training_frequency','first'),probability=('probability','mean'),score=('score','mean')).reset_index()
        h['method']=method;h['direction_up']=h.score>(0 if method=='archived20' else .5);frames.append(h)
    for method in ['training_frequency','neutral_50']:
        h=baselines.copy();h['method']=method;h['probability']=h.training_frequency if method=='training_frequency' else .5
        h['score']=h.probability;h['direction_up']=h.probability>.5;frames.append(h)
    ensemble=pd.concat(frames,ignore_index=True);assert len(ensemble)==564
    ensemble.to_csv(OUT/'all_ensemble_predictions.csv',index=False)
    run.update(finished_utc=now(),new_seed_predictions=423,reused_seed_predictions=423,ensemble_weeks=141,
        artifacts={n:sha(OUT/n) for n in ['classification_seed_predictions.csv','all_seed_predictions.csv','all_ensemble_predictions.csv']})
    save(path,run);print('Scoring complete: 423 classification probabilities; 4 ensemble methods on 141 weeks.',flush=True)

if __name__=='__main__':main()

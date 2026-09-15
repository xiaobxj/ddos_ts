from common17 import *

def main():
    check_frozen();fitting=check_phase('training');assert fitting['all_converged'] and fitting['new_primary_fits']==24
    path=OUT/'scoring_manifest.json';assert not path.exists(),'Preserve scoring';run=manifest('scoring');save(path,run)
    obs,price,returns=data();bars=previous.raw_bars(price);base=pd.read_csv(OUT/'classification_baselines.csv',float_precision='round_trip')
    refs={r['job']:r for r in read(V16/'results/validation_features.json')};sources={h['job']:h for h in read(OUT/'source_heads.json')}
    old=pd.read_csv(V16/'results/model_predictions.csv',float_precision='round_trip');old_ensemble=pd.read_csv(V16/'results/ensemble_predictions.csv',float_precision='round_trip')
    forecasts=[];clip_rows=[];clip_summary=[];training_metrics=pd.read_csv(OUT/'training_metrics.csv').set_index('job')
    for h in read(OUT/'heads.json'):
        fold=next(f for f in cfg()['folds'] if f['cutoff']==h['cutoff']);_,te=indices(obs,fold);d=load_npz(h);v=load_npz(refs[h['source_job']])
        np.testing.assert_array_equal(v['row_index'],te);f=v['features'].astype(float);x=apply_clip(f,d);z=design(x)@np.asarray(h['coefficients']);p=probability(z)
        g=base[base.cutoff.eq(h['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,te)
        g['method']=h['method'];g['seed']=h['seed'];g['logit']=z;g['score']=p;g['probability']=p;g['direction_up']=p>.5;forecasts.append(g)
        clipped=(f<d['lower'])|(f>d['upper']);before=(f-d['mean'])/d['sd'];nclipped=clipped.sum(axis=1)
        for i,row in enumerate(g.itertuples()):
            clip_rows.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],date=row.date,row_index=int(row.row_index),
                clipped_coordinates=int(nclipped[i]),coordinate_clip_fraction=float(clipped[i].mean()),
                maximum_standardized_distance_before=float(np.abs(before[i]).max()),maximum_standardized_distance_after=float(np.abs(x[i]).max())))
        ref=old[old.method.eq(CANDIDATES[h['method']])&old.cutoff.eq(h['cutoff'])&old.seed.eq(h['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(ref.row_index,te)
        clip_summary.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],validation_n=len(te),
            training_coordinate_clip_fraction=float(training_metrics.loc[h['job'],'coordinate_clip_fraction']),
            validation_coordinate_clip_fraction=float(clipped.mean()),validation_row_clip_fraction=float(clipped.any(axis=1).mean()),
            mean_maximum_distance_before=float(np.abs(before).max(axis=1).mean()),mean_maximum_distance_after=float(np.abs(x).max(axis=1).mean()),
            original_mean_probability=float(ref.probability.mean()),clipped_mean_probability=float(p.mean()),observed_up_fraction=float(g.actual_up.mean())))
    new=pd.concat(forecasts,ignore_index=True);assert len(new)==1044
    new.to_csv(OUT/'new_model_predictions.csv',index=False)
    all_predictions=pd.concat([old,new],ignore_index=True);assert len(all_predictions)==2871
    all_predictions.to_csv(OUT/'model_predictions.csv',index=False)
    e=new.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),year=('year','first'),
        joint_completed=('joint_completed','first'),actual_return=('actual_return','first'),actual_up=('actual_up','first'),
        training_frequency=('training_frequency','first'),score=('score','mean'),probability=('probability','mean')).reset_index()
    e['direction_up']=e.probability>.5;ensemble=pd.concat([old_ensemble,e],ignore_index=True)
    assert len(ensemble)==1827 and ensemble.groupby('method').size().eq(261).all()
    ensemble.to_csv(OUT/'ensemble_predictions.csv',index=False)
    thresholds={r['cutoff']:r for r in read(OUT/'state_thresholds.json')};states=[]
    for fold in cfg()['folds']:
        _,te=indices(obs,fold);tags=assign_states(descriptors(bars,obs.anchor.iloc[te].to_numpy()),thresholds[fold['cutoff']])
        tags.insert(0,'date',obs.date.iloc[te].to_numpy());tags.insert(0,'row_index',te);tags.insert(0,'cutoff',fold['cutoff']);states.append(tags)
    states=pd.concat(states,ignore_index=True);assert len(states)==261 and states.date.is_unique
    states.to_csv(OUT/'validation_states.csv',index=False)
    pd.DataFrame(clip_rows).to_csv(OUT/'clipping_rows.csv',index=False);pd.DataFrame(clip_summary).to_csv(OUT/'clipping_summary.csv',index=False)
    files=[OUT/n for n in ['new_model_predictions.csv','model_predictions.csv','ensemble_predictions.csv','validation_states.csv','clipping_rows.csv','clipping_summary.csv']]
    finish(run,files,new_probability_forecasts=1044,reused_model_forecasts=1827,model_records=2871,ensemble_records=1827,
        heldout_weeks=261,causal_market_state_rows=261,state_conditioned_forecasts=0)
    print('Scored1044 new probabilities; retained1827 model records;7 methods on261 unchanged dates. State labels are diagnostic only.',flush=True)

if __name__=='__main__':main()

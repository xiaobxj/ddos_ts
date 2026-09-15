from common16 import *


def main():
    legacy.initialize();check_frozen();fitting=check_phase('training');assert fitting['all_converged'] and fitting['new_primary_fits']==15
    path=OUT/'scoring_manifest.json';assert not path.exists(),'Preserve scoring';run=manifest('scoring');save(path,run)
    obs,price,returns=data();bars=raw_bars(price);base=pd.read_csv(OUT/'classification_baselines.csv')
    old=pd.read_csv(OUT/'native_seed_reference.csv');cached=pd.read_csv(OUT/'reused_probe_seed_reference.csv')
    forecasts=[];native_rows=[];checks=[];feature_refs=[];files=[];new_predictions=0
    for head in read(OUT/'heads.json'):
        fold=next(f for f in cfg()['folds'] if f['cutoff']==head['cutoff']);_,te=indices(obs,fold);d=load_features(head)
        if head['kind']=='learned':
            model,state,values,f,_=extract_learned(head,te)
            original=previous.archival_native_output(model,values).astype(float)
            native=original*state['scales']['returns_sd']+state['scales']['returns_mean']
            reference=old[old.cutoff.eq(head['cutoff'])&old.seed.eq(head['seed'])].sort_values('row_index')
            np.testing.assert_array_equal(reference.row_index,te)
            native_error=float(np.max(np.abs(native-reference.score.to_numpy())));assert native_error<1e-10
            native_rows.append(reference.copy())
            assert object_hash(model.state_dict())==head['model_sha256'] and sha(PROJECT/head['project_file'])==head['sha256']
            assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
            del model,values;torch.cuda.empty_cache()
        else:f=raw_features(bars,obs.anchor.iloc[te].to_numpy());native_error=None
        z=design((f.astype(float)-d['mean'])/d['sd'])@np.asarray(head['coefficients']);p=probability(z)
        g=base[base.cutoff.eq(head['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,te)
        g['method']=head['method'];g['seed']=head['seed'];g['logit']=z;g['score']=p;g['probability']=p;g['direction_up']=p>.5
        retained_error=None
        if head['reused']:
            reference=cached[cached.cutoff.eq(head['cutoff'])&cached.seed.eq(head['seed'])].sort_values('row_index')
            np.testing.assert_array_equal(reference.row_index,te)
            retained_error=max(float(np.max(np.abs(p-reference.probability.to_numpy()))),float(np.max(np.abs(z-reference.logit.to_numpy()))))
            assert retained_error<1e-10;np.testing.assert_array_equal(g.direction_up,reference.direction_up)
            g=reference.copy()
        else:new_predictions+=len(g)
        forecasts.append(g)
        cache=CACHE/f"validation_{head['job']}.npz";np.savez_compressed(cache,features=f,row_index=te);files.append(cache)
        feature_refs.append(dict(job=head['job'],cache_file=str(cache.relative_to(PROJECT)),cache_sha256=sha(cache)))
        checks.append(dict(job=head['job'],native_error=native_error,retained_probe_error=retained_error))
    all_predictions=pd.concat(forecasts+native_rows,ignore_index=True);assert len(all_predictions)==1827 and new_predictions==621
    assert len(pd.concat(native_rows))==783
    all_predictions.to_csv(OUT/'model_predictions.csv',index=False)
    ensemble=all_predictions.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),year=('year','first'),
        joint_completed=('joint_completed','first'),actual_return=('actual_return','first'),actual_up=('actual_up','first'),
        training_frequency=('training_frequency','first'),score=('score','mean'),probability=('probability','mean')).reset_index()
    ensemble['direction_up']=ensemble.score>np.where(ensemble.method.eq('native_mse'),0,.5)
    for method in ['training_frequency','neutral_50']:
        g=base.copy();g['method']=method;g['probability']=g.training_frequency if method=='training_frequency' else .5
        g['score']=g.probability;g['direction_up']=g.probability>.5;ensemble=pd.concat([ensemble,g],ignore_index=True)
    assert len(ensemble)==1305 and ensemble.groupby('method').size().eq(261).all()
    ensemble.to_csv(OUT/'ensemble_predictions.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'archival_parity.csv',index=False)
    save(OUT/'validation_features.json',feature_refs)
    files += [OUT/n for n in ['model_predictions.csv','ensemble_predictions.csv','archival_parity.csv','validation_features.json']]
    finish(run,files,new_probability_forecasts=621,reused_probability_forecasts=423,reused_native_forecasts=783,model_records=1827,ensemble_records=1305,
        heldout_weeks=261,learned_feature_rows=783,raw_feature_rows=261)
    print('Scoring complete:621 new probability forecasts;1206 retained forecasts;5 methods on261 dates.',flush=True)


if __name__=='__main__':main()

from common15 import *

def main():
    legacy.initialize();check_frozen();fit=check_artifacts('fitting');assert fit['all_converged'] and fit['primary_heads']==18
    check_artifacts('extraction');path=OUT/'scoring_manifest.json';assert not path.exists(),'Preserve scoring'
    run=manifest('scoring');save(path,run);heads=read(OUT/'heads.json');rows=[];files=[];features=[];native_errors=[]
    base=pd.read_csv(OUT/'classification_baselines.csv');old=pd.read_csv(OUT/'archived_seed_predictions.csv');meta=read(OUT/'training_metadata.json')
    for head in heads:
        model,state=load_backbone(head);values,te=previous.load_validation(head['cutoff']);f,native=extract_features(model,values)
        d=load_features(head);x=(f.astype(float)-d['mean'])/d['sd'];z=design(x)@np.asarray(head['coefficients'],float);p=probability(z)
        reference=old[old.method.eq(head['source'])&old.cutoff.eq(head['cutoff'])&old.seed.eq(head['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(reference.row_index,te)
        if head['source']=='direction_bce':native_score=probability(native.astype(float))
        else:
            scales=meta[head['cutoff']]['scales'];native_score=native.astype(float)*scales['returns_sd']+scales['returns_mean']
        error=float(np.max(np.abs(native_score-reference.score.to_numpy())));assert error<1e-10
        native_errors.append(dict(family=head['family'],cutoff=head['cutoff'],seed=head['seed'],maximum_error=error))
        path_feature=CACHE/f'validation_{identity(head)}.npz'
        np.savez_compressed(path_feature,features=f,row_index=te,native_output=native);files.append(path_feature)
        features.append(dict(family=head['family'],cutoff=head['cutoff'],seed=head['seed'],feature_file=str(path_feature.relative_to(ROOT)),sha256=sha(path_feature)))
        g=base[base.cutoff.eq(head['cutoff'])].copy();np.testing.assert_array_equal(g.row_index,te)
        g['method']=head['family'];g['seed']=head['seed'];g['probe_logit']=z;g['probability']=p;g['score']=p;g['direction_up']=p>.5;rows.append(g)
        assert object_hash(model.state_dict())==head['model_sha256'] and sha(PROJECT/head['project_file'])==head['sha256']
        del model,values;torch.cuda.empty_cache()
    new=pd.concat(rows,ignore_index=True);assert len(new)==846
    new.to_csv(OUT/'probe_seed_predictions.csv',index=False)
    pd.concat([old,new],ignore_index=True).to_csv(OUT/'all_seed_predictions.csv',index=False)
    ensemble=new.groupby(['method','date'],sort=True).agg(cutoff=('cutoff','first'),row_index=('row_index','first'),actual_return=('actual_return','first'),
        actual_up=('actual_up','first'),training_frequency=('training_frequency','first'),probability=('probability','mean')).reset_index()
    ensemble['score']=ensemble.probability;ensemble['direction_up']=ensemble.probability>.5
    archived=pd.read_csv(OUT/'archived_ensemble_predictions.csv')
    all_ensemble=pd.concat([archived,ensemble],ignore_index=True);assert len(all_ensemble)==846
    all_ensemble.to_csv(OUT/'all_ensemble_predictions.csv',index=False)
    save(OUT/'validation_features.json',features);pd.DataFrame(native_errors).to_csv(OUT/'native_prediction_parity.csv',index=False)
    files += [OUT/n for n in ['probe_seed_predictions.csv','all_seed_predictions.csv','all_ensemble_predictions.csv','validation_features.json','native_prediction_parity.csv']]
    run.update(finished_utc=now(),new_seed_predictions=846,reused_seed_predictions=846,validation_feature_rows=846,
        validation_weeks=141,ensemble_methods=6,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files});save(path,run)
    print('Scoring complete: 846 new +846 reused seed forecasts;6 ensemble methods on141weeks.',flush=True)

if __name__=='__main__':main()

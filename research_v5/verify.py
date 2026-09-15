"""Independent chronology, target, checkpoint and source replay checks."""
from models5 import *
import subprocess


def main():
    initialize()
    test=subprocess.run([sys.executable,str(ROOT/'test_contracts.py')],capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'test_results.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    assert test.returncode==0,test.stdout+test.stderr
    manifests=[json.loads((OUT/name).read_text(encoding='utf-8')) for name in
               ['preparation_manifest.json','capacity_manifest.json','validation_manifest.json']]
    previous=old_evidence()
    for run in manifests:
        assert run.get('finished_utc')
        assert run['protocol_sha256']==sha(ROOT/'protocol.json')
        assert run['old_evidence']==previous
        for name,digest in run['source_sha256'].items():assert sha(PROJECT/name)==digest,name
    prep=manifests[0]
    for name,digest in prep['cache_sha256'].items():assert sha(ROOT/name)==digest,name
    for r in prep['tokenizer_weight']['files']:assert sha(V3/r['file'])==r['sha256']
    assert sha(OUT/'observation_table.csv')==prep['observation_table_sha256']
    assert manifests[2]['preparation_manifest_sha256']==sha(OUT/'preparation_manifest.json')
    frame=pd.read_csv(V1/'data/1_000300.csv');obs=pd.read_csv(OUT/'observation_table.csv')
    old=pd.read_csv(V4/'results/observation_table.csv');targets=np.load(CACHE/'targets.npz')
    for key in old.columns:
        if pd.api.types.is_float_dtype(old[key]):
            np.testing.assert_allclose(obs[key],old.iloc[obs.v4_row][key],rtol=0,atol=1e-14)
        else:np.testing.assert_array_equal(obs[key],old.iloc[obs.v4_row][key])
    # Recompute all auxiliary coordinates directly, independently of prepare.py.
    for i,r in obs.iterrows():
        t=int(r.anchor);future=frame.iloc[t+1:t+6][['open','high','low','close','volume']].to_numpy(float)
        expected=np.empty((5,5))
        expected[:,:4]=np.log(future[:,:4]/frame.close.iloc[t])
        expected[:,4]=np.log1p(future[:,4])-np.mean(np.log1p(frame.volume.iloc[t-19:t+1].to_numpy(float)))
        np.testing.assert_allclose(expected.flatten(),targets['auxiliary'][i],rtol=0,atol=1e-12)
        actual=frame.open.iloc[int(r.exit)]/frame.open.iloc[int(r.entry)]-1
        assert abs(actual-targets['returns'][i])<1e-12
        assert r.joint_completed==max(r.completed,frame.date.iloc[t+5])
    indices=np.load(CACHE/'capacity_indices.npy')
    available=np.flatnonzero((obs.joint_completed<=cfg()['capacity']['sample_cutoff']).to_numpy())
    np.testing.assert_array_equal(indices,available[np.rint(np.linspace(0,len(available)-1,32)).astype(int)])
    capacity=pd.read_csv(OUT/'capacity_predictions.csv')
    cap_summary=json.loads((OUT/'capacity_summary.json').read_text(encoding='utf-8'))
    cap_checkpoints=json.loads((OUT/'capacity_checkpoint_manifest.json').read_text(encoding='utf-8'))
    maximum_capacity_error=0.;replayed_capacity=0
    for r in cap_checkpoints:
        path=ROOT/r['file'];assert sha(path)==r['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        network=make_model(r['representation'],r['variant'],r['seed'],capacity=True).eval()
        network.load_state_dict(saved['state_dict']);values=batch_tensors(r['representation'],indices)
        # Original final capacity predictions were computed with autograd active
        # to inspect final gradients; use the same forward mode for exact replay.
        output,_=predict(network,values,r['variant']=='joint_ohlcv')
        estimates=output.detach().cpu().numpy().astype(float)
        reference=capacity[(capacity.representation==r['representation'])&(capacity.variant==r['variant'])&
                           (capacity.label_kind==r['label_kind'])&(capacity.seed==r['seed'])]
        np.testing.assert_array_equal(reference.row_index,indices)
        error=float(np.max(np.abs(estimates-reference.prediction_standardized)))
        maximum_capacity_error=max(maximum_capacity_error,error)
        np.testing.assert_allclose(estimates,reference.prediction_standardized,rtol=0,atol=1e-10)
        y=(targets['returns'][indices]-saved['scales']['returns_mean'])/saved['scales']['returns_sd']
        y=y.astype(np.float32)
        if r['label_kind']=='permuted':y=y[np.random.default_rng(20260914).permutation(32)]
        np.testing.assert_allclose(y,reference.target_standardized,rtol=0,atol=1e-10)
        summary=next(x for x in cap_summary if all(x[k]==r[k] for k in ['representation','variant','seed','label_kind']))
        measured=stats(estimates,y)
        assert abs(measured['mse']-summary['mse'])<1e-10
        assert summary['capacity_pass']==(measured['mse']<=.1 and measured['correlation'] is not None and measured['correlation']>=.9)
        replayed_capacity+=1;del network,values,output;torch.cuda.empty_cache()
    seed=pd.read_csv(OUT/'validation_seed_predictions.csv')
    folds=json.loads((OUT/'validation_folds.json').read_text(encoding='utf-8'))
    checkpoints=json.loads((OUT/'validation_checkpoint_manifest.json').read_text(encoding='utf-8'))
    maximum_validation_error=0.;replayed_validation=0
    for fold in folds:
        tr=np.flatnonzero((obs.joint_completed<=fold['cutoff']).to_numpy())
        te=np.flatnonzero(((obs.date>fold['cutoff'])&(obs.date<=fold['end'])&(obs.weekday==4)&
                          (obs.joint_completed<='2020-12-31')).to_numpy())
        assert len(tr)==fold['train_n'] and len(te)==fold['test_n']
        assert obs.joint_completed.iloc[tr].max()==fold['last_train_label']<=fold['cutoff']<obs.date.iloc[te].min()
        assert fold['representation']=='raw'
        values=batch_tensors('raw',te)
        for r in [x for x in checkpoints if x['cutoff']==fold['cutoff']]:
            path=ROOT/r['file'];assert sha(path)==r['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['representation']=='raw' and saved['train_n']==len(tr)
            assert saved['protocol_sha256']==sha(ROOT/'protocol.json')
            for key in ['returns','auxiliary']:
                np.testing.assert_allclose(targets[key][tr].mean(axis=0),saved['scales'][key+'_mean'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(np.maximum(targets[key][tr].std(axis=0),1e-6),saved['scales'][key+'_sd'],rtol=0,atol=1e-12)
            network=make_model('raw',r['variant'],r['seed']).eval();network.load_state_dict(saved['state_dict'])
            with torch.inference_mode():
                out,_=predict(network,values,False)
                changed,_=predict(network,{k:torch.roll(v,1,dims=0) for k,v in values.items()},False)
            expected=out.cpu().numpy().astype(float)*saved['scales']['returns_sd']+saved['scales']['returns_mean']
            expected_shift=changed.cpu().numpy().astype(float)*saved['scales']['returns_sd']+saved['scales']['returns_mean']
            ref=seed[(seed.variant==r['variant'])&(seed.epoch==r['epoch'])&(seed.seed==r['seed'])&(seed.cutoff==r['cutoff'])]
            np.testing.assert_array_equal(ref.row_index,te)
            maximum_validation_error=max(maximum_validation_error,float(np.max(np.abs(expected-ref.predicted_return))))
            np.testing.assert_allclose(expected,ref.predicted_return,rtol=0,atol=1e-10)
            np.testing.assert_allclose(expected_shift,ref.reassigned_prediction,rtol=0,atol=1e-10)
            replayed_validation+=1;del network;torch.cuda.empty_cache()
        del values
    ensemble=pd.read_csv(OUT/'validation_ensemble_predictions.csv')
    for (variant,epoch),g in seed.groupby(['variant','epoch']):
        assert g.groupby('date').size().eq(3).all()
        mean=g.groupby('date').predicted_return.mean().sort_index()
        other=ensemble[(ensemble.variant==variant)&(ensemble.epoch==epoch)].sort_values('date')
        np.testing.assert_allclose(mean.to_numpy(),other.predicted_return,rtol=0,atol=1e-12)
    metrics=pd.read_csv(OUT/'validation_metrics.csv')
    selected=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected']
    order={v['name']:i for i,v in enumerate(cfg()['variants'])}
    winner=min(metrics.to_dict('records'),key=lambda r:(r['rmse'],r['epoch'],order[r['variant']]))
    assert (winner['variant'],winner['epoch'])==(selected['variant'],selected['epoch'])
    result=dict(status='PASS',unit_tests=4,shared_observations=len(obs),capacity_samples=len(indices),
                auxiliary_target_coordinates_recomputed=int(targets['auxiliary'].size),
                real_and_permuted_capacity_checkpoints_replayed=replayed_capacity,
                validation_checkpoints_replayed=replayed_validation,max_capacity_replay_error=maximum_capacity_error,
                max_validation_replay_error=maximum_validation_error,capacity_prediction_rows=len(capacity),
                validation_seed_prediction_rows=len(seed),validation_weeks=int(ensemble.groupby(['variant','epoch']).size().iloc[0]),
                old_evidence_files_preserved=len(previous),all_source_and_cache_hashes_match=True,
                pretrained_representation_excluded_from_historical_validation=True,all_training_labels_mature=True,
                selection_recomputed=True,checked_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

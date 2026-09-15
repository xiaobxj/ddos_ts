from common14 import *
import evaluate14 as evaluation

def close_tree(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:close_tree(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close_tree(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)

def main():
    legacy.initialize();start=time.time();prep=check_frozen()
    metadata=read(OUT/'training_metadata.json');obs=pd.read_csv(OUT/'observation_table.csv')
    predictions=pd.read_csv(OUT/'classification_seed_predictions.csv');archived=pd.read_csv(OUT/'archived_predictions.csv')
    curves=pd.read_csv(OUT/'training_curves.csv');draws=pd.read_csv(OUT/'training_loss_draws.csv')
    initial=read(OUT/'initial_states.json');max_prediction=0.;max_training=0.;epochs=0;loss_passes=0;forecast_count=0
    references=read(OUT/'models.json')+read(OUT/'archived_models.json');assert len(references)==18
    for ref in references:
        cutoff=ref['cutoff'];seed=ref['seed'];model,state=load_model(ref)
        tr=np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy());assert state['train_n']==len(tr)
        if ref.get('method')=='direction_bce':
            assert state['training_metadata']==metadata[cutoff] and state['protocol_sha256']==prep['protocol_sha256']
            assert object_hash(state['optimizer_state_dict'])==ref['optimizer_sha256']
            assert object_hash({k:state[k] for k in ['numpy_rng_state','cpu_rng_state','cuda_rng_state']})==ref['rng_sha256']
            c=curves[curves.cutoff.eq(cutoff)&curves.seed.eq(seed)].sort_values('epoch');assert len(c)==20
            rng=np.random.default_rng(seed)
            for row in c.itertuples():
                order=rng.permutation(len(tr));batches=prior.batches(order)
                assert row.order_sha256==array_hash(order) and row.boundary_sha256==object_hash([array_hash(b) for b in batches])
                assert row.steps==len(batches) and row.last_batch_n==len(batches[-1]) and row.presentations==len(tr)
                assert row.lr==.001 and row.weight_decay==.1;epochs+=1
            assert object_hash(rng.bit_generator.state)==object_hash(state['numpy_rng_state'])
            for value in state['optimizer_state_dict']['state'].values():assert int(value['step'])==20*((len(tr)+127)//128)
            values,labels,indices,meta=load_training(cutoff);np.testing.assert_array_equal(indices,tr)
            with np.load(V5/'cache/targets.npz') as raw:
                np.testing.assert_array_equal(labels['direction'].cpu().numpy(),(raw['returns'][tr]>0).astype(np.float32))
            for r in draws[draws.cutoff.eq(cutoff)&draws.seed.eq(seed)].itertuples():
                measured=training_loss(model,values,labels,r.mode=='dropout',int(r.audit_seed) if r.mode=='dropout' else None)
                for k,v in measured.items():
                    e=abs(v-getattr(r,k));max_training=max(max_training,e);assert e<1e-12
                loss_passes+=1
            del values,labels
            g=predictions[predictions.cutoff.eq(cutoff)&predictions.seed.eq(seed)].sort_values('row_index')
        else:
            assert state['scales']==metadata[cutoff]['scales']
            g=archived[archived.cutoff.eq(cutoff)&archived.seed.eq(seed)].sort_values('row_index')
        testing,te=load_validation(cutoff);np.testing.assert_array_equal(te,g.row_index)
        with torch.inference_mode():z,_=legacy.predict(model,testing,False)
        z=z.cpu().numpy().astype(float)
        if ref.get('method')=='direction_bce':
            p=probability(z)
            error=max(float(np.max(np.abs(z-g.logit.to_numpy()))),float(np.max(np.abs(p-g.probability.to_numpy()))))
            np.testing.assert_array_equal(g.actual_up,raw_direction(obs.exec_return.iloc[te]).astype(int))
            np.testing.assert_array_equal(g.direction_up,p>.5)
            np.testing.assert_allclose(g.training_frequency,metadata[cutoff]['frequency'],rtol=0,atol=1e-15)
        else:
            p=z*state['scales']['returns_sd']+state['scales']['returns_mean']
            error=float(np.max(np.abs(p-g.predicted_return.to_numpy())))
            np.testing.assert_allclose(g.actual,obs.exec_return.iloc[te],rtol=0,atol=1e-12)
        max_prediction=max(max_prediction,error);assert error<1e-10;forecast_count+=len(g)
        del model,testing;torch.cuda.empty_cache()
        print(f"Replay {ref.get('method','archived20')} {cutoff} seed {seed}: PASS",flush=True)
    assert epochs==180 and loss_passes==81 and forecast_count==846
    for r in initial:
        old=legacy.make_model('combined',r['seed']);old_hash=shared_initial_hash(old)
        new=make_classifier(r['seed'],metadata[r['cutoff']]['frequency'])
        assert old_hash==shared_initial_hash(new)==r['shared_initial_sha256']
        assert object_hash(new.state_dict())==r['initial_model_sha256']
        assert object_hash(torch.get_rng_state())==r['initial_cpu_rng_sha256'] and object_hash(torch.cuda.get_rng_state())==r['initial_cuda_rng_sha256']
        assert float(new.return_head.bias.detach().cpu()[0])==r['initial_bias']
        del old,new
    ensemble=pd.read_csv(OUT/'all_ensemble_predictions.csv');seed=pd.read_csv(OUT/'all_seed_predictions.csv')
    for method,g in ensemble.groupby('method'):
        g=g.sort_values('date');assert len(g)==141 and g.date.is_unique
        if method=='direction_bce':
            expected=predictions.groupby('date').probability.mean()
            np.testing.assert_allclose(g.probability,expected,rtol=0,atol=1e-14)
            np.testing.assert_array_equal(g.direction_up,g.probability>.5)
        elif method=='archived20':
            expected=archived.groupby('date').predicted_return.mean()
            np.testing.assert_allclose(g.score,expected,rtol=0,atol=1e-14)
            assert g.probability.isna().all();np.testing.assert_array_equal(g.direction_up,g.score>0)
        else:
            expected=g.cutoff.map({c:m['frequency'] for c,m in metadata.items()}).to_numpy() if method=='training_frequency' else np.full(141,.5)
            np.testing.assert_allclose(g.probability,expected,rtol=0,atol=1e-15)
            np.testing.assert_array_equal(g.direction_up,g.probability>.5)
    # Independent pairwise AUROC and elementary confusion/probability formulas.
    metric_tables=[(ensemble,pd.read_csv(OUT/'ensemble_metrics.csv'),['method']),
                   (seed,pd.read_csv(OUT/'seed_metrics.csv'),['method','seed'])]
    for source,summary,keys in metric_tables:
        for r in summary.to_dict('records'):
            g=source.copy()
            for k in keys:g=g[g[k].eq(r[k])]
            y=g.actual_up.to_numpy(bool);p=g.direction_up.to_numpy(bool);score=g.score.to_numpy()
            assert int((p==y).sum())==r['correct_directions']
            assert abs(float((p==y).mean())-r['accuracy'])<1e-14
            for k,v in [('tp',p&y),('tn',~p&~y),('fp',p&~y),('fn',~p&y)]:assert r[k]==int(v.sum())
            a=score[y,None];b=score[~y][None,:]
            auroc=float(((a>b)+.5*(a==b)).mean());assert abs(auroc-r['auroc'])<1e-14
            if g.probability.notna().all():
                q=g.probability.to_numpy();bounded=np.minimum(1-1e-12,np.maximum(1e-12,q))
                brier=float(np.average((q-y.astype(int))**2))
                nll=float(np.average(-np.where(y,np.log(bounded),np.log(1-bounded))))
                assert abs(brier-r['brier'])<1e-14 and abs(nll-r['log_loss'])<1e-14
            else:assert pd.isna(r['brier']) and pd.isna(r['log_loss'])
    frames,pairs,assessment=evaluation.compute()
    for name,frame in frames.items():pd.testing.assert_frame_equal(frame,pd.read_csv(OUT/f'{name}.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
    close_tree(pairs,read(OUT/'primary_comparisons.json'));close_tree(assessment,read(OUT/'assessment.json'))
    bins=pd.read_csv(OUT/'reliability_bins.csv');assert len(bins)==15
    for r in bins.itertuples():
        g=ensemble[ensemble.method.eq(r.method)];h=g[g.probability.ge(r.lower)&(g.probability.le(r.upper) if r.bin_index==4 else g.probability.lt(r.upper))]
        assert r.n==len(h)
        if len(h):assert abs(r.mean_probability-h.probability.mean())<1e-14 and abs(r.observed_frequency-h.actual_up.mean())<1e-14
        else:assert pd.isna(r.mean_probability) and pd.isna(r.observed_frequency)
    previous=read(OUT/'contract_verification.json')['completed_utc']
    for phase in ['training','scoring','evaluation']:
        run=read(OUT/f'{phase}_manifest.json');assert previous<=run['started_utc']<run['finished_utc'];previous=run['finished_utc']
        for name,digest in run['artifacts'].items():assert sha(OUT/name)==digest,name
    assert old_evidence()==prep['old_evidence']
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,model_states_replayed=18,initial_states_verified=9,
        training_loss_passes=81,epoch_orders_and_batches=180,classifier_seed_predictions=423,archived_seed_predictions=423,
        ensemble_methods=4,validation_weeks=141,primary_comparisons_recomputed=4,reliability_bins_recomputed=15,
        max_prediction_error=max_prediction,max_training_loss_error=max_training,previous_files_preserved=2415,
        protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes())
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

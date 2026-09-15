"""Independent coverage, inference, scoring and evidence verification."""
from common9 import *


def metrics_check(row,g):
    y=g.actual.to_numpy();p=g.predicted_return.to_numpy();mse=float(np.mean((p-y)**2))
    checks=dict(mse=mse,rmse=mse**.5,accuracy=float(((p>0)==(y>0)).mean()),
        mse_skill_vs_training_mean=1-mse/float(np.mean((g.training_mean-y)**2)),
        reassigned_mse=float(np.mean((g.reassigned_prediction-y)**2)))
    for key,value in checks.items():assert abs(row[key]-value)<1e-12,(key,row[key],value)


def verify_scores(seed,ensemble):
    for r in pd.read_csv(OUT/'ensemble_metrics.csv').to_dict('records'):metrics_check(r,ensemble[ensemble.rule==r['rule']])
    for r in pd.read_csv(OUT/'yearly_metrics.csv').to_dict('records'):
        metrics_check(r,ensemble[(ensemble.rule==r['rule'])&(ensemble.date.str[:4]==str(r['year']))])
    for r in pd.read_csv(OUT/'seed_metrics.csv').to_dict('records'):
        metrics_check(r,seed[(seed.rule==r['rule'])&(seed.seed==r['seed'])])
    a=ensemble[ensemble.rule=='balanced'].sort_values('date');b=ensemble[ensemble.rule=='legacy'].sort_values('date')
    assert a.date.tolist()==b.date.tolist()
    y=a.actual.to_numpy();pred=a.predicted_return.to_numpy();mean=a.training_mean.to_numpy()
    va=(pred-y)**2;old=(b.predicted_return.to_numpy()-y)**2;mu=(mean-y)**2
    comparisons=json.loads((OUT/'primary_comparisons.json').read_text(encoding='utf-8'));assert len(comparisons)==2
    ps=[]
    for r in comparisons:
        vb=old if r['reference']=='legacy' else mu;d=va-vb;n=len(d)
        rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))))
        ids=((starts[:,:,None]+np.arange(8))%n).reshape(10000,-1)[:,:n]
        boot=d[ids].mean(axis=1);centered=(d-d.mean())[ids].mean(axis=1)
        p=float((1+(np.abs(centered)>=abs(d.mean())).sum())/10001);ps.append(p)
        assert abs(r['mse']['p']-p)<1e-12 and abs(r['mse']['difference']-d.mean())<1e-12
        np.testing.assert_allclose(np.quantile(boot,[.025,.975]),[r['mse']['ci95_low'],r['mse']['ci95_high']],rtol=0,atol=1e-12)
        delta=np.sqrt(va[ids].mean(axis=1))-np.sqrt(vb[ids].mean(axis=1))
        np.testing.assert_allclose(np.quantile(delta,[.025,.975]),[r['rmse']['ci95_low'],r['rmse']['ci95_high']],rtol=0,atol=1e-12)
    adjusted=np.zeros(2);running=0.
    for i,j in enumerate(np.argsort(ps)):running=max(running,min(1.,ps[j]*(2-i)));adjusted[j]=running
    np.testing.assert_allclose(adjusted,[r['mse']['holm_adjusted_p'] for r in comparisons],rtol=0,atol=1e-12)
    count_mean=0;count_old=0;count_year=0
    for number in cfg()['training']['seeds']:
        x=seed[(seed.rule=='balanced')&(seed.seed==number)].sort_values('date')
        z=seed[(seed.rule=='legacy')&(seed.seed==number)].sort_values('date')
        count_mean+=int(np.mean((x.predicted_return-x.actual)**2)<np.mean((x.training_mean-x.actual)**2))
        count_old+=int(np.mean((x.predicted_return-x.actual)**2)<np.mean((z.predicted_return-z.actual)**2))
    for year in ['2018','2019','2020']:
        h=a.date.str[:4].eq(year).to_numpy();count_year+=int(va[h].mean()<old[h].mean())
    flags=dict(ensemble_beats_legacy=bool(va.mean()<old.mean()),ensemble_beats_mean=bool(va.mean()<mu.mean()),
        at_least_two_seeds_beat_mean=count_mean>=2,at_least_two_seeds_beat_legacy=count_old>=2,at_least_two_years_beat_legacy=count_year>=2)
    stored=json.loads((OUT/'assessment.json').read_text(encoding='utf-8'))
    assert stored['flags']==flags and stored['descriptive_screen_pass']==all(flags.values())
    assert stored['new_seeds_better_than_mean']==count_mean and stored['new_seeds_better_than_legacy']==count_old
    assert stored['new_years_better_than_legacy']==count_year
    return flags


def main():
    legacy.initialize();prep=check_frozen();start=time.time();assert old_evidence()==prep['old_evidence']
    run=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
    assert run.get('finished_utc') and run['fits']==9 and run['source_sha256']==source_hashes()
    assert run['input_sha256']==input_hashes()
    obs=pd.read_csv(OUT/'observation_table.csv');seed=pd.read_csv(OUT/'all_seed_predictions.csv')
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv');curves=pd.read_csv(OUT/'balanced_training_curves.csv')
    assert len(seed)==846 and len(ensemble)==282 and len(curves)==180
    assert seed.groupby(['rule','date']).size().eq(3).all()
    refs=json.loads((OUT/'legacy_checkpoint_manifest.json').read_text(encoding='utf-8'))+json.loads((OUT/'balanced_checkpoint_manifest.json').read_text(encoding='utf-8'))
    assert len(refs)==18 and len({(r['rule'],r['cutoff'],r['seed']) for r in refs})==18
    with np.load(V5/'cache/targets.npz') as arrays:target={k:arrays[k].copy() for k in ['returns','auxiliary']}
    max_replay=0.;max_shift=0.;order_count=0;boundary_count=0
    for number,r in enumerate(refs,1):
        path=PROJECT/r['project_file'];assert sha(path)==r['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['arm']=='combined' and state['epoch']==20
        tr,te=load_fold(r['cutoff']);n=len(tr)
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(r['cutoff']).to_numpy()))
        for key,values in target.items():
            x=values[tr].astype(float)
            np.testing.assert_allclose(x.mean(axis=0),state['scales'][key+'_mean'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.maximum(x.std(axis=0),1e-6),state['scales'][key+'_sd'],rtol=0,atol=1e-12)
        g=seed[(seed.rule==r['rule'])&(seed.cutoff==r['cutoff'])&(seed.seed==r['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index.to_numpy(),te)
        np.testing.assert_allclose(g.actual,target['returns'][te],rtol=0,atol=1e-12)
        np.testing.assert_allclose(g.training_mean,state['scales']['returns_mean'],rtol=0,atol=1e-12)
        assert obs.weekday.iloc[te].eq(4).all() and obs.joint_completed.iloc[te].le('2020-12-31').all()
        if r['rule']=='balanced':
            group=curves[(curves.cutoff==r['cutoff'])&(curves.seed==r['seed'])].sort_values('epoch')
            assert group.epoch.tolist()==list(range(1,21));rng=np.random.default_rng(r['seed'])
            m=(n+127)//128;q,rem=divmod(n,m);sizes=np.full(m,q);sizes[:rem]+=1
            boundaries=np.r_[0,np.cumsum(sizes)]
            coeff=sizes*m/n/sizes/m;np.testing.assert_allclose(coeff,1/n,rtol=0,atol=1e-15)
            for row in group.itertuples():
                order=rng.permutation(n)
                assert row.training_order_sha256==array_hash(tr[order]);order_count+=1
                assert row.batch_boundaries_sha256==array_hash(boundaries);boundary_count+=1
                assert row.presentations==n and row.train_n==n and row.optimizer_steps==m
                assert row.minimum_batch==sizes.min() and row.maximum_batch==sizes.max()
        model=legacy.make_model('combined',r['seed']).eval();model.load_state_dict(state['state_dict'])
        assert sum(p.numel() for p in model.parameters())==38551
        values=legacy.batch_tensors(te)
        with torch.inference_mode():
            p,_=legacy.predict(model,values,False)
            altered,_=legacy.predict(model,{k:torch.roll(v,1,dims=0) for k,v in values.items()},False)
        p=p.cpu().numpy().astype(float)*state['scales']['returns_sd']+state['scales']['returns_mean']
        shifted=altered.cpu().numpy().astype(float)*state['scales']['returns_sd']+state['scales']['returns_mean']
        np.testing.assert_allclose(p,g.predicted_return,rtol=0,atol=1e-10)
        np.testing.assert_allclose(shifted,g.reassigned_prediction,rtol=0,atol=1e-10)
        max_replay=max(max_replay,float(np.max(np.abs(p-g.predicted_return))))
        max_shift=max(max_shift,float(np.max(np.abs(shifted-g.reassigned_prediction))))
        del model,values;torch.cuda.empty_cache()
        if number%3==0:print(f'Replayed {number}/18 models',flush=True)
    assert order_count==180 and boundary_count==180
    for (rule,date),g in seed.groupby(['rule','date']):
        stored=ensemble[(ensemble.rule==rule)&(ensemble.date==date)].iloc[0]
        assert abs(stored.predicted_return-g.predicted_return.mean())<1e-12
        assert abs(stored.reassigned_prediction-g.reassigned_prediction.mean())<1e-12
    flags=verify_scores(seed,ensemble)
    probe=json.loads((OUT/'probe_manifest.json').read_text(encoding='utf-8'))
    assert probe['parameters_unchanged'] and probe['optimizer_steps_applied']==0 and probe['probe_states']==3
    assert probe['maximum_balanced_full_gradient_relative_error']<1e-4
    assert old_evidence()==prep['old_evidence'];check_frozen()
    result=dict(status='PASS',new_fits=9,reused_models=9,total_checkpoints_replayed=18,validation_weeks=141,
        epoch_order_hashes_recomputed=order_count,boundary_hashes_recomputed=boundary_count,
        maximum_prediction_replay_error=max_replay,maximum_reassignment_replay_error=max_shift,primary_comparisons_recomputed=2,
        flags=flags,source_input_protocol_preserved=True,previous_files_preserved=len(prep['old_evidence']),
        elapsed_seconds=time.time()-start,completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

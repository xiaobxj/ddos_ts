"""Post-fit evidence audit; no selection or training changes."""
from common8 import *


def independent_policy(obs, cutoff, name):
    full = np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy())
    tr = full.copy(); w = np.ones(len(tr))
    if name.startswith('disjoint'):
        phase = int(name[-1]); first = int(obs.anchor.iloc[full[0]])
        kept = []; previous_end = -1
        for i in full:
            a = int(obs.anchor.iloc[i]); entry = int(obs.entry.iloc[i])
            if (a-first)%6 == phase and entry>previous_end:
                kept.append(i); previous_end=max(int(obs.exit.iloc[i]),a+5)
        tr=np.asarray(kept); w=np.ones(len(tr))
    elif name=='quarter_equal':
        labels=pd.to_datetime(obs.date.iloc[full]).dt.to_period('Q').astype(str).tolist()
        quarters=set(labels); counts={q:labels.count(q) for q in quarters}
        w=np.array([len(full)/len(quarters)/counts[q] for q in labels])
    elif name=='recent3y':
        tr=full[obs.date.iloc[full].ge(f'{int(cutoff[:4])-2}-01-01').to_numpy()];w=np.ones(len(tr))
    else: assert name=='full'
    return tr,w,full


def assert_metrics(row, g):
    y=g.actual.to_numpy();p=g.predicted_return.to_numpy();mse=float(np.mean((p-y)**2))
    values=dict(mse=mse,rmse=mse**.5,accuracy=float(((p>0)==(y>0)).mean()),
        mse_skill_vs_training_mean=1-mse/float(np.mean((g.training_mean-y)**2)),
        mse_skill_vs_full_mean=1-mse/float(np.mean((g.full_training_mean-y)**2)),
        reassigned_mse=float(np.mean((g.reassigned_prediction-y)**2)))
    for k,v in values.items():assert abs(row[k]-v)<1e-12,(k,row[k],v)


def verify_statistics(ensemble,seed):
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    for r in metrics.to_dict('records'):
        assert_metrics(r,ensemble[(ensemble.model==r['model'])&(ensemble.policy==r['policy'])])
    for r in pd.read_csv(OUT/'seed_metrics.csv').to_dict('records'):
        assert_metrics(r,seed[(seed.policy==r['policy'])&(seed.seed==r['seed'])])
    for r in pd.read_csv(OUT/'yearly_metrics.csv').to_dict('records'):
        assert_metrics(r,ensemble[(ensemble.model==r['model'])&(ensemble.policy==r['policy'])&(ensemble.date.str[:4]==str(r['year']))])
    neural=ensemble[ensemble.model=='neural']; full=neural[neural.policy=='full'].sort_values('date')
    primary=json.loads((OUT/'primary_comparisons.json').read_text(encoding='utf-8'));assert len(primary)==10
    ps=[]
    for r in primary:
        g=neural[neural.policy==r['policy']].sort_values('date');y=g.actual.to_numpy();a=g.predicted_return.to_numpy()
        assert g.date.tolist()==full.date.tolist()
        b=full.predicted_return.to_numpy() if r['reference']=='full_neural' else g.training_mean.to_numpy()
        va=(a-y)**2;vb=(b-y)**2;d=va-vb;n=len(y)
        rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))))
        ids=((starts[:,:,None]+np.arange(8))%n).reshape(10000,-1)[:,:n]
        boot=d[ids].mean(axis=1);centered=(d-d.mean())[ids].mean(axis=1)
        p=float((1+(np.abs(centered)>=abs(d.mean())).sum())/10001);ps.append(p)
        assert abs(p-r['mse']['p'])<1e-12 and abs(d.mean()-r['mse']['difference'])<1e-12
        np.testing.assert_allclose(np.quantile(boot,[.025,.975]),[r['mse']['ci95_low'],r['mse']['ci95_high']],rtol=0,atol=1e-12)
        rmse=np.sqrt(va[ids].mean(axis=1))-np.sqrt(vb[ids].mean(axis=1))
        np.testing.assert_allclose(np.quantile(rmse,[.025,.975]),[r['rmse']['ci95_low'],r['rmse']['ci95_high']],rtol=0,atol=1e-12)
    adjusted=np.zeros(10);running=0.
    for i,j in enumerate(np.argsort(ps)):
        running=max(running,min(1.,ps[j]*(10-i)));adjusted[j]=running
    np.testing.assert_allclose(adjusted,[r['mse']['holm_adjusted_p'] for r in primary],rtol=0,atol=1e-12)
    screens={}
    table=pd.read_csv(OUT/'policy_comparisons.csv')
    full_mse=float(np.mean((full.predicted_return-full.actual)**2))
    for policy in cfg()['policies']:
        g=neural[neural.policy==policy];mse=float(np.mean((g.predicted_return-g.actual)**2))
        own=float(np.mean((g.training_mean-g.actual)**2));common=float(np.mean((g.full_training_mean-g.actual)**2))
        count=sum(np.mean((h.predicted_return-h.actual)**2)<np.mean((h.training_mean-h.actual)**2) for _,h in seed[seed.policy==policy].groupby('seed'))
        r=table[table.policy==policy].iloc[0]
        assert count==r.seeds_better_than_own_mean
        assert abs(1-mse/full_mse-r.mse_skill_vs_full_neural)<1e-12
        if policy!='full':screens[policy]=bool(mse<full_mse and mse<own and mse<common and count>=2)
    assessment=json.loads((OUT/'assessment.json').read_text(encoding='utf-8'))
    assert assessment['policy_screens']==screens
    assert assessment['disjoint_family_screen']==all(screens[f'disjoint{i}'] for i in range(3))
    return screens


def main():
    legacy.initialize();start=time.time();prep=check_frozen()
    assert old_evidence()==prep['old_evidence']
    assert (OUT/'observation_table.csv').read_bytes()==(V7/'results/observation_table.csv').read_bytes()
    obs=pd.read_csv(OUT/'observation_table.csv');seed=pd.read_csv(OUT/'all_seed_predictions.csv')
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv')
    assert len(seed)==2538 and len(ensemble)==1692
    assert set(seed.policy)==set(cfg()['policies']) and set(seed.seed)==set(cfg()['training']['seeds'])
    assert seed.groupby(['policy','date']).size().eq(3).all()
    assert all(len(g)==141 for _,g in seed.groupby(['policy','seed']))
    curves=pd.concat([pd.read_csv(OUT/f'worker{w}_training_curves.csv') for w in [0,1]],ignore_index=True)
    assert len(curves)==900
    refs=json.loads((OUT/'reused_checkpoint_manifest.json').read_text(encoding='utf-8'))
    for worker in [0,1]:
        run=json.loads((OUT/f'worker{worker}_manifest.json').read_text(encoding='utf-8'))
        assert run.get('finished_utc') and run['fits']==[23,22][worker]
        assert run['source_sha256']==source_hashes() and run['input_sha256']==input_hashes()
        refs+=json.loads((OUT/f'worker{worker}_checkpoint_manifest.json').read_text(encoding='utf-8'))
    assert len(refs)==54 and len({(r['policy'],r['cutoff'],r['seed']) for r in refs})==54
    with np.load(V5/'cache/targets.npz') as data: raw_targets={k:data[k].copy() for k in ['returns','auxiliary']}
    maximum=0.;maximum_shift=0.;orders=0
    for number,r in enumerate(refs,1):
        path=PROJECT/r['project_file'];assert sha(path)==r['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        tr,w,te,n=load_policy(r['cutoff'],r['policy']);independent,iw,full=independent_policy(obs,r['cutoff'],r['policy'])
        np.testing.assert_array_equal(tr,independent);np.testing.assert_allclose(w,iw,rtol=0,atol=1e-12)
        for k,yall in raw_targets.items():
            y=yall[tr].astype(float);mean=np.average(y,axis=0,weights=w)
            sd=np.maximum(np.sqrt(np.average((y-mean)**2,axis=0,weights=w)),1e-6)
            np.testing.assert_allclose(mean,saved['scales'][k+'_mean'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(sd,saved['scales'][k+'_sd'],rtol=0,atol=1e-12)
        g=seed[(seed.policy==r['policy'])&(seed.cutoff==r['cutoff'])&(seed.seed==r['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index.to_numpy(),te)
        np.testing.assert_allclose(g.actual,raw_targets['returns'][te],rtol=0,atol=1e-12)
        assert obs.joint_completed.iloc[te].le('2020-12-31').all()
        if r['policy']!='full':
            job=next(j for j in fit_plan() if j['policy']==r['policy'] and j['cutoff']==r['cutoff'] and j['seed']==r['seed'])
            assert saved['job_index']==job['job_index'] and saved['epoch']==20
            group=curves[curves.job_index==job['job_index']].sort_values('epoch')
            assert group.epoch.tolist()==list(range(1,21)) and group.train_n.eq(len(tr)).all()
            rng=np.random.default_rng(r['seed'])
            for row in group.itertuples():
                order=[]
                while len(order)<n:order.extend(rng.permutation(len(tr)).tolist())
                positions=tr[np.asarray(order[:n])]
                assert row.training_order_sha256==order_hash(positions)
                assert row.presentations==n and row.optimizer_steps==int(np.ceil(n/128));orders+=1
        model=legacy.make_model('combined',r['seed']).eval();model.load_state_dict(saved['state_dict'])
        values=legacy.batch_tensors(te)
        with torch.inference_mode():
            output,_=legacy.predict(model,values,False)
            altered,_=legacy.predict(model,{k:torch.roll(t,1,dims=0) for k,t in values.items()},False)
        p=output.cpu().numpy().astype(float)*saved['scales']['returns_sd']+saved['scales']['returns_mean']
        shift=altered.cpu().numpy().astype(float)*saved['scales']['returns_sd']+saved['scales']['returns_mean']
        np.testing.assert_allclose(p,g.predicted_return,rtol=0,atol=1e-10)
        np.testing.assert_allclose(shift,g.reassigned_prediction,rtol=0,atol=1e-10)
        maximum=max(maximum,float(np.max(np.abs(p-g.predicted_return))))
        maximum_shift=max(maximum_shift,float(np.max(np.abs(shift-g.reassigned_prediction))))
        del model,values;torch.cuda.empty_cache()
        if number%9==0:print(f'Replayed {number}/54 neural checkpoints',flush=True)
    assert orders==900
    with np.load(V5/'cache/packed_raw.npz') as data:x=data['patches'].reshape(len(data['patches']),-1).astype(float)
    linear=pd.read_csv(OUT/'linear_predictions.csv');linear_refs=json.loads((OUT/'linear_checkpoint_manifest.json').read_text(encoding='utf-8'))
    assert len(linear_refs)==18 and len(linear)==846
    max_linear=0.;max_normal=0.
    for r in linear_refs:
        assert sha(OUT.parent/r['file'])==r['sha256']
        tr,w,_,n=load_policy(r['cutoff'],r['policy']);w=w*n/w.sum()
        g=linear[(linear.policy==r['policy'])&(linear.cutoff==r['cutoff'])].sort_values('row_index')
        with np.load(ROOT/r['file']) as m:
            xm=np.average(x[tr],axis=0,weights=w);xs=np.maximum(np.sqrt(np.average((x[tr]-xm)**2,axis=0,weights=w)),1e-6)
            ym=np.average(raw_targets['returns'][tr],weights=w)
            np.testing.assert_allclose(m['feature_mean'],xm,rtol=0,atol=1e-12)
            np.testing.assert_allclose(m['feature_sd'],xs,rtol=0,atol=1e-12)
            assert abs(float(m['target_mean'])-ym)<1e-12 and float(m['alpha'])==10.
            z=(x[tr]-xm)/xs;beta=m['coefficients'];residual=raw_targets['returns'][tr]-ym-z@beta
            normal=float(np.max(np.abs(z.T@(w*residual)-10*beta)))
            assert normal<1e-7;assert abs(np.dot(w,residual))<1e-7;max_normal=max(max_normal,normal)
            p=(x[g.row_index.to_numpy()]-xm)/xs@beta+ym
        np.testing.assert_allclose(p,g.predicted_return,rtol=0,atol=1e-10)
        np.testing.assert_allclose(np.roll(p,1),g.reassigned_prediction,rtol=0,atol=1e-10)
        max_linear=max(max_linear,float(np.max(np.abs(p-g.predicted_return))))
    for policy,g in seed.groupby('policy'):
        averaged=g.groupby('date').predicted_return.mean().sort_index()
        stored=ensemble[(ensemble.model=='neural')&(ensemble.policy==policy)].set_index('date').predicted_return.sort_index()
        np.testing.assert_allclose(averaged,stored,rtol=0,atol=1e-12)
    screens=verify_statistics(ensemble,seed)
    train=pd.read_csv(OUT/'frozen_model_training_predictions.csv')
    assert len(train)==17292
    np.testing.assert_allclose((train.predicted_return-train.actual)**2,train.fitted_return_loss,rtol=0,atol=1e-12)
    assert old_evidence()==prep['old_evidence'];check_frozen()
    result=dict(status='PASS',total_neural_checkpoints_replayed=54,new_neural_fits=45,reused_neural_checkpoints=9,
        linear_models_replayed=18,training_order_hashes_recomputed=orders,maximum_neural_replay_error=maximum,
        maximum_reassignment_replay_error=maximum_shift,maximum_linear_replay_error=max_linear,
        maximum_linear_normal_equation_residual=max_normal,primary_comparisons_recomputed=10,policy_screens=screens,
        previous_seven_round_files_preserved=1794,withdrawn_preflight_files_preserved=62,
        previous_and_preflight_files_preserved=len(prep['old_evidence']),source_input_and_protocol_unchanged=True,
        no_preflight_predictions_used=True,elapsed_seconds=time.time()-start,completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

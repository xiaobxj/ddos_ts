"""Fresh stock annual backbones and fixed uniform/time-weighted heads."""
from stock import *

def fit(frame,cutoff):
    folder=OUT/'annual'/cutoff;folder.mkdir(parents=True,exist_ok=False)
    r,w,_=modules(True);a=annual_interface(frame,cutoff);ids=a['training_rows'];n=len(ids)
    values={k:r.torch.from_numpy(v).cuda() for k,v in a['values'].items()}
    labels={k:r.torch.from_numpy(v).cuda() for k,v in a['labels'].items()}
    csv(folder/'membership.csv',a['observations'].iloc[ids]);save(folder/'scales.json',a['scales'])
    raw,weights,age=w.weights_for(a['training_dates'],cutoff);settings=read(PROJECT/'research_v47/protocol.json')
    models=[];uniform=[];weighted=[];curves=[]
    # A simple train-only model determines whether the neural representation adds value.
    marketfit=r.fit_clip(a['market']);markettheta,_=r.fit_newton(marketfit['standardized'],a['y'],r.cfg()['probe'])
    np.savez_compressed(folder/'market4.npz',coefficients=markettheta,**{k:v for k,v in marketfit.items() if k in ['lower','upper','mean','sd']})
    for seed in SEEDS:
        model=r.legacy.make_model('combined',seed);optimizer=r.training.optimizer_for(model);rng=np.random.default_rng(seed)
        for epoch in range(1,21):
            model.train();order=rng.permutation(n);chunks=r.training.batches(order);sums=np.zeros(3)
            for positions in chunks:
                loss,ret,aux,_=r.training.update_step(model,optimizer,values,labels,positions)
                sums+=np.array([loss,ret,aux])*len(positions)
            curves.append(dict(seed=seed,epoch=epoch,train_n=n,steps=len(chunks),permutation_sha256=r.training.array_hash(order),
                joint_loss=sums[0]/n,return_loss=sums[1]/n,auxiliary_loss=sums[2]/n))
            if epoch%5==0:print(f'603986 {cutoff} seed={seed} epoch={epoch}/20 n={n} loss={sums[0]/n:.5f}',flush=True)
        state=r.training.capture(model,optimizer,rng,arm='combined',history='stock_rolling5',epoch=20,cutoff=cutoff,seed=seed,
            train_n=n,full_train_n=len(a['observations']),scales=a['scales'],stock='sh603986',protocol_sha256=sha(ROOT/'protocol.json'),freeze_sha256=sha(OUT/'freeze.json'))
        path=folder/f'model_{seed}.pt';r.torch.save(state,path)
        ref=dict(cutoff=cutoff,seed=seed,epoch=20,project_file=relative(path),sha256=sha(path),model_sha256=state['model_sha256']);models.append(ref)
        optimizer.zero_grad(set_to_none=True);model.eval();model.requires_grad_(False)
        features,native=r.neural.extract_features(model,values)
        d,coefs,_,_=r.fit_pipeline(features,a['market'],a['gate'],a['y']);d.update(row_index=ids,native_standardized=native)
        pipeline=folder/f'pipeline_{seed}.npz';np.savez_compressed(pipeline,**d)
        wc=[]
        for x in [d['x18'],d['x19'],d['x23']]:
            theta,_=w.fit_newton(x,a['y'],weights,settings['probe']);wc.append(theta)
        gamma,_=w.fit_offset(w.design(d['x19'])@wc[1],d['x23'][:,-1],a['y'],weights,settings['offset_probe'])
        wc.append(np.r_[wc[1][:-1],gamma,wc[1][-1]])
        for method,theta,wt in zip(METHODS,coefs,wc):
            shared=dict(method=method,seed=seed,cutoff=cutoff,cache_file=relative(pipeline),cache_sha256=sha(pipeline))
            uniform.append(dict(shared,coefficients=theta.tolist()));weighted.append(dict(shared,coefficients=wt.tolist()))
        del model,optimizer,features,state;r.torch.cuda.empty_cache()
    csv(folder/'training_curves.csv',pd.DataFrame(curves))
    save(folder/'annual.json',dict(stock='sh603986',annual_cutoff=cutoff,models=models,uniform=uniform,weighted=weighted,
        volatility_median=float(np.median(a['market'][:,1])),return_scale=a['scales'],training_up_frequency=float(a['y'].mean()),
        training_n=n,first_anchor=a['training_dates'].min(),last_anchor=a['training_dates'].max(),
        artifacts={relative(p):sha(p) for p in folder.iterdir() if p.is_file()}))

def main():
    check_freeze();frame=data()
    assert not (OUT/'training_started.json').exists(),'Existing run must be reviewed, never silently overwritten'
    save(OUT/'training_started.json',dict(started_utc=now(),freeze_sha256=sha(OUT/'freeze.json')))
    for cutoff in cfg()['cutoffs']:fit(frame,cutoff)
    check_freeze()
    save(OUT/'training_completed.json',dict(completed_utc=now(),annual_fits=4,neural_fits=12,epochs_per_fit=20,
        annual_manifests={c:sha(OUT/'annual'/c/'annual.json') for c in cfg()['cutoffs']}))

if __name__=='__main__':main()

"""Natural20 annual retraining and the unchanged R47 weighted heads."""
from common50 import *
from dataset50 import annual_interface

def fit_annual(frame,cutoff,folder):
    guard(cutoff.endswith('12-31'),'Annual refit requires Dec31');folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    recipe,weighted,_=modules(True);a=annual_interface(frame,cutoff);ids=a['training_rows'];n=len(ids)
    values={k:recipe.torch.from_numpy(v).cuda() for k,v in a['values'].items()};labels={k:recipe.torch.from_numpy(v).cuda() for k,v in a['labels'].items()}
    csv_save(folder/'membership.csv',a['observations'].iloc[ids]);save(folder/'scales.json',a['scales'])
    np.savez_compressed(folder/'training_interface.npz',**a['values'],**a['labels'],row_index=ids,market_features=a['market'],gate=a['gate'],direction=a['y'])
    raw,weights,age=weighted.weights_for(a['training_dates'],cutoff);settings=read(PROJECT/'research_v47/protocol.json');models=[];uniform=[];weightheads=[];curves=[];headchecks=[]
    for seed in SEEDS:
        model=recipe.legacy.make_model('combined',seed);optimizer=recipe.training.optimizer_for(model);rng=np.random.default_rng(seed)
        for epoch in range(1,21):
            model.train();order=rng.permutation(n);chunks=recipe.training.batches(order);sums=np.zeros(3)
            for positions in chunks:
                loss,ret,aux,_=recipe.training.update_step(model,optimizer,values,labels,positions);sums+=np.array([loss,ret,aux])*len(positions)
            curves.append(dict(seed=seed,epoch=epoch,train_n=n,steps=len(chunks),permutation_sha256=recipe.training.array_hash(order),joint_loss=sums[0]/n,return_loss=sums[1]/n,auxiliary_loss=sums[2]/n))
            if epoch%5==0:print(f'Annual replay/build {cutoff},seed{seed}: {epoch}/20 natural epochs.',flush=True)
        state=recipe.training.capture(model,optimizer,rng,arm='combined',history='rolling5',epoch=20,cutoff=cutoff,seed=seed,train_n=n,full_train_n=len(a['observations']),scales=a['scales'],protocol_sha256=sha(PROJECT/'research_v49/protocol.json'),implementation_protocol_sha256=sha(ROOT/'protocol.json'))
        path=folder/f'model_{seed}.pt';recipe.torch.save(state,path)
        ref=dict(history='rolling5',cutoff=cutoff,seed=seed,epoch=20,train_n=n,full_train_n=len(a['observations']),project_file=relative(path),sha256=sha(path),model_sha256=state['model_sha256'],optimizer_sha256=state['optimizer_sha256'],rng_sha256=state['rng_sha256']);models.append(ref)
        optimizer.zero_grad(set_to_none=True);model.eval();model.requires_grad_(False);features,native=recipe.neural.extract_features(model,values)
        d,coefs,traces,stats=recipe.fit_pipeline(features,a['market'],a['gate'],a['y']);d.update(row_index=ids,native_standardized=native)
        pipeline=folder/f'pipeline_{seed}.npz';np.savez_compressed(pipeline,**d)
        weighted_coefs=[]
        for x in [d['x18'],d['x19'],d['x23']]:
            theta,_=weighted.fit_newton(x,a['y'],weights,settings['probe']);_,grad,hess=weighted.objective(theta,weighted.design(x),a['y'],weights,.01)
            guard(abs(grad).max()<=1e-9 and np.linalg.eigvalsh(hess).min()>0,'Weighted head optimality failed');weighted_coefs.append(theta)
        offset=weighted.design(d['x19'])@weighted_coefs[1];gamma,_=weighted.fit_offset(offset,d['x23'][:,-1],a['y'],weights,settings['offset_probe']);weighted_coefs.append(np.r_[weighted_coefs[1][:-1],gamma,weighted_coefs[1][-1]])
        for method,theta,wt in zip(METHODS,coefs,weighted_coefs):
            shared=dict(method=method,seed=seed,cutoff=cutoff,train_n=n,cache_file=relative(pipeline),cache_sha256=sha(pipeline),model_project_file=ref['project_file'])
            uniform.append(dict(shared,history=U,coefficients=theta.tolist()));weightheads.append(dict(shared,history=W,coefficients=wt.tolist()))
        del model,optimizer,features,state;recipe.torch.cuda.empty_cache()
    np.savez_compressed(folder/'training_weights.npz',row_index=ids,raw_weight=raw,weight=weights,age_days=age)
    csv_save(folder/'training_curves.csv',pd.DataFrame(curves))
    result=dict(annual_cutoff=cutoff,models=models,uniform=uniform,weighted=weightheads,volatility_median=float(np.quantile(a['market'][:,1],.5,method='linear')),return_scale=a['scales'],training_up_frequency=float(a['y'].mean()),training_n=n,annual_artifacts=artifacts(folder))
    save(folder/'annual.json',result);return result

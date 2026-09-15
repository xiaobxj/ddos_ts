"""84 independent fresh fits, in two separate horizon worker processes."""
from core import *
import argparse, traceback

def step(model,opt,values,labels,y,idx,objective,h,r):
    torch=r.torch;batch=torch.tensor(idx,device='cuda');opt.zero_grad(set_to_none=True)
    pred,aux=r.legacy.predict(model,{k:v[batch] for k,v in values.items()},True)
    ret=((pred-labels['returns'][batch])**2).mean() if objective=='mse' else torch.nn.functional.binary_cross_entropy_with_logits(pred,y[batch])
    al=((aux[:,:5*h]-labels['auxiliary'][batch])**2).mean();loss=ret+.1*al
    assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return np.array([float(loss.detach()),float(ret.detach()),float(al.detach()),float(gn)])

def snapshot(model,opt,rng,a,values,labels,task,seed,epoch,folder,r,w):
    path=folder/f'model_{seed}_e{epoch}.pt'
    state=r.training.capture(model,opt,rng,**task,seed=seed,epoch=epoch,scales=a['scales'],
        train_n=len(a['ids']),protocol_sha256=sha(ROOT/'protocol.json'),freeze_sha256=sha(OUT/'freeze.json'))
    r.torch.save(state,path)
    # Keep the saved torch RNG and training flags unchanged by diagnostic inference.
    model.eval();model.requires_grad_(False)
    features,native=r.neural.extract_features(model,values)
    ids=a['ids'];d,coefs,traces,metrics=r.fit_pipeline(features[ids],a['market'][ids],a['gate'][ids],a['y'])
    d.update(row_index=ids)
    xs=r.apply_pipeline(features,a['market'],a['gate'],d)
    _,weights,_=w.weights_for(a['obs'].date.iloc[ids],task['cutoff']);settings=read(PROJECT/'research_v47/protocol.json')
    wc=[]
    for x in [d['x18'],d['x19'],d['x23']]:wc.append(w.fit_newton(x,a['y'],weights,settings['probe'])[0])
    gamma,_=w.fit_offset(w.design(d['x19'])@wc[1],d['x23'][:,-1],a['y'],weights,settings['offset_probe'])
    wc.append(np.r_[wc[1][:-1],gamma,wc[1][-1]])
    probabilities={};headrows=[]
    for method,x,u,weighted,metric in zip(METHODS,xs,coefs,wc,metrics):
        inter=u.copy();inter[-1]=weighted[-1]
        slopes=weighted.copy();slopes[-1]=u[-1]
        for scheme,theta in [('U',u),('W',weighted),('I',inter),('S',slopes)]:
            probabilities[f'{method}.{scheme}']=r.probability(r.design(x)@theta)
            headrows.append(dict(method=method,scheme=scheme,coefficients=theta.tolist(),uniform_solver=metric))
    # Scalar score calibration here is fit on training data, explicitly NOT OOS calibration.
    nfit=r.fit_clip(native[ids,None]);nt,_=r.fit_newton(nfit['standardized'],a['y'],r.cfg()['probe'])
    probabilities['scalar_probe.U']=r.probability(r.design(r.transform(native[:,None],nfit))@nt)
    if task['objective']=='bce':probabilities['native_bce.U']=r.probability(native)
    train_pred=native[ids]
    if task['objective']=='mse':loss=float(np.mean((train_pred-a['labels']['returns'])**2))
    else:loss=float(np.mean(np.logaddexp(0,train_pred)-a['y']*train_pred))
    # Deterministic input-use diagnostic, not a scored alternative strategy.
    selected=np.linspace(0,len(ids)-1,32,dtype=int);original=ids[selected]
    altered={k:v[original].clone() for k,v in values.items()}
    altered['patches']=altered['patches'].flip([1,2])
    pf,pn=r.neural.extract_features(model,altered)
    diag=dict(train_eval_primary_loss=loss,native_training_sd=float(train_pred.std()),
        representation_sd_mean=float(features[ids].std(0).mean()),
        time_reversal_native_mean_abs_change=float(np.mean(abs(pn-native[original]))),
        time_reversal_representation_mean_abs_change=float(np.mean(abs(pf-features[original]))),
        input_use_warning='nonzero change is not proof of predictive information',
        feature_requires_grad=False,training_feature_labels='in-sample, not OOF')
    cache=folder/f'cache_{seed}_e{epoch}.npz'
    np.savez_compressed(cache,**d,all_features=features,all_native=native,**{'p__'+k:v for k,v in probabilities.items()})
    save(folder/f'heads_{seed}_e{epoch}.json',dict(heads=headrows,diagnostics=diag,
        scalar_transform={k:np.asarray(v).tolist() for k,v in nfit.items() if k!='standardized'},scalar_coefficients=nt.tolist(),
        checkpoint_sha256=sha(path),cache_sha256=sha(cache),model_sha256=state['model_sha256']))
    model.requires_grad_(True);model.train()
    r.torch.set_rng_state(state['cpu_rng_state']);r.torch.cuda.set_rng_state(state['cuda_rng_state'])

def fit(task):
    folder=OUT/'fits'/task_id(task)
    if (folder/'completed.json').exists():
        d=read(folder/'completed.json')
        for p,s in d['files'].items():assert sha(folder/p)==s
        print('VERIFIED_RESUME '+task_id(task),flush=True);return
    folder.mkdir(parents=True,exist_ok=False)
    r,w,_=modules();a=interface(data(),task);ids=a['ids'];n=len(ids)
    csv(folder/'membership.csv',a['obs'].iloc[ids]);save(folder/'scales.json',a['scales'])
    values={k:r.torch.from_numpy(v).cuda() for k,v in a['values'].items()}
    training_values={k:v[ids] for k,v in values.items()}
    labels={k:r.torch.from_numpy(v).cuda() for k,v in a['labels'].items()};y=r.torch.tensor(a['y'],dtype=r.torch.float32,device='cuda')
    mk=r.fit_clip(a['market'][ids]);theta,_=r.fit_newton(mk['standardized'],a['y'],r.cfg()['probe'])
    marketprob=r.probability(r.design(r.transform(a['market'],mk))@theta)
    np.savez_compressed(folder/'controls.npz',market_probability=marketprob,market_theta=theta,
        market=a['market'],gate=a['gate'],**{k:v for k,v in mk.items() if k!='standardized'})
    curves=[];seedrefs=[]
    for seed in SEEDS:
        model=r.legacy.make_model('combined',seed)
        if task['objective']=='bce':
            p=float(np.clip(a['y'].mean(),1e-6,1-1e-6))
            with r.torch.no_grad():model.return_head.bias.fill_(float(np.log(p/(1-p))))
        opt=r.training.optimizer_for(model);rng=np.random.default_rng(seed)
        for epoch in range(1,21):
            order=rng.permutation(n);chunks=r.training.batches(order);sums=np.zeros(4);exposure=np.zeros(n)
            for idx in chunks:
                out=step(model,opt,training_values,labels,y,idx,task['objective'],task['h'],r)
                sums+=out*len(idx);exposure[idx]=1/len(idx)
            curves.append(dict(seed=seed,epoch=epoch,train_n=n,steps=len(chunks),last_batch=len(chunks[-1]),
                nominal_loss_weight_max_min=float(exposure.max()/exposure.min()),
                permutation_sha256=r.training.array_hash(order),joint_loss=sums[0]/n,return_loss=sums[1]/n,aux_loss=sums[2]/n,gradient_norm=sums[3]/n))
            if epoch%5==0:print(f'{task_id(task)} seed={seed} epoch={epoch}/20 n={n} loss={sums[0]/n:.5f}',flush=True)
            if epoch in [10,20]:snapshot(model,opt,rng,a,values,labels,task,seed,epoch,folder,r,w)
        del model,opt;r.torch.cuda.empty_cache()
    csv(folder/'training_curves.csv',pd.DataFrame(curves))
    save(folder/'completed.json',dict(**task,completed_utc=now(),training_n=n,training_frequency=float(a['y'].mean()),
        volatility_median=float(np.median(a['market'][ids,1])),first_anchor=a['obs'].date.iloc[ids].min(),
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--h',type=int,choices=[1,5],required=True);a=ap.parse_args()
    check_freeze();modules(True)
    for task in tasks(a.h):fit(task)
    check_freeze();save(OUT/f'training_h{a.h}_completed.json',dict(status='PASS',completed_utc=now(),fits=42,
        tasks={task_id(t):sha(OUT/'fits'/task_id(t)/'completed.json') for t in tasks(a.h)}))

if __name__=='__main__':
    try:main()
    except Exception:
        text=traceback.format_exc();p=OUT/f'failure_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.txt'
        p.parent.mkdir(exist_ok=True);p.write_text(text,encoding='utf-8');raise

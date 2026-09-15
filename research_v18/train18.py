from common18 import *

def main():
    check_frozen();path=OUT/'training_manifest.json';assert not path.exists(),'Preserve training'
    run=manifest('training');save(path,run);start=time.time();obs,price,returns=data();contexts=[];files=[]
    for fold in cfg()['folds']:
        tr,te=indices(obs,fold);f=market_rows('training',fold['cutoff'],tr);d=fit_market(f);y=(returns[tr]>0).astype(float)
        path=CACHE/f"market_training_{fold['cutoff']}.npz";np.savez_compressed(path,**d,row_index=tr,direction=y);files.append(path)
        contexts.append(dict(cutoff=fold['cutoff'],train_n=len(tr),validation_rows=te.tolist(),cache_file=str(path.relative_to(PROJECT)),cache_sha256=sha(path)))
    save(OUT/'market_contexts.json',contexts);files.append(OUT/'market_contexts.json');contexts={c['cutoff']:c for c in contexts}
    parents={h['job']:h for h in read(OUT/'parent_heads.json')};val_refs={r['job']:r for r in read(V16/'results/validation_features.json')}
    heads=[];traces=[];metrics=[];coefficients=[];train_rows=0
    for job in read(OUT/'jobs.json'):
        context=contexts[job['cutoff']];c=load_npz(context);x=inputs(job,'training',context,parents,val_refs);y=c['direction']
        theta,trace=fit_newton(x,y,cfg()['probe']);value,g,h=objective(theta,design(x),y,.01);assert np.max(np.abs(g))<=1e-9 and np.linalg.eigvalsh(h).min()>0
        rate=float(y.mean());constant=base16.previous.probe_metrics(np.full(len(y),np.log(rate/(1-rate))),y)
        parent_value=parents[job['parent_job']]['objective'] if job['parent_job'] is not None else constant['log_loss']
        assert value<=parent_value+1e-12
        path=CACHE/f"training_{job['job']}.npz";np.savez_compressed(path,standardized=x,direction=y,row_index=c['row_index']);files.append(path)
        head=dict(job,cache_file=str(path.relative_to(PROJECT)),cache_sha256=sha(path),coefficients=theta.tolist(),train_n=len(y),iterations=trace[-1]['iteration'],
            objective=value,gradient_inf=float(np.max(np.abs(g))),hessian_min_eigenvalue=float(np.linalg.eigvalsh(h).min()),l2_lambda=.01,parent_or_constant_objective=parent_value)
        heads.append(head);train_rows+=len(y);traces.extend(dict(job=job['job'],**r) for r in trace)
        measures=base16.previous.probe_metrics(design(x)@theta,y)
        metrics.append(dict(job=job['job'],method=job['method'],cutoff=job['cutoff'],seed=job['seed'],train_n=len(y),dimensions=x.shape[1],iterations=head['iterations'],
            objective=value,gradient_inf=head['gradient_inf'],parent_or_constant_objective=parent_value,**measures,
            constant_brier=constant['brier'],constant_log_loss=constant['log_loss'],constant_accuracy=constant['accuracy']))
        names=[]
        if job['representation_dimensions']:
            names=[f'decoded_{j:02d}' for j in range(25)] if job['method']=='learned_market' else list(base16.NAMES)
        names += [f'market_{MARKET_NAMES[i]}' for i in job['market_indices']]+['intercept']
        assert len(names)==len(theta)
        for i,(name,value) in enumerate(zip(names,theta)):
            block='intercept' if i==len(theta)-1 else 'representation' if i<job['representation_dimensions'] else 'market'
            coefficients.append(dict(job=job['job'],method=job['method'],cutoff=job['cutoff'],seed=job['seed'],coordinate=i,name=name,block=block,coefficient=float(value)))
        save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False)
        print(f"Converged {len(heads)}/30: {job['job']}; slopes={job['dimensions']}; iterations={head['iterations']}",flush=True)
    assert len(heads)==30 and train_rows==47325 and len(coefficients)==732
    pd.DataFrame(metrics).to_csv(OUT/'training_metrics.csv',index=False);pd.DataFrame(coefficients).to_csv(OUT/'coefficients.csv',index=False)
    files += [OUT/n for n in ['heads.json','solver_trace.csv','training_metrics.csv','coefficients.csv']]
    finish(run,files,new_primary_fits=30,new_newton_iterations=sum(h['iterations'] for h in heads),training_feature_rows=train_rows,market_training_rows=9465,
        all_converged=True,all_nested_training_objectives_pass=True,validation_scoring_during_fit=False,neural_forward_passes=0,neural_training_steps=0,elapsed_seconds=time.time()-start)
    print('All30 heads converged before new held-out scoring. No neural forward or training pass.',flush=True)

if __name__=='__main__':main()

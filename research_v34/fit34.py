from common34 import *

def main():
    legacy.initialize();check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');obs,price,targets=data();ids=np.arange(len(obs));mf=market(price,obs,ids);u=gates(price,obs,ids);y=(targets['returns']>0).astype(float)
    banks=read(OUT/'feature_banks.json');annuals=read(OUT/'source_heads.json');endpoints=[];policies=[];traces=[];drifts=[];training_refs=[];files=[]
    for j in csv('jobs').query('refit').itertuples():
        tr=training_rows(obs,j.head_cutoff)
        for seed in cfg()['seeds']:
            bankref=next(b for b in banks if b['cutoff']==j.encoder_cutoff and b['seed']==seed);bank=arrays(bankref)
            hs=[next(h for h in annuals if h['cutoff']==j.encoder_cutoff and h['seed']==seed and h['method']==m) for m in LEARNED];d=arrays(hs[0]);xx,labels=training_interface(bank,tr,mf,u,y,d);ts,tt,ms=fit_coefficients(xx,labels)
            path=CACHE/f'training_{j.head_cutoff}_{seed}.npz';np.savez_compressed(path,row_index=tr,direction=labels,x18=xx[0],x19=xx[1],x23=xx[2]);files.append(path);training_refs.append(dict(cutoff=j.head_cutoff,encoder_cutoff=j.encoder_cutoff,seed=seed,**ref(path)))
            for m,h,t,trace,met in zip(LEARNED,hs,ts,tt,ms):
                job=f'{m}_{j.head_cutoff}_{seed}';r=dict(method=m,cutoff=j.head_cutoff,encoder_cutoff=j.encoder_cutoff,transform_cutoff=j.encoder_cutoff,seed=seed,job=job,train_n=len(tr),coefficients=t.tolist(),iterations=trace[-1]['iteration'],cache_file=h['cache_file'],cache_sha256=h['cache_sha256'],annual_job=h['job'],**met);endpoints.append(r);traces.extend(dict(job=job,**v) for v in trace)
                a=np.asarray(h['coefficients'])
                for policy in cfg()['updates']:
                    theta=shrink(a,t,policy['alpha']);policies.append(dict(**{k:v for k,v in r.items() if k not in ['coefficients','objective','reduced_objective','gradient_inf','hessian_min','iterations']},history=policy['history'],alpha=policy['alpha'],coefficients=theta.tolist(),endpoint_job=job,fit_kind='derived_not_separately_optimized'))
                    drifts.append(dict(history=policy['history'],method=m,cutoff=j.head_cutoff,seed=seed,alpha=policy['alpha'],annual_norm=float(np.linalg.norm(a)),endpoint_drift=float(np.linalg.norm(t-a)),policy_drift=float(np.linalg.norm(theta-a)),intercept_change=float(theta[-1]-a[-1]),maximum_abs_change=float(abs(theta-a).max())))
            print(f'Fitted {len(endpoints)}/204 fixed-coordinate endpoints: {j.head_cutoff} seed {seed}; no weekly scoring.',flush=True)
    assert len(endpoints)==204 and len(policies)==612 and len(training_refs)==51
    for n,v in [('endpoints.json',endpoints),('policy_heads.json',policies),('training_designs.json',training_refs)]:p=OUT/n;save(p,v);files.append(p)
    for n,v in [('solver_trace',traces),('coefficient_drift',drifts)]:p=OUT/f'{n}.csv';pd.DataFrame(v).to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_neural_fits=0,new_feature_inference=0,new_transform_fits=0,endpoint_fits=204,derived_policy_heads=612,all_converged=True,weekly_scoring_during_fit=False)
if __name__=='__main__':main()

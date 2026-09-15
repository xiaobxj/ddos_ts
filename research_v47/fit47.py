from common47 import *
def main():
    check_frozen();assert not (OUT/'fitting_manifest.json').exists();run=manifest('fitting');obs=csv('observation_table');source=read(OUT/'source_heads.json');heads=[];traces=[];metrics=[];files=[]
    for cutoff in cfg()['decision_dates']:
        for seed in cfg()['seeds']:
            src=[next(h for h in source if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in METHODS];d=arrays(src[0]);ids,xx,y,raw,w,age=training_interface(obs,d,cutoff)
            coefs=[];ms=[];ts=[]
            for x in xx:
                theta,trace=fit_newton(x,y,w,cfg()['probe']);val,grad,hess=objective(theta,design(x),y,w,.01);coefs.append(theta);ts.append(trace);ms.append(dict(objective=val,gradient_inf=float(abs(grad).max()),hessian_min=float(np.linalg.eigvalsh(hess).min())))
            offset=design(xx[1])@coefs[1];gamma,trace=fit_offset(offset,xx[2][:,-1],y,w,cfg()['offset_probe']);theta=np.r_[coefs[1][:-1],gamma,coefs[1][-1]];val,grad,hess=offset_objective(gamma,offset,xx[2][:,-1],y,w,.01);coefs.append(theta);ts.append(trace);ms.append(dict(objective=val+.005*float(coefs[1][:-1]@coefs[1][:-1]),reduced_objective=val,gradient_inf=abs(grad),hessian_min=hess))
            assert ms[2]['objective']<=ms[3]['objective']+1e-12<=ms[1]['objective']+2e-12 and ms[1]['objective']<=ms[0]['objective']+1e-12
            path=CACHE/f'training_{cutoff}_{seed}.npz';np.savez_compressed(path,row_index=ids,x18=xx[0],x19=xx[1],x23=xx[2],direction=y,weight=w,raw_weight=raw,age_days=age);files.append(path)
            for method,theta,trace,stats,old in zip(METHODS,coefs,ts,ms,src):
                job=f'{method}_{cutoff}_{seed}';h=dict(history=NEW[0],method=method,cutoff=cutoff,seed=seed,job=job,train_n=len(ids),coefficients=theta.tolist(),iterations=trace[-1]['iteration'],frozen_pipeline_file=old['cache_file'],frozen_pipeline_sha256=old['cache_sha256'],model_project_file=old['model_project_file'],cache_file=str(path.relative_to(PROJECT)),cache_sha256=sha(path),**stats);heads.append(h);traces.extend(dict(job=job,**r) for r in trace);metrics.append(dict(job=job,method=method,cutoff=cutoff,seed=seed,**stats))
            print(f'Fitted {len(heads)}/72 weighted heads: {cutoff},seed{seed}; no weekly scoring.',flush=True)
    assert len(heads)==72
    p=OUT/'heads.json';save(p,heads);files.append(p)
    for name,data in [('solver_trace',traces),('training_metrics',metrics)]:p=OUT/f'{name}.csv';pd.DataFrame(data).to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_neural_fits=0,new_feature_inference=0,new_head_fits=72,joint_logistic_fits=54,conditional_scalar_fits=18,all_converged=True,weekly_scoring_during_fit=False)
    print('All72weighted heads frozen before new scoring.',flush=True)
if __name__=='__main__':main()

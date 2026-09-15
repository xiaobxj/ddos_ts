from common24 import *
import verify15 as independent

def main():
    check_frozen();assert not (OUT/'training_manifest.json').exists();run=manifest('training');save(OUT/'training_manifest.json',run)
    obs,price,returns=data();sources={s['job']:s for s in read(OUT/'source_heads.json')};parents={h['job']:h for h in read(V19/'results/heads.json')};heads=[];traces=[];coefs=[];metrics=[];files=[];total=0
    for job in read(OUT/'jobs.json'):
        source=sources[job['source_job']];x,tr=inputs(source,'training');y=(returns[tr]>0).astype(float);xx=scaled_inputs(x,job['multiplier']);parent=parents[source['source_job']]
        pt=np.asarray(parent['coefficients']);nested=np.r_[pt[:-1],0.,pt[-1]];value,_,_=objective(nested,design(xx),y,.01);assert abs(value-parent['objective'])<1e-12
        theta,trace=fit_newton(xx,y,cfg()['probe']);beta=original_coefficients(theta,job['multiplier']);value,grad,hess=objective(theta,design(xx),y,.01);dv,dg=direct_objective(beta,x,y,job['multiplier'])
        assert abs(value-dv)<1e-12 and abs(grad).max()<=1e-9 and np.linalg.eigvalsh(hess).min()>0 and value<=parent['objective']+1e-12
        path=CACHE/f"training_{job['job']}.npz";np.savez_compressed(path,standardized=xx,original_input=x,row_index=tr,direction=y);files.append(path)
        h=dict(job,**reference(path),coefficients=theta.tolist(),original_coefficients=beta.tolist(),train_n=len(tr),iterations=trace[-1]['iteration'],objective=value,gradient_inf=float(abs(grad).max()),original_gradient_inf=float(abs(dg).max()),hessian_min_eigenvalue=float(np.linalg.eigvalsh(hess).min()),parent_objective=parent['objective'],l2_lambda=.01)
        heads.append(h);total+=len(tr);traces.extend(dict(job=h['job'],**r) for r in trace);metrics.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],train_n=len(tr),objective=value,**independent.independent_metrics(design(x)@beta,y)))
        for i,b in enumerate(beta):coefs.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],coordinate=i,coefficient=float(b),solver_coefficient=float(theta[i]),penalty=float(penalty_vector(x.shape[1],h['multiplier'])[i]),block='intercept' if i==len(beta)-1 else 'order_interaction' if i==len(beta)-2 else 'original_interaction' if i==len(beta)-3 else 'parent'))
        save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False);print(f"Converged {len(heads)}/48: {h['job']}; updates={h['iterations']}",flush=True)
    assert len(heads)==48 and len(coefs)==1500 and total==75720
    shrink=[]
    for source in sources.values():
        hs=sorted([h for h in heads if h['source_job']==source['job']],key=lambda h:h['multiplier']);values=[float(source['coefficients'][-2])]+[h['original_coefficients'][-2] for h in hs]
        assert all(abs(values[i+1])<=abs(values[i])+1e-7 for i in range(2))
        shrink.append(dict(source_job=source['job'],source_method=source['method'],cutoff=source['cutoff'],seed=source['seed'],coefficient_1x=values[0],coefficient_4x=values[1],coefficient_16x=values[2],absolute_coefficient_monotone=True))
    for name,rows in [('training_metrics',metrics),('coefficients',coefs),('shrinkage_path',shrink)]:path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    files += [OUT/'heads.json',OUT/'solver_trace.csv'];finish(run,files,new_primary_fits=48,new_projection_fits=0,reused_projection_fits=24,new_newton_iterations=sum(h['iterations'] for h in heads),training_feature_rows=total,all_converged=True,validation_scoring_during_fit=False,new_hmm_fits=0,neural_forward_passes=0,neural_training_steps=0)
    print('All48heads completed before any new nextyear scoring.',flush=True)
if __name__=='__main__':main()

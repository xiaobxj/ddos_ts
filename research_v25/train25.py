from common25 import *
import verify15 as independent

def main():
    check_frozen();assert not (OUT/'training_manifest.json').exists();run=manifest('training');save(OUT/'training_manifest.json',run);obs,price,returns=data();sources={s['job']:s for s in read(OUT/'source_heads.json')};parents={p['job']:p for p in read(V19/'results/heads.json')};heads=[];metrics=[];traces=[];coefficients=[];files=[];total=0
    for job in read(OUT/'jobs.json'):
        source=sources[job['source_job']];parent=parents[job['parent_job']];x,tr=inputs(source,'training');y=(returns[tr]>0).astype(float);pt=np.asarray(parent['coefficients']);offset=design(x[:,:-1])@pt;feature=x[:,-1];constant=float(.005*np.square(pt[:-1]).sum())
        gamma,trace=fit_offset(offset,feature,y,cfg()['probe']);value,grad,hess=offset_objective(gamma,offset,feature,y);full=value+constant;zero=offset_objective(0.,offset,feature,y)[0]+constant
        assert abs(zero-parent['objective'])<1e-12 and full<=parent['objective']+1e-12 and source['objective']<=full+1e-12;theta=np.r_[pt[:-1],gamma,pt[-1]]
        path=CACHE/f"training_{job['job']}.npz";np.savez_compressed(path,standardized=x,row_index=tr,direction=y,offset=offset,order_feature=feature);files.append(path)
        h=dict(job,**reference(path),gamma=gamma,coefficients=theta.tolist(),frozen_parent_coefficients=pt.tolist(),train_n=len(tr),iterations=trace[-1]['iteration'],reduced_objective=value,parent_penalty_constant=constant,objective=full,parent_objective=parent['objective'],joint_objective=source['objective'],gradient_absolute=abs(grad),hessian=hess,parent_coefficient_l2_drift=0.,parent_intercept_drift=0.)
        heads.append(h);total+=len(tr);traces.extend(dict(job=h['job'],**r) for r in trace);metrics.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],train_n=len(tr),objective=full,reduced_objective=value,**independent.independent_metrics(offset+gamma*feature,y)))
        for i,c in enumerate(theta):coefficients.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],coordinate=i,coefficient=float(c),trainable=i==len(theta)-2,block='intercept' if i==len(theta)-1 else 'order_interaction' if i==len(theta)-2 else 'original_interaction' if i==len(theta)-3 else 'parent'))
        save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False);print(f"Converged {len(heads)}/24: {h['job']}; scalar updates={h['iterations']}",flush=True)
    assert len(heads)==24 and total==37860 and len(coefficients)==750 and sum(c['trainable'] for c in coefficients)==24
    for name,rows in [('coefficients',coefficients),('training_metrics',metrics)]:path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    files += [OUT/'heads.json',OUT/'solver_trace.csv'];finish(run,files,new_primary_fits=24,new_trainable_coefficients=24,frozen_parent_coefficients=726,new_projection_fits=0,reused_projection_fits=24,new_newton_iterations=sum(h['iterations'] for h in heads),training_feature_rows=total,all_converged=True,validation_scoring_during_fit=False,new_hmm_fits=0,neural_forward_passes=0,neural_training_steps=0)
    print('All24scalar fits completed before new nextyear scoring.',flush=True)
if __name__=='__main__':main()

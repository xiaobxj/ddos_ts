from common20 import *

def main():
    check_frozen();path=OUT/'training_manifest.json';assert not path.exists(),'Preserve training';run=manifest('training');save(path,run)
    obs,price,returns=data();sources={s['job']:s for s in read(OUT/'source_heads.json')};heads=[];lookup={};traces=[];metrics=[];coefficients=[];files=[];total_rows=0
    for job in read(OUT/'jobs.json'):
        source=sources[job['source_job']];f=next(f for f in cfg()['folds'] if f['cutoff']==job['cutoff']);x0,idx=source_inputs(source,job['source_round'],'training')
        pos=recent_positions(obs,idx,f);x=x0[pos];rows=idx[pos];y=(returns[rows]>0).astype(float);theta,trace=fit_newton(x,y,cfg()['probe'])
        value,grad,hess=objective(theta,design(x),y,.01);oldvalue,_,_=objective(np.asarray(source['coefficients']),design(x),y,.01)
        nested_value=None
        if job['interaction']:
            parent=lookup[job['recent_parent_job']];pt=np.asarray(parent['coefficients']);nested=np.r_[pt[:-1],0.,pt[-1]]
            nested_value,_,_=objective(nested,design(x),y,.01);assert abs(nested_value-parent['objective'])<1e-12 and value<=nested_value+1e-12
        assert np.max(np.abs(grad))<=1e-9 and np.linalg.eigvalsh(hess).min()>0 and value<=oldvalue+1e-12
        path=CACHE/f"training_{job['job']}.npz";np.savez_compressed(path,standardized=x,row_index=rows,full_row_positions=pos,direction=y);files.append(path)
        h=dict(job,cache_file=str(path.relative_to(PROJECT)),cache_sha256=sha(path),coefficients=theta.tolist(),train_n=len(rows),full_train_n=len(idx),recent_start=f['recent_start'],
            training_frequency=float(y.mean()),iterations=trace[-1]['iteration'],objective=value,source_on_recent_objective=oldvalue,nested_recent_additive_objective=nested_value,
            gradient_inf=float(np.max(np.abs(grad))),hessian_min_eigenvalue=float(np.linalg.eigvalsh(hess).min()),l2_lambda=.01)
        heads.append(h);lookup[h['job']]=h;traces.extend(dict(job=h['job'],**r) for r in trace);total_rows+=len(rows)
        measured=previous.previous.base16.previous.probe_metrics(design(x)@theta,y);rate=y.mean();constant=previous.previous.base16.previous.probe_metrics(np.full(len(y),np.log(rate/(1-rate))),y)
        metrics.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],train_n=len(rows),objective=value,source_on_recent_objective=oldvalue,
            nested_recent_additive_objective=nested_value,**measured,constant_brier=constant['brier'],constant_log_loss=constant['log_loss']))
        source_theta=np.asarray(source['coefficients'])
        for coordinate,coefficient in enumerate(theta):
            block='intercept' if coordinate==len(theta)-1 else 'interaction' if job['interaction'] and coordinate==len(theta)-2 else 'additive'
            coefficients.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],coordinate=coordinate,block=block,
                coefficient=float(coefficient),full_window_coefficient=float(source_theta[coordinate]),coefficient_delta=float(coefficient-source_theta[coordinate])))
        save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False)
        print(f"Converged {len(heads)}/48: {h['job']}; n={len(rows)}; updates={h['iterations']}",flush=True)
    assert len(heads)==48 and total_rows==31608 and len(coefficients)==1428
    for name,records in [('training_metrics',metrics),('coefficients',coefficients)]:
        path=OUT/f'{name}.csv';pd.DataFrame(records).to_csv(path,index=False);files.append(path)
    files += [OUT/'heads.json',OUT/'solver_trace.csv']
    finish(run,files,new_primary_fits=48,new_newton_iterations=sum(h['iterations'] for h in heads),training_feature_rows=total_rows,all_converged=True,
        all_source_objective_checks=True,all24_nested_recent_interaction_checks=True,validation_scoring_during_fit=False,new_projection_fits=0,neural_forward_passes=0,neural_training_steps=0)
    print('All48recent-window classifiers complete before any new held-out scoring.',flush=True)

if __name__=='__main__':main()

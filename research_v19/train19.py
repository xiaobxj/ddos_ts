from common19 import *

def main():
    check_frozen();path=OUT/'training_manifest.json';assert not path.exists(),'Preserve training'
    run=manifest('training');save(path,run);start=time.time();obs,price,returns=data();sources={h['job']:h for h in read(OUT/'source_heads.json')}
    heads=[];traces=[];metrics=[];diagnostics=[];coefficients=[];files=[];total_rows=0
    for job in read(OUT/'jobs.json'):
        source=sources[job['source_job']];x,v,rows=inherited_inputs(source,'training');y=(returns[rows]>0).astype(float)
        t=fit_interaction(x,v,np.asarray(source['coefficients'])[:25]);assert float(t['training_orthogonality_inf'])<1e-8
        theta,trace=fit_newton(t['standardized'],y,cfg()['probe']);value,g,h=objective(theta,design(t['standardized']),y,.01)
        assert np.max(np.abs(g))<=1e-9 and value<=source['objective']+1e-12 and np.linalg.eigvalsh(h).min()>0
        path=CACHE/f"training_{job['job']}.npz";np.savez_compressed(path,**t,row_index=rows,direction=y);files.append(path)
        head=dict(job,cache_file=str(path.relative_to(PROJECT)),cache_sha256=sha(path),coefficients=theta.tolist(),train_n=len(rows),iterations=trace[-1]['iteration'],
            objective=value,parent_objective=source['objective'],gradient_inf=float(np.max(np.abs(g))),hessian_min_eigenvalue=float(np.linalg.eigvalsh(h).min()),
            l2_lambda=.01,interaction_coefficient=float(theta[-2]),raw_product_coefficient=float(theta[-2]/t['residual_sd']))
        heads.append(head);traces.extend(dict(job=job['job'],**r) for r in trace);total_rows+=len(rows)
        measured=previous.base16.previous.probe_metrics(design(t['standardized'])@theta,y)
        rate=float(y.mean());constant=previous.base16.previous.probe_metrics(np.full(len(y),np.log(rate/(1-rate))),y)
        metrics.append(dict(job=job['job'],method=job['method'],cutoff=job['cutoff'],seed=job['seed'],train_n=len(y),dimensions=job['dimensions'],
            objective=value,parent_objective=source['objective'],gradient_inf=head['gradient_inf'],**measured,constant_brier=constant['brier'],constant_log_loss=constant['log_loss']))
        a=design(x);projection_prediction=a@t['projection'];variance=float(t['product'].var())
        diagnostics.append(dict(job=job['job'],method=job['method'],cutoff=job['cutoff'],seed=job['seed'],train_n=len(y),
            base_design_rank=int(t['base_design_rank']),augmented_design_rank=int(t['augmented_design_rank']),projection_rank=int(t['projection_rank']),
            base_design_condition=float(t['singular_values'][0]/t['singular_values'][-1]),product_std=float(t['product'].std()),residual_mean=float(t['residual_mean']),
            residual_raw_sd=float(t['residual_raw_sd']),residual_sd=float(t['residual_sd']),sd_floored=bool(t['residual_raw_sd']<1e-6),
            product_linear_r2=float(1-np.square(t['product']-projection_prediction).mean()/variance) if variance else None,
            orthogonality_inf=float(t['training_orthogonality_inf']),interaction_coefficient=head['interaction_coefficient'],raw_product_coefficient=head['raw_product_coefficient']))
        names=[f'decoded_{j:02d}' for j in range(25)] if source['method']=='learned_market' else list(previous.base16.NAMES)
        names += [f'market_{previous.MARKET_NAMES[i]}' for i in source['market_indices']]+['residualized_volatility_x_parent_signal','intercept']
        assert len(names)==len(theta)
        for i,(name,coefficient) in enumerate(zip(names,theta)):
            block='intercept' if i==len(theta)-1 else 'interaction' if i==len(theta)-2 else 'additive'
            coefficients.append(dict(job=job['job'],method=job['method'],cutoff=job['cutoff'],seed=job['seed'],coordinate=i,name=name,block=block,coefficient=float(coefficient)))
        save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False)
        print(f"Converged {len(heads)}/24: {job['job']}; interaction={head['interaction_coefficient']:+.6f}; iterations={head['iterations']}",flush=True)
    assert len(heads)==24 and total_rows==37860 and len(coefficients)==726
    for name,table in [('training_metrics',metrics),('interaction_diagnostics',diagnostics),('coefficients',coefficients)]:
        path=OUT/f'{name}.csv';pd.DataFrame(table).to_csv(path,index=False);files.append(path)
    files += [OUT/'heads.json',OUT/'solver_trace.csv']
    finish(run,files,new_primary_fits=24,input_projections=24,new_newton_iterations=sum(h['iterations'] for h in heads),training_feature_rows=total_rows,
        all_converged=True,all_nested_training_objectives_pass=True,validation_scoring_during_fit=False,neural_forward_passes=0,neural_training_steps=0,elapsed_seconds=time.time()-start)
    print('All24classifier fits finished before new held-out scoring;24training-only interaction projections frozen.',flush=True)

if __name__=='__main__':main()

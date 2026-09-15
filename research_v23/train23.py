from common23 import *
import verify15 as independent

def main():
    check_frozen();path=OUT/'training_manifest.json';assert not path.exists();run=manifest('training');save(path,run)
    obs,price,returns=data();files=[];signals=[];daily=[];stability=[];runs=[]
    for f in cfg()['folds']:
        tr,te=indices(obs,f);a,b=daily_extent(price,f);state=features(price.iloc[:a+1]);sg=signal_rows(state,obs,tr,f['cutoff']);signals.append(sg)
        ds=state.iloc[120:].copy();ds.insert(0,'cutoff',f['cutoff']);daily.append(ds)
        for gate in GATES:
            d,rs=run_diagnostics(ds,gate);stability.append(dict(cutoff=f['cutoff'],split='training',gate=gate,**d));runs.extend(dict(cutoff=f['cutoff'],split='training',gate=gate,**r) for r in rs)
    for name,table in [('training_signal_states',pd.concat(signals,ignore_index=True)),('training_daily_states',pd.concat(daily,ignore_index=True)),('training_stability',pd.DataFrame(stability)),('training_state_runs',pd.DataFrame(runs))]:
        path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    sources={s['job']:s for s in read(OUT/'source_heads.json')};heads=[];traces=[];coefs=[];metrics=[];novelty=[];total=0;projections=0
    for job in read(OUT/'jobs.json'):
        u,tr=gate_inputs(job,'training');y=(returns[tr]>0).astype(float)
        if job['source_job']:
            source=sources[job['source_job']];x,rows=base_inputs(source,'training');np.testing.assert_array_equal(rows,tr);d=fit_interaction(x,u,source_ray(source));xx=d['standardized'];projections+=1
            pt=np.asarray(source['coefficients']);nested=np.r_[pt[:-1],0.,pt[-1]];parent_objective=source['objective'];v,_,_=objective(nested,design(xx),y,.01);assert abs(v-parent_objective)<1e-12
            np.testing.assert_allclose(design(xx)@nested,design(x)@pt,rtol=0,atol=1e-12);assert d['augmented_design_rank']==d['base_design_rank']+1 and d['residual_raw_sd']>1e-6 and d['training_orthogonality_inf']<1e-8
            gateproj=np.linalg.lstsq(design(x),u,rcond=1e-12)[0];d['gate_projection']=gateproj
            novelty.append(dict(job=job['job'],method=job['method'],cutoff=job['cutoff'],seed=job['seed'],gate=job['gate'],gate_linear_r2=float(1-np.square(u-design(x)@gateproj).mean()/u.var()),
                product_linear_r2=float(1-np.square(d['residual']).mean()/d['product'].var()),residual_raw_sd=float(d['residual_raw_sd']),orthogonality_inf=float(d['training_orthogonality_inf'])))
        else:
            mean=float(u.mean());sd=max(float(u.std()),1e-6);xx=((u-mean)/sd)[:,None];d=dict(standardized=xx,gate=u,gate_mean=np.asarray(mean),gate_sd=np.asarray(sd));parent_objective=None
        theta,trace=fit_newton(xx,y,cfg()['probe']);value,grad,hess=objective(theta,design(xx),y,.01);assert abs(grad).max()<=1e-9 and np.linalg.eigvalsh(hess).min()>0
        if parent_objective is not None:assert value<=parent_objective+1e-12
        path=CACHE/f"training_{job['job']}.npz";np.savez_compressed(path,**d,row_index=tr,direction=y);files.append(path)
        h=dict(job,**reference(path),coefficients=theta.tolist(),train_n=len(tr),iterations=trace[-1]['iteration'],objective=value,gradient_inf=float(abs(grad).max()),hessian_min_eigenvalue=float(np.linalg.eigvalsh(hess).min()),parent_objective=parent_objective,l2_lambda=.01)
        heads.append(h);total+=len(tr);traces.extend(dict(job=h['job'],**r) for r in trace);metrics.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],train_n=len(tr),objective=value,**independent.independent_metrics(design(xx)@theta,y)))
        for i,t in enumerate(theta):coefs.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],coordinate=i,coefficient=float(t),block='intercept' if i==len(theta)-1 else 'interaction' if h['source_job'] and i==len(theta)-2 else 'original_interaction' if h['source_job'] and i==len(theta)-3 else 'parent' if h['source_job'] else 'state'))
        save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False);print(f"Converged {len(heads)}/30: {h['job']}; updates={h['iterations']}",flush=True)
    assert len(heads)==30 and projections==24 and total==47325 and len(coefs)==762
    for name,rows in [('training_metrics',metrics),('coefficients',coefs),('gate_novelty',novelty)]:path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    files += [OUT/'heads.json',OUT/'solver_trace.csv'];finish(run,files,new_primary_fits=30,new_projection_fits=24,new_newton_iterations=sum(h['iterations'] for h in heads),training_feature_rows=total,all_converged=True,
        validation_scoring_during_fit=False,new_hmm_fits=0,neural_forward_passes=0,neural_training_steps=0)
    print('All30heads finished before heldout scoring.',flush=True)

if __name__=='__main__':main()

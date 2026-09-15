from common21 import *
import verify15 as independent

def main():
    check_frozen();path=OUT/'training_manifest.json';assert not path.exists();run=manifest('training');save(path,run)
    obs,price,returns=data();daily=daily_returns(price);contexts=[];htraces=[];files=[];summaries=[]
    for f in cfg()['folds']:
        a,b=daily_extent(price,f);model,trace=fit_hmm(daily[:a],cfg()['hmm']);tr,te=indices(obs,f)
        path=CACHE/f"hmm_{f['cutoff']}.npz";np.savez_compressed(path,**model,daily_return=daily[:a],anchor=np.arange(1,a+1));files.append(path)
        context=dict(cutoff=f['cutoff'],training_end_anchor=a,iterations=trace[-1]['iteration'],gradient_inf=trace[-1]['gradient_inf'],**reference(path));contexts.append(context)
        g=model['filtered'][obs.anchor.iloc[tr].to_numpy(int)-1,1];v=model['variances'];p=model['transition']
        summaries.append(dict(cutoff=f['cutoff'],daily_n=a,missing=int(np.isnan(daily[:a]).sum()),iterations=context['iterations'],gradient_inf=context['gradient_inf'],log_likelihood=model['log_likelihood'],
            low_daily_sd=float(np.sqrt(v[0])),high_daily_sd=float(np.sqrt(v[1])),variance_ratio=float(v[1]/v[0]),stay_low=float(p[0,0]),stay_high=float(p[1,1]),
            implied_low_duration=1/(1-p[0,0]),implied_high_duration=1/(1-p[1,1]),training_low_soft_occupancy=float(model['smoothed_training'][:,0].sum()),
            training_high_soft_occupancy=float(model['smoothed_training'][:,1].sum()),classifier_training_mean_high_probability=float(g.mean())))
        htraces.extend(dict(cutoff=f['cutoff'],**r) for r in trace);save(OUT/'hmm_contexts.json',contexts);pd.DataFrame(htraces).to_csv(OUT/'hmm_solver_trace.csv',index=False)
        print(f"HMM {len(contexts)}/6 converged; {f['cutoff']}; EM updates={context['iterations']}; sd={np.sqrt(v)}",flush=True)
    sources={s['job']:s for s in read(OUT/'source_heads.json')};heads=[];traces=[];coeff=[];metrics=[];projections=0;total=0
    for job in read(OUT/'jobs.json'):
        f=next(f for f in cfg()['folds'] if f['cutoff']==job['cutoff']);tr,te=indices(obs,f);c=next(c for c in contexts if c['cutoff']==job['cutoff']);u=training_gate(c,obs,tr);y=(returns[tr]>0).astype(float)
        if job['source_job']:
            source=sources[job['source_job']];x,rows=base_inputs(source,'training');np.testing.assert_array_equal(rows,tr)
            d=fit_interaction(x,u,np.asarray(source['coefficients'])[:25]);xx=d['standardized'];projections+=1
            parent=np.asarray(source['coefficients']);nested=np.r_[parent[:-1],0.,parent[-1]];parent_objective=source['objective']
            value,_,_=objective(nested,design(xx),y,.01);assert abs(value-parent_objective)<1e-12
            np.testing.assert_allclose(design(xx)@nested,design(x)@parent,rtol=0,atol=1e-12)
            assert d['augmented_design_rank']==d['base_design_rank']+1 and d['residual_raw_sd']>1e-6 and d['training_orthogonality_inf']<1e-8
        else:
            mean=float(u.mean());sd=max(float(u.std()),1e-6);xx=((u-mean)/sd)[:,None];d=dict(standardized=xx,gate=u,gate_mean=np.asarray(mean),gate_sd=np.asarray(sd));parent_objective=None
        theta,trace=fit_newton(xx,y,cfg()['probe']);value,gradient,hessian=objective(theta,design(xx),y,.01)
        assert abs(gradient).max()<=1e-9 and np.linalg.eigvalsh(hessian).min()>0
        if parent_objective is not None:assert value<=parent_objective+1e-12
        path=CACHE/f"training_{job['job']}.npz";np.savez_compressed(path,**d,row_index=tr,direction=y);files.append(path)
        h=dict(job,**reference(path),coefficients=theta.tolist(),train_n=len(tr),iterations=trace[-1]['iteration'],objective=value,gradient_inf=float(abs(gradient).max()),
            hessian_min_eigenvalue=float(np.linalg.eigvalsh(hessian).min()),parent_objective=parent_objective,l2_lambda=.01);heads.append(h);total+=len(tr)
        traces.extend(dict(job=h['job'],**r) for r in trace);metrics.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],train_n=len(tr),objective=value,**independent.independent_metrics(design(xx)@theta,y)))
        for i,t in enumerate(theta):coeff.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],coordinate=i,coefficient=float(t),block='intercept' if i==len(theta)-1 else 'interaction' if h['source_job'] and i==len(theta)-2 else 'additive' if h['source_job'] else 'state'))
        save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False)
        print(f"Classifier {len(heads)}/30 converged; {h['job']}; updates={h['iterations']}",flush=True)
    assert len(heads)==30 and total==47325 and len(coeff)==738 and projections==24
    for name,rows in [('hmm_fit_summary',summaries),('coefficients',coeff),('training_metrics',metrics)]:path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    files += [OUT/n for n in ['heads.json','solver_trace.csv','hmm_contexts.json','hmm_solver_trace.csv']]
    finish(run,files,new_hmm_fits=6,new_primary_fits=30,new_projection_fits=24,new_newton_iterations=sum(h['iterations'] for h in heads),new_em_iterations=sum(c['iterations'] for c in contexts),
        training_feature_rows=total,all_converged=True,validation_scoring_during_fit=False,heldout_filtering_during_fit=False,neural_forward_passes=0,neural_training_steps=0)
    print('All6HMM and30classifier fits finished before heldout filtering/scoring.',flush=True)

if __name__=='__main__':main()

"""Independent QR, convex-solver, forecast and statistics checks for round19."""
from common19 import *
import evaluate19 as evaluation
import verify17 as audit
import verify15 as independent
import scipy
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.special import expit


def main():
    start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists(),'Preserve verification'
    contract=read(OUT/'contract_verification.json');assert contract['protocol_sha256']==prep['protocol_sha256']
    for name,digest in contract['artifacts'].items():assert sha(ROOT/name)==digest,name
    last=contract['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        run=check_phase(phase);assert last<=run['started_utc']<run['finished_utc'];last=run['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert run[key]==prep[key]
    fitting=read(OUT/'training_manifest.json')
    assert fitting['all_converged'] and fitting['new_primary_fits']==24 and fitting['input_projections']==24
    assert not fitting['validation_scoring_during_fit'] and fitting['neural_forward_passes']==fitting['neural_training_steps']==0
    prior=read(V18/'results/verification.json');assert prior['status']=='PASS' and prior['neural_states_with_inherited_round17_replay']==18
    obs,price,returns=data();heads=read(OUT/'heads.json');sources={h['job']:h for h in read(OUT/'source_heads.json')}
    original={h['job']:h for h in read(V18/'results/heads.json')};assert len(sources)==len(heads)==24
    jobs={j['job']:j for j in read(OUT/'jobs.json')};contexts={c['cutoff']:c for c in read(V18/'results/market_contexts.json')}
    for key,source in sources.items():assert source==original[key]
    predictions=pd.read_csv(OUT/'model_predictions.csv',float_precision='round_trip')
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv',float_precision='round_trip')
    old_predictions=pd.read_csv(V18/'results/model_predictions.csv',float_precision='round_trip')
    traces=pd.read_csv(OUT/'solver_trace.csv');training_metrics=pd.read_csv(OUT/'training_metrics.csv').set_index('job')
    coefficients=pd.read_csv(OUT/'coefficients.csv');diagnostics=pd.read_csv(OUT/'interaction_diagnostics.csv').set_index('job')
    components=pd.read_csv(OUT/'interaction_components.csv',float_precision='round_trip');summaries=pd.read_csv(OUT/'component_summary.csv')
    settings=cfg()['independent_solver'];ps=cfg()['projection_verification'];solutions=[];projections=[];forecasts=[]
    total_train=0;total_test=0;maximum_error=0.;maximum_parent_error=0.
    for h in heads:
        job=jobs[h['job']]
        for key,value in job.items():assert h[key]==value
        source=sources[h['source_job']];assert CANDIDATES[h['method']]==source['method'] and h['base_dimensions']==source['dimensions']
        variant=next(v for v in cfg()['variants'] if v['method']==h['method']);assert h['dimensions']==variant['dimensions']==h['base_dimensions']+1
        fold=next(f for f in cfg()['folds'] if f['cutoff']==h['cutoff']);tr,te=indices(obs,fold);d=load_npz(h)
        x,v,rows=inherited_inputs(source,'training');old=load_npz(source);context=load_npz(contexts[h['cutoff']])
        np.testing.assert_array_equal(rows,tr);np.testing.assert_array_equal(d['row_index'],tr);np.testing.assert_array_equal(context['row_index'],tr)
        np.testing.assert_array_equal(x,old['standardized']);np.testing.assert_array_equal(v,context['standardized'][:,1])
        y=(returns[tr]>0).astype(float);np.testing.assert_array_equal(d['direction'],y);assert h['train_n']==len(tr)
        ray=np.asarray(source['coefficients'])[:25];signal=x[:,:25]@ray;product=v*signal;a=np.column_stack([x,np.ones(len(x))])
        for value,key in [(ray,'ray'),(signal,'signal'),(product,'product'),(v,'volatility')]:np.testing.assert_array_equal(value,d[key])
        replay_transform=fit_interaction(x,v,ray)
        for key,value in replay_transform.items():np.testing.assert_array_equal(value,d[key])
        # Pivoted QR supplies an independent algorithm for the SVD projection.
        qr,_,qr_rank,_=lstsq(a,product,cond=ps['rcond'],lapack_driver='gelsy')
        qr_fit=a@qr;qr_residual=product-qr_fit;qr_mean=float(qr_residual.mean());qr_sd=max(float(np.sqrt(np.square(qr_residual-qr_mean).mean())),1e-6)
        qr_h=(qr_residual-qr_mean)/qr_sd
        fitted_gap=float(np.max(np.abs(qr_fit-a@d['projection'])));train_h_gap=float(np.max(np.abs(qr_h-d['interaction'])))
        assert fitted_gap<=ps['training_fitted_value_absolute_tolerance'] and train_h_gap<=ps['training_standardized_residual_absolute_tolerance']
        assert qr_rank==int(d['projection_rank'])==int(d['base_design_rank'])
        assert int(d['augmented_design_rank'])==int(d['base_design_rank'])+1 and float(d['residual_raw_sd'])>1e-6
        orthogonality=float(np.max(np.abs(a.T@d['interaction']))/len(x));assert orthogonality<1e-8
        xx0,vv,testrows=inherited_inputs(source,'validation');np.testing.assert_array_equal(testrows,te)
        xx,extra=apply_interaction(xx0,vv,d);qr_q=vv*(xx0[:,:25]@ray)
        qr_future=(qr_q-np.column_stack([xx0,np.ones(len(xx0))])@qr-qr_mean)/qr_sd
        future_h_gap=float(np.max(np.abs(qr_future-xx[:,-1])));assert future_h_gap<=ps['validation_standardized_residual_absolute_tolerance']
        parent_theta=np.asarray(source['coefficients']);parent_z=xx0@parent_theta[:-1]+parent_theta[-1]
        old_g=old_predictions[old_predictions.method.eq(source['method'])&old_predictions.cutoff.eq(h['cutoff'])&old_predictions.seed.eq(h['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(old_g.row_index,te)
        parent_gap=max(float(np.max(np.abs(parent_z-old_g.logit.to_numpy()))),float(np.max(np.abs(expit(parent_z)-old_g.probability.to_numpy()))))
        assert parent_gap<1e-12;maximum_parent_error=max(maximum_parent_error,parent_gap)
        projections.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],qr_rank=int(qr_rank),svd_rank=int(d['projection_rank']),
            training_fitted_value_gap=fitted_gap,training_standardized_residual_gap=train_h_gap,validation_standardized_residual_gap=future_h_gap,training_orthogonality_inf=orthogonality))
        xnew=d['standardized'];assert xnew.shape==(len(tr),h['dimensions']);np.testing.assert_array_equal(xnew[:,:h['base_dimensions']],x)
        theta=np.asarray(h['coefficients']);assert len(theta)==h['dimensions']+1 and h['l2_lambda']==.01
        value,grad,hess=objective(theta,design(xnew),y,.01);assert np.max(np.abs(grad))<=1e-9 and np.linalg.eigvalsh(hess).min()>0
        audit.near(value,h['objective'],1e-14);audit.near(float(np.max(np.abs(grad))),h['gradient_inf'],1e-14)
        audit.near(float(np.linalg.eigvalsh(hess).min()),h['hessian_min_eigenvalue'])
        theta_replay,trace=fit_newton(xnew,y,cfg()['probe']);np.testing.assert_allclose(theta,theta_replay,rtol=0,atol=1e-12)
        assert trace[-1]['iteration']==h['iterations'];saved=traces[traces.job.eq(h['job'])].sort_values('iteration')
        pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
        nested=np.r_[parent_theta[:-1],0.,parent_theta[-1]];nested_value,_=independent.independent_objective(nested,xnew,y,.01)
        audit.near(nested_value,source['objective']);audit.near(h['parent_objective'],source['objective']);assert value<=nested_value+1e-12
        np.testing.assert_allclose(xnew@nested[:-1]+nested[-1],x@parent_theta[:-1]+parent_theta[-1],rtol=0,atol=1e-12)
        initial=np.zeros(len(theta));initial[-1]=np.log(y.mean()/(1-y.mean()))
        alt=minimize(independent.independent_objective,initial,args=(xnew,y,.01),method=settings['method'],jac=True,options=settings['options'])
        av,ag=independent.independent_objective(alt.x,xnew,y,.01);gap=abs(av-value)
        pgap=float(np.max(np.abs(expit(xnew@alt.x[:-1]+alt.x[-1])-probability(design(xnew)@theta))));gn=float(np.max(np.abs(ag)))
        assert gap<=settings['objective_absolute_tolerance'] and pgap<=settings['training_probability_absolute_tolerance'] and gn<=settings['gradient_infinity_tolerance']
        solutions.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],alternate_iterations=int(alt.nit),alternate_success=bool(alt.success),
            alternate_message=str(alt.message),objective_absolute_gap=gap,maximum_training_probability_gap=pgap,alternate_gradient_inf=gn))
        for key,val in independent.independent_metrics(xnew@theta[:-1]+theta[-1],y).items():audit.near(val,training_metrics.loc[h['job'],key])
        coef=coefficients[coefficients.job.eq(h['job'])].sort_values('coordinate');np.testing.assert_allclose(coef.coefficient,theta,rtol=0,atol=1e-15)
        assert coef.block.eq('additive').sum()==h['base_dimensions'] and coef.block.eq('interaction').sum()==coef.block.eq('intercept').sum()==1
        audit.near(theta[-2],h['interaction_coefficient']);audit.near(float(theta[-2]/d['residual_sd']),h['raw_product_coefficient'])
        diag=diagnostics.loc[h['job']];expected=dict(base_design_rank=int(d['base_design_rank']),augmented_design_rank=int(d['augmented_design_rank']),projection_rank=int(d['projection_rank']),
            base_design_condition=float(d['singular_values'][0]/d['singular_values'][-1]),product_std=float(product.std()),residual_mean=float(d['residual_mean']),
            residual_raw_sd=float(d['residual_raw_sd']),residual_sd=float(d['residual_sd']),sd_floored=bool(d['residual_raw_sd']<1e-6),
            product_linear_r2=float(1-np.square(product-a@d['projection']).mean()/product.var()),orthogonality_inf=orthogonality,
            interaction_coefficient=float(theta[-2]),raw_product_coefficient=float(theta[-2]/d['residual_sd']))
        for key,val in expected.items():
            if isinstance(val,bool):assert val==bool(diag[key])
            else:audit.near(val,diag[key])
        z=xx@theta[:-1]+theta[-1];p=expit(z)
        g=predictions[predictions.method.eq(h['method'])&predictions.cutoff.eq(h['cutoff'])&predictions.seed.eq(h['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index,te);np.testing.assert_array_equal(g.date,obs.date.iloc[te]);np.testing.assert_array_equal(g.actual_up,returns[te]>0)
        np.testing.assert_array_equal(g.actual_return,returns[te]);np.testing.assert_array_equal(g.direction_up,p>.5)
        err=max(float(np.max(np.abs(z-g.logit.to_numpy()))),float(np.max(np.abs(p-g.probability.to_numpy()))));assert err<1e-12;maximum_error=max(maximum_error,err)
        comp=components[components.job.eq(h['job'])].sort_values('row_index');np.testing.assert_array_equal(comp.row_index,te)
        for key,val in dict(standardized_volatility=vv,parent_representation_signal=extra['signal'],product=extra['product'],standardized_interaction=xx[:,-1],
            refitted_additive_logit=xx0@theta[:h['base_dimensions']]+theta[-1],interaction_logit=xx[:,-1]*theta[-2],logit=z).items():
            np.testing.assert_allclose(comp[key],val,rtol=0,atol=1e-12)
        np.testing.assert_allclose(comp.logit,comp.refitted_additive_logit+comp.interaction_logit,rtol=0,atol=1e-12)
        for split,values,vol,sig,prod in [('training',xnew,v,signal,product),('validation',xx,vv,extra['signal'],extra['product'])]:
            aa=values[:,:h['base_dimensions']]@theta[:h['base_dimensions']]+theta[-1];bb=values[:,-1]*theta[-2]
            saved=summaries[summaries.job.eq(h['job'])&summaries.split.eq(split)].iloc[0];assert saved.n==len(values)
            expected=dict(volatility_mean=float(vol.mean()),signal_mean=float(sig.mean()),signal_std=float(sig.std()),product_mean=float(prod.mean()),product_std=float(prod.std()),
                interaction_feature_mean=float(values[:,-1].mean()),interaction_feature_std=float(values[:,-1].std()),refitted_additive_mean=float(aa.mean()),refitted_additive_std=float(aa.std()),
                interaction_logit_mean=float(bb.mean()),interaction_logit_std=float(bb.std()),total_logit_mean=float((aa+bb).mean()),total_logit_std=float((aa+bb).std()))
            for key,val in expected.items():audit.near(val,saved[key])
        total_train+=len(tr);total_test+=len(te)
        forecasts.append(dict(job=h['job'],base_dimensions=h['base_dimensions'],dimensions=h['dimensions'],training_rows=len(tr),validation_rows=len(te),maximum_forecast_error=err,parent_forecast_error=parent_gap))
    assert total_train==37860 and total_test==1044 and len(coefficients)==726 and len(summaries)==48 and len(diagnostics)==24
    assert len(predictions)==5220 and len(ensemble)==3132 and len(components)==1044
    for filename,frame in [('model_predictions.csv',predictions),('ensemble_predictions.csv',ensemble)]:
        old=pd.read_csv(V18/'results'/filename,float_precision='round_trip');g=frame[frame.method.isin(old.method.unique())];keys=['method','date']+(['seed'] if 'seed' in old else [])
        pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),g[old.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-15)
    new=pd.read_csv(OUT/'new_model_predictions.csv',float_precision='round_trip');assert len(new)==1044
    saved_new=predictions[predictions.method.isin(CANDIDATES)]
    pd.testing.assert_frame_equal(new.reset_index(drop=True),saved_new[new.columns].reset_index(drop=True),check_dtype=False,rtol=0,atol=0)
    for (method,date),g in new.groupby(['method','date']):
        e=ensemble[ensemble.method.eq(method)&ensemble.date.eq(date)].iloc[0];assert len(g)==(3 if method=='learned_vol_interaction' else 1)
        audit.near(float(g.probability.mean()),e.probability,1e-15);assert bool(e.direction_up)==(e.probability>.5)
    tables,pairs,assessments=evaluation.compute()
    for name,table in tables.items():pd.testing.assert_frame_equal(table,pd.read_csv(OUT/f'{name}.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));independent.close_tree(assessments,read(OUT/'assessments.json'));metric_groups=0
    for r in tables['ensemble_metrics'].to_dict('records'):
        w=next(w for w in evaluation.windows() if w['name']==r['window']);g=evaluation.select(ensemble,w);audit.verify_metrics(g[g.method.eq(r['method'])],r);metric_groups+=1
    for r in tables['yearly_metrics'].to_dict('records'):audit.verify_metrics(ensemble[ensemble.method.eq(r['method'])&ensemble.year.eq(r['year'])],r);metric_groups+=1
    for r in tables['seed_metrics'].to_dict('records'):
        w=next(w for w in evaluation.windows() if w['name']==r['window']);g=evaluation.select(predictions,w);audit.verify_metrics(g[g.method.eq(r['method'])&g.seed.eq(r['seed'])],r);metric_groups+=1
    states=pd.read_csv(OUT/'validation_states.csv');enriched=ensemble.merge(states,on=['cutoff','row_index','date'],validate='many_to_one')
    for r in tables['state_metrics'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);g=evaluation.select(enriched,w);s=g[g[r['partition']].eq(r['state'])&g.method.eq(r['method'])]
        audit.verify_metrics(s,r);assert r['sparse']==(len(s)<20);audit.near(len(s)/w['n'],r['sample_share']);metric_groups+=1
        f=g[g[r['partition']].eq(r['state'])&g.method.eq('training_frequency')]
        expected=float(audit.independent_loss(s,'brier').mean()-audit.independent_loss(f,'brier').mean()) if len(s) and s.probability.notna().all() else None
        audit.near(expected,r['brier_difference_vs_frequency'])
    assert metric_groups==441 and len(tables['state_metrics'])==288 and len(tables['reliability_bins'])==110
    for r in tables['reliability_bins'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);g=evaluation.select(ensemble,w);g=g[g.method.eq(r['method'])]
        s=g[g.probability.ge(r['lower']) & (g.probability.lt(r['upper']) if r['bin_index']<4 else g.probability.le(1))]
        assert len(s)==r['n'];audit.near(float(s.probability.mean()) if len(s) else None,r['mean_probability']);audit.near(float(s.actual_up.mean()) if len(s) else None,r['observed_frequency'])
    for r in tables['direction_changes'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);e=evaluation.select(ensemble,w)
        a=e[e.method.eq(r['method'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');aa=a.direction_up.to_numpy()==a.actual_up.to_numpy();bb=b.direction_up.to_numpy()==b.actual_up.to_numpy()
        assert r['changed']==np.count_nonzero(aa!=bb) and r['correct_to_wrong']==np.count_nonzero(bb&~aa) and r['wrong_to_correct']==np.count_nonzero(~bb&aa)
    assert len(tables['direction_changes'])==16
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))));ids=np.empty((10000,n),int)
        for k in range(n):ids[:,k]=(starts[:,k//8]+k%8)%n
        e=evaluation.select(ensemble,w)
        for r in [p for p in pairs if p['window']==w['name']]:
            a=e[e.method.eq(r['candidate'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');v=audit.independent_loss(a,r['metric'])-audit.independent_loss(b,r['metric'])
            mean=float(v.mean());boot=v[ids].mean(axis=1);centered=(v-mean)[ids].mean(axis=1)
            for actual,key in [(mean,'difference'),(float(np.quantile(boot,.025)),'ci95_low'),(float(np.quantile(boot,.975)),'ci95_high'),(float((1+np.count_nonzero(np.abs(centered)>=abs(mean)))/10001),'p')]:audit.near(actual,r[key])
    assert len(pairs)==16;order=sorted(range(16),key=lambda i:pairs[i]['p']);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(16-rank)*pairs[i]['p']));audit.near(running,pairs[i]['holm_adjusted_p'])
    usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(usage)==len(states)==261 and usage.previously_evaluated.all() and set(usage.date)==set(states.date)
    assert old_evidence()==prep['old_evidence'];files=[]
    for name,rows in [('independent_solver_verification',solutions),('projection_verification',projections),('forecast_verification',forecasts)]:
        path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,previous_files_preserved=2882,new_primary_fits=24,independent_solutions=24,
        independent_qr_projections=24,neural_states_with_inherited_round17_replay=18,new_neural_forward_passes=0,neural_training_steps=0,training_feature_rows=37860,
        new_probability_forecasts=1044,reused_model_records=4176,model_records=5220,ensemble_records=3132,metric_groups=441,state_metric_rows=288,reliability_bins=110,
        primary_contrasts=16,independent_holdout_dates=0,maximum_new_forecast_error=maximum_error,maximum_parent_forecast_error=maximum_parent_error,
        maximum_qr_training_fitted_value_gap=max(r['training_fitted_value_gap'] for r in projections),maximum_qr_training_residual_gap=max(r['training_standardized_residual_gap'] for r in projections),
        maximum_qr_validation_residual_gap=max(r['validation_standardized_residual_gap'] for r in projections),
        maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),
        maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in solutions),scipy=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)


if __name__=='__main__':main()

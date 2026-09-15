"""Independent state likelihood/posterior, local stationarity, QR, convex and forecast audits."""
from common21 import *
import evaluate21 as evaluation
import verify17 as audit
import verify15 as independent
import audit_hmm21 as ha
from contract21 import synthetic
import scipy
from scipy.optimize import minimize
from scipy.linalg import lstsq
from scipy.special import expit

def main():
    start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();synthetic()
    last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert r[key]==prep[key]
    fit=read(OUT/'training_manifest.json');assert fit['all_converged'] and fit['new_hmm_fits']==6 and fit['new_primary_fits']==30 and fit['new_projection_fits']==24
    assert not fit['validation_scoring_during_fit'] and not fit['heldout_filtering_during_fit'] and fit['neural_forward_passes']==fit['neural_training_steps']==0
    obs,price,returns=data();daily=daily_returns(price);contexts={c['cutoff']:c for c in read(OUT/'hmm_contexts.json')};vcontexts={c['cutoff']:c for c in read(OUT/'hmm_validation_contexts.json')}
    sources={s['job']:s for s in read(OUT/'source_heads.json')};original={s['job']:s for s in read(V18/'results/heads.json')};assert all(s==original[k] for k,s in sources.items())
    hchecks=[];gates={};trgates={};hsettings=cfg()['independent_hmm'];htrace=pd.read_csv(OUT/'hmm_solver_trace.csv',float_precision='round_trip');hs=pd.read_csv(OUT/'hmm_fit_summary.csv',float_precision='round_trip').set_index('cutoff')
    daily_saved=pd.read_csv(OUT/'hmm_daily_states.csv',float_precision='round_trip');signal_saved=pd.read_csv(OUT/'hmm_signal_states.csv',float_precision='round_trip')
    # Reconstruct daily returns without calling the production return function.
    independent_daily=[]
    for i in range(1,len(price)):
        a=price.iloc[i-1];b=price.iloc[i];ok=bool(a.valid_ohlc) and bool(b.valid_ohlc) and np.isfinite(a.close) and np.isfinite(b.close) and a.close>0 and b.close>0
        independent_daily.append(np.log(float(b.close)/float(a.close)) if ok else np.nan)
    np.testing.assert_allclose(daily,independent_daily,rtol=0,atol=2e-15,equal_nan=True)
    for f in cfg()['folds']:
        cutoff=f['cutoff'];c=contexts[cutoff];d=load_npz(c);vd=load_npz(vcontexts[cutoff]);a,b=daily_extent(price,f);tr,te=indices(obs,f)
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(cutoff)))
        np.testing.assert_array_equal(te,np.flatnonzero(obs.date.gt(cutoff)&obs.date.le(f['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')))
        assert len(tr)==f['train_n'] and len(te)==f['test_n'] and c['training_end_anchor']==a
        np.testing.assert_array_equal(d['daily_return'],daily[:a]);np.testing.assert_array_equal(vd['daily_return'],daily[a:b]);np.testing.assert_array_equal(d['anchor'],np.arange(1,a+1));np.testing.assert_array_equal(vd['anchor'],np.arange(a+1,b+1))
        v=d['variances'];p=d['transition'];ref=ha.log_reference(daily[:a],v,p);llgap=abs(ref['log_likelihood']-float(d['log_likelihood']));assert llgap<hsettings['log_domain_likelihood_tolerance']
        for key in ['filtered','smoothed_training']:np.testing.assert_allclose(ref[key],d[key],rtol=0,atol=hsettings['log_domain_probability_tolerance'])
        np.testing.assert_allclose(ref['transition_counts'],d['transition_counts'],rtol=0,atol=hsettings['log_domain_transition_count_tolerance'])
        initial=d['filtered'][-1]@p;np.testing.assert_array_equal(initial,vd['initial']);future=ha.log_reference(daily[a:b],v,p,initial,False)['filtered']
        np.testing.assert_allclose(future,vd['filtered'],rtol=0,atol=1e-10)
        replay,trace=fit_hmm(daily[:a],cfg()['hmm'])
        for key,value in replay.items():np.testing.assert_array_equal(value,d[key])
        saved=htrace[htrace.cutoff.eq(cutoff)].sort_values('iteration');pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
        assert trace[-1]['iteration']==c['iterations'] and np.diff(saved.log_likelihood).min()>=-1e-8
        theta=ha.encode(v,p);obj,grad=ha.independent_objective(theta,daily[:a]);assert abs(grad).max()<=1e-7
        eps=cfg()['hmm']['transition_floor'];bounds=[tuple(np.log(cfg()['hmm']['variance_bounds']))]*2+[(np.log(eps/(1-eps)),np.log((1-eps)/eps))]*2
        alt=minimize(ha.independent_objective,theta,args=(daily[:a],),jac=True,method=hsettings['method'],bounds=bounds,options=hsettings['options']);av,ag=ha.independent_objective(alt.x,daily[:a]);v2,p2=ha.decode(alt.x)
        alt_all=ha.log_reference(daily[:b],v2,p2,posterior=False)['filtered'];all_original=forward(daily[:b],v,p)['filtered'];alt_gap=float(abs(alt_all-all_original).max())
        assert abs(av-obj)<=hsettings['objective_absolute_tolerance'] and abs(ag).max()<=hsettings['gradient_infinity_tolerance'] and alt_gap<=hsettings['filtered_probability_absolute_tolerance']
        np.testing.assert_array_equal(all_original[:a],d['filtered']);np.testing.assert_array_equal(all_original[a:],vd['filtered'])
        # Perturb each future suffix; delivered prefix and current posterior cannot change.
        for n in [1,len(future)//2,len(future)-1]:
            altered=daily[a:b].copy();altered[n:]=.123;changed=forward(altered,v,p,initial=initial)['filtered'];np.testing.assert_array_equal(changed[:n],vd['filtered'][:n])
        missing=np.flatnonzero(~np.isfinite(daily[a:b]))
        for j in missing:np.testing.assert_allclose(vd['filtered'][j],vd['predicted'][j],rtol=0,atol=1e-14)
        tg=2*d['filtered'][obs.anchor.iloc[tr].to_numpy(int)-1,1]-1;fg=2*future[obs.anchor.iloc[te].to_numpy(int)-a-1,1]-1;trgates[cutoff]=tg;gates[cutoff]=fg
        sg=signal_saved[signal_saved.cutoff.eq(cutoff)].sort_values('row_index');np.testing.assert_array_equal(sg.row_index,te);np.testing.assert_array_equal(sg.date,obs.date.iloc[te]);np.testing.assert_allclose(sg.gate,fg,rtol=0,atol=1e-10)
        np.testing.assert_array_equal(sg.hmm_bucket,buckets((fg+1)/2));dg=daily_saved[daily_saved.cutoff.eq(cutoff)]
        np.testing.assert_array_equal(dg.anchor,np.arange(a+1,b+1));np.testing.assert_array_equal(dg.date,price.date.iloc[a+1:b+1]);np.testing.assert_allclose(dg.high_probability,future[:,1],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(dg.daily_return,daily[a:b]);np.testing.assert_array_equal(dg.prior_high_probability,vd['predicted'][:,1])
        expected=dict(daily_n=a,missing=int(np.isnan(daily[:a]).sum()),iterations=c['iterations'],gradient_inf=c['gradient_inf'],log_likelihood=float(d['log_likelihood']),low_daily_sd=float(np.sqrt(v[0])),high_daily_sd=float(np.sqrt(v[1])),variance_ratio=float(v[1]/v[0]),stay_low=p[0,0],stay_high=p[1,1],implied_low_duration=1/(1-p[0,0]),implied_high_duration=1/(1-p[1,1]),training_low_soft_occupancy=float(d['smoothed_training'][:,0].sum()),training_high_soft_occupancy=float(d['smoothed_training'][:,1].sum()),classifier_training_mean_high_probability=float((tg.mean()+1)/2))
        for key,value in expected.items():audit.near(value,hs.loc[cutoff,key],1e-10)
        hchecks.append(dict(cutoff=cutoff,training_steps=a,heldout_steps=b-a,log_likelihood_gap=llgap,log_domain_train_probability_gap=float(abs(ref['filtered']-d['filtered']).max()),log_domain_future_probability_gap=float(abs(future-vd['filtered']).max()),alternate_objective_gap=abs(av-obj),alternate_gradient_inf=float(abs(ag).max()),alternate_probability_gap=alt_gap,alternate_success=bool(alt.success),alternate_iterations=int(alt.nit),alternate_message=str(alt.message),prefix_checks=3))
        print(f"Independent HMM audit {len(hchecks)}/6 PASS; {cutoff}",flush=True)
    heads=read(OUT/'heads.json');jobs={j['job']:j for j in read(OUT/'jobs.json')};assert len(heads)==len(jobs)==30
    preds=pd.read_csv(OUT/'model_predictions.csv',float_precision='round_trip');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv',float_precision='round_trip');parts=pd.read_csv(OUT/'interaction_components.csv',float_precision='round_trip');summaries=pd.read_csv(OUT/'component_summary.csv',float_precision='round_trip')
    traces=pd.read_csv(OUT/'solver_trace.csv',float_precision='round_trip');coefs=pd.read_csv(OUT/'coefficients.csv',float_precision='round_trip');training=pd.read_csv(OUT/'training_metrics.csv',float_precision='round_trip').set_index('job');oldpred=pd.read_csv(V18/'results/model_predictions.csv',float_precision='round_trip')
    ps=cfg()['projection_verification'];settings=cfg()['independent_solver'];solutions=[];projections=[];forecasts=[];total=0;testtotal=0;maxerror=0.;parenterror=0.
    for h in heads:
        for key,value in jobs[h['job']].items():assert h[key]==value
        f=next(f for f in cfg()['folds'] if f['cutoff']==h['cutoff']);tr,te=indices(obs,f);d=load_npz(h);u=trgates[h['cutoff']];uu=gates[h['cutoff']];y=(returns[tr]>0).astype(float)
        np.testing.assert_array_equal(d['row_index'],tr);np.testing.assert_array_equal(d['direction'],y);xx=d['standardized'];theta=np.asarray(h['coefficients'])
        assert h['train_n']==len(tr) and xx.shape==(len(tr),h['dimensions']) and len(theta)==h['dimensions']+1 and h['l2_lambda']==.01
        if h['source_job']:
            source=sources[h['source_job']];assert source['method']==CANDIDATES[h['method']];x,rows=base_inputs(source,'training');xv,testrows=base_inputs(source,'validation');np.testing.assert_array_equal(rows,tr);np.testing.assert_array_equal(testrows,te)
            ray=np.asarray(source['coefficients'])[:25];replay=fit_interaction(x,u,ray)
            for key,value in replay.items():np.testing.assert_array_equal(value,d[key])
            np.testing.assert_array_equal(xx[:,:-1],x);product=u*(x[:,:25]@ray);a=design(x)
            qr,_,rank,_=lstsq(a,product,cond=ps['rcond'],lapack_driver='gelsy');res=product-a@qr;mean=res.mean();sd=max(float(res.std()),1e-6);qh=(res-mean)/sd
            fitgap=float(abs(a@qr-a@d['projection']).max());traingap=float(abs(qh-xx[:,-1]).max());future_h=(uu*(xv[:,:25]@ray)-design(xv)@qr-mean)/sd
            val,extra=apply_interaction(xv,uu,d);futuregap=float(abs(future_h-val[:,-1]).max())
            assert fitgap<=ps['training_fitted_value_absolute_tolerance'] and traingap<=ps['training_standardized_residual_absolute_tolerance'] and futuregap<=ps['validation_standardized_residual_absolute_tolerance']
            assert rank==int(d['projection_rank'])==int(d['base_design_rank']) and d['augmented_design_rank']==rank+1 and d['residual_raw_sd']>1e-6 and d['training_orthogonality_inf']<1e-8
            pt=np.asarray(source['coefficients']);nested=np.r_[pt[:-1],0.,pt[-1]];nv,_=independent.independent_objective(nested,xx,y,.01);audit.near(nv,source['objective']);audit.near(nv,h['parent_objective']);assert h['objective']<=nv+1e-12
            np.testing.assert_allclose(design(xx)@nested,design(x)@pt,rtol=0,atol=1e-12)
            pg=oldpred[oldpred.method.eq(source['method'])&oldpred.cutoff.eq(h['cutoff'])&oldpred.seed.eq(h['seed'])].sort_values('row_index');np.testing.assert_array_equal(pg.row_index,te)
            pe=float(abs(expit(design(xv)@pt)-pg.probability.to_numpy()).max());assert pe<1e-12;parenterror=max(parenterror,pe)
            projections.append(dict(job=h['job'],training_fitted_value_gap=fitgap,training_residual_gap=traingap,validation_residual_gap=futuregap,rank=int(rank)))
            comp=parts[parts.job.eq(h['job'])].sort_values('row_index');np.testing.assert_array_equal(comp.row_index,te);np.testing.assert_array_equal(comp.date,obs.date.iloc[te])
            for key,value in dict(gate=uu,parent_representation_signal=extra['signal'],product=extra['product'],interaction_feature=val[:,-1],refitted_additive_logit=xv@theta[:-2]+theta[-1],interaction_logit=val[:,-1]*theta[-2],logit=design(val)@theta).items():np.testing.assert_allclose(comp[key],value,rtol=0,atol=1e-9)
            for split,values in [('training',xx),('validation',val)]:
                s=summaries[summaries.job.eq(h['job'])&summaries.split.eq(split)].iloc[0];aa=values[:,:-1]@theta[:-2]+theta[-1];bb=values[:,-1]*theta[-2];assert s.n==len(values)
                expected=dict(interaction_coefficient=theta[-2],raw_product_coefficient=float(theta[-2]/d['residual_sd']),interaction_feature_mean=float(values[:,-1].mean()),interaction_feature_std=float(values[:,-1].std()),refitted_additive_mean=float(aa.mean()),interaction_logit_mean=float(bb.mean()),interaction_logit_std=float(bb.std()),total_logit_mean=float((aa+bb).mean()))
                for key,value in expected.items():audit.near(value,s[key],1e-9)
        else:
            np.testing.assert_array_equal(d['gate'],u);audit.near(float(u.mean()),float(d['gate_mean']));audit.near(max(float(u.std()),1e-6),float(d['gate_sd']))
            np.testing.assert_array_equal(xx,((u-float(d['gate_mean']))/float(d['gate_sd']))[:,None]);val=((uu-float(d['gate_mean']))/float(d['gate_sd']))[:,None]
        value,grad,hess=objective(theta,design(xx),y,.01);audit.near(value,h['objective']);assert abs(grad).max()<=1e-9 and np.linalg.eigvalsh(hess).min()>0
        audit.near(float(abs(grad).max()),h['gradient_inf']);audit.near(float(np.linalg.eigvalsh(hess).min()),h['hessian_min_eigenvalue'])
        t,trace=fit_newton(xx,y,cfg()['probe']);np.testing.assert_array_equal(t,theta);saved=traces[traces.job.eq(h['job'])].sort_values('iteration');pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14);assert trace[-1]['iteration']==h['iterations']
        initial=np.zeros(len(theta));initial[-1]=np.log(y.mean()/(1-y.mean()));alt=minimize(independent.independent_objective,initial,args=(xx,y,.01),method=settings['method'],jac=True,options=settings['options']);av,ag=independent.independent_objective(alt.x,xx,y,.01)
        gap=abs(av-value);pgap=float(abs(expit(design(xx)@alt.x)-probability(design(xx)@theta)).max());gn=float(abs(ag).max())
        assert gap<=settings['objective_absolute_tolerance'] and pgap<=settings['training_probability_absolute_tolerance'] and gn<=settings['gradient_infinity_tolerance']
        solutions.append(dict(job=h['job'],objective_absolute_gap=gap,maximum_training_probability_gap=pgap,alternate_gradient_inf=gn,alternate_success=bool(alt.success),alternate_iterations=int(alt.nit),alternate_message=str(alt.message)))
        for key,value in independent.independent_metrics(design(xx)@theta,y).items():audit.near(value,training.loc[h['job'],key])
        cg=coefs[coefs.job.eq(h['job'])].sort_values('coordinate');np.testing.assert_array_equal(cg.coefficient,theta);assert cg.block.eq('intercept').sum()==1 and cg.block.eq('interaction').sum()==int(bool(h['source_job']))
        z=design(val)@theta;p=expit(z);g=preds[preds.method.eq(h['method'])&preds.cutoff.eq(h['cutoff'])&preds.seed.eq(h['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index,te);np.testing.assert_array_equal(g.date,obs.date.iloc[te]);np.testing.assert_array_equal(g.actual_up,returns[te]>0);np.testing.assert_array_equal(g.actual_return,returns[te]);np.testing.assert_array_equal(g.direction_up,p>.5)
        err=max(float(abs(g.probability.to_numpy()-p).max()),float(abs(g.logit.to_numpy()-z).max()));assert err<1e-9;maxerror=max(maxerror,err);total+=len(tr);testtotal+=len(te)
        forecasts.append(dict(job=h['job'],training_rows=len(tr),heldout_rows=len(te),maximum_forecast_error=err))
    assert total==47325 and testtotal==1305 and len(coefs)==738 and len(projections)==24 and len(parts)==1044 and len(summaries)==48
    assert len(preds)==8613 and len(ensemble)==5220
    for filename,frame in [('model_predictions.csv',preds),('ensemble_predictions.csv',ensemble)]:
        old=pd.read_csv(V20/'results'/filename,float_precision='round_trip');g=frame[frame.method.isin(old.method.unique())];keys=['method','date']+(['seed'] if 'seed' in old else [])
        pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),g[old.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-15)
    new=pd.read_csv(OUT/'new_model_predictions.csv',float_precision='round_trip');assert len(new)==1305
    pd.testing.assert_frame_equal(new.reset_index(drop=True),preds[preds.method.isin(list(CANDIDATES)+['hmm_only'])][new.columns].reset_index(drop=True),check_dtype=False,rtol=0,atol=0)
    for (method,date),g in new.groupby(['method','date']):
        e=ensemble[ensemble.method.eq(method)&ensemble.date.eq(date)].iloc[0];assert len(g)==(3 if method=='learned_hmm_interaction' else 1);audit.near(g.probability.mean(),e.probability,1e-15);assert bool(e.direction_up)==(e.probability>.5)
    tables,pairs,assessments=evaluation.compute()
    for name,table in tables.items():pd.testing.assert_frame_equal(table,pd.read_csv(OUT/f'{name}.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));independent.close_tree(assessments,read(OUT/'assessments.json'));groups=0
    for r in tables['ensemble_metrics'].to_dict('records'):
        w=next(w for w in evaluation.windows() if w['name']==r['window']);g=evaluation.select(ensemble,w);audit.verify_metrics(g[g.method.eq(r['method'])],r);groups+=1
    for r in tables['yearly_metrics'].to_dict('records'):audit.verify_metrics(ensemble[ensemble.method.eq(r['method'])&ensemble.year.eq(r['year'])],r);groups+=1
    for r in tables['seed_metrics'].to_dict('records'):
        w=next(w for w in evaluation.windows() if w['name']==r['window']);g=evaluation.select(preds,w);audit.verify_metrics(g[g.method.eq(r['method'])&g.seed.eq(r['seed'])],r);groups+=1
    states=pd.read_csv(OUT/'validation_states.csv');enriched=ensemble.merge(states,on=['cutoff','row_index','date'],validate='many_to_one').merge(signal_saved,on=['cutoff','row_index','date'],validate='many_to_one')
    for name in ['state_metrics','hmm_state_metrics']:
        for r in tables[name].to_dict('records'):
            w=next(w for w in cfg()['windows'] if w['name']==r['window']);g=evaluation.select(enriched,w);s=g[g[r['partition']].eq(r['state'])&g.method.eq(r['method'])];audit.verify_metrics(s,r);groups+=1
            assert r['sparse']==(len(s)<20);audit.near(len(s)/w['n'],r['sample_share']);f=g[g[r['partition']].eq(r['state'])&g.method.eq('training_frequency')]
            value=float(audit.independent_loss(s,'brier').mean()-audit.independent_loss(f,'brier').mean()) if len(s) and s.probability.notna().all() else None;audit.near(value,r['brier_difference_vs_frequency'])
    assert groups==852 and len(tables['reliability_bins'])==190
    for r in tables['reliability_bins'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);g=evaluation.select(ensemble,w);g=g[g.method.eq(r['method'])];s=g[g.probability.ge(r['lower'])&(g.probability.lt(r['upper']) if r['bin_index']<4 else g.probability.le(1))]
        assert len(s)==r['n'];audit.near(float(s.probability.mean()) if len(s) else None,r['mean_probability']);audit.near(float(s.actual_up.mean()) if len(s) else None,r['observed_frequency'])
    for r in tables['direction_changes'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);e=evaluation.select(ensemble,w);a=e[e.method.eq(r['method'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');aa=a.direction_up.to_numpy()==a.actual_up.to_numpy();bb=b.direction_up.to_numpy()==b.actual_up.to_numpy()
        assert r['changed']==np.count_nonzero(aa!=bb) and r['correct_to_wrong']==np.count_nonzero(bb&~aa) and r['wrong_to_correct']==np.count_nonzero(~bb&aa)
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))));ids=np.empty((10000,n),int)
        for k in range(n):ids[:,k]=(starts[:,k//8]+k%8)%n
        e=evaluation.select(ensemble,w)
        for r in [r for r in pairs if r['window']==w['name']]:
            a=e[e.method.eq(r['candidate'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');v=audit.independent_loss(a,r['metric'])-audit.independent_loss(b,r['metric']);mean=float(v.mean());boot=v[ids].mean(axis=1);centered=(v-mean)[ids].mean(axis=1)
            for value,key in [(mean,'difference'),(float(np.quantile(boot,.025)),'ci95_low'),(float(np.quantile(boot,.975)),'ci95_high'),(float((1+np.count_nonzero(abs(centered)>=abs(mean)))/10001),'p')]:audit.near(value,r[key])
    assert len(pairs)==26;order=sorted(range(26),key=lambda i:pairs[i]['p']);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(26-rank)*pairs[i]['p']));audit.near(running,pairs[i]['holm_adjusted_p'])
    usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(usage)==len(states)==len(signal_saved)==261 and usage.previously_evaluated.all() and set(usage.date)==set(states.date)
    assert old_evidence()==prep['old_evidence'];files=[]
    for name,values in [('hmm_independent_verification',hchecks),('independent_solver_verification',solutions),('projection_verification',projections),('forecast_verification',forecasts)]:path=OUT/f'{name}.csv';pd.DataFrame(values).to_csv(path,index=False);files.append(path)
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,previous_files_preserved=3061,new_hmm_fits=6,new_primary_fits=30,independent_hmm_checks=6,independent_solutions=30,independent_qr_projections=24,
        neural_states_with_inherited_round17_replay=18,new_neural_forward_passes=0,neural_training_steps=0,training_feature_rows=47325,new_probability_forecasts=1305,reused_model_records=7308,model_records=8613,ensemble_records=5220,
        metric_groups=852,state_metric_rows=480,hmm_state_metric_rows=120,reliability_bins=190,primary_contrasts=26,independent_holdout_dates=0,maximum_new_forecast_error=maxerror,maximum_parent_forecast_error=parenterror,
        maximum_hmm_likelihood_gap=max(r['log_likelihood_gap'] for r in hchecks),maximum_hmm_alternate_probability_gap=max(r['alternate_probability_gap'] for r in hchecks),maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),
        maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in solutions),scipy=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)

if __name__=='__main__':main()

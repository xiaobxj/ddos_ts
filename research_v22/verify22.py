from common22 import *
import evaluate22 as evaluation
import verify17 as audit
import verify15 as independent
from contract22 import synthetic
from scipy.optimize import minimize
from scipy.linalg import lstsq
from scipy.special import expit
import scipy

def verify_runs(frame,gate,summary,saved):
    col='relative_volatility_state' if gate=='relative_volatility' else 'persistence_state';high='elevated' if gate=='relative_volatility' else 'persistent'
    valid=frame[frame[gate].notna()].reset_index(drop=True);xs=valid[gate].to_numpy();anchors=valid.anchor.to_numpy();states=valid[col].to_numpy();groups=[]
    for row in valid.to_dict('records'):
        if not groups or row['anchor']!=groups[-1][-1]['anchor']+1 or row[col]!=groups[-1][-1][col]:groups.append([])
        groups[-1].append(row)
    expected=[]
    for i,g in enumerate(groups):
        expected.append(dict(state=g[0][col],first_anchor=g[0]['anchor'],last_anchor=g[-1]['anchor'],first_date=g[0]['date'],last_date=g[-1]['date'],length=len(g),
            left_censored=i==0 or g[0]['anchor']!=groups[i-1][-1]['anchor']+1,right_censored=i==len(groups)-1 or groups[i+1][0]['anchor']!=g[-1]['anchor']+1))
    pd.testing.assert_frame_equal(pd.DataFrame(expected),saved[list(expected[0])].reset_index(drop=True),check_dtype=False)
    pairs=[i for i in range(1,len(xs)) if anchors[i]==anchors[i-1]+1];a=np.array([xs[i-1] for i in pairs]);b=np.array([xs[i] for i in pairs]);nchanges=sum(states[i]!=states[i-1] for i in pairs)
    corr=float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>0 and b.std()>0 else None
    values=dict(n=len(valid),missing=len(frame)-len(valid),mean=float(xs.mean()),std=float(xs.std()),p10=float(np.quantile(xs,.1)),p90=float(np.quantile(xs,.9)),high_state_share=float(np.mean(states==high)),adjacent_pairs=len(pairs),switches=nchanges,
        switch_rate=nchanges/len(pairs) if pairs else None,lag1_correlation=corr,runs=len(groups),censored_runs=sum(r['left_censored'] or r['right_censored'] for r in expected),median_observed_run=float(np.median([len(g) for g in groups])),max_observed_run=max(map(len,groups)))
    for k,value in values.items():audit.near(value,summary[k],1e-12)

def main():
    start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();synthetic();last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['source_sha256','protocol_sha256','input_sha256']:assert r[key]==prep[key]
    fit=read(OUT/'training_manifest.json');assert fit['all_converged'] and fit['new_primary_fits']==60 and fit['new_projection_fits']==48 and not fit['validation_scoring_during_fit']
    assert fit['neural_forward_passes']==fit['neural_training_steps']==fit['new_hmm_fits']==0
    obs,price,returns=data();states={};ind_states={};statechecks=[];maxstategap=0.;totalprefixchecks=0
    for split in ['training','validation']:
        signals=pd.read_csv(OUT/f'{split}_signal_states.csv',float_precision='round_trip');daily=pd.read_csv(OUT/f'{split}_daily_states.csv',float_precision='round_trip');stability=pd.read_csv(OUT/f'{split}_stability.csv',float_precision='round_trip');runs=pd.read_csv(OUT/f'{split}_state_runs.csv')
        assert len(signals)==(9465 if split=='training' else 261)
        for f in cfg()['folds']:
            tr,te=indices(obs,f);a,b=daily_extent(price,f);rows=tr if split=='training' else te;end=a if split=='training' else b
            np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(f['cutoff'])));np.testing.assert_array_equal(te,np.flatnonzero(obs.date.gt(f['cutoff'])&obs.date.le(f['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')))
            assert len(tr)==f['train_n'] and len(te)==f['test_n'];anchors=np.arange(120,end+1) if split=='training' else np.arange(a+1,b+1)
            d=daily[daily.cutoff.eq(f['cutoff'])].reset_index(drop=True);sg=signals[signals.cutoff.eq(f['cutoff'])].sort_values('row_index');np.testing.assert_array_equal(d.anchor,anchors);np.testing.assert_array_equal(d.date,price.date.iloc[anchors])
            scalar=scalar_features(price.iloc[:end+1],anchors);gap=float(np.nanmax(abs(scalar-d[NUMERIC].to_numpy())));assert gap<2e-12;maxstategap=max(maxstategap,gap)
            valid=np.isfinite(scalar).all(axis=1);rv=np.where(valid,np.where(scalar[:,2]>0,'elevated','subdued'),'missing');ps=np.where(valid,np.where(scalar[:,3]>=.2,'persistent','choppy'),'missing')
            np.testing.assert_array_equal(d.relative_volatility_state,rv);np.testing.assert_array_equal(d.persistence_state,ps);np.testing.assert_array_equal(d.relative_joint,np.where(valid,np.char.add(np.char.add(ps,'__'),rv),'missing'))
            replay=features(price.iloc[:end+1]);np.testing.assert_array_equal(d[NUMERIC],replay.iloc[anchors][NUMERIC]);np.testing.assert_array_equal(sg.row_index,rows);np.testing.assert_array_equal(sg.date,obs.date.iloc[rows]);np.testing.assert_array_equal(sg.anchor,obs.anchor.iloc[rows])
            np.testing.assert_array_equal(sg[NUMERIC],replay.iloc[obs.anchor.iloc[rows].to_numpy(int)][NUMERIC]);assert np.isfinite(sg[NUMERIC]).all().all()
            independent_signal=scalar_features(price.iloc[:end+1],obs.anchor.iloc[rows].to_numpy(int));states[(split,f['cutoff'])]=sg;ind_states[(split,f['cutoff'])]=dict(zip(NUMERIC,independent_signal.T))
            if split=='validation':
                for cut in [a,(a+b)//2,b-1]:
                    prefix=features(price.iloc[:cut+1]);pd.testing.assert_frame_equal(prefix,replay.iloc[:cut+1]);changed=price.iloc[:end+1].copy();changed.loc[cut+1:,'close']*=1.7;pd.testing.assert_frame_equal(features(changed).iloc[:cut+1],prefix);totalprefixchecks+=1
            for gate in GATES:
                summary=stability[stability.cutoff.eq(f['cutoff'])&stability.gate.eq(gate)].iloc[0];rr=runs[runs.cutoff.eq(f['cutoff'])&runs.gate.eq(gate)];verify_runs(d,gate,summary,rr)
            statechecks.append(dict(split=split,cutoff=f['cutoff'],daily_rows=len(d),valid_daily_rows=int(valid.sum()),signal_rows=len(sg),maximum_scalar_gap=gap))
    assert totalprefixchecks==18
    heads=read(OUT/'heads.json');jobs={j['job']:j for j in read(OUT/'jobs.json')};sources={s['job']:s for s in read(OUT/'source_heads.json')};original={s['job']:s for s in read(V18/'results/heads.json')};assert len(heads)==len(jobs)==60 and all(s==original[k] for k,s in sources.items())
    preds=pd.read_csv(OUT/'model_predictions.csv',float_precision='round_trip');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv',float_precision='round_trip');parts=pd.read_csv(OUT/'interaction_components.csv',float_precision='round_trip');summary=pd.read_csv(OUT/'component_summary.csv',float_precision='round_trip')
    traces=pd.read_csv(OUT/'solver_trace.csv',float_precision='round_trip');coefs=pd.read_csv(OUT/'coefficients.csv',float_precision='round_trip');tm=pd.read_csv(OUT/'training_metrics.csv',float_precision='round_trip').set_index('job');novelty=pd.read_csv(OUT/'gate_novelty.csv',float_precision='round_trip').set_index('job')
    oldpred=pd.read_csv(V18/'results/model_predictions.csv',float_precision='round_trip');settings=cfg()['independent_solver'];ps=cfg()['projection_verification'];solutions=[];projections=[];forecasts=[];total=0;testtotal=0;maxerror=0.;maxparent=0.
    for h in heads:
        for key,value in jobs[h['job']].items():assert h[key]==value
        f=next(f for f in cfg()['folds'] if f['cutoff']==h['cutoff']);tr,te=indices(obs,f);u=states[('training',h['cutoff'])][h['gate']].to_numpy();uu=ind_states[('validation',h['cutoff'])][h['gate']];d=load_npz(h);xx=d['standardized'];theta=np.asarray(h['coefficients']);y=(returns[tr]>0).astype(float)
        np.testing.assert_array_equal(d['row_index'],tr);np.testing.assert_array_equal(d['direction'],y);assert xx.shape==(len(tr),h['dimensions']) and len(theta)==h['dimensions']+1 and h['l2_lambda']==.01
        if h['source_job']:
            source=sources[h['source_job']];assert source['method']==CANDIDATES[h['method']] and h['gate']==GATE_MAP[h['method']];x,rows=base_inputs(source,'training');xv,testrows=base_inputs(source,'validation');np.testing.assert_array_equal(rows,tr);np.testing.assert_array_equal(testrows,te)
            ray=np.asarray(source['coefficients'])[:25];replay=fit_interaction(x,u,ray)
            for key,value in replay.items():np.testing.assert_array_equal(value,d[key])
            np.testing.assert_array_equal(xx[:,:-1],x);product=u*(x[:,:25]@ray);a=design(x);qr,_,rank,_=lstsq(a,product,cond=ps['rcond'],lapack_driver='gelsy');res=product-a@qr;mean=res.mean();sd=max(float(res.std()),1e-6);qh=(res-mean)/sd
            fitgap=float(abs(a@qr-a@d['projection']).max());traingap=float(abs(qh-xx[:,-1]).max());val,extra=apply_interaction(xv,uu,d);future=(uu*(xv[:,:25]@ray)-design(xv)@qr-mean)/sd;futuregap=float(abs(future-val[:,-1]).max())
            assert fitgap<=ps['training_fitted_value_absolute_tolerance'] and traingap<=ps['training_standardized_residual_absolute_tolerance'] and futuregap<=ps['validation_standardized_residual_absolute_tolerance']
            assert rank==int(d['projection_rank'])==int(d['base_design_rank']) and d['augmented_design_rank']==rank+1 and d['residual_raw_sd']>1e-6 and d['training_orthogonality_inf']<1e-8
            pt=np.asarray(source['coefficients']);nested=np.r_[pt[:-1],0.,pt[-1]];nv,_=independent.independent_objective(nested,xx,y,.01);audit.near(nv,source['objective']);audit.near(nv,h['parent_objective']);assert h['objective']<=nv+1e-12;np.testing.assert_allclose(design(xx)@nested,design(x)@pt,rtol=0,atol=1e-12)
            pg=oldpred[oldpred.method.eq(source['method'])&oldpred.cutoff.eq(h['cutoff'])&oldpred.seed.eq(h['seed'])].sort_values('row_index');np.testing.assert_array_equal(pg.row_index,te);pe=float(abs(expit(design(xv)@pt)-pg.probability.to_numpy()).max());assert pe<1e-12;maxparent=max(maxparent,pe)
            gqr,_,_,_=lstsq(a,u,cond=ps['rcond'],lapack_driver='gelsy');assert abs(a@gqr-a@d['gate_projection']).max()<1e-9;nov=novelty.loc[h['job']]
            for key,value in dict(gate_linear_r2=float(1-np.square(u-a@gqr).mean()/u.var()),product_linear_r2=float(1-np.square(res).mean()/product.var()),residual_raw_sd=float(res.std()),orthogonality_inf=float(d['training_orthogonality_inf'])).items():audit.near(value,nov[key],1e-10)
            projections.append(dict(job=h['job'],training_fitted_value_gap=fitgap,training_residual_gap=traingap,validation_residual_gap=futuregap,rank=int(rank)))
            comp=parts[parts.job.eq(h['job'])].sort_values('row_index');np.testing.assert_array_equal(comp.row_index,te);np.testing.assert_array_equal(comp.date,obs.date.iloc[te])
            for key,value in dict(gate=uu,parent_representation_signal=extra['signal'],product=extra['product'],interaction_feature=val[:,-1],refitted_additive_logit=xv@theta[:-2]+theta[-1],interaction_logit=val[:,-1]*theta[-2],logit=design(val)@theta).items():np.testing.assert_allclose(comp[key],value,rtol=0,atol=1e-9)
            for split,values in [('training',xx),('validation',val)]:
                s=summary[summary.job.eq(h['job'])&summary.split.eq(split)].iloc[0];aa=values[:,:-1]@theta[:-2]+theta[-1];bb=values[:,-1]*theta[-2];assert s.n==len(values)
                for key,value in dict(interaction_coefficient=theta[-2],raw_product_coefficient=float(theta[-2]/d['residual_sd']),interaction_feature_mean=float(values[:,-1].mean()),interaction_feature_std=float(values[:,-1].std()),refitted_additive_mean=float(aa.mean()),interaction_logit_mean=float(bb.mean()),interaction_logit_std=float(bb.std()),total_logit_mean=float((aa+bb).mean())).items():audit.near(value,s[key],1e-9)
        else:
            assert h['method']==CONTROLS[h['gate']];np.testing.assert_array_equal(d['gate'],u);audit.near(float(u.mean()),float(d['gate_mean']));audit.near(max(float(u.std()),1e-6),float(d['gate_sd']));np.testing.assert_array_equal(xx,((u-float(d['gate_mean']))/float(d['gate_sd']))[:,None]);val=((uu-float(d['gate_mean']))/float(d['gate_sd']))[:,None]
        value,grad,hess=objective(theta,design(xx),y,.01);audit.near(value,h['objective']);assert abs(grad).max()<=1e-9 and np.linalg.eigvalsh(hess).min()>0;audit.near(float(abs(grad).max()),h['gradient_inf']);audit.near(float(np.linalg.eigvalsh(hess).min()),h['hessian_min_eigenvalue'])
        t,trace=fit_newton(xx,y,cfg()['probe']);np.testing.assert_array_equal(t,theta);saved=traces[traces.job.eq(h['job'])].sort_values('iteration');pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14);assert trace[-1]['iteration']==h['iterations']
        initial=np.zeros(len(theta));initial[-1]=np.log(y.mean()/(1-y.mean()));alt=minimize(independent.independent_objective,initial,args=(xx,y,.01),method=settings['method'],jac=True,options=settings['options']);av,ag=independent.independent_objective(alt.x,xx,y,.01);gap=abs(av-value);pgap=float(abs(expit(design(xx)@alt.x)-probability(design(xx)@theta)).max());gn=float(abs(ag).max())
        assert gap<=settings['objective_absolute_tolerance'] and pgap<=settings['training_probability_absolute_tolerance'] and gn<=settings['gradient_infinity_tolerance'];solutions.append(dict(job=h['job'],objective_absolute_gap=gap,maximum_training_probability_gap=pgap,alternate_gradient_inf=gn,alternate_success=bool(alt.success),alternate_iterations=int(alt.nit),alternate_message=str(alt.message)))
        for key,value in independent.independent_metrics(design(xx)@theta,y).items():audit.near(value,tm.loc[h['job'],key])
        cg=coefs[coefs.job.eq(h['job'])].sort_values('coordinate');np.testing.assert_array_equal(cg.coefficient,theta);assert cg.block.eq('intercept').sum()==1 and cg.block.eq('interaction').sum()==int(bool(h['source_job']))
        z=design(val)@theta;p=expit(z);g=preds[preds.method.eq(h['method'])&preds.cutoff.eq(h['cutoff'])&preds.seed.eq(h['seed'])].sort_values('row_index');np.testing.assert_array_equal(g.row_index,te);np.testing.assert_array_equal(g.date,obs.date.iloc[te]);np.testing.assert_array_equal(g.actual_up,returns[te]>0);np.testing.assert_array_equal(g.actual_return,returns[te]);np.testing.assert_array_equal(g.direction_up,p>.5)
        err=max(float(abs(g.probability.to_numpy()-p).max()),float(abs(g.logit.to_numpy()-z).max()));assert err<1e-9;maxerror=max(maxerror,err);total+=len(tr);testtotal+=len(te);forecasts.append(dict(job=h['job'],training_rows=len(tr),heldout_rows=len(te),maximum_forecast_error=err))
    assert total==94650 and testtotal==2610 and len(coefs)==1476 and len(projections)==48 and len(parts)==2088 and len(summary)==96 and len(preds)==11223 and len(ensemble)==6786
    for filename,frame in [('model_predictions.csv',preds),('ensemble_predictions.csv',ensemble)]:
        old=pd.read_csv(V21/'results'/filename,float_precision='round_trip');g=frame[frame.method.isin(old.method.unique())];keys=['method','date']+(['seed'] if 'seed' in old else []);pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),g[old.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-15)
    new=pd.read_csv(OUT/'new_model_predictions.csv',float_precision='round_trip');assert len(new)==2610;pd.testing.assert_frame_equal(new.reset_index(drop=True),preds[preds.method.isin(list(CANDIDATES)+list(CONTROLS.values()))][new.columns].reset_index(drop=True),check_dtype=False,rtol=0,atol=0)
    for (method,date),g in new.groupby(['method','date']):
        e=ensemble[ensemble.method.eq(method)&ensemble.date.eq(date)].iloc[0];assert len(g)==(3 if method.startswith('learned') else 1);audit.near(g.probability.mean(),e.probability,1e-15);assert bool(e.direction_up)==(e.probability>.5)
    tables,pairs,assess=evaluation.compute()
    for name,table in tables.items():pd.testing.assert_frame_equal(table,pd.read_csv(OUT/f'{name}.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));independent.close_tree(assess,read(OUT/'assessments.json'));groups=0
    for r in tables['ensemble_metrics'].to_dict('records'):
        w=next(w for w in evaluation.windows() if w['name']==r['window']);g=evaluation.select(ensemble,w);audit.verify_metrics(g[g.method.eq(r['method'])],r);groups+=1
    for r in tables['yearly_metrics'].to_dict('records'):audit.verify_metrics(ensemble[ensemble.method.eq(r['method'])&ensemble.year.eq(r['year'])],r);groups+=1
    for r in tables['seed_metrics'].to_dict('records'):
        w=next(w for w in evaluation.windows() if w['name']==r['window']);g=evaluation.select(preds,w);audit.verify_metrics(g[g.method.eq(r['method'])&g.seed.eq(r['seed'])],r);groups+=1
    enriched=evaluation.enriched(ensemble)
    for r in tables['state_metrics'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);g=evaluation.select(enriched,w);s=g[g[r['partition']].eq(r['state'])&g.method.eq(r['method'])];audit.verify_metrics(s,r);groups+=1;assert r['sparse']==(len(s)<20);audit.near(len(s)/w['n'],r['sample_share'])
        f=g[g[r['partition']].eq(r['state'])&g.method.eq('training_frequency')];value=float(audit.independent_loss(s,'brier').mean()-audit.independent_loss(f,'brier').mean()) if len(s) and s.probability.notna().all() else None;audit.near(value,r['brier_difference_vs_frequency'])
    assert groups==1520 and len(tables['reliability_bins'])==250
    for r in tables['reliability_bins'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);g=evaluation.select(ensemble,w);g=g[g.method.eq(r['method'])];s=g[g.probability.ge(r['lower'])&(g.probability.lt(r['upper']) if r['bin_index']<4 else g.probability.le(1))];assert len(s)==r['n'];audit.near(float(s.probability.mean()) if len(s) else None,r['mean_probability']);audit.near(float(s.actual_up.mean()) if len(s) else None,r['observed_frequency'])
    for r in tables['direction_changes'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);e=evaluation.select(ensemble,w);a=e[e.method.eq(r['method'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');aa=a.direction_up.to_numpy()==a.actual_up.to_numpy();bb=b.direction_up.to_numpy()==b.actual_up.to_numpy();assert r['changed']==np.count_nonzero(aa!=bb) and r['correct_to_wrong']==np.count_nonzero(bb&~aa) and r['wrong_to_correct']==np.count_nonzero(~bb&aa)
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))));ids=np.empty((10000,n),int)
        for k in range(n):ids[:,k]=(starts[:,k//8]+k%8)%n
        e=evaluation.select(ensemble,w)
        for r in [r for r in pairs if r['window']==w['name']]:
            a=e[e.method.eq(r['candidate'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');v=audit.independent_loss(a,r['metric'])-audit.independent_loss(b,r['metric']);mean=float(v.mean());boot=v[ids].mean(axis=1);centered=(v-mean)[ids].mean(axis=1)
            for value,key in [(mean,'difference'),(float(np.quantile(boot,.025)),'ci95_low'),(float(np.quantile(boot,.975)),'ci95_high'),(float((1+np.count_nonzero(abs(centered)>=abs(mean)))/10001),'p')]:audit.near(value,r[key])
    assert len(pairs)==52;order=sorted(range(52),key=lambda i:pairs[i]['p']);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(52-rank)*pairs[i]['p']));audit.near(running,pairs[i]['holm_adjusted_p'])
    for r in pd.read_csv(OUT/'state_correlations.csv',float_precision='round_trip').itertuples():
        old=pd.read_csv(V18/'results'/f'{r.split}_states.csv',float_precision='round_trip');old=old[old.cutoff.eq(r.cutoff)].sort_values('row_index');u=states[(r.split,r.cutoff)][r.gate].to_numpy();assert r.n==len(u);audit.near(float(np.corrcoef(u,old.volatility20)[0,1]),r.correlation_volatility20);audit.near(float(np.corrcoef(u,abs(old.trend60))[0,1]),r.correlation_abs_trend60)
    usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(usage)==261 and usage.previously_evaluated.all() and set(usage.date)==set(ensemble.date);assert old_evidence()==prep['old_evidence'];files=[]
    for name,value in [('state_verification',statechecks),('independent_solver_verification',solutions),('projection_verification',projections),('forecast_verification',forecasts)]:path=OUT/f'{name}.csv';pd.DataFrame(value).to_csv(path,index=False);files.append(path)
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,previous_files_preserved=3166,new_primary_fits=60,independent_solutions=60,independent_qr_projections=48,state_split_checks=12,future_prefix_checks=18,
        new_hmm_fits=0,new_neural_forward_passes=0,neural_training_steps=0,inherited_neural_replay_states=18,training_feature_rows=94650,new_probability_forecasts=2610,reused_model_records=8613,model_records=11223,ensemble_records=6786,
        metric_groups=1520,state_metric_rows=1196,reliability_bins=250,primary_contrasts=52,independent_holdout_dates=0,maximum_new_forecast_error=maxerror,maximum_parent_forecast_error=maxparent,maximum_state_formula_gap=maxstategap,
        maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in solutions),
        scipy=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)

if __name__=='__main__':main()

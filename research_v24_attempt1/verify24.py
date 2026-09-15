"""Independent diagonal-penalty solutions, cache provenance, forecasts and metrics."""
from common24 import *
import evaluate24 as evaluation
import verify17 as audit
import verify15 as independent
from contract24 import synthetic,provenance
from scipy.optimize import minimize
from scipy.special import expit
import scipy

def main():
    start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();synthetic()
    pd.testing.assert_frame_equal(pd.DataFrame(provenance()),csv('temporal_provenance'),check_dtype=False)
    last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        run=check_phase(phase);assert last<=run['started_utc']<run['finished_utc'];last=run['finished_utc']
        for key in ['source_sha256','protocol_sha256','input_sha256']:assert run[key]==prep[key]
    fit=read(OUT/'training_manifest.json');assert fit['all_converged'] and fit['new_primary_fits']==48 and fit['new_projection_fits']==0 and fit['reused_projection_fits']==24 and not fit['validation_scoring_during_fit']
    assert fit['neural_forward_passes']==fit['neural_training_steps']==fit['new_hmm_fits']==0
    obs,price,returns=data();sources={h['job']:h for h in read(OUT/'source_heads.json')};parents={h['job']:h for h in read(V19/'results/heads.json')};heads=read(OUT/'heads.json');jobs={h['job']:h for h in read(OUT/'jobs.json')}
    predictions=csv('model_predictions');ensemble=csv('ensemble_predictions');parts=csv('interaction_components');summary=csv('component_summary');traces=csv('solver_trace');coefficients=csv('coefficients');tm=csv('training_metrics').set_index('job')
    settings=cfg()['independent_solver'];solutions=[];forecasts=[];replays=[];reconstructed={};maxstategap=0.
    # Reconstruct unchanged order coordinate directly in the original coordinate system.
    for source in sources.values():
        parent=parents[source['source_job']];d=load_npz(source);fold=next(f for f in cfg()['folds'] if f['cutoff']==source['cutoff']);tr,te=indices(obs,fold)
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(source['cutoff'])))
        np.testing.assert_array_equal(te,np.flatnonzero(obs.date.gt(source['cutoff'])&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')))
        for split,rows in [('training',tr),('validation',te)]:
            x,idx=previous.base_inputs(parent,split);np.testing.assert_array_equal(idx,rows);anchors=obs.anchor.iloc[rows].to_numpy(int)
            gate=previous.scalar_features(price.iloc[:anchors.max()+1],anchors)[:,-1]
            saved=csv(f'{split}_signal_states');saved=saved[saved.cutoff.eq(source['cutoff'])].sort_values('row_index');np.testing.assert_array_equal(saved.row_index,rows)
            gap=float(abs(gate-saved.excess_order.to_numpy()).max());assert gap<2e-12;maxstategap=max(maxstategap,gap)
            product=gate*(x[:,:25]@d['ray']);order=(product-design(x)@d['projection']-float(d['residual_mean']))/float(d['residual_sd']);xx=np.column_stack([x,order]);original,oldrows=inputs(source,split);np.testing.assert_array_equal(oldrows,rows)
            err=float(abs(xx-original).max());assert err<1e-9;reconstructed[(source['job'],split)]=(xx,rows)
            replays.append(dict(source_job=source['job'],split=split,n=len(rows),maximum_gate_gap=gap,maximum_input_gap=err))
    maxerror=0.;total=0;testtotal=0
    for h in heads:
        assert all(h[k]==v for k,v in jobs[h['job']].items());source=sources[h['source_job']];parent=parents[source['source_job']];d=load_npz(h);x,tr=inputs(source,'training');val,te=reconstructed[(source['job'],'validation')]
        y=(returns[tr]>0).astype(float);xx=scaled_inputs(x,h['multiplier']);theta=np.asarray(h['coefficients']);beta=np.asarray(h['original_coefficients'])
        assert h['order_lambda']==.01*h['multiplier'];np.testing.assert_array_equal(d['standardized'],xx);np.testing.assert_array_equal(d['original_input'],x);np.testing.assert_array_equal(d['row_index'],tr);np.testing.assert_array_equal(d['direction'],y)
        np.testing.assert_array_equal(beta,original_coefficients(theta,h['multiplier']));np.testing.assert_array_equal(xx[:,:-1],x[:,:-1]);assert abs(beta[-2])<=abs(source['coefficients'][-2])+1e-7
        value,grad,hess=objective(theta,design(xx),y,.01);dv,dg=direct_objective(beta,x,y,h['multiplier']);audit.near(value,dv,1e-12);audit.near(value,h['objective']);audit.near(float(abs(grad).max()),h['gradient_inf']);audit.near(float(abs(dg).max()),h['original_gradient_inf'])
        assert abs(grad).max()<=1e-9 and abs(dg).max()<=5e-9 and np.linalg.eigvalsh(hess).min()>0;audit.near(float(np.linalg.eigvalsh(hess).min()),h['hessian_min_eigenvalue'])
        pt=np.asarray(parent['coefficients']);nested=np.r_[pt[:-1],0.,pt[-1]];nv,_=direct_objective(nested,x,y,h['multiplier']);audit.near(nv,parent['objective']);audit.near(nv,h['parent_objective']);assert value<=nv+1e-12
        replay,trace=fit_newton(xx,y,cfg()['probe']);np.testing.assert_array_equal(replay,theta);saved=traces[traces.job.eq(h['job'])].sort_values('iteration');pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14);assert trace[-1]['iteration']==h['iterations']
        initial=np.zeros(len(beta));initial[-1]=np.log(y.mean()/(1-y.mean()));alt=minimize(direct_objective,initial,args=(x,y,h['multiplier']),jac=True,method=settings['method'],options=settings['options']);av,ag=direct_objective(alt.x,x,y,h['multiplier'])
        gap=abs(av-value);pgap=float(abs(expit(design(x)@alt.x)-expit(design(x)@beta)).max());gn=float(abs(ag).max());assert gap<=settings['objective_absolute_tolerance'] and pgap<=settings['training_probability_absolute_tolerance'] and gn<=settings['gradient_infinity_tolerance']
        solutions.append(dict(job=h['job'],objective_absolute_gap=gap,maximum_training_probability_gap=pgap,alternate_gradient_inf=gn,alternate_success=bool(alt.success),alternate_iterations=int(alt.nit),alternate_message=str(alt.message)))
        for key,value in independent.independent_metrics(design(x)@beta,y).items():audit.near(value,tm.loc[h['job'],key])
        cg=coefficients[coefficients.job.eq(h['job'])].sort_values('coordinate');np.testing.assert_array_equal(cg.coefficient,beta);np.testing.assert_array_equal(cg.solver_coefficient,theta);np.testing.assert_array_equal(cg.penalty,penalty_vector(x.shape[1],h['multiplier']))
        for block in ['intercept','order_interaction','original_interaction']:assert cg.block.eq(block).sum()==1
        z=design(val)@beta;p=expit(z);g=predictions[predictions.method.eq(h['method'])&predictions.cutoff.eq(h['cutoff'])&predictions.seed.eq(h['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index,te);np.testing.assert_array_equal(g.date,obs.date.iloc[te]);np.testing.assert_array_equal(g.actual_up,returns[te]>0);np.testing.assert_array_equal(g.actual_return,returns[te]);np.testing.assert_array_equal(g.direction_up,p>.5)
        err=max(float(abs(g.probability.to_numpy()-p).max()),float(abs(g.logit.to_numpy()-z).max()));assert err<1e-9;maxerror=max(maxerror,err);total+=len(tr);testtotal+=len(te);forecasts.append(dict(job=h['job'],training_rows=len(tr),heldout_rows=len(te),maximum_forecast_error=err))
        comp=parts[parts.job.eq(h['job'])].sort_values('row_index');np.testing.assert_array_equal(comp.row_index,te)
        for key,value in dict(order_feature=val[:,-1],refitted_base_logit=val[:,:-2]@beta[:-3]+beta[-1],original_interaction_logit=val[:,-2]*beta[-3],order_interaction_logit=val[:,-1]*beta[-2],logit=z).items():np.testing.assert_allclose(comp[key],value,rtol=0,atol=1e-9)
        np.testing.assert_allclose(comp.logit,comp.refitted_base_logit+comp.original_interaction_logit+comp.order_interaction_logit,rtol=0,atol=1e-12)
        for split,values in [('training',x),('validation',val)]:
            s=summary[summary.job.eq(h['job'])&summary.split.eq(split)].iloc[0];assert s.n==len(values);zz=design(values)@beta;oldz=design(values[:,:-1])@pt;weakz=design(values)@np.asarray(source['coefficients'])
            for key,value in dict(order_coefficient=beta[-2],original_interaction_coefficient=beta[-3],order_logit_sd=float((values[:,-1]*beta[-2]).std()),parent_coefficient_l2_drift=float(np.linalg.norm(np.r_[beta[:-2],beta[-1]]-pt)),rms_probability_change_vs_parent=float(np.sqrt(np.square(expit(zz)-expit(oldz)).mean())),rms_probability_change_vs_weak=float(np.sqrt(np.square(expit(zz)-expit(weakz)).mean()))).items():audit.near(value,s[key],1e-9)
    assert len(heads)==len(solutions)==48 and len(replays)==48 and total==75720 and testtotal==2088 and len(coefficients)==1500 and len(parts)==2088 and len(summary)==96 and len(predictions)==14616 and len(ensemble)==8613
    for row in csv('shrinkage_path').itertuples():
        s=sources[row.source_job];hs=sorted([h for h in heads if h['source_job']==row.source_job],key=lambda h:h['multiplier']);values=[s['coefficients'][-2]]+[h['original_coefficients'][-2] for h in hs]
        np.testing.assert_array_equal(values,[row.coefficient_1x,row.coefficient_4x,row.coefficient_16x]);assert abs(values[2])<=abs(values[1])+1e-7<=abs(values[0])+2e-7 and row.absolute_coefficient_monotone
    for filename,frame in [('model_predictions.csv',predictions),('ensemble_predictions.csv',ensemble)]:
        old=pd.read_csv(V23/'results'/filename,float_precision='round_trip');g=frame[frame.method.isin(old.method.unique())];keys=['method','date']+(['seed'] if 'seed' in old else []);pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),g[old.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-15)
    new=csv('new_model_predictions');assert len(new)==2088;pd.testing.assert_frame_equal(new.reset_index(drop=True),predictions[predictions.method.isin(CANDIDATES)][new.columns].reset_index(drop=True),check_dtype=False,rtol=0,atol=0)
    for (method,date),g in new.groupby(['method','date']):
        e=ensemble[ensemble.method.eq(method)&ensemble.date.eq(date)].iloc[0];assert len(g)==(3 if method.startswith('learned') else 1);audit.near(float(g.probability.mean()),e.probability,1e-15);assert bool(e.direction_up)==(e.probability>.5)
    # The independent metric/bootstrap checks below are inherited verbatim in structure,
    # with declared counts and current references updated before protocol freeze.
    preds=predictions
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
    assert groups==2130 and len(tables['reliability_bins'])==320
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
    assert len(pairs)==64;order=sorted(range(64),key=lambda i:pairs[i]['p']);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(64-rank)*pairs[i]['p']));audit.near(running,pairs[i]['holm_adjusted_p'])
    for row in tables['retention_targets'].itertuples():
        early=ensemble[ensemble.year.le(2017)];late=ensemble[ensemble.year.ge(2018)];parent=CANDIDATES[row.method];weak=WEAK[row.method]
        def correct(frame,method):
            g=frame[frame.method.eq(method)];return int(g.direction_up.eq(g.actual_up).sum())
        def brier(frame,method):return float(audit.independent_loss(frame[frame.method.eq(method)],'brier').mean())
        flags=[correct(early,row.method)>=correct(early,weak),correct(late,row.method)>=correct(late,parent),brier(early,row.method)<=brier(early,weak),brier(late,row.method)<=brier(late,weak)]
        assert flags==[row.early_accuracy_retained,row.late_parent_accuracy_retained,row.early_brier_retained,row.late_brier_retained] and row.all_targets_met==all(flags) and not row.independent_confirmation
    usage=csv('historical_date_usage');assert len(usage)==261 and usage.previously_evaluated.all() and set(usage.date)==set(ensemble.date);assert old_evidence()==prep['old_evidence'];files=[]
    for name,rows in [('inherited_feature_verification',replays),('independent_solver_verification',solutions),('forecast_verification',forecasts)]:path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,previous_files_preserved=3389,new_primary_fits=48,independent_solutions=48,new_projection_fits=0,reused_projection_fits=24,inherited_feature_split_checks=48,temporal_provenance_chains=24,
        new_hmm_fits=0,new_neural_forward_passes=0,neural_training_steps=0,inherited_neural_replay_states=18,training_feature_rows=75720,new_probability_forecasts=2088,reused_model_records=12528,model_records=14616,ensemble_records=8613,
        metric_groups=2130,state_metric_rows=1716,reliability_bins=320,primary_contrasts=64,independent_holdout_dates=0,maximum_new_forecast_error=maxerror,maximum_order_state_formula_gap=maxstategap,
        maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in solutions),
        scipy=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)

if __name__=='__main__':main()

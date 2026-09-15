"""Independently audit recent label membership, convex fits and all forecasts."""
from common20 import *
import evaluate20 as evaluation
import verify17 as audit
import verify15 as independent
import scipy
from scipy.optimize import minimize
from scipy.special import expit

def main():
    start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists(),'Preserve verification'
    last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert r[key]==prep[key]
    fitting=read(OUT/'training_manifest.json');assert fitting['all_converged'] and fitting['new_primary_fits']==48 and not fitting['validation_scoring_during_fit']
    assert fitting['new_projection_fits']==fitting['neural_forward_passes']==fitting['neural_training_steps']==0
    inherited=read(V19/'results/verification.json');assert inherited['status']=='PASS' and inherited['independent_qr_projections']==24 and inherited['neural_states_with_inherited_round17_replay']==18
    obs,price,returns=data();heads=read(OUT/'heads.json');head_map={h['job']:h for h in heads};sources={s['job']:s for s in read(OUT/'source_heads.json')}
    jobs={j['job']:j for j in read(OUT/'jobs.json')};assert len(heads)==len(sources)==len(jobs)==48
    original={h['job']:h for folder in [V18,V19] for h in read(folder/'results/heads.json')}
    for key,source in sources.items():assert source==original[key]
    membership=pd.read_csv(OUT/'window_membership.csv');fold_table=pd.read_csv(OUT/'fold_summaries.csv').set_index('cutoff');masks={};fold_audit=[]
    for f in cfg()['folds']:
        lower=f"{int(f['cutoff'][:4])-2}-01-01";assert f['recent_start']==lower
        tr=np.array([i for i,r in enumerate(obs.itertuples()) if r.joint_completed<=f['cutoff']],int)
        selected=np.array([i for i in tr if lower<=obs.date.iloc[i]<=f['cutoff']],int)
        te=np.array([i for i,r in enumerate(obs.itertuples()) if f['cutoff']<r.date<=f['end'] and r.weekday==4 and r.joint_completed<='2020-12-31'],int)
        assert len(tr)==f['train_n'] and len(selected)==f['recent_train_n'] and len(te)==f['test_n'] and not len(np.intersect1d(selected,te))
        g=membership[membership.cutoff.eq(f['cutoff'])].sort_values('row_index');np.testing.assert_array_equal(g.row_index,tr)
        np.testing.assert_array_equal(g.date,obs.date.iloc[tr]);np.testing.assert_array_equal(g.joint_completed,obs.joint_completed.iloc[tr])
        np.testing.assert_array_equal(g.included_recent,np.isin(tr,selected));masks[f['cutoff']]=(tr,selected,te)
        r=fold_table.loc[f['cutoff']];assert r.full_train_n==len(tr) and r.recent_train_n==len(selected) and r.removed_n==len(tr)-len(selected) and r.test_n==len(te)
        assert r.first_recent_date==obs.date.iloc[selected[0]] and r.last_recent_date==obs.date.iloc[selected[-1]] and r.last_recent_label_completed==obs.joint_completed.iloc[selected].max()
        audit.near(float((returns[tr]>0).mean()),r.full_frequency);audit.near(float((returns[selected]>0).mean()),r.recent_frequency)
        fold_audit.append(dict(cutoff=f['cutoff'],first_allowed_signal=lower,full_rows=len(tr),recent_rows=len(selected),test_rows=len(te),recent_up_labels=int((returns[selected]>0).sum()),latest_label_completed=obs.joint_completed.iloc[selected].max()))
    assert len(membership)==9465 and sum(len(v[1]) for v in masks.values())==3951
    predictions=pd.read_csv(OUT/'model_predictions.csv',float_precision='round_trip');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv',float_precision='round_trip')
    old_predictions=pd.read_csv(V19/'results/model_predictions.csv',float_precision='round_trip');traces=pd.read_csv(OUT/'solver_trace.csv')
    training_metrics=pd.read_csv(OUT/'training_metrics.csv').set_index('job');coefficients=pd.read_csv(OUT/'coefficients.csv')
    components=pd.read_csv(OUT/'interaction_components.csv',float_precision='round_trip');distribution=pd.read_csv(OUT/'interaction_input_distribution.csv')
    settings=cfg()['independent_solver'];solutions=[];forecast_checks=[];total_train=0;total_test=0;maximum_error=0.;maximum_source_error=0.;nested_checks=0
    for h in heads:
        for key,val in jobs[h['job']].items():assert h[key]==val
        tr,selected,te=masks[h['cutoff']];source=sources[h['source_job']];full,idx=source_inputs(source,h['source_round'],'training');np.testing.assert_array_equal(idx,tr)
        pos=np.searchsorted(tr,selected);x=full[pos];y=(returns[selected]>0).astype(float);d=load_npz(h)
        for key,val in [('standardized',x),('row_index',selected),('direction',y),('full_row_positions',pos)]:np.testing.assert_array_equal(d[key],val)
        assert x.shape==(len(selected),h['dimensions']) and h['train_n']==len(selected) and h['full_train_n']==len(tr)
        audit.near(h['training_frequency'],float(y.mean()));theta=np.asarray(h['coefficients']);assert len(theta)==h['dimensions']+1 and h['l2_lambda']==.01
        value,grad,hess=objective(theta,design(x),y,.01);assert np.max(np.abs(grad))<=1e-9 and np.linalg.eigvalsh(hess).min()>0
        audit.near(value,h['objective'],1e-14);audit.near(float(np.max(np.abs(grad))),h['gradient_inf'],1e-14);audit.near(float(np.linalg.eigvalsh(hess).min()),h['hessian_min_eigenvalue'])
        source_theta=np.asarray(source['coefficients']);sv,sg=independent.independent_objective(source_theta,x,y,.01)
        audit.near(sv,h['source_on_recent_objective']);assert value<=sv+1e-12
        if h['interaction']:
            parent=head_map[h['recent_parent_job']];p=load_npz(parent);np.testing.assert_array_equal(d['row_index'],p['row_index']);np.testing.assert_array_equal(x[:,:-1],p['standardized'])
            pt=np.asarray(parent['coefficients']);nested=np.r_[pt[:-1],0.,pt[-1]];nv,ng=independent.independent_objective(nested,x,y,.01)
            audit.near(nv,parent['objective']);audit.near(nv,h['nested_recent_additive_objective']);assert value<=nv+1e-12;nested_checks+=1
        else:assert h['nested_recent_additive_objective'] is None
        replay,trace=fit_newton(x,y,cfg()['probe']);np.testing.assert_allclose(theta,replay,rtol=0,atol=1e-12);assert trace[-1]['iteration']==h['iterations']
        saved=traces[traces.job.eq(h['job'])].sort_values('iteration');pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
        initial=np.zeros(len(theta));initial[-1]=np.log(y.mean()/(1-y.mean()))
        alt=minimize(independent.independent_objective,initial,args=(x,y,.01),method=settings['method'],jac=True,options=settings['options'])
        av,ag=independent.independent_objective(alt.x,x,y,.01);gap=abs(av-value);pgap=float(np.max(np.abs(expit(x@alt.x[:-1]+alt.x[-1])-probability(design(x)@theta))));gn=float(np.max(np.abs(ag)))
        assert gap<=settings['objective_absolute_tolerance'] and pgap<=settings['training_probability_absolute_tolerance'] and gn<=settings['gradient_infinity_tolerance']
        solutions.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],alternate_iterations=int(alt.nit),alternate_success=bool(alt.success),alternate_message=str(alt.message),
            objective_absolute_gap=gap,maximum_training_probability_gap=pgap,alternate_gradient_inf=gn))
        for key,val in independent.independent_metrics(x@theta[:-1]+theta[-1],y).items():audit.near(val,training_metrics.loc[h['job'],key])
        coef=coefficients[coefficients.job.eq(h['job'])].sort_values('coordinate')
        for column,values in [('coefficient',theta),('full_window_coefficient',source_theta),('coefficient_delta',theta-source_theta)]:np.testing.assert_allclose(coef[column],values,rtol=0,atol=1e-15)
        assert coef.block.eq('intercept').sum()==1 and coef.block.eq('interaction').sum()==int(h['interaction'])
        xx,testrows=source_inputs(source,h['source_round'],'validation');np.testing.assert_array_equal(testrows,te)
        if h['interaction']:
            parent=head_map[h['recent_parent_job']];additive,_=source_inputs(sources[parent['source_job']],18,'validation');np.testing.assert_array_equal(xx[:,:-1],additive)
            # Evaluate the inherited interaction formula directly, without refitting it.
            trans=load_npz(source);base18=sources[source['source_job']];bx,bv,br=previous.inherited_inputs(base18,'validation')
            direct_h=(bv*(bx[:,:25]@trans['ray'])-np.column_stack([bx,np.ones(len(bx))])@trans['projection']-float(trans['residual_mean']))/float(trans['residual_sd'])
            np.testing.assert_array_equal(xx[:,-1],direct_h)
        old_g=old_predictions[old_predictions.method.eq(source['method'])&old_predictions.cutoff.eq(h['cutoff'])&old_predictions.seed.eq(h['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(old_g.row_index,te);sz=xx@source_theta[:-1]+source_theta[-1]
        source_error=max(float(np.max(np.abs(sz-old_g.logit.to_numpy()))),float(np.max(np.abs(expit(sz)-old_g.probability.to_numpy()))));assert source_error<1e-12;maximum_source_error=max(maximum_source_error,source_error)
        z=xx@theta[:-1]+theta[-1];prob=expit(z);g=predictions[predictions.method.eq(h['method'])&predictions.cutoff.eq(h['cutoff'])&predictions.seed.eq(h['seed'])].sort_values('row_index')
        for actual,expected in [(g.row_index,te),(g.date,obs.date.iloc[te]),(g.actual_up,returns[te]>0),(g.actual_return,returns[te]),(g.direction_up,prob>.5)]:np.testing.assert_array_equal(actual,expected)
        err=max(float(np.max(np.abs(z-g.logit.to_numpy()))),float(np.max(np.abs(prob-g.probability.to_numpy()))));assert err<1e-12;maximum_error=max(maximum_error,err)
        if h['interaction']:
            comp=components[components.job.eq(h['job'])].sort_values('row_index');np.testing.assert_array_equal(comp.row_index,te)
            for column,values in [('inherited_interaction_feature',xx[:,-1]),('refitted_additive_logit',xx[:,:-1]@theta[:-2]+theta[-1]),('interaction_logit',xx[:,-1]*theta[-2]),('logit',z)]:
                np.testing.assert_allclose(comp[column],values,rtol=0,atol=1e-12)
            for split,values in [('full_training',full),('recent_training',x),('validation',xx)]:
                r=distribution[distribution.job.eq(h['job'])&distribution.split.eq(split)].iloc[0];assert r.n==len(values);hh=values[:,-1];aa=values[:,:-1]@theta[:-2]+theta[-1];bb=hh*theta[-2]
                expected=dict(interaction_feature_mean=float(hh.mean()),interaction_feature_std=float(hh.std()),interaction_feature_min=float(hh.min()),interaction_feature_max=float(hh.max()),
                    interaction_coefficient=float(theta[-2]),old_interaction_coefficient=float(source_theta[-2]),refitted_additive_mean=float(aa.mean()),interaction_logit_mean=float(bb.mean()),
                    interaction_logit_std=float(bb.std()),total_logit_mean=float((aa+bb).mean()),total_logit_std=float((aa+bb).std()))
                for key,val in expected.items():audit.near(val,r[key])
        total_train+=len(selected);total_test+=len(te);forecast_checks.append(dict(job=h['job'],training_rows=len(selected),validation_rows=len(te),dimensions=h['dimensions'],maximum_new_forecast_error=err,maximum_source_forecast_error=source_error))
    assert total_train==31608 and total_test==2088 and len(coefficients)==1428 and nested_checks==24 and len(components)==1044 and len(distribution)==72
    assert len(predictions)==7308 and len(ensemble)==4437
    for filename,frame in [('model_predictions.csv',predictions),('ensemble_predictions.csv',ensemble)]:
        old=pd.read_csv(V19/'results'/filename,float_precision='round_trip');g=frame[frame.method.isin(old.method.unique())];keys=['method','date']+(['seed'] if 'seed' in old else [])
        pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),g[old.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-15)
    new=pd.read_csv(OUT/'new_model_predictions.csv',float_precision='round_trip');assert len(new)==2088
    pd.testing.assert_frame_equal(new.reset_index(drop=True),predictions[predictions.method.isin(FULL)][new.columns].reset_index(drop=True),check_dtype=False,rtol=0,atol=0)
    for (method,date),g in new.groupby(['method','date']):
        e=ensemble[ensemble.method.eq(method)&ensemble.date.eq(date)].iloc[0];assert len(g)==(3 if method.startswith('learned') else 1)
        audit.near(float(g.probability.mean()),e.probability,1e-15);assert bool(e.direction_up)==(e.probability>.5)
    freq=pd.read_csv(OUT/'recent_frequency_predictions.csv',float_precision='round_trip');assert len(freq)==261
    for f in cfg()['folds']:
        tr,recent,te=masks[f['cutoff']];rate=float((returns[recent]>0).mean());g=freq[freq.cutoff.eq(f['cutoff'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index,te);np.testing.assert_array_equal(g.probability,np.full(len(te),rate));np.testing.assert_array_equal(g.direction_up,np.full(len(te),rate>.5))
    pd.testing.assert_frame_equal(freq.reset_index(drop=True),ensemble[ensemble.method.eq('recent_frequency')][freq.columns].reset_index(drop=True),check_dtype=False,rtol=0,atol=0)
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
        expected=float(audit.independent_loss(s,'brier').mean()-audit.independent_loss(f,'brier').mean()) if len(s) and s.probability.notna().all() else None;audit.near(expected,r['brier_difference_vs_frequency'])
    assert metric_groups==624 and len(tables['state_metrics'])==408 and len(tables['reliability_bins'])==160
    for r in tables['reliability_bins'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);g=evaluation.select(ensemble,w);g=g[g.method.eq(r['method'])]
        s=g[g.probability.ge(r['lower']) & (g.probability.lt(r['upper']) if r['bin_index']<4 else g.probability.le(1))]
        assert len(s)==r['n'];audit.near(float(s.probability.mean()) if len(s) else None,r['mean_probability']);audit.near(float(s.actual_up.mean()) if len(s) else None,r['observed_frequency'])
    for r in tables['direction_changes'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);e=evaluation.select(ensemble,w);a=e[e.method.eq(r['method'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date')
        aa=a.direction_up.to_numpy()==a.actual_up.to_numpy();bb=b.direction_up.to_numpy()==b.actual_up.to_numpy()
        assert r['changed']==np.count_nonzero(aa!=bb) and r['correct_to_wrong']==np.count_nonzero(bb&~aa) and r['wrong_to_correct']==np.count_nonzero(~bb&aa)
    assert len(tables['direction_changes'])==36
    for r in tables['factorial_differences'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);e=evaluation.select(ensemble,w);fa,fi,ra,ri=FACTORIAL[r['family']]
        ll={m:audit.independent_loss(e[e.method.eq(m)].sort_values('date'),r['metric']) for m in [fa,fi,ra,ri]}
        full=float((ll[fi]-ll[fa]).mean());recent=float((ll[ri]-ll[ra]).mean());audit.near(full,r['full_interaction_increment']);audit.near(recent,r['recent_interaction_increment']);audit.near(recent-full,r['difference_in_differences'])
    assert len(tables['factorial_differences'])==8
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))));ids=np.empty((10000,n),int)
        for k in range(n):ids[:,k]=(starts[:,k//8]+k%8)%n
        e=evaluation.select(ensemble,w)
        for r in [q for q in pairs if q['window']==w['name']]:
            a=e[e.method.eq(r['candidate'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');values=audit.independent_loss(a,r['metric'])-audit.independent_loss(b,r['metric'])
            mean=float(values.mean());boot=values[ids].mean(axis=1);centered=(values-mean)[ids].mean(axis=1)
            for actual,key in [(mean,'difference'),(float(np.quantile(boot,.025)),'ci95_low'),(float(np.quantile(boot,.975)),'ci95_high'),(float((1+np.count_nonzero(np.abs(centered)>=abs(mean)))/10001),'p')]:audit.near(actual,r[key])
    assert len(pairs)==32;order=sorted(range(32),key=lambda i:pairs[i]['p']);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(32-rank)*pairs[i]['p']));audit.near(running,pairs[i]['holm_adjusted_p'])
    usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(usage)==len(states)==261 and usage.previously_evaluated.all() and set(usage.date)==set(states.date)
    assert old_evidence()==prep['old_evidence'];files=[]
    for name,records in [('independent_solver_verification',solutions),('forecast_verification',forecast_checks),('window_verification',fold_audit)]:
        path=OUT/f'{name}.csv';pd.DataFrame(records).to_csv(path,index=False);files.append(path)
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,previous_files_preserved=2958,new_primary_fits=48,independent_solutions=48,
        neural_states_with_inherited_round17_replay=18,interaction_transforms_with_inherited_round19_qr_verification=24,new_neural_forward_passes=0,neural_training_steps=0,new_projection_fits=0,
        training_feature_rows=31608,full_fold_membership_rows=9465,recent_fold_membership_rows=3951,nested_recent_interaction_checks=24,
        new_probability_forecasts=2088,new_deterministic_baseline_forecasts=261,reused_model_records=5220,model_records=7308,ensemble_records=4437,
        metric_groups=624,state_metric_rows=408,reliability_bins=160,primary_contrasts=32,factorial_diagnostics=8,independent_holdout_dates=0,
        maximum_new_forecast_error=maximum_error,maximum_inherited_forecast_error=maximum_source_error,
        maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),
        maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in solutions),scipy=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:val for k,val in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)

if __name__=='__main__':main()

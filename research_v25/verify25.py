"""Verify frozen parent coefficients, scalar roots, projections, predictions and metrics."""
from common25 import *
import evaluate25 as evaluation
import verify17 as audit
import verify15 as independent
from contract25 import synthetic,inherited_provenance
from scipy.special import expit
import scipy

def main():
    start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists();synthetic();pd.testing.assert_frame_equal(pd.DataFrame(inherited_provenance()),csv('temporal_provenance'),check_dtype=False)
    last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        run=check_phase(phase);assert last<=run['started_utc']<run['finished_utc'];last=run['finished_utc']
        for key in ['source_sha256','protocol_sha256','input_sha256']:assert run[key]==prep[key]
    fit=read(OUT/'training_manifest.json');assert fit['all_converged'] and fit['new_primary_fits']==24 and fit['new_trainable_coefficients']==24 and fit['frozen_parent_coefficients']==726 and fit['new_projection_fits']==0 and fit['reused_projection_fits']==24 and not fit['validation_scoring_during_fit']
    assert fit['neural_forward_passes']==fit['neural_training_steps']==fit['new_hmm_fits']==0
    obs,price,returns=data();sources={h['job']:h for h in read(OUT/'source_heads.json')};parents={h['job']:h for h in read(V19/'results/heads.json')};heads=read(OUT/'heads.json');jobs={h['job']:h for h in read(OUT/'jobs.json')}
    predictions=csv('model_predictions');ensemble=csv('ensemble_predictions');parts=csv('interaction_components');summary=csv('component_summary');traces=csv('solver_trace');coefficients=csv('coefficients');tm=csv('training_metrics').set_index('job');settings=cfg()['independent_solver'];solutions=[];forecasts=[];replays=[];reconstructed={};maxstategap=0.
    for source in sources.values():
        parent=parents[source['source_job']];d=load_npz(source);fold=next(f for f in cfg()['folds'] if f['cutoff']==source['cutoff']);tr,te=indices(obs,fold)
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(source['cutoff'])))
        np.testing.assert_array_equal(te,np.flatnonzero(obs.date.gt(source['cutoff'])&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')));assert len(tr)==fold['train_n'] and len(te)==fold['test_n']
        for split,rows in [('training',tr),('validation',te)]:
            x,idx=previous.previous.base_inputs(parent,split);np.testing.assert_array_equal(idx,rows);anchors=obs.anchor.iloc[rows].to_numpy(int);gate=previous.previous.scalar_features(price.iloc[:anchors.max()+1],anchors)[:,-1]
            saved=csv(f'{split}_signal_states');saved=saved[saved.cutoff.eq(source['cutoff'])].sort_values('row_index');np.testing.assert_array_equal(saved.row_index,rows);gap=float(abs(gate-saved.excess_order.to_numpy()).max());assert gap<2e-12;maxstategap=max(maxstategap,gap)
            product=gate*(x[:,:25]@d['ray']);order=(product-design(x)@d['projection']-float(d['residual_mean']))/float(d['residual_sd']);xx=np.column_stack([x,order]);original,oldrows=inputs(source,split);np.testing.assert_array_equal(oldrows,rows)
            err=float(abs(xx-original).max());assert err<1e-9;reconstructed[(source['job'],split)]=(xx,rows);replays.append(dict(source_job=source['job'],split=split,n=len(rows),maximum_gate_gap=gap,maximum_input_gap=err))
    maxerror=0.;maxparent=0.;maxjoint=0.;total=0;testtotal=0
    for h in heads:
        assert all(h[k]==v for k,v in jobs[h['job']].items());assert h['free_parameters']==1 and h['free_intercepts']==0 and h['order_lambda']==.01;source=sources[h['source_job']];parent=parents[h['parent_job']];d=load_npz(h);x,tr=inputs(source,'training');val,te=reconstructed[(source['job'],'validation')]
        y=(returns[tr]>0).astype(float);pt=np.asarray(parent['coefficients']);joint=np.asarray(source['coefficients']);gamma=h['gamma'];theta=np.asarray(h['coefficients']);offset=design(x[:,:-1])@pt;feature=x[:,-1]
        np.testing.assert_array_equal(h['frozen_parent_coefficients'],pt);np.testing.assert_array_equal(theta,np.r_[pt[:-1],gamma,pt[-1]]);assert h['parent_coefficient_l2_drift']==h['parent_intercept_drift']==0
        for key,value in dict(standardized=x,row_index=tr,direction=y,offset=offset,order_feature=feature).items():np.testing.assert_array_equal(d[key],value)
        constant=float(.005*np.square(pt[:-1]).sum());value,grad,hess=offset_objective(gamma,offset,feature,y);zero=offset_objective(0.,offset,feature,y)[0]+constant
        full,fullgrad,_=previous.objective(theta,design(x),y,.01);audit.near(full,value+constant);audit.near(float(fullgrad[-2]),grad);assert abs(grad)<=1e-10 and hess>=.01
        for key,vv in dict(reduced_objective=value,parent_penalty_constant=constant,objective=full,parent_objective=parent['objective'],joint_objective=source['objective'],gradient_absolute=abs(grad),hessian=hess).items():audit.near(vv,h[key])
        audit.near(zero,parent['objective']);assert source['objective']<=full+1e-12<=zero+2e-12
        replay,trace=fit_offset(offset,feature,y,cfg()['probe']);assert replay==gamma;saved=traces[traces.job.eq(h['job'])].sort_values('iteration');pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14);assert trace[-1]['iteration']==h['iterations']
        alt=independent_root(offset,feature,y,cfg()['probe']);cgap=abs(alt['gamma']-gamma);ogap=abs(alt['objective']-value);pgap=float(abs(expit(offset+alt['gamma']*feature)-expit(offset+gamma*feature)).max())
        assert alt['converged'] and cgap<=settings['coefficient_absolute_tolerance'] and ogap<=settings['objective_absolute_tolerance'] and pgap<=settings['training_probability_absolute_tolerance'] and abs(alt['gradient'])<=settings['gradient_absolute_tolerance']
        solutions.append(dict(job=h['job'],root_gamma=alt['gamma'],coefficient_absolute_gap=cgap,objective_absolute_gap=ogap,maximum_training_probability_gap=pgap,alternate_gradient_absolute=abs(alt['gradient']),root_iterations=alt['iterations'],root_converged=alt['converged']))
        for key,vv in independent.independent_metrics(offset+gamma*feature,y).items():audit.near(vv,tm.loc[h['job'],key])
        cg=coefficients[coefficients.job.eq(h['job'])].sort_values('coordinate');np.testing.assert_array_equal(cg.coefficient,theta);np.testing.assert_array_equal(cg.trainable,cg.coordinate.eq(len(theta)-2));assert cg.trainable.sum()==1
        for block in ['intercept','order_interaction','original_interaction']:assert cg.block.eq(block).sum()==1
        parent_z=design(val[:,:-1])@pt;z=parent_z+gamma*val[:,-1];p=expit(z);joint_parent=design(val[:,:-1])@np.r_[joint[:-2],joint[-1]];joint_order=joint[-2]*val[:,-1];joint_z=joint_parent+joint_order
        g=predictions[predictions.method.eq(h['method'])&predictions.cutoff.eq(h['cutoff'])&predictions.seed.eq(h['seed'])].sort_values('row_index');np.testing.assert_array_equal(g.row_index,te);np.testing.assert_array_equal(g.date,obs.date.iloc[te]);np.testing.assert_array_equal(g.actual_up,returns[te]>0);np.testing.assert_array_equal(g.actual_return,returns[te]);np.testing.assert_array_equal(g.direction_up,p>.5)
        err=max(float(abs(g.probability.to_numpy()-p).max()),float(abs(g.logit.to_numpy()-z).max()));assert err<1e-9;maxerror=max(maxerror,err);total+=len(tr);testtotal+=len(te)
        parent_old=predictions[predictions.method.eq(parent['method'])&predictions.cutoff.eq(h['cutoff'])&predictions.seed.eq(h['seed'])].sort_values('row_index');joint_old=predictions[predictions.method.eq(source['method'])&predictions.cutoff.eq(h['cutoff'])&predictions.seed.eq(h['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(parent_old.row_index,te);np.testing.assert_array_equal(joint_old.row_index,te);perr=float(abs(expit(parent_z)-parent_old.probability.to_numpy()).max());jerr=float(abs(expit(joint_z)-joint_old.probability.to_numpy()).max());assert perr<1e-9 and jerr<1e-9;maxparent=max(maxparent,perr);maxjoint=max(maxjoint,jerr)
        forecasts.append(dict(job=h['job'],training_rows=len(tr),heldout_rows=len(te),maximum_forecast_error=err,maximum_parent_probability_gap=perr,maximum_joint_probability_gap=jerr))
        comp=parts[parts.job.eq(h['job'])].sort_values('row_index');np.testing.assert_array_equal(comp.row_index,te)
        expected=dict(order_feature=val[:,-1],frozen_base_logit=val[:,:-2]@pt[:-2]+pt[-1],frozen_original_interaction_logit=val[:,-2]*pt[-2],frozen_parent_logit=parent_z,order_interaction_logit=gamma*val[:,-1],logit=z,joint_parent_logit=joint_parent,joint_order_logit=joint_order,joint_logit=joint_z,parent_constraint_component=parent_z-joint_parent,order_coefficient_component=(gamma-joint[-2])*val[:,-1],total_logit_difference=z-joint_z)
        for key,vv in expected.items():np.testing.assert_allclose(comp[key],vv,rtol=0,atol=1e-9)
        np.testing.assert_allclose(comp.total_logit_difference,comp.parent_constraint_component+comp.order_coefficient_component,rtol=0,atol=1e-12);np.testing.assert_allclose(comp.logit,comp.frozen_parent_logit+comp.order_interaction_logit,rtol=0,atol=1e-12)
        for split,values in [('training',x),('validation',val)]:
            s=summary[summary.job.eq(h['job'])&summary.split.eq(split)].iloc[0];assert s.n==len(values);pz=design(values[:,:-1])@pt;zz=pz+gamma*values[:,-1];jp=design(values[:,:-1])@np.r_[joint[:-2],joint[-1]];jz=jp+joint[-2]*values[:,-1];dp=pz-jp;dg=(gamma-joint[-2])*values[:,-1]
            measured=dict(offset_gamma=gamma,joint_gamma=joint[-2],gamma_difference=gamma-joint[-2],frozen_original_interaction_coefficient=pt[-2],parent_coefficient_l2_drift=0.,parent_intercept_drift=0.,order_logit_sd=float((gamma*values[:,-1]).std()),parent_constraint_rms=float(np.sqrt(np.square(dp).mean())),order_coefficient_change_rms=float(np.sqrt(np.square(dg).mean())),total_logit_difference_rms=float(np.sqrt(np.square(zz-jz).mean())),rms_probability_change_vs_parent=float(np.sqrt(np.square(expit(zz)-expit(pz)).mean())),rms_probability_change_vs_joint=float(np.sqrt(np.square(expit(zz)-expit(jz)).mean())))
            for key,vv in measured.items():audit.near(vv,s[key],1e-9)
    assert len(heads)==len(solutions)==24 and len(replays)==48 and total==37860 and testtotal==1044 and len(coefficients)==750 and coefficients.trainable.sum()==24 and len(parts)==1044 and len(summary)==48 and len(predictions)==15660 and len(ensemble)==9135
    for filename,frame in [('model_predictions.csv',predictions),('ensemble_predictions.csv',ensemble)]:
        old=pd.read_csv(V24/'results'/filename,float_precision='round_trip');g=frame[frame.method.isin(old.method.unique())];keys=['method','date']+(['seed'] if 'seed' in old else []);pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),g[old.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-15)
    new=csv('new_model_predictions');assert len(new)==1044;pd.testing.assert_frame_equal(new.reset_index(drop=True),predictions[predictions.method.isin(CANDIDATES)][new.columns].reset_index(drop=True),check_dtype=False,rtol=0,atol=0)
    for (method,date),g in new.groupby(['method','date']):
        e=ensemble[ensemble.method.eq(method)&ensemble.date.eq(date)].iloc[0];assert len(g)==(3 if method.startswith('learned') else 1);audit.near(float(g.probability.mean()),e.probability,1e-15);assert bool(e.direction_up)==(e.probability>.5)
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
    assert groups==2261 and len(tables['reliability_bins'])==340
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
    assert len(pairs)==32;order=sorted(range(32),key=lambda i:pairs[i]['p']);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(32-rank)*pairs[i]['p']));audit.near(running,pairs[i]['holm_adjusted_p'])
    for row in tables['retention_targets'].itertuples():
        early=ensemble[ensemble.year.le(2017)];late=ensemble[ensemble.year.ge(2018)];parent=CANDIDATES[row.method];weak=WEAK[row.method]
        def correct(frame,method):
            g=frame[frame.method.eq(method)];return int(g.direction_up.eq(g.actual_up).sum())
        def brier(frame,method):return float(audit.independent_loss(frame[frame.method.eq(method)],'brier').mean())
        flags=[correct(early,row.method)>=correct(early,weak),correct(late,row.method)>=correct(late,parent),brier(early,row.method)<=brier(early,weak),brier(late,row.method)<=brier(late,weak)]
        assert flags==[row.early_accuracy_retained,row.late_parent_accuracy_retained,row.early_brier_retained,row.late_brier_retained] and row.all_targets_met==all(flags) and not row.independent_confirmation
    usage=csv('historical_date_usage');assert len(usage)==261 and usage.previously_evaluated.all() and set(usage.date)==set(ensemble.date);assert old_evidence()==prep['old_evidence'];files=[]
    for name,rows in [('inherited_feature_verification',replays),('independent_solver_verification',solutions),('forecast_verification',forecasts)]:path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,previous_files_preserved=3585,new_primary_fits=24,independent_roots=24,new_trainable_coefficients=24,frozen_parent_coefficients=726,parent_coefficients_bitwise_unchanged=True,parent_intercepts_bitwise_unchanged=True,new_projection_fits=0,reused_projection_fits=24,inherited_feature_split_checks=48,temporal_provenance_chains=24,
        new_hmm_fits=0,new_neural_forward_passes=0,neural_training_steps=0,inherited_neural_replay_states=18,training_feature_rows=37860,new_probability_forecasts=1044,reused_model_records=14616,model_records=15660,ensemble_records=9135,
        metric_groups=2261,state_metric_rows=1820,reliability_bins=340,primary_contrasts=32,independent_holdout_dates=0,maximum_new_forecast_error=maxerror,maximum_parent_probability_gap=maxparent,maximum_joint_probability_gap=maxjoint,maximum_order_state_formula_gap=maxstategap,
        maximum_alternate_coefficient_gap=max(r['coefficient_absolute_gap'] for r in solutions),maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),maximum_alternate_gradient_absolute=max(r['alternate_gradient_absolute'] for r in solutions),
        scipy=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)

if __name__=='__main__':main()

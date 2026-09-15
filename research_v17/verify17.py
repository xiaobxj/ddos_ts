"""Independent order statistics, solver, forecast, state and metric verification."""
from common17 import *
import evaluate17 as evaluation
import verify15 as independent
import scipy
from scipy.optimize import minimize
from scipy.special import expit

def quantile_reference(f,q):
    ordered=np.sort(np.asarray(f,float),axis=0);p=(len(ordered)-1)*q;lo=int(np.floor(p));hi=int(np.ceil(p))
    return ordered[lo]+(p-lo)*(ordered[hi]-ordered[lo])

def near(a,b,tol=1e-12):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    else:assert abs(a-b)<=tol,(a,b)

def verify_metrics(g,r):
    y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool);n=len(g);assert n==r['n']
    assert int((up==y).sum())==r['correct_directions']
    for key,mask in [('tp',up&y),('tn',~up&~y),('fp',up&~y),('fn',~up&y)]:assert int(mask.sum())==r[key]
    if not n:
        for k in ['accuracy','direction_error','balanced_accuracy','auroc','brier','log_loss']:assert pd.isna(r[k])
        return
    near(float((up==y).mean()),r['accuracy']);near(float((up!=y).mean()),r['direction_error'])
    near(float(up.mean()),r['predicted_up_fraction']);near(float(y.mean()),r['observed_up_fraction'])
    if y.any() and (~y).any():
        near(float((up[y].mean()+(~up[~y]).mean())/2),r['balanced_accuracy'])
        score=g.score.to_numpy();pairs=score[y,None]-score[~y][None,:];near(float(((pairs>0)+.5*(pairs==0)).mean()),r['auroc'])
    else:assert pd.isna(r['balanced_accuracy']) and pd.isna(r['auroc'])
    if g.probability.notna().all():
        p=g.probability.to_numpy();q=np.minimum(1-1e-12,np.maximum(1e-12,p));assert np.all((p>=0)&(p<=1))
        near(float(np.square(p-y.astype(float)).mean()),r['brier']);near(float(-np.where(y,np.log(q),np.log(1-q)).mean()),r['log_loss'])
        near(float(p.mean()),r['mean_probability']);near(float(p.std()),r['probability_std']);near(float(p.mean()-y.mean()),r['calibration_gap'])
        assert int(((p<1e-12)|(p>1-1e-12)).sum())==r['clipped_probabilities']
    else:assert g.probability.isna().all() and pd.isna(r['brier']) and pd.isna(r['log_loss'])

def independent_loss(g,name):
    if name=='direction_error':return g.direction_up.ne(g.actual_up).to_numpy(float)
    assert name=='brier';return np.square(g.probability.to_numpy()-g.actual_up.to_numpy(float))

def main():
    legacy.initialize();start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists(),'Preserve verification'
    last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    training=read(OUT/'training_manifest.json');assert training['new_primary_fits']==24 and training['all_converged']
    assert not training['validation_scoring_during_fit'] and not training['labels_used_to_fit_state_thresholds'] and training['neural_training_steps']==0
    obs,price,returns=data();bars=previous.raw_bars(price)
    heads=read(OUT/'heads.json');sources={h['job']:h for h in read(OUT/'source_heads.json')};refs={r['job']:r for r in read(V16/'results/validation_features.json')}
    predictions=pd.read_csv(OUT/'model_predictions.csv',float_precision='round_trip');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv',float_precision='round_trip')
    trace=pd.read_csv(OUT/'solver_trace.csv');train_metrics=pd.read_csv(OUT/'training_metrics.csv').set_index('job');settings=cfg()['independent_solver']
    solutions=[];replays=[];learned_train=0;raw_train=0;learned_test=0;raw_test=0;max_prediction=0.;native_max=0.
    for head in heads:
        source=sources[head['source_job']];fold=next(f for f in cfg()['folds'] if f['cutoff']==head['cutoff']);tr,te=indices(obs,fold)
        old=load_npz(source);new=load_npz(head);val=load_npz(refs[source['job']]);y=(returns[tr]>0).astype(float)
        for d,idx in [(old,tr),(new,tr),(val,te)]:np.testing.assert_array_equal(d['row_index'],idx)
        np.testing.assert_array_equal(old['direction'],y);np.testing.assert_array_equal(new['direction'],y)
        if source['kind']=='learned':
            model,state,values,f,_=previous.extract_learned(source,tr);assert state['train_n']==len(tr);learned_train+=len(tr);del values
            values=legacy.batch_tensors(te);vf,_=previous.previous.extract_features(model,values)
            native=previous.previous.archival_native_output(model,values).astype(float)*state['scales']['returns_sd']+state['scales']['returns_mean']
            saved=predictions[predictions.method.eq('native_mse')&predictions.cutoff.eq(head['cutoff'])&predictions.seed.eq(head['seed'])].sort_values('row_index')
            np.testing.assert_array_equal(saved.row_index,te);err=float(np.max(np.abs(native-saved.score.to_numpy())));assert err<1e-10;native_max=max(native_max,err)
            assert previous.object_hash(model.state_dict())==source['model_sha256'] and not any(p.requires_grad or p.grad is not None for p in model.parameters())
            learned_test+=len(te);del model,values;torch.cuda.empty_cache()
        else:
            f=previous.raw_features(bars,obs.anchor.iloc[tr].to_numpy());vf=previous.raw_features(bars,obs.anchor.iloc[te].to_numpy())
            np.testing.assert_allclose(f,previous.scalar_features(bars,obs.anchor.iloc[tr].to_numpy()),rtol=0,atol=1e-14)
            np.testing.assert_allclose(vf,previous.scalar_features(bars,obs.anchor.iloc[te].to_numpy()),rtol=0,atol=1e-14)
            raw_train+=len(tr);raw_test+=len(te)
        np.testing.assert_array_equal(f,old['features']);np.testing.assert_array_equal(vf,val['features'])
        low=quantile_reference(f,.01);high=quantile_reference(f,.99)
        np.testing.assert_allclose(low,new['lower'],rtol=0,atol=1e-12);np.testing.assert_allclose(high,new['upper'],rtol=0,atol=1e-12)
        clipped=np.maximum(new['lower'],np.minimum(new['upper'],f.astype(float)));mean=clipped.mean(axis=0);sd=np.maximum(np.sqrt(np.square(clipped-mean).mean(axis=0)),1e-6)
        np.testing.assert_array_equal(clipped,new['clipped_features']);np.testing.assert_array_equal(mean,new['mean']);np.testing.assert_array_equal(sd,new['sd'])
        x=(clipped-mean)/sd;np.testing.assert_array_equal(x,new['standardized']);theta=np.asarray(head['coefficients'])
        value,grad,hess=objective(theta,design(x),y,.01);assert np.max(np.abs(grad))<=1e-9 and np.linalg.eigvalsh(hess).min()>0
        near(value,head['objective'],1e-14);near(float(np.max(np.abs(grad))),head['gradient_inf'],1e-14)
        near(float(np.linalg.eigvalsh(hess).min()),head['hessian_min_eigenvalue'])
        replay,rt=fit_newton(x,y,cfg()['probe']);np.testing.assert_array_equal(replay,theta);assert rt[-1]['iteration']==head['iterations']
        saved_trace=trace[trace.job.eq(head['job'])].sort_values('iteration');pd.testing.assert_frame_equal(pd.DataFrame(rt),saved_trace[list(rt[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
        initial=np.zeros(26);initial[-1]=np.log(y.mean()/(1-y.mean()))
        alt=minimize(independent.independent_objective,initial,args=(x,y,.01),method=settings['method'],jac=True,options=settings['options'])
        av,ag=independent.independent_objective(alt.x,x,y,.01);gap=abs(av-value)
        pgap=float(np.max(np.abs(expit(x@alt.x[:-1]+alt.x[-1])-probability(design(x)@theta))));gnorm=float(np.max(np.abs(ag)))
        assert gap<=settings['objective_absolute_tolerance'] and pgap<=settings['training_probability_absolute_tolerance'] and gnorm<=settings['gradient_infinity_tolerance']
        solutions.append(dict(job=head['job'],method=head['method'],cutoff=head['cutoff'],seed=head['seed'],alternate_iterations=int(alt.nit),
            alternate_success=bool(alt.success),alternate_message=str(alt.message),objective_absolute_gap=gap,maximum_training_probability_gap=pgap,alternate_gradient_inf=gnorm))
        measured=independent.independent_metrics(x@theta[:-1]+theta[-1],y)
        for k,v in measured.items():near(v,train_metrics.loc[head['job'],k])
        xx=(np.maximum(new['lower'],np.minimum(new['upper'],vf.astype(float)))-mean)/sd;zz=xx@theta[:-1]+theta[-1];pp=expit(zz)
        g=predictions[predictions.method.eq(head['method'])&predictions.cutoff.eq(head['cutoff'])&predictions.seed.eq(head['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index,te);np.testing.assert_array_equal(g.actual_up,(returns[te]>0));np.testing.assert_array_equal(g.direction_up,pp>.5)
        err=max(float(np.max(np.abs(pp-g.probability.to_numpy()))),float(np.max(np.abs(zz-g.logit.to_numpy()))));assert err<1e-12;max_prediction=max(max_prediction,err)
        replays.append(dict(job=head['job'],training_rows=len(tr),validation_rows=len(te),feature_bitwise_equal=True,maximum_forecast_error=err))
    assert [learned_train,raw_train,learned_test,raw_test]==[28395,9465,783,261]
    assert len(heads)==len(solutions)==24 and len(predictions)==2871 and len(ensemble)==1827
    for filename,frame in [('model_predictions.csv',predictions),('ensemble_predictions.csv',ensemble)]:
        old=pd.read_csv(V16/'results'/filename,float_precision='round_trip');g=frame[frame.method.isin(old.method.unique())]
        keys=['method','date']+(['seed'] if 'seed' in old else [])
        pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),g[old.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-15)
    new_predictions=pd.read_csv(OUT/'new_model_predictions.csv',float_precision='round_trip');assert len(new_predictions)==1044
    for (method,date),g in new_predictions.groupby(['method','date']):
        e=ensemble[ensemble.method.eq(method)&ensemble.date.eq(date)].iloc[0];assert len(g)==(3 if method=='learned_clip' else 1)
        near(float(g.probability.mean()),e.probability,1e-15);assert bool(e.direction_up)==(e.probability>.5)
    # Reconstruct causal descriptors and independently interpret the frozen boundaries.
    thresholds={r['cutoff']:r for r in read(OUT/'state_thresholds.json')};all_tr=pd.read_csv(OUT/'training_states.csv',float_precision='round_trip');all_te=pd.read_csv(OUT/'validation_states.csv',float_precision='round_trip')
    names=['trend60','volatility20','range20','volume_change20'];state_train=0;state_test=0
    for fold in cfg()['folds']:
        tr,te=indices(obs,fold);t=thresholds[fold['cutoff']];train_desc=descriptors(bars,obs.anchor.iloc[tr].to_numpy())
        expected=dict(trend_lower=float(quantile_reference(train_desc.trend60,1/3)),trend_upper=float(quantile_reference(train_desc.trend60,2/3)),
            volatility_median=float(quantile_reference(train_desc.volatility20,.5)),range_median=float(quantile_reference(train_desc.range20,.5)),volume_median=float(quantile_reference(train_desc.volume_change20,.5)))
        for k,v in expected.items():near(v,t[k])
        assert t['train_n']==len(tr) and t['last_training_signal']==obs.date.iloc[tr].max() and t['last_training_maturity']==obs.joint_completed.iloc[tr].max()<=fold['cutoff']
        for rows,frame in [(tr,all_tr),(te,all_te)]:
            g=frame[frame.cutoff.eq(fold['cutoff'])].sort_values('row_index');np.testing.assert_array_equal(g.row_index,rows);np.testing.assert_array_equal(g.date,obs.date.iloc[rows])
            d=descriptors(bars,obs.anchor.iloc[rows].to_numpy());ref=scalar_descriptors(bars,obs.anchor.iloc[rows].to_numpy())
            np.testing.assert_allclose(d,ref,rtol=0,atol=1e-12);np.testing.assert_array_equal(g[names].to_numpy(),d.to_numpy())
            trend=['low_trend' if v<=t['trend_lower'] else 'mid_trend' if v<=t['trend_upper'] else 'high_trend' for v in d.trend60]
            vol=['low_vol' if v<=t['volatility_median'] else 'high_vol' for v in d.volatility20]
            ran=['low_range' if v<=t['range_median'] else 'high_range' for v in d.range20]
            volume=['low_volume_change' if v<=t['volume_median'] else 'high_volume_change' for v in d.volume_change20]
            for k,expected_labels in [('trend',trend),('volatility',vol),('range',ran),('volume',volume),('trend_volatility',[a+'__'+b for a,b in zip(trend,vol)])]:assert g[k].tolist()==expected_labels
        state_train+=len(tr);state_test+=len(te)
    assert state_train==9465 and state_test==261 and all_te.date.is_unique
    usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(usage)==261 and usage.previously_evaluated.all();assert set(usage.date)==set(all_te.date)
    # Recompute saved tables, then verify metric arithmetic independently of production metrics.
    tables,pairs,assessments=evaluation.compute()
    for name,table in tables.items():pd.testing.assert_frame_equal(table,pd.read_csv(OUT/f'{name}.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));independent.close_tree(assessments,read(OUT/'assessments.json'))
    metric_groups=0
    for r in tables['ensemble_metrics'].to_dict('records'):
        w=next(w for w in evaluation.windows() if w['name']==r['window']);g=evaluation.select(ensemble,w);verify_metrics(g[g.method.eq(r['method'])],r);metric_groups+=1
    for r in tables['yearly_metrics'].to_dict('records'):verify_metrics(ensemble[ensemble.method.eq(r['method'])&ensemble.year.eq(r['year'])],r);metric_groups+=1
    for r in tables['seed_metrics'].to_dict('records'):
        w=next(w for w in evaluation.windows() if w['name']==r['window']);g=evaluation.select(predictions,w);verify_metrics(g[g.method.eq(r['method'])&g.seed.eq(r['seed'])],r);metric_groups+=1
    enriched=ensemble.merge(all_te,on=['cutoff','row_index','date'],validate='many_to_one')
    for r in tables['state_metrics'].to_dict('records'):
        w=next(w for w in cfg()['windows'] if w['name']==r['window']);g=evaluation.select(enriched,w);s=g[g[r['partition']].eq(r['state'])&g.method.eq(r['method'])]
        verify_metrics(s,r);assert r['sparse']==(len(s)<20);near(len(s)/w['n'],r['sample_share']);metric_groups+=1
    assert metric_groups==258 and len(tables['state_metrics'])==168 and len(tables['reliability_bins'])==60
    for (window,partition,method),g in tables['state_metrics'].groupby(['window','partition','method']):
        w=next(w for w in cfg()['windows'] if w['name']==window);assert g.n.sum()==w['n'];near(float(g.sample_share.sum()),1.);near(float(g.error_share.sum()),1.)
    # Independent explicit circular block construction and centered contrast statistics.
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))))
        ids=np.empty((10000,n),int)
        for k in range(n):ids[:,k]=(starts[:,k//8]+k%8)%n
        e=evaluation.select(ensemble,w)
        for r in [p for p in pairs if p['window']==w['name']]:
            a=e[e.method.eq(r['candidate'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');v=independent_loss(a,r['metric'])-independent_loss(b,r['metric'])
            mean=float(v.mean());boot=v[ids].mean(axis=1);centered=(v-mean)[ids].mean(axis=1)
            near(mean,r['difference']);near(float(np.quantile(boot,.025)),r['ci95_low']);near(float(np.quantile(boot,.975)),r['ci95_high'])
            near(float((1+np.count_nonzero(np.abs(centered)>=abs(mean)))/10001),r['p'])
    order=sorted(range(12),key=lambda i:pairs[i]['p']);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(12-rank)*pairs[i]['p']));near(running,pairs[i]['holm_adjusted_p'])
    assert old_evidence()==prep['old_evidence']
    pd.DataFrame(solutions).to_csv(OUT/'independent_solver_verification.csv',index=False);pd.DataFrame(replays).to_csv(OUT/'feature_replay_verification.csv',index=False)
    files=[OUT/'independent_solver_verification.csv',OUT/'feature_replay_verification.csv']
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,previous_files_preserved=2716,new_primary_fits=24,
        independent_solutions=24,neural_states_replayed=18,neural_training_steps=0,learned_training_rows=learned_train,raw_training_rows=raw_train,
        learned_validation_rows=learned_test,raw_validation_rows=raw_test,new_probability_forecasts=1044,reused_model_forecasts=1827,
        model_records=2871,ensemble_records=1827,market_state_training_rows=state_train,market_state_validation_rows=state_test,
        metric_groups=metric_groups,state_metric_rows=168,reliability_bins=60,primary_contrasts=12,independent_holdout_dates=0,
        maximum_new_forecast_error=max_prediction,maximum_native_forecast_error=native_max,
        maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),
        maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in solutions),scipy=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)

if __name__=='__main__':main()

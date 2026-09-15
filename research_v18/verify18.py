"""Verify new feature joins, independent convex solutions and all forecasts."""
from common18 import *
import evaluate18 as evaluation
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
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    fitting=read(OUT/'training_manifest.json');assert fitting['all_converged'] and fitting['new_primary_fits']==30 and not fitting['validation_scoring_during_fit']
    assert fitting['neural_forward_passes']==fitting['neural_training_steps']==0
    prior=read(V17/'results/verification.json');assert prior['status']=='PASS' and prior['neural_states_replayed']==18
    obs,price,returns=data();bars=base16.raw_bars(price);heads=read(OUT/'heads.json');parents={h['job']:h for h in read(OUT/'parent_heads.json')}
    contexts={c['cutoff']:c for c in read(OUT/'market_contexts.json')};val_refs={r['job']:r for r in read(V16/'results/validation_features.json')}
    sources={h['job']:h for h in read(V17/'results/source_heads.json')};market_train={};market_test={};transforms=[];duplicate_rows=[]
    for fold in cfg()['folds']:
        cutoff=fold['cutoff'];tr,te=indices(obs,fold);ref=contexts[cutoff];c=load_npz(ref);np.testing.assert_array_equal(c['row_index'],tr);np.testing.assert_array_equal(ref['validation_rows'],te)
        np.testing.assert_array_equal(c['direction'],(returns[tr]>0).astype(float))
        f=market_rows('training',cutoff,tr);v=market_rows('validation',cutoff,te)
        for rows,actual in [(tr,f),(te,v)]:
            direct=descriptors(bars,obs.anchor.iloc[rows].to_numpy());scalar=scalar_descriptors(bars,obs.anchor.iloc[rows].to_numpy())
            np.testing.assert_array_equal(actual,direct.to_numpy());np.testing.assert_allclose(actual,scalar.to_numpy(),rtol=0,atol=1e-12)
        np.testing.assert_array_equal(f,c['features'])
        for q,name in [(.01,'lower'),(.99,'upper')]:np.testing.assert_allclose(audit.quantile_reference(f,q),c[name],rtol=0,atol=1e-12)
        clipped=np.maximum(c['lower'],np.minimum(c['upper'],f));mean=clipped.mean(axis=0);sd=np.maximum(np.sqrt(np.square(clipped-mean).mean(axis=0)),1e-6)
        for actual,expected in [(clipped,c['clipped_features']),(mean,c['mean']),(sd,c['sd'])]:np.testing.assert_array_equal(actual,expected)
        x=(clipped-mean)/sd;xx=(np.maximum(c['lower'],np.minimum(c['upper'],v))-mean)/sd;np.testing.assert_array_equal(x,c['standardized'])
        market_train[cutoff]=x;market_test[cutoff]=xx
        parent=next(h for h in parents.values() if h['cutoff']==cutoff and h['method']=='raw25_clip');raw=load_npz(sources[parent['source_job']]);norm=load_npz(parent)
        raw_val=load_npz(val_refs[parent['source_job']]);norm_val=(np.maximum(norm['lower'],np.minimum(norm['upper'],raw_val['features']))-norm['mean'])/norm['sd']
        # The raw family is deterministic; also reconstruct its source formulas directly.
        np.testing.assert_allclose(raw['features'],base16.scalar_features(bars,obs.anchor.iloc[tr].to_numpy()),rtol=0,atol=1e-14)
        np.testing.assert_allclose(raw_val['features'],base16.scalar_features(bars,obs.anchor.iloc[te].to_numpy()),rtol=0,atol=1e-14)
        for mi,ri in DUPLICATES.items():
            errors=[float(np.max(np.abs(f[:,mi]-raw['features'][:,ri]))),float(np.max(np.abs(v[:,mi]-raw_val['features'][:,ri]))),
                float(np.max(np.abs(x[:,mi]-norm['standardized'][:,ri]))),float(np.max(np.abs(xx[:,mi]-norm_val[:,ri])))]
            assert max(errors[:2])<1e-14 and max(errors[2:])<1e-12
            duplicate_rows.append(dict(cutoff=cutoff,market_column=MARKET_NAMES[mi],raw_column=base16.NAMES[ri],
                training_raw_error=errors[0],validation_raw_error=errors[1],training_standardized_error=errors[2],validation_standardized_error=errors[3],omitted_from_raw_append=True))
        for j,name in enumerate(MARKET_NAMES):
            transforms.append(dict(cutoff=cutoff,feature=name,train_n=len(tr),validation_n=len(te),lower=float(c['lower'][j]),upper=float(c['upper'][j]),
                training_clipped_mean=float(mean[j]),training_clipped_sd=float(sd[j]),validation_standardized_mean=float(xx[:,j].mean()),
                validation_lower_fraction=float((v[:,j]<c['lower'][j]).mean()),validation_upper_fraction=float((v[:,j]>c['upper'][j]).mean())))
    predictions=pd.read_csv(OUT/'model_predictions.csv',float_precision='round_trip');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv',float_precision='round_trip')
    traces=pd.read_csv(OUT/'solver_trace.csv');training_metrics=pd.read_csv(OUT/'training_metrics.csv').set_index('job');coefficients=pd.read_csv(OUT/'coefficients.csv')
    components=pd.read_csv(OUT/'logit_components.csv',float_precision='round_trip');summaries=pd.read_csv(OUT/'component_summary.csv');settings=cfg()['independent_solver']
    solutions=[];replays=[];total_train=0;total_test=0;maximum_error=0.
    for h in heads:
        fold=next(f for f in cfg()['folds'] if f['cutoff']==h['cutoff']);tr,te=indices(obs,fold);d=load_npz(h);np.testing.assert_array_equal(d['row_index'],tr)
        y=(returns[tr]>0).astype(float);np.testing.assert_array_equal(d['direction'],y);market=market_train[h['cutoff']][:,h['market_indices']];future=market_test[h['cutoff']][:,h['market_indices']]
        if h['parent_job'] is None:rep=np.empty((len(tr),0));testrep=np.empty((len(te),0))
        else:
            parent=parents[h['parent_job']];old=load_npz(parent);rep=old['standardized'];f=load_npz(val_refs[parent['source_job']])['features'].astype(float)
            testrep=(np.maximum(old['lower'],np.minimum(old['upper'],f))-old['mean'])/old['sd']
        x=np.concatenate((rep,market),axis=1);xx=np.concatenate((testrep,future),axis=1)
        assert x.shape==(len(tr),h['dimensions']);np.testing.assert_array_equal(x,d['standardized'])
        variant=next(r for r in cfg()['variants'] if r['method']==h['method']);assert h['market_indices']==variant['market_indices'] and h['dimensions']==variant['dimensions']
        theta=np.asarray(h['coefficients']);assert len(theta)==h['dimensions']+1
        value,grad,hess=objective(theta,design(x),y,.01);assert np.max(np.abs(grad))<=1e-9 and np.linalg.eigvalsh(hess).min()>0
        audit.near(value,h['objective'],1e-14);audit.near(float(np.max(np.abs(grad))),h['gradient_inf'],1e-14)
        audit.near(float(np.linalg.eigvalsh(hess).min()),h['hessian_min_eigenvalue'])
        replay,trace=fit_newton(x,y,cfg()['probe']);np.testing.assert_allclose(theta,replay,rtol=0,atol=1e-12);assert trace[-1]['iteration']==h['iterations']
        saved=traces[traces.job.eq(h['job'])].sort_values('iteration');pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
        if h['parent_job'] is not None:
            t=np.asarray(parent['coefficients']);nested=np.r_[t[:-1],np.zeros(len(h['market_indices'])),t[-1]]
            nested_v,_=independent.independent_objective(nested,x,y,.01);audit.near(nested_v,parent['objective']);assert value<=nested_v+1e-12
        initial=np.zeros(len(theta));initial[-1]=np.log(y.mean()/(1-y.mean()))
        alt=minimize(independent.independent_objective,initial,args=(x,y,.01),method=settings['method'],jac=True,options=settings['options'])
        av,ag=independent.independent_objective(alt.x,x,y,.01);gap=abs(av-value);pgap=float(np.max(np.abs(expit(x@alt.x[:-1]+alt.x[-1])-probability(design(x)@theta))));gn=float(np.max(np.abs(ag)))
        assert gap<=settings['objective_absolute_tolerance'] and pgap<=settings['training_probability_absolute_tolerance'] and gn<=settings['gradient_infinity_tolerance']
        solutions.append(dict(job=h['job'],method=h['method'],cutoff=h['cutoff'],seed=h['seed'],alternate_iterations=int(alt.nit),alternate_success=bool(alt.success),
            alternate_message=str(alt.message),objective_absolute_gap=gap,maximum_training_probability_gap=pgap,alternate_gradient_inf=gn))
        for k,v in independent.independent_metrics(x@theta[:-1]+theta[-1],y).items():audit.near(v,training_metrics.loc[h['job'],k])
        coef=coefficients[coefficients.job.eq(h['job'])].sort_values('coordinate');np.testing.assert_allclose(coef.coefficient,theta,rtol=0,atol=1e-15)
        assert int(coef.block.eq('market').sum())==len(h['market_indices']) and int(coef.block.eq('representation').sum())==h['representation_dimensions']
        z=xx@theta[:-1]+theta[-1];p=expit(z);g=predictions[predictions.method.eq(h['method'])&predictions.cutoff.eq(h['cutoff'])&predictions.seed.eq(h['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index,te);np.testing.assert_array_equal(g.date,obs.date.iloc[te]);np.testing.assert_array_equal(g.actual_up,returns[te]>0);np.testing.assert_array_equal(g.direction_up,p>.5)
        err=max(float(np.max(np.abs(z-g.logit.to_numpy()))),float(np.max(np.abs(p-g.probability.to_numpy()))));assert err<1e-12;maximum_error=max(maximum_error,err)
        width=h['representation_dimensions'];comp=components[components.job.eq(h['job'])].sort_values('row_index')
        np.testing.assert_array_equal(comp.row_index,te);np.testing.assert_allclose(comp.representation_logit,xx[:,:width]@theta[:width],rtol=0,atol=1e-12)
        np.testing.assert_allclose(comp.market_logit,xx[:,width:]@theta[width:-1],rtol=0,atol=1e-12);np.testing.assert_allclose(comp.logit,comp.representation_logit+comp.market_logit+comp.intercept,rtol=0,atol=1e-12)
        for split,values in [('training',x),('validation',xx)]:
            a=values[:,:width]@theta[:width];b=values[:,width:]@theta[width:-1];saved=summaries[summaries.job.eq(h['job'])&summaries.split.eq(split)].iloc[0];assert saved.n==len(values)
            expected=dict(representation_mean=float(a.mean()),representation_std=float(a.std()),market_mean=float(b.mean()),market_std=float(b.std()),
                component_correlation=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None,intercept=float(theta[-1]),total_logit_mean=float((a+b+theta[-1]).mean()),total_logit_std=float((a+b).std()))
            for k,v in expected.items():audit.near(v,saved[k])
        total_train+=len(tr);total_test+=len(te);replays.append(dict(job=h['job'],dimensions=h['dimensions'],training_rows=len(tr),validation_rows=len(te),maximum_forecast_error=err))
    assert len(heads)==30 and total_train==47325 and total_test==1305 and len(coefficients)==732 and len(summaries)==60
    assert len(predictions)==4176 and len(ensemble)==2610 and len(components)==1305
    for filename,frame in [('model_predictions.csv',predictions),('ensemble_predictions.csv',ensemble)]:
        old=pd.read_csv(V17/'results'/filename,float_precision='round_trip');g=frame[frame.method.isin(old.method.unique())];keys=['method','date']+(['seed'] if 'seed' in old else [])
        pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),g[old.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=0,atol=1e-15)
    new=pd.read_csv(OUT/'new_model_predictions.csv',float_precision='round_trip');assert len(new)==1305
    for (method,date),g in new.groupby(['method','date']):
        e=ensemble[ensemble.method.eq(method)&ensemble.date.eq(date)].iloc[0];assert len(g)==(3 if method=='learned_market' else 1)
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
        audit.verify_metrics(s,r);assert r['sparse']==(len(s)<20);metric_groups+=1
    assert metric_groups==366 and len(tables['state_metrics'])==240 and len(tables['reliability_bins'])==90
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))));ids=np.empty((10000,n),int)
        for k in range(n):ids[:,k]=(starts[:,k//8]+k%8)%n
        e=evaluation.select(ensemble,w)
        for r in [p for p in pairs if p['window']==w['name']]:
            a=e[e.method.eq(r['candidate'])].sort_values('date');b=e[e.method.eq(r['reference'])].sort_values('date');v=audit.independent_loss(a,r['metric'])-audit.independent_loss(b,r['metric'])
            mean=float(v.mean());boot=v[ids].mean(axis=1);centered=(v-mean)[ids].mean(axis=1)
            for actual,key in [(mean,'difference'),(float(np.quantile(boot,.025)),'ci95_low'),(float(np.quantile(boot,.975)),'ci95_high'),(float((1+np.count_nonzero(np.abs(centered)>=abs(mean)))/10001),'p')]:audit.near(actual,r[key])
    assert len(pairs)==14;order=sorted(range(14),key=lambda i:pairs[i]['p']);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(14-rank)*pairs[i]['p']));audit.near(running,pairs[i]['holm_adjusted_p'])
    usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(usage)==len(states)==261 and usage.previously_evaluated.all() and set(usage.date)==set(states.date)
    assert old_evidence()==prep['old_evidence'];files=[]
    for name,rows in [('independent_solver_verification',solutions),('feature_join_verification',replays),('duplicate_column_verification',duplicate_rows),('market_transform_audit',transforms)]:
        path=OUT/f'{name}.csv';pd.DataFrame(rows).to_csv(path,index=False);files.append(path)
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,previous_files_preserved=2795,new_primary_fits=30,independent_solutions=30,
        neural_states_with_inherited_round17_replay=18,new_neural_forward_passes=0,neural_training_steps=0,training_feature_rows=47325,market_training_rows=9465,
        market_validation_rows=261,omitted_duplicate_checks=18,new_probability_forecasts=1305,reused_model_records=2871,model_records=4176,ensemble_records=2610,
        metric_groups=366,state_metric_rows=240,reliability_bins=90,primary_contrasts=14,independent_holdout_dates=0,maximum_new_forecast_error=maximum_error,
        maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),
        maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in solutions),scipy=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)

if __name__=='__main__':main()

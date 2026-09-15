from common16 import *
import evaluate16 as evaluation
import verify15 as independent
import scipy
from scipy.optimize import minimize
from scipy.special import expit


def main():
    legacy.initialize();start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists(),'Preserve verification'
    last=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=last
    for phase in ['training','scoring','evaluation']:
        run=check_phase(phase);assert last<=run['started_utc']<run['finished_utc'];last=run['finished_utc']
        for key in ['protocol_sha256','source_sha256','input_sha256']:assert run[key]==prep[key]
    training=read(OUT/'training_manifest.json');assert training['all_converged'] and not training['validation_scoring_during_fit'] and training['neural_training_steps']==0
    obs,price,returns=data();bars=raw_bars(price);heads=read(OUT/'heads.json');assert len(heads)==24
    predictions=pd.read_csv(OUT/'model_predictions.csv');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv')
    native=pd.read_csv(OUT/'native_seed_reference.csv');retained=pd.read_csv(OUT/'reused_probe_seed_reference.csv')
    old_heads={(h['cutoff'],h['seed']):h for h in read(V15/'results/heads.json') if h['family']=='probe_mse'}
    features={r['job']:r for r in read(OUT/'validation_features.json')}
    train_metrics=pd.read_csv(OUT/'training_metrics.csv').set_index('job');diagnostics=pd.read_csv(OUT/'feature_diagnostics.csv').set_index('job')
    traces=pd.read_csv(OUT/'solver_trace.csv');solutions=[];replays=[];settings=cfg()['independent_solver']
    learned_train=0;raw_train=0;learned_test=0;raw_test=0;native_count=0;max_error=0.
    for head in heads:
        fold=next(f for f in cfg()['folds'] if f['cutoff']==head['cutoff']);tr,te=indices(obs,fold);d=load_features(head)
        y=(returns[tr]>0).astype(float);np.testing.assert_array_equal(d['row_index'],tr);np.testing.assert_array_equal(d['direction'],y)
        if head['kind']=='learned':
            model,state,values,f,_=extract_learned(head,tr);assert state['train_n']==len(tr)
            learned_train+=len(tr);del values
        else:
            f=raw_features(bars,obs.anchor.iloc[tr].to_numpy());reference=scalar_features(bars,obs.anchor.iloc[tr].to_numpy())
            np.testing.assert_allclose(f,reference,rtol=0,atol=1e-14);raw_train+=len(tr)
        np.testing.assert_array_equal(f,d['features'])
        mean=f.astype(float).mean(axis=0);sd=np.maximum(f.astype(float).std(axis=0),cfg()['standardization']['sd_floor']);x=(f.astype(float)-mean)/sd
        for actual,expected in [(mean,d['mean']),(sd,d['sd']),(x,d['standardized'])]:np.testing.assert_array_equal(actual,expected)
        theta=np.asarray(head['coefficients']);value,g,h=objective(theta,design(x),y,.01)
        assert np.max(np.abs(g))<=1e-9 and abs(value-head['objective'])<1e-14 and np.linalg.eigvalsh(h).min()>0
        assert abs(np.linalg.eigvalsh(h).min()-head['hessian_min_eigenvalue'])<1e-12
        replay,trace=fit_newton(x,y,cfg()['probe']);np.testing.assert_allclose(theta,replay,rtol=0,atol=1e-12)
        assert head['iterations']==trace[-1]['iteration']
        if not head['reused']:
            saved=traces[traces.job.eq(head['job'])].sort_values('iteration')
            pd.testing.assert_frame_equal(pd.DataFrame(trace),saved[list(trace[0])].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
        else:
            old=old_heads[(head['cutoff'],head['seed'])]
            np.testing.assert_array_equal(theta,old['coefficients'])
            assert head['cache_sha256']==old['feature_sha256'] and head['sha256']==old['sha256']
        initial=np.zeros(26);initial[-1]=np.log(y.mean()/(1-y.mean()))
        alt=minimize(independent.independent_objective,initial,args=(x,y,.01),method=settings['method'],jac=True,options=settings['options'])
        alt_value,alt_g=independent.independent_objective(alt.x,x,y,.01)
        gap=abs(alt_value-value);p_gap=float(np.max(np.abs(expit(x@alt.x[:-1]+alt.x[-1])-probability(design(x)@theta))));norm=float(np.max(np.abs(alt_g)))
        assert gap<=settings['objective_absolute_tolerance'] and p_gap<=settings['training_probability_absolute_tolerance'] and norm<=settings['gradient_infinity_tolerance']
        solutions.append(dict(job=head['job'],method=head['method'],cutoff=head['cutoff'],seed=head['seed'],reused=head['reused'],
            newton_iterations=head['iterations'],alternate_iterations=int(alt.nit),alternate_success=bool(alt.success),alternate_message=str(alt.message),
            objective_absolute_gap=gap,maximum_training_probability_gap=p_gap,alternate_gradient_inf=norm))
        measured=independent.independent_metrics(x@theta[:-1]+theta[-1],y);saved=train_metrics.loc[head['job']]
        for key,v in measured.items():assert abs(v-saved[key])<1e-12,(head['job'],key)
        assert abs(value-saved.objective)<1e-14 and abs(saved.constant_accuracy-float(((y.mean()>.5)==(y>0)).mean()))<1e-14
        diag=diagnostics.loc[head['job']];sv=np.linalg.svd(x,compute_uv=False)
        assert diag.numerical_rank==int((sv>1e-8).sum()) and diag.floored_coordinates==int((f.astype(float).std(axis=0)<1e-6).sum())
        assert abs(diag.minimum_singular_value-sv.min())<1e-12
        if head['kind']=='learned':
            values=legacy.batch_tensors(te);vf,_=previous.extract_features(model,values);learned_test+=len(te)
            original=previous.archival_native_output(model,values).astype(float)
            score=original*state['scales']['returns_sd']+state['scales']['returns_mean']
            old=native[native.cutoff.eq(head['cutoff'])&native.seed.eq(head['seed'])].sort_values('row_index')
            error=float(np.max(np.abs(score-old.score.to_numpy())));assert error<1e-10;max_error=max(max_error,error);native_count+=len(te)
            actual=predictions[predictions.method.eq('native_mse')&predictions.cutoff.eq(head['cutoff'])&predictions.seed.eq(head['seed'])].sort_values('row_index')
            pd.testing.assert_frame_equal(old.reset_index(drop=True),actual[old.columns].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
            assert object_hash(model.state_dict())==head['model_sha256'] and sha(PROJECT/head['project_file'])==head['sha256']
            assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
            del model,values;torch.cuda.empty_cache()
        else:
            vf=raw_features(bars,obs.anchor.iloc[te].to_numpy());reference=scalar_features(bars,obs.anchor.iloc[te].to_numpy())
            np.testing.assert_allclose(vf,reference,rtol=0,atol=1e-14);raw_test+=len(te)
        cache=features[head['job']];assert sha(PROJECT/cache['cache_file'])==cache['cache_sha256']
        with np.load(PROJECT/cache['cache_file']) as saved_features:
            np.testing.assert_array_equal(vf,saved_features['features']);np.testing.assert_array_equal(te,saved_features['row_index'])
        z=((vf.astype(float)-mean)/sd)@theta[:-1]+theta[-1];p=expit(z)
        predicted=predictions[predictions.method.eq(head['method'])&predictions.cutoff.eq(head['cutoff'])&predictions.seed.eq(head['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(predicted.row_index,te);np.testing.assert_array_equal(predicted.actual_up,returns[te]>0)
        np.testing.assert_allclose(predicted.actual_return,returns[te],rtol=0,atol=1e-14)
        np.testing.assert_array_equal(predicted.direction_up,p>.5)
        error=max(float(np.max(np.abs(p-predicted.probability.to_numpy()))),float(np.max(np.abs(z-predicted.logit.to_numpy()))))
        assert error<1e-10;max_error=max(max_error,error)
        if head['reused']:
            old=retained[retained.cutoff.eq(head['cutoff'])&retained.seed.eq(head['seed'])].sort_values('row_index')
            pd.testing.assert_frame_equal(old.reset_index(drop=True),predicted[old.columns].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
        replays.append(dict(job=head['job'],kind=head['kind'],train_n=len(tr),test_n=len(te),features_bitwise_equal=True,scaling_bitwise_equal=True,maximum_forecast_error=error))
        print(f"Feature and independent-solver replay {len(replays)}/24: {head['job']} PASS",flush=True)
    assert learned_train==28395 and raw_train==9465 and learned_test==native_count==783 and raw_test==261
    assert len(predictions)==1827 and len(ensemble)==1305 and len(solutions)==24
    for method,g in ensemble.groupby('method'):
        g=g.sort_values('date');assert len(g)==261 and g.date.is_unique
        if method in ['native_mse','learned_probe','raw25_probe']:
            source=predictions[predictions.method.eq(method)]
            assert source.groupby('date').size().eq(1 if method=='raw25_probe' else 3).all()
            field='score' if method=='native_mse' else 'probability';expected=source.groupby('date')[field].mean()
            np.testing.assert_allclose(g[field],expected,rtol=0,atol=1e-14)
        else:
            expected=g.training_frequency.to_numpy() if method=='training_frequency' else np.full(261,.5)
            np.testing.assert_allclose(g.probability,expected,rtol=0,atol=1e-15)
        if method=='native_mse':assert g.probability.isna().all();np.testing.assert_array_equal(g.direction_up,g.score>0)
        else:np.testing.assert_array_equal(g.direction_up,g.probability>.5)
    # Later reference/learned ensembles stay numerically equal to round15.
    old_ensemble=pd.read_csv(V15/'results/all_ensemble_predictions.csv')
    for method,prior_method in [('native_mse','archived20'),('learned_probe','probe_mse'),('training_frequency','training_frequency'),('neutral_50','neutral_50')]:
        a=ensemble[ensemble.method.eq(method)&ensemble.year.ge(2018)].sort_values('date');b=old_ensemble[old_ensemble.method.eq(prior_method)].sort_values('date')
        assert a.date.tolist()==b.date.tolist();np.testing.assert_allclose(a.score,b.score,rtol=0,atol=1e-14);np.testing.assert_array_equal(a.direction_up,b.direction_up)
    for w in evaluation.windows():
        e=evaluation.select(ensemble,w);m=evaluation.select(predictions,w)
        table=pd.read_csv(OUT/'ensemble_metrics.csv');independent.verify_metric_table(e,table[table.window.eq(w['name'])],['method'])
        table=pd.read_csv(OUT/'seed_metrics.csv');independent.verify_metric_table(m,table[table.window.eq(w['name'])],['method','seed'])
    independent.verify_metric_table(ensemble,pd.read_csv(OUT/'yearly_metrics.csv'),['method','year'])
    tables,pairs,assessments=evaluation.compute()
    for name,table in tables.items():pd.testing.assert_frame_equal(table,pd.read_csv(OUT/f'{name}.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
    independent.close_tree(pairs,read(OUT/'primary_comparisons.json'));independent.close_tree(assessments,read(OUT/'assessments.json'))
    bins=pd.read_csv(OUT/'reliability_bins.csv');assert len(bins)==40
    for r in bins.itertuples():
        w=next(w for w in cfg()['windows'] if w['name']==r.window);g=evaluation.select(ensemble[ensemble.method.eq(r.method)],w)
        g=g[g.probability.ge(r.lower)&(g.probability.le(r.upper) if r.bin_index==4 else g.probability.lt(r.upper))]
        assert len(g)==r.n
        if len(g):assert abs(g.probability.mean()-r.mean_probability)<1e-14 and abs(g.actual_up.mean()-r.observed_frequency)<1e-14
        else:assert pd.isna(r.mean_probability) and pd.isna(r.observed_frequency)
    usage=pd.read_csv(OUT/'historical_date_usage.csv');assert len(usage)==261 and usage.previously_evaluated.all()
    historical=read(OUT/'historical_usage_audit.json');assert historical['independent_holdout_dates']==0
    source_paths={'round13_rolling':V13/'results/rolling_seed_predictions.csv','round15_validation':V15/'results/all_seed_predictions.csv',
        'round2_validation':PROJECT/'research_v2/results/validation_predictions.csv','round3_validation':PROJECT/'research_v3/results/validation_grid_predictions.csv',
        'round1_later_development':PROJECT/'research/results/predictions.csv'}
    for name,path in source_paths.items():
        source=pd.read_csv(path);np.testing.assert_array_equal(usage['seen_'+name],usage.date.isin(set(source.date)))
    assert old_evidence()==prep['old_evidence']
    pd.DataFrame(solutions).to_csv(OUT/'independent_solver_verification.csv',index=False);pd.DataFrame(replays).to_csv(OUT/'feature_replay_verification.csv',index=False)
    files=[OUT/'independent_solver_verification.csv',OUT/'feature_replay_verification.csv']
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,neural_states_replayed=18,learned_training_rows=learned_train,
        raw_training_rows=raw_train,learned_validation_rows=learned_test,raw_validation_rows=raw_test,native_forecasts=783,
        model_forecasts=1827,independent_solutions=24,retained_heads=9,retained_late_probabilities=423,metric_groups=63,reliability_bins=40,
        paired_contrasts=12,historical_dates_audited=261,independent_holdout_dates=0,previous_files_preserved=2618,
        max_forecast_error=max_error,maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in solutions),
        maximum_alternate_probability_gap=max(r['maximum_training_probability_gap'] for r in solutions),
        maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in solutions),scipy=scipy.__version__,
        protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','artifacts']},indent=2),flush=True)


if __name__=='__main__':main()

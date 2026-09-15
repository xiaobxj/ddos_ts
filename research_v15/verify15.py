"""Replay frozen representations and forecasts; independently verify convex solutions."""
from common15 import *
import evaluate15 as evaluation
import scipy
from scipy.optimize import minimize
from scipy.special import expit


def close_tree(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:close_tree(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close_tree(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)


def independent_objective(theta,x,y,lam):
    z=x@theta[:-1]+theta[-1]
    loss=float(np.where(y>0,np.logaddexp(0.,-z),np.logaddexp(0.,z)).mean())
    error=expit(z)-y
    grad=np.r_[np.mean(x*error[:,None],axis=0)+lam*theta[:-1],error.mean()]
    return loss+lam/2*float(np.square(theta[:-1]).sum()),grad


def independent_metrics(z,y):
    p=expit(z);up=p>.5;label=y>0
    pair=p[label,None]-p[~label][None,:]
    return dict(log_loss=float(np.where(label,np.logaddexp(0.,-z),np.logaddexp(0.,z)).mean()),
        brier=float(np.square(p-y).mean()),accuracy=float((up==label).mean()),
        auroc=float(((pair>0)+.5*(pair==0)).mean()),mean_probability=float(p.mean()),probability_std=float(p.std()))


def verify_metric_table(source,table,keys):
    for r in table.to_dict('records'):
        g=source
        for k in keys:g=g[g[k].eq(r[k])]
        y=g.actual_up.to_numpy(bool);p=g.direction_up.to_numpy(bool);score=g.score.to_numpy()
        assert len(g)==r['n'] and int((p==y).sum())==r['correct_directions']
        assert abs(float((p==y).mean())-r['accuracy'])<1e-14
        assert abs(float((p!=y).mean())-r['direction_error'])<1e-14
        assert abs(float((p[y].mean()+(~p[~y]).mean())/2)-r['balanced_accuracy'])<1e-14
        assert abs(p.mean()-r['predicted_up_fraction'])<1e-14 and abs(y.mean()-r['observed_up_fraction'])<1e-14
        for k,v in [('tp',p&y),('tn',~p&~y),('fp',p&~y),('fn',~p&y)]:assert r[k]==int(v.sum())
        pair=score[y,None]-score[~y][None,:]
        assert abs(float(((pair>0)+.5*(pair==0)).mean())-r['auroc'])<1e-14
        if g.probability.notna().all():
            q=g.probability.to_numpy();bounded=np.minimum(1-1e-12,np.maximum(1e-12,q))
            assert np.all((q>=0)&(q<=1))
            brier=float(np.square(q-y.astype(float)).mean())
            nll=float(-np.where(y,np.log(bounded),np.log(1-bounded)).mean())
            assert abs(brier-r['brier'])<1e-14 and abs(nll-r['log_loss'])<1e-14
            assert abs(q.mean()-r['mean_probability'])<1e-14 and abs(q.std()-r['probability_std'])<1e-14
            assert abs(q.mean()-y.mean()-r['calibration_gap'])<1e-14
            assert int(((q<1e-12)|(q>1-1e-12)).sum())==r['clipped_probabilities']
        else:assert g.probability.isna().all() and pd.isna(r['brier']) and pd.isna(r['log_loss'])


def main():
    legacy.initialize();start=time.time();prep=check_frozen();assert not (OUT/'verification.json').exists(),'Preserve verification'
    finish=read(OUT/'contract_verification.json')['completed_utc'];assert prep['finished_utc']<=finish
    for phase in ['extraction','fitting','scoring','evaluation']:
        run=check_artifacts(phase);assert finish<=run['started_utc']<run['finished_utc'];finish=run['finished_utc']
        assert run['protocol_sha256']==prep['protocol_sha256'] and run['source_sha256']==prep['source_sha256']
        assert run['input_sha256']==prep['input_sha256']
    fitting=read(OUT/'fitting_manifest.json');assert fitting['all_converged'] and not fitting['validation_used'] and fitting['neural_training_steps']==0
    assert fitting['preflight_coefficients_bitwise_equal']
    heads=read(OUT/'heads.json');assert len(heads)==18
    preflight={identity(h):h for h in read(PREFLIGHT/'results/heads.json')}
    for head in heads:np.testing.assert_array_equal(head['coefficients'],preflight[identity(head)]['coefficients'])
    meta=read(OUT/'training_metadata.json');obs=pd.read_csv(OUT/'observation_table.csv')
    training=pd.read_csv(OUT/'training_metrics.csv');trace=pd.read_csv(OUT/'solver_trace.csv')
    diagnostics=pd.read_csv(OUT/'feature_diagnostics.csv')
    predictions=pd.read_csv(OUT/'probe_seed_predictions.csv');archived=pd.read_csv(OUT/'archived_seed_predictions.csv')
    validation={identity(r):r for r in read(OUT/'validation_features.json')}
    independent=[];feature_checks=[];max_prediction=0.;train_count=0;val_count=0;native_count=0
    lam=cfg()['probe']['l2_lambda'];settings=cfg()['independent_solver']
    with np.load(V5/'cache/targets.npz') as raw:raw_labels=(raw['returns']>0).astype(float)
    for head in heads:
        cutoff=head['cutoff'];number=head['seed'];family=head['family'];d=load_features(head)
        model,state=load_backbone(head);values,tr,y=training_data(cutoff);f,native=extract_features(model,values)
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy()))
        np.testing.assert_array_equal(tr,d['row_index']);np.testing.assert_array_equal(y,raw_labels[tr]);np.testing.assert_array_equal(y,d['direction'])
        np.testing.assert_array_equal(f,d['features']);np.testing.assert_array_equal(native,d['native_output'])
        mean=np.mean(f.astype(float),axis=0);sd=np.maximum(np.std(f.astype(float),axis=0),cfg()['standardization']['sd_floor'])
        x=(f.astype(float)-mean)/sd
        for measured,saved in [(mean,d['mean']),(sd,d['sd']),(x,d['standardized'])]:np.testing.assert_array_equal(measured,saved)
        np.testing.assert_array_equal(d['native_weights'],model.return_head.weight.detach().cpu().numpy().astype(float).flatten())
        assert float(d['native_bias'])==float(model.return_head.bias.detach().cpu()[0])
        r=diagnostics[diagnostics.family.eq(family)&diagnostics.cutoff.eq(cutoff)&diagnostics.seed.eq(number)].iloc[0]
        singular=np.linalg.svd(x,compute_uv=False)
        assert r.train_n==len(y) and r.numerical_rank==int((singular>1e-8).sum()) and r.unchanged_model_sha256==head['model_sha256']
        assert r.floored_coordinates==int((f.astype(float).std(axis=0)<1e-6).sum())
        np.testing.assert_allclose([r.minimum_singular_value,r.maximum_singular_value],[singular.min(),singular.max()],rtol=1e-12,atol=1e-12)
        theta=np.asarray(head['coefficients']);a=design(x)
        value,g,h=objective(theta,a,y,lam);assert np.max(np.abs(g))<=cfg()['probe']['gradient_infinity_tolerance']
        assert abs(value-head['objective'])<1e-14 and abs(np.linalg.eigvalsh(h).min()-head['hessian_min_eigenvalue'])<1e-12
        assert np.linalg.eigvalsh(h).min()>0 and head['l2_lambda']==lam
        replay,replay_trace=fit_newton(x,y,cfg()['probe']);np.testing.assert_allclose(theta,replay,rtol=0,atol=1e-12)
        saved_trace=trace[trace.family.eq(family)&trace.cutoff.eq(cutoff)&trace.seed.eq(number)].sort_values('iteration')
        pd.testing.assert_frame_equal(pd.DataFrame(replay_trace).reset_index(drop=True),saved_trace[list(replay_trace[0])].reset_index(drop=True),
            check_dtype=False,rtol=1e-12,atol=1e-14)
        assert head['iterations']==replay_trace[-1]['iteration'] and np.all(np.diff(saved_trace.objective)<=1e-13)
        initial=np.zeros(26);initial[-1]=np.log(y.mean()/(1-y.mean()))
        alternate=minimize(independent_objective,initial,args=(x,y,lam),method=settings['method'],jac=True,options=settings['options'])
        alt_value,alt_gradient=independent_objective(alternate.x,x,y,lam)
        objective_gap=abs(alt_value-value);probability_gap=float(np.max(np.abs(expit(x@alternate.x[:-1]+alternate.x[-1])-probability(a@theta))))
        alt_norm=float(np.max(np.abs(alt_gradient)))
        assert objective_gap<=settings['objective_absolute_tolerance'],(identity(head),objective_gap)
        assert probability_gap<=settings['training_probability_absolute_tolerance'],(identity(head),probability_gap)
        assert alt_norm<=settings['gradient_infinity_tolerance'],(identity(head),alt_norm)
        independent.append(dict(family=family,cutoff=cutoff,seed=number,primary_iterations=head['iterations'],alternate_iterations=int(alternate.nit),
            alternate_success=bool(alternate.success),alternate_message=str(alternate.message),primary_gradient_inf=float(np.max(np.abs(g))),
            alternate_gradient_inf=alt_norm,objective_absolute_gap=objective_gap,maximum_training_probability_gap=probability_gap))
        summary=training[training.family.eq(family)&training.cutoff.eq(cutoff)&training.seed.eq(number)].iloc[0]
        measured=independent_metrics(x@theta[:-1]+theta[-1],y)
        for key,v in measured.items():assert abs(v-summary[key])<1e-12,(identity(head),key)
        assert abs(value-summary.objective)<1e-14 and abs(lam/2*np.square(theta[:-1]).sum()-summary.penalty)<1e-14
        constant=independent_metrics(np.full(len(y),initial[-1]),y)
        assert abs(constant['log_loss']-summary.constant_log_loss)<1e-14 and abs(constant['brier']-summary.constant_brier)<1e-14
        assert value<=constant['log_loss']+1e-12
        if head['source']=='direction_bce':
            native_theta=np.r_[d['native_weights']*sd,float(d['native_bias'])+d['native_weights']@mean]
            native_value,_=independent_objective(native_theta,x,y,lam)
            assert abs(native_value-summary.native_ridge_objective)<1e-12 and value<=native_value+1e-12
            native_m=independent_metrics(native.astype(float),y)
            for key in ['log_loss','brier','probability_std','accuracy']:assert abs(native_m[key]-summary['native_'+key])<1e-12
        else:
            scales=meta[cutoff]['scales'];native_returns=native.astype(float)*scales['returns_sd']+scales['returns_mean']
            assert abs(((native_returns>0)==(y>0)).mean()-summary.native_accuracy)<1e-14
            assert pd.isna(summary.native_log_loss) and pd.isna(summary.native_brier)
        train_count+=len(y);del values
        testing,te=previous.load_validation(cutoff);vf,vnative=extract_features(model,testing)
        original_native=archival_native_output(model,testing)
        fold=next(v for v in cfg()['folds'] if v['cutoff']==cutoff)
        expected=np.flatnonzero((obs.date.gt(cutoff)&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')).to_numpy())
        np.testing.assert_array_equal(te,expected);assert not len(np.intersect1d(tr,te))
        vr=validation[identity(head)];assert sha(ROOT/vr['feature_file'])==vr['sha256']
        with np.load(ROOT/vr['feature_file']) as saved:
            np.testing.assert_array_equal(saved['features'],vf);np.testing.assert_array_equal(saved['native_output'],vnative);np.testing.assert_array_equal(saved['row_index'],te)
            np.testing.assert_array_equal(saved['archival_output'],original_native)
        z=((vf.astype(float)-mean)/sd)@theta[:-1]+theta[-1];p=expit(z)
        predicted=predictions[predictions.method.eq(family)&predictions.cutoff.eq(cutoff)&predictions.seed.eq(number)].sort_values('row_index')
        np.testing.assert_array_equal(predicted.row_index,te);np.testing.assert_array_equal(predicted.actual_up,raw_labels[te])
        np.testing.assert_allclose(predicted.actual_return,obs.exec_return.iloc[te],rtol=0,atol=1e-14)
        error=max(float(np.max(np.abs(z-predicted.probe_logit.to_numpy()))),float(np.max(np.abs(p-predicted.probability.to_numpy()))))
        assert error<1e-10;max_prediction=max(max_prediction,error)
        np.testing.assert_array_equal(predicted.direction_up,p>.5);np.testing.assert_allclose(predicted.score,p,rtol=0,atol=1e-12)
        np.testing.assert_allclose(predicted.training_frequency,meta[cutoff]['frequency'],rtol=0,atol=1e-15)
        old=archived[archived.method.eq(head['source'])&archived.cutoff.eq(cutoff)&archived.seed.eq(number)].sort_values('row_index')
        np.testing.assert_array_equal(old.row_index,te);np.testing.assert_array_equal(old.actual_up,raw_labels[te])
        if head['source']=='direction_bce':old_score=expit(original_native.astype(float));np.testing.assert_array_equal(old.direction_up,old_score>.5)
        else:
            scales=meta[cutoff]['scales'];old_score=original_native.astype(float)*scales['returns_sd']+scales['returns_mean']
            np.testing.assert_array_equal(old.direction_up,old_score>0);assert old.probability.isna().all()
        native_error=float(np.max(np.abs(old_score-old.score.to_numpy())));assert native_error<1e-10;max_prediction=max(max_prediction,native_error)
        native_count+=len(te);val_count+=len(te)
        assert object_hash(model.state_dict())==head['model_sha256'] and sha(PROJECT/head['project_file'])==head['sha256']
        assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        feature_checks.append(dict(family=family,cutoff=cutoff,seed=number,training_rows=len(tr),validation_rows=len(te),
            features_bitwise_equal=True,scaling_bitwise_equal=True,unchanged_model_sha256=head['model_sha256'],prediction_max_error=error,native_max_error=native_error))
        del model,testing;torch.cuda.empty_cache()
        print(f"Independent solution and feature/forecast replay {len(independent)}/18: {identity(head)} PASS",flush=True)
    assert train_count==34584 and val_count==native_count==846
    assert sum(h['iterations'] for h in heads)==fitting['total_newton_iterations']
    ensemble=pd.read_csv(OUT/'all_ensemble_predictions.csv');seed=pd.read_csv(OUT/'all_seed_predictions.csv')
    assert len(ensemble)==846 and len(seed)==1692 and len(predictions)==846
    for old,source in [(archived,seed),(pd.read_csv(OUT/'archived_ensemble_predictions.csv'),ensemble)]:
        keys=['method','date']+(['seed'] if 'seed' in old else [])
        actual=source[source.method.isin(old.method.unique())][old.columns].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,old.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-14)
    pd.testing.assert_frame_equal(seed[seed.method.str.startswith('probe_')][predictions.columns].reset_index(drop=True),predictions,
        check_dtype=False,rtol=1e-12,atol=1e-14)
    for method,g in ensemble.groupby('method'):
        g=g.sort_values('date');assert len(g)==141 and g.date.is_unique
        if method in seed.method.unique():
            field='score' if method=='archived20' else 'probability'
            sg=seed[seed.method.eq(method)];assert sg.groupby('date').size().eq(3).all()
            expected=sg.groupby('date')[field].mean();np.testing.assert_allclose(g[field],expected,rtol=0,atol=1e-14)
        else:
            expected=g.cutoff.map({c:m['frequency'] for c,m in meta.items()}).to_numpy() if method=='training_frequency' else np.full(141,.5)
            np.testing.assert_allclose(g.probability,expected,rtol=0,atol=1e-15)
        if method=='archived20':assert g.probability.isna().all();np.testing.assert_array_equal(g.direction_up,g.score>0)
        else:np.testing.assert_array_equal(g.direction_up,g.probability>.5)
    verify_metric_table(ensemble,pd.read_csv(OUT/'ensemble_metrics.csv'),['method'])
    verify_metric_table(seed,pd.read_csv(OUT/'seed_metrics.csv'),['method','seed'])
    annual=ensemble.assign(year=ensemble.date.str[:4].astype(int));verify_metric_table(annual,pd.read_csv(OUT/'yearly_metrics.csv'),['method','year'])
    frames,pairs,assessments=evaluation.compute()
    for name,frame in frames.items():pd.testing.assert_frame_equal(frame,pd.read_csv(OUT/f'{name}.csv'),check_dtype=False,rtol=1e-12,atol=1e-12)
    close_tree(pairs,read(OUT/'primary_comparisons.json'));close_tree(assessments,read(OUT/'assessments.json'))
    bins=pd.read_csv(OUT/'reliability_bins.csv');assert len(bins)==25
    for r in bins.itertuples():
        g=ensemble[ensemble.method.eq(r.method)];h=g[g.probability.ge(r.lower)&(g.probability.le(r.upper) if r.bin_index==4 else g.probability.lt(r.upper))]
        assert r.n==len(h)
        if len(h):assert abs(r.mean_probability-h.probability.mean())<1e-14 and abs(r.observed_frequency-h.actual_up.mean())<1e-14
        else:assert pd.isna(r.mean_probability) and pd.isna(r.observed_frequency)
    assert old_evidence()==prep['old_evidence']
    pd.DataFrame(independent).to_csv(OUT/'independent_solver_verification.csv',index=False)
    pd.DataFrame(feature_checks).to_csv(OUT/'feature_replay_verification.csv',index=False)
    files=[OUT/n for n in ['independent_solver_verification.csv','feature_replay_verification.csv']]
    result=dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,model_states_replayed=18,
        training_feature_rows=train_count,validation_feature_rows=val_count,independent_solutions=18,newton_solutions_replayed=18,
        new_seed_predictions=846,archived_seed_predictions=846,ensemble_methods=6,validation_weeks=141,
        primary_comparisons_recomputed=7,reliability_bins_recomputed=25,metrics_independently_verified=36,
        max_prediction_error=max_prediction,maximum_alternate_objective_gap=max(r['objective_absolute_gap'] for r in independent),
        maximum_alternate_training_probability_gap=max(r['maximum_training_probability_gap'] for r in independent),
        maximum_alternate_gradient_inf=max(r['alternate_gradient_inf'] for r in independent),previous_files_preserved=2527,
        previous_round_files_preserved=2479,preflight_files_preserved=48,preflight_coefficients_bitwise_equal=True,
        scipy_version=scipy.__version__,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

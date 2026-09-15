from common17 import *

def main():
    check_frozen();path=OUT/'training_manifest.json';assert not path.exists(),'Preserve training'
    run=manifest('training');save(path,run);start=time.time();obs,price,returns=data();bars=previous.raw_bars(price)
    files=[];thresholds=[];training_states=[]
    for fold in cfg()['folds']:
        tr,_=indices(obs,fold);d=descriptors(bars,obs.anchor.iloc[tr].to_numpy());t=fit_thresholds(d)
        thresholds.append(dict(cutoff=fold['cutoff'],train_n=len(tr),last_training_signal=obs.date.iloc[tr].max(),
            last_training_maturity=obs.joint_completed.iloc[tr].max(),**t))
        tags=assign_states(d,t);tags.insert(0,'date',obs.date.iloc[tr].to_numpy());tags.insert(0,'row_index',tr);tags.insert(0,'cutoff',fold['cutoff'])
        training_states.append(tags)
    save(OUT/'state_thresholds.json',thresholds);pd.concat(training_states,ignore_index=True).to_csv(OUT/'training_states.csv',index=False)
    files += [OUT/'state_thresholds.json',OUT/'training_states.csv']
    heads=[];traces=[];metrics=[];training_rows=0
    for source in read(OUT/'source_heads.json'):
        fold=next(f for f in cfg()['folds'] if f['cutoff']==source['cutoff']);tr,_=indices(obs,fold)
        d=load_npz(source);np.testing.assert_array_equal(d['row_index'],tr);y=(returns[tr]>0).astype(float);np.testing.assert_array_equal(d['direction'],y)
        c=fit_clip(d['features']);x=c['standardized'];theta,trace=fit_newton(x,y,cfg()['probe']);v,g,h=objective(theta,design(x),y,.01)
        assert np.max(np.abs(g))<=1e-9 and np.linalg.eigvalsh(h).min()>0
        job=source['job']+'_clip';method='learned_clip' if source['kind']=='learned' else 'raw25_clip'
        cache=CACHE/f'training_{job}.npz';np.savez_compressed(cache,**c,direction=y,row_index=tr);files.append(cache)
        head=dict(job=job,source_job=source['job'],method=method,kind=source['kind'],cutoff=source['cutoff'],seed=source['seed'],
            cache_file=str(cache.relative_to(PROJECT)),cache_sha256=sha(cache),coefficients=theta.tolist(),train_n=len(tr),
            iterations=trace[-1]['iteration'],objective=v,gradient_inf=float(np.max(np.abs(g))),hessian_min_eigenvalue=float(np.linalg.eigvalsh(h).min()),l2_lambda=.01)
        heads.append(head);training_rows+=len(tr)
        traces.extend(dict(job=job,**r) for r in trace)
        clipped=(d['features']<c['lower'])|(d['features']>c['upper'])
        measured=previous.previous.probe_metrics(design(x)@theta,y)
        constant=previous.previous.probe_metrics(np.full(len(y),np.log(y.mean()/(1-y.mean()))),y)
        metrics.append(dict(job=job,method=method,cutoff=source['cutoff'],seed=source['seed'],train_n=len(tr),
            iterations=head['iterations'],objective=v,gradient_inf=head['gradient_inf'],**measured,
            constant_brier=constant['brier'],constant_log_loss=constant['log_loss'],constant_accuracy=constant['accuracy'],
            coordinate_clip_fraction=float(clipped.mean()),row_clip_fraction=float(clipped.any(axis=1).mean()),
            floored_coordinates=int((c['clipped_features'].std(axis=0)<1e-6).sum())))
        save(OUT/'heads.json',heads);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False)
        print(f"Converged {len(heads)}/24: {job}; iterations={head['iterations']}; gradient={head['gradient_inf']:.2e}",flush=True)
    assert len(heads)==24 and training_rows==37860
    pd.DataFrame(metrics).to_csv(OUT/'training_metrics.csv',index=False)
    files += [OUT/'heads.json',OUT/'solver_trace.csv',OUT/'training_metrics.csv']
    finish(run,files,new_primary_fits=24,new_newton_iterations=sum(h['iterations'] for h in heads),training_feature_rows=training_rows,
        causal_state_training_rows=sum(len(g) for g in training_states),state_threshold_sets=6,all_converged=True,
        validation_scoring_during_fit=False,labels_used_to_fit_state_thresholds=False,neural_training_steps=0,elapsed_seconds=time.time()-start)
    print('All24 fits finished. Six label-free state threshold sets frozen. No new validation scores yet.',flush=True)

if __name__=='__main__':main()

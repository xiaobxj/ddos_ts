from common16 import *


def main():
    legacy.initialize();check_frozen();path=OUT/'training_manifest.json';assert not path.exists(),'Preserve training'
    run=manifest('training');save(path,run);start=time.time();obs,price,returns=data();bars=raw_bars(price)
    heads=[];curves=[];metrics=[];diagnostics=[];files=[];new_rows=0
    for job in read(OUT/'new_jobs.json'):
        fold=next(f for f in cfg()['folds'] if f['cutoff']==job['cutoff']);tr,_=indices(obs,fold);y=(returns[tr]>0).astype(float)
        if job['kind']=='learned':
            model,state,values,f,native=extract_learned(job,tr);assert state['train_n']==len(tr)
            assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
            del model,values;torch.cuda.empty_cache()
        else:f=raw_features(bars,obs.anchor.iloc[tr].to_numpy())
        x,mean,sd=normalize_train(f,cfg()['standardization']['sd_floor']);cache=CACHE/f"training_{job['job']}.npz"
        np.savez_compressed(cache,features=f,standardized=x,mean=mean,sd=sd,direction=y,row_index=tr);files.append(cache)
        theta,trace=fit_newton(x,y,cfg()['probe']);v,g,h=objective(theta,design(x),y,.01)
        assert np.max(np.abs(g))<=1e-9 and v<=trace[0]['objective']+1e-12 and np.linalg.eigvalsh(h).min()>0
        head=dict(job,cache_file=str(cache.relative_to(PROJECT)),cache_sha256=sha(cache),coefficients=theta.tolist(),train_n=len(tr),
            iterations=trace[-1]['iteration'],objective=v,gradient_inf=float(np.max(np.abs(g))),hessian_min_eigenvalue=float(np.linalg.eigvalsh(h).min()),l2_lambda=.01)
        heads.append(head);new_rows+=len(tr)
        for row in trace:curves.append(dict(job=job['job'],method=job['method'],cutoff=job['cutoff'],seed=job['seed'],**row))
        save(OUT/'new_heads.json',heads);pd.DataFrame(curves).to_csv(OUT/'solver_trace.csv',index=False)
        print(f"New head {len(heads)}/15: {job['job']}, iterations={head['iterations']}, gradient={head['gradient_inf']:.3e}",flush=True)
    assert len(heads)==15 and new_rows==20568
    heads+=read(OUT/'reused_heads.json');assert len(heads)==24
    for head in heads:
        d=load_features(head);fold=next(f for f in cfg()['folds'] if f['cutoff']==head['cutoff']);tr,_=indices(obs,fold)
        np.testing.assert_array_equal(d['row_index'],tr);np.testing.assert_array_equal(d['direction'],(returns[tr]>0).astype(float))
        row=training_metrics(head,d);assert row['gradient_inf']<=1e-9 and row['objective']<=row['constant_log_loss']+1e-12;metrics.append(row)
        sd=d['features'].astype(float).std(axis=0);sv=np.linalg.svd(d['standardized'],compute_uv=False)
        diagnostics.append(dict(job=head['job'],method=head['method'],cutoff=head['cutoff'],seed=head['seed'],reused=head['reused'],
            train_n=len(tr),dimensions=25,numerical_rank=int((sv>1e-8).sum()),minimum_singular_value=float(sv.min()),
            minimum_raw_sd=float(sd.min()),maximum_raw_sd=float(sd.max()),floored_coordinates=int((sd<1e-6).sum())))
    heads=sorted(heads,key=lambda h:h['job']);save(OUT/'heads.json',heads)
    pd.DataFrame(metrics).to_csv(OUT/'training_metrics.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'feature_diagnostics.csv',index=False)
    files += [OUT/n for n in ['new_heads.json','solver_trace.csv','heads.json','training_metrics.csv','feature_diagnostics.csv']]
    finish(run,files,elapsed_seconds=time.time()-start,new_primary_fits=15,reused_heads=9,all_converged=True,neural_training_steps=0,
        new_training_feature_rows=new_rows,new_newton_iterations=sum(h['iterations'] for h in heads if not h['reused']),validation_scoring_during_fit=False)
    print('All15 new heads converged;9 retained heads verified. No held-out scoring during fitting.',flush=True)


if __name__=='__main__':main()

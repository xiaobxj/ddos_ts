from common15 import *

def main():
    check_frozen();check_artifacts('extraction');path=OUT/'fitting_manifest.json';assert not path.exists(),'Preserve probe fits'
    run=manifest('fitting');save(path,run);start=time.time();heads=[];curves=[];summary=[]
    meta=read(OUT/'training_metadata.json')
    for ref in read(OUT/'feature_models.json'):
        d=load_features(ref);x=d['standardized'];y=d['direction'];theta,trace=fit_newton(x,y,cfg()['probe'])
        a=design(x);value,gradient,hessian=objective(theta,a,y,cfg()['probe']['l2_lambda'])
        assert np.max(np.abs(gradient))<=1e-9 and value<=trace[0]['objective']+1e-12
        head=dict(ref,coefficients=theta.tolist(),iterations=trace[-1]['iteration'],gradient_inf=float(np.max(np.abs(gradient))),
            objective=value,hessian_min_eigenvalue=float(np.linalg.eigvalsh(hessian).min()),l2_lambda=.01)
        heads.append(head)
        for row in trace:curves.append(dict(family=ref['family'],cutoff=ref['cutoff'],seed=ref['seed'],**row))
        freq=meta[ref['cutoff']]['frequency'];constant=probe_metrics(np.full(len(y),np.log(freq/(1-freq))),y)
        row=dict(family=ref['family'],cutoff=ref['cutoff'],seed=ref['seed'],train_n=len(y),iterations=head['iterations'],
            gradient_inf=head['gradient_inf'],objective=value,penalty=.005*float(theta[:-1]@theta[:-1]),
            hessian_min_eigenvalue=head['hessian_min_eigenvalue'],**probe_metrics(a@theta,y),
            constant_log_loss=constant['log_loss'],constant_brier=constant['brier'],native_log_loss=None,native_brier=None,native_ridge_objective=None,
            native_probability_std=None,native_accuracy=None)
        if ref['source']=='direction_bce':
            w=d['native_weights']*d['sd'];b=float(d['native_bias'])+float(d['native_weights']@d['mean']);native_theta=np.r_[w,b]
            native_value,_=objective(native_theta,a,y,.01,False)
            native=probe_metrics(d['native_output'].astype(float),y)
            assert value<=native_value+1e-12
            row.update(native_log_loss=native['log_loss'],native_brier=native['brier'],native_ridge_objective=native_value,
                native_probability_std=native['probability_std'],native_accuracy=native['accuracy'])
        else:
            scales=meta[ref['cutoff']]['scales'];returns=d['native_output'].astype(float)*scales['returns_sd']+scales['returns_mean']
            row['native_accuracy']=float(((returns>0)==(y>0)).mean())
        summary.append(row);save(OUT/'heads.json',heads)
        pd.DataFrame(curves).to_csv(OUT/'solver_trace.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'training_metrics.csv',index=False)
        print(f"Probe fit {len(heads)}/18: {identity(ref)}, iterations={head['iterations']}, gradient={head['gradient_inf']:.3e}",flush=True)
    assert len(heads)==18;check_artifacts('extraction')
    preflight={identity(h):h for h in read(PREFLIGHT/'results/heads.json')}
    for head in heads:np.testing.assert_array_equal(head['coefficients'],preflight[identity(head)]['coefficients'])
    files=[OUT/n for n in ['heads.json','solver_trace.csv','training_metrics.csv']]
    run.update(finished_utc=now(),elapsed_seconds=time.time()-start,primary_heads=18,neural_training_steps=0,
        total_newton_iterations=sum(h['iterations'] for h in heads),all_converged=True,validation_used=False,preflight_coefficients_bitwise_equal=True,
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in files});save(path,run)
    print('All18 frozen-feature probes converged before validation scoring.',flush=True)

if __name__=='__main__':main()

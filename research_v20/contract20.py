from common20 import *

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();obs,price,returns=data();tests=[];rows=[]
    # Inclusion is by signal date; label completion still enforces availability.
    toy=pd.DataFrame(dict(date=['2011-12-31','2012-01-01','2014-12-01','2015-01-01']))
    np.testing.assert_array_equal(recent_positions(toy,np.arange(4),dict(recent_start='2012-01-01')),[1,2,3])
    tests.append('signal-date lower boundary inclusive; caller supplies only label-mature full-fold rows')
    sources={s['job']:s for s in read(OUT/'source_heads.json')}
    for job in read(OUT/'jobs.json'):
        f=next(f for f in cfg()['folds'] if f['cutoff']==job['cutoff']);tr,te=indices(obs,f);pos=recent_positions(obs,tr,f)
        x,idx=source_inputs(sources[job['source_job']],job['source_round'],'training');np.testing.assert_array_equal(idx,tr)
        selected=tr[pos];assert len(selected)==f['recent_train_n'] and obs.date.iloc[selected].ge(f['recent_start']).all() and obs.joint_completed.iloc[selected].le(f['cutoff']).all()
        assert np.isfinite(x[pos]).all() and len(np.unique(returns[selected]>0))==2
        if job['interaction']:
            source=sources[job['source_job']];parent=sources[source['source_job']];base,bidx=source_inputs(parent,18,'training')
            np.testing.assert_array_equal(bidx,tr);np.testing.assert_array_equal(x[:,:-1],base)
            t=np.asarray(parent['coefficients']);nested=np.r_[t[:-1],0.,t[-1]]
            np.testing.assert_allclose(design(x[pos])@nested,design(base[pos])@t,rtol=0,atol=1e-12)
            val,_,_=objective(nested,design(x[pos]),(returns[selected]>0).astype(float),.01);bval,_,_=objective(t,design(base[pos]),(returns[selected]>0).astype(float),.01)
            assert abs(val-bval)<1e-12
        rows.append(dict(job=job['job'],full_rows=len(tr),recent_rows=len(pos),dimensions=x.shape[1],first_date=obs.date.iloc[selected[0]],last_completed=obs.joint_completed.iloc[selected].max()))
    tests.append('48training joins preserve frozen coordinates; only known-label recent membership changes;24interactions exactly nest their additive inputs')
    rng=np.random.default_rng(20262000)
    for dim in [26,27,29,30]:
        x=rng.normal(size=(107,dim));y=(rng.random(107)>.45).astype(float);theta,trace=fit_newton(x,y,cfg()['probe']);value,grad,hess=objective(theta,design(x),y,.01)
        assert np.max(np.abs(grad))<=1e-9 and np.linalg.eigvalsh(hess).min()>0
    tests.append('fourdimension convex-solver synthetic convergence; no real candidate fit or held-out score in contract')
    path=OUT/'contract_input_checks.csv';pd.DataFrame(rows).to_csv(path,index=False)
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(path.relative_to(ROOT)):sha(path)}))
    print(json.dumps(dict(status='PASS',training_input_checks=len(rows),tests=tests)),flush=True)

if __name__=='__main__':main()

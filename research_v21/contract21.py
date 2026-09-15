from common21 import *
from audit_hmm21 import log_reference,encode,independent_objective
import itertools

def synthetic():
    y=np.array([.01,-.03,np.nan,.015,-.004]);v=np.array([.0001,.0009]);p=np.array([[.92,.08],[.13,.87]])
    z=forward(y,v,p,posterior=True);ref=log_reference(y,v,p);weights=[];paths=list(itertools.product(range(2),repeat=len(y)))
    for path in paths:
        w=.5
        for t,state in enumerate(path):
            if t:w*=p[path[t-1],state]
            if np.isfinite(y[t]):w*=np.exp(-y[t]**2/(2*v[state]))/np.sqrt(2*np.pi*v[state])
        weights.append(w)
    weights=np.asarray(weights);ll=np.log(weights.sum());weights/=weights.sum();gamma=np.zeros((len(y),2));xi=np.zeros((2,2))
    for path,w in zip(paths,weights):
        for t,state in enumerate(path):
            gamma[t,state]+=w
            if t:xi[path[t-1],state]+=w
    assert abs(ll-z['log_likelihood'])<1e-12
    for key,values in [('smoothed_training',gamma),('transition_counts',xi)]:np.testing.assert_allclose(z[key],values,rtol=0,atol=1e-12)
    for key in ['filtered','smoothed_training','transition_counts']:np.testing.assert_allclose(ref[key],z[key],rtol=0,atol=1e-12)
    for n in range(1,len(y)+1):
        prefix=forward(y[:n],v,p,posterior=True);np.testing.assert_array_equal(prefix['filtered'],z['filtered'][:n])
        np.testing.assert_allclose(prefix['smoothed_training'][-1],z['filtered'][n-1],rtol=0,atol=1e-12)
    np.testing.assert_allclose(z['filtered'][2],z['filtered'][1]@p,rtol=0,atol=1e-12)
    cont=forward(y[3:],v,p,initial=z['filtered'][2]@p);np.testing.assert_array_equal(cont['filtered'],z['filtered'][3:])
    theta=encode(v,p);val,grad=independent_objective(theta,y);fd=[]
    for j in range(4):
        h=np.zeros(4);h[j]=1e-5;fd.append((independent_objective(theta+h,y)[0]-independent_objective(theta-h,y)[0])/2e-5)
    np.testing.assert_allclose(grad,fd,rtol=0,atol=1e-8);np.testing.assert_allclose(grad,likelihood_gradient(y,v,p,z),rtol=0,atol=1e-12)
    toy=pd.DataFrame(dict(close=[100,101,102,103,104],valid_ohlc=[True,False,True,True,True]));r=daily_returns(toy)
    assert np.isnan(r[:2]).all() and np.isfinite(r[2:]).all()
    return ['exhaustive32state paths verify likelihood,gamma,xi; independent log-domain recursion agrees','filtered prefixes and continuation invariant; missing emission advances transition','four likelihood derivatives agree with central finite differences; adjacent invalid bars masked']

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=synthetic();obs,price,returns=data();sources={s['job']:s for s in read(OUT/'source_heads.json')};checks=[]
    for f in cfg()['folds']:
        tr,te=indices(obs,f);a,b=daily_extent(price,f)
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(f['cutoff'])))
        expected=np.flatnonzero(obs.date.gt(f['cutoff'])&obs.date.le(f['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31'))
        np.testing.assert_array_equal(te,expected);assert len(tr)==f['train_n'] and len(te)==f['test_n'] and obs.anchor.iloc[tr].max()<=a
        assert obs.anchor.iloc[te].min()>a and obs.anchor.iloc[te].max()<=b
    for job in read(OUT/'jobs.json'):
        if job['source_job'] is None:continue
        s=sources[job['source_job']];x,rows=base_inputs(s,'training');f=next(f for f in cfg()['folds'] if f['cutoff']==job['cutoff']);tr,_=indices(obs,f)
        np.testing.assert_array_equal(rows,tr);assert x.shape==(len(tr),job['dimensions']-1) and np.isfinite(x).all()
        checks.append(dict(job=job['job'],n=len(tr),base_dimensions=x.shape[1],last_label=obs.joint_completed.iloc[tr].max()))
    rng=np.random.default_rng(20262100)
    for d in [1,27,30]:
        x=rng.normal(size=(111,d));y=(rng.random(111)>.45).astype(float);t,trace=fit_newton(x,y,cfg()['probe']);v,g,h=objective(t,design(x),y,.01)
        assert abs(g).max()<=1e-9 and np.linalg.eigvalsh(h).min()>0
    tests+=['six independent date/maturity/daily-anchor checks and24unchanged training input joins','three classifier-dimension synthetic convex convergence checks; no real model fit']
    path=OUT/'contract_input_checks.csv';pd.DataFrame(checks).to_csv(path,index=False)
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(path.relative_to(ROOT)):sha(path)}))
    print(json.dumps(dict(status='PASS',tests=tests)),flush=True)

if __name__=='__main__':main()

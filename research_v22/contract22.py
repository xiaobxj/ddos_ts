from common22 import *

def synthetic():
    rng=np.random.default_rng(20262200);dates=pd.date_range('2000-01-01',periods=301).strftime('%Y-%m-%d');close=np.exp(4+np.r_[0,rng.normal(0,.01,300).cumsum()]);p=pd.DataFrame(dict(date=dates,close=close,valid_ohlc=True))
    f=features(p);np.testing.assert_allclose(f[NUMERIC],scalar_features(p,np.arange(len(p))),rtol=0,atol=2e-13,equal_nan=True)
    for end in [120,183,250]:
        prefix=features(p.iloc[:end+1]);pd.testing.assert_frame_equal(prefix,f.iloc[:end+1]);altered=p.copy();altered.loc[end+1:,'close']*=3
        pd.testing.assert_frame_equal(features(altered).iloc[:end+1],prefix)
    bad=p.copy();bad.loc[150,'valid_ohlc']=False;b=features(bad);assert b[NUMERIC].iloc[150:271].isna().all().all() and b[NUMERIC].iloc[271].notna().all()
    flat=p.copy();flat['close']=100.;ff=features(flat);assert ff.relative_volatility.iloc[120:].eq(0).all() and ff.efficiency60.iloc[120:].eq(0).all()
    for sign in [-1,1]:
        monotone=p.copy();monotone['close']=100*np.exp(sign*.002*np.arange(len(p)));m=features(monotone);np.testing.assert_allclose(m.efficiency60.iloc[120:],1.,rtol=0,atol=1e-13)
    alt=p.copy();alt['close']=100*np.exp(np.where(np.arange(len(p))%2, .01,0));a=features(alt);np.testing.assert_allclose(a.efficiency60.iloc[120:],0.,rtol=0,atol=1e-13)
    scaled=p.copy();scaled['close']*=100;np.testing.assert_allclose(features(scaled)[NUMERIC],f[NUMERIC],rtol=0,atol=2e-13,equal_nan=True)
    # A missing feature separates runs; no invented adjacent transition.
    toy=pd.DataFrame(dict(anchor=[0,1,2,3,4],date=list('abcde'),relative_volatility=[-.1,-.2,np.nan,.2,.3],relative_volatility_state=['subdued','subdued','missing','elevated','elevated']))
    diag,runs=run_diagnostics(toy,'relative_volatility');assert diag['adjacent_pairs']==2 and diag['switches']==0 and [r['length'] for r in runs]==[2,2] and all(r['left_censored'] and r['right_censored'] for r in runs)
    return ['independent scalar versus rolling state formulas; future-prefix invariance at3boundaries','flat,monotone,alternating,price-scale and exact121bar bad-history tests','missing feature breaks run adjacency; run censoring checked']

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=synthetic();obs,price,returns=data();checks=[];sources={s['job']:s for s in read(OUT/'source_heads.json')}
    for f in cfg()['folds']:
        tr,te=indices(obs,f);a,b=daily_extent(price,f);state=features(price.iloc[:a+1]);sg=signal_rows(state,obs,tr,f['cutoff'])
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(f['cutoff'])));assert len(tr)==f['train_n'] and obs.anchor.iloc[tr].max()<=a
        np.testing.assert_allclose(sg[NUMERIC],scalar_features(price.iloc[:a+1],obs.anchor.iloc[tr].to_numpy(int)),rtol=0,atol=2e-12)
        for s in sources.values():
            if s['cutoff']!=f['cutoff']:continue
            x,rows=base_inputs(s,'training');np.testing.assert_array_equal(rows,tr);assert np.isfinite(x).all();checks.append(dict(job=s['job'],rows=len(tr),dimensions=x.shape[1],last_label=obs.joint_completed.iloc[tr].max()))
    rng=np.random.default_rng(22)
    for dim in [1,27,30]:
        x=rng.normal(size=(100,dim));y=(rng.random(100)>.45).astype(float);t,trace=fit_newton(x,y,cfg()['probe']);v,g,h=objective(t,design(x),y,.01);assert abs(g).max()<=1e-9 and np.linalg.eigvalsh(h).min()>0
    tests+=['all6training state/maturity joins independently reconstructed;24unchanged source joins','three synthetic convex fits; no real new classifier fit or heldout scoring']
    path=OUT/'contract_input_checks.csv';pd.DataFrame(checks).to_csv(path,index=False)
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(path.relative_to(ROOT)):sha(path)}));print(json.dumps(dict(status='PASS',tests=tests)),flush=True)

if __name__=='__main__':main()

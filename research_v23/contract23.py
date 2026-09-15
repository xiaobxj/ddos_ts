from common23 import *
from states23 import sign_statistic
import itertools

def synthetic():
    for signs in [[1,1,1,-1,-1,0],[1,1,-1,-1],[0,0,1,-1],[1]*6,[-1]*6,[0]*6]:
        permutations=list(set(itertools.permutations(signs)));observed=[sign_statistic(s)[0] for s in permutations];expected=sign_statistic(signs)[1]
        assert abs(np.mean(observed)-expected)<1e-14 and abs(np.mean([sign_statistic(s)[2] for s in permutations]))<1e-14
    a=np.r_[np.ones(40),-np.ones(20)];b=np.tile([1,-1,1,-1,1,1],10);assert a.sum()==b.sum() and np.square(a).sum()==np.square(b).sum()
    assert sign_statistic(a)[1]==sign_statistic(b)[1] and sign_statistic(a)[2]>0 and sign_statistic(b)[2]<0
    for s in [a,b]:assert sign_statistic(s)==sign_statistic(-s) and sign_statistic(s)==sign_statistic(s[::-1])
    for sign in [-1,0,1]:assert sign_statistic([sign]*60)[2]==0
    rng=np.random.default_rng(20262300);close=np.exp(4+np.r_[0,rng.normal(0,.01,300).cumsum()]);price=pd.DataFrame(dict(date=pd.date_range('2000-01-01',periods=301).strftime('%Y-%m-%d'),close=close,valid_ohlc=True));f=features(price)
    np.testing.assert_allclose(f[NUMERIC],scalar_features(price,np.arange(len(price))),rtol=0,atol=2e-13,equal_nan=True)
    for end in [120,183,250]:
        prefix=features(price.iloc[:end+1]);pd.testing.assert_frame_equal(prefix,f.iloc[:end+1]);altered=price.copy();altered.loc[end+1:,'close']*=3;pd.testing.assert_frame_equal(features(altered).iloc[:end+1],prefix)
    bad=price.copy();bad.loc[150,'valid_ohlc']=False;z=features(bad);assert z[NUMERIC].iloc[150:271].isna().all().all() and z[NUMERIC].iloc[271].notna().all()
    flat=price.copy();flat['close']=100.;z=features(flat);assert z.excess_order.iloc[120:].eq(0).all() and z.neutral_count60.iloc[120:].eq(60).all()
    scaled=price.copy();scaled['close']*=100;np.testing.assert_array_equal(features(scaled)[NUMERIC[5:]],f[NUMERIC[5:]])
    toy=pd.DataFrame(dict(anchor=[0,1,2,3,4,5],date=list('abcdef'),excess_order=[-.1,-.1,np.nan,0.,.1,.1],order_state=['negative_order','negative_order','missing','near_zero_order','positive_order','positive_order']))
    d,runs=run_diagnostics(toy,'excess_order');assert d['adjacent_pairs']==3 and d['switches']==1 and [r['length'] for r in runs]==[2,1,2] and d['high_state_share']==.4
    return ['exactmultiset permutation expectation with neutrals; six exhaustivecases','samecounts reordered sign paths changegate; reversal/signflip invariance; constant-signgate0','scalarrolling,3futureprefix,121invalidbar,flat and price-scale tests','three-state runbreak,censoring and neutral-state occupancy']

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=synthetic();obs,price,returns=data();checks=[];sources=read(OUT/'source_heads.json');r18={s['job']:s for s in read(V18/'results/heads.json')}
    for f in cfg()['folds']:
        tr,te=indices(obs,f);a,b=daily_extent(price,f);frame=features(price.iloc[:a+1]);sg=signal_rows(frame,obs,tr,f['cutoff']);np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(f['cutoff'])))
        assert len(tr)==f['train_n'];np.testing.assert_allclose(sg[NUMERIC],scalar_features(price.iloc[:a+1],obs.anchor.iloc[tr].to_numpy(int)),rtol=0,atol=2e-12)
        for s in sources:
            if s['cutoff']!=f['cutoff']:continue
            x,rows=base_inputs(s,'training');np.testing.assert_array_equal(rows,tr);np.testing.assert_array_equal(x,load_npz(s)['standardized']);np.testing.assert_array_equal(source_ray(s),np.asarray(r18[s['source_job']]['coefficients'])[:25]);assert x.shape[1]==s['dimensions'] and np.isfinite(x).all();checks.append(dict(job=s['job'],rows=len(tr),dimensions=x.shape[1],ray_source=s['source_job']))
    rng=np.random.default_rng(23)
    for dim in [1,28,31]:
        x=rng.normal(size=(100,dim));y=(rng.random(100)>.45).astype(float);t,tr=fit_newton(x,y,cfg()['probe']);v,g,h=objective(t,design(x),y,.01);assert abs(g).max()<=1e-9 and np.linalg.eigvalsh(h).min()>0
    tests+=['sixcausaltraining joins;24unchanged R19inputs with inherited R18ray','three syntheticconvex fits; no realcandidate fit or heldoutscoring']
    path=OUT/'contract_input_checks.csv';pd.DataFrame(checks).to_csv(path,index=False);save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),artifacts={str(path.relative_to(ROOT)):sha(path)}));print(json.dumps(dict(status='PASS',tests=tests)),flush=True)
if __name__=='__main__':main()

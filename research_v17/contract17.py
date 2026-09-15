from common17 import *

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=[]
    rng=np.random.default_rng(20260917);c=np.exp(4+np.cumsum(rng.normal(.0002,.01,240)))
    bars=np.column_stack([c*.998,c*1.015,c*.985,c,np.exp(rng.normal(9,.3,240))]);anchors=np.array([60,124,180,219])
    a=descriptors(bars,anchors);b=scalar_descriptors(bars,anchors)
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-13)
    for t in anchors:
        changed=bars.copy();changed[t+1:,:4]*=13;changed[t+1:,4]+=1e9
        np.testing.assert_array_equal(descriptors(bars,[t]),descriptors(changed,[t]))
        np.testing.assert_array_equal(descriptors(bars,[t]),descriptors(bars[:t+1],[t]))
    flat=bars.copy();flat[:,:4]=100.;flat[:,4]=1000.;assert np.isfinite(descriptors(flat,anchors).to_numpy()).all()
    tests.append('independent scalar market formulas, flat-price floor, future perturbation and truncated-history invariance')
    x=np.column_stack([np.arange(101.),np.arange(101.)[::-1],np.ones(101)]);d=fit_clip(x)
    np.testing.assert_array_equal(d['lower'],[1.,1.,1.]);np.testing.assert_array_equal(d['upper'],[99.,99.,1.])
    f=x.copy();f[0,0]=-1e12;f[-1,1]=1e12;frozen={k:v.copy() for k,v in d.items()};apply_clip(f,d)
    for k in d:np.testing.assert_array_equal(d[k],frozen[k])
    assert d['sd'][2]==1e-6
    tests.append('known order-statistic clipping bounds, clipped-training moments and immutable future transform')
    obs,price,returns=data();bars=previous.raw_bars(price)
    for fold in cfg()['folds']:
        tr,te=indices(obs,fold);desc=descriptors(bars,obs.anchor.iloc[tr].to_numpy());thresholds=fit_thresholds(desc)
        cutoff_anchor=int(np.flatnonzero(price.date.le(fold['cutoff']).to_numpy())[-1]);changed=bars.copy()
        changed[cutoff_anchor+1:,:4]*=11.;changed[cutoff_anchor+1:,4]+=1e10
        assert fit_thresholds(descriptors(changed,obs.anchor.iloc[tr].to_numpy()))==thresholds
        assert len(tr)==fold['train_n'] and obs.joint_completed.iloc[tr].le(fold['cutoff']).all()
        q=assign_states(desc,thresholds);assert set(q.trend_volatility)<=set(PARTITIONS['trend_volatility'])
    tests.append('six causal training masks and annual state-threshold invariance to all post-cutoff prices')
    desc=pd.DataFrame(dict(trend60=[-1.,0.,1.,2.],volatility20=[.1,.2,.3,.4],range20=[.1,.2,.3,.4],volume_change20=[-1.,0.,1.,2.]))
    thresholds=dict(trend_lower=0.,trend_upper=1.,volatility_median=.2,range_median=.2,volume_median=0.)
    tags=assign_states(desc,thresholds)
    assert tags.trend.tolist()==['low_trend','low_trend','mid_trend','high_trend'] and tags.volatility.tolist()==['low_vol','low_vol','high_vol','high_vol']
    syn=rng.normal(size=(71,25));y=(rng.random(71)>.48).astype(float);theta,trace=fit_newton(syn,y,cfg()['probe'])
    v,g,h=objective(theta,design(syn),y,.01);assert np.max(np.abs(g))<=1e-9 and np.linalg.eigvalsh(h).min()>0
    tests.append('label-free state function and exact boundary ties; inherited convex solver synthetic convergence')
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes()))
    print(json.dumps(dict(status='PASS',tests=tests)),flush=True)

if __name__=='__main__':main()

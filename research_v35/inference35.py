from common35 import *

def verify_inference(ensemble,pairs):
    ps=[]
    for w in cfg()['windows']:
        n=w['n'];rng=np.random.default_rng(20260910);starts=rng.integers(0,n,size=(10000,(n+7)//8));ids=np.array([[(int(s)+j)%n for s in row for j in range(8)][:n] for row in starts]);np.testing.assert_array_equal(ids,statistics.bootstrap_indices(n))
        for row in [r for r in pairs if r['window']==w['name']]:
            e=ensemble[ensemble.date.between(w['start'],w['end'])];a=e[e.history.eq(row['history'])&e.method.eq(row['candidate'])].sort_values('date');b=e[e.history.eq(row['reference_history'])&e.method.eq(row['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist() and len(a)==n
            if row['metric']=='direction_error':v=(a.direction_up.to_numpy()!=a.actual_up.to_numpy()).astype(float)-(b.direction_up.to_numpy()!=b.actual_up.to_numpy()).astype(float)
            else:v=(a.probability.to_numpy()-a.actual_up.to_numpy())**2-(b.probability.to_numpy()-b.actual_up.to_numpy())**2
            mu=float(v.mean());boot=v[ids].mean(1);center=(v-mu)[ids].mean(1);p=(1+int((np.abs(center)>=abs(mu)).sum()))/10001;lo,hi=np.quantile(boot,[.025,.975]);assert abs(mu-row['difference'])<1e-14 and abs(lo-row['ci95_low'])<1e-14 and abs(hi-row['ci95_high'])<1e-14 and p==row['p'];ps.append(p)
    assert ps==[r['p'] for r in pairs];last=0.;adjusted=[0.]*len(ps)
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):last=max(last,min(1.,ps[i]*(len(ps)-rank)));adjusted[i]=last
    np.testing.assert_array_equal(adjusted,[r['holm_adjusted_p'] for r in pairs])


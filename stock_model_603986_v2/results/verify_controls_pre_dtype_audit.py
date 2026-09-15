"""Independent checks of frequency, market-only and scalar probability controls."""
from core import *
from scipy.special import expit

def main():
    check_freeze();f=data();checks=[]
    for h in [1,5]:
        for task in tasks(h):
            a=interface(f,task);ids=a['ids'];folder=OUT/'fits'/task_id(task);done=read(folder/'completed.json')
            assert done['training_frequency']==float(a['y'].mean())
            with np.load(folder/'controls.npz') as d:
                lo=np.quantile(a['market'][ids],.01,axis=0);hi=np.quantile(a['market'][ids],.99,axis=0)
                clipped=np.clip(a['market'][ids],lo,hi);mean=clipped.mean(0);sd=np.maximum(clipped.std(0),1e-6)
                for name,v in [('lower',lo),('upper',hi),('mean',mean),('sd',sd)]:np.testing.assert_array_equal(d[name],v)
                x=(np.clip(a['market'],lo,hi)-mean)/sd;z=np.c_[x,np.ones(len(x))];theta=d['market_theta']
                probability=expit(z@theta);np.testing.assert_allclose(probability,d['market_probability'],rtol=0,atol=2e-15)
                gradient=z[ids].T@(probability[ids]-a['y'])/len(ids)+.01*np.r_[theta[:-1],0.]
                gap=float(abs(gradient).max());assert gap<3e-9
                checks.append(dict(task=task_id(task),control='market4',gradient_inf=gap))
            for seed in SEEDS:
                for epoch in [10,20]:
                    with np.load(folder/f'cache_{seed}_e{epoch}.npz') as d:
                        head=read(folder/f'heads_{seed}_e{epoch}.json');t=head['scalar_transform'];native=d['all_native'][:,None]
                        # Recompute the training-only scalar transform independently.
                        lo=np.quantile(native[ids],.01,axis=0);hi=np.quantile(native[ids],.99,axis=0)
                        clipped=np.clip(native[ids],lo,hi);mean=clipped.mean(0);sd=np.maximum(clipped.std(0),1e-6)
                        for name,v in [('lower',lo),('upper',hi),('mean',mean),('sd',sd)]:np.testing.assert_allclose(t[name],v,rtol=0,atol=1e-12)
                        x=(np.clip(native,np.asarray(t['lower']),np.asarray(t['upper']))-np.asarray(t['mean']))/np.asarray(t['sd'])
                        z=np.c_[x,np.ones(len(x))];theta=np.asarray(head['scalar_coefficients']);p=expit(z@theta)
                        np.testing.assert_allclose(p,d['p__scalar_probe.U'],rtol=0,atol=2e-15)
                        gradient=z[ids].T@(p[ids]-a['y'])/len(ids)+.01*np.r_[theta[:-1],0.]
                        gap=float(abs(gradient).max());assert gap<3e-9
                        checks.append(dict(task=task_id(task),control='scalar_probe',seed=seed,epoch=epoch,gradient_inf=gap))
                        if task['objective']=='bce':np.testing.assert_allclose(expit(native[:,0]),d['p__native_bce.U'],rtol=0,atol=1e-7)
    csv(OUT/'control_checks.csv',pd.DataFrame(checks))
    save(OUT/'controls_verified.json',dict(status='PASS',completed_utc=now(),controls=len(checks),
        verifier_sha256=sha(ROOT/'verify_controls.py'),max_gradient=max(c['gradient_inf'] for c in checks)))
    print('Controls PASS',len(checks))

if __name__=='__main__':main()

from common19 import *

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=[]
    # Balanced deterministic grid: product is orthogonal to both main effects.
    grid=np.array([(s,v) for s in [-2.,-1.,1.,2.] for v in [-1.,1.]])
    x=np.zeros((8,26));x[:,0]=grid[:,0];x[:,25]=grid[:,1];ray=np.zeros(25);ray[0]=1.
    t=fit_interaction(x,grid[:,1],ray);expected=grid[:,0]*grid[:,1]
    np.testing.assert_allclose(t['product'],expected,rtol=0,atol=0);np.testing.assert_allclose(t['interaction'],expected/expected.std(),rtol=0,atol=1e-12)
    assert int(t['augmented_design_rank'])==int(t['base_design_rank'])+1
    zero=fit_interaction(x,np.zeros(8),ray);assert float(zero['residual_sd'])==1e-6 and np.count_nonzero(zero['interaction'])==0
    frozen={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in t.items()};changed=x.copy();changed[:,:25]*=17
    apply_interaction(changed,np.full(8,12.),t)
    for k in t:np.testing.assert_array_equal(t[k],frozen[k])
    tests.append('known volatility/sign product, removal of linear redundancy, rank gain, zero-product floor and immutable future transform')
    rng=np.random.default_rng(20261900)
    for dim in [30,27]:
        a=rng.normal(size=(103,dim));y=(rng.random(103)>.48).astype(float);theta,trace=fit_newton(a,y,cfg()['probe']);v,g,h=objective(theta,design(a),y,.01)
        assert np.max(np.abs(g))<=1e-9 and np.linalg.eigvalsh(h).min()>0 and len(theta)==dim+1
    tests.append('inherited convex solver converges for30and27slopes without a real candidate fit')
    obs,price,returns=data();diagnostics=[]
    for s in read(OUT/'source_heads.json'):
        fold=next(f for f in cfg()['folds'] if f['cutoff']==s['cutoff']);tr,_=indices(obs,fold);x,v,rows=inherited_inputs(s,'training');np.testing.assert_array_equal(rows,tr)
        ray=np.asarray(s['coefficients'])[:25];t=fit_interaction(x,v,ray);assert float(t['training_orthogonality_inf'])<1e-8
        parent=np.asarray(s['coefficients']);nested=np.r_[parent[:-1],0.,parent[-1]];value,_,_=objective(nested,design(t['standardized']),(returns[tr]>0).astype(float),.01)
        assert abs(value-s['objective'])<1e-12
        np.testing.assert_allclose(design(t['standardized'])@nested,design(x)@parent,rtol=0,atol=1e-12)
        assert int(t['augmented_design_rank'])==int(t['base_design_rank'])+1 and float(t['residual_raw_sd'])>1e-6
        diagnostics.append(dict(source_job=s['job'],base_rank=int(t['base_design_rank']),augmented_rank=int(t['augmented_design_rank']),
            residual_sd=float(t['residual_raw_sd']),orthogonality_inf=float(t['training_orthogonality_inf'])))
    tests.append('24real training-only joins, nonredundant rank gains, orthogonality and exact parent nesting at zero interaction slope')
    pd.DataFrame(diagnostics).to_csv(OUT/'contract_transform_diagnostics.csv',index=False)
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        artifacts={'results\\contract_transform_diagnostics.csv':sha(OUT/'contract_transform_diagnostics.csv')}))
    print(json.dumps(dict(status='PASS',tests=tests)),flush=True)

if __name__=='__main__':main()

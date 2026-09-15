from common15 import *

def main():
    legacy.initialize();check_frozen(False);tests=[];rng=np.random.default_rng(20261500)
    f=rng.normal(size=(83,4));y=(rng.random(83)<.55).astype(float);x,mean,sd=normalize_train(f);a=design(x)
    theta=rng.normal(size=5)*.2;lam=cfg()['probe']['l2_lambda'];value,g,h=objective(theta,a,y,lam)
    delta=1e-5;numeric_g=[];numeric_h=[]
    for i in range(len(theta)):
        e=np.zeros(len(theta));e[i]=delta
        vp,gp=objective(theta+e,a,y,lam,False);vm,gm=objective(theta-e,a,y,lam,False)
        numeric_g.append((vp-vm)/(2*delta));numeric_h.append((gp-gm)/(2*delta))
    np.testing.assert_allclose(g,numeric_g,rtol=0,atol=1e-9)
    np.testing.assert_allclose(h,np.asarray(numeric_h).T,rtol=0,atol=1e-9)
    assert np.linalg.eigvalsh(h).min()>0
    z=np.zeros(5);z[-1]=1.3
    v0,g0=objective(z,a,y,0,False);v1,g1=objective(z,a,y,lam,False)
    assert v0==v1 and g0[-1]==g1[-1]
    tests.append('analytic BCE/ridge gradient and Hessian versus central differences; unpenalized intercept')
    fitted,trace=fit_newton(x,y,cfg()['probe']);v,gg,hh=objective(fitted,a,y,lam)
    assert np.max(np.abs(gg))<=cfg()['probe']['gradient_infinity_tolerance'] and v<=trace[0]['objective']
    assert all(b['objective']<=aa['objective']+1e-13 for aa,b in zip(trace,trace[1:]))
    constant=np.ones((83,3));cx,cm,cs=normalize_train(constant)
    np.testing.assert_array_equal(cx,np.zeros_like(cx));np.testing.assert_array_equal(cs,np.full(3,1e-6))
    ct,ct_trace=fit_newton(cx,y,cfg()['probe']);assert ct_trace[-1]['iteration']==0 and np.all(ct[:-1]==0)
    assert abs(probability(ct[-1])-y.mean())<1e-14
    np.testing.assert_allclose((f-mean)/sd,x,rtol=0,atol=0)
    tests.append('deterministic solver convergence and monotonic objective; constant-feature SD floor and probability baseline')
    ref=read(OUT/'backbones.json')[0];model,state=load_backbone(ref);values,tr,labels=training_data(ref['cutoff'])
    values={k:t[:64] for k,t in values.items()};before=object_hash(model.state_dict())
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();features,native=extract_features(model,values)
    with torch.inference_mode():out,_=legacy.predict(model,values,False)
    np.testing.assert_array_equal(native,out.cpu().numpy())
    assert object_hash(model.state_dict())==before and not any(p.requires_grad for p in model.parameters())
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    np.testing.assert_array_equal(labels,previous.raw_direction(pd.read_csv(OUT/'observation_table.csv').exec_return.iloc[tr]))
    tests.append('real frozen model feature/readout reconstruction, labels, weight/RNG immutability and disabled gradients')
    result=dict(status='PASS',tests=tests,completed_utc=now(),source_sha256=source_hashes(),protocol_sha256=sha(ROOT/'protocol.json'))
    save(OUT/'contract_verification.json',result);print(json.dumps(dict(status='PASS',tests=tests)),flush=True)

if __name__=='__main__':main()

from common16 import *


def main():
    legacy.initialize();check_frozen(False);tests=[]
    t=np.arange(200,dtype=float);close=np.exp(.001*t+4);bars=np.column_stack([close*.999,close*1.01,close*.99,close,1000+2*t])
    anchors=np.array([124,150,175]);actual=raw_features(bars,anchors);expected=scalar_features(bars,anchors)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
    for i,h in enumerate(HORIZONS):
        np.testing.assert_allclose(actual[:,5*i],.001*h,rtol=0,atol=1e-14)
        np.testing.assert_allclose(actual[:,5*i+1],0.,rtol=0,atol=1e-14)
    for anchor in anchors:
        changed=bars.copy();changed[anchor+1:,:4]*=7;changed[anchor+1:,4]+=1e9
        np.testing.assert_array_equal(raw_features(changed,[anchor]),raw_features(bars,[anchor]))
    tests.append('raw25 vector/scalar and known-path formulas; future perturbation invariance')
    obs,price,returns=data();tr,_=indices(obs,cfg()['folds'][0]);f=raw_features(raw_bars(price),obs.anchor.iloc[tr].to_numpy())
    np.testing.assert_allclose(f,scalar_features(raw_bars(price),obs.anchor.iloc[tr].to_numpy()),rtol=0,atol=1e-14)
    x,mean,sd=normalize_train(f);np.testing.assert_allclose(x,(f-mean)/sd,rtol=0,atol=0)
    # Synthetic solver problem; no actual experiment head fit in the contract.
    rng=np.random.default_rng(20261600);synthetic=rng.normal(size=(71,25));y=(rng.random(71)>.48).astype(float)
    theta,trace=fit_newton(synthetic,y,cfg()['probe']);v,g,h=objective(theta,design(synthetic),y,.01)
    assert np.max(np.abs(g))<=1e-9 and np.linalg.eigvalsh(h).min()>0
    assert all(b['objective']<=a['objective']+1e-13 for a,b in zip(trace,trace[1:]))
    tests.append('real causal raw training rows, normalization and inherited deterministic solver convergence')
    ref=read(OUT/'backbones.json')[0];model,state,values,f,native=extract_learned(ref,tr[:64]);before=object_hash(model.state_dict())
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    with torch.inference_mode():direct,_=legacy.predict(model,values,False)
    np.testing.assert_array_equal(native,direct.cpu().numpy())
    archival=previous.archival_native_output(model,values);model.requires_grad_(True)
    with torch.inference_mode():original,_=legacy.predict(model,values,False)
    model.requires_grad_(False);np.testing.assert_array_equal(archival,original.cpu().numpy())
    assert before==object_hash(model.state_dict())==ref['model_sha256'] and not any(p.requires_grad or p.grad is not None for p in model.parameters())
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    tests.append('early frozen backbone native/feature reconstruction, archival compatibility and tensor/RNG invariance')
    for fold in cfg()['folds']:
        tr,te=indices(obs,fold);assert obs.joint_completed.iloc[tr].le(fold['cutoff']).all() and obs.date.iloc[te].gt(fold['cutoff']).all()
    for h in read(OUT/'reused_heads.json'):
        d=load_features(h);_,g,_=objective(np.asarray(h['coefficients']),design(d['standardized']),d['direction'],.01)
        assert np.max(np.abs(g))<=1e-9 and h['l2_lambda']==.01
    tests.append('six exact maturity masks and nine retained converged heads under the identical objective')
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes()))
    print(json.dumps(dict(status='PASS',tests=tests)),flush=True)


if __name__=='__main__':main()

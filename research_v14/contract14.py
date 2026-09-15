from common14 import *

def main():
    legacy.initialize();check_frozen(False);tests=[]
    obs=pd.read_csv(OUT/'observation_table.csv');meta=read(OUT/'training_metadata.json')
    with np.load(V5/'cache/targets.npz') as raw,np.load(V5/'cache/packed_raw.npz') as packed:
        for fold in cfg()['folds']:
            cutoff=fold['cutoff'];tr=np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy())
            with np.load(CACHE/f'training_{cutoff}.npz') as d:
                np.testing.assert_array_equal(d['row_index'],tr)
                np.testing.assert_array_equal(d['direction'],(raw['returns'][tr]>0).astype(np.float32))
                for k in ['patches','geometry','valid']:np.testing.assert_array_equal(d[k],packed[k][tr])
                y=raw['auxiliary'][tr].astype(float);m=y.mean(axis=0);s=np.maximum(y.std(axis=0),1e-6)
                np.testing.assert_array_equal(d['auxiliary'],((y-m)/s).astype(np.float32))
                assert meta[cutoff]['frequency']==float((raw['returns'][tr]>0).mean())
            with np.load(CACHE/f'validation_{cutoff}.npz') as d:
                te=np.flatnonzero((obs.date.gt(cutoff)&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')).to_numpy())
                np.testing.assert_array_equal(d['row_index'],te)
                for k in ['patches','geometry','valid']:np.testing.assert_array_equal(d[k],packed[k][te])
    np.testing.assert_array_equal(raw_direction([-.001,0,.001]),[0,0,1])
    assert raw_direction([.001])[0]==1 and ((.001-.002)/.01)<0
    tests.append('raw-return class sign, zero ties, causal train/validation masks, exact packed inputs and auxiliary scales')
    z=np.array([-100.,-3.,0.,2.,100.]);y=np.array([0.,1.,1.,0.,1.])
    t=torch.tensor(z,dtype=torch.float64,requires_grad=True);target=torch.tensor(y,dtype=torch.float64)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(t,target);loss.backward()
    assert abs(float(loss.detach())-float(np.mean(np.logaddexp(0,z)-y*z)))<1e-13
    np.testing.assert_allclose(t.grad.numpy(),(probability(z)-y)/len(y),rtol=0,atol=1e-14)
    probabilities=probability(np.array([-3.,0.,1.]));assert abs(probabilities.mean()-probability(np.array([-3.,0.,1.]).mean()))>.01
    assert probability(0.)==.5 and not bool(probability(0.)>.5)
    tests.append('stable BCE/logistic formula and analytic gradient; probability-mean ensemble and strict threshold')
    cutoff=cfg()['folds'][0]['cutoff'];seed=cfg()['seeds'][0];frequency=meta[cutoff]['frequency']
    old=legacy.make_model('combined',seed);shared=shared_initial_hash(old);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    new=make_classifier(seed,frequency);assert shared_initial_hash(new)==shared
    assert torch.equal(torch.get_rng_state(),cpu) and torch.equal(torch.cuda.get_rng_state(),cuda)
    assert torch.count_nonzero(new.return_head.weight)==0
    assert abs(float(probability(float(new.return_head.bias.detach().cpu()[0])))-frequency)<1e-8
    tests.append('identical original shared initial tensors and random states; causal frequency-bias initialization')
    values,labels,tr,_=load_training(cutoff);idx=np.arange(128)
    a=make_classifier(seed,frequency);oa=prior.optimizer_for(a);a.train();update_step(a,oa,values,labels,idx)
    weight_hash=object_hash(a.state_dict());optimizer_hash=object_hash(oa.state_dict());rng=torch.cuda.get_rng_state().clone()
    b=make_classifier(seed,frequency);ob=prior.optimizer_for(b);b.train();batch=torch.tensor(idx,device='cuda')
    ob.zero_grad(set_to_none=True);logits,aux=legacy.predict(b,{k:t[batch] for k,t in values.items()},True)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels['direction'][batch])+.1*((aux-labels['auxiliary'][batch])**2).mean()
    expected.backward();torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True);ob.step()
    assert object_hash(b.state_dict())==weight_hash and object_hash(ob.state_dict())==optimizer_hash
    assert torch.equal(rng,torch.cuda.get_rng_state())
    modes=[m.training for m in a.modules()];cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    training_loss(a,values,labels,True,20261400)
    assert modes==[m.training for m in a.modules()] and object_hash(a.state_dict())==weight_hash
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    tests.append('real-model BCE/auxiliary update and AdamW parity; dropout audit preserves weights, modes and random states')
    g=pd.DataFrame(dict(actual_up=[0,0,1,1],direction_up=[False,True,False,True],score=[.1,.6,.4,.9],probability=[.1,.6,.4,.9]))
    m=metric(g);assert m['accuracy']==.5 and m['auroc']==.75 and m['tp']==m['fp']==m['tn']==m['fn']==1
    assert auc([0,0,0,0],[0,0,1,1])==.5
    assert abs(m['brier']-.185)<1e-14
    tests.append('known confusion matrix, tied-rank AUROC and probability-score metric example')
    result=dict(status='PASS',tests=tests,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),completed_utc=now())
    save(OUT/'contract_verification.json',result);print(json.dumps(dict(status='PASS',tests=tests)),flush=True)

if __name__=='__main__':main()

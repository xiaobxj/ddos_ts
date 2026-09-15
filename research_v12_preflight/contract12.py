"""Real-data contract: exact legacy updates and full stochastic optimizer-state restoration."""
from common12 import *
import io


def main():
    legacy.initialize();check_frozen(contract=False);checks=[]
    obs=pd.read_csv(V10/'results/observation_table.csv')
    for fold in cfg()['folds']:
        values,labels,tr,scales=load_training(fold['cutoff'])
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(fold['cutoff']).to_numpy()))
        expected,scale=legacy.standardized_targets(tr);assert scale==scales
        assert all(torch.equal(labels[k],expected[k]) for k in labels)
        original=legacy.batch_tensors(tr)
        assert all(torch.equal(values[k],original[k]) for k in values)
        validation,te=load_validation(fold['cutoff'])
        _,oldte=legacy.fold_indices(obs,fold);np.testing.assert_array_equal(te,oldte)
        expected_values=legacy.batch_tensors(te)
        assert all(torch.equal(validation[k],expected_values[k]) for k in validation)
        assert not len(np.intersect1d(tr,te))
        del values,labels,expected,original,validation,expected_values
    checks.append('Exact causal training arrays, mature scales and unchanged validation masks/features')
    values,labels,indices,_=load_training(cfg()['folds'][0]['cutoff'])
    values={k:v[:129] for k,v in values.items()};labels={k:v[:129] for k,v in labels.items()};indices=indices[:129]
    seed=cfg()['seeds'][0];fingerprints=[]
    for use_literal_legacy in [True,False]:
        model=legacy.make_model('combined',seed);optimizer=optimizer_for(model);rng=np.random.default_rng(seed)
        for epoch in range(2):
            model.train();order=rng.permutation(129)
            for positions in batches(order):
                if use_literal_legacy:
                    batch=torch.tensor(positions,device='cuda');v={k:t[batch] for k,t in values.items()}
                    optimizer.zero_grad(set_to_none=True);output,aux=legacy.predict(model,v,True)
                    loss=((output-labels['returns'][batch])**2).mean()
                    loss=loss+.1*((aux-labels['auxiliary'][batch])**2).mean()
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                else:update_step(model,optimizer,values,labels,positions)
        current=capture(model,optimizer,rng)
        fingerprints.append({k:current[k] for k in ['model_sha256','optimizer_sha256','rng_sha256']})
    assert fingerprints[0]==fingerprints[1]
    checks.append('Exact real-model MSE, clipping, optimizer and RNG parity with literal legacy updates')
    # Serialize a nontrivial training boundary entirely in memory; no fixture checkpoints are left behind.
    model=legacy.make_model('combined',seed);optimizer=optimizer_for(model);rng=np.random.default_rng(seed)
    job=dict(schedule='contract',cutoff='2017-12-31',seed=seed)
    run_epoch(model,optimizer,rng,values,labels,indices,1,job)
    snapshot=capture(model,optimizer,rng)
    memory=io.BytesIO();torch.save(snapshot,memory)
    run_epoch(model,optimizer,rng,values,labels,indices,2,job)
    uninterrupted=capture(model,optimizer,rng)
    memory.seek(0);loaded=torch.load(memory,map_location='cpu',weights_only=True)
    resumed,opt2,rng2,_=restore_payload(loaded,seed)
    before=capture(resumed,opt2,rng2)
    modes=[m.training for m in resumed.modules()]
    first=training_loss(resumed,values,labels,True,20261200)
    second=training_loss(resumed,values,labels,True,20261200)
    assert first==second and modes==[m.training for m in resumed.modules()]
    after=capture(resumed,opt2,rng2)
    assert all(before[k]==after[k] for k in ['model_sha256','optimizer_sha256','rng_sha256'])
    run_epoch(resumed,opt2,rng2,values,labels,indices,2,job)
    restored=capture(resumed,opt2,rng2)
    assert all(restored[k]==uninterrupted[k] for k in ['model_sha256','optimizer_sha256','rng_sha256'])
    checks.append('Serialized model/moments/counters/RNG exactly match uninterrupted dropout training')
    checks.append('Fixed-state dropout audits are repeatable and leave training state unchanged')
    candidate,opt3,rng3,_=restore_payload(loaded,seed)
    moment_hash=object_hash(opt3.state_dict()['state'])
    original_groups=cpu_copy(opt3.state_dict()['param_groups'])
    opt3.param_groups[0]['lr']=.0001
    changed=cpu_copy(opt3.state_dict()['param_groups']);changed[0]['lr']=.001
    assert changed==original_groups and object_hash(opt3.state_dict()['state'])==moment_hash
    for fold in cfg()['folds']:
        with np.load(CACHE/f'training_{fold["cutoff"]}.npz') as data:n=len(data['row_index'])
        orders=[]
        for schedule in cfg()['schedules']:
            r=np.random.default_rng(seed);orders.append([array_hash(r.permutation(n)) for epoch in range(40)])
        assert orders[0]==orders[1]
    checks.append('Only learning rate changes at branch boundary; moments and complete permutation budgets match')
    check_frozen(contract=False)
    result=dict(status='PASS',checks=checks,checks_passed=len(checks),legacy_update_fingerprints=fingerprints[0],
        source_sha256=source_hashes(),protocol_sha256=sha(ROOT/'protocol.json'),validation_predictions=0,
        completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT/'contract_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

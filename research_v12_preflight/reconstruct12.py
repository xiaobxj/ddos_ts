"""Recover nine exact epoch20 model/optimizer/RNG states without validation forward passes."""
from common12 import *


def main():
    legacy.initialize();check_frozen()
    assert not (OUT/'reconstruction_manifest.json').exists(),'Preserve previous reconstruction'
    start=time.time();run=manifest('shared_prefix_reconstruction');save(OUT/'reconstruction_manifest.json',run)
    directory=OUT/'prefix_checkpoints';directory.mkdir(exist_ok=True)
    refs=[];curves=[];monitors=[];draws=[];parity=[]
    for number,old in enumerate(read(OUT/'archived_models.json'),1):
        job=dict(cutoff=old['cutoff'],seed=old['seed'],schedule='prefix20')
        values,labels,tr,scales=load_training(job['cutoff'])
        model=legacy.make_model('combined',job['seed']);optimizer=optimizer_for(model);rng=np.random.default_rng(job['seed'])
        monitors.append(dict(**job,epoch=0,**training_loss(model,values,labels)))
        for epoch in range(1,21):
            curves.append(run_epoch(model,optimizer,rng,values,labels,tr,epoch,job))
            if epoch in cfg()['training']['monitor_epochs']:
                monitors.append(dict(**job,epoch=epoch,**training_loss(model,values,labels)))
        source=torch.load(PROJECT/old['project_file'],map_location='cpu',weights_only=True)
        current=cpu_copy(model.state_dict())
        mismatches=[name for name,v in current.items() if not torch.equal(v,source['state_dict'][name])]
        maximum=max(float((v-source['state_dict'][name]).abs().max()) for name,v in current.items())
        match=not mismatches and scales==source['scales']
        parity.append(dict(**job,exact=match,model_sha256=model_hash(model),archived_model_sha256=old['model_sha256'],
                           maximum_weight_difference=maximum,mismatched_tensors=mismatches))
        save(OUT/'reconstruction_parity.json',parity)
        assert match,(identity(job),maximum,mismatches)
        assert model_hash(model)==old['model_sha256']
        draws.extend(dict(**job,epoch=20,**r) for r in training_audit(model,values,labels))
        state=capture(model,optimizer,rng,**job,epoch=20,arm='combined',history='full',scales=scales,train_n=len(tr),
            protocol_sha256=sha(ROOT/'protocol.json'),reconstructed_from=old['project_file'],archived_sha256=old['sha256'])
        path=directory/(identity(job)+'.pt');torch.save(state,path);refs.append(checkpoint_ref(path,state))
        pd.DataFrame(curves).to_csv(OUT/'prefix_training_curves.csv',index=False)
        pd.DataFrame(monitors).to_csv(OUT/'prefix_training_monitors.csv',index=False)
        pd.DataFrame(draws).to_csv(OUT/'prefix_training_loss_draws.csv',index=False)
        save(OUT/'prefix_models.json',refs)
        print(f'Exact reconstruction {number}/9; {job["cutoff"]}, seed {job["seed"]}; {time.time()-start:.1f}s',flush=True)
        del model,optimizer,values,labels,source,current,state;torch.cuda.empty_cache()
    check_frozen()
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),exact_matches=9,shared_fits=9,epochs=180,
               checkpoints=9,new_validation_predictions=0,elapsed_seconds=time.time()-start)
    save(OUT/'reconstruction_manifest.json',run)


if __name__=='__main__':main()

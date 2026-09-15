"""Two fixed schedules restored from identical full epoch20 states; no validation passes."""
from common12 import *


def main():
    legacy.initialize();check_frozen()
    previous=read(OUT/'reconstruction_manifest.json')
    assert previous.get('finished_utc') and previous['exact_matches']==9
    assert not (OUT/'continuation_manifest.json').exists(),'Preserve previous continuation'
    start=time.time();run=manifest('paired_continuations');save(OUT/'continuation_manifest.json',run)
    directory=OUT/'final_checkpoints';directory.mkdir(exist_ok=True)
    refs=[];curves=[];monitors=[];draws=[];starts=[]
    prefixes=read(OUT/'prefix_models.json')
    for number,job in enumerate(read(OUT/'continuation_plan.json'),1):
        ref=next(r for r in prefixes if r['cutoff']==job['cutoff'] and r['seed']==job['seed'])
        values,labels,tr,scales=load_training(job['cutoff'])
        model,optimizer,rng,source=restore(ref)
        start_state=capture(model,optimizer,rng)
        assert start_state['rng_sha256']==source['rng_sha256']
        starts.append(dict(**meta(job),prefix_project_file=ref['project_file'],prefix_sha256=ref['sha256'],
            initial_model_sha256=start_state['model_sha256'],initial_optimizer_sha256=start_state['optimizer_sha256'],
            initial_rng_sha256=start_state['rng_sha256']))
        for group in optimizer.param_groups:group['lr']=job['lr']
        for epoch in range(21,41):
            curves.append(run_epoch(model,optimizer,rng,values,labels,tr,epoch,job))
            if epoch in cfg()['training']['monitor_epochs']:
                monitors.append(dict(**meta(job),epoch=epoch,**training_loss(model,values,labels)))
        draws.extend(dict(**meta(job),epoch=40,**r) for r in training_audit(model,values,labels))
        state=capture(model,optimizer,rng,**meta(job),epoch=40,arm='combined',history='full',scales=scales,train_n=len(tr),
            protocol_sha256=sha(ROOT/'protocol.json'),prefix_project_file=ref['project_file'],prefix_sha256=ref['sha256'],
            initial_model_sha256=start_state['model_sha256'],initial_optimizer_sha256=start_state['optimizer_sha256'],
            initial_rng_sha256=start_state['rng_sha256'],learning_rate=job['lr'])
        path=directory/(identity(job)+'.pt');torch.save(state,path);refs.append(checkpoint_ref(path,state))
        pd.DataFrame(curves).to_csv(OUT/'continuation_training_curves.csv',index=False)
        pd.DataFrame(monitors).to_csv(OUT/'continuation_training_monitors.csv',index=False)
        pd.DataFrame(draws).to_csv(OUT/'continuation_training_loss_draws.csv',index=False)
        save(OUT/'final_models.json',refs);save(OUT/'paired_starting_states.json',starts)
        print(f'Continuation {number}/18; {identity(job)}; {time.time()-start:.1f}s',flush=True)
        del model,optimizer,values,labels,source,state,start_state;torch.cuda.empty_cache()
    check_frozen()
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),continuations=18,checkpoints=18,epochs=360,
               new_validation_predictions=0,elapsed_seconds=time.time()-start)
    save(OUT/'continuation_manifest.json',run)


if __name__=='__main__':main()

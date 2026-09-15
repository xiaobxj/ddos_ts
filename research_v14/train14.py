from common14 import *

def main():
    legacy.initialize();check_frozen();path=OUT/'training_manifest.json';assert not path.exists(),'Preserve training run'
    run=manifest('training');save(path,run);start=time.time();directory=OUT/'checkpoints';directory.mkdir(exist_ok=True)
    refs=[];curves=[];audits=[];initial=[]
    for fold in cfg()['folds']:
        cutoff=fold['cutoff'];values,labels,tr,metadata=load_training(cutoff)
        for seed in cfg()['seeds']:
            model=make_classifier(seed,metadata['frequency']);optimizer=prior.optimizer_for(model);rng=np.random.default_rng(seed)
            initial.append(dict(cutoff=cutoff,seed=seed,shared_initial_sha256=shared_initial_hash(model),
                initial_model_sha256=object_hash(model.state_dict()),initial_bias=float(model.return_head.bias.detach().cpu()[0]),
                initial_cpu_rng_sha256=object_hash(torch.get_rng_state()),initial_cuda_rng_sha256=object_hash(torch.cuda.get_rng_state())))
            for epoch in range(1,21):
                model.train();order=rng.permutation(len(tr));batches=prior.batches(order);sums=np.zeros(3);norms=[]
                for batch in batches:
                    joint,bce,aux,norm=update_step(model,optimizer,values,labels,batch)
                    sums+=np.array([joint,bce,aux])*len(batch);norms.append(norm)
                curves.append(dict(cutoff=cutoff,seed=seed,epoch=epoch,train_n=len(tr),presentations=len(tr),steps=len(batches),
                    last_batch_n=len(batches[-1]),order_sha256=array_hash(order),boundary_sha256=object_hash([array_hash(b) for b in batches]),
                    online_joint_loss=float(sums[0]/len(tr)),online_bce=float(sums[1]/len(tr)),online_auxiliary_mse=float(sums[2]/len(tr)),
                    gradient_norm_mean=float(np.mean(norms)),clipped_fraction=float(np.mean(np.asarray(norms)>1)),lr=.001,weight_decay=.1))
            state=prior.capture(model,optimizer,rng,arm='combined',history='full',method='direction_bce',head_semantics='up_logit',
                cutoff=cutoff,seed=seed,epoch=20,train_n=len(tr),training_metadata=metadata,protocol_sha256=sha(ROOT/'protocol.json'))
            p=directory/f'direction_bce_{cutoff}_{seed}_epoch20.pt';torch.save(state,p)
            refs.append(dict(method='direction_bce',cutoff=cutoff,seed=seed,epoch=20,project_file=str(p.relative_to(PROJECT)),sha256=sha(p),
                model_sha256=state['model_sha256'],optimizer_sha256=state['optimizer_sha256'],rng_sha256=state['rng_sha256']))
            for audit in training_audit(model,values,labels):audits.append(dict(cutoff=cutoff,seed=seed,**audit))
            save(OUT/'models.json',refs);save(OUT/'initial_states.json',initial)
            pd.DataFrame(curves).to_csv(OUT/'training_curves.csv',index=False)
            pd.DataFrame(audits).to_csv(OUT/'training_loss_draws.csv',index=False)
            print(f'Direction BCE {len(refs)}/9: {cutoff} seed {seed}; {time.time()-start:.1f}s',flush=True)
            del model,optimizer;torch.cuda.empty_cache()
        del values,labels;torch.cuda.empty_cache()
    assert len(refs)==9 and len(curves)==180 and len(audits)==81;check_frozen()
    run.update(finished_utc=now(),elapsed_seconds=time.time()-start,fits=9,epochs=180,training_loss_passes=81,
        validation_during_training=False,artifacts={n:sha(OUT/n) for n in ['models.json','initial_states.json','training_curves.csv','training_loss_draws.csv']})
    save(path,run);print('Training complete: 9 fits, 180 epochs, 81 training loss passes; no validation scoring.',flush=True)

if __name__=='__main__':main()

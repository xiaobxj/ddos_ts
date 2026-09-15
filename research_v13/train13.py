from common13 import *

def main():
    legacy.initialize();check_frozen();path=OUT/'training_manifest.json'
    assert not path.exists(),'Preserve prior run'
    run=manifest('training');save(path,run);start=time.time()
    checkpoints=OUT/'inner_checkpoints';checkpoints.mkdir(exist_ok=True)
    refs=[];curves=[];losses=[]
    for cutoff in cfg()['new_training_cutoffs']:
        values,labels,tr,scales=load_training(cutoff)
        for seed in cfg()['seeds']:
            model=legacy.make_model('combined',seed);optimizer=prior.optimizer_for(model);rng=np.random.default_rng(seed)
            for epoch in range(1,21):
                model.train();order=rng.permutation(len(tr));online=0.;norms=[]
                batches=prior.batches(order)
                for batch in batches:
                    loss,ret,aux,norm=prior.update_step(model,optimizer,values,labels,batch)
                    online+=loss*len(batch);norms.append(norm)
                curves.append(dict(cutoff=cutoff,seed=seed,epoch=epoch,train_n=len(tr),presentations=len(tr),
                    steps=len(batches),last_batch_n=len(batches[-1]),order_sha256=array_hash(order),
                    boundary_sha256=object_hash([array_hash(b) for b in batches]),online_joint_loss=online/len(tr),
                    clipped_fraction=float(np.mean(np.array(norms)>1)),lr=.001,weight_decay=.1))
            state=prior.capture(model,optimizer,rng,arm='combined',history='full',epoch=20,cutoff=cutoff,seed=seed,
                train_n=len(tr),full_train_n=len(tr),scales=scales,protocol_sha256=sha(ROOT/'protocol.json'))
            p=checkpoints/f'combined_full_{cutoff}_{seed}_epoch20.pt';torch.save(state,p)
            ref=dict(cutoff=cutoff,seed=seed,epoch=20,project_file=str(p.relative_to(PROJECT)),sha256=sha(p),
                model_sha256=state['model_sha256'],optimizer_sha256=state['optimizer_sha256'],rng_sha256=state['rng_sha256'])
            refs.append(ref);losses.append(dict(cutoff=cutoff,seed=seed,**prior.training_loss(model,values,labels)))
            save(OUT/'inner_models.json',refs)
            pd.DataFrame(curves).to_csv(OUT/'training_curves.csv',index=False)
            pd.DataFrame(losses).to_csv(OUT/'final_training_losses.csv',index=False)
            print(f'Inner MSE20 {len(refs)}/9: {cutoff} seed {seed}; {time.time()-start:.1f}s',flush=True)
            del model,optimizer;torch.cuda.empty_cache()
        del values,labels;torch.cuda.empty_cache()
    assert len(refs)==9 and len(curves)==180
    check_frozen();run.update(finished_utc=now(),elapsed_seconds=time.time()-start,fits=9,epochs=180,
        rolling_or_outer_predictions_during_fit=0,artifacts={n:sha(OUT/n) for n in ['inner_models.json','training_curves.csv','final_training_losses.csv']})
    save(path,run);print('Training complete: 9 fits, 180 epochs, no forecast scoring.',flush=True)

if __name__=='__main__':main()

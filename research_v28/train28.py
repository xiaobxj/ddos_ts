from common28 import *
def main():
    legacy.initialize();check_frozen();assert not (OUT/'training_manifest.json').exists();run=manifest('training');save(OUT/'training_manifest.json',run);started=time.time()
    obs,price,targets=data();models=[];heads=[];curves=[];traces=[];metrics=[];files=[];scales=read(OUT/'training_scales.json');cp=OUT/'checkpoints';cp.mkdir(exist_ok=True)
    for history in MEMORIES:
        for fold in cfg()['folds']:
            cutoff=fold['cutoff'];values,labels,tr=load_training(history,cutoff);np.testing.assert_array_equal(tr,indices(obs,fold,history)[0]);mf=market(price,obs,tr);u=gates(price,obs,tr);y=(targets['returns'][tr]>0).astype(float);n=len(tr)
            for seed in cfg()['seeds']:
                model=legacy.make_model('combined',seed);assert sum(p.numel() for p in model.parameters())==38551
                optimizer=training.optimizer_for(model);rng=np.random.default_rng(seed)
                for epoch in range(1,21):
                    model.train();permutation=schedule(len(tr),n,rng);chunks=training.batches(permutation);sums=np.zeros(3);norms=[]
                    for batch in chunks:
                        loss,ret,aux,norm=training.update_step(model,optimizer,values,labels,batch);sums+=np.array([loss,ret,aux])*len(batch);norms.append(norm)
                    counts=np.bincount(permutation,minlength=len(tr));curves.append(dict(history=history,cutoff=cutoff,seed=seed,epoch=epoch,train_n=len(tr),presentations=n,optimizer_steps=len(chunks),last_batch_n=len(chunks[-1]),minimum_repetitions=int(counts.min()),maximum_repetitions=int(counts.max()),order_sha256=training.array_hash(permutation),joint_loss=sums[0]/n,return_loss=sums[1]/n,auxiliary_loss=sums[2]/n,clipped_fraction=float((np.array(norms)>1).mean())))
                    if epoch%10==0:print(f'Training {len(models)+1}/138 {history} {cutoff} seed {seed}: natural epoch {epoch}/20; elapsed {time.time()-started:.1f}s',flush=True)
                state=training.capture(model,optimizer,rng,arm='combined',history=history,epoch=20,cutoff=cutoff,seed=seed,train_n=len(tr),full_train_n=fold['train_n'],scales=scales[f'{history}_{cutoff}'],protocol_sha256=sha(ROOT/'protocol.json'))
                p=cp/f'combined_{history}_{cutoff}_{seed}_epoch20.pt';torch.save(state,p);files.append(p)
                r=dict(history=history,cutoff=cutoff,seed=seed,epoch=20,train_n=len(tr),full_train_n=fold['train_n'],project_file=str(p.relative_to(PROJECT)),sha256=sha(p),model_sha256=state['model_sha256'],optimizer_sha256=state['optimizer_sha256'],rng_sha256=state['rng_sha256']);models.append(r)
                optimizer.zero_grad(set_to_none=True);model.eval();model.requires_grad_(False);features,native=neural.extract_features(model,values)
                d,coefs,ts,ms=fit_pipeline(features,mf,u,y);d.update(row_index=tr,native_standardized=native);p=CACHE/f'pipeline_{history}_{cutoff}_{seed}.npz';np.savez_compressed(p,**d);files.append(p)
                for method,theta,trace,m in zip(LEARNED,coefs,ts,ms):
                    h=dict(history=history,method=method,cutoff=cutoff,seed=seed,job=f'{history}_{method}_{cutoff}_{seed}',coefficients=theta.tolist(),train_n=len(tr),iterations=trace[-1]['iteration'],model_project_file=r['project_file'],**ref(p),**m);heads.append(h)
                    traces.extend(dict(job=h['job'],**item) for item in trace);metrics.append(dict(job=h['job'],history=history,method=method,cutoff=cutoff,seed=seed,train_n=len(tr),iterations=h['iterations'],**m))
                save(OUT/'models.json',models);save(OUT/'heads.json',heads);pd.DataFrame(curves).to_csv(OUT/'training_curves.csv',index=False);pd.DataFrame(traces).to_csv(OUT/'solver_trace.csv',index=False)
                print(f'Completed {len(models)}/138 backbones and {len(heads)}/552 heads; no new weekly scoring.',flush=True)
                del model,optimizer,features,d,state;torch.cuda.empty_cache()
            del values,labels;torch.cuda.empty_cache()
    assert len(models)==138 and len(heads)==552 and len(curves)==2760 and sum(r['optimizer_steps'] for r in curves)==cfg()['budget']['new_optimizer_steps']
    pd.DataFrame(metrics).to_csv(OUT/'training_metrics.csv',index=False);files += [OUT/n for n in ['models.json','heads.json','training_curves.csv','solver_trace.csv','training_metrics.csv']]
    check_frozen();finish(run,files,elapsed_seconds=time.time()-started,neural_fits=138,natural_epochs=2760,optimizer_steps=cfg()['budget']['new_optimizer_steps'],head_fits=552,newton_iterations=sum(h['iterations'] for h in heads),projections=276,unique_training_feature_rows=sum(r['train_n'] for r in models),training_presentations=sum(r['presentations'] for r in curves),next_year_scoring_during_fit=False,all_converged=True)
    print('Training PASS:138backbones and552heads frozen before new scoring.',flush=True)
if __name__=='__main__':main()

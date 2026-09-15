"""Single worker for the nine prespecified balanced-batch fits."""
from common9 import *


def main():
    legacy.initialize();prep=check_frozen();start=time.time()
    probe=json.loads((OUT/'probe_manifest.json').read_text(encoding='utf-8'))
    assert probe.get('finished_utc') and probe['source_sha256']==source_hashes()
    destination=OUT/'training_manifest.json';assert not destination.exists(),'Preserve completed work'
    run=manifest('balanced_training');save(destination,run)
    directory=OUT/'balanced_checkpoints';directory.mkdir(exist_ok=True)
    settings=cfg()['training'];obs=pd.read_csv(OUT/'observation_table.csv')
    records=[];curves=[];refs=[]
    for number,job in enumerate(fit_plan(),1):
        tr,te=load_fold(job['cutoff']);n=len(tr)
        values=legacy.batch_tensors(tr);testing=legacy.batch_tensors(te)
        labels,scales=legacy.standardized_targets(tr)
        model=legacy.make_model('combined',job['seed'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        rng=np.random.default_rng(job['seed'])
        for epoch in range(1,21):
            model.train();order=legacy.epoch_order(n,n,rng);chunks=batches(order,'balanced')
            running=0.;norms=[]
            for positions in chunks:
                b=torch.tensor(positions,device='cuda');optimizer.zero_grad(set_to_none=True)
                p,a=legacy.predict(model,{k:t[b] for k,t in values.items()},True)
                loss=batch_loss(p,a,labels,b,n,'balanced');assert torch.isfinite(loss)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();norms.append(float(norm));running+=float(loss.detach())
            row=dict(**job,epoch=epoch,train_n=n,presentations=n,optimizer_steps=len(chunks),
                training_order_sha256=array_hash(tr[order]),batch_boundaries_sha256=boundary_hash(chunks),
                minimum_batch=min(len(b) for b in chunks),maximum_batch=max(len(b) for b in chunks),
                online_joint_loss=running/len(chunks),raw_gradient_norm_mean=float(np.mean(norms)),
                raw_gradient_norm_max=float(max(norms)),gradient_clipping_fraction=float(np.mean(np.asarray(norms)>1.)))
            if epoch==20:
                model.eval()
                with torch.inference_mode():
                    p,_=legacy.predict(model,testing,False)
                    altered,_=legacy.predict(model,{k:torch.roll(t,1,dims=0) for k,t in testing.items()},False)
                predictions=p.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
                shifted=altered.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
                for i,p,q in zip(te,predictions,shifted):
                    records.append(dict(**job,row_index=int(i),date=obs.date.iloc[i],actual=float(obs.exec_return.iloc[i]),
                        predicted_return=float(p),reassigned_prediction=float(q),training_mean=scales['returns_mean'],source='new_round9_fit'))
                train_mse,aux_mse=legacy.evaluate_training(model,values,labels)
                row.update(final_model_train_return_mse=train_mse,final_model_train_auxiliary_mse=aux_mse,
                           validation_mse=float(np.mean((predictions-obs.exec_return.iloc[te].to_numpy())**2)))
                path=directory/f'{job["cutoff"]}_{job["seed"]}_epoch20.pt'
                torch.save(dict(state_dict={k:t.detach().cpu() for k,t in model.state_dict().items()},**job,arm='combined',epoch=20,
                    scales=scales,train_n=n,protocol_sha256=sha(ROOT/'protocol.json')),path)
                refs.append(dict(**job,epoch=20,project_file=str(path.relative_to(PROJECT)),file=str(path.relative_to(ROOT)),sha256=sha(path)))
            curves.append(row)
        pd.DataFrame(records).to_csv(OUT/'balanced_predictions.csv',index=False)
        pd.DataFrame(curves).to_csv(OUT/'balanced_training_curves.csv',index=False)
        save(OUT/'balanced_checkpoint_manifest.json',refs)
        print(f'{number}/9 fits; {job["cutoff"]}, seed {job["seed"]}; {time.time()-start:.1f}s',flush=True)
        del model,optimizer,values,testing,labels;torch.cuda.empty_cache()
    check_frozen();assert old_evidence()==prep['old_evidence']
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),fits=9,checkpoints=len(refs),predictions=len(records),
               elapsed_seconds=time.time()-start,old_preserved=True)
    save(destination,run);print('Finished all nine fits.',flush=True)


if __name__=='__main__':main()

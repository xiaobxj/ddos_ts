"""Nine robust-return fits, preserving original data, batches, seeds and budget."""
from common10 import *


def main():
    legacy.initialize();prep=check_frozen();start=time.time()
    destination=OUT/'training_manifest.json';assert not destination.exists(),'Preserve previous run'
    run=manifest('huber_training');save(destination,run)
    directory=OUT/'huber_checkpoints';directory.mkdir(exist_ok=True)
    settings=cfg()['training'];obs=pd.read_csv(OUT/'observation_table.csv')
    rows=[];curves=[];refs=[]
    for number,job in enumerate(fit_plan(),1):
        tr,te=load_fold(job['cutoff']);n=len(tr)
        values=legacy.batch_tensors(tr);testing=legacy.batch_tensors(te);labels,scales=legacy.standardized_targets(tr)
        model=legacy.make_model('combined',job['seed'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        rng=np.random.default_rng(job['seed'])
        for epoch in range(1,21):
            model.train();order=legacy.epoch_order(n,n,rng);chunks=batches(order);running=0.;norms=[];linear=0
            for positions in chunks:
                b=torch.tensor(positions,device='cuda');optimizer.zero_grad(set_to_none=True)
                p,a=legacy.predict(model,{k:t[b] for k,t in values.items()},True)
                loss=joint_loss(p,a,labels,b);assert torch.isfinite(loss)
                linear+=int(((p-labels['returns'][b]).detach().abs()>1.).sum())
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();norms.append(float(norm));running+=float(loss.detach())*len(b)
            record=dict(**job,epoch=epoch,train_n=n,presentations=n,optimizer_steps=len(chunks),
                training_order_sha256=array_hash(tr[order]),batch_boundaries_sha256=boundary_hash(chunks),
                online_joint_loss=running/n,online_return_linear_fraction=linear/n,
                raw_gradient_norm_mean=float(np.mean(norms)),raw_gradient_norm_max=float(max(norms)),
                gradient_clipping_fraction=float(np.mean(np.asarray(norms)>1.)))
            if epoch==20:
                model.eval();robust=0.;count_linear=0
                with torch.inference_mode():
                    output,_=legacy.predict(model,testing,False)
                    shifted,_=legacy.predict(model,{k:torch.roll(t,1,dims=0) for k,t in testing.items()},False)
                    for lo in range(0,n,128):
                        p,_=legacy.predict(model,{k:t[lo:lo+128] for k,t in values.items()},False)
                        e=p-labels['returns'][lo:lo+128];robust+=float(robust_loss(e).sum());count_linear+=int((e.abs()>1.).sum())
                predictions=output.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
                altered=shifted.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
                for i,p,q in zip(te,predictions,altered):rows.append(dict(**job,row_index=int(i),date=obs.date.iloc[i],actual=float(obs.exec_return.iloc[i]),
                    predicted_return=float(p),reassigned_prediction=float(q),training_mean=scales['returns_mean'],source='new_round10_fit'))
                train_mse,aux_mse=legacy.evaluate_training(model,values,labels)
                record.update(final_model_train_return_mse=train_mse,final_model_train_auxiliary_mse=aux_mse,
                    final_model_train_scaled_huber=robust/n,final_model_train_linear_fraction=count_linear/n,
                    validation_mse=float(np.mean((predictions-obs.exec_return.iloc[te].to_numpy())**2)))
                path=directory/f'{job["cutoff"]}_{job["seed"]}_epoch20.pt'
                torch.save(dict(state_dict={k:t.detach().cpu() for k,t in model.state_dict().items()},**job,arm='combined',epoch=20,
                    scales=scales,train_n=n,protocol_sha256=sha(ROOT/'protocol.json'),loss='2*Huber(delta1)+0.1*aux_MSE'),path)
                refs.append(dict(**job,epoch=20,project_file=str(path.relative_to(PROJECT)),file=str(path.relative_to(ROOT)),sha256=sha(path)))
            curves.append(record)
        pd.DataFrame(rows).to_csv(OUT/'huber_predictions.csv',index=False)
        pd.DataFrame(curves).to_csv(OUT/'huber_training_curves.csv',index=False)
        save(OUT/'huber_checkpoint_manifest.json',refs)
        print(f'{number}/9 fits; {job["cutoff"]}, seed {job["seed"]}; {time.time()-start:.1f}s',flush=True)
        del model,optimizer,values,testing,labels;torch.cuda.empty_cache()
    check_frozen();assert old_evidence()==prep['old_evidence']
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),fits=9,checkpoints=9,predictions=len(rows),elapsed_seconds=time.time()-start,old_preserved=True)
    save(destination,run);print('Finished all nine robust fits.',flush=True)


if __name__=='__main__':main()

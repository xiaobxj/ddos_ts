"""Raw-only chronological validation of initialization and auxiliary objectives."""
from models5 import *


def evaluate_full(network,values,labels,auxiliary,batch_size=128):
    network.eval();sum_return=0.;sum_aux=0.;count=len(labels['returns'])
    with torch.inference_mode():
        for lo in range(0,count,batch_size):
            v={k:t[lo:lo+batch_size] for k,t in values.items()}
            out,aux=predict(network,v,auxiliary)
            sum_return+=float(((out-labels['returns'][lo:lo+batch_size])**2).sum())
            if auxiliary:sum_aux+=float(((aux-labels['auxiliary'][lo:lo+batch_size])**2).mean(dim=1).sum())
    return sum_return/count,sum_aux/count if auxiliary else None


def prediction_rows(obs,ids,variant,epoch,seed,cutoff,output,shifted,train_n):
    return [dict(variant=variant,epoch=epoch,seed=seed,cutoff=cutoff,train_n=train_n,
                 row_index=int(i),date=obs.date.iloc[i],actual=float(obs.exec_return.iloc[i]),
                 predicted_return=float(p),reassigned_prediction=float(s)) for i,p,s in zip(ids,output,shifted)]


def main():
    initialize();start=time.time();config=cfg();settings=config['validation']
    manifest=start_manifest('validation');save(OUT/'validation_manifest.json',manifest)
    (OUT/'validation_checkpoints').mkdir(exist_ok=True)
    prep=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
    assert prep.get('finished_utc') and prep['protocol_sha256']==sha(ROOT/'protocol.json')
    manifest['preparation_manifest_sha256']=sha(OUT/'preparation_manifest.json')
    obs=pd.read_csv(OUT/'observation_table.csv');records=[];curves=[];checkpoints=[];folds=[];baselines=[]
    for fold in settings['folds']:
        tr=np.flatnonzero((obs.joint_completed<=fold['cutoff']).to_numpy())
        te=np.flatnonzero(((obs.date>fold['cutoff'])&(obs.date<=fold['end'])&(obs.weekday==4)&
                          (obs.joint_completed<='2020-12-31')).to_numpy())
        values=batch_tensors('raw',tr);testing=batch_tensors('raw',te)
        reassigned={k:torch.roll(v,1,dims=0) for k,v in testing.items()}
        labels,scales=standardized_targets(tr)
        ymean=scales['returns_mean'];ysd=scales['returns_sd']
        for i in te:baselines.append(dict(date=obs.date.iloc[i],actual=float(obs.exec_return.iloc[i]),
                                          cutoff=fold['cutoff'],train_n=len(tr),training_mean=float(ymean),zero_return=0.))
        folds.append(dict(**fold,train_n=len(tr),test_n=len(te),last_train_label=obs.joint_completed.iloc[tr].max(),
                          first_signal=obs.date.iloc[te].min(),last_signal=obs.date.iloc[te].max(),representation='raw'))
        for variant in [v['name'] for v in config['variants']]:
            auxiliary=variant=='joint_ohlcv'
            for seed in config['seeds']:
                network=make_model('raw',variant,seed)
                optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
                rng=np.random.default_rng(seed)
                for epoch in range(1,settings['max_epochs']+1):
                    network.train();order=rng.permutation(len(tr));running=0.
                    for lower in range(0,len(tr),settings['batch_size']):
                        batch=torch.tensor(order[lower:lower+settings['batch_size']],device='cuda')
                        v={k:t[batch] for k,t in values.items()}
                        optimizer.zero_grad(set_to_none=True)
                        output,aux=predict(network,v,auxiliary)
                        loss=((output-labels['returns'][batch])**2).mean()
                        if auxiliary:loss=loss+.1*((aux-labels['auxiliary'][batch])**2).mean()
                        assert torch.isfinite(loss)
                        loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step();running+=float(loss.detach())*len(batch)
                    curve=dict(variant=variant,cutoff=fold['cutoff'],seed=seed,epoch=epoch,
                               online_joint_loss=running/len(tr))
                    if epoch in settings['epochs']:
                        network.eval()
                        with torch.inference_mode():
                            out,_=predict(network,testing,False);shift,_=predict(network,reassigned,False)
                        estimate=out.cpu().numpy().astype(float)*ysd+ymean
                        shifted=shift.cpu().numpy().astype(float)*ysd+ymean
                        records.extend(prediction_rows(obs,te,variant,epoch,seed,fold['cutoff'],estimate,shifted,len(tr)))
                        train_mse,train_aux=evaluate_full(network,values,labels,auxiliary)
                        curve.update(final_model_train_return_mse=train_mse,final_model_train_auxiliary_mse=train_aux,
                                     validation_return_mse=float(np.mean((estimate-obs.exec_return.iloc[te].to_numpy())**2)),
                                     reassigned_validation_return_mse=float(np.mean((shifted-obs.exec_return.iloc[te].to_numpy())**2)),
                                     validation_forecast_std=float(estimate.std()))
                        path=OUT/'validation_checkpoints'/f'{variant}_{fold["cutoff"]}_{seed}_epoch{epoch}.pt'
                        # Copy state tensors to CPU while keeping the active model and optimizer on GPU.
                        torch.save(dict(state_dict={k:t.detach().cpu() for k,t in network.state_dict().items()},
                                        representation='raw',variant=variant,seed=seed,epoch=epoch,cutoff=fold['cutoff'],
                                        train_n=len(tr),scales=scales,protocol_sha256=sha(ROOT/'protocol.json')),path)
                        checkpoints.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path),variant=variant,
                                                cutoff=fold['cutoff'],seed=seed,epoch=epoch))
                    curves.append(curve)
                pd.DataFrame(records).to_csv(OUT/'validation_seed_predictions.csv',index=False)
                pd.DataFrame(curves).to_csv(OUT/'validation_training_curves.csv',index=False)
                print(f"Validation {fold['cutoff']} {variant} seed {seed} finished; {time.time()-start:.1f}s",flush=True)
                del network,optimizer;torch.cuda.empty_cache()
        del values,testing,reassigned,labels;torch.cuda.empty_cache()
    pd.DataFrame(baselines).to_csv(OUT/'validation_baselines.csv',index=False)
    save(OUT/'validation_folds.json',folds);save(OUT/'validation_checkpoint_manifest.json',checkpoints)
    assert old_evidence()==manifest['old_evidence']
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.time()-start,
                    predictions=len(records),checkpoints=len(checkpoints),fits=27,old_preserved=True)
    save(OUT/'validation_manifest.json',manifest)
    print('All raw-only validation fits complete; candidate scoring may begin.',flush=True)


if __name__=='__main__':main()

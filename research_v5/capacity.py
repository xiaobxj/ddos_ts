"""Frozen real-window memorization diagnostics, including shuffled-label controls."""
from models5 import *


def main():
    initialize();settings=cfg()['capacity'];start=time.time()
    manifest=start_manifest('capacity');save(OUT/'capacity_manifest.json',manifest)
    (OUT/'capacity_checkpoints').mkdir(exist_ok=True)
    indices=np.load(CACHE/'capacity_indices.npy')
    labels,scales=standardized_targets(indices)
    permutation=np.random.default_rng(20260914).permutation(len(indices))
    rows=[];curves=[];predictions=[];checkpoints=[]
    variants=[v['name'] for v in cfg()['variants']]
    for representation in cfg()['representations']:
        values=batch_tensors(representation,indices)
        plans=[(variant,'real') for variant in variants]+[('mlp','real'),('mlp','permuted')]
        for variant,label_kind in plans:
            auxiliary=(variant=='joint_ohlcv')
            weight=.1 if auxiliary else 0.
            y=labels['returns'][permutation] if label_kind=='permuted' else labels['returns']
            for seed in cfg()['seeds']:
                network=make_model(representation,variant,seed,capacity=True)
                optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=0.)
                first_gradient=None;best=float('inf')
                identity=dict(representation=representation,variant=variant,label_kind=label_kind,seed=seed)
                for step in range(settings['steps']+1):
                    network.eval()
                    if step in [0,1,2,5,10,25,50,100,200,400]:
                        with torch.inference_mode():
                            output,aux=predict(network,values,auxiliary)
                            error=float(((output-y)**2).mean())
                            aux_error=float(((aux-labels['auxiliary'])**2).mean()) if auxiliary else None
                        curves.append(dict(**identity,step=step,return_mse=error,auxiliary_mse=aux_error))
                        best=min(best,error)
                    if step==settings['steps']:break
                    network.train();optimizer.zero_grad(set_to_none=True)
                    output,aux=predict(network,values,auxiliary)
                    loss=((output-y)**2).mean()
                    best=min(best,float(loss.detach()))
                    if auxiliary:loss=loss+weight*((aux-labels['auxiliary'])**2).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    if step==0:
                        first_gradient=dict(backbone=grad_norm(network.backbone.parameters()) if variant!='mlp' else None,
                                            readout=grad_norm(network.return_head.parameters()) if variant!='mlp' else grad_norm(network.net[-1].parameters()))
                    torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                network.eval();optimizer.zero_grad(set_to_none=True)
                final,aux=predict(network,values,auxiliary)
                final_loss=((final-y)**2).mean()
                if auxiliary:final_loss=final_loss+weight*((aux-labels['auxiliary'])**2).mean()
                final_loss.backward()
                last_gradient=dict(backbone=grad_norm(network.backbone.parameters()) if variant!='mlp' else None,
                                   readout=grad_norm(network.return_head.parameters()) if variant!='mlp' else grad_norm(network.net[-1].parameters()))
                result=stats(final.detach().cpu().numpy(),y.cpu().numpy())
                shifted={k:torch.roll(v,1,dims=0) for k,v in values.items()}
                with torch.inference_mode():reassigned,_=predict(network,shifted,False)
                shift_stats=stats(reassigned.cpu().numpy(),y.cpu().numpy())
                passed=result['mse']<=.1 and result['correlation'] is not None and result['correlation']>=.9
                row=dict(**identity,**result,capacity_pass=passed,minimum_evaluated_return_mse=best,
                         reassigned_mse=shift_stats['mse'],prediction_reassignment_rms=float(torch.mean((reassigned-final.detach())**2).sqrt()),
                         first_gradients=first_gradient,final_gradients=last_gradient,
                         trainable_parameters=sum(p.numel() for p in network.parameters()))
                rows.append(row)
                for j,(estimate,actual,shift) in enumerate(zip(final.detach().cpu().numpy(),y.cpu().numpy(),reassigned.cpu().numpy())):
                    predictions.append(dict(**identity,row_index=int(indices[j]),target_standardized=float(actual),
                                            prediction_standardized=float(estimate),reassigned_prediction=float(shift)))
                path=OUT/'capacity_checkpoints'/f'{representation}_{variant}_{label_kind}_{seed}.pt'
                torch.save(dict(state_dict=network.cpu().state_dict(),**identity,scales=scales,indices=indices.tolist(),
                                protocol_sha256=sha(ROOT/'protocol.json'),steps=settings['steps']),path)
                checkpoints.append(dict(**identity,file=str(path.relative_to(ROOT)),sha256=sha(path)))
                save(OUT/'capacity_summary.json',rows)
                pd.DataFrame(curves).to_csv(OUT/'capacity_curves.csv',index=False)
                pd.DataFrame(predictions).to_csv(OUT/'capacity_predictions.csv',index=False)
                print(f"Capacity {representation}/{variant}/{label_kind} seed {seed}: MSE {result['mse']:.6f}, PASS {passed}; {time.time()-start:.1f}s",flush=True)
                del network,optimizer;torch.cuda.empty_cache()
        del values
    save(OUT/'capacity_checkpoint_manifest.json',checkpoints)
    assert old_evidence()==manifest['old_evidence']
    manifest.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.time()-start,
                    capacity_fits=len(rows),predictions=len(predictions),old_preserved=True)
    save(OUT/'capacity_manifest.json',manifest)


if __name__=='__main__':main()

"""Fixed-state, dropout-off gradient probe; no optimizer or parameter update."""
from common9 import *


def main():
    legacy.initialize();check_frozen();destination=OUT/'probe_manifest.json';assert not destination.exists()
    run=manifest('training_only_gradient_probe');save(destination,run)
    refs=json.loads((OUT/'legacy_checkpoint_manifest.json').read_text(encoding='utf-8'))
    summaries=[];steps=[];max_identity=0.
    for r in [r for r in refs if r['seed']==20260910]:
        saved=torch.load(PROJECT/r['project_file'],map_location='cpu',weights_only=True)
        tr,_=load_fold(r['cutoff']);n=len(tr);values=legacy.batch_tensors(tr);labels,scales=legacy.standardized_targets(tr)
        assert scales==saved['scales']
        model=legacy.make_model('combined',r['seed']).eval();model.load_state_dict(saved['state_dict'])
        parameters=list(model.parameters());state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
        order=np.random.default_rng(20260910).permutation(n);gradients={};lengths={}
        for rule in ['legacy','balanced']:
            chunks=batches(order,rule);vectors=[]
            for number,positions in enumerate(chunks):
                b=torch.tensor(positions,device='cuda')
                p,a=legacy.predict(model,{k:t[b] for k,t in values.items()},True)
                loss=batch_loss(p,a,labels,b,n,rule)
                parts=torch.autograd.grad(loss,parameters,allow_unused=True)
                vector=torch.cat([(g if g is not None else torch.zeros_like(v)).reshape(-1) for g,v in zip(parts,parameters)]).detach().cpu().numpy().astype(float)
                vectors.append(vector)
                steps.append(dict(cutoff=r['cutoff'],seed=r['seed'],rule=rule,step=number+1,batch_n=len(b),
                    loss_scale=batch_factor(len(b),n,rule),raw_gradient_norm=float(np.linalg.norm(vector)),loss=float(loss.detach())))
            gradients[rule]=np.asarray(vectors);lengths[rule]=np.array([len(b) for b in chunks])
        reference=np.average(gradients['legacy'],axis=0,weights=lengths['legacy'])
        reference_norm=float(np.linalg.norm(reference))
        for rule,matrix in gradients.items():
            mean=matrix.mean(axis=0);norms=np.linalg.norm(matrix,axis=1)
            difference=float(np.linalg.norm(mean-reference));relative=difference/max(reference_norm,1e-12)
            summaries.append(dict(cutoff=r['cutoff'],seed=r['seed'],rule=rule,train_n=n,steps=len(matrix),
                gradient_norm_mean=float(norms.mean()),gradient_norm_max=float(norms.max()),last_step_gradient_norm=float(norms[-1]),
                batch_gradient_vector_variance=float(np.mean(np.sum((matrix-mean)**2,axis=1))),
                mean_gradient_difference_from_full_objective=difference,mean_gradient_relative_difference=relative,
                full_objective_gradient_norm=reference_norm,dropout_disabled=True,optimizer_steps_applied=0))
            if rule=='balanced':assert relative<1e-4;max_identity=max(max_identity,relative)
        assert all(torch.equal(state[k],v.detach().cpu()) for k,v in model.state_dict().items())
        del model,values,labels;torch.cuda.empty_cache()
    pd.DataFrame(summaries).to_csv(OUT/'fixed_state_gradient_summary.csv',index=False)
    pd.DataFrame(steps).to_csv(OUT/'fixed_state_gradient_steps.csv',index=False)
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),probe_states=3,optimizer_steps_applied=0,
               parameters_unchanged=True,dropout_disabled=True,maximum_balanced_full_gradient_relative_error=max_identity)
    save(destination,run);print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':main()

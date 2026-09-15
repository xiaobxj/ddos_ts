from common28 import *

def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence();obs,price,targets=data();files=[];membership=[];scales={};budgets=[]
    assert cfg()['probe']==prior.cfg()['probe'] and cfg()['offset_probe']==prior.cfg()['offset_probe']
    with np.load(PROJECT/'research_v5/cache/packed_raw.npz') as archive:packed={k:archive[k].copy() for k in ['patches','geometry','valid']}
    for history in MEMORIES:
        for fold in cfg()['folds']:
            tr,te=indices(obs,fold,history);cutoff=fold['cutoff'];labels,s=scales_for(targets,tr);scales[f'{history}_{cutoff}']=s
            p=CACHE/f'training_{history}_{cutoff}.npz';np.savez_compressed(p,**{k:v[tr] for k,v in packed.items()},**labels,row_index=tr);files.append(p)
            for split,ids in [('training',tr),('testing_union',te)]:
                for i in ids:membership.append(dict(history=history,cutoff=cutoff,split=split,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i]))
            n=len(tr);budgets.append(dict(history=history,cutoff=cutoff,unique_rows=n,presentations_per_epoch=n,natural_epochs=20,steps_per_seed=20*((n+127)//128),total_presentations_per_seed=n*20,equivalent_retained_passes=20,test_union_n=len(te),shared_with_annual=cutoff.endswith('12-31')))
    b=cfg()['budget'];assert sum(x['steps_per_seed'] for x in budgets)*3==b['new_optimizer_steps'];assert sum(x['total_presentations_per_seed'] for x in budgets)*3==b['new_sample_presentations']
    save(OUT/'training_scales.json',scales);pd.DataFrame(membership).to_csv(OUT/'membership.csv',index=False);pd.DataFrame(budgets).to_csv(OUT/'budgets.csv',index=False);routing(obs).to_csv(OUT/'routing.csv',index=False)
    baseline,ensemble=frozen_baselines();baseline.to_csv(OUT/'baseline_model_predictions.csv',index=False);ensemble.to_csv(OUT/'baseline_ensemble_predictions.csv',index=False)
    files += [OUT/n for n in ['training_scales.json','membership.csv','budgets.csv','routing.csv','baseline_model_predictions.csv','baseline_ensemble_predictions.csv']]
    finish(run,files,old_files=len(run['old_evidence']),new_neural_fits=138,new_head_fits=552,new_projections=276,shared_annual_models=36,weeks=272,new_scoring=False)
    print(json.dumps(dict(status='FROZEN',protocol=run['protocol_sha256'],**b)),flush=True)
if __name__=='__main__':main()

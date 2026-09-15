from common27 import *
def main():
    assert not (OUT/'preparation_manifest.json').exists();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence();obs,price,targets=data();files=[];membership=[];scales={};budgets=[]
    assert cfg()['probe']==prior.cfg()['probe'] and cfg()['offset_probe']==prior.cfg()['offset_probe']
    with np.load(PROJECT/'research_v5/cache/packed_raw.npz') as archive:packed={k:archive[k].copy() for k in ['patches','geometry','valid']}
    for history in HISTORIES:
        for fold in cfg()['folds']:
            tr,te=indices(obs,fold,history);cutoff=fold['cutoff'];labels,s=scales_for(targets,tr);scales[f'{history}_{cutoff}']=s
            if history in NEW:
                p=CACHE/f'training_{history}_{cutoff}.npz';np.savez_compressed(p,**{k:v[tr] for k,v in packed.items()},**labels,row_index=tr);files.append(p)
            for split,ids in [('training',tr),('testing',te)]:
                for i in ids:membership.append(dict(history=history,cutoff=cutoff,split=split,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i]))
            n=fold['train_n'];budgets.append(dict(history=history,cutoff=cutoff,unique_rows=len(tr),presentations_per_nominal_epoch=n,nominal_epochs=20,steps_per_seed=20*((n+127)//128),total_presentations_per_seed=n*20,equivalent_retained_passes=n*20/len(tr),test_n=len(te),reused=history=='full'))
    save(OUT/'training_scales.json',scales);pd.DataFrame(membership).to_csv(OUT/'membership.csv',index=False);pd.DataFrame(budgets).to_csv(OUT/'budgets.csv',index=False)
    a,b=calibration_availability(obs);a.to_csv(OUT/'calibration_availability.csv',index=False);b.to_csv(OUT/'calibration_membership_plan.csv',index=False)
    baseline,ensemble=frozen_baselines();baseline.to_csv(OUT/'full_model_predictions.csv',index=False);ensemble.to_csv(OUT/'full_ensemble_predictions.csv',index=False)
    files += [OUT/n for n in ['training_scales.json','membership.csv','budgets.csv','calibration_availability.csv','calibration_membership_plan.csv','full_model_predictions.csv','full_ensemble_predictions.csv']]
    finish(run,files,old_files=len(run['old_evidence']),new_neural_fits=36,new_head_fits=144,new_projections=72,reused_neural_models=18,weeks=272,new_scoring=False)
    print(json.dumps(dict(status='FROZEN',protocol=run['protocol_sha256'],old_files=len(run['old_evidence']),neural_fits=36,weeks=272)),flush=True)
if __name__=='__main__':main()

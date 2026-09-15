from common14 import *

def main():
    assert not (OUT/'preparation_manifest.json').exists(),'Preserve preparation'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence();local=[]
    for name in ['observation_table.csv','archived_models.json','archived_predictions.csv','validation_rows.csv']:
        shutil.copyfile(V13/'results'/name,OUT/name);local.append(OUT/name)
    obs=pd.read_csv(OUT/'observation_table.csv');validation=pd.read_csv(OUT/'validation_rows.csv')
    scales=read(V12/'results/training_scales.json');metadata={};training=[];budgets=[]
    with np.load(V5/'cache/targets.npz') as raw:
        for fold,n_expected,v_expected in zip(cfg()['folds'],[1678,1921,2165],[49,46,46]):
            cutoff=fold['cutoff'];tr=np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy());assert len(tr)==n_expected
            y=raw_direction(raw['returns'][tr]);frequency=float(y.mean(dtype=np.float64))
            with np.load(V12/'cache'/f'training_{cutoff}.npz') as data:
                np.testing.assert_array_equal(data['row_index'],tr)
                p=CACHE/f'training_{cutoff}.npz'
                np.savez_compressed(p,**{k:data[k] for k in ['patches','geometry','valid','auxiliary','row_index']},direction=y)
            local.append(p);vp=CACHE/f'validation_{cutoff}.npz';shutil.copyfile(V12/'cache'/vp.name,vp);local.append(vp)
            with np.load(vp) as data:te=data['row_index'].copy()
            expected=np.flatnonzero((obs.date.gt(cutoff)&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')).to_numpy())
            np.testing.assert_array_equal(te,expected);assert len(te)==v_expected and not len(np.intersect1d(tr,te))
            v=validation[validation.cutoff.eq(cutoff)];np.testing.assert_array_equal(v.row_index,te)
            np.testing.assert_allclose(v.actual,raw['returns'][te],rtol=0,atol=1e-12)
            b,l=probability_losses(np.full(len(y),frequency),y)
            metadata[cutoff]=dict(train_n=len(tr),up_count=int(y.sum()),zero_count=int((raw['returns'][tr]==0).sum()),
                frequency=frequency,initial_logit=float(np.log(frequency/(1-frequency))),scales=scales[cutoff],
                constant_training_brier=float(b.mean()),constant_training_log_loss=float(l.mean()),constant_training_accuracy=float(((frequency>.5)==(y>0)).mean()))
            for i,z in zip(tr,y):training.append(dict(cutoff=cutoff,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i],direction=int(z)))
            steps=(len(tr)+127)//128
            budgets.append(dict(cutoff=cutoff,train_n=len(tr),epochs=20,seeds=3,steps_per_epoch=steps,last_batch_n=len(tr)%128,
                total_steps_all_seeds=steps*20*3,presentations_all_seeds=len(tr)*20*3))
    save(OUT/'training_metadata.json',metadata)
    pd.DataFrame(training).to_csv(OUT/'training_rows.csv',index=False);pd.DataFrame(budgets).to_csv(OUT/'training_budgets.csv',index=False)
    baselines=validation[['cutoff','row_index','date','actual']].rename(columns={'actual':'actual_return'})
    baselines['actual_up']=raw_direction(baselines.actual_return).astype(int)
    baselines['training_frequency']=baselines.cutoff.map({c:v['frequency'] for c,v in metadata.items()})
    baselines.to_csv(OUT/'classification_baselines.csv',index=False)
    jobs=[dict(cutoff=f['cutoff'],seed=s,method='direction_bce') for f in cfg()['folds'] for s in cfg()['seeds']]
    save(OUT/'training_plan.json',jobs)
    local += [OUT/n for n in ['training_metadata.json','training_rows.csv','training_budgets.csv','classification_baselines.csv','training_plan.json']]
    run.update(finished_utc=now(),new_fits=9,epochs=180,validation_weeks=141,local_sha256={str(p.relative_to(ROOT)):sha(p) for p in local})
    save(OUT/'preparation_manifest.json',run)
    print(json.dumps(dict(status='FROZEN',previous_files=len(run['old_evidence']),new_fits=9,epochs=180,protocol_sha256=run['protocol_sha256'])),flush=True)

if __name__=='__main__':main()

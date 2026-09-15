from common13 import *

def main():
    assert not (OUT/'preparation_manifest.json').exists(),'Preserve frozen preparation'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence()
    obs=pd.read_csv(PROJECT/'research_v10/results/observation_table.csv')
    assert len(obs)==3780 and obs.date.is_unique
    obs.to_csv(OUT/'observation_table.csv',index=False)
    local=[OUT/'observation_table.csv'];scales={};training=[];budgets=[]
    with np.load(V5/'cache/targets.npz') as labels,np.load(V5/'cache/packed_raw.npz') as packed:
        np.testing.assert_allclose(labels['returns'],obs.exec_return,rtol=0,atol=1e-12)
        for cutoff,expected in zip(cfg()['new_training_cutoffs'],[1077,1190,1434]):
            tr=np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy());assert len(tr)==expected
            target={};s={}
            for key in ['returns','auxiliary']:
                y=labels[key][tr].astype(float);mean=y.mean(axis=0);sd=np.maximum(y.std(axis=0),1e-6)
                target[key]=((y-mean)/sd).astype(np.float32)
                s[key+'_mean']=np.asarray(mean).tolist();s[key+'_sd']=np.asarray(sd).tolist()
            scales[cutoff]=s;p=CACHE/f'training_{cutoff}.npz'
            np.savez_compressed(p,**{k:packed[k][tr] for k in ['patches','geometry','valid']},**target,row_index=tr);local.append(p)
            for i in tr:training.append(dict(cutoff=cutoff,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i]))
            budgets.append(dict(cutoff=cutoff,train_n=len(tr),epochs=20,seeds=3,steps_per_epoch=(len(tr)+127)//128,
                last_batch_n=len(tr)%128,total_steps_all_seeds=20*3*((len(tr)+127)//128),presentations_all_seeds=20*3*len(tr)))
    scales.update(read(V12/'results/training_scales.json'))
    save(OUT/'training_scales.json',scales)
    pd.DataFrame(training).to_csv(OUT/'training_rows.csv',index=False)
    pd.DataFrame(budgets).to_csv(OUT/'training_budgets.csv',index=False)
    refs=read(V12/'results/archived_models.json');assert len(refs)==9
    save(OUT/'archived_models.json',refs)
    for name in ['archived_predictions.csv','validation_rows.csv']:
        shutil.copyfile(V12/'results'/name,OUT/name)
    pool=[]
    for cutoff,expected in zip(cfg()['inner_cutoffs'],[22,48,50,49,46]):
        end=f'{int(cutoff[:4])+1}-12-31'
        ids=np.flatnonzero((obs.date.gt(cutoff)&obs.date.le(end)&obs.weekday.eq(4)).to_numpy())
        assert len(ids)==expected
        for i in ids:
            pool.append(dict(inner_cutoff=cutoff,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i],
                training_mean=scales[cutoff]['returns_mean'],training_sd=scales[cutoff]['returns_sd'],
                source='new_inner_fit' if cutoff in cfg()['new_training_cutoffs'] else 'archived_forecast'))
    pool=pd.DataFrame(pool);assert len(pool)==215 and pool.date.is_unique
    pool.to_csv(OUT/'rolling_plan.csv',index=False)
    membership=[]
    for fold,expected in zip(cfg()['outer_folds'],cfg()['calibration']['expected_counts']):
        cutoff=fold['cutoff'];start=f'{int(cutoff[:4])-2}-01-01'
        g=pool[pool.date.ge(start)&pool.date.le(cutoff)&pool.joint_completed.le(cutoff)].copy()
        assert len(g)==expected and g.date.str[:4].nunique()==3 and g.inner_cutoff.lt(g.date).all()
        g.insert(0,'outer_cutoff',cutoff);membership.append(g)
    pd.concat(membership,ignore_index=True).to_csv(OUT/'calibration_membership.csv',index=False)
    jobs=[dict(cutoff=c,seed=s) for c in cfg()['new_training_cutoffs'] for s in cfg()['seeds']]
    save(OUT/'training_plan.json',jobs)
    local += [OUT/n for n in ['training_scales.json','training_rows.csv','training_budgets.csv','archived_models.json',
        'archived_predictions.csv','validation_rows.csv','rolling_plan.csv','calibration_membership.csv','training_plan.json']]
    run.update(local_sha256={str(p.relative_to(ROOT)):sha(p) for p in local},new_fits=9,actual_epochs=180,
        rolling_dates=215,calibration_counts=cfg()['calibration']['expected_counts'],outer_weeks=141,finished_utc=now())
    save(OUT/'preparation_manifest.json',run)
    print(json.dumps(dict(status='FROZEN',previous_files=len(run['old_evidence']),new_fits=9,
        calibration_counts=run['calibration_counts'],protocol_sha256=run['protocol_sha256'])),flush=True)

if __name__=='__main__':main()

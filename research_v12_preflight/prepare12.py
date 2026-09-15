"""Freeze comparable inputs, schedules, baselines and reconstruction references."""
from common12 import *


def main():
    assert not (OUT/'preparation_manifest.json').exists(),'Preserve frozen preparation'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    run=manifest('preparation');run['old_evidence']=old_evidence()
    local=[]
    for name in ['training_scales.json','training_rows.csv']:
        shutil.copyfile(V11/'results'/name,OUT/name);local.append(OUT/name)
    scales=read(OUT/'training_scales.json')
    obs=pd.read_csv(V10/'results/observation_table.csv')
    rows=[];budgets=[]
    with np.load(V5/'cache/packed_raw.npz') as packed:
        for fold,n_expected,v_expected in zip(cfg()['folds'],[1678,1921,2165],[49,46,46]):
            cutoff=fold['cutoff'];p=CACHE/f'training_{cutoff}.npz'
            shutil.copyfile(V11/'cache'/p.name,p);local.append(p)
            with np.load(p) as arrays:tr=arrays['row_index'].copy()
            np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(cutoff).to_numpy()))
            assert len(tr)==n_expected
            with np.load(V10/f'cache/masks_{cutoff}.npz') as masks:te=masks['testing'].copy()
            expected=np.flatnonzero((obs.date.gt(cutoff)&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')).to_numpy())
            np.testing.assert_array_equal(te,expected)
            assert len(te)==v_expected and not len(np.intersect1d(tr,te))
            p=CACHE/f'validation_{cutoff}.npz'
            np.savez_compressed(p,**{k:packed[k][te] for k in ['patches','geometry','valid']},row_index=te);local.append(p)
            for i in te:
                rows.append(dict(cutoff=cutoff,row_index=int(i),date=obs.date.iloc[i],actual=float(obs.exec_return.iloc[i]),
                    training_mean=scales[cutoff]['returns_mean'],training_sd=scales[cutoff]['returns_sd'],zero_return=0.))
            steps=(len(tr)+127)//128
            for s in cfg()['schedules']:
                budgets.append(dict(cutoff=cutoff,schedule=s['name'],train_n=len(tr),total_epochs=40,
                    steps_per_epoch=steps,total_steps=steps*40,total_presentations=len(tr)*40,last_batch_n=len(tr)%128,
                    lr1_20=s['epochs1_20_lr'],lr21_40=s['epochs21_40_lr'],weight_decay=.1,
                    nominal_cumulative_decay_factor=(1-.1*s['epochs1_20_lr'])**(20*steps)*(1-.1*s['epochs21_40_lr'])**(20*steps)))
    validation=pd.DataFrame(rows);assert len(validation)==141 and validation.date.is_unique
    validation.to_csv(OUT/'validation_rows.csv',index=False)
    pd.DataFrame(budgets).to_csv(OUT/'schedule_budgets.csv',index=False)
    refs=[]
    for r in read(V10/'results/mse_checkpoint_manifest.json'):
        state=torch.load(PROJECT/r['project_file'],map_location='cpu',weights_only=True)
        assert state['epoch']==20 and state['arm']=='combined' and state['history']=='full'
        assert state['scales']==scales[r['cutoff']]
        refs.append(dict(schedule='archived20',cutoff=r['cutoff'],seed=r['seed'],epoch=20,
            project_file=r['project_file'],sha256=r['sha256'],model_sha256=object_hash(state['state_dict'])))
    assert len(refs)==9
    save(OUT/'archived_models.json',refs)
    old=pd.read_csv(V10/'results/mse_predictions.csv')
    columns=['cutoff','seed','row_index','date','actual','predicted_return','reassigned_prediction','training_mean']
    old=old[columns].copy();old['schedule']='archived20';old['epoch']=20
    old=old.merge(validation[['date','training_sd']],on='date',validate='many_to_one')
    assert len(old)==423 and old.groupby('date').size().eq(3).all()
    for (cutoff,seed),g in old.groupby(['cutoff','seed']):
        v=validation[validation.cutoff.eq(cutoff)]
        np.testing.assert_array_equal(g.row_index,v.row_index)
        np.testing.assert_allclose(g.actual,v.actual,rtol=0,atol=1e-12)
    old.to_csv(OUT/'archived_predictions.csv',index=False)
    jobs=[dict(cutoff=r['cutoff'],seed=r['seed'],schedule=s['name'],lr=s['epochs21_40_lr'])
          for r in refs for s in cfg()['schedules']]
    save(OUT/'continuation_plan.json',jobs)
    local += [OUT/n for n in ['validation_rows.csv','schedule_budgets.csv','archived_models.json','archived_predictions.csv','continuation_plan.json']]
    run.update(local_sha256={str(p.relative_to(ROOT)):sha(p) for p in local},
        archived_models=9,shared_reconstructions=9,continuations=18,full_training_paths=18,
        actual_training_epochs=540,validation_weeks=141,completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT/'preparation_manifest.json',run)
    print(json.dumps(dict(status='FROZEN',previous_files=len(run['old_evidence']),shared_reconstructions=9,
                         continuations=18,epochs_per_path=40,validation_weeks=141,protocol_sha256=run['protocol_sha256'])))


if __name__=='__main__':main()

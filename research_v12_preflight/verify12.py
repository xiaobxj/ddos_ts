"""Full checkpoint replay, optimizer/budget audits and independent metric arithmetic."""
from common12 import *


def verify_metric(row,g):
    p=g.predicted_return.to_numpy(float);y=g.actual.to_numpy(float);e=p-y
    mse=float(np.mean(e*e));positive=y>0
    centered=p-g.groupby('cutoff').predicted_return.transform('mean').to_numpy()
    total=float(np.sum((p-p.mean())**2))
    values=dict(n=len(g),mse=mse,rmse=mse**.5,accuracy=float(np.mean((p>0)==positive)),
        balanced_accuracy=float(((p[positive]>0).mean()+(p[~positive]<=0).mean())/2),
        predicted_up_fraction=float((p>0).mean()),forecast_std=float(p.std()),
        within_fold_forecast_std=float(np.sqrt(np.mean(centered**2))),mae=float(np.abs(e).mean()),
        mse_skill_vs_training_mean=1-mse/float(np.mean((g.training_mean.to_numpy()-y)**2)),
        reassigned_mse=float(np.mean((g.reassigned_prediction.to_numpy()-y)**2)))
    values['reassignment_mse_relative_change']=values['reassigned_mse']/mse-1
    values['between_fold_variance_share']=float(1-np.sum(centered**2)/total) if total else None
    values['correlation']=float(np.corrcoef(p,y)[0,1]) if p.std()>1e-12 else None
    for key,value in values.items():
        if value is None:assert pd.isna(row[key]),(key,row[key])
        else:assert abs(float(row[key])-value)<1e-11,(key,row[key],value)


def inference_audit(seed,ensemble):
    for r in pd.read_csv(OUT/'ensemble_metrics.csv').to_dict('records'):
        verify_metric(r,ensemble[ensemble.schedule.eq(r['schedule'])])
    for r in pd.read_csv(OUT/'yearly_metrics.csv').to_dict('records'):
        verify_metric(r,ensemble[ensemble.schedule.eq(r['schedule'])&ensemble.date.str[:4].eq(str(r['year']))])
    for r in pd.read_csv(OUT/'seed_metrics.csv').to_dict('records'):
        verify_metric(r,seed[seed.schedule.eq(r['schedule'])&seed.seed.eq(r['seed'])])
    refs={k:g.sort_values('date') for k,g in ensemble.groupby('schedule')}
    constants=pd.read_csv(OUT/'validation_rows.csv')
    for r in pd.read_csv(OUT/'baseline_metrics.csv').to_dict('records'):
        g=constants.copy();g['predicted_return']=g[r['reference']];g['reassigned_prediction']=g[r['reference']]
        verify_metric(r,g);refs[r['reference']]=g.sort_values('date')
    records=read(OUT/'primary_comparisons.json');assert len(records)==4
    n=141;rng=np.random.default_rng(20260910);starts=rng.integers(n,size=(10000,int(np.ceil(n/8))))
    ids=((starts[:,:,None]+np.arange(8))%n).reshape(10000,-1)[:,:n];ps=[]
    for r,planned in zip(records,cfg()['primary_comparisons']):
        assert r['candidate']==planned['candidate'] and r['reference']==planned['reference']
        a,b=refs[r['candidate']],refs[r['reference']];assert a.date.tolist()==b.date.tolist()
        y=a.actual.to_numpy();x=(a.predicted_return.to_numpy()-y)**2;z=(b.predicted_return.to_numpy()-y)**2
        error=x-z;accuracy=((a.predicted_return.to_numpy()>0)==(y>0)).astype(float)-((b.predicted_return.to_numpy()>0)==(y>0)).astype(float)
        for key,d in [('mse',error),('accuracy',accuracy)]:
            means=d[ids].mean(axis=1);centered=(d-d.mean())[ids].mean(axis=1)
            p=float((1+(np.abs(centered)>=abs(d.mean())).sum())/10001)
            assert abs(r[key]['difference']-d.mean())<1e-12 and abs(r[key]['p']-p)<1e-12
            np.testing.assert_allclose([r[key]['ci95_low'],r[key]['ci95_high']],np.quantile(means,[.025,.975]),rtol=0,atol=1e-12)
            if key=='mse':ps.append(p)
        delta=np.sqrt(x[ids].mean(axis=1))-np.sqrt(z[ids].mean(axis=1))
        assert abs(r['rmse']['difference']-(np.sqrt(x.mean())-np.sqrt(z.mean())))<1e-12
        np.testing.assert_allclose([r['rmse']['ci95_low'],r['rmse']['ci95_high']],np.quantile(delta,[.025,.975]),rtol=0,atol=1e-12)
    holm=np.empty(4);running=0.
    for i,j in enumerate(np.argsort(ps)):running=max(running,min(1.,ps[j]*(4-i)));holm[j]=running
    np.testing.assert_allclose(holm,[r['mse']['holm_adjusted_p'] for r in records],rtol=0,atol=1e-12)
    def mse(g):return float(np.mean((g.predicted_return-g.actual)**2))
    counts=dict(seeds_beat_mean=0,paired_seeds_beat_constant40=0,paired_seeds_beat_archived20=0,years_beat_constant40=0,years_beat_archived20=0)
    for number in cfg()['seeds']:
        a=seed[seed.schedule.eq('decay40')&seed.seed.eq(number)]
        counts['seeds_beat_mean']+=int(mse(a)<float(np.mean((a.training_mean-a.actual)**2)))
        for name in ['constant40','archived20']:
            b=seed[seed.schedule.eq(name)&seed.seed.eq(number)]
            counts['paired_seeds_beat_'+name]+=int(mse(a)<mse(b))
    for year in ['2018','2019','2020']:
        a=refs['decay40'][refs['decay40'].date.str[:4].eq(year)]
        for name in ['constant40','archived20']:
            b=refs[name][refs[name].date.str[:4].eq(year)]
            counts['years_beat_'+name]+=int(mse(a)<mse(b))
    flags=dict(ensemble_beats_constant40=mse(refs['decay40'])<mse(refs['constant40']),
        ensemble_beats_archived20=mse(refs['decay40'])<mse(refs['archived20']),ensemble_beats_mean=mse(refs['decay40'])<mse(refs['training_mean']),
        at_least_two_seeds_beat_mean=counts['seeds_beat_mean']>=2,
        at_least_two_paired_seeds_beat_constant40=counts['paired_seeds_beat_constant40']>=2,
        at_least_two_paired_seeds_beat_archived20=counts['paired_seeds_beat_archived20']>=2,
        at_least_two_years_beat_constant40=counts['years_beat_constant40']>=2,
        at_least_two_years_beat_archived20=counts['years_beat_archived20']>=2)
    result=read(OUT/'assessment.json');assert result['counts']==counts and result['flags']==flags
    assert result['descriptive_screen_pass']==all(flags.values())
    return flags


def main():
    legacy.initialize();prep=check_frozen();assert old_evidence()==prep['old_evidence'];start=time.time()
    assert read(OUT/'scoring_manifest.json').get('finished_utc')
    old=read(OUT/'archived_models.json');prefixes=read(OUT/'prefix_models.json');finals=read(OUT/'final_models.json')
    assert (len(old),len(prefixes),len(finals))==(9,9,18)
    parity=read(OUT/'reconstruction_parity.json');assert len(parity)==9 and all(r['exact'] and r['maximum_weight_difference']==0 for r in parity)
    curves=pd.concat([pd.read_csv(OUT/'prefix_training_curves.csv'),pd.read_csv(OUT/'continuation_training_curves.csv')],ignore_index=True)
    monitors=pd.concat([pd.read_csv(OUT/'prefix_training_monitors.csv'),pd.read_csv(OUT/'continuation_training_monitors.csv')],ignore_index=True)
    draws=pd.concat([pd.read_csv(OUT/'prefix_training_loss_draws.csv'),pd.read_csv(OUT/'continuation_training_loss_draws.csv')],ignore_index=True)
    seed=pd.read_csv(OUT/'all_seed_predictions.csv');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv')
    training_rows=pd.read_csv(OUT/'training_rows.csv');truth=pd.read_csv(OUT/'validation_rows.csv')
    assert len(curves)==540 and len(monitors)==117 and len(draws)==243 and len(seed)==1269
    max_prediction=0.;max_reassignment=0.;max_training_loss=0.;orders=0;audits=0
    for number,ref in enumerate(old+prefixes+finals,1):
        model,state=load_model(ref);values,labels,tr,scales=load_training(ref['cutoff'])
        t=training_rows[training_rows.cutoff.eq(ref['cutoff'])];np.testing.assert_array_equal(tr,t.row_index)
        assert t.joint_completed.le(ref['cutoff']).all() and state['scales']==scales
        assert object_hash(state['state_dict'])==ref['model_sha256']
        validation,te=load_validation(ref['cutoff']);g=truth[truth.cutoff.eq(ref['cutoff'])]
        np.testing.assert_array_equal(g.row_index,te)
        schedule='archived20' if ref['schedule']=='prefix20' else ref['schedule']
        expected=seed[seed.schedule.eq(schedule)&seed.cutoff.eq(ref['cutoff'])&seed.seed.eq(ref['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(expected.row_index,te)
        model.eval()
        with torch.inference_mode():
            p,_=legacy.predict(model,validation,False)
            q,_=legacy.predict(model,{k:torch.roll(v,1,dims=0) for k,v in validation.items()},False)
        p=p.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
        q=q.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']
        d=float(np.max(np.abs(p-expected.predicted_return)));e=float(np.max(np.abs(q-expected.reassigned_prediction)))
        assert d<1e-10 and e<1e-10;max_prediction=max(max_prediction,d);max_reassignment=max(max_reassignment,e)
        if ref['schedule']!='archived20':
            assert state['protocol_sha256']==sha(ROOT/'protocol.json')
            assert object_hash(state['optimizer_state_dict'])==state['optimizer_sha256']==ref['optimizer_sha256']
            assert object_hash({k:state[k] for k in ['numpy_rng_state','cpu_rng_state','cuda_rng_state']})==state['rng_sha256']==ref['rng_sha256']
            epoch=ref['epoch'];assert epoch==(20 if ref['schedule']=='prefix20' else 40)
            assert all(int(v['step'])==epoch*((len(tr)+127)//128) for v in state['optimizer_state_dict']['state'].values())
            lr=.0001 if ref['schedule']=='decay40' else .001
            assert state['optimizer_state_dict']['param_groups'][0]['lr']==lr
            assert state['optimizer_state_dict']['param_groups'][0]['weight_decay']==.1
            group=curves[curves.schedule.eq(ref['schedule'])&curves.cutoff.eq(ref['cutoff'])&curves.seed.eq(ref['seed'])].set_index('epoch')
            expected_epochs=list(range(1,21)) if epoch==20 else list(range(21,41))
            assert group.index.tolist()==expected_epochs;rng=np.random.default_rng(ref['seed'])
            for step in range(1,epoch+1):
                order=rng.permutation(len(tr))
                if step not in group.index:continue
                r=group.loc[step];assert r.training_order_sha256==array_hash(tr[order]);orders+=1
                assert r.batch_boundaries_sha256==array_hash(np.r_[np.arange(0,len(tr),128),len(tr)])
                assert r.learning_rate==lr and r.train_n==r.presentations==len(tr) and r.optimizer_steps==(len(tr)+127)//128
            assert rng.bit_generator.state==state['numpy_rng_state']
            current=training_audit(model,values,labels)
            group=draws[draws.schedule.eq(ref['schedule'])&draws.cutoff.eq(ref['cutoff'])&draws.seed.eq(ref['seed'])]
            assert len(group)==9
            for r in current:
                prev=group[group['mode'].eq(r['mode'])&group.draw.eq(r['draw'])].iloc[0]
                for key in ['return_mse','auxiliary_mse','joint_mse']:
                    delta=abs(prev[key]-r[key]);max_training_loss=max(max_training_loss,delta);assert delta<1e-11
                audits+=1
            m=monitors[monitors.schedule.eq(ref['schedule'])&monitors.cutoff.eq(ref['cutoff'])&monitors.seed.eq(ref['seed'])]
            expected_monitor=[0,5,10,15,20] if epoch==20 else [25,30,35,40]
            assert m.epoch.tolist()==expected_monitor
            for key in ['return_mse','auxiliary_mse','joint_mse']:assert abs(m[m.epoch.eq(epoch)].iloc[0][key]-current[0][key])<1e-11
            if epoch==20:
                archived=next(r for r in old if r['cutoff']==ref['cutoff'] and r['seed']==ref['seed'])
                assert ref['model_sha256']==archived['model_sha256']
            else:
                pre=next(r for r in prefixes if r['cutoff']==ref['cutoff'] and r['seed']==ref['seed'])
                for key in ['model_sha256','optimizer_sha256','rng_sha256']:assert state['initial_'+key]==pre[key]
                other=next(r for r in finals if r['cutoff']==ref['cutoff'] and r['seed']==ref['seed'] and r['schedule']!=ref['schedule'])
                assert other['rng_sha256']==ref['rng_sha256']
        print(f'Checkpoint replay {number}/36; {time.time()-start:.1f}s',flush=True)
        del model,values,labels,validation,state;torch.cuda.empty_cache()
    assert orders==540 and audits==243
    for (schedule,date),g in seed.groupby(['schedule','date']):
        assert len(g)==3
        r=ensemble[ensemble.schedule.eq(schedule)&ensemble.date.eq(date)].iloc[0]
        assert abs(r.predicted_return-g.predicted_return.mean())<1e-12 and abs(r.reassigned_prediction-g.reassigned_prediction.mean())<1e-12
    flags=inference_audit(seed,ensemble)
    losses=pd.read_csv(OUT/'training_loss_summary.csv');assert len(losses)==54
    for r in losses.to_dict('records'):
        g=draws[draws.schedule.eq(r['schedule'])&draws.cutoff.eq(r['cutoff'])&draws.seed.eq(r['seed'])&draws['mode'].eq(r['mode'])]
        assert r['draws']==len(g)
        for key in ['return_mse','auxiliary_mse','joint_mse']:
            assert abs(r[key]-g[key].mean())<1e-11
            assert abs(r[key+'_draw_sd']-(g[key].std(ddof=1) if len(g)>1 else 0.))<1e-11
    pairs=pd.read_csv(OUT/'paired_training_comparisons.csv');assert len(pairs)==54
    for r in pairs.to_dict('records'):
        base=losses[losses.cutoff.eq(r['cutoff'])&losses.seed.eq(r['seed'])&losses['mode'].eq(r['mode'])]
        a=base[base.schedule.eq(r['candidate'])].iloc[0];b=base[base.schedule.eq(r['reference'])].iloc[0]
        for key in ['return_mse','auxiliary_mse','joint_mse']:
            assert abs(r[key+'_candidate']-a[key])<1e-11 and abs(r[key+'_reference']-b[key])<1e-11
            assert abs(r[key+'_difference']-(a[key]-b[key]))<1e-11
    for r in pd.read_csv(OUT/'training_comparison_summary.csv').to_dict('records'):
        g=pairs[pairs.candidate.eq(r['candidate'])&pairs.reference.eq(r['reference'])&pairs['mode'].eq(r['mode'])]
        assert r['pairs']==len(g)==9
        for key in ['return_mse','auxiliary_mse','joint_mse']:
            assert r[key+'_lower_count']==int(g[key+'_difference'].lt(-1e-7).sum())
            assert abs(r[key+'_relative_change']-(g[key+'_candidate'].mean()/g[key+'_reference'].mean()-1))<1e-11
    check_frozen();assert old_evidence()==prep['old_evidence']
    result=dict(status='PASS',model_states_replayed=36,exact_legacy_reconstructions=9,full40_epoch_paths=18,
        physical_epoch_orders_and_boundaries=orders,training_loss_passes_replayed=audits,paired_training_comparisons=54,
        validation_seed_predictions=1269,validation_weeks=141,primary_comparisons_recomputed=4,flags=flags,
        maximum_prediction_replay_error=max_prediction,maximum_reassignment_replay_error=max_reassignment,
        maximum_training_loss_replay_error=max_training_loss,previous_files_preserved=len(prep['old_evidence']),
        elapsed_seconds=time.time()-start,completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

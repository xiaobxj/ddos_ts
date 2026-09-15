"""Independent replay, intercept optimality, budget and statistical audit."""
from common10 import *


def check_metrics(row,g):
    e=g.predicted_return.to_numpy()-g.actual.to_numpy();mse=float(np.mean(e**2));z=e/g.training_sd.to_numpy()
    checks=dict(mse=mse,rmse=mse**.5,accuracy=float(((g.predicted_return>0)==(g.actual>0)).mean()),mae=float(np.abs(e).mean()),
        scaled_huber_error=float(np.where(np.abs(z)<=1,z*z,2*np.abs(z)-1).mean()),
        mse_skill_vs_training_mean=1-mse/float(np.mean((g.training_mean-g.actual)**2)),
        mse_skill_vs_huber_intercept=1-mse/float(np.mean((g.huber_intercept-g.actual)**2)),
        reassigned_mse=float(np.mean((g.reassigned_prediction-g.actual)**2)))
    for key,value in checks.items():assert abs(row[key]-value)<1e-12,(key,row[key],value)


def statistics_audit(seed,ensemble):
    for r in pd.read_csv(OUT/'ensemble_metrics.csv').to_dict('records'):check_metrics(r,ensemble[ensemble.rule==r['rule']])
    for r in pd.read_csv(OUT/'yearly_metrics.csv').to_dict('records'):check_metrics(r,ensemble[(ensemble.rule==r['rule'])&(ensemble.date.str[:4]==str(r['year']))])
    for r in pd.read_csv(OUT/'seed_metrics.csv').to_dict('records'):check_metrics(r,seed[(seed.rule==r['rule'])&(seed.seed==r['seed'])])
    new=ensemble[ensemble.rule=='huber'].sort_values('date');old=ensemble[ensemble.rule=='mse'].sort_values('date')
    assert new.date.tolist()==old.date.tolist()
    y=new.actual.to_numpy();a=(new.predicted_return.to_numpy()-y)**2
    references={'mse':(old.predicted_return.to_numpy()-y)**2,'training_mean':(new.training_mean.to_numpy()-y)**2,
                'huber_intercept':(new.huber_intercept.to_numpy()-y)**2}
    primary=json.loads((OUT/'primary_comparisons.json').read_text(encoding='utf-8'));assert len(primary)==3
    ps=[]
    for r in primary:
        b=references[r['reference']];d=a-b;n=len(a);rng=np.random.default_rng(20260910)
        starts=rng.integers(n,size=(10000,int(np.ceil(n/8))))
        ids=((starts[:,:,None]+np.arange(8))%n).reshape(10000,-1)[:,:n]
        centered=(d-d.mean())[ids].mean(axis=1);p=float((1+(np.abs(centered)>=abs(d.mean())).sum())/10001);ps.append(p)
        assert abs(p-r['mse']['p'])<1e-12 and abs(d.mean()-r['mse']['difference'])<1e-12
        np.testing.assert_allclose(np.quantile(d[ids].mean(axis=1),[.025,.975]),[r['mse']['ci95_low'],r['mse']['ci95_high']],rtol=0,atol=1e-12)
        delta=np.sqrt(a[ids].mean(axis=1))-np.sqrt(b[ids].mean(axis=1))
        np.testing.assert_allclose(np.quantile(delta,[.025,.975]),[r['rmse']['ci95_low'],r['rmse']['ci95_high']],rtol=0,atol=1e-12)
    corrected=np.zeros(3);running=0.
    for i,j in enumerate(np.argsort(ps)):running=max(running,min(1.,ps[j]*(3-i)));corrected[j]=running
    np.testing.assert_allclose(corrected,[r['mse']['holm_adjusted_p'] for r in primary],rtol=0,atol=1e-12)
    counts={'mean':0,'huber':0,'both':0,'model':0};years=0
    for number in cfg()['training']['seeds']:
        g=seed[(seed.rule=='huber')&(seed.seed==number)].sort_values('date');h=seed[(seed.rule=='mse')&(seed.seed==number)].sort_values('date')
        mse=float(np.mean((g.predicted_return-g.actual)**2));m=mse<float(np.mean((g.training_mean-g.actual)**2))
        c=mse<float(np.mean((g.huber_intercept-g.actual)**2));counts['mean']+=int(m);counts['huber']+=int(c);counts['both']+=int(m and c)
        counts['model']+=int(mse<float(np.mean((h.predicted_return-h.actual)**2)))
    for year in ['2018','2019','2020']:
        mask=new.date.str[:4].eq(year).to_numpy();years+=int(a[mask].mean()<references['mse'][mask].mean())
    flags=dict(ensemble_beats_mse_model=bool(a.mean()<references['mse'].mean()),ensemble_beats_mean=bool(a.mean()<references['training_mean'].mean()),
        ensemble_beats_huber_intercept=bool(a.mean()<references['huber_intercept'].mean()),at_least_two_seeds_beat_both_constants=counts['both']>=2,
        at_least_two_seeds_beat_mse_models=counts['model']>=2,at_least_two_years_beat_mse_model=years>=2)
    saved=json.loads((OUT/'assessment.json').read_text(encoding='utf-8'));assert saved['flags']==flags and saved['descriptive_screen_pass']==all(flags.values())
    assert saved['new_seeds_better_than_mean']==counts['mean'] and saved['new_seeds_better_than_huber_intercept']==counts['huber']
    assert saved['new_seeds_better_than_both_constants']==counts['both'] and saved['new_seeds_better_than_mse_models']==counts['model']
    assert saved['new_years_better_than_mse_model']==years
    return flags


def main():
    legacy.initialize();start=time.time();prep=check_frozen();assert old_evidence()==prep['old_evidence']
    run=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'));assert run.get('finished_utc') and run['fits']==9
    assert run['source_sha256']==source_hashes() and run['input_sha256']==input_hashes()
    obs=pd.read_csv(OUT/'observation_table.csv');seed=pd.read_csv(OUT/'all_seed_predictions.csv');ensemble=pd.read_csv(OUT/'ensemble_predictions.csv')
    constants=pd.read_csv(OUT/'constant_predictions.csv');curves=pd.read_csv(OUT/'huber_training_curves.csv')
    assert len(seed)==846 and len(ensemble)==282 and len(curves)==180 and len(constants)==141
    assert seed.groupby(['rule','date']).size().eq(3).all()
    refs=json.loads((OUT/'mse_checkpoint_manifest.json').read_text(encoding='utf-8'))+json.loads((OUT/'huber_checkpoint_manifest.json').read_text(encoding='utf-8'))
    assert len(refs)==18 and len({(r['rule'],r['cutoff'],r['seed']) for r in refs})==18
    with np.load(V5/'cache/targets.npz') as data:targets={k:data[k].copy() for k in ['returns','auxiliary']}
    scales=json.loads((OUT/'training_scales_and_intercepts.json').read_text(encoding='utf-8'))
    max_score=0.
    for record in scales:
        tr,te=load_fold(record['cutoff']);y=targets['returns'][tr];mean=y.mean();sd=y.std();z=(y-mean)/sd
        left=float(z.min()-1);right=float(z.max()+1)
        for _ in range(120):
            center=(left+right)/2;score=float(np.mean(np.maximum(-1,np.minimum(1,center-z))))
            if score>0:right=center
            else:left=center
        center=(left+right)/2;location=mean+sd*center
        assert abs(record['huber_location_standardized']-center)<1e-12 and abs(record['huber_location_return']-location)<1e-12
        residual=float(np.mean(np.maximum(-1,np.minimum(1,center-z))));assert abs(residual)<1e-12;max_score=max(max_score,abs(residual))
        g=constants[constants.cutoff==record['cutoff']].sort_values('row_index');np.testing.assert_array_equal(g.row_index.to_numpy(),te)
        np.testing.assert_allclose(g.huber_intercept,location,rtol=0,atol=1e-12);np.testing.assert_allclose(g.training_mean,mean,rtol=0,atol=1e-12)
        np.testing.assert_allclose(g.training_sd,sd,rtol=0,atol=1e-12);np.testing.assert_allclose(g.actual,targets['returns'][te],rtol=0,atol=1e-12)
    maximum=0.;shift_maximum=0.;orders=0;boundaries=0
    for number,r in enumerate(refs,1):
        path=PROJECT/r['project_file'];assert sha(path)==r['sha256'];state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['arm']=='combined' and state['epoch']==20
        tr,te=load_fold(r['cutoff']);n=len(tr)
        np.testing.assert_array_equal(tr,np.flatnonzero(obs.joint_completed.le(r['cutoff']).to_numpy()))
        for key,data in targets.items():
            x=data[tr].astype(float);np.testing.assert_allclose(x.mean(axis=0),state['scales'][key+'_mean'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.maximum(x.std(axis=0),1e-6),state['scales'][key+'_sd'],rtol=0,atol=1e-12)
        g=seed[(seed.rule==r['rule'])&(seed.cutoff==r['cutoff'])&(seed.seed==r['seed'])].sort_values('row_index')
        np.testing.assert_array_equal(g.row_index.to_numpy(),te);np.testing.assert_allclose(g.actual,targets['returns'][te],rtol=0,atol=1e-12)
        assert obs.weekday.iloc[te].eq(4).all() and obs.joint_completed.iloc[te].le('2020-12-31').all()
        if r['rule']=='huber':
            group=curves[(curves.cutoff==r['cutoff'])&(curves.seed==r['seed'])].sort_values('epoch');assert group.epoch.tolist()==list(range(1,21))
            rng=np.random.default_rng(r['seed']);edge=np.r_[np.arange(0,n,128),n]
            for row in group.itertuples():
                order=rng.permutation(n);assert row.training_order_sha256==array_hash(tr[order]);orders+=1
                assert row.batch_boundaries_sha256==array_hash(edge);boundaries+=1
                assert row.presentations==n and row.train_n==n and row.optimizer_steps==(n+127)//128
        model=legacy.make_model('combined',r['seed']).eval();model.load_state_dict(state['state_dict'])
        assert sum(p.numel() for p in model.parameters())==38551;values=legacy.batch_tensors(te)
        with torch.inference_mode():
            p,_=legacy.predict(model,values,False);shift,_=legacy.predict(model,{k:torch.roll(t,1,dims=0) for k,t in values.items()},False)
        p=p.cpu().numpy().astype(float)*state['scales']['returns_sd']+state['scales']['returns_mean']
        shift=shift.cpu().numpy().astype(float)*state['scales']['returns_sd']+state['scales']['returns_mean']
        np.testing.assert_allclose(p,g.predicted_return,rtol=0,atol=1e-10);np.testing.assert_allclose(shift,g.reassigned_prediction,rtol=0,atol=1e-10)
        maximum=max(maximum,float(np.max(np.abs(p-g.predicted_return))));shift_maximum=max(shift_maximum,float(np.max(np.abs(shift-g.reassigned_prediction))))
        del model,values;torch.cuda.empty_cache()
        if number%3==0:print(f'Replayed {number}/18 checkpoints',flush=True)
    assert orders==180 and boundaries==180
    for (rule,date),g in seed.groupby(['rule','date']):
        row=ensemble[(ensemble.rule==rule)&(ensemble.date==date)].iloc[0]
        assert abs(row.predicted_return-g.predicted_return.mean())<1e-12 and abs(row.reassigned_prediction-g.reassigned_prediction.mean())<1e-12
    flags=statistics_audit(seed,ensemble)
    # Independently recompute the fixed-model concentration audit from its frozen input.
    old=pd.read_csv(V8/'results/frozen_model_training_predictions.csv');audit=pd.read_csv(OUT/'training_loss_audit.csv');audited=0
    for row in audit[audit.state=='frozen_mse_model'].itertuples():
        g=old[(old.cutoff==row.cutoff)&(old.seed==row.seed)];sd=next(r['returns_sd'] for r in scales if r['cutoff']==row.cutoff)
        e=(g.predicted_return.to_numpy()-g.actual.to_numpy())/sd;absolute=np.abs(e);count=int(np.ceil(len(e)*.01));top=np.argsort(absolute)[-count:]
        quadratic=e**2;robust=np.where(absolute<=1,e**2,2*absolute-1)
        assert abs(row.top_one_percent_mse_share-quadratic[top].sum()/quadratic.sum())<1e-12
        assert abs(row.top_one_percent_huber_share-robust[top].sum()/robust.sum())<1e-12
        assert abs(row.linear_region_fraction-np.mean(absolute>1))<1e-12;audited+=1
    assert audited==9;check_frozen();assert old_evidence()==prep['old_evidence']
    result=dict(status='PASS',new_fits=9,reused_models=9,total_checkpoints_replayed=18,validation_weeks=141,
        epoch_order_hashes_recomputed=orders,boundary_hashes_recomputed=boundaries,training_intercepts_recomputed=3,
        maximum_intercept_score_error=max_score,maximum_prediction_replay_error=maximum,maximum_reassignment_replay_error=shift_maximum,
        primary_comparisons_recomputed=3,frozen_training_loss_audits_recomputed=audited,flags=flags,
        previous_files_preserved=len(prep['old_evidence']),source_input_protocol_preserved=True,elapsed_seconds=time.time()-start,
        completed_utc=pd.Timestamp.now(tz='UTC').isoformat())
    save(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

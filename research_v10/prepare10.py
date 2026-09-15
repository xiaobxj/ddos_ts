"""Freeze one robust-loss candidate, training-only intercepts and loss audit."""
from common10 import *


def audit_rows(errors):
    e=np.asarray(errors,float);a=np.abs(e);count=max(1,int(np.ceil(len(e)*.01)));top=np.argsort(a)[-count:]
    mse=e**2;huber=numpy_robust(e);g=2*a;h=2*np.minimum(a,1.)
    return dict(n=len(e),linear_region_fraction=float(np.mean(a>1.)),standardized_mse=float(mse.mean()),
        scaled_huber_loss=float(huber.mean()),top_one_percent_n=count,
        top_one_percent_mse_share=float(mse[top].sum()/mse.sum()),top_one_percent_huber_share=float(huber[top].sum()/huber.sum()),
        mean_absolute_mse_derivative=float(g.mean()),mean_absolute_huber_derivative=float(h.mean()),
        top_one_percent_mse_derivative_share=float(g[top].sum()/g.sum()),top_one_percent_huber_derivative_share=float(h[top].sum()/h.sum()))


def main():
    legacy.initialize();OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    assert not (OUT/'preparation_manifest.json').exists()
    run=manifest('preparation');(OUT/'observation_table.csv').write_bytes((V9/'results/observation_table.csv').read_bytes())
    obs=pd.read_csv(OUT/'observation_table.csv')
    for path in (V9/'cache').glob('masks_*.npz'):(CACHE/path.name).write_bytes(path.read_bytes())
    with np.load(V5/'cache/targets.npz') as data:raw={k:data[k].astype(float) for k in ['returns','auxiliary']}
    scales=[];baselines=[];summaries=[];periods=[]
    training=pd.read_csv(V8/'results/frozen_model_training_predictions.csv');assert len(training)==17292
    for fold in cfg()['folds']:
        tr,te=load_fold(fold['cutoff']);record=dict(cutoff=fold['cutoff'],train_n=len(tr))
        for key,data in raw.items():
            mean=data[tr].mean(axis=0);sd=np.maximum(data[tr].std(axis=0),1e-6)
            record[key+'_mean']=np.asarray(mean).tolist();record[key+'_sd']=np.asarray(sd).tolist()
        y=raw['returns'][tr];z=(y-record['returns_mean'])/record['returns_sd']
        center=huber_location(z);location=record['returns_mean']+record['returns_sd']*center
        record.update(huber_location_standardized=center,huber_location_return=location,
            huber_location_score=float(np.mean(np.clip(center-z,-1.,1.))))
        scales.append(record)
        for i in te:baselines.append(dict(cutoff=fold['cutoff'],date=obs.date.iloc[i],row_index=int(i),actual=float(raw['returns'][i]),
            training_mean=record['returns_mean'],huber_intercept=location,zero_return=0.,training_sd=record['returns_sd']))
        summaries.append(dict(cutoff=fold['cutoff'],state='zero_standardized_prediction',seed=None,**audit_rows(-z)))
        g=training[training.cutoff==fold['cutoff']]
        for seed,h in g.groupby('seed'):
            h=h.sort_values('row_index');np.testing.assert_array_equal(h.row_index.to_numpy(),tr)
            np.testing.assert_allclose(h.actual,raw['returns'][tr],rtol=0,atol=1e-12)
            errors=(h.predicted_return.to_numpy()-h.actual.to_numpy())/record['returns_sd']
            summaries.append(dict(cutoff=fold['cutoff'],state='frozen_mse_model',seed=int(seed),**audit_rows(errors)))
            dates=h.date.str[:4].to_numpy();loss=errors**2;robust=numpy_robust(errors)
            for year in sorted(set(dates)):
                mask=dates==year
                periods.append(dict(cutoff=fold['cutoff'],seed=int(seed),year=int(year),rows=int(mask.sum()),
                    sample_share=float(mask.mean()),mse_loss_share=float(loss[mask].sum()/loss.sum()),
                    huber_loss_share=float(robust[mask].sum()/robust.sum())))
    save(OUT/'training_scales_and_intercepts.json',scales)
    pd.DataFrame(baselines).to_csv(OUT/'constant_predictions.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'training_loss_audit.csv',index=False)
    pd.DataFrame(periods).to_csv(OUT/'yearly_training_loss_attribution.csv',index=False)
    save(OUT/'fit_plan.json',fit_plan())
    old=pd.read_csv(V9/'results/legacy_predictions.csv');assert len(old)==423
    old['rule']='mse';old.to_csv(OUT/'mse_predictions.csv',index=False)
    refs=json.loads((V9/'results/legacy_checkpoint_manifest.json').read_text(encoding='utf-8'));assert len(refs)==9
    for r in refs:
        r['rule']='mse';assert sha(PROJECT/r['project_file'])==r['sha256']
        saved=torch.load(PROJECT/r['project_file'],map_location='cpu',weights_only=True)
        record=next(x for x in scales if x['cutoff']==r['cutoff'])
        assert saved['scales']=={k:record[k] for k in saved['scales']}
    save(OUT/'mse_checkpoint_manifest.json',refs)
    old_curves=pd.read_csv(V9/'results/legacy_training_curves.csv');old_curves['rule']='mse'
    old_curves.to_csv(OUT/'mse_training_curves.csv',index=False)
    frozen=list(CACHE.glob('*.npz'))+[OUT/n for n in ['observation_table.csv','training_scales_and_intercepts.json','constant_predictions.csv',
        'training_loss_audit.csv','yearly_training_loss_attribution.csv','fit_plan.json','mse_predictions.csv','mse_checkpoint_manifest.json','mse_training_curves.csv']]
    run.update(finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),planned_fits=9,reused_models=9,intercepts=3,
        local_sha256={str(p.relative_to(ROOT)):sha(p) for p in frozen})
    save(OUT/'preparation_manifest.json',run)
    print(pd.DataFrame(scales)[['cutoff','train_n','returns_mean','returns_sd','huber_location_return','huber_location_score']].to_string(index=False))
    print(pd.DataFrame(summaries)[['cutoff','state','seed','linear_region_fraction','top_one_percent_mse_share','top_one_percent_huber_share']].to_string(index=False))
    print(f'Frozen9 new fits,9 old models,3 intercepts; preserved {len(run["old_evidence"])} previous files.')


if __name__=='__main__':main()

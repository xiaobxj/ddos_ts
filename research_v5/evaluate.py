"""Score all frozen chronological candidates and prespecified paired diagnostics."""
from common5 import *
import importlib.util
sys.path.append(str(V3))

spec=importlib.util.spec_from_file_location('v3_statistics_for_v5',V3/'evaluate.py')
v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3)


def metric(g):
    y=g.actual.to_numpy(float);p=g.predicted_return.to_numpy(float)
    result=stats(p,y);positive=y>0
    result.update(n=len(y),balanced_accuracy=float(((p[positive]>0).mean()+(p[~positive]<=0).mean())/2),
                  forecast_positive_fraction=float((p>0).mean()),actual_positive_fraction=float(positive.mean()))
    if 'cutoff' in g:
        residual=p-g.groupby('cutoff').predicted_return.transform('mean').to_numpy()
        total=float(np.sum((p-p.mean())**2))
        result.update(within_fold_forecast_std=float(np.sqrt(np.mean(residual**2))),
                      between_fold_share_of_forecast_variation=float(1-np.sum(residual**2)/total) if total else None)
    if 'training_mean' in g:
        mse_mean=float(np.mean((g.training_mean.to_numpy()-y)**2))
        result.update(mse_skill_vs_training_mean=1-result['mse']/mse_mean)
    if 'reassigned_prediction' in g:
        result.update(reassigned_mse=float(np.mean((g.reassigned_prediction-y)**2)),
                      reassignment_mse_increase=float(np.mean((g.reassigned_prediction-y)**2))-result['mse'])
    return result


def main():
    run=json.loads((OUT/'validation_manifest.json').read_text(encoding='utf-8'))
    assert run.get('finished_utc')
    seed=pd.read_csv(OUT/'validation_seed_predictions.csv')
    baselines=pd.read_csv(OUT/'validation_baselines.csv')
    means=seed.groupby(['variant','epoch','date'],sort=False).agg(predicted_return=('predicted_return','mean'),
        reassigned_prediction=('reassigned_prediction','mean'),actual=('actual','first'),cutoff=('cutoff','first')).reset_index()
    means=means.merge(baselines[['date','training_mean']],on='date',validate='many_to_one')
    metrics=[];yearly=[];seed_metrics=[]
    for (variant,epoch),g in means.groupby(['variant','epoch']):
        assert len(g)==len(baselines)
        metrics.append(dict(variant=variant,epoch=int(epoch),**metric(g)))
        for year,h in g.groupby(g.date.str[:4]):yearly.append(dict(variant=variant,epoch=int(epoch),year=int(year),**metric(h)))
    for (variant,epoch,number),g in seed.groupby(['variant','epoch','seed']):
        g=g.merge(baselines[['date','training_mean']],on='date',validate='one_to_one')
        seed_metrics.append(dict(variant=variant,epoch=int(epoch),seed=int(number),**metric(g)))
    base=[]
    for name in ['training_mean','zero_return']:
        g=baselines.copy();g['predicted_return']=g[name]
        base.append(dict(method=name,**metric(g)))
    table=pd.DataFrame(metrics)
    order={v['name']:i for i,v in enumerate(cfg()['variants'])}
    selected=min(metrics,key=lambda r:(r['rmse'],r['epoch'],order[r['variant']]))
    save(OUT/'selection.json',dict(selected=selected,baseline_metrics=base,
         selection_rule=cfg()['validation']['selection'],candidate_count=len(metrics),validation_weeks=len(baselines),
         selection_after_all_models_finished=True,selection_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    pd.DataFrame(metrics).to_csv(OUT/'validation_metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(OUT/'validation_yearly_metrics.csv',index=False)
    pd.DataFrame(seed_metrics).to_csv(OUT/'validation_seed_metrics.csv',index=False)
    means.to_csv(OUT/'validation_ensemble_predictions.csv',index=False)
    pairs=[]
    for epoch in [5,20]:
        reference=means[(means.variant=='return_zero')&(means.epoch==epoch)].sort_values('date')
        ids=v3.bootstrap_indices(len(reference))
        for variant in ['return_nonzero','joint_ohlcv']:
            g=means[(means.variant==variant)&(means.epoch==epoch)].sort_values('date')
            assert g.date.tolist()==reference.date.tolist()
            y=g.actual.to_numpy();a=(g.predicted_return.to_numpy()-y)**2;b=(reference.predicted_return.to_numpy()-y)**2
            delta=np.sqrt(a[ids].mean(axis=1))-np.sqrt(b[ids].mean(axis=1))
            pairs.append(dict(variant=variant,reference='return_zero',epoch=epoch,
                              mse=v3.difference(a-b,ids),
                              rmse=dict(difference=float(np.sqrt(a.mean())-np.sqrt(b.mean())),
                                        ci95_low=float(np.quantile(delta,.025)),ci95_high=float(np.quantile(delta,.975))),
                              accuracy=v3.difference(((g.predicted_return>0)==(g.actual>0)).to_numpy(float)-
                                                     ((reference.predicted_return>0)==(reference.actual>0)).to_numpy(float),ids)))
    for r,p in zip(pairs,v3.holm([r['mse']['p'] for r in pairs])):r['mse']['holm_adjusted_p']=p
    save(OUT/'paired_comparisons.json',pairs)
    # Carry over the previously selected validation predictor without refitting.
    # Its old training sample set differs by the five stricter auxiliary exclusions.
    prior_choice=json.loads((V4/'results/selection.json').read_text(encoding='utf-8'))['selected_epochs']
    prior=pd.read_csv(V4/'results/validation_seed_predictions.csv')
    prior=prior[prior.epoch==prior_choice].groupby('date').agg(
        predicted_return=('predicted_return','mean'),actual=('actual','first'),cutoff=('cutoff','first')).reset_index()
    current=means[(means.variant==selected['variant'])&(means.epoch==selected['epoch'])].sort_values('date')
    prior=prior.sort_values('date')
    assert prior.date.tolist()==current.date.tolist()
    np.testing.assert_allclose(prior.actual,current.actual,rtol=0,atol=1e-12)
    save(OUT/'frozen_round4_reference.json',dict(method='round4_raw_fixed',epochs=prior_choice,
        **metric(prior),same_validation_dates=True,training_sample_set_changed=True,
        current_selected_minus_previous_rmse=selected['rmse']-metric(prior)['rmse'],
        role='Unchanged historical reference, not a controlled attribution of the five sample exclusions.'))
    obs=pd.read_csv(OUT/'observation_table.csv');overlap=[]
    for fold in cfg()['validation']['folds']:
        g=obs[obs.joint_completed<=fold['cutoff']]
        adjacent=np.diff(g.anchor.to_numpy())==1
        y=g.exec_return.to_numpy();a=y[:-1][adjacent];b=y[1:][adjacent]
        overlap.append(dict(cutoff=fold['cutoff'],daily_training_rows=len(g),adjacent_pairs=int(adjacent.sum()),
                            adjacent_label_correlation=float(np.corrcoef(a,b)[0,1]),
                            interpretation='Overlapping-label diagnostic; not an estimate of independent sample size.'))
    save(OUT/'training_label_overlap.json',overlap)
    capacity=json.loads((OUT/'capacity_summary.json').read_text(encoding='utf-8'))
    learning=pd.read_csv(OUT/'capacity_curves.csv')
    learning=learning[(learning.label_kind=='real')&(learning.variant!='mlp')&(learning.return_mse<=.1)]
    learning.groupby(['representation','variant','seed']).step.min().reset_index(
        name='first_logged_step_at_mse_0_1').to_csv(OUT/'capacity_learning_speed.csv',index=False)
    rows=[]
    for (rep,variant,label),g in pd.DataFrame(capacity).groupby(['representation','variant','label_kind']):
        rows.append(dict(representation=rep,variant=variant,label_kind=label,runs=len(g),passes=int(g.capacity_pass.sum()),
                         minimum_final_mse=float(g.mse.min()),median_final_mse=float(g.mse.median()),maximum_final_mse=float(g.mse.max())))
    pd.DataFrame(rows).to_csv(OUT/'capacity_aggregated.csv',index=False)
    print(table[['variant','epoch','n','accuracy','rmse','mse_skill_vs_training_mean','forecast_std','reassignment_mse_increase']].to_string(index=False))
    print(json.dumps(dict(selected=selected,baselines=base),indent=2))


if __name__=='__main__':main()

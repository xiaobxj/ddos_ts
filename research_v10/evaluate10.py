"""Fixed MSE primary endpoint and robust-location controls, without selection."""
from common10 import *
import evaluate6 as statistics


def metric(g):
    result=statistics.metric(g);e=g.predicted_return.to_numpy()-g.actual.to_numpy()
    result.update(mae=float(np.mean(np.abs(e))),scaled_huber_error=float(numpy_robust(e/g.training_sd.to_numpy()).mean()),
        mse_skill_vs_huber_intercept=1-result['mse']/float(np.mean((g.huber_intercept-g.actual)**2)),
        reassigned_mse=float(np.mean((g.reassigned_prediction-g.actual)**2)))
    result['reassignment_mse_relative_change']=result['reassigned_mse']/result['mse']-1
    return result


def paired(a,b,reference):
    a=a.sort_values('date');b=b.sort_values('date');assert a.date.tolist()==b.date.tolist()
    y=a.actual.to_numpy();x=(a.predicted_return.to_numpy()-y)**2;z=(b.predicted_return.to_numpy()-y)**2
    ids=statistics.bootstrap_indices(len(a));delta=np.sqrt(x[ids].mean(axis=1))-np.sqrt(z[ids].mean(axis=1))
    return dict(candidate='huber',reference=reference,n=len(a),mse=statistics.difference(x-z,ids),
        rmse=dict(difference=float(np.sqrt(x.mean())-np.sqrt(z.mean())),ci95_low=float(np.quantile(delta,.025)),ci95_high=float(np.quantile(delta,.975))),
        accuracy=statistics.difference(((a.predicted_return>0)==(a.actual>0)).to_numpy(float)-((b.predicted_return>0)==(b.actual>0)).to_numpy(float),ids))


def main():
    check_frozen();run=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'));assert run.get('finished_utc') and run['fits']==9
    seed=pd.concat([pd.read_csv(OUT/'mse_predictions.csv'),pd.read_csv(OUT/'huber_predictions.csv')],ignore_index=True)
    constants=pd.read_csv(OUT/'constant_predictions.csv')
    seed=seed.merge(constants[['date','huber_intercept','training_sd']],on='date',validate='many_to_one')
    assert len(seed)==846 and not seed.duplicated(['rule','seed','date']).any()
    assert seed.groupby(['rule','date']).size().eq(3).all()
    ensemble=seed.groupby(['rule','date'],sort=False).agg(predicted_return=('predicted_return','mean'),
        reassigned_prediction=('reassigned_prediction','mean'),actual=('actual','first'),cutoff=('cutoff','first'),training_mean=('training_mean','first'),
        huber_intercept=('huber_intercept','first'),training_sd=('training_sd','first')).reset_index()
    rows=[];years=[];seeds=[];sensitivity=[];base_metrics=[]
    for rule,g in ensemble.groupby('rule',sort=False):
        rows.append(dict(rule=rule,**metric(g)))
        for year,h in g.groupby(g.date.str[:4]):years.append(dict(rule=rule,year=int(year),**metric(h)))
    for (rule,number),g in seed.groupby(['rule','seed'],sort=False):seeds.append(dict(rule=rule,seed=int(number),**metric(g)))
    metrics=pd.DataFrame(rows);yearly=pd.DataFrame(years);seed_metrics=pd.DataFrame(seeds)
    old=ensemble[ensemble.rule=='mse'].sort_values('date');new=ensemble[ensemble.rule=='huber'].sort_values('date')
    references={'mse':old}
    for name in ['training_mean','huber_intercept','zero_return']:
        g=constants.copy();g['predicted_return']=g[name];g['reassigned_prediction']=g[name]
        references[name]=g;base_metrics.append(dict(reference=name,**metric(g)))
    primary=[paired(new,references[name],name) for name in ['mse','training_mean','huber_intercept']]
    for r,p in zip(primary,statistics.holm([r['mse']['p'] for r in primary])):r['mse']['holm_adjusted_p']=p
    matched=seed_metrics.pivot(index='seed',columns='rule',values='mse');yp=yearly.pivot(index='year',columns='rule',values='mse')
    a=metrics[metrics.rule=='huber'].iloc[0];b=metrics[metrics.rule=='mse'].iloc[0];individual=seed_metrics[seed_metrics.rule=='huber']
    both=(individual.mse_skill_vs_training_mean>0)&(individual.mse_skill_vs_huber_intercept>0)
    flags=dict(ensemble_beats_mse_model=bool(a.mse<b.mse),ensemble_beats_mean=bool(a.mse_skill_vs_training_mean>0),
        ensemble_beats_huber_intercept=bool(a.mse_skill_vs_huber_intercept>0),at_least_two_seeds_beat_both_constants=bool(both.sum()>=2),
        at_least_two_seeds_beat_mse_models=bool((matched.huber<matched.mse).sum()>=2),at_least_two_years_beat_mse_model=bool((yp.huber<yp.mse).sum()>=2))
    assessment=dict(flags=flags,descriptive_screen_pass=all(flags.values()),new_seeds_better_than_mean=int((individual.mse_skill_vs_training_mean>0).sum()),
        new_seeds_better_than_huber_intercept=int((individual.mse_skill_vs_huber_intercept>0).sum()),new_seeds_better_than_both_constants=int(both.sum()),
        new_seeds_better_than_mse_models=int((matched.huber<matched.mse).sum()),new_years_better_than_mse_model=int((yp.huber<yp.mse).sum()),
        mse_skill_vs_mse_model=1-float(a.mse/b.mse),primary_significant_improvements=sum(r['mse']['difference']<0 and r['mse']['holm_adjusted_p']<.05 for r in primary),
        primary_comparisons=3,exploratory_historical_reuse=True,no_selection=True,scored_utc=pd.Timestamp.now(tz='UTC').isoformat())
    for level,number,frame in [('ensemble',None,ensemble)]+[('seed',int(n),g) for n,g in seed.groupby('seed')]:
        a=frame[frame.rule=='mse'].sort_values('date');b=frame[frame.rule=='huber'].sort_values('date')
        assert a.date.tolist()==b.date.tolist();np.testing.assert_allclose(a.training_mean,b.training_mean,rtol=0,atol=1e-12)
        delta=b.predicted_return.to_numpy()-a.predicted_return.to_numpy()
        sensitivity.append(dict(level=level,seed=number,forecast_rms_difference=float(np.sqrt(np.mean(delta**2))),
            changed_direction_weeks=int(((a.predicted_return.to_numpy()>0)!=(b.predicted_return.to_numpy()>0)).sum()),mse_change=metric(b)['mse']-metric(a)['mse']))
    for name,frame in [('all_seed_predictions',seed),('ensemble_predictions',ensemble),('ensemble_metrics',metrics),
                       ('yearly_metrics',yearly),('seed_metrics',seed_metrics),('prediction_sensitivity',pd.DataFrame(sensitivity))]:frame.to_csv(OUT/f'{name}.csv',index=False)
    pd.DataFrame(base_metrics).to_csv(OUT/'baseline_metrics.csv',index=False)
    matched.to_csv(OUT/'matched_seed_mse.csv');yp.to_csv(OUT/'matched_year_mse.csv')
    save(OUT/'primary_comparisons.json',primary);save(OUT/'assessment.json',assessment)
    print(metrics.to_string(index=False));print(pd.DataFrame(base_metrics)[['reference','rmse','accuracy','mae','scaled_huber_error']].to_string(index=False))
    print(json.dumps(assessment,indent=2))


if __name__=='__main__':main()

"""One fixed candidate, two prespecified paired comparisons; no selection."""
from common9 import *
import evaluate6 as statistics


def metric(g):
    result=statistics.metric(g)
    result['reassigned_mse']=float(np.mean((g.reassigned_prediction-g.actual)**2))
    result['reassignment_mse_relative_change']=result['reassigned_mse']/result['mse']-1
    return result


def paired(a,b,reference):
    a=a.sort_values('date');b=b.sort_values('date');assert a.date.tolist()==b.date.tolist()
    y=a.actual.to_numpy();x=(a.predicted_return.to_numpy()-y)**2;z=(b.predicted_return.to_numpy()-y)**2
    ids=statistics.bootstrap_indices(len(a));delta=np.sqrt(x[ids].mean(axis=1))-np.sqrt(z[ids].mean(axis=1))
    return dict(candidate='balanced',reference=reference,n=len(a),mse=statistics.difference(x-z,ids),
        rmse=dict(difference=float(np.sqrt(x.mean())-np.sqrt(z.mean())),ci95_low=float(np.quantile(delta,.025)),ci95_high=float(np.quantile(delta,.975))),
        accuracy=statistics.difference(((a.predicted_return>0)==(a.actual>0)).to_numpy(float)-((b.predicted_return>0)==(b.actual>0)).to_numpy(float),ids))


def main():
    check_frozen();run=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
    assert run.get('finished_utc') and run['fits']==9
    seed=pd.concat([pd.read_csv(OUT/'legacy_predictions.csv'),pd.read_csv(OUT/'balanced_predictions.csv')],ignore_index=True)
    assert len(seed)==846 and not seed.duplicated(['rule','seed','date']).any()
    assert seed.groupby(['rule','date']).size().eq(3).all()
    ensemble=seed.groupby(['rule','date'],sort=False).agg(predicted_return=('predicted_return','mean'),
        reassigned_prediction=('reassigned_prediction','mean'),actual=('actual','first'),cutoff=('cutoff','first'),
        training_mean=('training_mean','first')).reset_index()
    rows=[];years=[];seeds=[];sensitivity=[]
    for rule,g in ensemble.groupby('rule',sort=False):
        rows.append(dict(rule=rule,**metric(g)))
        for year,h in g.groupby(g.date.str[:4]):years.append(dict(rule=rule,year=int(year),**metric(h)))
    for (rule,number),g in seed.groupby(['rule','seed'],sort=False):seeds.append(dict(rule=rule,seed=int(number),**metric(g)))
    metrics=pd.DataFrame(rows);yearly=pd.DataFrame(years);seed_metrics=pd.DataFrame(seeds)
    old=ensemble[ensemble.rule=='legacy'].sort_values('date');new=ensemble[ensemble.rule=='balanced'].sort_values('date')
    mu=old.copy();mu['predicted_return']=mu.training_mean;mu['reassigned_prediction']=mu.training_mean
    zero=mu.copy();zero['predicted_return']=0.;zero['reassigned_prediction']=0.
    comparisons=[paired(new,old,'legacy'),paired(new,mu,'training_mean')]
    for r,p in zip(comparisons,statistics.holm([r['mse']['p'] for r in comparisons])):r['mse']['holm_adjusted_p']=p
    matched=seed_metrics.pivot(index='seed',columns='rule',values='mse')
    year_pivot=yearly.pivot(index='year',columns='rule',values='mse')
    new_m=metrics[metrics.rule=='balanced'].iloc[0];old_m=metrics[metrics.rule=='legacy'].iloc[0]
    flags=dict(ensemble_beats_legacy=bool(new_m.mse<old_m.mse),ensemble_beats_mean=bool(new_m.mse_skill_vs_training_mean>0),
        at_least_two_seeds_beat_mean=bool((seed_metrics[seed_metrics.rule=='balanced'].mse_skill_vs_training_mean>0).sum()>=2),
        at_least_two_seeds_beat_legacy=bool((matched.balanced<matched.legacy).sum()>=2),
        at_least_two_years_beat_legacy=bool((year_pivot.balanced<year_pivot.legacy).sum()>=2))
    assessment=dict(flags=flags,descriptive_screen_pass=all(flags.values()),new_seeds_better_than_mean=int((seed_metrics[seed_metrics.rule=='balanced'].mse_skill_vs_training_mean>0).sum()),
        new_seeds_better_than_legacy=int((matched.balanced<matched.legacy).sum()),new_years_better_than_legacy=int((year_pivot.balanced<year_pivot.legacy).sum()),
        mse_skill_vs_legacy=1-float(new_m.mse/old_m.mse),primary_significant_improvements=sum(r['mse']['difference']<0 and r['mse']['holm_adjusted_p']<.05 for r in comparisons),
        primary_comparisons=2,exploratory_historical_reuse=True,no_selection=True,scored_utc=pd.Timestamp.now(tz='UTC').isoformat())
    for level,number,frame in [('ensemble',None,ensemble)]+[('seed',int(n),g) for n,g in seed.groupby('seed')]:
        a=frame[frame.rule=='legacy'].sort_values('date');b=frame[frame.rule=='balanced'].sort_values('date')
        assert a.date.tolist()==b.date.tolist()
        np.testing.assert_allclose(a.training_mean,b.training_mean,rtol=0,atol=1e-12)
        difference=b.predicted_return.to_numpy()-a.predicted_return.to_numpy()
        sensitivity.append(dict(level=level,seed=number,forecast_rms_difference=float(np.sqrt(np.mean(difference**2))),
            changed_direction_weeks=int(((a.predicted_return.to_numpy()>0)!=(b.predicted_return.to_numpy()>0)).sum()),
            mse_change=metric(b)['mse']-metric(a)['mse']))
    seed.to_csv(OUT/'all_seed_predictions.csv',index=False);ensemble.to_csv(OUT/'ensemble_predictions.csv',index=False)
    metrics.to_csv(OUT/'ensemble_metrics.csv',index=False);yearly.to_csv(OUT/'yearly_metrics.csv',index=False)
    seed_metrics.to_csv(OUT/'seed_metrics.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'prediction_sensitivity.csv',index=False)
    matched.to_csv(OUT/'matched_seed_mse.csv');year_pivot.to_csv(OUT/'matched_year_mse.csv')
    save(OUT/'baseline_metrics.json',dict(training_mean=metric(mu),zero=metric(zero)))
    save(OUT/'primary_comparisons.json',comparisons);save(OUT/'assessment.json',assessment)
    print(metrics.to_string(index=False));print(json.dumps(assessment,indent=2))


if __name__=='__main__':main()

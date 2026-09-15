"""Prespecified ten-test family; all sampling policies remain separately reported."""
from common8 import *
import evaluate6 as statistics


def metric(g):
    result = statistics.metric(g)
    result['mse_skill_vs_full_mean'] = 1-result['mse']/float(np.mean((g.full_training_mean-g.actual)**2))
    result['reassigned_mse'] = float(np.mean((g.reassigned_prediction-g.actual)**2))
    result['reassignment_mse_relative_change'] = result['reassigned_mse']/result['mse']-1
    return result


def paired(a, b, policy, reference):
    a = a.sort_values('date'); b = b.sort_values('date')
    assert a.date.tolist()==b.date.tolist()
    y = a.actual.to_numpy(); x = (a.predicted_return.to_numpy()-y)**2; z = (b.predicted_return.to_numpy()-y)**2
    ids = statistics.bootstrap_indices(len(a))
    delta = np.sqrt(x[ids].mean(axis=1))-np.sqrt(z[ids].mean(axis=1))
    return dict(policy=policy, reference=reference, n=len(a), mse=statistics.difference(x-z, ids),
                rmse=dict(difference=float(np.sqrt(x.mean())-np.sqrt(z.mean())), ci95_low=float(np.quantile(delta,.025)), ci95_high=float(np.quantile(delta,.975))),
                accuracy=statistics.difference(((a.predicted_return>0)==(a.actual>0)).to_numpy(float)-((b.predicted_return>0)==(b.actual>0)).to_numpy(float),ids))


def main():
    check_frozen()
    frames = [pd.read_csv(OUT / 'reused_full_predictions.csv')]
    for worker in [0,1]:
        run = json.loads((OUT / f'worker{worker}_manifest.json').read_text(encoding='utf-8'))
        assert run.get('finished_utc') and run['fits']==[23,22][worker]
        frames.append(pd.read_csv(OUT / f'worker{worker}_predictions.csv'))
    linear_run = json.loads((OUT / 'linear_manifest.json').read_text(encoding='utf-8'))
    assert linear_run.get('finished_utc') and linear_run['fits']==18
    seed = pd.concat(frames, ignore_index=True)
    assert len(seed)==2538 and not seed.duplicated(['policy','seed','date']).any()
    assert seed.groupby(['policy','date']).size().eq(3).all()
    common = seed[seed.policy=='full'][['date','training_mean']].drop_duplicates().rename(columns={'training_mean':'full_training_mean'})
    seed = seed.merge(common, on='date', validate='many_to_one')
    neural = seed.groupby(['policy','date'], sort=False).agg(predicted_return=('predicted_return','mean'),
        reassigned_prediction=('reassigned_prediction','mean'), actual=('actual','first'), cutoff=('cutoff','first'),
        training_mean=('training_mean','first'), full_training_mean=('full_training_mean','first')).reset_index()
    neural['model'] = 'neural'
    linear = pd.read_csv(OUT / 'linear_predictions.csv').merge(common,on='date',validate='many_to_one'); linear['model']='ridge'
    ensemble = pd.concat([neural,linear],ignore_index=True)
    rows = []; years = []; seeds = []; baselines = []; comparisons = []; primary = []
    for (model,policy), g in ensemble.groupby(['model','policy'],sort=False):
        assert len(g)==141
        rows.append(dict(model=model,policy=policy,**metric(g)))
        for year,h in g.groupby(g.date.str[:4]): years.append(dict(model=model,policy=policy,year=int(year),**metric(h)))
    for (policy,number),g in seed.groupby(['policy','seed'],sort=False): seeds.append(dict(policy=policy,seed=int(number),**metric(g)))
    metrics = pd.DataFrame(rows); seed_metrics = pd.DataFrame(seeds)
    full = neural[neural.policy=='full']; full_mse = float(np.mean((full.predicted_return-full.actual)**2))
    for policy in cfg()['policies']:
        g = neural[neural.policy==policy].copy(); m = metrics[(metrics.model=='neural')&(metrics.policy==policy)].iloc[0]
        linear_m = metrics[(metrics.model=='ridge')&(metrics.policy==policy)].iloc[0]
        mean = g.copy(); mean['predicted_return']=mean.training_mean
        own_mse = float(np.mean((g.training_mean-g.actual)**2)); common_mse=float(np.mean((g.full_training_mean-g.actual)**2))
        count = int((seed_metrics[seed_metrics.policy==policy].mse_skill_vs_training_mean>0).sum())
        flags = dict(beats_full_neural=bool(m.mse<full_mse), beats_own_mean=bool(m.mse<own_mse),
                     beats_common_full_mean=bool(m.mse<common_mse), at_least_two_seeds_beat_own_mean=count>=2)
        comparisons.append(dict(policy=policy,accuracy=float(m.accuracy),rmse=float(m.rmse),mse_skill_vs_full_neural=1-float(m.mse/full_mse),
            mse_skill_vs_own_mean=float(m.mse_skill_vs_training_mean),mse_skill_vs_common_mean=float(m.mse_skill_vs_full_mean),
            own_mean_rmse=own_mse**.5,full_mean_rmse=common_mse**.5,ridge_rmse=float(linear_m.rmse),
            mse_skill_vs_matched_ridge=1-float(m.mse/linear_m.mse),seeds_better_than_own_mean=count,
            **flags,descriptive_screen_pass=all(flags.values()) if policy!='full' else False))
        for name,column in [('own_mean','training_mean'),('full_mean','full_training_mean'),('zero',None)]:
            h=g.copy(); h['predicted_return']=h[column] if column else 0.; h['reassigned_prediction']=h.predicted_return
            baselines.append(dict(policy=policy,reference=name,**metric(h)))
        if policy!='full':
            primary += [paired(g,full,policy,'full_neural'),paired(g,mean,policy,'own_mean')]
    for r,p in zip(primary,statistics.holm([r['mse']['p'] for r in primary])): r['mse']['holm_adjusted_p']=p
    comparisons=pd.DataFrame(comparisons)
    flags={r['policy']:bool(r['descriptive_screen_pass']) for r in comparisons.to_dict('records') if r['policy']!='full'}
    save(OUT / 'assessment.json',dict(policy_screens=flags,disjoint_family_screen=all(flags[f'disjoint{i}'] for i in range(3)),
        primary_comparisons=10,primary_significant_improvements=sum(r['mse']['difference']<0 and r['mse']['holm_adjusted_p']<.05 for r in primary),
        no_policy_or_phase_selection=True,historical_reuse=True,scored_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    save(OUT / 'primary_comparisons.json',primary)
    pd.DataFrame(baselines).to_csv(OUT / 'baseline_metrics.csv',index=False)
    seed.to_csv(OUT / 'all_seed_predictions.csv',index=False)
    ensemble.to_csv(OUT / 'ensemble_predictions.csv',index=False)
    metrics.to_csv(OUT / 'ensemble_metrics.csv',index=False)
    seed_metrics.to_csv(OUT / 'seed_metrics.csv',index=False)
    pd.DataFrame(years).to_csv(OUT / 'yearly_metrics.csv',index=False)
    comparisons.to_csv(OUT / 'policy_comparisons.csv',index=False)
    print(comparisons.to_string(index=False))
    print(json.dumps(dict(policy_screens=flags,disjoint_family_screen=all(flags[f'disjoint{i}'] for i in range(3))),indent=2))


if __name__ == '__main__':
    main()

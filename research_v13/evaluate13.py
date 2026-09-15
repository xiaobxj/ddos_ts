from common13 import *
import evaluate6 as statistics

def metrics(g):
    r=statistics.metric(g)
    r['mae']=float(np.mean(np.abs(g.predicted_return-g.actual)))
    r['correct_directions']=int(((g.predicted_return>0)==(g.actual>0)).sum())
    if 'original_prediction' in g and g.original_prediction.notna().all():
        r['directions_changed_vs_original']=int(((g.predicted_return>0)!=(g.original_prediction>0)).sum())
    return r

def compute():
    seed=pd.read_csv(OUT/'outer_seed_predictions.csv');ensemble=pd.read_csv(OUT/'outer_ensemble_predictions.csv')
    base=pd.read_csv(OUT/'validation_rows.csv');frames=[ensemble]
    for name in ['training_mean','zero_return']:
        g=base.copy();g['method']=name;g['predicted_return']=g[name];frames.append(g)
    all_predictions=pd.concat(frames,ignore_index=True)
    summary=[];yearly=[];seeds=[]
    for method,g in all_predictions.groupby('method'):
        assert len(g)==141
        summary.append(dict(method=method,**metrics(g)))
        for year,h in g.groupby(g.date.str[:4]):yearly.append(dict(method=method,year=int(year),**metrics(h)))
    for (method,number),g in seed.groupby(['method','seed']):seeds.append(dict(method=method,seed=int(number),**metrics(g)))
    table=pd.DataFrame(summary);years=pd.DataFrame(yearly);seed_table=pd.DataFrame(seeds)
    pairs=[];ids=statistics.bootstrap_indices(141)
    for comparison in cfg()['primary_comparisons']:
        a=all_predictions[all_predictions.method.eq(comparison['candidate'])].sort_values('date')
        b=all_predictions[all_predictions.method.eq(comparison['reference'])].sort_values('date')
        assert a.date.tolist()==b.date.tolist();np.testing.assert_allclose(a.actual,b.actual,rtol=0,atol=1e-12)
        ea=(a.predicted_return.to_numpy()-a.actual.to_numpy())**2
        eb=(b.predicted_return.to_numpy()-b.actual.to_numpy())**2
        dr=np.sqrt(ea[ids].mean(axis=1))-np.sqrt(eb[ids].mean(axis=1))
        pairs.append(dict(**comparison,mse=statistics.difference(ea-eb,ids),rmse=dict(difference=float(np.sqrt(ea.mean())-np.sqrt(eb.mean())),
            ci95_low=float(np.quantile(dr,.025)),ci95_high=float(np.quantile(dr,.975))),
            accuracy=statistics.difference(((a.predicted_return>0)==(a.actual>0)).to_numpy(float)-((b.predicted_return>0)==(b.actual>0)).to_numpy(float),ids)))
    for r,p in zip(pairs,statistics.holm([r['mse']['p'] for r in pairs])):r['mse']['holm_adjusted_p']=p
    t=table.set_index('method');s=seed_table.pivot(index='seed',columns='method',values='mse')
    y=years.pivot(index='year',columns='method',values='mse')
    counts=dict(seeds_beating_original=int((s.rolling_shrink<s.archived20).sum()),
        seeds_beating_training_mean=int((s.rolling_shrink<t.loc['training_mean','mse']).sum()),
        years_beating_original=int((y.rolling_shrink<y.archived20).sum()),
        years_beating_training_mean=int((y.rolling_shrink<y.training_mean).sum()))
    flags=dict(ensemble_beats_original=bool(t.loc['rolling_shrink','mse']<t.loc['archived20','mse']),
        ensemble_beats_training_mean=bool(t.loc['rolling_shrink','mse']<t.loc['training_mean','mse']),
        at_least_two_seeds_beat_original=counts['seeds_beating_original']>=2,
        at_least_two_seeds_beat_training_mean=counts['seeds_beating_training_mean']>=2,
        at_least_two_years_beat_original=counts['years_beating_original']>=2,
        at_least_two_years_beat_training_mean=counts['years_beating_training_mean']>=2)
    assessment=dict(candidate='rolling_shrink',counts=counts,flags=flags,descriptive_screen_pass=all(flags.values()),
        mse_skill_vs_original=float(1-t.loc['rolling_shrink','mse']/t.loc['archived20','mse']),
        significant_candidate_improvements=sum(r['candidate']=='rolling_shrink' and r['mse']['difference']<0 and r['mse']['holm_adjusted_p']<.05 for r in pairs),
        independent_confirmation=False,strategy_promotion=False)
    return table,years,seed_table,pairs,assessment

def main():
    check_frozen();scoring=read(OUT/'scoring_manifest.json');assert scoring.get('finished_utc')
    for n,d in scoring['artifacts'].items():assert sha(OUT/n)==d,n
    assert not (OUT/'evaluation_manifest.json').exists(),'Preserve evaluation'
    run=manifest('evaluation');table,years,seeds,pairs,assessment=compute()
    table.to_csv(OUT/'ensemble_metrics.csv',index=False);years.to_csv(OUT/'yearly_metrics.csv',index=False)
    seeds.to_csv(OUT/'seed_metrics.csv',index=False);save(OUT/'primary_comparisons.json',pairs);save(OUT/'assessment.json',assessment)
    run.update(finished_utc=now(),primary_comparisons=4,artifacts={n:sha(OUT/n) for n in ['ensemble_metrics.csv','yearly_metrics.csv','seed_metrics.csv','primary_comparisons.json','assessment.json']})
    save(OUT/'evaluation_manifest.json',run)
    print(table[['method','accuracy','rmse','mse_skill_vs_training_mean']].to_string(index=False))
    print(json.dumps(assessment),flush=True)

if __name__=='__main__':main()

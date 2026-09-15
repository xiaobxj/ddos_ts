"""Four frozen contrasts, all seeds/years and training diagnostics; no candidate selection."""
from common12 import *
import evaluate6 as statistics


def metric(g):
    result=statistics.metric(g)
    result['mae']=float(np.mean(np.abs(g.predicted_return-g.actual)))
    result['reassigned_mse']=float(np.mean((g.reassigned_prediction-g.actual)**2))
    result['reassignment_mse_relative_change']=result['reassigned_mse']/result['mse']-1
    return result


def primary_pair(a,b,candidate,reference):
    a=a.sort_values('date');b=b.sort_values('date');assert a.date.tolist()==b.date.tolist()
    y=a.actual.to_numpy();x=(a.predicted_return.to_numpy()-y)**2;z=(b.predicted_return.to_numpy()-y)**2
    ids=statistics.bootstrap_indices(len(a));delta=np.sqrt(x[ids].mean(axis=1))-np.sqrt(z[ids].mean(axis=1))
    return dict(candidate=candidate,reference=reference,n=len(a),mse=statistics.difference(x-z,ids),
        rmse=dict(difference=float(np.sqrt(x.mean())-np.sqrt(z.mean())),ci95_low=float(np.quantile(delta,.025)),
                  ci95_high=float(np.quantile(delta,.975))),
        accuracy=statistics.difference(((a.predicted_return>0)==(a.actual>0)).to_numpy(float)-
                     ((b.predicted_return>0)==(b.actual>0)).to_numpy(float),ids))


def main():
    check_frozen();assert read(OUT/'scoring_manifest.json').get('finished_utc')
    seed=pd.read_csv(OUT/'all_seed_predictions.csv');assert len(seed)==1269
    ensemble=seed.groupby(['schedule','date'],sort=False).agg(predicted_return=('predicted_return','mean'),
        reassigned_prediction=('reassigned_prediction','mean'),actual=('actual','first'),cutoff=('cutoff','first'),
        training_mean=('training_mean','first'),training_sd=('training_sd','first')).reset_index()
    assert len(ensemble)==423
    summary=[];years=[];seeds=[]
    for schedule,g in ensemble.groupby('schedule',sort=False):
        summary.append(dict(schedule=schedule,**metric(g)))
        for year,h in g.groupby(g.date.str[:4]):years.append(dict(schedule=schedule,year=int(year),**metric(h)))
    for (schedule,number),g in seed.groupby(['schedule','seed'],sort=False):
        seeds.append(dict(schedule=schedule,seed=int(number),**metric(g)))
    metrics=pd.DataFrame(summary);yearly=pd.DataFrame(years);individual=pd.DataFrame(seeds)
    references={name:g for name,g in ensemble.groupby('schedule')};baselines=[]
    constants=pd.read_csv(OUT/'validation_rows.csv')
    for name in ['training_mean','zero_return']:
        g=constants.copy();g['predicted_return']=g[name];g['reassigned_prediction']=g[name]
        references[name]=g;baselines.append(dict(reference=name,**metric(g)))
    primary=[primary_pair(references[r['candidate']],references[r['reference']],**r) for r in cfg()['primary_comparisons']]
    for r,p in zip(primary,statistics.holm([r['mse']['p'] for r in primary])):r['mse']['holm_adjusted_p']=p
    by_seed=individual.pivot(index='seed',columns='schedule',values='mse')
    by_year=yearly.pivot(index='year',columns='schedule',values='mse')
    by_schedule=metrics.set_index('schedule');new=by_schedule.loc['decay40'];s=individual[individual.schedule.eq('decay40')]
    counts=dict(seeds_beat_mean=int(s.mse_skill_vs_training_mean.gt(0).sum()),
        paired_seeds_beat_constant40=int(by_seed.decay40.lt(by_seed.constant40).sum()),
        paired_seeds_beat_archived20=int(by_seed.decay40.lt(by_seed.archived20).sum()),
        years_beat_constant40=int(by_year.decay40.lt(by_year.constant40).sum()),
        years_beat_archived20=int(by_year.decay40.lt(by_year.archived20).sum()))
    flags=dict(ensemble_beats_constant40=bool(new.mse<by_schedule.loc['constant40','mse']),
        ensemble_beats_archived20=bool(new.mse<by_schedule.loc['archived20','mse']),ensemble_beats_mean=bool(new.mse_skill_vs_training_mean>0),
        at_least_two_seeds_beat_mean=counts['seeds_beat_mean']>=2,
        at_least_two_paired_seeds_beat_constant40=counts['paired_seeds_beat_constant40']>=2,
        at_least_two_paired_seeds_beat_archived20=counts['paired_seeds_beat_archived20']>=2,
        at_least_two_years_beat_constant40=counts['years_beat_constant40']>=2,
        at_least_two_years_beat_archived20=counts['years_beat_archived20']>=2)
    assessment=dict(candidate='decay40',flags=flags,descriptive_screen_pass=all(flags.values()),counts=counts,
        mse_skill_vs_constant40=1-float(new.mse/by_schedule.loc['constant40','mse']),
        mse_skill_vs_archived20=1-float(new.mse/by_schedule.loc['archived20','mse']),
        candidate_significant_improvements=sum(r['candidate']=='decay40' and r['mse']['difference']<0 and r['mse']['holm_adjusted_p']<.05 for r in primary),
        four_primary_comparisons=4,historical_reuse=True,no_selection=True,scored_utc=pd.Timestamp.now(tz='UTC').isoformat())
    for name,frame in [('ensemble_predictions',ensemble),('ensemble_metrics',metrics),('yearly_metrics',yearly),('seed_metrics',individual),
                       ('baseline_metrics',pd.DataFrame(baselines))]:frame.to_csv(OUT/(name+'.csv'),index=False)
    by_seed.to_csv(OUT/'matched_seed_mse.csv');by_year.to_csv(OUT/'matched_year_mse.csv')
    save(OUT/'primary_comparisons.json',primary);save(OUT/'assessment.json',assessment)
    # Prespecified fixed-state training comparisons; no training result controls fitting or scoring.
    draws=pd.concat([pd.read_csv(OUT/'prefix_training_loss_draws.csv'),pd.read_csv(OUT/'continuation_training_loss_draws.csv')],ignore_index=True)
    assert len(draws)==243
    rows=[]
    for (schedule,cutoff,number,mode),g in draws.groupby(['schedule','cutoff','seed','mode'],sort=False):
        r=dict(schedule=schedule,cutoff=cutoff,seed=int(number),mode=mode,draws=len(g))
        for key in ['return_mse','auxiliary_mse','joint_mse']:
            r[key]=float(g[key].mean());r[key+'_draw_sd']=float(g[key].std(ddof=1)) if len(g)>1 else 0.
        rows.append(r)
    losses=pd.DataFrame(rows);losses.to_csv(OUT/'training_loss_summary.csv',index=False)
    pairs=[]
    for left,right in [('constant40','prefix20'),('decay40','prefix20'),('decay40','constant40')]:
        a=losses[losses.schedule.eq(left)].set_index(['cutoff','seed','mode'])
        b=losses[losses.schedule.eq(right)].set_index(['cutoff','seed','mode'])
        for index,r in a.iterrows():
            s=b.loc[index];row=dict(candidate=left,reference=right,cutoff=index[0],seed=int(index[1]),mode=index[2])
            for key in ['return_mse','auxiliary_mse','joint_mse']:
                row[key+'_candidate']=float(r[key]);row[key+'_reference']=float(s[key]);row[key+'_difference']=float(r[key]-s[key])
            pairs.append(row)
    pairs=pd.DataFrame(pairs);pairs.to_csv(OUT/'paired_training_comparisons.csv',index=False)
    aggregates=[]
    for (candidate,reference,mode),g in pairs.groupby(['candidate','reference','mode']):
        r=dict(candidate=candidate,reference=reference,mode=mode,pairs=len(g))
        for key in ['return_mse','auxiliary_mse','joint_mse']:
            r[key+'_relative_change']=float(g[key+'_candidate'].mean()/g[key+'_reference'].mean()-1)
            r[key+'_lower_count']=int(g[key+'_difference'].lt(-1e-7).sum())
        aggregates.append(r)
    pd.DataFrame(aggregates).to_csv(OUT/'training_comparison_summary.csv',index=False)
    print(metrics[['schedule','accuracy','rmse','mse_skill_vs_training_mean']].to_string(index=False))
    print(json.dumps(assessment,indent=2))


if __name__=='__main__':main()

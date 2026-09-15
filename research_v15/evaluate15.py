from common15 import *
import evaluate6 as statistics

def weekly_loss(g,name):
    if name=='direction_error':return g.direction_up.to_numpy(bool)!=g.actual_up.to_numpy(bool)
    b,l=probability_losses(g.probability,g.actual_up);return b if name=='brier' else l

def compute():
    ensemble=pd.read_csv(OUT/'all_ensemble_predictions.csv');seed=pd.read_csv(OUT/'all_seed_predictions.csv')
    rows=[];years=[];seeds=[];reliability=[]
    for method,g in ensemble.groupby('method'):
        assert len(g)==141 and g.date.is_unique;rows.append(dict(method=method,**metric(g)))
        for year,h in g.groupby(g.date.str[:4]):years.append(dict(method=method,year=int(year),**metric(h)))
        if g.probability.notna().all():
            bins=np.minimum((g.probability.to_numpy()*5).astype(int),4)
            for i in range(5):
                h=g[bins==i];reliability.append(dict(method=method,bin_index=i,lower=i/5,upper=(i+1)/5,n=len(h),
                    mean_probability=float(h.probability.mean()) if len(h) else None,observed_frequency=float(h.actual_up.mean()) if len(h) else None))
    for (method,number),g in seed.groupby(['method','seed']):seeds.append(dict(method=method,seed=int(number),**metric(g)))
    table=pd.DataFrame(rows);year_table=pd.DataFrame(years);seed_table=pd.DataFrame(seeds);pairs=[];ids=statistics.bootstrap_indices(141)
    for c in cfg()['primary_comparisons']:
        a=ensemble[ensemble.method.eq(c['candidate'])].sort_values('date');b=ensemble[ensemble.method.eq(c['reference'])].sort_values('date')
        assert a.date.tolist()==b.date.tolist();np.testing.assert_array_equal(a.actual_up,b.actual_up)
        delta=weekly_loss(a,c['metric']).astype(float)-weekly_loss(b,c['metric']).astype(float)
        pairs.append(dict(**c,**statistics.difference(delta,ids)))
    for r,p in zip(pairs,statistics.holm([r['p'] for r in pairs])):r['holm_adjusted_p']=p
    t=table.set_index('method');s=seed_table.pivot(index='seed',columns='method',values='accuracy')
    sy=year_table.pivot(index='year',columns='method',values='accuracy');by=year_table.pivot(index='year',columns='method',values='brier')
    assessments=[];changes=[]
    for family in cfg()['families']:
        name=family['name'];source=family['source']
        counts=dict(seeds_beating_original_accuracy=int((s[name]>s.archived20).sum()),
            seeds_beating_frequency_brier=int((seed_table.loc[seed_table.method.eq(name),'brier']<t.loc['training_frequency','brier']).sum()),
            years_beating_original_accuracy=int((sy[name]>sy.archived20).sum()),years_beating_frequency_brier=int((by[name]<by.training_frequency).sum()))
        flags=dict(ensemble_accuracy_beats_original=bool(t.loc[name,'accuracy']>t.loc['archived20','accuracy']),
            ensemble_accuracy_beats_frequency=bool(t.loc[name,'accuracy']>t.loc['training_frequency','accuracy']),
            ensemble_brier_beats_frequency=bool(t.loc[name,'brier']<t.loc['training_frequency','brier']),
            ensemble_log_loss_beats_frequency=bool(t.loc[name,'log_loss']<t.loc['training_frequency','log_loss']),
            at_least_two_seeds_beat_original_accuracy=counts['seeds_beating_original_accuracy']>=2,
            at_least_two_seeds_beat_frequency_brier=counts['seeds_beating_frequency_brier']>=2,
            at_least_two_years_beat_original_accuracy=counts['years_beating_original_accuracy']>=2,
            at_least_two_years_beat_frequency_brier=counts['years_beating_frequency_brier']>=2)
        assessments.append(dict(method=name,source=source,counts=counts,flags=flags,descriptive_screen_pass=all(flags.values()),
            accuracy_difference_vs_original=float(t.loc[name,'accuracy']-t.loc['archived20','accuracy']),
            brier_skill_vs_training_frequency=float(1-t.loc[name,'brier']/t.loc['training_frequency','brier']),
            log_loss_skill_vs_training_frequency=float(1-t.loc[name,'log_loss']/t.loc['training_frequency','log_loss']),
            significant_improvements=sum(r['candidate']==name and r['difference']<0 and r['holm_adjusted_p']<.05 for r in pairs),
            independent_confirmation=False,strategy_promotion=False))
        a=ensemble[ensemble.method.eq(name)].set_index('date');b=ensemble[ensemble.method.eq(source)].set_index('date')
        for year in ['all','2018','2019','2020']:
            g=a if year=='all' else a[a.index.str.startswith(year)];h=b.loc[g.index]
            correct=g.direction_up.eq(g.actual_up);old=h.direction_up.eq(h.actual_up)
            changes.append(dict(method=name,reference=source,year=year,n=len(g),changed=int(g.direction_up.ne(h.direction_up).sum()),
                correct_to_wrong=int((old&~correct).sum()),wrong_to_correct=int((~old&correct).sum())))
    return dict(ensemble_metrics=table,yearly_metrics=year_table,seed_metrics=seed_table,reliability_bins=pd.DataFrame(reliability),
        direction_changes=pd.DataFrame(changes)),pairs,assessments

def main():
    check_frozen();check_artifacts('scoring');path=OUT/'evaluation_manifest.json';assert not path.exists(),'Preserve evaluation'
    run=manifest('evaluation');frames,pairs,assessments=compute();files=[]
    for name,frame in frames.items():p=OUT/f'{name}.csv';frame.to_csv(p,index=False);files.append(p)
    save(OUT/'primary_comparisons.json',pairs);save(OUT/'assessments.json',assessments)
    files += [OUT/'primary_comparisons.json',OUT/'assessments.json']
    run.update(finished_utc=now(),primary_comparisons=7,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files});save(path,run)
    print(frames['ensemble_metrics'][['method','accuracy','balanced_accuracy','auroc','brier','log_loss']].to_string(index=False))
    print(json.dumps(assessments),flush=True)

if __name__=='__main__':main()

from common16 import *
import evaluate6 as statistics


def windows():return cfg()['windows']+[dict(name='pooled_2015_2020',start='2015-01-01',end='2020-12-31',n=261)]
def select(frame,window):return frame[frame.date.ge(window['start'])&frame.date.le(window['end'])]
def loss(frame,name):
    if name=='direction_error':return frame.direction_up.ne(frame.actual_up).to_numpy(float)
    b,l=probability_losses(frame.probability,frame.actual_up);return b if name=='brier' else l


def compute():
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv');models=pd.read_csv(OUT/'model_predictions.csv')
    rows=[];years=[];seeds=[];bins=[];changes=[]
    for w in windows():
        for method,g in select(ensemble,w).groupby('method'):
            assert len(g)==w['n'] and g.date.is_unique;rows.append(dict(window=w['name'],method=method,**metric(g)))
            if w['name']!='pooled_2015_2020' and g.probability.notna().all():
                which=np.minimum((g.probability.to_numpy()*5).astype(int),4)
                for i in range(5):
                    h=g[which==i];bins.append(dict(window=w['name'],method=method,bin_index=i,lower=i/5,upper=(i+1)/5,n=len(h),
                        mean_probability=float(h.probability.mean()) if len(h) else None,observed_frequency=float(h.actual_up.mean()) if len(h) else None))
        for (method,seed),g in select(models[models.method.isin(['native_mse','learned_probe'])],w).groupby(['method','seed']):
            seeds.append(dict(window=w['name'],method=method,seed=int(seed),**metric(g)))
    for (method,year),g in ensemble.groupby(['method','year']):years.append(dict(method=method,year=int(year),**metric(g)))
    tables=dict(ensemble_metrics=pd.DataFrame(rows),yearly_metrics=pd.DataFrame(years),seed_metrics=pd.DataFrame(seeds),reliability_bins=pd.DataFrame(bins))
    pairs=[];assessments=[]
    for w in cfg()['windows']:
        e=select(ensemble,w);ids=statistics.bootstrap_indices(w['n'])
        for comparison in cfg()['primary_comparisons_per_window']:
            a=e[e.method.eq(comparison['candidate'])].sort_values('date');b=e[e.method.eq(comparison['reference'])].sort_values('date')
            assert a.date.tolist()==b.date.tolist();np.testing.assert_array_equal(a.actual_up,b.actual_up)
            pairs.append(dict(window=w['name'],**comparison,**statistics.difference(loss(a,comparison['metric'])-loss(b,comparison['metric']),ids)))
        t=tables['ensemble_metrics'][tables['ensemble_metrics'].window.eq(w['name'])].set_index('method')
        yr=tables['yearly_metrics'];yr=yr[yr.year.between(int(w['start'][:4]),int(w['end'][:4]))]
        acc=yr.pivot(index='year',columns='method',values='accuracy');brier=yr.pivot(index='year',columns='method',values='brier')
        for method in ['learned_probe','raw25_probe']:
            counts=dict(years_beating_native_accuracy=int((acc[method]>acc.native_mse).sum()),years_beating_frequency_brier=int((brier[method]<brier.training_frequency).sum()))
            flags=dict(accuracy_beats_native=bool(t.loc[method,'accuracy']>t.loc['native_mse','accuracy']),
                accuracy_beats_frequency=bool(t.loc[method,'accuracy']>t.loc['training_frequency','accuracy']),
                brier_beats_frequency=bool(t.loc[method,'brier']<t.loc['training_frequency','brier']),
                log_loss_beats_frequency=bool(t.loc[method,'log_loss']<t.loc['training_frequency','log_loss']),
                two_years_beat_native_accuracy=counts['years_beating_native_accuracy']>=2,two_years_beat_frequency_brier=counts['years_beating_frequency_brier']>=2)
            assessments.append(dict(window=w['name'],method=method,counts=counts,flags=flags,window_descriptive_pass=all(flags.values()),
                brier_skill_vs_frequency=float(1-t.loc[method,'brier']/t.loc['training_frequency','brier']),
                accuracy_difference_vs_native=float(t.loc[method,'accuracy']-t.loc['native_mse','accuracy']),independent_confirmation=False,strategy_promotion=False))
            a=e[e.method.eq(method)].set_index('date');b=e[e.method.eq('native_mse')].set_index('date').loc[a.index]
            new=a.direction_up.eq(a.actual_up);old=b.direction_up.eq(b.actual_up)
            changes.append(dict(window=w['name'],method=method,reference='native_mse',changed=int(a.direction_up.ne(b.direction_up).sum()),
                correct_to_wrong=int((old&~new).sum()),wrong_to_correct=int((~old&new).sum())))
    adjusted=statistics.holm([p['p'] for p in pairs])
    for p,value in zip(pairs,adjusted):p['holm_adjusted_p']=value
    for a in assessments:
        a['cross_period_descriptive_pass']=all(r['window_descriptive_pass'] for r in assessments if r['method']==a['method'])
        a['significant_improvements']=sum(p['window']==a['window'] and p['candidate']==a['method'] and p['difference']<0 and p['holm_adjusted_p']<.05 for p in pairs)
    tables['direction_changes']=pd.DataFrame(changes)
    return tables,pairs,assessments


def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists(),'Preserve evaluation'
    run=manifest('evaluation');tables,pairs,assessments=compute();files=[]
    for name,table in tables.items():p=OUT/f'{name}.csv';table.to_csv(p,index=False);files.append(p)
    save(OUT/'primary_comparisons.json',pairs);save(OUT/'assessments.json',assessments)
    files += [OUT/'primary_comparisons.json',OUT/'assessments.json']
    finish(run,files,primary_contrasts=12,methods=5,years=6,windows=2,pooled_descriptive_only=True)
    print(tables['ensemble_metrics'][['window','method','accuracy','auroc','brier','log_loss']].to_string(index=False),flush=True)
    print(json.dumps(assessments),flush=True)


if __name__=='__main__':main()

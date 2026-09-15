from common19 import *
import evaluate6 as statistics

def windows():return cfg()['windows']+[dict(name='pooled_2015_2020',start='2015-01-01',end='2020-12-31',n=261)]
def select(f,w):return f[f.date.ge(w['start'])&f.date.le(w['end'])]
def loss(f,name):
    if name=='direction_error':return f.direction_up.ne(f.actual_up).to_numpy(float)
    b,l=probability_losses(f.probability,f.actual_up);return b if name=='brier' else l

def compute():
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv',float_precision='round_trip');models=pd.read_csv(OUT/'model_predictions.csv',float_precision='round_trip')
    rows=[];years=[];seeds=[];bins=[];pairs=[];assessments=[];changes=[]
    for w in windows():
        for method,g in select(ensemble,w).groupby('method'):
            assert len(g)==w['n'] and g.date.is_unique;rows.append(dict(window=w['name'],method=method,**metric(g)))
            if w['name']!='pooled_2015_2020' and g.probability.notna().all():
                which=np.minimum((g.probability.to_numpy()*5).astype(int),4)
                for i in range(5):
                    s=g[which==i];bins.append(dict(window=w['name'],method=method,bin_index=i,lower=i/5,upper=(i+1)/5,n=len(s),
                        mean_probability=float(s.probability.mean()) if len(s) else None,observed_frequency=float(s.actual_up.mean()) if len(s) else None))
        for (method,seed),g in select(models[models.method.isin(['native_mse','learned_probe','learned_clip','learned_market','learned_vol_interaction'])],w).groupby(['method','seed']):
            seeds.append(dict(window=w['name'],method=method,seed=int(seed),**metric(g)))
    for (method,year),g in ensemble.groupby(['method','year']):years.append(dict(method=method,year=int(year),**metric(g)))
    tables=dict(ensemble_metrics=pd.DataFrame(rows),yearly_metrics=pd.DataFrame(years),seed_metrics=pd.DataFrame(seeds),reliability_bins=pd.DataFrame(bins))
    for w in cfg()['windows']:
        e=select(ensemble,w);ids=statistics.bootstrap_indices(w['n'])
        for comp in cfg()['primary_comparisons_per_window']:
            a=e[e.method.eq(comp['candidate'])].sort_values('date');b=e[e.method.eq(comp['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist()
            np.testing.assert_array_equal(a.actual_up,b.actual_up);pairs.append(dict(window=w['name'],**comp,**statistics.difference(loss(a,comp['metric'])-loss(b,comp['metric']),ids)))
        t=tables['ensemble_metrics'][tables['ensemble_metrics'].window.eq(w['name'])].set_index('method');yr=tables['yearly_metrics']
        yr=yr[yr.year.between(int(w['start'][:4]),int(w['end'][:4]))];acc=yr.pivot(index='year',columns='method',values='accuracy');brier=yr.pivot(index='year',columns='method',values='brier')
        for candidate,parent in CANDIDATES.items():
            anchor=ANCHORS[candidate];c=t.loc[candidate];counts=dict(years_beating_native_accuracy=int((acc[candidate]>acc.native_mse).sum()),years_beating_frequency_brier=int((brier[candidate]<brier.training_frequency).sum()))
            flags=dict(accuracy_beats_parent=bool(c.accuracy>t.loc[parent,'accuracy']),accuracy_beats_round17=bool(c.accuracy>t.loc[anchor,'accuracy']),
                accuracy_beats_native=bool(c.accuracy>t.loc['native_mse','accuracy']),accuracy_beats_frequency=bool(c.accuracy>t.loc['training_frequency','accuracy']),
                brier_beats_parent=bool(c.brier<t.loc[parent,'brier']),brier_beats_round17=bool(c.brier<t.loc[anchor,'brier']),brier_beats_frequency=bool(c.brier<t.loc['training_frequency','brier']),
                log_loss_beats_frequency=bool(c.log_loss<t.loc['training_frequency','log_loss']),two_years_beat_native_accuracy=counts['years_beating_native_accuracy']>=2,
                two_years_beat_frequency_brier=counts['years_beating_frequency_brier']>=2)
            assessments.append(dict(window=w['name'],method=candidate,parent=parent,round17_anchor=anchor,counts=counts,flags=flags,window_descriptive_pass=all(flags.values()),
                brier_skill_vs_frequency=float(1-c.brier/t.loc['training_frequency','brier']),independent_confirmation=False,strategy_promotion=False))
            for reference in [parent,anchor,'native_mse','training_frequency']:
                a=e[e.method.eq(candidate)].set_index('date');b=e[e.method.eq(reference)].set_index('date').loc[a.index];new=a.direction_up.eq(a.actual_up);old=b.direction_up.eq(b.actual_up)
                changes.append(dict(window=w['name'],method=candidate,reference=reference,changed=int(a.direction_up.ne(b.direction_up).sum()),
                    correct_to_wrong=int((old&~new).sum()),wrong_to_correct=int((~old&new).sum())))
    for p,q in zip(pairs,statistics.holm([p['p'] for p in pairs])):p['holm_adjusted_p']=q
    for a in assessments:
        a['cross_period_descriptive_pass']=all(r['window_descriptive_pass'] for r in assessments if r['method']==a['method'])
        a['significant_improvements']=sum(p['window']==a['window'] and p['candidate']==a['method'] and p['difference']<0 and p['holm_adjusted_p']<.05 for p in pairs)
    states=pd.read_csv(OUT/'validation_states.csv');enriched=ensemble.merge(states,on=['cutoff','row_index','date'],validate='many_to_one');diagnostics=[]
    for w in cfg()['windows']:
        e=select(enriched,w)
        for partition,labels in PARTITIONS.items():
            for label in labels:
                s=e[e[partition].eq(label)];f=s[s.method.eq('training_frequency')]
                for method in METHODS:
                    g=s[s.method.eq(method)];diagnostics.append(dict(window=w['name'],partition=partition,state=label,method=method,sparse=len(g)<20,sample_share=len(g)/w['n'],
                        brier_difference_vs_frequency=float(loss(g,'brier').mean()-loss(f,'brier').mean()) if len(g) and g.probability.notna().all() else None,**metric(g)))
    tables.update(direction_changes=pd.DataFrame(changes),state_metrics=pd.DataFrame(diagnostics));return tables,pairs,assessments

def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists(),'Preserve evaluation'
    run=manifest('evaluation');tables,pairs,assessments=compute();files=[]
    for name,table in tables.items():path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    save(OUT/'primary_comparisons.json',pairs);save(OUT/'assessments.json',assessments);files += [OUT/'primary_comparisons.json',OUT/'assessments.json']
    finish(run,files,primary_contrasts=16,methods=12,years=6,windows=2,pooled_descriptive_only=True,state_metric_rows=len(tables['state_metrics']))
    print(tables['ensemble_metrics'][['window','method','accuracy','auroc','brier','log_loss']].to_string(index=False),flush=True);print(json.dumps(assessments),flush=True)

if __name__=='__main__':main()

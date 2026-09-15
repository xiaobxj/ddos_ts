from common21 import *
import evaluate6 as statistics

def windows():return cfg()['windows']+[dict(name='pooled_2015_2020',start='2015-01-01',end='2020-12-31',n=261)]
def select(f,w):return f[f.date.ge(w['start'])&f.date.le(w['end'])]
def loss(f,name):
    if name=='direction_error':return f.direction_up.ne(f.actual_up).to_numpy(float)
    b,l=probability_losses(f.probability,f.actual_up);return b if name=='brier' else l

def assessment_flags(t,yr,candidate):
    parent=CANDIDATES[candidate];old=OLD_INTERACTIONS[candidate];c=t.loc[candidate]
    acc=yr.pivot(index='year',columns='method',values='accuracy');b=yr.pivot(index='year',columns='method',values='brier')
    flags={f'accuracy_beats_{ref}':bool(c.accuracy>t.loc[ref,'accuracy']) for ref in [parent,old,'native_mse','training_frequency']}
    flags.update({f'brier_beats_{ref}':bool(c.brier<t.loc[ref,'brier']) for ref in [parent,old,'training_frequency','hmm_only']})
    flags.update(log_loss_beats_frequency=bool(c.log_loss<t.loc['training_frequency','log_loss']),two_years_beat_native_accuracy=bool((acc[candidate]>acc.native_mse).sum()>=2),
        two_years_beat_frequency_brier=bool((b[candidate]<b.training_frequency).sum()>=2));assert len(flags)==11;return flags

def compute():
    ensemble=pd.read_csv(OUT/'ensemble_predictions.csv',float_precision='round_trip');models=pd.read_csv(OUT/'model_predictions.csv',float_precision='round_trip');rows=[];years=[];seeds=[];bins=[];pairs=[];assessments=[];changes=[]
    for w in windows():
        for method,g in select(ensemble,w).groupby('method'):
            assert len(g)==w['n'] and g.date.is_unique;rows.append(dict(window=w['name'],method=method,**metric(g)))
            if w['name']!='pooled_2015_2020' and g.probability.notna().all():
                which=np.minimum((g.probability.to_numpy()*5).astype(int),4)
                for i in range(5):
                    s=g[which==i];bins.append(dict(window=w['name'],method=method,bin_index=i,lower=i/5,upper=(i+1)/5,n=len(s),mean_probability=float(s.probability.mean()) if len(s) else None,observed_frequency=float(s.actual_up.mean()) if len(s) else None))
        for (method,seed),g in select(models[models.method.isin(SEED_METHODS)],w).groupby(['method','seed']):seeds.append(dict(window=w['name'],method=method,seed=int(seed),**metric(g)))
    for (method,year),g in ensemble.groupby(['method','year']):years.append(dict(method=method,year=int(year),**metric(g)))
    tables=dict(ensemble_metrics=pd.DataFrame(rows),yearly_metrics=pd.DataFrame(years),seed_metrics=pd.DataFrame(seeds),reliability_bins=pd.DataFrame(bins))
    for w in cfg()['windows']:
        e=select(ensemble,w);ids=statistics.bootstrap_indices(w['n'])
        for comp in cfg()['primary_comparisons_per_window']:
            a=e[e.method.eq(comp['candidate'])].sort_values('date');b=e[e.method.eq(comp['reference'])].sort_values('date');np.testing.assert_array_equal(a.date,b.date)
            pairs.append(dict(window=w['name'],**comp,**statistics.difference(loss(a,comp['metric'])-loss(b,comp['metric']),ids)))
        t=tables['ensemble_metrics'][tables['ensemble_metrics'].window.eq(w['name'])].set_index('method');yr=tables['yearly_metrics'];yr=yr[yr.year.between(int(w['start'][:4]),int(w['end'][:4]))]
        for candidate,parent in CANDIDATES.items():
            flags=assessment_flags(t,yr,candidate);assessments.append(dict(window=w['name'],method=candidate,parent=parent,old_interaction=OLD_INTERACTIONS[candidate],flags=flags,
                window_descriptive_pass=all(flags.values()),independent_confirmation=False,strategy_promotion=False,interaction_hypothesis_retained=True))
            for reference in [parent,OLD_INTERACTIONS[candidate],'native_mse','training_frequency','hmm_only']:
                a=e[e.method.eq(candidate)].sort_values('date');b=e[e.method.eq(reference)].sort_values('date');aa=a.direction_up.to_numpy()==a.actual_up.to_numpy();bb=b.direction_up.to_numpy()==b.actual_up.to_numpy()
                changes.append(dict(window=w['name'],method=candidate,reference=reference,changed=int(np.count_nonzero(aa!=bb)),correct_to_wrong=int(np.count_nonzero(bb&~aa)),wrong_to_correct=int(np.count_nonzero(~bb&aa))))
    for pair,q in zip(pairs,statistics.holm([r['p'] for r in pairs])):pair['holm_adjusted_p']=q
    for a in assessments:
        a['cross_period_descriptive_pass']=all(r['window_descriptive_pass'] for r in assessments if r['method']==a['method'])
        a['significant_improvements']=sum(r['window']==a['window'] and r['candidate']==a['method'] and r['difference']<0 and r['holm_adjusted_p']<.05 for r in pairs)
    states=pd.read_csv(OUT/'validation_states.csv');hmm=pd.read_csv(OUT/'hmm_signal_states.csv',float_precision='round_trip');enriched=ensemble.merge(states,on=['cutoff','row_index','date'],validate='many_to_one').merge(hmm,on=['cutoff','row_index','date'],validate='many_to_one')
    oldrows=[];hmmrows=[]
    for w in cfg()['windows']:
        e=select(enriched,w)
        for partition,labels in dict(PARTITIONS,hmm_bucket=['low_probability','uncertain','high_probability']).items():
            for label in labels:
                s=e[e[partition].eq(label)];f=s[s.method.eq('training_frequency')]
                for method in METHODS:
                    g=s[s.method.eq(method)];r=dict(window=w['name'],partition=partition,state=label,method=method,sparse=len(g)<20,sample_share=len(g)/w['n'],
                        brier_difference_vs_frequency=float(loss(g,'brier').mean()-loss(f,'brier').mean()) if len(g) and g.probability.notna().all() else None,**metric(g))
                    (hmmrows if partition=='hmm_bucket' else oldrows).append(r)
    tables.update(state_metrics=pd.DataFrame(oldrows),hmm_state_metrics=pd.DataFrame(hmmrows),direction_changes=pd.DataFrame(changes));return tables,pairs,assessments

def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs,assessments=compute();files=[]
    for name,table in tables.items():path=OUT/f'{name}.csv';table.to_csv(path,index=False);files.append(path)
    for name,values in [('primary_comparisons',pairs),('assessments',assessments)]:path=OUT/f'{name}.json';save(path,values);files.append(path)
    assert len(pairs)==26 and len(tables['state_metrics'])==480 and len(tables['hmm_state_metrics'])==120 and len(tables['reliability_bins'])==190
    finish(run,files,primary_contrasts=26,methods=20,metric_groups=852,reliability_bins=190,state_metric_rows=480,hmm_state_metric_rows=120,pooled_descriptive_only=True)
    print(tables['ensemble_metrics'][tables['ensemble_metrics'].method.isin(list(CANDIDATES)+list(CANDIDATES.values())+list(OLD_INTERACTIONS.values())+['hmm_only','training_frequency','native_mse'])][['window','method','accuracy','brier','log_loss']].to_string(index=False),flush=True)
    print(json.dumps(assessments),flush=True)

if __name__=='__main__':main()

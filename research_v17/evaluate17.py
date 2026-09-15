from common17 import *
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
        for (method,seed),g in select(models[models.method.isin(['native_mse','learned_probe','learned_clip'])],w).groupby(['method','seed']):
            seeds.append(dict(window=w['name'],method=method,seed=int(seed),**metric(g)))
    for (method,year),g in ensemble.groupby(['method','year']):years.append(dict(method=method,year=int(year),**metric(g)))
    tables=dict(ensemble_metrics=pd.DataFrame(rows),yearly_metrics=pd.DataFrame(years),seed_metrics=pd.DataFrame(seeds),reliability_bins=pd.DataFrame(bins))
    for w in cfg()['windows']:
        e=select(ensemble,w);ids=statistics.bootstrap_indices(w['n'])
        for comparison in cfg()['primary_comparisons_per_window']:
            a=e[e.method.eq(comparison['candidate'])].sort_values('date');b=e[e.method.eq(comparison['reference'])].sort_values('date')
            assert a.date.tolist()==b.date.tolist();np.testing.assert_array_equal(a.actual_up,b.actual_up)
            pairs.append(dict(window=w['name'],**comparison,**statistics.difference(loss(a,comparison['metric'])-loss(b,comparison['metric']),ids)))
        t=tables['ensemble_metrics'][tables['ensemble_metrics'].window.eq(w['name'])].set_index('method')
        yr=tables['yearly_metrics'];yr=yr[yr.year.between(int(w['start'][:4]),int(w['end'][:4]))]
        acc=yr.pivot(index='year',columns='method',values='accuracy');brier=yr.pivot(index='year',columns='method',values='brier')
        for c,r in CANDIDATES.items():
            counts=dict(years_beating_native_accuracy=int((acc[c]>acc.native_mse).sum()),years_beating_frequency_brier=int((brier[c]<brier.training_frequency).sum()))
            flags=dict(accuracy_beats_unclipped=bool(t.loc[c,'accuracy']>t.loc[r,'accuracy']),accuracy_beats_native=bool(t.loc[c,'accuracy']>t.loc['native_mse','accuracy']),
                accuracy_beats_frequency=bool(t.loc[c,'accuracy']>t.loc['training_frequency','accuracy']),brier_beats_unclipped=bool(t.loc[c,'brier']<t.loc[r,'brier']),
                brier_beats_frequency=bool(t.loc[c,'brier']<t.loc['training_frequency','brier']),log_loss_beats_frequency=bool(t.loc[c,'log_loss']<t.loc['training_frequency','log_loss']),
                two_years_beat_native_accuracy=counts['years_beating_native_accuracy']>=2,two_years_beat_frequency_brier=counts['years_beating_frequency_brier']>=2)
            assessments.append(dict(window=w['name'],method=c,counts=counts,flags=flags,window_descriptive_pass=all(flags.values()),
                brier_skill_vs_frequency=float(1-t.loc[c,'brier']/t.loc['training_frequency','brier']),independent_confirmation=False,strategy_promotion=False))
            for ref in [r,'native_mse']:
                a=e[e.method.eq(c)].set_index('date');b=e[e.method.eq(ref)].set_index('date').loc[a.index]
                new=a.direction_up.eq(a.actual_up);old=b.direction_up.eq(b.actual_up)
                changes.append(dict(window=w['name'],method=c,reference=ref,changed=int(a.direction_up.ne(b.direction_up).sum()),
                    correct_to_wrong=int((old&~new).sum()),wrong_to_correct=int((~old&new).sum())))
    for p,q in zip(pairs,statistics.holm([p['p'] for p in pairs])):p['holm_adjusted_p']=q
    for a in assessments:
        a['cross_period_descriptive_pass']=all(r['window_descriptive_pass'] for r in assessments if r['method']==a['method'])
        a['significant_improvements']=sum(p['window']==a['window'] and p['candidate']==a['method'] and p['difference']<0 and p['holm_adjusted_p']<.05 for p in pairs)
    states=pd.read_csv(OUT/'validation_states.csv');train_states=pd.read_csv(OUT/'training_states.csv');diagnostics=[];distributions=[];state_clips=[]
    enriched=ensemble.merge(states,on=['cutoff','row_index','date'],validate='many_to_one');minimum=cfg()['market_states']['minimum_descriptive_cell_n']
    clips=pd.read_csv(OUT/'clipping_rows.csv').merge(states,on=['cutoff','row_index','date'],validate='many_to_one')
    for w in cfg()['windows']:
        e=select(enriched,w);cl=select(clips,w)
        for partition,labels in PARTITIONS.items():
            for label in labels:
                s=e[e[partition].eq(label)];f=s[s.method.eq('training_frequency')];n=s[s.method.eq('native_mse')]
                for method in METHODS:
                    g=s[s.method.eq(method)];m=metric(g);total_errors=int(e[e.method.eq(method)].direction_up.ne(e[e.method.eq(method)].actual_up).sum())
                    diagnostics.append(dict(window=w['name'],partition=partition,state=label,method=method,sparse=len(g)<minimum,
                        sample_share=len(g)/w['n'],error_share=(len(g)-m['correct_directions'])/total_errors if total_errors else None,
                        brier_difference_vs_frequency=float(loss(g,'brier').mean()-loss(f,'brier').mean()) if len(g) and g.probability.notna().all() else None,
                        accuracy_difference_vs_native=float(m['accuracy']-metric(n)['accuracy']) if len(g) else None,**m))
                for method in CANDIDATES:
                    g=cl[cl[partition].eq(label)&cl.method.eq(method)]
                    state_clips.append(dict(window=w['name'],partition=partition,state=label,method=method,unique_dates=int(g.date.nunique()),head_rows=len(g),
                        mean_coordinate_clip_fraction=float(g.coordinate_clip_fraction.mean()) if len(g) else None,
                        head_row_clip_fraction=float(g.clipped_coordinates.gt(0).mean()) if len(g) else None))
    for fold in cfg()['folds']:
        tr=train_states[train_states.cutoff.eq(fold['cutoff'])];te=states[states.cutoff.eq(fold['cutoff'])]
        for partition,labels in PARTITIONS.items():
            for label in labels:
                a=int(tr[partition].eq(label).sum());b=int(te[partition].eq(label).sum())
                distributions.append(dict(cutoff=fold['cutoff'],year=int(fold['end'][:4]),partition=partition,state=label,
                    train_n=a,validation_n=b,training_fraction=a/len(tr),validation_fraction=b/len(te),fraction_difference=b/len(te)-a/len(tr)))
    tables.update(direction_changes=pd.DataFrame(changes),state_metrics=pd.DataFrame(diagnostics),state_distribution=pd.DataFrame(distributions),state_clipping=pd.DataFrame(state_clips))
    return tables,pairs,assessments

def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists(),'Preserve evaluation'
    run=manifest('evaluation');tables,pairs,assessments=compute();files=[]
    for name,table in tables.items():p=OUT/f'{name}.csv';table.to_csv(p,index=False);files.append(p)
    save(OUT/'primary_comparisons.json',pairs);save(OUT/'assessments.json',assessments);files += [OUT/'primary_comparisons.json',OUT/'assessments.json']
    finish(run,files,primary_contrasts=12,methods=7,years=6,windows=2,pooled_descriptive_only=True,state_metric_rows=len(tables['state_metrics']),state_selection=False)
    print(tables['ensemble_metrics'][['window','method','accuracy','auroc','brier','log_loss']].to_string(index=False),flush=True)
    print(json.dumps(assessments),flush=True)

if __name__=='__main__':main()

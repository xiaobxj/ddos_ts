from common26 import *
def select(f,w):return f[f.date.ge(w['start'])&f.date.le(w['end'])]
def loss(f,name):
    if name=='direction_error':return f.direction_up.ne(f.actual_up).to_numpy(float)
    b,l=previous.probability_losses(f.probability,f.actual_up);return b if name=='brier' else l
def compute(models,ensemble):
    metrics=[];years=[];seeds=[];bins=[];pairs=[];changed=[];changes=[]
    windows=cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end='2026-08-31',n=272)]
    for w in windows:
        for method,g in select(ensemble,w).groupby('method'):
            assert len(g)==w['n'] and g.date.is_unique;metrics.append(dict(window=w['name'],method=method,**metric(g)))
            if method!='native_mse' and w['name']!='pooled_2021_2026':
                idx=np.minimum((g.probability.to_numpy()*5).astype(int),4)
                for i in range(5):
                    s=g[idx==i];bins.append(dict(window=w['name'],method=method,bin_index=i,n=len(s),mean_probability=float(s.probability.mean()) if len(s) else None,observed_up=float(s.actual_up.mean()) if len(s) else None))
        for (method,seed),g in select(models[models.seed.ne(-1)],w).groupby(['method','seed']):seeds.append(dict(window=w['name'],method=method,seed=int(seed),**metric(g)))
    for (method,year),g in ensemble.groupby(['method','year']):years.append(dict(method=method,year=int(year),**metric(g)))
    seed_years=[dict(method=m,seed=int(s),year=int(y),**metric(g)) for (m,s,y),g in models[models.seed.ne(-1)].groupby(['method','seed','year'])]
    for w in cfg()['windows']:
        e=select(ensemble,w);ids=statistics.bootstrap_indices(w['n'])
        for comp in cfg()['primary_comparisons_per_window']:
            a=e[e.method.eq(comp['candidate'])].sort_values('date');b=e[e.method.eq(comp['reference'])].sort_values('date');np.testing.assert_array_equal(a.date,b.date)
            pairs.append(dict(window=w['name'],**comp,**statistics.difference(loss(a,comp['metric'])-loss(b,comp['metric']),ids)))
        for method,reference in [('learned_order_extension','learned_vol_interaction'),('learned_order_offset','learned_vol_interaction'),('learned_order_offset','learned_order_extension')]:
            a=e[e.method.eq(method)].sort_values('date').reset_index(drop=True);b=e[e.method.eq(reference)].sort_values('date').reset_index(drop=True);mask=a.direction_up.ne(b.direction_up);good=a.direction_up.eq(a.actual_up)
            changes.append(dict(window=w['name'],method=method,reference=reference,n=len(a),changed=int(mask.sum()),wrong_to_correct=int((mask&good).sum()),correct_to_wrong=int((mask&~good).sum())))
            for i in np.flatnonzero(mask):changed.append(dict(window=w['name'],method=method,reference=reference,date=a.date.iloc[i],actual_up=int(a.actual_up.iloc[i]),candidate_probability=float(a.probability.iloc[i]),reference_probability=float(b.probability.iloc[i]),candidate_correct=bool(good.iloc[i])))
    for row,p in zip(pairs,statistics.holm([p['p'] for p in pairs])):row['holm_adjusted_p']=p
    return dict(ensemble_metrics=pd.DataFrame(metrics),yearly_metrics=pd.DataFrame(years),seed_metrics=pd.DataFrame(seeds),seed_yearly_metrics=pd.DataFrame(seed_years),reliability_bins=pd.DataFrame(bins),direction_changes=pd.DataFrame(changes),changed_signal_details=pd.DataFrame(changed,columns=['window','method','reference','date','actual_up','candidate_probability','reference_probability','candidate_correct'])),pairs
def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs=compute(csv('model_predictions'),csv('ensemble_predictions'));files=[]
    for name,table in tables.items():p=OUT/f'{name}.csv';table.to_csv(p,index=False);files.append(p)
    save(OUT/'primary_comparisons.json',pairs);files.append(OUT/'primary_comparisons.json');assert len(pairs)==24
    finish(run,files,primary_contrasts=24,methods=6,historical_extension=True,independent_holdout=False)
    print(tables['ensemble_metrics'][['window','method','correct_directions','n','accuracy','brier','log_loss']].to_string(index=False),flush=True)
if __name__=='__main__':main()

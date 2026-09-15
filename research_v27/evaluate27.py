from common27 import *
def windows():return cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end='2026-08-31',n=272)]
def select(f,w):return f[f.date.ge(w['start'])&f.date.le(w['end'])]
def loss(g,name):
    if name=='direction_error':return g.direction_up.ne(g.actual_up).to_numpy(float)
    b,l=prior.previous.probability_losses(g.probability,g.actual_up);return b if name=='brier' else l
def seed_diagnostics(models,ensemble):
    weekly=[];summary=[]
    for (history,method,date),g in models[models.method.isin(LEARNED)].groupby(['history','method','date']):
        g=g.sort_values('seed');assert g.seed.tolist()==cfg()['seeds'];p=g.probability.to_numpy(float);q=float(p.mean());up=int((p>.5).sum());direction=q>.5;majority=up>=2
        row=dict(history=history,method=method,date=date,year=int(date[:4]),actual_up=int(g.actual_up.iloc[0]),up_seed_count=up,unanimous=up in [0,3],mean_probability=q,ensemble_up=direction,opposes_majority=bool(direction!=majority),seed_probability_variance=float(p.var()))
        row.update({f'probability_seed_{i+1}':float(p[i]) for i in range(3)});weekly.append(row)
    weekly=pd.DataFrame(weekly)
    for w in windows():
        for (history,method),g in select(weekly,w).groupby(['history','method']):
            p=g[[f'probability_seed_{i+1}' for i in range(3)]].to_numpy();y=g.actual_up.to_numpy(float);q=g.mean_probability.to_numpy();seed_acc=((p>.5)==y[:,None]).mean(0);ensemble_acc=((q>.5)==y).mean();mean_brier=float(((p-y[:,None])**2).mean());ensemble_brier=float(((q-y)**2).mean());variance=float(p.var(1).mean())
            assert abs(mean_brier-ensemble_brier-variance)<1e-14
            summary.append(dict(window=w['name'],history=history,method=method,n=len(g),seed_accuracy_min=float(seed_acc.min()),seed_accuracy_max=float(seed_acc.max()),mean_seed_accuracy=float(seed_acc.mean()),ensemble_accuracy=float(ensemble_acc),ensemble_below_all_seeds=bool(ensemble_acc<seed_acc.min()),disagreeing_weeks=int((~g.unanimous).sum()),opposes_majority_weeks=int(g.opposes_majority.sum()),opposes_majority_wrong=int((g.opposes_majority&g.ensemble_up.ne(g.actual_up)).sum()),mean_seed_brier=mean_brier,ensemble_brier=ensemble_brier,variance_reduction=variance))
    return weekly,pd.DataFrame(summary)
def compute(models,ensemble):
    metrics=[];seeds=[];bins=[];pairs=[];changes=[]
    for w in windows():
        for (history,method),g in select(ensemble,w).groupby(['history','method']):
            assert len(g)==w['n'] and g.date.is_unique;metrics.append(dict(window=w['name'],history=history,method=method,**metric(g)))
            if method!='native_mse' and w['name']!='pooled_2021_2026':
                idx=np.minimum((g.probability.to_numpy()*5).astype(int),4)
                for i in range(5):
                    s=g[idx==i];bins.append(dict(window=w['name'],history=history,method=method,bin_index=i,n=len(s),mean_probability=float(s.probability.mean()) if len(s) else None,observed_up=float(s.actual_up.mean()) if len(s) else None))
        for (history,method,seed),g in select(models[models.seed.ne(-1)],w).groupby(['history','method','seed']):seeds.append(dict(window=w['name'],history=history,method=method,seed=int(seed),**metric(g)))
    yearly=pd.DataFrame([dict(history=h,method=m,year=int(y),**metric(g)) for (h,m,y),g in ensemble.groupby(['history','method','year'])])
    seed_yearly=pd.DataFrame([dict(history=h,method=m,seed=int(s),year=int(y),**metric(g)) for (h,m,s,y),g in models[models.seed.ne(-1)].groupby(['history','method','seed','year'])])
    for w in cfg()['windows']:
        e=select(ensemble,w);ids=statistics.bootstrap_indices(w['n'])
        for comp in cfg()['primary_comparisons_per_window']:
            a=e[e.history.eq(comp['history'])&e.method.eq(comp['candidate'])].sort_values('date');b=e[e.history.eq(comp['reference_history'])&e.method.eq(comp['reference'])].sort_values('date');np.testing.assert_array_equal(a.date,b.date)
            pairs.append(dict(window=w['name'],**comp,**statistics.difference(loss(a,comp['metric'])-loss(b,comp['metric']),ids)))
        for history in NEW:
            for method in LEARNED:
                a=e[e.history.eq(history)&e.method.eq(method)].sort_values('date').reset_index(drop=True);b=e[e.history.eq('full')&e.method.eq(method)].sort_values('date').reset_index(drop=True);changed=a.direction_up.ne(b.direction_up);good=a.direction_up.eq(a.actual_up)
                changes.append(dict(window=w['name'],history=history,method=method,changed=int(changed.sum()),wrong_to_correct=int((changed&good).sum()),correct_to_wrong=int((changed&~good).sum())))
    for row,p in zip(pairs,statistics.holm([p['p'] for p in pairs])):row['holm_adjusted_p']=p
    weekly,diag=seed_diagnostics(models,ensemble)
    return dict(ensemble_metrics=pd.DataFrame(metrics),yearly_metrics=yearly,seed_metrics=pd.DataFrame(seeds),seed_yearly_metrics=seed_yearly,reliability_bins=pd.DataFrame(bins),direction_changes=pd.DataFrame(changes),seed_fusion_weekly=weekly,seed_fusion_diagnostics=diag),pairs
def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs=compute(csv('model_predictions'),csv('ensemble_predictions'));files=[]
    for name,table in tables.items():p=OUT/f'{name}.csv';table.to_csv(p,index=False);files.append(p)
    save(OUT/'primary_comparisons.json',pairs);files.append(OUT/'primary_comparisons.json');assert len(pairs)==32
    finish(run,files,primary_contrasts=32,histories=3,methods=6,seed_fusion_rules_changed=False,calibration_fits=0,historical_extension=True,independent_holdout=False)
    m=tables['ensemble_metrics'];print(m[m.method.isin(['learned_vol_interaction','native_mse','training_frequency'])][['window','history','method','correct_directions','n','accuracy','brier','log_loss']].to_string(index=False),flush=True)
if __name__=='__main__':main()

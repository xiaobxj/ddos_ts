from common39 import *
def compute(models,ensemble):
    metrics=[];sm=[];state=[];pairs=[];changes=[];weekly=[];ctx=csv('weekly_context')[['row_index','date','state']]
    for p in periods():
        e=ensemble[ensemble.date.between(p['start'],p['end'])];s=models[models.date.between(p['start'],p['end'])&models.seed.ne(-1)]
        for (h,m),g in e.groupby(['history','method']):metrics.append(dict(period=p['name'],history=h,method=m,**metric(g)))
        for (h,m,seed),g in s.groupby(['history','method','seed']):sm.append(dict(period=p['name'],history=h,method=m,seed=int(seed),**metric(g)))
        q=e[e.history.isin([ANNUAL]+NEW)&e.method.isin(LEARNED)].merge(ctx,on=['row_index','date'],validate='many_to_one')
        for h in [ANNUAL]+NEW:
            for m in LEARNED:
                for st in STATES:state.append(dict(period=p['name'],history=h,method=m,state=st,**metric(q[q.history.eq(h)&q.method.eq(m)&q.state.eq(st)])))
    for p in cfg()['windows']:
        e=ensemble[ensemble.date.between(p['start'],p['end'])];ids=statistics.bootstrap_indices(p['n'])
        for comp in cfg()['primary_comparisons_per_window']:
            a=e[e.history.eq(comp['history'])&e.method.eq(comp['candidate'])].sort_values('date');b=e[e.history.eq(comp['reference_history'])&e.method.eq(comp['reference'])].sort_values('date');np.testing.assert_array_equal(a.date,b.date)
            def loss(g):return g.direction_up.ne(g.actual_up).to_numpy(float) if comp['metric']=='direction_error' else (g.probability.to_numpy()-g.actual_up.to_numpy())**2
            pairs.append(dict(window=p['name'],**comp,**statistics.difference(loss(a)-loss(b),ids)))
    for r,p in zip(pairs,statistics.holm([r['p'] for r in pairs])):r['holm_adjusted_p']=p
    source=csv('seed_routing');routing=source[source.seed.eq(cfg()['seeds'][0])].drop(columns=['seed','offset','annual_probability','probability','actual_up']);ei=ensemble.set_index(['history','method','row_index'])
    for r in routing.itertuples():
        a=ei.loc[(r.history,r.method,r.row_index)];b=ei.loc[(ANNUAL,r.method,r.row_index)];good=a.direction_up==a.actual_up;old=b.direction_up==b.actual_up;case='regression' if old and not good else 'recovery' if good and not old else 'stable_correct' if old else 'stable_wrong';weekly.append(dict(**r._asdict(),year=int(r.date[:4]),probability=a.probability,annual_probability=b.probability,actual_up=int(a.actual_up),case=case,changed_probability=a.probability!=b.probability))
    weekly=pd.DataFrame(weekly).drop(columns='Index');assert len(weekly)==4352
    for p in periods():
        for (h,m),g in weekly[weekly.date.between(p['start'],p['end'])].groupby(['history','method']):changes.append(dict(period=p['name'],history=h,method=m,n=len(g),fit_eligible_weeks=int(g.fit_eligible.sum()),validation_accepted_weeks=int(g.validation_accepted.sum()),changed_probability_weeks=int(g.changed_probability.sum()),regressions=int(g['case'].eq('regression').sum()),recoveries=int(g['case'].eq('recovery').sum())))
    outcomes=[]
    for r in csv('gate_decisions').itertuples():
        g=weekly[weekly.cutoff.eq(r.cutoff)&weekly.method.eq(r.method)&weekly.family.eq(r.family)];g=g if r.component=='all' else g[g.state.eq(r.component)]
        for label in ['gated','direct']:
            h=POLICIES[r.family+'_'+label];q=g[g.history.eq(h)];n=len(q);outcomes.append(dict(cutoff=r.cutoff,method=r.method,family=r.family,component=r.component,accepted=r.accepted,history=h,n=n,brier_difference=float(((q.probability-q.actual_up)**2-(q.annual_probability-q.actual_up)**2).mean()) if n else None,regressions=int(q['case'].eq('regression').sum()),recoveries=int(q['case'].eq('recovery').sum())))
    return dict(ensemble_metrics=pd.DataFrame(metrics),seed_metrics=pd.DataFrame(sm),state_metrics=pd.DataFrame(state),weekly_policy_effects=weekly,direction_changes=pd.DataFrame(changes),quarter_outcomes=pd.DataFrame(outcomes),all_2026_cases=weekly[weekly.year.eq(2026)]),pairs
def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs=compute(csv('model_predictions'),csv('ensemble_predictions'));files=[]
    for name,g in tables.items():p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    p=OUT/'primary_comparisons.json';save(p,pairs);files.append(p);assert len(pairs)==84;finish(run,files,exploratory_comparisons=84,independent_holdout=False)
    m=tables['ensemble_metrics'];print(m[m.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])&m.history.isin([ANNUAL]+NEW)&m.method.isin(PRIMARY)][['period','history','method','correct_directions','n','brier']].to_string(index=False),flush=True)
if __name__=='__main__':main()

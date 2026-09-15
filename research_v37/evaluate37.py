from common37 import *
def compute(models,ensemble):
    metrics=[];seed_metrics=[];sm=[];pairs=[];weekly=[];support=csv('support_contract').set_index(['cutoff','state']);context=csv('weekly_context');ctx=context[['row_index','date','state','head_cutoff','encoder_cutoff']]
    for w in periods():
        a=ensemble[ensemble.date.between(w['start'],w['end'])];b=models[models.date.between(w['start'],w['end'])&models.seed.ne(-1)]
        for (h,m),g in a.groupby(['history','method']):metrics.append(dict(period=w['name'],history=h,method=m,**metric(g)))
        for (h,m,s),g in b.groupby(['history','method','seed']):seed_metrics.append(dict(period=w['name'],history=h,method=m,seed=int(s),**metric(g)))
        q=a[a.history.isin([ANNUAL]+NEW)&a.method.isin(LEARNED)].merge(ctx,on=['row_index','date'],validate='many_to_one')
        for h in [ANNUAL]+NEW:
            for m in LEARNED:
                for state in STATES:
                    g=q[q.history.eq(h)&q.method.eq(m)&q.state.eq(state)];sm.append(dict(period=w['name'],history=h,method=m,state=state,supported=len(g)>=10,**metric(g)))
    for w in cfg()['windows']:
        e=ensemble[ensemble.date.between(w['start'],w['end'])];ids=statistics.bootstrap_indices(w['n'])
        for comp in cfg()['primary_comparisons_per_window']:
            a=e[e.history.eq(comp['history'])&e.method.eq(comp['candidate'])].sort_values('date');b=e[e.history.eq(comp['reference_history'])&e.method.eq(comp['reference'])].sort_values('date');np.testing.assert_array_equal(a.date,b.date);assert len(a)==w['n']
            def loss(g):return g.direction_up.ne(g.actual_up).to_numpy(float) if comp['metric']=='direction_error' else (g.probability.to_numpy()-g.actual_up.to_numpy())**2
            pairs.append(dict(window=w['name'],**comp,**statistics.difference(loss(a)-loss(b),ids)))
    for r,p in zip(pairs,statistics.holm([r['p'] for r in pairs])):r['holm_adjusted_p']=p
    for history in NEW:
        for m in LEARNED:
            a=ensemble[ensemble.history.eq(history)&ensemble.method.eq(m)].sort_values('date').reset_index(drop=True);b=ensemble[ensemble.history.eq(ANNUAL)&ensemble.method.eq(m)].sort_values('date').reset_index(drop=True);np.testing.assert_array_equal(a.date,b.date)
            for r0,r1 in zip(b.itertuples(),a.itertuples()):
                c=context[context.row_index.eq(r1.row_index)].iloc[0];s=support.loc[(c.head_cutoff,c.state)];old=r0.direction_up==r0.actual_up;new=r1.direction_up==r1.actual_up;case='regression' if old and not new else 'recovery' if new and not old else 'stable_correct' if old else 'stable_wrong'
                weekly.append(dict(history=history,method=m,row_index=r1.row_index,date=r1.date,year=r1.year,head_cutoff=c.head_cutoff,encoder_cutoff=c.encoder_cutoff,state=c.state,new_state_n=int(s.new_n),eligible=bool(s.eligible),actual_up=r1.actual_up,annual_probability=r0.probability,probability=r1.probability,probability_change=r1.probability-r0.probability,signed_probability_change=(2*r1.actual_up-1)*(r1.probability-r0.probability),annual_correct=old,correct=new,case=case))
    weekly=pd.DataFrame(weekly);changes=[]
    for w in periods():
        for (h,m),g in weekly[weekly.date.between(w['start'],w['end'])].groupby(['history','method']):changes.append(dict(period=w['name'],history=h,method=m,n=len(g),eligible_weeks=int(g.eligible.sum()),fallback_weeks=int((~g.eligible).sum()),recoveries=int(g['case'].eq('recovery').sum()),regressions=int(g['case'].eq('regression').sum()),mean_signed_probability_change=float(g.signed_probability_change.mean())))
    return dict(ensemble_metrics=pd.DataFrame(metrics),seed_metrics=pd.DataFrame(seed_metrics),state_metrics=pd.DataFrame(sm),weekly_correction_effects=weekly,direction_changes=pd.DataFrame(changes),all_2026_cases=weekly[weekly.year.eq(2026)],flips_2026=weekly[weekly.year.eq(2026)&weekly['case'].isin(['recovery','regression'])]),pairs
def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs=compute(csv('model_predictions'),csv('ensemble_predictions'));files=[]
    for name,g in tables.items():p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    p=OUT/'primary_comparisons.json';save(p,pairs);files.append(p);assert len(pairs)==48;finish(run,files,exploratory_comparisons=48,policies_selected_from_results=0,independent_holdout=False)
    m=tables['ensemble_metrics'];print(m[m.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])&m.history.isin([ANNUAL]+NEW)&m.method.isin(PRIMARY)][['period','history','method','correct_directions','n','accuracy','brier']].to_string(index=False),flush=True)
if __name__=='__main__':main()

from common48 import *
def case(a,b,y):return 'recovery' if a==y and b!=y else 'regression' if a!=y and b==y else 'stable_correct' if a==y else 'stable_wrong'
def compute(models,ensemble):
    met=[];seeds=[];states=[];ctx=csv('weekly_context')[['row_index','date','state']]
    for w in periods():
        e=ensemble[ensemble.date.between(w['start'],w['end'])];s=models[models.date.between(w['start'],w['end'])&models.seed.ne(-1)]
        for (h,m),g in e.groupby(['history','method']):met.append(dict(period=w['name'],history=h,method=m,**metric(g)))
        for (h,m,seed),g in s.groupby(['history','method','seed']):seeds.append(dict(period=w['name'],history=h,method=m,seed=int(seed),**metric(g)))
        q=e[e.history.isin(REPORT_HIST)&e.method.isin(METHODS)].merge(ctx,on=['row_index','date'],validate='many_to_one')
        for h in REPORT_HIST:
            for m in METHODS:
                for state in STATES:states.append(dict(period=w['name'],history=h,method=m,state=state,**metric(q[q.history.eq(h)&q.method.eq(m)&q.state.eq(state)])))
    pairs=[]
    for w in cfg()['windows']:
        e=ensemble[ensemble.date.between(w['start'],w['end'])];ids=bootstrap_indices(w['n'])
        for spec in cfg()['primary_comparisons_per_window']:
            a=e[e.history.eq(spec['history'])&e.method.eq(spec['candidate'])].sort_values('date');b=e[e.history.eq(spec['reference_history'])&e.method.eq(spec['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist()
            def loss(x):return x.direction_up.ne(x.actual_up).to_numpy(float) if spec['metric']=='direction_error' else (x.probability.to_numpy()-x.actual_up.to_numpy())**2
            pairs.append(dict(window=w['name'],**spec,**difference(loss(a)-loss(b),ids)))
    for r,p in zip(pairs,holm([r['p'] for r in pairs])):r['holm_adjusted_p']=p
    ei=ensemble.set_index(['history','method','row_index']);q=ensemble[ensemble.history.isin(NEW)&ensemble.method.isin(METHODS)].merge(ctx,on=['row_index','date'],validate='many_to_one');weekly=[]
    for r in q.itertuples():
        out=dict(history=r.history,method=r.method,cutoff=r.cutoff,date=r.date,row_index=r.row_index,state=r.state,year=r.year,probability=r.probability,actual_up=r.actual_up)
        for label,h in [('annual',ANNUAL),('weighted',WEIGHTED),('quarter4',QUARTER)]:
            b=ei.loc[(h,r.method,r.row_index)];out.update({label+'_probability':b.probability,'changed_vs_'+label:r.probability!=b.probability,'case_vs_'+label:case(r.direction_up,b.direction_up,r.actual_up),'brier_vs_'+label:(r.probability-r.actual_up)**2-(b.probability-b.actual_up)**2})
        weekly.append(out)
    weekly=pd.DataFrame(weekly);comparisons=[]
    for w in periods():
        for (history,method),g in weekly[weekly.date.between(w['start'],w['end'])].groupby(['history','method']):
            for label,h in [('annual',ANNUAL),('weighted',WEIGHTED),('quarter4',QUARTER)]:comparisons.append(dict(period=w['name'],history=history,method=method,reference_history=h,n=len(g),changed_probability_weeks=int(g['changed_vs_'+label].sum()),recoveries=int(g['case_vs_'+label].eq('recovery').sum()),regressions=int(g['case_vs_'+label].eq('regression').sum()),brier_difference=float(g['brier_vs_'+label].mean())))
    factorial=[];summary=[]
    for method in METHODS:
        for idx in ensemble[ensemble.history.eq(ANNUAL)&ensemble.method.eq(method)].row_index:
            rows=[ei.loc[(h,method,idx)] for h in [ANNUAL,INTERCEPT,SLOPES,WEIGHTED]]
            for loss in ['direction_error','brier']:
                u,i,s,w=[loss_values(r,loss) for r in rows];factorial.append(dict(method=method,row_index=int(idx),date=rows[0].date,metric=loss,uniform_loss=u,intercept_loss=i,slopes_loss=s,weighted_loss=w,**factorial_values(u,i,s,w)))
    factorial=pd.DataFrame(factorial)
    columns=['total','intercept_at_uniform_slopes','slopes_at_uniform_intercept','intercept_at_weighted_slopes','slopes_at_weighted_intercept','interaction','symmetric_intercept','symmetric_slopes']
    for period in periods():
        for (method,loss),g in factorial[factorial.date.between(period['start'],period['end'])].groupby(['method','metric']):summary.append(dict(period=period['name'],method=method,metric=loss,n=len(g),**{k:float(g[k].mean()) for k in columns}))
    return dict(ensemble_metrics=pd.DataFrame(met),seed_metrics=pd.DataFrame(seeds),state_metrics=pd.DataFrame(states),weekly_policy_effects=weekly,reference_comparisons=pd.DataFrame(comparisons),weekly_factorial_effects=factorial,factorial_effects=pd.DataFrame(summary),all_2026_cases=weekly[weekly.year.eq(2026)]),pairs

def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs=compute(csv('model_predictions'),csv('ensemble_predictions'));files=[]
    for name,t in tables.items():p=OUT/f'{name}.csv';t.to_csv(p,index=False);files.append(p)
    p=OUT/'primary_comparisons.json';save(p,pairs);files.append(p);assert len(pairs)==72;check_frozen();finish(run,files,exploratory_comparisons=72,new_blind_holdout=False)
    m=tables['ensemble_metrics'];print(m[m.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])&m.history.isin(REPORT_HIST)&m.method.isin(PRIMARY)][['period','history','method','correct_directions','n','brier']].to_string(index=False),flush=True)
if __name__=='__main__':main()

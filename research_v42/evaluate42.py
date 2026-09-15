from common42 import *

def change_case(candidate,reference,actual):
    good=candidate==actual;old=reference==actual
    return 'recovery' if good and not old else 'regression' if old and not good else 'stable_correct' if good else 'stable_wrong'

def compute(models,ensemble):
    metrics=[];seedmetrics=[];states=[];pairs=[];ctx=csv('weekly_context')[['row_index','date','state']]
    for p in periods():
        e=ensemble[ensemble.date.between(p['start'],p['end'])];s=models[models.date.between(p['start'],p['end'])&models.seed.ne(-1)]
        for (h,m),g in e.groupby(['history','method']):metrics.append(dict(period=p['name'],history=h,method=m,**metric(g)))
        for (h,m,seed),g in s.groupby(['history','method','seed']):seedmetrics.append(dict(period=p['name'],history=h,method=m,seed=int(seed),**metric(g)))
        joined=e[e.history.isin(REPORT_HIST)&e.method.isin(LEARNED)].merge(ctx,on=['row_index','date'],validate='many_to_one')
        for history in REPORT_HIST:
            for method in LEARNED:
                for state in STATES:states.append(dict(period=p['name'],history=history,method=method,state=state,**metric(joined[joined.history.eq(history)&joined.method.eq(method)&joined.state.eq(state)])))
    for w in cfg()['windows']:
        e=ensemble[ensemble.date.between(w['start'],w['end'])];ids=statistics.bootstrap_indices(w['n'])
        for comp in cfg()['primary_comparisons_per_window']:
            a=e[e.history.eq(comp['history'])&e.method.eq(comp['candidate'])].sort_values('date');b=e[e.history.eq(comp['reference_history'])&e.method.eq(comp['reference'])].sort_values('date');np.testing.assert_array_equal(a.date,b.date)
            def loss(g):return g.direction_up.ne(g.actual_up).to_numpy(float) if comp['metric']=='direction_error' else (g.probability.to_numpy()-g.actual_up.to_numpy())**2
            pairs.append(dict(window=w['name'],**comp,**statistics.difference(loss(a)-loss(b),ids)))
    for r,p in zip(pairs,statistics.holm([r['p'] for r in pairs])):r['holm_adjusted_p']=p
    ei=ensemble.set_index(['history','method','row_index']);rows=[];seed=csv('seed_routing');first=seed[seed.seed.eq(cfg()['seeds'][0])]
    for r in first.itertuples():
        a=ei.loc[(NEW[0],r.method,r.row_index)];base=ei.loc[(ANNUAL,r.method,r.row_index)];month=ei.loc[(MONTHLY,r.method,r.row_index)];quarter=ei.loc[(QUARTER,r.method,r.row_index)]
        rows.append(dict(history=NEW[0],method=r.method,row_index=r.row_index,date=r.date,year=int(r.date[:4]),cutoff=r.cutoff,state=r.state,action=r.action,original_accepted=r.original_accepted,original_reason=r.original_reason,retention_reason=r.retention_reason,source_cutoff=r.source_cutoff,expiry=r.expiry,source_age_days=r.source_age_days,used_retained_source=r.used_retained_source,probability=a.probability,annual_probability=base.probability,monthly_probability=month.probability,quarterly_probability=quarter.probability,actual_up=int(a.actual_up),changed_vs_annual=a.probability!=base.probability,changed_vs_monthly=a.probability!=month.probability,changed_vs_quarterly=a.probability!=quarter.probability,case_vs_annual=change_case(a.direction_up,base.direction_up,a.actual_up),case_vs_monthly=change_case(a.direction_up,month.direction_up,a.actual_up),case_vs_quarterly=change_case(a.direction_up,quarter.direction_up,a.actual_up),brier_vs_annual=(a.probability-a.actual_up)**2-(base.probability-a.actual_up)**2,brier_vs_monthly=(a.probability-a.actual_up)**2-(month.probability-a.actual_up)**2,brier_vs_quarterly=(a.probability-a.actual_up)**2-(quarter.probability-a.actual_up)**2))
    weekly=pd.DataFrame(rows);assert len(weekly)==1088;coverage=[];comparisons=[]
    for w in periods():
        for method in LEARNED:
            g=weekly[weekly.method.eq(method)&weekly.date.between(w['start'],w['end'])];carry=g[g.used_retained_source]
            coverage.append(dict(period=w['name'],method=method,n=len(g),fresh_weeks=int(g.action.eq('fresh').sum()),carry_weeks=int(g.action.eq('carry').sum()),fallback_weeks=int(g.action.eq('fallback').sum()),used_retained_weeks=len(carry),changed_vs_annual=int(g.changed_vs_annual.sum()),changed_vs_monthly=int(g.changed_vs_monthly.sum()),changed_vs_quarterly=int(g.changed_vs_quarterly.sum()),maximum_carried_age=float(carry.source_age_days.max()) if len(carry) else None))
            for name,ref in [('annual',ANNUAL),('monthly',MONTHLY),('quarterly',QUARTER)]:comparisons.append(dict(period=w['name'],method=method,reference_history=ref,n=len(g),changed_probability_weeks=int(g['changed_vs_'+name].sum()),recoveries=int(g['case_vs_'+name].eq('recovery').sum()),regressions=int(g['case_vs_'+name].eq('regression').sum()),brier_difference=float(g['brier_vs_'+name].mean()) if len(g) else None))
    outcomes=[]
    for r in csv('retention_decisions').itertuples():
        g=weekly[weekly.cutoff.eq(r.cutoff)&weekly.method.eq(r.method)&weekly.state.eq(r.state)];outcomes.append(dict(cutoff=r.cutoff,method=r.method,state=r.state,action=r.action,retention_reason=r.retention_reason,source_cutoff=r.source_cutoff,expiry=r.expiry,n=len(g),changed_vs_monthly=int(g.changed_vs_monthly.sum()),recoveries_vs_monthly=int(g.case_vs_monthly.eq('recovery').sum()),regressions_vs_monthly=int(g.case_vs_monthly.eq('regression').sum()),brier_vs_monthly=float(g.brier_vs_monthly.mean()) if len(g) else None))
    return dict(ensemble_metrics=pd.DataFrame(metrics),seed_metrics=pd.DataFrame(seedmetrics),state_metrics=pd.DataFrame(states),weekly_policy_effects=weekly,retention_coverage=pd.DataFrame(coverage),reference_comparisons=pd.DataFrame(comparisons),decision_outcomes=pd.DataFrame(outcomes),carried_weeks=weekly[weekly.used_retained_source],all_2026_cases=weekly[weekly.year.eq(2026)]),pairs

def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs=compute(csv('model_predictions'),csv('ensemble_predictions'));files=[]
    for name,g in tables.items():p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    p=OUT/'primary_comparisons.json';save(p,pairs);files.append(p);assert len(pairs)==36;finish(run,files,exploratory_comparisons=36,independent_holdout=False)
    m=tables['ensemble_metrics'];print(m[m.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])&m.history.isin(REPORT_HIST)&m.method.isin(PRIMARY)][['period','history','method','correct_directions','n','brier']].to_string(index=False),flush=True)
if __name__=='__main__':main()

from common45 import *
def case(up,old,y):return 'recovery' if up==y and old!=y else 'regression' if up!=y and old==y else 'stable_correct' if up==y else 'stable_wrong'
def compute(models,ensemble):
    met=[];seeds=[];states=[];ctx=csv('weekly_context')[['row_index','date','state']]
    for w in periods():
        e=ensemble[ensemble.date.between(w['start'],w['end'])];s=models[models.date.between(w['start'],w['end'])&models.seed.ne(-1)]
        for (h,m),g in e.groupby(['history','method']):met.append(dict(period=w['name'],history=h,method=m,**metric(g)))
        for (h,m,seed),g in s.groupby(['history','method','seed']):seeds.append(dict(period=w['name'],history=h,method=m,seed=int(seed),**metric(g)))
        q=e[e.history.isin(REPORT_HIST)&e.method.isin(METHODS)].merge(ctx,on=['row_index','date'],validate='many_to_one')
        for h in REPORT_HIST:
            for m in METHODS:
                for st in STATES:states.append(dict(period=w['name'],history=h,method=m,state=st,**metric(q[q.history.eq(h)&q.method.eq(m)&q.state.eq(st)])))
    pairs=[]
    for w in cfg()['windows']:
        e=ensemble[ensemble.date.between(w['start'],w['end'])];ids=bootstrap_indices(w['n'])
        for spec in cfg()['primary_comparisons_per_window']:
            a=e[e.history.eq(spec['history'])&e.method.eq(spec['candidate'])].sort_values('date');b=e[e.history.eq(spec['reference_history'])&e.method.eq(spec['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist()
            def loss(x):return x.direction_up.ne(x.actual_up).to_numpy(float) if spec['metric']=='direction_error' else (x.probability.to_numpy()-x.actual_up.to_numpy())**2
            pairs.append(dict(window=w['name'],**spec,**difference(loss(a)-loss(b),ids)))
    for r,p in zip(pairs,holm([r['p'] for r in pairs])):r['holm_adjusted_p']=p
    ei=ensemble.set_index(['history','method','row_index']);source=csv('seed_routing');source=source[source.seed.eq(cfg()['seeds'][0])];weekly=[]
    for r in source.itertuples():
        a=ei.loc[(r.history,r.method,r.row_index)];b=ei.loc[(ANNUAL,r.method,r.row_index)];q=ei.loc[(QUARTER,r.method,r.row_index)];out={k:getattr(r,k) for k in ['history','cutoff','method','row_index','date','state','family','component','fit_eligible','validation_accepted','gate_reason']};out.update(year=int(r.date[:4]),probability=a.probability,actual_up=int(a.actual_up))
        for label,ref in [('annual',b),('quarter4',q)]:out.update({f'{label}_probability':ref.probability,f'changed_vs_{label}':a.probability!=ref.probability,f'case_vs_{label}':case(a.direction_up,ref.direction_up,a.actual_up),f'brier_vs_{label}':(a.probability-a.actual_up)**2-(ref.probability-ref.actual_up)**2})
        weekly.append(out)
    weekly=pd.DataFrame(weekly);assert len(weekly)==2176;coverage=[];comparisons=[]
    for w in periods():
        for (h,m),g in weekly[weekly.date.between(w['start'],w['end'])].groupby(['history','method']):
            coverage.append(dict(period=w['name'],history=h,method=m,n=len(g),fit_eligible_weeks=int(g.fit_eligible.sum()),validation_accepted_weeks=int(g.validation_accepted.sum()),changed_probability_weeks=int(g.changed_vs_annual.sum())))
            for label,ref in [('annual',ANNUAL),('quarter4',QUARTER)]:comparisons.append(dict(period=w['name'],history=h,method=m,reference_history=ref,n=len(g),changed_probability_weeks=int(g[f'changed_vs_{label}'].sum()),recoveries=int(g[f'case_vs_{label}'].eq('recovery').sum()),regressions=int(g[f'case_vs_{label}'].eq('regression').sum()),brier_difference=float(g[f'brier_vs_{label}'].mean())))
    gates=csv('gate_decisions');outcomes=[];subs=[]
    for r in gates.itertuples():
        h=POLICIES[r.family];g=weekly[weekly.history.eq(h)&weekly.method.eq(r.method)&weekly.cutoff.eq(r.cutoff)&weekly.component.eq(r.component)];outcomes.append(dict(cutoff=r.cutoff,history=h,method=r.method,family=r.family,component=r.component,accepted=bool(r.accepted),training_n=r.training_n,validation_n=r.validation_n,n=len(g),changed_probability_weeks=int(g.changed_vs_annual.sum()),recoveries=int(g.case_vs_annual.eq('recovery').sum()),regressions=int(g.case_vs_annual.eq('regression').sum()),brier_difference=float(g.brier_vs_annual.mean()) if len(g) else None))
        for st in [st for st in STATES if component(st,r.family)==r.component]:
            z=g[g.state.eq(st)];subs.append(dict(cutoff=r.cutoff,history=h,method=r.method,state=st,parent_component=r.component,accepted=bool(r.accepted),n=len(z),recoveries=int(z.case_vs_annual.eq('recovery').sum()),regressions=int(z.case_vs_annual.eq('regression').sum()),brier_difference=float(z.brier_vs_annual.mean()) if len(z) else None))
    support=csv('group_support');old=csv('original_four_state_support').rename(columns={'state':'component'});old['family']='four_state';combined=pd.concat([support,old[support.columns]],ignore_index=True);summ=[]
    for family,g in combined.groupby('family'):
        z=g[g['mode'].eq('ready')];summ.append(dict(family=family,total_cells=len(g),ready_cells=len(z),training_supported_cells=int(z.train_supported.sum()),validation_supported_cells=int(z.validation_supported.sum()),training_support_fraction=float(z.train_supported.mean()),validation_support_fraction=float(z.validation_supported.mean())))
    return dict(ensemble_metrics=pd.DataFrame(met),seed_metrics=pd.DataFrame(seeds),state_metrics=pd.DataFrame(states),weekly_policy_effects=weekly,policy_coverage=pd.DataFrame(coverage),reference_comparisons=pd.DataFrame(comparisons),decision_outcomes=pd.DataFrame(outcomes),substate_outcomes=pd.DataFrame(subs),support_comparison=combined,support_summary=pd.DataFrame(summ),all_2026_cases=weekly[weekly.year.eq(2026)]),pairs
def main():
    check_frozen();check_phase('scoring');assert not (OUT/'evaluation_manifest.json').exists();run=manifest('evaluation');tables,pairs=compute(csv('model_predictions'),csv('ensemble_predictions'));files=[]
    for name,g in tables.items():p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    p=OUT/'primary_comparisons.json';save(p,pairs);files.append(p);assert len(pairs)==48;check_frozen();finish(run,files,exploratory_comparisons=48,new_blind_holdout=False)
    m=tables['ensemble_metrics'];print(m[m.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])&m.history.isin(REPORT_HIST)&m.method.isin(PRIMARY)][['period','history','method','correct_directions','n','brier']].to_string(index=False),flush=True)
if __name__=='__main__':main()

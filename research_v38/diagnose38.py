from common38 import *
def observations():
    member=csv('membership');rows=[]
    for t in read(OUT/'training_inputs.json'):
        d=arrays(t)
        for j,m in enumerate(LEARNED):
            p=probability(d['annual_logits'][:,j])
            rows.extend(dict(cutoff=t['cutoff'],row_index=int(i),method=m,seed=t['seed'],probability=float(q),actual_up=int(y)) for i,q,y in zip(d['row_index'],p,d['direction']))
    daily=pd.DataFrame(rows);ensemble=daily.groupby(['cutoff','row_index','method'],sort=False).agg(probability=('probability','mean'),actual_up=('actual_up','first')).reset_index();ensemble['seed']=-1;daily=pd.concat([daily,ensemble],ignore_index=True)
    tr=member[member.view.ne('future')].merge(daily,on=['cutoff','row_index'],validate='many_to_many');models=csv('model_predictions');e=csv('ensemble_predictions');e['seed']=-1;future=pd.concat([models,e],ignore_index=True);future=future[future.history.eq(ANNUAL)&future.method.isin(LEARNED)];fu=member[member.view.eq('future')].merge(future[['row_index','method','seed','probability','actual_up']],on='row_index',validate='one_to_many')
    result=pd.concat([tr,fu],ignore_index=True);result['residual']=result.actual_up-result.probability;result['score']=result.probability;result['direction_up']=result.probability.gt(.5).astype(int);assert not result.duplicated(['cutoff','view','row_index','method','seed']).any();return result
def metrics(o):
    grouped={k:g for k,g in o.groupby(['cutoff','view','method','seed'],sort=False)};out=[]
    for cutoff in cfg()['decision_dates']:
        for view in VIEWS:
            for m in LEARNED:
                for seed in seeds():
                    g=grouped.get((cutoff,view,m,seed),o.iloc[:0])
                    for state in SCOPES:
                        s=g if state=='all' else g[g.state.eq(state)];r=metric(s);out.append(dict(cutoff=cutoff,year=int(previous_round.annual_for(cutoff)[:4])+1,view=view,method=m,seed=seed,state=state,residual_mean=float(s.residual.mean()) if len(s) else None,residual_sum=float(s.residual.sum()),up_n=int(s.actual_up.sum()),down_n=int(len(s)-s.actual_up.sum()),opposite_class_pairs=int(s.actual_up.sum()*(len(s)-s.actual_up.sum())),**r))
    return pd.DataFrame(out)
def transfers(t):
    ix=t.set_index(['cutoff','view','method','seed','state']);rows=[];dec=[]
    for cutoff in cfg()['decision_dates']:
        for m in LEARNED:
            for seed in seeds():
                for state in STATES:
                    d,f,w=[ix.loc[(cutoff,v,m,seed,state)] for v in VIEWS];rows.append(dict(cutoff=cutoff,year=int(previous_round.annual_for(cutoff)[:4])+1,method=m,seed=seed,state=state,daily_n=int(d.n),friday_n=int(f.n),future_n=int(w.n),daily_residual=d.residual_mean,friday_residual=f.residual_mean,future_residual=w.residual_mean,daily_friday=agreement(d.residual_mean,f.residual_mean),daily_future=agreement(d.residual_mean,w.residual_mean),friday_future=agreement(f.residual_mean,w.residual_mean),r37_eligible=d.n>=20,weekly_count_flag=f.n>=10 and w.n>=10,daily_auroc=d.auroc,friday_auroc=f.auroc,future_auroc=w.auroc))
                for source,target in [('daily','friday'),('friday','future'),('daily','future')]:
                    a=t[t.cutoff.eq(cutoff)&t.view.eq(source)&t.method.eq(m)&t.seed.eq(seed)&t.state.ne('all')].set_index('state').loc[STATES];b=t[t.cutoff.eq(cutoff)&t.view.eq(target)&t.method.eq(m)&t.seed.eq(seed)&t.state.ne('all')].set_index('state').loc[STATES];dec.append(dict(cutoff=cutoff,year=int(previous_round.annual_for(cutoff)[:4])+1,method=m,seed=seed,source=source,target=target,source_n=int(a.n.sum()),target_n=int(b.n.sum()),**decomposition(a.n,a.residual_sum,b.n,b.residual_sum)))
    return pd.DataFrame(rows),pd.DataFrame(dec)
def transfer_summary(t):
    out=[]
    for p in periods():
        for m in LEARNED:
            for seed in seeds():
                a=t[t.year.isin(period_years(p))&t.method.eq(m)&t.seed.eq(seed)]
                for view in ['daily','friday']:
                    for cohort in ['available','r37_eligible','weekly_count10']:
                        g=a[a[view+'_n'].gt(0)&a.future_n.gt(0)]
                        if cohort!='available':g=g[g.r37_eligible]
                        if cohort=='weekly_count10':g=g[g.weekly_count_flag]
                        categories=g[view+'_future'];out.append(dict(period=p['name'],method=m,seed=seed,view=view,cohort=cohort,cells=len(g),future_weeks=int(g.future_n.sum()),**{k+'_cells':int(categories.eq(k).sum()) for k in ['same','opposite','near_zero','undefined']},**{k+'_weeks':int(g.loc[categories.eq(k),'future_n'].sum()) for k in ['same','opposite','near_zero','undefined']}))
    return pd.DataFrame(out)
def archived_accounting():
    models=csv('model_predictions');e=csv('ensemble_predictions');e['seed']=-1;a=pd.concat([models,e],ignore_index=True);a=a[a.method.isin(LEARNED)&a.history.isin([ANNUAL]+list(POLICIES.values()))];ctx=csv('weekly_context')[['row_index','date','head_cutoff','encoder_cutoff','state','added_state_n']];a=a.merge(ctx,on=['row_index','date'],validate='many_to_one');b=a[a.history.eq(ANNUAL)][['row_index','method','seed','probability']].rename(columns={'probability':'annual_probability'});c=a[a.history.ne(ANNUAL)].merge(b,on=['row_index','method','seed'],validate='many_to_one');d,res,size,align,change=brier_terms(c.annual_probability.to_numpy(),c.probability.to_numpy(),c.actual_up.to_numpy());c['delta']=d;c['annual_residual']=res;c['squared_shift']=size;c['alignment_term']=align;c['brier_change']=change;c['eligible']=c.added_state_n.ge(20);c['annual_correct']=c.annual_probability.gt(.5).eq(c.actual_up);c['correct']=c.direction_up.eq(c.actual_up);summaries=[];rank=[]
    for p in periods():
        q=c[c.date.between(p['start'],p['end'])]
        for (h,m,seed),g in q.groupby(['history','method','seed']):
            for scope,s in [('all',g),('eligible',g[g.eligible])]:summaries.append(dict(period=p['name'],history=h,method=m,seed=seed,scope=scope,n=len(s),squared_shift=float(s.squared_shift.mean()) if len(s) else None,alignment_term=float(s.alignment_term.mean()) if len(s) else None,brier_change=float(s.brier_change.mean()) if len(s) else None,regressions=int((s.annual_correct&~s.correct).sum()),recoveries=int((~s.annual_correct&s.correct).sum())))
    lookup={k:g for k,g in c.groupby(['head_cutoff','history','method','seed','state'],sort=False)}
    for cutoff in cfg()['decision_dates']:
        for h in POLICIES.values():
            for m in LEARNED:
                for seed in seeds():
                    for state in STATES:
                        g=lookup.get((cutoff,h,m,seed,state),c.iloc[:0]);baseline=g.copy();baseline['probability']=g.annual_probability;baseline['score']=g.annual_probability;baseline['direction_up']=g.annual_probability.gt(.5).astype(int);r=metric(g);b=metric(baseline);rank.append(dict(cutoff=cutoff,history=h,method=m,seed=seed,state=state,n=len(g),annual_auroc=b['auroc'],corrected_auroc=r['auroc'],**ranking(g.annual_probability,g.probability)))
    return c,pd.DataFrame(summaries),pd.DataFrame(rank)
def main():
    check_frozen();assert not (OUT/'diagnosis_manifest.json').exists();run=manifest('diagnosis');o=observations();print('Archived annual diagnostic probabilities joined; no fit or feature inference.',flush=True);m=metrics(o);t,d=transfers(m);s=transfer_summary(t);w,b,r=archived_accounting();assert len(m)==5520 and len(t)==1472 and len(d)==1104 and len(r)==4416;files=[]
    for name,g in [('diagnostic_observations',o),('cohort_metrics',m),('residual_transfer',t),('residual_decomposition',d),('transfer_summary',s),('weekly_brier_accounting',w),('brier_summary',b),('within_cell_ranking',r),('residual_transfer_2026',t[t.year.eq(2026)]),('brier_2026',w[w.year.eq(2026)])]:p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_fits=0,new_feature_inference=0,new_forecast_policies=0,new_inferential_tests=0,metric_cells=len(m),transfer_cells=len(t),decomposition_cells=len(d),ranking_cells=len(r));print('R38 fixed diagnostic tables complete.',flush=True)
if __name__=='__main__':main()

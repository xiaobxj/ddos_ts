from common36 import *
COVERAGE=[f'{p}_{n}' for p in ['market5','decoded25'] for n in ['outside_any','outside_fraction','mean_squared_raw_z']]+['outside_'+n for n in NAMES]

def training_summaries():
    obs,_,targets=prior.data();y=(targets['returns']>0).astype(int);states=csv('state_observations');rows=[]
    for cutoff in cfg()['decision_dates']:
        annual=annual_for(cutoff);ss=states[states.encoder_cutoff.eq(annual)].set_index('row_index')
        for role,ids in roles(obs,cutoff).items():
            g=ss.loc[ids].copy();g['up']=y[ids]
            for state in STATES:
                c=g[g.state.eq(state)];n=len(c);rows.append(dict(cutoff=cutoff,encoder_cutoff=annual,role=role,state=state,n=n,role_n=len(g),up_n=int(c.up.sum()),up_rate=float(c.up.mean()) if n else np.nan,share=n/len(g) if len(g) else np.nan,supported=n>=cfg()['support']['training_cell_min'],**{'mean_'+k:float(c[k].mean()) if n else np.nan for k in COVERAGE+NAMES}))
    summary=pd.DataFrame(rows);comp=[];decomp=[]
    for cutoff in cfg()['decision_dates']:
        a=summary[summary.cutoff.eq(cutoff)&summary.role.eq('annual')].set_index('state').loc[STATES]
        for role in ['added','removed','both']:
            t=summary[summary.cutoff.eq(cutoff)&summary.role.eq(role)].set_index('state').loc[STATES]
            for state in STATES:
                ar=a.loc[state];tr=t.loc[state];comp.append(dict(cutoff=cutoff,encoder_cutoff=annual_for(cutoff),target_role=role,state=state,annual_n=int(ar.n),target_n=int(tr.n),annual_up_n=int(ar.up_n),target_up_n=int(tr.up_n),annual_share=ar.share,target_share=tr.share,share_delta=tr.share-ar.share,annual_up_rate=ar.up_rate,target_up_rate=tr.up_rate,up_rate_delta=tr.up_rate-ar.up_rate,supported=bool(ar.supported and tr.supported),**{k+'_delta':tr['mean_'+k]-ar['mean_'+k] for k in COVERAGE}))
            d=label_decomposition(a.n,a.up_n,t.n,t.up_n);decomp.append(dict(cutoff=cutoff,encoder_cutoff=annual_for(cutoff),target_role=role,annual_n=int(a.n.sum()),target_n=int(t.n.sum()),annual_up_rate=float(a.up_n.sum()/a.n.sum()),target_up_rate=float(t.up_n.sum()/t.n.sum()) if t.n.sum() else np.nan,supported_states=int((a.supported&t.supported).sum()),**d))
    assert len(summary)==460 and len(comp)==276 and len(decomp)==69
    return summary,pd.DataFrame(comp),pd.DataFrame(decomp)

def contexts(summary):
    route=csv('routing');states=csv('state_observations');c=route.merge(states,on=['encoder_cutoff','row_index','date'],validate='one_to_one');assert len(c)==272
    lookup=summary.set_index(['cutoff','role','state']);extra=[]
    for r in c.itertuples():
        a=lookup.loc[(r.head_cutoff,'annual',r.state)];n=lookup.loc[(r.head_cutoff,'added',r.state)]
        extra.append(dict(annual_state_n=int(a.n),added_state_n=int(n.n),annual_state_up_rate=a.up_rate,added_state_up_rate=n.up_rate,conditional_up_rate_delta=n.up_rate-a.up_rate,training_support=bool(a.supported and n.supported)))
    return pd.concat([c.reset_index(drop=True),pd.DataFrame(extra)],axis=1)

def prediction_tables(context):
    keys=['row_index','date','state'];e=csv('ensemble_predictions');m=csv('model_predictions');e=e[e.history.isin(ARMS.values())&e.method.isin(LEARNED)].merge(context[keys],on=['row_index','date'],validate='many_to_one');m=m[m.history.isin(ARMS.values())&m.method.isin(LEARNED)].merge(context[keys],on=['row_index','date'],validate='many_to_one');results=[];seed_results=[]
    for period in periods():
        p=e[e.date.between(period['start'],period['end'])];q=m[m.date.between(period['start'],period['end'])]
        for state in STATES:
            for arm,history in ARMS.items():
                for method in LEARNED:
                    g=p[p.state.eq(state)&p.history.eq(history)&p.method.eq(method)];results.append(dict(period=period['name'],state=state,arm=arm,history=history,method=method,supported=len(g)>=cfg()['support']['weekly_cell_min'],**metric(g)))
                    for seed in cfg()['seeds']:
                        g=q[q.state.eq(state)&q.history.eq(history)&q.method.eq(method)&q.seed.eq(seed)];seed_results.append(dict(period=period['name'],state=state,arm=arm,history=history,method=method,seed=seed,supported=len(g)>=cfg()['support']['weekly_cell_min'],**metric(g)))
    assert len(results)==576 and len(seed_results)==1728;return pd.DataFrame(results),pd.DataFrame(seed_results)

def effect_tables(context):
    c=context.drop(columns=['head_cutoff','encoder_cutoff']);w=csv('archived_weekly_effects').merge(c,on=['row_index','date'],validate='many_to_one');assert len(w)==1088;result=[]
    for p in periods():
        g=w[w.date.between(p['start'],p['end'])]
        for state in STATES:
            for method in LEARNED:
                v=g[g.state.eq(state)&g.method.eq(method)]
                for case in ['all','regression','recovery','stable_correct','stable_wrong']:
                    s=v if case=='all' else v[v['case'].eq(case)];n=len(s)
                    result.append(dict(period=p['name'],state=state,method=method,case=case,n=n,supported=n>=cfg()['support']['weekly_cell_min'],regressions=int(s['case'].eq('regression').sum()),recoveries=int(s['case'].eq('recovery').sum()),training_supported_weeks=int(s.training_support.sum()),**{'mean_'+k:float(s[k].mean()) if n else np.nan for k in EFFECTS},**{'mean_signed_'+k:float(s['signed_'+k].mean()) if n else np.nan for k in EFFECTS}))
    assert len(result)==720;return w,pd.DataFrame(result)

def main():
    check_frozen();check_phase('calibration');assert not (OUT/'diagnosis_manifest.json').exists();run=manifest('diagnosis');summary,comp,decomp=training_summaries();context=contexts(summary);pm,sm=prediction_tables(context);weekly,eff=effect_tables(context);latest=[]
    for annual in sorted(summary.encoder_cutoff.unique()):latest.append(summary[summary.encoder_cutoff.eq(annual)].cutoff.max())
    files=[]
    for n,g in [('training_state_summary',summary),('state_comparisons',comp),('label_mix_decomposition',decomp),('latest_annual_snapshot',summary[summary.cutoff.isin(latest)]),('weekly_context',context),('state_prediction_metrics',pm),('state_seed_metrics',sm),('weekly_state_effects',weekly),('state_effect_summary',eff),('all_2026_cases',weekly[weekly.year.eq(2026)]),('regressions_2026',weekly[weekly.year.eq(2026)&weekly['case'].eq('regression')])]:p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,training_state_cells=460,forecast_state_cells=576,weekly_method_rows=1088,new_predictions=0,new_fits=0,new_inferential_tests=0);print('State-conditioned labels, coverage and archived prediction diagnostics computed.',flush=True)
if __name__=='__main__':main()

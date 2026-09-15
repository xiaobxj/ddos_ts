from common42 import *
from contract42 import synthetic
from verify37 import independent_metric
from scipy.special import expit
import calendar,math

def independent_expiry(date):
    year,month,day=map(int,date.split('-'));ends=[f'{y}-{m:02d}-{calendar.monthrange(y,m)[1]}' for y in [year,year+1] for m in [3,6,9,12]];return min(d for d in ends if d>date)

def assert_records(a,b):
    assert list(a.columns)==list(b.columns) and len(a)==len(b)
    for aa,bb in zip(a.itertuples(index=False,name=None),b.itertuples(index=False,name=None)):
        for x,y in zip(aa,bb):eq(x,y)

def audit_routing():
    gates=csv('gate_decisions');dates=csv('schedule').cutoff.tolist();routes=csv('retention_decisions');ri=routes.set_index(['cutoff','method','state']);gi=gates.set_index(['cutoff','method','component']);assert len(routes)==1088 and not routes.duplicated(['cutoff','method','state']).any();checks=0
    for method in LEARNED:
        for state in STATES:
            selected=[]
            for i,date in enumerate(dates):
                g=gi.loc[(date,method,state)];r=ri.loc[(date,method,state)];source=None
                # Find the last formal acceptance, then inspect every intervening rejection.
                if g.accepted:source=date
                elif g['mode']=='ready' and g.reason=='insufficient_validation' and g.training_n>=10 and g.validation_n<5:
                    candidates=[j for j in range(i) if gi.loc[(dates[j],method,state),'accepted']]
                    if candidates:
                        j=max(candidates);candidate=dates[j];intervening=[gi.loc[(dates[k],method,state)] for k in range(j+1,i+1)]
                        if candidate[:4]==date[:4] and independent_expiry(candidate)>date and all(x['mode']=='ready' and x.reason=='insufficient_validation' and x.training_n>=10 and x.validation_n<5 for x in intervening):source=candidate
                previous=selected[-1] if selected else None;prev_expiry=independent_expiry(previous) if previous else None;expired=bool(previous and prev_expiry<=date);action='fresh' if g.accepted else 'carry' if source else 'fallback'
                reason='accepted_current' if g.accepted else 'retained_validation_shortage' if source else 'retention_expired' if g.reason=='insufficient_validation' and expired else 'no_valid_saved_acceptance' if g.reason=='insufficient_validation' else 'cleared_'+g.reason
                assert r.action==action and r.retention_reason==reason and r.previous_expired==expired and r.original_accepted==g.accepted and r.original_reason==g.reason and r['mode']==g['mode'] and r.training_n==g.training_n and r.validation_n==g.validation_n
                eq(r.previous_source_cutoff,previous);eq(r.previous_expiry,prev_expiry);eq(r.source_cutoff,source);eq(r.expiry,independent_expiry(source) if source else None);eq(r.source_encoder_cutoff,f'{int(source[:4])-1}-12-31' if source else None);selected.append(source);checks+=1
    # Prefix replay: overwrite all later gate evidence; earlier routing must stay unchanged.
    for i,cutoff in enumerate(dates):
        poisoned=gates.copy();future=poisoned.cutoff.gt(cutoff);poisoned.loc[future,'accepted']=False;poisoned.loc[future,'mode']='annual_reset';poisoned.loc[future,'reason']='annual_reset';poisoned.loc[future,['training_n','validation_n']]=0;poisoned['future_outcome_not_an_input']=-1e100
        got=retention_decisions(poisoned,dates[:i+1]);expected=routes[routes.cutoff.le(cutoff)].reset_index(drop=True);assert_records(got,expected)
    return dict(independent_decision_cells=checks,prefix_future_gate_poison_checks=len(dates),fresh_cells=int(routes.action.eq('fresh').sum()),carried_cells=int(routes.action.eq('carry').sum()),fallback_cells=int(routes.action.eq('fallback').sum()))

def audit_predictions():
    models=csv('model_predictions');ensemble=csv('ensemble_predictions');base=csv('baseline_model_predictions');be=csv('baseline_ensemble_predictions');details=csv('seed_routing');decisions=csv('retention_decisions').set_index(['cutoff','method','state']);bank=csv('weekly_signal_bank').set_index(['method','seed','row_index']);heads=read(OUT/'correction_heads.json');hi={(h['cutoff'],h['method'],h['seed'],h['component']):h for h in heads};mi=models.set_index(['history','method','seed','row_index']);ei=ensemble.set_index(['history','method','row_index']);gates=csv('gate_decisions').set_index(['cutoff','method','component']);fallback=0;noncarry=0;carried=0;maxgap=0.
    assert len(models)==100096 and len(ensemble)==37536 and len(details)==3264
    pd.testing.assert_frame_equal(models[~models.history.isin(NEW)].reset_index(drop=True),base,check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(ensemble[~ensemble.history.isin(NEW)].reset_index(drop=True),be,check_dtype=False,atol=1e-14,rtol=0)
    for r in details.itertuples():
        yy,mm,dd=map(int,r.date.split('-'));pm=mm-1 or 12;py=yy if mm>1 else yy-1;decision_date=f'{py}-{pm:02d}-{calendar.monthrange(py,pm)[1]}';assert r.cutoff==decision_date<r.date;d=decisions.loc[(r.cutoff,r.method,r.state)];b=bank.loc[(r.method,r.seed,r.row_index)];a=mi.loc[(NEW[0],r.method,r.seed,r.row_index)];monthly=mi.loc[(MONTHLY,r.method,r.seed,r.row_index)];used=isinstance(d.source_cutoff,str) and d.source_cutoff<r.date<=d.expiry and d.source_cutoff[:4]==r.date[:4]
        if used:
            h=hi[(d.source_cutoff,r.method,r.seed,r.state)];offset=h['offset'];assert gates.loc[(d.source_cutoff,r.method,r.state),'accepted'];assert d.source_encoder_cutoff==b.encoder_cutoff==f'{yy-1}-12-31';assert d.expiry==independent_expiry(d.source_cutoff);eq(r.source_age_days,(pd.Timestamp(r.date)-pd.Timestamp(d.source_cutoff)).days)
        else:offset=0.;assert pd.isna(r.source_age_days)
        expected=float(b.probability) if offset==0 else float(expit(float(b.annual_logit)+offset));gap=abs(a.probability-expected);maxgap=max(maxgap,gap);assert gap<1e-12 and r.offset==offset and r.probability==a.probability and r.annual_probability==b.probability and a.actual_up==b.actual_up and a.direction_up==int(a.probability>.5) and a.cutoff==r.cutoff
        assert r.action==d.action and r.original_accepted==d.original_accepted and r.original_reason==d.original_reason and r.retention_reason==d.retention_reason and r.state==b.state and r.used_retained_source==(used and d.action=='carry')
        eq(r.source_cutoff,d.source_cutoff if used else None);eq(r.expiry,d.expiry if used else None);eq(r.source_encoder_cutoff,d.source_encoder_cutoff if used else None)
        if not r.used_retained_source:assert a.probability==monthly.probability;noncarry+=1
        else:assert not d.original_accepted and d.original_reason=='insufficient_validation';carried+=1
        if offset==0:assert a.probability==b.probability;fallback+=1
        if mm<=3:assert offset==0 and d.action=='fallback'
    newcontrol=models[models.history.eq(NEW[0])&~models.method.isin(LEARNED)].drop(columns='history').reset_index(drop=True);oldcontrol=base[base.history.eq(ANNUAL)&~base.method.isin(LEARNED)].drop(columns='history').reset_index(drop=True);pd.testing.assert_frame_equal(newcontrol,oldcontrol,check_dtype=False,check_exact=True)
    for (history,method,idx),g in models.groupby(['history','method','row_index'],sort=False):
        n=1 if method=='training_frequency' else 3;assert len(g)==n;r=ei.loc[(history,method,idx)];eq(r.score,math.fsum(g.score)/n)
        if method=='native_mse':assert pd.isna(r.probability) and r.direction_up==int(r.score>0)
        else:eq(r.probability,math.fsum(g.probability)/n);assert r.direction_up==int(r.probability>.5)
    cells=0;state_source=ensemble.merge(csv('weekly_context')[['row_index','date','state']],on=['row_index','date'],validate='many_to_one')
    for name,source,keys in [('ensemble_metrics',ensemble,['history','method']),('seed_metrics',models,['history','method','seed']),('state_metrics',state_source,['history','method','state'])]:
        groups={k:g for k,g in source.groupby(keys,sort=False)}
        for r in csv(name).to_dict('records'):
            w=next(w for w in periods() if w['name']==r['period']);g=groups.get(tuple(r[k] for k in keys),source.iloc[:0]);g=g[g.date.between(w['start'],w['end'])]
            for k,v in independent_metric(g).items():eq(v,r[k])
            cells+=1
    return dict(maximum_prediction_gap=maxgap,independent_seed_predictions=len(details),exact_zero_offset_predictions=fallback,exact_original_monthly_noncarry_predictions=noncarry,carried_seed_predictions=carried,independent_metric_cells=cells)

def audit_outcomes():
    weekly=csv('weekly_policy_effects');e=csv('ensemble_predictions').set_index(['history','method','row_index']);assert len(weekly)==1088;di=csv('retention_decisions').set_index(['cutoff','method','state'])
    for r in weekly.itertuples():
        a=e.loc[(NEW[0],r.method,r.row_index)];assert r.probability==a.probability and r.actual_up==a.actual_up
        for label,h in [('annual',ANNUAL),('monthly',MONTHLY),('quarterly',QUARTER)]:
            b=e.loc[(h,r.method,r.row_index)];good=a.direction_up==a.actual_up;old=b.direction_up==b.actual_up;case='recovery' if good and not old else 'regression' if old and not good else 'stable_correct' if good else 'stable_wrong';assert getattr(r,label+'_probability')==b.probability and getattr(r,'changed_vs_'+label)==(a.probability!=b.probability) and getattr(r,'case_vs_'+label)==case;eq(getattr(r,'brier_vs_'+label),(a.probability-a.actual_up)**2-(b.probability-b.actual_up)**2)
        d=di.loc[(r.cutoff,r.method,r.state)];assert r.action==d.action and r.used_retained_source==(d.action=='carry')
    for r in csv('retention_coverage').itertuples():
        w=next(w for w in periods() if w['name']==r.period);g=weekly[weekly.method.eq(r.method)&weekly.date.between(w['start'],w['end'])];carry=g[g.used_retained_source];assert r.n==len(g) and r.fresh_weeks==sum(g.action=='fresh') and r.carry_weeks==sum(g.action=='carry') and r.fallback_weeks==sum(g.action=='fallback') and r.used_retained_weeks==len(carry)
        for label in ['annual','monthly','quarterly']:assert getattr(r,'changed_vs_'+label)==sum(g['changed_vs_'+label])
        eq(r.maximum_carried_age,max(carry.source_age_days) if len(carry) else None)
    for r in csv('reference_comparisons').itertuples():
        w=next(w for w in periods() if w['name']==r.period);g=weekly[weekly.method.eq(r.method)&weekly.date.between(w['start'],w['end'])];label={ANNUAL:'annual',MONTHLY:'monthly',QUARTER:'quarterly'}[r.reference_history];assert r.n==len(g) and r.changed_probability_weeks==sum(g['changed_vs_'+label]) and r.recoveries==sum(g['case_vs_'+label]=='recovery') and r.regressions==sum(g['case_vs_'+label]=='regression');eq(r.brier_difference,math.fsum(g['brier_vs_'+label])/len(g) if len(g) else None)
    for r in csv('decision_outcomes').itertuples():
        g=weekly[weekly.cutoff.eq(r.cutoff)&weekly.method.eq(r.method)&weekly.state.eq(r.state)];d=di.loc[(r.cutoff,r.method,r.state)];assert r.action==d.action and r.retention_reason==d.retention_reason and r.n==len(g) and r.changed_vs_monthly==sum(g.changed_vs_monthly) and r.recoveries_vs_monthly==sum(g.case_vs_monthly=='recovery') and r.regressions_vs_monthly==sum(g.case_vs_monthly=='regression');eq(r.source_cutoff,d.source_cutoff);eq(r.expiry,d.expiry);eq(r.brier_vs_monthly,math.fsum(g.brier_vs_monthly)/len(g) if len(g) else None)
    pd.testing.assert_frame_equal(csv('carried_weeks'),weekly[weekly.used_retained_source].reset_index(drop=True),check_dtype=False,check_exact=True);pd.testing.assert_frame_equal(csv('all_2026_cases'),weekly[weekly.year.eq(2026)].reset_index(drop=True),check_dtype=False,check_exact=True)
    return dict(independent_weekly_effects=len(weekly),independent_decision_outcomes=len(csv('decision_outcomes')))

def audit_inference():
    e=csv('ensemble_predictions');pairs=read(OUT/'primary_comparisons.json');ps=[]
    for w in cfg()['windows']:
        n=w['n'];starts=np.random.default_rng(20260910).integers(n,size=(10000,(n+7)//8));ids=np.array([[(int(s)+j)%n for s in row for j in range(8)][:n] for row in starts]);np.testing.assert_array_equal(ids,statistics.bootstrap_indices(n));p=e[e.date.between(w['start'],w['end'])]
        for r in [r for r in pairs if r['window']==w['name']]:
            a=p[p.history.eq(r['history'])&p.method.eq(r['candidate'])].sort_values('date');b=p[p.history.eq(r['reference_history'])&p.method.eq(r['reference'])].sort_values('date');assert a.date.tolist()==b.date.tolist() and len(a)==n;v=(a.direction_up.to_numpy()!=a.actual_up.to_numpy()).astype(float)-(b.direction_up.to_numpy()!=b.actual_up.to_numpy()).astype(float) if r['metric']=='direction_error' else (a.probability.to_numpy()-a.actual_up.to_numpy())**2-(b.probability.to_numpy()-b.actual_up.to_numpy())**2;mean=float(v.mean());boot=v[ids].mean(1);center=(v-mean)[ids].mean(1);pval=(1+int((abs(center)>=abs(mean)).sum()))/10001;lo,hi=np.quantile(boot,[.025,.975]);eq(mean,r['difference']);eq(lo,r['ci95_low']);eq(hi,r['ci95_high']);assert pval==r['p'];ps.append(pval)
    adjusted=[0.]*len(ps);last=0.
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):last=max(last,min(1.,ps[i]*(len(ps)-rank)));adjusted[i]=last
    np.testing.assert_array_equal(adjusted,[r['holm_adjusted_p'] for r in pairs]);assert len(pairs)==36

def main():
    prep=check_frozen();assert not (OUT/'verification.json').exists();last=read(OUT/'contract_verification.json')['completed_utc'];start=time.time()
    for phase in ['routing','scoring','evaluation']:
        r=check_phase(phase);assert last<=r['started_utc']<r['finished_utc'];last=r['finished_utc']
        for k in ['protocol_sha256','source_sha256','input_sha256']:assert r[k]==prep[k]
    synthetic();routing=audit_routing();print('Independent bounded retention and future-gate isolation PASS.',flush=True);pred=audit_predictions();diag=audit_outcomes();audit_inference();assert old_evidence()==prep['old_evidence']
    for src,name in COPIES:assert sha(src)==sha(OUT/name)
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5691,old_forecast_histories_preserved=22,new_fits=0,independent_comparisons=36,**routing,**pred,**diag));print('R42 independent sources,predictions,metrics,diagnostics and36comparisons PASS.',flush=True)
if __name__=='__main__':main()

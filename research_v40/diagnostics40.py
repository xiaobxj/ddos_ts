from common40 import *

def diagnostics(ensemble,weekly):
    ei=ensemble.set_index(['history','method','row_index']);route=csv('routing').set_index('row_index');rows=[]
    for r in weekly.itertuples():
        q=ei.loc[(QUARTER[r.history],r.method,r.row_index)];base=ei.loc[(ANNUAL,r.method,r.row_index)]
        ma=r.probability!=base.probability;qa=q.probability!=base.probability
        relation='both_active' if ma and qa else 'monthly_only' if ma else 'quarterly_only' if qa else 'neither'
        mc=int(r.probability>.5)==int(r.actual_up);qc=int(q.direction_up)==int(r.actual_up)
        case='recovery' if mc and not qc else 'regression' if qc and not mc else 'stable_correct' if mc else 'stable_wrong'
        rows.append(dict(history=r.history,reference_history=QUARTER[r.history],method=r.method,row_index=r.row_index,date=r.date,year=r.year,state=r.state,monthly_cutoff=r.cutoff,quarterly_cutoff=route.loc[r.row_index,'quarter_cutoff'],monthly_probability=r.probability,quarterly_probability=q.probability,annual_probability=base.probability,actual_up=r.actual_up,monthly_active=ma,quarterly_active=qa,activation_relation=relation,changed_vs_quarter=r.probability!=q.probability,monthly_correct=mc,quarterly_correct=qc,case=case,brier_difference=(r.probability-r.actual_up)**2-(q.probability-r.actual_up)**2))
    comparison=pd.DataFrame(rows);assert len(comparison)==2176;summaries=[]
    for p in periods():
        for h in NEW:
            for m in LEARNED:
                allrows=comparison[comparison.history.eq(h)&comparison.method.eq(m)&comparison.date.between(p['start'],p['end'])]
                for relation in ['all','both_active','monthly_only','quarterly_only','neither']:
                    g=allrows if relation=='all' else allrows[allrows.activation_relation.eq(relation)]
                    summaries.append(dict(period=p['name'],history=h,reference_history=QUARTER[h],method=m,relation=relation,n=len(g),changed_vs_quarter=int(g.changed_vs_quarter.sum()),monthly_active_weeks=int(g.monthly_active.sum()),quarterly_active_weeks=int(g.quarterly_active.sum()),monthly_correct=int(g.monthly_correct.sum()),quarterly_correct=int(g.quarterly_correct.sum()),recoveries=int(g['case'].eq('recovery').sum()),regressions=int(g['case'].eq('regression').sum()),brier_difference=float(g.brier_difference.mean()) if len(g) else None))
    gates=csv('gate_decisions');qi=csv('quarterly_gates').set_index(['cutoff','method','family','component']);quarters=csv('quarterly_schedule').cutoff.tolist();gi=gates.set_index(['cutoff','method','component']);dates=cfg()['decision_dates'];register=[]
    for i,c in enumerate(dates):
        qc=max(d for d in quarters if d<=c)
        for m in LEARNED:
            for state in STATES:
                r=gi.loc[(c,m,state)];before=bool(gi.loc[(dates[i-1],m,state),'accepted']) if i else False;accepted=bool(r.accepted)
                transition='enabled' if accepted and not before else 'stopped' if before and not accepted else 'retained_on' if accepted else 'retained_off'
                future=comparison[comparison.history.eq(POLICIES['state_gated'])&comparison.method.eq(m)&comparison.state.eq(state)&comparison.monthly_cutoff.eq(c)]
                register.append(dict(cutoff=c,previous_cutoff=dates[i-1] if i else None,quarterly_cutoff=qc,method=m,state=state,mode=r['mode'],extra_month=c not in quarters,accepted=accepted,previous_accepted=before,quarterly_accepted=bool(qi.loc[(qc,m,'state',state),'accepted']),transition=transition,reason=r.reason,training_n=int(r.training_n),validation_n=int(r.validation_n),future_n=len(future),future_brier_vs_annual=float(((future.monthly_probability-future.actual_up)**2-(future.annual_probability-future.actual_up)**2).mean()) if len(future) else None,future_brier_vs_quarter=float(future.brier_difference.mean()) if len(future) else None,future_recoveries_vs_quarter=int(future['case'].eq('recovery').sum()),future_regressions_vs_quarter=int(future['case'].eq('regression').sum())))
    return dict(cadence_weekly_comparison=comparison,cadence_summary=pd.DataFrame(summaries),gate_transitions=pd.DataFrame(register))

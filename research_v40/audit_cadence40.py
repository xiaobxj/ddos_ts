from common40 import *
import calendar,math

def eq(a,b):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(a-b)<1e-12,(a,b)

def audit_cadence():
    route=csv('routing');quarters=csv('quarterly_schedule').cutoff.tolist();dates=cfg()['decision_dates']
    # Independent calendar arithmetic, including signal dates exactly on month ends.
    for r in route.itertuples():
        yy,mm,dd=map(int,r.date.split('-'));pm=mm-1 or 12;py=yy if mm>1 else yy-1
        expected=f'{py}-{pm:02d}-{calendar.monthrange(py,pm)[1]}'
        assert expected==r.head_cutoff<r.date
        assert r.quarter_cutoff==max(d for d in quarters if d<r.date)
    assert len(dates)==68 and len(route)==272
    assert dates==[f'{y}-{m:02d}-{calendar.monthrange(y,m)[1]}' for y in range(2020,2027) for m in range(1,13) if (2020,12)<=(y,m)<=(2026,7)]
    members=csv('split_membership');oldmembers=csv('quarterly_members');schedule=csv('schedule').set_index('cutoff');oldschedule=csv('quarterly_schedule').set_index('cutoff')
    heads=read(OUT/'correction_heads.json');oldheads=read(OUT/'quarterly_heads.json');hi={(r['cutoff'],r['method'],r['seed'],r['family'],r['component']):r for r in heads};gates=csv('gate_decisions');oldgates=csv('quarterly_gates');gi=gates.set_index(['cutoff','method','family','component']);sharedheads=0
    for cutoff in quarters:
        for key in oldschedule.columns:eq(oldschedule.loc[cutoff,key],schedule.loc[cutoff,key])
        a=members[members.cutoff.eq(cutoff)].reset_index(drop=True);b=oldmembers[oldmembers.cutoff.eq(cutoff)].reset_index(drop=True);pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
    for r in oldheads:
        if r['family']!='state':continue
        x=hi[(r['cutoff'],r['method'],r['seed'],r['family'],r['component'])]
        assert set(x)==set(r)
        for key in r:eq(r[key],x[key])
        sharedheads+=1
    for r in oldgates[oldgates.family.eq('state')].to_dict('records'):
        x=gi.loc[(r['cutoff'],r['method'],r['family'],r['component'])]
        for key in x.index:eq(r[key],x[key])
    assert sharedheads==1104
    # Q1 is an exactannual fallback, including the extra January and February decisions.
    models=csv('model_predictions');mi=models.set_index(['history','method','seed','row_index']);base=csv('baseline_model_predictions').set_index(['history','method','seed','row_index']);firstmonth=0;q1=0
    ri=route.set_index('row_index');effects=csv('seed_routing')
    for r in effects.itertuples():
        actual=mi.loc[(r.history,r.method,r.seed,r.row_index)];annual=base.loc[(ANNUAL,r.method,r.seed,r.row_index)]
        assert actual.cutoff==r.cutoff==ri.loc[r.row_index,'head_cutoff']
        if int(r.date[5:7])<=3:
            assert r.offset==0 and actual.probability==annual.probability and not r.validation_accepted;q1+=1
        if r.cutoff==ri.loc[r.row_index,'quarter_cutoff']:
            quarterly=base.loc[(QUARTER[r.history],r.method,r.seed,r.row_index)]
            assert actual.probability==quarterly.probability and actual.direction_up==quarterly.direction_up;firstmonth+=1
    e=csv('ensemble_predictions').set_index(['history','method','row_index']);comp=csv('cadence_weekly_comparison')
    assert len(comp)==2176 and not comp.duplicated(['history','method','row_index']).any()
    for r in comp.itertuples():
        m=e.loc[(r.history,r.method,r.row_index)];q=e.loc[(QUARTER[r.history],r.method,r.row_index)];a=e.loc[(ANNUAL,r.method,r.row_index)];ma=m.probability!=a.probability;qa=q.probability!=a.probability;mc=m.direction_up==m.actual_up;qc=q.direction_up==q.actual_up
        assert r.monthly_probability==m.probability and r.quarterly_probability==q.probability and r.annual_probability==a.probability and r.actual_up==a.actual_up
        assert r.monthly_active==ma and r.quarterly_active==qa and r.changed_vs_quarter==(m.probability!=q.probability)
        assert r.monthly_correct==mc and r.quarterly_correct==qc
        assert r.activation_relation==('both_active' if ma and qa else 'monthly_only' if ma else 'quarterly_only' if qa else 'neither')
        assert r.case==('recovery' if mc and not qc else 'regression' if qc and not mc else 'stable_correct' if mc else 'stable_wrong')
        eq(r.brier_difference,(m.probability-m.actual_up)**2-(q.probability-q.actual_up)**2)
    summaries=csv('cadence_summary');assert len(summaries)==360
    for r in summaries.itertuples():
        period=next(p for p in periods() if p['name']==r.period);g=comp[comp.history.eq(r.history)&comp.method.eq(r.method)&comp.date.between(period['start'],period['end'])];g=g if r.relation=='all' else g[g.activation_relation.eq(r.relation)];assert r.n==len(g)
        for key,col in [('changed_vs_quarter','changed_vs_quarter'),('monthly_active_weeks','monthly_active'),('quarterly_active_weeks','quarterly_active'),('monthly_correct','monthly_correct'),('quarterly_correct','quarterly_correct')]:assert getattr(r,key)==sum(bool(v) for v in g[col])
        assert r.recoveries==sum(v=='recovery' for v in g['case']) and r.regressions==sum(v=='regression' for v in g['case'])
        eq(r.brier_difference,math.fsum(g.brier_difference)/len(g) if len(g) else None)
    tr=csv('gate_transitions');assert len(tr)==1088;oldgi=oldgates.set_index(['cutoff','method','family','component'])
    for r in tr.itertuples():
        i=dates.index(r.cutoff);g=gi.loc[(r.cutoff,r.method,'state',r.state)];oldcut=max(d for d in quarters if d<=r.cutoff);before=bool(gi.loc[(dates[i-1],r.method,'state',r.state),'accepted']) if i else False
        eq(r.previous_cutoff,dates[i-1] if i else None)
        assert r.accepted==g.accepted and r.previous_accepted==before and r.quarterly_cutoff==oldcut and r.quarterly_accepted==oldgi.loc[(oldcut,r.method,'state',r.state),'accepted'] and r.extra_month==(r.cutoff not in quarters)
        assert r.transition==('enabled' if g.accepted and not before else 'stopped' if before and not g.accepted else 'retained_on' if g.accepted else 'retained_off')
        assert r.reason==g.reason and r.mode==g['mode'] and r.training_n==g.training_n and r.validation_n==g.validation_n
        future=comp[comp.history.eq(POLICIES['state_gated'])&comp.method.eq(r.method)&comp.state.eq(r.state)&comp.monthly_cutoff.eq(r.cutoff)]
        assert r.future_n==len(future) and r.future_recoveries_vs_quarter==sum(z=='recovery' for z in future['case']) and r.future_regressions_vs_quarter==sum(z=='regression' for z in future['case'])
        eq(r.future_brier_vs_quarter,math.fsum(future.brier_difference)/len(future) if len(future) else None)
        delta=[(z.monthly_probability-z.actual_up)**2-(z.annual_probability-z.actual_up)**2 for z in future.itertuples()];eq(r.future_brier_vs_annual,math.fsum(delta)/len(delta) if delta else None)
    print('Shared quarterly candidates, Q1fallback, first-month equality and cadence diagnostics PASS.',flush=True)
    return dict(shared_quarter_head_cells=sharedheads,shared_quarter_gate_cells=int(oldgates.family.eq('state').sum()),exact_first_month_seed_predictions=firstmonth,q1_exact_annual_seed_predictions=q1,cadence_weekly_cases=len(comp),cadence_summary_cells=len(summaries),gate_transition_cells=len(tr))

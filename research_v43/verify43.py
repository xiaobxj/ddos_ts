from common43 import *
import math,calendar

def equal(a,b):
    if a is None or pd.isna(a): assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(float(a)-float(b))<2e-12,(a,b)
def classify(n,bd,cd,available):
    if not available:return 'unfitted'
    if not n:return 'no_samples'
    if n<5:return 'insufficient_validation'
    if bd>=-1e-12:return 'brier_not_better'
    return 'direction_worse' if cd<0 else 'pass'
def main():
    prep=check_frozen();check_phase('diagnosis');assert not (OUT/'verification.json').exists();started=now()
    subjects=csv('subjects');bank=csv('weekly_signal_bank');members=csv('split_membership');heads=read(OUT/'correction_heads.json');gates=csv('gate_decisions');dates=csv('schedule').cutoff.tolist()
    sd=csv('validation_seed_diagnostics');wd=csv('validation_week_diagnostics');cells=csv('diagnostic_cells');cohorts=csv('diagnostic_summary')
    si=subjects.set_index('subject_id');bi=bank.set_index(['method','seed','row_index']);hi={(h['cutoff'],h['method'],h['component'],h['seed']):h for h in heads}
    gi=gates.set_index(['cutoff','method','component']);state=bank[['row_index','state']].drop_duplicates().set_index('row_index').state
    # Separate chronological state machine, distinct from contract's last-acceptance prefix search.
    records={}; checked=0
    for date in dates:
        for method in METHODS:
            for st in STATES:
                key=(method,st);old=records.get(key);r=si.loc[f'{date}:{method}:{st}'];g=gi.loc[(date,method,st)]
                equal(r.source_cutoff,old[0] if old else None);equal(r.expiry,old[1] if old else None)
                status='absent' if old is None else 'annual_mismatch' if old[0][:4]!=date[:4] else 'expired' if date>=old[1] else 'in_life'
                assert r.availability==status
                if g.accepted:
                    y,m=int(date[:4]),int(date[5:7]); q=3*((m-1)//3+1)
                    if m==q:q+=3
                    if q>12:y+=1;q-=12
                    records[key]=(date,f'{y:04d}-{q:02d}-{calendar.monthrange(y,q)[1]:02d}')
                elif old and date<old[1] and old[0][:4]==date[:4] and g['mode']=='ready' and g.reason=='insufficient_validation' and g.training_n>=10 and g.validation_n<5:
                    pass
                else:records.pop(key,None)
                checked+=1
    assert checked==1088 and len(cells)==3264
    # Independently enumerate exact validation membership and seed Cartesian product.
    expected_keys=set();val_by={c:set(g.row_index) for c,g in members[members.role.eq('validation')].groupby('cutoff')}
    for r in subjects.itertuples():
        if r.availability not in ['in_life','expired']:continue
        ids={i for i in val_by.get(r.cutoff,set()) if state[i]==r.state}
        assert len(ids)==r.current_validation_n
        for idx in ids:
            for seed in cfg()['seeds']:expected_keys.add((r.subject_id,seed,int(idx)))
    assert len(sd)==len(expected_keys) and set(zip(sd.subject_id,sd.seed,sd.row_index))==expected_keys
    maxgap=0.
    for x in sd.itertuples():
        r=si.loc[x.subject_id];b=bi.loc[(r.method,x.seed,x.row_index)];old=hi[(r.source_cutoff,r.method,r.state,x.seed)];cur=hi[(r.cutoff,r.method,r.state,x.seed)]
        assert b.joint_completed<=r.cutoff and old['fit_eligible']
        assert x.date==b.date and x.joint_completed==b.joint_completed and x.annual_encoder_cutoff==b.encoder_cutoff and x.actual_up==b.actual_up
        assert x.matured_after_source==(b.joint_completed>r.source_cutoff)
        assert x.matured_since_previous_decision==(b.joint_completed>r.previous_decision_cutoff)
        assert x.in_source_validation==(x.row_index in val_by.get(r.source_cutoff,set()))
        assert not (x.matured_after_source and x.in_source_validation)
        equal(x.annual_probability,b.probability);equal(x.old_offset,old['offset'])
        for role,h in [('old',old),('current',cur)]:
            if role=='current' and not r.current_fit_eligible:
                assert pd.isna(x.current_offset) and pd.isna(x.current_probability);continue
            value=b.probability if h['offset']==0 else 1/(1+math.exp(-float(b.annual_logit)-h['offset']))
            gap=abs(value-getattr(x,role+'_probability'));maxgap=max(maxgap,gap);assert gap<2e-15
    groups={k:g for k,g in sd.groupby(['subject_id','row_index'])}
    assert len(wd)==len(groups) and len(sd)==3*len(wd)
    for x in wd.itertuples():
        g=groups[(x.subject_id,x.row_index)].sort_values('seed');assert g.seed.tolist()==cfg()['seeds']
        p0=math.fsum(g.annual_probability)/3;equal(p0,x.annual_probability)
        for role in ['old','current']:
            if role=='current' and not si.loc[x.subject_id,'current_fit_eligible']:
                assert pd.isna(x.current_probability) and pd.isna(x.current_brier_difference) and pd.isna(x.current_correct_difference);continue
            p=math.fsum(g[role+'_probability'])/3;equal(p,getattr(x,role+'_probability'))
            equal((p-x.actual_up)**2-(p0-x.actual_up)**2,getattr(x,role+'_brier_difference'))
            assert int((p>.5)==x.actual_up)-int((p0>.5)==x.actual_up)==getattr(x,role+'_correct_difference')
    wg={k:g for k,g in wd.groupby('subject_id')};empty=wd.iloc[:0]
    for c in cells.itertuples():
        r=si.loc[c.subject_id];g=wg.get(c.subject_id,empty)
        if c.view=='post_source_validation':g=g[g.joint_completed.gt(c.source_cutoff)] if len(g) else g
        elif c.view=='incremental_validation':g=g[g.joint_completed.gt(c.previous_decision_cutoff)] if len(g) else g
        assert len(g)==c.n and c.known_at_source_n==int((~g.matured_after_source).sum()) and c.source_validation_overlap_n==int(g.in_source_validation.sum()) and c.new_since_previous_n==int(g.matured_since_previous_decision.sum())
        for role in ['old','current']:
            available=role=='old' or c.current_fit_eligible
            bd=math.fsum(g[role+'_brier_difference'])/len(g) if len(g) and available else None
            cd=int(math.fsum(g[role+'_correct_difference'])) if available else None
            equal(bd,getattr(c,role+'_brier_difference'));equal(cd,getattr(c,role+'_correct_difference'))
            expected='no_saved_record' if c.availability=='absent' else 'annual_mismatch' if c.availability=='annual_mismatch' else classify(len(g),bd,cd,available)
            assert getattr(c,role+'_evidence')==expected
        old,cur=c.old_evidence,c.current_evidence
        expected='not_assessed' if old in ['no_saved_record','annual_mismatch'] else 'current_unfitted' if cur=='unfitted' else 'insufficient_evidence' if old in ['no_samples','insufficient_validation'] else ('both_pass' if old=='pass' else 'current_only_pass') if cur=='pass' else 'old_only_pass' if old=='pass' else 'both_reject'
        assert c.comparison==expected
        if c.view=='full_validation' and c.availability in ['in_life','expired'] and c.current_fit_eligible:
            original=gi.loc[(c.cutoff,c.method,c.state)]
            equal(c.current_brier_difference,original.brier_difference)
            assert c.current_correct_difference==original.candidate_correct-original.baseline_correct
            assert c.current_evidence==('pass' if original.accepted else 'no_samples' if original.validation_n==0 else original.reason)
        if c.view!='full_validation':assert c.known_at_source_n==0 and c.source_validation_overlap_n==0
    assert not cells[cells.n.lt(5)].old_evidence.eq('pass').any()
    # Summaries checked by record sets and unique source/week identities; repeated uses are not independent N.
    periods={p:(a,b) for p,a,b in period_definitions()}
    expected_summary_keys=set()
    for p,(start,end) in periods.items():
        for k,_ in cells[cells.cutoff.between(start,end)&cells.availability.isin(['in_life','expired'])].groupby(['method','availability','view']):expected_summary_keys.add((p,)+k)
    assert len(cohorts)==len(expected_summary_keys) and set(zip(cohorts.period,cohorts.method,cohorts.availability,cohorts.view))==expected_summary_keys
    for r in cohorts.itertuples():
        start,end=periods[r.period];g=cells[cells.cutoff.between(start,end)&cells.method.eq(r.method)&cells.availability.eq(r.availability)&cells.view.eq(r.view)];w=wd[wd.subject_id.isin(g.subject_id)]
        if r.view=='post_source_validation':w=w[w.matured_after_source]
        elif r.view=='incremental_validation':w=w[w.matured_since_previous_decision]
        pairs=set(zip(w.source_cutoff,w.row_index))
        assert [r.cells,r.validation_uses,r.unique_calendar_weeks,r.unique_source_week_pairs,r.repeated_source_week_uses]==[len(g),len(w),len(set(w.row_index)),len(pairs),len(w)-len(pairs)]
        for label in ['pass','brier_not_better','direction_worse','no_samples','insufficient_validation']:assert getattr(r,'old_'+label)==sum(g.old_evidence==label)
        for label in ['both_pass','old_only_pass','current_only_pass','both_reject','insufficient_evidence','current_unfitted']:assert getattr(r,label)==sum(g.comparison==label)
        q=g[g.original_reason.isin(['brier_not_better','direction_worse'])]
        assert [r.quality_clear_cells,r.quality_clear_old_pass,r.quality_clear_old_reject,r.quality_clear_insufficient]==[len(q),sum(q.old_evidence=='pass'),sum(q.old_evidence.isin(['brier_not_better','direction_worse'])),sum(q.old_evidence.isin(['no_samples','insufficient_validation']))]
    actual=csv('quality_clear_diagnostics');wanted=cells[cells.availability.eq('in_life')&cells.original_reason.isin(['brier_not_better','direction_worse'])]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True),wanted.reset_index(drop=True),check_dtype=False)
    print('Independent incoming records, seed math, evidence views, zero-sample handling and summaries PASS.',flush=True)
    # Change all as-yet immature outcomes/probabilities and all later coefficients; every current diagnostic must be invariant.
    from diagnose43 import diagnose
    isolation=0
    for date in dates:
        sub=subjects[subjects.cutoff.eq(date)];poison=bank.copy();future=poison.joint_completed.gt(date)
        poison.loc[future,'actual_up']=1-poison.loc[future,'actual_up'];poison.loc[future,'probability']=.999;poison.loc[future,'annual_logit']=12.
        ph=[dict(h,offset=-99.) if h['cutoff']>date else h for h in heads]
        a,b,c=diagnose(sub,poison,members,ph)
        for fresh,original in [(a,sd),(b,wd),(c,cells)]:
            expected=original[original.subject_id.isin(sub.subject_id)].reset_index(drop=True)
            if len(expected)==0:assert len(fresh)==0;continue
            fresh=fresh.reset_index(drop=True).copy()
            for name in expected.columns:
                if fresh[name].isna().all() and expected[name].isna().all():
                    fresh[name]=np.nan;expected[name]=np.nan
            pd.testing.assert_frame_equal(fresh.reset_index(drop=True),expected,check_dtype=False,check_exact=False,rtol=0,atol=2e-12)
        isolation+=1
    for name in ['model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv']:
        assert sha(OUT/name)==sha(PREV/'results'/name)
    ensemble=csv('ensemble_predictions');assert ensemble.history.nunique()==23 and len(ensemble)==37536
    assert len(csv('model_predictions'))==100096 and old_evidence()==prep['old_evidence']
    check_frozen();check_phase('diagnosis')
    save(OUT/'verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5754,unchanged_forecast_histories=23,new_fits=0,new_policies=0,new_pvalues=0,independent_incoming_records=1088,independent_seed_rows=len(sd),independent_week_rows=len(wd),independent_evidence_cells=len(cells),independent_summary_cells=len(cohorts),future_data_and_head_poison_checks=isolation,maximum_seed_probability_gap=maxgap,artifacts={}))
    print(f'R43 independent verification PASS, including {isolation} future-data isolation checks and 5754 unchanged old files.',flush=True)
if __name__=='__main__':main()

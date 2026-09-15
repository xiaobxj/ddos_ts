from common44 import *
import calendar

def eq(a,b):
    if a is None or pd.isna(a):assert b is None or pd.isna(b),(a,b)
    elif isinstance(a,(str,bool)):assert a==b,(a,b)
    else:assert abs(float(a)-float(b))<1e-12,(a,b)
def months(source,end):
    y,m=int(source[:4]),int(source[5:7]);r=[]
    while True:
        m+=1
        if m==13:y+=1;m=1
        d=f'{y:04d}-{m:02d}-{calendar.monthrange(y,m)[1]:02d}'
        if d>end:break
        r.append(d)
    return r
def endquarter(source):
    y=int(source[:4]);candidates=[f'{y}-{m:02d}-{calendar.monthrange(y,m)[1]:02d}' for m in [3,6,9,12]]+[f'{y+1}-03-31']
    return next(x for x in candidates if x>source)
def main():
    prep=check_frozen();check_phase('diagnosis');assert not (OUT/'verification.json').exists();start=now()
    origins=csv('origins');meta=csv('weekly_metadata');gates=csv('gate_decisions');bank=csv('weekly_signal_bank');checks=csv('checkpoints');members=csv('checkpoint_members');arrivals=csv('observed_arrivals');tracks=csv('origin_feasibility');summary=csv('feasibility_summary');events=csv('event_feasibility');end=cfg()['label_end']
    keys={(r.cutoff,r.method,r.component) for r in gates[gates.accepted].itertuples()};assert len(keys)==84
    assert len(origins)==84 and set(zip(origins.source_cutoff,origins.method,origins.state))==keys
    assert set(map(tuple,meta[META].itertuples(index=False,name=None)))==set(map(tuple,bank[META].itertuples(index=False,name=None))) and len(meta)==272
    pool=list(meta.itertuples(index=False));ci=checks.set_index(['origin_id','checkpoint']);ti=tracks.set_index(['origin_id','view']);expected_members=set();expected_checkpoints=set();expected_arrivals=set();count_cells=0
    assert len(tracks)==168 and not tracks[['origin_id','view']].duplicated().any()
    for o in origins.itertuples():
        expiry=endquarter(o.source_cutoff);assert o.expiry==expiry and o.event_id==f'{o.source_cutoff}:{o.state}'
        dates=months(o.source_cutoff,expiry);assert o.planned_checkpoints==len(dates) and o.planned_preexpiry_checkpoints==sum(d<expiry for d in dates)
        assert o.horizon_days==(pd.Timestamp(expiry)-pd.Timestamp(o.source_cutoff)).days
        available_arrivals=sorted([p for p in pool if o.source_cutoff<p.joint_completed<=min(expiry,end) and p.state==o.state and p.year==o.source_year],key=lambda p:(p.joint_completed,p.date,p.row_index))
        for rank,p in enumerate(available_arrivals,1):
            expected_arrivals.add((o.origin_id,p.row_index,rank));a=arrivals[arrivals.origin_id.eq(o.origin_id)&arrivals.row_index.eq(p.row_index)].iloc[0]
            assert a.arrival_rank==rank and a.date==p.date and a.joint_completed==p.joint_completed and a.signal_precedes_or_equals_source==(p.date<=o.source_cutoff) and a.maturity_age_days==(pd.Timestamp(p.joint_completed)-pd.Timestamp(o.source_cutoff)).days
        independent=[]
        for d in dates:
            expected_checkpoints.add((o.origin_id,d));r=ci.loc[(o.origin_id,d)];observed=d<=end
            assert r.observed==observed and r.preexpiry==(d<expiry) and r.metadata_only_extended_cutoff==(d>cfg()['last_original_candidate_cutoff'])
            if not observed:
                assert pd.isna(r.cumulative_n) and pd.isna(r.rolling13_n) and pd.isna(r.dropped_from_rolling13);continue
            past=[p for p in pool if p.joint_completed<=d];last13={p.row_index for p in sorted(past,key=lambda p:(p.date,p.row_index))[-13:]}
            selected=[p for p in past if p.joint_completed>o.source_cutoff and p.state==o.state and p.year==o.source_year]
            cumulative=len(selected);rolling=sum(p.row_index in last13 for p in selected)
            assert [r.cumulative_n,r.rolling13_n,r.dropped_from_rolling13]==[cumulative,rolling,cumulative-rolling]
            for p in selected:
                expected_members.add((o.origin_id,d,p.row_index));m=members[members.origin_id.eq(o.origin_id)&members.checkpoint.eq(d)&members.row_index.eq(p.row_index)].iloc[0]
                assert m.date==p.date and m.joint_completed==p.joint_completed and m.in_rolling13==(p.row_index in last13)
            independent.append((d,cumulative,rolling));count_cells+=2
        for vi,view in enumerate(VIEWS,1):
            t=ti.loc[(o.origin_id,view)];hits=[x[0] for x in independent if x[vi]>=5];early=[d for d in hits if d<expiry]
            label='reached_before_expiry' if early else 'observation_censored' if expiry>end else 'reached_only_at_expiry' if hits else 'not_reached_by_expiry'
            assert t.classification==label and t.horizon_complete==(expiry<=end)
            first=early[0] if early else None;eq(t.first_ready_decision,hits[0] if hits else None);eq(t.first_preexpiry_ready_decision,first)
            eq(t.fifth_maturity_date,available_arrivals[4].joint_completed if vi==1 and len(available_arrivals)>=5 else None)
            eq(t.days_until_preexpiry_ready,(pd.Timestamp(first)-pd.Timestamp(o.source_cutoff)).days if first else None);eq(t.days_remaining_at_preexpiry_ready,(pd.Timestamp(expiry)-pd.Timestamp(first)).days if first else None)
            assert t.observed_checkpoints==len(independent) and t.observed_preexpiry_checkpoints==sum(x[0]<expiry for x in independent)
            eq(t.last_observed_checkpoint,independent[-1][0] if independent else None);eq(t.last_observed_n,independent[-1][vi] if independent else None)
            eq(t.expiry_n,next((x[vi] for x in independent if x[0]==expiry),None))
            future=[p for p in pool if first and first<p.date<=min(expiry,end)]
            assert t.observed_followup_all_state_weeks==len(future) and t.observed_followup_same_state_weeks==sum(p.state==o.state for p in future) and t.followup_observation_censored==bool(first and expiry>end)
    assert len(checks)==len(expected_checkpoints) and set(zip(checks.origin_id,checks.checkpoint))==expected_checkpoints
    assert len(members)==len(expected_members) and set(zip(members.origin_id,members.checkpoint,members.row_index))==expected_members
    assert len(arrivals)==len(expected_arrivals) and set(zip(arrivals.origin_id,arrivals.row_index,arrivals.arrival_rank))==expected_arrivals
    periods={p:(a,b) for p,a,b in period_definitions()};sk=set()
    for p,(a,b) in periods.items():
        for k,_ in tracks[tracks.source_cutoff.between(a,b)].groupby(['method','view']):sk.add((p,)+k)
    assert len(summary)==len(sk) and set(zip(summary.period,summary.method,summary.view))==sk
    for r in summary.itertuples():
        a,b=periods[r.period];g=tracks[tracks.method.eq(r.method)&tracks.view.eq(r.view)&tracks.source_cutoff.between(a,b)];complete=g[g.horizon_complete];hit=g[g.classification.eq('reached_before_expiry')]
        assert [r.origins,r.complete_origins,r.incomplete_origins,r.complete_reached_before_expiry,r.origins_with_no_preexpiry_checkpoint,r.unique_source_state_events,r.ready_with_observed_same_state_followup,r.ready_with_no_observed_same_state_followup]==[len(g),len(complete),len(g)-len(complete),sum(complete.classification=='reached_before_expiry'),sum(g.planned_preexpiry_checkpoints==0),g.event_id.nunique(),sum(hit.observed_followup_same_state_weeks>0),sum(hit.observed_followup_same_state_weeks==0)]
        for label in CLASSES:assert getattr(r,label)==sum(g.classification==label)
        assert sum(getattr(r,label) for label in CLASSES)==r.origins
    ek=set(zip(tracks.event_id,tracks.view));assert len(events)==len(ek) and set(zip(events.event_id,events.view))==ek
    for r in events.itertuples():
        g=tracks[tracks.event_id.eq(r.event_id)&tracks.view.eq(r.view)];assert r.method_count==len(g) and r.methods==';'.join(sorted(g.method))
        for col in ['classification','first_preexpiry_ready_decision','horizon_complete','expiry_n','observed_followup_same_state_weeks','planned_preexpiry_checkpoints']:
            for value in g[col]:eq(value,getattr(r,col))
    print('Independent origin calendar, every count/member, first readiness, censoring and aggregate checks PASS.',flush=True)
    from diagnose44 import count_tracks
    poisoned=bank.copy()
    for col in ['probability','annual_probability','annual_logit','score','actual','actual_up','direction_up']:
        if col in poisoned:poisoned[col]=-987654321
    poisonmeta=canonical_metadata(poisoned);pd.testing.assert_frame_equal(poisonmeta,meta,check_dtype=False)
    outcome_checks=0
    for fresh,old in zip(count_tracks(origins,poisonmeta,end),[checks,members,arrivals,tracks]):
        fresh=fresh.copy();old=old.copy()
        for col in old.columns:
            if old[col].isna().all() and fresh[col].isna().all():old[col]=np.nan;fresh[col]=np.nan
        pd.testing.assert_frame_equal(fresh.reset_index(drop=True),old.reset_index(drop=True),check_dtype=False,check_exact=False,rtol=0,atol=1e-12);outcome_checks+=1
    isolation=0;prefix_origins=0
    for d in sorted(checks.loc[checks.observed,'checkpoint'].unique()):
        # Future gates can neither create prior origins nor reset their original acceptance timestamps.
        pg=gates.copy();later=pg.cutoff.ge(d);pg.loc[later,'accepted']=~pg.loc[later,'accepted'];pg.loc[~pg.accepted,'brier_difference']=12345.
        actual={(r.cutoff,r.method,r.component) for r in gates[gates.accepted&gates.cutoff.lt(d)].itertuples()}
        altered={(r.cutoff,r.method,r.component) for r in pg[pg.accepted&pg.cutoff.lt(d)].itertuples()};assert altered==actual;prefix_origins+=1
        truncated=meta[meta.joint_completed.le(d)]
        for r in checks[checks.checkpoint.eq(d)].itertuples():
            a=count_members(truncated,r.source_cutoff,r.state,d);expected=members[members.origin_id.eq(r.origin_id)&members.checkpoint.eq(d)]
            assert set(a.row_index)==set(expected.row_index) and set(a.loc[a.in_rolling13,'row_index'])==set(expected.loc[expected.in_rolling13,'row_index']);isolation+=1
    for n in ['model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv']:assert sha(OUT/n)==sha(PREV/'results'/n)
    assert len(csv('model_predictions'))==100096 and len(csv('ensemble_predictions'))==37536 and csv('ensemble_predictions').history.nunique()==23
    assert old_evidence()==prep['old_evidence'];check_frozen();check_phase('diagnosis')
    save(OUT/'verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5800,unchanged_forecast_histories=23,new_fits=0,new_policies=0,new_pvalues=0,independent_origins=84,independent_planned_checkpoints=len(checks),independent_count_cells=count_cells,independent_checkpoint_members=len(members),independent_arrivals=len(arrivals),independent_origin_view_cells=len(tracks),independent_summary_cells=len(summary),independent_event_view_cells=len(events),outcome_poison_output_tables=outcome_checks,future_metadata_isolation_checkpoints=isolation,future_gate_prefix_dates=prefix_origins,artifacts={}))
    print('R44 independent verification PASS; future information and predictive outcomes cannot drive counts. Old evidence unchanged.',flush=True)
if __name__=='__main__':main()

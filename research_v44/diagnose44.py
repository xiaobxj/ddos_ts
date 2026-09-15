from common44 import *

def count_tracks(origins,meta,end):
    checks=[];members=[];arrivals=[];tracks=[]
    for o in origins.itertuples(index=False):
        info=o._asdict();limit=min(o.expiry,end);arr=count_members(meta,o.source_cutoff,o.state,limit)
        for rank,a in enumerate(arr.itertuples(),1):
            arrivals.append(dict(origin_id=o.origin_id,event_id=o.event_id,source_cutoff=o.source_cutoff,method=o.method,state=o.state,row_index=int(a.row_index),date=a.date,joint_completed=a.joint_completed,arrival_rank=rank,signal_precedes_or_equals_source=a.date<=o.source_cutoff,maturity_age_days=int((pd.Timestamp(a.joint_completed)-pd.Timestamp(o.source_cutoff)).days)))
        fifth=arr.joint_completed.iloc[4] if len(arr)>=5 else None;local=[]
        for d in checkpoints(o.source_cutoff,o.expiry):
            observed=d<=end
            selected=count_members(meta,o.source_cutoff,o.state,d) if observed else meta.iloc[:0]
            r=dict(origin_id=o.origin_id,event_id=o.event_id,source_cutoff=o.source_cutoff,method=o.method,state=o.state,expiry=o.expiry,checkpoint=d,observed=observed,preexpiry=d<o.expiry,metadata_only_extended_cutoff=d>cfg()['last_original_candidate_cutoff'],cumulative_n=len(selected) if observed else None,rolling13_n=int(selected.in_rolling13.sum()) if observed else None,dropped_from_rolling13=int((~selected.in_rolling13).sum()) if observed else None)
            for a in selected.itertuples():members.append(dict(origin_id=o.origin_id,checkpoint=d,row_index=int(a.row_index),date=a.date,joint_completed=a.joint_completed,in_rolling13=bool(a.in_rolling13)))
            checks.append(r);local.append(r)
        for view in VIEWS:
            label,first,anyfirst=classify_track(local,o.expiry,end,view);key='cumulative_n' if view=='cumulative' else 'rolling13_n';known=[r for r in local if r['observed']]
            terminal=next((r[key] for r in known if r['checkpoint']==o.expiry),None)
            following=meta[meta.date.gt(first)&meta.date.le(limit)] if first else meta.iloc[:0]
            tracks.append(dict(**info,view=view,classification=label,first_ready_decision=anyfirst,first_preexpiry_ready_decision=first,fifth_maturity_date=fifth if view=='cumulative' else None,days_until_preexpiry_ready=int((pd.Timestamp(first)-pd.Timestamp(o.source_cutoff)).days) if first else None,days_remaining_at_preexpiry_ready=int((pd.Timestamp(o.expiry)-pd.Timestamp(first)).days) if first else None,horizon_complete=o.expiry<=end,observed_checkpoints=len(known),observed_preexpiry_checkpoints=sum(r['preexpiry'] for r in known),last_observed_checkpoint=known[-1]['checkpoint'] if known else None,last_observed_n=known[-1][key] if known else None,expiry_n=terminal,observed_followup_all_state_weeks=len(following),observed_followup_same_state_weeks=int(following.state.eq(o.state).sum()),followup_observation_censored=bool(first and o.expiry>end)))
    return pd.DataFrame(checks),pd.DataFrame(members),pd.DataFrame(arrivals),pd.DataFrame(tracks)

def summaries(tracks):
    rows=[]
    for p,start,end in period_definitions():
        for (method,view),g in tracks[tracks.source_cutoff.between(start,end)].groupby(['method','view']):
            complete=g[g.horizon_complete];hit=g[g.classification.eq('reached_before_expiry')]
            r=dict(period=p,method=method,view=view,origins=len(g),complete_origins=len(complete),incomplete_origins=len(g)-len(complete),complete_reached_before_expiry=int(complete.classification.eq('reached_before_expiry').sum()),origins_with_no_preexpiry_checkpoint=int(g.planned_preexpiry_checkpoints.eq(0).sum()),unique_source_state_events=g.event_id.nunique(),ready_with_observed_same_state_followup=int(hit.observed_followup_same_state_weeks.gt(0).sum()),ready_with_no_observed_same_state_followup=int(hit.observed_followup_same_state_weeks.eq(0).sum()))
            for label in CLASSES:r[label]=int(g.classification.eq(label).sum())
            rows.append(r)
    return pd.DataFrame(rows)

def event_table(tracks):
    rows=[]
    for (event,view),g in tracks.groupby(['event_id','view']):
        for col in ['classification','first_preexpiry_ready_decision','first_ready_decision','horizon_complete','last_observed_n','expiry_n','observed_followup_same_state_weeks']:assert g[col].fillna('UNDEFINED').nunique()==1
        a=g.iloc[0];rows.append(dict(event_id=event,source_cutoff=a.source_cutoff,state=a.state,expiry=a.expiry,view=view,methods=';'.join(sorted(g.method)),method_count=len(g),planned_preexpiry_checkpoints=int(a.planned_preexpiry_checkpoints),classification=a.classification,first_preexpiry_ready_decision=a.first_preexpiry_ready_decision,horizon_complete=bool(a.horizon_complete),expiry_n=a.expiry_n,observed_followup_same_state_weeks=int(a.observed_followup_same_state_weeks)))
    return pd.DataFrame(rows)

def main():
    check_frozen();assert not (OUT/'diagnosis_manifest.json').exists();run=manifest('diagnosis')
    checks,members,arrivals,tracks=count_tracks(csv('origins'),csv('weekly_metadata'),cfg()['label_end']);summary=summaries(tracks);events=event_table(tracks);files=[]
    for name,g in [('checkpoints',checks),('checkpoint_members',members),('observed_arrivals',arrivals),('origin_feasibility',tracks),('feasibility_summary',summary),('event_feasibility',events)]:
        p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,origins=84,planned_checkpoints=len(checks),observed_checkpoints=int(checks.observed.sum()),checkpoint_members=len(members),observed_origin_week_arrivals=len(arrivals),origin_view_cells=len(tracks),summary_cells=len(summary),unique_event_view_cells=len(events),new_fits=0,new_policies=0,new_pvalues=0)
    print(f'Feasibility counting complete:84 fixed origins, {len(checks)} planned checkpoints, two fixed views. No predictive outcomes evaluated.',flush=True)
if __name__=='__main__':main()

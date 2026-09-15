from common41 import *

def sensitivity(cells,effects):
    grouped={k:g for k,g in effects.groupby(['cadence','cutoff','method','state'])};summaries=[];leave=[]
    for c in cells.itertuples():
        g=grouped.get((c.cadence,c.cutoff,c.method,c.state),effects.iloc[:0]);n=len(g);bd=g.brier_difference.to_numpy();cd=g.correct_difference.to_numpy();cases=[]
        if c.metric_eligible:
            for r in g.itertuples():
                newbd=float((bd.sum()-r.brier_difference)/(n-1));newcd=int(cd.sum()-r.correct_difference);metric_accept=newbd < -1e-12 and newcd>=0;full,reason=cell_gate(c.mode,c.training_n,n-1,newbd,newcd)
                record=dict(cadence=c.cadence,cutoff=c.cutoff,method=c.method,state=c.state,removed_row_index=r.row_index,removed_date=r.date,original_accepted=c.accepted,remaining_n=n-1,removed_brier_difference=r.brier_difference,removed_correct_difference=r.correct_difference,brier_difference=newbd,correct_difference=newcd,metric_accepted=metric_accept,full_accepted=full,full_reason=reason,metric_flipped=metric_accept!=c.accepted,full_flipped=full!=c.accepted,mechanical_support_failure=n-1<5)
                cases.append(record);leave.append(record)
        flips=sum(r['metric_flipped'] for r in cases);fullflips=sum(r['full_flipped'] for r in cases);helpful=np.maximum(-bd,0.);positive_total=float(helpful.sum());share=float(helpful.max()/positive_total) if positive_total else None
        cls='unsupported' if not c.metric_eligible else 'accepted_metric_fragile' if c.accepted and flips else 'accepted_count_only_fragile' if c.accepted and n==5 else 'accepted_single_drop_stable' if c.accepted else 'rejected_metric_fragile' if flips else 'rejected_single_drop_stable'
        summaries.append(dict(**{k:v for k,v in c._asdict().items() if k!='Index'},loo_n=len(cases),metric_flip_count=flips,full_flip_count=fullflips,minimum_support_boundary=bool(c.metric_eligible and n==5),sensitivity_class=cls,helpful_brier_weeks=int((bd<0).sum()),harmful_brier_weeks=int((bd>0).sum()),largest_helpful_share=share))
    summary=pd.DataFrame(summaries);future=[]
    for cadence in CADENCES:
        g=csv(cadence+'_outcomes');g=g[g.family.eq('state')&g.history.eq(HISTORY[cadence][1])].rename(columns={'component':'state','n':'future_n','brier_difference':'future_brier_vs_annual','recoveries':'future_recoveries','regressions':'future_regressions'});g['cadence']=cadence;future.append(g[['cadence','cutoff','method','state','future_n','future_brier_vs_annual','future_recoveries','future_regressions']])
    summary=summary.merge(pd.concat(future,ignore_index=True),on=['cadence','cutoff','method','state'],validate='one_to_one');assert len(summary)==1456
    cohorts=[]
    for p in cfg()['summary_windows']:
        for (cadence,method),g in summary[summary.cutoff.between(p['start'],p['end'])].groupby(['cadence','method']):
            for label in ['all_supported','accepted_all','accepted_metric_fragile','accepted_count_only_fragile','accepted_single_drop_stable','rejected_metric_fragile','rejected_single_drop_stable']:
                z=g[g.metric_eligible] if label=='all_supported' else g[g.accepted] if label=='accepted_all' else g[g.sensitivity_class.eq(label)];nf=int(z.future_n.sum());defined=z[z.future_n.gt(0)]
                cohorts.append(dict(period=p['name'],cadence=cadence,method=method,cohort=label,cells=len(z),accepted_cells=int(z.accepted.sum()),metric_fragile_cells=int(z.metric_flip_count.gt(0).sum()),minimum_support_cells=int(z.minimum_support_boundary.sum()),loo_deletions=int(z.loo_n.sum()),metric_flips=int(z.metric_flip_count.sum()),future_nonempty_cells=len(defined),future_weeks=nf,future_worse_cells=int(defined.future_brier_vs_annual.gt(1e-12).sum()),future_better_cells=int(defined.future_brier_vs_annual.lt(-1e-12).sum()),future_week_weighted_brier=float((defined.future_brier_vs_annual*defined.future_n).sum()/nf) if nf else None,future_recoveries=int(z.future_recoveries.sum()),future_regressions=int(z.future_regressions.sum())))
    return summary,pd.DataFrame(leave),pd.DataFrame(cohorts)

def membership_flows():
    state=csv('weekly_signal_bank')[['row_index','state']].drop_duplicates().set_index('row_index').state;flows=[];rowflows=[]
    for cadence in CADENCES:
        schedule=csv(cadence+'_schedule');members=csv(cadence+'_members');sets={(c,role):g.set_index('row_index') for (c,role),g in members.groupby(['cutoff','role'])};empty=members.iloc[:0].set_index('row_index')
        for prev,current in zip(schedule.itertuples(),list(schedule.itertuples())[1:]):
            for role in ['training','validation']:
                aa=sets.get((prev.cutoff,role),empty);bb=sets.get((current.cutoff,role),empty)
                for st in ['all']+STATES:
                    a=aa if st=='all' else aa[np.array(aa.index.map(state))==st]
                    b=bb if st=='all' else bb[np.array(bb.index.map(state))==st]
                    old=set(map(int,a.index));new=set(map(int,b.index));shared=old&new;lost=old-new;added=new-old
                    for status,ids in [('retained',shared),('dropped',lost),('added',added)]:
                        for idx in sorted(ids):
                            r=b.loc[idx] if idx in new else a.loc[idx];rowflows.append(dict(cadence=cadence,previous_cutoff=prev.cutoff,cutoff=current.cutoff,role=role,state=st,row_index=idx,date=r.date,joint_completed=r.joint_completed,flow=status))
                    oldest=bool(not lost or len(bb) and all(a.loc[i,'date']<bb.date.min() for i in lost))
                    flows.append(dict(cadence=cadence,previous_cutoff=prev.cutoff,cutoff=current.cutoff,role=role,state=st,previous_mode=prev.mode,mode=current.mode,both_ready=prev.mode=='ready' and current.mode=='ready',previous_n=len(old),current_n=len(new),retained_n=len(shared),dropped_n=len(lost),added_n=len(added),retained_fraction_current=len(shared)/len(new) if new else None,jaccard=len(shared)/len(old|new) if old|new else None,dropped_older_than_current_window=oldest,support_dropout=st!='all' and role=='validation' and len(old)>=5 and len(new)<5))
    return pd.DataFrame(flows),pd.DataFrame(rowflows)

def transitions_and_runs(cells,effects,flows):
    ci=cells.set_index(['cadence','cutoff','method','state']);fi=flows.set_index(['cadence','cutoff','role','state']);records=[];runs=[];run_members=[]
    valsets={k:set(g.row_index.astype(int)) for k,g in effects.groupby(['cadence','cutoff','method','state'])}
    for cadence in CADENCES:
        dates=csv(cadence+'_schedule').cutoff.tolist()
        for method in LEARNED:
            for st in STATES:
                for before,after in zip(dates,dates[1:]):
                    a=ci.loc[(cadence,before,method,st)];b=ci.loc[(cadence,after,method,st)];f=fi.loc[(cadence,after,'validation',st)];transition='enabled' if b.accepted and not a.accepted else 'stopped' if a.accepted and not b.accepted else 'retained_on' if b.accepted else 'retained_off'
                    records.append(dict(cadence=cadence,method=method,state=st,previous_cutoff=before,cutoff=after,previous_accepted=bool(a.accepted),accepted=bool(b.accepted),previous_reason=a.reason,reason=b.reason,previous_mode=a['mode'],mode=b['mode'],both_ready=f.both_ready,transition=transition,previous_validation_n=f.previous_n,validation_n=f.current_n,retained_n=f.retained_n,dropped_n=f.dropped_n,added_n=f.added_n,retained_fraction_current=f.retained_fraction_current,support_dropout=f.support_dropout,stop_by_validation_support=transition=='stopped' and b.reason=='insufficient_validation',dropped_older_than_current_window=f.dropped_older_than_current_window))
                active=[]
                for d in dates+[None]:
                    if d is not None and ci.loc[(cadence,d,method,st),'accepted']:active.append(d);continue
                    if not active:continue
                    rid=f'{cadence}:{method}:{st}:{active[0]}';counts={};first={};last={}
                    for cutoff in active:
                        for idx in valsets[(cadence,cutoff,method,st)]:counts[idx]=counts.get(idx,0)+1;first.setdefault(idx,cutoff);last[idx]=cutoff
                    total=sum(counts.values());union=len(counts);runs.append(dict(run_id=rid,cadence=cadence,method=method,state=st,first_cutoff=active[0],last_cutoff=active[-1],decisions=len(active),total_validation_uses=total,unique_validation_weeks=union,repeated_uses=total-union,uses_per_unique_week=total/union,first_last_overlap=len(valsets[(cadence,active[0],method,st)]&valsets[(cadence,active[-1],method,st)])))
                    for idx,count in sorted(counts.items()):run_members.append(dict(run_id=rid,row_index=idx,uses=count,first_cutoff=first[idx],last_cutoff=last[idx]))
                    active=[]
    return pd.DataFrame(records),pd.DataFrame(runs),pd.DataFrame(run_members)

def main():
    check_frozen();assert not (OUT/'diagnosis_manifest.json').exists();run=manifest('diagnosis');cells=csv('gate_cells');effects=csv('validation_week_effects');summary,leave,cohorts=sensitivity(cells,effects);flows,rowflows=membership_flows();trans,runs,members=transitions_and_runs(cells,effects,flows)
    assert len(flows)==890 and len(trans)==1424 and int(runs.decisions.sum())==int(cells.accepted.sum());files=[]
    tables=dict(gate_sensitivity=summary,leave_one_week_out=leave,sensitivity_summary=cohorts,membership_flows=flows,membership_row_flows=rowflows,gate_transitions=trans,acceptance_runs=runs,acceptance_run_members=members)
    for name,g in tables.items():p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen();finish(run,files,new_fits=0,new_predictions=0,new_pvalues=0,gate_cells=len(cells),eligible_cells=int(cells.metric_eligible.sum()),leave_one_out_cases=len(leave),flow_cells=len(flows),transition_cells=len(trans),acceptance_runs=len(runs));print(f'Diagnosis complete: {len(leave)}single-week deletions,890membership flows,1424gate transitions. No policy changed.',flush=True)
if __name__=='__main__':main()

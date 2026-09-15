from common41 import *
from contract41 import synthetic
import math
from collections import Counter

def scalar_gate(mode,train_n,n,loss,correct):
    if mode!='ready':return False,mode
    if train_n<10:return False,'insufficient_state_train'
    if n<5:return False,'insufficient_validation'
    if loss>=-1e-12:return False,'brier_not_better'
    if correct<0:return False,'direction_worse'
    return True,'accepted'

def audit_sensitivity():
    cells=csv('gate_cells');effects=csv('validation_week_effects');summary=csv('gate_sensitivity');leave=csv('leave_one_week_out');ci=cells.set_index(['cadence','cutoff','method','state']);si=summary.set_index(['cadence','cutoff','method','state']);li=leave.set_index(['cadence','cutoff','method','state','removed_row_index']);count=0
    assert len(cells)==len(summary)==1456 and not leave.duplicated(['cadence','cutoff','method','state','removed_row_index']).any()
    for cadence in CADENCES:
        vp=csv(cadence+'_validation_predictions');vp=vp[vp.family.eq('state')];vi={k:g for k,g in vp.groupby(['cutoff','method','component','row_index'])};outcomes=csv(cadence+'_outcomes');oi=outcomes[outcomes.family.eq('state')&outcomes.history.eq(HISTORY[cadence][1])].set_index(['cutoff','method','component']);groups={k:g for k,g in effects[effects.cadence.eq(cadence)].groupby(['cutoff','method','state'])}
        for c in cells[cells.cadence.eq(cadence)].itertuples():
            key=(cadence,c.cutoff,c.method,c.state);g=groups.get((c.cutoff,c.method,c.state),effects.iloc[:0]);rows=list(g.itertuples());d=[];dc=[]
            for r in rows:
                raw=vi[(c.cutoff,c.method,c.state,r.row_index)];assert len(raw)==3;baseline=math.fsum(raw.annual_probability)/3;candidate=math.fsum(raw.candidate_probability)/3;y=int(raw.actual_up.iloc[0]);loss=(candidate-y)**2-(baseline-y)**2;cd=int((candidate>.5)==bool(y))-int((baseline>.5)==bool(y));eq(r.brier_difference,loss);assert r.correct_difference==cd;eq(r.annual_probability,baseline);eq(r.candidate_probability,candidate);d.append(loss);dc.append(cd)
            n=len(rows);orig_bd=math.fsum(d)/n if n else None;orig_cd=sum(dc);decision,reason=scalar_gate(c.mode,c.training_n,n,orig_bd,orig_cd);assert decision==c.accepted and reason==c.reason;eq(orig_bd,c.brier_difference);assert c.correct_difference==orig_cd
            eligible=c.mode=='ready' and c.training_n>=10 and n>=5;assert c.metric_eligible==eligible;flips=0;fullflips=0
            if eligible:
                for i,r in enumerate(rows):
                    # Explicitly reconstruct remaining observations, rather than producer sum-minus-one.
                    retained=[j for j in range(n) if j!=i];new_bd=math.fsum(d[j] for j in retained)/(n-1);new_cd=sum(dc[j] for j in retained);met=new_bd<-1e-12 and new_cd>=0;full,why=scalar_gate(c.mode,c.training_n,n-1,new_bd,new_cd);v=li.loc[(*key,r.row_index)]
                    eq(v.brier_difference,new_bd);assert v.correct_difference==new_cd and v.remaining_n==n-1 and v.metric_accepted==met and v.full_accepted==full and v.full_reason==why
                    assert v.metric_flipped==(met!=c.accepted) and v.full_flipped==(full!=c.accepted) and v.mechanical_support_failure==(n==5) and v.removed_date==r.date and v.removed_correct_difference==r.correct_difference;eq(v.removed_brier_difference,r.brier_difference)
                    flips+=met!=c.accepted;fullflips+=full!=c.accepted;count+=1
            r=si.loc[key];assert r.loo_n==(n if eligible else 0) and r.metric_flip_count==flips and r.full_flip_count==fullflips and r.minimum_support_boundary==(eligible and n==5)
            cls='unsupported' if not eligible else 'accepted_metric_fragile' if c.accepted and flips else 'accepted_count_only_fragile' if c.accepted and n==5 else 'accepted_single_drop_stable' if c.accepted else 'rejected_metric_fragile' if flips else 'rejected_single_drop_stable';assert r.sensitivity_class==cls
            positive=[-v for v in d if v<0];eq(r.largest_helpful_share,max(positive)/math.fsum(positive) if positive else None);assert r.helpful_brier_weeks==sum(v<0 for v in d) and r.harmful_brier_weeks==sum(v>0 for v in d)
            o=oi.loc[(c.cutoff,c.method,c.state)];assert r.future_n==o.n and r.future_recoveries==o.recoveries and r.future_regressions==o.regressions;eq(r.future_brier_vs_annual,o.brier_difference)
    assert count==len(leave)
    cohorts=csv('sensitivity_summary');assert len(cohorts)==168
    for r in cohorts.itertuples():
        period=next(p for p in cfg()['summary_windows'] if p['name']==r.period);g=summary[summary.cadence.eq(r.cadence)&summary.method.eq(r.method)&summary.cutoff.between(period['start'],period['end'])];g=g[g.metric_eligible] if r.cohort=='all_supported' else g[g.accepted] if r.cohort=='accepted_all' else g[g.sensitivity_class.eq(r.cohort)];defined=g[g.future_n.gt(0)];n=int(g.future_n.sum())
        assert r.cells==len(g) and r.accepted_cells==sum(g.accepted) and r.metric_fragile_cells==sum(g.metric_flip_count>0) and r.minimum_support_cells==sum(g.minimum_support_boundary) and r.loo_deletions==sum(g.loo_n) and r.metric_flips==sum(g.metric_flip_count)
        assert r.future_nonempty_cells==len(defined) and r.future_weeks==n and r.future_worse_cells==sum(defined.future_brier_vs_annual>1e-12) and r.future_better_cells==sum(defined.future_brier_vs_annual<-1e-12);eq(r.future_week_weighted_brier,math.fsum(float(z.future_brier_vs_annual)*int(z.future_n) for z in defined.itertuples())/n if n else None)
        assert r.future_recoveries==sum(g.future_recoveries) and r.future_regressions==sum(g.future_regressions)
    return dict(independent_validation_week_cells=len(effects),independent_gate_cells=len(cells),independent_deletion_cases=count,independent_sensitivity_cohorts=len(cohorts))

def audit_membership_and_runs():
    flows=csv('membership_flows');rowflows=csv('membership_row_flows');trans=csv('gate_transitions');runs=csv('acceptance_runs');rm=csv('acceptance_run_members');cells=csv('gate_cells');ci=cells.set_index(['cadence','cutoff','method','state']);bank=csv('weekly_signal_bank');state=bank[['row_index','state']].drop_duplicates().set_index('row_index').state;expectedrows=set();fi=flows.set_index(['cadence','cutoff','role','state']);actualruns=set()
    assert len(flows)==890 and len(trans)==1424
    caches={}
    for cadence in CADENCES:
        members=csv(cadence+'_members');schedule=csv(cadence+'_schedule');dates=schedule.cutoff.tolist();modes=schedule.set_index('cutoff')['mode'].to_dict();sets={}
        for cutoff in dates:
            for role in ['training','validation']:
                g=members[members.cutoff.eq(cutoff)&members.role.eq(role)];sets[(cutoff,role,'all')]={int(r.row_index):r for r in g.itertuples()}
                for st in STATES:sets[(cutoff,role,st)]={int(r.row_index):r for r in g.itertuples() if state.loc[r.row_index]==st}
        caches[cadence]=sets
        for i in range(1,len(dates)):
            before,after=dates[i-1:i+1]
            for role in ['training','validation']:
                for st in ['all']+STATES:
                    a=sets[(before,role,st)];b=sets[(after,role,st)];same=[k for k in a if k in b];lost=[k for k in a if k not in b];new=[k for k in b if k not in a];r=fi.loc[(cadence,after,role,st)]
                    assert r.previous_cutoff==before and r.previous_mode==modes[before] and r['mode']==modes[after] and r.both_ready==(modes[before]=='ready' and modes[after]=='ready')
                    assert [r.previous_n,r.current_n,r.retained_n,r.dropped_n,r.added_n]==[len(a),len(b),len(same),len(lost),len(new)] and len(b)==len(a)-len(lost)+len(new)
                    eq(r.retained_fraction_current,len(same)/len(b) if b else None);eq(r.jaccard,len(same)/len(set(a)|set(b)) if a or b else None)
                    pool=sets[(after,role,'all')];oldest=all(a[k].date<min(v.date for v in pool.values()) for k in lost) if lost and pool else not lost;assert r.dropped_older_than_current_window==oldest
                    assert r.support_dropout==(st!='all' and role=='validation' and len(a)>=5 and len(b)<5)
                    for status,ids in [('retained',same),('dropped',lost),('added',new)]:
                        for idx in ids:
                            x=b[idx] if idx in b else a[idx];expectedrows.add((cadence,before,after,role,st,idx,x.date,x.joint_completed,status))
        for method in LEARNED:
            for st in STATES:
                accepted=[i for i,d in enumerate(dates) if ci.loc[(cadence,d,method,st),'accepted']];segments=[]
                for i in accepted:
                    if segments and i==segments[-1][-1]+1:segments[-1].append(i)
                    else:segments.append([i])
                for segment in segments:
                    dd=[dates[i] for i in segment];rid=f'{cadence}:{method}:{st}:{dd[0]}';actualruns.add(rid);r=runs[runs.run_id.eq(rid)].iloc[0];uses=Counter(idx for d in dd for idx in sets[(d,'validation',st)]);assert [r.decisions,r.total_validation_uses,r.unique_validation_weeks,r.repeated_uses]==[len(dd),sum(uses.values()),len(uses),sum(uses.values())-len(uses)]
                    assert r.first_cutoff==dd[0] and r.last_cutoff==dd[-1] and r.first_last_overlap==len(set(sets[(dd[0],'validation',st)])&set(sets[(dd[-1],'validation',st)]));eq(r.uses_per_unique_week,sum(uses.values())/len(uses))
                    member=rm[rm.run_id.eq(rid)].set_index('row_index');assert set(member.index)==set(uses)
                    for idx,count in uses.items():
                        present=[d for d in dd if idx in sets[(d,'validation',st)]];assert member.loc[idx,'uses']==count and member.loc[idx,'first_cutoff']==present[0] and member.loc[idx,'last_cutoff']==present[-1]
    assert actualruns==set(runs.run_id) and runs.run_id.is_unique and int(runs.decisions.sum())==int(cells.accepted.sum())
    actualrows=set(rowflows.itertuples(index=False,name=None));assert actualrows==expectedrows and len(actualrows)==len(rowflows)
    for r in trans.itertuples():
        a=ci.loc[(r.cadence,r.previous_cutoff,r.method,r.state)];b=ci.loc[(r.cadence,r.cutoff,r.method,r.state)];f=fi.loc[(r.cadence,r.cutoff,'validation',r.state)];transition='enabled' if b.accepted and not a.accepted else 'stopped' if a.accepted and not b.accepted else 'retained_on' if b.accepted else 'retained_off'
        assert r.transition==transition and r.previous_accepted==a.accepted and r.accepted==b.accepted and r.previous_reason==a.reason and r.reason==b.reason and r.previous_mode==a['mode'] and r.mode==b['mode'] and r.both_ready==f.both_ready
        for key,other in [('previous_validation_n','previous_n'),('validation_n','current_n'),('retained_n','retained_n'),('dropped_n','dropped_n'),('added_n','added_n'),('retained_fraction_current','retained_fraction_current'),('support_dropout','support_dropout'),('dropped_older_than_current_window','dropped_older_than_current_window')]:eq(getattr(r,key),f[other])
        assert r.stop_by_validation_support==(transition=='stopped' and b.reason=='insufficient_validation')
    return dict(independent_role_flows=len(flows),independent_member_flows=len(rowflows),independent_gate_transitions=len(trans),independent_acceptance_runs=len(runs),accepted_cells_accounted=int(runs.decisions.sum()))

def main():
    prep=check_frozen();assert not (OUT/'verification.json').exists();run=check_phase('diagnosis');assert read(OUT/'contract_verification.json')['completed_utc']<=run['started_utc']<run['finished_utc']
    for key in ['protocol_sha256','source_sha256','input_sha256']:assert run[key]==prep[key]
    start=time.time();synthetic();s=audit_sensitivity();print('Independent single-week sensitivity, mechanical support and archived future-cohort checks PASS.',flush=True);m=audit_membership_and_runs();print('Independent membership turnover, support attrition and accepted-run reuse checks PASS.',flush=True)
    for src,name in COPIES:assert sha(src)==sha(OUT/name)
    assert old_evidence()==prep['old_evidence'];save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),elapsed_seconds=time.time()-start,protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],old_files_preserved=5633,unchanged_forecast_histories=22,new_fits=0,new_predictions=0,new_pvalues=0,**s,**m));print('R41 independent verification PASS.',flush=True)
if __name__=='__main__':main()

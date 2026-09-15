from common43 import *
import math,calendar

def expiry(d):
    y,m=map(int,[d[:4],d[5:7]]); m=(m//3+1)*3
    if m>12:y+=1;m-=12
    return f'{y:04d}-{m:02d}-{calendar.monthrange(y,m)[1]:02d}'
def incoming(gates,dates,method,state,cutoff):
    prefix=gates[gates.method.eq(method)&gates.component.eq(state)&gates.cutoff.lt(cutoff)].sort_values('cutoff')
    accepted=prefix[prefix.accepted]
    if not len(accepted):return None,None
    src=accepted.cutoff.iloc[-1]; end=expiry(src)
    after=prefix[prefix.cutoff.gt(src)]
    if any(r.cutoff>=end or r.mode!='ready' or r.reason!='insufficient_validation' or r.training_n<10 or r.validation_n>=5 for r in after.itertuples()):return None,None
    return src,end
def main():
    prep=check_frozen(False); assert not (OUT/'contract_verification.json').exists()
    b=csv('weekly_signal_bank'); mem=csv('split_membership'); schedule=csv('schedule'); gates=csv('gate_decisions'); dec=csv('retention_decisions'); heads=read(OUT/'correction_heads.json'); vp=csv('validation_predictions')
    hi={(h['cutoff'],h['method'],h['component'],h['seed']):h for h in heads}
    bi=b.set_index(['method','seed','row_index']); pool=b[b.method.eq(METHODS[0])&b.seed.eq(cfg()['seeds'][0])].sort_values('date')
    assert len(b)==3264 and len(gates)==1088 and len(heads)==3264 and len(dec)==1088
    assert b.groupby('row_index').state.nunique().eq(1).all() and b.groupby('row_index').joint_completed.nunique().eq(1).all()
    state=pool.set_index('row_index').state
    for c in schedule.itertuples():
        val=pool[pool.joint_completed.le(c.cutoff)].tail(13)
        train=pool[pool.joint_completed.lt(val.date.min())].tail(52) if len(val) else pool.iloc[:0]
        for role,g in [('training',train),('validation',val)]:
            src=mem[mem.cutoff.eq(c.cutoff)&mem.role.eq(role)]
            assert src.row_index.tolist()==g.row_index.tolist()
        assert not set(val.row_index)&set(train.row_index)
    for row in vp.itertuples():
        base=bi.loc[(row.method,row.seed,row.row_index)];h=hi[(row.cutoff,row.method,row.component,row.seed)]
        assert row.joint_completed<=row.cutoff and row.actual_up==base.actual_up and row.annual_probability==base.probability and row.offset==h['offset']
        p=base.probability if h['offset']==0 else 1/(1+math.exp(-base.annual_logit-h['offset']))
        assert abs(p-row.candidate_probability)<2e-15
    subjects=[]; dates=schedule.cutoff.tolist(); prev={d:dates[i-1] if i else None for i,d in enumerate(dates)}
    for r in dec.itertuples():
        src,end=incoming(gates,dates,r.method,r.state,r.cutoff)
        assert (src is None and pd.isna(r.previous_source_cutoff)) or src==r.previous_source_cutoff
        assert (end is None and pd.isna(r.previous_expiry)) or end==r.previous_expiry
        g=gates[gates.cutoff.eq(r.cutoff)&gates.method.eq(r.method)&gates.component.eq(r.state)].iloc[0]
        available=src is not None and src[:4]==r.cutoff[:4]
        availability='absent' if src is None else 'annual_mismatch' if not available else 'expired' if r.cutoff>=end else 'in_life'
        fit=all(hi[(r.cutoff,r.method,r.state,s)]['fit_eligible'] for s in cfg()['seeds'])
        assert fit==(g['mode']=='ready' and g.training_n>=10)
        if src:
            old=gates[gates.cutoff.eq(src)&gates.method.eq(r.method)&gates.component.eq(r.state)].iloc[0]
            assert old.accepted and src<r.cutoff and old.training_n>=10
            assert all(hi[(src,r.method,r.state,s)]['fit_eligible'] for s in cfg()['seeds'])
        subjects.append(dict(subject_id=f'{r.cutoff}:{r.method}:{r.state}',cutoff=r.cutoff,previous_decision_cutoff=prev[r.cutoff],method=r.method,state=r.state,mode=r.mode,original_accepted=bool(r.original_accepted),original_reason=r.original_reason,original_action=r.action,training_n=int(r.training_n),current_validation_n=int(r.validation_n),current_fit_eligible=fit,source_cutoff=src,expiry=end,source_age_days=int((pd.Timestamp(r.cutoff)-pd.Timestamp(src)).days) if src else None,availability=availability))
    # Eligibility boundary contract: no samples/too few are not passes, even if every observed loss improves.
    assert evidence(0,None,0)=='no_samples' and evidence(4,-.1,4)=='insufficient_validation'
    assert evidence(5,-.1,0)=='pass' and evidence(5,-1e-12,0)=='brier_not_better' and evidence(5,-.1,-1)=='direction_worse'
    assert evidence(5,-.1,1,False)=='unfitted'
    p=OUT/'subjects.csv';pd.DataFrame(subjects).to_csv(p,index=False)
    check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],monthly_cutoffs=68,independent_incoming_records=1088,original_validation_seed_rows=len(vp),future_outcomes_used=False,artifacts={str(p.relative_to(ROOT)):sha(p)}))
    print('Contract PASS: causal monthly membership, original parameters and all incoming saved records. Subjects frozen before revalidation.',flush=True)
if __name__=='__main__':main()

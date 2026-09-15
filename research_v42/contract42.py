from common42 import *
from scipy.special import expit

def synthetic():
    def simulate(events):
        rows=[]
        for cutoff,reason in events:
            for method in LEARNED:
                for state in STATES:
                    mode=reason if reason in ['annual_reset','q1_guard','insufficient_history'] else 'ready';rows.append(dict(cutoff=cutoff,method=method,family='state',component=state,mode=mode,reason=reason,accepted=reason=='accepted',training_n=9 if reason=='insufficient_state_train' else 20,validation_n=3 if reason=='insufficient_validation' else 8))
        return retention_decisions(pd.DataFrame(rows),[d for d,r in events])
    def first(t):return t[t.method.eq(LEARNED[0])&t.state.eq(STATES[0])].set_index('cutoff')
    assert expiry_after('2024-03-31')=='2024-06-30' and expiry_after('2024-05-31')=='2024-06-30' and expiry_after('2024-02-29')=='2024-03-31' and expiry_after('2024-12-31')=='2025-03-31'
    t=first(simulate([('2024-03-31','accepted'),('2024-04-30','insufficient_validation'),('2024-05-31','insufficient_validation'),('2024-06-30','insufficient_validation'),('2024-07-31','insufficient_validation')]))
    assert t.action.tolist()==['fresh','carry','carry','fallback','fallback']
    for d in ['2024-04-30','2024-05-31']:assert t.loc[d,'source_cutoff']=='2024-03-31' and t.loc[d,'expiry']=='2024-06-30'
    assert t.loc['2024-06-30','retention_reason']=='retention_expired' and t.loc['2024-07-31','retention_reason']=='no_valid_saved_acceptance'
    for reason in ['brier_not_better','direction_worse','insufficient_state_train','insufficient_history']:
        t=first(simulate([('2024-03-31','accepted'),('2024-04-30',reason),('2024-05-31','insufficient_validation')]))
        assert t.action.tolist()==['fresh','fallback','fallback'] and pd.isna(t.source_cutoff.iloc[-1])
    t=first(simulate([('2024-03-31','accepted'),('2024-04-30','insufficient_validation'),('2024-05-31','accepted'),('2024-06-30','accepted')]))
    assert t.loc['2024-05-31','source_cutoff']=='2024-05-31' and t.loc['2024-06-30','expiry']=='2024-09-30'
    t=first(simulate([('2024-11-30','accepted'),('2024-12-31','annual_reset'),('2025-01-31','q1_guard'),('2025-02-28','q1_guard'),('2025-03-31','insufficient_validation')]))
    assert t.action.tolist()==['fresh','fallback','fallback','fallback','fallback']
    assert usable('2024-03-31','2024-06-30','2024-06-30') and not usable('2024-03-31','2024-06-30','2024-07-01') and not usable('2024-03-31','2024-06-30','2024-03-31')
    assert v40.monthly_for('2023-06-30')=='2023-05-31' and v40.monthly_for('2023-07-01')=='2023-06-30'
    return ['strict_next_quarter_expiry','carry_cannot_extend_expiry','expiry_clears_source','no_resurrection_after_quality_rejection','training_and_history_rejection_clear','fresh_acceptance_replaces_source','annual_and_Q1clear','inclusive_signal_expiry_and_strict_decision_cutoff']

def causal_sources():
    obs,_,targets=data();bank=csv('weekly_signal_bank');bi=bank.set_index(['method','seed','row_index']);state=bank[['row_index','state']].drop_duplicates().set_index('row_index').state;members=csv('split_membership');schedule=csv('schedule');route=csv('routing');gates=csv('gate_decisions');heads=read(OUT/'correction_heads.json');hi={(h['cutoff'],h['method'],h['seed'],h['component']):h for h in heads};vp=csv('validation_predictions');assert len(gates)==1088 and len(heads)==3264 and len(bank)==3264
    for r in route.itertuples():assert r.head_cutoff==v40.monthly_for(r.date)<r.date and r.encoder_cutoff==annual_for(r.date)
    for c in schedule.itertuples():
        tr=members[members.cutoff.eq(c.cutoff)&members.role.eq('training')];va=members[members.cutoff.eq(c.cutoff)&members.role.eq('validation')]
        assert len(tr)==c.train_n and len(va)==c.validation_n and not set(tr.row_index)&set(va.row_index)
        if len(tr):assert tr.joint_completed.max()<va.date.min() and tr.joint_completed.max()<=c.fit_cutoff
        if len(va):assert va.joint_completed.le(c.cutoff).all()
        for g in [tr,va]:
            for r in g.itertuples():assert r.date==obs.iloc[r.row_index].date and r.joint_completed==obs.iloc[r.row_index].joint_completed
    grouped={k:g for k,g in vp.groupby(['cutoff','method','component'])}
    for r in gates.itertuples():
        ids=members[members.cutoff.eq(r.cutoff)&members.role.eq('training')].row_index;val=members[members.cutoff.eq(r.cutoff)&members.role.eq('validation')].row_index;assert r.training_n==int(ids.map(state).eq(r.component).sum());val=val[val.map(state).eq(r.component)];g=grouped.get((r.cutoff,r.method,r.component),vp.iloc[:0]);assert set(g.row_index)==set(val) and len(g)==3*r.validation_n
        bases=[];probs=[];labels=[]
        for idx in val:
            x=g[g.row_index.eq(idx)].sort_values('seed');assert x.seed.tolist()==cfg()['seeds']
            for s in x.itertuples():
                b=bi.loc[(r.method,s.seed,idx)];h=hi[(r.cutoff,r.method,s.seed,r.component)];assert h['offset']==s.offset and s.annual_probability==b.probability and s.actual_up==int(targets['returns'][idx]>0);eq(s.candidate_probability,b.probability if s.offset==0 else float(expit(b.annual_logit+s.offset)))
            bases.append(float(x.annual_probability.mean()));probs.append(float(x.candidate_probability.mean()));labels.append(int(x.actual_up.iloc[0]))
        p0=np.array(bases);p=np.array(probs);y=np.array(labels);bd=float(np.mean((p-y)**2-(p0-y)**2)) if len(y) else None;cd=int(((p>.5)==y).sum()-((p0>.5)==y).sum());eq(r.brier_difference,bd)
        reason=r.mode if r.mode!='ready' else 'insufficient_state_train' if r.training_n<10 else 'insufficient_validation' if r.validation_n<5 else 'brier_not_better' if not bd < -1e-12 else 'direction_worse' if cd<0 else 'accepted';assert r.reason==reason and r.accepted==(reason=='accepted')
    return dict(monthly_cutoffs=len(schedule),original_gate_cells=len(gates),original_head_cells=len(heads),weekly_seed_rows=len(bank))

def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();start=now();tests=synthetic();counts=causal_sources();check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],synthetic_cases=tests,**counts,artifacts={}));print('Contract PASS: expiry/reset/no-resurrection boundaries and original causal candidates/gates.',flush=True)
if __name__=='__main__':main()

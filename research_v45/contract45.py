from common45 import *
def synthetic():
    assert [component(s,'trend') for s in STATES]==['negative','negative','nonnegative','nonnegative']
    assert [component(s,'volatility') for s in STATES]==['low','high','low','high']
    assert quarter_for('2023-06-30')=='2023-03-31' and quarter_for('2023-07-01')=='2023-06-30'
    y=np.array([0,1,0,1,0]);p0=np.array([.4,.6,.4,.6,.4]);p=np.array([.3,.7,.3,.7,.3])
    assert gate('ready',10,p0,p,y)['accepted'] and gate('ready',9,p0,p,y)['reason']=='insufficient_state_train'
    assert gate('ready',10,p0[:4],p[:4],y[:4])['reason']=='insufficient_validation' and not gate('ready',10,p0,p0,y)['accepted']
    assert gate('annual_reset',52,p0,p,y)['reason']=='annual_reset'
    z=np.array([.51,.99,.6,.6,.6]);q=np.array([.49,.6,.6,.6,.6]);yy=np.array([1,0,1,1,1]);assert gate('ready',10,z,q,yy)['reason']=='direction_worse'
    np.testing.assert_array_equal(corrected(np.zeros(3),np.array([.1,.5,.9]),0),np.array([.1,.5,.9]))
    assert solve(np.zeros(10),np.ones(10))['offset']>0 and solve(np.zeros(100),np.ones(100))['offset']==.5
    return ['fixed_partition_unions','strict_quarter_route','training10_boundary','validation5_boundary','Brier_tie_rejected','direction_regression_rejected','annual_reset','zero_offset_exact_copy','scalar_bound']
def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=synthetic();bank=csv('weekly_signal_bank');route=csv('routing');schedule=csv('schedule');members=csv('split_membership');pool=bank[bank.method.eq(METHODS[0])&bank.seed.eq(cfg()['seeds'][0])].sort_values('date')
    assert len(bank)==3264 and len(pool)==272 and len(route)==272 and len(schedule)==23 and schedule['mode'].eq('ready').sum()==13
    assert bank.groupby('row_index').state.nunique().eq(1).all() and bank.groupby('row_index').joint_completed.nunique().eq(1).all()
    np.testing.assert_allclose(probability(bank.annual_logit),bank.probability,rtol=0,atol=1e-12)
    annual=csv('baseline_model_predictions');annual=annual[annual.history.eq(ANNUAL)&annual.method.isin(METHODS)].set_index(['method','seed','row_index']);bi=bank.set_index(['method','seed','row_index'])
    for key,b in bi.iterrows():
        a=annual.loc[key];assert a.probability==b.probability and a.actual_up==b.actual_up and a.actual==b.actual and a.joint_completed==b.joint_completed
    for r in route.itertuples():assert quarter_for(r.date)==r.head_cutoff<r.date and r.encoder_cutoff==f'{int(r.date[:4])-1}-12-31'
    support=[]
    for c in schedule.itertuples():
        va=pool[pool.joint_completed.le(c.cutoff)].tail(13);tr=pool[pool.joint_completed.lt(va.date.min())].tail(52) if len(va) else pool.iloc[:0]
        for role,g in [('training',tr),('validation',va)]:assert g.row_index.tolist()==members[members.cutoff.eq(c.cutoff)&members.role.eq(role)].row_index.tolist()
        mode='annual_reset' if c.cutoff.endswith('12-31') else 'ready' if len(tr)==52 and len(va)==13 else 'insufficient_history';assert mode==c.mode
        if len(tr):assert tr.joint_completed.max()<va.date.min() and tr.exit.max()<=va.entry.min()
        for family,groups in FAMILIES.items():
            for group in groups:
                nt=int(tr.state.map(lambda s:component(s,family)).eq(group).sum());nv=int(va.state.map(lambda s:component(s,family)).eq(group).sum());support.append(dict(cutoff=c.cutoff,family=family,component=group,mode=c.mode,training_n=nt,validation_n=nv,train_supported=mode=='ready' and nt>=10,validation_supported=mode=='ready' and nt>=10 and nv>=5))
    p=OUT/'group_support.csv';pd.DataFrame(support).to_csv(p,index=False);assert len(support)==92
    check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],synthetic_cases=tests,ready_quarters=13,cutoffs=23,group_support_cells=92,artifacts={str(p.relative_to(ROOT)):sha(p)}));print('Contract PASS: original52/13maturity splits, strict routing and two fixed unions; support frozen.',flush=True)
if __name__=='__main__':main()

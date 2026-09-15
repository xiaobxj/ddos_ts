from common40 import *
from base_contract35 import independent_route
def synthetic():
    dates=pd.date_range('2018-01-05',periods=80,freq='W-FRI');p=pd.DataFrame(dict(row_index=np.arange(80),date=dates.strftime('%Y-%m-%d'),joint_completed=(dates+pd.Timedelta(days=10)).strftime('%Y-%m-%d')));cutoff=(dates[-1]+pd.Timedelta(days=10)).strftime('%Y-%m-%d');tr,va,r=split(p,cutoff);assert len(tr)==52 and len(va)==13 and tr.joint_completed.max()<va.date.min() and r['purged_between_n']==1;assert tr.row_index.max()==65 and va.row_index.min()==67
    poison=p.copy();poison.loc[poison.joint_completed.gt(cutoff),'joint_completed']='2099-01-01';a,b,_=split(poison,cutoff);pd.testing.assert_frame_equal(a,tr);pd.testing.assert_frame_equal(b,va)
    assert split(p.iloc[:20],cutoff)[2]['mode']=='insufficient_history';assert split(p,'2020-12-31')[2]['mode']=='annual_reset'
    z=np.zeros(52);y=np.r_[np.ones(40),np.zeros(12)];g=pd.DataFrame(dict(annual_logit=z,actual_up=y,state=[STATES[0]]*10+[STATES[1]]*9+[STATES[2]]*33));h=fitted(g,'ready');assert h[0]['fit_eligible'] and not h[1]['fit_eligible'] and h[1]['offset']==0;assert all(x['offset']==0 for x in fitted(g,'annual_reset'))
    p0=np.array([.4,.6,.4,.6,.4]);y=np.array([0,1,0,1,0]);better=np.array([.3,.7,.3,.7,.3]);assert gate_decision('ready','state',10,5,p0,better,y)['accepted'];assert not gate_decision('ready','state',10,5,p0,p0,y)['accepted'];assert not gate_decision('ready','state',10,4,p0[:4],better[:4],y[:4])['accepted'];assert not gate_decision('ready','state',9,5,p0,better,y)['accepted']
    p0=np.array([.51,.99]);new=np.array([.49,.6]);y=np.array([1.,0.]);r=gate_decision('ready','state',10,2,p0,new,y);assert not r['accepted']
    p0=np.r_[p0,np.full(3,.6)];new=np.r_[new,np.full(3,.6)];y=np.r_[y,np.ones(3)];r=gate_decision('ready','state',10,5,p0,new,y);assert r['brier_difference']<0 and r['candidate_correct']<r['baseline_correct'] and r['reason']=='direction_worse'
    p=np.array([.1,.5,.9]);np.testing.assert_array_equal(corrected_probability(np.zeros(3),p,np.zeros(3)),p)
    assert monthly_for('2026-07-31')=='2026-06-30' and monthly_for('2026-08-01')=='2026-07-31' and monthly_for('2024-03-01')=='2024-02-29'
    pool=pd.DataFrame(dict(row_index=np.arange(80),date=dates.strftime('%Y-%m-%d'),joint_completed=(dates+pd.Timedelta(days=10)).strftime('%Y-%m-%d')))
    assert split(pool,'2026-01-31')[2]['mode']=='q1_guard' and split(pool,'2026-02-28')[2]['mode']=='q1_guard'
    frames_equal_missing(pd.DataFrame({'brier_difference':[None]}),pd.DataFrame({'brier_difference':[np.nan]}),check_dtype=False)
    for aa,bb in [(None,0.),(.1,.2)]:
        try:frames_equal_missing(pd.DataFrame({'brier_difference':[aa]}),pd.DataFrame({'brier_difference':[bb]}),check_dtype=False)
        except AssertionError:pass
        else:raise AssertionError('Missing compatibility masked a substantive difference')
    return ['month_end_friday_strict_cutoff','leap_month_end','January_February_Q1guard','missing_only_normalization','purge_unmatured_adjacent_training_label','strict_training_before_validation','52plus13count_requirement','annual_reset','state_train10boundary','state_validation5boundary','Brier_tie_rejected','Brier_gain_cannot_override_worse_direction','zero_offset_exact_copy']
def build():
    obs,_,targets=data();base=csv('baseline_model_predictions');base=base[base.history.eq(ANNUAL)&base.method.isin(LEARNED)].copy();fx=csv('source_seed_effects');fx=fx[fx.history.eq('annual_state_offset')][['row_index','method','seed','annual_logit','annual_probability']];ctx=csv('weekly_context')[['row_index','date','head_cutoff','encoder_cutoff','state']];bank=base.merge(fx,on=['row_index','method','seed'],validate='one_to_one').merge(ctx,on=['row_index','date'],validate='many_to_one');assert len(bank)==3264;np.testing.assert_array_equal(bank.probability,bank.annual_probability);np.testing.assert_allclose(probability(bank.annual_logit),bank.probability,rtol=0,atol=1e-12)
    bank=bank.merge(obs.reset_index(names='row_index')[['row_index','entry','exit']],on='row_index',validate='many_to_one');route=csv('routing');pool=route[['row_index','date']].merge(obs[['date','joint_completed','entry','exit']],on='date',validate='one_to_one');np.testing.assert_array_equal(bank.actual_up,(targets['returns'][bank.row_index.to_numpy(int)]>0).astype(int));assert len(pool)==272 and pool.row_index.is_unique
    for r in route.itertuples():assert (r.encoder_cutoff,r.quarter_cutoff)==independent_route(r.date);assert r.head_cutoff==monthly_for(r.date)<r.date
    schedules=[];members=[];support=[]
    for cutoff in cfg()['decision_dates']:
        tr,va,r=split(pool,cutoff);schedules.append(r)
        if len(tr):assert tr.joint_completed.le(r['fit_cutoff']).all() and tr.joint_completed.max()<va.date.min() and int(tr.exit.max())<=int(va.entry.min())
        assert not set(tr.row_index)&set(va.row_index);te=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);assert obs.date.iloc[te].gt(cutoff).all()
        for role,rows in [('training',tr),('validation',va)]:
            for row in rows.itertuples():members.append(dict(cutoff=cutoff,role=role,used=r['mode']=='ready',row_index=int(row.row_index),date=row.date,joint_completed=row.joint_completed))
        for state in STATES:
            s=ctx.set_index('row_index').state;nt=int(s.loc[tr.row_index].eq(state).sum());nv=int(s.loc[va.row_index].eq(state).sum());support.append(dict(cutoff=cutoff,state=state,mode=r['mode'],training_n=nt,validation_n=nv,train_supported=r['mode']=='ready' and nt>=10,validation_supported=r['mode']=='ready' and nt>=10 and nv>=5))
    schedules=pd.DataFrame(schedules);members=pd.DataFrame(members)
    oldbank=pd.read_csv(V39/'results/weekly_signal_bank.csv',float_precision='round_trip');pd.testing.assert_frame_equal(bank,oldbank,check_dtype=False,check_exact=True)
    oldmembers=csv('quarterly_members')
    for cutoff in csv('quarterly_schedule').cutoff:
        pd.testing.assert_frame_equal(members[members.cutoff.eq(cutoff)].reset_index(drop=True),oldmembers[oldmembers.cutoff.eq(cutoff)].reset_index(drop=True),check_dtype=False,check_exact=True)
    return bank,schedules,members,pd.DataFrame(support)
def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();start=now();cases=synthetic();tables=build();files=[]
    for name,g in zip(['weekly_signal_bank','schedule','split_membership','state_support'],tables):p=OUT/f'{name}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],synthetic_cases=cases,ready_months=int(tables[1]['mode'].eq('ready').sum()),first_ready= tables[1][tables[1]['mode'].eq('ready')].cutoff.min(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('Contract PASS: purged52/13 splits, archived point-in-time signals and support frozen.',flush=True)
if __name__=='__main__':main()

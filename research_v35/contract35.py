from common35 import *
from base_contract35 import base_memberships_check,boundary_cases,independent_route
def synthetic_cases():
    fake=pd.DataFrame(dict(date=['2016-01-01','2016-04-01','2020-12-01','2020-12-30','2021-03-01','2021-03-02'],joint_completed=['2016-01-12','2016-04-12','2020-12-15','2021-01-12','2021-03-31','2021-04-01']))
    s=member_sets(fake,'2021-03-31')
    for arm,expected in {'annual':[0,1,2],'add':[0,1,2,3,4],'remove':[1,2],'both':[1,2,3,4]}.items():np.testing.assert_array_equal(s[arm],expected)
    for arm,ids in member_sets(fake,'2020-12-31').items():np.testing.assert_array_equal(ids,[0,1,2])
    p0=np.array([.4,.7]);pa=np.array([.6,.5]);pr=np.array([.45,.65]);pb=np.array([.5,.8]);e=effects(p0,pa,pr,pb)
    np.testing.assert_allclose(e['add_effect']+e['remove_effect'],pb-p0,rtol=0,atol=1e-15)
    np.testing.assert_allclose(e['add_without_remove']+e['remove_after_add'],pb-p0,rtol=0,atol=1e-15)
    np.testing.assert_allclose(e['remove_without_add']+e['add_after_remove'],pb-p0,rtol=0,atol=1e-15)
    z=effects(p0,p0,p0,p0);assert all(np.all(v==0) for v in z.values())
    from scipy.special import expit
    zz=np.array([-3.,1.,1.]);assert expit(zz).mean()>.5 and expit(zz.mean())<.5;assert not probability(np.array([0.]))[0]>.5
    return ['four_member_sets','newly_matured_old_date_included','future_maturity_excluded','annual_reset_identity','both_order_sum_identity','symmetric_effect_sum','identical_sets_zero_effect','seed_probability_mean_not_mean_logit','strict_half_is_down']
def ablation_checks():
    obs,price,targets=data();summary=csv('membership_summary');members=csv('ablation_membership');banks=read(OUT/'feature_banks.json');checks=[]
    assert len(summary)==92 and summary.new_fit.sum()==34 and cfg()['probe']==previous_round.cfg()['probe'] and cfg()['offset_probe']==previous_round.cfg()['offset_probe']
    for cutoff in cfg()['decision_dates']:
        annual=annual_for(cutoff)
        def independent_set(c):
            lo=(pd.Timestamp(c)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            return {i for i,r in obs.iterrows() if r.date>=lo and isinstance(r.joint_completed,str) and r.joint_completed<=c}
        a=independent_set(annual);q=independent_set(cutoff);expected={'annual':a,'add':a|q,'remove':a&q,'both':q};computed=member_sets(obs,cutoff)
        assert (a|q)-a==q-a and a-(a&q)==a-q and ((a|q)-(a-q))==q
        te=csv('routing').query('head_cutoff==@cutoff').row_index.to_numpy(int)
        for arm,ids in expected.items():
            rows=sorted(ids);np.testing.assert_array_equal(computed[arm],rows);g=members[members.arm.eq(arm)&members.cutoff.eq(cutoff)];np.testing.assert_array_equal(g.row_index,rows)
            np.testing.assert_array_equal(g.date,obs.date.iloc[rows]);np.testing.assert_array_equal(g.joint_completed,obs.joint_completed.iloc[rows])
            r=summary[summary.arm.eq(arm)&summary.cutoff.eq(cutoff)].iloc[0];assert r.train_n==len(rows) and r.maximum_maturity==obs.joint_completed.iloc[rows].max()<=cutoff and r.minimum_date==obs.date.iloc[rows].min() and r.maximum_date==obs.date.iloc[rows].max()
            assert abs(r.up_fraction-(targets['returns'][rows]>0).mean())<1e-14 and not set(rows)&set(te)
            for b in [b for b in banks if b['cutoff']==annual]:assert ids<=set(arrays(b)['row_index'])
            checks.append(dict(cutoff=cutoff,arm=arm,train_n=len(rows),all_mature=True,no_test_overlap=True,bank_complete=True))
    # Member masks above are independent; recompute serialized train-only summaries as well.
    for n,g in zip(['membership_summary','ablation_membership','set_changes'],membership_tables()):pd.testing.assert_frame_equal(g,csv(n),check_dtype=False,rtol=0,atol=1e-14)
    return pd.DataFrame(checks)
def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();base=base_memberships_check();checks=ablation_checks();cases=boundary_cases()+synthetic_cases();files=[]
    for n,g in [('base_membership_checks',base),('ablation_checks',checks)]:p=OUT/f'{n}.csv';g.to_csv(p,index=False);files.append(p)
    check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],base_cutoffs=23,ablation_sets=92,synthetic_cases=cases,independent_descriptors=True,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},new_head_fits=0));print('Contract PASS:92 independent member sets, causal boundaries, descriptors and accounting identities.',flush=True)
if __name__=='__main__':main()

from common36 import *
from base_contract35 import independent_route,boundary_cases
def synthetic_cases():
    x=np.array([[-1,1,0,0,0],[-1,2,0,0,0],[0,1,0,0,0],[1,2,0,0,0]],float);np.testing.assert_array_equal(assign(x,dict(volatility_median=1)),STATES)
    c,_=coverage(np.array([[0.,1.],[-.1,1.1]]),np.array([0.,0.]),np.array([1.,1.]),np.zeros(2),np.ones(2));np.testing.assert_array_equal(c['outside_any'],[0,1]);np.testing.assert_array_equal(c['outside_fraction'],[0,1])
    d=label_decomposition([10,10,10,10],[2,4,6,8],[0,10,20,0],[0,5,16,0]);assert d['status']=='defined' and abs(d['overall_delta']-.2)<1e-14
    assert label_decomposition([1,1],[0,1],[0,0],[0,0])['status']=='empty_target'
    assert label_decomposition([1,0],[0,0],[0,1],[0,1])['status']=='target_state_absent_in_annual'
    same=label_decomposition([10,10],[3,7],[10,10],[3,7]);assert same['composition']==0 and same['within_state']==0
    raw=np.arange(50,dtype=float).reshape(10,5);t=calibrate(raw,np.arange(5));poison=raw.copy();poison[5:]=1e10;assert t==calibrate(poison,np.arange(5))
    return ['trend_zero_nonnegative','median_equality_low','strict_coverage_bounds','empty_target_no_imputation','missing_annual_state_flag','label_mix_sum_identity','identical_labels_zero_decomposition','future_descriptors_excluded_from_thresholds']
def checks():
    obs,price=raw_data();raw=descriptor_matrix(obs,price);bars=price[['open','high','low','close','volume']].to_numpy(float);scalar=prior.additive.scalar_descriptors(bars,obs.anchor.to_numpy(int))[NAMES[:4]].to_numpy(float);np.testing.assert_allclose(raw[:,:4],scalar,rtol=1e-12,atol=1e-12)
    order=[]
    for t in obs.anchor:
        closes=bars[t-60:t+1,3];sign=np.array([int(b>a)-int(b<a) for a,b in zip(closes[:-1],closes[1:])]);order.append(sum(sign[:-1]*sign[1:])/59-(sum(sign)**2-sum(sign**2))/(60*59))
    np.testing.assert_allclose(raw[:,4],order,rtol=0,atol=1e-12);members=csv('membership_roles');required=csv('required_rows');route=csv('routing');checks=[]
    assert len(route)==272
    canonical=[i for i,r in obs.iterrows() if r.weekday==4 and r.date>='2021-01-01' and isinstance(r.joint_completed,str) and r.joint_completed<=cfg()['label_end']];assert route.row_index.tolist()==canonical
    for r in route.itertuples():assert (r.encoder_cutoff,r.head_cutoff)==independent_route(r.date)
    for cutoff in cfg()['decision_dates']:
        annual=annual_for(cutoff)
        def rows(c):
            lo=(pd.Timestamp(c)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).strftime('%Y-%m-%d');return {i for i,r in obs.iterrows() if r.date>=lo and isinstance(r.joint_completed,str) and r.joint_completed<=c}
        a=rows(annual);q=rows(cutoff);expected=dict(annual=a,added=q-a,removed=a-q,retained=a&q,both=q)
        for role,ids in expected.items():
            g=members[members.cutoff.eq(cutoff)&members.role.eq(role)];assert g.row_index.tolist()==sorted(ids);assert all(obs.joint_completed.iloc[i]<=cutoff for i in ids);checks.append(dict(cutoff=cutoff,role=role,n=len(ids),mature=True))
    for a,g in required.groupby('encoder_cutoff'):
        needed=set(members[members.encoder_cutoff.eq(a)].row_index)|set(route[route.encoder_cutoff.eq(a)].row_index);assert g.row_index.tolist()==sorted(needed)
        for br in [b for b in read(OUT/'feature_banks.json') if b['cutoff']==a]:
            bank=arrays(br);assert needed<=set(bank['row_index']);h=next(h for h in read(OUT/'annual_heads.json') if h['cutoff']==a and h['seed']==br['seed']);d=arrays(h);np.testing.assert_array_equal(d['row_index'],training_rows(obs,a));np.testing.assert_array_equal(bank_slice(bank,d['row_index']),d['features'])
    for n,g in zip(['membership_roles','required_rows'],memberships()):pd.testing.assert_frame_equal(g,csv(n),check_dtype=False,check_exact=True)
    return pd.DataFrame(checks)
def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();table=checks();cases=boundary_cases()+synthetic_cases();p=OUT/'membership_checks.csv';table.to_csv(p,index=False);check_frozen(False)
    save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],membership_sets=115,independent_descriptors=True,synthetic_cases=cases,artifacts={str(p.relative_to(ROOT)):sha(p)}));print('Contract PASS: causal descriptors,115 member sets and diagnostic edge cases.',flush=True)
if __name__=='__main__':main()

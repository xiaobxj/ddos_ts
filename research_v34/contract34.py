from common34 import *
def independent_route(date):
    d=pd.Timestamp(date);month=3*((d.month-1)//3)+1;quarter=str((pd.Timestamp(d.year,month,1)-pd.Timedelta(days=1)).date());return f'{d.year-1}-12-31',quarter
def boundary_cases():
    expected={'2021-03-31':('2020-12-31','2020-12-31'),'2021-04-01':('2020-12-31','2021-03-31'),'2021-06-30':('2020-12-31','2021-03-31'),'2021-07-01':('2020-12-31','2021-06-30'),'2021-12-31':('2020-12-31','2021-09-30'),'2022-01-01':('2021-12-31','2021-12-31')}
    for date,v in expected.items():assert independent_route(date)==v
    fake=pd.DataFrame(dict(date=['2016-03-31','2016-04-01','2021-03-01','2021-03-02'],joint_completed=['2016-04-08','2016-04-11','2021-03-31','2021-04-01']));np.testing.assert_array_equal(training_rows(fake,'2021-03-31'),[1,2])
    return ['quarter_end_still_old_head','next_day_new_head','year_end_still_old_encoder','new_year_new_encoder','five_year_lower_boundary_inclusive','maturity_equality_included_future_excluded']
def memberships_check():
    obs,price,targets=data();jobs=csv('jobs');members=csv('membership');route=csv('routing');banks=csv('bank_membership');assert len(jobs)==23 and jobs.refit.sum()==17 and len(route)==272
    assert cfg()['probe']==prior.cfg()['probe'] and cfg()['offset_probe']==prior.cfg()['offset_probe'];assert read(V28/'results/verification.json')['status']=='PASS' and read(V31/'results/verification.json')['status']=='PASS'
    canonical_rows=[]
    for i,r in obs.iterrows():
        if r.weekday==4 and r.date>='2021-01-01' and isinstance(r.joint_completed,str) and r.joint_completed<=cfg()['label_end']:canonical_rows.append(i)
    assert route.row_index.tolist()==canonical_rows
    for r in route.itertuples():assert (r.encoder_cutoff,r.head_cutoff)==independent_route(r.date) and r.encoder_cutoff<=r.head_cutoff<r.date
    checks=[];alltr=[]
    for j in jobs.itertuples():
        c=pd.Timestamp(j.head_cutoff);lower=c-pd.DateOffset(years=5)+pd.Timedelta(days=1);tr=np.flatnonzero(pd.to_datetime(obs.date).ge(lower)&pd.to_datetime(obs.joint_completed).le(c));te=route[route.head_cutoff.eq(j.head_cutoff)].row_index.to_numpy(int)
        np.testing.assert_array_equal(tr,training_rows(obs,j.head_cutoff));assert len(tr)==j.train_n and len(te)==j.test_n and not np.intersect1d(tr,te).size
        for split,ids in [('training',tr),('testing',te)]:np.testing.assert_array_equal(members[members.head_cutoff.eq(j.head_cutoff)&members.split.eq(split)].row_index,ids)
        assert j.encoder_cutoff==annual_for(j.head_cutoff) and j.encoder_cutoff.endswith('12-31');assert obs.joint_completed.iloc[tr].max()<=j.head_cutoff<obs.date.iloc[te].min();alltr.extend(tr);checks.append(dict(head_cutoff=j.head_cutoff,encoder_cutoff=j.encoder_cutoff,training_rows=len(tr),testing_rows=len(te),causal=True))
    for model in read(OUT/'source_models.json'):
        assert sha(PROJECT/model['project_file'])==model['sha256'];annual=model['cutoff'];seed=model['seed'];h=next(h for h in read(OUT/'source_heads.json') if h['cutoff']==annual and h['seed']==seed and h['method']=='learned_market');t=next(t for t in read(OUT/'source_testing_features.json') if t['cutoff']==annual and t['seed']==seed);d=arrays(h);td=arrays(t);np.testing.assert_array_equal(d['row_index'],training_rows(obs,annual));assert not np.intersect1d(d['row_index'],td['row_index']).size
        oldtrain=set(d['row_index']);oldtest=set(td['row_index']);needed=set(members[members.encoder_cutoff.eq(annual)].row_index);b=banks[banks.encoder_cutoff.eq(annual)];assert b.row_index.tolist()==sorted(oldtrain|oldtest|needed)
        for r in b.itertuples():assert r.source==(0 if r.row_index in oldtrain else 1 if r.row_index in oldtest else 2)
    ids=np.unique(alltr);anchors=obs.anchor.iloc[ids].to_numpy(int);actual=market(price,obs,ids);scalar=prior.additive.scalar_descriptors(price[['open','high','low','close','volume']].to_numpy(float),anchors)[prior.additive.MARKET_NAMES].to_numpy(float);np.testing.assert_allclose(actual,scalar,rtol=1e-12,atol=1e-12)
    independent=[]
    for a in anchors:
        close=price.close.iloc[a-60:a+1].to_numpy(float);sign=[int(close[i]>close[i-1])-int(close[i]<close[i-1]) for i in range(1,61)];independent.append(sum(sign[i]*sign[i-1] for i in range(1,60))/59-((sum(sign)**2)-sum(s*s for s in sign))/(60*59))
    np.testing.assert_allclose(gates(price,obs,ids),independent,rtol=0,atol=1e-12);return pd.DataFrame(checks)
def coefficient_cases():
    from scipy.special import expit
    a=np.array([-.7, .3, -.2]);b=np.array([.8, -.4, .6]);x=np.array([[1.,2.],[-1.,.5]])
    for alpha in [0.,.25,.5,1.]:
        t=shrink(a,b,alpha);np.testing.assert_allclose(design(x)@t,(1-alpha)*(design(x)@a)+alpha*(design(x)@b),rtol=0,atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(t-a),alpha*np.linalg.norm(b-a),rtol=0,atol=1e-15)
    np.testing.assert_array_equal(shrink(a,b,0),a);np.testing.assert_array_equal(shrink(a,b,1),b)
    assert abs(expit(.25*3+.75*(-1))-(.25*expit(3)+.75*expit(-1)))>.01
    z=np.array([-3.,1.,1.]);assert expit(z).mean()>.5 and expit(z.mean())<.5
    assert not probability(np.array([0.]))[0]>.5
    rng=np.random.default_rng(20260910);xx=[rng.normal(size=(90,n)) for n in [3,4,5]];xx.append(xx[-1]);y=(rng.random(90)>.48).astype(float)
    ts,_,ms=fit_coefficients(xx,y);np.testing.assert_array_equal(np.r_[ts[3][:-2],ts[3][-1]],ts[1])
    for j in range(3):assert ms[j]['gradient_inf']<=1e-9
    assert ms[3]['gradient_inf']<=1e-10
    return ['fractional_coefficient_drift','within_seed_logit_identity','exact_alpha0_alpha1','not_probability_blend','not_sigmoid_mean_logit','strict_half_is_down','synthetic_fixed_design_solvers_and_R25_offset']

def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();checks=memberships_check();cases=boundary_cases()+coefficient_cases();check_frozen(False);checks.to_csv(OUT/'membership_checks.csv',index=False)
    save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],head_dates=23,annual_sources=18,synthetic_boundary_cases=cases,independent_descriptors=True,artifacts={'membership_checks.csv':sha(OUT/'membership_checks.csv')},new_head_fits=0));print('Contract PASS: causal rolling memberships, annual/quarter boundaries and independent descriptors.',flush=True)
if __name__=='__main__':main()

from common33 import *
def synthetic_cases():
    rng=np.random.default_rng(20260910);xa=rng.normal(size=(7,31));xn=rng.normal(size=(7,31));ta=rng.normal(size=32);tn=rng.normal(size=32);a,b,c,t=product_components(xa,ta,xn,tn,'learned_order_extension');np.testing.assert_allclose((b-a).sum(1),(xn@tn[:-1]+tn[-1])-(xa@ta[:-1]+ta[-1]),rtol=0,atol=1e-12);cases=['group_logit_identity','symmetric_product_identity']
    a,b,c,t=product_components(xa,ta,xa,ta,'learned_order_extension');assert not np.any(b-a) and not np.any(c) and not np.any(t);cases.append('identical_pipeline_zero')
    x=np.ones((1,29));aa=np.zeros(30);bb=np.zeros(30);bb[-1]=1.;a,b,c,t=product_components(x,aa,x,bb,'learned_market');assert np.array_equal(b-a,np.array([[1,0,0,0,0]]));cases.append('intercept_only_and_missing_groups')
    aa=np.zeros(30);bb=aa.copy();aa[0]=1;bb[0]=.5;a,b,c,t=product_components(x,aa,x*2,bb,'learned_market');assert np.array_equal(a,b) and c[0,1]==-.75 and t[0,1]==.75;cases.append('coordinate_rescaling_cancels')
    za=np.array([-4.,0.,2.,.7]);zn=np.array([3.,0.,2.+1e-11,-.8]);w=secant(za,zn);np.testing.assert_allclose(w*(zn-za),expit(zn)-expit(za),rtol=0,atol=1e-14);assert (w>0).all();cases += ['secant_probability_identity','zero_and_nearzero_secant']
    z=np.array([-3.,1.,1.]);assert expit(z).mean()>.5 and expit(z.mean())<.5;cases.append('probability_ensemble_not_logit_ensemble')
    assert not bool(expit(0)>.5);cases.append('probability_half_is_down')
    return cases
def sources_check():
    assert read(V32/'results/verification.json')['status']=='PASS';obs,_,_=data();route=csv('routing');np.testing.assert_array_equal(route.row_index,previous_round.canonical(obs));assert len(route)==272
    for g in [csv('model_predictions'),csv('ensemble_predictions')]:
        assert g.date.tolist()==obs.date.iloc[g.row_index].tolist();assert g.joint_completed.tolist()==obs.joint_completed.iloc[g.row_index].tolist();assert g.groupby(['history','method']).date.nunique().eq(272).all()
    heads=read(OUT/'annual_heads.json');new=read(OUT/'quarter_heads.json');banks=read(OUT/'feature_banks.json');assert len(heads)==72 and len(new)==204 and len(banks)==18
    refs={r['cache_file']:r for r in heads+new+banks}
    for r in refs.values():assert sha(PROJECT/r['cache_file'])==r['cache_sha256']
    for r in new:
        tr=training_rows(obs,r['cutoff']);np.testing.assert_array_equal(arrays(r)['row_index'],tr);assert r['encoder_cutoff']<r['cutoff'] and r['train_n']==len(tr)
    return dict(annual_heads=72,quarter_heads=204,feature_banks=18,source_cache_files=len(refs),seed_week_pairs=3264,ensemble_week_pairs=1088)
def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();checks=sources_check();cases=synthetic_cases();check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],checks=checks,synthetic_cases=cases,new_fit_or_decomposition=False));print('Contract PASS: source membership and nine algebra/ensemble boundary checks.',flush=True)
if __name__=='__main__':main()

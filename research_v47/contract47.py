from common47 import *
from datetime import date
import math

def independent_inputs(f,m,u,d):
    rep=(np.clip(f,d['rep__lower'],d['rep__upper'])-d['rep__mean'])/d['rep__sd'];mk=(np.clip(m,d['market__lower'],d['market__upper'])-d['market__mean'])/d['market__sd']
    a=np.column_stack([rep,mk]);v=mk[:,1]*(rep@d['vol__ray'])-a@d['vol__projection'][:-1]-d['vol__projection'][-1];v=(v-d['vol__residual_mean'])/d['vol__residual_sd'];b=np.column_stack([a,v])
    h=u*(rep@d['order__ray'])-b@d['order__projection'][:-1]-d['order__projection'][-1];h=(h-d['order__residual_mean'])/d['order__residual_sd'];return [a,b,np.column_stack([b,h])]

def synthetic():
    sys.path.insert(0,str(PROJECT/'research_v15'));import solver15
    rng=np.random.default_rng(471);x=rng.normal(size=(37,5));y=rng.integers(0,2,size=37).astype(float);theta=rng.normal(size=6)*.1;a=design(x);w=np.full(37,1/37)
    new=objective(theta,a,y,w);old=solver15.objective(theta,a,y)
    for v,u in zip(new,old):np.testing.assert_allclose(v,u,rtol=0,atol=1e-14)
    t,_=fit_newton(x,y,w,cfg()['probe']);old_t,_=solver15.fit_newton(x,y,cfg()['probe']);np.testing.assert_allclose(t,old_t,rtol=0,atol=1e-10)
    w=rng.uniform(.1,1,size=37);w/=w.sum();val,g,h=objective(theta,a,y,w);eps=1e-5
    for j in range(len(theta)):
        t1=theta.copy();t0=theta.copy();t1[j]+=eps;t0[j]-=eps;v1,g1=objective(t1,a,y,w,hessian=False);v0,g0=objective(t0,a,y,w,hessian=False)
        assert abs((v1-v0)/(2*eps)-g[j])<1e-9;np.testing.assert_allclose((g1-g0)/(2*eps),h[:,j],rtol=0,atol=1e-9)
    offset=rng.normal(size=37);feature=rng.normal(size=37);gamma=.13;val,g,h=offset_objective(gamma,offset,feature,y,w)
    v1,g1,_=offset_objective(gamma+eps,offset,feature,y,w);v0,g0,_=offset_objective(gamma-eps,offset,feature,y,w);assert abs((v1-v0)/(2*eps)-g)<1e-9 and abs((g1-g0)/(2*eps)-h)<1e-9
    raw=np.exp2(-np.array([0,730.5,1461])/730.5);np.testing.assert_array_equal(raw,np.array([1,.5,.25]));np.testing.assert_allclose(raw/raw.sum(),(raw*17)/(raw*17).sum(),rtol=0,atol=1e-15)
    o=pd.DataFrame(dict(date=['2020-12-31','2021-01-01','2025-12-01','2025-12-20'],joint_completed=['2021-01-11','2021-01-12','2025-12-10','2026-01-01']))
    assert training_rows(o,'2025-12-31').tolist()==[1,2]
    return ['equal_weight_objective_gradient_Hessian','equal_weight_solver_matches_original','weighted_full_gradient_Hessian_finite_differences','weighted_offset_derivatives','exact_half_life','raw_weight_scale_invariance','five_year_and_joint_maturity_boundaries']

def audit_sources():
    obs=csv('observation_table');weights=csv('training_weights');summary=csv('weight_summary').set_index('cutoff');heads=read(OUT/'source_heads.json');models=read(OUT/'source_models.json');tests=read(OUT/'source_testing_features.json');base=csv('baseline_model_predictions');checks=[];maxgap=0.;total=0
    with np.load(PROJECT/'research_v5/cache/targets.npz') as t:np.testing.assert_allclose(obs.exec_return,t['returns'],rtol=0,atol=1e-12)
    annual28=[f for f in read(V28/'protocol.json')['folds'] if f['cutoff'].endswith('12-31')];assert cfg()['decision_dates']==[f['cutoff'] for f in annual28]
    for cutoff in cfg()['decision_dates']:
        year=int(cutoff[:4]);lower=f'{year-4}-01-01';ids=[i for i,r in enumerate(obs.itertuples()) if lower<=r.date and r.joint_completed<=cutoff]
        z=weights[weights.cutoff.eq(cutoff)];assert z.row_index.tolist()==ids and len(ids)==next(f for f in annual28 if f['cutoff']==cutoff)['training_counts']['rolling5'];ages=[(date.fromisoformat(cutoff)-date.fromisoformat(obs.date.iloc[i])).days for i in ids];raw=[math.exp(-math.log(2)*a/730.5) for a in ages];den=math.fsum(raw);w=np.array([a/den for a in raw]);np.testing.assert_allclose(z.raw_weight,raw,rtol=0,atol=1e-15);np.testing.assert_allclose(z.weight,w,rtol=0,atol=1e-16);assert z.age_days.tolist()==ages and z.joint_completed.max()<=cutoff
        s=summary.loc[cutoff];assert abs(s.weight_sum-1)<1e-14 and abs(s.kish_effective_n-1/math.fsum(a*a for a in w))<1e-9 and s.n==len(ids)
        assert abs(s.uniform_up_frequency-float(obs.exec_return.iloc[ids].gt(0).mean()))<1e-14 and abs(s.weighted_up_frequency-math.fsum(a*int(obs.exec_return.iloc[i]>0) for a,i in zip(w,ids)))<1e-14
        for seed in cfg()['seeds']:
            hs=[next(h for h in heads if h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in METHODS];model=next(m for m in models if m['cutoff']==cutoff and m['seed']==seed);assert sha(PROJECT/model['project_file'])==model['sha256'] and all(h['model_project_file']==model['project_file'] for h in hs)
            d=arrays(hs[0]);te=arrays(next(t for t in tests if t['cutoff']==cutoff and t['seed']==seed));np.testing.assert_array_equal(d['row_index'],ids);y=obs.exec_return.iloc[ids].gt(0).to_numpy(float);np.testing.assert_array_equal(d['direction'],y);assert np.isfinite(d['features']).all()
            for prefix,features in [('rep',d['features']),('market',d['market_features'])]:
                features=np.asarray(features,float)
                low,high=np.quantile(features,[.01,.99],axis=0);clipped=np.clip(features,low,high);mean=clipped.mean(0);sd=np.maximum(clipped.std(0),1e-6)
                for key,val in [('lower',low),('upper',high),('mean',mean),('sd',sd)]:np.testing.assert_allclose(d[prefix+'__'+key],val,rtol=0,atol=1e-12)
            t18=np.asarray(hs[0]['coefficients']);np.testing.assert_array_equal(d['vol__ray'],t18[:25]);np.testing.assert_array_equal(d['order__ray'],t18[:25])
            for arr in [d,te]:
                xx=independent_inputs(arr['features'],arr['market_features'],arr['gate'],d)
                for key,x in zip(['x18','x19','x23'],xx):gap=float(abs(x-arr[key]).max());maxgap=max(maxgap,gap);assert gap<1e-9
            expected_te=np.flatnonzero(obs.date.gt(cutoff)&obs.date.le(f'{year+1}-12-31')&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end']));np.testing.assert_array_equal(te['row_index'],expected_te)
            for j,h in enumerate(hs):
                key=['x18','x19','x23','x23'][j];theta=np.array(h['coefficients']);uniform=np.full(len(y),1/len(y))
                if j<3:val,grad,hess=objective(theta,design(d[key]),y,uniform);assert abs(grad).max()<=1.01e-9 and np.linalg.eigvalsh(hess).min()>0
                else:
                    t19=np.array(hs[1]['coefficients']);np.testing.assert_array_equal(np.r_[theta[:-2],theta[-1]],t19);val,grad,hess=offset_objective(theta[-2],design(d['x19'])@t19,d['x23'][:,-1],y,uniform);val+=.005*(t19[:-1]@t19[:-1]);assert abs(grad)<=1.01e-10
                assert abs(val-h['objective'])<1e-12
                q=base[base.history.eq(ANNUAL)&base.method.eq(h['method'])&base.seed.eq(seed)&base.cutoff.eq(cutoff)].sort_values('row_index');np.testing.assert_array_equal(q.row_index,te['row_index']);prob=probability(design(te[key])@theta);np.testing.assert_allclose(q.probability,prob,rtol=0,atol=1e-14);np.testing.assert_array_equal(q.direction_up,prob>.5);total+=1
            # A bank containing future rows and labels must expose exactly the same training interface.
            bank={key:np.concatenate([d[key],te[key]]) for key in ['row_index','x18','x19','x23']};original=training_interface(obs,bank,cutoff);poison={k:a.copy() for k,a in bank.items()};outside=~np.isin(bank['row_index'],ids)
            for key in ['x18','x19','x23']:poison[key][outside]=123456.
            bad=obs.copy();bad.loc[bad.joint_completed.gt(cutoff),'exec_return']=-999.;changed=training_interface(bad,poison,cutoff)
            for j,(a,b) in enumerate(zip(original,changed)):
                if j==1:
                    for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
                else:np.testing.assert_array_equal(a,b)
            checks.append(dict(cutoff=cutoff,seed=seed,n=len(ids),test_n=len(te['row_index']),future_rows_poisoned=int(outside.sum()),uniform_head_replay=True,frozen_designs_verified=True,training_interface_causal=True))
    assert total==72 and len(checks)==18
    return checks,maxgap

def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=synthetic();rows,gap=audit_sources();p=OUT/'source_contract_checks.csv';pd.DataFrame(rows).to_csv(p,index=False);check_frozen(False)
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],synthetic_cases=tests,annual_jobs=18,uniform_head_replays=72,maximum_frozen_design_gap=gap,artifacts={str(p.relative_to(ROOT)):sha(p)}));print('Contract PASS:weights,mature5-yearmembers,frozenfeatures,72uniformheadreplays and18future-poison checks.',flush=True)
if __name__=='__main__':main()

from common48 import *
import copy,math

def independent_inputs(f,m,u,d):
    rep=(np.clip(f,d['rep__lower'],d['rep__upper'])-d['rep__mean'])/d['rep__sd'];mk=(np.clip(m,d['market__lower'],d['market__upper'])-d['market__mean'])/d['market__sd']
    a=np.column_stack([rep,mk]);v=mk[:,1]*(rep@d['vol__ray'])-a@d['vol__projection'][:-1]-d['vol__projection'][-1];v=(v-d['vol__residual_mean'])/d['vol__residual_sd'];b=np.column_stack([a,v])
    h=u*(rep@d['order__ray'])-b@d['order__projection'][:-1]-d['order__projection'][-1];h=(h-d['order__residual_mean'])/d['order__residual_sd'];return [a,b,np.column_stack([b,h])]

def synthetic():
    u=[];w=[]
    for cutoff in cfg()['decision_dates']:
        for seed in cfg()['seeds']:
            for method in METHODS:
                a=dict(cutoff=cutoff,seed=seed,method=method,job='u',train_n=12,cache_file='same',cache_sha256='same',model_project_file='same',coefficients=[.2,-.3,.1]);u.append(a);w.append(dict(a,job='w',coefficients=[-.1,.4,-.2]))
    records=assemble_records(u,w);assert len(records)==144
    for h in records:assert h['coefficients']==([.2,-.3,-.2] if h['history']==INTERCEPT else [-.1,.4,.1])
    identical=assemble_records(u,u);assert all(h['coefficients']==[.2,-.3,.1] for h in identical)
    poison=copy.deepcopy(w)
    for h in poison:h['future_label']=123456;h['future_accuracy']=1.
    assert assemble_records(u,poison)==records
    x=np.array([[1.,-1.],[.1,.3],[-.2,.7]]);u=np.array([.2,-.3,.1]);w=np.array([-.1,.4,-.2]);z0=design(x)@u;z1=design(x)@w;i=design(x)@np.r_[u[:-1],w[-1]];s=design(x)@np.r_[w[:-1],u[-1]]
    np.testing.assert_allclose(z1-z0,w[-1]-u[-1]+x@(w[:-1]-u[:-1]),rtol=0,atol=1e-15);np.testing.assert_allclose(z1-z0,(i-z0)+(s-z0),rtol=0,atol=1e-15)
    assert (probability(np.array([-.001,0,.001]))>.5).tolist()==[False,False,True]
    for loss in [(0.,1.,0.,1.),(.2,.3,.1,.25)]:
        f=factorial_values(*loss);assert abs(f['symmetric_intercept']+f['symmetric_slopes']-f['total'])<1e-14 and abs(f['intercept_at_uniform_slopes']+f['slopes_at_uniform_intercept']+f['interaction']-f['total'])<1e-14
    return ['exact_coordinate_swaps','identical_endpoints_recover_original','unused_outcomes_cannot_affect_assembly','logit_additivity','strict_half_probability_boundary','symmetric_loss_identity']

def audit_sources():
    obs=csv('observation_table');u=read(OUT/'uniform_heads.json');w=read(OUT/'weighted_heads.json');models=read(OUT/'source_models.json');tests=read(OUT/'source_testing_features.json');base=csv('baseline_model_predictions');records=[];gapmax=0.;replays=0
    assert u==read(PREV/'results/source_heads.json') and w==read(PREV/'results/heads.json')
    for cutoff in cfg()['decision_dates']:
        year=int(cutoff[:4]);ids=np.array([i for i,r in enumerate(obs.itertuples()) if f'{year-4}-01-01'<=r.date and r.joint_completed<=cutoff]);np.testing.assert_array_equal(ids,training_rows(obs,cutoff));teids=np.flatnonzero(obs.date.gt(cutoff)&obs.date.le(f'{year+1}-12-31')&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end']))
        for seed in cfg()['seeds']:
            us=[next(h for h in u if (h['cutoff'],h['seed'],h['method'])==(cutoff,seed,m)) for m in METHODS];ws=[next(h for h in w if (h['cutoff'],h['seed'],h['method'])==(cutoff,seed,m)) for m in METHODS]
            d=arrays(us[0]);wd=arrays(ws[0]);testref=next(t for t in tests if (t['cutoff'],t['seed'])==(cutoff,seed));te=arrays(testref);model=next(m for m in models if (m['cutoff'],m['seed'])==(cutoff,seed));assert sha(PROJECT/model['project_file'])==model['sha256']
            np.testing.assert_array_equal(d['row_index'],ids);np.testing.assert_array_equal(te['row_index'],teids)
            for key in ['row_index','direction','x18','x19','x23']:np.testing.assert_array_equal(d[key],wd[key])
            np.testing.assert_array_equal(d['direction'],obs.exec_return.iloc[ids].gt(0).to_numpy(float));np.testing.assert_array_equal(d['vol__ray'],us[0]['coefficients'][:25]);np.testing.assert_array_equal(d['order__ray'],us[0]['coefficients'][:25])
            for a,b,dimension in zip(us,ws,[30,31,32,32]):
                assert len(a['coefficients'])==len(b['coefficients'])==dimension and a['train_n']==b['train_n']==len(ids)
                assert b['frozen_pipeline_file']==a['cache_file'] and b['frozen_pipeline_sha256']==a['cache_sha256'] and a['model_project_file']==b['model_project_file']==model['project_file']
            for hs in [us,ws]:np.testing.assert_array_equal(np.r_[hs[3]['coefficients'][:-2],hs[3]['coefficients'][-1]],hs[1]['coefficients'])
            for arr in [d,te]:
                xx=independent_inputs(arr['features'],arr['market_features'],arr['gate'],d)
                for key,x in zip(['x18','x19','x23'],xx):gap=float(abs(x-arr[key]).max());gapmax=max(gapmax,gap);assert gap<1e-9
            for history,hs in [(ANNUAL,us),(WEIGHTED,ws)]:
                for method,h,key in zip(METHODS,hs,['x18','x19','x23','x23']):
                    theta=np.asarray(h['coefficients']);p=probability(design(te[key])@theta);g=base[base.history.eq(history)&base.method.eq(method)&base.seed.eq(seed)&base.cutoff.eq(cutoff)].sort_values('row_index');np.testing.assert_array_equal(g.row_index,teids);np.testing.assert_allclose(g.probability,p,rtol=0,atol=1e-14);np.testing.assert_array_equal(g.direction_up,p>.5);replays+=1
            records.append(dict(cutoff=cutoff,seed=seed,train_n=len(ids),test_n=len(teids),last_joint_completed=obs.joint_completed.iloc[ids].max(),exact_shared_designs=True,source_coefficients_verified=True,endpoint_heads_replayed=8))
    assert replays==144 and len(records)==18
    return records,gapmax

def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();tests=synthetic();records,gap=audit_sources();p=OUT/'source_contract_checks.csv';pd.DataFrame(records).to_csv(p,index=False);check_frozen(False)
    save(OUT/'contract_verification.json',dict(status='PASS',completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],synthetic_cases=tests,source_jobs=18,endpoint_head_replays=144,maximum_frozen_design_gap=gap,artifacts={str(p.relative_to(ROOT)):sha(p)}));print('Contract PASS:sharedcoordinates,18matureannualjobs and144endpointheadreplays;swaps use no labels.',flush=True)
if __name__=='__main__':main()

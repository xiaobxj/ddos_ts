from common27 import *
def memberships():
    obs,price,targets=data();budgets=csv('budgets');records=[]
    # R26 already reconstructed these exact input bytes and all old recipes.
    old=read(V26/'results/verification.json');assert old['status']=='PASS' and old['data_contract']['maximum_raw_gap']==0 and old['data_contract']['maximum_auxiliary_gap']==0
    for name,digest in read(V26/'results/preparation_manifest.json')['input_sha256'].items():assert sha(PROJECT/name)==digest
    with np.load(PROJECT/'research_v5/cache/packed_raw.npz') as archive:packed={k:archive[k].copy() for k in ['patches','geometry','valid']}
    allrows=[]
    for history in HISTORIES:
        for fold in cfg()['folds']:
            tr,te=indices(obs,fold,history);lower='2010-01-01' if history=='full' else str(pd.Timestamp(fold['cutoff']).year-int(history[-1])+1)+'-01-01'
            independent=np.flatnonzero(pd.to_datetime(obs.date).ge(pd.Timestamp(lower))&pd.to_datetime(obs.joint_completed).le(pd.Timestamp(fold['cutoff'])))
            np.testing.assert_array_equal(tr,independent);allrows.extend(tr.tolist())
            assert obs.joint_completed.iloc[tr].max()<=fold['cutoff']<obs.date.iloc[te].min();assert len(tr)>700
            labels,s=scales_for(targets,tr);assert s==read(OUT/'training_scales.json')[f"{history}_{fold['cutoff']}"]
            if history in NEW:
                with np.load(CACHE/f"training_{history}_{fold['cutoff']}.npz") as d:
                    np.testing.assert_array_equal(d['row_index'],tr)
                    for k in labels:np.testing.assert_array_equal(labels[k],d[k])
                    for k in packed:np.testing.assert_array_equal(packed[k][tr],d[k])
            for seed in cfg()['seeds']:
                rng=np.random.default_rng(seed);other=np.random.default_rng(seed)
                for epoch in range(1,21):
                    a=schedule(len(tr),fold['train_n'],rng);n=fold['train_n'];chunks=[]
                    while n:
                        draw=other.permutation(len(tr));take=min(n,len(tr));chunks.append(draw[:take]);n-=take
                    b=np.concatenate(chunks);np.testing.assert_array_equal(a,b);counts=np.bincount(a,minlength=len(tr));assert len(a)==fold['train_n'] and counts.max()-counts.min()<=1
                    if history=='full':
                        original=pd.read_csv(V26/'results/training_curves.csv');r=original[original.cutoff.eq(fold['cutoff'])&original.seed.eq(seed)&original.epoch.eq(epoch)].iloc[0];assert r.order_sha256==training.array_hash(a)
                assert rng.bit_generator.state==other.bit_generator.state
            records.append(dict(history=history,cutoff=fold['cutoff'],lower_anchor_date=lower,training_rows=len(tr),testing_rows=len(te),matched_updates=20*((fold['train_n']+127)//128),causal=True))
    ids=np.unique(allrows);anchors=obs.anchor.iloc[ids].to_numpy(int);mf=market(price,obs,ids)
    bars=price[['open','high','low','close','volume']].to_numpy(float);scalar=prior.additive.scalar_descriptors(bars,anchors)[prior.additive.MARKET_NAMES].to_numpy(float);np.testing.assert_allclose(mf,scalar,atol=1e-12,rtol=1e-12)
    u=[]
    for a in anchors:
        c=price.close.iloc[a-60:a+1].to_numpy(float);s=[1 if c[j]>c[j-1] else -1 if c[j]<c[j-1] else 0 for j in range(1,61)]
        u.append(sum(s[j]*s[j-1] for j in range(1,60))/59-((sum(s)**2)-sum(v*v for v in s))/(60*59))
    np.testing.assert_allclose(gates(price,obs,ids),u,rtol=0,atol=1e-12)
    a,b=calibration_availability(obs);pd.testing.assert_frame_equal(a,csv('calibration_availability'),check_dtype=False);pd.testing.assert_frame_equal(b,csv('calibration_membership_plan'),check_dtype=False)
    return records
def synthetic_recipe():
    rng=np.random.default_rng(20260912);f=rng.normal(size=(400,25));m=rng.normal(size=(400,4));u=rng.normal(size=400);y=(rng.random(400)<.45).astype(float)
    d,coef,trace,metrics=fit_pipeline(f,m,u,y);xx=apply_pipeline(f,m,u,d)
    for x,key in zip(xx[:3],['x18','x19','x23']):np.testing.assert_allclose(x,d[key],atol=1e-12,rtol=0)
    for c,x in zip(coef,xx):assert np.isfinite(probability(design(x)@c)).all()
    return dict(synthetic_head_fits=4,projections=2,maximum_gradient=max(m['gradient_inf'] for m in metrics))
def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();rows=memberships();s=synthetic_recipe();pd.DataFrame(rows).to_csv(OUT/'membership_verification.csv',index=False)
    p=check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=p['protocol_sha256'],source_sha256=p['source_sha256'],inherited_raw_input_verification=str(V26/'results/verification.json'),new_membership_checks=18,schedule_checks=1080,synthetic=s,new_neural_fits=0))
    print('Contract PASS:18 memberships,1080 matched schedules,causal transforms and inherited source hashes.',flush=True)
if __name__=='__main__':main()

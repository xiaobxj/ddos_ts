from common28 import *

def independent_route(date,cadence):
    d=pd.Timestamp(date)
    if cadence=='annual':return f'{d.year-1}-12-31'
    month=((d.month-1)//3)*3+1
    # Signal on a quarter-end still uses the preceding quarter's model.
    return str((pd.Timestamp(d.year,month,1)-pd.Timedelta(days=1)).date())

def verify_routes(unique,pred,obs):
    for policy in NEW:
        memory,c=policy.split('_');cadence=c.removesuffix('20');g=pred[pred.history.eq(policy)]
        expected=g.date.map(lambda d:independent_route(d,cadence));np.testing.assert_array_equal(g.cutoff,expected)
        source=unique[unique.history.eq(memory)].drop(columns='history');other=g.drop(columns='history').merge(source,on=['row_index','method','seed','cutoff'],suffixes=('_used','_source'),validate='one_to_one')
        assert len(other)==4352
        for col in ['date','joint_completed','actual','year','score','probability','direction_up','actual_up']:
            a=other[col+'_used'];b=other[col+'_source']
            if col in ['date','joint_completed']:assert a.equals(b)
            else:np.testing.assert_allclose(a,b,rtol=0,atol=1e-14,equal_nan=True)

def memberships():
    obs,price,targets=data();records=[];allrows=[];scales=read(OUT/'training_scales.json');members=csv('membership');budgets=csv('budgets');route=csv('routing')
    old=read(V26/'results/verification.json');assert old['status']=='PASS' and old['data_contract']['maximum_raw_gap']==0 and old['data_contract']['maximum_auxiliary_gap']==0
    for name,digest in read(V26/'results/preparation_manifest.json')['input_sha256'].items():assert sha(PROJECT/name)==digest
    with np.load(PROJECT/'research_v5/cache/packed_raw.npz') as archive:packed={k:archive[k].copy() for k in ['patches','geometry','valid']}
    canonical=obs[(obs.weekday==4)&obs.date.between('2021-01-01',cfg()['label_end'])&obs.joint_completed.le(cfg()['label_end'])];assert len(canonical)==272
    for history in MEMORIES:
        for fold in cfg()['folds']:
            tr,te=indices(obs,fold,history);cutoff=pd.Timestamp(fold['cutoff']);years=int(history[-1]);lower=cutoff.replace(year=cutoff.year-years)+pd.Timedelta(days=1)
            independent=np.flatnonzero(pd.to_datetime(obs.date).ge(lower)&pd.to_datetime(obs.joint_completed).le(cutoff));np.testing.assert_array_equal(tr,independent)
            routes=route[(route.memory==history)&(route.cutoff==fold['cutoff'])];np.testing.assert_array_equal(te,np.sort(routes.row_index.unique()))
            for split,ids in [('training',tr),('testing_union',te)]:
                recorded=members[(members.history==history)&(members.cutoff==fold['cutoff'])&(members.split==split)];np.testing.assert_array_equal(ids,recorded.row_index)
            assert obs.joint_completed.iloc[tr].max()<=fold['cutoff']<obs.date.iloc[te].min();assert len(tr)>700
            if fold['cutoff'].endswith('12-31'):
                oldfold=next(f for f in previous_round.cfg()['folds'] if f['cutoff']==fold['cutoff']);a,b=previous_round.indices(obs,oldfold,history);np.testing.assert_array_equal(a,tr);np.testing.assert_array_equal(b,te)
            labels,s=scales_for(targets,tr);assert s==scales[f"{history}_{fold['cutoff']}"]
            for k in ['returns','auxiliary']:
                y=np.asarray(targets[k][independent],dtype=np.float64);mean=np.sum(y,axis=0)/len(y);sd=np.sqrt(np.sum((y-mean)**2,axis=0)/len(y));np.testing.assert_allclose(s[k+'_mean'],mean,rtol=0,atol=1e-14);np.testing.assert_allclose(s[k+'_sd'],np.maximum(sd,1e-6),rtol=0,atol=1e-14)
            with np.load(CACHE/f"training_{history}_{fold['cutoff']}.npz") as d:
                np.testing.assert_array_equal(d['row_index'],tr)
                for k in labels:np.testing.assert_array_equal(labels[k],d[k])
                for k in packed:np.testing.assert_array_equal(packed[k][tr],d[k])
            for seed in cfg()['seeds']:
                rng=np.random.default_rng(seed);other=np.random.default_rng(seed)
                for epoch in range(1,21):
                    a=schedule(len(tr),len(tr),rng);b=other.permutation(len(tr));np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(np.sort(a),np.arange(len(tr)))
                    chunks=training.batches(a);assert len(chunks)==(len(tr)+127)//128 and sum(map(len,chunks))==len(tr)
            allrows.extend(tr.tolist());allrows.extend(te.tolist());records.append(dict(history=history,cutoff=fold['cutoff'],lower_anchor_date=str(lower.date()),training_rows=len(tr),testing_union_rows=len(te),steps_per_seed=20*((len(tr)+127)//128),passes_per_retained_row=20,causal=True))
    for policy,g in route.groupby('history'):
        assert len(g)==272 and g.date.tolist()==canonical.date.tolist();assert g.row_index.tolist()==canonical.index.tolist()
        np.testing.assert_array_equal(g.cutoff,g.apply(lambda r:independent_route(r.date,r.cadence),axis=1));assert g.cutoff.lt(g.date).all()
    ids=np.unique(allrows);anchors=obs.anchor.iloc[ids].to_numpy(int);mf=market(price,obs,ids)
    bars=price[['open','high','low','close','volume']].to_numpy(float);scalar=prior.additive.scalar_descriptors(bars,anchors)[prior.additive.MARKET_NAMES].to_numpy(float);np.testing.assert_allclose(mf,scalar,atol=1e-12,rtol=1e-12)
    u=[]
    for a in anchors:
        c=price.close.iloc[a-60:a+1].to_numpy(float);s=[1 if c[j]>c[j-1] else -1 if c[j]<c[j-1] else 0 for j in range(1,61)]
        u.append(sum(s[j]*s[j-1] for j in range(1,60))/59-((sum(s)**2)-sum(v*v for v in s))/(60*59))
    np.testing.assert_allclose(gates(price,obs,ids),u,rtol=0,atol=1e-12)
    assert len(records)==46 and int(budgets.steps_per_seed.sum())*3==cfg()['budget']['new_optimizer_steps']
    return records

def main():
    check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();rows=memberships()
    # Synthetic recipe contract from immutable R27, no historical model fit.
    sys.path.insert(0,str(V27));from contract27 import synthetic_recipe
    s=synthetic_recipe();pd.DataFrame(rows).to_csv(OUT/'membership_verification.csv',index=False);p=check_frozen(False)
    save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=p['protocol_sha256'],source_sha256=p['source_sha256'],inherited_raw_input_verification=str(V26/'results/verification.json'),new_membership_checks=46,schedule_checks=2760,routed_dates=1088,synthetic=s,new_neural_fits=0))
    print('Contract PASS:46 memberships,2760 natural schedules,1088 independent calendar routes,causal transforms.',flush=True)
if __name__=='__main__':main()

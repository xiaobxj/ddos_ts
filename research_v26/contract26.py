from common26 import *
from scipy.optimize import check_grad
def data_contract():
    obs,price,targets=data();bars=price[['open','high','low','close','volume']].to_numpy(float);maxraw=0.;maxaux=0.
    with np.load(PROJECT/'research_v5/cache/packed_raw.npz') as d:
        assert d['valid'].all();np.testing.assert_allclose(d['geometry'][:,:,0],.04,atol=1e-8,rtol=0)
        np.testing.assert_allclose(d['geometry'][:,:,1],np.broadcast_to(np.arange(1,26)/25,(len(obs),25)),atol=3e-8,rtol=0)
        for i,row in obs.iterrows():
            a=int(row.anchor);w=bars[a-124:a+1].copy();assert len(w)==125 and np.isfinite(w).all()
            # Preserve the original downloader's 0.011-point rounding tolerance.
            assert (w[:,:4]>0).all() and (w[:,4]>=0).all() and price.valid_ohlc.iloc[a-124:a+1].all()
            assert (w[:,1]+.011>=w[:,[0,2,3]].max(axis=1)).all() and (w[:,2]-.011<=w[:,[0,1,3]].min(axis=1)).all()
            w[:,:4]=np.log(w[:,:4]);w[:,4]=np.log1p(w[:,4]);w=((w-w.mean(0))/np.maximum(w.std(0),1e-6)).astype(np.float32)
            gap=float(abs(w.reshape(25,5,5)-d['patches'][i]).max());maxraw=max(maxraw,gap);assert gap==0
            actual=price.open.iloc[int(row.exit)]/price.open.iloc[int(row.entry)]-1
            assert abs(actual-row.exec_return)<1e-12
            fut=bars[a+1:a+6];aux=np.c_[np.log(fut[:,:4]/bars[a,3]),np.log1p(fut[:,4])-np.log1p(bars[a-19:a+1,4]).mean()].reshape(-1)
            gap=float(abs(aux-targets['auxiliary'][i]).max());maxaux=max(maxaux,gap);assert gap<1e-12
            assert row.joint_completed==max(row.completed,price.date.iloc[a+5])
    for fold in cfg()['folds']:
        tr,te=indices(obs,fold);assert obs.joint_completed.iloc[tr].max()<=fold['cutoff']<obs.date.iloc[te].min()
        labels,s=scales_for(targets,tr);assert s==read(OUT/'training_scales.json')[fold['cutoff']]
        with np.load(CACHE/f"training_{fold['cutoff']}.npz") as d:
            np.testing.assert_array_equal(d['row_index'],tr)
            for k in labels:np.testing.assert_array_equal(d[k],labels[k])
        both=np.r_[tr,te];a=obs.anchor.iloc[both].to_numpy(int)
        m=market(price,obs,both);ind=additive.scalar_descriptors(bars,a)[additive.MARKET_NAMES].to_numpy(float)
        np.testing.assert_allclose(m,ind,atol=1e-12,rtol=1e-12)
        u=gates(price,obs,both);scalar=order.scalar_features(price.iloc[:a.max()+1],a)[:,-1];np.testing.assert_allclose(u,scalar,atol=1e-12,rtol=0)
        # Removing every future bar must leave both market inputs unchanged.
        np.testing.assert_array_equal(market(price.iloc[:obs.anchor.iloc[tr].max()+1],obs,tr),m[:len(tr)])
        np.testing.assert_array_equal(gates(price.iloc[:obs.anchor.iloc[tr].max()+1],obs,tr),u[:len(tr)])
    return dict(raw_windows=len(obs),maximum_raw_gap=maxraw,maximum_auxiliary_gap=maxaux,annual_folds=6)
def inherited_contract():
    obs,price,_=data();records=[]
    maps={v:{(h['cutoff'],h['seed']):h for h in read(PROJECT/f'research_v{v}/results/heads.json') if h['method'] in LEARNED} for v in [18,19,23,25]}
    sources=[h for h in read(PROJECT/'research_v16/results/heads.json') if h['kind']=='learned']
    for h in sources:
        key=(h['cutoff'],h['seed']);d=arrays(h);rows=d['row_index'];rep=fit_clip(d['features']);mk=fit_clip(market(price,obs,rows));x18=np.column_stack([rep['standardized'],mk['standardized']]);t18=np.asarray(maps[18][key]['coefficients'])
        v=order.fit_interaction(x18,mk['standardized'][:,1],t18[:25]);x19=v['standardized'];o=order.fit_interaction(x19,gates(price,obs,rows),t18[:25]);x23=o['standardized']
        gaps=[]
        for ver,x in [(18,x18),(19,x19),(23,x23)]:
            old=arrays(maps[ver][key])['standardized'];gap=float(abs(old-x).max());assert gap<1e-9;gaps.append(gap)
        t19=np.asarray(maps[19][key]['coefficients']);t25=np.asarray(maps[25][key]['coefficients']);np.testing.assert_array_equal(t19,np.r_[t25[:-2],t25[-1]])
        records.append(dict(cutoff=key[0],seed=key[1],train_n=len(rows),maximum_input_gap=max(gaps)))
    assert len(records)==18
    # One real old feature extractor parity check, using its historical batch boundaries.
    h=sources[0];model,_=neural.load_backbone(h);d=arrays(h);values=legacy.batch_tensors(d['row_index']);features,_=neural.extract_features(model,values)
    np.testing.assert_array_equal(features,d['features']);del model,values;torch.cuda.empty_cache()
    return records
def synthetic():
    rng=np.random.default_rng(7);x=rng.normal(size=(320,29));y=(rng.random(320)<.47).astype(float)
    a=design(x);theta=rng.normal(size=30)*.1
    err=check_grad(lambda t:objective(t,a,y)[0],lambda t:objective(t,a,y)[1],theta);assert err<1e-6
    t,_,m=fit_stage(x,y);assert m['gradient_inf']<1e-9
    h=rng.normal(size=320);offset=a@t;g,_=previous.fit_offset(offset,h,y,cfg()['offset_probe']);alt=previous.independent_root(offset,h,y,cfg()['offset_probe']);assert abs(g-alt['gamma'])<1e-7
    return dict(gradient_check_error=float(err),scalar_solver_gap=abs(g-alt['gamma']))
def main():
    legacy.initialize();r=check_frozen(False);assert not (OUT/'contract_verification.json').exists();start=now()
    d=data_contract();s=synthetic();old=inherited_contract();pd.DataFrame(old).to_csv(OUT/'inherited_recipe_parity.csv',index=False)
    check_frozen(False);save(OUT/'contract_verification.json',dict(status='PASS',started_utc=start,completed_utc=now(),protocol_sha256=r['protocol_sha256'],source_sha256=r['source_sha256'],data=d,synthetic=s,inherited_recipes=18,maximum_old_recipe_gap=max(x['maximum_input_gap'] for x in old),new_year_neural_fits=0,old_neural_replays=1,verification_only_synthetic_fits=2))
    print('Contract PASS: raw windows, labels, causal fold scales, old recipe parity and synthetic solvers.',flush=True)
if __name__=='__main__':main()

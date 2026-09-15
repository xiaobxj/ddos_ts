from common29 import *

def quantile_linear(values,q):
    a=np.sort(np.asarray(values,float),axis=0);t=(len(a)-1)*q;i=int(t);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(t-i)
def independent_check(obs,mf,active,decision):
    dt=pd.to_datetime(obs.date);lo=pd.Timestamp(active).replace(year=int(active[:4])-5)+pd.Timedelta(days=1)
    tr=np.flatnonzero((dt>=lo)&(pd.to_datetime(obs.joint_completed)<=pd.Timestamp(active)));h=np.flatnonzero((dt>=lo)&(dt<=pd.Timestamp(active)));current=np.flatnonzero(dt<=pd.Timestamp(decision))[-60:]
    raw=mf[tr];lower=quantile_linear(raw,.01);upper=quantile_linear(raw,.99);clipped=np.minimum(np.maximum(raw,lower),upper);mean=clipped.sum(0)/len(tr);sd=np.maximum(np.sqrt(((clipped-mean)**2).sum(0)/len(tr)),1e-6)
    transform=lambda x:(np.minimum(np.maximum(x,lower),upper)-mean)/sd
    z=transform(mf[h]);lag=int(current[-1]-h[-1]);end=np.arange(59+lag,len(h));start=end-lag
    dist=np.array([sum((z[e-59:e+1].sum(0)/60-z[s-59:s+1].sum(0)/60)**2)/4 for s,e in zip(start,end)])
    refmean=transform(mf[h[-60:]]).sum(0)/60;currentmean=transform(mf[current]).sum(0)/60;distance=float(sum((currentmean-refmean)**2)/4);threshold=float(quantile_linear(dist,.9))
    saved=dict(lower=lower,upper=upper,mean=mean,sd=sd,training_rows=tr,history_rows=h,current_rows=current,reference_rows=h[-60:],calibration_end_rows=h[end],calibration_start_rows=h[start],calibration_distances=dist,reference_mean=refmean,current_mean=currentmean)
    return saved,dict(distance=distance,threshold=threshold,ratio=distance/threshold,lag_observations=lag,calibration_n=len(dist),historical_percentile=float((dist<distance).mean()),refit=distance>threshold)

def checks():
    obs,price,_=data();events=csv('events').fillna({'previous_model_cutoff':'','reference_last_date':'','current_last_date':''});refs=read(OUT/'state_refs.json');heads=read(OUT/'source_heads.json')
    with np.load(CACHE/'observed_market.npz') as a:mf=a['features'].copy();np.testing.assert_array_equal(a['row_index'],np.arange(len(obs)))
    bars=price[['open','high','low','close','volume']].to_numpy(float);scalar=prior.additive.scalar_descriptors(bars,obs.anchor.to_numpy(int))[prior.additive.MARKET_NAMES].to_numpy(float);np.testing.assert_allclose(mf,scalar,rtol=1e-12,atol=1e-12)
    meta=obs[['date','joint_completed']];active=None;maxgap=0.;independent_events=[];audit=[]
    for e in events.itertuples():
        assert e.previous_model_cutoff==(active or '')
        if e.cutoff.endswith('12-31'):
            assert e.mandatory and e.refit and e.selected_model_cutoff==e.cutoff;active=e.cutoff
        else:
            assert not e.mandatory;d,result=independent_check(meta,scalar,active,e.cutoff);saved=arrays(next(r for r in refs if r['cutoff']==e.cutoff))
            for k,v in d.items():
                if k.endswith('_rows'):np.testing.assert_array_equal(v,saved[k])
                else:
                    gap=float(np.max(np.abs(v-saved[k])));maxgap=max(maxgap,gap);np.testing.assert_allclose(v,saved[k],rtol=1e-10,atol=1e-11)
            for k in ['distance','threshold','ratio','historical_percentile']:assert abs(result[k]-getattr(e,k))<1e-10
            assert result['refit']==e.refit and result['lag_observations']==e.lag_observations and result['calibration_n']==e.calibration_n
            assert obs.date.iloc[d['history_rows']].max()<=active and obs.date.iloc[d['current_rows']].max()<=e.cutoff and obs.joint_completed.iloc[d['training_rows']].max()<=active
            archived=arrays(next(h for h in heads if h['cutoff']==active and h['seed']==cfg()['seeds'][0] and h['method']==LEARNED[0]))
            for key in ['lower','upper','mean','sd']:np.testing.assert_allclose(saved[key],archived['market__'+key],rtol=0,atol=1e-12)
            altered=mf.copy();altered[obs.date.gt(e.cutoff)]=1234567.89;new,new_d=assess(meta,altered,active,e.cutoff)
            original,_=assess(meta,mf,active,e.cutoff);assert new==original
            for k in saved:np.testing.assert_array_equal(new_d[k],saved[k])
            selected=e.cutoff if result['refit'] else active;assert e.selected_model_cutoff==selected
            audit.append(dict(decision=e.cutoff,active_model=active,triggered=result['refit'],pairs=len(d['calibration_distances']),history_ends_before_decision=True,future_perturbation_invariant=True));active=selected
        independent_events.append((e.cutoff,active))
    assert len(events)==23 and len(audit)==17 and int(events.mandatory.sum())==6
    route=csv('routing');canonical=obs[(obs.weekday==4)&obs.date.between('2021-01-01',cfg()['label_end'])&obs.joint_completed.le(cfg()['label_end'])]
    np.testing.assert_array_equal(route.row_index,canonical.index);np.testing.assert_array_equal(route.date,canonical.date);assert len(route)==272
    for r in route.itertuples():
        possible=[(date,c) for date,c in independent_events if date<r.date];date,selected=possible[-1];assert date==r.decision_cutoff and selected==r.cutoff and r.cutoff<r.date
    exp=required_extensions(obs,events);pd.testing.assert_frame_equal(exp,csv('extension_membership'),check_dtype=False)
    selected=set(route.cutoff);models=read(OUT/'source_models.json');assert len(models)==3*len(selected) and len(heads)==12*len(selected)
    for r in models:
        assert r['cutoff'] in selected and sha(PROJECT/r['project_file'])==r['sha256'];tr=training_rows(obs,r['cutoff']);np.testing.assert_array_equal(tr,previous_round.indices(obs,fold_for(r['cutoff']),'rolling5')[0])
    old=read(V28/'results/verification.json');assert old['status']=='PASS' and old['checkpoint_replays']==138 and old['independent_head_solutions']==552
    return dict(optional_checks=17,maximum_independent_state_gap=maxgap,selected_networks=len(models),future_perturbation_checks=17,extra_inference_weeks=len(exp)),pd.DataFrame(audit)

def main():
    prep=check_frozen(False);assert not (OUT/'contract_verification.json').exists();started=now();result,audit=checks();audit.to_csv(OUT/'independent_trigger_checks.csv',index=False);check_frozen(False)
    save(OUT/'contract_verification.json',dict(status='PASS',started_utc=started,completed_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],**result,new_neural_fits=0,new_head_fits=0));print('Contract PASS: independent market states, trigger calibration, future perturbations and model routing.',flush=True)
if __name__=='__main__':main()

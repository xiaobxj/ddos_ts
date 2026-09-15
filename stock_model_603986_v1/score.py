"""Stock-only Friday reconstruction and mature out-of-sample scoring."""
from stock import *
from quarter50 import fit_quarter
from validate50 import validate_quarter

def context(frame,anchors,median):
    r,_,_=modules();obs=pd.DataFrame({'anchor':anchors});ids=np.arange(len(obs))
    mf=r.market(frame,obs,ids);gate=r.gates(frame,obs,ids)
    states=[('negative' if x[0]<0 else 'nonnegative')+('_high' if x[1]>median else '_low') for x in mf]
    return mf,gate,states

def main():
    check_freeze();assert read(OUT/'verification_models.json')['status']=='PASS'
    f=data();r,_,_=modules(True);_,obs,_=observations(f,cfg()['data_end'])
    input_good=f.valid_ohlc.rolling(125,min_periods=125).min().fillna(0).astype(bool)
    anchors=np.flatnonzero(input_good&f.date.ge(cfg()['test_start'])&pd.to_datetime(f.date).dt.dayofweek.eq(4))
    common=pd.DataFrame(dict(anchor=anchors,date=f.date.iloc[anchors].to_numpy()))
    common=common.merge(obs.drop(columns=['anchor']),on='date',how='left',validate='one_to_one')
    common['year']=common.date.str[:4].astype(int)
    seedrows=[];controls=[];contexts=[]
    for cutoff in cfg()['cutoffs']:
        folder=OUT/'annual'/cutoff;a=read(folder/'annual.json');g=common[common.year.eq(int(cutoff[:4])+1)].reset_index(drop=True)
        mf,gate,states=context(f,g.anchor.to_numpy(int),a['volatility_median'])
        for i,row in g.iterrows():contexts.append(dict(row,state=states[i],trend60=mf[i,0],volatility20=mf[i,1],encoder_cutoff=cutoff))
        values={k:r.torch.from_numpy(v).cuda() for k,v in packed(f,g.anchor.to_numpy(int)).items()}
        with np.load(folder/'market4.npz') as z:market4={k:z[k] for k in z.files}
        p4=r.probability(r.design(r.transform(mf,market4))@market4['coefficients'])
        for i,row in g.iterrows():
            for method,p,predret in [('training_frequency',a['training_up_frequency'],np.nan),('always_up',1.,np.nan),
                    ('market4',float(p4[i]),np.nan),('training_mean_return',np.nan,a['return_scale']['returns_mean'])]:
                controls.append(dict(row,history='control',method=method,probability=p,predicted_return=predret))
        for ref in a['models']:
            seed=ref['seed'];model,_=r.load_model(ref);features,native=r.neural.extract_features(model,values)
            predret=native.astype(float)*a['return_scale']['returns_sd']+a['return_scale']['returns_mean']
            hs=[next(h for h in a['uniform'] if h['seed']==seed and h['method']==m) for m in METHODS]
            ws=[next(h for h in a['weighted'] if h['seed']==seed and h['method']==m) for m in METHODS]
            d=r.arrays(hs[0]);xx=r.apply_pipeline(features,mf,gate,d)
            for method,x,h,wh in zip(METHODS,xx,hs,ws):
                t=np.array(h['coefficients']);wt=np.array(wh['coefficients'])
                for history,theta in [(U,t),(W,wt),(I,np.r_[t[:-1],wt[-1]]),(S,np.r_[wt[:-1],t[-1]])]:
                    logits=r.design(x)@theta;probs=r.probability(logits)
                    for i,row in g.iterrows():seedrows.append(dict(row,history=history,method=method,seed=seed,
                        state=states[i],encoder_cutoff=cutoff,annual_logit=float(logits[i]),probability=float(probs[i]),predicted_return=np.nan))
            for i,row in g.iterrows():seedrows.append(dict(row,history='control',method='native_mse',seed=seed,
                state=states[i],encoder_cutoff=cutoff,annual_logit=np.nan,probability=np.nan,predicted_return=float(predret[i])))
            del model;r.torch.cuda.empty_cache()
    allseed=pd.DataFrame(seedrows);bank=allseed[allseed.history.eq(U)&allseed.actual_up.notna()].copy()
    csv(OUT/'annual_stock_signal_bank.csv',bank)
    qrows=[];quarterchecks=[]
    for qcut in sorted({(pd.Timestamp(d).to_period('Q').start_time-pd.Timedelta(days=1)).strftime('%Y-%m-%d') for d in common.date}):
        result=fit_quarter(bank,qcut);check=validate_quarter(bank,qcut,result)
        save(OUT/f'quarter_{qcut}.json',result);quarterchecks.append(dict(cutoff=qcut,**check))
        qend=(pd.Timestamp(qcut)+pd.Timedelta(days=1)).to_period('Q').end_time.strftime('%Y-%m-%d')
        selected=allseed[allseed.history.eq(U)&allseed.date.gt(qcut)&allseed.date.le(qend)]
        offsets={(x['method'],x['seed'],x['state']):x['applied_offset'] for x in result['quarter_offsets']}
        for row in selected.to_dict('records'):
            delta=offsets[(row['method'],row['seed'],row['state'])]
            row.update(history=Q,quarter_cutoff=qcut,applied_offset=delta,
                probability=row['probability'] if delta==0 else float(r.probability(row['annual_logit']+delta)))
            qrows.append(row)
    allseed=pd.concat([allseed,pd.DataFrame(qrows)],ignore_index=True)
    csv(OUT/'seed_predictions.csv',allseed)
    ensemble=allseed.groupby(['date','history','method'],sort=False).agg(probability=('probability','mean'),predicted_return=('predicted_return','mean'),seed_n=('seed','nunique')).reset_index()
    assert ensemble.seed_n.eq(3).all()
    ensemble=ensemble.merge(common,on='date',how='left',validate='many_to_one')
    ensemble=pd.concat([ensemble,pd.DataFrame(controls)],ignore_index=True)
    ensemble['predicted_up']=np.where(ensemble.probability.notna(),ensemble.probability.gt(.5),ensemble.predicted_return.gt(0)).astype(int)
    ensemble['status']=np.where(ensemble.actual_up.notna(),'mature_retrospective','pending_retrospective')
    csv(OUT/'ensemble_predictions.csv',ensemble);csv(OUT/'weekly_context.csv',pd.DataFrame(contexts))
    metrics=[]
    for period,sub in [('pooled',ensemble)]+[(str(y),ensemble[ensemble.year.eq(y)]) for y in range(2023,2027)]:
        for (history,method),g in sub[sub.actual_up.notna()].groupby(['history','method']):
            y=g.actual_up.to_numpy();p=g.probability.to_numpy();pr=g.predicted_return.to_numpy()
            row=dict(period=period,history=history,method=method,weeks=len(g),correct=int((g.predicted_up==y).sum()),
                accuracy=float((g.predicted_up==y).mean()),actual_up_rate=float(y.mean()),predicted_up_rate=float(g.predicted_up.mean()))
            # Always-up is a hard direction comparator, not a calibrated p=1 forecast.
            if np.isfinite(p).all() and method!='always_up':
                p=np.clip(p,1e-12,1-1e-12);row.update(brier=float(((p-y)**2).mean()),logloss=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()))
            if np.isfinite(pr).all():
                error=pr-g.actual_return.to_numpy();row.update(mae=float(abs(error).mean()),rmse=float(np.sqrt((error**2).mean())))
            metrics.append(row)
    csv(OUT/'metrics.csv',pd.DataFrame(metrics))
    # Fixed comparisons only; no ranking or outcome-dependent model selection.
    main=ensemble[ensemble.history.eq(U)&ensemble.method.eq('learned_vol_interaction')&ensemble.actual_up.notna()].sort_values('date')
    y=main.actual_up.to_numpy();mainloss=(main.probability.to_numpy()-y)**2;n=len(main)
    rng=np.random.default_rng(20260915);starts=rng.integers(0,n,size=(10000,int(np.ceil(n/8))))
    indices=((starts[:,:,None]+np.arange(8))%n).reshape(10000,-1)[:,:n]
    comparisons=[]
    for method in ['training_frequency','market4']:
        base=ensemble[ensemble.history.eq('control')&ensemble.method.eq(method)&ensemble.actual_up.notna()].sort_values('date')
        assert main.date.tolist()==base.date.tolist()
        delta=mainloss-(base.probability.to_numpy()-y)**2;samples=delta[indices].mean(axis=1);mean=float(delta.mean())
        p=(1+int((abs(samples-mean)>=abs(mean)).sum()))/10001
        comparisons.append(dict(reference=method,n=n,brier_difference=mean,ci95_low=float(np.quantile(samples,.025)),
            ci95_high=float(np.quantile(samples,.975)),p_centered_two_sided=p))
    order=np.argsort([x['p_centered_two_sided'] for x in comparisons]);running=0.
    for rank,j in enumerate(order):
        running=max(running,min(1.,(2-rank)*comparisons[j]['p_centered_two_sided']));comparisons[j]['p_holm_two']=running
    save(OUT/'comparisons.json',dict(main_history=U,main_method='learned_vol_interaction',block_weeks=8,replicates=10000,
        interpretation='Descriptive retrospective uncertainty, not untouched confirmation or trading profitability.',comparisons=comparisons))
    save(OUT/'quarter_verification.json',dict(status='PASS',quarters=quarterchecks))
    save(OUT/'scoring_completed.json',dict(completed_utc=now(),mature_weeks=n,latest_signal=common.date.max(),
        pending_dates=common.loc[common.actual_up.isna(),'date'].tolist(),seed_rows=len(allseed),ensemble_rows=len(ensemble)))
    check_freeze();print('Scored stock weeks:',n,'latest',common.date.max(),flush=True)

if __name__=='__main__':main()

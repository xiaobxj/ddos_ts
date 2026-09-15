"""Independent checkpoint, fitting, calendar and scoring verification."""
from stock import *
from scipy.special import expit

def models():
    check_freeze();frame=data();r,w,_=modules(True);checks=[]
    for cutoff in cfg()['cutoffs']:
        folder=OUT/'annual'/cutoff;annual=read(folder/'annual.json');a=annual_interface(frame,cutoff);ids=a['training_rows']
        for name,digest in annual['artifacts'].items():assert sha(PROJECT/name)==digest,name
        assert a['scales']==annual['return_scale'] and annual['stock']=='sh603986'
        pd.testing.assert_frame_equal(load_csv(folder/'membership.csv'),a['observations'].iloc[ids].reset_index(drop=True),check_exact=True)
        _,weights,_=w.weights_for(a['training_dates'],cutoff);values={k:r.torch.from_numpy(v).cuda() for k,v in a['values'].items()}
        maxgrad=0.;featuregap=0.
        for ref in annual['models']:
            model,state=r.load_model(ref);assert state['stock']=='sh603986' and state['freeze_sha256']==sha(OUT/'freeze.json')
            assert state['scales']==a['scales'] and state['train_n']==len(ids)
            hs=[next(h for h in annual['uniform'] if h['seed']==ref['seed'] and h['method']==m) for m in METHODS]
            d=r.arrays(hs[0]);f,_=r.neural.extract_features(model,values)
            gap=float(abs(f-d['features']).max());featuregap=max(featuregap,gap);assert gap<2e-5
            np.testing.assert_array_equal(d['row_index'],ids);np.testing.assert_array_equal(d['direction'],a['y'])
            np.testing.assert_array_equal(d['market_features'],a['market']);np.testing.assert_array_equal(d['gate'],a['gate'])
            xx=r.apply_pipeline(d['features'],a['market'],a['gate'],d)
            for key,x in zip(['x18','x19','x23'],xx):np.testing.assert_allclose(d[key],x,rtol=0,atol=1e-10)
            for collection,weight in [('uniform',np.full(len(ids),1/len(ids))),('weighted',weights)]:
                hs=[next(h for h in annual[collection] if h['seed']==ref['seed'] and h['method']==m) for m in METHODS]
                for i,(h,x) in enumerate(zip(hs,xx)):
                    theta=np.asarray(h['coefficients']);design=np.c_[x,np.ones(len(x))];residual=weight*(expit(design@theta)-a['y'])
                    if i<3:
                        gradient=design.T@residual;gradient[:-1]+=.01*theta[:-1];g=float(abs(gradient).max())
                    else:
                        parent=np.asarray(hs[1]['coefficients']);np.testing.assert_array_equal(theta[:-2],parent[:-1]);assert theta[-1]==parent[-1]
                        g=abs(float(x[:,-1]@residual+.01*theta[-2]))
                    maxgrad=max(maxgrad,g);assert g<2e-9
            del model;r.torch.cuda.empty_cache()
        with np.load(folder/'market4.npz') as d:
            x=r.transform(a['market'],d);theta=d['coefficients'];gradient=np.c_[x,np.ones(len(x))].T@(expit(np.c_[x,np.ones(len(x))]@theta)-a['y'])/len(x)
            gradient[:-1]+=.01*theta[:-1];assert abs(gradient).max()<2e-9
        curve=load_csv(folder/'training_curves.csv');assert len(curve)==60
        assert curve.groupby('seed').epoch.agg(list).apply(lambda x:x==list(range(1,21))).all()
        checks.append(dict(cutoff=cutoff,train_n=len(ids),checkpoint_replays=3,heads_checked=25,max_gradient=maxgrad,max_feature_gap=featuregap))
    save(OUT/'verification_models.json',dict(status='PASS',completed_utc=now(),checks=checks))
    print('All 12 stock checkpoints and 100 heads verified',flush=True)

def scoring():
    check_freeze();p=load_csv(OUT/'ensemble_predictions.csv');s=load_csv(OUT/'seed_predictions.csv');m=load_csv(OUT/'metrics.csv');checks=[]
    mature=p[p.actual_up.notna()];common=None
    for (h,method),g in mature.groupby(['history','method']):
        dates=sorted(g.date);assert len(dates)==len(set(dates))
        if common is None:common=dates
        assert common==dates
        metric=m[m.period.eq('pooled')&m.history.eq(h)&m.method.eq(method)].iloc[0]
        assert abs(float((g.predicted_up==g.actual_up).mean())-metric.accuracy)<1e-14
        if method not in ['always_up','native_mse','training_mean_return']:
            assert abs(float(((g.probability-g.actual_up)**2).mean())-metric.brier)<1e-14
    assert set(p[p.actual_up.isna()].date)=={'2026-09-11'}
    for keys,g in s.groupby(['date','history','method']):
        assert set(g.seed)==set(SEEDS) and len(g)==3
        e=p[p.date.eq(keys[0])&p.history.eq(keys[1])&p.method.eq(keys[2])].iloc[0]
        value=g.predicted_return.mean() if keys[2]=='native_mse' else g.probability.mean()
        expected=e.predicted_return if keys[2]=='native_mse' else e.probability
        assert abs(value-expected)<1e-14
        assert g.encoder_cutoff.lt(g.date).all()
    checks.append('All 25 output combinations use identical mature weeks; independent accuracy/Brier and three-seed means')
    f=data();_,obs,_=observations(f,cfg()['data_end'])
    labels=p[['date','actual_return']].drop_duplicates().dropna().merge(obs[['date','actual_return']],on='date',suffixes=('_saved','_rebuilt'))
    np.testing.assert_array_equal(labels.actual_return_saved,labels.actual_return_rebuilt)
    checks.append('Saved stock labels reproduce source reconstruction exactly; latest Friday remains unscored')
    # Replay representative first/last dates from every annual checkpoint,
    # truncating all market data at the signal date before inference.
    from score import context
    r,_,_=modules(True);maxgap=0.;replays=0
    for cutoff in cfg()['cutoffs']:
        annual=read(OUT/'annual'/cutoff/'annual.json');dates=sorted(s.loc[s.encoder_cutoff.eq(cutoff),'date'].unique())
        for ref in annual['models']:
            model,_=r.load_model(ref)
            for date in [dates[0],dates[-1]]:
                prefix=f[f.date.le(date)].reset_index(drop=True);anchor=np.array([len(prefix)-1])
                mf,gate,states=context(prefix,anchor,annual['volatility_median'])
                values={k:r.torch.from_numpy(v).cuda() for k,v in packed(prefix,anchor).items()}
                features,_=r.neural.extract_features(model,values)
                hs=[next(h for h in annual['uniform'] if h['seed']==ref['seed'] and h['method']==m) for m in METHODS]
                xx=r.apply_pipeline(features,mf,gate,r.arrays(hs[0]))
                for method,x,h in zip(METHODS,xx,hs):
                    expected=float(expit(np.c_[x,np.ones(len(x))]@np.array(h['coefficients']))[0])
                    row=s[s.date.eq(date)&s.history.eq(U)&s.method.eq(method)&s.seed.eq(ref['seed'])].iloc[0]
                    gap=abs(row.probability-expected);maxgap=max(maxgap,gap);assert gap<2e-5 and row.state==states[0];replays+=1
            del model;r.torch.cuda.empty_cache()
    checks.append('Causal truncated-date inference replays across all annual checkpoints')
    from quarter50 import fit_quarter
    bank=load_csv(OUT/'annual_stock_signal_bank.csv')
    for qfile in sorted(OUT.glob('quarter_2*.json')):
        qcut=qfile.stem.removeprefix('quarter_');baseline=read(qfile)
        prefix=bank[bank.joint_completed.le(qcut)].copy()
        assert fit_quarter(prefix,qcut)==baseline,'Quarter consumed future stock evidence'
    checks.append('Every state calibration identical with all future rows removed')
    save(OUT/'verification.json',dict(status='PASS',completed_utc=now(),freeze_sha256=sha(OUT/'freeze.json'),checks=checks,
        model_verification_sha256=sha(OUT/'verification_models.json'),quarter_verification_sha256=sha(OUT/'quarter_verification.json'),
        common_mature_weeks=len(common),causal_prediction_replays=replays,max_probability_gap=maxgap))
    print('Stock scoring verification PASS',len(common),'common weeks',flush=True)

if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='scoring':scoring()
    else:models()

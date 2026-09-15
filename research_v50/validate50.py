"""Independent numerical gates for newly generated annual and quarterly packages."""
from common50 import *
from scipy.special import expit
from scipy.optimize import brentq
from dataset50 import annual_interface,observations

def validate_quarter(bank,cutoff,result):
    pool=bank[['date','joint_completed']].drop_duplicates().sort_values('date');past=pool[pool.joint_completed<=cutoff];val=past.tail(13)
    fit_cutoff=(pd.Timestamp(val.date.iloc[0])-pd.Timedelta(days=1)).strftime('%Y-%m-%d') if len(val) else None
    train=pool[pool.joint_completed<=fit_cutoff].tail(52) if fit_cutoff else pool.iloc[:0]
    mode='annual_reset' if cutoff.endswith('12-31') else 'ready' if len(train)==52 and len(val)==13 else 'insufficient_history'
    guard(result['train_dates']==train.date.tolist() and result['validation_dates']==val.date.tolist(),'Quarter purge mismatch')
    guard(result['schedule']['mode']==mode,'Quarter reset/readiness mismatch');maximum=0.;checks=0
    for method in METHODS:
        for state in STATES:
            ps=[];bases=[];labels=None;eligible=None;nt=None
            for seed in SEEDS:
                subset=bank[bank.method.eq(method)&bank.seed.eq(seed)&bank.state.eq(state)]
                tr=subset[subset.date.isin(train.date)];va=subset[subset.date.isin(val.date)].sort_values('date');nt=len(tr);eligible=mode=='ready' and nt>=10
                if eligible:
                    z=tr.annual_logit.to_numpy();y=tr.actual_up.to_numpy();gradient=lambda d:float(np.sum(expit(z+d)-y)+20*d)
                    delta=-.5 if gradient(-.5)>=0 else .5 if gradient(.5)<=0 else brentq(gradient,-.5,.5,xtol=1e-14)
                else:delta=0.
                h=next(h for h in result['heads'] if h['method']==method and h['seed']==seed and h['component']==state)
                gap=abs(delta-h['offset']);maximum=max(maximum,gap);guard(gap<1e-11 and h['fit_eligible']==eligible,'Quarter offset independent solve failed')
                p0=va.probability.to_numpy();p=p0.copy() if h['offset']==0 else expit(va.annual_logit.to_numpy()+h['offset']);bases.append(p0);ps.append(p);labels=va.actual_up.to_numpy();checks+=1
            p=np.mean(ps,axis=0);p0=np.mean(bases,axis=0);gain=float(np.mean((p-labels)**2-(p0-labels)**2)) if len(labels) else None
            accepted=bool(mode=='ready' and nt>=10 and len(labels)>=5 and gain< -1e-12 and np.sum((p>.5)==labels)>=np.sum((p0>.5)==labels))
            decision=next(d for d in result['decisions'] if d['method']==method and d['state']==state)
            guard(decision['accepted']==accepted,'Quarter held-out gate mismatch')
            for seed in SEEDS:
                off=next(r for r in result['quarter_offsets'] if r['method']==method and r['seed']==seed and r['state']==state)
                guard(off['accepted']==accepted and off['applied_offset']==(off['raw_offset'] if accepted else 0.),'Quarter fallback mismatch')
    return dict(status='PASS',offsets_checked=checks,gate_cells=16,maximum_offset_gap=maximum,training_n=len(train),validation_n=len(val),mode=mode)
def validate_annual(frame,annual):
    r,w,_=modules(True);cutoff=annual['annual_cutoff'];a=annual_interface(frame,cutoff);ids=a['training_rows'];maxgrad=0.;featuregap=0.;checked=0
    guard(a['scales']==annual['return_scale'],'Annual target scale mismatch')
    guard(float(a['y'].mean())==annual['training_up_frequency'],'Annual frequency mismatch')
    guard(float(np.quantile(a['market'][:,1],.5,method='linear'))==annual['volatility_median'],'Annual state threshold mismatch')
    _,weights,_=w.weights_for(a['training_dates'],cutoff);values={k:r.torch.from_numpy(v).cuda() for k,v in a['values'].items()}
    for modelref in annual['models']:
        seed=modelref['seed'];model,state=r.load_model(modelref)
        guard(state['scales']==a['scales'] and state['train_n']==len(ids),'Checkpoint training metadata mismatch')
        f,_=r.neural.extract_features(model,values);h0=next(h for h in annual['uniform'] if h['seed']==seed);d=r.arrays(h0)
        np.testing.assert_array_equal(d['row_index'],ids);np.testing.assert_array_equal(d['direction'],a['y']);gap=float(np.max(abs(f-d['features'])));featuregap=max(featuregap,gap);guard(gap<2e-5,'Checkpoint feature replay failed')
        for name,current in [('market_features',a['market']),('gate',a['gate'])]:np.testing.assert_array_equal(d[name],current)
        xx=r.apply_pipeline(d['features'],a['market'],a['gate'],d)
        for key,x in zip(['x18','x19','x23'],xx):np.testing.assert_allclose(d[key],x,rtol=0,atol=1e-10)
        for collection,weight in [('uniform',np.full(len(ids),1/len(ids))),('weighted',weights)]:
            hs=[next(h for h in annual[collection] if h['seed']==seed and h['method']==method) for method in METHODS]
            for index,(h,x) in enumerate(zip(hs,xx)):
                theta=np.array(h['coefficients']);design=np.c_[x,np.ones(len(x))];residual=weight*(expit(design@theta)-a['y'])
                if index<3:
                    gradient=design.T@residual;gradient[:-1]+=.01*theta[:-1];g=float(abs(gradient).max())
                else:
                    parent=np.asarray(hs[1]['coefficients']);np.testing.assert_array_equal(theta[:-2],parent[:-1]);guard(theta[-1]==parent[-1],'R25 parent intercept changed');g=abs(float(x[:,-1]@residual+.01*theta[-2]))
                maxgrad=max(maxgrad,g);guard(g<2e-9,'Independent head stationarity failed');checked+=1
        del model; r.torch.cuda.empty_cache()
    return dict(status='PASS',training_n=len(ids),heads_checked=checked,maximum_gradient=maxgrad,maximum_feature_gap=featuregap,checkpoint_replays=3)
def validate_package_schema(p):
    guard(set(m['seed'] for m in p['models'])==set(SEEDS) and len(p['models'])==3,'Incomplete model set')
    for name in ['uniform','weighted']:
        guard(len(p[name])==12 and {(h['method'],h['seed']) for h in p[name]}=={(m,s) for m in METHODS for s in SEEDS},'Incomplete head set')
        for h in p[name]:guard(h['cutoff']==p['annual_cutoff'] and np.isfinite(h['coefficients']).all(),'Invalid annual head')
    guard(len(p['quarter_offsets'])==48 and {(h['method'],h['seed'],h['state']) for h in p['quarter_offsets']}=={(m,s,t) for m in METHODS for s in SEEDS for t in STATES},'Incomplete quarterly offsets')
    for o in p['quarter_offsets']:
        guard(abs(o['raw_offset'])<=.5 and o['applied_offset']==(o['raw_offset'] if o['accepted'] else 0.),'Invalid quarterly applied offset')
    guard(p['annual_cutoff']==annual_for(p['valid_from']) and p['quarter_cutoff']==quarter_for(p['valid_from']) and p['valid_until']==quarter_end(p['quarter_cutoff']),'Invalid package routing')
    for n,h in p['dependencies'].items():guard(sha(checked_path(n))==h,f'Changed package dependency: {n}')

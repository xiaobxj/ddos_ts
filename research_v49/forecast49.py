"""Label-free inference from a complete observed prefix and a fixed parameter package."""
from common49 import *
from data49 import prefix

def legacy_modules():
    sys.path.insert(0,str(PROJECT/'research_v26'))
    import common26 as legacy
    return legacy
def package_check(package,date,at=None,production=True):
    guard(package['annual_cutoff']==annual_for(date),'Annual parameter package is stale')
    guard(package['quarter_cutoff']==quarter_for(date),'Quarter parameter package is stale')
    guard(package['valid_from']<=date<=package['valid_until'],'Parameter package expired')
    if production:
        guard(at is not None and datetime.fromisoformat(package['assembled_utc'])<=at,'Package assembled in the future')
        guard(date in cfg()['signal_slots'],'Date is outside prospective cohort')
    for n,h in package['dependencies'].items():guard(sha(PROJECT/n)==h,f'Package dependency changed: {n}')
def validate_forecast(result):
    expected={(h,m,s) for h in HISTORIES for m in METHODS for s in SEEDS}
    rows=result['seed_predictions'];keys=[(r['history'],r['method'],r['seed']) for r in rows]
    guard(len(keys)==len(set(keys)) and set(keys)==expected,'Incomplete seed prediction matrix')
    for r in rows:
        p=r['probability'];guard(isinstance(p,(float,int)) and np.isfinite(p) and 0<=p<=1,'Invalid prediction probability')
        guard(r['direction_up']==int(p>.5) and np.isfinite(r['logit']),'Direction/probability mismatch')
    expected={(h,m) for h in HISTORIES for m in METHODS};ers=result['ensemble_predictions']
    keys=[(r['history'],r['method']) for r in ers]
    guard(len(keys)==len(set(keys)) and set(keys)==expected,'Incomplete ensemble prediction matrix')
    for r in ers:
        g=[next(s['probability'] for s in rows if (s['history'],s['method'],s['seed'])==(r['history'],r['method'],seed)) for seed in SEEDS]
        guard(r['probability']==float(pd.Series(g).mean()) and r['direction_up']==int(r['probability']>.5),'Ensemble averaging changed')
    guard(result['state'] in STATES,'Unknown market state')
def inputs(frame,date):
    f=prefix(frame,date);w=f.iloc[-125:][['open','high','low','close','volume']].to_numpy(float,copy=True)
    w[:,:4]=np.log(w[:,:4]);w[:,4]=np.log1p(w[:,4]);w=((w-w.mean(0))/np.maximum(w.std(0),1e-6)).astype(np.float32)
    patches=w.reshape(1,25,5,5);geometry=np.zeros((1,25,2),np.float32)
    for j in range(25):geometry[0,j]=[5/125,(j+1)*5/125]
    return f,dict(patches=patches,geometry=geometry,valid=np.ones((1,25),bool))
class Engine:
    def __init__(self,package):
        self.p=package;self.legacy=legacy_modules();self.legacy.legacy.initialize()
        self.models={};self.pipelines={}
        for m in package['models']:
            self.models[m['seed']]=self.legacy.load_model(m)[0]
            ref=next(h for h in package['uniform'] if h['seed']==m['seed'])
            self.pipelines[m['seed']]=self.legacy.arrays(ref)
    def predict(self,frame,date,production=True):
        package_check(self.p,date,utc(),production);f,v=inputs(frame,date);old=self.legacy
        obs=pd.DataFrame(dict(anchor=[len(f)-1]));market=old.market(f,obs,[0]);gate=old.gates(f,obs,[0])
        state=('negative' if market[0,0]<0 else 'nonnegative')+('_low' if market[0,1]<=self.p['volatility_median'] else '_high')
        values={k:old.torch.from_numpy(a).cuda() for k,a in v.items()};seed_rows=[];native=[];diagnostics=[]
        for seed in SEEDS:
            model=self.models[seed];features,_=old.neural.extract_features(model,values)
            z_native=old.neural.archival_native_output(model,values)
            native.append(float(z_native[0])*self.p['return_scale']['returns_sd']+self.p['return_scale']['returns_mean'])
            xx=old.apply_pipeline(features,market,gate,self.pipelines[seed])
            for method,x in zip(METHODS,xx):
                uh=next(h for h in self.p['uniform'] if h['seed']==seed and h['method']==method)
                wh=next(h for h in self.p['weighted'] if h['seed']==seed and h['method']==method)
                u=np.asarray(uh['coefficients']);w=np.asarray(wh['coefficients']);a=np.r_[x[0],1.]
                theta={U:u,Q:u,W:w,I:np.r_[u[:-1],w[-1]],S:np.r_[w[:-1],u[-1]]}
                base_z=float(a@u);base_p=float(old.probability(np.array([base_z]))[0])
                q=next(q for q in self.p['quarter_offsets'] if q['method']==method and q['seed']==seed and q['state']==state)
                for history in HISTORIES:
                    z=float(a@theta[history]);delta=q['applied_offset'] if history==Q else 0.
                    p=base_p if history==Q and delta==0 else float(old.probability(np.array([z+delta]))[0])
                    seed_rows.append(dict(history=history,method=method,seed=seed,probability=p,direction_up=int(p>.5),logit=z+delta))
                diagnostics.append(dict(seed=seed,method=method,features=features[0].astype(float).tolist(),design=x[0].tolist()))
        table=pd.DataFrame(seed_rows);ensemble=[]
        for history in HISTORIES:
            for method in METHODS:
                g=table[table.history.eq(history)&table.method.eq(method)]
                guard(g.seed.tolist()==SEEDS,'Missing or reordered seeds');p=float(g.probability.mean())
                ensemble.append(dict(history=history,method=method,probability=p,direction_up=int(p>.5)))
        control_p=self.p['training_up_frequency'];native_mean=float(pd.Series(native).mean())
        result=dict(signal_date=date,state=state,market_features=market[0].tolist(),order_gate=float(gate[0]),seed_predictions=seed_rows,ensemble_predictions=ensemble,controls=dict(native_returns_by_seed=dict(zip(map(str,SEEDS),native)),native_mean_return=native_mean,native_direction_up=int(native_mean>0),training_frequency=control_p,training_frequency_up=int(control_p>.5)),inference_diagnostics=diagnostics)
        validate_forecast(result);return result

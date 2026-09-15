"""Annual historical extension of the frozen R18/R19/R23/R25 learned recipes."""
from pathlib import Path
import sys,json,time,importlib.util
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache'
sys.path.insert(0,str(PROJECT/'research_v25'))
import common25 as previous
import common12 as training
import common15 as neural
import common18 as additive
import common23 as order
import evaluate6 as statistics
np=previous.np;pd=previous.pd;torch=neural.torch;legacy=neural.legacy
read=previous.read;save=previous.save;sha=previous.sha;metric=previous.metric
design=previous.design;probability=previous.probability;objective=neural.objective;fit_newton=neural.fit_newton
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset','native_mse','training_frequency']
LEARNED=METHODS[:4];PRIMARY=METHODS[1:4]
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def source_hashes():
    r=previous.source_hashes()
    paths=list(ROOT.glob('*.py'))+[PROJECT/'research_v4/data.py',PROJECT/'research_v5/prepare.py',PROJECT/'research/download_tencent.py']
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def input_hashes():
    paths=[PROJECT/n for n in ['research/data/1_000300.csv','research_v5/results/observation_table.csv','research_v5/cache/packed_raw.npz','research_v5/cache/targets.npz','research_v6/protocol.json','research_v15/protocol.json','research_v25/protocol.json','research_v25/results/delivery_manifest.json']]
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
def old_evidence():
    r=dict(read(PROJECT/'research_v25/results/preparation_manifest.json')['old_evidence'])
    for n,d in read(PROJECT/'research_v25/results/delivery_manifest.json')['files'].items():r[str((PROJECT/'research_v25'/n).relative_to(PROJECT))]=d
    p=PROJECT/'research_v25/results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p)
    assert len(r)==3672
    archive=PROJECT/'research_v26_attempt1';preserved=read(archive/'preservation.json')
    for n,d in preserved['files'].items():r[str((archive/n).relative_to(PROJECT))]=d
    path=archive/'preservation.json';r[str(path.relative_to(PROJECT))]=sha(path);assert len(r)==3697
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    r=read(OUT/'preparation_manifest.json')
    assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['source_sha256']==source_hashes() and r['input_sha256']==input_hashes()
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:assert read(OUT/'contract_verification.json')['status']=='PASS'
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def data():
    obs=pd.read_csv(PROJECT/'research_v5/results/observation_table.csv');price=pd.read_csv(PROJECT/'research/data/1_000300.csv')
    with np.load(PROJECT/'research_v5/cache/targets.npz') as d:targets={k:d[k].copy() for k in d.files}
    np.testing.assert_array_equal(obs.date,price.date.iloc[obs.anchor]);np.testing.assert_allclose(obs.exec_return,targets['returns'],rtol=0,atol=1e-12)
    return obs,price,targets
def indices(obs,fold):
    tr=np.flatnonzero(obs.joint_completed.le(fold['cutoff']))
    te=np.flatnonzero(obs.date.gt(fold['cutoff'])&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end']))
    assert len(tr)==fold['train_n'] and len(te)==fold['test_n'] and len(np.intersect1d(tr,te))==0
    return tr,te
def scales_for(targets,rows):
    labels={};scales={}
    for k in ['returns','auxiliary']:
        y=targets[k][rows].astype(float);mean=y.mean(axis=0);sd=np.maximum(y.std(axis=0),1e-6)
        labels[k]=((y-mean)/sd).astype(np.float32);scales[k+'_mean']=np.asarray(mean).tolist();scales[k+'_sd']=np.asarray(sd).tolist()
    return labels,scales
def ref(path):return dict(cache_file=str(path.relative_to(PROJECT)),cache_sha256=sha(path))
def arrays(ref):
    p=PROJECT/ref['cache_file'];assert sha(p)==ref['cache_sha256']
    with np.load(p) as d:return {k:d[k].copy() for k in d.files}
def market(price,obs,rows):return additive.descriptors(price[['open','high','low','close','volume']].to_numpy(float),obs.anchor.iloc[rows].to_numpy(int))[additive.MARKET_NAMES].to_numpy(float)
def gates(price,obs,rows):
    # The strict sample mask already requires 125 valid bars. Sign changes use only 61 trailing closes.
    a=obs.anchor.iloc[rows].to_numpy(int);close=price.close.to_numpy(float)
    s=np.sign(close[a[:,None]-np.arange(59,-1,-1)]-close[a[:,None]-np.arange(60,0,-1)])
    return (s[:,:-1]*s[:,1:]).mean(axis=1)-(s.sum(axis=1)**2-(s*s).sum(axis=1))/(60*59)
def fit_clip(f):return additive.fit_market(f)
def transform(f,d):return additive.transform(f,d)
def fit_stage(x,y):
    theta,trace=fit_newton(x,y,cfg()['probe']);value,g,h=objective(theta,design(x),y,.01)
    assert abs(g).max()<=1e-9 and np.linalg.eigvalsh(h).min()>0
    return theta,trace,dict(objective=value,gradient_inf=float(abs(g).max()),hessian_min=float(np.linalg.eigvalsh(h).min()))
def fit_pipeline(features,market_features,u,y):
    rep=fit_clip(features);mk=fit_clip(market_features);x18=np.column_stack([rep['standardized'],mk['standardized']])
    t18,tr18,m18=fit_stage(x18,y);ray=t18[:25]
    v=order.fit_interaction(x18,mk['standardized'][:,1],ray);x19=v['standardized'];t19,tr19,m19=fit_stage(x19,y)
    o=order.fit_interaction(x19,u,ray);x23=o['standardized'];t23,tr23,m23=fit_stage(x23,y)
    offset=design(x19)@t19;gamma,tr25=previous.fit_offset(offset,x23[:,-1],y,cfg()['offset_probe'])
    t25=np.r_[t19[:-1],gamma,t19[-1]];val,gr,he=previous.offset_objective(gamma,offset,x23[:,-1],y,.01)
    m25=dict(objective=val+.005*float(t19[:-1]@t19[:-1]),reduced_objective=val,gradient_inf=abs(gr),hessian_min=he)
    assert m23['objective']<=m25['objective']+1e-12<=m19['objective']+2e-12 and m19['objective']<=m18['objective']+1e-12
    for d in [v,o]:assert d['training_orthogonality_inf']<1e-8 and d['residual_raw_sd']>1e-6 and d['augmented_design_rank']==d['base_design_rank']+1
    saved=dict(features=features,market_features=market_features,gate=u,direction=y,x18=x18,x19=x19,x23=x23)
    for prefix,d in [('rep',rep),('market',mk),('vol',v),('order',o)]:saved.update({prefix+'__'+k:val for k,val in d.items()})
    coefs=[t18,t19,t23,t25];traces=[tr18,tr19,tr23,tr25];metrics=[m18,m19,m23,m25]
    return saved,coefs,traces,metrics
def part(d,prefix):return {k[len(prefix)+2:]:v for k,v in d.items() if k.startswith(prefix+'__')}
def apply_pipeline(features,market_features,u,d):
    x18=np.column_stack([transform(features,part(d,'rep')),transform(market_features,part(d,'market'))])
    v=transform(market_features,part(d,'market'))[:,1]
    x19,_=order.apply_interaction(x18,v,part(d,'vol'));x23,_=order.apply_interaction(x19,u,part(d,'order'))
    return [x18,x19,x23,x23]
def load_training(cutoff):
    with np.load(CACHE/f'training_{cutoff}.npz') as d:
        values={k:torch.from_numpy(d[k].copy()).cuda() for k in ['patches','geometry','valid']}
        labels={k:torch.from_numpy(d[k].copy()).cuda() for k in ['returns','auxiliary']}
        return values,labels,d['row_index'].copy()
def load_model(r):
    assert sha(PROJECT/r['project_file'])==r['sha256'];state=torch.load(PROJECT/r['project_file'],map_location='cpu',weights_only=True)
    assert state['cutoff']==r['cutoff'] and state['seed']==r['seed'] and state['epoch']==20
    assert training.object_hash(state['state_dict'])==r['model_sha256']
    model=legacy.make_model('combined',r['seed']);model.load_state_dict(state['state_dict']);model.eval();model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==38551
    return model,state
def forecasts(model,rows,scale):
    values=legacy.batch_tensors(rows);f,_=neural.extract_features(model,values)
    native=neural.archival_native_output(model,values).astype(float)*scale['returns_sd']+scale['returns_mean']
    return f,native
def rows_for_predictions(obs,rows,cutoff,method,seed,score,prob=None):
    p=np.full(len(rows),np.nan) if prob is None else np.asarray(prob,float)
    result=obs.iloc[rows][['date','joint_completed','exec_return']].copy().rename(columns={'exec_return':'actual'})
    result.insert(0,'row_index',rows);result['year']=result.date.str[:4].astype(int);result['cutoff']=cutoff;result['method']=method;result['seed']=seed
    result['score']=score;result['probability']=p;result['direction_up']=(np.asarray(score)>0 if prob is None else p>.5).astype(int);result['actual_up']=(result.actual>0).astype(int)
    return result
def ensemble_from(models):
    rows=[]
    for (method,date),g in models.groupby(['method','date'],sort=False):
        assert len(g)==(1 if method=='training_frequency' else 3)
        r=g.iloc[0].to_dict();r.pop('seed');r['score']=float(g.score.mean());r['probability']=float(g.probability.mean()) if method!='native_mse' else np.nan
        r['direction_up']=int(r['score']>0 if method=='native_mse' else r['probability']>.5);rows.append(r)
    return pd.DataFrame(rows)

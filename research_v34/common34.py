"""Causal quarterly endpoints in immutable annual coordinates, with frozen step fractions."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache'
V33=PROJECT/'research_v33';V32=PROJECT/'research_v32';V31=PROJECT/'research_v31';V28=PROJECT/'research_v28'
sys.path.insert(0,str(V33));import common33 as previous_round
base32=previous_round.previous_round;prior=base32.prior
np=prior.np;pd=prior.pd;torch=prior.torch;legacy=prior.legacy;neural=prior.neural;training=prior.training
read=prior.read;save=prior.save;sha=prior.sha;metric=prior.metric;data=prior.data;statistics=prior.statistics;arrays=prior.arrays;ref=prior.ref;market=prior.market;gates=prior.gates
apply_pipeline=prior.apply_pipeline;design=prior.design;probability=prior.probability;objective=prior.objective
METHODS=prior.METHODS;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY
NEW=['fixed_transform_step25','fixed_transform_step50','fixed_transform_step100'];HISTORIES=base32.HISTORIES+NEW
training_rows=base32.training_rows;ensemble_from=base32.ensemble_from;annual_for=base32.annual_for;bank_slice=base32.bank_slice

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V33/'protocol.json']+[V33/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']]
    paths += [V32/'results'/n for n in ['jobs.csv','membership.csv','bank_membership.csv','source_models.json','source_testing_features.json']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V33/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V33/'results/delivery_manifest.json')['files'].items():r[str((V33/n).relative_to(PROJECT))]=d
    p=V33/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==4946
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    r=read(OUT/'preparation_manifest.json');assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['source_sha256']==source_hashes() and r['input_sha256']==input_hashes()
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==r['source_sha256'] and c['protocol_sha256']==r['protocol_sha256']
        for n,d in c['artifacts'].items():assert sha(OUT/n)==d,n
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def baselines():return tuple(pd.read_csv(V32/'results'/n,float_precision='round_trip') for n in ['model_predictions.csv','ensemble_predictions.csv'])
def shrink(a,b,alpha):
    a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape and 0<=alpha<=1
    if alpha==0:return a.copy()
    if alpha==1:return b.copy()
    return a+alpha*(b-a)
def training_interface(bank,rows,mf,u,y,annual):
    # Select only causal members before applying frozen annual transformations.
    f=bank_slice(bank,rows);return apply_pipeline(f,mf[rows],u[rows],annual),y[rows]
def fit_coefficients(xx,y):
    ts=[];traces=[];metrics=[]
    for x in xx[:3]:
        t,trace=neural.fit_newton(x,y,cfg()['probe']);val,gr,he=objective(t,design(x),y,.01)
        assert abs(gr).max()<=1e-9 and np.linalg.eigvalsh(he).min()>0
        ts.append(t);traces.append(trace);metrics.append(dict(objective=val,gradient_inf=float(abs(gr).max()),hessian_min=float(np.linalg.eigvalsh(he).min())))
    offset=design(xx[1])@ts[1];gamma,trace=prior.previous.fit_offset(offset,xx[3][:,-1],y,cfg()['offset_probe'])
    t=np.r_[ts[1][:-1],gamma,ts[1][-1]];val,gr,he=prior.previous.offset_objective(gamma,offset,xx[3][:,-1],y,.01)
    ts.append(t);traces.append(trace);metrics.append(dict(objective=val+.005*float(ts[1][:-1]@ts[1][:-1]),reduced_objective=val,gradient_inf=abs(gr),hessian_min=he))
    assert abs(gr)<=1e-10
    return ts,traces,metrics
def candidate_predictions(obs,heads,banks,route,base):
    parts=[];sources=[];_,price,_=data()
    for policy in cfg()['updates']:
        history=policy['history']
        for cutoff in cfg()['decision_dates']:
            rows=route[route.head_cutoff.eq(cutoff)].row_index.to_numpy(int);annual=annual_for(cutoff)
            if cutoff==annual:
                g=base[base.history.eq('rolling5_annual20')&base.method.isin(LEARNED)&base.row_index.isin(rows)].copy();g['history']=history;parts.append(g)
            else:
                for seed in cfg()['seeds']:
                    hs=[next(h for h in heads if h['history']==history and h['cutoff']==cutoff and h['seed']==seed and h['method']==m) for m in LEARNED]
                    bank=arrays(next(b for b in banks if b['cutoff']==annual and b['seed']==seed));f=bank_slice(bank,rows);xx=apply_pipeline(f,market(price,obs,rows),gates(price,obs,rows),arrays(hs[0]))
                    for h,x in zip(hs,xx):
                        p=probability(design(x)@np.asarray(h['coefficients']));g=prior.rows_for_predictions(obs,rows,cutoff,h['method'],seed,p,p);g.insert(0,'history',history);parts.append(g)
            for seed in cfg()['seeds']:sources.append(dict(history=history,alpha=policy['alpha'],head_cutoff=cutoff,encoder_cutoff=annual,transform_cutoff=annual,seed=seed,test_n=len(rows),head_source='annual_reuse' if cutoff==annual else 'fraction_of_quarter_endpoint'))
        for method,source in [('native_mse','rolling5_annual20'),('training_frequency','rolling5_quarterly20')]:
            g=base[base.history.eq(source)&base.method.eq(method)].copy();g['history']=history;parts.append(g)
    r=pd.concat(parts,ignore_index=True);assert len(r)==13056 and not r.duplicated(['history','method','seed','date']).any();return r,pd.DataFrame(sources)

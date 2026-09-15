"""Fixed intercept/slope swaps between verified annual equal-weight and time-weighted heads."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';PREV=PROJECT/'research_v47';V28=PROJECT/'research_v28'
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset'];PRIMARY=METHODS[1:]
STATES=['negative_low','negative_high','nonnegative_low','nonnegative_high']
ANNUAL='rolling5_annual20';QUARTER='weekly_state_validated';WEIGHTED='annual_head_timeweight2y';INTERCEPT='annual_head_weighted_intercept';SLOPES='annual_head_weighted_slopes';NEW=[INTERCEPT,SLOPES];REPORT_HIST=[ANNUAL,WEIGHTED,INTERCEPT,SLOPES,QUARTER];REFS=[ANNUAL,WEIGHTED,QUARTER]
COPIES=[(PREV/'results'/f'{a}.csv',f'{b}.csv') for a,b in [('model_predictions','baseline_model_predictions'),('ensemble_predictions','baseline_ensemble_predictions'),('ensemble_metrics','baseline_metrics'),('weekly_context','weekly_context')]]+[(PROJECT/'research_v5/results/observation_table.csv','observation_table.csv')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def cfg():return read(ROOT/'protocol.json')
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def source_hashes():
    r=dict(read(PREV/'results/preparation_manifest.json')['source_sha256']);r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.glob('*.py')});return r
def source_inventory():
    uniform=read(PREV/'results/source_heads.json');weighted=read(PREV/'results/heads.json');models=read(PREV/'results/source_models.json');tests=read(PREV/'results/source_testing_features.json')
    assert len(uniform)==len(weighted)==72 and len(models)==len(tests)==18
    return uniform,weighted,models,tests
def input_hashes():
    uniform,weighted,models,tests=source_inventory();paths=[a for a,b in COPIES]+[PREV/'protocol.json',V28/'protocol.json']
    paths += [PREV/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md','source_heads.json','heads.json','source_models.json','source_testing_features.json','training_weights.csv','weight_summary.csv']]
    paths += [PROJECT/h['cache_file'] for h in uniform+weighted]+[PROJECT/t['cache_file'] for t in tests]+[PROJECT/m['project_file'] for m in models]
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
def old_evidence():
    r=dict(read(PREV/'results/preparation_manifest.json')['old_evidence'])
    r.update({str((PREV/n).relative_to(PROJECT)):h for n,h in read(PREV/'results/delivery_manifest.json')['files'].items()})
    p=PREV/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==6143
    for n,h in r.items():assert sha(PROJECT/n)==h,n
    return r

def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['source_sha256']==source_hashes() and p['protocol_sha256']==sha(ROOT/'protocol.json') and p['input_sha256']==input_hashes()
    for n,h in p['artifacts'].items():assert sha(ROOT/n)==h,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==p['source_sha256'] and c['protocol_sha256']==p['protocol_sha256']
        for n,h in c['artifacts'].items():assert sha(ROOT/n)==h,n
    return p
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r['finished_utc']
    for n,h in r['artifacts'].items():assert sha(ROOT/n)==h,n
    return r

def arrays(ref):
    path=PROJECT/ref['cache_file'];assert sha(path)==ref['cache_sha256']
    with np.load(path) as d:return {k:d[k].copy() for k in d.files}
def training_rows(obs,cutoff):
    lower=(pd.Timestamp(cutoff)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return np.flatnonzero(obs.date.ge(lower)&obs.joint_completed.le(cutoff))
def prediction_rows(obs,ids,history,cutoff,method,seed,p):
    r=obs.iloc[ids][['date','joint_completed','exec_return']].copy().rename(columns={'exec_return':'actual'})
    r.insert(0,'row_index',ids);r.insert(0,'history',history);r['year']=r.date.str[:4].astype(int);r['cutoff']=cutoff;r['method']=method;r['seed']=seed;r['score']=p;r['probability']=p;r['direction_up']=(p>.5).astype(int);r['actual_up']=(r.actual>0).astype(int)
    return r
def probability(z):return np.exp(-np.logaddexp(0.,-np.asarray(z,float)))
def design(x):return np.column_stack([np.asarray(x,float),np.ones(len(x))])
def assemble_records(uniform,weighted):
    ui={(h['cutoff'],h['seed'],h['method']):h for h in uniform};wi={(h['cutoff'],h['seed'],h['method']):h for h in weighted};assert set(ui)==set(wi);rows=[]
    for cutoff in cfg()['decision_dates']:
        for seed in cfg()['seeds']:
            for method in METHODS:
                key=(cutoff,seed,method);u=ui[key];w=wi[key];a=np.asarray(u['coefficients']);b=np.asarray(w['coefficients']);assert a.shape==b.shape
                for history,theta in [(INTERCEPT,np.r_[a[:-1],b[-1]]),(SLOPES,np.r_[b[:-1],a[-1]])]:
                    rows.append(dict(history=history,cutoff=cutoff,seed=seed,method=method,job=f'{history}_{method}_{cutoff}_{seed}',coefficients=theta.tolist(),uniform_job=u['job'],weighted_job=w['job'],train_n=u['train_n'],frozen_pipeline_file=u['cache_file'],frozen_pipeline_sha256=u['cache_sha256'],model_project_file=u['model_project_file'],slope_source='uniform' if history==INTERCEPT else 'weighted',intercept_source='weighted' if history==INTERCEPT else 'uniform'))
    return rows
def loss_values(row,metric):return float(row.direction_up!=row.actual_up) if metric=='direction_error' else float((row.probability-row.actual_up)**2)
def factorial_values(u,i,s,w):
    total=w-u;interaction=w-i-s+u;phi_i=.5*((i-u)+(w-s));phi_s=.5*((s-u)+(w-i))
    return dict(total=total,intercept_at_uniform_slopes=i-u,slopes_at_uniform_intercept=s-u,intercept_at_weighted_slopes=w-s,slopes_at_weighted_intercept=w-i,interaction=interaction,symmetric_intercept=phi_i,symmetric_slopes=phi_s)
def periods():return cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end=cfg()['label_end'],n=272)]+[dict(name=f'year_{y}',start=f'{y}-01-01',end=f'{y}-12-31') for y in range(2021,2027)]
def metric(g):
    n=len(g);y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool);pos=int(y.sum());neg=n-pos;tp=int((y&up).sum());tn=int((~y&~up).sum());fp=int((~y&up).sum());fn=int((y&~up).sum())
    r=dict(n=n,accuracy=(tp+tn)/n if n else None,direction_error=(fp+fn)/n if n else None,correct_directions=tp+tn,balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None,predicted_up_fraction=float(up.mean()) if n else None,observed_up_fraction=float(y.mean()) if n else None,tp=tp,tn=tn,fp=fp,fn=fn,auroc=None,brier=None,log_loss=None,mean_probability=None,probability_std=None,calibration_gap=None,clipped_probabilities=None)
    if pos and neg:
        ranks=pd.Series(g.score.to_numpy(float)).rank(method='average').to_numpy();r['auroc']=float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))
    if n and g.probability.notna().all():
        p=g.probability.to_numpy(float);q=np.clip(p,1e-12,1-1e-12);yf=y.astype(float);r.update(brier=float(np.mean((p-y)**2)),log_loss=float(np.mean(-yf*np.log(q)-(1-yf)*np.log1p(-q))),mean_probability=float(p.mean()),probability_std=float(p.std()),calibration_gap=float(p.mean()-y.mean()),clipped_probabilities=int(((p<1e-12)|(p>1-1e-12)).sum()))
    return r
def bootstrap_indices(n):
    starts=np.random.default_rng(20260910).integers(n,size=(10000,int(np.ceil(n/8))));return ((starts[:,:,None]+np.arange(8))%n).reshape(10000,-1)[:,:n]
def difference(v,ids):
    v=np.asarray(v,float);mean=v.mean();boot=v[ids].mean(1);center=(v-mean)[ids].mean(1)
    return dict(difference=float(mean),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),p=float((1+(abs(center)>=abs(mean)).sum())/10001))
def holm(ps):
    result=np.empty(len(ps));last=0.
    for rank,i in enumerate(np.argsort(ps)):last=max(last,min(1.,ps[i]*(len(ps)-rank)));result[i]=last
    return result.tolist()

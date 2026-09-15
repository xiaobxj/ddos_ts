"""Two fixed coarser quarterly calibration groups with immutable annual models."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';PREV=PROJECT/'research_v44';Q39=PROJECT/'research_v39'
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset'];PRIMARY=METHODS[1:]
STATES=['negative_low','negative_high','nonnegative_low','nonnegative_high'];FAMILIES={'trend':['negative','nonnegative'],'volatility':['low','high']}
POLICIES={'trend':'quarter_trend2_validated','volatility':'quarter_volatility2_validated'};NEW=list(POLICIES.values());ANNUAL='rolling5_annual20';QUARTER='weekly_state_validated';REFS=[QUARTER,ANNUAL];REPORT_HIST=[ANNUAL,QUARTER]+NEW
COPIES=[(PREV/'results'/f'{a}.csv',f'{b}.csv') for a,b in [('model_predictions','baseline_model_predictions'),('ensemble_predictions','baseline_ensemble_predictions'),('ensemble_metrics','baseline_metrics'),('weekly_signal_bank','weekly_signal_bank')]]+[(Q39/'results'/f'{a}.csv',f'{b}.csv') for a,b in [('schedule','schedule'),('split_membership','split_membership'),('routing','routing'),('weekly_context','weekly_context'),('gate_decisions','original_quarter_gates'),('state_support','original_four_state_support')]]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def cfg():return read(ROOT/'protocol.json')
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def source_hashes():
    r=dict(read(PREV/'results/preparation_manifest.json')['source_sha256']);r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.glob('*.py')});return r
def input_hashes():
    paths=[a for a,b in COPIES]+[PREV/'protocol.json']+[PREV/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md']]
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}
def old_evidence():
    r=dict(read(PREV/'results/preparation_manifest.json')['old_evidence']);r.update({str((PREV/n).relative_to(PROJECT)):h for n,h in read(PREV/'results/delivery_manifest.json')['files'].items()});p=PREV/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5848
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
def component(state,family):return state.split('_')[0] if family=='trend' else state.split('_')[1]
def quarter_for(date):return (pd.Timestamp(date)-pd.offsets.QuarterEnd(1)).strftime('%Y-%m-%d')
def probability(z):return np.exp(-np.logaddexp(0.,-np.asarray(z,float)))
def corrected(z,p0,d):return np.where(np.asarray(d)==0,np.asarray(p0),probability(np.asarray(z)+d))
def select(bank,ids,method,seed):return bank[bank.method.eq(method)&bank.seed.eq(seed)].set_index('row_index').loc[list(ids)].reset_index()
def solve(z,y):
    z=np.asarray(z,float);y=np.asarray(y,float);lo=-.5;hi=.5
    def grad(d):return float((probability(z+d)-y).sum()+20*d)
    if grad(lo)>=0:d=lo;status='lower_bound'
    elif grad(hi)<=0:d=hi;status='upper_bound'
    else:
        for _ in range(60):
            mid=(lo+hi)/2
            if grad(mid)>0:hi=mid
            else:lo=mid
        d=(lo+hi)/2;status='interior'
    q=z+d;objective=float(np.where(y>0,np.logaddexp(0.,-q),np.logaddexp(0.,q)).sum()+10*d*d)
    return dict(offset=float(d),status=status,objective=objective,gradient=grad(d))
def fit_records(bank,members,schedule):
    heads=[]
    for c in schedule.itertuples():
        ids=members[members.cutoff.eq(c.cutoff)&members.role.eq('training')].row_index.tolist()
        for m in METHODS:
            for seed in cfg()['seeds']:
                g=select(bank,ids,m,seed)
                for family,groups in FAMILIES.items():
                    for group in groups:
                        x=g[g.state.map(lambda s:component(s,family)).eq(group)];n=len(x);eligible=c.mode=='ready' and n>=10
                        result=solve(x.annual_logit,x.actual_up) if eligible else dict(offset=0.,status=c.mode if c.mode!='ready' else 'insufficient_state_train',objective=None,gradient=None)
                        heads.append(dict(cutoff=c.cutoff,fit_cutoff=c.fit_cutoff if isinstance(c.fit_cutoff,str) else None,mode=c.mode,method=m,seed=seed,family=family,component=group,training_n=n,training_up_n=int(x.actual_up.sum()),fit_eligible=eligible,**result))
    return heads
def gate(mode,nt,p0,p,y):
    n=len(y);bd=float(np.mean((p-y)**2-(p0-y)**2)) if n else None;bc=int(((p0>.5)==y).sum());cc=int(((p>.5)==y).sum())
    reason=mode if mode!='ready' else 'insufficient_state_train' if nt<10 else 'insufficient_validation' if n<5 else 'brier_not_better' if not bd < -1e-12 else 'direction_worse' if cc<bc else 'accepted'
    return dict(accepted=reason=='accepted',reason=reason,validation_n=n,baseline_correct=bc,candidate_correct=cc,brier_difference=bd)
def validate_records(bank,members,schedule,heads):
    hi={(h['cutoff'],h['method'],h['seed'],h['family'],h['component']):h for h in heads};gates=[];rows=[]
    for c in schedule.itertuples():
        ids=members[members.cutoff.eq(c.cutoff)&members.role.eq('validation')].row_index.tolist()
        for m in METHODS:
            for family,groups in FAMILIES.items():
                for group in groups:
                    ps=[];bases=[]
                    for seed in cfg()['seeds']:
                        g=select(bank,ids,m,seed);g=g[g.state.map(lambda s:component(s,family)).eq(group)];h=hi[(c.cutoff,m,seed,family,group)];p=corrected(g.annual_logit,g.probability,h['offset']);ps.append(p);bases.append(g.probability.to_numpy())
                        for r,q in zip(g.itertuples(),p):rows.append(dict(cutoff=c.cutoff,method=m,seed=seed,family=family,component=group,row_index=r.row_index,date=r.date,joint_completed=r.joint_completed,state=r.state,annual_probability=r.probability,candidate_probability=float(q),actual_up=int(r.actual_up),offset=h['offset']))
                    p=np.mean(ps,axis=0);p0=np.mean(bases,axis=0);gates.append(dict(cutoff=c.cutoff,mode=c.mode,method=m,family=family,component=group,training_n=h['training_n'],**gate(c.mode,h['training_n'],p0,p,g.actual_up.to_numpy())))
    return pd.DataFrame(gates),pd.DataFrame(rows)
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

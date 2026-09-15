"""Frozen annual observable states and member coverage, without new prediction policies."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V35=PROJECT/'research_v35'
sys.path.insert(0,str(V35));import common35 as previous_round
prior=previous_round.prior;np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;arrays=prior.arrays;metric=prior.metric
ARMS=previous_round.ARMS;LEARNED=previous_round.LEARNED;PRIMARY=previous_round.PRIMARY;EFFECTS=previous_round.EFFECTS
training_rows=previous_round.training_rows;annual_for=previous_round.annual_for;bank_slice=previous_round.bank_slice;member_sets=previous_round.member_sets
STATES=['negative_low','negative_high','nonnegative_low','nonnegative_high'];NAMES=['trend60','volatility20','range20','volume_change20','order60'];ROLES=['annual','added','removed','retained','both']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V35/'protocol.json']+[V35/'results'/n for n in ['delivery_manifest.json','preparation_manifest.json','verification.json','结果解读与下一步.md','source_heads.json','feature_banks.json','routing.csv','membership_summary.csv','set_changes.csv','model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','yearly_metrics.csv','weekly_effects.csv','primary_comparisons.json']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V35/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V35/'results/delivery_manifest.json')['files'].items():r[str((V35/n).relative_to(PROJECT))]=d
    p=V35/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5250
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
        for n,d in c['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def raw_data():
    obs=pd.read_csv(PROJECT/'research_v5/results/observation_table.csv',usecols=['date','anchor','weekday','joint_completed']);price=pd.read_csv(PROJECT/'research/data/1_000300.csv');np.testing.assert_array_equal(obs.date,price.date.iloc[obs.anchor]);return obs,price
def descriptor_matrix(obs,price):
    ids=np.arange(len(obs));return np.column_stack([prior.market(price,obs,ids),prior.gates(price,obs,ids)])
def roles(obs,cutoff):
    sets=member_sets(obs,cutoff);a=sets['annual'];q=sets['both'];return dict(annual=a,added=np.setdiff1d(q,a),removed=np.setdiff1d(a,q),retained=np.intersect1d(a,q),both=q)
def memberships():
    obs,_=raw_data();route=csv('routing');members=[];needed={c:set() for c in cfg()['decision_dates'] if c.endswith('12-31')}
    for cutoff in cfg()['decision_dates']:
        annual=annual_for(cutoff)
        for role,ids in roles(obs,cutoff).items():
            needed[annual].update(ids.tolist())
            for i in ids:members.append(dict(cutoff=cutoff,encoder_cutoff=annual,role=role,row_index=int(i),date=obs.date.iloc[i],joint_completed=obs.joint_completed.iloc[i]))
    for r in route.itertuples():needed[r.encoder_cutoff].add(r.row_index)
    required=[dict(encoder_cutoff=a,row_index=int(i),date=obs.date.iloc[i]) for a,ids in sorted(needed.items()) for i in sorted(ids)]
    return pd.DataFrame(members),pd.DataFrame(required)
def calibrate(raw,rows):
    # Deliberately no outcome argument; only annual selected raw descriptors.
    x=np.asarray(raw[rows],float);lo,hi=np.quantile(x,[.01,.99],axis=0,method='linear')
    return dict(volatility_median=float(np.quantile(x[:,1],.5,method='linear')),lower=lo.tolist(),upper=hi.tolist(),mean=x.mean(0).tolist(),sd=np.maximum(x.std(0),1e-6).tolist())
def assign(raw,threshold):
    negative=raw[:,0]<0;low=raw[:,1]<=threshold['volatility_median'];return np.where(negative,np.where(low,STATES[0],STATES[1]),np.where(low,STATES[2],STATES[3]))
def coverage(x,lower,upper,mean,sd):
    x=np.asarray(x,float);outside=(x<lower)|(x>upper);return dict(outside_any=outside.any(1).astype(float),outside_fraction=outside.mean(1),mean_squared_raw_z=(((x-mean)/sd)**2).mean(1)),outside
def label_decomposition(annual_n,annual_up,target_n,target_up):
    an=np.asarray(annual_n,float);au=np.asarray(annual_up,float);tn=np.asarray(target_n,float);tu=np.asarray(target_up,float)
    if tn.sum()==0:return dict(status='empty_target',overall_delta=np.nan,composition=np.nan,within_state=np.nan)
    delta=float(tu.sum()/tn.sum()-au.sum()/an.sum())
    if np.any((an==0)&(tn>0)):return dict(status='target_state_absent_in_annual',overall_delta=delta,composition=np.nan,within_state=np.nan)
    ra=np.divide(au,an,out=np.zeros_like(au),where=an>0);rt=np.divide(tu,tn,out=np.zeros_like(tu),where=tn>0);wa=an/an.sum();wt=tn/tn.sum()
    comp=float(((wt-wa)*ra).sum());within=float((wt*(rt-ra)).sum());assert abs(comp+within-delta)<1e-14
    return dict(status='defined',overall_delta=delta,composition=comp,within_state=within)
def periods():return cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end=cfg()['label_end'])]+[dict(name=f'year_{y}',start=f'{y}-01-01',end=f'{y}-12-31') for y in range(2021,2027)]

"""Read-only residual-transfer diagnostics of the frozen R37 experiment."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V37=PROJECT/'research_v37'
sys.path.insert(0,str(V37));import common37 as previous_round
prior=previous_round.prior;np=prior.np;pd=prior.pd;read=prior.read;save=prior.save;sha=prior.sha;data=prior.data;arrays=prior.arrays;probability=prior.probability;metric=prior.metric
STATES=previous_round.STATES;LEARNED=previous_round.LEARNED;PRIMARY=previous_round.PRIMARY;ANNUAL=previous_round.ANNUAL;POLICIES=previous_round.POLICIES
VIEWS=['daily','friday','future'];SCOPES=['all']+STATES
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def seeds():return cfg()['seeds']+[-1]
def periods():return cfg()['windows']+[dict(name='pooled_2021_2026',start='2021-01-01',end=cfg()['label_end'],n=272)]+[dict(name=f'year_{y}',start=f'{y}-01-01',end=f'{y}-12-31') for y in range(2021,2027)]
def period_years(p):return range(int(p['start'][:4]),int(p['end'][:4])+1)
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes();paths=[V37/'protocol.json']+[V37/'results'/n for n in ['preparation_manifest.json','delivery_manifest.json','verification.json','结果解读与下一步.md','training_inputs.json','correction_heads.json','correction_parameters.csv','routing.csv','state_thresholds.json','state_observations.csv','weekly_context.csv','model_predictions.csv','ensemble_predictions.csv','ensemble_metrics.csv','primary_comparisons.json','seed_correction_effects.csv','weekly_correction_effects.csv']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V37/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V37/'results/delivery_manifest.json')['files'].items():r[str((V37/n).relative_to(PROJECT))]=d
    p=V37/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==5428
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==p['source_sha256'] and c['protocol_sha256']==p['protocol_sha256']
        for n,d in c['artifacts'].items():assert sha(ROOT/n)==d,n
    return p
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def sign(v):return None if pd.isna(v) else 0 if abs(v)<=cfg()['settings']['residual_zero_tolerance'] else 1 if v>0 else -1
def agreement(a,b):
    a=sign(a);b=sign(b);return 'undefined' if a is None or b is None else 'near_zero' if a==0 or b==0 else 'same' if a==b else 'opposite'
def overlap(g):
    n=len(g)
    if not n:return dict(n=0,unique_intervals=0,total_edge_uses=0,union_edges=0,mean_edge_multiplicity=None,maximum_edge_multiplicity=0,overlapping_pairs=0,maximum_disjoint_intervals=0)
    a=g.entry.to_numpy(int);b=g.exit.to_numpy(int);assert (b>a).all();diff=np.zeros(int(b.max())+1,int);np.add.at(diff,a,1);np.add.at(diff,b,-1);counts=np.cumsum(diff);union=int((counts>0).sum());pairs=int(np.triu(np.maximum(a[:,None],a[None,:])<np.minimum(b[:,None],b[None,:]),1).sum());end=-1;disjoint=0
    for j in np.lexsort((a,b)):
        if a[j]>=end:disjoint+=1;end=int(b[j])
    return dict(n=n,unique_intervals=len(set(zip(a,b))),total_edge_uses=int((b-a).sum()),union_edges=union,mean_edge_multiplicity=float((b-a).sum()/union),maximum_edge_multiplicity=int(counts.max()),overlapping_pairs=pairs,maximum_disjoint_intervals=disjoint)
def decomposition(sn,ss,tn,ts):
    sn=np.asarray(sn,float);ss=np.asarray(ss,float);tn=np.asarray(tn,float);ts=np.asarray(ts,float)
    if not sn.sum() or not tn.sum():return dict(status='empty_source' if not sn.sum() else 'empty_target',overall_delta=None,composition=None,within_state=None)
    delta=float(ts.sum()/tn.sum()-ss.sum()/sn.sum())
    if ((tn>0)&(sn==0)).any():return dict(status='target_state_absent_in_source',overall_delta=delta,composition=None,within_state=None)
    rs=np.divide(ss,sn,out=np.zeros(4),where=sn>0);rt=np.divide(ts,tn,out=np.zeros(4),where=tn>0);ws=sn/sn.sum();wt=tn/tn.sum();c=float(((wt-ws)*rs).sum());w=float((wt*(rt-rs)).sum());assert abs(c+w-delta)<1e-12
    return dict(status='defined',overall_delta=delta,composition=c,within_state=w)
def ranking(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);i,j=np.triu_indices(len(a),1);s=np.sign(a[i]-a[j]);t=np.sign(b[i]-b[j]);return dict(all_pairs=len(i),inversions=int((s*t<0).sum()),tie_changes=int(((s==0)!=(t==0)).sum()))
def brier_terms(p,q,y):
    delta=q-p;e=y-p;size=delta**2;alignment=-2*delta*e;change=(q-y)**2-(p-y)**2;np.testing.assert_allclose(size+alignment,change,rtol=0,atol=1e-14);return delta,e,size,alignment,change

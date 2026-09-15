"""Fixed-window, natural-20-pass annual versus quarterly historical WFO."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V26=PROJECT/'research_v26';V27=PROJECT/'research_v27'
sys.path.insert(0,str(V27));import common27 as previous_round
prior=previous_round.prior
np=prior.np;pd=prior.pd;torch=prior.torch;legacy=prior.legacy;training=prior.training;neural=prior.neural
read=prior.read;save=prior.save;sha=prior.sha;metric=prior.metric;design=prior.design;probability=prior.probability;objective=prior.objective
data=prior.data;scales_for=prior.scales_for;arrays=prior.arrays;ref=prior.ref;market=prior.market;gates=prior.gates
fit_pipeline=prior.fit_pipeline;apply_pipeline=prior.apply_pipeline;fit_clip=prior.fit_clip;load_model=prior.load_model;forecasts=prior.forecasts
statistics=prior.statistics;METHODS=prior.METHODS;LEARNED=prior.LEARNED;PRIMARY=prior.PRIMARY
MEMORIES=['rolling5','rolling3']
NEW=[h+'_'+c+'20' for h in MEMORIES for c in ['annual','quarterly']]
HISTORIES=['full','rolling5_matched','rolling3_matched']+NEW

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def source_hashes():
    r=previous_round.source_hashes();r.update({str(p.relative_to(PROJECT)):sha(p) for p in ROOT.rglob('*.py')});return r
def input_hashes():
    r=previous_round.input_hashes()
    paths=[V27/'protocol.json']+[V27/'results'/n for n in ['model_predictions.csv','ensemble_predictions.csv','verification.json','delivery_manifest.json']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def old_evidence():
    r=dict(read(V27/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V27/'results/delivery_manifest.json')['files'].items():r[str((V27/n).relative_to(PROJECT))]=d
    p=V27/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p);assert len(r)==4002
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json')
    assert p['protocol_sha256']==sha(ROOT/'protocol.json') and p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:assert read(OUT/'contract_verification.json')['status']=='PASS'
    return p
def check_phase(phase):
    p=read(OUT/f'{phase}_manifest.json');assert p.get('finished_utc')
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    return p
def indices(obs,fold,history):
    cutoff=fold['cutoff'];lower=fold['lower'][history]
    tr=np.flatnonzero(obs.date.ge(lower)&obs.joint_completed.le(cutoff))
    te=np.flatnonzero(obs.date.gt(cutoff)&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end']))
    assert len(tr)==fold['training_counts'][history] and len(te)==fold['test_n'] and not len(np.intersect1d(tr,te))
    return tr,te
def load_training(history,cutoff):
    with np.load(CACHE/f'training_{history}_{cutoff}.npz') as d:
        values={k:torch.from_numpy(d[k].copy()).cuda() for k in ['patches','geometry','valid']};labels={k:torch.from_numpy(d[k].copy()).cuda() for k in ['returns','auxiliary']}
        return values,labels,d['row_index'].copy()
def schedule(count,presentations,rng):
    assert count==presentations
    return rng.permutation(count)
def predict_rows(obs,rows,history,cutoff,method,seed,score,prob=None):
    r=prior.rows_for_predictions(obs,rows,cutoff,method,seed,score,prob);r.insert(0,'history',history);return r
def ensemble_from(models):return previous_round.ensemble_from(models)
def frozen_baselines():
    m=pd.read_csv(V27/'results/model_predictions.csv',float_precision='round_trip');e=pd.read_csv(V27/'results/ensemble_predictions.csv',float_precision='round_trip')
    rename={'rolling5':'rolling5_matched','rolling3':'rolling3_matched'}
    for d in [m,e]:d['history']=d.history.replace(rename)
    return m,e
def routing(obs):
    ids=np.flatnonzero(obs.date.ge('2021-01-01')&obs.date.le(cfg()['label_end'])&obs.weekday.eq(4)&obs.joint_completed.le(cfg()['label_end']))
    assert len(ids)==272
    rows=[]
    for history in MEMORIES:
        for cadence in ['annual','quarterly']:
            for i in ids:
                date=obs.date.iloc[i];eligible=[f['cutoff'] for f in cfg()['folds'] if f['cutoff']<date and (cadence=='quarterly' or f['cutoff'].endswith('12-31'))]
                rows.append(dict(history=history+'_'+cadence+'20',memory=history,cadence=cadence,row_index=int(i),date=date,cutoff=max(eligible)))
    return pd.DataFrame(rows)
def route_predictions(unique):
    route=csv('routing');parts=[]
    for policy in NEW:
        r=route[route.history.eq(policy)];g=unique[unique.history.eq(r.memory.iloc[0])].drop(columns='history').merge(r[['row_index','date','cutoff']],on=['row_index','date','cutoff'],how='inner',validate='many_to_one');g.insert(0,'history',policy)
        assert len(g)==4352 and not g.duplicated(['method','seed','date']).any();parts.append(g)
    return pd.concat(parts,ignore_index=True)

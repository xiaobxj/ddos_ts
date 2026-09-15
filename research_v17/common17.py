"""Round17: immutable round16 features, training-only clipping and causal states."""
from pathlib import Path
import sys,json,time,shutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
V16=PROJECT/'research_v16'
OUT=ROOT/'results'
CACHE=ROOT/'cache'
sys.path.insert(0,str(V16))
import common16 as previous
np=previous.np;pd=previous.pd;torch=previous.torch;legacy=previous.legacy
read=previous.read;save=previous.save;sha=previous.sha
probability=previous.probability;normalize_train=previous.normalize_train;design=previous.design
objective=previous.objective;fit_newton=previous.fit_newton;probability_losses=previous.probability_losses
from states17 import descriptors,scalar_descriptors,fit_thresholds,assign_states,PARTITIONS

CANDIDATES={'learned_clip':'learned_probe','raw25_clip':'raw25_probe'}
METHODS=['native_mse','learned_probe','learned_clip','raw25_probe','raw25_clip','training_frequency','neutral_50']
CORE=['common17.py','states17.py','prepare17.py','contract17.py','train17.py','score17.py','evaluate17.py','verify17.py']

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def load_npz(ref):
    p=PROJECT/ref['cache_file'];assert sha(p)==ref['cache_sha256']
    with np.load(p) as d:return {k:d[k].copy() for k in d.files}
def old_evidence():
    r=dict(read(V16/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V16/'results/delivery_manifest.json')['files'].items():r[str((V16/n).relative_to(PROJECT))]=d
    p=V16/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p)
    assert len(r)==2716
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def source_hashes():
    r=previous.source_hashes()
    r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE})
    return r
def input_hashes():
    files=[previous.PRICE,previous.V5/'cache/targets.npz',previous.V5/'cache/packed_raw.npz',V16/'protocol.json']
    files += [V16/'results'/n for n in ['heads.json','validation_features.json','classification_baselines.csv','observation_table.csv',
        'model_predictions.csv','ensemble_predictions.csv','historical_date_usage.csv','verification.json','delivery_manifest.json']]
    for h in read(V16/'results/heads.json'):
        files.append(PROJECT/h['cache_file'])
        if h['kind']=='learned':files.append(PROJECT/h['project_file'])
    files += [PROJECT/r['cache_file'] for r in read(V16/'results/validation_features.json')]
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}
def manifest(phase):
    return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__)
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra)
    save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    r=read(OUT/'preparation_manifest.json')
    assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['source_sha256']==source_hashes() and r['input_sha256']==input_hashes()
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==r['source_sha256']
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def data():return previous.data()
def indices(obs,fold):return previous.indices(obs,fold)
def fit_clip(f):
    f=np.asarray(f,float);q=cfg()['clipping']['quantiles']
    low,high=np.quantile(f,q,axis=0,method='linear');clipped=np.clip(f,low,high)
    x,mean,sd=normalize_train(clipped,cfg()['standardization']['sd_floor'])
    return dict(lower=low,upper=high,mean=mean,sd=sd,standardized=x,clipped_features=clipped)
def apply_clip(f,d):return (np.clip(np.asarray(f,float),d['lower'],d['upper'])-d['mean'])/d['sd']
def metric(g):
    # The inherited metric assumes both label classes. Keep sparse diagnostic cells explicit.
    if len(g) and g.actual_up.nunique()==2:return previous.metric(g)
    if not len(g):
        return dict(n=0,accuracy=None,direction_error=None,correct_directions=0,balanced_accuracy=None,
            predicted_up_fraction=None,observed_up_fraction=None,tp=0,tn=0,fp=0,fn=0,auroc=None,brier=None,
            log_loss=None,mean_probability=None,probability_std=None,calibration_gap=None,clipped_probabilities=None)
    y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool)
    r=dict(n=len(g),accuracy=float((up==y).mean()),direction_error=float((up!=y).mean()),correct_directions=int((up==y).sum()),
        balanced_accuracy=None,predicted_up_fraction=float(up.mean()),observed_up_fraction=float(y.mean()),
        tp=int((up&y).sum()),tn=int((~up&~y).sum()),fp=int((up&~y).sum()),fn=int((~up&y).sum()),auroc=None,
        brier=None,log_loss=None,mean_probability=None,probability_std=None,calibration_gap=None,clipped_probabilities=None)
    if g.probability.notna().all():
        p=g.probability.to_numpy();b,l=probability_losses(p,y)
        r.update(brier=float(b.mean()),log_loss=float(l.mean()),mean_probability=float(p.mean()),probability_std=float(p.std()),
            calibration_gap=float(p.mean()-y.mean()),clipped_probabilities=int(((p<1e-12)|(p>1-1e-12)).sum()))
    return r

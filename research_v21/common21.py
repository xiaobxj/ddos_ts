"""Frozen-past two-state volatility recognition and one shared signal modifier."""
from pathlib import Path
import sys,json,time,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V20=PROJECT/'research_v20';V19=PROJECT/'research_v19';V18=PROJECT/'research_v18'
sys.path.insert(0,str(V20));import common20 as previous
np=previous.np;pd=previous.pd;read=previous.read;save=previous.save;sha=previous.sha;load_npz=previous.load_npz
probability=previous.probability;design=previous.design;objective=previous.objective;fit_newton=previous.fit_newton
metric=previous.metric;probability_losses=previous.probability_losses;PARTITIONS=previous.PARTITIONS
from hmm21 import daily_returns,forward,fit_hmm,likelihood_gradient
fit_interaction=previous.previous.fit_interaction;apply_interaction=previous.previous.apply_interaction
CANDIDATES={'learned_hmm_interaction':'learned_market','raw_hmm_interaction':'raw_trend'}
OLD_INTERACTIONS={'learned_hmm_interaction':'learned_vol_interaction','raw_hmm_interaction':'raw_vol_interaction'}
METHODS=previous.METHODS+list(CANDIDATES)+['hmm_only'];SEED_METHODS=previous.SEED_METHODS+['learned_hmm_interaction']
CORE=['hmm21.py','audit_hmm21.py','common21.py','prepare21.py','contract21.py','train21.py','score21.py','evaluate21.py','verify21.py']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def data():return previous.data()
def indices(obs,fold):return previous.indices(obs,fold)
def old_evidence():
    r=dict(read(V20/'results/preparation_manifest.json')['old_evidence'])
    for name,digest in read(V20/'results/delivery_manifest.json')['files'].items():r[str((V20/name).relative_to(PROJECT))]=digest
    path=V20/'results/delivery_manifest.json';r[str(path.relative_to(PROJECT))]=sha(path);assert len(r)==3061
    for name,digest in r.items():assert sha(PROJECT/name)==digest,name
    return r
def source_hashes():
    r=previous.source_hashes();r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE});return r
def input_hashes():
    r=previous.input_hashes();files=[V20/'protocol.json']+[V20/'results'/n for n in ['model_predictions.csv','ensemble_predictions.csv','classification_baselines.csv',
        'training_states.csv','validation_states.csv','historical_date_usage.csv','verification.json','delivery_manifest.json']]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in files});return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),
    executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,device='CPU float64; HMM and classifier research only')
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    r=read(OUT/'preparation_manifest.json');assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['source_sha256']==source_hashes() and r['input_sha256']==input_hashes()
    for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==r['source_sha256'] and c['protocol_sha256']==r['protocol_sha256']
        for name,digest in c['artifacts'].items():assert sha(ROOT/name)==digest,name
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for name,digest in r['artifacts'].items():assert sha(ROOT/name)==digest,name
    return r
def base_inputs(source,split):return previous.source_inputs(source,18,split)
def buckets(prob):return np.where(np.asarray(prob)<.2,'low_probability',np.where(np.asarray(prob)>.8,'high_probability','uncertain'))

def daily_extent(price,fold):
    a=int(np.flatnonzero(price.date.le(fold['cutoff']))[-1]);b=int(np.flatnonzero(price.date.le(fold['end']))[-1]);return a,b

def reference(path):return dict(cache_file=str(path.relative_to(PROJECT)),cache_sha256=sha(path))

def training_gate(context,obs,rows):
    return 2*load_npz(context)['filtered'][obs.anchor.iloc[rows].to_numpy(int)-1,1]-1

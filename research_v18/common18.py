"""Shared additive heads with fixed, causal continuous market inputs."""
from pathlib import Path
import sys,json,time,shutil

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;V17=PROJECT/'research_v17';V16=PROJECT/'research_v16'
OUT=ROOT/'results';CACHE=ROOT/'cache'
sys.path.insert(0,str(V17))
import common17 as previous
base16=previous.previous
np=previous.np;pd=previous.pd
read=previous.read;save=previous.save;sha=previous.sha;load_npz=previous.load_npz
probability=previous.probability;normalize_train=previous.normalize_train;design=previous.design
objective=previous.objective;fit_newton=previous.fit_newton;metric=previous.metric;probability_losses=previous.probability_losses
descriptors=previous.descriptors;scalar_descriptors=previous.scalar_descriptors;PARTITIONS=previous.PARTITIONS
MARKET_NAMES=['trend60','volatility20','range20','volume_change20']
DUPLICATES={1:11,2:13,3:14}
CANDIDATES={'learned_market':'learned_clip','raw_trend':'raw25_clip'}
NEW_METHODS=['learned_market','raw_trend','market4']
METHODS=previous.METHODS+NEW_METHODS
CORE=['common18.py','prepare18.py','contract18.py','train18.py','score18.py','evaluate18.py','verify18.py']

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def data():return previous.data()
def indices(obs,fold):return previous.indices(obs,fold)
def old_evidence():
    r=dict(read(V17/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V17/'results/delivery_manifest.json')['files'].items():r[str((V17/n).relative_to(PROJECT))]=d
    p=V17/'results/delivery_manifest.json';r[str(p.relative_to(PROJECT))]=sha(p)
    assert len(r)==2795
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def source_hashes():
    r=previous.source_hashes();r[str((V17/'verify17.py').relative_to(PROJECT))]=sha(V17/'verify17.py')
    r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE});return r
def input_hashes():
    r=previous.input_hashes();files=[V17/'protocol.json']
    files += [V17/'results'/n for n in ['heads.json','source_heads.json','training_states.csv','validation_states.csv','classification_baselines.csv',
        'model_predictions.csv','ensemble_predictions.csv','verification.json','delivery_manifest.json','historical_date_usage.csv']]
    files += [PROJECT/h['cache_file'] for h in read(V17/'results/heads.json')]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in files});return r
def manifest(phase):
    return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),
        executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,device='CPU float64; no new neural forward or training pass')
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
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
def fit_market(f):
    f=np.asarray(f,float);low,high=np.quantile(f,[.01,.99],axis=0,method='linear');clipped=np.clip(f,low,high)
    x,mean,sd=normalize_train(clipped,1e-6)
    return dict(features=f,lower=low,upper=high,clipped_features=clipped,mean=mean,sd=sd,standardized=x)
def transform(f,d):return (np.clip(np.asarray(f,float),d['lower'],d['upper'])-d['mean'])/d['sd']
def market_rows(split,cutoff,rows):
    frame=pd.read_csv(OUT/f'{split}_states.csv',float_precision='round_trip');g=frame[frame.cutoff.eq(cutoff)].sort_values('row_index')
    np.testing.assert_array_equal(g.row_index,rows);return g[MARKET_NAMES].to_numpy(float)
def inputs(job,split,context,parents,validation_refs):
    c=load_npz(context);xmarket=c['standardized'] if split=='training' else transform(market_rows('validation',job['cutoff'],context['validation_rows']),c)
    if job['parent_job'] is None:representation=np.empty((len(xmarket),0))
    else:
        parent=parents[job['parent_job']];d=load_npz(parent)
        if split=='training':representation=d['standardized']
        else:representation=transform(load_npz(validation_refs[parent['source_job']])['features'],d)
    assert representation.shape[1]==job['representation_dimensions']
    x=np.column_stack([representation,xmarket[:,job['market_indices']]])
    assert x.shape[1]==job['dimensions'] and np.isfinite(x).all();return x

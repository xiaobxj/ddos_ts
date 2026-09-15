"""Fixed, selective shrinkage of the existing sign-order extension."""
from pathlib import Path
import sys,json,time,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V23=PROJECT/'research_v23';V19=PROJECT/'research_v19';V18=PROJECT/'research_v18';V17=PROJECT/'research_v17'
sys.path.insert(0,str(V23));import common23 as previous
np=previous.np;pd=previous.pd;read=previous.read;save=previous.save;sha=previous.sha;load_npz=previous.load_npz
probability=previous.probability;design=previous.design;objective=previous.objective;fit_newton=previous.fit_newton;metric=previous.metric;probability_losses=previous.probability_losses
data=previous.data;indices=previous.indices;reference=previous.reference
CORE=['common24.py','prepare24.py','contract24.py','train24.py','score24.py','evaluate24.py','verify24.py']
MULTIPLIERS={f'{family}_order_shrink{factor}':factor for family in ['learned','raw'] for factor in [4,16]}
CANDIDATES={m:('learned_vol_interaction' if m.startswith('learned') else 'raw_vol_interaction') for m in MULTIPLIERS}
WEAK={m:('learned_order_extension' if m.startswith('learned') else 'raw_order_extension') for m in MULTIPLIERS}
OLD_INTERACTIONS={m:('learned_market' if m.startswith('learned') else 'raw_trend') for m in MULTIPLIERS}
GATE_MAP={m:'excess_order' for m in MULTIPLIERS};CONTROLS={'excess_order':'order_only'}
METHODS=previous.METHODS+list(CANDIDATES);SEED_METHODS=previous.SEED_METHODS+[m for m in CANDIDATES if m.startswith('learned')];PARTITIONS=previous.PARTITIONS

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def source_hashes():
    r=previous.source_hashes();r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE});return r
def old_evidence():
    r=dict(read(V23/'results/preparation_manifest.json')['old_evidence'])
    for name,digest in read(V23/'results/delivery_manifest.json')['files'].items():r[str((V23/name).relative_to(PROJECT))]=digest
    path=V23/'results/delivery_manifest.json';r[str(path.relative_to(PROJECT))]=sha(path);assert len(r)==3389
    for name,digest in r.items():assert sha(PROJECT/name)==digest,name
    return r
def input_hashes():
    r=previous.input_hashes()
    paths=[V23/'protocol.json']+[V23/'results'/name for name in ['heads.json','source_heads.json','training_signal_states.csv','validation_signal_states.csv','model_predictions.csv','ensemble_predictions.csv','verification.json','delivery_manifest.json']]
    paths += [PROJECT/s['cache_file'] for s in read(V23/'results/heads.json')]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in paths});return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,device='CPU float64; no neural execution')
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
def scaled_inputs(x,multiplier):
    result=np.array(x,dtype=float,copy=True);result[:,-1]/=np.sqrt(multiplier);return result
def original_coefficients(theta,multiplier):
    result=np.array(theta,dtype=float,copy=True);result[-2]/=np.sqrt(multiplier);return result
def penalty_vector(dim,multiplier):
    result=np.full(dim+1,.01);result[-2]=.01*multiplier;result[-1]=0.;return result
def inputs(source,split):
    cache=load_npz(source)
    if split=='training':return cache['standardized'],cache['row_index']
    parent=next(s for s in read(V23/'results/source_heads.json') if s['job']==source['source_job'])
    x,rows=previous.base_inputs(parent,'validation')
    frame=pd.read_csv(V23/'results/validation_signal_states.csv',float_precision='round_trip');frame=frame[frame.cutoff.eq(source['cutoff'])].sort_values('row_index');np.testing.assert_array_equal(frame.row_index,rows)
    xx,_=previous.apply_interaction(x,frame.excess_order.to_numpy(),cache);return xx,rows
def direct_objective(beta,x,y,multiplier):
    # Independent coordinate system: unchanged R23 columns and a diagonal penalty.
    z=x@beta[:-1]+beta[-1];p=1/(1+np.exp(-z));pen=penalty_vector(x.shape[1],multiplier)
    value=float(np.mean(np.logaddexp(0,z)-y*z)+.5*np.dot(pen,beta*beta))
    error=p-y;grad=np.r_[x.T@error/len(y),error.mean()]+pen*beta
    return value,grad

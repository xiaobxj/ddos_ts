from pathlib import Path
import sys,json,hashlib,time,copy,shutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
V5=PROJECT/'research_v5'
V6=PROJECT/'research_v6'
V12=PROJECT/'research_v12'
OUT=ROOT/'results'
CACHE=ROOT/'cache'
sys.path.insert(0,str(V12))
import common12 as prior
legacy=prior.legacy
torch=prior.torch
np=prior.np
pd=prior.pd
read=prior.read
save=prior.save
sha=prior.sha
array_hash=prior.array_hash
object_hash=prior.object_hash
cpu_copy=prior.cpu_copy

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def old_evidence():
    result=dict(read(V12/'results/preparation_manifest.json')['old_evidence'])
    for name,digest in read(V12/'results/delivery_manifest.json')['files'].items():
        result[str((V12/name).relative_to(PROJECT))]=digest
    p=V12/'results/delivery_manifest.json';result[str(p.relative_to(PROJECT))]=sha(p)
    for name,digest in result.items():assert sha(PROJECT/name)==digest,name
    assert len(result)==2341,len(result)
    return result

def source_hashes():
    files=[ROOT/n for n in ['common13.py','prepare13.py','contract13.py','train13.py','rolling13.py','calibrate13.py','score13.py','evaluate13.py','verify13.py']]
    files += [V12/'common12.py']+[V6/n for n in ['common6.py','models6.py','train6.py','evaluate6.py']]
    files += [PROJECT/'research_v4/architecture.py']+sorted((PROJECT/'research_v4/vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}

def input_hashes():
    files=[V6/'protocol.json',V5/'cache/packed_raw.npz',V5/'cache/targets.npz',
           PROJECT/'research_v10/results/observation_table.csv']
    files += [V12/'results'/n for n in ['archived_models.json','archived_predictions.csv','validation_rows.csv','training_scales.json','reconstruction_parity.json','delivery_manifest.json']]
    files += [PROJECT/r['project_file'] for r in read(V12/'results/archived_models.json')]
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}

def manifest(phase):
    return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),
        source_sha256=source_hashes(),input_sha256=input_hashes(),python=sys.version,
        executable=sys.executable,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__)

def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json')
    assert p['protocol_sha256']==sha(ROOT/'protocol.json')
    assert p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for name,digest in p['local_sha256'].items():assert sha(ROOT/name)==digest,name
    if contract:
        c=read(OUT/'contract_verification.json')
        assert c['status']=='PASS' and c['source_sha256']==source_hashes()
    return p

def load_training(cutoff):
    with np.load(CACHE/f'training_{cutoff}.npz') as d:
        assert set(d.files)=={'patches','geometry','valid','returns','auxiliary','row_index'}
        values={k:torch.from_numpy(d[k].copy()).cuda() for k in ['patches','geometry','valid']}
        labels={k:torch.from_numpy(d[k].copy()).cuda() for k in ['returns','auxiliary']}
        return values,labels,d['row_index'].copy(),read(OUT/'training_scales.json')[cutoff]

def load_model(ref):
    assert sha(PROJECT/ref['project_file'])==ref['sha256']
    state=torch.load(PROJECT/ref['project_file'],map_location='cpu',weights_only=True)
    assert state['epoch']==20 and state['arm']=='combined' and state['history']=='full'
    assert state['cutoff']==ref['cutoff'] and state['seed']==ref['seed']
    assert object_hash(state['state_dict'])==ref['model_sha256']
    model=legacy.make_model('combined',ref['seed']);model.load_state_dict(state['state_dict']);model.eval()
    assert sum(p.numel() for p in model.parameters())==38551
    return model,state

def forecast(model,indices,scales):
    values=legacy.batch_tensors(np.asarray(indices,dtype=int))
    model.eval()
    with torch.inference_mode():p,_=legacy.predict(model,values,False)
    return p.cpu().numpy().astype(float)*scales['returns_sd']+scales['returns_mean']

def coefficient(g,minimum_years=3):
    x=g.predicted_return.to_numpy(float)-g.training_mean.to_numpy(float)
    z=g.actual.to_numpy(float)-g.training_mean.to_numpy(float)
    numerator=float(np.dot(x,z));denominator=float(np.dot(x,x))
    sufficient=len(g)>=cfg()['calibration']['minimum_weeks'] and g.date.str[:4].nunique()>=minimum_years
    fallback=not sufficient or denominator<=cfg()['calibration']['denominator_floor']
    raw=numerator/denominator if denominator>cfg()['calibration']['denominator_floor'] else None
    alpha=0. if fallback else float(np.clip(raw,0.,1.))
    return dict(n=len(g),signal_years=int(g.date.str[:4].nunique()),numerator=numerator,
        denominator=denominator,raw_alpha=raw,alpha=alpha,fallback=fallback,
        lower_clipped=bool(not fallback and raw<0),upper_clipped=bool(not fallback and raw>1))

def transform(p,m,alpha):return np.asarray(m,float)+alpha*(np.asarray(p,float)-np.asarray(m,float))

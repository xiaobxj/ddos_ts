from pathlib import Path
import sys,json,time,shutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
V5=PROJECT/'research_v5'
V6=PROJECT/'research_v6'
V12=PROJECT/'research_v12'
V14=PROJECT/'research_v14'
OUT=ROOT/'results'
CACHE=ROOT/'cache'
sys.path.insert(0,str(V14))
import common14 as previous
prior=previous.prior
legacy=previous.legacy
torch=previous.torch
np=previous.np
pd=previous.pd
read=previous.read
save=previous.save
sha=previous.sha
object_hash=previous.object_hash
metric=previous.metric
probability_losses=previous.probability_losses
from solver15 import probability,normalize_train,design,objective,fit_newton

def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def identity(ref):return f"{ref['family']}_{ref['cutoff']}_{ref['seed']}"

def old_evidence():
    result=dict(read(V14/'results/preparation_manifest.json')['old_evidence'])
    for name,digest in read(V14/'results/delivery_manifest.json')['files'].items():
        result[str((V14/name).relative_to(PROJECT))]=digest
    p=V14/'results/delivery_manifest.json';result[str(p.relative_to(PROJECT))]=sha(p)
    assert len(result)==2479,len(result)
    for name,digest in result.items():assert sha(PROJECT/name)==digest,name
    return result

def source_hashes():
    files=[ROOT/n for n in ['common15.py','solver15.py','prepare15.py','contract15.py','extract15.py','fit15.py','score15.py','evaluate15.py','verify15.py']]
    files += [V14/'common14.py',V12/'common12.py']+[V6/n for n in ['common6.py','models6.py','evaluate6.py']]
    files += [PROJECT/'research_v4/architecture.py']+sorted((PROJECT/'research_v4/vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}

def input_hashes():
    files=[V5/'cache/packed_raw.npz',V5/'cache/targets.npz',V6/'protocol.json']
    files += [V14/'results'/n for n in ['models.json','archived_models.json','observation_table.csv','training_rows.csv',
        'training_metadata.json','classification_baselines.csv','all_seed_predictions.csv','all_ensemble_predictions.csv','delivery_manifest.json']]
    files += sorted((V14/'cache').glob('*.npz'))
    for name in ['models.json','archived_models.json']:files += [PROJECT/r['project_file'] for r in read(V14/'results'/name)]
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}

def manifest(phase):
    return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        input_sha256=input_hashes(),executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__)

def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json')
    assert p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['local_sha256'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==source_hashes()
    return p

def check_artifacts(name):
    r=read(OUT/f'{name}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r

def load_backbone(ref):
    model,state=previous.load_model(ref);model.eval();model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    return model,state

def extract_features(model,values):
    features=[];native=[];before=object_hash(model.state_dict())
    with prior.audit_mode(model,False),torch.inference_mode():
        for lo in range(0,len(values['patches']),128):
            v={k:t[lo:lo+128] for k,t in values.items()}
            f=model.decoded_features(v['patches'],v['geometry'],v['valid']).flatten(1)
            features.append(f.cpu().numpy().copy());native.append(model.return_head(f).squeeze(-1).cpu().numpy().copy())
    assert object_hash(model.state_dict())==before
    f=np.concatenate(features);z=np.concatenate(native)
    assert f.shape[1]==25 and np.isfinite(f).all() and np.isfinite(z).all()
    return f,z

def training_data(cutoff):
    with np.load(V14/'cache'/f'training_{cutoff}.npz') as data:
        values={k:torch.from_numpy(data[k].copy()).cuda() for k in ['patches','geometry','valid']}
        return values,data['row_index'].copy(),data['direction'].astype(float)

def load_features(ref):
    assert sha(ROOT/ref['feature_file'])==ref['feature_sha256']
    with np.load(ROOT/ref['feature_file']) as d:return {k:d[k].copy() for k in d.files}

def probe_metrics(z,y):
    p=probability(z);b,l=probability_losses(p,y)
    return dict(log_loss=float(np.mean(np.logaddexp(0.,z)-y*z)),brier=float(b.mean()),accuracy=float(((p>.5)==(y>0)).mean()),
        auroc=previous.auc(p,y),mean_probability=float(p.mean()),probability_std=float(p.std()))

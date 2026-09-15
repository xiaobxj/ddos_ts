from pathlib import Path
import sys,json,hashlib,time

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
V5=PROJECT/'research_v5';V6=PROJECT/'research_v6';V8=PROJECT/'research_v8';V9=PROJECT/'research_v9'
OUT=ROOT/'results';CACHE=ROOT/'cache'
sys.path.insert(0,str(V6))
import models6 as legacy
import numpy as np
import pandas as pd
torch=legacy.torch


def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def old_evidence():
    result=dict(json.loads((V9/'results/preparation_manifest.json').read_text(encoding='utf-8'))['old_evidence'])
    delivery=json.loads((V9/'results/delivery_manifest.json').read_text(encoding='utf-8'))
    for name,digest in delivery['files'].items():result[str((V9/name).relative_to(PROJECT))]=digest
    result[str((V9/'results/delivery_manifest.json').relative_to(PROJECT))]=sha(V9/'results/delivery_manifest.json')
    for name,digest in result.items():assert sha(PROJECT/name)==digest,name
    return result


def source_hashes():
    files=[ROOT/n for n in ['common10.py','prepare10.py','contract10.py','train10.py','evaluate10.py']]
    files += [V6/n for n in ['common6.py','models6.py','train6.py','evaluate6.py']]
    v4=PROJECT/'research_v4';files += [v4/'architecture.py']+list((v4/'vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}


def input_hashes():
    files=[V5/'cache/packed_raw.npz',V5/'cache/targets.npz',V6/'protocol.json',
        V9/'results/observation_table.csv',V9/'results/legacy_checkpoint_manifest.json',V9/'results/legacy_predictions.csv',
        V9/'results/legacy_training_curves.csv',V8/'results/frozen_model_training_predictions.csv']
    files += list((V9/'cache').glob('masks_*.npz'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}


def manifest(phase):
    return dict(phase=phase,started_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),
        source_sha256=source_hashes(),input_sha256=input_hashes(),old_evidence=old_evidence(),executable=sys.executable,
        python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device=torch.cuda.get_device_name())


def check_frozen(contract=True):
    prep=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
    assert prep['protocol_sha256']==sha(ROOT/'protocol.json')
    assert prep['source_sha256']==source_hashes() and prep['input_sha256']==input_hashes()
    for name,digest in prep['local_sha256'].items():assert sha(ROOT/name)==digest,name
    if contract:
        test=json.loads((OUT/'contract_verification.json').read_text(encoding='utf-8'))
        assert test['status']=='PASS' and test['source_sha256']==source_hashes()
    return prep


def fit_plan():return [dict(**fold,seed=seed,rule='huber') for fold in cfg()['folds'] for seed in cfg()['training']['seeds']]


def load_fold(cutoff):
    with np.load(CACHE/f'masks_{cutoff}.npz') as data:return data['training'].copy(),data['testing'].copy()


def batches(order):return [order[i:i+128] for i in range(0,len(order),128)]


def array_hash(array):return hashlib.sha256(np.asarray(array,dtype='<i8').tobytes()).hexdigest()


def boundary_hash(chunks):return array_hash(np.cumsum([0]+[len(b) for b in chunks]))


def robust_loss(errors):return torch.where(errors.abs()<=1.,errors**2,2.*errors.abs()-1.)


def numpy_robust(errors):
    errors=np.asarray(errors,float)
    return np.where(np.abs(errors)<=1.,errors**2,2.*np.abs(errors)-1.)


def joint_loss(output,auxiliary,labels,indices,rule='huber'):
    e=output-labels['returns'][indices]
    returns=robust_loss(e).mean() if rule=='huber' else (e**2).mean()
    return returns+.1*((auxiliary-labels['auxiliary'][indices])**2).mean()


def huber_location(z):
    z=np.asarray(z,float);lo=float(z.min()-1.);hi=float(z.max()+1.)
    for _ in range(160):
        mid=(lo+hi)/2
        if np.mean(np.clip(mid-z,-1.,1.))>0:hi=mid
        else:lo=mid
    value=(lo+hi)/2
    assert abs(np.mean(np.clip(value-z,-1.,1.)))<1e-12
    return value

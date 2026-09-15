from pathlib import Path
import sys,json,hashlib,time

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
V5=PROJECT/'research_v5';V6=PROJECT/'research_v6';V8=PROJECT/'research_v8'
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
    result=dict(json.loads((V8/'results/preparation_manifest.json').read_text(encoding='utf-8'))['old_evidence'])
    delivery=json.loads((V8/'results/delivery_manifest.json').read_text(encoding='utf-8'))
    for name,digest in delivery['files'].items():result[str((V8/name).relative_to(PROJECT))]=digest
    result[str((V8/'results/delivery_manifest.json').relative_to(PROJECT))]=sha(V8/'results/delivery_manifest.json')
    for name,digest in result.items():assert sha(PROJECT/name)==digest,name
    return result


def source_hashes():
    files=[ROOT/n for n in ['common9.py','prepare9.py','contract9.py','probe9.py','train9.py','evaluate9.py']]
    files += [V6/n for n in ['common6.py','models6.py','train6.py','evaluate6.py']]
    v4=PROJECT/'research_v4'
    files += [v4/'architecture.py']+list((v4/'vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}


def input_hashes():
    files=[V5/'cache/packed_raw.npz',V5/'cache/targets.npz',V6/'protocol.json',
           V8/'results/observation_table.csv',V8/'results/reused_checkpoint_manifest.json',
           V8/'results/reused_full_predictions.csv',V6/'results/main_training_curves.csv']
    files += list((V8/'cache').glob('policies_*.npz'))
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


def fit_plan():return [dict(**f,seed=s,rule='balanced') for f in cfg()['folds'] for s in cfg()['training']['seeds']]


def load_fold(cutoff):
    with np.load(CACHE/f'masks_{cutoff}.npz') as data:return data['training'].copy(),data['testing'].copy()


def batches(order,rule):
    n=len(order);steps=(n+127)//128
    if rule=='legacy':return [order[i:i+128] for i in range(0,n,128)]
    assert rule=='balanced'
    return list(np.array_split(order,steps))


def batch_factor(size,n,rule):return size*((n+127)//128)/n if rule=='balanced' else 1.


def batch_loss(output,auxiliary,labels,indices,n,rule):
    base=((output-labels['returns'][indices])**2).mean()+.1*((auxiliary-labels['auxiliary'][indices])**2).mean()
    return base*batch_factor(len(indices),n,rule)


def array_hash(array):return hashlib.sha256(np.asarray(array,dtype='<i8').tobytes()).hexdigest()


def boundary_hash(chunks):return array_hash(np.cumsum([0]+[len(b) for b in chunks]))

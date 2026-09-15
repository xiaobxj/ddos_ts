from pathlib import Path
from contextlib import contextmanager
import sys,json,hashlib,time,copy,shutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
V5=PROJECT/'research_v5'
V6=PROJECT/'research_v6'
V10=PROJECT/'research_v10'
V11=PROJECT/'research_v11'
OUT=ROOT/'results'
CACHE=ROOT/'cache'
sys.path.insert(0,str(V6))
import models6 as legacy
import numpy as np
import pandas as pd
torch=legacy.torch


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def cfg():return read(ROOT/'protocol.json')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def array_hash(array):return hashlib.sha256(np.asarray(array,dtype='<i8').tobytes()).hexdigest()


def object_hash(value):
    h=hashlib.sha256()
    def walk(v):
        if torch.is_tensor(v):
            h.update(b'tensor');h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=lambda x:str(x)):walk(k);walk(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
            for x in v:walk(x)
        else:h.update(type(v).__name__.encode());h.update(repr(v).encode())
    walk(value)
    return h.hexdigest()


def model_hash(model):return object_hash(model.state_dict())


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def old_evidence():
    result=dict(read(V11/'results/preparation_manifest.json')['old_evidence'])
    for name,digest in read(V11/'results/delivery_manifest.json')['files'].items():
        result[str((V11/name).relative_to(PROJECT))]=digest
    p=V11/'results/delivery_manifest.json';result[str(p.relative_to(PROJECT))]=sha(p)
    archive=PROJECT/'research_v12_preflight'
    preflight=read(archive/'preflight_manifest.json')
    for name,digest in preflight['files'].items():result[str((archive/name).relative_to(PROJECT))]=digest
    p=archive/'preflight_manifest.json';result[str(p.relative_to(PROJECT))]=sha(p)
    for name,digest in result.items():assert sha(PROJECT/name)==digest,name
    assert len(result)==2248,len(result)
    return result


def source_hashes():
    files=[ROOT/n for n in ['common12.py','prepare12.py','contract12.py','reconstruct12.py','continue12.py','score12.py','evaluate12.py','verify12.py']]
    files += [V6/n for n in ['common6.py','models6.py','train6.py','evaluate6.py']]
    files += [PROJECT/'research_v4/architecture.py']+sorted((PROJECT/'research_v4/vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}


def input_hashes():
    files=[V6/'protocol.json',V5/'cache/packed_raw.npz',V5/'cache/targets.npz',
        V10/'results/observation_table.csv',V10/'results/mse_checkpoint_manifest.json',V10/'results/mse_predictions.csv',
        V11/'results/training_scales.json',V11/'results/training_rows.csv']
    files += sorted((V11/'cache').glob('training_*.npz'))+sorted((V10/'cache').glob('masks_*.npz'))
    files += [PROJECT/r['project_file'] for r in read(V10/'results/mse_checkpoint_manifest.json')]
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}


def manifest(phase):
    return dict(phase=phase,started_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),
        source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,
        torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device=torch.cuda.get_device_name())


def check_frozen(contract=True):
    prep=read(OUT/'preparation_manifest.json')
    assert prep['protocol_sha256']==sha(ROOT/'protocol.json')
    assert prep['source_sha256']==source_hashes() and prep['input_sha256']==input_hashes()
    for name,digest in prep['local_sha256'].items():assert sha(ROOT/name)==digest,name
    if contract:
        test=read(OUT/'contract_verification.json')
        assert test['status']=='PASS' and test['source_sha256']==source_hashes()
    return prep


def load_training(cutoff):
    with np.load(CACHE/f'training_{cutoff}.npz') as data:
        assert set(data.files)=={'patches','geometry','valid','returns','auxiliary','row_index'}
        values={k:torch.from_numpy(data[k].copy()).cuda() for k in ['patches','geometry','valid']}
        labels={k:torch.from_numpy(data[k].copy()).cuda() for k in ['returns','auxiliary']}
        tr=data['row_index'].copy()
    return values,labels,tr,read(OUT/'training_scales.json')[cutoff]


def load_validation(cutoff):
    with np.load(CACHE/f'validation_{cutoff}.npz') as data:
        values={k:torch.from_numpy(data[k].copy()).cuda() for k in ['patches','geometry','valid']}
        return values,data['row_index'].copy()


def optimizer_for(model,lr=.001):return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.1)
def batches(order):return [order[i:i+128] for i in range(0,len(order),128)]
def identity(ref):return f"{ref['schedule']}_{ref['cutoff']}_{ref['seed']}"
def meta(ref):return {k:ref[k] for k in ['schedule','cutoff','seed']}


def update_step(model,optimizer,values,labels,positions):
    # Preserve the round6 expression and reduction order exactly.
    batch=torch.tensor(positions,device='cuda')
    optimizer.zero_grad(set_to_none=True)
    output,aux=legacy.predict(model,{k:t[batch] for k,t in values.items()},True)
    return_loss=((output-labels['returns'][batch])**2).mean()
    auxiliary_loss=((aux-labels['auxiliary'][batch])**2).mean()
    loss=return_loss+.1*auxiliary_loss
    assert torch.isfinite(loss)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()),float(return_loss.detach()),float(auxiliary_loss.detach()),float(norm)


@contextmanager
def audit_mode(model,stochastic=False,seed=None):
    modes=[(m,m.training) for m in model.modules()]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model.train(stochastic)
        if seed is not None:torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        try:yield
        finally:
            for m,mode in modes:m.training=mode


def training_loss(model,values,labels,stochastic=False,seed=None):
    errors,auxiliary=[],[]
    with audit_mode(model,stochastic,seed),torch.inference_mode():
        for lo in range(0,len(labels['returns']),128):
            p,a=legacy.predict(model,{k:t[lo:lo+128] for k,t in values.items()},True)
            errors.append(((p-labels['returns'][lo:lo+128])**2).cpu().numpy().astype(float))
            auxiliary.append(((a-labels['auxiliary'][lo:lo+128])**2).mean(dim=1).cpu().numpy().astype(float))
    r=float(np.concatenate(errors).mean());a=float(np.concatenate(auxiliary).mean())
    return dict(return_mse=r,auxiliary_mse=a,joint_mse=r+.1*a)


def training_audit(model,values,labels):
    rows=[dict(mode='eval',draw=-1,audit_seed=-1,**training_loss(model,values,labels))]
    for draw in range(cfg()['training_diagnostics']['dropout_draws']):
        seed=cfg()['training_diagnostics']['dropout_seed_base']+draw
        rows.append(dict(mode='dropout',draw=draw,audit_seed=seed,**training_loss(model,values,labels,True,seed)))
    return rows


def capture(model,optimizer,rng,**metadata):
    state=dict(state_dict=cpu_copy(model.state_dict()),optimizer_state_dict=cpu_copy(optimizer.state_dict()),
        numpy_rng_state=copy.deepcopy(rng.bit_generator.state),cpu_rng_state=torch.get_rng_state().clone(),
        cuda_rng_state=torch.cuda.get_rng_state().clone(),**metadata)
    state['model_sha256']=object_hash(state['state_dict'])
    state['optimizer_sha256']=object_hash(state['optimizer_state_dict'])
    state['rng_sha256']=object_hash({k:state[k] for k in ['numpy_rng_state','cpu_rng_state','cuda_rng_state']})
    return state


def load_model(ref):
    path=PROJECT/ref['project_file'];assert sha(path)==ref['sha256']
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=legacy.make_model('combined',ref['seed']);model.load_state_dict(state['state_dict'])
    assert sum(p.numel() for p in model.parameters())==38551
    return model,state


def restore_payload(state,seed):
    model=legacy.make_model('combined',seed);model.load_state_dict(state['state_dict'])
    optimizer=optimizer_for(model)
    # AdamW can retain the CPU step tensor by reference; resumed updates must not mutate the source snapshot.
    optimizer.load_state_dict(cpu_copy(state['optimizer_state_dict']))
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['numpy_rng_state'])
    torch.set_rng_state(state['cpu_rng_state']);torch.cuda.set_rng_state(state['cuda_rng_state'])
    assert object_hash(optimizer.state_dict())==state['optimizer_sha256']
    assert model_hash(model)==state['model_sha256']
    return model,optimizer,rng,state


def restore(ref):
    path=PROJECT/ref['project_file'];assert sha(path)==ref['sha256']
    return restore_payload(torch.load(path,map_location='cpu',weights_only=True),ref['seed'])


def run_epoch(model,optimizer,rng,values,labels,tr,epoch,job):
    model.train();n=len(tr);order=legacy.epoch_order(n,n,rng);chunks=batches(order)
    sums=np.zeros(3);norms=[]
    for b in chunks:
        loss,ret,aux,norm=update_step(model,optimizer,values,labels,b)
        sums+=np.asarray([loss,ret,aux])*len(b);norms.append(norm)
    return dict(**meta(job),epoch=epoch,train_n=n,presentations=n,optimizer_steps=len(chunks),
        learning_rate=optimizer.param_groups[0]['lr'],training_order_sha256=array_hash(tr[order]),
        batch_boundaries_sha256=array_hash(np.cumsum([0]+[len(b) for b in chunks])),
        online_joint_mse=float(sums[0]/n),online_return_mse=float(sums[1]/n),online_auxiliary_mse=float(sums[2]/n),
        raw_gradient_norm_mean=float(np.mean(norms)),raw_gradient_norm_max=float(max(norms)),
        gradient_clipping_fraction=float(np.mean(np.asarray(norms)>1.)))


def checkpoint_ref(path,state):
    return dict(**meta(state),epoch=state['epoch'],project_file=str(path.relative_to(PROJECT)),
        file=str(path.relative_to(ROOT)),sha256=sha(path),model_sha256=state['model_sha256'],
        optimizer_sha256=state['optimizer_sha256'],rng_sha256=state['rng_sha256'])

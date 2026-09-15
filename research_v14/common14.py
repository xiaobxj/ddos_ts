from pathlib import Path
import sys,json,hashlib,time,copy,shutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
V5=PROJECT/'research_v5'
V6=PROJECT/'research_v6'
V12=PROJECT/'research_v12'
V13=PROJECT/'research_v13'
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
def probability(logits):return np.exp(-np.logaddexp(0.,-np.asarray(logits,float)))
def raw_direction(returns):return (np.asarray(returns)>0).astype(np.float32)

def old_evidence():
    result=dict(read(V13/'results/preparation_manifest.json')['old_evidence'])
    for name,digest in read(V13/'results/delivery_manifest.json')['files'].items():
        result[str((V13/name).relative_to(PROJECT))]=digest
    p=V13/'results/delivery_manifest.json';result[str(p.relative_to(PROJECT))]=sha(p)
    assert len(result)==2415,len(result)
    for name,digest in result.items():assert sha(PROJECT/name)==digest,name
    return result

def source_hashes():
    files=[ROOT/n for n in ['common14.py','prepare14.py','contract14.py','train14.py','score14.py','evaluate14.py','verify14.py']]
    files += [V12/'common12.py']+[V6/n for n in ['common6.py','models6.py','train6.py','evaluate6.py']]
    files += [PROJECT/'research_v4/architecture.py']+sorted((PROJECT/'research_v4/vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}

def input_hashes():
    files=[V6/'protocol.json',V5/'cache/packed_raw.npz',V5/'cache/targets.npz']
    files += [V13/'results'/n for n in ['observation_table.csv','archived_models.json','archived_predictions.csv','validation_rows.csv','delivery_manifest.json']]
    files += [V12/'results/training_scales.json']+sorted((V12/'cache').glob('training_*.npz'))+sorted((V12/'cache').glob('validation_*.npz'))
    files += [PROJECT/r['project_file'] for r in read(V13/'results/archived_models.json')]
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}

def manifest(phase):
    return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),
        input_sha256=input_hashes(),executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__)

def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json')
    assert p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for name,digest in p['local_sha256'].items():assert sha(ROOT/name)==digest,name
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==source_hashes()
    return p

def load_training(cutoff):
    with np.load(CACHE/f'training_{cutoff}.npz') as d:
        assert set(d.files)=={'patches','geometry','valid','direction','auxiliary','row_index'}
        values={k:torch.from_numpy(d[k].copy()).cuda() for k in ['patches','geometry','valid']}
        labels={k:torch.from_numpy(d[k].copy()).cuda() for k in ['direction','auxiliary']}
        return values,labels,d['row_index'].copy(),read(OUT/'training_metadata.json')[cutoff]

def load_validation(cutoff):
    with np.load(CACHE/f'validation_{cutoff}.npz') as d:
        assert set(d.files)=={'patches','geometry','valid','row_index'}
        return {k:torch.from_numpy(d[k].copy()).cuda() for k in ['patches','geometry','valid']},d['row_index'].copy()

def shared_initial_hash(model):return object_hash({k:v for k,v in model.state_dict().items() if k!='return_head.bias'})

def make_classifier(seed,frequency):
    model=legacy.make_model('combined',seed)
    p=float(np.clip(frequency,1e-6,1-1e-6))
    with torch.no_grad():model.return_head.bias.fill_(float(np.log(p/(1-p))))
    assert sum(p.numel() for p in model.parameters())==38551
    return model

def load_model(ref):
    assert sha(PROJECT/ref['project_file'])==ref['sha256']
    state=torch.load(PROJECT/ref['project_file'],map_location='cpu',weights_only=True)
    assert state['cutoff']==ref['cutoff'] and state['seed']==ref['seed'] and state['epoch']==20
    model=legacy.make_model('combined',ref['seed']);model.load_state_dict(state['state_dict']);model.eval()
    assert object_hash(state['state_dict'])==ref['model_sha256']
    if ref.get('method')=='direction_bce':assert state['head_semantics']=='up_logit' and state['method']=='direction_bce'
    else:assert state['arm']=='combined' and state['history']=='full'
    return model,state

def update_step(model,optimizer,values,labels,positions):
    b=torch.tensor(positions,device='cuda');optimizer.zero_grad(set_to_none=True)
    logits,aux=legacy.predict(model,{k:t[b] for k,t in values.items()},True)
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels['direction'][b])
    auxiliary=((aux-labels['auxiliary'][b])**2).mean();loss=bce+.1*auxiliary
    assert torch.isfinite(loss);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    return float(loss.detach()),float(bce.detach()),float(auxiliary.detach()),float(norm)

def training_loss(model,values,labels,stochastic=False,seed=None):
    logits=[];auxiliary=[]
    with prior.audit_mode(model,stochastic,seed),torch.inference_mode():
        for lo in range(0,len(labels['direction']),128):
            z,a=legacy.predict(model,{k:t[lo:lo+128] for k,t in values.items()},True)
            logits.append(z.cpu().numpy().astype(float))
            auxiliary.append(((a-labels['auxiliary'][lo:lo+128])**2).mean(dim=1).cpu().numpy().astype(float))
    z=np.concatenate(logits);y=labels['direction'].cpu().numpy().astype(float);p=probability(z)
    bce=float(np.mean(np.logaddexp(0,z)-y*z));a=float(np.concatenate(auxiliary).mean())
    return dict(log_loss=bce,brier=float(np.mean((p-y)**2)),accuracy=float(((p>.5)==(y>0)).mean()),
        auxiliary_mse=a,joint_loss=bce+.1*a,mean_probability=float(p.mean()),probability_std=float(p.std()))

def training_audit(model,values,labels):
    rows=[dict(mode='eval',draw=-1,audit_seed=-1,**training_loss(model,values,labels))]
    for draw in range(cfg()['training_diagnostics']['dropout_draws']):
        seed=cfg()['training_diagnostics']['dropout_seed_base']+draw
        rows.append(dict(mode='dropout',draw=draw,audit_seed=seed,**training_loss(model,values,labels,True,seed)))
    return rows

def probability_losses(p,y):
    p=np.asarray(p,float);y=np.asarray(y,float);q=np.clip(p,1e-12,1-1e-12)
    return (p-y)**2,-y*np.log(q)-(1-y)*np.log1p(-q)

def auc(scores,y):
    y=np.asarray(y,bool);n1=int(y.sum());n0=len(y)-n1
    if n1==0 or n0==0:return None
    ranks=pd.Series(np.asarray(scores,float)).rank(method='average').to_numpy()
    return float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0))

def metric(g):
    y=g.actual_up.to_numpy(bool);up=g.direction_up.to_numpy(bool)
    tp=int((up&y).sum());tn=int((~up&~y).sum());fp=int((up&~y).sum());fn=int((~up&y).sum())
    result=dict(n=len(g),accuracy=float((up==y).mean()),direction_error=float((up!=y).mean()),
        correct_directions=tp+tn,balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),
        predicted_up_fraction=float(up.mean()),observed_up_fraction=float(y.mean()),tp=tp,tn=tn,fp=fp,fn=fn,
        auroc=auc(g.score,y),brier=None,log_loss=None,mean_probability=None,probability_std=None,calibration_gap=None,clipped_probabilities=None)
    if g.probability.notna().all():
        p=g.probability.to_numpy(float);assert np.all((p>=0)&(p<=1))
        b,l=probability_losses(p,y)
        result.update(brier=float(b.mean()),log_loss=float(l.mean()),mean_probability=float(p.mean()),probability_std=float(p.std()),
            calibration_gap=float(p.mean()-y.mean()),clipped_probabilities=int(((p<1e-12)|(p>1-1e-12)).sum()))
    else:assert g.probability.isna().all()
    return result

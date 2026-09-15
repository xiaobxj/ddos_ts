from pathlib import Path
import sys,json,time,shutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
V5=PROJECT/'research_v5'
V6=PROJECT/'research_v6'
V13=PROJECT/'research_v13'
V15=PROJECT/'research_v15'
OUT=ROOT/'results'
CACHE=ROOT/'cache'
PRICE=PROJECT/'research/data/1_000300.csv'
sys.path.insert(0,str(V15))
import common15 as previous
from features16 import raw_features,scalar_features,NAMES,HORIZONS,KINDS
np=previous.np;pd=previous.pd;torch=previous.torch;legacy=previous.legacy
read=previous.read;save=previous.save;sha=previous.sha;object_hash=previous.object_hash
probability=previous.probability;normalize_train=previous.normalize_train;design=previous.design
objective=previous.objective;fit_newton=previous.fit_newton;metric=previous.metric;probability_losses=previous.probability_losses


def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def job_id(kind,cutoff,seed=-1):return f'{kind}_{cutoff}_{seed}'


def old_evidence():
    result=dict(read(V15/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V15/'results/delivery_manifest.json')['files'].items():result[str((V15/n).relative_to(PROJECT))]=d
    p=V15/'results/delivery_manifest.json';result[str(p.relative_to(PROJECT))]=sha(p)
    assert len(result)==2618
    for n,d in result.items():assert sha(PROJECT/n)==d,n
    return result


def source_hashes():
    files=[ROOT/n for n in ['common16.py','features16.py','prepare16.py','contract16.py','train16.py','score16.py','evaluate16.py','verify16.py']]
    files += [V15/'common15.py',V15/'solver15.py',V15/'verify15.py',V15/'evaluate15.py',PROJECT/'research_v14/common14.py',PROJECT/'research_v12/common12.py']
    files += [V6/n for n in ['common6.py','models6.py','evaluate6.py']]+[PROJECT/'research_v4/architecture.py']
    files += sorted((PROJECT/'research_v4/vendor/Crossformer/cross_models').rglob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}


def input_hashes():
    files=[PRICE,V5/'cache/packed_raw.npz',V5/'cache/targets.npz',V6/'protocol.json',V15/'protocol.json']
    files += [V13/'results'/n for n in ['inner_models.json','rolling_seed_predictions.csv','calibration_membership.csv']]
    files += [V15/'results'/n for n in ['observation_table.csv','heads.json','all_seed_predictions.csv','all_ensemble_predictions.csv','training_metrics.csv','delivery_manifest.json']]
    files += [PROJECT/n for n in ['research/protocol.json','research_v2/protocol.json','research_v3/protocol.json','research_v4/protocol.json',
        'research/results/predictions.csv','research_v2/results/validation_predictions.csv','research_v3/results/validation_grid_predictions.csv']]
    refs=read(V13/'results/inner_models.json')+[h for h in read(V15/'results/heads.json') if h['family']=='probe_mse']
    files += [PROJECT/r['project_file'] for r in refs]
    files += [V15/h['feature_file'] for h in refs if 'feature_file' in h]
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}


def manifest(phase):
    return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),
        executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__)


def check_frozen(contract=True):
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json')
    assert p['source_sha256']==source_hashes() and p['input_sha256']==input_hashes()
    for n,d in p['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==source_hashes()
    return p


def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r


def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra)
    save(OUT/f"{run['phase']}_manifest.json",run)


def data():
    obs=pd.read_csv(OUT/'observation_table.csv');price=pd.read_csv(PRICE)
    np.testing.assert_array_equal(price.date.iloc[obs.anchor].to_numpy(),obs.date.to_numpy())
    with np.load(V5/'cache/targets.npz') as r:returns=r['returns'].copy()
    np.testing.assert_allclose(returns,obs.exec_return,rtol=0,atol=1e-12)
    return obs,price,returns


def indices(obs,fold):
    tr=np.flatnonzero(obs.joint_completed.le(fold['cutoff']).to_numpy())
    te=np.flatnonzero((obs.date.gt(fold['cutoff'])&obs.date.le(fold['end'])&obs.weekday.eq(4)&obs.joint_completed.le('2020-12-31')).to_numpy())
    assert len(tr)==fold['train_n'] and len(te)==fold['test_n'] and not len(np.intersect1d(tr,te))
    return tr,te


def load_features(head):
    p=PROJECT/head['cache_file'];assert sha(p)==head['cache_sha256']
    with np.load(p) as d:return {k:d[k].copy() for k in d.files}


def raw_bars(price):return price[['open','high','low','close','volume']].to_numpy(float)


def extract_learned(ref,rows):
    model,state=previous.load_backbone(ref);values=legacy.batch_tensors(rows)
    f,native=previous.extract_features(model,values)
    assert object_hash(model.state_dict())==ref['model_sha256']
    return model,state,values,f,native


def training_metrics(head,d):
    theta=np.asarray(head['coefficients']);a=design(d['standardized']);y=d['direction'];z=a@theta
    value,g,h=objective(theta,a,y,cfg()['probe']['l2_lambda']);rate=float(y.mean())
    constant=previous.probe_metrics(np.full(len(y),np.log(rate/(1-rate))),y)
    return dict(job=head['job'],method=head['method'],cutoff=head['cutoff'],seed=head['seed'],reused=head['reused'],train_n=len(y),
        iterations=head['iterations'],objective=value,gradient_inf=float(np.max(np.abs(g))),hessian_min_eigenvalue=float(np.linalg.eigvalsh(h).min()),
        **previous.probe_metrics(z,y),constant_log_loss=constant['log_loss'],constant_brier=constant['brier'],constant_accuracy=constant['accuracy'])

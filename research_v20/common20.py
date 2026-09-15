"""Fixed coordinates; only the classifier fitting window changes in round20."""
from pathlib import Path
import sys,json,time,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache'
V19=PROJECT/'research_v19';V18=PROJECT/'research_v18';sys.path.insert(0,str(V19))
import common19 as previous
np=previous.np;pd=previous.pd;read=previous.read;save=previous.save;sha=previous.sha;load_npz=previous.load_npz
probability=previous.probability;design=previous.design;objective=previous.objective;fit_newton=previous.fit_newton
metric=previous.metric;probability_losses=previous.probability_losses;PARTITIONS=previous.PARTITIONS
SPECS=[dict(method='learned_recent_additive',source_method='learned_market',source_round=18,family='learned',interaction=False,dimensions=29,fits=18),
    dict(method='raw_recent_additive',source_method='raw_trend',source_round=18,family='raw',interaction=False,dimensions=26,fits=6),
    dict(method='learned_recent_interaction',source_method='learned_vol_interaction',source_round=19,family='learned',interaction=True,dimensions=30,fits=18),
    dict(method='raw_recent_interaction',source_method='raw_vol_interaction',source_round=19,family='raw',interaction=True,dimensions=27,fits=6)]
FULL={s['method']:s['source_method'] for s in SPECS}
INTERACTIONS={'learned_recent_interaction':'learned_recent_additive','raw_recent_interaction':'raw_recent_additive'}
FACTORIAL={'learned':['learned_market','learned_vol_interaction','learned_recent_additive','learned_recent_interaction'],
    'raw':['raw_trend','raw_vol_interaction','raw_recent_additive','raw_recent_interaction']}
METHODS=previous.METHODS+list(FULL)+['recent_frequency']
SEED_METHODS=['native_mse','learned_probe','learned_clip','learned_market','learned_vol_interaction','learned_recent_additive','learned_recent_interaction']
CORE=['common20.py','prepare20.py','contract20.py','train20.py','score20.py','evaluate20.py','verify20.py']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def data():return previous.data()
def indices(obs,fold):return previous.indices(obs,fold)
def recent_positions(obs,rows,fold):
    return np.flatnonzero(obs.date.iloc[rows].ge(fold['recent_start']).to_numpy())
def old_evidence():
    r=dict(read(V19/'results/preparation_manifest.json')['old_evidence'])
    for name,digest in read(V19/'results/delivery_manifest.json')['files'].items():r[str((V19/name).relative_to(PROJECT))]=digest
    path=V19/'results/delivery_manifest.json';r[str(path.relative_to(PROJECT))]=sha(path);assert len(r)==2958
    for name,digest in r.items():assert sha(PROJECT/name)==digest,name
    return r
def source_hashes():
    r=previous.source_hashes();r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE});return r
def input_hashes():
    r=previous.input_hashes();files=[V19/'protocol.json']
    files += [V19/'results'/n for n in ['heads.json','source_heads.json','model_predictions.csv','ensemble_predictions.csv','classification_baselines.csv',
        'training_states.csv','validation_states.csv','historical_date_usage.csv','verification.json','delivery_manifest.json']]
    files += [PROJECT/h['cache_file'] for h in read(V19/'results/heads.json')]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in files});return r
def manifest(phase):
    return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),
        executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,device='CPU float64; final classifier fitting only')
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
def source_inputs(source,source_round,split):
    if split=='training':
        d=load_npz(source);return d['standardized'],d['row_index']
    if source_round==18:
        x,v,rows=previous.inherited_inputs(source,'validation');return x,rows
    base=next(s for s in read(V18/'results/heads.json') if s['job']==source['source_job'])
    x,v,rows=previous.inherited_inputs(base,'validation');xx,_=previous.apply_interaction(x,v,load_npz(source));return xx,rows


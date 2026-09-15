"""Frozen annual additive inputs plus a single volatility interaction."""
from pathlib import Path
import sys,json,time,shutil

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;V18=PROJECT/'research_v18';V17=PROJECT/'research_v17';V16=PROJECT/'research_v16'
OUT=ROOT/'results';CACHE=ROOT/'cache';sys.path.insert(0,str(V18))
import common18 as previous
np=previous.np;pd=previous.pd;read=previous.read;save=previous.save;sha=previous.sha;load_npz=previous.load_npz
probability=previous.probability;design=previous.design;objective=previous.objective;fit_newton=previous.fit_newton
metric=previous.metric;probability_losses=previous.probability_losses;PARTITIONS=previous.PARTITIONS
from interaction19 import fit_interaction,apply_interaction

CANDIDATES={'learned_vol_interaction':'learned_market','raw_vol_interaction':'raw_trend'}
ANCHORS={'learned_vol_interaction':'learned_clip','raw_vol_interaction':'raw25_clip'}
METHODS=previous.METHODS+list(CANDIDATES)
CORE=['common19.py','interaction19.py','prepare19.py','contract19.py','train19.py','score19.py','evaluate19.py','verify19.py']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def data():return previous.data()
def indices(obs,fold):return previous.indices(obs,fold)
def old_evidence():
    r=dict(read(V18/'results/preparation_manifest.json')['old_evidence'])
    for name,d in read(V18/'results/delivery_manifest.json')['files'].items():r[str((V18/name).relative_to(PROJECT))]=d
    path=V18/'results/delivery_manifest.json';r[str(path.relative_to(PROJECT))]=sha(path);assert len(r)==2882
    for name,d in r.items():assert sha(PROJECT/name)==d,name
    return r
def source_hashes():
    r=previous.source_hashes();r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE});return r
def input_hashes():
    r=previous.input_hashes();files=[V18/'protocol.json']
    files += [V18/'results'/n for n in ['heads.json','parent_heads.json','market_contexts.json','training_states.csv','validation_states.csv',
        'model_predictions.csv','ensemble_predictions.csv','classification_baselines.csv','historical_date_usage.csv','verification.json','delivery_manifest.json']]
    files += [PROJECT/h['cache_file'] for h in read(V18/'results/heads.json')]+[PROJECT/h['cache_file'] for h in read(V18/'results/market_contexts.json')]
    r.update({str(p.relative_to(PROJECT)):sha(p) for p in files});return r
def manifest(phase):
    return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),
        executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,device='CPU float64; no neural forward or training')
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    r=read(OUT/'preparation_manifest.json');assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['source_sha256']==source_hashes() and r['input_sha256']==input_hashes()
    for name,d in r['artifacts'].items():assert sha(ROOT/name)==d,name
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==r['source_sha256']
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for name,d in r['artifacts'].items():assert sha(ROOT/name)==d,name
    return r
def inherited_inputs(source,split):
    context=next(c for c in read(V18/'results/market_contexts.json') if c['cutoff']==source['cutoff']);c=load_npz(context)
    if split=='training':
        d=load_npz(source);np.testing.assert_array_equal(d['row_index'],c['row_index']);return d['standardized'],c['standardized'][:,1],d['row_index']
    parents={h['job']:h for h in read(V18/'results/parent_heads.json')};refs={r['job']:r for r in read(V16/'results/validation_features.json')}
    x=previous.inputs(source,'validation',context,parents,refs)
    market=previous.transform(previous.market_rows('validation',source['cutoff'],context['validation_rows']),c)
    return x,market[:,1],np.asarray(context['validation_rows'],int)

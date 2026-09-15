from pathlib import Path
import sys,json,time,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V22=PROJECT/'research_v22';V18=PROJECT/'research_v18';V19=PROJECT/'research_v19'
sys.path.insert(0,str(V22));import common22 as previous
np=previous.np;pd=previous.pd;read=previous.read;save=previous.save;sha=previous.sha;load_npz=previous.load_npz
probability=previous.probability;design=previous.design;objective=previous.objective;fit_newton=previous.fit_newton;metric=previous.metric;probability_losses=previous.probability_losses
data=previous.data;indices=previous.indices;daily_extent=previous.daily_extent;reference=previous.reference
fit_interaction=previous.fit_interaction;apply_interaction=previous.apply_interaction
from states23 import features,scalar_features,run_diagnostics,GATES,NUMERIC,STATE_PARTITIONS
import common20 as parent_inputs
def base_inputs(source,split):return parent_inputs.source_inputs(source,19,split)
def source_ray(source):return load_npz(source)['ray']
CANDIDATES={'learned_order_extension':'learned_vol_interaction','raw_order_extension':'raw_vol_interaction'}
GATE_MAP={m:'excess_order' for m in CANDIDATES};CONTROLS={'excess_order':'order_only'}
OLD_INTERACTIONS={'learned_order_extension':'learned_market','raw_order_extension':'raw_trend'}
METHODS=previous.METHODS+list(CANDIDATES)+list(CONTROLS.values());SEED_METHODS=previous.SEED_METHODS+['learned_order_extension']
PARTITIONS=dict(previous.PARTITIONS,**STATE_PARTITIONS)
CORE=['states23.py','common23.py','prepare23.py','contract23.py','train23.py','score23.py','evaluate23.py','verify23.py']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def old_evidence():
    r=dict(read(V22/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V22/'results/delivery_manifest.json')['files'].items():r[str((V22/n).relative_to(PROJECT))]=d
    path=V22/'results/delivery_manifest.json';r[str(path.relative_to(PROJECT))]=sha(path);assert len(r)==3292
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def source_hashes():
    r=previous.source_hashes();r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE});return r
def input_hashes():
    r=previous.input_hashes()
    for path in [V22/'protocol.json']+[V22/'results'/n for n in ['model_predictions.csv','ensemble_predictions.csv','validation_signal_states.csv','heads.json','verification.json','delivery_manifest.json']]:r[str(path.relative_to(PROJECT))]=sha(path)
    return r
def manifest(phase):return dict(phase=phase,started_utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source_hashes(),input_sha256=input_hashes(),executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,device='CPU float64; fixed trailing states and convex heads')
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
def check_frozen(contract=True):
    r=read(OUT/'preparation_manifest.json');assert r['protocol_sha256']==sha(ROOT/'protocol.json') and r['source_sha256']==source_hashes() and r['input_sha256']==input_hashes()
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    if contract:
        c=read(OUT/'contract_verification.json');assert c['status']=='PASS' and c['source_sha256']==r['source_sha256'] and c['protocol_sha256']==r['protocol_sha256']
        for n,d in c['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def check_phase(phase):
    r=read(OUT/f'{phase}_manifest.json');assert r.get('finished_utc')
    for n,d in r['artifacts'].items():assert sha(ROOT/n)==d,n
    return r
def signal_rows(frame,obs,rows,cutoff):
    g=frame.iloc[obs.anchor.iloc[rows].to_numpy(int)].copy();g.insert(0,'row_index',rows);g.insert(0,'cutoff',cutoff)
    assert np.isfinite(g[NUMERIC].to_numpy()).all();np.testing.assert_array_equal(g.date,obs.date.iloc[rows]);return g
def gate_inputs(h,split):
    states=pd.read_csv(OUT/f'{split}_signal_states.csv',float_precision='round_trip');g=states[states.cutoff.eq(h['cutoff'])].sort_values('row_index');return g[h['gate']].to_numpy(),g.row_index.to_numpy(int)

from pathlib import Path
import sys,json,time,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';CACHE=ROOT/'cache';V21=PROJECT/'research_v21';V18=PROJECT/'research_v18';V19=PROJECT/'research_v19'
sys.path.insert(0,str(V21));import common21 as previous
np=previous.np;pd=previous.pd;read=previous.read;save=previous.save;sha=previous.sha;load_npz=previous.load_npz
probability=previous.probability;design=previous.design;objective=previous.objective;fit_newton=previous.fit_newton;metric=previous.metric;probability_losses=previous.probability_losses
base_inputs=previous.base_inputs;data=previous.data;indices=previous.indices;daily_extent=previous.daily_extent;reference=previous.reference
fit_interaction=previous.fit_interaction;apply_interaction=previous.apply_interaction
from states22 import features,scalar_features,assign,run_diagnostics,GATES,NUMERIC,STATE_PARTITIONS
CANDIDATES={f'{family}_{gate}_interaction':parent for gate in ['relativevol','persistence'] for family,parent in [('learned','learned_market'),('raw','raw_trend')]}
GATE_MAP={m:('relative_volatility' if 'relativevol' in m else 'persistence') for m in CANDIDATES}
CONTROLS={'relative_volatility':'relativevol_only','persistence':'persistence_only'}
OLD_INTERACTIONS={m:('learned_vol_interaction' if m.startswith('learned') else 'raw_vol_interaction') for m in CANDIDATES}
METHODS=previous.METHODS+list(CANDIDATES)+list(CONTROLS.values());SEED_METHODS=previous.SEED_METHODS+[m for m in CANDIDATES if m.startswith('learned')]
PARTITIONS=dict(previous.PARTITIONS,hmm_bucket=['low_probability','uncertain','high_probability'],**STATE_PARTITIONS)
CORE=['states22.py','common22.py','prepare22.py','contract22.py','train22.py','score22.py','evaluate22.py','verify22.py']
def cfg():return read(ROOT/'protocol.json')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def old_evidence():
    r=dict(read(V21/'results/preparation_manifest.json')['old_evidence'])
    for n,d in read(V21/'results/delivery_manifest.json')['files'].items():r[str((V21/n).relative_to(PROJECT))]=d
    path=V21/'results/delivery_manifest.json';r[str(path.relative_to(PROJECT))]=sha(path);assert len(r)==3166
    for n,d in r.items():assert sha(PROJECT/n)==d,n
    return r
def source_hashes():
    r=previous.source_hashes();r.update({str((ROOT/n).relative_to(PROJECT)):sha(ROOT/n) for n in CORE});return r
def input_hashes():
    r=previous.input_hashes()
    for path in [V21/'protocol.json']+[V21/'results'/n for n in ['model_predictions.csv','ensemble_predictions.csv','hmm_signal_states.csv','verification.json','delivery_manifest.json']]:r[str(path.relative_to(PROJECT))]=sha(path)
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

"""Versioned parameter updates for the unchanged R49 prospective cohort."""
from pathlib import Path
import sys,uuid,copy
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'research_v49'))
import common49 as base
from common49 import pd,np,read,save,sha,guard,exclusive,encoded,iso,utc,TZ,Journal,HISTORIES,METHODS,PRIMARY,SEEDS,STATES,U,Q,W,I,S,annual_for,quarter_for,record_window
OUT=ROOT/'results';PARAMS=PROJECT/'prospective_r50'
def cfg():return read(ROOT/'protocol.json')
def cohort():return base.cfg()
def old_evidence():
    r=dict(read(PROJECT/'research_v49/results/freeze.json')['old_evidence'])
    previous=PROJECT/'research_v49';d=read(previous/'results/delivery_manifest.json')
    r.update({str((previous/n).relative_to(PROJECT)):h for n,h in d['files'].items()})
    r[str((previous/'results/delivery_manifest.json').relative_to(PROJECT))]=sha(previous/'results/delivery_manifest.json')
    r.update(read(previous/'results/runtime_initial_manifest.json')['files'])
    guard(len(r)==6230,'Unexpected old evidence count')
    for n,h in r.items():guard(sha(PROJECT/n)==h,f'Old evidence changed: {n}')
    return r
def check_freeze(verified=False):
    base.check_freeze();m=read(OUT/'freeze.json');guard(sha(ROOT/'protocol.json')==m['protocol_sha256'],'R50 protocol changed')
    for n,h in m['immutable_files'].items():guard(sha(PROJECT/n)==h,f'R50 frozen source/input changed: {n}')
    if verified:
        v=read(OUT/'verification.json');guard(v['status']=='PASS' and v['freeze_sha256']==sha(OUT/'freeze.json'),'R50 not verified')
    return m
def relative(p):return str(Path(p).resolve().relative_to(PROJECT))
def checked_path(name):
    p=(PROJECT/name).resolve();guard(p.is_relative_to(PROJECT),'Artifact escapes project');return p
def csv_save(path,frame):exclusive(path,frame.to_csv(index=False,lineterminator='\n').encode('utf-8'))
def load_csv(path):return pd.read_csv(path,float_precision='round_trip')
def quarter_end(cutoff):return (pd.Timestamp(cutoff)+pd.Timedelta(days=1)).to_period('Q').end_time.strftime('%Y-%m-%d')
_modules=None;_gpu=False
def modules(gpu=False):
    global _modules,_gpu
    if _modules is None:
        sys.path.insert(0,str(PROJECT/'research_v26'));import common26 as recipe
        sys.path.insert(0,str(PROJECT/'research_v47'));import solver47 as weighted
        sys.path.insert(0,str(PROJECT/'research_v39'));import common39 as quarter
        _modules=(recipe,weighted,quarter)
    if gpu and not _gpu:_modules[0].legacy.initialize();_gpu=True
    return _modules
def artifacts(folder):return {relative(p):sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}

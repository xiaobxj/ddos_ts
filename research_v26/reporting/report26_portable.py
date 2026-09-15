"""Run the unchanged report body in the installed Python plotting environment.

The GPU environment has no matplotlib. This adapter supplies file-only report
helpers, verifies frozen sources/inputs, and executes the exact frozen report
body. It neither imports the model runtime nor changes any model or score.
The original report failure is retained in results/logs/report.log.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset','native_mse','training_frequency']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def cfg():return read(ROOT/'protocol.json')
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def check_frozen():
    p=read(OUT/'preparation_manifest.json');assert sha(ROOT/'protocol.json')==p['protocol_sha256']
    for key in ['source_sha256','input_sha256']:
        for name,digest in p[key].items():assert sha(PROJECT/name)==digest,name
    for name,digest in p['artifacts'].items():assert sha(ROOT/name)==digest,name
    for phase in ['training','scoring','evaluation']:
        for name,digest in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/name)==digest,name
    assert read(OUT/'verification.json')['status']=='PASS';return p
def manifest(phase):
    p=check_frozen()
    return dict(phase=phase,started_utc=now(),protocol_sha256=p['protocol_sha256'],source_sha256=p['source_sha256'],input_sha256=p['input_sha256'],executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,report_only_adapter=str(Path(__file__).relative_to(ROOT)),report_only_adapter_sha256=sha(__file__),report_body_sha256=sha(ROOT/'report26.py'),reason='Existing GPU environment lacks matplotlib; use installed Python314 plotting stack. Frozen report body, all training/score/evaluation artifacts and original error log preserved.')
def finish(run,files,**extra):
    run.update(finished_utc=now(),artifacts={str(p.relative_to(ROOT)):sha(p) for p in files},**extra);save(OUT/f"{run['phase']}_manifest.json",run)
source=(ROOT/'report26.py').read_text(encoding='utf-8')
assert source.startswith('from common26 import *\n')
check_frozen()
exec(compile(source.split('\n',1)[1],str(ROOT/'report26.py'),'exec'),globals())

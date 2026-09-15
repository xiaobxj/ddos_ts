from pathlib import Path
import os
import sys
import json
import hashlib
import time

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent;V1=PROJECT/'research';V3=PROJECT/'research_v3';V4=PROJECT/'research_v4'
OUT=ROOT/'results';CACHE=ROOT/'cache'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(V4))
import numpy as np
import pandas as pd


def cfg():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def old_evidence():
    from util import previous_evidence
    result=previous_evidence()
    manifest=json.loads((V4/'results/delivery_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        assert sha(V4/name)==digest,name
        result[str((V4/name).relative_to(PROJECT))]=digest
    result[str((V4/'results/delivery_manifest.json').relative_to(PROJECT))]=sha(V4/'results/delivery_manifest.json')
    return result


def initialize():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    return torch


def stats(pred,y):
    pred=np.asarray(pred,float);y=np.asarray(y,float)
    return dict(mse=float(np.mean((pred-y)**2)),rmse=float(np.sqrt(np.mean((pred-y)**2))),
                correlation=float(np.corrcoef(pred,y)[0,1]) if pred.std()>1e-12 else None,
                forecast_std=float(pred.std()),accuracy=float(((pred>0)==(y>0)).mean()))


def source_hashes(phase):
    entry={'preparation':'prepare.py','capacity':'capacity.py','validation':'validate.py'}[phase]
    files=[ROOT/name for name in ['common5.py','models5.py',entry]]
    files+=[V4/'architecture.py',V4/'data.py',V4/'util.py']
    files+=list((V4/'vendor/Crossformer/cross_models').glob('*.py'))
    files+=list((V3/'vendor/Kronos/model').glob('*.py'))
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}


def start_manifest(name):
    torch=__import__('torch')
    return dict(name=name,started_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),
                source_sha256=source_hashes(name),old_evidence=old_evidence(),python=sys.version,executable=sys.executable,
                torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device=torch.cuda.get_device_name())

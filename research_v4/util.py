from pathlib import Path
import os
import sys
import hashlib
import json

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
V1=PROJECT/'research';V2=PROJECT/'research_v2';V3=PROJECT/'research_v3'
OUT=ROOT/'results';CACHE=ROOT/'cache'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
os.environ['HF_HUB_OFFLINE']='1'
sys.path.append(str(V3))
import numpy as np
import pandas as pd
from common import observation_table,ridge_path


def cfg():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def previous_evidence():
    return {str(p.relative_to(PROJECT)):sha(p) for folder in [V1,V2,V3] for p in folder.rglob('*')
            if p.is_file() and not any(part in p.parts for part in ['__pycache__','.git','vendor_py','vendor','models'])}


def initialize_torch():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Frozen protocol requires validated CUDA environment')
    return torch


def prediction_row(row,method,prediction,cutoff,train_n,seed=None):
    return dict(method=method,date=row.date,anchor=int(row.anchor),year=int(row.date[:4]),
                entry=int(row.entry),exit=int(row.exit),exit_date=row.exit_date,actual=float(row.exec_return),
                predicted_return=float(prediction),position=int(prediction>0),
                correct=int((prediction>0)==(row.exec_return>0)),cutoff=cutoff,train_n=int(train_n),seed=seed)

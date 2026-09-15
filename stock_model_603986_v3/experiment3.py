"""Isolated continuation of sealed stock v2; no writes to previous evidence."""
from pathlib import Path
import sys,json,datetime,importlib.util
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';V2=PROJECT/'stock_model_603986_v2'
sys.path.insert(0,str(V2))
import core as v2
np=v2.np;pd=v2.pd;old=v2.old;SEEDS=v2.SEEDS;METHODS=v2.METHODS
sha=v2.sha;read=v2.read;save=v2.save;csv=v2.csv;now=v2.now
modules=v2.modules;observations=v2.observations;membership=v2.membership;interface=v2.interface;task_id=v2.task_id
maximal_nonoverlap=v2.maximal_nonoverlap
def load_csv(p):return pd.read_csv(p,float_precision='round_trip',low_memory=False)
def cfg():return read(ROOT/'protocol.json')
def data():return load_csv(V2/'data/model_frame.csv')
def rel(p):return str(Path(p).resolve().relative_to(PROJECT))
def tasks(h=5):return [dict(h=5,objective='mse',years=5,cutoff=c,cadence='quarterly_full') for c in cfg()['new_cutoffs']]
def check_freeze():
    d=read(OUT/'freeze.json')
    for p,s in d['files'].items():assert sha(PROJECT/p)==s,p
    return d
def fit_folder(c,h=5):
    name=f'h{h}_mse_5y_{c}';p=OUT/'fits'/name
    return p if (p/'completed.json').exists() else V2/'results/fits'/name
def load_model(folder,seed,epoch=20):
    r,_,_=modules();path=folder/f'model_{seed}_e{epoch}.pt';s=r.torch.load(path,map_location='cpu',weights_only=True)
    base=ROOT if folder.is_relative_to(ROOT) else V2
    assert s['protocol_sha256']==sha(base/'protocol.json') and s['freeze_sha256']==sha(base/'results/freeze.json')
    assert r.training.object_hash(s['state_dict'])==s['model_sha256']
    m=r.legacy.make_model('combined',seed);m.load_state_dict(s['state_dict']);m.eval();m.requires_grad_(False)
    return m,s

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

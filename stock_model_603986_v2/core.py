"""Daily H1/H5 stock experiment; legacy numerical functions are read-only."""
from pathlib import Path
import sys, json, hashlib, datetime
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; OUT=ROOT/'results'
sys.path.insert(0,str(PROJECT/'stock_model_603986_v1'))
import stock as old
import numpy as np
import pandas as pd
sha=old.sha; read=old.read; save=old.save; csv=old.csv; now=old.now; load_csv=old.load_csv
SEEDS=[20260910,20260911,20260912]
METHODS=['add','vol','order_joint','order_fixed']

def cfg():return read(ROOT/'protocol.json')
def rel(p):return str(Path(p).resolve().relative_to(PROJECT))
def modules(init=False):return old.modules(init)
def data():return load_csv(ROOT/'data/model_frame.csv')
def check_freeze():
    d=read(OUT/'freeze.json')
    for p,s in d['files'].items():assert sha(PROJECT/p)==s,p
    assert read(OUT/'contract.json')['status']=='PASS'
    return d

def observations(frame,h):
    """Input-valid anchors include pending targets. Labels require every future bar."""
    assert h in [1,5]
    b=frame[['open','high','low','close','volume']].to_numpy(float)
    good=frame.valid_ohlc.to_numpy(bool); rows=[]; aux=[]; rets=[]
    for t in range(124,len(frame)):
        if not good[t-124:t+1].all():continue
        e=t+h; matured=e<len(frame) and good[t+1:e+1].all()
        ret=np.nan; label=np.full(5*h,np.nan); completed=''
        if matured:
            # Start at close t: cash distributions on t are already ex entitlement.
            denominator=frame.shares.iloc[t]*frame.raw_close.iloc[t]
            numerator=frame.shares.iloc[e]*frame.raw_close.iloc[e]+frame.cash.iloc[e]-frame.cash.iloc[t]
            ret=float(numerator/denominator-1); completed=frame.date.iloc[e]
            future=b[t+1:e+1]
            label=np.c_[np.log(future[:,:4]/b[t,3]),np.log1p(future[:,4])-np.log1p(b[t-19:t+1,4]).mean()].reshape(-1)
        rows.append(dict(row_index=len(rows),anchor=t,date=frame.date.iloc[t],exit=e,
            joint_completed=completed,matured=bool(matured),actual_return=ret,
            actual_up=float(ret>0) if matured else np.nan,year=int(frame.date.iloc[t][:4])))
        aux.append(label);rets.append(ret)
    return pd.DataFrame(rows),dict(returns=np.asarray(rets),auxiliary=np.asarray(aux))

def membership(obs,cutoff,years):
    lower=(pd.Timestamp(cutoff)-pd.DateOffset(years=years)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    ids=np.flatnonzero(obs.date.ge(lower)&obs.matured&obs.joint_completed.le(cutoff))
    assert len(ids)>=600,(cutoff,years,len(ids))
    assert obs.date.iloc[ids].lt(cutoff).all()
    return ids,lower

def tasks(h):
    result=[]
    for objective,years in [('mse',5),('bce',5),('mse',3)]:
        for c in cfg()['annual_cutoffs']:
            result.append(dict(h=h,objective=objective,years=years,cutoff=c,cadence='annual'))
    for c in cfg()['quarterly_full_cutoffs']:
        result.append(dict(h=h,objective='mse',years=5,cutoff=c,cadence='quarterly_2026'))
    return result

def task_id(t):return f"h{t['h']}_{t['objective']}_{t['years']}y_{t['cutoff']}"
def interface(frame,t):
    obs,target=observations(frame,t['h']);ids,lower=membership(obs,t['cutoff'],t['years'])
    r,_,_=modules();labels,scales=r.scales_for(target,ids)
    return dict(obs=obs,targets=target,ids=ids,lower=lower,labels=labels,scales=scales,
        values=old.packed(frame,obs.anchor.to_numpy(int)),market=r.market(frame,obs,np.arange(len(obs))),
        gate=r.gates(frame,obs,np.arange(len(obs))),y=target['returns'][ids].__gt__(0).astype(float))

def model_load(path):
    r,_,_=modules();s=r.torch.load(path,map_location='cpu',weights_only=True)
    assert s['protocol_sha256']==sha(ROOT/'protocol.json')
    assert s['freeze_sha256']==sha(OUT/'freeze.json')
    assert r.training.object_hash(s['state_dict'])==s['model_sha256']
    m=r.legacy.make_model('combined',s['seed']);m.load_state_dict(s['state_dict']);m.eval();m.requires_grad_(False)
    assert sum(p.numel() for p in m.parameters())==38551
    return m,s

def maximal_nonoverlap(rows):
    """Maximum disjoint return-edge intervals, NOT effective independent N."""
    end=-1; count=0
    for row in rows.sort_values(['exit','anchor']).itertuples():
        if row.anchor>=end:count+=1;end=row.exit
    return count

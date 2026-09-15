"""Stock-only data contract and adapters to the frozen numerical recipe."""
from pathlib import Path
import sys, json, hashlib, datetime
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; OUT=ROOT/'results'
sys.path.insert(0,str(PROJECT/'research_v50'))
from common50 import modules, SEEDS, METHODS, HISTORIES, STATES, U, Q, W, I, S
import numpy as np
import pandas as pd

CALENDAR=PROJECT/'prospective_r49/blobs/07ec96b57823f2db39058d65d0fa5a9634ec22d9c20cde893b1ca7a1d11bb5f5.csv'
ACTIONS=[('2017-05-22',2.,.53),('2018-05-21',1.4,.393),('2019-06-03',1.,.285),
         ('2020-05-07',1.4,.38),('2021-05-21',1.4,.56),('2022-06-02',1.,1.06),
         ('2023-06-20',1.,.62),('2025-07-03',1.,.34),('2026-05-26',1.,.75)]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def csv(p,f):
    with Path(p).open('x',encoding='utf-8',newline='') as stream:f.to_csv(stream,index=False,lineterminator='\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load_csv(p):return pd.read_csv(p,float_precision='round_trip')
def cfg():return read(ROOT/'protocol.json')
def relative(p):return str(Path(p).resolve().relative_to(PROJECT))

def build_frame(raw,calendar):
    """Forward wealth OHLC: split shares * raw price + cumulative nominal cash.

    At time t only actions on/before t enter. Volume is IPO-equivalent lots.
    Missing exchange sessions remain NaN and prohibit an input/label window.
    """
    assert raw.date.is_unique and raw.date.is_monotonic_increasing
    f=pd.DataFrame({'date':calendar}).merge(raw,on='date',how='left',validate='one_to_one')
    shares=np.ones(len(f));cash=np.zeros(len(f));s=1.;c=0.
    for date,ratio,dividend in ACTIONS:
        c+=s*dividend;s*=ratio;mask=f.date.ge(date)
        shares[mask]=s;cash[mask]=c
    for col in ['open','high','low','close']:
        f['raw_'+col]=f[col];f[col]=f[col]*shares+cash
    f['volume']=f.volume/shares;f['shares']=shares;f['cash']=cash
    good=np.isfinite(f[['open','high','low','close','volume']]).all(axis=1)
    good &= (f[['open','high','low','close']]>0).all(axis=1)&f.volume.gt(0)
    good &= f.high.ge(f[['open','close','low']].max(axis=1))&f.low.le(f[['open','close','high']].min(axis=1))
    f['valid_ohlc']=good
    return f

def data():return load_csv(ROOT/'data/model_frame.csv')

def observations(frame,cutoff):
    f=frame[frame.date.le(cutoff)].reset_index(drop=True);n=len(f)
    days=pd.to_datetime(f.date).dt.dayofweek.to_numpy();nxt=np.full(n,-1,int)
    for day in range(5):
        ids=np.flatnonzero(days==day);nxt[ids[:-1]]=ids[1:]
    bars=f[['open','high','low','close','volume']].to_numpy(float);good=f.valid_ohlc.to_numpy(bool)
    rows=[];aux=[];returns=[]
    for t in range(124,n-5):
        e=int(nxt[t])+1
        if nxt[t]<0 or e>=n:continue
        end=max(t+5,e)
        if not good[t-124:end+1].all():continue
        future=bars[t+1:t+6]
        prices=np.log(future[:,:4]/bars[t,3]);vol=np.log1p(future[:,4])-np.log1p(bars[t-19:t+1,4]).mean()
        # One share bought at next open; split-adjusted shares and entitled cash
        # through exit. Cash on entry ex-date is excluded; cash on exit included.
        entry_value=float(f.shares.iloc[t+1]*f.raw_open.iloc[t+1])
        ret=float((f.shares.iloc[e]*f.raw_open.iloc[e]+f.cash.iloc[e]-f.cash.iloc[t+1])/entry_value-1)
        rows.append(dict(anchor=t,date=f.date.iloc[t],weekday=int(days[t]),entry=t+1,exit=e,
            entry_date=f.date.iloc[t+1],exit_date=f.date.iloc[e],auxiliary_completed=f.date.iloc[t+5],
            joint_completed=f.date.iloc[end],actual_return=ret,actual_up=int(ret>0),
            holding_calendar_days=int((pd.Timestamp(f.date.iloc[e])-pd.Timestamp(f.date.iloc[t+1])).days)))
        aux.append(np.c_[prices,vol].reshape(-1));returns.append(ret)
    obs=pd.DataFrame(rows);obs.insert(0,'row_index',np.arange(len(obs)))
    return f,obs,dict(returns=np.asarray(returns,float),auxiliary=np.asarray(aux,float))

def packed(frame,anchors):
    bars=frame[['open','high','low','close','volume']].to_numpy(float);windows=[]
    for t in anchors:
        w=bars[t-124:t+1].copy();assert w.shape==(125,5) and np.isfinite(w).all()
        w[:,:4]=np.log(w[:,:4]);w[:,4]=np.log1p(w[:,4])
        windows.append(((w-w.mean(0))/np.maximum(w.std(0),1e-6)).astype(np.float32))
    a=np.asarray(windows);geometry=np.zeros((len(a),25,2),np.float32)
    for j in range(25):geometry[:,j]=[5/125,(j+1)*5/125]
    return dict(patches=a.reshape(len(a),25,5,5),geometry=geometry,valid=np.ones((len(a),25),bool))

def annual_interface(frame,cutoff):
    f,obs,targets=observations(frame,cutoff)
    lower=(pd.Timestamp(cutoff)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    ids=np.flatnonzero(obs.date.ge(lower)&obs.joint_completed.le(cutoff));assert len(ids)>=800
    r,_,_=modules();labels,scales=r.scales_for(targets,ids)
    return dict(frame=f,observations=obs,training_rows=ids,training_dates=obs.date.iloc[ids].reset_index(drop=True),
        values=packed(f,obs.anchor.iloc[ids].to_numpy(int)),labels=labels,scales=scales,
        market=r.market(f,obs,ids),gate=r.gates(f,obs,ids),y=targets['returns'][ids].__gt__(0).astype(float),lower=lower)

def check_freeze():
    freeze=read(OUT/'freeze.json')
    for name,digest in freeze['files'].items():assert sha(PROJECT/name)==digest,name
    assert read(OUT/'contract.json')['status']=='PASS'
    return freeze

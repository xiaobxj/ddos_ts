"""Reconstruct the original daily joint target using only a cutoff prefix."""
from common50 import *
from data49 import clean,unchanged_prefix

def observations(frame,cutoff):
    f=clean(frame[frame.date.le(cutoff)].reset_index(drop=True))
    dates=pd.to_datetime(f.date);days=dates.dt.dayofweek.to_numpy();n=len(f);nxt=np.full(n,-1,int)
    for day in range(5):
        ids=np.flatnonzero(days==day);nxt[ids[:-1]]=ids[1:]
    rows=[];aux=[];returns=[];bars=f[['open','high','low','close','volume']].to_numpy(float)
    good=f.valid_ohlc.to_numpy(bool);v4row=-1
    for t in range(124,n-5):
        e=int(nxt[t])+1
        if nxt[t]<0 or e>=n:continue
        end=max(t+5,e)
        if not good[t-124:end+1].all():continue
        v4row+=1;future=bars[t+1:t+6];o,h,l,c,volume=future.T
        if not ((h>=np.maximum(o,c)).all() and (l<=np.minimum(o,c)).all() and (h>=l).all()):continue
        prices=np.log(future[:,:4]/bars[t,3]);vol=np.log1p(volume)-np.log1p(bars[t-19:t+1,4]).mean()
        target=np.c_[prices,vol].reshape(-1);ret=float(bars[e,0]/bars[t+1,0]-1)
        rows.append(dict(anchor=t,date=f.date.iloc[t],weekday=int(days[t]),entry=t+1,exit=e,exit_date=f.date.iloc[e],completed=f.date.iloc[end],auxiliary_completed=f.date.iloc[t+5],joint_completed=f.date.iloc[end],exec_return=ret,v4_row=v4row));aux.append(target);returns.append(ret)
    obs=pd.DataFrame(rows);guard(len(obs)>0,'No mature joint observations')
    obs.insert(0,'row_index',np.arange(len(obs)))
    return f,obs,dict(returns=np.asarray(returns,float),auxiliary=np.asarray(aux,float))
def packed(frame,anchors):
    bars=frame[['open','high','low','close','volume']].to_numpy(float);windows=[]
    for t in anchors:
        w=bars[t-124:t+1].copy();guard(w.shape==(125,5),'Incomplete training window')
        w[:,:4]=np.log(w[:,:4]);w[:,4]=np.log1p(w[:,4]);windows.append(((w-w.mean(0))/np.maximum(w.std(0),1e-6)).astype(np.float32))
    a=np.asarray(windows);geometry=np.zeros((len(a),25,2),np.float32)
    for j in range(25):geometry[:,j]=[5/125,(j+1)*5/125]
    return dict(patches=a.reshape(len(a),25,5,5),geometry=geometry,valid=np.ones((len(a),25),bool))
def annual_interface(frame,cutoff):
    f,obs,targets=observations(frame,cutoff);lower=(pd.Timestamp(cutoff)-pd.DateOffset(years=5)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    ids=np.flatnonzero(obs.date.ge(lower)&obs.joint_completed.le(cutoff));guard(len(ids)>1000,'Insufficient five-year mature training data')
    recipe,_,_=modules();labels,scales=recipe.scales_for(targets,ids);v=packed(f,obs.anchor.iloc[ids].to_numpy(int))
    mf=recipe.market(f,obs,ids);g=recipe.gates(f,obs,ids)
    return dict(frame=f,observations=obs,training_rows=ids,training_dates=obs.date.iloc[ids].reset_index(drop=True),values=v,labels=labels,scales=scales,market=mf,gate=g,y=(targets['returns'][ids]>0).astype(float),lower=lower)

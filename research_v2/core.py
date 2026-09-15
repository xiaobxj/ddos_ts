"""Causal features and completed-label masks for the frozen second experiment."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "research"
sys.path.insert(0, str(V1))
from segmentation import ending_costs, reverse_lengths, fixed_lengths, encode_patches, transformed_bars


def observation_table(frame):
    dates = pd.to_datetime(frame.date)
    weekday = dates.dt.dayofweek.to_numpy()
    n = len(frame)
    next_anchor = np.full(n, -1, dtype=int)
    for day in range(5):
        ids = np.flatnonzero(weekday==day)
        next_anchor[ids[:-1]] = ids[1:]
    o, c = frame.open.to_numpy(float), frame.close.to_numpy(float)
    good = frame.valid_ohlc.to_numpy(bool)
    rows=[]
    for t in range(124, n-5):
        e = next_anchor[t]+1
        if next_anchor[t]<0 or e>=n:
            continue
        end=max(t+5,e)
        # Common sample mask for target ablations, preserving trading calendar.
        if not good[t-124:end+1].all():
            continue
        rows.append(dict(anchor=t, date=str(frame.date.iloc[t]), weekday=int(weekday[t]),
                         entry=t+1, exit=e, exit_date=str(frame.date.iloc[e]),
                         close_end=t+5, completed=str(frame.date.iloc[end]),
                         close_return=c[t+5]/c[t]-1, exec_return=o[e]/o[t+1]-1,
                         sessions=e-(t+1)))
    return pd.DataFrame(rows)


def economic_window(window):
    a=window[["open","high","low","close","volume"]].to_numpy(float,copy=True)
    op,hi,lo,cl,vol=a.T
    close_log=np.log(cl)
    ret=np.r_[0.,np.diff(close_log)]
    gap=np.r_[0.,np.log(op[1:]/cl[:-1])]
    body=np.log(cl/op)
    spread=np.log(hi/lo)
    upper=np.log(hi/np.maximum(op,cl))
    lower=np.log(np.minimum(op,cl)/lo)
    # Prefix-only volume scale, never a full-series normalization.
    cs=np.r_[0.,np.cumsum(vol)]
    scale=np.array([(cs[k+1]-cs[max(0,k-19)])/(k+1-max(0,k-19)) for k in range(len(vol))])
    relative_volume=np.log1p(vol)-np.log1p(scale)
    location=(2*cl-hi-lo)/np.maximum(hi-lo,1e-8)
    channels=np.column_stack([ret,gap,body,spread,upper,lower,relative_volume,location])
    feature=list(channels[-5:].reshape(-1))
    for length in [5,10,20,60,125]:
        r=ret[-length:]
        feature.extend([r.sum(),r.std(),np.sqrt(np.mean(np.minimum(r,0)**2)),spread[-length:].mean(),
                        relative_volume[-length:].mean(),cl[-1]/cl[-length:].max()-1,
                        close_log[-1]-close_log[-length:].mean()])
    assert len(feature)==75
    return channels,np.asarray(feature)


def stats_for_segment(segment):
    return [segment[:,0].sum(),segment[:,0].std(),segment[:,3].mean(),segment[:,6].mean()]


def multi_features(channels,base):
    out=list(base)
    for length in [5,10,20]:
        start=0
        for size in fixed_lengths(125,length):
            out.extend(stats_for_segment(channels[start:start+size]))
            start+=size
    assert len(out)==255
    return np.asarray(out)


def adaptive_features(channels,base,lengths):
    out=np.zeros((25,7))
    offset=25-len(lengths)
    start=0
    for j,size in enumerate(lengths):
        out[offset+j,:4]=stats_for_segment(channels[start:start+size])
        out[offset+j,4:]=[size/125,(start+size)/125,1.]
        start+=size
    assert start==125
    return np.r_[base,out.reshape(-1)]


def static_features(frame,obs):
    econ,multi=[],[]
    for t in obs.anchor.to_numpy(int):
        channels,base=economic_window(frame.iloc[t-124:t+1])
        econ.append(base)
        multi.append(multi_features(channels,base))
    return dict(econ=np.array(econ),multi=np.array(multi))


def prototype_features(frame,obs,prototypes):
    _,norm=ending_costs(frame.close.to_numpy(float),prototypes)
    bars=transformed_bars(frame,["open","high","low","close","volume"])
    legacy,adapt=[],[]
    for t in obs.anchor.to_numpy(int):
        lengths=reverse_lengths(norm,t+1)
        window=bars[t-124:t+1]
        z=(window-window.mean(axis=0))/np.maximum(window.std(axis=0),1e-6)
        legacy.append(encode_patches(z,lengths))
        channels,base=economic_window(frame.iloc[t-124:t+1])
        adapt.append(adaptive_features(channels,base,lengths))
    return dict(legacy=np.array(legacy),adaptive=np.array(adapt))


def fold_masks(obs,cutoff,period_end,memory):
    train=obs.completed.to_numpy()<=cutoff
    if memory:
        lower=(pd.Timestamp(cutoff)+pd.Timedelta(days=1)-pd.DateOffset(years=memory)).strftime('%Y-%m-%d')
        train &= obs.date.to_numpy()>=lower
    test=(obs.date.to_numpy()>cutoff)&(obs.date.to_numpy()<=period_end)&(obs.weekday.to_numpy()==4)
    assert train.any()
    assert obs.completed[train].max()<=cutoff
    return train,test


def scale_train(x,xt):
    mean=x.mean(axis=0)
    sd=x.std(axis=0)
    sd[sd<1e-6]=1.
    return (x-mean)/sd,(xt-mean)/sd


def ridge_path(x,y,xt,lambdas):
    z,zt=scale_train(x,xt)
    ym=y.mean()
    ev,vec=linalg.eigh(z.T@z/len(z),check_finite=False)
    rhs=vec.T@(z.T@(y-ym)/len(z))
    left=zt@vec
    return {str(l):left@(rhs/(np.maximum(ev,0)+l))+ym for l in lambdas}


def logistic_path(x,y,xt,lambdas):
    z,zt=scale_train(x,xt)
    target=(y>0).astype(float)
    initial=np.r_[np.log((target.mean()+1e-6)/(1-target.mean()+1e-6)),np.zeros(z.shape[1])]
    predictions,diagnostics={},[]
    for lam in reversed(lambdas):
        def fun(theta):
            scores=theta[0]+z@theta[1:]
            loss=np.mean(np.logaddexp(0,scores)-target*scores)+lam/2*np.sum(theta[1:]**2)
            error=expit(scores)-target
            grad=np.r_[error.mean(),z.T@error/len(z)+lam*theta[1:]]
            return loss,grad
        fit=minimize(fun,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-11,gtol=1e-7))
        if not fit.success:
            raise RuntimeError(f'logistic solver did not converge: {fit.message}')
        initial=fit.x
        predictions[str(lam)]=expit(initial[0]+zt@initial[1:])
        diagnostics.append(dict(lam=lam,iterations=int(fit.nit),success=bool(fit.success)))
    return predictions,diagnostics

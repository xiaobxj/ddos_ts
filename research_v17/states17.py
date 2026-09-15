"""Observable market descriptors and fixed, label-free annual partitions."""
import numpy as np
import pandas as pd

PARTITIONS={'trend_volatility':[f'{t}__{v}' for t in ['low_trend','mid_trend','high_trend'] for v in ['low_vol','high_vol']],
    'volatility':['low_vol','high_vol'],'range':['low_range','high_range'],'volume':['low_volume_change','high_volume_change']}

def descriptors(bars,anchors):
    bars=np.asarray(bars,float);a=np.asarray(anchors,int);assert bars.shape[1]==5 and np.all(a>=60)
    pos=a[:,None]-np.arange(59,-1,-1)[None,:]
    r=np.log(bars[pos,3])-np.log(bars[pos-1,3]);vol60=r.std(axis=1)
    trend=(np.log(bars[a,3])-np.log(bars[a-60,3]))/(np.maximum(vol60,1e-8)*np.sqrt(60.))
    pos20=pos[:,-20:];w=bars[pos20]
    out=pd.DataFrame(dict(trend60=trend,volatility20=r[:,-20:].std(axis=1),
        range20=(np.log(w[:,:,1])-np.log(w[:,:,2])).mean(axis=1),
        volume_change20=np.log1p(bars[a,4])-np.log1p(bars[a-20,4])))
    assert np.isfinite(out.to_numpy()).all();return out

def scalar_descriptors(bars,anchors):
    rows=[]
    for t in anchors:
        r=np.diff(np.log(bars[t-60:t+1,3]));r20=np.diff(np.log(bars[t-20:t+1,3]))
        rows.append(dict(trend60=(np.log(bars[t,3])-np.log(bars[t-60,3]))/(max(float(np.std(r,ddof=0)),1e-8)*np.sqrt(60)),
            volatility20=float(np.std(r20,ddof=0)),range20=float(np.mean(np.log(bars[t-19:t+1,1])-np.log(bars[t-19:t+1,2]))),
            volume_change20=float(np.log1p(bars[t,4])-np.log1p(bars[t-20,4]))))
    return pd.DataFrame(rows)

def fit_thresholds(d):
    t=np.quantile(d.trend60.to_numpy(),[1/3,2/3],method='linear');assert t[0]<t[1]
    return dict(trend_lower=float(t[0]),trend_upper=float(t[1]),volatility_median=float(d.volatility20.median()),
        range_median=float(d.range20.median()),volume_median=float(d.volume_change20.median()))

def assign_states(d,thresholds):
    t=thresholds;out=d.copy()
    out['trend']=np.where(d.trend60<=t['trend_lower'],'low_trend',np.where(d.trend60<=t['trend_upper'],'mid_trend','high_trend'))
    out['volatility']=np.where(d.volatility20<=t['volatility_median'],'low_vol','high_vol')
    out['range']=np.where(d.range20<=t['range_median'],'low_range','high_range')
    out['volume']=np.where(d.volume_change20<=t['volume_median'],'low_volume_change','high_volume_change')
    out['trend_volatility']=out.trend+'__'+out.volatility
    return out

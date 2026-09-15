"""Permutation-centered adjacent sign products; fixed60bar order descriptor."""
import numpy as np
import pandas as pd
import states22 as prior
NUMERIC=prior.NUMERIC+['positive_count60','negative_count60','neutral_count60','observed_lag_product60','permutation_expectation60','excess_order']
GATES=['excess_order'];STATE_PARTITIONS={'order_state':['negative_order','near_zero_order','positive_order']}

def sign_statistic(s):
    s=np.asarray(s,float);n=len(s);assert n>=2 and np.isin(s,[-1,0,1]).all()
    observed=float(np.mean(s[:-1]*s[1:]));expected=float((s.sum()**2-np.square(s).sum())/(n*(n-1)))
    return observed,expected,observed-expected

def features(price):
    f=prior.features(price);close=price.close.to_numpy(float);values=np.full((len(price),6),np.nan)
    if len(price)>60:
        signs=np.sign(np.diff(close));w=np.lib.stride_tricks.sliding_window_view(signs,60);observed=(w[:,:-1]*w[:,1:]).mean(axis=1);expected=(np.square(w.sum(axis=1))-np.square(w).sum(axis=1))/(60*59)
        z=np.column_stack([(w>0).sum(axis=1),(w<0).sum(axis=1),(w==0).sum(axis=1),observed,expected,observed-expected]);values[60:]=z
    valid=f.relative_volatility.notna().to_numpy();values[~valid]=np.nan
    for j,c in enumerate(NUMERIC[5:]):f[c]=values[:,j]
    f['order_state']=np.where(valid,np.where(f.excess_order<-.05,'negative_order',np.where(f.excess_order>.05,'positive_order','near_zero_order')),'missing');return f

def scalar_features(price,anchors):
    base=prior.scalar_features(price,anchors);result=[]
    for anchor,row in zip(anchors,base):
        if not np.isfinite(row).all():result.append([np.nan]*6);continue
        close=price.close.iloc[anchor-60:anchor+1].to_numpy(float);s=[1 if close[i]>close[i-1] else -1 if close[i]<close[i-1] else 0 for i in range(1,61)]
        positive=s.count(1);negative=s.count(-1);neutral=s.count(0);observed=sum(s[i]*s[i-1] for i in range(1,60))/59
        expected=(positive*(positive-1)+negative*(negative-1)-2*positive*negative)/(60*59)
        result.append([positive,negative,neutral,observed,expected,observed-expected])
    return np.column_stack([base,np.asarray(result)])

def run_diagnostics(frame,gate):
    # Reuse checked run-boundary implementation with the new three-level labels.
    temporary=frame.copy();temporary['persistence_state']=np.where(temporary.order_state.eq('positive_order'),'persistent',temporary.order_state)
    d,runs=prior.run_diagnostics(temporary,gate)
    for r in runs:
        if r['state']=='persistent':r['state']='positive_order'
    return d,runs

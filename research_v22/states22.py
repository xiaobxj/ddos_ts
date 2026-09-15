"""Fixed trailing log-price state coordinates; no fitted state parameters."""
import numpy as np
import pandas as pd
NUMERIC=['sd20','sd120','relative_volatility','efficiency60','persistence']
GATES=['relative_volatility','persistence']
STATE_PARTITIONS={'relative_volatility_state':['subdued','elevated'],'persistence_state':['choppy','persistent'],
    'relative_joint':[f'{a}__{b}' for a in ['choppy','persistent'] for b in ['subdued','elevated']]}

def features(price):
    close=price.close.to_numpy(float);valid=price.valid_ohlc.to_numpy(bool)&np.isfinite(close)&(close>0);n=len(close);out=np.full((n,5),np.nan)
    if n>120:
        safe=np.where(valid,close,1.);d=np.diff(np.log(safe));windows=np.lib.stride_tricks.sliding_window_view(d,120)
        ok=np.lib.stride_tricks.sliding_window_view(valid,121).all(axis=1);s20=windows[:,-20:].std(axis=1);s120=windows.std(axis=1)
        rv=(s20-s120)/np.maximum(s20+s120,1e-8);er=np.clip(abs(windows[:,-60:].sum(axis=1))/np.maximum(abs(windows[:,-60:]).sum(axis=1),1e-8),0.,1.)
        values=np.column_stack([s20,s120,rv,er,2*er-1]);out[120:][ok]=values[ok]
    frame=pd.DataFrame(out,columns=NUMERIC);frame.insert(0,'date',price.date.to_numpy());frame.insert(0,'anchor',np.arange(n));return assign(frame)

def assign(frame):
    f=frame.copy();ok=f.relative_volatility.notna()&f.efficiency60.notna()
    f['relative_volatility_state']=np.where(ok,np.where(f.relative_volatility>0,'elevated','subdued'),'missing')
    f['persistence_state']=np.where(ok,np.where(f.efficiency60>=.2,'persistent','choppy'),'missing')
    f['relative_joint']=np.where(ok,f.persistence_state+'__'+f.relative_volatility_state,'missing');return f

def scalar_features(price,anchors):
    result=[]
    for i in anchors:
        if i<120:result.append([np.nan]*5);continue
        segment=price.iloc[i-120:i+1];c=segment.close.to_numpy(float)
        if not (segment.valid_ohlc.all() and np.isfinite(c).all() and (c>0).all()):result.append([np.nan]*5);continue
        r=np.array([np.log(c[j]/c[j-1]) for j in range(1,121)])
        def sd(v):return np.sqrt(sum(float(x-v.mean())**2 for x in v)/len(v))
        short=sd(r[-20:]);long=sd(r);rv=(short-long)/max(short+long,1e-8);eff=min(1.,max(0.,abs(np.log(c[-1]/c[-61]))/max(sum(abs(x) for x in r[-60:]),1e-8)))
        result.append([short,long,rv,eff,2*eff-1])
    return np.asarray(result)

def run_diagnostics(frame,gate):
    column='relative_volatility_state' if gate=='relative_volatility' else 'persistence_state';high='elevated' if gate=='relative_volatility' else 'persistent'
    valid=frame[gate].notna().to_numpy();idx=np.flatnonzero(valid);g=frame.iloc[idx];adj=(np.diff(g.anchor.to_numpy())==1)
    state=g[column].to_numpy();values=g[gate].to_numpy();runs=[];start=0
    for k in range(1,len(g)+1):
        if k==len(g) or not adj[k-1] or state[k]!=state[k-1]:
            left=(start==0 or not adj[start-1]);right=(k==len(g) or not adj[k-1]);runs.append(dict(state=state[start],first_anchor=int(g.anchor.iloc[start]),last_anchor=int(g.anchor.iloc[k-1]),
                first_date=g.date.iloc[start],last_date=g.date.iloc[k-1],length=k-start,left_censored=bool(left),right_censored=bool(right)));start=k
    changes=int(np.sum((state[1:]!=state[:-1])&adj));corr=float(np.corrcoef(values[:-1][adj],values[1:][adj])[0,1]) if adj.sum()>1 and values[:-1][adj].std()>0 and values[1:][adj].std()>0 else None
    return dict(n=len(g),missing=int((~valid).sum()),mean=float(values.mean()),std=float(values.std()),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),high_state_share=float((state==high).mean()),
        adjacent_pairs=int(adj.sum()),switches=changes,switch_rate=changes/int(adj.sum()) if adj.sum() else None,lag1_correlation=corr,runs=len(runs),censored_runs=sum(r['left_censored'] or r['right_censored'] for r in runs),
        median_observed_run=float(np.median([r['length'] for r in runs])),max_observed_run=max(r['length'] for r in runs)),runs

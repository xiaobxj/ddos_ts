"""Fixed causal, non-learned OHLCV summaries; no fitted feature parameters."""
import numpy as np

HORIZONS=(5,10,20,60,120)
KINDS=('close_log_change','close_log_return_population_sd','mean_intraday_log_return','mean_high_low_log_range','log1p_volume_change')
NAMES=[f'{kind}_{h}' for h in HORIZONS for kind in KINDS]


def raw_features(bars,anchors):
    bars=np.asarray(bars,float);anchors=np.asarray(anchors,int);result=[]
    assert bars.shape[1]==5 and np.all(anchors>=120)
    for h in HORIZONS:
        positions=anchors[:,None]-np.arange(h-1,-1,-1)[None,:]
        window=bars[positions];prior=bars[positions-1,3]
        assert np.all(window[:,:,:4]>0) and np.all(window[:,:,4]>=0) and np.all(prior>0)
        log=np.log(window[:,:,:4]);changes=log[:,:,3]-np.log(prior)
        result.extend([np.log(bars[anchors,3])-np.log(bars[anchors-h,3]),changes.std(axis=1),
            (log[:,:,3]-log[:,:,0]).mean(axis=1),(log[:,:,1]-log[:,:,2]).mean(axis=1),
            np.log1p(bars[anchors,4])-np.log1p(bars[anchors-h,4])])
    result=np.column_stack(result);assert result.shape==(len(anchors),25) and np.isfinite(result).all()
    return result


def scalar_features(bars,anchors):
    # Separate straightforward reference implementation for formula checks.
    rows=[]
    for t in anchors:
        row=[]
        for h in HORIZONS:
            w=bars[t-h+1:t+1];closes=bars[t-h:t+1,3]
            row.extend([np.log(closes[-1])-np.log(closes[0]),np.std(np.diff(np.log(closes)),ddof=0),
                np.mean(np.log(w[:,3])-np.log(w[:,0])),np.mean(np.log(w[:,1])-np.log(w[:,2])),
                np.log1p(bars[t,4])-np.log1p(bars[t-h,4])])
        rows.append(row)
    return np.asarray(rows,float)

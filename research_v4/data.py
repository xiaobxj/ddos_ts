"""Observed-prefix window transforms and explicit variable-patch packing."""
from util import *


def observed_window(frame,anchor):
    return frame.iloc[anchor-124:anchor+1][['open','high','low','close','volume']].to_numpy(float,copy=True)


def raw_transform(window):
    values=window.copy()
    values[:,:4]=np.log(values[:,:4]);values[:,4]=np.log1p(values[:,4])
    return ((values-values.mean(axis=0))/np.maximum(values.std(axis=0),1e-6)).astype(np.float32)


def tokenizer_transform(window):
    values=np.c_[window,np.zeros(len(window))].astype(np.float32)
    values=(values-values.mean(axis=0))/(values.std(axis=0)+np.float32(1e-5))
    return np.clip(values,-5,5).astype(np.float32)


def pack(windows,lengths):
    n,lookback,dims=windows.shape
    assert lookback==125 and lengths.shape==(n,25)
    x=np.zeros((n,25,5,dims),dtype=np.float32)
    geometry=np.zeros((n,25,2),dtype=np.float32)
    valid=np.zeros((n,25),dtype=bool)
    for i in range(n):
        pieces=lengths[i][lengths[i]>0]
        assert pieces.sum()==125
        start=0;offset=25-len(pieces)
        for j,size in enumerate(pieces):
            positions=np.linspace(0,int(size)-1,5)
            left=positions.astype(int);right=np.minimum(left+1,size-1)
            weight=(positions-left)[:,None]
            x[i,offset+j]=(1-weight)*windows[i,start+left]+weight*windows[i,start+right]
            geometry[i,offset+j]=[size/125,(start+size)/125]
            valid[i,offset+j]=True
            start+=int(size)
    return dict(patches=x,geometry=geometry,valid=valid)


def packing_key(arm,year):
    return f"{arm['name']}_{'all' if arm['segmentation']=='fixed' else year}"


def packed_path(arm,year):
    return CACHE/f'{packing_key(arm,year)}.npz'


def build_packed(arm,year):
    path=packed_path(arm,year)
    if path.exists():return path
    windows=np.load(CACHE/'window_representations.npz')[arm['representation']]
    if arm['segmentation']=='fixed':
        lengths=np.full((len(windows),25),5,dtype=np.int16)
    else:
        lengths=np.load(CACHE/f'lengths_{year}.npz')[arm['segmentation']]
    data=pack(windows,lengths)
    np.savez_compressed(path,**data)
    return path

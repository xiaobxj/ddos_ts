"""Archive observed OHLCV; future labels are resolved separately from forecasts."""
from common49 import *
COLS=['date','open','high','low','close','volume']
def clean(frame):
    guard(set(COLS)<=set(frame.columns),'Missing OHLCV columns')
    f=frame[COLS].copy();dates=pd.to_datetime(f.date,format='%Y-%m-%d',errors='raise')
    guard(dates.dt.strftime('%Y-%m-%d').tolist()==f.date.tolist(),'Noncanonical dates')
    guard(f.date.is_unique and f.date.is_monotonic_increasing,'Duplicate or unsorted dates')
    guard((dates.dt.dayofweek<5).all(),'Weekend daily bar')
    for c in COLS[1:]:f[c]=pd.to_numeric(f[c],errors='raise').astype(float)
    guard(np.isfinite(f[COLS[1:]].to_numpy()).all(),'Nonfinite OHLCV')
    guard((f[['open','high','low','close']]>0).all().all() and (f.volume>=0).all(),'Invalid price or volume')
    tol=.011;f['valid_ohlc']=(f.high+tol>=f[['open','close','low']].max(axis=1))&(f.low-tol<=f[['open','close','high']].min(axis=1))
    return f
def prefix(frame,date):
    # Caller must supply an observed prefix. Reject, rather than silently hide, future rows.
    f=clean(frame);guard(len(f)>=125 and f.date.iloc[-1]==date,'Snapshot must end on signal date with >=125 bars')
    guard(f.valid_ohlc.iloc[-125:].all(),'Invalid OHLC within 125-bar input window')
    return f
def unchanged_prefix(original,updated,date):
    observed=clean(updated);observed=observed[observed.date<=date].reset_index(drop=True)
    guard(clean(original).reset_index(drop=True).equals(observed),'Snapshot revised or omitted the recorded prefix')
def snapshot(journal,frame,source,raw=None):
    check_freeze();f=clean(frame);at=utc();local=at.astimezone(TZ)
    base=clean(pd.read_csv(PROJECT/'research/data/1_000300.csv',float_precision='round_trip'))
    unchanged_prefix(base,f,base.date.iloc[-1])
    guard(len(f)>0 and f.date.iloc[-1]<=local.strftime('%Y-%m-%d'),'Future daily bar')
    if f.date.iloc[-1]==local.strftime('%Y-%m-%d'):guard(local.hour>=18,'Same-day snapshot before 18:00 CST')
    csvbytes=f.to_csv(index=False,lineterminator='\n').encode('utf-8');name,h=journal.blob(csvbytes,'.csv');blobs={name:h}
    if raw is not None:
        rawname,rawhash=journal.blob(raw,'.response.json');blobs[rawname]=rawhash
    return journal.append('snapshot',f'{iso(at)}:{h}',dict(source=source,csv_blob=name,start=f.date.iloc[0],end=f.date.iloc[-1],n=len(f),provider_vintage='retrieved snapshot; not an independently notarized point-in-time feed'),blobs)
def fetch(journal):
    import requests
    check_freeze();at=utc();end=at.astimezone(TZ).strftime('%Y-%m-%d')
    # The original verified project used this public Tencent raw index/day branch.
    url=f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000300,day,2025-01-01,{end},2000,qfq'
    response=requests.get(url,timeout=30);response.raise_for_status();payload=response.json();rows=payload['data']['sh000300']['day']
    guard(125<=len(rows)<2000,'Missing or truncated provider response')
    new=clean(pd.DataFrame([r[:6] for r in rows],columns=['date','open','close','high','low','volume']))
    base=clean(pd.read_csv(PROJECT/'research/data/1_000300.csv',float_precision='round_trip'))
    overlap=base.merge(new,on='date',suffixes=('_old','_new'))
    guard(len(overlap)>=125,'Provider does not overlap frozen history sufficiently')
    for c in COLS[1:]:guard(np.array_equal(overlap[c+'_old'].to_numpy(),overlap[c+'_new'].to_numpy()),f'Provider revised frozen overlap: {c}')
    combined=pd.concat([base,new[new.date>base.date.iloc[-1]]],ignore_index=True)
    return snapshot(journal,combined,dict(provider='Tencent',url=url,adjustment='raw index day branch',overlap_rows=len(overlap),overlap_exact=True),response.content)
def load_snapshot(journal,event):
    guard(event['kind']=='snapshot','Not a snapshot event');journal.events()
    return clean(pd.read_csv(journal.root/'blobs'/event['payload']['csv_blob'],float_precision='round_trip'))
def label_for(frame,signal):
    f=clean(frame);matches=np.flatnonzero(f.date.eq(signal));guard(len(matches)==1,'Signal missing from label snapshot')
    t=int(matches[0]);day=pd.Timestamp(signal).dayofweek
    later=np.flatnonzero((f.date>signal)&pd.to_datetime(f.date).dt.dayofweek.eq(day))
    if not len(later):return dict(status='PENDING',reason='next_same_weekday_not_observed')
    e=int(later[0])+1;end=max(t+5,e)
    if end>=len(f):return dict(status='PENDING',reason='exit_or_auxiliary_not_observed')
    w=f.iloc[t+1:t+6];strict=(w.high>=w[['open','close']].max(axis=1))&(w.low<=w[['open','close']].min(axis=1))&(w.high>=w.low)
    valid=t>=124 and f.valid_ohlc.iloc[t-124:end+1].all() and strict.all()
    r=dict(status='MATURE' if valid else 'INVALID',signal_date=signal,entry_date=f.date.iloc[t+1],next_same_weekday=f.date.iloc[e-1],exit_date=f.date.iloc[e],joint_completed=f.date.iloc[end],reason='valid' if valid else 'inherited_OHLC_eligibility_failed')
    if valid:r.update(exec_return=float(f.open.iloc[e]/f.open.iloc[t+1]-1),actual_up=int(f.open.iloc[e]>f.open.iloc[t+1]))
    return r

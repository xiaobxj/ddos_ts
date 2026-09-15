"""Archive public stock OHLCV; never change the index data or journals."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
END = '2026-09-14'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def fetch(task):
    mode, start, end = task
    p = ROOT / 'data' / f'tencent_{mode or "raw"}_{start}_{end}.json'
    url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh603986,day,{start},{end},2000,{mode}'
    if not p.exists():
        r = requests.get(url, timeout=35); r.raise_for_status()
        data = r.json(); assert data['code'] == 0, data
        p.write_bytes(r.content)
    data = json.loads(p.read_bytes())['data']['sh603986']
    key = mode + 'day' if mode else 'day'
    rows = data[key]
    assert 0 < len(rows) < 2000
    print(mode or 'raw', start, end, len(rows), rows[0][:6], rows[-1][:6], flush=True)
    return task, rows, dict(file=p.name, sha256=sha(p), url=url, key=key,
        retrieved_utc=dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(), rows=len(rows))

def main():
    (ROOT / 'data').mkdir(parents=True, exist_ok=True)
    # Some adjusted branches silently cap long ranges at 640 rows.
    # Annual requests, with a January overlap, stay below that cap.
    tasks = [(m,f'{y}-01-01',min(f'{y+1}-01-31',END))
             for m in ['', 'hfq', 'qfq'] for y in range(2016,2027)]
    groups = {m:[] for m in ['', 'hfq', 'qfq']}; sources=[]
    for (mode,_,_), rows, source in ThreadPoolExecutor(3).map(fetch,tasks):
        groups[mode].extend([r[:6] for r in rows]); sources.append(source)
    summaries = []
    for mode, rows in groups.items():
        f=pd.DataFrame(rows,columns=['date','open','close','high','low','volume'])
        f=f[f.date.le(END)].drop_duplicates()
        assert f.date.is_unique, 'Conflicting overlapping bars'
        f=f.sort_values('date').reset_index(drop=True)
        for c in f.columns[1:]: f[c]=pd.to_numeric(f[c],errors='raise')
        f=f[['date','open','high','low','close','volume']]
        valid=(f.high>=f[['open','close','low']].max(axis=1)) & (f.low<=f[['open','close','high']].min(axis=1))
        assert valid.all() and (f.iloc[:,1:5]>0).all().all() and (f.volume>0).all()
        f['valid_ohlc']=valid
        p=ROOT/'data'/f'{mode or "raw"}.csv'; f.to_csv(p,index=False,lineterminator='\n')
        summaries.append(dict(mode=mode or 'raw',file=p.name,sha256=sha(p),rows=len(f),start=f.date.iloc[0],end=f.date.iloc[-1]))
    save(ROOT/'data'/'manifest.json',dict(stock='sh603986',end=END,sources=sources,series=summaries,
        vintage='Downloaded retrospective vendor history; not an archived point-in-time data feed.',volume_unit='Tencent lots (100 shares); no amount fabricated'))

if __name__=='__main__': main()

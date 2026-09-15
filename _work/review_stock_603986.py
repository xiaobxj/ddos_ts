"""Archive public stock data and calculate descriptive short-horizon indicators."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib,json
import numpy as np
import pandas as pd
import requests

ROOT=Path('D:/ddos_v3/stock_reviews/603986/2026-09-15')
TZ=timezone(timedelta(hours=8));at=datetime.now(TZ)
assert at.strftime('%Y-%m-%d')=='2026-09-15'
ROOT.mkdir(parents=True,exist_ok=False)
urls={
    'daily':'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh603986,day,2025-01-01,2026-09-15,700,qfq',
    'quote':'https://qt.gtimg.cn/q=sh603986',
    'interim':'https://static.cninfo.com.cn/finalpage/2026-08-19/1225480384.PDF',
    'buyback':'https://paper.cnstock.com/html/2026-09/02/content_2264110.htm',
}

def save(name,value):
    with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

r=requests.get(urls['daily'],timeout=30);r.raise_for_status()
(ROOT/'tencent_daily_response.json').write_bytes(r.content)
d=r.json()['data']['sh603986'];branch='qfqday' if 'qfqday' in d else 'day'
f=pd.DataFrame([row[:6] for row in d[branch]],columns=['date','open','close','high','low','volume_lots'])
for c in f.columns[1:]:f[c]=f[c].astype(float)
assert f.date.is_unique and f.date.is_monotonic_increasing and len(f)>=60
assert np.isfinite(f.iloc[:,1:].to_numpy()).all()
assert (f.high>=f[['open','close','low']].max(axis=1)-.011).all()
assert (f.low<=f[['open','close','high']].min(axis=1)+.011).all()
daily_end=f.date.iloc[-1]
r=requests.get(urls['quote'],timeout=30);r.raise_for_status();raw=r.content.decode('gbk')
(ROOT/'tencent_quote_response.txt').write_text(raw,encoding='utf-8')
fields=raw.split('"')[1].split('~')
assert fields[2]=='603986'
quote=dict(name=fields[1],ticker=fields[2],price=float(fields[3]),previous_close=float(fields[4]),
           open=float(fields[5]),volume_lots=float(fields[6]),timestamp=fields[30],change=float(fields[31]),
           change_percent=float(fields[32]),high=float(fields[33]),low=float(fields[34]),
           turnover_percent=float(fields[38]),provider_pe_ttm=float(fields[39]),provider_pb=float(fields[46]))
quote_date=datetime.strptime(quote['timestamp'],'%Y%m%d%H%M%S').strftime('%Y-%m-%d')
assert quote_date==at.strftime('%Y-%m-%d') and datetime.strptime(quote['timestamp'],'%Y%m%d%H%M%S').hour>=15
assert daily_end<=quote_date and quote['low']<=quote['price']<=quote['high']
quote_added=daily_end<quote_date
if quote_added:
    assert abs(float(f.close.iloc[-1])-quote['previous_close'])<.011, 'Adjusted prior close disagrees with quote'
    f=pd.concat([f,pd.DataFrame([dict(date=quote_date,open=quote['open'],close=quote['price'],high=quote['high'],low=quote['low'],volume_lots=quote['volume_lots'])])],ignore_index=True)
else:assert abs(float(f.close.iloc[-1])-quote['price'])<.011
close=f.close
metrics={f'sma{n}':float(close.iloc[-n:].mean()) for n in [5,10,20,60]}
metrics.update({f'return_{n}_sessions':float(close.iloc[-1]/close.iloc[-n-1]-1) for n in [5,10,20]})
metrics['volume_vs_prior5']=float(f.volume_lots.iloc[-1]/f.volume_lots.iloc[-6:-1].mean())
metrics['volume_vs_prior20']=float(f.volume_lots.iloc[-1]/f.volume_lots.iloc[-21:-1].mean())
metrics['volume_vs_previous']=float(f.volume_lots.iloc[-1]/f.volume_lots.iloc[-2])
metrics['low_last10']=float(f.low.iloc[-10:].min())
metrics['high_last20']=float(f.high.iloc[-20:].max())
metrics['drawdown_from_last20_high']=float(close.iloc[-1]/metrics['high_last20']-1)
tr=pd.concat([f.high-f.low,(f.high-close.shift()).abs(),(f.low-close.shift()).abs()],axis=1).max(axis=1)
metrics['mean_true_range14']=float(tr.iloc[-14:].mean())
metrics['mean_true_range14_pct']=metrics['mean_true_range14']/float(close.iloc[-1])
metrics['interim_nonrecurring_net_share']=1973752934.75/6856786413.68
save('analysis.json',dict(scope='single_stock_descriptive_review_not_a_trained_forecast',horizon='1-4 weeks',
    retrieved_cst=at.isoformat(),daily_branch=branch,daily_endpoint_last_date=daily_end,
    quote_appended_for_descriptive_indicators=quote_added,quote=quote,metrics=metrics,
    latest_bars=f.tail(25).to_dict(orient='records'),sources=urls,
    existing_CSI300_model_used=False,orders_sent=False,
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file()}))
print(json.dumps(dict(quote=quote,metrics=metrics,daily_last_date=daily_end,quote_added=quote_added,folder=str(ROOT)),ensure_ascii=False,indent=2))

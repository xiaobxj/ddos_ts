"""Manual, genuine after-close forecasts. Research parameters; no order execution."""
from experiment3 import *
from slow3 import apply_record
from scipy.special import expit
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os,hashlib,requests,argparse,html,traceback

LIVE=PROJECT/'stock_daily_603986'
CST=datetime.timezone(datetime.timedelta(hours=8))
CAL_SOURCE='https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml'
HOLIDAYS=[('2026-01-01','2026-01-03'),('2026-02-15','2026-02-23'),('2026-04-04','2026-04-06'),
          ('2026-05-01','2026-05-05'),('2026-06-19','2026-06-21'),('2026-09-25','2026-09-27'),('2026-10-01','2026-10-07')]
MODELS=[('h1.annual.raw',1,'2025-12-31'),('h1.quarterly.raw',1,'2026-06-30'),
        ('h5.annual.raw',5,'2025-12-31'),('h5.quarterly.raw',5,'2026-06-30')]
NAMES={'h1.annual.raw':'快 · 次日／年度更新','h1.quarterly.raw':'快 · 次日／季度更新',
       'h5.annual.raw':'慢 · 5 个交易日／年度更新','h5.quarterly.raw':'慢 · 5 个交易日／季度原始',
       'h5.quarterly.platt':'慢 · 5 个交易日／现有校准','h5.quarterly.shrink':'慢 · 5 个交易日／受约束收缩',
       'h1.frequency':'快 · 训练上涨频率基准','h5.frequency':'慢 · 训练上涨频率基准'}

def canonical(o):return json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(b):return hashlib.sha256(b).hexdigest()
def utc_clock():return datetime.datetime.now(datetime.timezone.utc)
def sessions2026():
    dates=pd.bdate_range('2026-01-01','2026-12-31').strftime('%Y-%m-%d')
    return [d for d in dates if not any(a<=d<=b for a,b in HOLIDAYS)]

@contextmanager
def locked(root):
    root.mkdir(parents=True,exist_ok=True);p=root/'run.lock'
    try:fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('已有运行锁。请等待当前窗口完成；异常退出的锁需核实进程后处理。')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(json.dumps(dict(pid=os.getpid(),utc=now())))
        yield
    finally:p.unlink()

def blob(root,b):
    key=digest(b);p=root/'blobs'/key;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==b,'Blob hash collision or damage'
    else:
        with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    return key

def getblob(root,key):
    b=(root/'blobs'/key).read_bytes();assert digest(b)==key,'Data snapshot damaged';return b

def chain(root):
    events=[];previous='0'*64
    for i,p in enumerate(sorted((root/'events').glob('*.json')),1):
        assert p.name==f'{i:08d}.json','Event sequence gap'
        e=read(p);claimed=e.pop('sha256');assert e['sequence']==i and e['previous']==previous
        assert digest(canonical(e))==claimed,'Event hash mismatch'
        e['sha256']=claimed;events.append(e);previous=claimed
    if events:assert events[0]['kind']=='genesis' and sum(e['kind']=='genesis' for e in events)==1
    last_time=None;forecast_refs={}
    for e in events:
        recorded=datetime.datetime.fromisoformat(e['recorded_utc']);assert recorded.tzinfo is not None
        assert last_time is None or recorded>=last_time,'Clock moved backward in event history'
        last_time=recorded;local=recorded.astimezone(CST);p=e['payload']
        if e['kind']=='forecast':
            assert p['signal_date']==local.date().isoformat() and local.hour>=18,'Backdated/out-of-window forecast'
            assert len({r['name'] for r in p['rows']})==len(p['rows'])
            assert all(r['h'] in [1,5] and 0<=r['probability']<=1 for r in p['rows'])
            forecast_refs[e['sha256']]=p
        elif e['kind']=='label':
            prior=forecast_refs[p['forecast_sha256']]
            assert prior['signal_date']==p['signal_date'] and p['signal_date']<p['target_date']<=local.date().isoformat()
    forecasts=[e['payload']['signal_date'] for e in events if e['kind']=='forecast']
    labels=[(e['payload']['signal_date'],e['payload']['h']) for e in events if e['kind']=='label']
    assert len(forecasts)==len(set(forecasts)) and len(labels)==len(set(labels)),'Duplicate forecast or label'
    return events

def append(root,events,kind,payload,issued):
    e=dict(sequence=len(events)+1,previous=events[-1]['sha256'] if events else '0'*64,
           recorded_utc=issued.isoformat(),kind=kind,payload=payload)
    e['sha256']=digest(canonical(e));p=root/'events'/f'{e["sequence"]:08d}.json';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(canonical(e));f.flush();os.fsync(f.fileno())
    events.append(e);return e

def package_check():
    pkg=read(OUT/'daily_package.json')
    for name,s in pkg['files'].items():assert sha(PROJECT/name)==s,('Verified package changed',name)
    return pkg

def parse_bars(payload,symbol,mode=''):
    d=json.loads(payload);assert d['code']==0
    rows=d['data'][symbol][mode+'day'];assert 0<len(rows)<500
    f=pd.DataFrame([r[:6] for r in rows],columns=['date','open','close','high','low','volume'])
    for c in f.columns[1:]:f[c]=pd.to_numeric(f[c],errors='raise')
    f=f[['date','open','high','low','close','volume']]
    assert f.date.is_unique and f.date.is_monotonic_increasing
    assert f.iloc[:,1:].notna().all().all() and (f.iloc[:,1:]>0).all().all()
    assert (f.high>=f[['open','close','low']].max(axis=1)).all() and (f.low<=f[['open','close','high']].min(axis=1)).all()
    return f

def download(root,local):
    assert local.year==2026,'本参数包仅支持 2026 年日历，需更新已核对的日历包。'
    urls={name:f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,2026-01-01,{local:%Y-%m-%d},500,{mode}'
          for name,sym,mode in [('raw','sh603986',''),('hfq','sh603986','hfq'),('market','sh000300','')]}
    urls['quote']='https://qt.gtimg.cn/q=sh603986'
    def fetch(item):
        name,url=item;response=requests.get(url,timeout=30);response.raise_for_status();b=response.content
        return name,b,dict(url=url,retrieved_utc=now(),blob=blob(root,b))
    results=list(ThreadPoolExecutor(4).map(fetch,urls.items()))
    content={n:b for n,b,s in results};sources={n:s for n,b,s in results}
    raw=parse_bars(content['raw'],'sh603986');hfq=parse_bars(content['hfq'],'sh603986','hfq');market=parse_bars(content['market'],'sh000300')
    quote=content['quote'].decode('gbk').split('="')[1].split('";')[0].split('~')
    assert quote[2]=='603986' and len(quote[30])==14
    complete_end=local.date().isoformat() if local.hour>=18 else (local.date()-datetime.timedelta(days=1)).isoformat()
    return dict(raw=raw[raw.date.le(complete_end)].reset_index(drop=True),hfq=hfq[hfq.date.le(complete_end)].reset_index(drop=True),
                market=market[market.date.le(complete_end)].reset_index(drop=True),quote=quote,sources=sources,complete_end=complete_end)

def merge_unchanged(base,recent,through,cols):
    common=base[base.date.isin(recent.date)].merge(recent,on='date',suffixes=['_old','_new'])
    for c in cols:np.testing.assert_allclose(common[c+'_old'],common[c+'_new'],rtol=0,atol=1e-9,err_msg='历史行情发生修订，停止并保留旧预测')
    old_current=base[base.date.ge('2026-01-01')&base.date.le(through)]
    assert old_current.date.isin(recent.date).all(),'供应商缺失已保存行情，停止追加'
    assert recent[recent.date.le(through)].date.isin(base.date).all(),'历史缺失日被补写，需单独核查'
    return pd.concat([base[['date']+cols],recent[recent.date.gt(through)][['date']+cols]],ignore_index=True)

def frames(root,events,fetched):
    import io
    snaps=[e['payload'] for e in events if e['kind']=='snapshot']
    if snaps:
        previous=snaps[-1]
        raw=pd.read_csv(io.BytesIO(getblob(root,previous['raw_blob'])),float_precision='round_trip')
        cal=pd.read_csv(io.BytesIO(getblob(root,previous['calendar_blob'])),float_precision='round_trip')
    else:
        raw=load_csv(V2/'data/raw.csv');cal=load_csv(ROOT/'data/external_close.csv')[['date','csi300']].rename(columns={'csi300':'close'})
    end=fetched['complete_end'];through=cal.date.iloc[-1]
    assert end>=through,'数据包日期晚于已完成的市场日期，不能历史补录'
    expected=[d for d in sessions2026() if d<=end]
    assert fetched['market'].date.tolist()==expected,'沪深300日线与已核对交易日历不完整，请稍后重试'
    newcal=merge_unchanged(cal,fetched['market'],through,['close'])
    newraw=merge_unchanged(raw,fetched['raw'],through,['open','high','low','close','volume'])
    assert newraw.date.isin(newcal.date).all()
    f=old.build_frame(newraw,newcal.date)
    check=f[f.date.isin(fetched['hfq'].date)&f.valid_ohlc].merge(fetched['hfq'],on='date',suffixes=['_model','_vendor'])
    assert len(check)==len(fetched['raw']) and fetched['raw'].date.tolist()==fetched['hfq'].date.tolist()
    gap=max(float(abs(check[c+'_model']-check[c+'_vendor']).max()) for c in ['open','high','low','close'])
    assert gap<.005,('复权/权益记录不匹配，停止预测及结算并等待核对',gap)
    snapshot=dict(through=f.date.iloc[-1],raw_blob=blob(root,newraw.to_csv(index=False,lineterminator='\n').encode()),
                  calendar_blob=blob(root,newcal.to_csv(index=False,lineterminator='\n').encode()),
                  frame_blob=blob(root,f.to_csv(index=False,lineterminator='\n').encode()),sources=fetched['sources'],hfq_max_gap=gap)
    return f,snapshot

def issue_reason(local,frame,quote,pkg,events):
    today=local.date().isoformat()
    if today>pkg['valid_until']:return '参数包已到期：仅结算旧预测，需完成新季度训练验证后才能新增预测。'
    if not (18<=local.hour<24):return '请在交易日 18:00 后运行，届时才保存当天预测。'
    if today not in sessions2026():return '今天休市，仅检查已有预测是否到期。'
    if any(e['kind']=='forecast' and e['payload']['signal_date']==today for e in events):return '今天的预测已保存；重复运行不会覆盖或重复计数。'
    if frame.date.iloc[-1]!=today or not frame.valid_ohlc.iloc[-1]:return '尚无当天完整个股日线，不生成或回填预测。'
    if quote[30][:8]!=today.replace('-','') or quote[30][8:]<'150000':return '股票收盘快照日期/时间未确认，请稍后重试。'
    for col,k in [('raw_open',5),('raw_close',3),('raw_high',33),('raw_low',34)]:
        if abs(float(frame[col].iloc[-1])-float(quote[k]))>1e-9:return '收盘快照与日线不一致，请稍后重试。'
    if not frame.valid_ohlc.iloc[-125:].all() or len(frame)<125:return '连续输入窗口存在缺失，不生成预测。'
    return ''

def infer(frame):
    r,_,_=modules(True);obs=pd.DataFrame({'anchor':[len(frame)-1]})
    x=old.packed(frame,obs.anchor.to_numpy(int));values={k:r.torch.from_numpy(v).cuda() for k,v in x.items()}
    market=r.market(frame,obs,np.array([0]));gate=r.gates(frame,obs,np.array([0]));rows=[]
    for name,h,c in MODELS:
        folder=fit_folder(c,h);done=read(folder/'completed.json');prob=[]
        for seed in SEEDS:
            m,_=load_model(folder,seed);feat,_=r.neural.extract_features(m,values)
            with np.load(folder/f'cache_{seed}_e20.npz') as z:d={k:z[k] for k in z.files}
            scaled=r.apply_pipeline(feat,market,gate,d)[1]
            head=read(folder/f'heads_{seed}_e20.json');theta=np.asarray(next(v['coefficients'] for v in head['heads'] if v['method']=='vol' and v['scheme']=='U'))
            prob.append(float(expit(np.c_[scaled,np.ones(1)]@theta)[0]));del m
        rows.append(dict(name=name,h=h,probability=float(np.mean(prob)),seed_probabilities=prob,training_cutoff=c))
        if name in ['h1.annual.raw','h5.quarterly.raw']:
            rows.append(dict(name=f'h{h}.frequency',h=h,probability=done['training_frequency'],training_cutoff=c))
    records=read(OUT/'calibration_records.json');raw=next(v for v in rows if v['name']=='h5.quarterly.raw')
    for scheme in ['platt','shrink']:
        record=next(v for v in records if v['cadence']=='quarterly' and v['method']=='vol' and v['scheme']==scheme and v['cutoff']=='2026-06-30')
        rows.append(dict(name='h5.quarterly.'+scheme,h=5,probability=float(apply_record([raw['probability']],record)[0]),
                         training_cutoff=raw['training_cutoff'],calibration_accepted=record['accepted'],calibration_reason=record['reason']))
    r.torch.cuda.empty_cache();return rows

def target_date(date,h):
    dates=sessions2026();i=dates.index(date);return dates[i+h] if i+h<len(dates) else None

def settle(events,frame):
    labels=[];done={(e['payload']['signal_date'],e['payload']['h']) for e in events if e['kind']=='label'}
    dates=frame.date.tolist()
    for e in events:
        if e['kind']!='forecast':continue
        forecast=e['payload'];signal=forecast['signal_date'];a=dates.index(signal)
        for h in sorted({r['h'] for r in forecast['rows']}):
            if (signal,h) in done or a+h>=len(frame):continue
            z=a+h;path=frame.iloc[a:z+1];valid=bool(path.valid_ohlc.all())
            label=dict(signal_date=signal,h=h,target_date=dates[z],forecast_sha256=e['sha256'],status='MATURED' if valid else 'UNSCORABLE_MISSING_BAR')
            if valid:
                start=float(frame.raw_close.iloc[a]);shares=1.;cash=0.
                for date,ratio,dividend in old.ACTIONS:
                    if signal<date<=dates[z]:cash+=shares*dividend;shares*=ratio
                ret=(shares*float(frame.raw_close.iloc[z])+cash)/start-1
                formula=(frame.shares.iloc[z]*frame.raw_close.iloc[z]+frame.cash.iloc[z]-frame.cash.iloc[a])/(frame.shares.iloc[a]*start)-1
                assert abs(ret-formula)<1e-12
                label.update(actual_return=ret,actual_up=int(ret>0))
            labels.append(label)
    return labels

def render(root,events,status,local):
    forecasts=[e for e in events if e['kind']=='forecast'];labels=[e['payload'] for e in events if e['kind']=='label']
    latest=forecasts[-1]['payload'] if forecasts else None;lines=[]
    if latest:
        labelmap={(v['signal_date'],v['h']):v for v in labels}
        for row in latest['rows']:
            result=labelmap.get((latest['signal_date'],row['h']));outcome='待到期'
            if result:outcome=f"{'上涨' if result['actual_up'] else '未上涨'} {result['actual_return']:+.2%}" if result['status']=='MATURED' else '缺失行情，无法评分'
            lines.append(f"<tr><td>{html.escape(NAMES[row['name']])}</td><td>{row['probability']:.2%}</td><td>{latest['targets'][str(row['h'])]}</td><td>{outcome}</td></tr>")
    score=[]
    for name in NAMES:
        pairs=[]
        for e in forecasts:
            p=e['payload'];row=next((r for r in p['rows'] if r['name']==name),None)
            if row is None:continue
            lab=next((l for l in labels if l['signal_date']==p['signal_date'] and l['h']==row['h'] and l['status']=='MATURED'),None)
            if lab:pairs.append((row['probability'],lab['actual_up']))
        if pairs:
            a=np.array(pairs);score.append(f"<tr><td>{NAMES[name]}</td><td>{len(a)}</td><td>{np.mean((a[:,0]>.5)==a[:,1]):.2%}</td><td>{np.mean((a[:,0]-a[:,1])**2):.4f}</td></tr>")
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>603986 每日快慢预测</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:1080px;margin:40px auto;padding:0 20px;color:#183042;background:#f5f7fa}}article{{background:white;padding:24px;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:12px 8px;border-bottom:1px solid #ddd}}.status{{background:#e8f0fa;padding:16px}}small{{color:#526274}}</style>
<h1>603986 兆易创新 · 每日快慢预测</h1><p class="status">{html.escape(status)}</p><p>本次检查：{local:%Y-%m-%d %H:%M:%S}（北京时间）。数据由程序运行时下载。</p>
<article><h2>最近一次真实记录：{latest['signal_date'] if latest else '尚无'}</h2><p>显示的是研究模型输出；历史概率仍未通过可靠性验证，不能直接解读为买入胜率。</p>
<table><tr><th>模型</th><th>上涨概率输出</th><th>目标收盘日</th><th>结果</th></tr>{''.join(lines)}</table>
<p>快：次日收盘相对今天收盘；慢：第 5 个交易日收盘相对今天收盘。现金分红和送转按持股权益计入。此处不含交易成本，也不代表成交收益。</p></article>
<article><h2>真实前瞻记录</h2><p>已保存 {len(forecasts)} 个信号日，已到期 {sum(l['status']=='MATURED' for l in labels)} 个“日期×周期”标签；缺失而无法评分 {sum(l['status']!='MATURED' for l in labels)} 个。没有事后补写历史预测。</p>
<table><tr><th>模型</th><th>已成熟预测数</th><th>方向正确率</th><th>Brier（越低越好）</th></tr>{''.join(score)}</table>
<p>当前统计只作描述。5 日标签相互重叠，不能当成独立样本；少量前瞻记录不能证明有效。</p></article>
<article><h2>怎样使用</h2><p>每个交易日北京时间 18:00 后，双击“打开个股每日预测.cmd”。程序会下载日线、检查历史是否被修订、结算已到期记录，再保存当天的新预测。重复运行不会覆盖今天的记录；漏跑的日期不补录。</p>
<p>目前参数包有效至 2026-09-30。之后仍可结算已有记录；新增预测需要下一季度经过验证的参数包。当天无完整数据或权益校验失败时，程序会明确停止新增记录。</p>
<p><a href="../stock_model_603986_v3/REPORT.md">查看本轮研究报告</a> · <a href="events/">原始事件记录</a></p><small>记录采用本地哈希链和数据快照，能检查内部篡改；没有外部时间戳公证。请保留整个 stock_daily_603986 文件夹。</small></article></html>'''
    (root/'latest.html').write_text(document,encoding='utf-8')

def run(root=LIVE):
    with locked(root):
        pkg=package_check();events=chain(root);start=utc_clock();local=start.astimezone(CST)
        if not events:append(root,events,'genesis',dict(package_sha256=sha(OUT/'daily_package.json'),models=list(NAMES),no_backfill=True,label='close-to-close total shareholder return, H1/H5 exchange sessions'),start)
        fetched=download(root,local);frame,snapshot=frames(root,events,fetched)
        snap=append(root,events,'snapshot',snapshot,utc_clock())
        for label in settle(events,frame):
            label['snapshot_sha256']=snap['sha256'];append(root,events,'label',label,utc_clock())
        reason=issue_reason(local,frame,fetched['quote'],pkg,events)
        if not reason:
            rows=infer(frame);issued=utc_clock();endlocal=issued.astimezone(CST)
            assert endlocal.date()==local.date(),'计算跨过午夜，停止新增记录；不可回填昨天'
            assert not issue_reason(endlocal,frame,fetched['quote'],pkg,events)
            today=endlocal.date().isoformat();payload=dict(signal_date=today,rows=rows,targets={str(h):target_date(today,h) for h in [1,5]},
                raw_close=float(frame.raw_close.iloc[-1]),snapshot_sha256=snap['sha256'],package_sha256=sha(OUT/'daily_package.json'))
            append(root,events,'forecast',payload,issued);reason='已保存今天的快、慢两组真实预测，等待未来收盘到期评分。'
        assert chain(root)==events
        render(root,events,reason,utc_clock().astimezone(CST));print(reason,flush=True)
        print('结果：'+str(root/'latest.html'),flush=True)
        return events

def main():
    ap=argparse.ArgumentParser(description='603986 收盘后真实前瞻记录');ap.add_argument('--no-open',action='store_true');args=ap.parse_args()
    try:run()
    except Exception as ex:
        LIVE.mkdir(parents=True,exist_ok=True);p=LIVE/('error_'+utc_clock().strftime('%Y%m%dT%H%M%S%f')+'.txt')
        p.write_text(traceback.format_exc(),encoding='utf-8');print('本次停止：'+str(ex)+'\n诊断文件：'+str(p),flush=True)
        try:render(LIVE,chain(LIVE),'本次运行未完成：'+str(ex),utc_clock().astimezone(CST))
        except Exception:pass
        if not args.no_open and (LIVE/'latest.html').exists():os.startfile(LIVE/'latest.html')
        raise SystemExit(1)
    if not args.no_open:os.startfile(LIVE/'latest.html')

if __name__=='__main__':main()

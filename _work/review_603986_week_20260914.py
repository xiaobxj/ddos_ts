"""Archive a current-week stock review without changing frozen model evidence."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib
import requests
import pandas as pd
import numpy as np

PROJECT=Path('D:/ddos_v3')
MODEL=PROJECT/'stock_model_603986_v1'
ROOT=PROJECT/'stock_weekly_reviews/603986/2026-09-14_asof_2026-09-15'
TZ=timezone(timedelta(hours=8))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    now=datetime.now(TZ);assert now.strftime('%Y-%m-%d')=='2026-09-15'
    ROOT.mkdir(parents=True,exist_ok=False)
    manifest=read(MODEL/'results/delivery_manifest.json')
    required=['results/ensemble_predictions.csv','results/seed_predictions.csv','results/weekly_context.csv','results/summary.json','protocol.json','results/verification.json']
    for name in required:
        key=next(k for k in manifest['files'] if k.replace('\\','/')==name)
        assert sha(MODEL/name)==manifest['files'][key]
    assert read(MODEL/'results/verification.json')['status']=='PASS'
    urls=dict(daily='https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh603986,day,2026-01-01,2026-09-15,500,',
        quote='https://qt.gtimg.cn/q=sh603986',calendar='https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml')
    response=requests.get(urls['daily'],timeout=30);response.raise_for_status()
    (ROOT/'tencent_daily.json').write_bytes(response.content)
    rows=response.json()['data']['sh603986']['day']
    f=pd.DataFrame([r[:6] for r in rows],columns=['date','open','close','high','low','volume_lots'])
    for c in f.columns[1:]:f[c]=f[c].astype(float)
    assert f.date.iloc[-1]=='2026-09-15' and f.date.is_unique and f.date.is_monotonic_increasing
    assert np.isfinite(f.iloc[:,1:].to_numpy()).all()
    assert f.high.ge(f[['open','close','low']].max(axis=1)).all() and f.low.le(f[['open','close','high']].min(axis=1)).all()
    response=requests.get(urls['quote'],timeout=30);response.raise_for_status()
    (ROOT/'tencent_quote_gbk.bin').write_bytes(response.content)
    text=response.content.decode('gbk');(ROOT/'tencent_quote.txt').write_text(text,encoding='utf-8')
    fields=text.split('"')[1].split('~');assert fields[2]=='603986'
    q=dict(timestamp=fields[30],price=float(fields[3]),previous_close=float(fields[4]),open=float(fields[5]),
        high=float(fields[33]),low=float(fields[34]),volume_lots=float(fields[6]),daily_change_percent=float(fields[32]))
    assert q['timestamp'].startswith('20260915') and int(q['timestamp'][8:10])>=15
    for k in ['open','high','low','volume_lots']:assert q[k]==float(f.iloc[-1][k])
    assert q['price']==float(f.close.iloc[-1]) and q['previous_close']==float(f.close.iloc[-2])
    f.to_csv(ROOT/'daily.csv',index=False,lineterminator='\n')
    predictions=pd.read_csv(MODEL/'results/ensemble_predictions.csv',float_precision='round_trip')
    p=predictions[predictions.date.eq('2026-09-11')].copy();assert len(p)==25 and p.status.eq('pending_retrospective').all()
    p.to_csv(ROOT/'week_model_outputs.csv',index=False,lineterminator='\n')
    close=float(f.close.iloc[-1]);prior=float(f.loc[f.date.eq('2026-09-11'),'close'].iloc[0]);entry=float(f.loc[f.date.eq('2026-09-14'),'open'].iloc[0])
    week=f[f.date.ge('2026-09-14')]
    metrics=dict(last_price=close,previous_friday_close=prior,model_entry_open=entry,
        week_change_vs_previous_friday=close/prior-1,change_vs_model_entry_open=close/entry-1,
        current_week_low=float(week.low.min()),current_week_high=float(week.high.max()),
        sma5=float(f.close.tail(5).mean()),sma10=float(f.close.tail(10).mean()),sma20=float(f.close.tail(20).mean()),
        volume_vs_previous=float(f.volume_lots.iloc[-1]/f.volume_lots.iloc[-2]),
        volume_vs_prior5=float(f.volume_lots.iloc[-1]/f.volume_lots.iloc[-6:-1].mean()))
    def prob(history,method='learned_vol_interaction'):
        return float(p.loc[p.history.eq(history)&p.method.eq(method),'probability'].iloc[0])
    probabilities=dict(main=prob('rolling5_annual20'),state=prob('weekly_state_validated'),recent_weighted=prob('annual_head_timeweight2y'),
        market4=prob('control','market4'),training_frequency=prob('control','training_frequency'))
    summary=read(MODEL/'results/summary.json')
    save(ROOT/'review.json',dict(stock='603986',week=['2026-09-14','2026-09-18'],asof_cst=now.isoformat(),quote=q,metrics=metrics,
        model_input_date='2026-09-11',model_constructed_date='2026-09-15',model_entry_date='2026-09-14',scheduled_model_exit_date='2026-09-21',
        target='Gross opening-price return including entitled distributions; no execution or cost claim; scheduled exit assumes stock tradable.',
        probabilities=probabilities,model_status='Historical reconstruction, not a forecast actually issued on Sep11. Label pending, no win/loss recorded.',
        historical_main_accuracy=summary['main_accuracy'],historical_common_weeks=summary['mature_weeks'],
        selected_levels=dict(lower_watch=[355.55,356.00],first_repair=[371.,377.],upper_repair=[380.,388.]),
        interpretation='Model leans toward rebound, while prices remain below 5/10/20-day averages; support/resistance levels are conditional observations, not guaranteed barriers.',
        model_source_hashes={n:sha(MODEL/n) for n in required},sources=urls,model_retrained=False,journals_modified=False))
    lines=['# 603986 本周观察：2026年9月14—18日','',
        '截至9月15日收盘，我的判断是：模型倾向反弹，但价格仍处于偏弱修复阶段，尚未确认转强。','',
        f'最新收盘{close:.2f}元，当日{q["daily_change_percent"]:+.2f}%；相对上周五{prior:.2f}元，本周累计{metrics["week_change_vs_previous_friday"]:.2%}。数据来自[腾讯行情快照]({urls["quote"]})，日K已更新至9月15日，并与快照核对OHLC和成交量。','',
        '| 固定模型 | 输出的上涨概率 |','|---|---:|',
        f'| 主模型：年度滚动＋波动交互 | {probabilities["main"]:.1%} |',
        f'| 近期样本加权＋波动交互 | {probabilities["recent_weighted"]:.1%} |',
        f'| 简单行情特征模型 | {probabilities["market4"]:.1%} |',
        f'| 训练期上涨频率参照 | {probabilities["training_frequency"]:.1%} |','',
        '这些是9月15日才完成的模型对9月11日输入的历史重算，不是上周五实际发布的预测。新取得的9月14—15日行情仅用于观察本周进展，没有改写这些概率。','',
        f'模型对应9月14日开盘{entry:.2f}元至计划9月21日开盘的收益方向；它没有直接预测9月18日收盘。当前相对模型起点为{metrics["change_vs_model_entry_open"]:+.2%}，但结果尚未成熟，不能计为预测正确。主模型历史175周准确率只有52.0%，未超过简单基线；输出的63.3%不能当作已验证的真实胜率。','',
        '## 后半周观察区间','',
        '| 价格区间 | 依据与条件性判断 |','|---|---|',
        '| 355—356元 | 周一最低355.55、周二最低356.00；关注是否仍有承接。若收盘跌破355.55且难以收回，反弹判断转弱。 |',
        f'| 371—377元 | 今日高点371.66，5日均线{metrics["sma5"]:.2f}、10日均线{metrics["sma10"]:.2f}。收回该区并有成交量配合，才是更有说服力的修复迹象。 |',
        f'| 380—388元 | 前期交易区与20日均线{metrics["sma20"]:.2f}附近，是进一步修复时的观察区域；不是本周必达目标。 |','',
        f'今天成交量较昨天增加{metrics["volume_vs_previous"]-1:.1%}，但仅为此前5日均量的{metrics["volume_vs_prior5"]:.2f}倍；暂不把单日反弹当成明确的放量反转。以上价位是根据近期OHLC和均线作出的分析推断，不是统计验证过的支撑保证。','',
        '周内观察与模型的开盘到开盘标签使用不同起点，因此可以同时出现“本周相对上周五仍下跌”和“相对周一开盘暂时上涨”。原始数据、全部25个固定输出及依赖哈希一并保存；未修改个股训练集、模型或指数前瞻记录。','']
    (ROOT/'本周观察.md').write_text('\n'.join(lines),encoding='utf-8')
    save(ROOT/'manifest.json',dict(created_cst=now.isoformat(),files={p.name:sha(p) for p in ROOT.iterdir() if p.is_file()},
        script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(metrics=metrics,probabilities=probabilities,quote=q,folder=str(ROOT)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

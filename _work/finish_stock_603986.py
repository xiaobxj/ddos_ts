from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

root=Path('D:/ddos_v3/stock_reviews/603986/2026-09-15')
data=json.loads((root/'analysis.json').read_text(encoding='utf-8'));q=data['quote'];m=data['metrics']
payload=json.loads((root/'tencent_daily_response.json').read_text(encoding='utf-8'))['data']['sh603986']
rows=[r[:6] for r in payload[data['daily_branch']]]
if data['quote_appended_for_descriptive_indicators']:
    rows.append(['2026-09-15',q['open'],q['price'],q['high'],q['low'],q['volume_lots']])
values=np.array([r[1:] for r in rows],float);dates=[r[0] for r in rows];close=values[:,1]
assert abs(close[-1]-q['price'])<1e-10
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
fig,(ax,volume)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[4,1]},layout='constrained')
fig.patch.set_facecolor('#f8faf8');ax.set_facecolor('white');volume.set_facecolor('white')
n=25;tail=values[-n:];x=np.arange(n)
for i,(o,c,h,l,v) in enumerate(tail):
    color='#ae5b49' if c>=o else '#43846b'
    ax.vlines(i,l,h,color=color,lw=1.1)
    ax.add_patch(Rectangle((i-.26,min(o,c)),.52,max(abs(c-o),.12),color=color,alpha=.85))
    volume.bar(i,v/10000,color=color,width=.6,alpha=.7)
for window,color in [(5,'#b99042'),(10,'#6882b5'),(20,'#6d557e')]:
    average=np.convolve(close,np.ones(window)/window,mode='valid')
    ax.plot(x,average[-n:],lw=1.45,label=f'{window}日均价 {average[-1]:.2f}',color=color)
ax.axhspan(355,356,color='#8dafaa',alpha=.18,label='近端低点观察区 355—356')
ax.axhspan(371,377,color='#dec798',alpha=.16,label='首道修复观察区 371—377')
ax.scatter([n-1],[q['price']],s=42,color='#182e35',zorder=5)
ax.annotate(f"9/15报价 {q['price']:.2f}",xy=(n-1,q['price']),xytext=(n-7,343),
            arrowprops={'arrowstyle':'-','color':'#536862'},fontsize=10,color='#182e35')
ax.set_title('兆易创新 603986｜未来1—4周观察：反弹尚未站回短期均线',loc='left',fontsize=15,pad=17)
ax.set_ylabel('价格（元）');volume.set_ylabel('成交量\n（万手）')
volume.set_xticks(x[::3],[d[5:] for d in dates[-n:][::3]],rotation=0)
ax.legend(loc='upper right',fontsize=8,frameon=False,ncol=2)
for axis in [ax,volume]:
    axis.grid(axis='y',alpha=.15)
    axis.spines[['top','right']].set_visible(False)
    axis.spines[['left','bottom']].set_color('#d9e2dc')
    axis.tick_params(colors='#536862')
ax.set_ylim(bottom=min(340,float(tail[:,3].min())-7))
ax.set_xlim(-1,n)
fig.supxlabel('腾讯前复权日线截至9/14；9/15使用15:40报价补充，供描述性分析。观察区间不是已验证的买卖信号。',fontsize=9,color='#65756d')
fig.savefig(root/'603986_短期观察.png',dpi=150)
plt.close(fig)

report=f'''按未来1—4周看，兆易创新（603986）目前更适合放在反弹观察名单：已有单日反弹，但价格仍低于短期及中期均线，尚未确认趋势转强。这个判断来自该股行情与公告分析，没有套用原沪深300模型，也没有给出未经验证的个股上涨胜率。

截至2026年9月15日15:40:21，腾讯报价为{q['price']:.2f}元，当日变化{q['change_percent']:+.2f}%，日内高点{q['high']:.2f}、低点{q['low']:.2f}。下载的前复权日线截至9月14日；下面的均线与成交量描述另加入了9月15日报价中的日内OHLCV。该追加行仅用于本次个股观察，没有写入原研究账本。[腾讯行情接口](https://qt.gtimg.cn/q=sh603986)

| 描述指标 | 数值 |
|---|---:|
| 5日均价 | {m['sma5']:.2f}元 |
| 10日均价 | {m['sma10']:.2f}元 |
| 20日均价 | {m['sma20']:.2f}元 |
| 最近5个交易日价格变化 | {m['return_5_sessions']:+.2%} |
| 最近20个交易日价格变化 | {m['return_20_sessions']:+.2%} |
| 本次报价成交量／此前5日平均成交量 | {m['volume_vs_prior5']:.2f}倍 |
| 本次报价成交量／此前20日平均成交量 | {m['volume_vs_prior20']:.2f}倍 |
| 近14日平均真实波幅 | {m['mean_true_range14']:.2f}元，约现价的{m['mean_true_range14_pct']:.2%} |

这些指标由存档行情计算，属于描述统计。当前呈现价格低于5日均价、5日低于10日、10日低于20日的排列。今天成交量略高于此前5日均量，但低于此前20日均量，因此单凭今天上涨还不足以确认持续转强。平均真实波幅采用14日简单平均，不是Wilder平滑ATR；它描述过去日内及跳空波动，不能作为未来波幅保证。

| 未来价格行为 | 我的观察与处理倾向 |
|---|---|
| 355—371元区间反复，无法收复短期均线 | 继续观察，不把单日上涨视作趋势反转 |
| 重新站上371—377元，随后1—2个交易日保持，成交量超过此前5日均量 | 反弹修复得到更多支持，再检查380—388元附近的上方阻力 |
| 收盘跌破355—356元且随后不能迅速收回 | 近端低点未能守住，原反弹观察假设削弱，重新评估而非机械补仓 |

这些价位由9月14日低点355.55元、9月15日低点356.00元、上周价格区域及当前均线推导，是待观察的条件，不是经过个股回测确认的入场／止损系统，也不是保证有效的支撑或目标价。若未持有，我目前倾向等待价格修复确认后再考虑新开仓；若已有持仓，需要结合实际成本、仓位和可接受损失制定退出规则，本报告不假定这些信息。

基本面提供背景：2026年上半年公司营收115.66亿元、同比增长178.67%，扣非归母净利润48.83亿元、同比增长796.90%。同时，非经常性净收益约19.74亿元，占归母净利润约28.8%，主要涉及金融资产公允价值及处置损益；因此不能把归母净利润增幅全部视为主营业务可持续增速。半年报未经审计。[公司2026年半年报](https://static.cninfo.com.cn/finalpage/2026-08-19/1225480384.PDF)

回购也是应跟踪的事项。公司披露10亿—20亿元回购计划，截至8月31日累计已使用约6.54亿元，回购股份将用于注销。回购计划与已发生回购有区别，不能据此认定短期价格不会下跌；本次采用的是9月2日已披露进展，并未假定9月新增回购金额。[公司回购进展公告](https://paper.cnstock.com/html/2026-09/02/content_2264110.htm)

腾讯报价附带的市盈率TTM约32.24倍，为数据商口径，未独立重建完整TTM利润和股本调整。考虑到报表中的非经常性收益以及你1—4周的期限，我不据此认定现价便宜，也不把静态估值替代量价确认。

你的1—4周观察期跨越两段休市。上交所公布9月25日至27日中秋休市、10月1日至7日国庆休市，分别于9月28日、10月8日恢复交易。因此不能把4个自然周简单当作20个可交易日；若参与，应提前考虑是否持仓跨假期，以及复市跳空导致计划退出价无法成交的可能。[上交所2026年休市安排](https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml)

[价格与成交量图](603986_短期观察.png) · [计算结果与来源](analysis.json)
'''
(root/'短期分析.md').write_text(report,encoding='utf-8')
assert '\ufffd' not in report and '????' not in report
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.name!='manifest.json'}
(root/'manifest.json').write_text(json.dumps(dict(scope='descriptive_stock_review',files=manifest,orders_sent=False),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='DONE',report=str(root/'短期分析.md'),chart=str(root/'603986_短期观察.png')),ensure_ascii=False))

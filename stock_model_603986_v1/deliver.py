"""Readable report from verified outputs; never select or activate a winner."""
from stock import *

NAMES={'learned_market':'神经特征＋自身行情状态','learned_vol_interaction':'波动交互','learned_order_extension':'涨跌顺序交互','learned_order_offset':'涨跌顺序偏移修正'}
HNAMES={U:'固定五年／年度重训',Q:'加季度状态校准',W:'近期样本加权',I:'仅更新加权截距',S:'仅更新加权斜率'}
CONTROL={'training_frequency':'训练期上涨频率','always_up':'始终猜涨','market4':'四个简单行情特征','native_mse':'网络原始收益方向','training_mean_return':'训练期平均收益方向'}

def main():
    check_freeze();verification=read(OUT/'verification.json');assert verification['status']=='PASS'
    metrics=load_csv(OUT/'metrics.csv');pooled=metrics[metrics.period.eq('pooled')]
    cp=read(OUT/'comparisons.json');done=read(OUT/'scoring_completed.json');datareport=read(OUT/'data_validation.json')
    predictions=load_csv(OUT/'ensemble_predictions.csv');context=load_csv(OUT/'weekly_context.csv')
    mainrow=pooled[pooled.history.eq(U)&pooled.method.eq('learned_vol_interaction')].iloc[0]
    freq=pooled[pooled.method.eq('training_frequency')].iloc[0];market=pooled[pooled.method.eq('market4')].iloc[0]
    lines=[
        '# 603986：首轮个股周度建模结果', '',
        f'已完成真正的个股训练：4个年度截点 × 3个随机种子，共12个网络，每个训练20轮。检验{done["mature_weeks"]}个成熟周度样本，期间为2023年至2026年9月4日；行情冻结至2026年9月14日。', '',
        '此前研究与每周预测程序以沪深300为目标；本次从零训练603986的参数，沿用网络结构、固定五年滚动训练、状态校准及交互方法。未使用沪深300的训练样本或已训练权重。', '',
        f'预先指定的主模型（年度重训＋波动交互）方向准确率为 **{mainrow.accuracy:.1%}**，Brier为 **{mainrow.brier:.4f}**。训练期上涨频率基线为{freq.accuracy:.1%}／{freq.brier:.4f}，简单行情模型为{market.accuracy:.1%}／{market.brier:.4f}。Brier越低越好。', '',
        '![逐年检验与全部固定组合](603986_滚动检验.png)', '',
        '## 怎样测试', '',
        '- 每年只使用截至上一年末、标签已经完整兑现的最近五年有效样本；每天的输入窗口为125个交易日。实际年度训练样本为809、1023、1206、1206条。',
        '- 每年从零初始化3个网络；不根据测试成绩更改训练轮数、种子、正则或窗口。20种固定组合全部报告，不自动选最优模型。',
        '- 目标是信号日之后下一交易日开盘买入，持有至下一次相同星期几之后的交易日开盘，计算包含转增和现金分红的税前收益方向。节假日可使持有期超过一周；尚未训练独立四周目标。',
        '- 预测正确率按周计数，3个种子先平均，不能算成3倍样本。这里是按历史时间切分的样本外检验，但使用了当前下载的历史数据，且股票由用户在看过市场后选定，不声称完全未见的盲测。', '',
        '## 主模型与基线', '',
        '| 方法 | 正确周数／总周数 | 方向准确率 | Brier |', '|---|---:|---:|---:|']
    chosen=[mainrow]+[pooled[pooled.method.eq(m)].iloc[0] for m in CONTROL]
    for row in chosen:
        name='主模型：波动交互' if row.method=='learned_vol_interaction' else CONTROL[row.method]
        brier=f'{row.brier:.4f}' if pd.notna(row.get('brier')) else '不适用'
        lines.append(f'| {name} | {int(row.correct)}/{int(row.weeks)} | {row.accuracy:.1%} | {brier} |')
    lines+=['','## 分年度表现','','| 年度 | 周数 | 主模型准确率 | 上涨频率基线 | 简单行情模型 | 主模型Brier |','|---|---:|---:|---:|---:|---:|']
    for year in range(2023,2027):
        g=metrics[metrics.period.eq(str(year))];a=g[g.history.eq(U)&g.method.eq('learned_vol_interaction')].iloc[0]
        b=g[g.method.eq('training_frequency')].iloc[0];c=g[g.method.eq('market4')].iloc[0]
        lines.append(f'| {year}{"（截至9月4日信号）" if year==2026 else ""} | {int(a.weeks)} | {a.accuracy:.1%} | {b.accuracy:.1%} | {c.accuracy:.1%} | {a.brier:.4f} |')
    lines+=['','## 全部固定组合','','此表是描述性比较；表中最高数值不等于获得独立验证的可用策略。状态取自个股自身的趋势和波动，本版尚未加入沪深300、行业指数或公告特征。“涨跌顺序”指价格符号序列，不是订单簿。','','| 滚动／校准方式 | 特征方法 | 准确率 | Brier |','|---|---|---:|---:|']
    for history in HISTORIES:
        for method in METHODS:
            a=pooled[pooled.history.eq(history)&pooled.method.eq(method)].iloc[0]
            lines.append(f'| {HNAMES[history]} | {NAMES[method]} | {a.accuracy:.1%} | {a.brier:.4f} |')
    quarters=[read(p) for p in sorted(OUT.glob('quarter_2*.json'))]
    accepted=sum(sum(x['accepted'] for x in q['decisions']) for q in quarters)
    ready=[q['schedule']['cutoff'] for q in quarters if q['schedule']['mode']=='ready']
    lines += ['',f'季度状态校准只用该股此前的样本外周度信号：52周拟合，按标签成熟日期隔离后13周验证；状态样本不足时回退为零修正。共有{len(quarters)}个季度截点，{accepted}个“季度×方法×状态”单元通过门槛；首个具备完整训练与验证周数的截点为{ready[0] if ready else "无"}。年初固定重置，不在样本不足时强行拟合。', '',
        '## 差异是否稳定', '',
        '预先指定主模型的Brier减去基线Brier；负值才表示主模型较好。按8周连续块进行10000次重采样，以下区间和检验仍受历史重用、样本量及制度变化限制。', '',
        '| 参考模型 | Brier差值 | 95%块重采样区间 | 两项比较Holm校正p |', '|---|---:|---|---:|']
    for c in cp['comparisons']:
        lines.append(f'| {CONTROL[c["reference"]]} | {c["brier_difference"]:+.4f} | [{c["ci95_low"]:+.4f}, {c["ci95_high"]:+.4f}] | {c["p_holm_two"]:.4f} |')
    robust=all(c['brier_difference']<0 and c['ci95_high']<0 and c['p_holm_two']<.05 for c in cp['comparisons'])
    conclusion='主模型在这批历史数据的Brier指标上优于两种基线，但仍需新的前瞻样本确认。' if robust else '本轮没有建立主模型稳定优于简单基线的证据，不能据此将它作为已经验证的选股或买卖工具。'
    lines+=['', '**结论：'+conclusion+'**', '', '## 最近一周的输出性质', '']
    latest=predictions[predictions.date.eq(done['latest_signal'])];cx=context[context.date.eq(done['latest_signal'])].iloc[0]
    lines.append(f'{done["latest_signal"]}收盘后的输入被用于今天才进行的历史重算，状态为`{cx.state}`。以下概率不是当日实际发布的预测，也不是今天盘中的即时预测；标签尚未成熟，不纳入准确率。')
    lines+=['','| 固定方案（波动交互） | 三种子平均上涨概率 |','|---|---:|']
    for history in [U,Q,W]:
        a=latest[latest.history.eq(history)&latest.method.eq('learned_vol_interaction')].iloc[0]
        lines.append(f'| {HNAMES[history]} | {a.probability:.1%} |')
    lines+=['','## 个股特有的数据处理与检查','',
        f'- 下载到{datareport["raw_rows"]}条真实日线；按交易所交易日对齐后保留{datareport["missing_stock_sessions"]}个缺失个股交易日，凡输入或标签跨缺口都排除。缺失不被伪造为成交。',
        '- 按除权除息日累计股份倍数和现金，逐日构建仅依赖当时已发生事件的价格序列；成交量换成初始股份单位，降低转增造成的机械跳变。当前前复权行情仅作下载核对，不进入训练。',
        f'- 九次权益事件与腾讯后复权价格的最大差为{datareport["vendor_hfq_max_difference"]:.6f}，在训练前冻结的累计舍入容差内。训练前曾调整过舍入容差，经过及原始失败记录保留在准备阶段文件中。标签另按实际持股数和现金逐项重算，最大差为2.22e-16。权益记录来源：[经济通分红记录](https://www.etnet.com.hk/www/sc/ashares/quote_dividend.php?code=603986)，并用[2017年度公司权益分派实施公告](https://pdf.dfcfw.com/pdf/H2_AN201805141143166795_1.pdf)核对转增和除权日期。',
        '- 训练前做了未来数据扰动、截点标签、停牌窗口及网络更新检查；训练后重载全部12个模型，独立检查100个拟合头，并复核季度门槛。所有检查通过。',
        '- 收益未扣费税，没有模拟涨跌停、成交量约束、滑点或实际成交；不据此给出可交易收益率。原始行情仍是当前下载的历史版本，不能证明供应商没有修订。', '',
        '## 后续研究方向', '',
        '先加入同时期可获得的行业与大盘信息，区分“跟随半导体／市场上涨”和个股自身强弱，并明确测试绝对收益还是相对行业收益；再为四周持有期单独构建标签，重新做按成熟日期隔离的滚动检验。两个变化应分开比较，先锁定检验方案，再查看新增结果。', '',
        '## 文件', '',
        '- `protocol.json`：训练前固定的个股实验定义。',
        '- `results/annual/`：12个真实网络检查点、各年度成员表、缩放参数与概率头。',
        '- `results/ensemble_predictions.csv`：全部逐周集成输出；`seed_predictions.csv`保留种子结果。',
        '- `results/metrics.csv`、`comparisons.json`：完整指标和固定比较。',
        '- `results/verification.json`：最终验证回执。',
        '- 本目录是独立研究版；原有“打开每周预测.cmd”仍是沪深300程序，尚未接入个股自动更新。', '']
    path=ROOT/'建模结果.md'
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines))
    save(OUT/'summary.json',dict(stock='603986',neural_fits=12,mature_weeks=done['mature_weeks'],main_accuracy=float(mainrow.accuracy),
        main_correct=int(mainrow.correct),main_brier=float(mainrow.brier),frequency_accuracy=float(freq.accuracy),frequency_brier=float(freq.brier),
        market4_accuracy=float(market.accuracy),market4_brier=float(market.brier),conclusion=conclusion,
        latest_signal=done['latest_signal'],latest_status='retrospective_pending',built_utc=now()))
    print(read(OUT/'summary.json'),flush=True)

if __name__=='__main__':main()

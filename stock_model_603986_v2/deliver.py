"""Readable report from verified, unchanged evaluation artifacts."""
from core import *

NAMES={
 'mse_5y_e20.vol.U':'主模型：5年／20遍／波动交互',
 'mse_5y_e10.vol.U':'同轨迹第10遍',
 'mse_3y_e20.vol.U':'整条模型改为3年窗口',
 'bce_5y_e20.native_bce.U':'直接BCE原生概率',
 'bce_5y_e20.vol.U':'BCE表示＋波动交互头',
 'mse_5y_e20.scalar_probe.U':'MSE标量分数的概率头',
 'bce_5y_e20.scalar_probe.U':'BCE标量分数的概率头',
 'mse_5y_e20.add.U':'仅加性市场条件',
 'mse_5y_e20.order_joint.U':'联合增加价格顺序交互',
 'mse_5y_e20.order_fixed.U':'固定波动父模型＋顺序项',
 'mse_5y_e20.vol.W':'近期时间加权全部系数',
 'mse_5y_e20.vol.I':'仅替换加权截距',
 'mse_5y_e20.vol.S':'仅替换加权斜率',
 'mse_q2026_e20.vol.U':'2026年季度整网更新',
 'cal.quarter_four_state':'季度四状态验证修正',
 'cal.month_four_state':'月度四状态验证修正',
 'cal.quarter_pooled':'季度合并组验证修正',
 'cal.quarter_ungated':'季度四状态无验证门控',
 'cal.quarter_platt':'季度OOS Platt校准',
 'frequency_5y':'同训练成员上涨频率',
 'market4_5y':'仅四项行情特征',
 'constant_0.5':'固定50%概率',
 'always_up':'始终判断上涨',
 'always_down':'始终判断不涨'}

def table(rows,columns):
    return ['| '+' | '.join(x[0] for x in columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+[
        '| '+' | '.join(str(fn(row)) for _,fn in columns)+' |' for row in rows]

def main():
    assert read(OUT/'models_verified.json')['status']=='PASS'
    assert read(OUT/'outputs_verified.json')['status']=='PASS'
    assert read(OUT/'scoring_completed.json')['status']=='PASS'
    metrics=load_csv(OUT/'metrics.csv');years=load_csv(OUT/'year_metrics.csv');boots=load_csv(OUT/'bootstrap.csv')
    counts=read(OUT/'sample_counts.json');latest=read(OUT/'latest_probabilities.json')
    raw=load_csv(ROOT/'data/raw.csv');last=float(raw.close.iloc[-1]);mainmethod=cfg()['primary']
    headline=[]
    for h in [1,5]:
        m=metrics[(metrics.h==h)&metrics.method.eq(mainmethod)].iloc[0].to_dict()
        p=next(r for r in latest['rows'] if r['h']==h and r['method']==mainmethod)
        baseline=metrics[(metrics.h==h)&metrics.method.eq('frequency_5y')].iloc[0]
        main_boot=boots[(boots.h==h)&boots.control.eq('frequency_5y')&boots.block.eq(20)].iloc[0]
        headline.append(dict(h=h,n=int(m['n']),accuracy=m['accuracy'],brier=m['brier'],logloss=m['logloss'],
            next_probability=p['probability'],target_end='2026-09-16' if h==1 else '2026-09-22',
            baseline_accuracy=float(baseline.accuracy),baseline_brier=float(baseline.brier),
            delta_brier=float(main_boot.delta_brier),holm_p=float(main_boot.p_holm)))
    save(OUT/'delivery_summary.json',dict(status='VERIFIED_RESEARCH',completed_utc=now(),stock='603986',
        source_close=last,asof='2026-09-15',headlines=headline,
        statistical_note='Exploratory reused history; Holm corrects four declared primary contrasts, not all past research selection'))
    lines=['603986 兆易创新的快慢模型已完成本轮固定规则测试：84 次从头训练、168 个第 10／20 遍检查点。输入都是截至当日收盘的 125 根日线；快模型看下一交易日，慢模型看未来 5 个交易日，每个交易日分别生成概率。',
        '', '主要结论：快模型尚未显示稳定方向优势；慢模型有一些方向识别线索，但原始概率明显失真。2026 年季度重训的改善值得保留，尚不足以把整套概率视为可靠投资依据。',
        '', '这里的主模型在看成绩之前就固定为“5 年窗口、自然 20 遍、年度重训、波动交互、三种子平均概率”。下面没有按结果改选冠军。',
        '', '**主结果和最新模型概率**', '']
    lines+=table(headline,[('周期',lambda r:'快：次日' if r['h']==1 else '慢：5个交易日'),('成熟样本',lambda r:r['n']),
        ('方向准确率',lambda r:f"{r['accuracy']:.2%}"),('Brier',lambda r:f"{r['brier']:.6f}"),
        ('对数损失',lambda r:f"{r['logloss']:.6f}"),('9月15日收盘后的上涨概率',lambda r:f"{r['next_probability']:.2%}")])
    lines += ['',f'数据截至 2026-09-15 收盘，原始收盘价为 {last:.2f} 元。快模型目标为 9 月 15 日收盘至 9 月 16 日收盘，慢模型目标为 9 月 15 日收盘至 9 月 22 日收盘。这两个最新目标均未成熟，尚未计分。日历依据[上交所 2026 年休市安排](https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml)，价格与[腾讯行情](https://qt.gtimg.cn/q=sh603986)核对。',
        '', '“上涨概率”是模型输出，不是这套模型的历史准确率，也不是收益保证。Brier 和对数损失越低越好；恒定 50% 概率的 Brier 为 0.25。涨跌标签使用含分红送转权益的收盘至收盘总收益 > 0，不假设收盘后仍能按已知收盘价成交。旧版的开盘至开盘周目标保持为另一项研究，不能把两版准确率之差归因于改成日频。',
        '', '主模型的快／慢 Brier 为 0.255063／0.265044，均高于各自频率基准的 0.250047／0.250570。快模型 10 遍为 50.84%，20 遍降至 49.83%；改为 3 年窗口为 48.94%。慢模型 20 遍的方向好于 10 遍，说明训练轮数的作用随目标而变，不能把一个周期的经验直接套到另一个周期。',
        '', '近期季度更新确有值得保留的线索：2026 年快模型由 94/170＝55.29% 提高到 105/170＝61.76%，慢模型由 101/166＝60.84% 提高到 111/166＝66.87%。快模型该年 Brier 却从 0.248621 升至 0.249919；慢模型从 0.258467 降至 0.251742，仍略高于频率基准 0.250076。这是固定的两个季度重训对照，未据此自动替换年度主模型。',
        '', '慢模型季度 Platt 校准的全期 Brier 降至 0.254481、方向准确率为 56.33%，比原主模型改善；但仍未超过简单频率的概率误差。当前季度该校准未启用，所以今天的慢模型概率依然是原始输出。快模型标量概率头只有很小的 Brier 领先，属于 152 个固定方案中的探索性线索，不能据此认定稳定优势。',
        '', '需要特别留意高概率失真：慢模型历史上 70%—80% 概率箱有 63 个日信号，平均预测 74.05%，实际上涨仅 22/63＝34.92%。这些标签相互重叠，并非 63 个独立周。它不等于“今天应改判为 34.92%”，但足以说明不能把今天约 70.7% 的模型值当作七成把握。',
        '', '因此，后续研究应保留季度滚动更新的近期改善，同时优先验证慢模型的概率收缩／校准与极端概率失真；快模型需要先证明能提取超出简单参照的次日信号，而不是继续增加训练遍数。这些是待检验方向，本轮没有按已见成绩再追加调参。',
        '', '**各项固定对照**','', '所有概率候选使用各自周期完全相同的成熟预测日期。下表列关键固定对照；全部候选、种子、年份和状态结果均保存在 results 中，未隐藏失败项。','']
    for h in [1,5]:
        lines += [f'快模型 H1：' if h==1 else '慢模型 H5：','']
        records=[metrics[(metrics.h==h)&metrics.method.eq(name)].iloc[0].to_dict() for name in NAMES]
        lines += table(records,[('方案',lambda r:NAMES[r['method']]),('准确率',lambda r:f"{r['accuracy']:.2%}"),
            ('平衡准确率',lambda r:f"{r['balanced_accuracy']:.2%}"),('AUROC',lambda r:f"{r['auc']:.4f}"),
            ('Brier',lambda r:f"{r['brier']:.6f}"),('对数损失',lambda r:f"{r['logloss']:.6f}")])
        lines += ['']
    lines += ['窗口对照固定的是自然遍数，实际样本数和优化步数随 3／5 年窗口变化。它检验固定遍数下的完整训练方案，不把差异全部归因于数据窗口本身，也不重复小样本来凑齐旧模型步数。季度整网更新仅在 2026 年 3 月和 6 月截止重训，2023—2025 年和 2026 年第一季度沿用年度模型。',
        '', '**逐年表现**','','先看同一期的参照，再判断“越近是否越低”。2026 年只覆盖截至 9 月 15 日已经成熟的标签。','']
    chosen=years[years.method.isin([mainmethod,'frequency_5y','market4_5y','cal.quarter_four_state','mse_q2026_e20.vol.U'])]
    lines += table(chosen.sort_values(['h','year','method']).to_dict('records'),[('周期',lambda r:f"H{r['h']}"),('年份',lambda r:r['year']),
        ('方案',lambda r:NAMES[r['method']]),('样本',lambda r:r['n']),('准确率',lambda r:f"{r['accuracy']:.2%}"),('Brier',lambda r:f"{r['brier']:.6f}")])
    lines += ['','![快慢模型的年度概率误差与校准图](快慢模型测试.png)',
        '', '**重叠标签、验证和统计证据**','']
    lines += table(counts,[('周期',lambda r:f"H{r['h']}"),('成熟日信号',lambda r:r['mature_daily']),
        ('最大不重叠标签数',lambda r:r['greedy_max_disjoint']),('覆盖的日收益边',lambda r:r['covered_return_edges']),
        ('平均复用倍数',lambda r:f"{r['mean_edge_reuse']:.3f}"),('最后成熟信号日',lambda r:r['last_mature_signal'])])
    lines += ['', '“最大不重叠标签数”是区间组合计数，不是有效独立样本量。H5 不能因为每天出一个信号就声称独立周样本增加了五倍。主要比较使用同日期配对的 20 日移动块重采样，40 日块作为敏感性检查；两周期各对频率和简单行情基准，共四项主要 Brier 比较作 Holm 校正。区间仍受时间依赖、非平稳及反复研究历史的限制，不是新前瞻证据。','']
    lines += table(boots.to_dict('records'),[('周期',lambda r:f"H{r['h']}"),('参照',lambda r:NAMES[r['control']]),('块长',lambda r:r['block']),
        ('主模型Brier减参照',lambda r:f"{r['delta_brier']:+.6f}"),('95%块重采样区间',lambda r:f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]"),('Holm p',lambda r:f"{r['p_holm']:.4f}")])
    coverage=load_csv(OUT/'calibration_coverage.csv')
    lines += ['', '校准使用最近 252 个成熟日信号训练、63 个成熟日信号验证；训练标签终点必须严格早于验证首个信号日。每状态至少有 10／5 个最大不重叠训练／验证标签，且验证 Brier 更低、方向正确数不减少才启用，之后不再重拟合。年度切换后的第一季度回到年度模型。校准的正则强度沿用固定数值 20，但日样本损失总和与旧周样本的相对正则强度并不相同，不能称为旧策略的逐元素复制。','']
    lines += table(coverage.to_dict('records'),[('周期',lambda r:f"H{r['h']}"),('修正',lambda r:NAMES[r['method']]),
        ('启用日期',lambda r:r['adjusted_dates']),('概率实际改变日期',lambda r:r['probability_changed_dates']),('方向实际改变日期',lambda r:r['direction_changed_dates'])])
    lines += ['', '每次门控的训练／验证日期、最大不重叠数、删除区段敏感性及相邻窗口复用均存档。多个候选共享同一天的正确判断，不构成多个独立成功事件；通过了验证却后来没有遇到该状态，也不算实际覆盖。',
        '', '**以前的问题这次如何处理**','', '详见[历史问题逐项核对](历史问题核对.md)。特别保留原来波动与价格顺序交互的局部改善，同时重测窗口、轮数、分类目标、近期加权、季度整网更新和校准频率。HMM、DTW、路径效率、Huber、旧状态保留等本轮没有完整重跑，已逐项注明。',
        '', '个股 v1 的近期改善同样保留：[原周目标结果](../stock_model_603986_v1/建模结果.md)中，2026 年主模型为 20/32＝62.5%，全期为 91/175＝52.0%。本次换了收盘至收盘目标、增加每天的预测日期，又加入截至 9 月 15 日的数据，因此不能拿这个 62.5% 和新日频成绩直接计算“提升／下降幅度”。',
        '', '**验证与使用边界**','', f"[模型独立复核](results/models_verified.json)：{read(OUT/'models_verified.json')['checkpoints']} 个检查点与 {read(OUT/'models_verified.json')['heads']} 个概率头数值检查；[输出与因果校准复核](results/outputs_verified.json)通过。每个训练截止日前缀、预测前缀、目标成熟度和原始收益标签均核对，旧研究证据哈希保持。",
        '', '通过数值和因果检查仅说明实现按规则运行，不等于预测有效。趋势、波动、振幅和量能特征均来自 603986 自身；尚未加入大盘、半导体板块、新闻或财报因子。',
        '', '原始行情是供应商当前历史版本；前向分红送转构造具有因果性，但缺少当年逐日发布的行情修订档案。206 个缺失股票交易日保留为缺失，涉及它们的输入或目标窗口排除；不伪造停牌价格或原因。2023 年开始评价，是为了统一足够的训练历史；单只用户指定股票不能代表整个股票池。',
        '', '这是一轮可复算的研究测试，历史逐日概率不是当年真实发出的预测。本轮最新输出也没有追记进旧前瞻账本。目录中的数据与参数冻结至 9 月 15 日，没有把新的自动每日下载／运行入口接到原每周指数程序。',
        '', '[运行说明](README.md) · [全部方案指标](results/metrics.csv) · [各年指标](results/year_metrics.csv) · [最新全部概率](results/latest_probabilities.json) · [冻结方案](protocol.json)']
    (ROOT/'快慢模型测试报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    instructions='''本目录是 603986 兆易创新的 H1／H5 日频预测研究。请先看「快慢模型测试报告.md」和「历史问题核对.md」。

H1：下一交易日收盘涨跌；H5：未来第 5 个交易日收盘涨跌。两者每天使用最新 125 根日线生成输入；年度滚动训练与每天更新输入是两件事。这里已经算完 2023 年至 2026 年 9 月 15 日的固定历史回放。

研究数据和结果保存在本目录，原指数程序、前瞻账本与个股 v1 保持原样。打开报告不会自动下载数据。本版本没有安装自动任务，也没有接入原「打开每周预测.cmd」。

使用 GPU Python 3.13 解释器：
`D:\\ddos_v3\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -X utf8`

已完成的流水线为 prepare.py → train.py --h 1／--h 5 → score.py → verify.py models／outputs → audit.py → deliver.py。两个训练进程按周期隔离。已完成的拟合必须先通过哈希核对才可复用；评分和报告不覆盖旧结果。若要检验下一组新规则，应建立新研究目录并先冻结方案，不直接编辑本目录参数后覆盖成绩。

文件说明：
- protocol.json、results/freeze.json：训练前规则与源码／输入哈希。
- results/evaluation_implementation_freeze.json：评分前实现与数值容差。
- results/fits：84 个网络的成员、曲线、168 个检查点、头系数和原始概率缓存。
- results/predictions_h1.csv／predictions_h5.csv：全部逐日方案概率，matured=False 不计分。
- results/latest_probabilities.json：9 月 15 日输入的所有模型输出；按实际计算时刻标记。
- results/metrics.csv、year_metrics.csv、bootstrap.csv：统一样本评价。
- results/prior_problem_coverage.json：前 51 轮问题的逐项覆盖和未重测边界。
- results/delivery_manifest.json：最终交付文件哈希。

主模型预先固定为 mse_5y_e20.vol.U。其余候选是对照，不是可在结果最好时自动切换的投资策略。原生收益回归分数独立保存在 native_returns 中，不直接伪装成概率。
'''
    (ROOT/'README.md').write_text(instructions,encoding='utf-8')
    print(json.dumps(headline,indent=2))

if __name__=='__main__':main()

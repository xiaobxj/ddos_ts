"""Generate an auditable Chinese report and standalone research figures."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
ARMS=['raw_fixed','raw_adaptive','raw_permuted','kronos_fixed','kronos_adaptive','kronos_permuted','random_tokenizer_adaptive']
NAMES=dict(zip(ARMS,['A 原始数值＋固定分段','B 原始数值＋自适应分段','C 原始数值＋打乱分段',
                     'D Kronos 编码＋固定分段','E Kronos 编码＋自适应分段','F Kronos 编码＋打乱分段','G 随机编码＋自适应分段']))
NAMES.update(ridge_matched_schedule='Ridge：同日期重训',training_mean='训练样本平均收益',buy_hold='买入持有',
             zero_return_cash='零收益预测／空仓',round3_quarterly_ridge='第三轮 Ridge：季度重训',
             round3_direct_kronos='第三轮 Kronos-small：直接预测')
ENGLISH=dict(zip(ARMS,['A Raw / fixed','B Raw / adaptive','C Raw / permuted',
                       'D Kronos / fixed','E Kronos / adaptive','F Kronos / permuted','G Random / adaptive']))
ENGLISH.update(ridge_matched_schedule='Ridge / matched refits',buy_hold='Buy & hold')
COLORS=dict(zip(ARMS,['#2774c7','#14846e','#cb7c27','#2774c7','#14846e','#cb7c27','#9559a8']))
COLORS.update(ridge_matched_schedule='#e24b59',buy_hold='#6b7280')


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def pct(value):
    return f'{100*value:.2f}%'


def md(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def figures(table,curves,seeds,pairs):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig,axes=plt.subplots(2,1,figsize=(12,9.6),layout='constrained',sharex=True)
    for ax,names,title in zip(axes,[ARMS[:3],ARMS[3:]],
                             ['Raw OHLCV representation','Frozen tokenizer representations']):
        for name in names+['ridge_matched_schedule','buy_hold']:
            g=curves[(curves.window=='common')&(curves.method==name)]
            ax.plot(pd.to_datetime(g.date),g.wealth,color=COLORS[name],label=ENGLISH[name],
                    linewidth=1.7,linestyle='--' if name=='buy_hold' else '-')
        ax.axhline(1,color='#999999',linewidth=.7,linestyle=':')
        ax.set(title=title,ylabel='Wealth, initial capital = 1')
        ax.legend(loc='upper left',fontsize=9,ncol=3,frameon=False)
        ax.margins(y=.2)
    fig.suptitle('103 weekly decisions | CSI300 next-open long/cash proxy\n10 bps one-way friction; common dates, matched refits',fontsize=14)
    fig.savefig(OUT/'ablation_equity.png',dpi=180)
    fig.savefig(OUT/'ablation_equity.svg')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.3),layout='constrained')
    positions=np.arange(len(ARMS));seed_numbers=sorted(seeds.seed.unique())
    for offset,number,color in zip([-.16,0,.16],seed_numbers,['#7da2c6','#467ea9','#204d76']):
        values=[seeds[(seeds.window=='common')&(seeds.method==name)&(seeds.seed==number)].total_return.iloc[0] for name in ARMS]
        ax.scatter(values,positions+offset,label=f'Seed {number}',color=color,s=38,zorder=3)
    ensemble=[table[(table.window=='common')&(table.method==name)].total_return.iloc[0] for name in ARMS]
    ax.scatter(ensemble,positions,marker='D',facecolors='none',edgecolors='#e24b59',s=85,linewidths=1.7,label='Three-seed mean forecast',zorder=4)
    hold=table[(table.window=='common')&(table.method=='buy_hold')].total_return.iloc[0]
    ax.axvline(hold,color='#737373',linestyle='--',linewidth=1,label='Buy & hold')
    ax.axvline(0,color='#a0a0a0',linewidth=.7)
    ax.set_yticks(positions,[ENGLISH[n] for n in ARMS]);ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel='Cumulative net return, 10 bps one-way friction',
           title='Sensitivity to downstream training seed | 103 weekly decisions')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=9,frameon=False)
    fig.savefig(OUT/'seed_sensitivity.png',dpi=180)
    fig.savefig(OUT/'seed_sensitivity.svg')
    plt.close(fig)
    family=[r for r in pairs if r['window']=='common']
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    labels=[]
    for i,r in enumerate(family):
        d=r['rmse'];v=d['difference']*100
        ax.errorbar(v,i,xerr=[[v-d['ci95_low']*100],[d['ci95_high']*100-v]],fmt='o',
                    color='#2774c7',capsize=4,linewidth=1.5)
        labels.append(ENGLISH[r['method']]+' vs '+ENGLISH[r['reference']])
    ax.axvline(0,color='#e24b59',linewidth=1)
    ax.set_yticks(range(len(family)),labels);ax.invert_yaxis()
    ax.set(xlabel='RMSE difference, percentage points (negative favors first method)',
           title='Seven prespecified comparisons | 95% block-bootstrap intervals\n103 observations; block length 8; 10,000 replicates')
    fig.savefig(OUT/'paired_rmse.png',dpi=180)
    fig.savefig(OUT/'paired_rmse.svg')
    plt.close(fig)


def main():
    config=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    selection=read('selection.json');verification=read('verification.json');run=read('training_manifest.json')
    findings=read('findings.json')
    assert verification['status']=='PASS'
    metrics=pd.read_csv(OUT/'metrics.csv');costs=pd.read_csv(OUT/'cost_sensitivity.csv')
    table=metrics.merge(costs.query('cost_bps==10'),on=['window','method'])
    curves=pd.read_csv(OUT/'equity_curves_10bps.csv');seeds=pd.read_csv(OUT/'seed_metrics.csv')
    years=pd.read_csv(OUT/'yearly_metrics.csv');pairs=read('paired_comparisons.json')
    stability=read('seed_stability.json');impacts=read('influence_diagnostics.json');folds=read('folds.json')
    variation=pd.read_csv(OUT/'prediction_variation_diagnostics.csv')
    figures(table,curves,seeds,pairs)
    def results(window,names):
        rows=[]
        for name in names:
            r=table[(table.window==window)&(table.method==name)].iloc[0]
            rmse='—' if name=='buy_hold' else f'{r.return_rmse:.5f}'
            corr='—' if pd.isna(r.correlation) else f'{r.correlation:.3f}'
            rows.append([NAMES[name],pct(r.accuracy),rmse,corr,pct(r.total_return),pct(r.max_drawdown),pct(r.exposure)])
        return md(['模型／策略','方向准确率','收益 RMSE','收益相关性','累计净收益','最大回撤','持仓时间占比'],rows)
    def comparisons(window):
        rows=[]
        for r in pairs:
            if r['window']!=window:continue
            q=r['rmse'];a=r['accuracy']
            rows.append([NAMES[r['method']].split(' ')[0]+' − '+NAMES[r['reference']].split(' ')[0],
                         f"{q['difference']:.6f}",f"[{q['ci95_low']:.6f}, {q['ci95_high']:.6f}]",
                         f"{r['mse']['holm_adjusted_p']:.4f}",
                         f"{a['difference']*100:+.2f}",f"[{a['ci95_low']*100:.2f}, {a['ci95_high']*100:.2f}]"])
        return md(['前者减后者','ΔRMSE','RMSE 差 95% 区间','MSE Holm p','Δ准确率／百分点','准确率差 95% 区间／百分点'],rows)
    val=md(['训练轮数','2018–2020 验证 RMSE','方向准确率'],
           [[r['epochs'],f"{r['rmse']:.6f}",pct(r['accuracy'])] for r in selection['rankings']])
    seedrows=[]
    for name in ARMS:
        g=seeds[(seeds.window=='common')&(seeds.method==name)].sort_values('seed')
        stable=next(s for s in stability if s['window']=='common' and s['method']==name)
        seedrows.append([NAMES[name]]+[pct(r) for r in g.total_return]+[pct(stable['unanimous_direction_rate'])])
    seedtable=md(['模型','种子 20260910 净收益','种子 20260911 净收益','种子 20260912 净收益','三种子方向一致率'],seedrows)
    yearrows=[]
    for name in ARMS+['ridge_matched_schedule','buy_hold']:
        g=years[(years.window=='common')&(years.method==name)].sort_values('year')
        yearrows.append([NAMES[name]]+[pct(v) for v in g.period_net_return_10bps])
    yeartable=md(['模型','2024 年 7 月起','2025 年','2026 年至 8 月'],yearrows)
    costrows=[]
    for name in ARMS+['ridge_matched_schedule','buy_hold']:
        g=costs[(costs.window=='common')&(costs.method==name)].sort_values('cost_bps')
        costrows.append([NAMES[name]]+[pct(v) for v in g.total_return])
    costtable=md(['模型','0 bp','5 bp','10 bp','20 bp'],costrows)
    foldtable=md(['训练截止日','完成标签的训练样本数','最晚训练标签完成日','预测信号范围','预测周数'],
                 [[r['cutoff'],r['train_n'],r['last_label'],r['first_signal']+' 至 '+r['last_signal'],r['test_n']]
                  for r in folds if r['phase']=='development' and r['method']=='raw_fixed'])
    impactrows=[]
    for r in impacts:
        if r['window']=='common' and r['reference']=='ridge_matched_schedule':
            impactrows.append([NAMES[r['method']],f"{r['total_log_wealth_gap']:.4f}",r['largest_positive_date'],
                               f"{r['largest_positive_log_contribution']:.4f}",f"{r['gap_excluding_largest_positive']:.4f}",
                               f"{r['gap_excluding_top_three']:.4f}"])
    impacttable=md(['模型相对同重训 Ridge','总对数净值差','贡献最大信号周','该周贡献','扣该周贡献后差值','扣前三周贡献后差值'],impactrows)
    variationrows=[]
    for name in ARMS:
        g=variation[variation.method==name].sort_values('cutoff')
        variationrows.append([NAMES[name]]+[f'{v*10000:.2f}' for v in g.within_fold_forecast_sd]+
                             [f'{g.between_fold_share_of_forecast_variation.iloc[0]*100:.3f}%',
                              f'{g.seed_mean_last_epoch_training_loss.min():.4f}–{g.seed_mean_last_epoch_training_loss.max():.4f}'])
    variationtable=md(['模型','2024 折内预测 SD／bp','2025 折内预测 SD／bp','2026 折内预测 SD／bp',
                       '折间均值解释的预测方差','各折末轮训练损失范围'],variationrows)
    # Narrative is written after all frozen runs finish; it does not alter the protocol.
    text=build_report(config,selection,verification,run,findings,results,comparisons,
                      val,seedtable,yeartable,costtable,foldtable,impacttable,variationtable)
    report=OUT/'第四轮测试报告.md'
    report.write_text(text,encoding='utf-8')
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if target.startswith(('https://','http://')):continue
        assert (OUT/target).resolve().exists(),target
        links.append(target)
    files={str(p.relative_to(ROOT)).replace('\\','/'):hash_file(p) for p in ROOT.rglob('*')
           if p.is_file() and not any(part in p.parts for part in ['.venv_gpu','vendor','__pycache__','.git'])
           and p.name!='delivery_manifest.json'}
    (OUT/'delivery_manifest.json').write_text(json.dumps(dict(report='results/第四轮测试报告.md',
        report_links_checked=len(links),files=files),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(report),characters=len(text),local_links_checked=len(links),
                         hashed_files=len(files)),ensure_ascii=False))


# Report prose is kept separately from numeric evaluation and training.
def build_report(config,selection,verification,run,findings,results,comparisons,
                 val,seedtable,yeartable,costtable,foldtable,impacttable,variationtable):
    summary='\n\n'.join(findings['conclusions'])
    next_steps='\n\n'.join(findings['next_steps'])
    main_results=results('common',ARMS+['ridge_matched_schedule','training_mean','buy_hold','zero_return_cash'])
    sensitivity=results('post_release_sensitivity',ARMS+['ridge_matched_schedule','buy_hold'])
    context=results('common',['round3_quarterly_ridge','round3_direct_kronos'])
    return f'''# 第四轮方法测试：Crossformer、分段和 Kronos 编码

实验日期：2026-09-10。研究目录：research_v4。完成七组模型、三个随机种子、三个滚动区间的 63 次开发期训练，并保存全部检查点。

## 结果与判断

{summary}

本轮是补齐缺失接口后进行的探索性组合实验。前三轮已经检查过这段历史，因此这里的结果属于开发期证据，不能当作全新留出集的确认结果，也不能宣称精确复现原文的 57.81%。

## 共同区间：103 次周度决策

信号日为 2024-07-05 至 2026-08-21，首笔可执行开盘为 2024-07-08，最终结算为 2026-08-31 开盘。所有模型使用完全相同的信号与实际成交代理价格。各神经网络的主结果均取三个种子的收益预测均值，然后按预测大于零持有，否则空仓；不选择收益最好的种子。

下表累计收益和最大回撤均包含单边 10 bp 假设摩擦。方向准确率按预测收益符号计算，不是所有交易的盈利比例。买入持有的“准确率”只是实际上涨周的比例，RMSE 留空，因为它是仓位基准。

{main_results}

![不同分段与编码方式的净值](ablation_equity.png)

图中净值包含区间内每日开盘价格波动，不只画周度结算点；最大回撤也按同一每日开盘净值口径计算。持仓时间占比按回测中的交易日开盘区间计，可能不同于预测为上涨的周数占比。价格指数不含分红，现金利息取零。

第三轮两条冻结预测仅作背景参照。季度 Ridge 与本轮同日期重训 Ridge 的训练截止日不同，直接 Kronos-small 与本轮 tokenizer＋Crossformer 也属于不同模型，不能混称同一控制组。

{context}

## 七个预先指定的配对比较

同一信号日配对，采用长度为 8 个观察值的循环分块 bootstrap，重复 10,000 次。RMSE 差值小于零代表前者更好；准确率差值大于零代表前者更好。MSE 的检验在每个评价窗口内对七项主要比较做 Holm 校正。RMSE 区间是边际 95% 区间，不是七项同时置信区间；方向及收益指标属于探索性诊断。

{comparisons('common')}

![配对预测误差比较](paired_rmse.png)

完整结果还包含净周收益差的区间，以及每组模型相对 Ridge、零收益和买入持有的诊断比较。bootstrap 在观察周之间分块，未涵盖新市场时期的不确定性，也没有把三个训练种子当作三个独立市场样本。

## 预测有没有随样本变化

下表采用三种子集成预测，分别在三个固定模型区间内计算预测标准差。末列为每折最后一轮、三种子平均的训练期标准化 MSE；训练期按标签均值输出的标准化 MSE 基准为 1。训练日志中的损失含训练模式 dropout 和逐批更新，因此它是优化过程诊断，不能替代在固定最终权重上的完整训练集误差。

{variationtable}

折间解释比例按“1 − 各折内预测平方离差之和／全区间预测平方离差”计算。D 组几乎完全依靠重训后的整体偏移改变方向，E 组也主要受折间均值变化影响。这是完成实验后的描述性诊断，没有据此修改模型或预算。有效条件期望的波动本就可以远小于实际收益波动；这里需要验证的是新增预测能否降低误差，而不是强行放大预测幅度。

## 种子、年份与成本敏感性

以下是各个训练种子单独执行相同仓位规则的结果。集成结果是先平均预测再决定仓位，不能用三个单种子净收益的均值替代。随机 tokenizer 本身只用预先固定的一个初始化种子；因此 G 组只能检验该随机编码对照，尚未覆盖随机编码器初始化的分布。

{seedtable}

![训练随机种子敏感性](seed_sensitivity.png)

各年份使用连续持仓路径切片计算，没有每年重新空仓或重复收取期末清算成本；2024 和 2026 均不是完整年份。

{yeartable}

单边成本同时作用于首次建仓、每次仓位变化和最后清算；持续持仓不重复收费。这里只设定 0、5、10、20 bp 的总摩擦情景，没有把它们解释成交易所法定费率或某家券商报价。

{costtable}

相对同日期重训 Ridge 的收益差还做了单周贡献检查。下面只是从已实现的对数收益差中扣除贡献，并未删除价格数据后重跑交易，更不是可事先执行的避开某周策略。

{impacttable}

## 公开发布后的敏感性窗口：48 次决策

起点按第三轮冻结口径保留为 2025-08-18，落入样本的第一个周五信号是 2025-08-22。该切片从空仓重新计算首笔成本。它有助于检查较短窗口的变化，但仍是已观察历史；本轮 Crossformer 并不是当时已发布、已保存的预测模型。

{sensitivity}

{comparisons('post_release_sensitivity')}

## 数据、训练和因果边界

复用前三轮冻结的腾讯日线与执行标签，没有补充新价格或改动早期结果。沪深 300 共 4,046 条日线，覆盖 2010-01-04 至 2026-08-31；2015-03-27 的 OHLC 异常行仍保留在交易日历中，输入或标签触及异常的样本不参与训练。全部模型使用 3,785 个候选日度窗口中在各截止日已经完成标签的样本。

输入均为信号日收盘后可观察的最近 125 根 K 线。标签是“当前信号后的实际开盘到下一个同星期信号后的实际开盘”的收益；训练可用周一至周五的成熟样本，评估只用冻结的周五样本。遇到周五休市，持有至下一个实际周五信号的下一交易日开盘，因此不是强行假设每次都相隔五个交易日。

{foldtable}

所有新增模型都在同一日期从头训练，先用 2024-06-30 截止的历史，再于 2024 和 2025 年底重训。训练标签必须在截止日前完成。区间内不调参、不按测试收益改轮数、阈值或种子。Ridge 使用第三轮相同的 255 个经济特征、lambda=10，以及本轮相同的完成标签和重训日期。

训练轮数只由 A 组原始数值固定分段模型在 2018–2020 年的 141 个周五样本选择。每个年度、每个种子都训练至 10 轮，记录第 2、5、10 轮验证结果；按三种子预测均值的 RMSE 选择一次，再应用于全部七组。

{val}

最终固定为 **{selection['selected_epochs']} 轮**。验证期训练均值基准 RMSE 为 {selection['training_mean_validation_rmse']:.6f}，零收益基准为 {selection['zero_validation_rmse']:.6f}。验证准确率不是开发期准确率，所选轮数也只代表本轮小型网络和固定训练预算的结果。未针对编码模型分别寻优，因此这是共同预算下的消融，不是各方法最优性能排行榜。

AdamW 学习率 0.001、weight decay 0.01、batch 128、梯度范数裁剪 1.0，三个下游种子为 20260910、20260911、20260912。只打乱已经完成标签的训练样本，验证和开发信号保持时间顺序。GPU 使用 float32、确定性运算和关闭 TF32 的设置。

## 方法接口与控制组

**分段。** 固定分段为 25 个连续的五日片段。自适应分段复用第一轮冻结的年度 P0 和逆向 DTW：候选长度 5–25，L1 最小总路径距离再按该路径长度归一化，保留最前端不足五日的余段。P0 来自已取得的七个其他指数、八个原型，而不是原文缺失名单的 78 指数完整 P0。2024、2025、2026 年的原型分别只使用截至上年末的数据；训练样本也使用该折截止日可取得的特征提取器。

打乱分段沿用每个样本自适应分段的长度集合和有效片段数，用预先固定的 20260909＋anchor 随机种子重排长度，再切割原始时间顺序的窗口。它保留压缩程度，但破坏与形态匹配的边界。B 对 C、E 对 F 比较比单纯与固定 25 段比较更能检查边界选择的作用；这种对照仍非所有可能分段的穷举检验。

每段线性重采样到五个点，最多 25 个槽位右对齐，不足部分补零。额外传入片段长度／125、片段终点／125 和有效性掩码。原始数值为 log OHLC、log1p 成交量，各自在已观察的 125 日窗口内标准化。

**Kronos 编码。** 使用官方冻结 Kronos-Tokenizer-base，而非把高位 token ID 当连续数值。OHLCVA 采用官方 float32 窗口均值、标准差＋1e-5 及 ±5 裁剪；缺少的成交额明确置零。每个 125 日窗口单独调用编码器，再由官方 indices_to_bits 还原为 20 维、幅度 1／√20 的双极坐标。严禁先编码完整时间序列再切历史窗口。对分段内这些坐标做插值后，它们是连续向量特征，不再被解释为有效离散 token。

G 组使用同一个官方 tokenizer 架构、随机初始化种子 20260913、相同归一化和量化接口，随后冻结。E 与 G 的下游维度和参数量一致；原始数值与 tokenizer 组分别是 5 维和 20 维，比较它们同时改变了表达维度，不能把全部差异都归因于预训练。

编码器沿用第三轮按 LFS 核验过的公开权重。论文描述的预训练结束日为 2024 年 6 月，但该具体权重没有独立公布更细的训练截止证据，所以共同窗口依赖论文日期假设。仓库版本、权重实际发布时间和历史上能否使用是不同问题；本实验不是当时可用性的历史重建。预训练编码器没有参与 2018–2020 年轮数选择。

**Crossformer。** 保留官方 DSW 嵌入、时间／维度双阶段注意力、router、两层层次编码器和三层叠加解码器。使用 d_model=32、d_ff=64、4 heads、router factor=4、dropout=0.1、合并窗口 2；有效编码尺度为 25 槽到 13 槽。新增两维片段几何嵌入，以及从解码器 5×输入维度特征到一个收益值的线性读出层。收益标签按训练期均值／标准差缩放，预测后还原。

读出层零初始化，其余网络参数正常训练。解码器中间值是潜在特征，**没有训练成五日 OHLCAV 预测**；这是与原文目标函数的明确差别。原始数组三组各 138,441 个可训练参数，20 维编码四组各 150,996 个。固定、自适应、打乱分段在各自表示下参数量一致；预训练与随机编码对照也一致。

官方 Crossformer 原版没有变长补零掩码。本轮在时间注意力与解码器历史交叉注意力中屏蔽无效键，并在注意力、维度路由和分段合并后压制无效状态；合并后的有效性按参与片段取 OR。奇数槽位沿用官方末段复制规则。新增几何特征关闭且所有片段有效时，解码结果与未改动的官方模型通过数值一致性检查。

## 验证与可复查证据

最终验证状态：**{verification['status']}**。九项单元测试覆盖官方等价性、掩码数值不变性、无效输入梯度为零、仅一个有效片段、合成小样本可训练、未来价格扰动不影响当前窗口变换、固定重组一致性、打乱长度保持性以及编码坐标合法性。另独立重编码 12 个预训练／随机窗口，并加载 **{verification['development_checkpoints_replayed']} 个开发期检查点**回放全部种子预测，最大差异为 {verification['max_replayed_prediction_error']:.3g}。

独立重算全部 {verification['main_predictions']} 行主预测标签、{verification['seed_predictions']} 行种子预测的集成规则、训练边界、验证轮数选择和持仓方向；每日净值、周度复利与买入持有解析计算相符。输入、源码、原型、权重和检查点哈希均通过校验；前三轮 {verification['old_evidence_files_preserved']} 个纳入保全的证据文件未改变。外部 vendor 和权重目录不计入该证据文件数，它们另外按来源清单核验。

首次重编码复验发现 float32 均值／标准差因数组内存布局而有约 1e-5 的差异：准备程序先将全部窗口堆成 C 连续数组，最初验证程序直接处理 pandas 的列连续数组。验证程序恢复原有 C 连续布局后，全部 3,785 个窗口输入逐值完全一致，抽样编码也逐位一致；没有放宽比较精度或改动数据、训练及预测。首次失败日志和定位结果一并保留。

模型训练耗时 {run['elapsed_seconds']/60:.2f} 分钟，不含环境安装、输入准备、统计分析和检查点复验；设备为 {run['device']}，torch {run['torch']}。本轮创建独立 GPU 虚拟环境，使用电脑已有驱动，没有修改前几轮源码、数据或账户配置。

- [冻结实验协议](../protocol.json)、[训练选择](selection.json)、[训练来源清单](training_manifest.json)、[输入准备清单](preparation_manifest.json)。
- [主预测逐行记录](predictions.csv)、[全部种子预测](development_seed_predictions.csv)、[训练曲线](training_curves.csv)、[滚动区间](folds.json)。
- [评价指标](metrics.csv)、[成本敏感性](cost_sensitivity.csv)、[年度结果](yearly_metrics.csv)、[种子结果](seed_metrics.csv)。
- [预测变化诊断](prediction_variation_diagnostics.csv)、[归一化布局定位](normalization_layout_diagnostic.json)、[首次复验日志](verification_first_attempt.txt)。
- [七项配对统计](paired_comparisons.json)、[基准诊断比较](diagnostic_comparisons.json)、[单周贡献诊断](influence_diagnostics.json)。
- [最终复验](verification.json)、[测试日志](test_results.txt)、[检查点与哈希](checkpoint_manifest.json)、[编码器重算核验](tokenizer_verification.json)。
- 图表提供 PNG 与可编辑矢量版：[净值 SVG](ablation_equity.svg)、[种子 SVG](seed_sensitivity.svg)、[配对误差 SVG](paired_rmse.svg)。
- [执行与复验说明](../README.md)。交付文件哈希保存在同目录 delivery_manifest.json；它不把自己递归纳入哈希。

## 后续方向

{next_steps}

## 原始来源

方法结构依据 [Crossformer 官方仓库](https://github.com/Thinklab-SJTU/Crossformer/tree/c10c8eadb153d1dd9798250967747ca3ebb81383)，官方组件保持在 vendor 中，本轮的掩码和读出层位于 architecture.py。Kronos 编码器依据 [官方仓库固定版本](https://github.com/shiyu-coder/Kronos/tree/67b630e67f6a18c9e9be918d9b4337c960db1e9a) 与 [Kronos 原论文](https://arxiv.org/abs/2508.02739)，权重为 [Kronos-Tokenizer-base 固定版本](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base/tree/0e0117387f39004a9016484a186a908917e22426)。本报告中的数值均来自本地冻结实验，不是论文中的报告结果。
'''


if __name__=='__main__':
    main()

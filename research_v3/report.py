"""Standalone Chinese research report and exportable scientific figures."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
NAMES={
    'ridge_target':'Ridge：原有特征／完整历史',
    'hgb_target':'小型树模型：原有特征／完整历史',
    'ridge_target_matched':'Ridge：原有特征／匹配历史',
    'hgb_target_matched':'小型树模型：原有特征／匹配历史',
    'ridge_cross':'Ridge：加入跨指数特征',
    'hgb_cross':'小型树模型：加入跨指数特征',
    'kronos_ohlcv':'官方 Kronos-small：OHLCV',
    'kronos_ohlc_diagnostic':'官方 Kronos-small：仅 OHLC（诊断）',
    'zero_return_cash':'零收益预测／空仓',
    'buy_hold':'买入持有',
    'training_mean':'训练样本平均收益',
    'round2_frozen':'第二轮冻结基线'
}
ENGLISH={'ridge_target':'Ridge', 'hgb_target':'Small HGB', 'ridge_cross':'Ridge + cross-index (2015+ train)',
         'hgb_cross':'HGB + cross-index (2015+ train)', 'kronos_ohlcv':'Kronos OHLCV',
         'kronos_ohlc_diagnostic':'Kronos OHLC only', 'buy_hold':'Buy & hold'}
COLORS={'ridge_target':'#2563eb','hgb_target':'#ef7d24','ridge_cross':'#198b74',
        'hgb_cross':'#9b4bce','kronos_ohlcv':'#dc4444','kronos_ohlc_diagnostic':'#915faf','buy_hold':'#64748b'}


def pct(x, digits=2):
    return f'{x*100:.{digits}f}%'


def markdown(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)


def main():
    config=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    verify=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    local=json.loads((OUT/'local_run_manifest.json').read_text(encoding='utf-8'))
    kronos=json.loads((OUT/'kronos_run_manifest.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(OUT/'metrics.csv')
    costs=pd.read_csv(OUT/'cost_sensitivity.csv')
    table=metrics.merge(costs.query('cost_bps==10'),on=['window','method'])
    pred=pd.read_csv(OUT/'predictions.csv')
    curves=pd.read_csv(OUT/'equity_curves_10bps.csv')
    yearly=pd.read_csv(OUT/'yearly_metrics.csv')
    seeds=pd.read_csv(OUT/'kronos_seed_metrics.csv')
    sampling=json.loads((OUT/'kronos_sampling_diagnostics.json').read_text())
    pairs=json.loads((OUT/'paired_comparisons.json').read_text())
    impacts=json.loads((OUT/'influence_diagnostics.json').read_text())
    quality=json.loads((OUT/'kronos_output_quality.json').read_text())
    validation=pd.read_csv(OUT/'validation_summary.csv')
    obs=pd.read_csv(OUT/'observation_table.csv')
    panels=['ridge_target','hgb_target','ridge_target_matched','ridge_cross','hgb_target_matched','hgb_cross']
    def row(window,method):
        return table[(table.window==window)&(table.method==method)].iloc[0]
    def result_table(window,names):
        result=[]
        for name in names:
            r=row(window,name)
            rmse='—' if name=='buy_hold' else f'{r.return_rmse:.5f}'
            corr='—' if pd.isna(r.correlation) else f'{r.correlation:.3f}'
            result.append([NAMES[name],int(r.n),pct(r.accuracy),rmse,corr,pct(r.total_return),pct(r.max_drawdown),pct(r.exposure)])
        return markdown(['模型／策略','周数','方向准确率','收益 RMSE','收益相关性','累计净收益','最大回撤','持仓时间占比'],result)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.2})
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    plot_sets=[('development_full',['ridge_target','hgb_target','ridge_cross','hgb_cross','buy_hold'],
                'Previously inspected development history | 272 weekly decisions'),
               ('post_pretrain_common',['ridge_target','hgb_target','kronos_ohlcv','kronos_ohlc_diagnostic','buy_hold'],
                'Common dates after paper-stated pretraining cutoff | 103 weekly decisions')]
    for ax,(window,names,title) in zip(axes,plot_sets):
        for name in names:
            g=curves[(curves.window==window)&(curves.method==name)]
            ax.plot(pd.to_datetime(g.date),g.wealth,label=ENGLISH[name],color=COLORS[name],linewidth=1.6)
        ax.axhline(1.,color='#94a3b8',linewidth=0.8,linestyle=':')
        ax.set(title=title,ylabel='Wealth, start = 1.0')
        ax.legend(loc='best',ncol=2 if window=='development_full' else 3,fontsize=8,frameon=False)
    fig.suptitle('CSI300 next-open long/cash proxy | 10 bps one-way friction',fontsize=14)
    fig.savefig(OUT/'comparison_equity.png',dpi=180)
    fig.savefig(OUT/'comparison_equity.svg')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,name in zip(axes,['ridge_target','hgb_target','kronos_ohlcv']):
        g=pred[(pred.method==name)&(pred.date>='2024-07-01')]
        ax.scatter(g.predicted_return,g.actual,s=21,alpha=.7,color=COLORS[name],edgecolors='none')
        ax.axhline(0,color='#94a3b8',linewidth=.8);ax.axvline(0,color='#94a3b8',linewidth=.8)
        ax.set(title=f"{ENGLISH[name]} | corr = {row('post_pretrain_common',name).correlation:.3f}",
               xlabel='Forecast execution return',ylabel='Actual execution return')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1));ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(-.09,.23)
    fig.suptitle('Same 103 weekly intervals | prediction quality, before transaction costs',fontsize=13)
    fig.savefig(OUT/'forecast_scatter.png',dpi=180)
    fig.savefig(OUT/'forecast_scatter.svg')
    plt.close(fig)
    full='development_full';common='post_pretrain_common';release='post_release_sensitivity'
    r=row(full,'ridge_target');h=row(full,'hgb_target');cross=row(full,'ridge_cross')
    k=row(common,'kronos_ohlcv');kb=row(common,'buy_hold')
    train_all=int(((obs.completed<='2020-12-31')).sum())
    train_match=int(((obs.completed<='2020-12-31')&obs.panel_valid).sum())
    text=f'''# 第三轮方法测试：非线性模型、官方 Kronos 与跨指数信息

生成日期：{pd.Timestamp.now().strftime('%Y-%m-%d')}。本轮是已查看历史上的开发验证，不能当作全新盲测或实盘业绩。

## 结果概览

本轮完成六组本地模型对照，以及真实官方 Kronos-small 的两种输入测试。完整历史共 272 个周度持仓区间；包含 Kronos 的共同区间从 2024-07-05 信号开始，共 {int(k.n)} 个区间，最后统一在 2026-08-31 开盘结束。下述收益均使用单边 10 bp 的假设摩擦、零现金利息及不加杠杆的指数多头／空仓口径。

- 小型非线性模型：完整历史方向准确率 {pct(h.accuracy)}，累计净收益 {pct(h.total_return)}；相同完整训练历史的 Ridge 为 {pct(r.accuracy)}、{pct(r.total_return)}。
- 新增跨指数信息：Ridge 的准确率为 {pct(cross.accuracy)}、净收益 {pct(cross.total_return)}。因其他指数历史较短，必须与下表的“匹配历史”对照比较信息增量，不能只对比完整历史结果。
- 官方 Kronos OHLCV：共同区间准确率 {pct(k.accuracy)}，净收益 {pct(k.total_return)}；同区间买入持有为 {pct(kb.total_return)}。这与五年多的完整历史收益不能直接比较。
- 计算校验状态：**{verify['status']}**。此前两轮的 {verify['previous_files_preserved']} 个证据文件保持原哈希。

**研究判断：本轮没有找到足以稳定替换第二轮 Ridge 基线的证据。**六个本地模型在完整区间的收益预测均方误差都没有优于零收益预测。小型树模型在 2024 年 7 月后的区间取得 {pct(row(common,'hgb_target').total_return)}，但在完整区间为 {pct(h.total_return)}，且共同区间相对 Ridge 的误差／周收益差置信区间均跨零。它相对买入持有的累计对数净值优势为 0.0339；单是 2024-07-19 信号区间的正贡献就有 0.0376，显示短区间超额对个别行情敏感。这是已冻结结果的贡献拆解，不是删除该周后的新回测。

跨指数特征使匹配历史的 Ridge 准确率从 {pct(row(full,'ridge_target_matched').accuracy)} 升至 {pct(cross.accuracy)}，但净收益从 {pct(row(full,'ridge_target_matched').total_return)} 变为 {pct(cross.total_return)}，主要误差检验也未达到校正后的显著改善。官方 Kronos OHLCV 的共同区间收益 RMSE 为 {k.return_rmse:.5f}，高于 Ridge 的 {row(common,'ridge_target').return_rmse:.5f}；这一“误差更大”的差异在本轮探索性 MSE 检验中 Holm 校正 p=0.030。Kronos 的较低回撤同时伴随较低持仓比例，不能只据回撤认定预测更优。

仅 OHLC 的 Kronos 诊断准确率达到 {pct(row(common,'kronos_ohlc_diagnostic').accuracy)}，但净收益只有 {pct(row(common,'kronos_ohlc_diagnostic').total_return)}。准确率、预测收益大小、持仓时机与成本需要共同评估。上述结论只针对已运行的模型、输入和标签，不代表全部非线性模型或微调后 Kronos 的能力上限。

![两个统一比较区间的净值](comparison_equity.png)

## 1. 六组传统模型：完整开发区间

{result_table(full,panels+['training_mean','buy_hold','zero_return_cash'])}

“准确率”统一按预测的持仓收益符号与实际开盘至开盘收益符号比较。空仓行是零收益预测按非上涨分类的机械参照；其准确率不代表预测能力。买入持有是交易基准，未将极小正数编码视为有意义的收益预测。

完整历史 Ridge 的验证参数再次选中 λ=10，其逐期预测与第二轮冻结基线数值一致。“完整历史”和“匹配历史”都采用季度扩展窗口更新；季度更新沿用第二轮方案，本轮没有再次寻找最优更新频率。

## 2. 官方 Kronos 的共同区间

{result_table(common,['ridge_target','hgb_target','ridge_target_matched','ridge_cross','hgb_target_matched','hgb_cross','kronos_ohlcv','kronos_ohlc_diagnostic','buy_hold','zero_return_cash'])}

![相同日期上的收益预测散点](forecast_scatter.png)

Kronos 使用官方 tokenizer 和 24.7M 参数的 small 模型，全部权重冻结，调用官方 `KronosPredictor.predict`。这次确实运行了预训练模型，但还不是方法文章提出的自适应分段＋Kronos tokenizer＋Crossformer 组合。

模型读取每个决策时点之前的 125 根日线。预测区间覆盖入场开盘直至下一次实际周五信号后的退出开盘，通常生成 6 根日线；遇节假日按已冻结交易日历延长。只传入未来日期，不传入未来价格。交易收益预测为“预测退出 open / 预测入场 open − 1”，分母也来自模型预测。实际次日开盘仅用于事后标签与净值。

这是本地季度重训练方法与官方预训练模型直接迁移的方案对照；二者预训练数据、表示与目标适配方式不同，不能把差异全部解释成模型架构优劣，也不能外推到 Kronos 微调后的表现。

每种输入、每个日期固定使用三个种子，每个种子由官方接口平均 8 条采样路径。先将三个种子的 OHLCVA 预测均值合并，再计算收益。所有种子的结果都保留，没有选择表现最好的种子或调整交易阈值。

## 3. 数据与时间边界

数据仍为前两轮归档的腾讯 OHLCV 快照，截止 2026-08-31。沿用所有原始质量标记，保留交易日行位置；目标输入、标签跨越严重错误 OHLC 的样本被排除。没有重新下载或改写价格。

七个其他指数为上证综指、上证 50、中证 500、中证 1000、深证成指、中小 100、创业板指。新增 72 个特征：各指数在 5／20／60 日的相对收益、波动差和收益相关性，以及跨指数上涨比例、平均收益、收益离散程度。加上原有 255 个经济与固定多尺度特征，总计 327 维。所有窗口只使用当时已观察的数据，并按确切日期对齐，没有向前或向后填补缺失。

跨指数共同有效输入从 {local['panel_first_valid']} 开始。在 2020-12-31 训练截止时，完整历史有 {train_all} 个完成标签的日度训练样本，匹配历史有 {train_match} 个；这些周度目标在日度训练中互相重叠，不能视为同数量的独立周样本。两种模型都增加了“同样缩短训练历史、仍只用目标指数特征”的对照，因而跨指数增量比较的样本与参数完全一致。指数之间的相关性也没有被当作独立样本数量。

原始数据没有真实成交额。当前官方预测代码在只缺 amount 时会自动计算“volume × OHLC 均价”，因此本轮显式将 amount 通道置零，阻止自动合成；这里的零代表缺失占位。OHLC-only 是额外的输入诊断，同时将 volume 置零。两者都存在与六通道完整数据不同的输入条件，不能宣称是文章的完整 OHLCAV 复现。

[Kronos 论文附录](https://arxiv.org/html/2508.02739v1)声明预训练数据截止 2024 年 6 月。本轮据此将包含 Kronos 的主要共同比较限定为 2024 年 7 月以后。模型卡没有独立披露该权重专属的训练截止日期，因此这里是依照论文声明划分的历史检验，不能独立认证其所有预训练数据未覆盖测试期。下载的模型及 tokenizer 权重哈希都与 2025-06-30 的首次模型上传记录一致；后续提交为说明文档更新。

另列 2025-08-18 以后的保守发布后敏感性窗口，晚于公开论文与微调脚本公告。使用的官方代码修订时间为 2026 年 4 月，故这个切片也不等同于历史时点可获得软件的逐版本重演。归档行情本身也不是逐时点修订数据库。

## 4. 发布后窗口与成本敏感性

{result_table(release,['ridge_target','hgb_target','ridge_cross','hgb_cross','kronos_ohlcv','kronos_ohlc_diagnostic','buy_hold','zero_return_cash'])}

每个子区间均从空仓和净值 1 开始计算，并在同一结束开盘平仓。指数收益只是可执行时点的价格代理；本轮成本是单边 0／5／10／20 bp 的情景假设，不是券商或交易所实际收费，也不包含 ETF 跟踪误差、分红、冲击与真实买卖价差。最大回撤按每日开盘净值计算。

'''
    for window,title,names in [(full,'完整开发区间成本敏感性',['ridge_target','hgb_target','ridge_cross','hgb_cross','buy_hold']),
                               (common,'Kronos 共同区间成本敏感性',['ridge_target','hgb_target','kronos_ohlcv','kronos_ohlc_diagnostic','buy_hold'])]:
        rows=[]
        for name in names:
            g=costs[(costs.window==window)&(costs.method==name)].set_index('cost_bps')
            rows.append([NAMES[name]]+[pct(g.loc[b,'total_return']) for b in [0,5,10,20]])
        text+=f'### {title}\n\n'+markdown(['模型','0 bp','5 bp','10 bp','20 bp'],rows)+'\n\n'
    text+='## 5. 预先约定的配对比较\n\n负的 RMSE 差表示模型 A 误差更小；收益差使用平均每个周度持仓区间的净收益，单位为百分点。置信区间来自每块 8 个周度观察的循环区块、10,000 次重采样；节假日会使少数持仓区间长于一周。RMSE 对应的均方误差检验在每个窗口内进行 Holm 校正。所有检验都属于已查看历史上的探索性结果。\n\n'
    for window,title in [(full,'完整开发区间'),(common,'论文训练截止后的共同区间')]:
        rows=[]
        for p in pairs:
            if p['window']!=window:continue
            rms=p['rmse'];net=p['mean_weekly_net_10bps']
            rows.append([NAMES[p['method']]+' vs '+NAMES[p['reference']],
                         f"{rms['difference']:.6f} [{rms['ci95_low']:.6f}, {rms['ci95_high']:.6f}]",
                         f"{p['mse']['holm_adjusted_p']:.3f}",
                         f"{net['difference']*100:.3f} [{net['ci95_low']*100:.3f}, {net['ci95_high']*100:.3f}]"])
        text+=f'### {title}\n\n'+markdown(['A vs B','RMSE 差及 95% 区间','MSE 校正 p','平均周净收益差及 95% 区间（pp）'],rows)+'\n\n'
    text+='## 6. 年份、单周贡献与采样波动\n\n'
    rows=[]
    for year in sorted(yearly[yearly.window==common].year.unique()):
        for name in ['ridge_target','hgb_target','kronos_ohlcv','buy_hold']:
            r=yearly[(yearly.window==common)&(yearly.year==year)&(yearly.method==name)].iloc[0]
            rows.append([int(year),NAMES[name],int(r.n),pct(r.accuracy),pct(r.period_net_return_10bps)])
    text+=markdown(['年份','模型','周数','准确率','该段净收益'],rows)+'\n\n2024 与 2026 为不完整年份。该段净收益按持续持仓路径中的周收益分组复利，包含实际边界的交易成本，没有每年重新清仓。完整区间所有模型逐年结果在 `yearly_metrics.csv`。\n\n'
    rows=[]
    for name in ['kronos_ohlcv','kronos_ohlc_diagnostic']:
        for r in seeds[(seeds.window==common)&(seeds.method==name)].itertuples():
            rows.append([NAMES[name],int(r.seed),pct(r.accuracy),f'{r.return_rmse:.5f}',pct(r.total_return)])
    text+=markdown(['输入','种子（各平均 8 条路径）','准确率','RMSE','累计净收益'],rows)+'\n\n'
    for x in sampling:
        if x['window']==common:
            text+=f"{NAMES[x['method']]}：三个种子方向完全一致的日期占 {pct(x['unanimous_direction_rate'])}，逐日期预测收益在种子间的平均标准差为 {pct(x['mean_across_seed_forecast_std'])}。\n\n"
    rows=[]
    for x in impacts:
        if x['window']!=common or x['reference']!='buy_hold':continue
        rows.append([NAMES[x['method']],x['largest_positive_date'],f"{x['total_log_wealth_gap']:.4f}",
                     f"{x['gap_excluding_largest_positive']:.4f}",f"{x['gap_excluding_top_three']:.4f}"])
    text+=markdown(['相对买入持有','最大正贡献信号日','累计对数净值差','去掉最大正贡献','去掉前三个正贡献'],rows)
    text+='\n\n这些贡献分析仅拆解已经冻结的预测，不是删除某周后重新训练或重新执行的回测；数值不能当作另一项可交易业绩。完整历史与所有预先约定比较的贡献记录见 `influence_diagnostics.json`。\n\n'
    for x in quality:
        text+=f"{NAMES[x['method']]} 的集成预测中，{x['bars']} 根输出 K 线有 {x['invalid_bars']} 根违反高低价约束（{pct(x['invalid_fraction'])}）。没有按输出质量删样本或事后投影，预测开盘价均通过有限正数检查。\n\n"
    text+='## 7. 冻结参数、验证集与复现\n\n'
    text+=f"Ridge λ={selection['parameters']['ridge']['setting']}；小型树模型配置为 `{selection['parameters']['hgb']['setting']}`，学习率 0.03、叶节点最少 100 个日度训练样本、L2=10，关闭随机验证集早停。只在 2018—2020 年的 141 个周度验证日期选择参数，然后将相同参数复制给匹配历史和跨指数模型。\n\n"
    text+=markdown(['验证集模型','周数','收益 RMSE','方向准确率'],[[NAMES[r.method],int(r.n),f'{r.return_rmse:.5f}',pct(r.accuracy)] for r in validation.itertuples()])
    text+=f'''

模型选择记录时间：`{selection['frozen_utc']}`。协议哈希：`{local['protocol_sha256']}`。运行中曾在第一次本地验证前遇到 pandas 只读掩码兼容问题，修复只涉及复制数组；初始源码与失败清单在 `../sources/`，没有据开发结果改动方案。

关键产物：

- [冻结协议](../protocol.json)、[选参记录](selection.json)、[校验结果](verification.json)、[测试输出](test_results.txt)。
- [逐期预测](predictions.csv)、[全窗口指标](metrics.csv)、[成本情景](cost_sensitivity.csv)、[逐年结果](yearly_metrics.csv)。
- [配对检验](paired_comparisons.json)、[额外诊断](diagnostic_comparisons.json)、[单周贡献](influence_diagnostics.json)。
- [Kronos 采样种子指标](kronos_seed_metrics.csv)、[输入审计](kronos_input_audit.json)、`kronos_paths/` 中全部预测路径均值。
- [本地运行清单](local_run_manifest.json)、[Kronos 运行清单](kronos_run_manifest.json)、[权重来源与哈希](../sources/weights_manifest.json)。

核验覆盖 {verify['refit_folds_checked']} 个训练／验证折、{verify['predictions']} 条主预测、{verify['official_seed_paths_recomputed']} 份官方分种子预测路径；重新计算全部标签、预测收益、验证集选择、同样本信息增量比较及净值。净值同时与逐日计算、逐周复利和买入持有解析式核对。

路径 CSV 按官方 float32 输出保存。初次复算直接用 float64 解释十进制文本，出现小于 1e-7 的收益差；按原始 float32 恢复后，618 份路径的入场与退出预测价格全部精确一致，所有集成 OHLCVA 输出均逐位一致，收益复算差小于 1e-12。此修正只涉及校验读取精度，没有改动预测或评价结果。

复现命令见 [README](../README.md)。本机使用已有 CPU PyTorch 环境；新增依赖仅位于本轮目录，没有修改共享 Python 环境或显卡驱动。官方源码 commit：`{config['kronos']['code_revision']}`。模型 commit：`{config['kronos']['model_revision']}`；tokenizer commit：`{config['kronos']['tokenizer_revision']}`。各文件 SHA256 保存在清单中。

研究范围仍是沪深 300 周度择时。自适应分段配合神经网络、Crossformer 以及跨资产排序尚未测试，本报告不对这些未运行的组合给出效果结论。

## 参考来源

- [Kronos 官方实现及接口](https://github.com/shiyu-coder/Kronos/tree/{config['kronos']['code_revision']})。
- [Kronos 论文及训练截止声明](https://arxiv.org/html/2508.02739v1)。
- [官方模型权重提交历史](https://huggingface.co/NeoQuasar/Kronos-small/commits/main)、[官方 tokenizer 提交历史](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base/commits/main)。
- [scikit-learn HistGradientBoostingRegressor 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)。
'''
    (OUT/'第三轮测试报告.md').write_text(text,encoding='utf-8')
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in ROOT.rglob('*') if p.is_file() and not any(s in p.parts for s in ['.git','__pycache__','vendor_py','models','vendor'])
           and p.name!='delivery_manifest.json'}
    (OUT/'delivery_manifest.json').write_text(json.dumps(dict(files=files,generated_utc=pd.Timestamp.now(tz='UTC').isoformat()),indent=2),encoding='utf-8')
    print('Report and figures generated:',OUT/'第三轮测试报告.md')


if __name__=='__main__':
    main()

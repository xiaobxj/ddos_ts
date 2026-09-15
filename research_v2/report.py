"""Chinese report and standalone research charts, based only on saved results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
NAMES={
 'a_close_annual':'A 旧表示＋直接收盘收益', 'b_exec_annual':'B 旧表示＋持仓收益',
 'c_econ_annual':'C 经济特征75维', 'd_multi_exp_annual':'D 多尺度／扩展历史／年度',
 'e_adaptive_annual':'E 经济特征＋自适应分段', 'f_multi_exp_quarter':'F 多尺度／扩展历史／季度',
 'g_multi_3y_annual':'G 多尺度／最近3年／年度', 'h_multi_5y_annual':'H 多尺度／最近5年／年度',
 'i_multi_3y_quarter':'I 多尺度／最近3年／季度', 'j_multi_5y_quarter':'J 多尺度／最近5年／季度',
 'k_logit_exp_annual':'K 上涨概率／扩展历史／年度', 'l_logit_5y_quarter':'L 上涨概率／最近5年／季度',
 'always_up':'持有指数', 'cash':'恒定零收益／空仓',
 'v1_adaptive_normalized':'上一轮自适应模型', 'v1_fixed_10':'上一轮固定10日模型',
 'selected_threshold_10bps':'F＋10基点阈值', 'selected_threshold_20bps':'F＋20基点阈值',
 'selected_threshold_40bps':'F＋40基点阈值'}


def pct(x,d=2):
    return f'{x*100:.{d}f}%'


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)


def main():
    cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    selection=json.loads((OUT/'selection.json').read_text())
    chosen=selection['validation_selected_pipeline']
    metric=pd.read_csv(OUT/'metrics.csv').set_index('method')
    costs=pd.read_csv(OUT/'cost_sensitivity.csv')
    cost=costs[costs.cost_bps==10].set_index('method')
    pred=pd.read_csv(OUT/'predictions.csv')
    year=pd.read_csv(OUT/'yearly_metrics.csv')
    curve=pd.read_csv(OUT/'equity_curves_10bps.csv')
    weekly=pd.read_csv(OUT/'weekly_net_returns_10bps.csv').pivot(index='date',columns='method',values='net_return')
    validation=pd.read_csv(OUT/'validation_predictions.csv')
    v=validation[(validation.method==chosen)&np.isclose(validation.lam,selection['selected_lambdas'][chosen]['lam'])]
    validation_zero=float(np.sqrt(np.mean(v.actual**2)))
    position=pred.pivot(index='date',columns='method',values='position')
    log_gap=np.log1p(weekly.f_multi_exp_quarter)-np.log1p(weekly.d_multi_exp_annual)
    top=log_gap.idxmax()
    influence=dict(comparison='f_multi_exp_quarter vs d_multi_exp_annual',
                   differing_position_intervals=int((position.f_multi_exp_quarter!=position.d_multi_exp_annual).sum()),
                   largest_positive_contribution_signal=top,largest_log_return_contribution=float(log_gap.max()),
                   total_log_return_gap=float(log_gap.sum()),gap_excluding_largest_interval=float(log_gap.sum()-log_gap.max()),
                   interpretation='descriptive decomposition of fixed forecasts, not re-training or a new selected backtest',
                   validation_selected_rmse=float(np.sqrt(np.mean((v.predicted_return-v.actual)**2))),
                   validation_zero_rmse=validation_zero)
    (OUT/'influence_diagnostics.json').write_text(json.dumps(influence,indent=2),encoding='utf-8')

    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    colors={chosen:'#087E8B','v1_adaptive_normalized':'#B4603A','always_up':'#64748B'}
    fig,axes=plt.subplots(2,2,figsize=(14,9.5),layout='constrained')
    ax=axes[0,0]
    for method in colors:
        g=curve[curve.method==method]
        name={'f_multi_exp_quarter':'验证集选定新方案','v1_adaptive_normalized':'上一轮自适应','always_up':'持有指数'}[method]
        ax.plot(pd.to_datetime(g.date),g.wealth,lw=1.6,color=colors[method],label=name)
    ax.axhline(1,lw=.8,ls='--',color='#999999')
    ax.set(title='同一执行区间、单边10基点：净值',ylabel='初始净值 = 1')
    ax.legend(fontsize=9)
    ax.grid(axis='y',alpha=.2)
    ax=axes[0,1]
    mapping=[['d_multi_exp_annual','f_multi_exp_quarter'],['g_multi_3y_annual','i_multi_3y_quarter'],['h_multi_5y_annual','j_multi_5y_quarter']]
    values=np.array([[cost.loc[m,'total_return'] for m in row] for row in mapping])
    ax.imshow(values,cmap='RdYlGn',vmin=-.4,vmax=.1,aspect='auto')
    ax.set_xticks([0,1],['年度更新','季度更新'])
    ax.set_yticks([0,1,2],['扩展历史','最近3年','最近5年'])
    for i in range(3):
        for j in range(2):
            ax.text(j,i,pct(values[i,j]),ha='center',va='center',fontsize=16,fontweight='bold')
    ax.set_title('只改变训练窗口与更新频率：累计收益')
    ax=axes[1,0]
    show=['b_exec_annual','c_econ_annual','d_multi_exp_annual','e_adaptive_annual',chosen,'i_multi_3y_quarter','j_multi_5y_quarter']
    vals=metric.loc[show,'mse_skill_vs_zero'].to_numpy()
    ax.barh(np.arange(len(show)),vals,color=['#087E8B' if m==chosen else '#94A3B8' for m in show])
    ax.set_yticks(np.arange(len(show)),['B 旧表示／持仓标签','C 经济特征','D 固定多尺度／年度','E 经济自适应','F 固定多尺度／季度','I 最近3年／季度','J 最近5年／季度'])
    ax.invert_yaxis()
    ax.axvline(0,color='#555555',lw=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title='收益均方误差改善率：所有方案仍低于零基准',xlabel='相对恒定零收益预测；正值才表示改善')
    ax=axes[1,1]
    threshold_methods=[chosen,'selected_threshold_10bps','selected_threshold_20bps','selected_threshold_40bps']
    x=np.arange(4)
    bars=ax.bar(x,cost.loc[threshold_methods,'total_return'],color=['#087E8B','#94A3B8','#94A3B8','#B4603A'],width=.55)
    ax.axhline(0,color='#999999',lw=.8)
    for i,m in enumerate(threshold_methods):
        val=cost.loc[m,'total_return']
        ax.text(i,val+(.008 if val>=0 else -.015),pct(val),ha='center',va='bottom' if val>=0 else 'top',fontsize=10)
    ax.set_xticks(x,[f'{bps}基点\n持仓{cost.loc[m,"exposure"]:.1%}' for bps,m in zip([0,10,20,40],threshold_methods)])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title='预先指定的入场阈值：盈利伴随大幅降低敞口',ylabel='累计收益',ylim=(-.33,.09))
    fig.suptitle('第二轮开发验证 | 272个持仓区间 | 2021-2026.08',fontsize=17,fontweight='bold')
    fig.savefig(OUT/'round2_overview.png',dpi=180)
    plt.close(fig)

    rows=[]
    order=list(NAMES)[:12]+['v1_adaptive_normalized','v1_fixed_10','always_up','cash']
    for m in order:
        r=metric.loc[m];c=cost.loc[m]
        rows.append([NAMES[m],pct(r.accuracy),pct(r.balanced_accuracy),pct(r.return_rmse,4),pct(c.total_return),pct(c.max_drawdown),pct(c.exposure)])
    results_table=table(['方案','方向准确率','平衡准确率','持仓收益RMSE','累计净收益','最大回撤','持仓时间'],rows)
    threshold_rows=[]
    for bps,m in zip([0,10,20,40],threshold_methods):
        g=pred[(pred.method==m)&(pred.position==1)]
        threshold_rows.append([bps,len(g),pct((g.actual>0).mean()),pct(cost.loc[m,'exposure']),pct(cost.loc[m,'total_return']),pct(cost.loc[m,'max_drawdown'])])
    threshold_table=table(['入场阈值（基点）','持仓周度区间数','持仓区间上涨比例','持仓时间','累计净收益','最大回撤'],threshold_rows)
    cost_table=table(['单边成本（基点）','F累计收益','F＋40基点阈值累计收益','持有指数累计收益'],
                     [[bps,*[pct(costs[(costs.method==m)&(costs.cost_bps==bps)].total_return.iloc[0]) for m in [chosen,'selected_threshold_40bps','always_up']]] for bps in [0,5,10,20]])
    annual_rows=[]
    for y in sorted(year.year.unique()):
        g=year[year.year==y].set_index('method')
        annual_rows.append([y,int(g.loc[chosen,'n']),pct(g.loc[chosen,'accuracy']),pct(g.loc['v1_adaptive_normalized','accuracy']),pct(g.loc['always_up','accuracy']),f'{g.loc[chosen,"return_correlation"]:.3f}'])
    annual_table=table(['年份','周度区间数','F准确率','上一轮自适应，同标签','始终看涨','F收益相关系数'],annual_rows)
    comparisons=json.loads((OUT/'paired_comparisons.json').read_text())
    pair_table=table(['受控比较','RMSE变化（基点，负值好）','95%区间（基点）','MSE差异Holm校正p值'],
                     [[f'{x["method"].split("_")[0].upper()} 对 {x["reference"].split("_")[0].upper()}',
                       f'{x["rmse"]["difference"]*10000:+.3f}',
                       f'[{x["rmse"]["ci95_low"]*10000:+.3f}, {x["rmse"]["ci95_high"]*10000:+.3f}]',
                       f'{x["mse"]["holm_adjusted_p"]:.4f}'] for x in comparisons])
    validation_rows=[]
    for m,s in selection['selected_lambdas'].items():
        validation_rows.append([NAMES[m],s['lam'],pct(s['return_rmse'],4),'log loss' if s['model']=='logistic' else '持仓收益RMSE'])
    validation_table=table(['方案','选定惩罚系数λ','验证期收益RMSE','选参依据'],validation_rows)
    audit=json.loads((OUT/'verification.json').read_text())
    g=pred[pred.method==chosen]
    chosen_r=metric.loc[chosen];chosen_c=cost.loc[chosen]
    chosen_comp=json.loads((OUT/'selected_pipeline_comparisons.json').read_text())
    old_comp=next(x for x in chosen_comp if x['reference']=='v1_adaptive_normalized')
    selected_ci=old_comp['mean_weekly_net_return_10bps']
    report=f"""# 第二轮测试：目标对齐、经济特征与滚动训练

日期：2026-09-09。完成12个建模方案、4个旧模型／常数基准、3个阈值变体，共19组、5168条预测。**这一轮改善了部分净值表现，但没有找到可以确认的稳定预测优势。**

验证集选出的方案是 **F：经济特征＋固定5/10/20日多尺度＋扩展历史＋季度更新**。在下一交易日开盘执行、单边10基点假设成本下，累计收益从上一轮自适应的 **{pct(cost.loc['v1_adaptive_normalized','total_return'])}** 改为 **{pct(chosen_c.total_return)}**，持有指数为 **{pct(cost.loc['always_up','total_return'])}**。然而F方向准确率只有 **{pct(chosen_r.accuracy)}**，收益相关系数 **{chosen_r.return_correlation:.3f}**，RMSE仍高于恒定零收益基准。净收益改善也未通过区块bootstrap检验。

所有2021—2026年结果都属于**已看过历史的开发验证**。本轮的方案与参数网格在运行前冻结，参数只由2018—2020年选定；这不能把旧区间恢复成从未使用过的最终留出集。

## 1. 测试协议与可比较性

- 沿用上一轮腾讯OHLCV快照，未重新下载价格。主输入125根日线；异常OHLC的日历行保留，含异常的训练输入／标签区间排除。
- 保持上一轮的272个周五信号日：2021-01-08至2026-08-21。首笔可执行开盘为2021-01-11，最后平仓开盘为2026-08-31。
- 新标签为“下一开盘到下一次周五信号之后开盘”的简单收益，跨节假日允许持仓天数变化。训练使用所有工作日，对每一天建立“本日之后开盘到下次同星期几之后开盘”的类似周度标签；测试只取周五。
- A和B训练样本严格一致，须同时满足旧收盘标签与新持仓标签已完全实现。训练截止日之前无法完成的标签不进入训练。最近3年／5年窗口按**样本起点日期**筛选，输入125日可以伸到训练窗口起点之前，但不能越过当前预测时点。
- 每个折的标准化参数只取训练行。P0只有旧表示和经济自适应方案需要：验证期用2017年底以前的已保存P0，2021年后用上一轮对应年度P0；其他模型不使用P0。
- 所有回归方案比较λ∈{{0.001,0.01,0.1,1,10}}，目标为平均平方损失加λ乘系数平方和，随样本数缩放，保证短窗口的惩罚强度可比。10个ridge方案都选了λ=10，即网格最大值；它提示模型倾向很强的收缩，**不能据此声称已经找到最优惩罚系数**。本轮没有看结果后扩大网格。
- 概率模型用L2 logistic regression，按验证log loss选λ，仓位取上涨概率>0.5；其收益幅度预测用概率乘训练期条件均值，只是另一个诊断输出，不与概率阈值混淆。

**准确率口径发生了变化。** 上轮51.47%针对五日收盘收益；本轮统一对实际开盘持仓收益评分。原有自适应模型不改预测和仓位，按新标签准确率为53.31%。因此应将F的51.10%与53.31%比较，不能把两个不同标签下的数字直接当成改进。两个标签有27个区间方向不同。

原版多输出ridge各输出独立拟合；A/B的变化是价格标准化目标改为直接收益及持仓标签对齐，不能解释成“删掉其他输出释放了模型容量”。上一轮导出结果是外部参考；A和B才是仅改变标签的严格对照。

## 2. 表示与训练方式

| 方案 | 修改内容 | 特征维数 |
|---|---|---|
| A → B | 旧自适应表示保持相同，收盘收益改为实际持仓收益 | 700 |
| B → C | 改成经济特征：收益、开盘跳空、实体、振幅、上下影线、相对量、收盘位置 | 75 |
| C → D | 在经济特征上加入固定5/10/20日多尺度片段统计 | 255 |
| C → E | 在经济特征上加入自适应片段统计与时长／位置／有效位 | 250 |
| D/F/G/H/I/J | 同一固定多尺度表示；扩展／3年／5年 × 年度／季度更新 | 255 |
| K/L | 概率预测：扩展年度、5年季度 | 255 |

经济基底由最近5根K线的8个通道，以及5/10/20/60/125日的7个统计组成。片段额外统计为收益和、收益标准差、平均振幅、平均相对成交量。每次仅使用125日窗口内部数据；第一根无前收盘，收益和跳空设为0，量均值只用已知前缀，不回填未来。经济自适应的插值采样改成统计聚合，仍保留时长和实际结束位置。

## 3. 全部开发验证结果

下表全部按同一持仓收益标签评分、同一执行起止时间核算，成本为每次单边仓位变化10基点（0.10%）。平衡准确率平均上涨与非上涨两类召回率；恒定空仓的52.94%只是样本中非上涨比例，平衡准确率为50%。

{results_table}

F仅在其他被测ridge经济表示候选中由验证RMSE选出。验证期F的RMSE为 **{pct(influence['validation_selected_rmse'],6)}**，恒定零收益为 **{pct(validation_zero,6)}**，F甚至没有优于这个弱基准。因此“验证集选中”只表示它是候选模型中选出的方案，不能解释成已经通过了有效性门槛。

F相对上一轮自适应，平均每个周度区间的净收益差为 **{selected_ci['difference']*10000:+.2f}基点**，8观测区块bootstrap的95%区间为 **[{selected_ci['ci95_low']*10000:+.2f}，{selected_ci['ci95_high']*10000:+.2f}]基点**，p={selected_ci['p']:.4f}。它不是总累计收益差的置信区间；两者不能混用。

![第二轮结果总览](round2_overview.png)

## 4. 哪些改法得到了支持

**标签对齐：完成了任务口径修正，未带来明确统计增益。** B相对A的RMSE略增，两个方案方向准确率相同，净值仅小幅不同。后续研究仍应沿用可执行持仓标签，理由是任务一致，而非本轮证明它提高了预测能力。

**经济特征：误差有改善迹象，强度有限。** C相对B的RMSE降低，未校正区间为负，但在10项预设比较做Holm校正后p约0.221。C几乎一直持仓（97.58%时间），其净值接近持有指数。E相对C/D的改善也不明确，不能声称自适应分段已证实有效。

**滚动窗口：没有支持最近3年／5年更好。** 在相同多尺度特征下，短窗口并没有系统性改善误差与收益，5年季度方案表现更差。F与D的季度／年度差异在收益误差上极小。

**季度净值改善集中在少数区间。** F与D仅有 **{influence['differing_position_intervals']}个**周度仓位不同。最大的正向净对数收益差来自 **{top}** 信号对应持仓区间，贡献{influence['largest_log_return_contribution']:.4f}；全部区间合计差为{influence['total_log_return_gap']:.4f}。扣除这一个区间的贡献，剩余差为{influence['gap_excluding_largest_interval']:.4f}，已经转负。这只是对固定预测结果做贡献分解，未删样本重训，也未据此改变规则；它说明净值改善对单段行情敏感。

**概率模型：本轮未优于收益回归。** K/L的持仓方向准确率分别为49.63%和46.32%，没有显示概率输出本身就能解决问题。也未进行测试期校准或搜索新的概率阈值。

预先规定的10组ridge比较如下，RMSE变化以收益基点表示，负数才是改善。区块bootstrap为8个周度观测、10000次，MSE差异p值做Holm多重比较校正。区间为各比较的未校正95%区间，因此有时区间不跨0而校正p值仍大于0.05；本轮都属于探索性开发证据。

{pair_table}

## 5. 交易阈值：不能把空仓判对算成预测突破

在验证集选出的F上，预先规定0/10/20/40基点阈值：预期持仓收益超过阈值才持仓，其余空仓。没有从2021年后的表现中选定最终阈值。0阈值就是F，10/20/40阈值都保留。

{threshold_table}

40基点方案累计盈利1.94%，但仅在39个周度区间持仓，实际持仓时间14.20%。39段中22段上涨（56.41%）；以所有272段的“仓位是否对应上涨”计算得到54.78%，该数字受大量空仓影响，**不能称为模型原始方向准确率提升到54.78%**。实际连续开平仓是15个往返（30次单边变化），比39个有敞口区间更少。该方案年化收益仅0.34%，开盘净值最大回撤12.64%，Sharpe约0.084。

{cost_table}

成本提高到单边20基点时，40基点阈值方案转为亏损1.08%。10和20基点入场阈值都较差，没有呈现稳定的阈值附近表现；本轮不将40基点选成最终策略。

以上是价格指数代理，仓位仅为指数多头或现金，现金利息为0；成本为假设综合摩擦，不代表ETF／期货实际费用。没有模拟ETF分红、跟踪误差、期货基差或成交冲击。净值按下一开盘到下一开盘记账，回撤不是盘中回撤。

## 6. 年份表现与验证选参

{annual_table}

每年区间数量仍然很少。总体相关系数接近0，没有多个年份持续显著的预测表现。验证期共有141个完整周度标签，2018—2020按对应年度／季度时点重新训练，用完整验证预测路径选择λ。

{validation_table}

尽管K的验证收益RMSE略低，K按分类损失选择，预先定义的最终ridge方案选择不包含概率模型；没有在开发期看到结果后把两类模型合并重新挑选。

## 7. 文件、验证与结论

冻结规则：[protocol.json](../protocol.json)。全部结果都在本次 `research_v2/`，上一轮的 **{audit['old_files_preserved']}份**源码／数据／结果文件经SHA-256对比未改变。

- [predictions.csv](predictions.csv)：19组×272次，包含预测、实际持仓收益、概率、仓位、入场／退出日索引、训练截止日。
- [selection.json](selection.json)、[validation_predictions.csv](validation_predictions.csv)：验证选择及全部参数候选预测。
- [folds.json](folds.json)：212个训练折，包含窗口起点、最后已完成标签、测试起止日。
- [paired_comparisons.json](paired_comparisons.json)、[selected_pipeline_comparisons.json](selected_pipeline_comparisons.json)：完整统计。
- [cost_sensitivity.csv](cost_sensitivity.csv)、[weekly_net_returns_10bps.csv](weekly_net_returns_10bps.csv)：成本与周度记账。
- [verification.json](verification.json)、[test_results.txt](test_results.txt)：验证通过。8项单元检查、5168条持仓标签独立重算、所有训练截止日和滚动窗口边界核对；逐日净值与周度复利一致，上一轮模型原始净值逐成本场景对账一致。
- [influence_diagnostics.json](influence_diagnostics.json)：季度／年度差异贡献与零预测验证基准，属于结果诊断。

**本轮完成了授权的目标、特征、训练窗口／频率测试，并附带概率与阈值对照。结果不支持宣称找到稳定有效的方法。** 标签对齐与经济特征可以保留为下一步实验基础；缩短历史和季度更新的收益优势没有得到可靠证明。40基点阈值的小额盈利应保留为待验证现象。

这轮没有训练Kronos、Crossformer、树模型或时序神经网络。非线性／预训练组合仍是后续独立实验；需要重新冻结范围、训练边界和新的验证安排，不能将这次已看过的开发结果当作其最终留出测试。
"""
    (OUT/'第二轮测试报告.md').write_text(report,encoding='utf-8')
    files=[*ROOT.glob('*.py'),ROOT/'protocol.json',*OUT.glob('*.csv'),*OUT.glob('*.json'),*OUT.glob('*.png'),OUT/'第二轮测试报告.md',OUT/'test_results.txt']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='delivery_manifest.json'}
    (OUT/'delivery_manifest.json').write_text(json.dumps(hashes,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Report:',OUT/'第二轮测试报告.md')
    print(json.dumps(influence,indent=2))


if __name__=='__main__':
    main()

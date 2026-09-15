"""Produce a report and static plots from the completed, verified experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
LABELS={'native_mse':'原始 MSE','learned_clip':'学习特征：上一轮','learned_market':'学习特征＋四项市场条件',
    'raw25_clip':'简单特征：上一轮','raw_trend':'简单特征＋趋势得分','market4':'仅四项市场特征','training_frequency':'训练频率',
    'learned_probe':'未截尾学习特征','raw25_probe':'未截尾简单特征','neutral_50':'固定 50%'}
WINDOWS={'early_2015_2017':'2015—2017','late_2018_2020':'2018—2020','pooled_2015_2020':'2015—2020 合并'}
MAIN=['native_mse','learned_clip','learned_market','raw25_clip','raw_trend','market4','training_frequency']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(v):return '—' if pd.isna(v) else f'{v*100:.2f}%'
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def savefig(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)

def figures(metrics,yearly):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13.8,7.1),layout='constrained')
    for col,w in enumerate(['early_2015_2017','late_2018_2020']):
        t=metrics[metrics.window.eq(w)].set_index('method')
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col];scale=100 if key=='accuracy' else 1
            for xs,methods,color in [([0,1],['learned_clip','learned_market'],'#246c9e'),([3,4],['raw25_clip','raw_trend'],'#bd722c'),([6],['market4'],'#687582')]:
                ys=[t.loc[m,key]*scale for m in methods];ax.plot(xs,ys,'o-',lw=1.8,color=color,markersize=7)
                for x,y in zip(xs,ys):ax.annotate(f'{y:.2f}%' if key=='accuracy' else f'{y:.4f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center',color=color)
            ax.axhline(t.loc['training_frequency',key]*scale,color='#477443',ls='--',lw=1.2,label='训练频率')
            if key=='accuracy':ax.axhline(t.loc['native_mse',key]*scale,color='#845a87',ls=':',lw=1.5,label='原始 MSE')
            ax.set_xticks([0,1,3,4,6],['学习：前','学习：加市场','简单：前','简单：加趋势','仅市场'],fontsize=9)
            ax.set_xlim(-.65,6.7);ax.set_ylim((38,68) if key=='accuracy' else (.231,.301));ax.grid(axis='y',alpha=.15)
            ax.set_ylabel('方向准确率（%）' if key=='accuracy' else 'Brier（越低越好）')
            if row==0:ax.set_title(f"{WINDOWS[w]}，n={int(t.loc['native_mse','n'])}",fontweight='bold')
    axes[0,0].legend(loc='upper left',ncols=2,frameon=False,fontsize=9)
    fig.suptitle('连续市场条件：前后比较；“前”指第十七轮截尾模型',fontsize=15)
    savefig(fig,'market_context_comparison')
    fig,axes=plt.subplots(1,2,figsize=(13.5,4.9),layout='constrained');x=np.arange(6);years=list(range(2015,2021))
    for method,label,color in [('learned_clip','上一轮学习特征','#7d96b0'),('learned_market','学习特征＋市场条件','#246c9e'),('market4','仅市场特征','#d08839')]:
        g=yearly[yearly.method.eq(method)].sort_values('year');axes[0].plot(x,g.mean_probability*100,'o-',color=color,label=label,lw=1.6)
    g=yearly[yearly.method.eq('training_frequency')].sort_values('year');axes[0].plot(x,g.observed_up_fraction*100,'D--',color='#343a40',label='实际上涨比例',lw=1.4)
    axes[0].set(xticks=x,xticklabels=years,ylim=(36,76),ylabel='上涨概率 / 上涨比例（%）',title='年内平均概率：2017 年出现明显向下偏差')
    axes[0].legend(loc='upper left',fontsize=8.5,ncols=2,frameon=False)
    t=yearly.pivot(index='year',columns='method',values='brier')
    for offset,candidate,parent,color,label in [(-.17,'learned_market','learned_clip','#246c9e','学习特征增加市场条件'),(.17,'raw_trend','raw25_clip','#bd722c','简单特征增加趋势得分')]:
        d=(t[candidate]-t[parent]).to_numpy();axes[1].bar(x+offset,d,width=.32,color=color,label=label)
        for i,v in enumerate(d):axes[1].text(i+offset,v+(.0006 if v>=0 else -.0006),f'{v:+.4f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
    axes[1].axhline(0,color='#4f555b',lw=1);axes[1].set(xticks=x,xticklabels=years,ylim=(-.020,.025),ylabel='新增模型 Brier − 上一轮 Brier',title='逐年概率误差：零线下方表示改善')
    axes[1].legend(loc='upper left',fontsize=8.5,frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('逐年诊断；2015 年只保留 22 周，全部日期此前已被查看',fontsize=14)
    savefig(fig,'annual_probability_diagnostics')

def main():
    v=read(OUT/'verification.json');assert v['status']=='PASS';p=read(ROOT/'protocol.json');fitting=read(OUT/'training_manifest.json')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv');yearly=pd.read_csv(OUT/'yearly_metrics.csv');seed=pd.read_csv(OUT/'seed_metrics.csv');states=pd.read_csv(OUT/'state_metrics.csv')
    pairs=read(OUT/'primary_comparisons.json');assessments=read(OUT/'assessments.json');training=pd.read_csv(OUT/'training_metrics.csv')
    coefficients=pd.read_csv(OUT/'coefficients.csv');components=pd.read_csv(OUT/'component_summary.csv');duplicate=pd.read_csv(OUT/'duplicate_column_verification.csv')
    figures(metrics,yearly)
    lines=['# 第十八轮方法测试：连续市场特征的加性条件输入','',
        '本轮完成 30 个新分类头、1,305 条新概率预测和全部独立复核。**直接把连续市场特征加入分类头，没有获得稳定的跨时期改善。** 学习特征模型在 2018—2020 年多预测对 1 周，但在 2015—2017 年少预测对 4 周；简单特征模型后期多预测对 2 周，早期少预测对 1 周。仅使用四项市场特征的小模型也没有稳定超过训练频率参照。','',
        '市场状态可以被因果识别，但这不保证它们能以固定的加性关系预测未来涨跌。本轮不采用新增版本替换研究基线。所有 261 个测试日期此前都已经被查看，因此仍是探索性结果，**不能作为独立确认**。','',
        '## 1. 本轮只检验一个明确的问题','',
        '在保留上一轮特征处理、标签、年度截止日和分类阈值的前提下，增加少量连续市场条件，能否改善原有预测？四项条件沿用第十七轮定义：60 日趋势得分、20 日波动、20 日平均振幅和 20 日量能变化。它们全部来自信号日收盘时已知的 OHLCV 历史，不引入新的外部数据。','',
        table(['版本','上一轮表示维数','新增市场维数','斜率总数','拟合数'],[
            ['学习特征＋四项市场条件',25,4,29,18],['简单特征＋趋势得分',25,1,26,6],['仅四项市场特征',0,4,4,6]]),'',
        '三个版本均额外含一个不惩罚的截距。使用相同的均值二元对数损失，加 λ/2 倍斜率平方和，λ=0.01；梯度无穷范数收敛阈值为 1e-9。学习特征按原有三个种子平均概率，另外两个版本每年仅一个确定性分类头。所有版本严格以 p>0.5 预测上涨。没有搜索种子、阈值、正则强度或指标组合。','',
        '模型形式为“表示项＋市场项＋截距”的加性 logit。表示项在所有市场条件下共用相同权重，没有市场与表示的乘积交互，也没有六套状态专家或按测试结果切换模型。这个实验检验的是共享分类头中的加性条件，不覆盖所有可能的状态建模方法。','',
        '## 2. 为什么简单特征只增加一项','',
        '原有 25 个简单特征已经包含 20 日波动、振幅和量能变化。重复拼接会使同一信号得到两个受惩罚的系数，从而改变有效的岭惩罚，不能把这种效果解释为增加了市场信息。因此，本轮在评分前固定省略这三个重复列，只增加趋势得分。','',
        table(['市场条件','已有简单特征位置（从 0 开始）','本轮处理'],[
            ['20 日波动',11,'不重复加入'],['20 日平均振幅',13,'不重复加入'],['20 日量能变化',14,'不重复加入'],['60 日趋势得分','由已有 60 日收益、波动构成非线性比值','增加一维']]),'',
        f"六个年度的训练与测试重复列均已核对，原始数值最大差为 {max(duplicate.training_raw_error.max(),duplicate.validation_raw_error.max()):.3e}，标准化数值最大差为 {max(duplicate.training_standardized_error.max(),duplicate.validation_standardized_error.max()):.3e}。标准化中的微小浮点差异不改变这三个变量在定义上完全重复的事实。",'',
        '四项市场条件使用每个年度训练输入的第 1、99 百分位数截尾，再用截尾后的训练均值、总体标准差标准化，标准差下限为 1e-6。下一年度沿用冻结参数。原有 25 维表示的截尾与标准化完全沿用第十七轮缓存。新增分类头共同重拟合所有系数，原有表示本身不变。','',
        '市场条件与学习表示的维数不相同，新增版本也比父模型多了参数。本轮比较的是这些固定扩展方案的实用增益，不是等参数容量的架构因果比较。','',
        '## 3. 数据与冻结顺序','',
        table(['训练截止日','训练行数','下一年测试周数'],[[f['cutoff'],f['train_n'],f['test_n']] for f in p['folds']]),'',
        '沿用沪深 300 原始数据、125 根历史输入、无效 OHLC 排除规则及既有周收益标签。训练行的联合标签完成时间必须不晚于年度截止日；测试为下一年度符合原规则的周五信号，执行标签从下一交易时段开盘开始。2015 年受原有排除规则影响仅有 22 周，其年度统计不代表完整年份。早期为 120 周，后期为 141 周。','',
        f"30 个分类头全部收敛后才开始新的测试评分，合计 {fitting['new_newton_iterations']} 次 Newton 更新。沿用原 18 个神经网络对应的冻结表示缓存，没有新的神经网络训练或前向推理；其提取过程在第十七轮已回放复核，本轮通过哈希保持该证据链。",'',
        '## 4. 主要结果','',
        '表中的“上一轮”指第十七轮截尾版本。Brier、对数损失越低越好；原始 MSE 输出收益分数，不能直接当作概率计算这两项。未截尾模型和固定 50% 参照也原样保留在完整结果文件中。']
    for w in ['early_2015_2017','late_2018_2020']:
        t=metrics[metrics.window.eq(w)].set_index('method');lines += ['',f'### {WINDOWS[w]}（{int(t.iloc[0].n)} 周）','',
            table(['方法','正确周数','准确率','平衡准确率','AUROC','Brier','对数损失'],[[LABELS[m],int(t.loc[m,'correct_directions']),pct(t.loc[m,'accuracy']),pct(t.loc[m,'balanced_accuracy']),num(t.loc[m,'auroc']),num(t.loc[m,'brier']),num(t.loc[m,'log_loss'])] for m in MAIN])]
    lines += ['', '![连续市场条件前后对比](market_context_comparison.png)','',
        '学习特征模型后期准确率从 56.03% 到 56.74%，对应 79/141 到 80/141，仍低于原始 MSE 的 84/141（59.57%）。新增版本改变 7 个方向，其中 4 个由错变对、3 个由对变错。Brier 从 0.241944 到 0.241936，差异仅约 −0.0000075；对数损失从 0.676837 到 0.676994，略有变差。','',
        '早期学习特征模型改变 32 个方向，其中 18 个由对变错、14 个由错变对，准确率从 52.50% 降至 49.17%，Brier 从 0.262637 升至 0.271899。简单特征加趋势得分后，早期准确率从 44.17% 到 43.33%，后期从 53.90% 到 55.32%。','',
        '仅四项市场特征的模型，早期为 50.00%（60/120），后期为 55.32%（78/141）；两段时期的准确率均低于训练频率，Brier 均高于训练频率。其后期 AUROC 为 0.509596，合并 AUROC 为 0.503395，没有显示出稳定的单独排序能力。','',
        '合并 261 周仅作描述：原始 MSE 为 54.79%（143/261），学习特征＋市场条件为 53.26%（139/261），简单特征＋趋势为 49.81%（130/261），仅市场特征为 52.87%（138/261），训练频率为 54.41%（142/261）。不能用合并成绩替代两段时期分别验收。','',
        '## 5. 逐年与市场状态诊断','',
        table(['年份','周数']+[LABELS[m] for m in MAIN],[[year,int(g.iloc[0].n)]+[pct(g.set_index('method').loc[m,'accuracy']) for m in MAIN] for year,g in yearly.groupby('year')]),'',
        '新增学习模型的早期损失主要出现在 2017 年：准确率从 54.00% 到 44.00%，少预测对 5/50 周；2016 年多对 1/48 周，2015 年正确周数不变。新增学习模型后期净增的 1 周来自 2018 年，2019、2020 年的正确周数均未增加。','',
        '![逐年概率诊断](annual_probability_diagnostics.png)','',
        '2017 年全部保留周样本都处于该折训练中位数以下的低波动状态。学习特征模型加入市场条件后，年度平均上涨概率由 50.79% 降至 45.11%，实际上涨比例为 56.00%。仅市场特征模型的平均概率为 43.80%，50 周里只预测 1 周上涨，年度 AUROC 为 0.246753。它识别到了低波动，却把这些样本映射到了偏低的上涨概率。','',
        '按两个时期分别合并低、高波动组，结果如下。这里的分组、阈值和样本支持下限沿用上一轮，没有根据本轮结果重新划分或筛选。','',
        table(['时期','波动组','周数','上一轮学习准确率','新增学习准确率','上一轮学习 Brier','新增学习 Brier','实际上涨比例'],[
            [WINDOWS[w],label,int(g.set_index('method').loc['learned_market','n']),pct(g.set_index('method').loc['learned_clip','accuracy']),pct(g.set_index('method').loc['learned_market','accuracy']),
                num(g.set_index('method').loc['learned_clip','brier']),num(g.set_index('method').loc['learned_market','brier']),pct(g.set_index('method').loc['learned_market','observed_up_fraction'])]
            for w in ['early_2015_2017','late_2018_2020'] for state,label in [('low_vol','低波动'),('high_vol','高波动')]
            for g in [states[states.window.eq(w)&states.partition.eq('volatility')&states.state.eq(state)]]]),'',
        '所有六状态交叉分组及振幅、量能二分组均在 [state_metrics.csv](state_metrics.csv)。原有 12 个“时期 × 六状态”单元仍有 5 个不足 20 周，不能据这些小组的高准确率挑选专家。状态结果只作描述，没有分组显著性检验或状态选优后的合成预测。','',
        '## 6. 条件项确实被拟合，但训练改善没有稳定迁移','',
        '每个扩展模型把新增市场系数设为零时，都能精确嵌套对应父模型：表示、截距及正则目标不变。24 个扩展分类头收敛后的训练正则目标均不高于父模型。这个检查确认新特征被有效优化，不能把测试不稳定简单归结为未收敛；训练目标降低本身也不是泛化证据。','',
        table(['版本','训练截止日','训练准确率','训练对数损失','训练正则目标','父模型/常数目标'],[
            [LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.log_loss.mean()),num(g.objective.mean()),num(g.parent_or_constant_objective.mean())]
            for (method,cutoff),g in training.groupby(['method','cutoff'])]),'',
        '学习模型各年度市场项的系数均值及测试期 logit 贡献如下。系数以各年度标准化坐标为单位，取三个种子的均值；市场变量与表示可能相关，系数符号不能作为因果关系或变量重要性。','']
    coef=coefficients[coefficients.method.eq('learned_market')&coefficients.block.eq('market')].groupby(['cutoff','name']).coefficient.mean().unstack('name')
    comp=components[components.method.eq('learned_market')&components.split.eq('validation')].groupby('cutoff').mean(numeric_only=True)
    lines += [table(['测试年','趋势系数','波动系数','振幅系数','量能系数','市场项平均 logit','完整平均 logit'],[
        [int(cutoff[:4])+1,num(r['market_trend60']),num(r['market_volatility20']),num(r['market_range20']),num(r['market_volume_change20']),num(comp.loc[cutoff,'market_mean']),num(comp.loc[cutoff,'total_logit_mean'])] for cutoff,r in coef.iterrows()]),'',
        '在 2017 年三个学习模型的均值中，市场项贡献 −0.276859，表示项为 +0.028576，截距为 +0.029271，合计平均 logit 为 −0.219012。该等式是固定模型的代数核算；新旧模型的表示权重也重新拟合，不能把整个性能差异都归因于这一市场项。平均 logit 也不等于平均概率的 logit。','',
        '学习模型后期三个种子的方向准确率均为 56.03%，但 Brier 分别为 0.248522、0.256863、0.246597；概率质量仍有差异。平均概率集成后的 56.74% 不等于种子准确率的均值，因为集成是在阈值判断之前完成。完整结果见 [seed_metrics.csv](seed_metrics.csv)。','',
        '## 7. 固定统计检验与判定','',
        '预先固定 14 项对比，两个时期各 7 项：新增模型相对父模型的 Brier、方向错误率；两个新增模型相对仅市场特征模型的 Brier；仅市场特征模型相对训练频率的 Brier。差值为候选损失减参照损失，负数更好。','',
        '沿用 8 个保留观测为一块的循环区块重采样，10,000 次、种子 20260910，两个时期分别重采样，14 个 p 值统一进行 Holm 校正。由于保留日期有缺口，区块不一定对应连续 8 个自然周。区间及 p 值不覆盖训练、设计选择和历轮重复查看这些日期带来的不确定性。','',
        table(['时期','候选 / 参照','指标','差值','95% 区间','原始 p','Holm p'],[[WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],
            'Brier' if r['metric']=='brier' else '方向错误率',f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"] for r in pairs]),'',
        '本轮没有任何差异通过 Holm 校正。最小校正 p 为 0.2324，对应仅市场特征模型在早期的 Brier 劣于训练频率，并非改善。后期学习模型相对父模型的 Brier 差异约为 −0.0000075，原始 p=0.9965。','',
        '新增模型按九项严格条件验收：准确率高于父模型、原始 MSE、训练频率；Brier 低于父模型、仅市场特征、训练频率；对数损失低于训练频率；至少两年准确率高于 MSE，至少两年 Brier 低于训练频率。两个时期都须九项全通过，平局不算通过。','',
        table(['时期','新增模型','通过项数','跨时期通过'],[[WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/9",'是' if a['cross_period_descriptive_pass'] else '否'] for a in assessments]),'',
        '## 8. 结论与下一步方向','',
        '当前证据不支持采用这套直接拼接市场条件的加性方案，也不支持用四项市场特征的小模型替换已有参照。更早时期的成绩说明，后期的一两周改善不足以支持稳定提升的结论。','',
        '如果继续研究，优先检验一个范围有限的新问题：**波动状态是否改变原有信号的有效强度**。可以预先限定单一波动条件的正则化交互，保留当前加性模型和原始参照，检查它能否同时控制早期退化和后期误差。该交互没有在本轮测试，不能把它当作已经有效的改进。','',
        '这条后续路线需要把“状态直接预测涨跌”与“状态调节已有信号”分开检验。现有状态识别和因果标准化可以保留为研究工具。所有本轮日期都已被用于研究设计，进一步历史实验仍只能筛选假设，最终应通过未用于选型的数据或前瞻预测检验确认。','',
        '## 9. 独立复核与交付','',
        table(['项目','结果'],[
            ['新分类头 / 独立 L-BFGS-B 解','30 / 30'],['Newton 更新',fitting['new_newton_iterations']],['斜率与截距系数合计','732'],
            ['新神经网络训练 / 前向推理','0 / 0；沿用第十七轮 18 个网络回放证据'],['合并训练特征行','47,325'],['市场条件训练行 / 测试日期','9,465 / 261'],
            ['重复变量逐折核对','18 组，均省略追加'],['新概率 / 沿用模型记录','1,305 / 2,871'],['模型记录 / 集成记录','4,176 / 2,610'],['指标组 / 状态指标组','366 / 240'],
            ['概率分箱 / 主要对比','90 / 14'],['最大新预测复核误差',f"{v['maximum_new_forecast_error']:.3e}"],['独立求解最大目标差',f"{v['maximum_alternate_objective_gap']:.3e}"],
            ['独立求解最大训练概率差',f"{v['maximum_alternate_probability_gap']:.3e}"],['独立求解最大梯度',f"{v['maximum_alternate_gradient_inf']:.3e}"],['历史文件原样保留','2,795 个，哈希不变'],['独立保留测试日期','0']]),'',
        '复核状态为 **PASS**：逐行市场公式与独立标量实现一致，未来价格扰动不影响既有训练输入；训练分位数由独立次序统计量重算；全部新解均由独立目标、梯度和 L-BFGS-B 复核；新预测、集成、市场项分解、各类指标与区块对比均重新核算。旧预测保持原值。训练行包含相互重叠的每日滚动标签，47,325 行不是同等数量的独立市场事件。','',
        f"协议 SHA-256：`{v['protocol_sha256']}`。",'',
        '主要证据：[冻结协议](../protocol.json)；[总体指标](ensemble_metrics.csv)；[逐年指标](yearly_metrics.csv)；[主要对比](primary_comparisons.json)；[九项判定](assessments.json)；[市场变换核对](market_transform_audit.csv)；[重复列核对](duplicate_column_verification.csv)；[系数](coefficients.csv)；[逐周 logit 分解](logit_components.csv)；[分解汇总](component_summary.csv)；[独立求解](independent_solver_verification.csv)；[完整复核](verification.json)。','',
        '两张图均保存为 PNG 和 SVG。交付后可在工作目录运行 `python research_v18/delivery18.py` 进行只读交付核验。','']
    report='\n'.join(lines);assert '\ufffd' not in report;(OUT/'第十八轮测试报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(status='RENDERED',report_characters=len(report),figures=2),ensure_ascii=True))

if __name__=='__main__':main()

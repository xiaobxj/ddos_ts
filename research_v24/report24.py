"""Build the round24 report and export figures from verified artifacts."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
sys.path.insert(0,str(ROOT.parent/'research_v23'));from report23 import pct,num,line,table
LABELS={'learned_market':'学习：加性模型','learned_vol_interaction':'学习：原波动交互','learned_order_extension':'学习：顺序项 1 倍（上轮）','learned_order_shrink4':'学习：顺序项 4 倍惩罚','learned_order_shrink16':'学习：顺序项 16 倍惩罚',
    'raw_trend':'简单：加性趋势','raw_vol_interaction':'简单：原波动交互','raw_order_extension':'简单：顺序项 1 倍（上轮）','raw_order_shrink4':'简单：顺序项 4 倍惩罚','raw_order_shrink16':'简单：顺序项 16 倍惩罚','order_only':'仅顺序指标','native_mse':'原 MSE 模型','training_frequency':'训练期上涨频率'}
MAIN=list(LABELS);WINDOWS={'early_2015_2017':'2015–2017','late_2018_2020':'2018–2020','pooled_2015_2020':'合并描述'}
FAMILIES=[['learned_vol_interaction','learned_order_extension','learned_order_shrink4','learned_order_shrink16'],['raw_vol_interaction','raw_order_extension','raw_order_shrink4','raw_order_shrink16']]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def yes(value):return '是' if value else '否'
def metric_row(r):return [LABELS[r.method],f'{r.correct_directions}/{r.n}',pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)]
def pair_row(r):return [WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"]
def target_row(r):return [LABELS[r.method],yes(r.early_accuracy_retained),yes(r.late_parent_accuracy_retained),yes(r.early_brier_retained),yes(r.late_brier_retained),yes(r.all_targets_met)]

def figures(metrics,years,path):
    fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained');colors=['#187b8e','#b17b24'];x=np.arange(3)
    for col,family in enumerate(FAMILIES):
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col]
            for wi,window in enumerate(['early_2015_2017','late_2018_2020']):
                t=metrics[metrics.window.eq(window)].set_index('method');vals=t.loc[family[1:],key].to_numpy()*(100 if row==0 else 1);parent=t.loc[family[0],key]*(100 if row==0 else 1)
                ax.plot(x,vals,marker='o',lw=2,color=colors[wi],label=WINDOWS[window]+'：顺序模型');ax.axhline(parent,ls='--',lw=1,color=colors[wi],label=WINDOWS[window]+'：原交互')
                for i,v in enumerate(vals):ax.annotate(f'{v:.2f}%' if row==0 else f'{v:.6f}',(i,v),xytext=(0,9 if wi==1 else -16),textcoords='offset points',ha='center',fontsize=9,color=colors[wi])
                ax.text(2.12,parent,f'{parent:.2f}%' if row==0 else f'{parent:.6f}',va='center',color=colors[wi],fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
            ax.set_xlim(-.2,2.72);ax.set_xticks(x,['1 倍（上轮）','4 倍','16 倍']);ax.set_xlabel('新增顺序项的惩罚倍率');ax.set_ylabel('准确率（%）' if row==0 else 'Brier 误差');ax.set_title(('学习表征' if col==0 else '简单特征')+('：方向准确率 ↑' if row==0 else '：概率误差 ↓'),loc='left',weight='bold');ax.grid(alpha=.15)
            ax.set_ylim((40,62) if row==0 else (.234,.279) if col==0 else (.242,.291))
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('统一加强顺序项收缩，未同时保住前段改善与原交互后段表现\n虚线为各时期原波动交互；两档方案完整报告，未选择最佳倍率',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'shrinkage_comparison.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    for col,family in enumerate(FAMILIES):
        ax=axes[0,col];g=path[path.source_method.eq(family[1])]
        ratios=np.column_stack([np.ones(len(g)),abs(g.coefficient_4x/g.coefficient_1x),abs(g.coefficient_16x/g.coefficient_1x)])
        for values in ratios:ax.plot(x,values,color='#7e9097',alpha=.32,lw=.8)
        ax.plot(x,ratios.mean(axis=0),marker='o',color=colors[col],lw=2,label='各头比例均值')
        for i,value in enumerate(ratios.mean(axis=0)):ax.annotate(pct(value),(i,value),xytext=(0,9),textcoords='offset points',ha='center')
        ax.set_xticks(x,['1 倍','4 倍','16 倍']);ax.set_ylim(0,1.13);ax.set_ylabel('|顺序系数| / 上轮 |顺序系数|');ax.set_title(('学习表征' if col==0 else '简单特征')+'：系数缩小，效果不按倍率等比缩小',loc='left',weight='bold');ax.legend(frameon=False);ax.grid(alpha=.15)
        ax=axes[1,col];t=years.pivot(index='year',columns='method',values='correct_directions');xp=np.arange(6)
        for j,m in enumerate(family[2:]):
            values=t[m]-t[family[1]];bars=ax.bar(xp+(j-.5)*.32,values,.32,color=colors[j],label=['4 倍','16 倍'][j]);ax.bar_label(bars,labels=[f'{int(v):+d}' for v in values],padding=3)
        ax.set_xticks(xp,range(2015,2021));ax.set_ylim(-2.7,1.8);ax.axhline(0,color='#555555',lw=.8);ax.set_ylabel('比上轮顺序模型多对的周数');ax.set_title('年度变化：'+('学习表征' if col==0 else '简单特征'),loc='left',weight='bold');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('收缩实现有效，泛化目标仍未达到\n学习分支 16 倍：2018 年多对 1 周，2019 年少对 1 周，后段净变化为 0',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'shrinkage_mechanism.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

def main():
    verification=read(OUT/'verification.json');assert verification['status']=='PASS';protocol=read(ROOT/'protocol.json');fit=read(OUT/'training_manifest.json');metrics=csv('ensemble_metrics');years=csv('yearly_metrics');path=csv('shrinkage_path');summary=csv('component_summary');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json');targets=csv('retention_targets');changes=csv('direction_changes')
    figures(metrics,years,path)
    facts=dict(significant_improvements=sum(r['difference']<0 and r['holm_adjusted_p']<.05 for r in pairs),significant_deteriorations=sum(r['difference']>0 and r['holm_adjusted_p']<.05 for r in pairs),retention_target_passes=int(targets.all_targets_met.sum()),newton_updates=fit['new_newton_iterations'],total_primary_fit_executions=verification['total_round_primary_fit_executions'],attempt_forecasts_identical=verification['attempt_forecasts_identical'],coefficient_ratios={})
    for method,g in path.groupby('source_method'):
        facts['coefficient_ratios'][method]={str(factor):dict(min=float(abs(g[f'coefficient_{factor}x']/g.coefficient_1x).min()),mean=float(abs(g[f'coefficient_{factor}x']/g.coefficient_1x).mean()),max=float(abs(g[f'coefficient_{factor}x']/g.coefficient_1x).max())) for factor in [4,16]}
    (OUT/'report_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    chunks=['# 第二十四轮方法测试：仅加强新增顺序项的收缩','2026-09-12｜沪深 300｜4 倍、16 倍固定惩罚｜48 个最终分类头｜年度向前检验',
        '**本轮没有达到“保住前段改善、恢复原交互后段表现”的目标。** 学习分支的前段准确率由上轮 61/120（50.83%），变为 4 倍的 60/120（50.00%）和 16 倍的 59/120（49.17%）。两档后段均为 81/141（57.45%），仍低于原交互 82/141（58.16%）。',
        '新增系数确实缩小：学习分支 4 倍惩罚后的系数绝对值平均约为上轮的 87.51%，16 倍约为 58.51%。更强惩罚没有使系数等比缩小，也没有带来一致的评分改善。原交互、上轮顺序项以及更早路径效率的研究记录均保留，本轮两档模型暂不替代它们。',
        '64 项固定比较经 Holm 校正后，没有显著改善或恶化。四个候选均未满足跨时期描述筛选，也未满足本轮的全部四项保留目标。实现独立复算 PASS 与预测能力改善是不同结论；这些历史样本已反复使用，不能作为独立确认。',
        '## 固定方案与可归因的变化',
        '完整复用第 23 轮的 31 维学习输入或 28 维简单输入，包括原波动交互和新增顺序交互。60 日顺序持续度、监督射线、残差投影、训练均值和标准差都保持原样。只把最后一维顺序项的惩罚从 0.01 增至 0.04 或 0.16；原波动交互和所有其他斜率仍用 0.01，截距不惩罚。',
        '目标函数为：平均二元对数损失 + 0.01/2 × Σ（父模型斜率²）+ λ_order/2 × 顺序斜率²。所有系数一起重拟合，包括原波动交互。设顺序系数为零时精确包含第 19 轮原模型；无限大顺序惩罚的极限才是移除该项。本轮没有锁定父模型系数、没有预测后混合，也没有按年份启用或停用。',
        '实现将原顺序特征 h 除以 √m，再运行已有的 0.01 岭逻辑回归；输出系数 θ_order 还原为 β_order=θ_order/√m。因而 θ_order²=mβ_order²，等价于所需的差异惩罚。缩小后的特征没有再做标准化，否则会抵消这一操作。原波动交互坐标没有缩放。',
        '独立求解器直接使用未缩放输入和逐坐标惩罚向量；有限差分、梯度变换、目标值一致性以及六个人工数据拟合均核验通过。最终 48 个头还通过了原坐标中的独立 L-BFGS-B 解核验。',
        '模型结构和样本固定时，提高唯一顺序系数的二次惩罚，最优解的该系数绝对值应不增。24 个来源模型全部满足这一性质。但总预测也受其他重拟合系数影响，样本外准确率、Brier 或离父模型的距离不必单调。',
        '## 数据与时间顺序',
        '沿用六个年底截止折：2014–2019 年底分别训练，预测下一年。每折训练数为 1077、1190、1434、1678、1921、2165；测试周数为 22、48、50、49、46、46。前段为 2015–2017 年 120 周，后段为 2018–2020 年 141 周。“后段”不是新的 2026 年留出集。',
        '训练标签必须 joint_completed≤cutoff，保持原观察窗口、周收益目标、无效 OHLC 排除和严格 p>0.5 的判涨规则。学习分支取三个种子概率均值。没有新增数据、日期筛选、阈值选择、窗口选择或状态专家训练。两档参数都在本轮训练前固定并完整报告，没有依据年度测试结果选择一个最佳倍率。',
        '已核对 24 条 R23→R19→R18→R17→R16 来源链、缓存行号、训练截止日与标签成熟时间，并核对学习检查点哈希。原神经表征、监督射线和交互变换在对应整段训练数据上估计，属于训练内拟合。只切分最后一个分类头不能把上游变成 OOF；本轮没有进行整条流程的时间顺序 OOF 重建或 OOF 参数选择。',
        '上一轮通过的 18 个神经状态复核作为冻结证据继续使用，本轮神经前向、神经训练和 HMM 拟合次数均为零。顺序项仍是相邻涨跌符号积减去固定符号数量下的随机重排期望，忽略幅度且对符号翻转、时间反转不变；继承的 121 根日线有效条件与 125 根分类观察窗口也不变。',
        '## 主要结果',
        '准确率、平衡准确率和 AUROC 越高越好，Brier 与对数损失越低越好。原 MSE 模型不是概率输出，因此不报告其概率损失。']
    for window in protocol['windows']:
        t=metrics[metrics.window.eq(window['name'])].set_index('method').loc[MAIN].reset_index();chunks += ['### '+WINDOWS[window['name']],table(['方法','正确/样本','准确率','平衡准确率','AUROC','Brier','对数损失'],[metric_row(r) for r in t.itertuples()])]
    chunks += ['![收缩强度与预测表现](shrinkage_comparison.png)',
        '学习分支 4 倍惩罚的前段 Brier 从 0.267551 变为 0.267696，后段从 0.244092 微降至 0.244089。16 倍的两段分别为 0.268138、0.244096。两档仍较原交互的 Brier 略低，但没有同时优于上轮顺序模型；后段也均不如原加性模型的 0.241936。',
        '简单特征分支的前段准确率为 4 倍 41.67%、16 倍 42.50%，后段均为 55.32%。后段 Brier 随收缩略有下降，但仍高于原交互和训练频率。这个分支没有提供一致支持。',
        '### 各年正确周数']
    for family in FAMILIES:
        rows=[]
        for year,g in years.groupby('year'):
            t=g.set_index('method');rows.append([year,int(g.iloc[0]['n'])]+[int(t.loc[m,'correct_directions']) for m in family+['native_mse','training_frequency']])
        chunks.append(table(['年份','样本']+[LABELS[m] for m in family+['native_mse','training_frequency']],rows))
    chunks += ['学习分支相对上轮：4 倍只在 2017 年少对 1 周，后段的方向逐条完全一致；16 倍在 2017 年少对 2 周，2018 年多对 1 周、2019 年少对 1 周。16 倍确实修复了一个 2018 年判断，但同时损失一个 2019 年判断，后段净正确数未提高。',
        '### 相对原交互与上轮的方向变化',
        table(['时期','候选','对照','改变数','对→错','错→对'],[[WINDOWS[r.window],LABELS[r.method],LABELS[r.reference],r.changed,r.correct_to_wrong,r.wrong_to_correct] for r in changes[changes.reference.isin([m for family in FAMILIES for m in family[:2]])].itertuples()]),
        '## 系数与影响幅度',
        '系数统一在未缩放的第 23 轮原输入坐标中报告，不能直接比较求解器缩放坐标里的数值。学习分支按三个种子取系数均值。',
        table(['来源分支','预测年','上轮 1 倍系数','4 倍系数','16 倍系数'],[[LABELS[method],int(cutoff[:4])+1,num(g.coefficient_1x.mean()),num(g.coefficient_4x.mean()),num(g.coefficient_16x.mean())] for (method,cutoff),g in path.groupby(['source_method','cutoff'])]),
        table(['来源分支','惩罚倍率','系数绝对值比例最小','均值','最大'],[[LABELS[method],factor,pct(stats['min']),pct(stats['mean']),pct(stats['max'])] for method,values in facts['coefficient_ratios'].items() for factor,stats in values.items()]),
        '![收缩幅度与年度得失](shrinkage_mechanism.png)',
        '下表把下一年顺序 logit 的标准差，以及相对原交互／上轮顺序模型的概率变化均方根按各头取均值。它们描述改动幅度，不是风险、净收益或因果重要性；跨年度合并的均值也不是模型选择标准。',
        table(['候选','下一年顺序 logit 标准差均值','相对原交互概率变化 RMS 均值','相对上轮概率变化 RMS 均值'],[[LABELS[method],num(g.order_logit_sd.mean()),num(g.rms_probability_change_vs_parent.mean()),num(g.rms_probability_change_vs_weak.mean())] for method,g in summary[summary.split.eq('validation')].groupby('method')]),
        '## 预先固定的保留目标',
        '四项要求分别是：前段准确率不低于该分支上轮顺序模型；后段准确率不低于该分支原交互；前段和后段 Brier 各自不高于上轮。它们是描述性目标，不代表统计显著或独立确认。',
        table(['候选','保住前段准确率','恢复原交互后段准确率','保住前段 Brier','保住后段 Brier','全部满足'],[target_row(r) for r in targets.itertuples()]),
        '保留原有 11 条严格筛选：准确率超过原交互、原加性、原 MSE、训练频率；Brier 低于原交互、原加性、训练频率、仅顺序；对数损失低于训练频率；至少两年准确率超过原 MSE、至少两年 Brier 低于训练频率。相等不通过，两段都要全部满足。',
        table(['时期','候选','满足条件','跨时期通过'],[[WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",yes(a['cross_period_descriptive_pass'])] for a in assess]),
        '## 固定比较与不确定性',
        '每候选每时期相对原交互、上轮顺序模型、原加性模型，各比较 Brier 和方向错误率；另比较训练频率与仅顺序的 Brier。共 4×2×8=64 项，全部一起 Holm 校正。候选减对照损失，负值较好。循环区块为 8 个保留观测，10000 次重采样，种子 20260910，各时期单独生成；区间为边际 95% 区间，p 为双侧中心化结果。没有合并时期的主要检验。',
        table(['时期','候选 / 对照','损失','差值','95% 区间','原始 p','Holm p'],[pair_row(r) for r in pairs]),
        '这些校正没有覆盖之前多轮历史设计选择，不能把“没有显著恶化”解释为等效或非劣证明。逐状态的 26 个固定单元、33 个方法、两个时期全部保留为 1716 行诊断；少于 20 个样本继续标记稀疏，未据此筛选日期或选择倍率。',
        '## 实现核验与完整运行记录',
        f"最终交付包含 48 个分类头、1500 个分类系数、75720 行累计训练输入，执行 {fit['new_newton_iterations']} 次 Newton 更新；24 个已有顺序投影复用，没有新增投影拟合。训练在任何本轮新评分之前完成。梯度无穷范数≤10⁻⁹、Hessian 正定，48 个独立原坐标解均通过。",
        f"新预测最大复算误差 {verification['maximum_new_forecast_error']:.3e}；独立目标值最大差 {verification['maximum_alternate_objective_gap']:.3e}、训练概率最大差 {verification['maximum_alternate_probability_gap']:.3e}。另核对 48 组训练／下一年输入、2130 组汇总／年度／种子／状态指标、320 个可靠性分箱和 64 项比较。独立求解器的系数没有替代交付预测。",
        '第一遍已完成 48 个头及 2088 条新评分，但评估函数中的局部变量 flags 遮蔽同名函数，产生 UnboundLocalError，未生成评估结论。完整的 85 个文件及归档清单保存在 research_v24_attempt1。修复仅涉及变量名，并增加归档复核；模型设计、惩罚、样本和求解参数不变。随后重新冻结并完整重跑，两个版本的头记录与全部预测逐条一致。',
        '因此本轮实际执行了 96 次主要拟合（首遍 48＋最终 48），共 372 次主要 Newton 更新；重复评分不增加独立样本数。最终独立核验为 48 个解，另外的人工数据检查单列。3389 个前轮证据文件加 86 个失败运行归档文件，共 3475 个哈希保持不变。',
        '最终预测表有 2088 条本轮单头概率与 12528 条旧记录，共 14616 条；33 个方法各有 261 条集成记录，共 8613 条。训练内指标只作拟合诊断，不是样本外成绩。',
        table(['方法','截止日','训练准确率','训练 Brier','带惩罚目标'],[[LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())] for (method,cutoff),g in csv('training_metrics').groupby(['method','cutoff'])]),
        '## 研究决定与下一步',
        '保留原交互后段 58.16%，保留上轮顺序扩展的前段 50.83% 及小幅概率改善。本轮两档收缩作为完整的未达目标记录保留。它们说明统一压小顺序项的幅度不足以解决已观察到的年度取舍，不能据此否定所有状态识别方法，也不能宣称已经证明需要按某一年切换。',
        '下一步优先检验一个可分辨原因的改动：固定原交互模型的全部系数，将其 logit 作为固定基底，仅拟合一个顺序补充系数，观察后段损失来自新增项本身还是父模型系数重拟合的联动。这个实验尚未执行，固定基底也不保证方向准确率不会下降。暂不继续扩大统一惩罚的参数网格。',
        '若之后希望让模型根据市场阶段自动启用补充项，先建立整个上游流程都按历史边界估计的时间顺序校验，再用当时可获得的证据确定规则。当前的年度向前结果可以用于探索，但不足以支撑一个经独立验证的状态选择器。',
        '## 限制与文件',
        '**不能作为独立确认**：261 个日期已被反复查看，2015 年覆盖不完整，标签在日频训练样本间重叠，固定长度区块重采样只是特定假设下的近似。没有交易成本、执行或净收益测试，没有完整复现论文方法。已经使用过的 2021–2026 年记录不能重新命名为独立留出集。',
        f"本轮最终冻结协议 SHA-256：`{verification['protocol_sha256']}`。",
        '主要文件：[协议](../protocol.json)、[独立验证](verification.json)、[时间来源链](temporal_provenance.csv)、[人工等价性测试](synthetic_penalty_checks.csv)、[收缩路径](shrinkage_path.csv)、[系数](coefficients.csv)、[影响幅度](component_summary.csv)、[逐条 logit 分解](interaction_components.csv)、[集成预测](ensemble_predictions.csv)、[全部单模型预测](model_predictions.csv)、[汇总指标](ensemble_metrics.csv)、[年度指标](yearly_metrics.csv)、[种子指标](seed_metrics.csv)、[状态指标](state_metrics.csv)、[固定比较](primary_comparisons.json)、[保留目标](retention_targets.csv)、[筛选结果](assessments.json)、[方向变化](direction_changes.csv)、[运行归档](../../research_v24_attempt1/attempt_preservation.json)。',
        '图表矢量版：[表现 SVG](shrinkage_comparison.svg)、[机制 SVG](shrinkage_mechanism.svg)。复现请在独立副本及空结果目录中，保留前轮和首遍归档依赖，按 prepare24.py → contract24.py → train24.py → score24.py → evaluate24.py → verify24.py 顺序运行；计算解释器为 research_v4/.venv_gpu/Scripts/python.exe，报告绘图使用默认 Python。不得覆盖当前冻结记录。']
    report='\n\n'.join(chunks)+'\n';(OUT/'第二十四轮测试报告.md').write_text(report,encoding='utf-8')
    note='# 第二十四轮研究留存\n\n2026-09-12：新增顺序项采用固定 4 倍和 16 倍惩罚，48 个最终分类头独立核验通过。前段准确率为 50.00%／49.17%，后段均为 57.45%；未达到保住上轮前段 50.83%、恢复原交互后段 58.16% 的目标。\n\n原交互、上轮顺序模型和路径效率的研究记录继续保留，本轮不替换模型。64 项比较没有经 Holm 校正后显著改善，历史反复使用，不能作为独立确认。\n\n首遍评估变量名冲突已修复并完整归档；两遍头记录与预测完全一致。实际主要拟合执行 96 次（48＋48），新增独立日期为 0。下一步优先固定原模型系数，仅检验一个顺序补充系数的影响。\n\n[完整报告](results/第二十四轮测试报告.md)｜[协议](protocol.json)｜[验证](results/verification.json)｜[首遍归档](../research_v24_attempt1/attempt_preservation.json)\n'
    (ROOT/'README.md').write_text(note,encoding='utf-8');print(json.dumps(dict(status='BUILT_NOT_YET_VISUALLY_REVIEWED',report_characters=len(report),figures=2,facts=facts),indent=2))
if __name__=='__main__':main()

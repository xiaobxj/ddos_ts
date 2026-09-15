"""Generate the reviewable Chinese report from frozen comparison artifacts."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
POLICIES=['full','disjoint0','disjoint1','disjoint2','quarter_equal','recent3y']
NAMES={'full':'全历史等权','disjoint0':'不重叠相位0','disjoint1':'不重叠相位1','disjoint2':'不重叠相位2',
       'quarter_equal':'季度等总权重','recent3y':'最近三年'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    c=pd.read_csv(OUT/'policy_comparisons.csv').set_index('policy').loc[POLICIES]
    years=pd.read_csv(OUT/'yearly_metrics.csv');metrics=pd.read_csv(OUT/'ensemble_metrics.csv')
    seed=pd.read_csv(OUT/'seed_metrics.csv');d=pd.read_csv(OUT/'policy_definitions.csv')
    audit=pd.read_csv(OUT/'training_audit_summary.csv');loss=pd.read_csv(OUT/'training_loss_attribution.csv')
    sampling=pd.read_csv(OUT/'sampling_label_diagnostics.csv');exposure=pd.read_csv(OUT/'nominal_batch_exposure.csv')
    pairs=json.loads((OUT/'primary_comparisons.json').read_text(encoding='utf-8'))
    assessment=json.loads((OUT/'assessment.json').read_text(encoding='utf-8'))
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'));assert verification['status']=='PASS'
    interpretations=(ROOT/'interpretation8.md').read_text(encoding='utf-8')
    counts=table(['验证年','全历史','相位0','相位1','相位2','最近三年','每轮更新次数／末批大小'],[
        [int(f[:4])+1]+[int(d[(d.cutoff==f)&(d.policy==p)].train_n.iloc[0]) for p in ['full','disjoint0','disjoint1','disjoint2','recent3y']]+
        [f"{int(d[d.cutoff==f].optimizer_steps_per_epoch.iloc[0])}／{int(sampling[sampling.cutoff==f].last_batch_size.iloc[0])}"]
        for f in ['2017-12-31','2018-12-31','2019-12-31']])
    main_table=table(['训练方案','方向准确率','RMSE（bp）','MSE改善／统一均值','MSE改善／本方案均值','优于本方案均值的种子','Ridge RMSE（bp）'],[
        [NAMES[p],f'{c.loc[p,"accuracy"]*100:.2f}%',f'{c.loc[p,"rmse"]*1e4:.2f}',f'{c.loc[p,"mse_skill_vs_common_mean"]*100:+.2f}%',
         f'{c.loc[p,"mse_skill_vs_own_mean"]*100:+.2f}%',f'{int(c.loc[p,"seeds_better_than_own_mean"])}/3',f'{c.loc[p,"ridge_rmse"]*1e4:.2f}'] for p in POLICIES])
    comparison_table=table(['新方案','对照','MSE差值','RMSE差（bp）','RMSE差95%区间（bp）','原始p','Holm p'],[
        [NAMES[r['policy']],'原全历史模型' if r['reference']=='full_neural' else '本方案训练均值',f'{r["mse"]["difference"]:+.8f}',
         f'{r["rmse"]["difference"]*1e4:+.2f}',f'[{r["rmse"]["ci95_low"]*1e4:+.2f}, {r["rmse"]["ci95_high"]*1e4:+.2f}]',
         f'{r["mse"]["p"]:.4f}',f'{r["mse"]["holm_adjusted_p"]:.4f}'] for r in pairs])
    yearly_table=table(['训练方案','2018／49周','2019／46周','2020／46周'],[
        [NAMES[p]]+[f'{years[(years.model=="neural")&(years.policy==p)&(years.year==y)].mse_skill_vs_full_mean.iloc[0]*100:+.2f}%'
                     for y in [2018,2019,2020]] for p in POLICIES])
    seeds_table=table(['训练方案','种子20260910','种子20260911','种子20260912'],[
        [NAMES[p]]+[f'{seed[(seed.policy==p)&(seed.seed==s)].mse_skill_vs_training_mean.iloc[0]*100:+.2f}%'
                     for s in [20260910,20260911,20260912]] for p in POLICIES])
    diagnostics=table(['训练方案','折内预测标准差（bp）','输入整窗错位后的MSE变化'],[
        [NAMES[p],f'{metrics[(metrics.model=="neural")&(metrics.policy==p)].within_fold_forecast_std.iloc[0]*1e4:.2f}',
         f'{metrics[(metrics.model=="neural")&(metrics.policy==p)].reassignment_mse_relative_change.iloc[0]*100:+.2f}%'] for p in POLICIES])
    a2015=loss[(loss.grouping=='year')&(loss.period.astype(str)=='2015')]
    a2017=loss[(loss.grouping=='year')&(loss.period.astype(str)=='2017')]
    disjoint=sampling[sampling.policy.str.startswith('disjoint')]
    curves=pd.concat([pd.read_csv(OUT/f'worker{w}_training_curves.csv') for w in [0,1]])
    end=curves[curves.epoch==20]
    training_table=table(['新方案','保留样本上的最终标准化收益MSE','标准化辅助MSE'],[
        [NAMES[p],f'{end[end.policy==p].final_model_train_return_mse.mean():.4f}',
         f'{end[end.policy==p].final_model_train_auxiliary_mse.mean():.4f}'] for p in POLICIES[1:]])
    report=f'''# 第八轮方法测试：标签重叠、时期权重与采样对照

实验日期：2026-09-11。验证对象为沪深300原始 OHLCV 的固定 Crossformer 配置。全历史模型仅作为固定实验对照；上一轮已经发现它的历史稳定性不足。本轮正式完成45次神经网络训练、18次线性拟合，复用9个全历史模型；全部54个神经网络检查点和18个线性模型均已重放核验。

{interpretations}

## 1. 本轮比较范围

继续使用此前冻结的3780个有效样本，125根已知K线、5维OHLCV、25×5分段；信号时点之后的下一开盘价进场、下一同星期锚点之后的开盘价退出，收益标签与未来五根OHLCV辅助标签共同满足截止日成熟条件。训练量分别为1678、1921、2165；验证仍为2018、2019、2020年共141个星期五样本，年度数量49／46／46。

这些验证日期已在前几轮反复查看，本轮仍是现有数据上的探索性比较。论文完整复现所需的78指数数据、模式划分及全部训练细节仍未补齐。多轮筛选后的最高准确率需要新的独立证据才能确认。

固定模型为 combined20：38,551参数，d_model16／d_ff32，dropout0.3，AdamW权重衰减0.1，学习率0.001，20轮、batch128、梯度裁剪1，三个固定种子。损失保持标准化可执行收益MSE＋0.1×标准化未来OHLCV MSE。每种方案仅用自己的保留训练标签、按自己的权重拟合均值和总体标准差。

五个新方案均保持全历史对照的优化器更新次数。样本不足时，连续生成保留样本的新随机排列，重复到全历史样本数后截断；因此去重并不等于减少训练工作量。重复样本也不会增加独立信息。

{counts}

## 2. 采样规则及线性对照

- **不重叠相位0／1／2**：按原始交易日序号筛选 `(anchor−首个全历史anchor) mod 6 = 相位`，然后按时间保留未来联合区间不重叠的样本。联合区间为 `[entry, max(exit, anchor+5)]`，两端含边界；下一条的entry必须严格晚于上一条末端。节假日导致区间变长时会跳过后续候选。三个相位的训练锚点交集均为0，但其输入窗口和经济时期仍重叠；它们不是独立数据集，也没有穷尽全部六个相位。
- **季度等总权重**：保留所有训练行，行权重为 `N/(季度数×该季度有效行数)`，每个已观测季度总权重相同；包括不完整季度。加权由日期和样本资格决定，不使用验证收益或训练损失。
- **最近三年**：保留截止年及此前两年内的成熟训练锚点，固定三年，不搜索记忆长度。更早K线仍可能出现在125根输入窗口中。
- **线性参考**：每个方案、每个年度各拟合一个Ridge，共18个。输入是同一原始窗口的625维展开值；特征标准化和截距只使用本方案训练数据，权重总和统一为全历史N，岭惩罚固定10。没有辅助目标。这是本轮未调参的线性控制，不等同于第二轮使用工程特征的Ridge，也不是声称最优的线性模型。

季度权重有一个明确局限：2015Q3在严格资格筛选后仅剩4行；等季度方案将单行权重提高到14.47–14.63倍。该季度权重占比从约0.18%–0.24%变为2.70%–3.45%。所以本轮检验的是直接使每个季度总权重相等的方法，其表现不能代表带有效覆盖约束、权重上限或其他时期权重设计。

## 3. 训练数据审计

全历史按日期相邻标签相关性为{audit.adjacent_return_correlation.min():.3f}–{audit.adjacent_return_correlation.max():.3f}；按原始标签使用的收益增量 `[entry, exit)` 统计，每根被覆盖的增量平均被约4.96–4.97条标签使用。将未来辅助OHLCV及端点也计入，覆盖的未来K线平均被约5.98条标签使用。

去重后，相邻保留标签相关性为{disjoint.adjacent_retained_return_correlation.min():.3f}至{disjoint.adjacent_retained_return_correlation.max():.3f}，联合未来区间交叠为0。相关性下降证明采样操作生效；它不证明样本独立，也不预先保证预测会更好。原方案的训练标签成熟约束仍成立，标签重叠本身不等于使用验证期未来信息。

冻结全历史模型的训练误差审计覆盖9个模型、17,292条训练预测；先计算各模型的逐行平方误差，再按种子取平均并分组汇总。误差最高约1%的行贡献{audit.top_one_percent_fitted_loss_share.min()*100:.2f}%–{audit.top_one_percent_fitted_loss_share.max()*100:.2f}%的拟合平方损失。训练均值基准的损失另行计算并保存在审计表中。

2015年训练样本占{a2015.sample_share.min()*100:.2f}%–{a2015.sample_share.max()*100:.2f}%，拟合损失占{a2015.fitted_loss_share.min()*100:.2f}%–{a2015.fitted_loss_share.max()*100:.2f}%；2017年训练样本占{a2017.sample_share.min()*100:.2f}%–{a2017.sample_share.max()*100:.2f}%，拟合损失仅占{a2017.fitted_loss_share.min()*100:.2f}%–{a2017.fitted_loss_share.max()*100:.2f}%。损失确实随时期变化，但不能说2015年独自占据了大部分训练损失。

![训练数据与误差贡献](training_audit.png)

## 4. 完整验证结果

神经网络结果为同一方案三个固定种子预测的算术平均。各相位单列，不额外跨相位集成。MSE改善定义为 `1−模型MSE/对照MSE`，正数较好。RMSE以收益基点bp表示，1bp=0.0001；它是预测误差，不能当成策略收益。统一全历史训练均值的RMSE为{c.full_mean_rmse.iloc[0]*1e4:.2f}bp。

{main_table}

方向准确率仅是诊断指标；主比较预先固定为MSE。因此准确率与MSE出现不同方向时，不能临时选择较好看的指标宣称成功。

![各采样方案完整比较](policy_results.png)

按年度、相对同年度统一全历史训练均值的MSE改善如下：

{yearly_table}

![逐年表现](yearly_performance.png)

每个种子分别相对本方案训练均值的MSE改善如下；不能只取最好的种子：

{seeds_table}

## 5. 预先固定的十项主比较

五个新方案各与原全历史模型、本方案训练均值进行配对周度MSE比较。连续8个观测的循环块自举，10,000次，固定种子20260910；中心化双侧p值，对全部十项做Holm校正。下表差值均为新方案减对照，负数表示误差较低。百分位置信区间与中心化检验并非严格互逆；判断显著性遵循预设的校正p值。

{comparison_table}

十项中，校正后p<0.05且MSE改善的比较有{assessment['primary_significant_improvements']}项。即使出现显著值，多轮使用相同日期的探索属性也不会消失。

描述性筛选要求同时满足：集成MSE低于原全历史神经模型、低于本方案训练均值、低于共同全历史训练均值，且至少2／3个种子优于本方案均值。该筛选不等于统计确认；三个去重相位必须全部满足，才能记为相位家族通过。

{table(['方案','描述性筛选'],[[NAMES[p],'通过' if assessment['policy_screens'][p] else '未通过'] for p in POLICIES[1:]])}

三个去重相位的共同筛选：**{'通过' if assessment['disjoint_family_screen'] else '未通过'}**。没有据此选取最佳相位、采样方案或种子作为新的正式策略。

## 6. 解释模型行为的辅助指标

每个年度内部将验证输入窗口循环错位一位，目标与原行标签保持不变。该操作只做推理诊断，未进入训练或方案选择；错位不是新的真实交易策略，也不是独立预测检验。

{diagnostics}

训练终点的损失按各方案自己的标签标准化及权重计算，以下为9次拟合的平均值；标准化尺度不同，不能把跨方案的训练损失差异直接解释为同一任务上的优劣。

{training_table}

继承的末批大小分别为14、1、117。特别是2019验证年度对应的训练组，每轮有15个128样本批次和1个单样本批次。它们都执行一次更新。训练前向损失的名义样本系数会因此不均；相关诊断保存在[nominal_batch_exposure.csv](nominal_batch_exposure.csv)。这不是实际梯度或AdamW参数影响的测量，尚未证明它导致了表现变化。若后续继续优化，应将末批处理作为单独的固定预算对照，不能在本轮中悄悄改变它。

## 7. 设计预检修正与可追溯性

最初的去重实现只改变日期排序样本的起点。训练掩码检查发现，三个起点在第一次资格缺口后几乎立即合流：首折的共同锚点达到277个，后两折达到318、360个，无法提供有意义的相位对照。发现依据是训练日期和掩码，不是新模型验证得分。

当时已有4个完整神经网络拟合及18个线性拟合，训练程序已自动产生验证预测文件。没有查看这些预测值或计算它们的比较成绩后再修改方案。相关文件全部移入research_v8_preflight、记录退出原因和哈希；4个神经模型及预检线性模型全部排除，正式45＋18次拟合均从头进行。不能把预检计入本轮成功复核的54个正式神经模型。

修正后通过了相位锚点交集为0、每种方案内部联合标签区间不重叠、标签成熟、权重质量、重复预算、等权损失和一轮训练状态与旧实现逐张量一致、加权Ridge正规方程等检查。训练前3项单元检查通过；训练后54个神经检查点、18个线性模型均重放通过，并重新计算900个每轮抽样顺序哈希及全部十项统计比较。

神经预测最大重放差{verification['maximum_neural_replay_error']:.3g}，线性预测最大重放差{verification['maximum_linear_replay_error']:.3g}；前七轮1,794个证据文件及预检62个文件均保持哈希一致。详细记录见[verification.json](verification.json)、[冻结协议](../protocol.json)、[预检归档清单](../preflight_archive.json)。

本轮数据未更新；正式研究验证结果未扩展到2024–2026，也没有据本轮结果修改任何交易程序。

可复核明细：[方案指标](policy_comparisons.csv)、[全部逐种子预测](all_seed_predictions.csv)、[年度指标](yearly_metrics.csv)、[统计检验](primary_comparisons.json)、[训练误差归因](training_loss_attribution.csv)、[标签与采样诊断](sampling_label_diagnostics.csv)。静态图均另存同名SVG以便导出。
'''
    path=OUT/'第八轮测试报告.md';path.write_text(report,encoding='utf-8')
    print(json.dumps(dict(report=str(path),characters=len(report),verification='PASS'),ensure_ascii=False))


if __name__=='__main__':main()

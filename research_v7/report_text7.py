"""Narrative template; findings are filled only after all frozen runs finish."""
def report_text(audit,findings,assessment,t):
    observations='\n\n'.join(findings['observations'])
    next_steps='\n'.join(f'{i}. {line}' for i,line in enumerate(findings['next_steps'],1))
    status='满足' if assessment['consistency_screen_pass'] else '不满足'
    return f'''# 第七轮方法测试：固定 20 轮候选的多种子时间块检验

{findings['headline']}

本轮补齐第六轮的检验缺口：第六轮选中的是组合模型第 20 轮，而此前删除历史块的测试只跑到第 10 轮且只有一个种子。这里固定候选，新增 **54 次拟合**，每种删除方式均使用三个种子，并保留原基准和训练均值作对照。本轮没有重新挑选模型、轮数、种子或删除位置。

依然使用 **2018—2020 年的 141 个周五信号**。目标是从信号后的下一个实际开盘，到下一同星期信号后的实际开盘之间的收益。该历史已经被多轮查看，新增拟合次数并不代表新增独立市场样本。准确率、RMSE 和 MSE 均为预测指标，本轮没有生成交易净收益结论。

## 1. 固定候选的结果

下表为每周三个种子的平均预测。完整历史一行直接复用第六轮冻结结果；新增检验是其余三行。

{t['comparison']}

三种删除历史中，候选集成优于训练均值的数量为 **{assessment['deleted_histories_better_than_mean']}/3**，优于原基准 10 轮的数量为 **{assessment['deleted_histories_better_than_baseline10']}/3**。

![固定候选的多种子稳定性](candidate_stability.png)

训练前固定的三个描述性一致性条件如下。本轮对该一致性检查的结论为 **{status}**。即使全部满足，也只是允许进一步研究的本地一致性线索，不等同于统计确认或可部署结论。

{t['flags']}

{observations}

## 2. 比较角色和数据边界

| 角色 | 设置 | 作用 |
| --- | --- | --- |
| 固定候选 | 组合模型，20 轮；d_model 16 / d_ff 32；dropout 0.3；权重衰减 0.1；38,551 个参数 | 唯一接受稳定性检验的候选 |
| 主要神经网络对照 | 原模型，10 轮；d_model 32 / d_ff 64；dropout 0.1；权重衰减 0.01；138,471 个参数 | 使用第五轮已经选定的训练长度，保留其既有运行设置 |
| 同预算参考 | 原模型，20 轮 | 检查与相同训练轮数的模型相比如何；它较易过拟合，单独优于它不足以说明候选有效 |
| 训练前缀诊断 | 组合模型，10 轮 | 检查从近常数输出到第 20 轮的变化；不参与候选选择 |
| 简单参考 | 保留的成熟训练样本均值、零收益预测 | 判断模型是否提供超过简单预测的信号 |

两种架构均实际训练到第 20 轮，从中保留第 10、20 轮检查点。主要对照比较的是各自此前选定的训练长度，因此“原基准 10 轮”与“候选 20 轮”的最终预测更新次数不同；原模型第 20 轮则提供同轮数参照。

所有模型直接调用已冻结的第六轮模型实现。输入和目标来自第五轮相同的 3,780 个合格样本：125 根原始 OHLCV 观察窗口、固定 25 × 5 分段、五维输入。共同训练目标为标准化收益 MSE ＋ 0.1 × 未来五根 K 线 OHLCV 辅助目标 MSE，保持相同标签定义、严格辅助标签过滤和零初始化收益头。未引入预训练表示或新的行情。

| 年度截止日 | 验证年 | 完整训练样本数 | 验证周数 |
| --- | --- | --- | --- |
| 2017-12-31 | 2018 | 1,678 | 49 |
| 2018-12-31 | 2019 | 1,921 | 46 |
| 2019-12-31 | 2020 | 2,165 | 46 |

每折只使用 `joint_completed ≤ cutoff` 的训练标签，验证标签统一要求在 2020-12-31 前完成。完整历史和三个删除历史使用相同验证日期。

## 3. 删除方式和计算预算

直接复用第六轮的样本掩码：把各折成熟训练锚点按时间顺序分成五段，分别删除第 1、3、5 段，约为最早、中间、最近的 20%。这是删除标签锚点，不会从剩余样本的观察窗口中完全清除相应底层行情。各折删除的实际日期区间保存在[时间块定义](fold_history_definitions.csv)。

每个名义训练轮次仍展示完整历史对应的样本数量。先无放回打乱保留样本，不足部分用新的随机排列补齐，因此部分保留样本重复出现。这保持了同折同轮数下的更新预算，属于样本组成压力测试；不能解释为普通的无权重减样本重训。收益和辅助标签的标准化只使用当前保留的成熟训练样本。

新增训练预算为 **2 种架构 × 3 种删除 × 3 个年度 × 3 个种子 = 54 次拟合**，每次 20 轮，保存两个轮数，共 108 个新检查点。三个种子固定为 20260910、20260911、20260912；AdamW 学习率 0.001、batch 128、梯度裁剪 1.0。完整历史复用既有 36 个检查点，不重新拟合。没有在看见分数后增加训练或修改设置。

两个架构用独立进程同时运行，各自拥有模型、优化器、随机状态、CUDA 上下文和输出。相同种子与历史的样本展示顺序逐轮一致。使用原有独立 GPU 环境、CUDA float32、确定性算法、关闭 TF32；没有修改环境依赖或此前代码。

## 4. 预先指定的误差比较

主要比较是三种删除历史下，固定候选分别对原基准 10 轮和训练均值，合计六项。每项在相同 141 周上比较三个种子的平均预测。采用连续 8 个验证观察值的循环区块 bootstrap，10,000 次，随机种子 20260910；中心化双侧 p，六项 MSE 比较共同进行 Holm 调整。

负的 ΔRMSE 表示候选更好，正数表示更差。完整历史参与过候选筛选，只作描述性参考；原模型 20 轮与组合模型 10 轮是诊断参照。三个删除实验共用相同验证历史，并非三个独立样本集，统计区间也不能消除多轮历史选择带来的限制。

{t['primary']}

这里的置信区间是 bootstrap 百分位区间，p 值来自中心化双侧检验；两种计算并不是同一检验的相互反演，因此临界附近可能出现区间与原始 p 值看似不同的判断。显著性结论以协议预先指定的六项 MSE Holm 调整 p 值为准，不能把未调整区间或辅助准确率比较替代为主要结论。

## 5. 单独种子和年度结果

下面首先列出每个候选种子的完整 141 周结果。集成结果不能替代单种子一致性；也不能根据此表选择表现最好的种子。

{t['seeds']}

年度结果使用同一历史条件的三种子平均预测。每年对应独立的历史训练截止日，但这些年份已在先前研究中被观察。

{t['years']}

## 6. 预测变化与近常数诊断

将同一年度验证样本的完整输入窗口循环重配一次，保持各窗口内的顺序、几何信息和掩码完整。该诊断只在模型推理时进行，不进入训练、候选筛选或稳定性门槛。重配后误差上升说明原始输入配对在该样本上有帮助，仍不能单独证明可泛化的因果关系。

{t['all']}

“折内预测 SD”移除了各年度预测均值，单位 bp 为收益率万分之一。较小的预测波动可能来自输出接近常数；它需要与预测误差一起看。

![预测变化和近常数诊断](forecast_diagnostics.png)

以下比较删除历史后与相同角色、相同三种子集成完整历史预测之间的差异。去除训练均值的版本用于分开观察均值漂移；它不能消除标准差和其他训练效应的影响。

{t['shifts']}

{findings['diagnostics']}

## 7. 实现复验

**{audit['contract_tests']} 项行为检查通过；{audit['total_checkpoints_replayed']} 个检查点全部重放通过**，其中新检查点 {audit['new_checkpoints_replayed']} 个、复用检查点 {audit['reused_checkpoints_replayed']} 个。最大预测重放差异 {audit['maximum_replay_error']:.3g}，处于 CSV 浮点存储精度范围。

第 10 轮、种子 20260910 对应的 **{audit['prior_prefix_checkpoints_with_identical_weights']} 个历史删除检查点，全部权重张量与第六轮逐项相同**；{audit['prior_prefix_predictions_matched']} 条对应预测重现，最大差异 {audit['maximum_prefix_prediction_error']:.3g}。这核对了移除未使用的第 2、5 轮评估不会改变训练结果。

独立重建了所有删除掩码、成熟标签条件、标签标准化与 **{audit['training_epoch_order_hashes_recomputed']} 个训练轮次的样本顺序哈希**，并检查成对架构使用相同采样顺序与预算。新保存 {audit['new_seed_predictions']:,} 条逐种子预测，复用 {audit['reused_seed_predictions']:,} 条，合计 6,768 条；集成后为 2,256 条角色／历史／日期记录。

此前 **{audit['prior_evidence_files_preserved']:,} 个证据文件**及本轮冻结输入、训练源码均通过哈希核对。首次行为测试曾在准备流程尚未完成时启动，因复用预测文件未生成而报错；等待准备完成后，相同测试通过，没有更改协议或断言。保留[首次调度错误说明](initial_contract_scheduling_note.json)和[原始错误输出](initial_contract_scheduling_failure.txt)。

复验通过代表实现符合本轮协议，不代表模型通过了统计有效性或市场有效性检验。

可审计材料：

- [冻结协议](../protocol.json)、[准备清单](preparation_manifest.json)、[54 次拟合计划](fit_plan.json)、[时间块定义](fold_history_definitions.csv)。
- [固定候选比较](candidate_comparisons.csv)、[一致性条件与结论](stability_assessment.json)、[六项预设比较](primary_comparisons.json)。
- [逐种子预测](all_seed_predictions.csv)、[集成预测](ensemble_predictions.csv)、[完整角色指标](ensemble_metrics.csv)、[种子指标](seed_metrics.csv)、[年度指标](yearly_metrics.csv)。
- [均值和零收益参考](baseline_metrics.json)、[预测敏感性明细](prediction_sensitivity.csv)。
- [原模型训练曲线](baseline_training_curves.csv)、[组合模型训练曲线](combined_training_curves.csv)。
- [复用检查点清单](reused_checkpoint_manifest.json)、[原模型新检查点](baseline_checkpoint_manifest.json)、[组合模型新检查点](combined_checkpoint_manifest.json)。
- [完整复验结论](verification.json)、[行为检查输出](contract_test_output.txt)。
- 矢量图：[候选稳定性](candidate_stability.svg)、[预测诊断](forecast_diagnostics.svg)。

## 8. 下一步取舍

{next_steps}
'''

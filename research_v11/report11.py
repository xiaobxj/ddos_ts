"""Assemble an auditable Chinese report from training-only artifacts."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
NAMES={'archived_mse':'原 MSE20 权重','archived_huber':'原 Huber20 权重','joint':'联合损失重启','return_only':'仅收益损失重启'}


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(str(v) for v in row)+'|' for row in rows])


def main():
    losses=pd.read_csv(OUT/'training_loss_summary.csv')
    comparisons=pd.read_csv(OUT/'comparison_summary.csv')
    gradients=pd.read_csv(OUT/'gradient_summary.csv')
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    contract=json.loads((OUT/'contract_verification.json').read_text(encoding='utf-8'))
    run=json.loads((OUT/'training_manifest.json').read_text(encoding='utf-8'))
    prep=json.loads((OUT/'preparation_manifest.json').read_text(encoding='utf-8'))
    assert verification['status']=='PASS'
    text=['# 第十一轮方法测试：训练优化与辅助任务诊断',
          '本轮只检查训练过程。新增验证集预测和评分均为 **0**；没有用新的准确率或收益结果挑选方案。18 个旧模型状态、18 次配对重启训练和全部复核均已完成。',
          (ROOT/'interpretation11.md').read_text(encoding='utf-8'),
          '## 冻结的实验范围',
          '三个训练截止日为 2017-12-31、2018-12-31、2019-12-31；对应 1,678、1,921、2,165 条样本。每条样本的收益标签和辅助标签均须在该截止日前完成。各折沿用原来的 125 根日线窗口、原始 OHLCV 特征、训练均值与总体标准差。运行阶段只加载本轮单独保存的训练数组。',
          '固定检查 3 个种子（20260910、20260911、20260912）。原模型均为 combined 架构、38,551 个参数、dropout 0.3。每批 128 条，最后一批仍为 14、1、117 条；每轮分别 14、16、17 步。第九轮的均衡分批没有被采纳。',
          '收益误差按原训练标准差标准化。固定目标为缩放 Huber：当 |e|≤1 时为 e²，否则为 2|e|−1；共同对照目标为收益 Huber + 0.1×辅助 MSE。两种旧模型都按这一目标重新检查训练损失，因此可直接比较。',
          '每个旧 Huber20 检查点派生两次重启：联合损失组辅助系数为 0.1，仅收益组为 0。两组从相同权重出发，采用相同的新批次顺序和初始 dropout 随机序列；新 AdamW、学习率 0.0001、weight decay 0.1、梯度裁剪范数 1、固定 20 轮。第 0/5/10/15/20 轮只记录训练轨迹，第 20 轮为预先指定终点。',
          '旧检查点没有优化器状态，因此这不是原第 21 轮的精确续训。低学习率、重置优化器和额外预算共同构成这次局部优化检查；本轮不能分别归因于其中一个因素。仅收益组仍计算辅助输出，辅助梯度为零张量，AdamW 的权重衰减仍作用于辅助头。',
          '## 开启 dropout 后的训练损失',
          '每个状态做 1 次关闭 dropout 的评估，以及 8 次固定种子的全训练集 dropout 前向计算。每次按样本数加权；下表再对 9 个模型等权平均。各折样本重叠，这些均值和计数仅用于描述。']
    for mode,title in [('eval','关闭 dropout'),('dropout','开启 dropout，8 次固定抽样的均值')]:
        rows=[]
        for state in NAMES:
            g=losses[losses.state.eq(state)&losses['mode'].eq(mode)]
            rows.append([NAMES[state],*[f'{g[k].mean():.6f}' for k in ['return_huber','auxiliary_mse','common_joint','return_mse']]])
        text += [f'**{title}**',table(['模型状态','收益 Huber','辅助 MSE','共同联合损失','收益 MSE'],rows)]
    p=comparisons[comparisons.comparison.eq('archived_huber_minus_mse')]
    variability=pd.read_csv(OUT/'paired_dropout_variability.csv')
    variability=variability[variability.comparison.eq('archived_huber_minus_mse')]
    rows=[]
    for r in p.itertuples():
        rows.append(['关闭 dropout' if r.mode=='eval' else '8 次 dropout 均值',
                     f'{r.common_joint_difference_mean:+.6f}',f'{int(r.common_joint_lower_count)}/9'])
    text += [table(['旧 Huber20 − 旧 MSE20','共同联合损失差','Huber20 更低的配对数'],rows),
             f'按每个配对的全部 8 次抽样逐次检查，只有 {int(variability.common_joint_difference_min.gt(0).sum())}/9 组每次都是 Huber20 联合损失更高；{int(variability.common_joint_difference_max.lt(0).sum())}/9 组每次更低，其余存在符号变化。均值接近零的个别配对不宜视为稳定差异。',
             '有限次 dropout 平均不是精确期望，也不是置信区间。它与训练日志的在线损失不同：这里权重固定，在线日志中每个批次的权重持续变化。AdamW 的解耦权重衰减也不能由这张表中的数据损失完整表述。',
             '![固定权重损失与梯度](frozen_training_diagnosis.png)',
             '## 辅助任务是否与收益任务冲突',
             '对全部 18 个旧模型，分别计算全训练集收益 Huber 梯度和辅助 MSE 梯度：关闭 dropout 计算一次，开启 dropout 计算四次并先平均梯度，再计算夹角。仅将共享骨干及 geometry 参数用于下表；完整参数梯度同时留存，两个独立输出头不参与共享参数的夹角计算。',
             '“辅助范数比”是 ||0.1g_aux|| / ||g_return||；“辅助投影”是 0.1〈g_return,g_aux〉 / ||g_return||²。投影为负表示局部欧氏梯度冲突；小于 −1 才表示共同梯度在这个局部指标下整体背离收益下降方向。此处尚未经过全局裁剪或 AdamW，不等于实际更新归因。']
    rows=[]
    for r in gradients[gradients.parameter_group.eq('shared')].itertuples():
        rows.append([NAMES[r.state], '关闭 dropout' if r.mode=='eval' else '4 次梯度平均',
                     f'{int(r.negative_cosine_count)}/9',f'{r.cosine_mean:+.3f}',
                     f'{r.weighted_auxiliary_norm_ratio_mean:.3f}',f'{r.auxiliary_projection_fraction_mean:+.3f}',
                     f'{int(r.full_joint_step_opposes_return_count)}/9'])
    text += [table(['旧权重','模式','余弦为负数','平均 cosine','辅助范数比','辅助投影','整体背离数'],rows),
             '## 18 次配对重启的固定终点',
             '下表以 8 次 dropout 抽样均值比较第 20 轮。损失变化为负表示降低；百分比由九组平均损失之比计算，不是九个百分比的均值。所有配对完整保留，不挑选种子或中间轮次。']
    names={'joint_restart_minus_start':'联合重启 − 原 Huber20','return_only_restart_minus_start':'仅收益重启 − 原 Huber20',
           'return_only_minus_joint':'仅收益重启 − 联合重启','joint_restart_minus_archived_mse':'联合重启 − 原 MSE20'}
    rows=[]
    for name,label in names.items():
        r=comparisons[comparisons.comparison.eq(name)&comparisons['mode'].eq('dropout')].iloc[0]
        rows.append([label,f'{r.return_huber_mean_relative_change*100:+.2f}%',f'{int(r.return_huber_lower_count)}/9',
                     f'{r.common_joint_mean_relative_change*100:+.2f}%',f'{int(r.common_joint_lower_count)}/9',
                     f'{r.auxiliary_mse_mean_relative_change*100:+.2f}%'])
    text += [table(['配对比较','收益 Huber 变化','收益更低数','共同联合损失变化','联合更低数','辅助 MSE 变化'],rows),
             '![配对重启训练](paired_training_restarts.png)',
             '## 可复核性与边界',
             f'- 训练前 {contract["checks_passed"]} 项检查通过：因果缓存、标准化、Huber 公式与导数、分批/整体梯度一致性、诊断的随机状态隔离及配对更新一致性。梯度分批相对误差最大 {max(contract["gradient_partition_relative_errors"]):.3e}。',
             f'- 18 次重启共 360 个训练轮次，训练与终点抽样耗时 {run["elapsed_seconds"]:.1f} 秒。新增检查点包含优化器与随机状态，可支持后续精确恢复。',
             f'- 复核 36 个模型状态、36 次确定性训练评估、288 次 dropout 抽样、90 次梯度计算、360 组批次顺序及边界、90 行配对结果，全部通过。训练输出最大复算误差 {verification["maximum_output_replay_error"]:.3e}；梯度最大误差 {verification["maximum_gradient_replay_error"]:.3e}。',
             f'- 此前 {verification["previous_files_preserved"]} 个证据文件的哈希保持一致。冻结协议 SHA-256：`{prep["protocol_sha256"]}`。',
             '- 本轮不进行新的验证预测，不计算验证准确率、收益率或显著性。训练损失下降只能支持优化层面的判断；不能证明泛化改善、完全收敛或可交易性。',
             '- 历史验证区间已被前十轮反复查看。之后若进入预测验证，仍应视为探索，并完整报告固定种子和基准；不能把它重新称为独立测试。',
             '- 原论文复现所需的完整资产名单、形态构造和部分训练接口仍不齐全，本系列是可审计的方法实验，不是完整原论文复现。',
             '## 证据与复现入口',
             '[冻结协议](../protocol.json) · [训练前检查](contract_verification.json) · [最终复核](verification.json) · [结论数据](assessment.json)',
             '[训练样本索引](training_rows.csv) · [每次抽样损失](archived_loss_draws.csv) · [重启抽样损失](restart_loss_draws.csv) · [损失汇总](training_loss_summary.csv)',
             '[配对抽样波动](paired_dropout_variability.csv) · [旧训练目标复算核对](supplementary_verification.json)',
             '[全部配对结果](paired_training_comparisons.csv) · [梯度明细](gradient_diagnostics.csv) · [训练轨迹](restart_training_monitors.csv) · [训练预算与顺序](restart_training_curves.csv)',
             '[旧模型清单](archived_models.json) · [重启模型清单](restart_models.json) · [检查程序](../contract11.py) · [探针程序](../probe11.py) · [训练程序](../train11.py) · [复核程序](../verify11.py)',
             'GPU 运行环境沿用 `research_v4/.venv_gpu/Scripts/python.exe`。脚本顺序为 prepare11 → contract11 → probe11 → train11 → analyze11 → verify11；已有冻结结果禁止覆盖，重跑应建立新的研究目录。图表与报告分别由 figures11.py、report11.py 生成。']
    report='\n\n'.join(text)+'\n'
    (OUT/'第十一轮测试报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(report_characters=len(report),training_only=True)))


if __name__=='__main__':
    main()

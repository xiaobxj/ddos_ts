"""Render a self-contained Chinese report from verified quantitative evidence."""
from pathlib import Path
import json
import hashlib
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
ARMS = ['baseline', 'small', 'dropout', 'decay', 'combined']
NAMES = {'baseline': '原联合目标基准', 'small': '缩小模型', 'dropout': '增加 dropout',
         'decay': '增加权重衰减', 'combined': '三项组合'}
HISTORY = {'full': '完整历史', 'drop_early': '删除最早约 20%',
           'drop_middle': '删除中间约 20%', 'drop_recent': '删除最近约 20%'}


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def pct(x):
    return f'{x*100:.2f}%'


def md(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(map(str, row)) + ' |' for row in rows)


def main():
    audit = read('verification.json')
    assert audit['status'] == 'PASS'
    selection = read('selection.json')
    selected = selection['selected']
    findings = read('findings.json')
    metrics = pd.read_csv(OUT / 'main_metrics.csv')
    years = pd.read_csv(OUT / 'main_yearly_metrics.csv')
    seeds = pd.read_csv(OUT / 'main_seed_metrics.csv')
    train = pd.read_csv(OUT / 'main_training_curves.csv')
    blocks = pd.read_csv(OUT / 'block_metrics.csv')
    shifts = pd.read_csv(OUT / 'block_prediction_sensitivity.csv')
    parameters = read('model_parameters.json')
    references = read('previous_reference.json')
    mean = next(x for x in selection['baseline_metrics'] if x['method'] == 'training_mean')
    baseline = metrics[(metrics.arm == 'baseline') & (metrics.epoch == 10)].iloc[0]
    previous4 = references['round4_reference']
    header_table = md(['方法', '训练轮数', '方向准确率', '执行收益 RMSE', '相对训练均值的 MSE 改善'], [
        [NAMES[selected['arm']]+'（本轮按 RMSE 选中）', selected['epoch'], pct(selected['accuracy']), f"{selected['rmse']:.6f}", pct(selected['mse_skill_vs_training_mean'])],
        [NAMES['baseline']+'（第五轮选中方案）', 10, pct(baseline.accuracy), f'{baseline.rmse:.6f}', pct(baseline.mse_skill_vs_training_mean)],
        ['第四轮已冻结参考*', previous4['epochs'], pct(previous4['accuracy']), f"{previous4['rmse']:.6f}", '训练集略有不同'],
        ['成熟训练样本均值', '—', pct(mean['accuracy']), f"{mean['rmse']:.6f}", '0.00%']])
    arm_table = md(['组别', 'd_model / d_ff', 'dropout', 'AdamW 权重衰减', '参数量'], [
        [NAMES[r['name']], f"{r['d_model']} / {r['d_ff']}", r['dropout'], r['weight_decay'], f"{r['parameter_count']:,}"] for r in parameters])
    rows = []
    for arm in ARMS:
        for r in metrics[metrics.arm == arm].sort_values('epoch').itertuples():
            rows.append([NAMES[arm], r.epoch, pct(r.accuracy), f'{r.rmse:.6f}',
                         pct(r.mse_skill_vs_training_mean), f'{r.within_fold_forecast_std*10000:.2f}'])
    all_table = md(['组别', '训练轮数', '准确率', 'RMSE', '相对均值 MSE 改善', '折内预测标准差／bp'], rows)
    primary = md(['同为第 10 轮：相对原基准', 'ΔRMSE', '95% 区间', 'MSE 原始 p', 'MSE Holm p'], [
        [NAMES[r['arm']], f"{r['rmse']['difference']:+.6f}",
         f"[{r['rmse']['ci95_low']:+.6f}, {r['rmse']['ci95_high']:+.6f}]",
         f"{r['mse']['p']:.4f}", f"{r['mse']['holm_adjusted_p']:.4f}"] for r in read('primary_comparisons.json')])
    chosen_years = years[(years.arm == selected['arm']) & (years.epoch == selected['epoch'])]
    year_table = md(['年份', '周数', '准确率', 'RMSE', '相对均值 MSE 改善'], [
        [r.year, r.n, pct(r.accuracy), f'{r.rmse:.6f}', pct(r.mse_skill_vs_training_mean)] for r in chosen_years.itertuples()])
    chosen_seeds = seeds[(seeds.arm == selected['arm']) & (seeds.epoch == selected['epoch'])]
    seed_table = md(['训练种子', '准确率', 'RMSE', '相对均值 MSE 改善'], [
        [r.seed, pct(r.accuracy), f'{r.rmse:.6f}', pct(r.mse_skill_vs_training_mean)] for r in chosen_seeds.itertuples()])
    train_rows = []
    for arm in ARMS:
        for epoch in [5, 10, 20]:
            t = train[(train.arm == arm) & (train.epoch == epoch)]
            v = metrics[(metrics.arm == arm) & (metrics.epoch == epoch)].iloc[0]
            train_rows.append([NAMES[arm], epoch, f'{t.final_model_train_return_mse.mean():.4f}',
                               f'{t.final_model_train_auxiliary_mse.mean():.4f}', f'{v.rmse:.6f}'])
    training_table = md(['组别', '轮数', '训练收益标准化 MSE', '训练辅助目标标准化 MSE', '验证收益 RMSE'], train_rows)
    block_rows = []
    for arm in ARMS:
        for history in HISTORY:
            r = blocks[(blocks.arm == arm) & (blocks.history == history)].iloc[0]
            block_rows.append([NAMES[arm], HISTORY[history], pct(r.accuracy), f'{r.rmse:.6f}',
                               pct(r.mse_skill_vs_training_mean), f'{r.mse_difference_vs_matched_baseline:+.7f}', int(r.rank_within_history)])
    block_table = md(['组别', '训练历史', '准确率', 'RMSE', '相对保留样本均值 MSE 改善', 'ΔMSE 对同历史原基准', '同历史排名'], block_rows)
    stability_table = md(['组别', '优于均值的历史数', '优于同历史原基准的历史数', '排名范围', '最大预测 RMS 改变／bp', '最大方向改变比例'], [
        [NAMES[r['arm']], f"{r['better_than_training_mean_histories']}/4", f"{r['better_than_matched_baseline_histories']}/4" if r['arm'] != 'baseline' else '自身',
         f"{r['best_rank']}–{r['worst_rank']}", f"{r['maximum_prediction_rms_shift']*10000:.2f}", pct(r['maximum_direction_disagreement'])]
        for r in read('block_stability_summary.json')])
    folds = pd.read_csv(OUT / 'fold_history_definitions.csv')
    fold_table = md(['验证年', '完整训练数', '验证周数', '删除早／中／近后训练数', '每轮更新次数'], [
        [str(cutoff[:4])+' → '+str(int(cutoff[:4])+1), int(g[g.history=='full'].train_n.iloc[0]), int(g.test_n.iloc[0]),
         ' / '.join(str(int(g[g.history==h].train_n.iloc[0])) for h in list(HISTORY)[1:]), int(g.optimizer_steps_per_epoch.iloc[0])]
        for cutoff, g in folds.groupby('cutoff', sort=False)])
    deletion_table = md(['截止日', '删除位置', '删除标签锚点日期范围', '删除数量'], [
        [r.cutoff, HISTORY[r.history], f'{r.removed_start} 至 {r.removed_end}', r.removed_n]
        for r in folds[folds.history!='full'].itertuples()])
    text = build_text(audit, findings, selection, references, header_table, arm_table, all_table, primary,
                      year_table, seed_table, training_table, block_table, stability_table, fold_table, deletion_table)
    report = OUT / '第六轮测试报告.md'
    report.write_text(text, encoding='utf-8')
    links = re.findall(r'\]\(([^)]+)\)', text)
    for target in links:
        if not target.startswith(('https://', 'http://')):
            assert (OUT / target).resolve().exists(), target
    hashes = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name == 'delivery_manifest.json':
            continue
        with path.open('rb') as stream:
            hashes[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (OUT / 'delivery_manifest.json').write_text(json.dumps(dict(report=str(report.relative_to(ROOT)),
         report_links_checked=len(links), files=hashes), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(report=str(report), characters=len(text), hashed_files=len(hashes), checked_links=len(links))))


def build_text(audit, findings, selection, references, header, arms, all_metrics, primary, years, seeds,
               training, blocks, stability, folds, deletions):
    main_run = read('main_manifest.json')
    block_run = read('blocks_manifest.json')
    narrative = '\n\n'.join(findings['interpretation'])
    next_steps = '\n'.join(f'{i}. {line}' for i, line in enumerate(findings['next_steps'], 1))
    return f'''# 第六轮方法测试：模型大小、正则化与训练历史稳定性

{findings['headline']}

本轮使用同一组 **141 个周五信号（2018—2020 年）**。所有模型预测从信号后的下一个实际开盘，到下一同星期信号后的实际开盘之间的收益；这是前几轮已经查看过的历史验证。准确率、RMSE 均为预测指标，没有重新计算交易费用或策略净收益。

{header}

*第四轮参考仍使用其原先的训练集，比本轮多五个早期样本；作为冻结参考展示，不能据此严格归因正则化变化。第五轮联合目标基准与本轮训练样本、模型和目标完全一致，**{references['round5_baseline_rows_matched']} 条逐种子预测逐条复现，最大差异 {references['maximum_round5_prediction_difference']:.3g}**。

## 1. 本轮回答什么

上一轮发现模型能拟合少量真实样本，但训练延长后验证误差增加。本轮固定输入和联合目标，只改变模型宽度与正则化，再检查删除历史样本块时的表现。

{arms}

小模型参数从 138,471 减至 38,551，减少约 {(1-38551/138471)*100:.1f}%。保持 2 层编码器、4 个注意力头、4 个路由向量和原有解码结构。缩小宽度也改变每个注意力头的维度；没有测试缩减层数。前三个修改可各自与原基准作单因素比较；“三项组合”是捆绑候选，不是完整因子实验，不能分离所有交互作用。

共同目标为标准化收益 MSE ＋ 0.1 × 未来五根 K 线 OHLCV 辅助目标 MSE；收益头零初始化，辅助头使用独立随机作用域。未来价格使用相对当前收盘的对数变化，未来成交量相对已观察到的最近 20 根 K 线作变换，再使用本折保留的成熟训练标签计算均值和标准差。该辅助目标属于本地明确定义的研究目标，不是缺失 OHLCAV 训练细节的论文原样复现。

原始输入、25 × 5 分段、125 根观察窗口以及全部 3,780 个样本都直接复用第五轮文件。没有引入量化 token、预训练表示、重新下载的数据或自适应分段变化。严格辅助标签过滤额外排除的五个早期样本仍然排除。

## 2. 时间划分和预先固定的预算

{folds}

训练只使用 `joint_completed ≤ cutoff` 的日频标签。验证使用下一年度周五信号，最终标签须在 2020-12-31 前完成。训练日期和验证日期没有随机混合。训练标签互相重叠，因此训练行数不能当成独立样本数。

主实验共 **45 次拟合**：5 组 × 3 年 × 3 个种子，每次完成 20 轮，保存第 2、5、10、20 轮，共 180 个检查点。种子为 20260910、20260911、20260912。AdamW 学习率 0.001，batch 128，梯度裁剪 1.0。

全部主实验完成后，按每周三个种子的平均预测计算 RMSE，选择 20 个候选中最低者；同分时优先更少轮数，再按协议组别顺序。方向准确率和样本块删除结果均不参与选择。计算用时约 {main_run['elapsed_seconds']/60:.1f} 分钟。

主要推断比较预先固定为第 10 轮的四项变化对原基准：8 个相邻验证观察值的循环区块 bootstrap，10,000 次，随机种子 20260910，中心化双侧 p，四项 MSE 比较作 Holm 调整。区间仍受有限历史和多轮研究选择的限制。

## 3. 全部候选结果

{all_metrics}

![模型大小和正则化的训练及验证表现](regularization_comparison.png)

图左为三种子集成的验证 RMSE，图右为九次拟合中最终模型训练标准化收益 MSE 的均值。图中的 bp 为收益率万分之一，便于读数，并非费用或盈利。训练和验证两栏的标准化口径不同，不应直接比较纵轴绝对数值。

同为第 10 轮，负的 ΔRMSE 表示优于原联合目标基准：

{primary}

{narrative}

## 4. 选中候选的年度和种子稳定性

选中设置为 **{NAMES[selection['selected']['arm']]}，训练 {selection['selected']['epoch']} 轮**。下表年度数据均来自相应年度只使用历史训练资料的预测。

{years}

下表分别计算单个种子的 141 周表现，主结果表使用三个种子的平均预测；两者不能混用。

{seeds}

训练误差随轮数的变化如下。每行训练值是三个年度 × 三个种子的均值，验证值为同一设置的三种子集成：

{training}

## 5. 删除训练历史块的压力测试

各年度把成熟训练样本按日期分成五段，分别删除第 1、3、5 段，约为最早、中间、最近的 20%。这是删除标签锚点；剩余样本的 125 根 K 线窗口仍可能覆盖部分被删除日期。没有宣称完全清除了这段底层行情。

{deletions}

为保持优化更新预算一致，每轮仍展示完整历史对应的样本数量。先无放回打乱保留样本，不足部分再从新的随机排列补齐，所以部分保留样本会重复。各组用同一训练掩码与采样顺序；标准化只看保留样本。这项设计检验样本组成变化，不能解释为普通删除样本后的无权重重训。

删除实验固定 **10 轮、种子 20260910**，共 45 次拟合，约 {block_run['elapsed_seconds']/60:.1f} 分钟。完整历史参考取自主实验的相同种子和轮数，没有用集成预测替代。删除位置和轮数都不参与候选筛选。该项只有一个种子，不能分离所有样本组成与初始化的交互。

本轮主实验最终选中的是 **组合设置第 20 轮**，而删除测试按训练前协议固定为第 10 轮。因此下述删除结果直接描述第 10 轮各组的敏感性，**没有检验选中候选第 20 轮的删除稳定性**，也不能用第 10 轮的结果替它判定通过或失败。

部分主实验与删除实验使用独立进程同时运行，各自保持独立随机状态与输出，因此两项用时不能直接相加为总耗时，详见[运行说明](execution_overlap_note.json)。

{blocks}

相对均值的改善分别使用当前保留样本的训练均值。“ΔMSE 对同历史原基准”则保持相同训练历史，负数表示该变化优于原模型。排名只在同一种训练历史下的五个模型之间计算。

四种训练历史共用同一段 141 周验证，彼此高度相关。“优于基准的历史数”是敏感性描述，不是四次独立验证，也没有为这些删除结果另作显著性宣称。

删除训练样本也会改变目标均值与标准差。例如，截至 2018 年底的训练集删除中间块后，平均收益从约 +11.46 bp 变为 −6.74 bp。因此绝对预测的改变同时包含训练标签分布变化和模型响应变化，不能全部归于网络不稳定。[标签分布诊断](training_history_label_statistics.csv)及[预测变化分解](block_prediction_sensitivity.csv)分别保留了训练均值变化和扣除均值后的预测差异；后者仍不分离标准差或所有交互的影响。

![删除历史样本块的误差和预测变化](block_stability.png)

{stability}

{findings['block_interpretation']}

## 6. 复验和证据

**{audit['contract_tests']} 项行为检查通过；{audit['total_fits']} 次拟合、{audit['total_checkpoints_replayed']} 个检查点全部完成预测重放。** 主实验最大重放误差 {audit['main']['maximum_prediction_error']:.3g}，删除实验 {audit['blocks']['maximum_prediction_error']:.3g}，属于文本浮点存储精度差异。

核对了全部折的成熟标签条件、独立重建的删除掩码、保留标签的标准化参数、每轮样本展示数和更新次数。主实验保存 {audit['main']['prediction_rows']:,} 条逐种子预测，删除实验保存 {audit['blocks']['prediction_rows']:,} 条。既有 {audit['prior_evidence_files_preserved']:,} 个证据文件的哈希保持不变，输入、训练源码和冻结协议均一致。

基准行为检查覆盖权重、CPU/GPU 随机状态及一次完整训练更新；所有组别检查辅助目标对主干的梯度连接；模型宽度和 dropout 配置、保留样本采样与预算亦作了确定性检查。复验结论是实现符合本轮协议，不等同于市场有效性证明。

可复查文件：

- [训练前固定协议](../protocol.json)、[来源与输入清单](preparation_manifest.json)、[模型参数量](model_parameters.json)。
- [完整样本及成熟日期](observation_table.csv)、[各折与删除定义](fold_history_definitions.csv)。
- [主实验逐种子预测](main_seed_predictions.csv)、[集成预测](main_ensemble_predictions.csv)、[全部指标](main_metrics.csv)、[年度指标](main_yearly_metrics.csv)、[种子指标](main_seed_metrics.csv)、[训练曲线](main_training_curves.csv)。
- [预先指定的四项比较](primary_comparisons.json)、[选中规则与结果](selection.json)、[旧基准精确复现](previous_reference.json)。
- [删除实验预测](blocks_seed_predictions.csv)、[完整及删除历史对照预测](block_comparison_predictions.csv)、[删除实验指标](block_metrics.csv)、[预测变化](block_prediction_sensitivity.csv)、[稳定性汇总](block_stability_summary.json)。
- [完整检查结果](verification.json)、[行为检查输出](contract_test_output.txt)、[主实验检查点哈希](main_checkpoint_manifest.json)、[删除实验检查点哈希](blocks_checkpoint_manifest.json)。
- 矢量图：[训练与验证](regularization_comparison.svg)、[训练历史稳定性](block_stability.svg)。

## 7. 下一步研究取舍

{next_steps}

本轮未对 2024—2026 年结果重新评分，也没有推进实盘或形成可部署的参数建议。历史验证已经经过多轮查看，即使某个候选优于旧结果，也需要在事先固定的后续检验中验证。
'''


if __name__ == '__main__':
    main()

"""Trace every prior research round to an explicit test, guard or retained limitation."""
from core import *

# Coverage denotes what this experiment actually does; guard != rerun of every old candidate.
GROUPS=[
([1,3,4],'论文复现边界、预训练未来信息、模型塌缩和简单参照',
 '保护＋诊断','只重新训练原始 OHLCV 小型模型，无预训练权重；记录输入改变敏感性、预测方差和简单基准。未重跑 DTW／Kronos。','protocol.json; results/metrics.csv; results/fits/*/heads_*.json'),
([2],'输入日频、标签周期、预测日与执行收益口径混淆',
 '重新验证','H1／H5 收盘至收盘标签逐行独立计算；逐日预测；旧开盘至开盘目标不作直接优劣对照。','results/contract.json; results/sample_counts.json'),
([5,6,11,12],'训练误差下降不等于泛化改善，轮数和优化设定混在一起',
 '受控重测＋边界','同一轨迹保留 10／20 遍；固定小模型和正则；不重新搜索 Huber、学习率或辅助权重。','results/year_metrics.csv; results/native_metrics.csv; results/fits/*/training_curves.csv'),
([7],'选择种子或删除样本后的局部胜出被当成普遍效果',
 '受控重测＋保护','三固定种子等权概率；公布全部种子波动；不按测试成绩选种子，不做事后删样本。','results/seed_metrics.csv'),
([8,9],'重叠标签、小批次损失权重以及删样本后重复曝光',
 '诊断＋保护','记录末批大小与名义权重比；自然遍数不补齐步数；保留逐日样本，报告收益边复用和最大不重叠数，分块推断。未重跑均衡批次候选。','results/sample_counts.json; results/bootstrap.csv; results/fits/*/training_curves.csv'),
([10],'稳健损失看似降低极端误差却丢失方向信息',
 '保护，候选未重跑','本轮不默认采用 Huber；原始收益 MSE、相关性、方差与概率指标分开报告。','results/native_metrics.csv; results/metrics.csv'),
([13],'把样本内残差当成可用的历史校准预测',
 '重新验证','季度／月度校准只用真正因果的年度折外概率银行；训练头的 scalar_probe 明示为样本内监督拟合；校准训练验证按标签终点清除交叠。','calibration.py; results/outputs_verified.json'),
([14],'直接分类不一定优于回归；标准化收益正负误作涨跌',
 '受控重测','同成员、种子和结构比较 MSE／BCE；BCE 始终用原始收益 > 0；原生 MSE 不伪装成概率。','results/synthetic_contract_checks.json; results/metrics.csv; results/native_metrics.csv'),
([15],'冻结特征的浮点路径改变，以及训练表示不是 OOF',
 '重新验证','本轮统一 requires_grad=False 提取；逐检查点回放，界定数值容差；训练表示和监督交互不称 OOF。','results/checkpoint_checks.csv; results/head_gradient_checks.csv'),
([16,17],'复杂表征未超简单参照，截尾标准化或状态阈值偷看未来',
 '受控重测＋验证','频率与仅四项行情特征同训练成员；分位数、均值、尺度、波动中位数来自截止前训练行。未重跑全部 25 项简单特征族。','results/metrics.csv; results/causal_training_prefix_checks.json'),
([18,19],'市场条件能识别不代表可预测；交互近期方向收益伴随概率误差',
 '受控重测','加性／波动交互并列，逐年 Brier、对数损失与方向一起比较；交互投影正交、增秩与残差尺度复核。','results/year_metrics.csv; results/head_gradient_checks.csv'),
([20,27,28],'只缩短最后分类头却声称全模型近期训练；窗口与步数／重训混淆',
 '受控重测','3／5 年均从头训练整条表示和头，自然 10／20 遍；季度整网更新只检验 2026 年两个季度，不外推全部年份。','results/training_membership_plan.csv; results/year_metrics.csv'),
([21],'HMM 平滑泄漏，训练期过滤误称逐期折外识别',
 '保护，HMM 未重跑','本轮使用当时可知的趋势／波动，不使用 HMM 平滑；不把训练内状态参数解释为 OOF。','protocol.json; results/causal_training_prefix_checks.json'),
([22],'路径效率与既有趋势信息重复，微小概率改善被遗忘',
 '保留证据，候选未重跑','保留原路径效率概率改善线索；本轮未重新测试该候选，不把价格顺序项等同于路径效率，也不把未采用说成无效。','results/prior_report_sources.json; 历史问题核对.md'),
([23,24,25],'价格顺序扩展后跨期退步、惩罚搜索与共同重拟合归因',
 '受控重测','并列联合顺序头与固定波动父头仅加顺序系数；固定同一岭强度，拒绝看结果后加罚搜索。','results/metrics.csv; results/head_gradient_checks.csv'),
([26],'历史扩展后原来的近期优势消失',
 '受控重测','2023—2026 逐年展示，同期简单基准；2026 为截至 9 月的部分年；不能只报最近好年份。','results/year_metrics.csv'),
([29,30,31],'漂移／误差触发、影子试用和融合都未稳定超过年度更新',
 '保护＋有限更新重测','年度主模型固定；另测 2026 季度整网重训，不启用新触发器、影子择优或看成绩后融合。','protocol.json; results/year_metrics.csv'),
([32,33,34,35],'更新分类头的坐标和系数混淆，删旧标签不等于骨干忘记旧数据',
 '受控重测＋边界','时间加权及截距／斜率混合只在同一冻结坐标中对照；3 年整网重新训练。未重跑所有季度头更新／样本移除分解。','results/head_gradient_checks.csv; results/metrics.csv'),
([36,37,38],'状态样本过少、日标签约五倍重叠、过拟合状态偏移',
 '受控重测＋诊断','四状态共享基模型偏移；门控／无门控／合并组对照；训练与验证分别要求最大不重叠标签数。该数不叫有效独立样本量。','results/calibration_coverage.csv; results/sample_counts.json; results/calibration_decisions_h1.json; results/calibration_decisions_h5.json'),
([39,40],'最近验证通过不保证未来有效，月度更新增加覆盖可能变差',
 '受控重测','季度四状态、月度四状态、季度合并组和 Platt 校准固定对照；Brier 改善且方向不退步才启用，验证后不重拟合。','results/metrics.csv; results/calibration_coverage.csv; results/outputs_verified.json'),
([41,43,44],'验证依赖少数观察，重复使用旧样本，却误称新增独立证据',
 '诊断','删除 H 日信号区段的指标敏感性、相邻窗口复用、新加入日期均存档；标签跨区段仍可能重叠，不作独立重复试验。','results/calibration_decisions_h1.json; results/calibration_decisions_h5.json; results/calibration_reuse_h1.csv; results/calibration_reuse_h5.csv'),
([42],'旧状态参数保留与自动续期，收益集中在同一天',
 '保护，保留政策未重跑','本轮不延续旧状态修正；每次按当时验证重新决定，Q1 回到年度基模型，独立记录实际改变日期。','results/synthetic_contract_checks.json; results/calibration_coverage.csv'),
([45,46],'合并状态增加样本支持却不一定提升预测，同一天的跨方法重复',
 '有限受控重测','四状态与全市场合并校准对照；本轮未重跑趋势／波动两组子状态验证；按真实日期而不是模型数量解释覆盖。','results/calibration_coverage.csv; results/metrics.csv'),
([47,48],'近期加权整体、截距、斜率作用不同，2026 局部改善不能替代全期',
 '受控重测','固定 730.5 日半衰期，在同一特征坐标比较 U／W／I／S；逐年概率和方向分别报告。','results/year_metrics.csv; results/head_gradient_checks.csv'),
([49,50,51],'回填历史预测、参数提前生效、未成熟标签计分、改写旧证据',
 '保护＋工程验证','前缀重放、成熟度排除和参数截止检查；历史结果只存研究目录，旧指数账本无新增预测或标签；本轮不部署自动日更入口。','results/outputs_verified.json; results/latest_probabilities.json; results/preserved_files.json')
]

def main():
    assert read(OUT/'models_verified.json')['status']=='PASS' and read(OUT/'outputs_verified.json')['status']=='PASS'
    sources={r['round']:r for r in read(OUT/'prior_report_sources.json')};rows=[]
    for rounds,lesson,coverage,action,evidence in GROUPS:
        for n in rounds:rows.append(dict(round=n,lesson=lesson,coverage=coverage,action=action,evidence=evidence,source=sources[n]))
    assert sorted(r['round'] for r in rows)==list(range(1,52))
    save(OUT/'prior_problem_coverage.json',sorted(rows,key=lambda r:r['round']))
    lines=['本次把前 51 轮的问题逐项映射到重测、工程检查或明确保留的边界。下表的“保护”不等于重新跑过原轮次的每一个候选。',
        '', '以前有改善的线索继续保留：R19 的 2018—2020 年波动交互为 82/141，高于加性的 80/141，但概率误差更高；R22 的路径效率略降 Brier 而学习分支方向不变；R39／R46 的改善多次落在相同少数日期；R48 的近期加权斜率也有 2026 年局部改善。它们的完整旧报告保持原样。本次全部新结果使用 603986 的新目标，不能把旧指数分数当成可直接对比的个股基准。',
        '', '| 原轮次 | 曾遇到的问题 | 本次覆盖 | 验证或边界 |','|---|---|---|---|']
    for rounds,lesson,coverage,action,evidence in GROUPS:
        links='、'.join(f"[R{n}](../{sources[n]['file'].replace(chr(92),'/')})" for n in rounds)
        lines.append(f'| {links} | {lesson} | {coverage} | {action} |')
    lines.extend(['','机器可审计的逐轮来源哈希和证据文件见 [51 轮映射](results/prior_problem_coverage.json)。',
        '', '没有为了覆盖数量再跑全部 HMM、DTW、预训练、样本删除、Huber、学习率、触发器、旧状态保留等组合。这些候选在本次标为未重测，不能据此宣布它们对个股无效。',
        '', '工程准备时保留了两次未启动训练的失败：数据列顺序导致严格比较拒绝，以及旧清单文件名定位错误；另保留了评分模块同名导入的测试失败。修复均发生在对应训练／评价冻结之前，未改动预测目标、模型参数或评价阈值。'])
    (ROOT/'历史问题核对.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()

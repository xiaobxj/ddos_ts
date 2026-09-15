"""Chinese report from verified classification outputs and explicit interpretation."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
NAMES={'archived20':'原收益 MSE20','direction_bce':'直接方向分类','training_frequency':'训练期上涨频率','neutral_50':'固定概率 0.5'}
FLAGS={'ensemble_accuracy_beats_original':'集成方向准确率优于原收益模型',
       'ensemble_accuracy_beats_frequency':'集成方向准确率优于训练频率判断',
       'ensemble_brier_beats_frequency':'集成 Brier 优于训练频率概率',
       'ensemble_log_loss_beats_frequency':'集成对数损失优于训练频率概率',
       'at_least_two_seeds_beat_original_accuracy':'至少 2/3 配对种子的准确率优于原模型',
       'at_least_two_seeds_beat_frequency_brier':'至少 2/3 种子的 Brier 优于训练频率',
       'at_least_two_years_beat_original_accuracy':'至少 2/3 年度的准确率优于原模型',
       'at_least_two_years_beat_frequency_brier':'至少 2/3 年度的 Brier 优于训练频率'}

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def number(value,decimals=6):return f'{value:.{decimals}f}' if pd.notna(value) else '—'
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])

def main():
    verification=read('verification.json');assert verification['status']=='PASS'
    assessment=read('assessment.json');primary=read('primary_comparisons.json');metadata=read('training_metadata.json')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv').set_index('method');years=pd.read_csv(OUT/'yearly_metrics.csv')
    seeds=pd.read_csv(OUT/'seed_metrics.csv');training=pd.read_csv(OUT/'training_summary.csv')
    changes=pd.read_csv(OUT/'direction_changes.csv');bins=pd.read_csv(OUT/'reliability_bins.csv');budgets=pd.read_csv(OUT/'training_budgets.csv')
    text=['# 第十四轮方法测试：直接预测上涨概率',
          (ROOT/'interpretation14.md').read_text(encoding='utf-8'),
          '## 本轮检验的问题',
          '第十三轮发现，向正的训练均值收缩，会把部分非上涨预测改成上涨；小幅降低收益误差没有带来方向准确率提高。本轮只设一个分类候选，检验直接学习上涨概率是否比学习收益大小更有助于方向预测。',
          '所有研究选择在新增训练前冻结：固定分类损失、初始概率、20 轮预算、三个种子、50% 判断阈值、概率集成规则、四项配对检验和八项描述性条件。没有搜索阈值、置信度筛选、概率校准、最佳种子或第二个分类变体。',
          '输入仍为原 raw OHLCV 五日分段表示，骨干结构、几何层和辅助 OHLCV 头沿用原 combined 模型：38,551 个参数、dropout 0.3。原来的 25→1 线性收益头在本轮输出上涨 logit，参数形状不变。整个模型从相同种子初始化并训练，骨干权重可以更新；这不是固定旧骨干的线性探针。',
          '原 MSE20 的收益头以零权重、零标准化收益偏置开始，实际收益对应训练均值。分类头也用零权重，其偏置设为 logit（训练期上涨频率），得到因果常数概率的起点。除这一个偏置张量外，初始张量与相同种子的原模型完全一致；赋值不消耗随机数。',
          '目标标签必须由原始可执行收益是否严格大于零产生，不能对标准化收益取正负。零收益属于非上涨。训练仍要求收益与辅助标签共同完成时间不晚于各年度截止日，受无效 OHLC 影响的窗口继续按原规则排除。',
          table(['训练截止日','训练样本','上涨样本','训练上涨频率','初始 logit','每轮更新步数','末批样本'],[
              [r.cutoff,r.train_n,metadata[r.cutoff]['up_count'],f"{metadata[r.cutoff]['frequency']*100:.4f}%",
               f"{metadata[r.cutoff]['initial_logit']:.6f}",r.steps_per_epoch,r.last_batch_n] for r in budgets.itertuples()]),
          '分类主损失为未加权二元交叉熵（BCEWithLogits）；联合目标为 BCE + 0.1×原标准化辅助 OHLCV MSE。未加入类别权重、标签平滑、额外收益损失或损失缩放。AdamW 学习率 0.001、weight decay 0.1、梯度裁剪范数 1、批大小 128、原全样本排列均保持不变。',
          '固定种子为 20260910、20260911、20260912。新增 9 次训练，各 20 轮，共 180 轮、2,820 次优化器更新、345,840 次样本呈现。原 9 个收益模型和 423 条预测直接复用。全部分类训练完成后才生成验证预测。',
          'BCE 与标准化收益 MSE 的尺度和梯度形态不同，因此名义辅助系数仍为 0.1，并不意味着辅助任务相对主任务的梯度作用保持相同。这是固定预算下的目标与输出语义对照，不能把所有差异归因于一个完全隔离的因素。',
          '## 分类模型在训练期学到了什么',
          '每个最终状态分别关闭 dropout 评估一次，并用 8 个固定随机种子进行 dropout 抽样；权重不变，诊断过程保护 CPU/CUDA 随机状态和训练模式。以下为九个年度/种子组合的等权均值，训练常数参考为各折已完成标签的上涨频率。']
    rows=[]
    for mode,label in [('eval','关闭 dropout'),('dropout','8 次 dropout 抽样均值')]:
        g=training[training['mode'].eq(mode)]
        rows.append([label,number(g.log_loss.mean()),number(g.constant_log_loss.mean()),number(g.brier.mean()),
            number(g.constant_brier.mean()),f'{g.accuracy.mean()*100:.2f}%',number(g.auxiliary_mse.mean()),number(g.joint_loss.mean())])
    text += [table(['训练诊断','对数损失','常数对数损失','Brier','常数 Brier','方向准确率','辅助 MSE','联合损失'],rows),
             '训练联合损失由稳定的 logit 对数损失加 0.1×辅助 MSE 计算。有限次 dropout 平均只是训练诊断，不能代替泛化结果；重复训练折和种子也不构成独立数据集。',
             '## 固定 141 周的结果',
             '沿用 2018、2019、2020 年的 49、46、46 个周五信号。收益标签仍为原定义的可执行开盘到开盘周收益；分类标签为该收益是否上涨。分类集成先平均三个种子的概率，再按严格 >0.5 判断上涨；没有平均 logit。原收益模型仍按三个原始收益预测的平均值是否 >0 判断。',
             '概率参考包括各训练截止日的上涨频率，以及固定概率 0.5。由于沿用严格大于阈值的规则，概率恰好为 0.5 时判断为非上涨；它不是随机抽签产生的方向预测。原收益模型没有原生概率，不计算它的 Brier、对数损失或概率校准指标，也不对收益值套用任意 sigmoid。']
    rows=[]
    for name in NAMES:
        r=metrics.loc[name]
        rows.append([NAMES[name],f'{r.accuracy*100:.2f}% ({int(r.correct_directions)}/141)',f'{r.balanced_accuracy*100:.2f}%',
            number(r.auroc,4),number(r.brier),number(r.log_loss),f'{r.predicted_up_fraction*100:.2f}%'])
    text += [table(['方案','方向准确率','平衡准确率','AUROC','Brier','对数损失','预测上涨比例'],rows),
             'Brier = 平均（预测上涨概率−实际上涨标签）²；对数损失使用自然对数。两者越低越好；Brier 与原始收益 MSE 的目标和单位不同，不能直接比较数值。AUROC 反映分数排序；原收益模型在该项使用原始收益预测作为分数。',
             '报告的概率对数损失将概率裁剪到 [1e−12, 1−1e−12]，仅用于数值稳定；不改变概率或方向判断。概率方法在本次验证中的裁剪计数为：'+
             '、'.join(f'{NAMES[n]} {int(metrics.loc[n,"clipped_probabilities"])}' for n in ['direction_bce','training_frequency','neutral_50'])+'。',
             '![方向与概率误差](direction_and_probability.png)',
             table(['方案','平均上涨概率','实际上涨频率','概率减实际频率','概率标准差'],[
                 [NAMES[n],number(metrics.loc[n,'mean_probability']),number(metrics.loc[n,'observed_up_fraction']),
                  number(metrics.loc[n,'calibration_gap']),number(metrics.loc[n,'probability_std'])] for n in ['direction_bce','training_frequency','neutral_50']]),
             '## 年度、种子和方向变化',
             table(['年度','方案','方向准确率','Brier','对数损失','AUROC'],[
                 [r.year,NAMES[r.method],f'{r.accuracy*100:.2f}%',number(r.brier),number(r.log_loss),number(r.auroc,4)]
                 for r in years[years.method.ne('neutral_50')].sort_values(['year','method']).itertuples()]),
             table(['固定种子','方案','方向准确率','Brier','对数损失'],[
                 [r.seed,NAMES[r.method],f'{r.accuracy*100:.2f}%',number(r.brier),number(r.log_loss)]
                 for r in seeds.sort_values(['seed','method']).itertuples()]),
             table(['范围','改变方向周数','原正确→新错误','原错误→新正确','上涨→非上涨','非上涨→上涨'],[
                 ['全部' if r.year=='all' else r.year,r.changed,r.correct_to_wrong,r.wrong_to_correct,r.up_to_non_up,r.non_up_to_up] for r in changes.itertuples()]),
             '![年度表现与固定概率分箱](annual_and_calibration.png)',
             '分类概率按预设五个等宽区间分箱。空箱保留；最后一箱包含概率 1。样本数量有限且存在时间相关，这些点只作描述，不能据此拟合新阈值或声称概率已可靠校准。',
             table(['概率区间','周数','平均预测概率','实际上涨频率'],[
                 [f'[{r.lower:.1f}, {r.upper:.1f}'+(']' if r.bin_index==4 else ')'),r.n,number(r.mean_probability),number(r.observed_frequency)]
                 for r in bins[bins.method.eq('direction_bce')].itertuples()]),
             '## 四项预设配对比较',
             '四项统一表达为“候选损失−参考损失”，负值表示改善。方向误判率 = 1−准确率，表中该项单位为百分点；另外两项使用原始 Brier 和对数损失单位。对同一周作配对，采用长度 8 的循环观测块、10,000 次 bootstrap、种子 20260910、中心化双侧检验，并在完整四项指标族内做 Holm 校正。']
    rows=[]
    for r in primary:
        scale=100 if r['metric']=='direction_error' else 1;digits=2 if scale==100 else 6
        name={'direction_error':'方向误判率（百分点）','brier':'Brier','log_loss':'对数损失'}[r['metric']]
        fmt=lambda v:f'{v*scale:+.{digits}f}'
        rows.append([NAMES[r['candidate']]+' − '+NAMES[r['reference']],name,fmt(r['difference']),
            f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"])
    text += [table(['配对比较','指标','差值','描述性 95% 区间','双侧 p','Holm p'],rows),
             '百分位区间不必与中心化检验互为反演。检验固定已训练模型的预测，没有重新训练神经网络，且未覆盖多轮研究反复查看这些历史日期带来的选择影响。因此，局部统计结果仍不能作为独立确认。',
             table(['预设描述性条件','结果'],[[FLAGS[k],'通过' if v else '未通过'] for k,v in assessment['flags'].items()]),
             '本轮候选必须同时满足以上八项条件，才通过局部描述性筛选。通过筛选也不等于独立验证、可交易或自动替换研究基准。',
             '## 复核与文件',
             f"重放 9 个分类模型与 9 个原收益模型，核对 423 条分类 logit/概率和 423 条原收益预测，最大绝对预测误差 {verification['max_prediction_error']:.3e}。另外复核 9 个初始状态与共享张量指纹、180 轮样本排列和批次、优化器步数与随机状态、81 次最终训练损失，最大训练诊断差异 {verification['max_training_loss_error']:.3e}。",
             '独立核对概率集成规则、因果常数参考、方向混淆矩阵、逐对 AUROC、概率误差、15 个方法/概率分箱、四项配对检验及全部描述性条件。前 2,415 个研究证据文件的 SHA-256 全部保持一致。',
             table(['材料','文件'],[
                 ['冻结研究方案','[protocol.json](../protocol.json)'],
                 ['因果缓存、代码与旧证据','[preparation_manifest.json](preparation_manifest.json)'],
                 ['训练频率与标准化','[training_metadata.json](training_metadata.json)'],
                 ['模型、预算与训练记录','[models.json](models.json)、[training_budgets.csv](training_budgets.csv)、[training_curves.csv](training_curves.csv)'],
                 ['固定状态训练诊断','[training_loss_draws.csv](training_loss_draws.csv)、[training_summary.csv](training_summary.csv)'],
                 ['完整集成与种子预测','[all_ensemble_predictions.csv](all_ensemble_predictions.csv)、[all_seed_predictions.csv](all_seed_predictions.csv)'],
                 ['概率分箱与方向变化','[reliability_bins.csv](reliability_bins.csv)、[direction_changes.csv](direction_changes.csv)'],
                 ['四项检验与筛选','[primary_comparisons.json](primary_comparisons.json)、[assessment.json](assessment.json)'],
                 ['完整复核','[verification.json](verification.json)'],
                 ['可编辑矢量图','[方向与概率 SVG](direction_and_probability.svg)、[年度与校准 SVG](annual_and_calibration.svg)']]),
             '本轮不涉及新增行情、实盘或交易收益。原方法完整的 78 指数清单、P0 生成规则及若干接口细节仍不齐备；本轮是既有 raw OHLCV 研究系统上的分类目标对照，不能称为原论文报告指标的完整复现。']
    report='\n\n'.join(text)+'\n';(OUT/'第十四轮测试报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(status='WRITTEN',report_characters=len(report)),ensure_ascii=False))

if __name__=='__main__':main()

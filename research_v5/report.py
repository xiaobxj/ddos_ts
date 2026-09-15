"""Standalone Chinese diagnostic report and exportable scientific figures."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
VARIANTS=['return_zero','return_nonzero','joint_ohlcv']
NAMES={'return_zero':'原收益目标／零初始化','return_nonzero':'收益目标／非零初始化','joint_ohlcv':'收益＋OHLCV 辅助目标','mlp':'展平 MLP'}
REPS={'raw':'原始 OHLCV','bits':'20 维量化坐标','continuous':'20 维连续坐标'}
ENGLISH={'return_zero':'Return / zero head','return_nonzero':'Return / nonzero head','joint_ohlcv':'Return + OHLCV auxiliary','mlp':'Flattened MLP'}
COLORS={'return_zero':'#286bb2','return_nonzero':'#ce7529','joint_ohlcv':'#128477','mlp':'#9467bd'}


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(x):return f'{x*100:.2f}%'
def md(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)


def figures(cap,curves,metrics,base):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained',sharey=True)
    for ax,rep,title in zip(axes,['raw','bits','continuous'],['Raw OHLCV','Quantized 20D','Continuous prequantization 20D']):
        for variant in VARIANTS+['mlp']:
            g=curves[(curves.representation==rep)&(curves.variant==variant)&(curves.label_kind=='real')]
            summary=g.groupby('step').return_mse.agg(['median','min','max'])
            ax.plot(summary.index,np.maximum(summary['median'],1e-9),color=COLORS[variant],label=ENGLISH[variant],linewidth=1.7)
            ax.fill_between(summary.index,np.maximum(summary['min'],1e-9),np.maximum(summary['max'],1e-9),color=COLORS[variant],alpha=.12)
        ax.axhline(.1,color='#777777',linestyle='--',linewidth=1)
        ax.set(title=title,xlabel='Full-batch optimization step',yscale='log',ylim=(1e-9,3.))
    axes[0].set_ylabel('In-sample standardized return MSE')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9,frameon=False)
    fig.suptitle('Capacity diagnostic only | 32 real historical windows, 3 training seeds\nLines: seed median; bands: seed range; dashed line: prespecified MSE threshold',fontsize=13)
    fig.savefig(OUT/'capacity_learning.png',dpi=180);fig.savefig(OUT/'capacity_learning.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for variant in VARIANTS:
        g=metrics[metrics.variant==variant].sort_values('epoch')
        axes[0].plot(g.epoch,g.rmse,marker='o',color=COLORS[variant],label=ENGLISH[variant])
        axes[1].plot(g.epoch,g.within_fold_forecast_std*10000,marker='o',color=COLORS[variant],label=ENGLISH[variant])
    mean=next(r['rmse'] for r in base if r['method']=='training_mean')
    axes[0].axhline(mean,color='#666666',linestyle='--',label='Mature training mean')
    axes[0].axhline(read('frozen_round4_reference.json')['rmse'],color='#9467bd',linestyle=':',
                    label='Round-4 frozen ref. (old train set)')
    axes[0].set(ylabel='Execution-return RMSE',xlabel='Training epochs',title='Chronological validation error')
    axes[1].set(ylabel='Within-fold forecast standard deviation (bp)',xlabel='Training epochs',title='Forecast variation after removing fold means')
    for ax in axes:ax.set_xticks([2,5,10,20]);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('2018–2020 raw-only validation | three-seed mean forecasts\nHistorical candidate selection, not an untouched confirmation set',fontsize=13)
    fig.savefig(OUT/'validation_comparison.png',dpi=180);fig.savefig(OUT/'validation_comparison.svg');plt.close(fig)


def main():
    audit=read('verification.json');assert audit['status']=='PASS'
    selection=read('selection.json');findings=read('findings.json')
    capacity=pd.read_csv(OUT/'capacity_aggregated.csv');curves=pd.read_csv(OUT/'capacity_curves.csv')
    metrics=pd.read_csv(OUT/'validation_metrics.csv');yearly=pd.read_csv(OUT/'validation_yearly_metrics.csv')
    training=pd.read_csv(OUT/'validation_training_curves.csv');pairs=read('paired_comparisons.json')
    figures(capacity,curves,metrics,selection['baseline_metrics'])
    captable=md(['表示','模型／目标','训练标签','通过种子数','最终 MSE 中位数','最终 MSE 范围'],
        [[REPS[r.representation],NAMES[r.variant],'真实' if r.label_kind=='real' else '打乱',f'{r.passes}/{r.runs}',
          f'{r.median_final_mse:.6g}',f'[{r.minimum_final_mse:.6g}, {r.maximum_final_mse:.6g}]'] for r in capacity.itertuples()])
    valtable=md(['训练设置','轮数','验证准确率','验证 RMSE','相对训练均值 MSE 改善','折内预测 SD／bp'],
        [[NAMES[r.variant],r.epoch,pct(r.accuracy),f'{r.rmse:.6f}',pct(r.mse_skill_vs_training_mean),f'{r.within_fold_forecast_std*10000:.2f}']
          for r in metrics.itertuples()])
    pairtable=md(['比较','相同轮数','ΔRMSE','95% 区间','MSE Holm p'],
        [[NAMES[r['variant']]+' − 原收益目标',r['epoch'],f"{r['rmse']['difference']:.6f}",
          f"[{r['rmse']['ci95_low']:.6f}, {r['rmse']['ci95_high']:.6f}]",f"{r['mse']['holm_adjusted_p']:.4f}"] for r in pairs])
    rows=[]
    for variant in VARIANTS:
        for epoch in [5,20]:
            g=training[(training.variant==variant)&(training.epoch==epoch)]
            m=metrics[(metrics.variant==variant)&(metrics.epoch==epoch)].iloc[0]
            rows.append([NAMES[variant],epoch,f'{g.final_model_train_return_mse.mean():.4f}',
                         f'{m.mse:.7f}',f'{m.reassigned_mse:.7f}',f'{m.reassignment_mse_increase:.7f}'])
    learntable=md(['训练设置','轮数','最终模型训练标准化 MSE 均值','原样本验证 MSE','重配输入验证 MSE','重配后 MSE 增量'],rows)
    selected=selection['selected'];chosen=yearly[(yearly.variant==selected['variant'])&(yearly.epoch==selected['epoch'])]
    yeartable=md(['年份','周数','准确率','RMSE','相对训练均值 MSE 改善'],
                [[r.year,r.n,pct(r.accuracy),f'{r.rmse:.6f}',pct(r.mse_skill_vs_training_mean)] for r in chosen.itertuples()])
    content=report_text(audit,selection,findings,captable,valtable,pairtable,learntable,yeartable)
    (OUT/'第五轮测试报告.md').write_text(content,encoding='utf-8')
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',content):
        if target.startswith(('https://','http://')):continue
        assert (OUT/target).resolve().exists(),target
        links.append(target)
    files={}
    for p in ROOT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json':
            with p.open('rb') as stream:files[str(p.relative_to(ROOT)).replace('\\','/')]=hashlib.file_digest(stream,'sha256').hexdigest()
    (OUT/'delivery_manifest.json').write_text(json.dumps(dict(report='results/第五轮测试报告.md',
        report_links_checked=len(links),files=files),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(report='research_v5/results/第五轮测试报告.md',characters=len(content),links=len(links),hashed_files=len(files)),ensure_ascii=False))


def report_text(audit,selection,findings,captable,valtable,pairtable,learntable,yeartable):
    conclusions='\n\n'.join(findings['conclusions']);directions='\n\n'.join(findings['directions'])
    selected=selection['selected'];base=selection['baseline_metrics']
    prior=read('frozen_round4_reference.json')
    mean=next(r for r in base if r['method']=='training_mean');zero=next(r for r in base if r['method']=='zero_return')
    capacity=read('capacity_manifest.json');validation=read('validation_manifest.json')
    folds=read('validation_folds.json')
    foldtable=md(['训练截止日','成熟日度训练样本','最晚训练标签完成日','验证信号范围','验证周数'],
        [[r['cutoff'],r['train_n'],r['last_train_label'],r['first_signal']+' 至 '+r['last_signal'],r['test_n']] for r in folds])
    return f'''# 第五轮方法测试：训练能力、连续表示与辅助预测目标

实验日期：2026-09-10。研究目录：research_v5。本轮完成 45 次真实／打乱标签的小样本拟合，以及 27 次按时间先后进行的训练验证；保存并复验全部 153 个模型检查点。

## 主要判断

{conclusions}

本轮回答“第四轮的近似常数预测是否来自拟合能力或训练目标问题”。小样本拟合、历史验证、未来确认属于不同证据：能够记住训练标签不是可预测收益的证明；2018–2020 年也已经参与前几轮研究，因此本轮验证仍是探索性候选筛选。预训练编码器没有进入早年的预测验证。

## 第一层：真实窗口拟合能力

从截至 2017-12-31 已完成标签的合格样本中，按时间排序等距抽取 32 个窗口，不根据涨跌或收益大小选择。每个窗口含 125 根 K 线，固定为 25 个五日片段。所有容量测试使用同一批窗口、400 次全批量更新、学习率 0.001、梯度裁剪 1.0，并关闭 dropout 和 weight decay。这是检查能否拟合输入的受控条件，不等同于第四轮大样本训练配置。

原始数值是沿用第四轮的窗口内标准化 log OHLC、log1p 成交量。量化表示是官方 Kronos 的 20 维双极坐标；连续表示取同一冻结编码器在取符号之前、L2 归一化后的 20 维向量。两者每根 K 线向量范数都为 1，避免同时改变维度和总幅度。全体 3,785 个原窗口的 9,462,500 个坐标符号一致；连续范数最大误差约 1.79e-7。

预训练编码器包含晚于这些样本的训练数据。这里使用它，只检验既定输入能否被下游网络拟合；不能把预训练表示的拟合结果称为 2017 年或更早可取得的预测能力。

比较三种 Crossformer 训练设置：

- **原收益目标／零初始化**：保留第四轮的收益读出层初始化和单一收益 MSE。
- **收益目标／非零初始化**：只把收益读出权重改为 gain=0.1 的 Xavier uniform 初始化，偏置仍为零。
- **收益＋OHLCV 辅助目标**：收益读出仍为零初始化，在收益 MSE 上增加 0.1 倍未来五日 OHLCV 的平均标准化 MSE。

每种输入另用一个展平输入、64 个 GELU 隐藏单元的 MLP 拟合真实及打乱收益标签。标签打乱使用唯一固定种子 20260914，且在表示与训练种子之间保持相同排列。MLP 是容量对照，参数量不要求与 Crossformer 相同。

预先规定的拟合通过标准是：第 400 步最终标准化收益 MSE ≤ 0.1，且收益相关性 ≥ 0.9。表中通过数是三个训练种子中满足该条件的数量，不使用中途最好检查点或最好种子替代最终结果。

{captable}

![真实样本拟合曲线](capacity_learning.png)

图中的线为三种子中位数，阴影为种子范围；为绘图可读性，低于 1e-9 的 MSE 画在该下界，原始数值完整保留。打乱标签控制单独列在表中，它检验记忆能力，不衡量预测泛化。

## 第二层：2018–2020 年时间顺序验证

本层只使用原始 OHLCV，避免预训练编码器对早年验证造成时间污染。三种训练设置都完整运行三个年度滚动区间和三个种子，训练至 20 轮并记录第 2、5、10、20 轮。容量测试表现没有改变候选列表、训练预算或任何阈值。

{foldtable}

全部训练样本的收益标签和辅助标签都已在对应截止日前完成；测试只用周五信号，最终标签还必须在 2020-12-31 前完成。候选结果由三个种子预测先取均值后计算，故种子平均准确率不一定等于这里的集成准确率。

{valtable}

同折成熟训练收益均值的验证 RMSE 为 **{mean['rmse']:.6f}**，零收益基准为 **{zero['rmse']:.6f}**。相对训练均值 MSE 改善大于零才代表误差更低；模型输出更多波动本身不是目标。

折内预测标准差先减去各年度模型自身的预测均值，再对残差合并计算，避免把年度重训造成的整体偏移误当成逐样本信号；全区间预测标准差和折间方差占比另存于指标文件。

预先固定的最小 RMSE 选择规则选出 **{NAMES[selected['variant']]}，{selected['epoch']} 轮**，准确率 {pct(selected['accuracy'])}，RMSE {selected['rmse']:.6f}。这是这 12 个候选中的验证集选择结果，包含选参偏差，不是独立确认的最优模型。

还必须保留第四轮已经选出的原始数值固定分段模型作为背景参照：在完全相同的 141 个验证日上，它的冻结 {prior['epochs']} 轮预测准确率为 **{pct(prior['accuracy'])}**、RMSE 为 **{prior['rmse']:.6f}**，仍优于本轮选中候选。本轮辅助目标要求使共享训练集多排除了五个早期样本，因此这不是用于归因某个结构变化的严格控制比较，但足以说明不能宣称已超越之前的验证结果。

![按时间验证的误差与预测幅度](validation_comparison.png)

选中候选的年度表现如下，各年样本数及市场变化不同，不能仅凭一个年度高分外推稳定收益。

{yeartable}

## 同预算比较与输入依赖诊断

主要统计只比较同为 5 轮或 20 轮时，非零初始化和辅助目标相对原收益目标的误差。使用长度为 8 个观察值的循环分块 bootstrap、10,000 次重复、种子 20260910，并对四项 MSE 比较做 Holm 校正。RMSE 差值小于零代表新增设置更好；95% 区间为各项边际区间。它们仍属于历史验证上的探索性统计。

{pairtable}

另外用每个已完成的最终检查点、关闭 dropout 后重算完整训练集损失，避免把逐批优化时的损失误当作最终模型误差。验证区间内还将整个输入窗口循环错配一位，保留窗口内容、几何和掩码，检查预测是否依赖与该标签对应的输入。这种错配没有参与模型选择，也不是可实施交易策略。

{learntable}

训练期标准化收益 MSE 的均值输出基准为 1；上表在九次“年度×种子”拟合之间取平均。预测随输入变化与预测变化有用是两个问题：错配后误差上升才提供与输入对应关系有关的诊断信息，而且需要结合原始验证误差和不确定性来判断。

从平方误差本身看，最佳的常数预测就是训练收益均值；标准化后这个常数为零。如果模型在当前数据与训练条件下没有学到更有用的条件差异，回到均值附近并不等于代码出错。需要判断的是它能否在后续时间样本上稳定超过这个简单解。

## 实现与因果口径

Crossformer 复用第四轮已经验证的 DSW 嵌入、时间／维度注意力、层次编码解码和掩码实现，尺寸保持 d_model=32、d_ff=64、4 heads、2 层编码器、router factor=4。每个变体都包含相同的 D→5 辅助线性头，以维持参数结构；原收益设置不计算辅助损失。原始数值模型各 138,471 个参数，20 维表示模型各 151,101 个参数。

辅助头采用独立随机数作用域，初始化完成后恢复 CPU 和 GPU 随机状态；非零收益读出也使用独立作用域。接口检查确认新增但未使用的辅助头不会改变原骨干参数、初始解码结果或后续随机数状态。零初始化下第一步纯收益损失不会把梯度传到骨干，这是链式求导的预期结果；辅助目标可在第一步传入骨干梯度。是否影响持续学习，需要看实际拟合与验证结果，不能仅根据第一步梯度判定故障。

收益标签沿用先前的“当前信号后的开盘到下一个同星期信号后的开盘”收益。辅助目标则为未来五根 K 线的 OHLCV：四个价格取相对当前收盘价的对数比，成交量取未来 log1p 成交量减去最近 20 根已观察成交量的 log1p 均值。25 个坐标只按成熟训练标签的均值和标准差缩放。总损失为标准化收益 MSE＋0.1×辅助坐标平均 MSE，没有把解码器输出冒充已训练的原文 OHLCAV 预测。

共享样本共 3,780 个，比第四轮多排除 5 个早期日度样本：本轮对未来辅助 OHLCV 使用严格的高低价包围检查，2010-07-20 的原始最低价 2685.460 略高于开盘价 2685.459，因而影响 2010-07-13 至 2010-07-19 的五个输入锚点。此前数据质量标记允许这种报价舍入差异。本轮保留原始数据，对全部候选使用同一排除规则；没有只给辅助目标模型减少训练样本。因此新的原收益基线会重新训练，不把它与第四轮预测冒充逐值相同。

验证阶段采用 AdamW、学习率 0.001、weight decay=0.01、dropout=0.1、batch 128、梯度范数裁剪 1.0；仅打乱成熟的训练样本。使用已有的独立 GPU 环境、确定性 float32、关闭 TF32、四个 CPU 线程，没有安装新依赖。

## 复验与文件

最终状态：**{audit['status']}**。四项接口检查通过；独立重算全部 {audit['auxiliary_target_coordinates_recomputed']} 个辅助目标坐标和收益标签，核对所有训练标签成熟日期、等距抽样和三种子集成。共回放 **{audit['real_and_permuted_capacity_checkpoints_replayed']} 个容量检查点**及 **{audit['validation_checkpoints_replayed']} 个验证检查点**，最大预测误差分别为 {audit['max_capacity_replay_error']:.3g}、{audit['max_validation_replay_error']:.3g}。源码、输入缓存、权重和检查点哈希均匹配，前四轮 {audit['old_evidence_files_preserved']} 个纳入保全的证据文件未改变。

容量测试运行 {capacity['elapsed_seconds']/60:.2f} 分钟，历史验证运行 {validation['elapsed_seconds']/60:.2f} 分钟；不含数据准备、接口检查、统计和复验。两个阶段完整运行，结论文字写在训练结束后，未反向影响配置或预测。

- [冻结协议](../protocol.json)、[执行说明](../README.md)、[输入准备清单](preparation_manifest.json)、[表示核验](representation_verification.json)。
- [容量测试全部结果](capacity_summary.json)、[容量训练曲线](capacity_curves.csv)、[固定真实样本](capacity_samples.csv)、[真实及打乱标签预测](capacity_predictions.csv)。
- [验证候选指标](validation_metrics.csv)、[种子指标](validation_seed_metrics.csv)、[年度指标](validation_yearly_metrics.csv)、[逐条种子预测](validation_seed_predictions.csv)、[集成预测](validation_ensemble_predictions.csv)。
- [训练损失与错配诊断](validation_training_curves.csv)、[主要配对统计](paired_comparisons.json)、[选择结果](selection.json)。
- [第四轮冻结验证参照](frozen_round4_reference.json)、[重叠训练标签诊断](training_label_overlap.json)、[小样本达到误差阈值的步数](capacity_learning_speed.csv)。
- [最终复验](verification.json)、[测试日志](test_results.txt)、[容量检查点清单](capacity_checkpoint_manifest.json)、[验证检查点清单](validation_checkpoint_manifest.json)。
- 矢量图：[容量学习曲线 SVG](capacity_learning.svg)、[历史验证 SVG](validation_comparison.svg)。交付文件哈希保存在同目录 delivery_manifest.json。

## 后续判断

{directions}

结构和表示的原始定义参照 [Crossformer 固定官方版本](https://github.com/Thinklab-SJTU/Crossformer/tree/c10c8eadb153d1dd9798250967747ca3ebb81383)、[Kronos tokenizer 官方实现](https://github.com/shiyu-coder/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/model/kronos.py) 及 [BSQuantizer 官方实现](https://github.com/shiyu-coder/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/model/module.py)。本轮改动是本地诊断实验，数值均来自保存的本地计算。
'''


if __name__=='__main__':main()

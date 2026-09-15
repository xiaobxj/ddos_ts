"""Render a Chinese research report and static figures from verified results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
LABELS={'native_mse':'原始 MSE','learned_probe':'学习特征原版','learned_clip':'学习特征截尾','raw25_probe':'简单特征原版','raw25_clip':'简单特征截尾','training_frequency':'训练频率','neutral_50':'固定 50%'}
WINDOWS={'early_2015_2017':'2015—2017','late_2018_2020':'2018—2020','pooled_2015_2020':'2015—2020 合并'}
STATES={'low_trend__low_vol':'低趋势 × 低波动','low_trend__high_vol':'低趋势 × 高波动','mid_trend__low_vol':'中趋势 × 低波动',
    'mid_trend__high_vol':'中趋势 × 高波动','high_trend__low_vol':'高趋势 × 低波动','high_trend__high_vol':'高趋势 × 高波动'}
ORDER=['native_mse','learned_probe','learned_clip','raw25_probe','raw25_clip','training_frequency','neutral_50']
def pct(v):return '—' if pd.isna(v) else f'{v*100:.2f}%'
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def savefig(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)

def figures(metrics,states,distribution,tails):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.5,7.2),layout='constrained')
    for col,w in enumerate(['early_2015_2017','late_2018_2020']):
        t=metrics[metrics.window.eq(w)].set_index('method')
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col];scale=100 if key=='accuracy' else 1
            for xs,methods,color in [([0,1],['learned_probe','learned_clip'],'#2868a0'),([3,4],['raw25_probe','raw25_clip'],'#be6d2a')]:
                ys=[t.loc[m,key]*scale for m in methods];ax.plot(xs,ys,'o-',lw=1.8,color=color,markersize=7)
                for x,y in zip(xs,ys):ax.annotate(f'{y:.2f}%' if key=='accuracy' else f'{y:.4f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10,color=color)
            ax.axhline(t.loc['training_frequency',key]*scale,color='#497b47',ls='--',lw=1.2,label='训练频率')
            if key=='accuracy':ax.axhline(t.loc['native_mse',key]*scale,color='#805382',ls=':',lw=1.5,label='原始 MSE')
            ax.set_xticks([0,1,3,4],['学习：原版','学习：截尾','简单：原版','简单：截尾']);ax.set_xlim(-.6,4.6);ax.grid(axis='y',alpha=.15)
            ax.set_ylabel('方向准确率（%）' if key=='accuracy' else 'Brier（越低越好）')
            ax.set_ylim((38,69) if key=='accuracy' else (.230,.313))
            if row==0:ax.set_title(f"{WINDOWS[w]}，n={int(t.loc['native_mse','n'])}",fontweight='bold')
    axes[0,0].legend(loc='upper left',ncols=2,frameon=False,fontsize=9)
    fig.suptitle('固定 1%—99% 截尾：两段时期分别比较',fontsize=15)
    savefig(fig,'clipping_comparison')
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.7),layout='constrained');cmap=plt.colormaps['RdBu'].copy();cmap.set_bad('#ececec')
    methods=['native_mse','learned_clip','raw25_clip'];order=list(STATES)
    for ax,w in zip(axes,['early_2015_2017','late_2018_2020']):
        g=states[states.window.eq(w)&states.partition.eq('trend_volatility')]
        acc=g.pivot(index='state',columns='method',values='accuracy').loc[order];n=g[g.method.eq('native_mse')].set_index('state').loc[order,'n'].to_numpy()
        data=np.column_stack([(acc[m]-acc.training_frequency).to_numpy()*100 for m in methods]);masked=np.ma.array(data,mask=np.repeat((n<20)[:,None],3,axis=1))
        im=ax.imshow(masked,cmap=cmap,norm=colors.TwoSlopeNorm(vmin=-30,vcenter=0,vmax=30),aspect='auto')
        for i in range(6):
            for j in range(3):ax.text(j,i,f'{data[i,j]:+.1f}',ha='center',va='center',color='#777777' if n[i]<20 else ('white' if abs(data[i,j])>18 else '#222222'),fontsize=12)
        ax.set_xticks(range(3),['原始 MSE','学习特征截尾','简单特征截尾']);ax.set_yticks(range(6),[f'{STATES[s]}  n={nn}'+(' *' if nn<20 else '') for s,nn in zip(order,n)],fontsize=9)
        ax.set_title(WINDOWS[w],fontweight='bold');ax.tick_params(length=0)
    fig.colorbar(im,ax=axes,shrink=.75,label='相对训练频率基线的准确率差（百分点）')
    fig.suptitle('状态诊断：灰色格为 n<20；颜色不代表统计显著',fontsize=14)
    savefig(fig,'state_diagnostics')
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5),layout='constrained')
    d=distribution[distribution.partition.eq('volatility')&distribution.state.eq('high_vol')].sort_values('year');x=np.arange(6)
    axes[0].bar(x,d.validation_fraction*100,color='#397caf',width=.6)
    axes[0].plot(x,d.training_fraction*100,color='#547745',ls='--',label='各年训练期占比')
    totals=distribution[distribution.partition.eq('volatility')].groupby('year').validation_n.sum()
    for i,r in enumerate(d.itertuples()):axes[0].text(i,r.validation_fraction*100+3,f'{r.validation_n}/{int(totals.loc[r.year])}',ha='center',fontsize=10)
    axes[0].set(xticks=x,xticklabels=d.year.tolist(),ylim=(0,112),ylabel='高波动状态占比（%）',title='保留周样本的市场状态确实发生变化');axes[0].legend(frameon=False,loc='upper right',fontsize=9)
    t=tails[tails.method.eq('raw25_clip')].groupby('cutoff')[['validation_lower_fraction','validation_upper_fraction']].mean()
    axes[1].bar(x,t.validation_lower_fraction*100,color='#477ea8',label='低于训练期第 1 百分位',width=.6)
    axes[1].bar(x,t.validation_upper_fraction*100,bottom=t.validation_lower_fraction*100,color='#ce8642',label='高于训练期第 99 百分位',width=.6)
    for i,r in enumerate(t.itertuples()):axes[1].text(i,(r.validation_lower_fraction+r.validation_upper_fraction)*100+.8,f'{100*(r.validation_lower_fraction+r.validation_upper_fraction):.1f}%',ha='center',fontsize=10)
    axes[1].set(xticks=x,xticklabels=d.year.tolist(),ylim=(0,30),ylabel='简单特征坐标触发截尾的比例（%）',title='分布偏移既可能在上尾，也可能在下尾');axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('状态与分布变化；2015 年仅保留 22 周，右图为评分后描述',fontsize=14)
    savefig(fig,'market_distribution_shift')

def main():
    v=read(OUT/'verification.json');assert v['status']=='PASS';p=read(ROOT/'protocol.json');fitting=read(OUT/'training_manifest.json')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv');yearly=pd.read_csv(OUT/'yearly_metrics.csv');seeds=pd.read_csv(OUT/'seed_metrics.csv');states=pd.read_csv(OUT/'state_metrics.csv')
    distribution=pd.read_csv(OUT/'state_distribution.csv');tails=pd.read_csv(OUT/'posthoc_tail_details.csv');clipping=pd.read_csv(OUT/'clipping_summary.csv')
    pairs=read(OUT/'primary_comparisons.json');assessments=read(OUT/'assessments.json');thresholds=read(OUT/'state_thresholds.json')
    figures(metrics,states,distribution,tails)
    lines=['# 第十七轮方法测试：训练期截尾与因果市场状态识别',
        '', '本轮已经完成 24 个新分类头、1,044 条新概率预测和市场状态诊断。结论是：固定截尾缓解了部分简单特征的概率偏差，但尚未获得稳定的跨时期提升。市场状态可以在预测时因果识别，并帮助解释样本分布变化；本轮没有证据支持直接按六种状态切换模型。',
        '', '2018—2020 年，学习特征截尾后的准确率为 **56.03%（79/141）**，简单特征为 **53.90%（76/141）**，分别比各自原版多预测对 2 周。原始 MSE 仍为 **59.57%（84/141）**，训练频率为 **56.03%（79/141）**。2015—2017 年，两种截尾模型的准确率都没有提高。',
        '', '全部 261 个测试日期在前几轮已经被查看。本轮设计承接上一轮发现，属于重复使用历史样本的探索性研究；年度训练与预测符合因果顺序，但这些结果**不能作为独立确认**，也不能由方向准确率推断可交易收益。',
        '', '## 1. 冻结范围与因果约束', '',
        '数据、无效 OHLC 排除规则、125 根历史输入、下一开盘至下一周对应开盘的标签，以及每个年度的训练截止日全部沿用第十六轮。训练行必须满足联合标签完成时间不晚于截止日。市场指标只使用信号日收盘时及以前的数据，交易标签从次日开盘开始。2015 年受既有数据排除规则影响，只保留 22 周；后面关于 2015 年的描述均仅指这些保留样本。', '',
        table(['训练截止日','训练行数','下一年测试周数'],[[f['cutoff'],f['train_n'],f['test_n']] for f in p['folds']]), '',
        '学习特征来自原有 18 个 MSE20 神经网络的 25 维表示；简单特征沿用 25 个固定 OHLCV 汇总量。对每个分类头的每一维，使用本年度训练数据的第 1、99 百分位数截尾，再用截尾后的训练均值和总体标准差标准化，重新求解相同的岭逻辑回归。正则系数 λ=0.01，截距不惩罚，严格以 p>0.5 预测上涨，未搜索截尾比例、正则强度或阈值。', '',
        '这是“训练/未来特征截尾、重新标准化、重新拟合分类头”这一整套固定处理的比较，不能单独把差异归因于预测阶段截尾。学习特征每个年度沿用三个种子并平均概率；简单特征每年度只有一个确定性分类头。18 个神经网络均未重训。24 个新分类头全部拟合完成后才开始新的测试评分；合计 90 次 Newton 更新。', '',
        '## 2. 市场状态如何识别', '',
        table(['描述量','计算规则'],[
            ['趋势得分','过去 60 个交易日的累计收盘对数收益 ÷（60 日单日对数收益总体标准差 × √60）；标准差下限 1e-8。'],
            ['波动','最近 20 个交易日的单日收盘对数收益总体标准差。'],
            ['振幅','最近 20 日 log(最高价/最低价) 的均值。'],
            ['量能变化','log(1+当日成交量) − log(1+20 个交易日前成交量)。']]), '',
        '每个年度只用该年度训练输入确定阈值，不读其未来涨跌标签：趋势按训练期三分位数分成低、中、高，波动按训练期中位数分为低、高，交叉得到六种主要状态。振幅与量能变化分别按训练中位数分成两组，作为补充诊断。边界相等时归较低组。全部阈值在下一年度内固定。', '',
        '“低/中/高趋势”是相对于本年度训练分布的得分，不等同于事后牛市、熊市标签；“高量能变化”也不等同于绝对成交量大。状态没有进入本轮预测函数，也没有被用来删样本、调整阈值、重新加权或选择模型。', '',
        table(['训练截止日','趋势下界','趋势上界','日波动中位数','振幅中位数','量能变化中位数'],[[t['cutoff'],num(t['trend_lower']),num(t['trend_upper']),num(t['volatility_median']),num(t['range_median']),num(t['volume_median'])] for t in thresholds]), '',
        '## 3. 总体及逐年效果', '', 'Brier 和对数损失越低越好；AUROC 反映排序能力。原始 MSE 输出收益分数，没有概率指标。']
    for w in ['early_2015_2017','late_2018_2020']:
        t=metrics[metrics.window.eq(w)].set_index('method');lines += ['',f'### {WINDOWS[w]}（{int(t.iloc[0].n)} 周）','',
            table(['方法','正确周数','准确率','平衡准确率','AUROC','Brier','对数损失'],[[LABELS[m],int(t.loc[m,'correct_directions']),pct(t.loc[m,'accuracy']),pct(t.loc[m,'balanced_accuracy']),num(t.loc[m,'auroc']),num(t.loc[m,'brier']),num(t.loc[m,'log_loss'])] for m in ORDER])]
    lines += ['', '![固定截尾对比](clipping_comparison.png)', '',
        '后期学习特征准确率增加 1.42 个百分点，但 AUROC 从 0.592487 到 0.592283，Brier 仅从 0.242043 到 0.241944；这不足以表明排序能力提高。它只改变了 2 个方向，且两次都由错误变正确。简单特征后期改变 8 个方向，其中 5 个由错变对、3 个由对变错。早期学习特征一个方向都没变，简单特征改变 8 个方向，正负改善各 4 个。', '',
        '逐年方向准确率如下；完整年度概率指标见 [yearly_metrics.csv](yearly_metrics.csv)。', '',
        table(['年份','周数']+[LABELS[m] for m in ORDER[:-1]],[[year,int(g.iloc[0].n)]+[pct(g.set_index('method').loc[m,'accuracy']) for m in ORDER[:-1]] for year,g in yearly.groupby('year')]), '',
        '学习特征截尾后的种子结果如下。三个种子共享相同市场样本，不能把它们当作三份独立市场证据。简单特征不存在随机种子重复。', '',
        table(['时期','种子','准确率','Brier','对数损失'],[[WINDOWS[r.window],r.seed,pct(r.accuracy),num(r.brier),num(r.log_loss)] for r in seeds[seeds.method.eq('learned_clip')&seeds.window.ne('pooled_2015_2020')].itertuples()]), '',
        '合并 261 周只作描述：原始 MSE 为 54.79%（143/261）；学习特征截尾为 54.41%（142/261），其原版为 53.64%；简单特征截尾为 49.43%（129/261），其原版为 48.66%；训练频率为 54.41%。合并结果不能替代两段时期分别通过。', '',
        '## 4. 固定检验与判定', '',
        '本轮预先固定 12 项对比，均为“候选损失 − 参照损失”，负数更好。沿用 8 个保留观测为一块的循环区块重采样，10,000 次、种子 20260910；两个时期分别重采样，12 个 p 值统一做 Holm 校正。由于日期有排除缺口，8 个保留观测不一定是连续 8 个自然周。区间及 p 值不包含模型/设计选择和历轮反复查看数据带来的不确定性。', '',
        table(['时期','候选/参照','指标','差值','95% 区间','原始 p','Holm p'],[[WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"] for r in pairs]), '',
        '没有一项改善通过本轮 Holm 校正。唯一校正后 p<0.05 的差异是**早期简单特征截尾的 Brier 仍劣于训练频率**：差值 +0.033837，Holm p=0.0180。这个结果也仅限上述探索性检验框架，不是整个研究流程的独立显著性结论。', '',
        '预设的八项描述性条件涵盖：准确率胜过自身原版、原始 MSE、训练频率；Brier 胜过自身原版及训练频率；对数损失胜过训练频率；至少两年准确率胜过 MSE、至少两年 Brier 胜过训练频率。两段时期都需八项全通过，平局算未通过。', '',
        table(['时期','候选','通过项数','跨时期通过'],[[WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/8",'是' if a['cross_period_descriptive_pass'] else '否'] for a in assessments]), '',
        '## 5. 市场状态诊断带来的信息', '',
        '最清晰的是分布变化：2015 年保留的 22 周中，21 周（95.45%）位于训练中位数以上的高波动组；2016 年为 11/48 周（22.92%）；2017 年为 0/50 周，全部位于训练定义的低波动组。训练期自身高低波动约各半。因此，简单地沿用全训练期的特征范围确实会遇到明显的后续分布偏移。', '',
        '![市场状态与尾部分布](market_distribution_shift.png)', '',
        '以下展示全部六种状态，星号表示本时期 n<20。两段时期共 12 个“时期 × 状态”单元，有 **5 个稀疏单元**，不能据此挑选最优专家。表内结果是原样诊断，没有用它们重新拼接一条看起来更好的预测序列。']
    for w in ['early_2015_2017','late_2018_2020']:
        g=states[states.window.eq(w)&states.partition.eq('trend_volatility')];rows=[]
        for s,label in STATES.items():
            t=g[g.state.eq(s)].set_index('method');n=int(t.loc['native_mse','n'])
            rows.append([label+(' *' if n<20 else ''),n,pct(t.loc['native_mse','observed_up_fraction'])]+[pct(t.loc[m,'accuracy']) for m in ['native_mse','learned_clip','raw25_clip','training_frequency']])
        lines += ['',f'### {WINDOWS[w]}：状态内准确率','',table(['状态','周数','实际上涨占比','原始 MSE','学习截尾','简单截尾','训练频率'],rows)]
    lines += ['', '![市场状态诊断](state_diagnostics.png)', '',
        '高波动状态下，原始 MSE 在早期为 43.75%（14/32），低波动为 51.14%（45/88）；后期高波动为 60.29%（41/68），低波动为 58.90%（43/73）。这不是一个可以直接照搬的“高波动就停用模型”规则。不同分组的实际涨跌比例也不同，准确率差异本身不能证明状态具有预测因果作用。', '',
        '学习特征截尾在早期高波动组的 Brier 为 0.288869，训练频率为 0.250680；后期分别为 0.247568 与 0.250876。其相对表现也随时期变化。后期“低趋势 × 低波动”中的原始 MSE 达到 81.25%，但只有 16 周，已按事先规则标记为稀疏，不能据此承诺高准确率。', '',
        '振幅和量能变化的二分组也全部保留在 [state_metrics.csv](state_metrics.csv)。例如，后期高量能变化组的原始 MSE 准确率为 65.71%（46/70），早期同名组为 50.98%（26/51）。这些只是多个诊断切面中的描述；没有进行分组显著性检验，也没有据此选取交易状态。', '',
        '## 6. 为什么截尾没有解决问题', '',
        '2015 年简单特征有 23.64% 的测试坐标触发截尾，22 个周样本每一个都有坐标被截尾；训练期坐标触发比例约为 2.04%。用截尾训练统计量衡量的每行最大绝对标准化距离，均值从 6.63 降至 2.82。平均上涨概率从 30.45% 回到 39.71%，但实际上涨比例为 59.09%，准确率仍为 9/22。其 Brier 从 0.331839 改善为 0.305033，AUROC 却从 0.529915 降为 0.470085：缓和概率极端程度不等同于提高排序能力。', '',
        '逐折截尾比例如下。学习特征取三个种子比例的均值，简单特征取确定性单头比例。坐标比例的分母是“周数 ×25”，不是独立样本数。', '',
        table(['测试年','特征','训练坐标截尾比例','测试坐标截尾比例','测试行至少一维截尾比例','原版平均概率','截尾平均概率'],[[int(cutoff[:4])+1,LABELS[method],pct(g.training_coordinate_clip_fraction.mean()),pct(g.validation_coordinate_clip_fraction.mean()),pct(g.validation_row_clip_fraction.mean()),pct(g.original_mean_probability.mean()),pct(g.clipped_mean_probability.mean())] for (method,cutoff),g in clipping.groupby(['method','cutoff'])]), '',
        '评分后额外进行的上下尾描述性核算发现：2017 年简单特征触发截尾的 17.84% 坐标全部来自下尾；120 日波动、120 日平均振幅在保留的 50 周中均低于该折训练第 1 百分位，因此这两维在测试期被压成各自相同的下边界。它说明偏移也可以表现为持续低波动。这个核算选在评分之后，只解释固定结果，没有再训练、再预测或修改参数。详见 [posthoc_tail_details.csv](posthoc_tail_details.csv) 与 [provenance](posthoc_tail_provenance.json)。', '',
        '截尾会限制模型在历史范围以外的外推，也可能抹掉新状态内部的差异。本轮不足以把失效完全归因于特征偏移；表示质量、标签噪声和跨年份关系变化仍可能同时存在。', '',
        '## 7. 后续改进方向', '',
        '下一步更适合测试**共享模型的市场条件输入**：将这四个连续、因果的状态指标加入现有分类头，先检验统一模型是否从这些条件获得额外信息。只使用少量预先限定的加性项；若要引入交互，也应先限定形式和数量，再完成整轮评分。当前六状态的样本分布不足以支撑直接挑选或分别训练六套专家。', '',
        '比较中应保留原有模型、训练频率，以及单独使用市场特征的简单参照，以辨别收益来自状态本身还是与学习表示的互补。阈值和标准化继续只用年度训练期数据，两个时期分别验收；这些复用日期上的进一步结果仍属于探索，最终需要未用于选型的数据或后续前瞻预测检验。', '',
        '本轮不升级研究基线，不宣称交易收益。可保留状态识别与分布诊断代码，作为后续条件输入实验的可复核基础。', '',
        '## 8. 复核与证据', '',
        f"独立复核状态为 **{v['status']}**。分位数边界用排序后的次序统计量重算；状态公式与逐行实现比较，并验证扰动未来数据和截断未来历史不改变当时结果。24 个分类头均用独立 L-BFGS-B 目标/梯度重求解，18 个原有神经网络的训练和测试特征全部回放，权重不变。", '',
        table(['复核内容','结果'],[
            ['新分类头 / Newton 更新',f"24 / {fitting['new_newton_iterations']}"],['独立 L-BFGS-B 验证','24 个解'],['神经网络回放 / 新训练步数','18 / 0'],
            ['学习特征训练行 / 测试行','28,395 / 783'],['简单特征训练行 / 测试行','9,465 / 261'],['市场状态训练行 / 测试日期','9,465 / 261'],
            ['新增概率 / 沿用模型记录','1,044 / 1,827'],['模型记录 / 集成记录','2,871 / 1,827'],['独立复算指标组 / 状态指标组','258 / 168'],['概率分箱 / 主要对比','60 / 12'],
            ['最大新预测复核误差',f"{v['maximum_new_forecast_error']:.3e}"],['独立求解最大目标差',f"{v['maximum_alternate_objective_gap']:.3e}"],
            ['独立求解最大训练概率差',f"{v['maximum_alternate_probability_gap']:.3e}"],['独立求解最大梯度',f"{v['maximum_alternate_gradient_inf']:.3e}"],['历史证据文件保留','2,716 个，哈希不变'],['独立保留测试日期','0']]), '',
        '训练分数仅为监督训练集内的诊断：按折平均，学习特征截尾的训练准确率约 60.32%—62.89%，简单特征约 55.90%—61.00%。这些每日滚动标签相互重叠，训练行数也不是同等数量的独立周事件。训练指标见 [training_metrics.csv](training_metrics.csv)。所有新概率的报告截断计数为 0。', '',
        f"冻结协议 SHA-256：`{v['protocol_sha256']}`。", '',
        '主要文件：[协议](../protocol.json)；[总体指标](ensemble_metrics.csv)；[逐年指标](yearly_metrics.csv)；[种子指标](seed_metrics.csv)；[主要对比](primary_comparisons.json)；[八项判定](assessments.json)；[状态阈值](state_thresholds.json)；[逐周状态](validation_states.csv)；[状态分布](state_distribution.csv)；[状态指标](state_metrics.csv)；[截尾汇总](clipping_summary.csv)；[独立求解](independent_solver_verification.csv)；[完整复核](verification.json)。', '',
        '三张图均同时保存为 PNG 和 SVG。交付后可在工作目录运行 `python research_v17/delivery17.py` 进行只读哈希及报告核验。', '']
    report='\n'.join(lines);assert '\ufffd' not in report
    (OUT/'第十七轮测试报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(report_characters=len(report),figures=3,status='RENDERED'),ensure_ascii=True))

if __name__=='__main__':main()

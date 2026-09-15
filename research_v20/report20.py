"""Write the completed factorial comparison, retaining the earlier interaction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
LABELS={'native_mse':'原始 MSE','learned_probe':'学习：未截尾','learned_clip':'学习：第十七轮','learned_market':'学习：全历史加性',
    'learned_vol_interaction':'学习：全历史交互','learned_recent_additive':'学习：近三年加性','learned_recent_interaction':'学习：近三年交互',
    'raw25_probe':'简单：未截尾','raw25_clip':'简单：第十七轮','raw_trend':'简单：全历史加性','raw_vol_interaction':'简单：全历史交互',
    'raw_recent_additive':'简单：近三年加性','raw_recent_interaction':'简单：近三年交互','market4':'仅市场特征',
    'training_frequency':'全历史训练频率','recent_frequency':'近三年训练频率','neutral_50':'固定 50%'}
WINDOWS={'early_2015_2017':'2015—2017','late_2018_2020':'2018—2020','pooled_2015_2020':'2015—2020 合并'}
LEARNED=['learned_market','learned_vol_interaction','learned_recent_additive','learned_recent_interaction']
RAW=['raw_trend','raw_vol_interaction','raw_recent_additive','raw_recent_interaction']
MAIN=['native_mse']+LEARNED+RAW+['training_frequency','recent_frequency']
FULL={'learned_recent_additive':'learned_market','learned_recent_interaction':'learned_vol_interaction','raw_recent_additive':'raw_trend','raw_recent_interaction':'raw_vol_interaction'}
INTERACTIONS={'learned_recent_interaction':'learned_recent_additive','raw_recent_interaction':'raw_recent_additive'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(v):return '—' if pd.isna(v) else f'{v*100:.2f}%'
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def savefig(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)

def figures(metrics,yearly):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.8,7.6),layout='constrained')
    for col,w in enumerate(['early_2015_2017','late_2018_2020']):
        t=metrics[metrics.window.eq(w)].set_index('method')
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col];scale=100 if key=='accuracy' else 1
            for xs,methods,color in [([0,1],LEARNED[:2],'#256b9a'),([3,4],LEARNED[2:],'#b8682c')]:
                ys=[t.loc[m,key]*scale for m in methods];ax.plot(xs,ys,'o-',color=color,lw=2,markersize=7)
                for x,y in zip(xs,ys):ax.annotate(f'{y:.2f}%' if key=='accuracy' else f'{y:.4f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center',color=color,bbox=dict(facecolor='white',edgecolor='none',pad=.7))
            ax.axhline(t.loc['recent_frequency',key]*scale,color='#477744',ls='--',lw=1.3,label='近三年训练频率')
            if key=='accuracy':ax.axhline(t.loc['native_mse',key]*scale,color='#7b607f',ls=':',lw=1.4,label='原始 MSE')
            ax.set_xticks([0,1,3,4],['全历史\n加性','全历史\n交互','近三年\n加性','近三年\n交互'],fontsize=9)
            ax.set_xlim(-.55,4.55);ax.set_ylim((42,65) if key=='accuracy' else (.232,.290));ax.grid(axis='y',alpha=.15)
            ax.set_ylabel('方向准确率（%）' if key=='accuracy' else 'Brier（越低越好）')
            if row==0:ax.set_title(f"{WINDOWS[w]}，n={int(t.loc['native_mse','n'])}",fontweight='bold')
    axes[0,0].legend(loc='upper left',ncols=2,frameon=False,fontsize=8.5)
    fig.suptitle('学习特征：保留交互对照，近三年分类头训练改善前期、削弱后期',fontsize=14)
    savefig(fig,'window_interaction_comparison')
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.0),layout='constrained');x=np.arange(6);years=list(range(2015,2021))
    for method,color,label in [('learned_vol_interaction','#256b9a','全历史交互'),('learned_recent_interaction','#b8682c','近三年交互'),('recent_frequency','#477744','近三年训练频率')]:
        g=yearly[yearly.method.eq(method)].sort_values('year');axes[0].plot(x,g.mean_probability*100,'o-',color=color,label=label,lw=1.7)
    g=yearly[yearly.method.eq('recent_frequency')].sort_values('year');axes[0].plot(x,g.observed_up_fraction*100,'D--',color='#373d43',label='实际上涨比例',lw=1.4)
    axes[0].set(xticks=x,xticklabels=years,ylim=(35,83),ylabel='平均上涨概率 / 实际上涨比例（%）',title='学习交互：2018 年偏向上涨明显增强')
    axes[0].legend(loc='upper left',ncols=2,fontsize=8.5,frameon=False)
    for method,color,offset in [('learned_recent_interaction','#b8682c',9),('recent_frequency','#373d43',-17)]:
        r=yearly[yearly.year.eq(2018)&yearly.method.eq(method)].iloc[0];value=(r.mean_probability if method.startswith('learned') else r.observed_up_fraction)*100
        axes[0].annotate(f'{value:.2f}%',(3,value),xytext=(0,offset),textcoords='offset points',ha='center',color=color)
    t=yearly.pivot(index='year',columns='method',values='brier')
    for offset,candidate,parent,color,label in [(-.19,'learned_recent_interaction','learned_vol_interaction','#256b9a','学习交互'),(.19,'raw_recent_interaction','raw_vol_interaction','#b8682c','简单交互')]:
        delta=(t[candidate]-t[parent]).to_numpy();axes[1].bar(x+offset,delta,width=.35,color=color,label=label)
        for i,val in enumerate(delta):axes[1].text(i+offset+np.sign(offset)*.04,val+(.002 if val>=0 else -.002),f'{val:+.4f}',ha='center',va='bottom' if val>=0 else 'top',fontsize=7.5)
    axes[1].axhline(0,color='#555b61',lw=1);axes[1].set(xticks=x,xticklabels=years,ylim=(-.055,.115),ylabel='近三年交互 Brier − 全历史交互 Brier',title='缩短训练窗口：零线下方表示概率误差改善')
    axes[1].legend(loc='upper left',ncols=2,frameon=False,fontsize=8.5)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('逐年检查；2015 年仅 22 周，所有日期均为已经使用过的历史样本',fontsize=13.5)
    savefig(fig,'annual_window_diagnostics')

def main():
    assert not (OUT/'delivery_manifest.json').exists(),'Preserve delivered report'
    v=read(OUT/'verification.json');assert v['status']=='PASS';p=read(ROOT/'protocol.json');fit=read(OUT/'training_manifest.json')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv');yearly=pd.read_csv(OUT/'yearly_metrics.csv');folds=pd.read_csv(OUT/'fold_summaries.csv')
    training=pd.read_csv(OUT/'training_metrics.csv');seeds=pd.read_csv(OUT/'seed_metrics.csv');changes=pd.read_csv(OUT/'direction_changes.csv');factorial=pd.read_csv(OUT/'factorial_differences.csv')
    coefficients=pd.read_csv(OUT/'coefficients.csv');dist=pd.read_csv(OUT/'interaction_input_distribution.csv');pairs=read(OUT/'primary_comparisons.json');assessments=read(OUT/'assessments.json')
    figures(metrics,yearly)
    lines=['# 第二十轮方法测试：保留交互，比较近三年与全历史训练','',
        '**已保留第十九轮交互模型，完成四种新版本共 48 个分类头的训练及独立复核。本轮不采用三年硬截断窗口，继续保留全历史交互作为后续对照。** 学习交互改用近三年样本拟合分类头后，前期准确率从 46.67% 升至 55.83%，后期却从 58.16% 降至 51.06%。这一变化没有延续原交互在后期的方向改善。','',
        '用户指出的改善仍明确保留：在全历史训练下，第十九轮交互使 2018—2020 年准确率由加性模型的 56.74% 提高到 58.16%，多预测对 2 周。它是值得跟踪的局部线索；当时概率误差上升、前期退步的限制也同时保留。未通过跨时期筛选，不等于删除这个研究方向。','',
        '本轮“近三年”只限定最后分类头拟合损失使用的标签行。神经表示、截尾标准化、交互信号方向及投影均沿用每年冻结的全历史版本，仍包含更早数据的影响；这不是整条模型只用近三年数据重新训练的实验。','',
        '## 1. 四格对照与实际训练范围','',
        table(['分类头训练样本','不含乘积交互','含第十九轮乘积交互'],[
            ['截止日前全部可用记录','第十八轮加性模型，保留','第十九轮交互模型，保留'],
            ['截止日前三个日历年','本轮重新拟合','本轮重新拟合']]),'',
        '两种特征族分别完整比较这四格。学习特征在每个年度使用三个原有种子并平均概率；简单价格量能特征每年一个确定性头。新加性头维数分别为 29、26，新交互头分别为 30、27；每个头再含一个截距。全体采用 λ=0.01 的岭逻辑回归，截距不惩罚，严格 p>0.5 判涨。','',
        '所有新头使用原有冻结坐标，唯一改变是拟合损失的样本成员。三年窗口按信号日期筛选：从截止年份往前数第三年的 1 月 1 日起，并且收益及辅助标签已经在截止日完成。没有搜索窗口长度、正则强度、种子、状态、阈值或验证期最优权重。','',
        table(['截止日','近三年起点','全历史行数','近三年行数','全历史上涨频率','近三年上涨频率','下一年周数'],[
            [r.cutoff,r.recent_start,r.full_train_n,r.recent_train_n,pct(r.full_frequency),pct(r.recent_frequency),r.test_n] for r in folds.itertuples()]),'',
        '加入“近三年上涨频率”常数概率参照，它使用与新分类头完全相同的标签。六个频率均超过 0.5，所以它在所有测试日期均预测上涨。前期正确 67/120=55.83%，后期正确 79/141=56.03%。这避免将样本上涨比例变化误当作复杂模型的收益。旧全历史频率也继续保留。','',
        '原有沪深 300 OHLCV、125 根历史输入、开盘到开盘的周度收益标签及样本过滤保持一致。测试信号是下一年度周五，标签须于 2020-12-31 前完整；训练标签须不晚于各截止日。2015 年只有 22 周，继续沿用异常 OHLC 与受影响历史窗口的排除规则。前期 120 周、后期 141 周，合并 261 周仅作描述。','',
        '## 2. 交互如何保留，哪些内容没有重估','',
        '沿用第十九轮的一维交互：在加性坐标 X 中取前 25 项与原加性头前 25 个系数的乘积 r，乘以冻结标准化的 20 日波动 v，得到 q=v·r；再减去原训练折内 q 对 [X,1] 的最小二乘投影，并按原训练残差均值和标准差标准化，得到 h。','',
        '本轮读取原来的 [X,h] 缓存，筛选近三年的行用于新分类头训练。测试年采用完全相同的原 h 定义。没有重估信号方向、投影、残差尺度、基础特征分位数或神经表示。h 在近三年子集中不必均值为 0、标准差为 1，也不保证继续正交；这是固定坐标对照的预期性质。','',
        '原交互方向由全历史监督拟合的父模型决定，包含更早标签的信息，并不是训练内折外信号。因此，本轮可以讨论“既定特征坐标下缩短分类头训练窗口”的效果，不能把它推广为所有近期训练方案的结论。24 个近期交互头仍精确包含各自近期加性头：将交互系数置零并使用相同加性系数，就得到同一预测和正则化目标。','',
        '## 3. 两个时期的完整关键对照','',
        'Brier 与对数损失均越低越好。原始 MSE 输出收益分数，不补造概率损失。其余六个历史参照也全部保留在 [17 种方法总体指标](ensemble_metrics.csv)，下表展示本轮四格及必要基准。','']
    for w in p['windows']:
        t=metrics[metrics.window.eq(w['name'])].set_index('method')
        lines += [f"### {WINDOWS[w['name']]}，{w['n']} 周",'',table(['方法','正确周数','准确率','平衡准确率','AUROC','Brier','对数损失'],[
            [LABELS[m],int(t.loc[m,'correct_directions']),pct(t.loc[m,'accuracy']),pct(t.loc[m,'balanced_accuracy']),num(t.loc[m,'auroc']),num(t.loc[m,'brier']),num(t.loc[m,'log_loss'])] for m in MAIN]),'']
    lines += ['![窗口与交互对照](window_interaction_comparison.png)','',
        '学习版在近三年窗口内增加交互，前期从 53.33% 升至 55.83%，增加 3 个正确周；后期从 52.48% 降至 51.06%，减少 2 个正确周。前期 55.83% 恰好等于近三年频率基准，而 Brier 0.260195 高于基准的 0.247304。后期 Brier 0.276218 也高于旧全历史交互的 0.244168。','',
        '简单版改用近三年样本后，加性模型前期从 43.33% 升至 49.17%，后期从 55.32% 降至 51.06%；交互模型前期从 43.33% 升至 48.33%，后期从 54.61% 降至 48.94%。近期信息并未在两个时期带来同方向改善。','',
        '合并 261 周，近三年学习交互准确率为 53.26%，旧全历史交互为 52.87%，但 Brier 从 0.255860 上升到 0.268851。只看合并准确率会掩盖后期损失，不能据此选择新版本。','',
        '## 4. 逐年检查：2017 改善，2018 明显退步','']
    for title,methods in [('学习特征',LEARNED),('简单特征',RAW)]:
        lines += [f'### {title}方向准确率','',table(['年份','周数']+[LABELS[m] for m in methods]+['原始 MSE','近三年频率'],[
            [year,int(g.iloc[0].n)]+[pct(g.set_index('method').loc[m,'accuracy']) for m in methods+['native_mse','recent_frequency']] for year,g in yearly.groupby('year')]),'']
    lines += ['![逐年窗口诊断](annual_window_diagnostics.png)','',
        '近三年学习交互相对旧全历史交互的正确周数变化为：2015 年 0、2016 年 +1、2017 年 +10、2018 年 −11、2019 年 +3、2020 年 −2。2017 年从 21/50 提高到 31/50，是前期改善的主要来源；2018 年从 32/49 降到 21/49，是后期退步的主要来源。不能只留下其中一个年度作为整体结论。','',
        '2018 年近三年加性与交互模型都在 49 周中的 47 周预测上涨，平均上涨概率分别为 71.11%、71.59%；实际只有 42.86% 的周上涨。旧全历史交互平均概率为 54.60%，其方向准确率为 65.31%，新近三年交互则为 42.86%。新加性模型也已退步，因此后期问题并非只有交互版本出现。','',
        '截至 2017 年末，近三年训练只保留 595 行，全历史为 1,678 行；训练上涨频率从 52.03% 变为 56.64%。这与 2018 年的预测偏移同时出现，但不足以证明全部误差由上涨频率或样本量造成。固定表示下的全部斜率和截距也重新拟合，不能从单项描述直接作因果归因。','',
        table(['时期','近期交互版本','参照','改变方向周数','正确→错误','错误→正确'],[
            [WINDOWS[r.window],LABELS[r.method],LABELS[r.reference],r.changed,r.correct_to_wrong,r.wrong_to_correct] for r in changes.itertuples()
            if r.method in INTERACTIONS and r.reference in [FULL[r.method],INTERACTIONS[r.method]]]),'',
        '近三年学习交互相对近三年加性模型，后期只改变 2 周方向，均从正确变错误；相对旧全历史交互，则改动 28 周，其中 19 周改错、9 周改对。方向变化与窗口变化的规模不同，四格对照应继续保留。','',
        '## 5. 交互增量随训练窗口如何变化','',
        '下面用同一批逐周损失做代数比较：各窗口内先计算交互减加性的损失增量，再计算近期增量减全历史增量。最后一列负值表示近期窗口中的交互增量更有利。它没有额外拟合或挑选日期，也不是因果机制或新增显著性检验。','',
        table(['时期','特征族','指标','全历史交互增量','近三年交互增量','增量之差'],[
            [WINDOWS[r.window],'学习' if r.family=='learned' else '简单','Brier' if r.metric=='brier' else '方向错误率',f'{r.full_interaction_increment:+.6f}',f'{r.recent_interaction_increment:+.6f}',f'{r.difference_in_differences:+.6f}'] for r in factorial.itertuples()]),'',
        '学习交互在全历史后期降低方向错误率 1.42 个百分点，在近三年后期却增加 1.42 个百分点；前期则由增加 2.50 个百分点变为降低 2.50 个百分点。交互的作用依赖这次训练样本选择，当前证据不支持把某个窗口的改善外推到所有时期。','',
        '下面记录交互系数变化与近期输入分布。学习版取三个头的均值，标准差一列也是各头标准差的均值，不是集成变量的标准差。每个截止日的新旧系数使用同一冻结坐标，可以比较系数变化；跨年度坐标仍不同。','']
    dc=[]
    for (method,cutoff),g in coefficients[coefficients.block.eq('interaction')].groupby(['method','cutoff']):
        recent=dist[dist.method.eq(method)&dist.cutoff.eq(cutoff)&dist.split.eq('recent_training')]
        test=dist[dist.method.eq(method)&dist.cutoff.eq(cutoff)&dist.split.eq('validation')]
        dc.append([LABELS[method],int(cutoff[:4])+1,num(g.full_window_coefficient.mean()),num(g.coefficient.mean()),num(recent.interaction_feature_mean.mean()),num(recent.interaction_feature_std.mean()),num(test.interaction_logit_mean.mean())])
    lines += [table(['版本','测试年','原交互系数','新交互系数','近期 h 均值','近期 h 标准差','测试交互平均 logit'],dc),'',
        '2018 年近三年学习交互模型的测试平均 logit 为 1.040194，其中重新拟合的加性部分（含截距）为 1.021934，交互项为 0.018260。这是当前系数下的恒等核算，不等于删去交互后的重训结果，也不能把全部性能变化归因于加性部分。未额外生成删项或测试期校准预测。','',
        '## 6. 训练收敛与种子检查','',
        f"48 个新分类头共完成 {fit['new_newton_iterations']} 次 Newton 更新，全部达到梯度无穷范数不超过 1e-9 的阈值。新目标均不高于旧系数在相同近期样本上的目标；24 个近期交互头的目标也均不高于对应近期加性头。由此可以排除未收敛解释，不能据此证明泛化改善。",'',
        table(['版本','截止日','训练准确率','训练 Brier','新正则目标','旧系数在近期的目标'],[
            [LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean()),num(g.source_on_recent_objective.mean())] for (method,cutoff),g in training.groupby(['method','cutoff'])]),'',
        table(['时期','学习版','种子','准确率','Brier','AUROC'],[
            [WINDOWS[r.window],LABELS[r.method],r.seed,pct(r.accuracy),num(r.brier),num(r.auroc)] for r in seeds[seeds.method.isin(['learned_recent_additive','learned_recent_interaction'])&seeds.window.ne('pooled_2015_2020')].itertuples()]),'',
        '种子共享样本和很多历史信息，不能当成独立市场重复实验。分类头按概率集成后再过阈值，集成准确率也不等于各种子准确率的均值。','',
        '## 7. 冻结的统计对比与验收','',
        '两个特征族、两个时期分别执行 8 项对比，共 32 项：近期加性对全历史加性的 Brier 与方向错误率；近期交互对全历史交互的两项；近期交互对近期加性的两项；近期交互对两个训练频率基准各自的 Brier。差值均为候选损失减参照损失，负数更好。','',
        '沿用 8 个保留观测为一块的循环区块重采样，10,000 次、种子 20260910，两个时期分别重采样。报告百分位 95% 区间、中心化双侧 p，对 32 个 p 统一做 Holm 校正。区间不是同时置信区间，日期缺口可能使一块跨越更多自然周。它们不覆盖历轮设计选择、重复查看历史和跨年度标签复用的全部不确定性。','',
        table(['时期','候选 / 参照','指标','差值','95% 区间','原始 p','Holm p'],[
            [WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',
                f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"] for r in pairs]),'',
        '32 项对比均未通过 Holm 校正。最小校正 p=0.2208，对应后期学习交互相对近期加性的 Brier 退步。前期学习交互相对旧全历史交互的方向错误率降低 9.17 个百分点，原始 p=0.0165，Holm p=0.5114，仍不构成独立确认。','',
        '对两个近期交互候选固定十二项描述条件：准确率高于旧全历史交互、近期加性、原始 MSE、全历史频率、近期频率；Brier 低于旧全历史交互、近期加性及两个频率基准；对数损失低于近期频率；至少两年准确率高于 MSE、至少两年 Brier 低于近期频率。平局不通过，两个时期均须十二项全过。近期加性是四格对照的一部分，不在看完结果后自动晋升为新候选。','',
        table(['时期','近期交互候选','通过项数','跨时期通过','交互假设继续保留'],[
            [WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/12",'是' if a['cross_period_descriptive_pass'] else '否','是'] for a in assessments]),'',
        '## 8. 市场状态诊断与适用边界','',
        '全部 17 个方法继续使用既定趋势×波动六格、波动高低、振幅高低、量能高低分组，共 408 个状态指标组，见 [状态指标](state_metrics.csv)。分组阈值沿用全历史定义，以便保持前后可比；没有按近期数据重新标记状态、选择有利分组或输出只在有利状态交易的成绩。少于 20 条的组均标记为稀疏。','',
        '所有 261 个测试日期此前已经被查看，本轮方案承接这些诊断，因此结果**不能作为独立确认**。已用过的 2021—2026 部分历史也不能直接改名为新测试集。当前实验不等于完整论文复现，也未检验成本、仓位或交易盈利。','',
        '## 9. 本轮决定与后续方向','',
        '**保留全历史交互，不采用本轮三年硬截断替换它。** 交互在原全历史后期确有多预测对 2 周的记录；在本轮近期前期也比对应加性模型多预测对 3 周。这些局部线索继续写入研究记录，同时保留概率质量和跨时期稳定性不足的结果。','',
        '如果继续训练，下一步可围绕全历史交互做一个温和的时间衰减对照：让较早样本仍参与拟合，仅逐渐降低其权重，保留相应无交互模型，提前固定衰减形式和预算。它是尚未测试的假设；本轮没有估计最优衰减速度，也不能保证它会优于全历史等权。最终应使用未用于选型的数据或冻结后的前瞻记录确认。','',
        '## 10. 核验与证据','',
        table(['项目','结果'],[
            ['新分类头 / 独立 L-BFGS-B 解','48 / 48'],['Newton 更新',fit['new_newton_iterations']],['含截距系数','1,428'],['近期交互精确嵌套检查','24'],
            ['全历史成员行 / 近期成员行（按折累计）','9,465 / 3,951'],['跨模型训练特征行','31,608'],['新分类概率 / 新频率基准值','2,088 / 261'],
            ['沿用模型记录 / 合计模型记录','5,220 / 7,308'],['集成记录','4,437 = 17 × 261'],['指标组 / 状态组','624 / 408'],['概率分箱 / 主要对比 / 四格代数对比','160 / 32 / 8'],
            ['新神经训练 / 前向推理 / 投影拟合','0 / 0 / 0'],['继承的网络回放 / 独立投影证据','18 / 24'],['最大新预测复核误差',f"{v['maximum_new_forecast_error']:.3e}"],
            ['独立分类目标最大差',f"{v['maximum_alternate_objective_gap']:.3e}"],['独立解训练概率最大差',f"{v['maximum_alternate_probability_gap']:.3e}"],
            ['独立解梯度最大值',f"{v['maximum_alternate_gradient_inf']:.3e}"],['旧文件哈希不变','2,958'],['独立保留日期','0']]),'',
        '复核 PASS 覆盖独立日期及标签成熟度筛选、48 组冻结输入和原预测回放、近期模型精确嵌套、Newton 回放及另一求解器、2,088 条新分类概率、261 条频率值、全部旧记录、集成、指标和预设统计量。训练计数包含跨折、种子及重叠周标签的重复使用，不代表同量独立观测。计算 PASS 与性能验收分开。','',
        f"协议 SHA-256：`{v['protocol_sha256']}`。",'',
        '主要证据：[协议与保留交互的约定](../protocol.json)；[窗口成员](window_membership.csv)；[年度训练范围与频率](fold_summaries.csv)；[总体指标](ensemble_metrics.csv)；[逐年指标](yearly_metrics.csv)；[逐种子指标](seed_metrics.csv)；[方向变化](direction_changes.csv)；[四格增量](factorial_differences.csv)；[系数](coefficients.csv)；[交互输入及贡献](interaction_input_distribution.csv)；[主要统计对比](primary_comparisons.json)；[十二项判定](assessments.json)；[独立求解](independent_solver_verification.csv)；[完整复核](verification.json)。','',
        '两张图另存 SVG。交付后可运行 `python research_v20/delivery20.py` 做只读核验。','']
    report='\n'.join(lines);assert '\ufffd' not in report;(OUT/'第二十轮测试报告.md').write_text(report,encoding='utf-8')
    note='\n'.join(['# 后续研究保留项','',
        '用户于 2026-09-12 明确要求保留波动交互及其近期年份改善的线索。',
        '第十九轮全历史学习交互在 2018—2020 年方向准确率 58.16%，对应加性版本 56.74%；Brier 则较差，前期也未改善。不得只保留其中一面。',
        '第二十轮保留全部旧记录，完成“全历史／近三年分类头窗口 × 加性／交互”四格对照。近期交互前期 55.83%，后期 51.06%；未采用三年硬截断替换原模型。',
        '交互继续作为后续对照，尚未升级为通过独立确认的策略。温和时间衰减只是后续假设，本轮未测试。','',
        '[本轮报告](results/第二十轮测试报告.md)','[冻结协议](protocol.json)',''])
    (ROOT/'README.md').write_text(note,encoding='utf-8')
    print(json.dumps(dict(status='RENDERED',report_characters=len(report),figures=2,interaction_hypothesis_retained=True),ensure_ascii=True))

if __name__=='__main__':main()

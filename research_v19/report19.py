"""Render descriptive report and figures from the verified frozen experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
LABELS={'native_mse':'原始 MSE','learned_probe':'未截尾学习特征','learned_clip':'学习特征：第十七轮',
    'learned_market':'学习特征：加性条件','learned_vol_interaction':'学习特征：波动交互',
    'raw25_probe':'未截尾简单特征','raw25_clip':'简单特征：第十七轮','raw_trend':'简单特征：加性趋势',
    'raw_vol_interaction':'简单特征：波动交互','market4':'仅市场特征','training_frequency':'训练频率','neutral_50':'固定 50%'}
WINDOWS={'early_2015_2017':'2015—2017','late_2018_2020':'2018—2020','pooled_2015_2020':'2015—2020 合并'}
MAIN=['native_mse','learned_clip','learned_market','learned_vol_interaction','raw25_clip','raw_trend','raw_vol_interaction','training_frequency']
ANNUAL=['native_mse','learned_market','learned_vol_interaction','raw_trend','raw_vol_interaction','training_frequency']
CANDIDATES={'learned_vol_interaction':'learned_market','raw_vol_interaction':'raw_trend'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(v):return '—' if pd.isna(v) else f'{v*100:.2f}%'
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def savefig(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)


def figures(metrics,yearly):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.7,7.2),layout='constrained')
    for col,w in enumerate(['early_2015_2017','late_2018_2020']):
        t=metrics[metrics.window.eq(w)].set_index('method')
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col];scale=100 if key=='accuracy' else 1
            for xs,methods,color in [([0,1],['learned_market','learned_vol_interaction'],'#256b9a'),([3,4],['raw_trend','raw_vol_interaction'],'#b8682c')]:
                ys=[t.loc[m,key]*scale for m in methods];ax.plot(xs,ys,'o-',color=color,lw=2,markersize=7)
                for x,y in zip(xs,ys):ax.annotate(f'{y:.2f}%' if key=='accuracy' else f'{y:.4f}',(x,y),xytext=(0,11),textcoords='offset points',ha='center',color=color,bbox=dict(facecolor='white',edgecolor='none',pad=.7))
            ax.axhline(t.loc['training_frequency',key]*scale,color='#477744',ls='--',lw=1.3,label='训练频率')
            if key=='accuracy':ax.axhline(t.loc['native_mse',key]*scale,color='#7b607f',ls=':',lw=1.4,label='原始 MSE')
            ax.set_xticks([0,1,3,4],['学习：加性','学习：交互','简单：加性','简单：交互'],fontsize=9)
            ax.set_xlim(-.55,4.55);ax.set_ylim((39,65) if key=='accuracy' else (.234,.298));ax.grid(axis='y',alpha=.15)
            ax.set_ylabel('方向准确率（%）' if key=='accuracy' else 'Brier（越低越好）')
            if row==0:ax.set_title(f"{WINDOWS[w]}，n={int(t.loc['native_mse','n'])}",fontweight='bold')
    axes[0,0].legend(loc='upper left',ncols=2,frameon=False,fontsize=9)
    fig.suptitle('增加一个波动交互：后期方向略升，跨时期表现不稳定',fontsize=14)
    savefig(fig,'interaction_comparison')
    fig,axes=plt.subplots(1,2,figsize=(13.4,4.9),layout='constrained');x=np.arange(6);years=list(range(2015,2021))
    for ax,key in zip(axes,['correct_directions','brier']):
        t=yearly.pivot(index='year',columns='method',values=key)
        for offset,candidate,parent,color,label in [(-.19,'learned_vol_interaction','learned_market','#256b9a','学习特征：增加波动交互'),(.19,'raw_vol_interaction','raw_trend','#b8682c','简单特征：增加波动交互')]:
            d=(t[candidate]-t[parent]).to_numpy();ax.bar(x+offset,d,width=.35,color=color,label=label)
            for i,value in enumerate(d):
                pad=.1 if key=='correct_directions' else .0002
                label_x=i+offset+(np.sign(offset)*.045 if key=='brier' else 0)
                ax.text(label_x,value+(pad if value>=0 else -pad),f'{value:+.0f}' if key=='correct_directions' else f'{value:+.4f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=7.5 if key=='brier' else 8)
        ax.axhline(0,color='#555b61',lw=1);ax.set_xticks(x,years);ax.set_xlim(-.6,5.6);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set(ylim=(-3,3.2),yticks=list(range(-3,4)),ylabel='新模型正确周数 − 加性模型正确周数',title='方向判断：零线上方表示多预测对')
    axes[1].set(ylim=(-.009,.009),ylabel='新模型 Brier − 加性模型 Brier',title='概率误差：零线下方表示改善')
    axes[0].legend(loc='upper left',frameon=False,fontsize=8.5)
    fig.suptitle('逐年比较：准确率与概率质量并未同步改善；2015 年仅 22 周',fontsize=13.5)
    savefig(fig,'annual_interaction_diagnostics')


def main():
    assert not (OUT/'delivery_manifest.json').exists(),'Preserve delivered report'
    v=read(OUT/'verification.json');assert v['status']=='PASS';p=read(ROOT/'protocol.json');fitting=read(OUT/'training_manifest.json')
    metrics=pd.read_csv(OUT/'ensemble_metrics.csv');yearly=pd.read_csv(OUT/'yearly_metrics.csv');seeds=pd.read_csv(OUT/'seed_metrics.csv')
    states=pd.read_csv(OUT/'state_metrics.csv');training=pd.read_csv(OUT/'training_metrics.csv');diag=pd.read_csv(OUT/'interaction_diagnostics.csv')
    summary=pd.read_csv(OUT/'component_summary.csv');changes=pd.read_csv(OUT/'direction_changes.csv')
    pairs=read(OUT/'primary_comparisons.json');assessments=read(OUT/'assessments.json');figures(metrics,yearly)
    lines=['# 第十九轮方法测试：波动与已有信号的单维交互','',
        '**本轮交互没有带来稳定改善，不采用它替换现有研究基线。** 学习特征模型在后期的准确率从 56.74% 升至 58.16%，多预测对 2 周；前期从 49.17% 降至 46.67%，少预测对 3 周。后期虽然方向准确率提高，Brier 概率误差却从 0.241936 上升至 0.244168。简单特征模型前期准确率持平，后期下降。','',
        '24 个新分类头、24 组独立投影复算、1,044 条新概率和全部核验已经完成。计算复核为 PASS；两个候选均未通过预先固定的跨时期判定。16 项对比中没有显著改善，唯一通过本轮 Holm 校正的是简单特征交互模型在前期的 Brier **差于**训练频率。','',
        '所有 261 个测试日期此前均被查看，本轮设计也承接前期诊断，因此这些结果**不能作为独立确认**。本轮只检验这一种受限交互，不能据此断言所有市场状态方法无效。','',
        '## 1. 固定的问题和模型','',
        '第十八轮把市场条件直接加入分类头，本轮进一步检验：20 日波动能否调节已有信号的强度？沿用原来的加性模型，仅为每个分类头追加一个交互坐标，保留全部旧参照。','',
        table(['版本','直接父模型','原斜率数','新增斜率','总斜率 / 含截距系数','新拟合数'],[
            ['学习特征：波动交互','学习特征＋四项市场条件',29,1,'30 / 31',18],
            ['简单特征：波动交互','简单特征＋趋势得分',26,1,'27 / 28',6]]),'',
        '学习特征来自冻结的 Crossformer 解码表示，每个截止日沿用三个既有种子；按三个头的预测概率平均，再以 p>0.5 判涨。简单特征每个截止日一个确定性头。均使用平均二元对数损失＋λ/2 倍斜率平方和，λ=0.01，截距不惩罚。阈值、种子、正则强度、交互形式和统计标准均在新拟合、新评分前冻结。','',
        '基础输入 X 完全沿用第十八轮。学习版包含 25 个解码坐标与 4 个市场条件，简单版包含 25 个价格量能特征与趋势得分。简单版原有特征已经包含波动、振幅和量能变化，本轮没有重复添加这些主效应。','',
        '## 2. 交互如何构造，如何防止线性重复','',
        '在每个年度训练折内，令 v 为第十八轮已经截尾并标准化的 20 日波动；令 b 为对应父模型前 25 个系数。先取已有表示对父模型 logit 的贡献 r=X[:, :25]b，再计算一个乘积 q=v·r。父模型后附的市场系数和截距不进入 r；简单版的原始 25 维本身仍含三项市场摘要。','',
        '对训练数据计算 A=[X,1]，以无惩罚最小二乘得到 γ=argmin‖q−Aγ‖²。再用训练残差 e=q−Aγ 的均值 μe、总体标准差 se 构造 h=(e−μe)/max(se,1e−6)。模型使用 [X,h] 联合重新拟合。下一年直接沿用该折冻结的 b、γ、μe、se；不读取下一年的涨跌标签来定义这些变换。','',
        '这样每个分类头仅增加一个不能由训练集原加性坐标线性表示的方向。旧系数保持原值且新系数为零时，精确恢复父模型及其正则化目标。24 组设计矩阵的秩均增加 1，均未触及标准差下限。训练投影去掉的是线性重复部分，并非剔除所有统计依赖。','',
        '两个限制需要明确：第一，方向 b 来自同一训练折上已经监督拟合的父模型，不是训练内折外预测；投影本身不再用标签，但 q 仍通过 b 间接使用训练标签。第二，残差化后的岭惩罚与直接追加未残差化乘积的惩罚不同，本轮没有测试后一种模型。未来残差也没有再截尾，只保留组成变量原有的训练分位数截尾。所有旧斜率均会重拟合，不能把新旧性能差全部归因于单个交互项。','',
        '## 3. 数据、时间顺序与可比性','',
        '沿用沪深 300 日线 OHLCV、原有 125 根历史输入、样本掩码和开盘到开盘的周度可执行收益标签；正收益为上涨。训练仅纳入 joint_completed 不晚于截止日的样本。测试信号日期位于下一年度的周五，收益及辅助标签须在 2020-12-31 前完成。早期测试样本只有在标签完成后才可进入更晚年度训练。','',
        table(['训练截止日','训练行数','下一年测试周数'],[[f['cutoff'],f['train_n'],f['test_n']] for f in p['folds']]),'',
        '前期为 2015—2017 年 120 周，后期为 2018—2020 年 141 周；合并 261 周仅作描述。2015 年仍只保留 22 周，继承包括 2015-03-27 异常 OHLC 在内的历史过滤，不补齐或重新挑选日期。训练采用每日滚动的重叠周标签，各折及种子共享不少观测；合计 37,860 条训练特征行不是同量的独立市场事件。','',
        '本轮 24 个分类头全部训练结束后，才生成任何新的下一年概率。未新增神经网络训练或前向推理，使用第十七轮已经回放、第十八轮已经验证且哈希不变的缓存和市场变换。原有 4,176 条模型记录保留；新增 1,044 条后共 5,220 条，最终 12 种方法各覆盖同样 261 个日期。','',
        '本轮没有补齐论文尚缺的 78 指标清单、P0、Kronos 接口或全部超参数，也未引入未来预训练信息，仍属于现有原始 OHLCV 对照链的研究测试，不代表论文完整复现或交易收益验证。','',
        '## 4. 两个时期的实际结果','',
        'Brier 是上涨概率与实际 0/1 标签的均方误差；它和对数损失均越低越好。原始 MSE 给出收益分数，不虚构其概率或概率损失。表内列出关键参照，全部 12 种方法保存在 [完整总体指标](ensemble_metrics.csv)。','']
    for w in p['windows']:
        t=metrics[metrics.window.eq(w['name'])].set_index('method')
        lines += [f"### {WINDOWS[w['name']]}，{w['n']} 周",'',table(['方法','正确周数','准确率','平衡准确率','AUROC','Brier','对数损失'],[
            [LABELS[m],int(t.loc[m,'correct_directions']),pct(t.loc[m,'accuracy']),pct(t.loc[m,'balanced_accuracy']),num(t.loc[m,'auroc']),num(t.loc[m,'brier']),num(t.loc[m,'log_loss'])] for m in MAIN]),'']
    lines += ['![交互前后比较](interaction_comparison.png)','',
        '学习版前期 Brier 比加性父模型降低 0.002302，但仍明显差于第十七轮学习版及训练频率；后期 Brier 比父模型增加 0.002232，AUROC 从 0.602082 降至 0.592895。后期的 58.16% 也仍低于原始 MSE 的 59.57%。简单版前期准确率同为 43.33%，后期从 55.32% 降至 54.61%。','',
        '合并两个时期，学习交互为 138/261=52.87%，加性父模型为 139/261=53.26%；简单交互为 129/261=49.43%，加性父模型为 130/261=49.81%。原始 MSE 为 143/261=54.79%，训练频率为 142/261=54.41%。合并结果不用于掩盖前后时期差异，也不用于新增主要检验。','',
        '## 5. 逐年结果与方向变化','',
        table(['测试年','周数']+[LABELS[m] for m in ANNUAL],[[str(year),str(int(g.iloc[0].n))]+[pct(g.set_index('method').loc[m,'accuracy']) for m in ANNUAL] for year,g in yearly.groupby('year')]),'',
        '![逐年交互诊断](annual_interaction_diagnostics.png)','',
        '学习版后期相对加性模型的正确周数变化依次为 +2、−1、+1；2018、2019、2020 三年的 Brier 却分别增加 0.004789、0.000465、0.001275。准确率仅由概率是否越过 0.5 决定，不能代替对全部概率误差的检查。2018 和 2020 年学习交互的正确周数与原始 MSE 持平，2019 年少 2 周，没有一年严格超过原始 MSE。','',
        table(['时期','候选','相对加性父模型改变方向','原本对→错','原本错→对'],[
            [WINDOWS[r.window],LABELS[r.method],r.changed,r.correct_to_wrong,r.wrong_to_correct] for r in changes.itertuples() if CANDIDATES[r.method]==r.reference]),'',
        '学习版后期只改动 4 周的方向，3 周改对、1 周改错；前期改动 7 周，2 周改对、5 周改错。这是逐周记录的核对，不是额外选样本计算的新绩效。','',
        '三个学习种子的完整结果如下。集成在阈值判断之前平均概率，因此集成准确率不必位于单个种子准确率之间。','',
        table(['时期','种子','准确率','Brier','AUROC'],[[WINDOWS[r.window],r.seed,pct(r.accuracy),num(r.brier),num(r.auroc)] for r in seeds[seeds.method.eq('learned_vol_interaction')&seeds.window.ne('pooled_2015_2020')].itertuples()]),'',
        '## 6. 交互是否被学到，以及能否迁移','',
        f"全部分类头经过 {fitting['new_newton_iterations']} 次 Newton 更新达到预定收敛精度，训练正则化目标均不高于父模型。独立 L-BFGS-B 解与主解的目标差最大为 {v['maximum_alternate_objective_gap']:.3e}。可以排除本轮分类头未收敛这一解释，但训练目标降低不构成泛化证据。",'',
        table(['版本','截止日','训练准确率','训练 Brier','父模型目标','新目标'],[
            [LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.parent_objective.mean()),num(g.objective.mean())] for (method,cutoff),g in training.groupby(['method','cutoff'])]),'',
        '下面按年展示交互系数与下一年残差坐标的分布。学习版取三个种子的均值；“测试 h 标准差”是各头标准差的均值，不能当成集成变量的标准差。训练 h 的均值约为 0、标准差为 1。','']
    diagnostic_rows=[]
    for (method,cutoff),g in diag.groupby(['method','cutoff']):
        s=summary[summary.method.eq(method)&summary.cutoff.eq(cutoff)&summary.split.eq('validation')]
        diagnostic_rows.append([LABELS[method],int(cutoff[:4])+1,num(g.interaction_coefficient.mean()),num(s.interaction_feature_mean.mean()),
            num(s.interaction_feature_std.mean()),num(s.interaction_logit_mean.mean()),num(s.interaction_logit_std.mean())])
    lines += [table(['版本','测试年','交互系数','测试 h 均值','测试 h 标准差','交互平均 logit','交互 logit 标准差'],diagnostic_rows),'',
        '训练残差虽已正交化，下一年的分布仍会变化。简单版 2015 年 h 均值为 −2.272247、标准差为 2.490806，交互平均 logit 为 −0.074332；同年平均上涨概率从父模型的 39.97% 降到 38.31%，实际上涨比例为 59.09%。学习版前期各年的 h 标准差均值约为 1.91—2.12，也大于训练值 1。这些描述支持继续检查分布迁移，不能单凭它们证明所有误差的原因。','',
        '学习版交互系数随年度和种子变化，部分年度出现符号分歧。各年度的信号方向 b 和标准化基底本身也会变化，不能把系数变号解释为某个经济机制必然反转。所有 logit 分解仅用于核对固定模型；没有新做“删掉交互项但保留重拟合系数”的评分，也没有把这种不完整反事实当成归因。','',
        '## 7. 保留全部预设市场状态诊断','',
        '沿用第十七轮按训练数据确定的趋势三分位、波动中位数，展示趋势×波动的六个分组。ΔBrier 表示交互模型减加性父模型，负数更好。星号表示少于 20 个样本；这 12 个时期分组中有 5 个稀疏组。所有行均展示，不据此选状态、调阈值或计算挑选后的总体成绩。','']
    state_rows=[]
    state_names={'low_trend__low_vol':'低趋势 / 低波动','low_trend__high_vol':'低趋势 / 高波动','mid_trend__low_vol':'中趋势 / 低波动',
        'mid_trend__high_vol':'中趋势 / 高波动','high_trend__low_vol':'高趋势 / 低波动','high_trend__high_vol':'高趋势 / 高波动'}
    # Preserve the existing partition and state labels verbatim where no display alias exists.
    chosen=states[states.partition.eq('trend_volatility')]
    for (window,state),g in chosen.groupby(['window','state'],sort=False):
        t=g.set_index('method');n=int(t.iloc[0].n)
        state_rows.append([WINDOWS[window],state_names.get(state,state),f'{n}'+('*' if n<20 else ''),pct(t.loc['learned_vol_interaction','accuracy']),
            f"{t.loc['learned_vol_interaction','brier']-t.loc['learned_market','brier']:+.6f}",pct(t.loc['raw_vol_interaction','accuracy']),f"{t.loc['raw_vol_interaction','brier']-t.loc['raw_trend','brier']:+.6f}"])
    assert len(state_rows)==12
    lines += [table(['时期','状态','周数','学习交互准确率','学习 ΔBrier','简单交互准确率','简单 ΔBrier'],state_rows),'',
        '波动、振幅、量能各自高低两组的结果也全部保存在 [状态指标](state_metrics.csv)，合计 288 个指标组。它们重复使用相同日期，不是额外独立样本。','',
        '## 8. 预先固定的统计对比与判定','',
        '两个时期各 8 项、共 16 项主要对比：两个交互候选相对直接父模型的 Brier 与方向错误率；相对各自第十七轮参照的 Brier；相对训练频率的 Brier。差值均为候选损失减参照损失，负数更好。','',
        '两个时期分别使用 8 个保留观测为一块的循环区块重采样，10,000 次，种子 20260910；报告百分位 95% 区间、中心化双侧 p，并对全部 16 个 p 做 Holm 校正。区间本身未作同时性校正。日期缺口可能令一块跨越更多自然周，统计量也没有涵盖历轮设计选择和反复查看数据的不确定性。','',
        table(['时期','候选 / 参照','指标','差值','95% 区间','原始 p','Holm p'],[[WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],
            'Brier' if r['metric']=='brier' else '方向错误率',f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"] for r in pairs]),'',
        '没有任何改善通过 Holm 校正。唯一校正 p<0.05 的对比是简单交互在前期的 Brier 高于训练频率，差值 +0.033765，Holm p=0.0304。学习交互后期相对父模型的 Brier 退步，原始 p=0.0048，Holm p=0.0720，未通过本轮校正。这些 p 值仍是探索性结果，不能补偿已经用过的历史日期。','',
        '描述性验收使用十项严格条件：准确率超过直接父模型、第十七轮参照、原始 MSE、训练频率；Brier 低于直接父模型、第十七轮参照、训练频率；对数损失低于训练频率；至少两年准确率高于原始 MSE；至少两年 Brier 低于训练频率。每个时期十项全过且两个时期都过才满足条件，平局不算通过。','',
        table(['时期','候选','通过项数','跨时期通过'],[[WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/10",'是' if a['cross_period_descriptive_pass'] else '否'] for a in assessments]),'',
        '## 9. 对改进方向的判断','',
        '当前不采用这一波动乘积交互。市场条件识别和因果变换仍能帮助定位迁移问题，但新增状态变量、固定加性项或这个单维交互，都尚未形成可靠的跨时期提升。继续扩大同类历史特征搜索的优先级应降低。','',
        '下一步更值得检验的是训练分布的适应性：在保持当前标签、网络表示和评估参照的条件下，预先限定一个近期训练窗口，与当前扩展窗口直接比较；任何窗口或收缩强度选择应放在训练内按时间顺序的验证中。它是新的待检验假设，当前分布诊断并不能保证它有效，也不能把此前校准试验的结果移植过来。','',
        '研究结论要进一步确认，应冻结少量候选并积累未用于选型的未来预测记录，或使用经过历史使用记录核对的独立数据。当前 261 周以及更晚的部分历史区间在前轮已有使用，不能直接改名为新测试集。本轮没有执行上述后续实验。','',
        '## 10. 独立复核和交付证据','',
        table(['检查项目','结果'],[
            ['新分类头 / 独立 L-BFGS-B 解','24 / 24'],['SVD 投影 / 独立主元 QR 复核','24 / 24'],['Newton 更新',fitting['new_newton_iterations']],['系数合计（含截距）','726'],
            ['新增神经训练 / 前向推理','0 / 0；继承 18 个既有网络回放证据'],['训练特征行','37,860'],['新概率 / 沿用模型记录','1,044 / 4,176'],
            ['模型记录 / 集成记录','5,220 / 3,132'],['指标组 / 状态指标组','441 / 288'],['概率分箱 / 主要对比','110 / 16'],
            ['最大新预测误差',f"{v['maximum_new_forecast_error']:.3e}"],['QR 训练拟合值最大差',f"{v['maximum_qr_training_fitted_value_gap']:.3e}"],
            ['QR 训练标准化残差最大差',f"{v['maximum_qr_training_residual_gap']:.3e}"],['QR 下一年标准化残差最大差',f"{v['maximum_qr_validation_residual_gap']:.3e}"],
            ['独立分类目标最大差',f"{v['maximum_alternate_objective_gap']:.3e}"],['独立解训练概率最大差',f"{v['maximum_alternate_probability_gap']:.3e}"],
            ['独立解梯度最大值',f"{v['maximum_alternate_gradient_inf']:.3e}"],['旧证据哈希不变','2,882 个文件'],['独立保留日期','0']]),'',
        '复核覆盖训练索引、原始父模型权重、冻结标准化输入、SVD 与主元 QR 的投影一致性、父模型精确嵌套、Newton 回放和另一求解器、逐条预测与集成、交互贡献分解、441 组指标、110 个概率分箱、16 项区块对比及 Holm 校正。未来输入变换不改动训练参数，旧预测数值不变。复核 PASS 说明这些计算与协议一致，不代表方法性能通过。','',
        f"协议 SHA-256：`{v['protocol_sha256']}`。",'',
        '主要证据：[冻结协议](../protocol.json)；[总体指标](ensemble_metrics.csv)；[逐年指标](yearly_metrics.csv)；[逐种子指标](seed_metrics.csv)；[方向变化](direction_changes.csv)；[交互诊断](interaction_diagnostics.csv)；[逐周交互分解](interaction_components.csv)；[训练和测试分布](component_summary.csv)；[独立投影](projection_verification.csv)；[独立分类解](independent_solver_verification.csv)；[主要对比](primary_comparisons.json)；[十项判定](assessments.json)；[完整复核](verification.json)。','',
        '两张图均保存为 PNG 与 SVG。交付后可运行 `python research_v19/delivery19.py` 进行只读核验。','']
    report='\n'.join(lines);assert '\ufffd' not in report;(OUT/'第十九轮测试报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(status='RENDERED',report_characters=len(report),figures=2),ensure_ascii=True))


if __name__=='__main__':main()

"""Research report and figures from verified round23 artifacts only."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
sys.path.insert(0,str(ROOT.parent/'research_v22'));from report22 import pct,num,line,table
LABELS={'learned_market':'学习表征＋加性市场特征','learned_vol_interaction':'学习表征＋原波动交互','learned_order_extension':'学习表征＋原交互＋顺序项',
    'raw_trend':'简单特征＋加性趋势','raw_vol_interaction':'简单特征＋原波动交互','raw_order_extension':'简单特征＋原交互＋顺序项','order_only':'仅顺序指标','native_mse':'原 MSE 模型','training_frequency':'全训练期上涨频率'}
MAIN=list(LABELS);WINDOWS={'early_2015_2017':'2015–2017','late_2018_2020':'2018–2020','pooled_2015_2020':'合并描述'}
FAMILIES=[['learned_market','learned_vol_interaction','learned_order_extension'],['raw_trend','raw_vol_interaction','raw_order_extension']]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def metric_row(r):return [LABELS[r.method],f'{r.correct_directions}/{r.n}',pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)]
def pair_row(r):return [WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"]

def plots(metrics,years,daily,signal,novelty):
    fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#85919a','#b17b24','#187b8e'];x=np.arange(2);fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for col,fam in enumerate(FAMILIES):
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col]
            for j,m in enumerate(fam):
                vals=np.array([metrics[metrics.method.eq(m)&metrics.window.eq(w)].iloc[0][key] for w in ['early_2015_2017','late_2018_2020']])*(100 if row==0 else 1)
                bars=ax.bar(x+(j-1)*.23,vals,.23,color=colors[j],label=['加性模型','原波动交互','原交互＋顺序项'][j]);ax.bar_label(bars,labels=[f'{v:.2f}%' if row==0 else f'{v:.4f}' for v in vals],fontsize=9,padding=3)
            for i,w in enumerate(['early_2015_2017','late_2018_2020']):
                t=metrics[metrics.window.eq(w)].set_index('method');y=float(t.loc['training_frequency',key])*(100 if row==0 else 1);ax.plot([i-.42,i+.42],[y,y],ls='--',color='#bf5962',lw=1.2,label='训练频率' if i==0 else None)
                if row==0:ax.plot([i-.42,i+.42],[100*t.loc['native_mse',key]]*2,ls=':',color='#333333',lw=1.2,label='原 MSE' if i==0 else None)
            ax.set_xticks(x,['2015–2017 · 120 周','2018–2020 · 141 周']);ax.set_ylim(0,72 if row==0 else .325);ax.set_title(('学习表征' if col==0 else '简单特征')+('：方向准确率 ↑' if row==0 else '：Brier 误差 ↓'),loc='left',weight='bold');ax.set_ylabel('准确率（%）' if row==0 else '概率均方误差');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    hs,ls=axes[0,0].get_legend_handles_labels();fig.legend(hs,ls,loc='outside lower center',ncol=5,frameon=False);fig.suptitle('学习分支前段改善，后段尚未保持原交互的方向表现\n顺序项作为补充接受检验；原交互及其 58.16% 继续保留',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'order_extension_comparison.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');ax=axes[0,0];ax.plot(pd.to_datetime(daily.date),daily.excess_order,color='#187b8e',lw=.9)
    ax.axhline(0,color='#666666',lw=.7)
    for threshold in [-.05,.05]:ax.axhline(threshold,color='#a8874a',lw=.8,ls='--')
    ax.set_title('过去 60 日：实际相邻符号积 − 重排期望',loc='left',weight='bold');ax.set_ylabel('顺序持续度 u');ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.grid(alpha=.15)
    ax=axes[0,1];xp=np.arange(6)
    for j,(method,parent,color) in enumerate([('learned_order_extension','learned_vol_interaction','#187b8e'),('raw_order_extension','raw_vol_interaction','#b17b24')]):
        t=years.pivot(index='year',columns='method',values='correct_directions');delta=t[method]-t[parent];bars=ax.bar(xp+(j-.5)*.32,delta,.32,color=color,label='学习表征' if j==0 else '简单特征');ax.bar_label(bars,labels=[f'{v:+d}' for v in delta],padding=3)
    ax.set_xticks(xp,range(2015,2021));ax.axhline(0,color='#555555',lw=.8);ax.set_ylim(-4,5.5);ax.set_ylabel('比原交互多对的周数');ax.set_title('年度差异：增益没有跨年一致出现',loc='left',weight='bold');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1,0];pts=ax.scatter(signal.efficiency60,signal.excess_order,c=pd.to_datetime(signal.date).dt.year,cmap='viridis',s=18,alpha=.8);fig.colorbar(pts,ax=ax,ticks=range(2015,2021),label='年份');ax.set_xlabel('上轮 60 日路径效率');ax.set_ylabel('本轮顺序持续度');ax.set_title('路径效率相近时，涨跌排列仍可不同',loc='left',weight='bold');ax.grid(alpha=.15)
    ax=axes[1,1]
    for m,color,label in [('learned_order_extension','#187b8e','学习表征'),('raw_order_extension','#b17b24','简单特征')]:
        g=novelty[novelty.method.eq(m)].groupby('cutoff').gate_linear_r2.mean();ax.plot(range(2015,2021),g.to_numpy(),marker='o',color=color,label=label)
    ax.set_xticks(range(2015,2021));ax.set_ylim(0,1);ax.set_ylabel('训练门控解释度 R²');ax.set_xlabel('下一年度');ax.set_title('原输入可解释部分变化，非完全重复',loc='left',weight='bold');ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.suptitle('顺序信息与已有幅度指标有所不同，但不等于稳定可预测\n状态相关和 R² 均为诊断；未据此筛选日期或训练专家',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'order_state_diagnostics.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

def main():
    v=read(OUT/'verification.json');assert v['status']=='PASS';p=read(ROOT/'protocol.json');fit=read(OUT/'training_manifest.json');metrics=csv('ensemble_metrics');years=csv('yearly_metrics');daily=csv('validation_daily_states');signal=csv('validation_signal_states');stability=csv('validation_stability');nov=csv('gate_novelty');cor=csv('state_correlations');state=csv('state_metrics');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json');coef=csv('coefficients');train=csv('training_metrics')
    plots(metrics,years,daily,signal,nov)
    a=np.r_[np.ones(40),-np.ones(20)];b=np.tile([1,-1,1,-1,1,1],10)
    def values(s):
        observed=float(np.mean(s[:-1]*s[1:]));expected=float((s.sum()**2-np.square(s).sum())/(len(s)*(len(s)-1)));return dict(observed=observed,permutation_expectation=expected,excess_order=observed-expected)
    facts=dict(example_clustered=values(a),example_interleaved=values(b),same_positive_count=40,same_negative_count=20,significant_improvements=sum(r['difference']<0 and r['holm_adjusted_p']<.05 for r in pairs),significant_deteriorations=sum(r['difference']>0 and r['holm_adjusted_p']<.05 for r in pairs))
    (OUT/'report_facts.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
    chunks=['# 第二十三轮方法测试：在原交互上增加涨跌顺序信息','2026-09-12｜沪深 300｜30 个新分类头｜原交互保留｜历史研究实验',
        '本轮有局部改善，尚不足以替代原模型。学习分支从原交互的前段 56/120（46.67%）提高到 61/120（50.83%）：改变的 5 个方向全部由错变对。后段从 82/141（58.16%）降到 81/141（57.45%），5 次变化中 2 次由错变对、3 次由对变错。原交互和新增顺序项的结果均继续保留。',
        '学习分支的 Brier 前段从 0.269597 降到 0.267551，后段从 0.244168 微降到 0.244092；后段仍不如原加性模型的 0.241936。简单特征分支前段准确率由 43.33% 降到 42.50%，后段由 54.61% 升到 55.32%，没有一致改善。',
        '26 项固定比较中没有经 Holm 校正后显著的改善或恶化。学习分支前段相对原交互的方向改善，原始 p=0.0212，校正后 p=0.5087。两个候选都未通过冻结的跨时期描述筛选。数值复算 PASS 只说明实现和计算通过核验。',
        '## 本轮检验什么',
        '上一轮路径效率只利用收益总和与绝对收益总和，不能区分同一组日收益仅改变先后顺序的不同排列。本轮只增加一个“顺序持续度”门控，用相邻涨跌符号是否更容易连续，提供含时间顺序的信息。',
        '本轮父模型直接采用第十九轮原波动交互，而第廿二轮父模型是第十八轮加性模型。因此本轮对原交互的比较是受控的新增项检验；不能把与上轮不同方法的表现差异全部归因于状态定义。新增项沿用第十九轮缓存中的原监督射线，不改用第十九轮重新拟合的前 25 个系数。',
        '在每个年度，保留原交互的全部输入坐标和变换，追加一个经过训练期残差化的顺序乘积。所有分类系数（包括原波动交互系数）允许重新拟合；新增顺序系数设为零时精确包含原交互模型。原模型预测本身另外完整保存，并未覆盖。',
        '## 顺序持续度的定义与边界',
        '取截至当天的 60 个日涨跌符号 s_i：收盘价比前一日大为 +1、小为 −1、完全相等为 0。使用直接价格比较，不设微小变动阈值。',
        '实际相邻符号积均值 A=(Σ_{i=2}^{60}s_i s_{i−1})/59。同号且非零贡献 +1，异号贡献 −1，其中任何一天平价贡献 0。保持这 60 个符号的数量不变、均匀随机重排时，相邻符号积的精确期望为 B=((Σs_i)²−Σs_i²)/(60×59)。新门控 u=A−B。',
        'B 的公式来自两项无放回抽取：所有有序不同位置的符号乘积之和是 (Σs)²−Σs²。每个相邻对的期望相同，所以相邻乘积的平均期望也是 B。本轮直接使用解析值，没有逐窗口随机打乱、没有从未来估计参数。含平价符号的小样本全排列检查已经通过。',
        'u>0 表示相对同一涨跌数量的随机排列，同号相邻更多；u<0 表示交替更多。它不是状态后验概率，也不是标准相关系数或随机性检验的 p 值。连续段与随机性的背景可参见 [NIST Runs Test 说明](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm)，本轮使用的是自定义中心化符号积，而非调用该检验。',
        table(['固定 60 日例子','上涨数','下跌数','实际 A','重排期望 B','顺序持续度 u'],[['先 40 次涨、后 20 次跌',40,20,num(facts['example_clustered']['observed']),num(facts['example_clustered']['permutation_expectation']),num(facts['example_clustered']['excess_order'])],['重复（涨、跌、涨、跌、涨、涨）',40,20,num(facts['example_interleaved']['observed']),num(facts['example_interleaved']['permutation_expectation']),num(facts['example_interleaved']['excess_order'])]]),
        '这个指标对任意重排通常敏感，但仍有边界：全部上涨、全部下跌或全部平价都会得到 u=0，因为给定这种符号数量后已没有排列变化；它不等于没有趋势。它忽略收益幅度，对整体涨跌符号翻转、时间反转不变，也不刻画所有更长滞后的依赖关系。',
        '只有一个固定 60 日窗口和一阶相邻关系，不搜索窗口、滞后或阈值。描述组固定为 u<−0.05、−0.05≤u≤0.05、u>0.05；“接近零”不代表已证实随机。模型使用连续值，分组只用于诊断。',
        '## 数据、因果性与模型',
        '所有样本和评估口径沿用旧实验：沪深 300 数据源共 4046 根日线（2010-01-04 至 2026-08-31），本轮只评分 2015–2020 年反复使用过的 261 个周信号。前段 120 周、后段 141 周。后段指 2018–2020，不是新的 2026 留出集。',
        '六折的训练样本数为 1077、1190、1434、1678、1921、2165；下一年测试为 22、48、50、49、46、46。训练只含 joint_completed≤cutoff 的标签，目标为已有执行口径下周收益严格大于 0。较早测试日的标签成熟后可进入后续训练折。',
        '公式本身需要 61 个收盘价。为保持与上轮日状态诊断的有效历史一致，本轮仍要求 121 根 OHLC 日线有效；旧分类观察窗口要求 125 根，更严格，所以训练／评分样本没有变化。2015-03-27 无效条导致 121 个日步的状态缺失，2015 年状态诊断只剩 123 个有效日，周评分仍为 22 个。没有填补或压缩缺失日，连续段在缺失处中断。',
        '符号门控是完全确定的过去窗口函数，修改未来价格不改变此前特征。原神经表征、监督射线、OLS 变换和分类头仍按整段训练样本拟合，不能称整个模型为时间顺序 OOF。',
        '令 X_parent 为原第十九轮标准化输入，包括第十八轮加性坐标和原波动交互；r=X_parent[:25]·ray，ray 精确复用第十九轮缓存中的第十八轮监督射线。对 q=u·r 在训练期做 q~[X_parent,1] 的 OLS，残差按训练均值和总体标准差（下限 10⁻⁶）变成 h_order。把它追加到原输入，下一年冻结射线、投影和归一化参数。',
        '学习分支 18 个头由 30 变成 31 个斜率，简单分支 6 个头由 27 变成 28 个斜率。仅顺序对照有 6 个头，每个把 u 按训练均值／标准差变成单坐标，再拟合一个斜率和截距。共 30 个头、762 个分类系数（含截距）、24 个新乘积投影、47325 行累计训练输入；投影参数另计。',
        '统一采用 λ=0.01 的等权岭逻辑回归，截距不惩罚，严格 p>0.5 判涨。学习分支平均三个种子的概率。本轮不新增神经训练、神经前向或 HMM 拟合。训练内目标的下降是嵌套模型增加自由度后的拟合结果，不是泛化证明。',
        '## 主要结果',
        '准确率与 AUROC 越高越好，Brier 与对数损失越低越好。原 MSE 的收益分数未经概率校准，不报告 Brier 或对数损失。']
    for w in p['windows']:
        t=metrics[metrics.window.eq(w['name'])].set_index('method').loc[MAIN].reset_index();chunks+=['### '+WINDOWS[w['name']],table(['方法','正确/样本','准确率','平衡准确率','AUROC','Brier','对数损失'],[metric_row(r) for r in t.itertuples()])]
    chunks+=['![原模型与新增顺序项对比](order_extension_comparison.png)',
        '前段学习分支相对原加性为 +2 个正确方向，相对原交互为 +5；后段相对加性为 +1，相对原交互为 −1。这是不同对照给出的不同结论，不能只挑一个有利的对照。仅顺序指标前／后段准确率为 50.00%／55.32%，Brier 均高于训练频率基准，未显示单独可用的稳定方向能力。',
        '### 每年正确周数']
    for fam in FAMILIES:
        selected=fam+['order_only','native_mse','training_frequency'];rows=[]
        for year,g in years.groupby('year'):
            t=g.set_index('method');rows.append([year,int(g.iloc[0]['n'])]+[int(t.loc[m,'correct_directions']) for m in selected])
        chunks.append(table(['年份','样本']+[LABELS[m] for m in selected],rows))
    chunks+=['学习分支相对原交互，2015–2020 各年的正确数变化依次为 0、+1、+4、−3、+2、0。前段的改善主要来自 2017 年；后段 2018 年的损失抵消了 2019 年的收益。不能据此事后规定某几个年份启用或停用顺序项。',
        '## 状态信息与连续性诊断',
        '本轮顺序持续度与 |trend60| 的训练相关系数约为 −0.074 至 0.095，评分周逐年约为 −0.447 至 0.575，没有上一轮路径效率接近 1 的相关模式。它提供了不同角度的描述，但低相关不保证含有未来标签的额外信息。',
        table(['预测年','与原波动20相关','与 |trend60| 相关','与路径效率60相关','与相对波动相关'],[[int(r.cutoff[:4])+1,f'{r.correlation_volatility20:.6f}',f'{r.correlation_abs_trend60:.6f}',f'{r.correlation_efficiency60:.6f}',f'{r.correlation_relativevol:.6f}'] for r in cor[cor.split.eq('validation')].itertuples()]),
        '原输入线性解释顺序门控的训练 R² 约为 0.199–0.416，说明部分变化已有线性解释、部分未被线性解释。这个 R² 不是预测涨跌的 R²，也没有被用作自动选择条件。',
        table(['候选','门控解释度 R² 最小','均值','最大'],[[LABELS[m],num(g.gate_linear_r2.min()),num(g.gate_linear_r2.mean()),num(g.gate_linear_r2.max())] for m,g in nov.groupby('method')]),
        table(['预测年','有效日','缺失日','正顺序组占比','相邻相关','切换/相邻对','观察段中位数','最长观察段','删失段/总段'],[[int(r.cutoff[:4])+1,r.n,r.missing,pct(r.high_state_share),f'{r.lag1_correlation:.4f}',f'{r.switches}/{r.adjacent_pairs}',f'{r.median_observed_run:g}',r.max_observed_run,f'{r.censored_runs}/{r.runs}'] for r in stability.itertuples()]),
        '60 日滚动窗口相互重叠会提高状态自相关。连续段按有效相邻交易锚点计数，缺失和年度边界导致删失，表中长度只是观察到的段长，不能当成完整真实市场阶段的持续期。',
        '![年度差异及顺序状态诊断](order_state_diagnostics.png)',
        '### 固定顺序组的结果',
        '新分组前段样本数为 47／30／43，后段为 43／54／44。下表全部报告；这些仍是相关的历史样本，没有按状态选择交易或挑选有利子组。旧分组及新分组共 26 个状态单元、29 个方法、两个时期，共 1508 行状态指标，少于 20 的旧单元继续标为稀疏。']
    for w in p['windows']:
        rows=[]
        for label in ['negative_order','near_zero_order','positive_order']:
            for m in ['learned_vol_interaction','learned_order_extension','raw_order_extension','order_only','training_frequency']:
                r=state[state.window.eq(w['name'])&state.partition.eq('order_state')&state.state.eq(label)&state.method.eq(m)].iloc[0];rows.append([label,LABELS[m],int(r.n),pct(r.accuracy),num(r.brier)])
        chunks+=['### '+WINDOWS[w['name']]+' 顺序组',table(['顺序组','方法','样本','准确率','Brier'],rows)]
    chunks+=['## 固定比较与筛选',
        '每个时期 13 项，共 26 项，候选减对照损失，负值较好。循环区块长度 8 个保留观测，10000 次重采样，随机种子 20260910，两段单独生成。95% 区间为边际区间，双侧中心化 p 在全部 26 项上进行 Holm 校正；没有校正之前多轮历史设计选择，也没有使用合并时期作主要检验。',
        table(['时期','候选 / 对照','损失','差值','95% 区间','原始 p','Holm p'],[pair_row(r) for r in pairs]),
        '11 条描述条件仍冻结为：准确率超过原交互、原加性、原 MSE、训练频率（4 条）；Brier 低于原交互、原加性、训练频率、仅顺序（4 条）；对数损失低于训练频率（1 条）；至少两年准确率超过原 MSE、至少两年 Brier 低于训练频率（2 条）。相等不通过，两段必须全部满足；仅顺序对照不自动晋升。',
        table(['时期','候选','满足条件','跨时期通过','原交互留存'],[[WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",'是' if a['cross_period_descriptive_pass'] else '否','是'] for a in assess]),
        '## 原交互是否得到保留',
        '原模型本身的输入、系数、预测、报告与哈希记录保持原样。在新增模型里，原波动交互的特征也保留，但系数随整个分类头一起重新拟合；新模型精确嵌套原模型，并不等于把原模型所有系数永久锁定。',
        '下表学习分支为三种子系数均值，简单分支为单次值。系数依赖训练标准化坐标，不能跨变换直接当作因果重要性。逐条总 logit 已拆成重拟合的基础部分、原波动交互部分、新顺序部分，且复算和为总值。',
        table(['候选','预测年','重拟合原交互系数','新增顺序系数'],[[LABELS[m],int(cutoff[:4])+1,num(g[g.block.eq('original_interaction')].coefficient.mean()),num(g[g.block.eq('interaction')].coefficient.mean())] for (m,cutoff),g in coef[coef.method.ne('order_only')].groupby(['method','cutoff'])]),
        '## 实现复核',
        f"30 个头共执行 {fit['new_newton_iterations']} 次 Newton 更新，均达到梯度无穷范数≤10⁻⁹且 Hessian 正定。独立 L-BFGS-B 核验全部 30 个分类目标；独立 QR 核验 24 个新投影及门控线性解释度。24 个模型均在新增系数为零时精确重现第十九轮父模型及其训练目标。",
        f"逐条新预测的最大误差为 {v['maximum_new_forecast_error']:.3e}；状态表全部数值（包含继承的波动／路径效率）的独立标量公式最大差为 {v['maximum_state_formula_gap']:.3e}。独立分类器目标差最大 {v['maximum_alternate_objective_gap']:.3e}、训练概率差最大 {v['maximum_alternate_probability_gap']:.3e}。复算系数没有用于交付预测。",
        '核验包括六类小样本符号全排列期望、平价、同号、符号翻转／时间反转、价格缩放、三种状态的连续段中断与删失，以及 18 个真实历史边界的未来后缀扰动。12 组训练／测试状态序列和全部 1868 组指标、280 个可靠性分箱、26 项比较均复核通过。',
        '3292 个旧证据文件哈希保持不变。新增 1305 条单模型概率，加上 11223 条旧记录共 12528 条；29 个方法各有 261 条集成记录，共 7569 条。复用原来已核验的 18 个神经状态与缓存，本轮没有重新运行神经网络。',
        '以下为训练内拟合诊断，不能当成样本外成绩；学习模型为三个种子均值。',
        table(['方法','截止日','训练准确率','训练 Brier','带惩罚目标'],[[LABELS[m],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())] for (m,cutoff),g in train.groupby(['method','cutoff'])]),
        '## 研究决定与下一步',
        '保留全历史交互及其后段 58.16%，保留本轮学习顺序扩展前段的 +5 个正确方向和两个时期 Brier 小幅改善，作为尚待检验的研究线索。新增模型暂不替代原模型，简单特征分支也没有足够支持。上轮路径效率的小幅概率改善同样保留，不因本轮结果被删除。',
        '下一轮优先检验顺序补充项能否以更小、更稳定的幅度工作。可以在严格按时间先后划分的训练数据上校验新增项，并对新增系数加强收缩，同时把原交互作为固定比较对象。选择是否使用顺序项，应根据当时可用的训练证据，不能看完 2017／2018 年结果后按年份开关。这个收缩实验尚未在本轮执行。',
        '若下一步声称使用 OOF 条件校准，需要追溯上游表征、预处理、射线和父交互的估计日期；只把最后分类头做时间切分，不能自动使整个流程变成 OOF。有限的顺序信息研究仍优先于立即扩展两个大专家网络。',
        '## 限制与文件',
        '本轮 261 个历史评分日期已多次查看，**不能作为独立确认**。周标签在日频训练样本间重叠，8 观测区块重采样仅提供特定设定下的近似。2015 年覆盖不完整，状态没有外部真实标签。本轮没有交易成本、执行或净收益检验，也不等于完整论文方法复现；已经使用过的 2021–2026 年日期不能仅通过重新命名成为新的独立留出集。',
        f"冻结协议 SHA-256：`{v['protocol_sha256']}`。",
        '主要文件：[协议](../protocol.json)、[顺序公式](../states23.py)、[计算验证](verification.json)、[训练状态](training_signal_states.csv)、[评分状态](validation_signal_states.csv)、[日状态](validation_daily_states.csv)、[持续性](validation_stability.csv)、[连续段](validation_state_runs.csv)、[相关性](state_correlations.csv)、[输入解释度](gate_novelty.csv)、[逐条 logit 分解](interaction_components.csv)、[全部单模型预测](model_predictions.csv)、[集成预测](ensemble_predictions.csv)、[汇总指标](ensemble_metrics.csv)、[年度指标](yearly_metrics.csv)、[种子指标](seed_metrics.csv)、[状态指标](state_metrics.csv)、[可靠性分箱](reliability_bins.csv)、[方向变化](direction_changes.csv)、[固定比较](primary_comparisons.json)、[筛选结果](assessments.json)、[公式例子](report_facts.json)。',
        '图表矢量版：[模型对比 SVG](order_extension_comparison.svg)、[顺序诊断 SVG](order_state_diagnostics.svg)。独立副本的空结果目录中按 prepare23.py → contract23.py → train23.py → score23.py → evaluate23.py → verify23.py 执行，科学计算解释器为 research_v4/.venv_gpu/Scripts/python.exe。当前冻结记录不可覆盖；报告绘图使用默认 Python。']
    report='\n\n'.join(chunks)+'\n';(OUT/'第二十三轮测试报告.md').write_text(report,encoding='utf-8')
    (ROOT/'README.md').write_text('# 第二十三轮研究留存\n\n2026-09-12：30 个分类头、24 个新交互投影均完成且独立复算通过。本轮把顺序项追加到原第十九轮波动交互，保留旧输入、射线和旧模型记录。\n\n保留全历史交互后段 58.16%。新学习扩展前段 50.83%（+5 个正确方向），后段 57.45%（−1 个），两段 Brier 略降。没有经 Holm 校正后显著改善，尚不替代原模型。\n\n本轮顺序信息及前段改善留作后续研究线索。下一步优先按时间校验更小的补充幅度，追溯上游 OOF 条件；不根据已见测试年份开关。保留第二十二轮路径效率的小幅概率改善记录。\n\n[完整报告](results/第二十三轮测试报告.md)｜[冻结协议](protocol.json)｜[验证](results/verification.json)\n',encoding='utf-8')
    print(json.dumps(dict(status='BUILT_NOT_YET_VISUALLY_REVIEWED',report_characters=len(report),figures=2,facts=facts),indent=2))
if __name__=='__main__':main()

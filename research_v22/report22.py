"""Build report and scientific figures from verified artifacts; no model calls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
LABELS={'learned_market':'学习表征＋加性市场特征','learned_vol_interaction':'学习表征＋原波动交互','learned_hmm_interaction':'学习表征＋HMM 交互','learned_relativevol_interaction':'学习表征＋相对波动交互','learned_persistence_interaction':'学习表征＋路径效率交互',
    'raw_trend':'简单特征＋加性趋势','raw_vol_interaction':'简单特征＋原波动交互','raw_hmm_interaction':'简单特征＋HMM 交互','raw_relativevol_interaction':'简单特征＋相对波动交互','raw_persistence_interaction':'简单特征＋路径效率交互',
    'relativevol_only':'仅相对波动','persistence_only':'仅路径效率','native_mse':'原 MSE 模型','training_frequency':'全训练期上涨频率'}
MAIN=list(LABELS);WINDOWS={'early_2015_2017':'2015–2017','late_2018_2020':'2018–2020','pooled_2015_2020':'合并描述'}
FAMILIES=[['learned_market','learned_vol_interaction','learned_hmm_interaction','learned_relativevol_interaction','learned_persistence_interaction'],['raw_trend','raw_vol_interaction','raw_hmm_interaction','raw_relativevol_interaction','raw_persistence_interaction']]
GATES={'relative_volatility':'相对波动','persistence':'路径效率'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def pct(x):return '—' if pd.isna(x) else f'{100*x:.2f}%'
def num(x):return '—' if pd.isna(x) else f'{x:.6f}'
def line(xs):return '| '+' | '.join(map(str,xs))+' |'
def table(headers,rows):return '\n'.join([line(headers),line(['---']*len(headers))]+[line(r) for r in rows])
def metric_row(r):return [LABELS[r.method],f'{r.correct_directions}/{r.n}',pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)]
def pair_row(r):return [WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"]

def figures(metrics,stability,daily,signal):
    fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#8b969d','#b67c22','#ad9ca5','#4079a0','#25917e'];names=['加性模型','原波动交互','HMM 交互','相对波动交互','路径效率交互']
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');x=np.arange(2);width=.15
    for col,fam in enumerate(FAMILIES):
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col]
            for j,m in enumerate(fam):
                vals=[float(metrics[metrics.method.eq(m)&metrics.window.eq(w)].iloc[0][key]) for w in ['early_2015_2017','late_2018_2020']];vals=np.array(vals)*(100 if key=='accuracy' else 1)
                bars=ax.bar(x+(j-2)*width,vals,width,color=colors[j],label=names[j]);ax.bar_label(bars,labels=[f'{v:.2f}' if key=='accuracy' else f'{v:.4f}' for v in vals],fontsize=8,padding=3)
            for i,w in enumerate(['early_2015_2017','late_2018_2020']):
                t=metrics[metrics.window.eq(w)].set_index('method');value=float(t.loc['training_frequency',key])*(100 if key=='accuracy' else 1);ax.plot([i-.42,i+.42],[value,value],ls='--',color='#c4555a',lw=1.2,label='训练频率' if i==0 else None)
                if key=='accuracy':ax.plot([i-.42,i+.42],[100*t.loc['native_mse',key]]*2,ls=':',color='#292929',lw=1.2,label='原 MSE' if i==0 else None)
            ax.set_xticks(x,['2015–2017 · 120 周','2018–2020 · 141 周']);ax.set_title(('学习表征' if col==0 else '简单特征')+('：方向准确率 ↑' if row==0 else '：Brier 误差 ↓'),loc='left',weight='bold');ax.set_ylim(0,72 if row==0 else .325);ax.set_ylabel('准确率（%）' if row==0 else '概率均方误差');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    hs,ls=axes[0,0].get_legend_handles_labels();fig.legend(hs,ls,loc='outside lower center',ncol=4,frameon=False);fig.suptitle('固定含义的状态指标，尚未形成稳定的额外预测增益\n原交互继续保留；52 项固定检验无经 Holm 校正后显著的改善',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'fixed_state_comparison.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');date=pd.to_datetime(daily.date)
    for ax,col,label,threshold,color in [(axes[0,0],'relative_volatility','相对波动：(SD20−SD120)/(SD20＋SD120)',0,'#4079a0'),(axes[0,1],'efficiency60','60 日对数价格路径效率',.2,'#25917e')]:
        ax.plot(date,daily[col],lw=.8,color=color);ax.axhline(threshold,ls='--',lw=1,color='#8e5556');ax.set_title(label,loc='left',weight='bold');ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.grid(alpha=.15)
        ax.set_ylim((-.65,.65) if col=='relative_volatility' else (0,1));ax.set_ylabel('固定公式值')
    ax=axes[1,0];x=np.arange(6)
    for j,(gate,color) in enumerate([('relative_volatility','#4079a0'),('persistence','#25917e')]):
        s=stability[stability.gate.eq(gate)].sort_values('cutoff');bars=ax.bar(x+(j-.5)*.33,100*s.high_state_share,.33,color=color,label='短期波动高于长期' if j==0 else '路径效率 ≥ 0.2');ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set_xticks(x,['2015\n有效 123 日','2016\n244 日','2017\n244 日','2018\n243 日','2019\n244 日','2020\n243 日']);ax.set_ylim(0,70);ax.set_ylabel('日状态占比（%）');ax.set_title('状态占比仍会随年份变化',loc='left',weight='bold');ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    old=pd.read_csv(ROOT.parent/'research_v18/results/validation_states.csv',float_precision='round_trip');merged=signal.merge(old,on=['cutoff','row_index','date'],validate='one_to_one');ax=axes[1,1]
    points=ax.scatter(abs(merged.trend60),merged.efficiency60,c=pd.to_datetime(merged.date).dt.year,cmap='viridis',s=17,alpha=.8);fig.colorbar(points,ax=ax,label='年份',ticks=np.arange(2015,2021));ax.set_xlabel('已有 60 日趋势强度的绝对值');ax.set_ylabel('新 60 日路径效率');ax.set_title('路径效率与已有趋势强度高度相关',loc='left',weight='bold');ax.grid(alpha=.15)
    fig.suptitle('状态的含义固定，仍须检查信息重叠和预测价值\n滚动窗口重叠会提高自相关；2015 年缺失历史形成空白，未填补',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'state_stability_and_overlap.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

def main():
    v=read(OUT/'verification.json');assert v['status']=='PASS';p=read(ROOT/'protocol.json');fit=read(OUT/'training_manifest.json');metrics=csv('ensemble_metrics');years=csv('yearly_metrics');stability=csv('validation_stability');daily=csv('validation_daily_states');signal=csv('validation_signal_states');cor=csv('state_correlations');nov=csv('gate_novelty');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json');changes=csv('direction_changes');train=csv('training_metrics')
    figures(metrics,stability,daily,signal)
    # An algebraic counterexample; no fitting, forecasts or historical selection.
    path_a=np.r_[np.full(40,.01),np.full(20,-.01)];path_b=np.tile([.01,-.01,.01,-.01,.01,.01],10)
    er=lambda x:float(abs(x.sum())/abs(x).sum());same=lambda x:int(np.sum(np.sign(x[1:])==np.sign(x[:-1])))
    assert abs(er(path_a)-er(path_b))<1e-14
    facts=dict(permutation_example=dict(n=60,efficiency_a=er(path_a),efficiency_b=er(path_b),same_sign_neighbors_a=same(path_a),same_sign_neighbors_b=same(path_b),adjacent_pairs=59),
        heldout_efficiency_abs_trend_correlation_min=float(cor[cor.split.eq('validation')&cor.gate.eq('persistence')].correlation_abs_trend60.min()),heldout_efficiency_abs_trend_correlation_max=float(cor[cor.split.eq('validation')&cor.gate.eq('persistence')].correlation_abs_trend60.max()),
        significant_improvements=sum(r['difference']<0 and r['holm_adjusted_p']<.05 for r in pairs),significant_deteriorations=sum(r['difference']>0 and r['holm_adjusted_p']<.05 for r in pairs))
    (OUT/'report_facts.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
    chunks=['# 第二十二轮方法测试：相对波动、路径效率与状态信息重叠','2026-09-12｜沪深 300｜固定历史划分｜60 个分类头全部拟合后才进行本轮新评分',
        '本轮没有发现足以替换原方案的稳定增益。学习表征的路径效率交互在前段为 59/120（49.17%）、后段为 80/141（56.74%），两段的逐条方向均与原加性模型完全相同。它的 Brier 从 0.271899／0.241936 小幅降到 0.270341／0.241881。原全历史波动交互后段 82/141（58.16%）继续完整保留。',
        '相对波动交互的学习分支下降到 47.50%／55.32%；简单特征分支为 42.50%／56.03%。路径效率交互的简单分支前段从加性的 43.33% 提升至 45.83%，后段却由 55.32% 降到 53.19%，没有跨时期改善。四个候选均未通过冻结的描述筛选。',
        '52 项固定比较中没有经 Holm 校正后显著的改善。前段简单特征相对波动交互的 Brier 相对全训练频率较差，校正 p=0.0260。学习路径效率相对原交互的后段 Brier 虽然原始 p=0.0075，但校正后为 0.3750；不能只取未校正值宣称成功。',
        '本轮更有用的发现是信息重叠：路径效率与已有趋势强度绝对值的逐年评估相关系数约为 0.995–0.999。指标名称改变和数值平滑，并不自动产生新的方向信息。',
        '## 冻结设计',
        '本轮分别测试两个含义固定、无需训练状态识别器的连续指标。每个指标单独与原信号相乘，经训练期 OLS 残差化后，只增加一个分类斜率；同时设置各自的单指标逻辑回归对照。原来的全历史加性、波动交互、HMM 交互、三年窗口及全部旧方法均保留。',
        '相对波动：令 σ20、σ120 分别为截至当天的 20、120 个日对数收益的总体标准差，u_v=(σ20−σ120)/max(σ20+σ120,10⁻⁸)。它描述短期波动相对本身较长历史的增减，范围约为 [−1,1]，不是高波动的后验概率，也不保证不同资产或时期拥有相同分布。',
        '路径效率：ER60=|Σr_i|/max(Σ|r_i|,10⁻⁸)，其中 r_i 是截至当天的 60 个日对数收益，结果截到 [0,1]；交互门控为 u_e=2ER60−1。它衡量净位移占总路径长度的比例，取绝对值后不区分上涨和下跌。这是对数价格版本的路径效率，和通常直接用价格变化的效率比率有区别；参考 [StockSharp 指标实现说明](https://doc.stocksharp.com/en/topics/api/indicators/list_of_indicators/kaufman_efficiency_ratio)。',
        '平价历史使相对波动为 0、ER 为 0；分母下限为固定 10⁻⁸。窗口 20／60／120 沿用已有研究尺度，没有选择网格或结果后调参。诊断标签固定为 u_v>0 时“短期波动升高”，ER≥0.2 时“较高路径效率”；标签阈值只用于表格和连续段统计，模型始终使用连续数值。',
        '这两个状态特征在历史任一日期都只依赖当时及更早的价格，不用年度末重新拟合其定义。已检查整个价格前缀的一致性与未来后缀扰动。因此消除了上一轮 HMM 状态参数在训练期由整段样本估计的那一层不一致。原神经表征、监督信号射线、投影和分类器仍是在各自整段训练样本上拟合；整个模型不能称为时间顺序 OOF。',
        '时间评估仍按已成熟标签的历史训练、下一年度应用进行，避免随机打乱时间顺序。该原则可参见 [scikit-learn 时间序列交叉验证说明](https://scikit-learn.org/stable/modules/cross_validation.html)。本轮没有调用 TimeSeriesSplit 自动生成新划分，而是复用原来的六个固定年度边界。',
        '## 样本和训练口径',
        '数据源为 4046 根沪深 300 日线（2010-01-04 至 2026-08-31）。本轮评估仍是 2015–2020 年已经多次使用的 261 周，前段 120 周、后段 141 周；“后段／最近年份”在本报告中指 2018–2020，不能理解为新取得的 2026 样本。',
        '六折训练样本数为 1077、1190、1434、1678、1921、2165，测试为 22、48、50、49、46、46。目标是已有执行口径的周收益严格大于 0；训练要求 joint_completed≤cutoff。以前的测试日期在标签成熟后可以合法进入后来的训练折，不能称其为永久留出数据。',
        '状态计算要求截至当天连续 121 根 OHLC 日线都有效且收盘价有限、为正。2015-03-27 的无效条导致该条至其后第 120 条的状态缺失，共 121 个日步；前 120 个历史锚点也无法计算。旧 125 日观察窗口更严格，因此没有新增分类训练或测试剔除。2015 年状态日诊断只有 123 个有效日、评分只有原来的 22 周，未填补缺失历史。',
        '保留原 R18 的标准化加性 X 和前 25 个监督系数射线 r=X[:25]·w_parent[:25]。每个门控 u 单独形成 q=u·r，对 [X,1] 做训练期 OLS 投影，再以训练残差均值和总体标准差（下限 10⁻⁶）得到 h。下一年冻结这些变换并追加一个 h。新增交互系数为 0 时精确包含原加性模型；全部加性系数和截距可以重新拟合。',
        '学习分支 36 次拟合，每个 30 个斜率；简单分支 12 次拟合，每个 27 个斜率；两个单状态对照共 12 次拟合，每个 1 个斜率。共 60 个分类头、1476 个分类系数（含截距）、48 个交互投影，94650 行累计训练输入（跨折与方法重复计数）。射线和残差化的参数量不包含在分类系数计数中。',
        '所有分类头使用相同 λ=0.01、等权平均二元对数损失、截距不惩罚、严格 p>0.5 判涨。学习分支平均三个种子的预测概率。本轮没有神经权重训练、神经前向计算、HMM 拟合或联合双门控模型。',
        '## 主要结果',
        '准确率和 AUROC 越高越好，Brier 和对数损失越低越好。原 MSE 仅有收益分数，不填未经校准的概率指标。']
    for w in p['windows']:
        t=metrics[metrics.window.eq(w['name'])].set_index('method').loc[MAIN].reset_index();chunks+=['### '+WINDOWS[w['name']],table(['方法','正确/样本','准确率','平衡准确率','AUROC','Brier','对数损失'],[metric_row(r) for r in t.itertuples()])]
    chunks+=['![历史模型和两个新状态交互的对比](fixed_state_comparison.png)',
        '学习路径效率交互相对加性模型，两段都没有改变任何方向。它只是轻微移动概率，后段 Brier 差约 −0.000055，不能把这称作方向能力提高。相对原学习交互，前段 7 个方向不同（5 个由错变对、2 个由对变错），后段 4 个方向不同（1 个由错变对、3 个由对变错）。',
        '学习相对波动交互相对加性模型，前段 6 次方向变化中 4 次由对变错、2 次由错变对；后段 2 次均由对变错。单独使用相对波动或路径效率的模型在两个时期的 Brier 都高于全训练期上涨频率，尚未显示状态本身能提供稳定的方向预测。',
        '### 年度正确周数']
    for fam in FAMILIES:
        selected=fam+['native_mse','training_frequency'];rows=[]
        for year,g in years.groupby('year'):
            t=g.set_index('method');rows.append([year,int(g.iloc[0]['n'])]+[int(t.loc[m,'correct_directions']) for m in selected])
        chunks.append(table(['年份','样本']+[LABELS[m] for m in selected],rows))
    chunks+=['## 状态持续性与新增信息',
        '固定公式避免了 HMM 状态名称在年度重拟合时改变尺度，但实际分布和占比依然变化。相对波动的下一年度一阶自相关约 0.955–0.982，路径效率约 0.895–0.967；这些长窗口存在大量重叠，自相关高并不能证明识别了经济上真实、可预测的市场阶段。',
        '连续段统计按相邻交易锚点计算，缺失状态会中断，不跨缺失连线。年度／历史边界与缺失边界导致左右删失。表中的连续段长度包含删失段，仅是观察到的长度，不能直接解释为完整状态寿命。路径效率的固定阈值附近还会反复切换，例如 2015 年中位连续段仅 1 日。',
        table(['预测年','指标','有效日','缺失日','较高组占比','相邻相关','切换/有效相邻对','段长中位数','最长观察段','删失段/总段'],[[int(r.cutoff[:4])+1,GATES[r.gate],r.n,r.missing,pct(r.high_state_share),f'{r.lag1_correlation:.4f}',f'{r.switches}/{r.adjacent_pairs}',f'{r.median_observed_run:g}',r.max_observed_run,f'{r.censored_runs}/{r.runs}'] for r in stability.itertuples()]),
        '### 与已有特征的关系',
        '下表是评分周上的相关系数，只用于检查信息重叠。路径效率与 |trend60| 的高度相关不代表其与原有线性输入完全等价：绝对值本身是非线性变换；相乘和残差化也会形成新的函数。但它更像已有趋势幅度的重新表达，不能仅凭名字把它当成独立的市场识别来源。',
        table(['预测年','指标','与原 volatility20 的相关','与 |trend60| 的相关'],[[int(r.cutoff[:4])+1,GATES[r.gate],f'{r.correlation_volatility20:.6f}',f'{r.correlation_abs_trend60:.6f}'] for r in cor[cor.split.eq('validation')].itertuples()]),
        '代数上，已有 |trend60|=|Σr60|/(σ60√60)，ER60=|Σr60|/Σ|r60|。两者共享分子，只是用不同尺度归一化。这解释了为何它们可能高度相关；是否产生增量预测仍由上述比较决定。',
        '相对波动在简单特征分支中的训练线性可解释度 R² 达到约 0.906–0.965，提示它的大部分变化已能由现有标准化输入线性表达。学习分支没有达到同样程度，但这轮也没有显示更好的预测。这里的 R² 是训练输入解释度，不是涨跌预测 R²。',
        table(['新增交互','门控被已有 X 解释的 R² 最小','均值','最大'],[[LABELS[m],num(g.gate_linear_r2.min()),num(g.gate_linear_r2.mean()),num(g.gate_linear_r2.max())] for m,g in nov.groupby('method')]),
        '![状态分布与信息重叠](state_stability_and_overlap.png)',
        '### 路径效率不等于涨跌排列的持续性',
        'ER 只取收益总和与绝对收益总和，在窗口内重排收益不会改变它，因此没有直接编码涨跌出现的顺序。一个固定的代数例子：60 日中有 40 次 +1% 对数收益和 20 次 −1% 对数收益，先涨后跌与交错排列都得到 ER=1/3；相邻同号对却可以分别为 58/59 与 19/59。此例只说明指标定义的局限，没有使用标签、训练新模型或扩展本轮候选。',
        '因此本轮虽然用 persistence 命名代码字段，严格解释应是“路径效率代理”，不能宣称已经检验了所有趋势持续性方法。这也是下一轮应关注真正含时间顺序的信息，而不是继续替换同类归一化公式的原因。',
        '## 固定统计比较',
        '每个时期 26 项，共 52 项，候选减对照损失，负值较好。循环区块为 8 个保留观测、10000 次重采样、种子 20260910，两段单独生成。表中 95% 区间为边际区间，不是同时区间；双侧中心化 p 在全部 52 项上作 Holm 校正。没有校正此前多轮历史选择，也没有做合并时期的主要检验。',
        table(['时期','候选 / 对照','指标','差值','95% 区间','原始 p','Holm p'],[pair_row(r) for r in pairs]),
        '## 冻结描述筛选',
        '每个候选每段 11 条严格条件：准确率超过原加性、原交互、原 MSE、训练频率（4 条）；Brier 低于原加性、原交互、训练频率、匹配的单状态对照（4 条）；对数损失低于训练频率（1 条）；至少两年准确率超过原 MSE、至少两年 Brier 低于训练频率（2 条）。相等不通过，两段必须均满足全部条件，不自动挑选两个门控中表现最好的一项。',
        table(['时期','候选','满足条件','跨时期通过','原交互留存'],[[WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",'是' if a['cross_period_descriptive_pass'] else '否','是'] for a in assess]),
        '状态诊断完整保留旧 12 个分组单元、HMM 的 3 个后验组、新指标的 2＋2 个单指标组和 4 个交叉组，共 23 个状态单元。26 个方法、两个时期共有 1196 行状态指标，少于 20 个样本均标为稀疏。交叉组只作诊断，没有训练联合门控、按组挑样本或按组决定交易。',
        '## 复算与交付证据',
        f"60 个头共执行 {fit['new_newton_iterations']} 次 Newton 更新，全部达到梯度无穷范数≤10⁻⁹，Hessian 正定；48 个交互精确嵌套原加性模型。独立 L-BFGS-B 对全部 60 个分类目标、独立 QR 对全部 48 个交互投影及门控解释度进行复算，均通过。",
        f"逐条新预测的最大误差为 {v['maximum_new_forecast_error']:.3e}；两个状态公式相对独立标量算法的最大差为 {v['maximum_state_formula_gap']:.3e}。独立分类优化的目标最大差为 {v['maximum_alternate_objective_gap']:.3e}、训练概率最大差为 {v['maximum_alternate_probability_gap']:.3e}。复算系数未用于交付预测。",
        '已验证 12 组训练／评估状态序列、18 个实际历史未来后缀扰动、缺失值、平价／单向／交替收益、价格缩放与连续段删失处理。全部 1520 组指标、250 个可靠性分箱和 52 项比较均经过复核。旧证据 3166 个文件哈希未变。新增 2610 条单模型概率，加上 8613 条旧记录共 11223 条；26 个方法各 261 条集成记录，共 6786 条。计算验证 PASS 不代表研究效果通过。',
        '训练内指标只作拟合诊断；学习分支为三种子平均，简单及单状态分支为单次拟合。',
        table(['方法','截止日','训练准确率','训练 Brier','惩罚目标'],[[LABELS[m],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())] for (m,cutoff),g in train.groupby(['method','cutoff'])]),
        '## 研究决定与后续方向',
        '保留全历史交互及其后段 58.16% 的方向结果，保留原加性作为概率误差对照。本轮路径效率交互的轻微 Brier 改善也作为研究记录留存，但不替换原方案。相对波动版本未获得支持，暂不增加它的复杂度。',
        '下一步优先检查含时间顺序的持续性信息，例如固定窗口内的同号转移或符号自相关，先核对它与已有特征的重叠程度，再考察原交互误差是否随这种状态改变。为保留已有交互成果，可把原交互作为明确对照，另测试少量状态偏置／校准参数；不要一开始就分成两个大网络。以上是研究建议，本轮未测试这些指标或证明它们有效。',
        '监督射线与条件校准也应逐步使用时间顺序 OOF 预测来构建，避免在监督训练内误差上识别“适用状态”。无论指标更平滑还是状态更容易解释，都要以增量预测和跨时期稳健性来判断，而不能把稳定的状态曲线当成模型成功。',
        '## 限制与复现入口',
        '全部 261 个评分日期已被多轮查看，**不能作为独立确认**。日频训练标签对应的周区间相互重叠，不是独立样本；8 观测区块重采样只是在特定设定下的近似。2015 年覆盖不完整。没有市场状态真值、没有交易成本／执行／净收益检验，也不是完整论文方法复现。已用过的 2021–2026 日期不能通过改名成为新留出集。',
        f"冻结协议 SHA-256：`{v['protocol_sha256']}`。",
        '主要记录：[协议](../protocol.json)、[独立验证](verification.json)、[门控公式](../states22.py)、[训练状态](training_signal_states.csv)、[评估状态](validation_signal_states.csv)、[日状态](validation_daily_states.csv)、[连续段](validation_state_runs.csv)、[持续性统计](validation_stability.csv)、[状态相关性](state_correlations.csv)、[输入解释度](gate_novelty.csv)、[单模型预测](model_predictions.csv)、[集成预测](ensemble_predictions.csv)、[汇总指标](ensemble_metrics.csv)、[年度指标](yearly_metrics.csv)、[种子指标](seed_metrics.csv)、[全部状态指标](state_metrics.csv)、[可靠性分箱](reliability_bins.csv)、[方向变化](direction_changes.csv)、[固定比较](primary_comparisons.json)、[描述筛选](assessments.json)、[定义反例](report_facts.json)。',
        '图表：[模型对比 SVG](fixed_state_comparison.svg)、[状态与重叠 SVG](state_stability_and_overlap.svg)。在独立副本和空结果目录按 prepare22.py → contract22.py → train22.py → score22.py → evaluate22.py → verify22.py 执行，不覆盖当前冻结记录。科学计算解释器为 research_v4/.venv_gpu/Scripts/python.exe；报告绘图使用默认 Python。']
    report='\n\n'.join(chunks)+'\n';(OUT/'第二十二轮测试报告.md').write_text(report,encoding='utf-8')
    (ROOT/'README.md').write_text('# 第二十二轮研究留存\n\n2026-09-12：两个固定因果状态指标已完成 60 个分类头、48 个交互投影的拟合与复核。均未通过跨时期描述筛选。\n\n保留全历史交互，特别是 2018–2020 年 58.16% 的方向准确率；保留原加性概率对照。学习路径效率交互的轻微 Brier 改善留存，但两段逐条方向与加性完全相同。相对波动交互暂不采用。\n\n路径效率与已有 |trend60| 高度相关，且收益排列不改变它；下一步应优先检查含时间顺序的信息和 OOF 条件校准，不把状态平滑当成成功。\n\n[完整报告](results/第二十二轮测试报告.md)｜[冻结协议](protocol.json)｜[验证](results/verification.json)\n',encoding='utf-8')
    print(json.dumps(dict(status='BUILT_NOT_YET_VISUALLY_REVIEWED',report_characters=len(report),figures=2,facts=facts),indent=2))

if __name__=='__main__':main()

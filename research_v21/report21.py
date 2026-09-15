"""Build the research report from verified frozen results; no fitting or scoring."""
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
LABELS={'learned_market':'学习表征＋加性市场特征','learned_vol_interaction':'学习表征＋原波动交互','learned_hmm_interaction':'学习表征＋HMM 状态交互',
    'raw_trend':'简单特征＋加性趋势','raw_vol_interaction':'简单特征＋原波动交互','raw_hmm_interaction':'简单特征＋HMM 状态交互',
    'hmm_only':'仅 HMM 状态概率','native_mse':'原 MSE 模型','training_frequency':'全训练期上涨频率'}
MAIN=list(LABELS);WINDOWS={'early_2015_2017':'2015–2017','late_2018_2020':'2018–2020','pooled_2015_2020':'2015–2020 合并（描述）'}
FAMILIES=[['learned_market','learned_vol_interaction','learned_hmm_interaction'],['raw_trend','raw_vol_interaction','raw_hmm_interaction']]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def pct(v):return '—' if pd.isna(v) else f'{100*v:.2f}%'
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def line(values):return '| '+' | '.join(map(str,values))+' |'
def table(headers,rows):return '\n'.join([line(headers),line(['---']*len(headers))]+[line(r) for r in rows])
def metric_line(r):return [LABELS[r.method],f'{r.correct_directions}/{r.n}',pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)]
def pair_line(r):return [WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"]

def figures(metrics,years,hmm,signals,daily):
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#74818b','#b27d23','#126c8d'];x=np.arange(2);width=.23
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for col,family in enumerate(FAMILIES):
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col]
            for j,method in enumerate(family):
                vals=[float(metrics[(metrics.window.eq(w))&metrics.method.eq(method)].iloc[0][key]) for w in ['early_2015_2017','late_2018_2020']]
                plotvals=np.array(vals)*(100 if key=='accuracy' else 1)
                bars=ax.bar(x+(j-1)*width,plotvals,width,color=colors[j],label=['加性模型','原波动交互','HMM 状态交互'][j])
                ax.bar_label(bars,labels=[f'{v:.2f}%' if key=='accuracy' else f'{v:.4f}' for v in plotvals],fontsize=9,padding=3)
            for i,w in enumerate(['early_2015_2017','late_2018_2020']):
                t=metrics[metrics.window.eq(w)].set_index('method');ref=float(t.loc['training_frequency',key]);ref*=100 if key=='accuracy' else 1
                ax.plot([i-.4,i+.4],[ref,ref],color='#be4960',ls='--',lw=1.5,label='训练频率基准' if i==0 else None)
                if key=='accuracy':
                    value=100*float(t.loc['native_mse',key]);ax.plot([i-.4,i+.4],[value,value],color='#333333',ls=':',lw=1.5,label='原 MSE 模型' if i==0 else None)
            ax.set_xticks(x,['2015–2017 · 120 周','2018–2020 · 141 周']);ax.set_title(('学习表征' if col==0 else '简单特征')+('：方向准确率 ↑' if row==0 else '：Brier 概率误差 ↓'),loc='left',weight='bold')
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if key=='accuracy':ax.set_ylim(0,72);ax.set_ylabel('准确率（%）')
            else:ax.set_ylim(0,.325);ax.set_ylabel('均方概率误差')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False)
    fig.suptitle('两状态交互尚未形成稳定增益\n全部为已反复使用的历史日期；原交互继续保留',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'state_interaction_comparison.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for ax,r in zip(axes.flat,hmm.itertuples()):
        g=daily[daily.cutoff.eq(r.cutoff)];s=signals[signals.cutoff.eq(r.cutoff)];date=pd.to_datetime(g.date)
        ax.plot(date,g.high_probability,color='#126c8d',lw=1,alpha=.9,label='逐日向前过滤')
        ax.scatter(pd.to_datetime(s.date),s.high_probability,color='#ba7930',s=13,zorder=3,label='纳入评分的周信号')
        ax.axhline(.2,color='#87949b',ls='--',lw=.7);ax.axhline(.8,color='#87949b',ls='--',lw=.7);ax.set_ylim(-.03,1.05)
        ax.set_title(f"{int(r.cutoff[:4])+1} 年 · 评分 {len(s)} 周",loc='left',weight='bold')
        ax.text(.02,.94,f'拟合日波动：{100*r.low_daily_sd:.2f}% / {100*r.high_daily_sd:.2f}%\n隐含持续期：{r.implied_low_duration:.1f} / {r.implied_high_duration:.1f} 个交易日',transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m 月'));ax.set_ylabel('高波动状态后验概率');ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('同一名称的高、低波动状态，其尺度和持续时间会随年度拟合改变\n每年只用前一年及更早数据拟合；当年参数冻结，不使用未来平滑概率',fontsize=14,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'annual_filtered_states.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

def main():
    v=read(OUT/'verification.json');assert v['status']=='PASS';protocol=read(ROOT/'protocol.json');fit=read(OUT/'training_manifest.json')
    metrics=csv('ensemble_metrics');years=csv('yearly_metrics');hmm=csv('hmm_fit_summary');signals=csv('hmm_signal_states');daily=csv('hmm_daily_states');states=csv('hmm_state_metrics');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json');parts=csv('component_summary');train=csv('training_metrics')
    figures(metrics,years,hmm,signals,daily)
    counts=pd.crosstab(signals.cutoff,signals.hmm_bucket).reindex(columns=['low_probability','uncertain','high_probability'],fill_value=0);countrows=[]
    for r in hmm.itertuples():
        g=signals[signals.cutoff.eq(r.cutoff)];countrows.append([str(int(r.cutoff[:4])+1),len(g),pct(g.high_probability.mean()),*counts.loc[r.cutoff].astype(int).tolist()])
    chunks=['# 第二十一轮方法测试：两状态波动识别与共享信号调节','2026-09-12｜沪深 300｜历史研究实验｜所有新模型拟合完成后才进行下一年度过滤与评分',
        '本轮没有找到可以替换原方案的稳定增益。学习表征的 HMM 状态交互在 2015–2017 年为 58/120（48.33%），2018–2020 年为 80/141（56.74%）。原全历史波动交互的后段 82/141（58.16%）继续保留。新方法后段 Brier 为 0.243117，比原交互的 0.244168 略低，但仍高于加性模型的 0.241936；方向与概率质量需要分开判断。',
        '两种新增交互都未通过冻结的跨时期描述筛选。26 项固定比较中没有经 Holm 校正后显著的改善；简单特征状态交互在前段相对训练频率和仅状态模型的 Brier 较差，校正 p 分别为 0.0156 和 0.0225。这里的统计检验只针对本轮固定比较族，不校正此前多轮历史设计选择。',
        '已完成 6 个年度 HMM、30 个分类头和 24 个交互投影的拟合与核验。计算验证 PASS 不等于方法效果通过，也不构成收益率或可交易性结论。',
        '## 实验范围与状态的含义',
        '本轮延续“先识别状态，再调节共享模型”的思路：用日收益识别两个方差状态，仅增加一个沿原信号方向的条件调节项。没有把日历年份直接当作牛熊标签；各年度只是固定的参数估计和下一年评估边界。训练保留全部截止日前已成熟标签且等权。上一轮三年硬窗口结果、原交互及其全部对照均完整保留。',
        '两状态零均值高斯 HMM 假设：r_t | S_t=s ~ N(0, σ_s²)，状态转移矩阵为 P。两种均值都固定为 0，初始分布固定为 (0.5, 0.5)，估计两个方差和两个自保持概率。训练后按方差排序；“高波动”是这个模型内部相对另一状态的名称，不是经过外部标注的真实市场状态，也不直接代表上涨、下跌、趋势或震荡。',
        '模型使用固定单次初始化和 Baum–Welch EM 求似然局部驻点。EM 可有局部最优，独立 L-BFGS-B 复算只检验附近的数值稳定性，没有证明全局最优，也没有执行多初值搜索。算法背景见 [hmmlearn 官方教程](https://hmmlearn.readthedocs.io/en/stable/tutorial.html)。',
        '状态特征采用向前过滤概率。过滤使用截至当时的观测；平滑会利用之后的样本。本轮平滑只用于截止日前 HMM 的训练求解，下一年全部采用冻结参数向前过滤。相关区别见 [statsmodels 官方 Markov switching 示例](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html)。',
        '训练期也使用过滤概率，但参数在整段截止日前样本上拟合，所以它是训练内特征，**不是历史逐日 OOF／预序贯状态识别**。下一年的参数、投影、标准化和分类头均已冻结，首日先从最后训练后验乘以转移矩阵，再结合当天收益；随后逐交易日更新。已验证修改未来后缀不会改变此前概率。',
        '## 数据、缺失值和模型形式',
        '价格源有 4046 根日线，日期为 2010-01-04 至 2026-08-31；本轮评分仍只使用 2015–2020 年已经反复查看的 261 个周信号。日 HMM 用每个截止日前全部日线，分类训练仍按 joint_completed≤cutoff 保证标签成熟。后段的“最近年份”在这组对比里是 2018–2020，并非新取得的 2026 数据。',
        '日收益为相邻有效收盘价的对数差，两根相邻 OHLC 日线都有效才使用该收益。2015-03-27 无效条及其下一条对应的收益均视为缺失。缺失收益不提供发射似然信息，但保留该交易日并推进一次状态转移，既不填充值也不压缩时间。6 次 HMM 拟合累计 10924 个日步（含 10 个缺失），下一年累计过滤 1462 个日步（含 2 个缺失）；这些累计计数含不同折重复使用的历史。',
        '每折分类训练样本为 1077、1190、1434、1678、1921、2165；测试分别为 22、48、50、49、46、46 周。2015 年因已有无效 OHLC／125 日历史窗口规则只有 22 周可评分，不能把这一年解释成完整覆盖。所有旧样本筛选规则保持不变。',
        '令 p_t 为高方差状态的过滤后验，u_t=2p_t−1，r_t^signal=X_t[:25]·w_parent[:25]，q_t=u_t·r_t^signal。在训练期把 q 对 [X,1] 做 OLS 投影，去掉线性可表示部分；残差按训练均值和总体标准差标准化为 h。下一年冻结这些参数，追加 h 到原加性 X，然后使用岭逻辑回归。原监督信号射线同样是训练内估计，不能称为 OOF 预测。',
        '学习分支由 29 个斜率变成 30 个，简单特征分支由 26 个变成 27 个；每个模型只增加一个交互斜率。全部加性斜率和截距可重新拟合，令交互斜率为零仍精确包含原加性模型。残差化改变了岭惩罚的坐标含义，本轮没有另测未经残差化的乘积。',
        '仅状态对照把 u 按分类训练样本均值／标准差标准化后，用一个斜率和截距拟合。所有分类模型使用相同 λ=0.01、等权二元对数损失、截距不惩罚、严格 p>0.5 判涨。学习分支平均 3 个种子的预测概率。共拟合 738 个分类系数（含截距），另有 24 个 HMM 自由参数；投影和标准化参数另行冻结，不包含在这两个计数里。',
        '## 主要结果',
        '准确率越高越好；Brier 和对数损失越低越好。原 MSE 模型输出未经概率校准的收益分数，因此不填概率指标。原交互保留是研究假设留存，不等于已证明它整体优于所有对照。']
    for w in protocol['windows']:
        t=metrics[metrics.window.eq(w['name'])].set_index('method').loc[MAIN].reset_index()
        chunks += ['### '+WINDOWS[w['name']],table(['方法','正确/样本','准确率','平衡准确率','AUROC','Brier','对数损失'],[metric_line(r) for r in t.itertuples()])]
    chunks+=['![两个时期及两种特征分支的结果](state_interaction_comparison.png)',
        '相对学习加性模型，新交互前段只改变 3 个方向：2 个由对变错、1 个由错变对；后段 **141 个方向全部相同**。因此后段的 56.74% 不是相同正确数偶然抵消，而是逐条方向相同；它只移动了概率。相对原学习交互，后段 4 个方向不同，3 个由对变错、1 个由错变对，净少对 2 周。',
        '仅状态对照后段 141 周全部判涨，与训练频率基准的方向相同，Brier 却由 0.248705 增至 0.250205。这组结果没有显示“仅知道高低波动”可以提供额外的方向预测能力。它也不足以证明加入趋势或其他状态特征必然有效。',
        '### 年度细分',
        '以下列出正确周数，避免只看汇总比例。2018 年原学习交互仍为 32/49，而状态交互为 30/49；2019 年状态交互多对 1 周，2020 年少对 1 周。']
    for family in FAMILIES:
        selected=family+['hmm_only','native_mse','training_frequency'];rows=[]
        for year,g in years.groupby('year'):
            t=g.set_index('method');rows.append([year,int(g.iloc[0]['n'])]+[int(t.loc[m,'correct_directions']) for m in selected])
        chunks.append(table(['年份','样本']+[LABELS[m] for m in selected],rows))
    chunks+=['## 状态识别诊断',
        '最值得继续查的是状态含义的年度稳定性。用于预测 2015 年的模型，低／高状态日标准差约 0.53%／1.63%，隐含持续期只有 1.20／2.36 个交易日；用于预测 2016 年的模型变成约 1.23%／3.14%，持续期约 46.62／6.96 日。首折更接近快速切换的收益方差混合，不能仅凭有两个隐变量就称为长期市场阶段。',
        '本轮预先设置的是数值收敛、方差可分和最低软占用要求，没有强制长期持续的状态；首折通过计算合同不能证明经济解释稳定。这个现象可能涉及零均值高斯发射、单初值局部解、样本增加或状态尺度变化，本轮没有把这些可能原因做因果区分，也没有看到结果后加限制重跑。',
        table(['预测年','低状态日 SD','高状态日 SD','低自保持 P','高自保持 P','低持续日','高持续日','EM 步数'],[[int(r.cutoff[:4])+1,pct(r.low_daily_sd),pct(r.high_daily_sd),f'{r.stay_low:.4f}',f'{r.stay_high:.4f}',f'{r.implied_low_duration:.2f}',f'{r.implied_high_duration:.2f}',r.iterations] for r in hmm.itertuples()]),
        '持续期 1/(1−P_ss) 是拟合马尔可夫链的几何期望，不是人工标注的实际市场阶段长度。',
        table(['预测年','评分周数','平均高状态后验','低后验 <0.2','不确定 0.2–0.8','高后验 >0.8'],countrows),
        '2017 年评分的 50 周全部落在低后验组。2018–2020 年高后验组每年只有 4 周，合计 12/141；前段为 13/120。软门控仍使用全部连续概率，本表的 0.2/0.8 只用于描述，没有用于模型训练或交易切换。稀疏组无法支持可靠的单组策略选择。',
        '![冻结参数后的逐日状态概率](annual_filtered_states.png)',
        '### 固定状态组的表现',
        '全部三个后验组都报告，少于 20 个样本标为稀疏。这里的子组结果只用于诊断，不据此增加或删除训练日期。旧趋势／波动／区间／成交量分组的 480 行指标也全部保留。']
    for w in protocol['windows']:
        rows=[]
        for state in ['low_probability','uncertain','high_probability']:
            for m in ['learned_market','learned_vol_interaction','learned_hmm_interaction','raw_hmm_interaction','hmm_only','training_frequency']:
                r=states[states.window.eq(w['name'])&states.state.eq(state)&states.method.eq(m)].iloc[0];rows.append([state,LABELS[m],int(r.n),pct(r.accuracy),num(r.brier),'是' if r['sparse'] else '否'])
        chunks += ['### '+WINDOWS[w['name']]+' 状态组',table(['状态组','方法','样本','准确率','Brier','稀疏'],rows)]
    chunks+=['## 预定比较与描述筛选',
        '每个时期 13 项比较，共 26 项，候选减对照损失，小于 0 表示改善。循环区块长度为 8 个保留观测，10000 次重采样，随机种子 20260910，每个时期单独生成。95% 区间是边际区间，不是同时区间；双侧中心化 p 在全部 26 项内做 Holm 校正。没有用合并时期做主要假设检验。',
        table(['时期','候选 / 对照','损失','差值','95% 区间','原始 p','Holm p'],[pair_line(r) for r in pairs]),
        '11 条冻结的描述条件：准确率严格超过原加性、原交互、原 MSE、训练频率（4 条）；Brier 低于原加性、原交互、训练频率、仅状态（4 条）；对数损失低于训练频率（1 条）；至少两年准确率超过原 MSE、至少两年 Brier 低于训练频率（2 条）。平局不通过，两段均需全部通过；仅状态对照不自动晋升为候选策略。',
        table(['时期','候选','满足条件','跨时期通过','原交互继续保留'],[[WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",'是' if a['cross_period_descriptive_pass'] else '否','是'] for a in assess]),
        '## 训练与实现复核',
        f"6 个 HMM 共执行 {fit['new_em_iterations']} 次 EM 更新；30 个分类头共执行 {fit['new_newton_iterations']} 次 Newton 更新。全部满足冻结的收敛阈值，24 个交互投影秩增加 1 且精确嵌套父模型。独立 QR 重算投影、独立 L-BFGS-B 重算分类目标均通过。HMM 另经穷举小样本状态路径、有限差分导数、独立对数域递推、附近的似然复算，以及未来后缀扰动验证。",
        f"逐条新预测的最大复算差异为 {v['maximum_new_forecast_error']:.3e}；HMM 对数似然最大复算差异为 {v['maximum_hmm_likelihood_gap']:.3e}；HMM 独立局部优化后的全序列状态概率最大变化为 {v['maximum_hmm_alternate_probability_gap']:.3e}（冻结阈值 0.001）。独立分类器最大目标差为 {v['maximum_alternate_objective_gap']:.3e}、训练概率差为 {v['maximum_alternate_probability_gap']:.3e}。复算优化参数从未用于提交的预测。",
        '本轮没有新神经网络前向或权重训练；复用先前已核验的 18 个冻结神经状态和对应缓存。新增 1305 条单模型概率，完整保留 7308 条旧模型记录，共 8613 条；20 个方法各 261 条集成记录，共 5220 条。独立复核了 852 组指标、190 个可靠性分箱和 26 项固定比较。旧证据 3061 个文件的内容哈希保持不变。',
        '训练内表现仅作拟合诊断，不能当作泛化效果。下表学习分支为三种子训练指标均值，简单分支和仅状态为单次拟合。',
        table(['方法','截止日','训练准确率','训练 Brier','带惩罚目标'],[[LABELS[m],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())] for (m,cutoff),g in train.groupby(['method','cutoff'])]),
        '## 研究决定与下一步',
        '保留全历史交互及后段 58.16% 的结果，保留加性模型作更强的概率误差对照。本轮 HMM 状态交互不替代它们；也不根据高后验小组结果筛选交易或扩大专家网络。市场状况不同这一研究方向仍然合理，但本轮只检验“日收益方差状态 × 一个原有信号方向”，不足以否定或确认所有状态方法。',
        '下一轮优先研究状态是否可重复识别、是否提供现有连续波动特征以外的信息。可以把相对波动和趋势持续性分别作为小规模候选，先在截止日前训练内的时间顺序划分上检查占用、持续期、年度尺度稳定性和增量预测；再考虑少量状态偏置或斜率。若继续 HMM，持续性约束或更适合尾部的发射分布需要预先冻结并与当前无约束版本对照。这里是后续研究建议，本轮没有实施或证明它们有效。',
        '状态标识和监督射线的训练内特征也应逐步改为时间顺序生成的 OOF 特征，减少训练与下一年识别条件的不一致。每次只改变一个方面，继续保留原交互，避免同时更换状态、时间权重、神经表征和分类目标后无法归因。',
        '## 限制与可追溯文件',
        '这 261 个历史日期已经用于多轮方法设计，**不能作为独立确认**。训练周收益标签在日频样本间重叠，样本并不独立；8 观测区块重采样只能提供特定假设下的区间。状态后验没有外部真实标签，识别概率的校准程度未知；本轮既不是完整论文复现，也没有测试交易成本、执行或净收益。2021–2026 年的部分日期同样有历史使用记录，不能仅通过改名取得新留出集。',
        f"冻结协议 SHA-256：`{v['protocol_sha256']}`。",
        '主要文件：[冻结协议](../protocol.json)、[数值验证](verification.json)、[HMM 拟合参数](hmm_fit_summary.csv)、[日状态](hmm_daily_states.csv)、[周状态](hmm_signal_states.csv)、[逐模型预测](model_predictions.csv)、[集成预测](ensemble_predictions.csv)、[全部汇总指标](ensemble_metrics.csv)、[年度指标](yearly_metrics.csv)、[种子指标](seed_metrics.csv)、[旧状态组](state_metrics.csv)、[HMM 状态组](hmm_state_metrics.csv)、[可靠性分箱](reliability_bins.csv)、[方向变化](direction_changes.csv)、[交互贡献](interaction_components.csv)、[固定比较](primary_comparisons.json)、[筛选结果](assessments.json)。',
        '图表矢量版：[结果比较 SVG](state_interaction_comparison.svg)、[逐年状态 SVG](annual_filtered_states.svg)。复现依次执行 prepare21.py → contract21.py → train21.py → score21.py → evaluate21.py → verify21.py；在独立副本和空结果目录运行，不覆盖当前冻结证据。科学计算解释器为 research_v4/.venv_gpu/Scripts/python.exe，报告和绘图使用默认 Python。']
    report='\n\n'.join(chunks)+'\n';(OUT/'第二十一轮测试报告.md').write_text(report,encoding='utf-8')
    note='# 第二十一轮研究留存\n\n2026-09-12：已完成 6 个两状态 HMM 与 30 个分类头。当前 HMM 状态交互未通过跨时期描述筛选。保留全历史交互，尤其 2018–2020 年学习分支的 58.16%，作为后续研究假设；不覆盖第十九、二十轮证据。\n\n新状态交互后段准确率 56.74%，与原加性模型逐条方向一致。下一步优先检查状态尺度／持续期稳定性和训练内时间顺序 OOF 生成，再测试少量条件参数。\n\n[完整报告](results/第二十一轮测试报告.md)｜[协议](protocol.json)｜[验证](results/verification.json)\n'
    (ROOT/'README.md').write_text(note,encoding='utf-8');print(json.dumps(dict(report_characters=len(report),figures=2,status='BUILT_NOT_YET_VISUALLY_REVIEWED')))

if __name__=='__main__':main()

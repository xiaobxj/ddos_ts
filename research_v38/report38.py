from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
ML={'learned_market':'R18','learned_vol_interaction':'R19','learned_order_extension':'R23','learned_order_offset':'R25'};PRIMARY=list(ML)[1:]
SL={'negative_low':'负趋势／低波动','negative_high':'负趋势／高波动','nonnegative_low':'非负趋势／低波动','nonnegative_high':'非负趋势／高波动'}
VL={'daily':'新增全部日样本','friday':'新增周五子集','future':'随后评估周信号'};HL={'annual_state_offset':'分状态修正','annual_shared_offset':'统一修正','annual_state_offset_half':'状态修正减半'}
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def pp(v):return '—' if pd.isna(v) else f'{v*100:+.2f}'
def pct(v):return '—' if pd.isna(v) else f'{v:.2%}'
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    prep=read('preparation_manifest.json');assert prep['protocol_sha256']==sha(ROOT/'protocol.json')
    for k in ['source_sha256','input_sha256']:
        for n,d in prep[k].items():assert sha(PROJECT/n)==d
    for n,d in read('diagnosis_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    v=read('verification.json');assert v['status']=='PASS';assert not (OUT/'report_manifest.json').exists();start=pd.Timestamp.now(tz='UTC').isoformat();m=csv('cohort_metrics');t=csv('residual_transfer');s=csv('transfer_summary');d=csv('residual_decomposition');o=csv('target_overlap');b=csv('brier_summary');rank=csv('within_cell_ranking');files=[]
    parts=['# 第三十八轮：新增日样本偏差能否迁移到后续周信号','',
    '本轮完成只读诊断，没有新增神经训练、分类／修正参数拟合、特征推理、预测策略或显著性检验。全部 R37 概率和成绩保持原值；用已保存的年度训练 logit 和周预测重新计算描述指标。所有历史结果此前已经查看，下面的后续标签只用于回顾诊断，未参与模型或门槛选择。','',
    '## 三个固定样本视角','',
    '每个季度比较：相对该年度训练集新增且联合标签成熟的全部日样本、其中周五子集、以及原路由分配给下一季度的周五评估信号。三者使用相同的年度模型和 R36 四状态边界。新增集按年度累计，跨季度重复；不能汇总其出现次数当成独立样本。年初新增集为空的单元也保留。','',
    '模型残差定义为实际上涨标签 y 减年度上涨概率 p。正值表示该样本中年度预测偏低，负值表示偏高；不是实际上涨率减年度训练标签率。融合残差使用三个种子概率的等权均值，未对平均 logit 取 sigmoid。','',
    '日样本和周样本比较可以显示取样差异；周五训练子集与随后周信号比较可进一步观察时间变化。但训练子集更小，差异不能自动解释为因果、可靠的启用规则或需要反向交易。','',
    '## 执行标签区间重叠','',
    '按原标签 entry 到 exit 的交易日边区间 [entry, exit) 计算覆盖。只共享边界不算收益区间重叠。“最多可取不重叠区间数”是区间调度计数，并非独立有效样本数；即便没有机械重叠，收益仍可能相关。这里没有量化辅助标签和 125 日输入上下文的重叠。','']
    latest=[]
    for year in range(2021,2027):latest.append(o[o.year.eq(year)].cutoff.max())
    g=o[o.cutoff.isin(latest)&o.state.eq('all')&o.view.isin(['daily','friday'])]
    parts += [table(['截止','样本视角','n','收益区间平均覆盖次数','最大覆盖次数','重叠区间对数','最多可取不重叠区间数'],[[r.cutoff,VL[r.view],r.n,num(r.mean_edge_multiplicity),r.maximum_edge_multiplicity,r.overlapping_pairs,r.maximum_disjoint_intervals] for r in g.itertuples()]),'',
    '全部 23 个截止、三个视角和四状态／整体的 345 个区间单元均保存。训练与随后预测的执行标签区间没有交叉，前后比较不是把同一段未成熟收益同时放进两边。','',
    '## 偏差方向是否延续','',
    '以下每个单元是一个“季度截止×状态”，仅统计 R37 原规则允许修正的单元：新增日样本不少于 20 个，而且所选训练视角及随后状态周信号均非空。相同方向表示训练残差和随后残差同号；反向表示异号。接近零（绝对值≤1e−12）单列。周数是这些单元覆盖的实际后续周数，不是预测准确率；单元之间的训练集重复，不能当作独立试验。','']
    for period,label in [('extension_2021_2023','2021—2023'),('recent_2024_2026','2024—2026 年 8 月'),('year_2026','2026 年已有 30 周')]:
        g=s[s.period.eq(period)&s.seed.eq(-1)&s.method.isin(PRIMARY)&s.cohort.eq('r37_eligible')];parts += [f'### {label}','',table(['方法','训练视角','可比较单元','同向单元','反向单元','近零单元','同向／覆盖后续周'],[[ML[r.method],VL[r.view],r.cells,r.same_cells,r.opposite_cells,r.near_zero_cells,f'{r.same_weeks}/{r.future_weeks}'] for r in g.itertuples()]),'']
    strict=s[s.period.eq('pooled_2021_2026')&s.seed.eq(-1)&s.method.isin(PRIMARY)&s.cohort.eq('weekly_count10')&s.view.eq('friday')]
    parts += ['### 小样本限制','',table(['方法','同时有至少10个新增周五和10个随后周信号的单元','覆盖后续周'],[[ML[r.method],r.cells,r.future_weeks] for r in strict.itertuples()]),'',
    '这只是更严格的计数标记，没有用于生成新预测。所有单元、种子、非空比较及空单元在 residual_transfer.csv 和 transfer_summary.csv 中完整保存；未以同号率筛选状态。','',
    '## 2026 年：三个视角下的状态残差','']
    for cutoff in ['2026-03-31','2026-06-30']:
        g=t[t.cutoff.eq(cutoff)&t.method.eq('learned_order_extension')&t.seed.eq(-1)];parts += [f'### R23，截至 {cutoff}','',table(['状态','新增日 n／残差百分点','新增周五 n／残差百分点','随后周 n／残差百分点','日→随后','周五→随后'],[[SL[r.state],f'{r.daily_n}／{pp(r.daily_residual)}',f'{r.friday_n}／{pp(r.friday_residual)}',f'{r.future_n}／{pp(r.future_residual)}',r.daily_future,r.friday_future] for r in g.itertuples()]),'']
    parts += ['## 取样与时间变化的组成分解','',
    '对残差均值差异，先固定来源状态内残差，计算状态占比变化项 Σ(w目标−w来源)r来源；再以目标占比加权状态内残差变化。两项相加等于总残差变化。若目标出现来源中没有样本的状态，则两项不可定义，不补造该状态残差。该分解依赖所规定顺序，只是记账。','',
    table(['截止','来源→目标','来源 n','目标 n','总变化／百分点','占比项','组内项','状态'],[[r.cutoff,VL[r.source]+'→'+VL[r.target],r.source_n,r.target_n,pp(r.overall_delta),pp(r.composition),pp(r.within_state),r.status] for r in d[d.year.eq(2026)&d.method.eq('learned_order_extension')&d.seed.eq(-1)].itertuples()]),'',
    '## 排序能力：截距修正能改变什么','',
    '同一季度、同一状态、同一种子内，加一个共同 logit 截距再取 sigmoid 是严格递增变换，无法改善该单元内部的样本排序。融合三个种子则不一定是原平均概率的共同单调变换；跨状态、跨季度汇总也可能改变排序。因此不能从局部排序不变推断年度 AUROC 不变。','',
    f"本轮 {len(rank)} 个排序单元中，单种子共 {v['per_seed_rank_inversions']} 对逆序，融合预测共 {v['ensemble_rank_inversions']} 对逆序；所有并列改变和单类／空单元均保存。AUROC 缺失表示无法在该单元比较正负样本，不作 0.5 填充。",'',
    table(['截止','视角','状态','n','上涨／下跌','AUROC','Brier','残差／百分点'],[[r.cutoff,VL[r.view],SL.get(r.state,'整体'),r.n,f'{r.up_n}/{r.down_n}',num(r.auroc),num(r.brier),pp(r.residual_mean)] for r in m[m.year.eq(2026)&m.method.eq('learned_order_extension')&m.seed.eq(-1)&m.state.eq('all')].itertuples()]),'',
    '## R37 概率损失变化的精确记账','',
    '令 Δp=修正概率−年度概率，e=y−年度概率，则 Brier 变化为 Δp²−2Δp·e。第一项总非负；第二项用随后已观察残差衡量修正是否朝向标签。两项严格还原原 R37 的 Brier 差，不能用其事后符号建立本轮新策略。','',
    table(['时段','方法','原 R37 方案','周数','幅度平方项','与后续残差对齐项','Brier 总变化'],[[r.period,ML[r.method],HL[r.history],r.n,num(r.squared_shift),num(r.alignment_term),num(r.brier_change)] for r in b[b.seed.eq(-1)&b.scope.eq('all')&b.method.isin(PRIMARY)&b.period.isin(['recent_2024_2026','year_2026'])].itertuples()]),'',
    '正的 Brier 总变化表示变差。这里只重述原概率产生的损失，没有重新训练或改变预测。全部周、种子及仅启用周的分解另存。','',
    '## 复核与研究边界','',
    f"独立区间计数和动态规划复核 {v['overlap_cells']} 个区间单元；独立概率来源最大差 {v['maximum_source_probability_gap']:.3g}；{v['metric_cells']} 个指标单元、{v['transfer_cells']} 个迁移单元、{v['decomposition_cells']} 个分解单元和 {v['ranking_cells']} 个排序单元复核通过。Brier 恒等式最大误差 {v['maximum_brier_identity_gap']:.3g}。",'',
    '5,428 个旧证据文件保持不变，R37 的 48 项探索比较原样保留。本轮无新增 p 值，不把同号计数解释为未来提升，不从这些已查看历史中选择启用阈值。旧 R25 近期 73/125、58.4% 仍单独保留；自然 20 遍年度 R25 对照是 71/125、56.8%。','',
    '![训练残差与随后残差](residual_transfer.png)','',
    '![2026 年 Brier 变化分解](brier_accounting_2026.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十八轮诊断报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files.append(p)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    eligible=t[t.seed.eq(-1)&t.method.isin(PRIMARY)&t.r37_eligible&t.future_n.gt(0)];xlimit=max(10,float(np.ceil(eligible.daily_residual.abs().max()*10)*10+5));ylimit=max(10,float(np.ceil(eligible.future_residual.abs().max()*10)*10+5))
    fig,axs=plt.subplots(1,3,figsize=(16,6.8),sharex=True,sharey=True);fig.subplots_adjust(left=.075,right=.88,top=.80,bottom=.20,wspace=.14);norm=matplotlib.colors.BoundaryNorm(np.arange(2020.5,2027.5),6);cmap=matplotlib.colormaps['viridis'].resampled(6)
    for ax,method in zip(axs,PRIMARY):
        g=t[t.seed.eq(-1)&t.method.eq(method)&t.r37_eligible&t.future_n.gt(0)];sc=ax.scatter(g.daily_residual*100,g.future_residual*100,s=25+g.future_n*13,c=g.year,cmap=cmap,norm=norm,edgecolors='white',linewidth=.6,alpha=.85);ax.axhline(0,color='#777777',lw=1);ax.axvline(0,color='#777777',lw=1);ax.set_title(ML[method]);ax.set_xlabel('新增全部日样本：残差均值（百分点）');ax.set_xlim(-xlimit,xlimit);ax.set_ylim(-ylimit,ylimit);ax.grid(alpha=.15);ax.spines[['right','top']].set_visible(False)
    axs[0].set_ylabel('随后周信号：残差均值（百分点）');cax=fig.add_axes([.905,.25,.016,.49]);fig.colorbar(sc,cax=cax,ticks=range(2021,2027),label='预测年度');fig.suptitle('相同状态内，新增样本的偏差是否延续到后续？',fontsize=18,y=.96);fig.text(.5,.865,'每点为一个 R37 可启用的季度／状态单元；点越大，覆盖的随后周数越多。',ha='center',fontsize=11);fig.text(.5,.045,'左上与右下象限表示残差反向。小样本未删除；全部历史已查看，图中同号率不是准确率。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'residual_transfer.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,6.5),sharey=True);fig.subplots_adjust(left=.075,right=.965,top=.80,bottom=.28,wspace=.14);hist=list(HL);x=np.arange(3)
    for ax,method in zip(axs,PRIMARY):
        g=b[b.seed.eq(-1)&b.method.eq(method)&b.scope.eq('all')&b.period.eq('year_2026')].set_index('history').loc[hist];ax.bar(x-.17,g.squared_shift,.32,color='#0077bb',label='幅度平方项');ax.bar(x+.17,g.alignment_term,.32,color='#ee7733',label='与后续残差对齐项');ax.plot(x,g.brier_change,'ko',label='Brier 总变化')
        for j,val in enumerate(g.brier_change):ax.annotate(f'{val:+.4f}',(j,val),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks(x,['分状态','统一','状态减半']);ax.set_title(ML[method]);ax.axhline(0,color='#555555',lw=1);ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False);ax.margins(y=.3)
    axs[0].set_ylabel('相对年度基线的 Brier 变化（正值变差）');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False);fig.suptitle('2026 年 30 周：原 R37 修正为何改变概率损失',fontsize=18,y=.95);fig.text(.5,.028,'两项相加严格等于总变化。使用原预测作事后记账，本轮没有新增预测。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'brier_accounting_2026.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=start,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R38 Chinese report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

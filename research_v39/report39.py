from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';ANNUAL='rolling5_annual20'
HL={ANNUAL:'年度基线','weekly_state_validated':'状态修正＋验证','weekly_state_ungated':'状态修正直接使用','weekly_global_validated':'统一修正＋验证','weekly_global_ungated':'统一修正直接使用'}
ML={'learned_market':'R18','learned_vol_interaction':'R19','learned_order_extension':'R23','learned_order_offset':'R25'};PRIMARY=list(ML)[1:]
SL={'negative_low':'负趋势／低波动','negative_high':'负趋势／高波动','nonnegative_low':'非负趋势／低波动','nonnegative_high':'非负趋势／高波动','all':'全部状态'}
RL={'annual_reset':'年度重置','insufficient_history':'历史不足','insufficient_state_train':'状态训练不足','insufficient_validation':'状态验证不足','brier_not_better':'Brier 未改善','direction_worse':'方向正确数减少','accepted':'通过预定规则'}
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def pct(v):return '—' if pd.isna(v) else f'{v:.2%}'
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    prep=read('preparation_manifest.json');assert prep['protocol_sha256']==sha(ROOT/'protocol.json')
    for k in ['source_sha256','input_sha256']:
        for n,d in prep[k].items():assert sha(PROJECT/n)==d
    for phase in ['fitting','validation','scoring','evaluation']:
        for n,d in read(f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    v=read('verification.json');assert v['status']=='PASS';assert not (OUT/'report_manifest.json').exists();start=pd.Timestamp.now(tz='UTC').isoformat();schedule=csv('schedule');met=csv('ensemble_metrics');sm=csv('seed_metrics');gates=csv('gate_decisions');changes=csv('direction_changes');qo=csv('quarter_outcomes');pairs=read('primary_comparisons.json');fit=read('fitting_manifest.json');files=[]
    parts=['# 第三十九轮：成熟周样本训练、后续保留验证与修正启用','',
    '本轮实际拟合了新的概率修正参数，运行按历史时间顺序的保留验证，再计算之后的历史预测。年度网络、年度分类主系数、交互项和年度状态边界全部保留。全部历史此前已查看，本轮是探索性历史回放，不是新盲测，也不存在当年真实发出的本轮修正交易记录。','',
    '## 预先固定的规则','',
    '每个非年末季度截止，取最后 13 个联合标签已成熟的原年度周信号作验证。用第一个验证信号前一天作为修正器训练截止，只选该日前已经成熟的最近 52 个周标签作训练。相邻旧周即使信号日期更早，标签尚未成熟也要隔离，不能简单将相邻 65 行切成 52＋13。52 和 13 是信号数量，不是严格的一年和一季度。','',
    '历史不足时精确使用年度概率。年末保持年度重置，下一年第一季度不启用修正。每季度都独立训练当前候选，不累加旧修正。所有参数先冻结，再用保留验证标签决定是否启用；通过后也不将验证标签加入重拟合。','',
    '统一修正：每种子用 52 个训练周拟合一个截距。分状态修正：每状态训练周至少 10 个才拟合一个截距，否则为零。两者均使用 Σ二分类对数损失＋20d²/2，且 |d|≤0.5。强度和幅度沿用 R37，未搜索。','',
    '验证先分别修正三个种子概率，再等权平均。统一方案在全部 13 个验证周上检查；状态方案分别要求该状态至少 5 个验证周。只有 Brier 比年度低超过 1e−12，且方向正确周数不减少，才允许在随后信号使用该修正。严格大于 0.5 判上涨。计数和双指标门槛是固定启用规则，不代表显著性或模型已被证明可靠。','',
    '每个家族都保留“不经验证直接使用”的对照，它与通过验证的方案使用完全相同的较早 52 周训练参数。两者差异可以检查验证门控的作用；两种家族之间还存在支持样本和总惩罚量差异，不能当作仅改变状态数量的纯消融。原 native_mse 和训练频率均复制年度对照。','',
    '历史训练输入采用当时年度模型的原始周预测和当时年度边界给出的相对状态。没有用当前网络回算过去，也没有用当前中位数重新标过去的状态。校准样本因此跨越不同年度模型和不同绝对波动阈值；保留验证检验的是这个顺序校准流程，不能宣称当前／新年度网络本身经过独立验证。','',
    f"本轮有 {v['ready_quarters']} 个非年末季度满足训练和验证总量要求；实际拟合 {fit['scalar_fits']} 个标量参数，其中状态 {fit['state_fits']}、统一 {fit['global_fits']}。新增神经训练、变换拟合和特征推理均为 0。",'',
    '## 时间边界与可用样本','',table(['决定日','处理','累计成熟周','训练周','验证周','修正训练截止','验证最早信号','边界隔离旧周'],[[r.cutoff,RL.get(r.mode,r.mode),r.available,r.train_n,r.validation_n,r.fit_cutoff if isinstance(r.fit_cutoff,str) else '—',r.validation_first if isinstance(r.validation_first,str) else '—',r.purged_between_n] for r in schedule.itertuples()]),'',
    '年度重置行中的数量仅用于审计潜在历史可用性，不表示实际进行了拟合或启用。训练最晚成熟日早于验证最早信号；验证最晚成熟日不晚于决定日；所有之后评估信号的日期严格晚于决定日。','']
    for period,label in [('extension_2021_2023','2021—2023：147 周'),('recent_2024_2026','2024—2026 年 8 月：125 周'),('pooled_2021_2026','全段：272 周')]:
        rows=[]
        for m in PRIMARY:
            for h in HL:
                r=met[met.period.eq(period)&met.history.eq(h)&met.method.eq(m)].iloc[0];rows.append([ML[m],HL[h],f'{r.correct_directions}/{r.n}',pct(r.accuracy),num(r.brier),num(r.log_loss),num(r.auroc)])
        parts += [f'## {label}','',table(['方法','方案','正确／周数','准确率','Brier↓','对数损失↓','AUROC'],rows),'']
    parts += ['## 各年方向结果','']
    for m in PRIMARY:
        rows=[]
        for year in range(2021,2027):
            g=met[met.period.eq(f'year_{year}')&met.method.eq(m)].set_index('history');rows.append([year,int(g.loc[ANNUAL,'n'])]+[f"{int(g.loc[h,'correct_directions'])}（{pct(g.loc[h,'accuracy'])}）" for h in HL])
        parts += [f'### {ML[m]}','',table(['年份','周数']+list(HL.values()),rows),'']
    parts += ['## 启用覆盖与改善／退步','',table(['时期','方法','方案','可拟合覆盖周','验证通过覆盖周','概率实际变化周','错→对','对→错'],[[r.period,ML[r.method],HL[r.history],r.fit_eligible_weeks,r.validation_accepted_weeks,r.changed_probability_weeks,r.recoveries,r.regressions] for r in changes[changes.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])&changes.method.isin(PRIMARY)].itertuples()]),'',
    '验证通过覆盖数也附在直接使用对照上，用于比较同一批候选通过了哪些检查；直接使用对照不会据此回退。概率改变不一定跨过 0.5，也不一定改变方向。','',
    '## 所有通过验证的状态／统一候选','',table(['决定日','方法','家族','状态','训练周','验证周','验证 Brier 差','验证正确数：年度→候选'],[[r.cutoff,ML[r.method],'分状态' if r.family=='state' else '统一',SL[r.component],r.training_n,r.validation_n,num(r.brier_difference),f'{r.baseline_correct}→{r.candidate_correct}'] for r in gates[gates.accepted&gates.method.isin(PRIMARY)].itertuples()]),'',
    '全部 460 个门控单元包含年度重置、历史不足、状态训练不足、状态验证不足、Brier 未改善和方向减少等拒绝原因。所有决定均在后续预测评分前冻结。','',
    '### R23 在 2026 年的全部决定','',table(['决定日','家族','状态','训练周','验证周','验证 Brier 差','验证正确数：年度→候选','决定'],[[r.cutoff,'分状态' if r.family=='state' else '统一',SL[r.component],r.training_n,r.validation_n,num(r.brier_difference),f'{r.baseline_correct}→{r.candidate_correct}',RL[r.reason]] for r in gates[gates.method.eq('learned_order_extension')&gates.cutoff.isin(['2026-03-31','2026-06-30'])].itertuples()]),'',
    '## 验证后的下一季度','',
    '在门控冻结以后，才计算下表的后续表现。通过但后续没有该状态信号的单元也保留为 n=0。这些后续结果没有反过来改变决定。完整季度表同时包含未通过候选的直接使用结果。','',
    table(['决定日','方法','家族','状态','后续周','通过验证后 Brier 差','错→对','对→错'],[[r.cutoff,ML[r.method],'分状态' if r.family=='state' else '统一',SL[r.component],r.n,num(r.brier_difference),r.recoveries,r.regressions] for r in qo[qo.accepted&qo.method.isin(PRIMARY)&qo.history.isin(['weekly_state_validated','weekly_global_validated'])].itertuples()]),'',
    '## 种子范围','',table(['时期','方法','方案','种子最低／最高准确率','融合准确率'],[[p,ML[m],HL[h],pct(sm[sm.period.eq(p)&sm.method.eq(m)&sm.history.eq(h)].accuracy.min())+'／'+pct(sm[sm.period.eq(p)&sm.method.eq(m)&sm.history.eq(h)].accuracy.max()),pct(met[met.period.eq(p)&met.method.eq(m)&met.history.eq(h)].accuracy.iloc[0])] for p in ['extension_2021_2023','recent_2024_2026'] for m in PRIMARY for h in HL]),'',
    '## 预定探索比较','',
    '4 个候选各自对年度、两个验证版各自对直接使用版、状态验证对统一验证，乘三个主要方法、方向错误和 Brier 两种损失、两个时期，共 84 项。固定 8 周循环区块，10,000 次配对 bootstrap，中心化双侧 p，再统一 Holm 调整。差值为候选损失减参考损失，负值有利。该调整不覆盖历轮探索，也不能消除已查看历史的选择影响。','',
    f"84 项中调整后 p<0.05 的数量：{sum(r['holm_adjusted_p']<.05 for r in pairs)}；最小调整 p：{min(r['holm_adjusted_p'] for r in pairs):.6f}。",'',
    table(['时期','方法','候选','参考','损失','差值','95% 区间','原 p','Holm p'],[[r['window'],ML[r['candidate']],HL[r['history']],HL[r['reference_history']],r['metric'],num(r['difference']),f"[{num(r['ci95_low'])}, {num(r['ci95_high'])}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
    '## 审计与限制','',
    f"独立复核 {v['independent_scalar_solutions']} 个标量解，最大参数差 {v['maximum_scalar_gap']:.3g}；最大预测差 {v['maximum_prediction_gap']:.3g}。训练接口 {v['training_isolation_interfaces']} 次非成员标签／概率污染检查、23 个验证接口的非验证行污染检查均通过。所有训练／验证／评估边界、门控、精确回退、{v['independent_metric_cells']} 个指标单元和 84 项比较均复核。",'',
    '5,484 个旧证据文件保持不变。原 R25 近期 73/125、58.4% 继续单独保留；当前自然 20 遍年度 R25 为 71/125、56.8%。R37 在 2025 年的局部改善及 2026 年退步均保留。','',
    '本轮只检验固定的周样本校准与验证规则。验证期领先仍可能不延续，拒绝候选也可能错失后续改善；样本不足与年度回退会降低启用覆盖。不能以未显著认定等效，也不能把局部较高准确率当作稳定进步。未做交易收益、成本或实盘部署测试。','',
    '![各年准确率](yearly_accuracy.png)','',
    '![修正覆盖与验证门控](correction_coverage.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十九轮实验报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files.append(p);font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    colors=['#222222','#0077bb','#66ccee','#cc6677','#ddaa33'];styles=['-','-','--','-','--'];fig,axs=plt.subplots(1,3,figsize=(16.5,6.7),sharey=True);fig.subplots_adjust(left=.065,right=.965,top=.8,bottom=.27,wspace=.14)
    for ax,m in zip(axs,PRIMARY):
        for (h,label),color,ls in zip(HL.items(),colors,styles):
            g=met[met.method.eq(m)&met.history.eq(h)&met.period.str.startswith('year_')].sort_values('period');ax.plot(range(2021,2027),g.accuracy,marker='o',lw=2,color=color,ls=ls,label=label)
        ax.set_title(ML[m]);ax.set_xticks(range(2021,2027),['2021','2022','2023','2024','2025','2026*'],rotation=20);ax.set_ylim(.2,.8);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False);fig.suptitle('先训练、再验证：周样本修正能否改善后续预测？',fontsize=18,y=.95);fig.text(.5,.02,'每年周数 50、49、48、47、48、30；* 2026 年截至 8 月。早期样本不足及每年 Q1 保留年度预测。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'yearly_accuracy.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6.3),sharey=True);fig.subplots_adjust(left=.075,right=.965,top=.79,bottom=.25,wspace=.15);x=np.arange(3);width=.19;hists=list(HL)[1:]
    for ax,period,label in zip(axs,['extension_2021_2023','recent_2024_2026'],['2021—2023：147 周','2024—2026 年 8 月：125 周']):
        for j,(h,color) in enumerate(zip(hists,colors[1:])):
            g=changes[changes.period.eq(period)&changes.history.eq(h)].set_index('method').loc[PRIMARY];bars=ax.bar(x+(j-1.5)*width,g.changed_probability_weeks,width,label=HL[h],color=color)
            for rect,val in zip(bars,g.changed_probability_weeks):ax.annotate(str(int(val)),(rect.get_x()+width/2,val),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
        ax.set_xticks(x,[ML[m] for m in PRIMARY]);ax.set_title(label);ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False);ax.margins(y=.25)
    axs[0].set_ylabel('相对年度概率实际改变的周数');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False);fig.suptitle('验证门控实际保留了多少修正？',fontsize=18,y=.95);fig.text(.5,.01,'概率改变不等于方向改变，也不等于预测改善。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'correction_coverage.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=start,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R39 Chinese report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

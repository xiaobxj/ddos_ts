from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';ANNUAL='rolling5_annual20'
HL={ANNUAL:'年度基线','weekly_state_validated':'季度修正＋验证','monthly_state_validated':'月度修正＋验证','weekly_state_ungated':'季度直接使用','monthly_state_ungated':'月度直接使用'}
ML={'learned_market':'R18','learned_vol_interaction':'R19','learned_order_extension':'R23','learned_order_offset':'R25'};PRIMARY=list(ML)[1:]
SL={'negative_low':'负趋势／低波动','negative_high':'负趋势／高波动','nonnegative_low':'非负趋势／低波动','nonnegative_high':'非负趋势／高波动'}
RL={'annual_reset':'年度重置','q1_guard':'第一季度回退','insufficient_history':'历史不足','insufficient_state_train':'状态训练不足','insufficient_validation':'状态验证不足','brier_not_better':'Brier 未改善','direction_worse':'方向正确数减少','accepted':'通过预定规则'}
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
    v=read('verification.json');assert v['status']=='PASS';assert not (OUT/'report_manifest.json').exists();start=pd.Timestamp.now(tz='UTC').isoformat()
    schedule=csv('schedule');met=csv('ensemble_metrics');seeds=csv('seed_metrics');gates=csv('gate_decisions');changes=csv('direction_changes');trans=csv('gate_transitions');cadence=csv('cadence_summary');pairs=read('primary_comparisons.json');fit=read('fitting_manifest.json');files=[]
    parts=['# 第四十轮：成熟周样本状态修正，月度与季度更新比较','',
        '本轮只改变修正器训练与保留验证的联合更新频率。年度模型、交互项、状态定义、52 周训练／13 周验证、成熟隔离、状态训练至少 10 周／验证至少 5 周、正则 20 和截距上限 0.5 均沿用 R39。验证要求三种子等权概率的 Brier 降低超过 1e−12，且方向正确周数不减少；严格大于 0.5 判上涨。','',
        '验证通过后不重拟合，不叠加旧修正。月度验证与月度直接使用共享同一批训练参数。样本不足或验证拒绝时精确复制年度概率。所有年份第一季度均保留年度预测，因此年末重置之外，1 月和 2 月的月底决定也不启用修正。月末周五信号仍使用此前月底决定，不能使用当天结束后才形成的修正。','',
        '这是训练和验证一起变频，不是只改变停止时机。额外月度更新会改变训练／验证成员、拟合参数和启用决定。所有历史结果此前已查看，本轮是探索性历史回放，没有新盲测、没有新增神经网络训练，也没有进行收益、交易成本或部署测试。','',
        f"共 {len(schedule)} 个月底决定，其中 {int(schedule['mode'].eq('ready').sum())} 个满足总量要求；实际拟合 {fit['scalar_fits']} 个状态标量，{int(gates.accepted.sum())}/{len(gates)} 个门控单元通过预定规则（包括诊断方法 R18）。同一状态和方法的多个月份并非独立试验。",'',
        '## 预先冻结的时间安排','',table(['决定日','处理','训练截止','训练周','验证周','验证首信号','验证最晚成熟'],[(r.cutoff,RL.get(r.mode,r.mode),r.fit_cutoff if isinstance(r.fit_cutoff,str) else '—',r.train_n,r.validation_n,r.validation_first if isinstance(r.validation_first,str) else '—',r.validation_max_maturity if isinstance(r.validation_max_maturity,str) else '—') for r in schedule.itertuples()]),'',
        '训练标签必须在第一个验证信号之前已经成熟；验证标签必须不晚于决定日成熟；评分信号严格晚于决定日。重置或回退行保留潜在成员数量仅作审计，不代表实际拟合。52 和 13 是周信号数量，不是严格的日历年和季度。','']
    for period,label in [('extension_2021_2023','2021—2023：147 周'),('recent_2024_2026','2024—2026 年 8 月：125 周'),('pooled_2021_2026','全段：272 周'),('year_2026','2026 年截至 8 月：30 周')]:
        rows=[]
        for m in PRIMARY:
            for h in HL:
                r=met[met.period.eq(period)&met.method.eq(m)&met.history.eq(h)].iloc[0];rows.append((ML[m],HL[h],f'{int(r.correct_directions)}/{int(r.n)}',pct(r.accuracy),num(r.brier),num(r.log_loss),num(r.auroc)))
        parts+=['## '+label,'',table(['方法','方案','正确／周数','准确率','Brier↓','对数损失↓','AUROC'],rows),'']
    parts+=['## 逐年方向正确数','']
    for m in PRIMARY:
        rows=[]
        for y in range(2021,2027):
            g=met[met.period.eq(f'year_{y}')&met.method.eq(m)].set_index('history');rows.append([y,int(g.loc[ANNUAL,'n'])]+[int(g.loc[h,'correct_directions']) for h in HL])
        parts+=['### '+ML[m],'',table(['年份','周数']+list(HL.values()),rows),'']
    rows=[]
    for p in ['extension_2021_2023','recent_2024_2026','year_2026']:
        for r in changes[changes.period.eq(p)&changes.method.isin(PRIMARY)].itertuples():rows.append([p,ML[r.method],HL[r.history],r.fit_eligible_weeks,r.validation_accepted_weeks,r.changed_probability_weeks,r.recoveries,r.regressions])
    parts+=['## 月度方案相对年度的覆盖与方向变化','',table(['时期','方法','方案','可拟合周','验证通过覆盖周','概率实际改变周','错→对','对→错'],rows),'',
        '“验证通过覆盖周”也附在直接使用对照上供核对；直接使用不会据此回退。概率改变不一定改变方向。','',
        '## 月度相对对应季度：改善与退步','',table(['时期','方法','月度方案','比较周','概率变化周','错→对','对→错','Brier 差↓'],[(r.period,ML[r.method],HL[r.history],r.n,r.changed_vs_quarter,r.recoveries,r.regressions,num(r.brier_difference)) for r in cadence[cadence.relation.eq('all')&cadence.method.isin(PRIMARY)&cadence.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])].itertuples()]),'',
        '## 额外月份的状态启用／停用登记','',
        '下表保留所有主要方法在额外月底发生的启用或停用，包含随后无该状态信号的单元。启用和停用按相邻月底的验证决定变化定义；第一季度回退单独标明。后续 Brier 仅作事后诊断，未参与门控。正的后续 Brier 差不等于已估计出统计误报率。','',
        table(['决定日','方法','状态','处理','变化','季度门控启用','后续周','后续 Brier 相对季度'],[(r.cutoff,ML[r.method],SL[r.state],RL.get(r.mode,r.mode),r.transition,'是' if r.quarterly_accepted else '否',r.future_n,num(r.future_brier_vs_quarter)) for r in trans[trans.extra_month&trans.method.isin(PRIMARY)&trans.transition.isin(['enabled','stopped'])].itertuples()]),'',
        '## R23 在 2026 年的月度验证决定','',table(['决定日','状态','训练周','验证周','验证 Brier 差','验证正确数：年度→候选','原因'],[(r.cutoff,SL[r.component],r.training_n,r.validation_n,num(r.brier_difference),f'{r.baseline_correct}→{r.candidate_correct}',RL.get(r.reason,r.reason)) for r in gates[gates.method.eq('learned_order_extension')&gates.cutoff.between('2026-03-01','2026-07-31')].itertuples()]),'',
        '## 近期各种子表现','',table(['方法','方案','种子','正确数／125','Brier↓'],[(ML[r.method],HL[r.history],r.seed,r.correct_directions,num(r.brier)) for r in seeds[seeds.period.eq('recent_2024_2026')&seeds.method.isin(PRIMARY)&seeds.history.isin(HL)].itertuples()]),'',
        '## 预先固定的 60 项探索性比较','',
        '月度验证对季度验证、月度直接对季度直接、月度验证对月度直接，以及两个新方案分别对年度：每种比较使用三个主要方法、方向误差与 Brier、两个固定时期。循环 8 周区块重采样 10,000 次，固定种子 20260910，中心化双侧 p 值，60 项统一 Holm 校正。误差差值越低越好。校正不覆盖此前多轮研究和自适应提出本轮假设。','',
        f"本轮最小校正 p 值为 {min(r['holm_adjusted_p'] for r in pairs):.6f}；校正后小于 0.05 的比较有 {sum(r['holm_adjusted_p']<.05 for r in pairs)} 项。",'',
        table(['时期','方法','候选','参照','损失','差','95% 区间','原始 p','Holm p'],[(r['window'],ML[r['candidate']],HL[r['history']],HL[r['reference_history']],r['metric'],num(r['difference']),f"[{num(r['ci95_low'])}, {num(r['ci95_high'])}]",num(r['p']),num(r['holm_adjusted_p'])) for r in pairs]),'',
        '## 独立审计与解释边界','',
        f"独立核对 {v['independent_scalar_solutions']} 个标量解、{v['independent_metric_cells']} 个指标单元与 60 项比较。季度重合点的 {v['shared_quarter_head_cells']} 个参数单元、{v['shared_quarter_gate_cells']} 个门控单元与 R39 一致；{v['exact_first_month_seed_predictions']} 条对应首月种子预测与季度方案完全相同，第一季度 {v['q1_exact_annual_seed_predictions']} 条种子预测精确回退年度。",'',
        f"训练非成员污染接口 {v['training_isolation_interfaces']} 次、验证非成员污染检查 {v['validation_isolation_cutoffs']} 个截止日均通过。独立复核 {v['cadence_weekly_cases']} 个周比较、{v['cadence_summary_cells']} 个汇总单元和 {v['gate_transition_cells']} 个状态转移登记。5,554 个旧证据文件保持不变。",'',
        '旧 R25 近期 73/125、58.4% 的训练预算结果继续单独保留；本轮自然 20 遍年度 R25 仍为 71/125、56.8%。上一轮状态验证 R23 的 73/125、58.4% 同样保留，不与旧 R25 混同。R37 的 2025 年局部改善及 2026 年退步均保留。','',
        '较多更新机会可能带来更及时的调整，也可能带来偶然通过。启用次数、样本覆盖和平均准确率都需结合逐年、逐周变化看待。少量验证周不足以证明状态可稳定迁移。训练输入跨不同年度模型与年度相对状态，不能宣称当前年度网络自身获得了独立验证。','',
        '![各年准确率](yearly_accuracy.png)','', '![月度与季度覆盖](cadence_coverage.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第四十轮实验报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files.append(p)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    colors=['#222222','#0077bb','#cc6677','#66ccee','#ddaa33'];styles=['-','-','-','--','--'];fig,axs=plt.subplots(1,3,figsize=(16.5,6.7),sharey=True);fig.subplots_adjust(left=.065,right=.965,top=.8,bottom=.27,wspace=.14)
    for ax,m in zip(axs,PRIMARY):
        for (h,label),color,ls in zip(HL.items(),colors,styles):
            g=met[met.method.eq(m)&met.history.eq(h)&met.period.str.startswith('year_')].sort_values('period');ax.plot(range(2021,2027),g.accuracy,marker='o',lw=2,color=color,ls=ls,label=label)
        ax.set_title(ML[m]);ax.set_xticks(range(2021,2027),['2021','2022','2023','2024','2025','2026*'],rotation=20);ax.set_ylim(.2,.8);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False);fig.suptitle('固定训练与验证规则：月度更新是否优于季度更新？',fontsize=18,y=.95);fig.text(.5,.02,'每年周数 50、49、48、47、48、30；* 2026 年截至 8 月。所有方案保留每年第一季度年度预测。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'yearly_accuracy.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6.3),sharey=True);fig.subplots_adjust(left=.075,right=.965,top=.79,bottom=.25,wspace=.15);x=np.arange(3);width=.19;oldweekly=csv('quarterly_weekly_effects');newweekly=csv('weekly_policy_effects');allweekly=pd.concat([oldweekly,newweekly],ignore_index=True)
    for ax,low,high,label in zip(axs,['2021-01-01','2024-01-01'],['2023-12-31','2026-08-31'],['2021—2023：147 周','2024—2026 年 8 月：125 周']):
        for j,(h,color) in enumerate(zip(list(HL)[1:],colors[1:])):
            counts=[int(allweekly[allweekly.history.eq(h)&allweekly.method.eq(m)&allweekly.date.between(low,high)].changed_probability.sum()) for m in PRIMARY];bars=ax.bar(x+(j-1.5)*width,counts,width,label=HL[h],color=color)
            for rect,val in zip(bars,counts):ax.annotate(str(val),(rect.get_x()+width/2,val),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
        ax.set_xticks(x,[ML[m] for m in PRIMARY]);ax.set_title(label);ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False);ax.margins(y=.25)
    axs[0].set_ylabel('相对年度概率实际改变的周数');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False);fig.suptitle('更多复核机会带来了多少实际修正？',fontsize=18,y=.95);fig.text(.5,.01,'覆盖更多不等于预测改善；月度方案同步重新训练并验证。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'cadence_coverage.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=start,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R40 Chinese report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

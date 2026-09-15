from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
ML={'learned_market':'R18','learned_vol_interaction':'R19','learned_order_extension':'R23','learned_order_offset':'R25'};PRIMARY=list(ML)[1:];CL={'monthly':'月度','quarterly':'季度'}
SL={'negative_low':'负趋势／低波动','negative_high':'负趋势／高波动','nonnegative_low':'非负趋势／低波动','nonnegative_high':'非负趋势／高波动'}
SENS={'accepted_metric_fragile':'删一周会改变指标判断','accepted_count_only_fragile':'仅最低数量门槛敏感','accepted_single_drop_stable':'删任意一周仍通过','rejected_metric_fragile':'拒绝：删一周可改变指标判断','rejected_single_drop_stable':'拒绝：删任意一周判断不变','all_supported':'全部支持充足单元','accepted_all':'全部通过单元'}
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def pct(v):return '—' if pd.isna(v) else f'{v:.1%}'
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    prep=read('preparation_manifest.json');assert prep['protocol_sha256']==sha(ROOT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for n,d in prep[key].items():assert sha(PROJECT/n)==d
    for n,d in read('diagnosis_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    verification=read('verification.json');assert verification['status']=='PASS';assert not (OUT/'report_manifest.json').exists();start=pd.Timestamp.now(tz='UTC').isoformat();files=[]
    cells=csv('gate_sensitivity');cohorts=csv('sensitivity_summary');flows=csv('membership_flows');trans=csv('gate_transitions');runs=csv('acceptance_runs');leave=csv('leave_one_week_out')
    parts=['# 第四十一轮：验证门控的单周敏感性、样本复用与支持流失','',
        '本轮是只读诊断。沿用 R39 季度与 R40 月度状态修正的全部参数、验证概率、门控决定和已评分的后续预测；没有训练、参数更新、门控重选、预测重算或新增显著性检验。原有 22 条预测历史和 5,633 个旧证据文件保持不变。','',
        '原验证规则为：较早 52 个成熟周训练、随后 13 个成熟周验证，按原年度相对趋势／波动状态分组；状态训练至少 10 周、验证至少 5 周；三种子等权概率的 Brier 降低且方向正确数不减少。训练标签在验证开始之前成熟，验证标签不晚于决定日成熟。第一季度回退规则保持不变。','',
        '## 单周敏感性怎样测','',
        '仅对当时总量和状态支持均足够的单元，逐一删除一个验证周，以原来已冻结的参数和三种子概率重新计算剩余验证周的指标。训练集、参数与后续预测均不改变。支持不足的原单元仍保留登记，不通过删除样本获得资格。','',
        '分别记录两种判断：①暂时保持原支持资格，仅检查剩余周的 Brier 与方向正确数；②完整执行原门控，重新要求剩余样本至少 5 周。若原来刚好 5 周，删一周后只有 4 周，第二种必然拒绝；这与指标是否依赖单个观察值分开报告。删除一个周是一项敏感性诊断，不是新的资格认证，也不是置信区间。','',
        f"共 {len(cells)} 个原门控单元，其中 {int(cells.metric_eligible.sum())} 个满足敏感性检查所需的原始支持条件；实际核对 {len(leave)} 次单周删除。季度和月度、不同方法之间存在共享信息，不能将单元数或删除次数当作独立样本量。",'',
        '## 全部历史：通过与拒绝的敏感性','',
        table(['频率','方法','类别','单元数','指标敏感单元','原样本正好 5 周','删除次数','改变指标判断次数'],[(CL[r.cadence],ML[r.method],SENS[r.cohort],r.cells,r.metric_fragile_cells,r.minimum_support_cells,r.loo_deletions,r.metric_flips) for r in cohorts[cohorts.period.eq('decision_all_2020_2026')&cohorts.method.isin(PRIMARY)&cohorts.cohort.isin(SENS)].itertuples()]),'',
        '“删一周会改变指标判断”按原接受／拒绝状态计算；原通过单元变成拒绝，或原支持充足的拒绝单元变成通过。分类“仅最低数量门槛敏感”要求指标判断对每一次删除都保持通过，但原样本数正好 5。','',
        '## 分时期的已通过单元','',table(['决定日期时期','频率','方法','通过单元','指标敏感单元','正好 5 周单元'],[(r.period,CL[r.cadence],ML[r.method],r.cells,r.metric_fragile_cells,r.minimum_support_cells) for r in cohorts[cohorts.cohort.eq('accepted_all')&cohorts.method.isin(PRIMARY)&~cohorts.period.eq('decision_all_2020_2026')].itertuples()]),'',
        '时期按门控决定日归属，不是按之后信号日期归属。分时期诊断不能直接当作原固定测试窗口的准确率差。','',
        '## 相邻更新重复用了多少样本','',
        '每个频率分别比较相邻决定日，对训练／验证、全部样本／四个原状态，逐一登记保留、滚出和新进入的行。满足 当前数量＝之前数量−滚出数量＋新进入数量。以下主表只汇总前后两个决定日都处于 ready 的相邻对，所有回退和历史不足行仍保留在明细。','']
    rows=[]
    for cadence in ['monthly','quarterly']:
        for role,label in [('training','训练'),('validation','验证')]:
            g=flows[flows.cadence.eq(cadence)&flows.role.eq(role)&flows.state.eq('all')&flows.both_ready];rows.append([CL[cadence],label,len(g),f'{g.current_n.mean():.2f}',f'{g.retained_n.mean():.2f}',f'{g.added_n.mean():.2f}',pct(g.retained_n.sum()/g.current_n.sum())])
    parts+=[table(['频率','角色','相邻对数','平均当前数量','平均保留数量','平均新进数量','保留占当前比例'],rows),'',
        '复用比例仅计数同一个历史周是否再次出现；它不是有效独立样本量，也没有校正标签之间可能存在的相关性。参数也可能随更新改变，因此同标签复用不等于整项试验完全重复。','',
        '## 通过后因验证支持不足而停用','',
        table(['频率','方法','之前决定日','停用日','状态','前后均 ready','之前周数','保留','滚出','新进','当前周数'],[(CL[r.cadence],ML[r.method],r.previous_cutoff,r.cutoff,SL[r.state],'是' if r.both_ready else '否',r.previous_validation_n,r.retained_n,r.dropped_n,r.added_n,r.validation_n) for r in trans[trans.stop_by_validation_support&trans.method.isin(PRIMARY)].itertuples()]),'',
        '这里的停用原因来自原门控，数量流失来自明确的成员对账；不能把原状态样本滚出短窗口解释为经济状态已经失效，也不能事后假设保留原修正一定更好。支持不足是信息不充分，既可能防止错误启用，也可能错失之后的改善。','',
        '## R23 连续通过的全部记录','',
        table(['频率','状态','开始','结束','连续决定数','验证使用次数','不同验证周','重复使用次数','使用次数／不同周'],[(CL[r.cadence],SL[r.state],r.first_cutoff,r.last_cutoff,r.decisions,r.total_validation_uses,r.unique_validation_weeks,r.repeated_uses,f'{r.uses_per_unique_week:.2f}') for r in runs[runs.method.eq('learned_order_extension')].itertuples()]),'',
        '每个通过单元恰好归入一段连续记录；包含只有一次通过的记录。运行中的累计使用次数不能代替独立验证次数。强制年度重置和第一季度回退会打断连续记录。','',
        '## 敏感性与已经发生的后续表现','',
        '下表只连接原来已经评分的候选直接使用结果；通过单元与原验证方案相同。保留后续没有对应信号的单元。后续表现未用于删除样本、修改门控或构造新策略。方法之间和月度／季度之间重复使用了数据，表中比例不是独立胜率。','',
        table(['频率','方法','通过类别','单元','有后续信号','后续周','Brier 变差单元','Brier 变好单元','按周加权 Brier 差↓'],[(CL[r.cadence],ML[r.method],SENS[r.cohort],r.cells,r.future_nonempty_cells,r.future_weeks,r.future_worse_cells,r.future_better_cells,num(r.future_week_weighted_brier)) for r in cohorts[cohorts.period.eq('decision_all_2020_2026')&cohorts.method.isin(PRIMARY)&cohorts.cohort.isin(['accepted_metric_fragile','accepted_count_only_fragile','accepted_single_drop_stable'])].itertuples()]),'',
        '## 旧结果与审计','',
        '本轮没有产生新的准确率。R40 近期月度验证 R19／R23／R25 为 68／71／70 个正确周，R39 季度验证为 69／73／72，均以 125 周为分母。2026 年两种频率的验证方案逐周预测相同。原 R25 近期 73/125、58.4% 属于旧训练预算协议；自然 20 遍年度 R25 是 71/125、56.8%；R39 季度验证 R23 的 73/125、58.4% 单独保留。R37 的 2025 年局部改善和 2026 年退步也保留。','',
        f"独立复核 {verification['independent_validation_week_cells']} 个验证周贡献、{verification['independent_deletion_cases']} 次删除、{verification['independent_sensitivity_cohorts']} 个汇总、{verification['independent_role_flows']} 个成员流汇总、{verification['independent_member_flows']} 条成员流明细、{verification['independent_gate_transitions']} 个门控转移及 {verification['independent_acceptance_runs']} 段连续通过记录；共对账 {verification['accepted_cells_accounted']} 个已通过单元。",'',
        '全部历史此前已查看，本轮是探索性诊断，没有新增盲测、p 值或因果结论。删除单周后的判断稳定也可能在后续失败，不能直接用本轮敏感性分类作为新的启用门槛。','',
        '![通过门控的敏感性](gate_sensitivity.png)','', '![相邻更新样本复用](sample_reuse.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第四十一轮诊断报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files.append(p)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(13.5,6.5),sharey=True);fig.subplots_adjust(left=.07,right=.965,top=.8,bottom=.25,wspace=.15);colors=['#cc6677','#ddaa33','#0077bb'];classes=['accepted_metric_fragile','accepted_count_only_fragile','accepted_single_drop_stable'];labels=['删除一周改变指标判断','仅最低数量门槛敏感','删除任意一周仍通过'];x=np.arange(3)
    for ax,cadence in zip(axs,['monthly','quarterly']):
        bottom=np.zeros(3)
        for cls,color,label in zip(classes,colors,labels):
            values=np.array([int((cells.cadence.eq(cadence)&cells.method.eq(m)&cells.sensitivity_class.eq(cls)).sum()) for m in PRIMARY]);bars=ax.bar(x,values,.58,bottom=bottom,color=color,label=label)
            for rect,n,b in zip(bars,values,bottom):
                if n:ax.text(rect.get_x()+rect.get_width()/2,b+n/2,str(n),ha='center',va='center',color='white' if color!='#ddaa33' else '#222222',fontsize=11)
            bottom+=values
        for j,total in enumerate(bottom):ax.text(j,total+.4,f'共 {int(total)}',ha='center',fontsize=10)
        ax.set_xticks(x,[ML[m] for m in PRIMARY]);ax.set_title(CL[cadence]);ax.margins(y=.22);ax.grid(axis='y',alpha=.15);ax.spines[['right','top']].set_visible(False)
    axs[0].set_ylabel('原来通过的门控单元数');handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False);fig.suptitle('删去一个验证周，原来通过的门控有多稳定？',fontsize=18,y=.95);fig.text(.5,.02,'保持原参数；指标变化与样本从 5 周降至 4 周分开。单元数量不是独立试验数。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'gate_sensitivity.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12.5,6.3));fig.subplots_adjust(left=.08,right=.96,top=.8,bottom=.25,wspace=.25)
    for ax,role,title in zip(axs,['training','validation'],['52 周训练窗口','13 周验证窗口']):
        gg=[flows[flows.cadence.eq(c)&flows.role.eq(role)&flows.state.eq('all')&flows.both_ready] for c in ['monthly','quarterly']];ret=np.array([g.retained_n.mean() for g in gg]);new=np.array([g.added_n.mean() for g in gg]);ax.bar([0,1],ret,.52,label='从上次保留',color='#0077bb');ax.bar([0,1],new,.52,bottom=ret,label='本次新进入',color='#66ccee')
        for j,(a,b) in enumerate(zip(ret,new)):ax.text(j,a/2,f'{a:.2f}',ha='center',va='center',color='white');ax.text(j,a+b/2,f'{b:.2f}',ha='center',va='center');ax.text(j,a+b+.5,f'复用 {a/(a+b):.1%}',ha='center')
        ax.set_xticks([0,1],[f'月度（{len(gg[0])} 对）',f'季度（{len(gg[1])} 对）']);ax.set_title(title);ax.set_ylabel('每次更新的平均周样本数');ax.set_ylim(0,(ret+new).max()*1.2);ax.grid(axis='y',alpha=.15);ax.spines[['right','top']].set_visible(False)
    handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False);fig.suptitle('相邻更新增加了多少新样本？',fontsize=18,y=.95);fig.text(.5,.015,'仅前后两个截止日都满足总量要求的相邻对；复用比例不代表有效独立样本量。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'sample_reuse.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=start,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R41 Chinese diagnostic report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

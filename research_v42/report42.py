from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results';ANNUAL='rolling5_annual20';MONTHLY='monthly_state_validated';QUARTER='weekly_state_validated';NEW='monthly_state_retained'
HL={ANNUAL:'年度基线',QUARTER:'季度修正＋验证',MONTHLY:'月度立即回退',NEW:'月度有限期保留'};ML={'learned_market':'R18','learned_vol_interaction':'R19','learned_order_extension':'R23','learned_order_offset':'R25'};PRIMARY=list(ML)[1:]
SL={'negative_low':'负趋势／低波动','negative_high':'负趋势／高波动','nonnegative_low':'非负趋势／低波动','nonnegative_high':'非负趋势／高波动'}
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def pct(v):return '—' if pd.isna(v) else f'{v:.2%}'
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    prep=read('preparation_manifest.json');assert prep['protocol_sha256']==sha(ROOT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for n,d in prep[key].items():assert sha(PROJECT/n)==d
    for phase in ['routing','scoring','evaluation']:
        for n,d in read(phase+'_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    v=read('verification.json');assert v['status']=='PASS';assert not (OUT/'report_manifest.json').exists();start=pd.Timestamp.now(tz='UTC').isoformat();files=[]
    met=csv('ensemble_metrics');seeds=csv('seed_metrics');decisions=csv('retention_decisions');outcomes=csv('decision_outcomes');weekly=csv('weekly_policy_effects');carried=csv('carried_weeks');coverage=csv('retention_coverage');comp=csv('reference_comparisons');pairs=read('primary_comparisons.json')
    parts=['# 第四十二轮：验证状态样本不足时，有限期保留上次通过的修正','',
        '本轮新增一条历史预测路由规则，复用 R40 已冻结的月度训练参数和验证门控。没有重新训练神经网络或标量修正，也没有改变年度模型、交互项、四状态定义、52／13 周训练验证、成熟隔离、正则 20、截距上限 0.5 或验证指标门槛。所有历史此前已查看，本轮不是新盲测。','',
        '## 预先冻结的行为','',
        '每种方法和状态独立保存最近一次正式通过验证的三种子原参数。当前正式通过时，用当前参数替换记录；固定到期日为该正式通过日之后的第一个季度末。仅当当前训练支持仍足够、原门控唯一拒绝原因是验证状态周数少于 5 时，才可暂时使用尚未到期且仍属于同一年度模型／状态边界的记录。','',
        '保留不刷新来源日期，也不延长到期日。其他拒绝原因均清空记录，包括 Brier 未改善、方向正确数减少、训练或整体历史不足。记录清空后不能因下一次缺样本而复活。年末与第一季度回退同样清空；绝不把当前未经验证的新参数当成被保留参数。','',
        '到期日对预测信号包含当天：例如 3 月 31 日通过，可用于日期不晚于 6 月 30 日的后续信号。月末／季末当天的周信号只能使用此前月底决定；6 月 30 日形成的新决定面向更晚的信号，所以旧记录此时已不能继续保留。该日重新正式通过可以建立到期为 9 月 30 日的新记录。','',
        f"全部 1,088 个决定单元先冻结后评分：{v['fresh_cells']} 个使用当前正式通过的参数，{v['carried_cells']} 个保留旧参数，{v['fallback_cells']} 个回退年度（含 R18 诊断方法）。参数来源和到期日没有使用后续结果。新方案只针对支持不足的处理，不保证挡住当前验证通过但随后失效的修正。",'']
    for period,label in [('extension_2021_2023','2021—2023：147 周'),('recent_2024_2026','2024—2026 年 8 月：125 周'),('pooled_2021_2026','全段：272 周'),('year_2026','2026 年截至 8 月：30 周')]:
        rows=[]
        for method in PRIMARY:
            for history in HL:
                r=met[met.period.eq(period)&met.method.eq(method)&met.history.eq(history)].iloc[0];rows.append([ML[method],HL[history],f'{int(r.correct_directions)}/{int(r.n)}',pct(r.accuracy),num(r.brier),num(r.log_loss),num(r.auroc)])
        parts+=['## '+label,'',table(['方法','方案','正确／周数','准确率','Brier↓','对数损失↓','AUROC'],rows),'']
    parts+=['## 逐年正确数','']
    for method in PRIMARY:
        rows=[]
        for year in range(2021,2027):
            g=met[met.method.eq(method)&met.period.eq(f'year_{year}')].set_index('history');rows.append([year,int(g.loc[ANNUAL,'n'])]+[int(g.loc[h,'correct_directions']) for h in HL])
        parts+=['### '+ML[method],'',table(['年份','周数']+list(HL.values()),rows),'']
    parts+=['## 所有主要方法的保留决定及后续覆盖','',
        '保留决定包括之后没有对应状态信号的单元。保留单元数、覆盖周数、概率实际变化周数和方向变化周数不是同一个量。来源年龄从最初正式通过日计算，而非从最近一次保留日计算。','',
        table(['决定日','方法','状态','参数来源','固定到期','后续周','概率变化周','错→对','对→错','Brier 相对月度差↓'],[(r.cutoff,ML[r.method],SL[r.state],r.source_cutoff,r.expiry,r.n,r.changed_vs_monthly,r.recoveries_vs_monthly,r.regressions_vs_monthly,num(r.brier_vs_monthly)) for r in outcomes[outcomes.action.eq('carry')&outcomes.method.isin(PRIMARY)].itertuples()]),'',
        '## 所有主要方法实际使用保留参数的周','',
        table(['信号日','方法','状态','参数来源','到期','来源年龄／日','原月度概率','保留概率','实际上涨','相对月度方向变化'],[(r.date,ML[r.method],SL[r.state],r.source_cutoff,r.expiry,int(r.source_age_days),num(r.monthly_probability),num(r.probability),r.actual_up,r.case_vs_monthly) for r in carried[carried.method.isin(PRIMARY)].itertuples()]),'',
        '## 各时期覆盖','',table(['时期','方法','总周','新通过周','保留周','回退周','相对月度概率变化周','保留来源最大年龄'],[(r.period,ML[r.method],r.n,r.fresh_weeks,r.carry_weeks,r.fallback_weeks,r.changed_vs_monthly,'—' if pd.isna(r.maximum_carried_age) else int(r.maximum_carried_age)) for r in coverage[coverage.method.isin(PRIMARY)&coverage.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])].itertuples()]),'',
        '## 相对各参照的方向改善与退步','',table(['时期','方法','参照','概率变化周','错→对','对→错','Brier 差↓'],[(r.period,ML[r.method],HL[r.reference_history],r.changed_probability_weeks,r.recoveries,r.regressions,num(r.brier_difference)) for r in comp[comp.method.isin(PRIMARY)&comp.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])].itertuples()]),'',
        '## 近期单种子诊断','',table(['方法','方案','种子','正确周／125','Brier↓'],[(ML[r.method],HL[r.history],r.seed,r.correct_directions,num(r.brier)) for r in seeds[seeds.method.isin(PRIMARY)&seeds.history.isin(HL)&seeds.period.eq('recent_2024_2026')].itertuples()]),'',
        '门控由原三种子集成决定，保留时三个种子使用同一次正式通过的各自原参数。单种子结果只作诊断，没有重新挑选种子。','',
        '## 36 项预先固定的探索性比较','',
        '新方案分别对原月度立即回退、季度验证和年度基线，三个主要方法、方向误差和 Brier、两个固定时期。循环 8 周区块重采样 10,000 次，固定种子 20260910，中心化双侧 p 值，36 项统一 Holm 校正。校正不覆盖此前多轮探索和提出本轮假设的自适应过程。','',
        f"最小校正 p 值 {min(r['holm_adjusted_p'] for r in pairs):.6f}；校正后小于 0.05 的比较有 {sum(r['holm_adjusted_p']<.05 for r in pairs)} 项。",'',
        table(['时期','方法','参照','损失','差↓','95% 区间','原始 p','Holm p'],[(r['window'],ML[r['candidate']],HL[r['reference_history']],r['metric'],num(r['difference']),f"[{num(r['ci95_low'])}, {num(r['ci95_high'])}]",num(r['p']),num(r['holm_adjusted_p'])) for r in pairs]),'',
        '## 独立审计与限制','',
        f"独立重建 {v['independent_decision_cells']} 个保留决定，完成 {v['prefix_future_gate_poison_checks']} 个截止日前缀的后续门控污染检查。复核 {v['independent_seed_predictions']} 条新增学习方法种子预测、{v['independent_metric_cells']} 个指标单元、{v['independent_weekly_effects']} 个逐周比较和 {v['independent_decision_outcomes']} 个包含空后续的决定单元。",'',
        f"{v['exact_original_monthly_noncarry_predictions']} 条非保留种子预测与原月度精确相同；{v['carried_seed_predictions']} 条使用保留来源。最大独立预测误差 {v['maximum_prediction_gap']:.3g}。零偏移、第一季度、到期、其他拒绝清空及禁止复活均通过核对。5,691 个旧证据文件和原 22 条预测历史完整保留。",'',
        '本轮并未获得未见过的未来数据。促成本假设的 2024 年支持流失案例已在之前检查，不能把针对该历史现象的改善称为盲测验证。有限期保留仍可能延续失效参数；即使没有新增方向退步，也需要看概率损失和更广时间覆盖。没有进行交易收益、成本或部署测试。','',
        '原 R25 近期 73/125、58.4% 属于旧训练预算协议；当前自然 20 遍年度 R25 为 71/125、56.8%；R39 季度验证 R23 的 73/125、58.4% 单独保留。R37 的 2025 年局部改善及 2026 年退步仍保留。','',
        '![逐年准确率](yearly_accuracy.png)','', '![保留的方向影响](retention_direction_effect.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第四十二轮实验报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files.append(p)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    colors=['#222222','#0077bb','#cc6677','#009988'];fig,axs=plt.subplots(1,3,figsize=(16.5,6.7),sharey=True);fig.subplots_adjust(left=.065,right=.965,top=.8,bottom=.27,wspace=.14)
    for ax,method in zip(axs,PRIMARY):
        for (history,label),color in zip(HL.items(),colors):
            g=met[met.method.eq(method)&met.history.eq(history)&met.period.str.startswith('year_')].sort_values('period');ax.plot(range(2021,2027),g.accuracy,marker='o',lw=2,color=color,label=label)
        ax.set_title(ML[method]);ax.set_xticks(range(2021,2027),['2021','2022','2023','2024','2025','2026*'],rotation=20);ax.set_ylim(.2,.8);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=2,frameon=False);fig.suptitle('状态验证样本不足：有限期保留是否有帮助？',fontsize=18,y=.95);fig.text(.5,.025,'每年周数 50、49、48、47、48、30；* 2026 年截至 8 月。保持同一批年度模型与候选参数。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'yearly_accuracy.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12.5,6.1),sharey=True);fig.subplots_adjust(left=.075,right=.965,top=.8,bottom=.24,wspace=.16);x=np.arange(3)
    for ax,period,title in zip(axs,['extension_2021_2023','recent_2024_2026'],['2021—2023：147 周','2024—2026 年 8 月：125 周']):
        g=comp[comp.period.eq(period)&comp.reference_history.eq(MONTHLY)].set_index('method').loc[PRIMARY]
        for shift,col,color,label in [(-.18,'recoveries','#009988','错→对'),(.18,'regressions','#cc6677','对→错')]:
            bars=ax.bar(x+shift,g[col],.34,color=color,label=label)
            for rect,val in zip(bars,g[col]):ax.annotate(str(int(val)),(rect.get_x()+rect.get_width()/2,val),xytext=(0,4),textcoords='offset points',ha='center')
        ax.set_xticks(x,[ML[m] for m in PRIMARY]);ax.set_title(title);ax.set_ylim(0,max(2,float(comp[comp.reference_history.eq(MONTHLY)][['recoveries','regressions']].to_numpy().max())+1));ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True));ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False)
    axs[0].set_ylabel('相对原月度立即回退的方向变化周数');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False);fig.suptitle('保留参数恢复了哪些正确预测，又增加了多少退步？',fontsize=17,y=.95);fig.text(.5,.02,'概率改变不一定改变方向；完整逐周概率损失与来源年龄见报告。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'retention_direction_effect.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=start,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('R42 Chinese report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

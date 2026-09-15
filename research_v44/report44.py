from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'report_manifest.json').exists();v=json.loads((OUT/'verification.json').read_text(encoding='utf-8'));assert v['status']=='PASS'
    t=pd.read_csv(OUT/'origin_feasibility.csv');s=pd.read_csv(OUT/'feasibility_summary.csv');e=pd.read_csv(OUT/'event_feasibility.csv');o=pd.read_csv(OUT/'origins.csv')
    methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];mn=dict(zip(methods,['R19','R23','R25']))
    classes=['reached_before_expiry','reached_only_at_expiry','not_reached_by_expiry','observation_censored'];names=['到期前可决策','仅到期日才够','完整期限内不足','尚未观察完'];colors=['#009E8E','#E6A03B','#B6BCC4','#7472AD']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,view,title in zip(axes,['cumulative','rolling13'],['固定来源后累计新成熟周','其中仍位于当月最近 13 周的样本']):
        g=s[s.period.eq('all_2020_2026')&s.view.eq(view)].set_index('method').loc[methods];bottom=np.zeros(3)
        for cls,label,color in zip(classes,names,colors):
            vals=g[cls].to_numpy();ax.bar(range(3),vals,bottom=bottom,color=color,label=label)
            for i,z in enumerate(vals):
                if z:ax.text(i,bottom[i]+z/2,str(int(z)),ha='center',va='center')
            bottom+=vals
        ax.set_xticks(range(3),[mn[m] for m in methods]);ax.set_title(title,fontsize=13);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(bottom)*1.13);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.12)
    axes[0].set_ylabel('原正式通过记录数（不同来源可能共享样本）');fig.suptitle('固定参数来源后，能否在原到期前攒够 5 个新成熟状态周？',fontsize=17)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,.055));fig.text(.5,.014,'到期日决定不能继续使用原参数；未知后续不记为失败。结果只表示样本可用性。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.23,wspace=.20)
    for ext in ['png','svg']:fig.savefig(OUT/f'feasibility_by_method.{ext}',dpi=170)
    plt.close(fig)
    g=e[e.view.eq('cumulative')];fig,ax=plt.subplots(figsize=(10,5));bottom=np.zeros(3)
    for cls,label,color in zip(classes,names,colors):
        vals=np.array([sum(g.planned_preexpiry_checkpoints.eq(n)&g.classification.eq(cls)) for n in range(3)]);ax.bar(range(3),vals,bottom=bottom,color=color,label=label)
        for i,z in enumerate(vals):
            if z:ax.text(i,bottom[i]+z/2,str(int(z)),ha='center',va='center')
        bottom+=vals
    ax.set_xticks(range(3),['0 次','1 次','2 次']);ax.set_xlabel('原通过日至到期日之间，严格早于到期日的月度决策次数');ax.set_ylabel('不同正式通过日期×状态事件数');ax.set_title('月度决策机会与信息积累期限',fontsize=17);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(1,max(bottom))*1.16)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,.035));fig.text(.5,.007,'跨方法相同日期／状态已合并；不同来源之间仍可能重叠，不能视为独立市场样本。',ha='center',fontsize=9)
    fig.subplots_adjust(left=.1,right=.98,top=.86,bottom=.30)
    for ext in ['png','svg']:fig.savefig(OUT/f'feasibility_by_opportunity.{ext}',dpi=170)
    plt.close(fig)
    lines=['# 第四十四轮：固定原通过日后的新成熟状态样本积累','',
    '本轮只统计信息到达，不训练或评价新的预测策略。对原 84 次正式通过（R19／R23／R25 共 65 次，另有 R18 诊断方法），分别固定原日期、状态和参数来源。后来的月度通过、清空或替换不会重置该观察记录；这不表示原预测流程仍在使用它。',
    '', '到期日保持为原通过日之后的第一个季度末。按月末检查，只有严格早于到期日的决定才算有机会使用原记录；到期日凑够 5 周单列。新样本按 joint_completed 严格晚于原通过日且不晚于检查日计算，不要求信号日期也晚于原通过日，相关区别已逐周记录。',
    '', '累计口径保留来源后全部新成熟状态周；最近 13 周口径再与当月全市场最近 13 个成熟周取交集。状态、5 周门槛、年度模型边界及成熟定义未改变。计算仅使用时间和状态元数据，不读取收益标签或概率。数据截止为 2026-08-31，8 月底是元数据检查，未新增当月训练或候选门控。',
    '', '## 每种方法的可用性','',
    '| 方法 | 口径 | 原通过次数 | 到期前可决策 | 仅到期日才够 | 完整期限不足 | 尚未观察完且未达到 | 完整期限数 | 已可决策且还有同状态可评价信号 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        for view,label in [('cumulative','累计'),('rolling13','最近 13 周')]:
            r=s[s.period.eq('all_2020_2026')&s.method.eq(m)&s.view.eq(view)].iloc[0]
            lines.append(f'| {mn[m]} | {label} | {r.origins} | {r.reached_before_expiry} | {r.reached_only_at_expiry} | {r.not_reached_by_expiry} | {r.observation_censored} | {r.complete_origins} | {r.ready_with_observed_same_state_followup} |')
    lines+=['','尚未观察完整期限的来源若已达到门槛，会归入已可决策；“尚未观察完且未达到”不是失败。未来同状态信号只统计原冻结数据中已可评价的周，截止后机会未知；不同来源的机会可能重叠，不能相加当作新样本量。','',
    '## 到期前已达到门槛的原来源（累计口径）','',
    '| 方法 | 原通过日 | 状态 | 第 5 周成熟日 | 最早可决策日 | 原到期日 | 剩余日数 | 已观察到的后续同状态可评价周 | 后续期限尚未观察完 |',
    '|---|---|---|---|---|---|---:|---:|---|']
    hit=t[t.method.isin(methods)&t.view.eq('cumulative')&t.classification.eq('reached_before_expiry')]
    for r in hit.sort_values(['method','source_cutoff','state']).itertuples():
        lines.append(f'| {mn[r.method]} | {r.source_cutoff} | {r.state} | {r.fifth_maturity_date} | {r.first_preexpiry_ready_decision} | {r.expiry} | {r.days_remaining_at_preexpiry_ready:g} | {r.observed_followup_same_state_weeks} | {r.followup_observation_censored} |')
    if not len(hit):lines.append('| — | — | — | — | — | — | — | 0 | — |')
    lines+=['','## 按原通过年份（累计口径）','', '| 年份 | 方法 | 来源数 | 到期前可决策 | 仅到期日才够 | 完整期限不足 | 尚未观察完且未达到 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in s[s.period.str.startswith('year_')&s.method.isin(methods)&s.view.eq('cumulative')].sort_values(['period','method']).itertuples():
        lines.append(f'| {r.period[5:]} | {mn[r.method]} | {r.origins} | {r.reached_before_expiry} | {r.reached_only_at_expiry} | {r.not_reached_by_expiry} | {r.observation_censored} |')
    lines+=['','没有原正式通过记录的年份不生成虚构来源。R18、早期／近期、每个检查点、空样本及尚未观察的未来检查点均保存在 CSV。','',
    '## 边界与审计','',
    f"独立复核 {v['independent_origins']} 个来源、{v['independent_planned_checkpoints']} 个预定检查点、{v['independent_count_cells']} 个已观察计数、{v['independent_checkpoint_members']} 条检查点成员、{v['independent_arrivals']} 条来源样本到达记录和 {v['independent_origin_view_cells']} 个来源／口径结果。复核 {v['independent_summary_cells']} 个汇总与 {v['independent_event_view_cells']} 个合并事件／口径单元。",
    '', f"所有收益和概率列被替换后，四张计数输出表保持不变；对 {v['future_metadata_isolation_checkpoints']} 个已观察检查点移除之后才成熟的行，当前计数不变。对 {v['future_gate_prefix_dates']} 个时间前缀更改之后的通过门控，已有来源集合不变。5,800 个旧文件和原 23 条预测历史保持不变。",
    '', '达到 5 周仅表示有机会做验证，不表示旧参数或新参数能通过概率误差与方向门槛，也不是收益改善。累计新信息来自已看过的历史，不是新增盲测；原接受条件本身仍有选择效应。固定观察起点可以发现被月度重置遮蔽的积累过程，但不能据此恢复已经清空的参数或延长期限。',
    '', '原 R25 近期 73/125、58.4% 属于旧训练预算协议；自然 20 遍年度 R25 为 71/125、56.8%；季度验证 R23 的 73/125、58.4% 分开保留。R42 近期 R19／R23／R25 正确周数 69／72／71 不变。本轮没有新的准确率、p 值、模型训练或部署。',
    '', '![分方法可用性](feasibility_by_method.png)','', '![月度机会与可用性](feasibility_by_opportunity.png)','',
    '[逐来源结果](origin_feasibility.csv) · [全部检查点（含未来未知）](checkpoints.csv) · [样本到达记录](observed_arrivals.csv) · [检查点成员](checkpoint_members.csv) · [合并日期／状态事件](event_feasibility.csv) · [独立复核](verification.json)']
    report=OUT/'第四十四轮可行性报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    artifacts=[report]+[OUT/f'{n}.{ext}' for n in ['feasibility_by_method','feasibility_by_opportunity'] for ext in ['png','svg']]
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts}),ensure_ascii=False,indent=2),encoding='utf-8')
    print('R44 report and two figures generated; visual inspection pending.',flush=True)
if __name__=='__main__':main()

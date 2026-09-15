from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hashlib

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    assert not (OUT/'report_manifest.json').exists();v=read(OUT/'verification.json');assert v['status']=='PASS'
    c=pd.read_csv(OUT/'diagnostic_cells.csv');s=pd.read_csv(OUT/'diagnostic_summary.csv');subjects=pd.read_csv(OUT/'subjects.csv')
    methods=['learned_vol_interaction','learned_order_extension','learned_order_offset'];names=dict(zip(methods,['R19','R23','R25']))
    views=['full_validation','post_source_validation','incremental_validation'];vn=['完整验证窗口','原通过后才成熟','上月底后才成熟']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11})
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,m in zip(axes,methods):
        a=s[s.period.eq('all_2020_2026')&s.method.eq(m)&s.availability.eq('in_life')].set_index('view').reindex(views)
        data=[a.old_pass.to_numpy(),(a.old_brier_not_better+a.old_direction_worse).to_numpy(),(a.old_no_samples+a.old_insufficient_validation).to_numpy()];bottom=np.zeros(3)
        for values,color,label in zip(data,['#009E8E','#CC6278','#B6BCC4'],['旧参数通过','旧参数指标未通过','样本不足（含零样本）']):
            ax.bar(np.arange(3),values,bottom=bottom,color=color,label=label)
            for i,z in enumerate(values):
                if z:ax.text(i,bottom[i]+z/2,str(int(z)),ha='center',va='center',fontsize=10)
            bottom+=values
        ax.set_xticks(range(3),vn,rotation=12);ax.set_title(names[m]);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(1,max(bottom))*1.2);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel('原方案尚未到期的保存记录检查次数');fig.suptitle('旧参数再次通过，有多少依赖已知验证周？',fontsize=17)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.04));fig.text(.5,.012,'同一记录、同一周可能重复使用；样本门槛固定为 5 周。计数不代表独立市场事件。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.07,right=.98,top=.81,bottom=.25,wspace=.15)
    for ext in ['png','svg']:fig.savefig(OUT/f'evidence_freshness.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,view,title in zip(axes,views[:2],vn[:2]):
        vals=[]
        for m in methods:
            g=c[c.availability.eq('in_life')&c.method.eq(m)&c.view.eq(view)&c.original_reason.isin(['brier_not_better','direction_worse'])]
            vals.append([sum(g.old_evidence=='pass'),sum(g.old_evidence.isin(['brier_not_better','direction_worse'])),sum(g.old_evidence.isin(['no_samples','insufficient_validation']))])
        bottom=np.zeros(3)
        for j,color,label in zip(range(3),['#009E8E','#CC6278','#B6BCC4'],['旧参数通过','旧参数指标未通过','样本不足（含零样本）']):
            a=np.array(vals)[:,j];ax.bar(range(3),a,bottom=bottom,color=color,label=label)
            for i,z in enumerate(a):
                if z:ax.text(i,bottom[i]+z/2,str(int(z)),ha='center',va='center')
            bottom+=a
        for i,z in enumerate(bottom):ax.text(i,z+.12,f'共 {int(z)}',ha='center',fontsize=10)
        ax.set_ylim(0,max(1,max(bottom))*1.25+.3);ax.set_xticks(range(3),[names[m] for m in methods]);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('当月新候选因指标未通过而清空旧参数的次数');fig.suptitle('新候选未通过时，旧参数的证据是什么？',fontsize=17)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,.055));fig.text(.5,.018,'只统计尚未到期记录；不把到期清空、缺样本或新候选通过混入质量拒绝。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.09,right=.98,top=.80,bottom=.23,wspace=.20)
    for ext in ['png','svg']:fig.savefig(OUT/f'quality_clear_evidence.{ext}',dpi=170)
    plt.close(fig)
    lines=['# 第四十三轮：旧参数与新候选在新增成熟验证数据上的对照','',
    '本轮是冻结历史上的参数诊断。没有新增训练、预测策略、准确率历史或 p 值；原 23 条预测历史保持不变。检查对象是 R42 每月决定发生前实际保存的旧记录，绝不寻找已清空的更早记录。到期记录单列描述，不改变到期规则。',
    '', '同时计算三个固定口径：完整的当月最近 13 个成熟周中属于该状态的周；其中 joint_completed 严格晚于旧参数原通过日的周；其中 joint_completed 严格晚于上月底的周。后两个口径不会重新使用原通过时已成熟的标签。旧参数和当月新候选使用同一组周、同一批原年度概率，先对三个种子概率等权平均，再计算指标。',
    '', '旧参数证据通过要求至少 5 个状态周、Brier 相对年度降低超过 1e-12，且方向正确数不减少。零样本和不足 5 周单独记录，即便观察到的损失较低，也不记为通过。当月未具备拟合条件的候选记为未拟合。此处是证据分类，不构成可投入使用的新门控；训练支持、到期及年度规则仍独立存在。',
    '', '原训练使用更早 52 个成熟周，末尾 13 周用于验证，并隔离未成熟相邻标签；自然 20 遍、滚动 5 年的年度模型及四状态定义均不变。完整窗口使用原时间点保存的年度预测，可能包含年度切换前的模型版本，没有用最新模型重算历史。',
    '', '## 尚未到期的旧记录：完整窗口与新证据','',
    '| 方法 | 证据口径 | 检查次数 | 旧通过 | 指标未通过 | 不足 5 周（含零） | 零样本 | 验证周使用次数 | 不同来源×周 | 重复来源×周使用 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        for view,label in zip(views,vn):
            r=s[s.period.eq('all_2020_2026')&s.method.eq(m)&s.availability.eq('in_life')&s.view.eq(view)].iloc[0]
            lines.append(f'| {names[m]} | {label} | {r.cells} | {r.old_pass} | {r.old_brier_not_better+r.old_direction_worse} | {r.old_no_samples+r.old_insufficient_validation} | {r.old_no_samples} | {r.validation_uses} | {r.unique_source_week_pairs} | {r.repeated_source_week_uses} |')
    lines+=['','计数可跨月份重复，三个方法也共用市场日期；不同来源×周不等于独立样本量。完整窗口中的已知标签、原验证重合数与新增标签数在逐单元文件中列明。','',
    '## 当月新候选因质量拒绝而清空的旧记录','',
    '| 方法 | 月底 | 状态 | 旧来源 | 证据口径 | 周数 | 旧 Brier 差↓ | 旧正确数差 | 旧证据 | 新 Brier 差↓ | 新正确数差 | 新证据 |',
    '|---|---|---|---|---|---:|---:|---:|---|---:|---:|---|']
    q=c[c.availability.eq('in_life')&c.method.isin(methods)&c.original_reason.isin(['brier_not_better','direction_worse'])&c.view.isin(views[:2])]
    def f(x):return '—' if pd.isna(x) else f'{x:.6f}'
    for r in q.sort_values(['method','cutoff','state','view']).itertuples():
        lines.append(f'| {names[r.method]} | {r.cutoff} | {r.state} | {r.source_cutoff} | {vn[views.index(r.view)]} | {r.n} | {f(r.old_brier_difference)} | {r.old_correct_difference:g} | {r.old_evidence} | {f(r.current_brier_difference)} | {r.current_correct_difference:g} | {r.current_evidence} |')
    if not len(q):lines.append('| — | — | — | — | — | 0 | — | — | — | — | — | — |')
    lines+=['','## 原方案保存记录的覆盖','', '| 方法 | 尚未到期 | 到期 | 没有旧记录 |', '|---|---:|---:|---:|']
    for m in methods:
        g=subjects[subjects.method.eq(m)];lines.append(f"| {names[m]} | {sum(g.availability=='in_life')} | {sum(g.availability=='expired')} | {sum(g.availability=='absent')} |")
    lines+=['','全部 1,088 个原决定均保留，每个决定三个证据口径，共 3,264 个诊断单元；没有旧记录的单元记为未评估，不记为失败或通过。包括 R18 的所有方法、年度、近期和早期汇总均见 CSV。',
    '', '## 审计与限制','',
    f"独立复核 {v['independent_incoming_records']} 个原保存记录、{v['independent_seed_rows']} 条诊断种子概率、{v['independent_week_rows']} 条逐周集成贡献、{v['independent_evidence_cells']} 个证据单元和 {v['independent_summary_cells']} 个汇总。对全部 {v['future_data_and_head_poison_checks']} 个月底修改之后才成熟的标签／概率和之后的候选参数，当前诊断不变。5,754 个旧文件及全部原预测、指标文件保持不变。",
    '', '完整验证窗口是重复使用的历史，不能作为旧参数新一次独立验证。通过后才成熟的子集虽然在原接受时不可见，但本轮设计前这些历史也已经查看；没有新增盲测。子集通过不保证之后有效，子集不足也不能证明参数已经失效。本轮没有生成基于重验证的新路由，未计算其收益或新预测准确率，也未修改研究基线。',
    '', '原 R25 近期 73/125、58.4% 属于旧训练预算协议；当前自然 20 遍年度 R25 为 71/125、56.8%；R39 季度验证 R23 的 73/125、58.4% 分开保留。R42 近期 R19／R23／R25 为 69／72／71 个正确周，原结果未覆盖。',
    '', '![新旧验证证据](evidence_freshness.png)','', '![质量清空对照](quality_clear_evidence.png)','',
    '[全部诊断单元](diagnostic_cells.csv) · [逐周证据](validation_week_diagnostics.csv) · [种子概率与参数来源](validation_seed_diagnostics.csv) · [分期汇总](diagnostic_summary.csv) · [原保存记录](subjects.csv) · [独立复核](verification.json)']
    report=OUT/'第四十三轮诊断报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    artifacts=[report]+[OUT/f'{n}.{e}' for n in ['evidence_freshness','quality_clear_evidence'] for e in ['png','svg']]
    (OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts}),ensure_ascii=False,indent=2),encoding='utf-8')
    print('R43 Chinese report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

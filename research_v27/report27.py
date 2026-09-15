"""Report-only process using the existing Python314 plotting installation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
METHODS=['learned_market','learned_vol_interaction','learned_order_extension','learned_order_offset','native_mse','training_frequency']
LABELS=dict(zip(METHODS,['加性 R18','波动交互 R19','联合顺序项 R23','固定原模型＋顺序项 R25','原 MSE','训练上涨频率']))
HISTORIES=['full','rolling5','rolling3'];HL={'full':'全历史','rolling5':'最近 5 年','rolling3':'最近 3 年'}
WL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def check():
    prep=read(OUT/'preparation_manifest.json');assert sha(ROOT/'protocol.json')==prep['protocol_sha256']
    for k in ['source_sha256','input_sha256']:
        for n,d in prep[k].items():assert sha(PROJECT/n)==d,n
    for phase in ['training','scoring','evaluation']:
        for n,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d,n
    assert read(OUT/'verification.json')['status']=='PASS';return prep
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def num(x):return '—' if pd.isna(x) else f'{x:.6f}'
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');m=csv('ensemble_metrics');yr=csv('yearly_metrics');diag=csv('seed_fusion_diagnostics');avail=csv('calibration_availability');budget=csv('budgets');pairs=read(OUT/'primary_comparisons.json');fit=read(OUT/'training_manifest.json');v=read(OUT/'verification.json');sig=[r for r in pairs if r['holm_adjusted_p']<.05]
    parts=['# 第二十七轮：固定更新预算的训练窗口比较','',f'生成时间：{now()}。36 个新神经网络、144 个分类解及全部评分独立复核通过。复用第 26 轮全历史对照，旧证据保持冻结。','',
        '本轮比较全历史、最近 5 年、最近 3 年的训练记忆。三个模型配方 R19/R23/R25 保留，同时报告加性 R18、原 MSE 和各自训练窗口的上涨频率基准。所有模型仍在同一 272 个周信号上评分；这是已查看过的历史研究，不是新盲测。','',
        '## 训练预算与口径','',
        '每个预测年以前年年底为截止日。短窗口按信号日期截取最近 3 或 5 个完整日历年，并要求全部标签已成熟。每轮名义训练呈现与全历史相同数量的样本、使用相同 batch 边界和更新次数；短窗口内随机打乱后重复抽取，最后一遍按剩余数量截断。每条样本在同一名义轮被抽取次数相差最多一次。','',
        '**因此本轮是固定更新预算的窗口比较。短窗口的有效遍历次数更多，并非每个保留样本恰好训练 20 次。** 结果不能直接外推到自然 20 遍的短窗口训练。所有标签尺度、特征截尾/标准化、监督方向、交互投影和逻辑回归都仅由该窗口的唯一成熟样本拟合；分类拟合不会再次使用神经训练的重复抽样权重。','',
        table(['预测年','窗口','唯一训练样本','每种子更新次数','等效保留样本遍历'],[[int(r.cutoff[:4])+1,HL[r.history],r.unique_rows,r.steps_per_seed,f'{r.equivalent_retained_passes:.2f}'] for r in budget.itertuples()]),'',
        '窗口限制针对带标签的训练信号，125 日输入上下文可以早于窗口起始日；所有输入都早于其对应信号日期。原有 0.011 点 OHLC 舍入规则、无效样本排除和最终标签成熟边界不变。','',
        '## 分时期结果','']
    for w in cfg['windows']:
        g=m[m.window.eq(w['name'])].set_index(['history','method']);parts += [f"### {WL[w['name']]}（{w['n']} 周）",'',table(['窗口','方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[HL[h],LABELS[method],f"{int(g.loc[(h,method),'correct_directions'])}/{int(g.loc[(h,method),'n'])}",f"{g.loc[(h,method),'accuracy']:.2%}",f"{g.loc[(h,method),'balanced_accuracy']:.2%}",num(g.loc[(h,method),'auroc']),num(g.loc[(h,method),'brier']),num(g.loc[(h,method),'log_loss'])] for h in HISTORIES for method in METHODS]),'']
        for h in HISTORIES[1:]:
            r=g.loc[(h,'learned_vol_interaction')];base=g.loc[('full','learned_vol_interaction')];parts += [f"{HL[h]}的 R19 相对全历史：正确方向变化 {int(r.correct_directions-base.correct_directions):+d} 周，Brier 变化 {r.brier-base.brier:+.6f}。",'']
    parts += [f'32 项预设配对比较统一 Holm 校正，校正后 p<0.05 的比较有 {len(sig)} 项。完整比较如下；负损失差值表示候选更好。','',table(['时期','候选窗口/方法','参照窗口/方法','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WL[r['window']],HL[r['history']]+'/'+LABELS[r['candidate']],HL[r['reference_history']]+'/'+LABELS[r['reference']],r['metric'],num(r['difference']),f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
        '每段独立使用 10,000 次循环移动区块重采样，每块 8 个有效周观察、固定种子 20260910。区间为边际区间。该校正只覆盖本轮比较，不能消除多轮查看历史或拟合本身带来的不确定性。合并 272 周只作补充描述。','',
        '## 各年度：原波动交互 R19','',table(['信号年','周数']+[HL[h] for h in HISTORIES],[[year,int(g.n.iloc[0])]+[f"{g.set_index('history').loc[h,'accuracy']:.2%}" for h in HISTORIES] for year,g in yr[yr.method.eq('learned_vol_interaction')].groupby('year')]),'',
        '全部 18 个窗口/方法组合的年度结果保存在 yearly_metrics.csv，种子与年度种子结果保存在 seed_metrics.csv、seed_yearly_metrics.csv；方向改变的净贡献见 direction_changes.csv。','',
        '## 种子融合诊断','',
        '主结果始终使用三个概率的等权均值、严格 p>0.5。下列诊断不改变融合规则，不选择最佳种子，也不把多数票另列为获胜候选。种子的概率幅度会影响均值方向；“与多数方向相反周数”用来定位这些分歧。','',
        table(['时期','窗口','方法','三个种子准确率范围','融合准确率','种子方向有分歧','均值与多数方向相反','Brier 方差减少量'],[[WL[r.window],HL[r.history],LABELS[r.method],f'{r.seed_accuracy_min:.2%}–{r.seed_accuracy_max:.2%}',f'{r.ensemble_accuracy:.2%}',r.disagreeing_weeks,r.opposes_majority_weeks,num(r.variance_reduction)] for r in diag[diag.window.ne('pooled_2021_2026')&diag.method.eq('learned_vol_interaction')].itertuples()]),'',
        '逐周核对三个概率、方向分歧及均值。对平方概率误差有恒等式：平均单种子 Brier − 平均概率后的 Brier = 种子概率的平均方差。它说明平均概率会降低相对于平均单种子的 Brier，但方向准确率没有相同保证。这也不保证融合优于频率基准。完整四个学习方法的诊断均已保存。','',
        '## 未来校准所需的过去验证数据','',
        '本轮仅审计过去预测的可用日期，没有拟合校准器或融合权重。每个外层预测年检查之前三年的已成熟周预测，要求每条预测由其信号之前的年度模型产生。全历史可衔接 R25/R26 的旧预测；新短窗口只能使用本轮自身更早年度的预测，不能借用全历史模型的概率代替。','',
        table(['预测年','窗口','过去可用周数','覆盖信号年数','至少三个信号年'],[[r.prediction_year,HL[r.history],r.available_weeks,r.available_signal_years,'是' if r.three_signal_years_available else '否'] for r in avail.itertuples()]),'',
        '可用预测受旧评分边界限制，不能自动视作对应三年全部可产生的信号。若下一轮要求所有窗口在每个预测年都具有同等校准历史，还需先补足短窗口的较早年度模型与过去预测。','',
        '## 核验与后续判断','',
        f"实际新神经训练 {fit['elapsed_seconds']/60:.2f} 分钟，共 {fit['optimizer_steps']:,} 次更新、{fit['training_presentations']:,} 次样本呈现。36 个检查点、144 个分类解由独立回放和另一套求解器核验；预测最大差异 {v['maximum_forecast_gap']:.3g}。保留 {v['old_files_preserved']:,} 份既有文件。协议 SHA-256：`{prep['protocol_sha256']}`。",'',
        '本轮只改变训练记忆及其必然带来的重复抽样，不加入新的状态选择器、超参数搜索或结果驱动的种子权重。先判断两个短窗口的改善是否跨时期、跨年度、跨种子出现，再考虑自然遍历预算、校准或状态条件化的独立验证。全部候选继续保留，不自动替换原模型。此前 2018–2020 年 R19 的 58.16% 和 2015–2017 年 R25 的 51.67% 记录保持不变。','',
        '预测指标不能直接推断交易成本后的盈利。真正前向验证仍需在目标结果发生之前带时间戳保存预测。','',
        '![训练窗口比较](training_memory.png)','',
        '核心审计文件：membership.csv、budgets.csv、models.json、heads.json、primary_comparisons.json、verification.json、delivery_manifest.json。']
    report=OUT/'第二十七轮测试报告.md';report.write_text('\n'.join(parts),encoding='utf-8')
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors=['#888888','#177e89','#b6622f']
    for row,method in enumerate(['learned_vol_interaction','learned_order_offset']):
        for h,color in zip(HISTORIES,colors):
            g=yr[yr.history.eq(h)&yr.method.eq(method)].sort_values('year');axs[row,0].plot(g.year,g.accuracy*100,'o-',label=HL[h],color=color);axs[row,1].plot(g.year,g.brier,'o-',label=HL[h],color=color)
        axs[row,0].set_title(LABELS[method]+'：方向准确率');axs[row,1].set_title(LABELS[method]+'：Brier');axs[row,0].axhline(50,ls='--',color='#ccc');axs[row,1].axhline(.25,ls='--',color='#ccc');axs[row,0].set_ylabel('%');axs[row,0].legend(ncol=3)
    for ax in axs.flat:ax.set_xticks(range(2021,2027));ax.grid(alpha=.2)
    fig.suptitle('同等更新预算下的训练窗口比较；2026 年仅截至 8 月')
    fig.savefig(OUT/'training_memory.png',dpi=170);fig.savefig(OUT/'training_memory.svg');plt.close(fig)
    facts=OUT/'report_facts.json';save(facts,dict(window_metrics=m.to_dict('records'),significant_comparisons=sig,independent_holdout=False,automatic_promotion=False,matched_updates=True,calibration_fits=0))
    save(OUT/'report_manifest.json',dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],source_sha256=prep['source_sha256'],input_sha256=prep['input_sha256'],executable=sys.executable,python=sys.version,artifacts={str(p.relative_to(ROOT)):sha(p) for p in [report,OUT/'training_memory.png',OUT/'training_memory.svg',facts]}))
    print('Report and figures complete.',flush=True)
if __name__=='__main__':main()

"""Read-only report and figures from the frozen feedback-policy experiment."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
HL={'rolling5_matched':'年度／旧重复量','rolling5_annual20':'年度／自然 20 遍','rolling5_quarterly20':'季度／自然 20 遍','state90':'市场状态触发＋年度','error16':'成熟预测误差触发＋年度'}
LABELS={'learned_market':'加性 R18','learned_vol_interaction':'波动交互 R19','learned_order_extension':'联合顺序项 R23','learned_order_offset':'固定原模型＋顺序项 R25','native_mse':'原 MSE','training_frequency':'训练上涨频率'}
WL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'}
RL={'annual':'年末例行','warmup':'不足 16 周','persistent_underperformance':'两段均落后','no_persistent_underperformance':'未持续落后'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def num(x):return '—' if pd.isna(x) else f'{x:.6f}'
def check():
    prep=read(OUT/'preparation_manifest.json');assert sha(ROOT/'protocol.json')==prep['protocol_sha256']
    for key in ['source_sha256','input_sha256']:
        for name,d in prep[key].items():assert sha(PROJECT/name)==d,name
    for phase in ['extension','scoring','evaluation']:
        for name,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/name)==d,name
    assert read(OUT/'verification.json')['status']=='PASS';return prep
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');m=csv('ensemble_metrics');yr=csv('yearly_metrics');ev=csv('events');op=ev[~ev.mandatory];b=csv('budgets');pairs=read(OUT/'primary_comparisons.json');sig=[r for r in pairs if r['holm_adjusted_p']<.05];v=read(OUT/'verification.json');diag=csv('next_quarter_diagnostics')
    parts=['# 第三十轮：成熟预测误差触发模型更新','',
        f"预先固定的规则在 17 次非年末季度检查中触发 {int(op.refit.sum())} 次更新；其中 {int((~op.ready).sum())} 次检查因当前模型不足 16 个成熟周预测而保持原模型。另有 6 次固定年末更新。计算和因果顺序复核全部通过。",'',
        '本轮复用已冻结的 5 年窗口、自然 20 遍神经网络和分类头，实际新增网络训练、分类拟合均为 0。测试的是何时使用新模型，模型结构、训练量和融合方式保持原配方。全部 272 个历史周此前已被查看，本轮不是新盲测。','',
        '## 固定的失效判断','',
        '统一用波动交互 R19 的三种子平均概率监测。每个非年末季末，取当前模型实际使用期间、到当天全部标签已成熟的最近 16 个周预测。分成相邻且不重叠的前 8 周和后 8 周，分别计算“模型 Brier − 当时训练上涨频率基准 Brier”的平均值。只有两段都严格大于 0 才更新。', '',
        '这意味着持续落后于频率基准，不要求误差单调增加。两个区间是同一检查内不重叠的 8 周，不能解释成两次独立检验。零差值或只有一段落后都不触发；不足 16 周则保持原模型。每次年末或触发更新后，只积累新模型实际发出的预测，旧模型的误差不纳入新模型判断。','',
        '标签成熟沿用执行收益及辅助标签均已完成的 joint_completed 口径。模型训练截止日严格早于信号日；季末当天的信号仍使用原模型，新决定仅影响之后的信号。没有用训练集残差补历史记录，也没有把新模型回算过去样本当作新模型已经发出的预测。', '',
        '所有 R18/R19/R23/R25、原 MSE 和频率基准共享 R19 触发日历，因此结果是在检验一个共同更新政策。该监测指标未必适合其他预测头，不能事后换成当轮表现最好的监测方法。16 周、两个 8 周和零阈值均预先固定，本轮没有搜索其他参数。','',
        '## 预测覆盖与顺序','',
        f"在模拟触发前，先按日历固定 {prep['coverage_model_weeks']} 个模型周的通用覆盖；复用旧预测，补算其中 {prep['missing_model_weeks']} 个模型周、{v['extension_replays']} 个已有网络的推理。覆盖包含某些政策不会选到的模型，但触发函数只接收已发出、当前模型、已成熟的记录。",'',
        '随后逐个季末回放，保存触发日历、已发出预测账本、每次检查使用的周与区间，并在汇总比较前冻结。最后才计算下一季度的新旧模型对照，用于判断触发是否有用；这些未来结果不参与当前或之后的触发计算。','',
        '## 触发记录','',table(['检查日','检查前模型','成熟周数','前 8 周超额 Brier','后 8 周超额 Brier','原因','检查后模型'],[[r.cutoff,r.previous_model_cutoff if isinstance(r.previous_model_cutoff,str) else '—',r.available_mature_weeks,num(r.older8_excess),num(r.recent8_excess),RL[r.reason],r.selected_model_cutoff] for r in ev.itertuples()]),'',
        '逐次使用的样本和区间见 monitor_membership.csv；已发出概率与当时频率基准见 published_ledger.csv。年末行只作例行重置，不计算 16 周触发统计。','',
        '## 逐期运行时的训练预算','',table(['政策','更新日期数','三种子网络训练数','优化器更新次数'],[[HL[r.history],r.refit_dates,r.neural_fits_if_run_online,r.optimizer_steps_if_run_online] for r in b.itertuples()]),'',
        '此表是如果按历史逐期运行该政策所需的训练量；此次研究复用权重，实际新增网络/分类训练为 0，补充推理的计算也包括未被最终政策选中的路径。','']
    for w in cfg['windows']:
        g=m[m.window.eq(w['name'])].set_index(['history','method']);parts += [f"## {WL[w['name']]}（{w['n']} 周）",'',table(['政策','方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[HL[h],LABELS[method],f"{int(g.loc[(h,method),'correct_directions'])}/{w['n']}",f"{g.loc[(h,method),'accuracy']:.2%}",f"{g.loc[(h,method),'balanced_accuracy']:.2%}",num(g.loc[(h,method),'auroc']),num(g.loc[(h,method),'brier']),num(g.loc[(h,method),'log_loss'])] for h in HL for method in LABELS]),'']
    parts += ['## 40 项预设配对比较','',f"统一 Holm 校正后，p<0.05 的改善 {sum(r['difference']<0 for r in sig)} 项、恶化 {sum(r['difference']>0 for r in sig)} 项。负损失差表示误差触发政策更好。",'',table(['时期','方法','参照政策/方法','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WL[r['window']],LABELS[r['candidate']],HL[r['reference_history']]+'/'+LABELS[r['reference']],r['metric'],num(r['difference']),f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
        '两段各用 10,000 次循环移动区块重采样，每块 8 个有效周观察、种子 20260910；p 为中心化双侧值。区间为边际 95% 区间。该校正只覆盖本轮，不能消除多轮历史研究中的选择影响。合并 272 周仅作补充描述。','']
    for method in ['learned_vol_interaction','learned_order_extension','learned_order_offset']:
        parts += [f'## 逐年：{LABELS[method]}','',table(['信号年','周数']+[HL[h] for h in HL],[[year,int(g.n.iloc[0])]+[f"{g.set_index('history').loc[h,'accuracy']:.2%}" for h in HL] for year,g in yr[yr.method.eq(method)].groupby('year')]),'']
    parts += ['## 诊断：触发后换模型是否有帮助','',
        '对每个非年末检查，事后比较下一个季度内继续用原模型与改用当季新模型的 R19 预测。下表是描述性诊断，没有添加候选规则或额外显著性筛选，也没有反向修改触发记录。暖机不足的行仍列出，便于看到检查机制的覆盖限制。', '',
        table(['检查日','判定','下一段周数','继续原模型正确数','改用新模型正确数','新减旧 Brier'],[[r.cutoff,RL[r.reason],r.next_n,r.incumbent_correct,r.fresh_correct,num(r.fresh_minus_incumbent_brier)] for r in diag.itertuples()]),'']
    sd=csv('seed_fusion_diagnostics');parts += ['## 种子稳定性','',table(['时期','政策','方法','三个种子准确率范围','融合准确率'],[[WL[r.window],HL[r.history],LABELS[r.method],f'{r.seed_accuracy_min:.2%}–{r.seed_accuracy_max:.2%}',f'{r.ensemble_accuracy:.2%}'] for r in sd[sd.window.ne('pooled_2021_2026')&sd.history.isin(['rolling5_annual20','error16'])&sd.method.isin(['learned_vol_interaction','learned_order_offset'])].itertuples()]),'',
        '全部方法、年度和种子的结果见 yearly_metrics.csv、seed_metrics.csv 和 seed_yearly_metrics.csv。未选择最好种子，也未修改概率均值融合。','',
        '## 复核与限制','',
        f"69 个旧检查点的来源与状态已核验；新增推理复现 {v['extension_replays']} 个已有网络，新特征代数最大差 {v['maximum_feature_algebra_gap']:.3g}、概率最大差 {v['maximum_probability_gap']:.3g}。独立标量回放逐条核对触发日历、272 条已发出周预测和监测区间。",'',
        '10 个合成检查覆盖暖机、零阈值、只有一段落后、标签成熟边界、重置及最新 16 周。17 次非年末检查分别扰动尚未成熟的标签、未来预测概率和其他模型的误差，确认原决定不变。所有指标、40 项比较、下一季度诊断和 4,600 个旧证据文件已核验。', '',
        '持续落后于一个简单基准可能来自噪声、长期缺少预测信息或模型失效；本轮规则不能自动区分这些原因。更换模型也可能无效。16 周的积累会延迟反应，且一次实际更新会改变未来的监测样本组成。复核通过不等于预测优势或经济收益得到确认。本轮不自动替换基线，原有近期改善记录继续保留。','',
        '![误差触发与下一季度效果](forecast_error_trigger.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    path=OUT/'第三十轮测试报告.md';path.write_text('\n'.join(parts),encoding='utf-8');files=[path]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,1,figsize=(15,10),layout='constrained');x=np.arange(len(op));ax=axs[0]
    ax.bar(x-.18,op.older8_excess,.36,label='前 8 周',color='#4c78a8');ax.bar(x+.18,op.recent8_excess,.36,label='后 8 周',color='#f2a541');ax.axhline(0,color='#555555',lw=1);ax.set_xticks(x,op.cutoff,rotation=45,ha='right',fontsize=9);ax.set_ylabel('模型 Brier − 当时频率基准 Brier');ax.set_title('两个区间都大于零才更新；× 表示不足 16 个成熟周，红色点标示触发');ax.legend(frameon=False,loc='upper right')
    for j,r in enumerate(op.itertuples()):
        if not r.ready:ax.scatter(j,0,marker='x',color='#666666',zorder=5)
        elif r.refit:ax.scatter(j,max(r.older8_excess,r.recent8_excess)+.004,color='#c44e52',s=25,zorder=5)
    ax=axs[1];years=sorted(yr.year.unique());policies=['rolling5_annual20','rolling5_quarterly20','state90','error16'];colors=['#4c78a8','#72b7b2','#999999','#c44e52'];width=.20
    for j,h in enumerate(policies):
        g=yr[yr.history.eq(h)&yr.method.eq('learned_vol_interaction')].set_index('year').loc[years];pos=np.arange(6)+(j-1.5)*width;ax.bar(pos,g.accuracy,width,color=colors[j],label=HL[h]);[ax.text(xx,val+.008,f'{val:.0%}',ha='center',fontsize=8) for xx,val in zip(pos,g.accuracy)]
    ax.set_xticks(np.arange(6),[str(y) if y<2026 else '2026 至 8 月' for y in years]);ax.set_ylim(0,.88);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.set_title('波动交互 R19：同一批周信号的逐年准确率');ax.legend(ncol=2,loc='upper center',frameon=False)
    for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('5 年窗口、自然 20 遍：成熟预测误差能否帮助选择更新时机？',fontsize=17);fig.supxlabel('历史研究；三种子等权融合。只有已发出、当前模型、已成熟的预测可用于触发。',fontsize=10)
    for ext in ['png','svg']:
        p=OUT/f'forecast_error_trigger.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);save(OUT/'report_manifest.json',dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('Report and figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

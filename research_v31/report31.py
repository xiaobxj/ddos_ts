"""Standalone read-only Chinese report and scientific figure from frozen results."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
HL={'rolling5_matched':'年度／旧重复量','rolling5_annual20':'年度／自然 20 遍','rolling5_quarterly20':'季度／自然 20 遍','state90':'市场状态触发','error16':'成熟误差触发','shadow_trial':'新旧试运行后切换','annual_quarter_half':'年度与季度各半'}
ML={'learned_market':'加性 R18','learned_vol_interaction':'波动交互 R19','learned_order_extension':'联合顺序项 R23','learned_order_offset':'固定原模型＋顺序项 R25','native_mse':'原 MSE','training_frequency':'训练上涨频率'}
WL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'}
RL={'annual':'年末重置','same_model':'两者相同','warmup':'共同成熟周不足','promote':'采用试运行模型','hold':'保持原模型'}
PRIMARY=list(ML)[1:4]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def num(x):return '—' if pd.isna(x) else f'{x:.6f}'
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def check():
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for name,d in p[key].items():assert sha(PROJECT/name)==d
    for phase in ['scoring','evaluation']:
        for name,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/name)==d
    assert read(OUT/'verification.json')['status']=='PASS';return p
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');m=csv('ensemble_metrics');yr=csv('yearly_metrics');ev=csv('events');diag=csv('next_quarter_diagnostics');budgets=csv('budgets');pairs=read(OUT/'primary_comparisons.json');v=read(OUT/'verification.json');sig=[r for r in pairs if r['holm_adjusted_p']<.05]
    parts=['# 第三十一轮：先比较新旧模型，再决定是否替换','',
      f"固定试运行规则在 11 次不同模型的非年末比较中采用了 {int(ev.promote.sum())} 次候选模型，另有 6 次强制年末重置。其他 6 次非年末检查发生在 3 月末，新旧模型相同。本轮同时检验年度模型与季度模型各占一半的固定融合。",'',
      '计算与因果时序复核通过。复用 69 个旧神经网络及 276 个分类头，新增训练、分类拟合和神经推理均为 0。所有 272 周历史结果此前已被查看，本轮是历史研究，不是新盲测。文中的“发出预测”指逐期回放中预先指定模型的预测，并非声称存在当年真实运行日志。','',
      '## 预先固定的两项政策','',
      '试运行政策：每个季度内，旧模型和上个季末训练的新模型并行预测。到本季末，仅取本轮试运行期间双方共同发出、所有标签已成熟的周预测，要求至少 8 周；新模型的平均 Brier 必须严格更低，方向正确周数必须严格更多，才在下一季度接替旧模型。两项指标任一打平都保持原模型。所有方法共享 R19 三种子等权概率的判断，不事后改用表现最好的监测头。','',
      '例如，6 月末评估的是 3 月模型在第二季度的预测；通过后第三季度采用 3 月模型。刚在 6 月末训练的模型只进入第三季度试运行，不能直接视为已经通过评估。一次试运行只比较这个季度的固定新旧模型对，其他模型和前轮试运行误差不参与。','',
      '年末始终使用新年度模型重置，下一季度的试运行模型与之相同。因此 3 月末没有不同模型可采用；9 月启动的试运行也不能绕过年末强制重置。沿用这个年度规则是与此前方案保持可比的选择，不是本轮发现的最优规则。标签成熟统一要求 joint_completed 不晚于检查日，决策只影响之后的信号，季末当天的信号仍归前一个季度。','',
      '固定融合政策：每个信号日取最近的年度模型与最近的季度模型，二者训练截止日都必须严格早于信号日，各占 1/2。先在同一种子内平均概率，再等权平均三个种子；原 MSE 只平均原始收益预测并用零阈值，不构造概率；频率基准按相同权重平均。没有拟合融合权重，也没有根据未来结果切换成分。','',
      '8 周是一个季度内的最低共同证据量，使用该季度全部成熟周，未搜索最优周数。双指标门槛针对上一轮概率误差和方向准确率不一致的问题提出；它是研究假设，不是显著性检验，也不保证新模型持续更好。','',
      '## 证据与冻结顺序','',
      '先冻结协议、全部源码、输入和 4,696 个旧证据文件，再核对 671 个模型周的既有预测覆盖及 16 项合成边界检查。随后做一次逐期回放，冻结新旧预测账本、试运行成员和两种政策的模型来源，再计算汇总指标及下一季度诊断。旧预测、模型和历史报告没有改写。','',
      'paired_ledger.csv 记录每周新旧模型及概率；trial_membership.csv 记录实际用于每次检查的成熟周；shadow_routing.csv 记录采用的模型；blend_routing.csv 记录固定融合的两个截止日及权重。融合预测表的 cutoff 仅表示较新的成分训练截止日，不代表单一模型。','',
      '## 试运行决策','',table(['检查日','原模型','已试运行模型','共同成熟周','新减旧 Brier','正确周数增量','决定','下一季度采用'],[[r.cutoff,r.incumbent_cutoff if isinstance(r.incumbent_cutoff,str) else '—',r.evaluated_challenger_cutoff if isinstance(r.evaluated_challenger_cutoff,str) else '—',r.paired_mature_weeks,num(r.challenger_minus_incumbent_brier),r.challenger_correct_gain,RL[r.reason],r.selected_model_cutoff] for r in ev.itertuples()]),'',
      '年末行只记录重置，未应用比较门槛；其差值不用于晋升。所有新候选均在检查后开始下一季度试运行。','',
      '## 逐期运行时所需训练量','',table(['政策','训练日期数','三种子网络数','分类头拟合数','优化器更新次数'],[[HL[r.history],r.refit_dates,r.neural_fits_if_run_online,r.head_fits_if_run_online,r.optimizer_steps_if_run_online] for r in budgets.itertuples()]),'',
      '两项新政策都需要完整的季度候选模型，因此与每季度训练的预算相同，即 23 个截止日、69 个网络、276 个分类头及 13,800 次优化器更新。试运行中拒绝某个候选不会退回已发生的训练成本。表中是逐期运行的假设预算，本轮实际新增训练为 0。','']
    for w in cfg['windows']:
        g=m[m.window.eq(w['name'])].set_index(['history','method']);parts += [f"## {WL[w['name']]}（{w['n']} 周）",'',table(['政策','方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[HL[h],ML[method],f"{int(g.loc[(h,method),'correct_directions'])}/{w['n']}",f"{g.loc[(h,method),'accuracy']:.2%}",f"{g.loc[(h,method),'balanced_accuracy']:.2%}",num(g.loc[(h,method),'auroc']),num(g.loc[(h,method),'brier']),num(g.loc[(h,method),'log_loss'])] for h in HL for method in ML]),'']
    parts += ['## 68 项预设比较','',f"两段时期、两个新政策相对年度与季度方案、各自简单基准，以及两个新政策之间的预设比较共 68 项。统一 Holm 校正后，显著改善 {sum(r['difference']<0 for r in sig)} 项、显著恶化 {sum(r['difference']>0 for r in sig)} 项。",'',table(['时期','政策','方法','参照政策/方法','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WL[r['window']],HL[r['history']],ML[r['candidate']],HL[r['reference_history']]+'/'+ML[r['reference']],r['metric'],num(r['difference']),f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
      '负损失差表示候选更好。每段用 10,000 次循环移动区块重采样、每块 8 个有效周观察、种子 20260910；p 为中心化双侧值，区间为边际 95% 区间。本轮多重校正不能消除跨轮历史研究中的选择影响。合并 272 周只作补充描述，状态触发、误差触发和旧训练量年度方案保留为描述性对照。','']
    for method in PRIMARY:
        parts += [f'## 逐年：{ML[method]}','',table(['年份','周数']+[HL[h] for h in HL],[[year,int(g.n.iloc[0])]+[f"{g.set_index('history').loc[h,'accuracy']:.2%}" for h in HL] for year,g in yr[yr.method.eq(method)].groupby('year')]),'']
    parts += ['## 事后诊断：已比较的候选在下一季度如何','',
      '下表仅比较 11 次不同模型的非年末检查之后，下一个季度候选与原模型的表现，采用与未采用均列出。它评估试运行效果能否延续，未用于改变任何当期或后续决策，也未据此增加政策或选择最佳季度。','',table(['检查日','当时是否采用','下一段周数','原模型正确数','候选正确数','候选减原模型 Brier'],[[r.cutoff,'是' if r.promoted else '否',r.next_n,r.incumbent_correct,r.challenger_correct,num(r.challenger_minus_incumbent_brier)] for r in diag.itertuples()]),'']
    sd=csv('seed_fusion_diagnostics');parts += ['## 种子稳定性','',table(['时期','政策','方法','三个种子准确率范围','融合准确率'],[[WL[r.window],HL[r.history],ML[r.method],f'{r.seed_accuracy_min:.2%}–{r.seed_accuracy_max:.2%}',f'{r.ensemble_accuracy:.2%}'] for r in sd[sd.window.ne('pooled_2021_2026')&sd.history.isin(['rolling5_annual20','shadow_trial','annual_quarter_half'])&sd.method.isin(PRIMARY)].itertuples()]),'',
      '全部方法、年度、种子、概率分箱和方向变化见对应 CSV。没有选择最好种子或按结果修改融合规则。','',
      '## 复核与限制','',
      f"69 个旧检查点的模型与优化器状态、来源和训练尺度均已核验。16 项合成检查、17 次未来信息/未发出模型/旧试运行记录扰动检查、272 条配对预测的独立标量回放均通过。固定融合独立代数最大差 {v['maximum_blend_algebra_gap']:.3g}；所有指标、68 项统计比较及 11 项下一季度诊断通过复核。",'',
      '试运行可以限制未验证的新模型立即接替，但一个季度的证据仍少，且通过评估时模型已经更旧。共同门槛可能错过方向持平而概率更好的候选，也未必适合所有预测头。年末强制重置及共同标签成熟要求会进一步限制响应时机。固定融合则可能稀释较好成分或保留较差成分，不能预先认定其更稳健。','',
      '全部结果来自已查看历史；本轮门槛还受到上一轮失败现象的启发。计算可复现、顺序因果并不等于得到新的独立验证。本轮不自动替换研究基线，不据此声称交易收益，之前交互方法和 R25 近期 58.4% 的记录继续保留。','',
      '![三种交互方法的逐年比较](shadow_and_blend_yearly.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十一轮测试报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files=[p]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,3,figsize=(17,6),sharey=True,layout='constrained');policies=['rolling5_annual20','rolling5_quarterly20','shadow_trial','annual_quarter_half'];colors=['#4c78a8','#999999','#c44e52','#2b927b'];styles=['-','--','-','-'];markers=['o','s','^','D'];years=sorted(yr.year.unique())
    for ax,method in zip(axs,PRIMARY):
        for h,color,style,marker in zip(policies,colors,styles,markers):
            g=yr[yr.history.eq(h)&yr.method.eq(method)].set_index('year').loc[years];ax.plot(np.arange(6),g.accuracy,color=color,ls=style,marker=marker,lw=2,label=HL[h])
        ax.set_title(ML[method],fontsize=13);ax.set_xticks(np.arange(6),['2021','2022','2023','2024','2025','2026*'],rotation=35);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=11);fig.suptitle('5 年窗口、自然 20 遍：延迟切换与固定融合的逐年表现',fontsize=17);fig.supxlabel('同一批历史周；三种子等权。*2026 年截至 8 月（30 周）；其余年份 47–50 周。',fontsize=10)
    for ext in ['png','svg']:
        p=OUT/f'shadow_and_blend_yearly.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('Report and figure generated; visual review pending.',flush=True)
if __name__=='__main__':main()

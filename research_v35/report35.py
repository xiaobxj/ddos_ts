"""Standalone research report and plots for controlled training-member interventions."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
HL={'rolling5_matched':'年度／旧重复量','rolling5_annual20':'年度基线','rolling5_quarterly20':'整套季度','state90':'市场状态触发','error16':'成熟误差触发','shadow_trial':'新旧试运行切换','annual_quarter_half':'年度与季度概率各半','annual_net_quarter_head':'R32 季度分类端','fixed_transform_step25':'固定变换／25% 更新','fixed_transform_step50':'固定变换／50% 更新','fixed_transform_step100':'加入＋移出／完整滚动','fixed_transform_add_only':'只加入新成熟成员','fixed_transform_remove_only':'只移出旧成员'}
ML={'learned_market':'加性 R18','learned_vol_interaction':'波动交互 R19','learned_order_extension':'联合顺序项 R23','learned_order_offset':'固定 R19＋顺序项 R25','native_mse':'原 MSE','training_frequency':'训练上涨频率'}
WL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'}
ARMS={'annual':'rolling5_annual20','add':'fixed_transform_add_only','remove':'fixed_transform_remove_only','both':'fixed_transform_step100'}
AL={'annual':'年度','add':'只加入','remove':'只移出','both':'加入＋移出'};PRIMARY=list(ML)[1:4]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def num(x):return '—' if pd.isna(x) else f'{x:.6f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def check():
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for n,d in p[key].items():assert sha(PROJECT/n)==d
    for phase in ['fitting','scoring','evaluation','diagnosis']:
        for n,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    assert read(OUT/'verification.json')['status']=='PASS';return p
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');m=csv('ensemble_metrics');yr=csv('yearly_metrics');pairs=read(OUT/'primary_comparisons.json');v=read(OUT/'verification.json');summary=csv('effect_summary');members=csv('membership_summary');changes=csv('set_changes');flips=csv('direction_flips');sig=[r for r in pairs if r['holm_adjusted_p']<.05]
    parts=['# 第三十五轮：滚动窗口加入与移出的受控对照','',
      '本轮新增 102 组样本操作与季度种子组合的拟合，共 408 个分类头；复用年度 72 个头和上一轮完整滚动 204 个头。年度神经网络、全部缩放、交互方向和投影、原始特征缓存不变，无新神经训练或特征推理。训练前冻结四组成员及全部比较，全部新头冻结后才进行周预测。','',
      '## 四组训练成员','',
      '对每个季度，A 为当年年度训练成员，Q 为本季度完整五年滚动成员，N=Q\\A 为新增成员，O=A\\Q 为移出成员，R=A∩Q 为保留成员。每年重新定义 A。','',
      table(['组别','成员','作用'],[['年度','A','直接复用年度系数'],['只加入','A∪Q=A∪N','保留年度成员，再加入本季度新增成员'],['只移出','A∩Q=A\\O','移出超出当前窗口的旧成员，不加入新成员'],['加入＋移出','Q=R∪N','完整滚动，直接复用 R34 固定年度变换的 100% 系数更新']]),'',
      '所有组别按行身份去重，成员标签必须在季度截止时全部成熟。新增指相对年度集合新获得资格，可能包含日期早于本季度、但标签刚成熟的观察。只加入组在年内保留年度下界，可能超过五年；只移出组样本减少，且不引入年度截止后新成熟的标签，因此这两组是诊断性成员干预，不能统称为固定五年训练。','',
      '本轮只改变分类系数损失函数中的样本成员。年度神经网络、特征截尾和标准化、监督交互方向、残差投影及尺度仍保留原年度训练信息，包括被移出成员的信息。因此“只移出”不等于整套模型忘记这些数据，也不是整条流程的留出实验。','',
      'R18、R19、R23 沿用 lambda=0.01、只惩罚斜率的平均逻辑损失；R25 固定同组同季度的 R19，完整拟合顺序系数 gamma。本轮采用完整系数拟合，不叠加上一轮 25% 或 50% 插值。相同正则强度下，样本操作仍同时改变数量、类别构成、覆盖范围及平均损失中各样本的权重，无法单独区分数量与质量。','',
      '## 样本数量变化','',
      table(['季度截止','年度数','新增数','移出数','只加入数','只移出数','完整滚动数'],[[r.cutoff,r.annual_n,r.added_n,r.removed_n,r.annual_n+r.added_n,r.annual_n-r.removed_n,r.both_n] for r in changes.itertuples()]),'',
      '全部 92 个组别与截止日组合的最早日期、最晚日期、标签最晚成熟时间、上涨标签比例见 membership_summary.csv；逐个成员见 ablation_membership.csv。','',
      '## 预测与比较口径','',
      '同一批 272 个周信号，2021–2023 为 147 周，2024–2026 年 8 月为 125 周。预测目标仍为原定义的随后执行周收益方向，三个固定种子的概率等权平均，概率严格 >0.5 才判断上涨。各组第一季度与年度方案完全相同；原 MSE 全段沿用年度结果，训练频率对照则使用每组自己的成熟成员标签比例。11 个旧政策完整保留。','',
      '所有历史结果此前已被查看，本轮假设来自此前滚动训练与归因结果，不是新的盲测，也没有根据本轮表现选择新的训练窗口、更新强度或种子。','']
    for w in cfg['windows']:
        g=m[m.window.eq(w['name'])].set_index(['history','method']);parts += [f"## {WL[w['name']]}（{w['n']} 周）",'',table(['政策','方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[HL[h],ML[method],f"{int(g.loc[(h,method),'correct_directions'])}/{w['n']}",f"{g.loc[(h,method),'accuracy']:.2%}",f"{g.loc[(h,method),'balanced_accuracy']:.2%}",num(g.loc[(h,method),'auroc']),num(g.loc[(h,method),'brier']),num(g.loc[(h,method),'log_loss'])] for h in HL for method in ML]),'']
    for method in PRIMARY:
        g=yr[yr.method.eq(method)];parts += [f'## 逐年：{ML[method]}','',table(['年份','周数']+[AL[a] for a in ARMS],[[int(year),int(s.n.iloc[0])]+[f"{s.set_index('history').loc[h,'accuracy']:.2%}" for h in ARMS.values()] for year,s in g.groupby('year')]),'']
    parts += ['## 两种操作如何改变预测概率','',
      '记 p0、pA、pR、pB 分别为年度、只加入、只移出、同时操作的上涨概率。加入的条件影响为 pA−p0（尚未移出时）及 pB−pR（已经移出时）；移出的条件影响为 pR−p0（尚未加入时）及 pB−pA（已经加入时）。','',
      '对两种顺序取等权平均：加入贡献=[(pA−p0)+(pB−pR)]/2；移出贡献=[(pR−p0)+(pB−pA)]/2；两者之和严格等于 pB−p0。非加和差异为 pB−pA−pR+p0，反映两种操作的条件影响不同，其中包含重新拟合和概率转换的非线性；它不是模型内部的市场或顺序交互特征。','',
      '先按每个种子的实际概率计算，再平均三个种子的贡献，保持原概率融合定义。该归因是约定的操作顺序平均，不是唯一自然因果分配，因此两个条件路径也全部报告。符号按随后实际涨跌方向调整，负值表示概率向错误方向移动，单位为概率百分点，不是准确率百分点。','']
    for period in ['extension_2021_2023','recent_2024_2026','year_2026']:
        title=WL.get(period,'2026 年');g=summary[summary.period.eq(period)&summary.method.isin(PRIMARY)]
        parts += [f'### {title}：全部周及完整滚动的退步／改善周','',table(['方法','周类型','周数','加入平均贡献','移出平均贡献','总变化','非加和差异','加入最大不利周','移出最大不利周'],[[ML[r.method],{'all':'全部','regression':'对→错','recovery':'错→对'}[r.case],r.n,f'{100*r.mean_signed_add_effect:+.3f}',f'{100*r.mean_signed_remove_effect:+.3f}',f'{100*r.mean_signed_total_delta:+.3f}',f'{100*r.mean_signed_nonadditive:+.3f}',r.dominant_add,r.dominant_remove] for r in g[g['case'].isin(['all','regression','recovery'])].itertuples()]),'']
    g=summary[summary.period.eq('year_2026')&summary.method.isin(PRIMARY)&summary['case'].eq('regression')]
    parts += ['### 2026 年退步周：两个条件路径','',table(['方法','周数','先加入','移出后再加入','先移出','加入后再移出'],[[ML[r.method],r.n,f'{100*r.mean_signed_add_without_remove:+.3f}',f'{100*r.mean_signed_add_after_remove:+.3f}',f'{100*r.mean_signed_remove_without_add:+.3f}',f'{100*r.mean_signed_remove_after_add:+.3f}'] for r in g.itertuples()]),'',
      '“退步周”按完整滚动组相对年度组定义，使用相同集合比较加减影响。稳定正确和稳定错误、所有年度的同口径汇总亦保存在 effect_summary.csv。系数逐坐标对照与同样的操作分解见 coefficient_effects.csv；不能仅凭系数位移范数推断预测好坏。','',
      '## 2026 年各组相对年度的翻转','',table(['组别','方法','对→错','错→对'],[[AL[arm],ML[method],int((flips.arm.eq(arm)&flips.method.eq(method)&flips.year.eq(2026)&flips.arm_case.eq('regression')).sum()),int((flips.arm.eq(arm)&flips.method.eq(method)&flips.year.eq(2026)&flips.arm_case.eq('recovery')).sum())] for arm in ['add','remove','both'] for method in PRIMARY]),'',
      '## 60 项预设比较','',f"加入相对年度、移出相对年度、完整滚动相对只移出、完整滚动相对只加入、完整滚动相对年度，共五组关系；两个时期、三个主方法、方向误差和 Brier，共 60 项统一 Holm 校正。显著改善 {sum(r['difference']<0 for r in sig)} 项，显著恶化 {sum(r['difference']>0 for r in sig)} 项。",'',
      table(['时期','候选','方法','参照','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WL[r['window']],HL[r['history']],ML[r['candidate']],HL[r['reference_history']],r['metric'],num(r['difference']),f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
      '负损失差表示候选更好。沿用 10,000 次循环区块重采样，每块 8 个有效周观察，种子 20260910；中心化双侧 p，边际 95% 区间。校正只覆盖本轮，不能抵消跨轮选择。完整滚动相对年度的关系已在上一轮出现，不是新增独立证据。原 R34 的 60 项比较原样归档。','']
    sd=csv('seed_fusion_diagnostics');parts += ['## 种子稳定性','',table(['时期','组别','方法','三种子准确率范围','融合准确率'],[[WL[r.window],HL[r.history],ML[r.method],f'{r.seed_accuracy_min:.2%}–{r.seed_accuracy_max:.2%}',f'{r.ensemble_accuracy:.2%}'] for r in sd[sd.window.ne('pooled_2021_2026')&sd.history.isin(ARMS.values())&sd.method.isin(PRIMARY)].itertuples()]),'',
      '## 独立复核与限制','',
      f"408 个新分类头及 204 个复用完整滚动分类头通过独立 L-BFGS-B 或 Brent 检验。153 组拟合接口通过非训练行特征、市场描述、顺序描述和标签扰动检查，92 个样本集合独立重建。测试概率最大差 {v['maximum_probability_gap']:.3g}，固定年度坐标重建最大差 {v['maximum_feature_algebra_gap']:.3g}，两种操作分解最大代数差 {v['maximum_effect_identity_gap']:.3g}。",'',
      '3,264 组种子周、1,088 组融合周、6,375 个系数坐标的操作分解、全部指标和 60 项统计比较均复核。第一季度年度一致、各组频率来自自身成员、原 MSE 不生成虚假概率。5,063 个旧证据文件保持不变。','',
      '这项受控研究识别的是固定年度表征下、分类端加入与移出这些特定样本集合的影响。不能据此证明单个样本、特定市场状态或者一般意义上的滚动训练导致了某个市场结果。部分表征对分类训练样本仍属训练内表征，也不能声称全流程折外。旧 R25 近期 73/125、58.4% 保留，不自动替换研究基线。','',
      '![四组成员的逐年表现](member_ablation_yearly.png)','',
      '![2026 年完整滚动退步周的操作贡献](member_effects_2026.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十五轮测试报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files=[p]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,3,figsize=(18,6.4),sharey=True);fig.subplots_adjust(left=.05,right=.99,top=.83,bottom=.24,wspace=.13);colors=['#333333','#0077bb','#dd8844','#aa4466'];styles=['-','-','--',':'];markers=['o','s','^','D'];years=sorted(yr.year.unique())
    for ax,method in zip(axs,PRIMARY):
        for (arm,h),col,style,mk in zip(ARMS.items(),colors,styles,markers):
            g=yr[yr.history.eq(h)&yr.method.eq(method)].set_index('year').loc[years];ax.plot(np.arange(6),g.accuracy,color=col,ls=style,marker=mk,lw=2,label=AL[arm])
        ax.set_title(ML[method],fontsize=13);ax.set_xticks(np.arange(6),['2021','2022','2023','2024','2025','2026*'],rotation=30);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,frameon=False,fontsize=12)
    fig.suptitle('固定年度表征与变换：分别加入新成熟成员、移出旧成员',fontsize=18,y=.97);fig.text(.5,.025,'同一批已查看历史；完整系数重拟合；三种子概率等权。*2026 年截至 8 月，共 30 周。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'member_ablation_yearly.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    g=summary[summary.period.eq('year_2026')&summary['case'].eq('regression')].set_index('method');fig,ax=plt.subplots(figsize=(10.5,6.5));fig.subplots_adjust(left=.12,right=.97,top=.80,bottom=.25);x=np.arange(3);width=.32
    for shift,key,label,col in [(-width/2,'mean_signed_add_effect','加入贡献','#0077bb'),(width/2,'mean_signed_remove_effect','移出贡献','#dd8844')]:
        vals=np.array([g.loc[method,key]*100 for method in PRIMARY]);bars=ax.bar(x+shift,vals,width,label=label,color=col)
        for b,val in zip(bars,vals):ax.annotate(f'{val:+.2f}',(b.get_x()+b.get_width()/2,val),xytext=(0,5 if val>=0 else -14),textcoords='offset points',ha='center',fontsize=12)
    ax.axhline(0,color='#555555',lw=1);ax.set_xticks(x,[ML[method]+f"\n{int(g.loc[method,'n'])} 个退步周" for method in PRIMARY]);ax.set_ylabel('按实际方向调整的概率变化（百分点）');ax.margins(y=.25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False,fontsize=12)
    fig.suptitle('2026 年完整滚动退步周：加入与移出的平均贡献',fontsize=16,y=.97);fig.text(.5,.875,'两种操作顺序取平均；负值表示概率背离实际方向，非准确率变化。',ha='center',fontsize=10);fig.text(.5,.025,'相同退步周集合内比较；两项相加等于完整滚动相对年度的概率变化。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'member_effects_2026.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('Chinese report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

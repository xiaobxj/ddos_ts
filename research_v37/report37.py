from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
ANNUAL='rolling5_annual20';STATE='annual_state_offset';GLOBAL='annual_shared_offset';HALF='annual_state_offset_half'
HL={ANNUAL:'年度基线',STATE:'分状态修正（主方案）',GLOBAL:'统一修正（匹配对照）',HALF:'状态修正减半','fixed_transform_add_only':'R35 只加入重拟合','fixed_transform_step100':'R35 完整滚动系数'}
ML={'learned_market':'R18','learned_vol_interaction':'R19','learned_order_extension':'R23','learned_order_offset':'R25'};PRIMARY=list(ML)[1:]
SL={'negative_low':'负趋势／低波动','negative_high':'负趋势／高波动','nonnegative_low':'非负趋势／低波动','nonnegative_high':'非负趋势／高波动'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def pct(x):return '—' if pd.isna(x) else f'{x:.2%}'
def num(x):return '—' if pd.isna(x) else f'{x:.6f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def check():
    prep=read(OUT/'preparation_manifest.json');assert prep['protocol_sha256']==sha(ROOT/'protocol.json')
    for k in ['source_sha256','input_sha256']:
        for n,d in prep[k].items():assert sha(PROJECT/n)==d
    for phase in ['fitting','scoring','evaluation']:
        for n,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    assert read(OUT/'verification.json')['status']=='PASS';return prep
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();start=pd.Timestamp.now(tz='UTC').isoformat();met=csv('ensemble_metrics');seed=csv('seed_metrics');state=csv('state_metrics');chg=csv('direction_changes');params=csv('correction_parameters');support=csv('support_contract');pairs=read(OUT/'primary_comparisons.json');v=read(OUT/'verification.json');fit=read(OUT/'fitting_manifest.json');cfg=read(ROOT/'protocol.json');files=[]
    parts=['# 第三十七轮：年度模型上的受约束状态修正','',
    '本轮训练了少量新增修正参数，并生成新的历史滚动预测；年度神经网络、变换、分类主系数和交互系数全部固定。四状态沿用 R36 的年度边界，没有搜索状态数或波动阈值。全部 272 个历史周结果此前已查看，本轮是探索性历史比较，不是新增盲测。','',
    '## 固定规则','',
    '每季度使用相对该年度训练集新增、且联合标签已成熟的日样本。对每个方法和种子，用年度模型计算这些样本的原 logit z；每个有至少 20 个新增日成员的状态只拟合一个截距修正 d。目标为 Σ[log(1+exp(z+d))−y(z+d)]+20d²/2，且 |d|≤0.5。小于 20 个时 d=0，预测精确回退年度概率。每日标签有重叠，20 是预定启用规则，并非独立有效样本数或可靠性保证。','',
    '收缩强度 20、上限 0.5 在新预测前固定，未搜索。0.5 logit 上限最多改变约 12.44 个概率百分点；这不是固定加减 12.44 个百分点，也不是准确率保证。修正从零重算，季度之间不累加；每次年度模型切换后清零。新增训练样本集合跨季度累计重叠，不能将各季度计数相加当成独立样本。','',
    '统一修正对照使用相同的达标状态和样本，并把所有达标状态的修正强制相等。若有 K 个达标状态，共享截距的惩罚为 20K d²/2。因此它是分状态目标的等值约束版本；预测落在不达标状态时同样回退。只剩一个达标状态时两者相同。它不是对全部新样本无条件拟合的截距。','',
    '减半对照直接取已拟合状态修正的 0.5 倍，无新增拟合。三种子分别修正再转为概率，最后等权平均概率，并以严格大于 0.5 判上涨。R25 保持年度 R19 非顺序系数及原年度 gamma，只额外添加本轮截距；修正后的 R25 是新候选。原 native_mse 和训练频率对照直接复制年度结果，不做状态修正。','',
    f"本轮实际拟合 {fit['state_fits']} 个状态标量及 {fit['shared_fits']} 个共享标量；复用 18 个年度特征库，0 次新神经训练、0 次新变换拟合、0 次新特征推理。生成 9,792 条新学习模型种子预测；另复制年度控制行 3,264 条。",'']
    for period,label in [('extension_2021_2023','2021—2023：147 周'),('recent_2024_2026','2024—2026 年 8 月：125 周'),('pooled_2021_2026','全段：272 周')]:
        rows=[]
        for m in PRIMARY:
            for h in HL:
                r=met[met.period.eq(period)&met.history.eq(h)&met.method.eq(m)].iloc[0];rows.append([ML[m],HL[h],f'{r.correct_directions}/{r.n}',pct(r.accuracy),num(r.brier),num(r.log_loss),pct(r.balanced_accuracy),num(r.auroc)])
        parts += [f'## {label}','',table(['方法','方案','正确／周数','准确率','Brier↓','对数损失↓','平衡准确率','AUROC'],rows),'']
    parts += ['## 各年结果','']
    for m in PRIMARY:
        rows=[]
        for year in range(2021,2027):
            g=met[met.period.eq(f'year_{year}')&met.method.eq(m)].set_index('history');rows.append([year,int(g.loc[ANNUAL,'n'])]+[f"{int(g.loc[h,'correct_directions'])}（{pct(g.loc[h,'accuracy'])}）" for h in [ANNUAL,STATE,GLOBAL,HALF]])
        parts += [f'### {ML[m]}','',table(['年份','周数','年度','分状态','统一','减半'],rows),'']
    parts += ['## 修正触发与方向改变','',table(['时段','方法','方案','有支持／全部周','错→对','对→错','平均实际方向概率变化'],[[r.period,ML[r.method],HL[r.history],f'{r.eligible_weeks}/{r.n}',r.recoveries,r.regressions,f'{100*r.mean_signed_probability_change:+.3f} 个百分点'] for r in chg[chg.method.isin(PRIMARY)&chg.period.isin(['extension_2021_2023','recent_2024_2026','year_2026'])].itertuples()]),'',
    '触发取决于该信号状态在当期新增训练样本中的计数。修正后即使概率改变，方向也可能不变；方向变化同时列出改善与退步，不只展示成功案例。逐周概率、种子修正和全部 2026 年案例保存在 CSV。','',
    '## 2026 年修正参数与小样本','']
    for cutoff in ['2026-03-31','2026-06-30']:
        rows=[]
        for m in PRIMARY:
            for s in SL:
                g=params[params.cutoff.eq(cutoff)&params.method.eq(m)&params.kind.eq('state')&params.state.eq(s)];assert len(g)==3
                rows.append([ML[m],SL[s],int(g.n.iloc[0]),'启用' if g.eligible.iloc[0] else '回退',pct(g.up_rate.iloc[0]),pct(g.annual_mean_probability.mean()),f'{g.offset.mean():+.4f}',f'{g.offset.min():+.4f}～{g.offset.max():+.4f}'])
        parts += [f'### 截至 {cutoff}','',table(['方法','状态','新增日样本','处理','新增上涨率','年度预测均值（三种子）','平均 logit 修正','种子范围'],rows),'']
    parts += ['标签上涨率与年度预测均值的偏差决定修正的训练方向；R36 的年度标签均值差异不能直接代替模型残差。本轮训练集数值用于描述拟合，不是周预测成绩。','',
    '### 2026 年按信号状态评估','',table(['方法','状态','方案','正确／周数','准确率','Brier↓','样本标记'],[[ML[r.method],SL[r.state],HL[r.history],f'{r.correct_directions}/{r.n}',pct(r.accuracy),num(r.brier),'至少10周' if r.supported else '不足10周'] for r in state[state.period.eq('year_2026')&state.method.isin(PRIMARY)].itertuples()]),'',
    '## 种子稳健性','',table(['时段','方法','方案','种子最低／最高准确率','融合准确率'],[[p,ML[m],HL[h],f"{pct(seed[seed.period.eq(p)&seed.method.eq(m)&seed.history.eq(h)].accuracy.min())}／{pct(seed[seed.period.eq(p)&seed.method.eq(m)&seed.history.eq(h)].accuracy.max())}",pct(met[met.period.eq(p)&met.method.eq(m)&met.history.eq(h)].accuracy.iloc[0])] for p in ['extension_2021_2023','recent_2024_2026'] for m in PRIMARY for h in [ANNUAL,STATE,GLOBAL,HALF]]),'',
    '## 预定探索比较','',
    '三种新方案各自对年度，以及分状态对统一：三个主要方法、两个时段、方向错误与 Brier，共 48 项。采用固定 8 周循环区块、10,000 次配对 bootstrap，中心化双侧 p，再对 48 项统一作 Holm 调整。差值为候选损失减参考损失，负值有利于候选。该调整不覆盖历轮探索，也不能消除已查看历史结果后提出假设的选择偏差。','',
    f"48 项中 Holm 调整 p<0.05 的数量为 {sum(r['holm_adjusted_p']<.05 for r in pairs)}，最小调整 p 为 {min(r['holm_adjusted_p'] for r in pairs):.6f}。",'',
    table(['时段','方法','候选','参考','损失','差值','95% 区间','原始 p','Holm p'],[[r['window'],ML[r['candidate']],HL[r['history']],HL[r['reference_history']],r['metric'],num(r['difference']),f"[{num(r['ci95_low'])}, {num(r['ci95_high'])}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
    '## 审计与解释边界','',
    f"独立复核 {v['independent_scalar_solutions']} 个标量最优解；最大参数差 {v['maximum_scalar_solution_gap']:.3g}，最大新预测概率差 {v['maximum_prediction_gap']:.3g}。51 个训练接口对所有非成员特征／标签污染保持不变；23 个季度段通过未来价格扰动检查。全部概率融合、方向、稀疏指标、48 项比较、时间路由和精确回退均复核。",'',
    '5,315 个旧证据文件保持不变。原 R25 近期 73/125、58.4% 属于此前实验口径，继续单独保留；本轮直接比较的自然 20 遍年度 R25 是 71/125、56.8%，两者不能混为同一基线。该实验只检验受约束截距能否帮助方向分类，未测试交易成本、仓位或收益。','',
    '若结果改善，也只能作为继续研究的候选；若没有改善，也只否定本轮具体状态划分、记忆和截距修正规则，不能据此否定所有市场状态方法。不得按已知错误周修改强度或删状态后沿用本轮显著性结果。','',
    '![各年方向准确率](yearly_accuracy.png)','',
    '![近期准确率和概率损失变化](recent_changes.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十七轮实验报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files.append(p)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    colors=['#333333','#0077bb','#ee7733','#aa4499'];names=['年度基线','分状态修正','统一修正','状态修正减半'];pol=[ANNUAL,STATE,GLOBAL,HALF]
    fig,axs=plt.subplots(1,3,figsize=(16,6),sharey=True);fig.subplots_adjust(left=.065,right=.965,top=.80,bottom=.23,wspace=.15)
    for ax,m in zip(axs,PRIMARY):
        for h,label,color in zip(pol,names,colors):
            g=met[met.method.eq(m)&met.history.eq(h)&met.period.str.startswith('year_')].copy();g['year']=g.period.str[-4:].astype(int);g=g.sort_values('year');ax.plot(g.year,g.accuracy,marker='o',lw=2,label=label,color=color)
        ax.set_title(ML[m]);ax.set_xticks(range(2021,2027),['2021','2022','2023','2024','2025','2026*'],rotation=20);ax.set_ylim(.2,.8);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.06),ncol=4,frameon=False);fig.suptitle('固定年度模型后，状态修正能否改善各年预测？',fontsize=18,y=.95);fig.text(.5,.015,'每年周数：50、49、48、47、48、30；* 2026 年仅至 8 月。全部为已查看历史。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'yearly_accuracy.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6.5));fig.subplots_adjust(left=.09,right=.96,top=.79,bottom=.23,wspace=.28);x=np.arange(3);width=.24
    for j,(h,label,color) in enumerate(zip(pol[1:],names[1:],colors[1:])):
        a=met[met.period.eq('recent_2024_2026')&met.history.eq(h)].set_index('method');b=met[met.period.eq('recent_2024_2026')&met.history.eq(ANNUAL)].set_index('method');da=[100*(a.loc[m,'accuracy']-b.loc[m,'accuracy']) for m in PRIMARY];db=[a.loc[m,'brier']-b.loc[m,'brier'] for m in PRIMARY]
        for ax,values in zip(axs,[da,db]):
            bars=ax.bar(x+(j-1)*width,values,width,label=label,color=color)
            for rect,val in zip(bars,values):ax.annotate(f'{val:+.1f}' if ax is axs[0] else f'{val:+.4f}',(rect.get_x()+width/2,val),xytext=(0,4 if val>=0 else -5),textcoords='offset points',ha='center',va='bottom' if val>=0 else 'top',fontsize=9)
    for ax in axs:ax.set_xticks(x,[ML[m] for m in PRIMARY]);ax.axhline(0,color='#555555',lw=1);ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False);ax.margins(y=.28)
    axs[0].set_title('准确率变化（百分点，上升有利）');axs[1].set_title('Brier 变化（下降有利）');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=3,frameon=False);fig.suptitle('近期 125 周：三种修正相对年度基线',fontsize=18,y=.95);fig.text(.5,.018,'固定方案一次性比较；数值差异不等于未来提升或统计显著。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'recent_changes.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=start,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('Chinese full report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

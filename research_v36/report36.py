"""Chinese diagnostic report; no new forecast policy or fit."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
SL={'negative_low':'负趋势／低波动','negative_high':'负趋势／高波动','nonnegative_low':'非负趋势／低波动','nonnegative_high':'非负趋势／高波动'}
ML={'learned_market':'加性 R18','learned_vol_interaction':'波动交互 R19','learned_order_extension':'联合顺序项 R23','learned_order_offset':'固定 R19＋顺序项 R25'}
AL={'annual':'年度','add':'只加入','remove':'只移出','both':'加入＋移出'};RL={'annual':'年度成员','added':'新增成员','removed':'移出成员','retained':'保留成员','both':'滚动成员'}
WL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'};PRIMARY=list(ML)[1:]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def pct(x):return '—' if pd.isna(x) else f'{x:.2%}'
def num(x):return '—' if pd.isna(x) else f'{x:.6f}'
def pp(x):return '—' if pd.isna(x) else f'{100*x:+.2f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def check():
    p=read(OUT/'preparation_manifest.json');assert p['protocol_sha256']==sha(ROOT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for n,d in p[key].items():assert sha(PROJECT/n)==d
    for phase in ['calibration','diagnosis']:
        for n,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    assert read(OUT/'verification.json')['status']=='PASS';return p
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');th=read(OUT/'state_thresholds.json');t=csv('training_state_summary');comp=csv('state_comparisons');dec=csv('label_mix_decomposition');latest=csv('latest_annual_snapshot');pm=csv('state_prediction_metrics');eff=csv('state_effect_summary');w=csv('weekly_state_effects');v=read(OUT/'verification.json')
    parts=['# 第三十六轮：新增训练成员的市场状态与特征覆盖','',
      '本轮只对已有数据、模型概率和 R35 操作分解做诊断，没有新增神经训练、分类训练、特征推理、预测或显著性检验。6 组年度状态边界和标签无关的覆盖指标先冻结，再汇总状态内标签与预测结果。所有原准确率及旧 R25 近期 73/125、58.4% 记录保持不变。','',
      '## 四个粗状态与时间边界','',
      '趋势采用已有 60 日收盘对数变化除以同期日收益标准差与 √60 的描述量，按符号划分：<0 为负趋势，=0 归入非负趋势。20 日波动按年度五年成熟训练成员的中位数划分，等于中位数归低波动。组合形成四组，不搜索状态数或阈值；这些是可观察粗分组，不是已验证的隐含市场状态分类器。','',
      table(['使用年度','年度校准截止','成熟训练日样本','20 日波动中位数'],[[int(r['encoder_cutoff'][:4])+1,r['encoder_cutoff'],r['train_n'],num(r['volatility_median'])] for r in th]),'',
      '每年的边界固定到下一年度模型切换前，对所有季度、方法和种子一致。预测状态只用信号日及之前的价格量数据，边界在信号之前已确定。历史训练成员使用该年度坐标作组内比较，不能把这种诊断解释为在每个历史训练日已经知道年末边界。原 R17 的六格划分和 R29 的状态变化触发器均未改变。','',
      '每个训练状态单元少于 20 个日样本、每个预测状态单元少于 10 个周信号，会标记为“小样本”。这是描述性标记，不是有效独立样本数、统计显著性或模型资格判定；日标签重叠，周序列也存在时间依赖。空单元不平滑、不补标签比例、不与别组自动合并。','',
      '## 新增成员的状态与标签','',
      '年度 A、新增 Q\\A、移出 A\\Q、保留 A∩Q、完整滚动 Q 均沿用 R35 的成熟成员。季度集合相对年度累计，跨季度、角色相互重叠。下面按每年最后可用季度作单次快照；所有季度保存在 training_state_summary.csv，不能将这些日成员次数直接相加当成独立样本。','']
    dates=sorted(latest.cutoff.unique())
    for cutoff in dates:
        a=latest[latest.cutoff.eq(cutoff)&latest.role.eq('annual')].set_index('state');n=latest[latest.cutoff.eq(cutoff)&latest.role.eq('added')].set_index('state')
        parts += [f'### 截至 {cutoff}：年度与新增成员','',table(['状态','年度 n／上涨比例','新增 n／上涨比例','年度占比','新增占比','组内上涨比例差','样本标记'],[[SL[s],f"{int(a.loc[s,'n'])}／{pct(a.loc[s,'up_rate'])}",f"{int(n.loc[s,'n'])}／{pct(n.loc[s,'up_rate'])}",pct(a.loc[s,'share']),pct(n.loc[s,'share']),pp(n.loc[s,'up_rate']-a.loc[s,'up_rate']),'计数达到门槛' if bool(a.loc[s,'supported'] and n.loc[s,'supported']) else '小样本／缺失'] for s in SL]),'']
    parts += ['## 标签比例变化：状态占比与组内变化','',
      '设 w 为状态占比、r 为状态内上涨比例。先保持年度组内比例，计算占比变化项 Σ(w新−w年)r年；再计算组内变化项 Σw新(r新−r年)。两项严格还原总体新减年度上涨比例。该顺序固定了年度组内比例，因此是约定的记账方式，不能当作因果分解。','',
      '目标某状态没有样本时，其权重为零，不补造该状态上涨率；目标整体为空则不计算。如果目标出现年度没有覆盖的状态，则两项标为不可定义。完整结果同时包括移出、滚动成员，但下表只展示新增成员。','',
      table(['季度截止','新增日样本','年度上涨比例','新增上涨比例','总体差／百分点','占比变化项','组内变化项','达到计数门槛状态数'],[[r.cutoff,r.target_n,pct(r.annual_up_rate),pct(r.target_up_rate),pp(r.overall_delta),pp(r.composition),pp(r.within_state),f'{r.supported_states}/4'] for r in dec[dec.target_role.eq('added')].itertuples()]),'',
      '## 特征覆盖的含义','',
      '市场 5 维包含趋势、20 日波动、区间振幅、量变化，以及 60 日符号顺序描述。用年度成员原始描述的 1%／99% 分位数作常见范围，报告严格越界的坐标比例和任一坐标越界率。神经 25 维使用对应种子原年度模型已经保存的截尾边界，检查原始解码特征截尾前的越界情况。','',
      '同时记录原始坐标按年度均值／标准差缩放后的均方距离；市场使用年度原始均值与标准差，神经坐标使用原模型截尾训练均值与标准差。没有修改预测用缩放或阈值。神经覆盖先按每个种子计算，再平均种子指标，并非增加了三倍独立样本。5 维与 25 维的任一越界率不能直接比较；这些指标也不是多维异常概率或预测失效判定。','']
    c=latest[latest.cutoff.eq('2026-06-30')&latest.role.isin(['annual','added'])]
    parts += ['### 2026 年 6 月末：按状态的覆盖','',table(['角色','状态','日样本','市场任一越界','市场坐标越界','神经任一越界均值','神经坐标越界均值'],[[RL[r.role],SL[r.state],r.n,pct(r.mean_market5_outside_any),pct(r.mean_market5_outside_fraction),pct(r.mean_decoded25_outside_any),pct(r.mean_decoded25_outside_fraction)] for r in c.itertuples()]),'',
      '全部季度、角色及每个市场描述的越界率、均值、均方距离见 training_state_summary.csv；逐个神经种子覆盖见 decoded_coverage_by_seed.csv。','',
      '## 预测发生时的状态','',
      '以下用信号日状态对 R35 原概率分组。它描述全体新增成员重拟合后，哪些预测状态出现改善或退步，不能说明训练成员中的哪个状态子集造成了该变化。本轮没有按训练状态删样本或重拟合。四组概率及三种子平均规则都保持原值。','']
    for period in ['extension_2021_2023','recent_2024_2026','year_2026']:
        label=WL.get(period,'2026 年');g=pm[pm.period.eq(period)&pm.method.isin(PRIMARY)];parts += [f'### {label}：原预测的分组表现','',table(['方法','状态','组别','正确／周数','准确率','Brier↓','AUROC','样本标记'],[[ML[r.method],SL[r.state],AL[r.arm],f'{r.correct_directions}/{r.n}',pct(r.accuracy),num(r.brier),num(r.auroc),'计数达到门槛' if r.supported else '小样本／缺失'] for r in g.itertuples()]),'']
    g=eff[eff.period.eq('year_2026')&eff.method.isin(PRIMARY)&eff['case'].eq('all')];parts += ['### 2026 年：加入与移出影响按信号状态分组','',table(['方法','信号状态','周数','完整滚动对→错','错→对','加入贡献／概率百分点','移出贡献','新增同状态训练计数达标周'],[[ML[r.method],SL[r.state],r.n,r.regressions,r.recoveries,pp(r.mean_signed_add_effect),pp(r.mean_signed_remove_effect),r.training_supported_weeks] for r in g.itertuples()]),'',
      '加入／移出贡献沿用 R35 的两种操作顺序平均，再按随后实际方向调整符号；负值表示背离实际方向。表内是概率百分点，不是准确率变化。训练计数达标表示该周之前的年度与新增同状态成员均不少于 20 个；不意味着标签关系稳定。所有年度、状态及稳定正确／稳定错误／退步／改善分类完整保留。','',
      '## 复核与限制','',
      f"6 组年度中位数、分位数与均值尺度用独立算术核对；23 个季度段的描述量通过未来价格扰动检查。市场与神经覆盖按逐坐标标量算法复核，最大绝对差 {v['maximum_coverage_gap']:.3g}。460 个训练状态单元、576 个融合预测单元、1,728 个种子预测单元，以及标签分解与所有空／单类指标均复核。",'',
      '按状态相加可恢复原全段正确周数和 Brier，1,088 组周方法的原概率和操作效应不变。5,250 个旧证据文件保持不变。本轮没有新增显著性检验；R35 的 60 项探索比较原样归档。','',
      '状态划分粗、部分单元小、同一年度的新增集合累计重叠；条件上涨比例变化还可能来自未纳入状态的因素。所有历史结果已经查看，不能用本轮分组结果证明状态权重的未来收益，也不能直接按已知错误周设置权重或排除样本。年度主对照、交互设计及旧 R25 58.4% 记录继续保留。','',
      '![新增成员相对年度的状态内标签差异](state_label_shift.png)','',
      '![2026 年原概率按信号状态分组](state_forecast_2026.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十六轮诊断报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files=[p]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    mat=[];annot=[]
    for cutoff in dates:
        a=latest[latest.cutoff.eq(cutoff)&latest.role.eq('annual')].set_index('state');n=latest[latest.cutoff.eq(cutoff)&latest.role.eq('added')].set_index('state');mat.append([(n.loc[s,'up_rate']-a.loc[s,'up_rate'])*100 for s in SL]);annot.append([('—' if pd.isna(n.loc[s,'up_rate']) else f"{100*(n.loc[s,'up_rate']-a.loc[s,'up_rate']):+.1f}")+f"\nn={int(n.loc[s,'n'])}"+('*' if not (a.loc[s,'supported'] and n.loc[s,'supported']) else '') for s in SL])
    fig,ax=plt.subplots(figsize=(12.4,7.6));fig.subplots_adjust(left=.15,right=.88,top=.82,bottom=.16);mat=np.array(mat);limit=max(1,float(np.nanmax(abs(mat))));im=ax.imshow(mat,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    for i in range(len(dates)):
        for j in range(4):ax.text(j,i,annot[i][j],ha='center',va='center',fontsize=11,color='white' if np.isfinite(mat[i,j]) and abs(mat[i,j])>.6*limit else 'black')
    ax.set_xticks(np.arange(4),SL.values(),rotation=15);ax.set_yticks(np.arange(len(dates)),dates);fig.colorbar(im,ax=ax,fraction=.045,pad=.025,label='组内上涨比例差（百分点）');fig.suptitle('新增成员与年度成员：相同粗状态内的标签差异',fontsize=17,y=.96);fig.text(.5,.875,'每年仅取最后可用季度快照；格内 n 为新增日样本数。',ha='center',fontsize=11);fig.text(.5,.035,'* 年度或新增状态单元不足 20 个日样本；标签重叠，计数不代表独立样本量。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'state_label_shift.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(18,6.5),sharey=True);fig.subplots_adjust(left=.05,right=.99,top=.82,bottom=.30,wspace=.12);x=np.arange(4);colors=['#333333','#0077bb','#dd8844','#aa4466'];markers=['o','s','^','D']
    for ax,method in zip(axs,PRIMARY):
        for (arm,label),color,marker in zip(AL.items(),colors,markers):
            g=pm[pm.period.eq('year_2026')&pm.method.eq(method)&pm.arm.eq(arm)].set_index('state').loc[list(SL)];ax.plot(x,g.accuracy,color=color,marker=marker,lw=2,label=label)
        counts=pm[pm.period.eq('year_2026')&pm.method.eq(method)&pm.arm.eq('annual')].set_index('state').loc[list(SL),'n'];ax.set_xticks(x,[SL[s]+f'\nn={int(counts.loc[s])}'+('*' if counts.loc[s]<10 else '') for s in SL],rotation=20,fontsize=9);ax.set_title(ML[method],fontsize=13);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=4,frameon=False,fontsize=12);fig.suptitle('2026 年已有预测：按信号日趋势与波动分组',fontsize=18,y=.96);fig.text(.5,.025,'共 30 个周信号；* 单元不足 10 周。未产生新预测，分组表现不能证明状态加权有效。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'state_forecast_2026.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('Chinese diagnostic report and two figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

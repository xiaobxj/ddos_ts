"""Verified fixed-parent experiment report and scientific figures."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
sys.path.insert(0,str(ROOT.parent/'research_v24'));from report24 import pct,num,line,table,yes
LABELS={'learned_market':'学习：加性模型','learned_vol_interaction':'学习：原波动交互','learned_order_extension':'学习：顺序项联合拟合（R23）','learned_order_shrink4':'学习：4 倍收缩（R24）','learned_order_shrink16':'学习：16 倍收缩（R24）','learned_order_offset':'学习：固定原模型＋顺序项',
    'raw_trend':'简单：加性趋势','raw_vol_interaction':'简单：原波动交互','raw_order_extension':'简单：顺序项联合拟合（R23）','raw_order_shrink4':'简单：4 倍收缩（R24）','raw_order_shrink16':'简单：16 倍收缩（R24）','raw_order_offset':'简单：固定原模型＋顺序项','order_only':'仅顺序指标','native_mse':'原 MSE 模型','training_frequency':'训练期上涨频率'}
MAIN=list(LABELS);WINDOWS={'early_2015_2017':'2015–2017','late_2018_2020':'2018–2020','pooled_2015_2020':'合并描述'}
FAMILIES=[['learned_vol_interaction','learned_order_extension','learned_order_offset'],['raw_vol_interaction','raw_order_extension','raw_order_offset']]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csv(name):return pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip')
def metric_row(r):return [LABELS[r.method],f'{r.correct_directions}/{r.n}',pct(r.accuracy),pct(r.balanced_accuracy),num(r.auroc),num(r.brier),num(r.log_loss)]
def pair_row(r):return [WINDOWS[r['window']],LABELS[r['candidate']]+' / '+LABELS[r['reference']],'Brier' if r['metric']=='brier' else '方向错误率',f"{r['difference']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",f"{r['p']:.4f}",f"{r['holm_adjusted_p']:.4f}"]
def target_row(r):return [LABELS[r.method],yes(r.early_accuracy_retained),yes(r.late_parent_accuracy_retained),yes(r.early_brier_retained),yes(r.late_brier_retained),yes(r.all_targets_met)]

def plot(metrics,years,summary):
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#b17b24','#85919a','#187b8e'];fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');xp=np.arange(2)
    for col,family in enumerate(FAMILIES):
        for row,key in enumerate(['accuracy','brier']):
            ax=axes[row,col]
            for j,method in enumerate(family):
                values=np.array([metrics[metrics.window.eq(w)&metrics.method.eq(method)].iloc[0][key] for w in ['early_2015_2017','late_2018_2020']])*(100 if row==0 else 1)
                bars=ax.bar(xp+(j-1)*.23,values,.23,color=colors[j],label=['原波动交互','顺序项联合拟合','固定原模型＋顺序项'][j]);ax.bar_label(bars,labels=[f'{v:.2f}%' if row==0 else f'{v:.4f}' for v in values],padding=3,fontsize=9)
            for i,w in enumerate(['early_2015_2017','late_2018_2020']):
                t=metrics[metrics.window.eq(w)].set_index('method');value=t.loc['training_frequency',key]*(100 if row==0 else 1);ax.plot([i-.42,i+.42],[value]*2,ls='--',color='#bf5962',lw=1.1,label='训练频率' if i==0 else None)
                if row==0:ax.plot([i-.42,i+.42],[100*t.loc['native_mse',key]]*2,ls=':',color='#333333',lw=1.1,label='原 MSE' if i==0 else None)
            ax.set_xticks(xp,['2015–2017 · 120 周','2018–2020 · 141 周']);ax.set_ylim(0,72 if row==0 else .325);ax.set_ylabel('准确率（%）' if row==0 else 'Brier 概率误差');ax.set_title(('学习表征' if col==0 else '简单特征')+('：方向准确率 ↑' if row==0 else '：Brier 误差 ↓'),loc='left',weight='bold');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False);fig.suptitle('固定原模型后，学习分支前段再多对 1 周；后段仍未恢复 58.16%\n只训练一个顺序补充系数，原模型全部斜率和截距保持不变',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'fixed_parent_comparison.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    for col,family in enumerate(FAMILIES):
        method=family[-1];ax=axes[0,col];g=summary[summary.method.eq(method)&summary.split.eq('training')].groupby('cutoff')[['offset_gamma','joint_gamma']].mean()
        ax.plot(range(2015,2021),g.joint_gamma,ls='--',marker='s',color='#b17b24',label='联合拟合系数');ax.plot(range(2015,2021),g.offset_gamma,marker='o',color='#187b8e',label='固定原模型系数');ax.set_ylim(0,.24 if col==0 else .11);ax.set_xlabel('预测年份');ax.set_ylabel('顺序系数 γ（相同输入坐标）');ax.set_title(('学习表征' if col==0 else '简单特征')+'：新增系数与联合拟合接近',loc='left',weight='bold');ax.legend(frameon=False);ax.grid(alpha=.15)
    ax=axes[1,0];t=years.pivot(index='year',columns='method',values='correct_directions');xp=np.arange(6)
    for j,family in enumerate(FAMILIES):
        delta=t[family[-1]]-t[family[1]];bars=ax.bar(xp+(j-.5)*.32,delta,.32,color=['#187b8e','#b17b24'][j],label=['学习表征','简单特征'][j]);ax.bar_label(bars,labels=[f'{int(v):+d}' for v in delta],padding=3)
    ax.set_xticks(xp,range(2015,2021));ax.set_ylim(-1.5,1.6);ax.set_yticks([-1,0,1]);ax.axhline(0,color='#555555',lw=.8);ax.set_ylabel('比联合拟合多对的周数');ax.set_title('方向变化只发生在前段的两个周信号',loc='left',weight='bold');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1,1];g=summary[summary.split.eq('validation')].groupby('method')[['parent_constraint_rms','order_coefficient_change_rms','total_logit_difference_rms']].mean().loc[['learned_order_offset','raw_order_offset']];xp=np.arange(2)
    for j,(key,label,color) in enumerate([('parent_constraint_rms','父模型系数约束部分','#85919a'),('order_coefficient_change_rms','顺序系数变化部分','#b17b24'),('total_logit_difference_rms','两部分合计后的差异','#187b8e')]):
        values=g[key].to_numpy();bars=ax.bar(xp+(j-1)*.23,values,.23,color=color,label=label);ax.bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=3,fontsize=9)
    ax.set_xticks(xp,['学习表征','简单特征']);ax.set_ylim(0,.012);ax.set_ylabel('下一年 logit 差异 RMS 的各头均值');ax.set_title('两部分在 logit 上相加，RMS 不能直接相加',loc='left',weight='bold');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('父模型系数被冻结，顺序项仍保留相近的影响\n与第 23 轮相比，两分支后段方向逐条相同；分解仅用于诊断',fontsize=15,weight='bold')
    for ext in ['png','svg']:fig.savefig(OUT/f'fixed_parent_diagnostics.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

def main():
    v=read(OUT/'verification.json');assert v['status']=='PASS';p=read(ROOT/'protocol.json');fit=read(OUT/'training_manifest.json');metrics=csv('ensemble_metrics');years=csv('yearly_metrics');summary=csv('component_summary');pairs=read(OUT/'primary_comparisons.json');assess=read(OUT/'assessments.json');targets=csv('retention_targets');changes=csv('direction_changes');ensemble=csv('ensemble_predictions');heads=read(OUT/'heads.json')
    plot(metrics,years,summary);details=[]
    for parent,joint,fixed in FAMILIES:
        a=ensemble[ensemble.method.eq(fixed)].set_index('date');b=ensemble[ensemble.method.eq(joint)].set_index('date');c=ensemble[ensemble.method.eq(parent)].set_index('date')
        for date in a.index[a.direction_up.ne(b.direction_up)]:details.append(dict(method=fixed,date=date,actual_up=int(a.loc[date,'actual_up']),parent_probability=float(c.loc[date,'probability']),joint_probability=float(b.loc[date,'probability']),fixed_probability=float(a.loc[date,'probability']),joint_correct=bool(b.loc[date,'direction_up']==a.loc[date,'actual_up']),fixed_correct=bool(a.loc[date,'direction_up']==a.loc[date,'actual_up'])))
    pd.DataFrame(details).to_csv(OUT/'changed_signal_details.csv',index=False)
    facts=dict(significant_improvements=sum(r['difference']<0 and r['holm_adjusted_p']<.05 for r in pairs),significant_deteriorations=sum(r['difference']>0 and r['holm_adjusted_p']<.05 for r in pairs),retention_target_passes=int(targets.all_targets_met.sum()),newton_updates=fit['new_newton_iterations'],changed_signals_vs_joint=len(details),parent_coefficients_unchanged=v['parent_coefficients_bitwise_unchanged'],parent_intercepts_unchanged=v['parent_intercepts_bitwise_unchanged'],minimum_joint_fit_advantage=min(h['objective']-h['joint_objective'] for h in heads),maximum_joint_fit_advantage=max(h['objective']-h['joint_objective'] for h in heads))
    (OUT/'report_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    chunks=['# 第二十五轮方法测试：固定原交互，只拟合一个顺序系数','2026-09-12｜沪深 300｜24 个单系数拟合｜原模型系数与截距冻结｜年度向前检验',
        '**学习分支前段再改善 1 周，后段问题仍在。** 固定原波动交互后，前段准确率从第 23 轮联合拟合的 61/120（50.83%）提高到 62/120（51.67%）；相对原交互的 56/120（46.67%），共多对 6 周。后段为 81/141（57.45%），仍低于原交互的 82/141（58.16%），且与第 23 轮的方向逐条相同。',
        '这说明在本次固定方案和历史样本上，后段损失并非只由父模型系数重拟合造成：完全锁定它们后，顺序补充项依然改变了原模型的 5 次后段判断，其中 3 次由对变错、2 次由错变对。这个结果不证明损失在未来必然重现，也不排除其他状态定义或训练方式。',
        '学习分支前段 Brier 从 0.267551 降至 0.267098，后段从 0.244092 升至 0.244132。前段仍没有超过训练频率基准的 52.50% 准确率和 0.249839 Brier；后段仍没有恢复原交互准确率。因此保留本轮局部改善，暂不替换原模型。',
        '32 项固定比较经 Holm 校正后，没有显著改善或恶化。前段学习分支相对原交互的方向改善原始 p=0.0058，校正后 p=0.1740。两个候选均未满足全部保留目标或跨时期描述筛选。实现核验 PASS 不等于预测能力已得到确认。',
        '## 本轮只改变什么',
        '原模型指每个年度截止日已经拟合并冻结的第 19 轮波动交互模型。保留它的全部斜率和截距，记其输出为 z_parent。沿用第 23 轮的顺序特征 h，只拟合一个 γ：z_fixed=z_parent+γh。没有新增截距、重新校准父模型、缩放原系数、预测后混合或按年份开关。',
        'λ=0.01，与第 23 轮联合拟合相同。沿用同一 h、监督射线、投影、训练归一化、样本和标签。因此本轮与第 23 轮比较的是“父模型系数是否固定”这项约束。第 24 轮的两档收缩继续完整保留为描述对照，本轮没有再次搜索惩罚强度。',
        '对每折训练数据，优化 J(γ)=平均二元对数损失(z_parent+γh,y)+0.01γ²/2。γ=0 时精确还原原模型。父模型惩罚常数 C=0.01×Σ父斜率²/2 在求解中省略，比较完整目标时加回。截距一直固定且不惩罚。',
        '一阶导为 mean[h·(sigmoid(z_parent+γh)−y)]+0.01γ；二阶导为 mean[h²·p·(1−p)]+0.01，严格为正。因此这个一维问题有唯一最优解。主要求解器从 γ=0 做带回溯的 Newton 更新，独立求解器用另一套梯度／损失计算和区间求根核对。',
        '训练内应满足：第 23 轮全部系数联合拟合的完整目标 ≤ 本轮固定父模型目标 ≤ 第 19 轮原模型目标。全部 24 个模型通过此检查。这只是可选系数范围不同导致的训练内性质，不能据此保证下一年表现。',
        '原父模型每个学习头有 31 个分类系数（含截距），每个简单头有 28 个；本轮冻结共 726 个这样的系数，仅新增 24 个可训练 γ。组合模型序列化为 750 个分类坐标。上游神经表征和投影参数另计，不能把整个预测流程称为只有一个参数。',
        '## 数据、状态与时间顺序',
        '六个截止日为 2014–2019 年底，各预测下一年；训练数为 1077、1190、1434、1678、1921、2165，测试数为 22、48、50、49、46、46。前段是 2015–2017 年 120 周，后段是 2018–2020 年 141 周；没有把新近年份当作新增独立数据。',
        '训练标签必须 joint_completed≤cutoff，预测严格采用 p>0.5。学习分支平均三个种子的概率，简单分支每年一个模型。原执行口径下周收益目标、周五信号、125 根观察窗口、OHLC 无效条排除和不完整的 2015 年覆盖保持不变。没有新增样本筛选。',
        '顺序特征沿用 60 日涨跌符号相邻积减去同一符号数量随机排列下的期望，再与原监督信号相乘，并按训练期残差投影和标准差变换。它仍忽略幅度，对整体符号翻转和时间反转不变；日状态仍要求 121 根有效 OHLC。所有 24 个顺序投影复用，没有重新拟合。',
        '24 条上游来源链、训练行号、标签成熟时间和检查点哈希继续核对。原神经表征、射线、父交互和残差化都在各自完整训练折上估计；本轮是年度向前评分，**不是整条流程的时间顺序 OOF**。本轮没有运行神经前向、神经训练或 HMM 拟合，18 个旧神经状态复算依赖冻结证据。',
        '## 主要结果',
        '准确率、平衡准确率、AUROC 越高越好；Brier 和对数损失越低越好。原 MSE 输出不作为概率，不报告它的概率损失。']
    for w in p['windows']:
        t=metrics[metrics.window.eq(w['name'])].set_index('method').loc[MAIN].reset_index();chunks += ['### '+WINDOWS[w['name']],table(['方法','正确/样本','准确率','平衡准确率','AUROC','Brier','对数损失'],[metric_row(r) for r in t.itertuples()])]
    chunks += ['![固定父模型与联合拟合的表现](fixed_parent_comparison.png)',
        '简单特征分支前段从第 23 轮的 42.50% 降为 41.67%，后段保持 55.32%，两段 Brier 略降但仍高于训练频率基准。没有出现与学习分支一致的方向改善。',
        '### 每年正确周数']
    for family in FAMILIES:
        rows=[]
        for year,g in years.groupby('year'):
            t=g.set_index('method');rows.append([year,int(g.iloc[0]['n'])]+[int(t.loc[m,'correct_directions']) for m in family+['native_mse','training_frequency']])
        chunks.append(table(['年份','样本']+[LABELS[m] for m in family+['native_mse','training_frequency']],rows))
    chunks += ['学习分支比第 23 轮多对的 1 周在 2016 年，其他年份正确数不变；简单分支少对的 1 周在 2015 年。两分支后段不仅总准确率相同，方向预测也逐条相同。',
        '### 相对原交互与联合拟合的变化',
        table(['时期','候选','对照','改变数','对→错','错→对'],[[WINDOWS[r.window],LABELS[r.method],LABELS[r.reference],r.changed,r.correct_to_wrong,r.wrong_to_correct] for r in changes[changes.reference.isin([m for family in FAMILIES for m in family[:2]])].itertuples()]),
        '以下是与联合拟合相比发生方向变化的全部两条信号，而非挑选有利案例。概率是模型估计值，接近 0.5 表示在固定阈值附近；不能由此另行选择阈值。',
        table(['候选','日期','实际方向','原交互概率','联合拟合概率','本轮概率','联合拟合正确','本轮正确'],[[LABELS[r['method']],r['date'],'涨' if r['actual_up'] else '非涨',num(r['parent_probability']),num(r['joint_probability']),num(r['fixed_probability']),yes(r['joint_correct']),yes(r['fixed_correct'])] for r in details]),
        '学习分支在 2016-12-09 从略高于 0.5 降到略低于 0.5，由错变对；简单分支在 2015-11-20 则反向跨过阈值，由对变错。它们说明方向准确率可能被很小的概率变化改变，不能把多对 1 周视为大幅结构突破。',
        '## 系数与差异分解',
        '下面的学习分支系数是同一年三个种子的均值，简单分支为单模型。h 的坐标完全相同，所以这里的 γ 可以直接比较；父模型斜率和截距在本轮均逐项保持原值。',
        table(['候选','预测年','联合拟合 γ','固定父模型 γ','差值'],[[LABELS[method],int(cutoff[:4])+1,num(g.joint_gamma.mean()),num(g.offset_gamma.mean()),f'{g.gamma_difference.mean():+.6f}'] for (method,cutoff),g in summary[summary.split.eq('training')].groupby(['method','cutoff'])]),
        '两种模型的差异精确写为：z_fixed−z_joint=(z_parent−z_joint_parent)+(γ_fixed−γ_joint)h。第一部分来自父模型系数约束，第二部分来自这一约束下 γ 的变化。约束会改变 γ 的最优值，不能把本轮描述为只把联合模型的父系数机械替换、其他一切不动。',
        '逐个模型、逐条样本的上述分解已核验。下面按各头等权取下一年均方根差异的均值；RMS 不能相加，因此两部分的 RMS 不应被解释为百分比贡献。学习分支的系数均值图也不是三种子平均概率的 logit。',
        table(['候选','父系数约束部分 RMS','顺序系数变化部分 RMS','总 logit 差异 RMS','相对原交互概率变化 RMS','相对联合拟合概率变化 RMS'],[[LABELS[method],num(g.parent_constraint_rms.mean()),num(g.order_coefficient_change_rms.mean()),num(g.total_logit_difference_rms.mean()),num(g.rms_probability_change_vs_parent.mean()),num(g.rms_probability_change_vs_joint.mean())] for method,g in summary[summary.split.eq('validation')].groupby('method')]),
        '![系数和差异诊断](fixed_parent_diagnostics.png)',
        '这些分解仅说明两个已训练模型如何不同，不是市场因果归因，也没有把未训练的拆分版本作为额外预测策略评分。',
        '## 固定保留目标与描述筛选',
        '四项保留目标沿用前轮：前段准确率不低于第 23 轮对应分支；后段准确率不低于原交互；前段和后段 Brier 分别不高于第 23 轮。全部满足才算达到该描述目标。',
        table(['候选','保住前段准确率','恢复原交互后段准确率','保住前段 Brier','保住后段 Brier','全部满足'],[target_row(r) for r in targets.itertuples()]),
        '另保留 11 条严格筛选：准确率分别超过原交互、原加性、原 MSE、训练频率；Brier 分别低于原交互、原加性、训练频率、仅顺序；对数损失低于训练频率；至少两年准确率超过原 MSE、至少两年 Brier 低于训练频率。相等不通过，两时期都要全部满足。',
        table(['时期','候选','满足条件','跨时期通过'],[[WINDOWS[a['window']],LABELS[a['method']],f"{sum(a['flags'].values())}/11",yes(a['cross_period_descriptive_pass'])] for a in assess]),
        '## 固定比较与统计限制',
        '每候选每时期相对原交互、联合拟合、原加性模型各比较 Brier 和方向错误率，另比较训练频率与仅顺序的 Brier，共 2×2×8=32 项。候选减对照的损失，负值较好。循环区块长度 8 个保留观测，10000 次重采样，随机种子 20260910，各时期单独生成；双侧中心化 p 一起做 Holm 校正。区间为边际 95% 区间，没有合并时期主要检验。',
        table(['时期','候选 / 对照','损失','差值','95% 区间','原始 p','Holm p'],[pair_row(r) for r in pairs]),
        '本轮校正不覆盖以前多轮设计选择，也没有修正上游拟合不确定性。没有显著恶化不代表等效或非劣。旧的 26 个固定状态单元、35 个方法、两个时期共 1820 行状态诊断全部保留，少于 20 个样本继续标稀疏，未用于日期或模型选择。',
        '## 实现与验证',
        f"24 个标量模型共执行 {fit['new_newton_iterations']} 次 Newton 更新，梯度绝对值≤10⁻¹⁰、曲率≥0.01。独立区间求根验证全部 24 个解，系数最大差 {v['maximum_alternate_coefficient_gap']:.3e}、目标最大差 {v['maximum_alternate_objective_gap']:.3e}、训练概率最大差 {v['maximum_alternate_probability_gap']:.3e}。独立解没有用于交付预测。",
        f"1044 条新单模型概率的最大复算误差为 {v['maximum_new_forecast_error']:.3e}。原模型 γ=0 的概率复核最大差为 {v['maximum_parent_probability_gap']:.3e}；同输入联合拟合概率复核最大差为 {v['maximum_joint_probability_gap']:.3e}。冻结的父斜率和截距均逐项一致。",
        '六组人工数据覆盖普通、零特征、常数特征、全涨标签、全非涨标签和极端固定 logit；检查一阶／二阶有限差分、符号翻转、零特征精确复原。真实数据复核 48 组训练／下一年输入及 24 条上游来源链。训练完成后才生成本轮新评分。',
        '总计 37860 行累计训练输入，24 个新增可训练系数；保留 726 个父模型分类系数，组合序列化坐标为 750。3585 个旧证据文件哈希保持不变。新单模型记录 1044 条加旧记录 14616 条，共 15660 条；35 个方法各有 261 条集成记录，共 9135 条。2261 组指标、340 个可靠性分箱、32 项比较均复核通过。',
        '训练内指标如下，仅用于拟合诊断。完整带惩罚目标包含冻结父模型的惩罚常数，才能与旧模型对比。',
        table(['方法','截止日','训练准确率','训练 Brier','完整带惩罚目标'],[[LABELS[method],cutoff,pct(g.accuracy.mean()),num(g.brier.mean()),num(g.objective.mean())] for (method,cutoff),g in csv('training_metrics').groupby(['method','cutoff'])]),
        '## 研究决定与下一步',
        '保留原交互后段 58.16%，保留第 23 轮顺序项和第 22 轮路径效率的改善记录，也保留本轮学习分支前段 51.67% 的结果。本轮固定父模型是一个参数更受限的补充方案，尚未达到跨时期替换条件。简单分支的方向表现也没有给出一致支持。',
        '在本次设定下，仅提高统一惩罚、或锁定全部父模型系数，都没有修复学习分支的后段净损失。后续应优先检验顺序补充项的效用是否随当时可见的市场条件稳定变化，而非继续按已见测试年调整参数。先建立整个上游都只使用当时历史的时间顺序训练／验证证据，再决定是否需要条件化补充或允许 γ 接近零。这个条件化模型尚未在本轮训练。',
        '不能根据本轮的两个临界信号事后改变 0.5 阈值，也不能根据 2017／2018 年得失设置年份开关。固定父模型可以限制改动范围，但不保证原模型方向准确率不会下降。',
        '## 限制与文件',
        '**不能作为独立确认**：261 个日期已多次查看，2015 年覆盖不完整，日频训练样本的周标签存在重叠，固定长度区块仅提供近似不确定性。没有交易成本、执行或净收益检验，也没有完整复现论文；已使用的 2021–2026 年日期不能重新命名成独立留出集。',
        f"冻结协议 SHA-256：`{v['protocol_sha256']}`。",
        '主要文件：[协议](../protocol.json)、[单系数实现](../common25.py)、[独立验证](verification.json)、[时间来源链](temporal_provenance.csv)、[标量测试](synthetic_offset_checks.csv)、[系数](coefficients.csv)、[逐条差异分解](interaction_components.csv)、[影响幅度](component_summary.csv)、[全部方向变化信号](changed_signal_details.csv)、[集成预测](ensemble_predictions.csv)、[全部单模型预测](model_predictions.csv)、[汇总指标](ensemble_metrics.csv)、[年度指标](yearly_metrics.csv)、[种子指标](seed_metrics.csv)、[状态指标](state_metrics.csv)、[比较](primary_comparisons.json)、[保留目标](retention_targets.csv)、[筛选](assessments.json)。',
        '图表矢量版：[模型对比 SVG](fixed_parent_comparison.svg)、[诊断 SVG](fixed_parent_diagnostics.svg)。复现请在保留所有前轮依赖的独立副本及空结果目录中，按 prepare25.py → contract25.py → train25.py → score25.py → evaluate25.py → verify25.py 执行；计算解释器为 research_v4/.venv_gpu/Scripts/python.exe，报告绘图使用默认 Python。不得覆盖当前冻结记录。']
    report='\n\n'.join(chunks)+'\n';(OUT/'第二十五轮测试报告.md').write_text(report,encoding='utf-8')
    note='# 第二十五轮研究留存\n\n2026-09-12：固定原波动交互全部系数和截距，仅拟合 24 个顺序补充系数，独立区间求根及预测核验通过。\n\n学习分支前段 51.67%（比 R23 多对 1 周），后段仍为 57.45%，方向与 R23 逐条相同。原交互后段 58.16% 继续保留。本轮局部改善留存，暂不替代模型；32 项比较没有 Holm 校正后显著改善。\n\n后段损失并非只由父模型重拟合造成。下一步优先建立完整因果时间验证，再检验补充项效用的条件变化；本轮没有训练条件选择器，历史结果不能作为独立确认。\n\n[完整报告](results/第二十五轮测试报告.md)｜[协议](protocol.json)｜[验证](results/verification.json)\n'
    (ROOT/'README.md').write_text(note,encoding='utf-8');print(json.dumps(dict(status='BUILT_NOT_YET_VISUALLY_REVIEWED',report_characters=len(report),figures=2,facts=facts),indent=2))
if __name__=='__main__':main()

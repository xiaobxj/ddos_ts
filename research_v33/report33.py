"""Descriptive report only: no new prediction policy or fitted model."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
ML={'learned_market':'加性 R18','learned_vol_interaction':'波动交互 R19','learned_order_extension':'联合顺序项 R23','learned_order_offset':'固定 R19＋顺序项 R25'}
GL={'intercept':'截距','decoded':'解码线性','market':'市场线性','vol':'波动交互','order':'顺序交互'}
CL={'regression':'对→错','recovery':'错→对','stable_correct':'仍然正确','stable_wrong':'仍然错误'}
PL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'}
PRIMARY=list(ML)[1:]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csv(n):return pd.read_csv(OUT/f'{n}.csv',float_precision='round_trip')
def now():return pd.Timestamp.now(tz='UTC').isoformat()
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pp(x):return '—' if pd.isna(x) else f'{x*100:+.2f}'
def check():
    prep=read(OUT/'preparation_manifest.json');assert prep['protocol_sha256']==sha(ROOT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for n,d in prep[key].items():assert sha(PROJECT/n)==d
    for n,d in read(OUT/'diagnosis_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    assert read(OUT/'verification.json')['status']=='PASS';return prep
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();perf=csv('performance_summary');weekly=csv('weekly_attribution');comp=csv('component_summary');dom=csv('dominant_summary');shift=csv('training_window_changes');v=read(OUT/'verification.json');p2026=perf[perf.period.eq('year_2026')];flips=csv('flips_2026')
    parts=['# 第三十三轮：季度分类端为何改变原有判断','',
      '本轮完成已有预测的数值贡献分解，没有新增训练、特征推理、候选策略或显著性检验。逐条分析年度分类端与季度分类端在同一个年度神经网络下的预测差异，覆盖全部 272 周、4 个学习方法、3 个种子；2026 年单列全部 30 周及所有改善、恶化的判断。原有准确率没有变化。','',
      '本报告中的贡献依赖明确的分组、坐标和分配路径，是既有预测差异的账目分解，不是某个参数的独立因果效应，也不是唯一的特征重要性。训练样本变化可能同时改变多个部分，不能仅凭其中一个系数变大就断言它造成错误。','',
      '## 2026 年的预测翻转','',table(['方法','周数','年度正确','季度分类端正确','对→错','错→对','仍正确','仍错误'],[[ML[r.method],r.n,r.annual_correct,r.quarter_head_correct,r.regression,r.recovery,r.stable_correct,r.stable_wrong] for r in p2026.itertuples()]),'',
      '正确数的净变化等于“错→对”减“对→错”。各方法的翻转周可能重合，不能把方法间的计数相加解释为独立市场事件。其余未翻转的预测也保留在 all_2026_cases.csv。','',
      '## 如何分解','',
      '对每个种子分别重建涨跌对数几率：z = 截距 + 25 个解码特征线性项 + 4 个市场描述线性项 + 波动交互项 + 顺序交互项。方法中不存在的交互项记为零。原始神经特征和当日市场描述完全相同，但年度和季度分类端各用自己的缩放、监督方向、残差投影与系数。','',
      '每项变化为新贡献减旧贡献。对于乘积 θx，进一步对称拆分：系数部分 = (θ新−θ旧)(x新+x旧)/2，输入变换部分 = (θ新+θ旧)(x新−x旧)/2。二者相加恰好还原 θ新x新−θ旧x旧；截距全记为系数部分。输入变换部分包含缩放/截尾更新，以及交互项中监督方向、投影和残差尺度的更新。','',
      '为保留非线性的概率转换，每个种子使用 w = [sigmoid(z新)−sigmoid(z旧)]/(z新−z旧)，把各项对数几率差乘以 w。总差为零时取 sigmoid 导数，近零时使用稳定的等价公式。然后对三个种子的每项概率贡献取平均，得到融合概率的精确差额；没有用 sigmoid(平均 z) 替代原来的平均概率。','',
      '独立核验另用 64 点数值积分计算线性路径上 sigmoid 导数的平均值，与上述权重核对。五项概率贡献之和、每项系数和输入变换两部分之和均可还原存档预测，只有浮点舍入差异。该分配是选定线性路径的账目分解，没有进行置换 Shapley 或混合参数的新策略评分。','',
      '下面的有符号概率贡献再乘以实际涨跌方向：正值表示朝实际结果移动，负值表示背离实际结果。单位均为概率百分点。实际结果只用于事后解释，不参与重新产生预测。贡献之间可能相互抵消，因此不报告分母接近零的贡献百分比。','']
    for method in PRIMARY:
        c=comp[comp.period.eq('year_2026')&comp.method.eq(method)&comp.case.eq('regression')].set_index('group');d=dom[dom.period.eq('year_2026')&dom.method.eq(method)&dom.case.eq('regression')].set_index('group');n=int(c.n.iloc[0]);parts += [f'## {ML[method]}：{n} 个对→错周的平均贡献','',table(['项','总贡献（百分点）','系数变化部分','输入变换部分','作为最大不利项的周数'],[[GL[k],pp(c.loc[k,'mean_signed_contribution']),pp(c.loc[k,'mean_signed_coefficient']),pp(c.loc[k,'mean_signed_transform']),int(d.loc[k,'dominant_adverse_count'])] for k in GL]),'',
          '“最大不利项”仅指本轮分解中最负的单项，不能解释为唯一原因或单独改变它就一定能修复预测。系数变化与输入变换的拆分也依赖参数坐标；例如把输入扩大一倍、系数减半，预测可完全不变而两部分互相抵消。','']
        g=flips[flips.method.eq(method)];parts += [f'### {ML[method]}：全部 2026 年翻转周','',table(['信号日','实际','变化','年度上涨概率','季度上涨概率']+[GL[k] for k in GL]+['合计有符号贡献'],[[r.date,'涨' if r.actual_up else '跌',CL[r.case],f'{r.old_probability:.2%}',f'{r.new_probability:.2%}']+[pp(getattr(r,'signed_'+k)) for k in GL]+[pp(r.signed_probability_change)] for r in g.itertuples()]),'']
    parts += ['## 改善周也完整保留','',table(['方法','错→对周数']+[GL[k]+'平均贡献' for k in GL],[[ML[method],int(comp[(comp.period=='year_2026')&(comp.method==method)&(comp.case=='recovery')].n.iloc[0])]+[pp(comp[(comp.period=='year_2026')&(comp.method==method)&(comp.case=='recovery')].set_index('group').loc[k,'mean_signed_contribution']) for k in GL] for method in PRIMARY]),'',
      '没有错→对样本时显示横线，不把缺失均值当作零效应。每一方法的全部稳定正确和稳定错误周也纳入 component_summary.csv，避免仅由新增错误反推结论。','',
      '## 训练窗口如何改变','',
      '下表逐项核对 2026 年两个季度分类截止日与 2025 年末年度训练样本的差异。新增、移出、保留按样本集合定义；新增可以包含此前日期但标签刚成熟的样本。这里只描述样本变化，没有进行加入/移除某批样本的控制重训，不能确认哪批样本单独造成系数变化。','',
      table(['分类截止日','年度样本','季度样本','保留','新增','移出','年度上涨占比','季度上涨占比','新增样本上涨占比','移出样本上涨占比'],[[r.head_cutoff,r.annual_train_n,r.quarter_train_n,r.retained_n,r.added_n,r.removed_n,f'{r.annual_up_fraction:.2%}',f'{r.quarter_up_fraction:.2%}',f'{r.added_up_fraction:.2%}',f'{r.removed_up_fraction:.2%}'] for r in shift[shift.head_cutoff.str.startswith('2026')].itertuples()]),'',
      '23 个截止日的完整汇总见 training_window_changes.csv，每条保留/新增/移出样本见 training_membership_changes.csv。缩放、截尾、监督方向与投影的变化见 transform_changes.csv；各组系数范数和截距见 parameter_changes.csv。不同坐标与不同维数下的范数不能直接横向当作影响大小。','',
      '## 其他年份与两段时期','']
    for period in ['extension_2021_2023','recent_2024_2026']+[f'year_{y}' for y in range(2021,2027)]:
        g=perf[perf.period.eq(period)];parts += [f"### {PL.get(period,period.replace('year_',''))}",'',table(['方法','周数','年度准确率','季度分类端准确率','对→错','错→对','年度 Brier↓','季度分类端 Brier↓'],[[ML[r.method],r.n,f'{r.annual_accuracy:.2%}',f'{r.quarter_head_accuracy:.2%}',r.regression,r.recovery,f'{r.annual_brier:.6f}',f'{r.quarter_head_brier:.6f}'] for r in g.itertuples()]),'']
    parts += ['## 复核与用途边界','',
      f"3,264 组种子周比较、16,320 条分项贡献和 1,088 组融合周比较全部独立复核。标量项差最大 {v['maximum_term_gap']:.3g}，存档概率差最大 {v['maximum_archived_probability_gap']:.3g}，积分分配差最大 {v['maximum_probability_allocation_gap']:.3g}，融合概率求和残差最大 {v['maximum_ensemble_sum_gap']:.3g}。第一季度所有归因项严格为零。",'',
      '九项合成检查覆盖加和、系数/输入拆分、相同模型、只改截距、坐标抵消、零/近零差以及平均概率与平均对数几率不等价。训练窗口和汇总计数均核对源文件，4,895 个旧证据文件保持不变。','',
      '本轮没有新增统计显著性检验。第 32 轮原有 28 项比较完整保留在 baseline_comparisons.json；其中经多重校正没有显著改善或恶化。本轮寻找某项贡献出现在哪些错误周，不会增加独立验证证据。全部结果来自已查看历史，不能据此筛选后再称为新盲测。','',
      '年度基线、R19/R23/R25 交互设计与旧训练重复量 R25 近期 58.4% 的记录继续保留。本轮仅解释预测变化，不自动取消某个特征、改变阈值、冻结部分系数或替换模型。','',
      '![2026 年退步周的概率贡献](probability_contributions_2026.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十三轮归因报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files=[p]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,3,figsize=(17,6),sharey=True);fig.subplots_adjust(left=.06,right=.99,top=.79,bottom=.23,wspace=.12);selected=comp[comp.period.eq('year_2026')&comp.method.isin(PRIMARY)&comp.case.eq('regression')];lo=min(0,float(selected.mean_signed_contribution.min()*100));hi=max(0,float(selected.mean_signed_contribution.max()*100));span=max(1,hi-lo)
    for ax,method in zip(axs,PRIMARY):
        g=selected[selected.method.eq(method)].set_index('group').loc[list(GL)];values=g.mean_signed_contribution.to_numpy()*100;x=np.arange(5);ax.bar(x,values,color=['#c44e52' if v<0 else '#2b927b' for v in values],width=.68);ax.axhline(0,color='#444444',lw=1);ax.set_title(f'{ML[method]}（{int(g.n.iloc[0])} 周）',fontsize=13);ax.set_xticks(x,list(GL.values()),rotation=25);ax.set_ylim(lo-.20*span,hi+.25*span);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        for xx,value in zip(x,values):ax.text(xx,value+(.035*span if value>=0 else -.035*span),f'{value:+.2f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=10)
    axs[0].set_ylabel('平均有符号概率贡献（百分点）');fig.suptitle('2026 年：年度判断正确、季度分类端判断错误的周',fontsize=17,y=.97);fig.text(.5,.88,'负值表示背离实际涨跌；正值表示部分抵消。按每种方法自身的退步周计算均值。',ha='center',fontsize=11);fig.text(.5,.09,'这是选定分解路径上的账目贡献，包含相互抵消；不能直接解释为独立因果效应。',ha='center',fontsize=11);fig.text(.5,.035,'全部历史结果已被查看；改善周、未翻转周及其他年份均完整保留。',ha='center',fontsize=10)
    for ext in ['png','svg']:
        p=OUT/f'probability_contributions_2026.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('Descriptive report and probability-contribution figure generated.',flush=True)
if __name__=='__main__':main()

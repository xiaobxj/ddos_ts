from common26 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
LABELS={'learned_market':'加性模型 R18','learned_vol_interaction':'原波动交互 R19','learned_order_extension':'顺序项联合拟合 R23','learned_order_offset':'固定原模型＋顺序项 R25','native_mse':'原 MSE','training_frequency':'训练期上涨频率'}
WINDOWS={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述：2021–2026 年 8 月'}
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def num(v):return '—' if pd.isna(v) else f'{v:.6f}'
def main():
    check_frozen();assert read(OUT/'verification.json')['status']=='PASS';assert not (OUT/'report_manifest.json').exists();run=manifest('report');m=csv('ensemble_metrics');years=csv('yearly_metrics');seeds=csv('seed_metrics');changes=csv('direction_changes');pairs=read(OUT/'primary_comparisons.json');fit=read(OUT/'training_manifest.json');v=read(OUT/'verification.json')
    parts=['# 第二十六轮：2021–2026 年历史扩展测试','',f'生成时间：{now()}。全部 18 次神经网络训练、72 次分类拟合及 272 周评分已经完成，独立复核通过。','',
        '本轮沿用此前明确的三个学习分支，重新训练六个年度的完整上游模型。先完成全部训练，再统一评分，没有根据某一年的新结果调整后续设置。第 1–3 轮已查看全部 272 个信号；因此本轮属于当前配方的历史扩展检验，不是研究层面的全新独立盲测。','',
        '## 主要结果','']
    for w in cfg()['windows']:
        g=m[m.window.eq(w['name'])].set_index('method');parts += [f"### {WINDOWS[w['name']]}（{w['n']} 周）",'',table(['方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[LABELS[method],f"{int(g.loc[method,'correct_directions'])}/{int(g.loc[method,'n'])}",f"{g.loc[method,'accuracy']:.2%}",f"{g.loc[method,'balanced_accuracy']:.2%}",num(g.loc[method,'auroc']),num(g.loc[method,'brier']),num(g.loc[method,'log_loss'])] for method in METHODS]),'']
        for method in ['learned_order_extension','learned_order_offset']:
            delta=int(g.loc[method,'correct_directions']-g.loc['learned_vol_interaction','correct_directions']);bd=g.loc[method,'brier']-g.loc['learned_vol_interaction','brier'];parts += [f"{LABELS[method]}相对原波动交互：正确方向变化 {delta:+d} 周，Brier 变化 {bd:+.6f}。",'']
    sig=[p for p in pairs if p['holm_adjusted_p']<.05];parts += [f'24 项预设配对比较统一进行 Holm 校正，校正后 p<0.05 的比较共 {len(sig)} 项。显著性只覆盖本轮预设比较，不修正多轮查看历史和模型选择的影响。', '']
    if sig:parts += [table(['时期','候选 / 参照','损失','差值（负值更好）','校正 p'],[[WINDOWS[p['window']],LABELS[p['candidate']]+' / '+LABELS[p['reference']],p['metric'],num(p['difference']),num(p['holm_adjusted_p'])] for p in sig]),'']
    parts += ['## 各年度与随机种子','',table(['年份','周数']+[LABELS[x] for x in METHODS],[[str(year),int(g.n.iloc[0])]+[f"{g.set_index('method').loc[x,'accuracy']:.2%}" for x in METHODS] for year,g in years.groupby('year')]),'',
        '完整年度指标见 yearly_metrics.csv；各随机种子及其年度结果见 seed_metrics.csv、seed_yearly_metrics.csv。下表给出各方法三种子准确率的最小值与最大值；范围不构成置信区间。','',
        table(['时期','方法','三种子准确率范围'],[[WINDOWS[w],LABELS[method],f'{g.accuracy.min():.2%}–{g.accuracy.max():.2%}'] for (w,method),g in seeds[seeds.window.ne('pooled_2021_2026')].groupby(['window','method'])]),'',
        '## 改动影响了哪些信号','',table(['时期','候选 / 参照','方向改变','错→对','对→错'],[[WINDOWS[r.window],LABELS[r.method]+' / '+LABELS[r.reference],r.changed,r.wrong_to_correct,r.correct_to_wrong] for r in changes.itertuples()]),'',
        '逐条日期和两侧概率保存在 changed_signal_details.csv。方向相同不代表概率相同；须同时检查 Brier 和对数损失。','',
        '## 固定配方与时间边界','',
        '- 六个预测年分别以 2020–2025 年底为训练截止日，训练必须满足 joint_completed≤截止日。标签未成熟的年末信号不会提前进入训练。',
        '- 原始数据 2010-01-04 至 2026-08-31，共 4,046 根日线。当前最后可完整评分信号为 2026-08-21。2026 年只有前八个月，不能视作完整年度。',
        '- 18 个全新 combined MSE20 Crossformer，每个 38,551 参数，三个固定种子；20 轮、batch 128、AdamW 学习率 0.001、weight decay 0.1、dropout 0.3、梯度裁剪 1。损失为标准化收益 MSE＋0.1×未来五根 OHLCV 辅助 MSE。',
        '- 输入为沿用的原始 OHLCV 125 日窗口，25×5 固定分段，无预训练 tokenizer。每个窗口仅用自身历史归一化；标签尺度只由该折成熟训练样本得到。',
        '- 25 维学习特征及四个市场描述量分别用训练期 1%/99% 分位截尾，再按训练均值/总体标准差归一化。',
        '- R18 加性分类系数的前 25 维定义监督方向。R19 增加波动×该方向的一项训练残差；R23 再增加 60 日符号顺序×同一方向的一项训练残差，并联合拟合。R25 固定 R19 全部系数和截距，只拟合与 R23 相同的顺序特征系数。',
        '- 所有逻辑回归斜率惩罚 λ=0.01，截距不惩罚；R25 只惩罚新增标量。没有超参数、阈值、窗口或种子搜索，没有状态择时或模型切换。',
        '- 分类模型按三个种子的概率平均，再使用严格 p>0.5；原 MSE 平均原始收益预测再判正负，未人为转换为概率。训练上涨频率由同一成熟样本得到。',
        '- 神经表征、监督方向、投影均在各折训练期内拟合，训练指标属于样本内诊断；本轮不声称完整上游的交叉拟合或独立确认。','',
        '## 统计口径','',
        '分别检验 2021–2023（147 周）和 2024–2026 年 8 月（125 周）。每段固定 12 个配对比较，总共 24 项统一 Holm 校正。采用 10,000 次循环移动区块重采样，每块连续 8 个有效周观察，种子 20260910；报告候选减参照的损失差值、边际 95% 区间和中心化双侧 p。合并 272 周只作描述，随机种子不当作独立周样本。','',
        table(['时期','候选 / 参照','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WINDOWS[p['window']],LABELS[p['candidate']]+' / '+LABELS[p['reference']],p['metric'],num(p['difference']),f"[{p['ci95_low']:+.6f}, {p['ci95_high']:+.6f}]",num(p['p']),num(p['holm_adjusted_p'])] for p in pairs]),'',
        '## 核验与保留结论','',
        f"训练共 {fit['optimizer_steps']} 个神经网络更新，{fit['head_fits']} 次分类拟合、36 次训练期交互投影；训练耗时 {fit['elapsed_seconds']/60:.2f} 分钟。72 个分类解由独立 L-BFGS-B / 标量括区间求根核验，18 个检查点全部回放。",'',
        f"原始窗口与标签全量重算；训练与测试特征按保存的检查点逐一复现。独立代数预测最大差异 {v['maximum_forecast_gap']:.3g}；{v['old_files_preserved']:,} 份既有证据文件哈希保持一致（含首次预检查归档）。协议 SHA-256：`{sha(ROOT/'protocol.json')}`。",'',
        '首次预检查因新校验误用零舍入容差而停止，尚未进行新年度训练或评分。原数据下载规则允许 0.011 点 OHLC 舍入误差。本轮恢复原有校验口径，保留所有样本与模型设置；首次尝试的 24 份文件及保全记录位于 research_v26_attempt1。', '',
        '本轮不自动替换原模型。此前 2018–2020 年原波动交互 58.16%、R25 为 57.45%，以及 2015–2017 年 R25 为 51.67% 的记录继续保留；本轮扩展结果用于判断这些改进是否跨时期延续，不能抹去旧区间的得失。', '',
        '下一步先结合年度、种子和逐条信号差异判断稳定性。市场状态的条件化使用需要另立固定实验，训练与选择只能使用当时已经成熟的数据。真正的前向验证应在目标结果发生前带时间戳保存预测，等待标签成熟后评分。预测准确率不直接代表扣除交易成本后的盈利。','',
        '![年度方向与概率误差](annual_extension.png)','',
        '核心文件：protocol.json、membership.csv、models.json、heads.json、model_predictions.csv、ensemble_predictions.csv、primary_comparisons.json、verification.json。']
    text='\n'.join(parts);path=OUT/'第二十六轮测试报告.md';path.write_text(text,encoding='utf-8')
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,1,figsize=(12,9),layout='constrained');colors=['#8a929c','#996313','#117c91','#ac4670','#593ba0','#333333']
    for method,color in zip(METHODS,colors):
        g=years[years.method.eq(method)].sort_values('year');axs[0].plot(g.year,g.accuracy*100,'o-',color=color,label=LABELS[method],lw=1.6)
        if method!='native_mse':axs[1].plot(g.year,g.brier,'o-',color=color,label=LABELS[method],lw=1.6)
    axs[0].set_ylabel('方向准确率（%）');axs[0].axhline(50,color='#ccc',ls='--');axs[0].set_title('固定模型配方，按年度训练：2021–2026 年历史扩展');axs[0].legend(ncol=3,fontsize=9)
    axs[1].set_ylabel('Brier（越低越好）');axs[1].set_xlabel('信号年份；2026 年仅截至 8 月');axs[1].axhline(.25,color='#ccc',ls='--')
    for ax in axs:ax.set_xticks(range(2021,2027));ax.grid(alpha=.2)
    fig.savefig(OUT/'annual_extension.png',dpi=170);fig.savefig(OUT/'annual_extension.svg');plt.close(fig)
    facts=dict(windows=m.to_dict('records'),yearly=years.to_dict('records'),significant_comparisons=sig,independent_confirmation=False,automatic_promotion=False,verification_status='PASS');save(OUT/'report_facts.json',facts)
    finish(run,[path,OUT/'annual_extension.png',OUT/'annual_extension.svg',OUT/'report_facts.json'],report_status='READY_FOR_VISUAL_REVIEW')
    print('Report and figures generated.',flush=True)
if __name__=='__main__':main()

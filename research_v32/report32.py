"""Standalone Chinese report for annual representation / quarterly head split."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
HL={'rolling5_matched':'年度／旧重复量','rolling5_annual20':'整套年度／自然 20 遍','rolling5_quarterly20':'整套季度／自然 20 遍','state90':'市场状态触发','error16':'成熟误差触发','shadow_trial':'新旧试运行后切换','annual_quarter_half':'年度与季度各半','annual_net_quarter_head':'年度网络＋季度分类端'}
ML={'learned_market':'加性 R18','learned_vol_interaction':'波动交互 R19','learned_order_extension':'联合顺序项 R23','learned_order_offset':'固定 R19＋顺序项 R25','native_mse':'原 MSE','training_frequency':'训练上涨频率'}
WL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'}
PRIMARY=list(ML)[1:4];CANDIDATE='annual_net_quarter_head'
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
        for n,d in p[key].items():assert sha(PROJECT/n)==d
    for phase in ['extraction','fitting','scoring','evaluation']:
        for n,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    assert read(OUT/'verification.json')['status']=='PASS';return p
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');m=csv('ensemble_metrics');yr=csv('yearly_metrics');jobs=csv('jobs');banks=read(OUT/'feature_banks.json');budgets=csv('budgets');pairs=read(OUT/'primary_comparisons.json');v=read(OUT/'verification.json');sig=[r for r in pairs if r['holm_adjusted_p']<.05]
    parts=['# 第三十二轮：年度神经网络固定，季度重拟合分类端','',
      '本轮实际新增 51 组季度分类端训练，共 204 个分类头和 102 个交互残差投影；18 个年度神经网络保持不变，没有新增神经网络训练。年末 72 个分类头及其变换直接复用旧证据。训练、评分和独立复核全部完成。','',
      '这是一项对更新层次的拆分研究：与整套年度更新相比，增加季度分类端适应；与整套季度更新相比，保留年度神经网络表征。全部 272 周历史结果此前已被查看，本轮不是新盲测。','',
      '## 固定规则与可解释范围','',
      '每个信号年固定使用上一年 12 月末训练的五年窗口、自然 20 遍神经网络，网络权重、归一化目标尺度和推理模式不变。每个非年末季末，仅对当时五个自然年锚点窗口内、全部标签已成熟的日观察重拟合分类端。每个种子的神经网络和分类端保持对应，最终概率仍为三种子等权平均。','',
      '“分类端”包括 25 个已解码特征与 4 个市场描述的 1/99 截尾和标准化、R18 的监督方向、波动与顺序交互的残差投影、R18/R19/R23 的系数，以及 R25 的单一顺序项系数。R25 保留的是该季度新拟合的 R19 系数，并非一直保留年初 R19。正则化、特征定义和拟合目标保持原配方，没有搜索新的训练窗口、阈值、种子或融合权重。','',
      '因此本轮检验整个分类端更新的效果，不能将差异单独归因于某一个系数、缩放或投影。神经特征对部分分类训练样本是训练内表征，与原流程相同；本轮不声称整条流程使用了折外特征。','',
      '年末的分类端直接复用年度模型，因此每年第一季度必须与年度方案逐周相同。季度更新只作用于检查日之后的信号，季度末当天仍使用上一季度分类端。标签成熟沿用 joint_completed 口径，收益和辅助目标都必须完成。训练包括所有合格工作日观察，测试仍为相同的周五信号。','',
      '原 MSE 对照只使用冻结年度网络的原始收益预测，逐周等于年度对照且没有构造概率。训练频率对照随季度训练样本更新，逐周等于整套季度方案的频率预测。routing.csv 分别记录网络与分类截止日；预测表 cutoff 在学习方法中表示分类端截止日，在原 MSE 中表示年度网络截止日。','',
      '## 特征来源与训练顺序','',
      f"每个年度网络保留其已有训练特征和该年周测试特征的原值；只为缺失的日训练观察补算特征。18 个网络共补算 {sum(b['new_rows'] for b in banks)} 条特征，建立 {sum(b['bank_rows'] for b in banks)} 条含来源标记的网络特征记录。",'',
      '特征库包含为计算复用准备的后续样本，不代表当前分类器可用它们。每个拟合接口仅接收本季度成熟训练成员的特征、市场描述、顺序描述和涨跌标签；全部 51 个接口都通过了库内非训练行及未成熟标签扰动检查。来源 0 表示旧年度训练缓存，1 表示旧年度测试缓存，2 表示本轮新推理。缓存是计算角色，不是允许提前使用标签的角色。','',
      '先冻结协议、源码、输入和 4,759 个旧证据文件，再检查样本和边界、补齐特征并冻结特征库；随后完成并冻结全部 204 个新分类头，最后进行周评分及统计比较。训练期间没有用周测试表现选择模型。','',
      '## 每个季度的样本及来源','',table(['分类截止日','固定网络截止日','日训练样本','随后周测试数','分类端来源'],[[r.head_cutoff,r.encoder_cutoff,r.train_n,r.test_n,'本轮重拟合' if r.refit else '旧年度复用'] for r in jobs.itertuples()]),'',
      '五年窗口约束的是训练样本锚点，125 根日线的输入上下文可以早于该窗口下界；这与已有对照一致。','',
      '## 逐期运行预算','',table(['政策','参数更新日期数','神经网络训练数','分类头拟合数','神经优化器更新数'],[[HL[r.history],r.refit_dates,r.neural_fits_if_run_online,r.head_fits_if_run_online,r.optimizer_steps_if_run_online] for r in budgets.itertuples()]),'',
      '本轮候选逐期运行时需要 6 个年度网络截止日、三种子共 18 个网络，分类端则在 23 个截止日共拟合 276 个头。神经优化器更新 3,600 次，与年度方案相同，少于整套季度的 13,800 次；分类端拟合次数与季度方案相同。特征提取与分类拟合也有成本，这个表不是总耗时或总算力的等价比较。本轮实际仅新增 204 个分类头，未重训这 18 个神经网络。','']
    for w in cfg['windows']:
        g=m[m.window.eq(w['name'])].set_index(['history','method']);parts += [f"## {WL[w['name']]}（{w['n']} 周）",'',table(['政策','方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[HL[h],ML[method],f"{int(g.loc[(h,method),'correct_directions'])}/{w['n']}",f"{g.loc[(h,method),'accuracy']:.2%}",f"{g.loc[(h,method),'balanced_accuracy']:.2%}",num(g.loc[(h,method),'auroc']),num(g.loc[(h,method),'brier']),num(g.loc[(h,method),'log_loss'])] for h in HL for method in ML]),'']
    parts += ['## 28 项预设比较','',f"两个时期内，三个主配方分别相对整套年度及整套季度比较方向误差和 Brier，并加入 R19 相对本政策频率 Brier 及原 MSE 方向误差，共 28 项。统一 Holm 校正后，显著改善 {sum(r['difference']<0 for r in sig)} 项、显著恶化 {sum(r['difference']>0 for r in sig)} 项。",'',table(['时期','方法','参照政策/方法','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WL[r['window']],ML[r['candidate']],HL[r['reference_history']]+'/'+ML[r['reference']],r['metric'],num(r['difference']),f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
      '负损失差表示本轮候选更好。每段采用 10,000 次循环移动区块重采样，每块 8 个有效周观察、种子 20260910；p 为中心化双侧值，区间为边际 95% 区间。校正只覆盖本轮，不能消除跨轮历史试验的选择影响。其他七项旧政策、全部方法、年度和种子均保留，合并 272 周只作补充描述。','']
    for method in PRIMARY:
        parts += [f'## 逐年：{ML[method]}','',table(['年份','周数']+[HL[h] for h in ['rolling5_matched','rolling5_annual20','rolling5_quarterly20',CANDIDATE]],[[year,int(g.n.iloc[0])]+[f"{g.set_index('history').loc[h,'accuracy']:.2%}" for h in ['rolling5_matched','rolling5_annual20','rolling5_quarterly20',CANDIDATE]] for year,g in yr[yr.method.eq(method)].groupby('year')]),'']
    sd=csv('seed_fusion_diagnostics');parts += ['## 种子稳定性','',table(['时期','政策','方法','三种子准确率范围','融合准确率'],[[WL[r.window],HL[r.history],ML[r.method],f'{r.seed_accuracy_min:.2%}–{r.seed_accuracy_max:.2%}',f'{r.ensemble_accuracy:.2%}'] for r in sd[sd.window.ne('pooled_2021_2026')&sd.history.isin(['rolling5_annual20','rolling5_quarterly20',CANDIDATE])&sd.method.isin(PRIMARY)].itertuples()]),'',
      '全部逐年与逐种子结果、方向改变及概率分箱见对应 CSV。没有改用最好种子或另一套概率融合规则。','',
      '## 独立复核','',
      f"18 组缺失日特征推理按位复现，旧训练/测试特征和网络状态保持不变。51 组训练输入均通过因果成员检查；204 个分类头分别用独立 L-BFGS-B 或 Brent 求根验证，截尾、标准化、监督方向和交互残差投影也逐项重建。测试特征代数最大差 {v['maximum_feature_algebra_gap']:.3g}、概率最大差 {v['maximum_probability_gap']:.3g}。",'',
      '六个年份的第一季度预测逐周等于年度方案；原 MSE 全段逐周等于年度方案，频率基准全段逐周等于季度方案。全部指标、28 项区块重采样及 Holm 校正由独立算术复核，4,759 个旧证据文件保持不变。','',
      '计算与因果顺序通过不等于预测优势或交易收益成立。所有历史周已被查看，当前拆分假设又来自此前年度与季度表现的差异，因此还需要与真正后续未查看数据区别对待。本轮不自动替换基线，旧 R25 近期 58.4% 的局部结果及原交互设计继续保留。','',
      '![年度网络与季度分类端的逐年表现](annual_net_quarter_head_yearly.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十二轮测试报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files=[p]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,3,figsize=(17,6),sharey=True);fig.subplots_adjust(left=.055,right=.99,top=.82,bottom=.23,wspace=.12);policies=['rolling5_annual20','rolling5_quarterly20',CANDIDATE];colors=['#4c78a8','#999999','#c44e52'];styles=['-','--','-'];markers=['o','s','^'];years=sorted(yr.year.unique())
    for ax,method in zip(axs,PRIMARY):
        for h,color,style,marker in zip(policies,colors,styles,markers):
            g=yr[yr.history.eq(h)&yr.method.eq(method)].set_index('year').loc[years];ax.plot(np.arange(6),g.accuracy,color=color,ls=style,marker=marker,lw=2,label=HL[h])
        ax.set_title(ML[method],fontsize=13);ax.set_xticks(np.arange(6),['2021','2022','2023','2024','2025','2026*'],rotation=35);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False,fontsize=11);fig.suptitle('5 年窗口：固定年度神经网络，只在季度更新分类端',fontsize=17,y=.965);fig.text(.5,.035,'同一批历史周；三种子等权。*2026 年截至 8 月（30 周）；其余年份 47–50 周。',ha='center',fontsize=10)
    for ext in ['png','svg']:
        p=OUT/f'annual_net_quarter_head_yearly.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('Report and figure generated; visual review pending.',flush=True)
if __name__=='__main__':main()

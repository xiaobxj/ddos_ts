"""Standalone Chinese report, with every frozen step and historical comparator."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
HL={'rolling5_matched':'年度／旧重复量','rolling5_annual20':'年度基线','rolling5_quarterly20':'整套季度','state90':'市场状态触发','error16':'成熟误差触发','shadow_trial':'新旧试运行切换','annual_quarter_half':'年度与季度概率各半','annual_net_quarter_head':'R32 季度分类端','fixed_transform_step25':'固定变换／25% 更新','fixed_transform_step50':'固定变换／50% 更新','fixed_transform_step100':'固定变换／100% 更新'}
ML={'learned_market':'加性 R18','learned_vol_interaction':'波动交互 R19','learned_order_extension':'联合顺序项 R23','learned_order_offset':'固定 R19＋顺序项 R25','native_mse':'原 MSE','training_frequency':'训练上涨频率'}
WL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'}
PRIMARY=list(ML)[1:4];NEW=list(HL)[-3:];FOCUS=['rolling5_annual20','annual_net_quarter_head']+NEW
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
    for phase in ['fitting','scoring','evaluation']:
        for n,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/n)==d
    assert read(OUT/'verification.json')['status']=='PASS';return p
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');m=csv('ensemble_metrics');yr=csv('yearly_metrics');pairs=read(OUT/'primary_comparisons.json');v=read(OUT/'verification.json');sig=[r for r in pairs if r['holm_adjusted_p']<.05];e=csv('ensemble_predictions');flips=[]
    for history in NEW:
        for method in list(ML)[:4]:
            a=e[e.history.eq(history)&e.method.eq(method)].sort_values('date').reset_index(drop=True);b=e[e.history.eq('rolling5_annual20')&e.method.eq(method)].sort_values('date').reset_index(drop=True);assert a.date.tolist()==b.date.tolist()
            for i in range(len(a)):
                gooda=a.direction_up.iloc[i]==a.actual_up.iloc[i];goodb=b.direction_up.iloc[i]==b.actual_up.iloc[i]
                if gooda!=goodb:flips.append(dict(history=history,method=method,date=a.date.iloc[i],year=int(a.year.iloc[i]),actual_up=int(a.actual_up.iloc[i]),annual_probability=float(b.probability.iloc[i]),candidate_probability=float(a.probability.iloc[i]),case='recovery' if gooda else 'regression'))
    flip=pd.DataFrame(flips);p=OUT/'direction_flips.csv';flip.to_csv(p,index=False);files=[p]
    parts=['# 第三十四轮：固定年度变换，限制季度系数更新','',
      '本轮完成 51 组季度与种子组合的训练，新增 204 个固定坐标下的分类拟合端点，生成 612 个政策系数向量。18 个年度神经网络、年度缩放和交互投影、原始特征缓存保持不变；没有新神经训练或特征推理。主候选 25%，敏感性 50%，固定变换完整系数更新对照 100%，三者均在看到本轮结果之前冻结。','',
      '## 更新规则','',
      '每个季度使用五个自然年锚点窗口内 joint_completed 不晚于截止日的全部日训练样本。年度神经网络、神经目标尺度、25 个解码特征与 4 个市场描述的截尾和标准化、R18 监督方向、波动与顺序交互的投影及残差尺度，全年保持年度值。新的季度 R18 系数不用于重建交互方向；年度交互在季度样本上不再要求正交，也不重新残差化。','',
      '在固定年度坐标上，沿用原 lambda=0.01、斜率惩罚而截距不惩罚的逻辑回归，重拟合 R18、R19、R23。R25 在该季度完整更新的 R19 上拟合单一顺序系数 gamma。然后按 θ(α)=θ年度+α(θ季度完整−θ年度) 更新所有系数，包括截距。每个季度都以同一个年度模型为起点，下一年重置年度起点。','',
      '25% 和 50% 是拟合完成后的系数插值，不是分别求解的受约束逻辑回归最优解；系数位移范数在同一坐标下严格缩小到完整位移的对应比例。每个种子的预测等价于先插值新旧 logit，再转换成概率；最后对三个种子概率等权平均。它与新旧概率平均、先平均种子 logit 均不同。','',
      '每个更新幅度下，R25 非顺序系数及截距严格等于相同幅度的 R19；gamma 也按该幅度插值，没有在插值后的 R19 上再次求最优 gamma。所有交互结构继续保留。年度基线相当于 α=0；第一季度直接复用年度预测。原 MSE 逐周等于年度基线，训练频率逐周等于完整季度方案。','',
      '## 研究口径','',
      '同一批 272 个周信号，2021–2023 为 147 周，2024–2026 年 8 月为 125 周。信号日之后的执行周收益方向仍用原定义，严格概率 >0.5 才判涨；种子固定为 20260910、20260911、20260912。全部历史此前已查看，这项实验受到先前归因结果启发，不是新的盲测。','',
      '25% 与年度比较检查有限适应，25% 与固定变换 100% 比较检查同一坐标下缩小系数更新的影响；固定变换 100% 与 R32 的差别包括年度截尾、缩放、监督方向及交互投影的整体冻结。不能将结果单独归因于某个市场或顺序系数。主候选始终是预定的 25%，不按本轮结果另选幅度。','']
    for w in cfg['windows']:
        g=m[m.window.eq(w['name'])].set_index(['history','method'])
        parts += [f"## {WL[w['name']]}（{w['n']} 周）",'',table(['政策','方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[HL[h],ML[method],f"{int(g.loc[(h,method),'correct_directions'])}/{w['n']}",f"{g.loc[(h,method),'accuracy']:.2%}",f"{g.loc[(h,method),'balanced_accuracy']:.2%}",num(g.loc[(h,method),'auroc']),num(g.loc[(h,method),'brier']),num(g.loc[(h,method),'log_loss'])] for h in HL for method in ML]),'']
    for method in PRIMARY:
        g=yr[yr.method.eq(method)];parts += [f'## 逐年：{ML[method]}','',table(['年份','周数']+[HL[h] for h in FOCUS],[[int(year),int(s.n.iloc[0])]+[f"{s.set_index('history').loc[h,'accuracy']:.2%}" for h in FOCUS] for year,s in g.groupby('year')]),'']
    parts += ['## 2026 年相对年度的方向翻转','',table(['政策','方法','从对变错','从错变对'],[[HL[h],ML[method],int(((flip.history==h)&(flip.method==method)&(flip.year==2026)&(flip['case']=='regression')).sum()),int(((flip.history==h)&(flip.method==method)&(flip.year==2026)&(flip['case']=='recovery')).sum())] for h in NEW for method in PRIMARY]),'',
      '完整各年翻转见 direction_flips.csv，不仅记录先前 7 个退步周。2026 年只有 30 个周信号，1 周对应约 3.33 个准确率百分点。','',
      '## 60 项预设比较','',f"两个时期内，三个幅度、三个主方法的方向误差和 Brier 分别对年度比较；另加主候选 25% 相对固定变换 100% 和 R32 的比较，共 60 项统一 Holm 校正。显著改善 {sum(r['difference']<0 for r in sig)} 项、显著恶化 {sum(r['difference']>0 for r in sig)} 项。",'',
      table(['时期','候选','方法','参照','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WL[r['window']],HL[r['history']],ML[r['candidate']],HL[r['reference_history']],r['metric'],num(r['difference']),f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
      '负损失差表示候选更好。沿用 10,000 次循环区块重采样，每块 8 个有效周观察，种子 20260910；中心化双侧 p 和边际 95% 区间。校正仅覆盖本轮 60 项，不能消除跨轮反复查看历史的影响。原 R32 的 28 项比较原样保存在 baseline_comparisons.json。','']
    sd=csv('seed_fusion_diagnostics');parts += ['## 种子稳定性','',table(['时期','政策','方法','三种子准确率范围','融合准确率'],[[WL[r.window],HL[r.history],ML[r.method],f'{r.seed_accuracy_min:.2%}–{r.seed_accuracy_max:.2%}',f'{r.ensemble_accuracy:.2%}'] for r in sd[sd.window.ne('pooled_2021_2026')&sd.history.isin(FOCUS)&sd.method.isin(PRIMARY)].itertuples()]),'',
      '全部方法、种子、年份、方向翻转、概率分箱和系数位移见相应 CSV。没有改用最佳种子或按年份挑选幅度。','',
      '## 独立复核与限制','',
      f"204 个拟合端点通过独立 L-BFGS-B 或 Brent 求根；612 个系数向量逐项检查年度坐标来源、更新比例和 R25 对 R19 的约束。51 个拟合接口全部通过非训练行的特征、市场描述、顺序描述及标签扰动检查。变换代数最大差 {v['maximum_feature_algebra_gap']:.3g}，测试概率最大差 {v['maximum_probability_gap']:.3g}，年度起点概率最大差 {v['maximum_annual_anchor_probability_gap']:.3g}。",'',
      '六年第一季度逐周等于年度，原 MSE 与频率基准符合原定义，全部指标及 60 项区块重采样与校正均由独立算术复核。4,946 个旧证据文件未变。没有重新执行已通过复核的旧神经网络推理；本轮通过来源哈希继承该证据，并重建固定变换后的全部训练和测试设计。','',
      '较小参数更新不保证更好预测，年度模型本身也会判断错误。部分神经特征对分类训练样本仍为训练内表征，不能声称整条流程折外。旧重复量 R25 近期 73/125、58.4% 的记录继续保留；本轮不自动替换基线或声称交易收益改善。','',
      '![各年固定变换与不同系数更新幅度](fixed_transform_steps_yearly.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    p=OUT/'第三十四轮测试报告.md';p.write_text('\n'.join(parts),encoding='utf-8');files.append(p)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,3,figsize=(18,6.5),sharey=True);fig.subplots_adjust(left=.05,right=.99,top=.83,bottom=.26,wspace=.13)
    colors=['#333333','#bb5566','#0077bb','#33aa88','#ee9933'];styles=['-','--','-','--',':'];markers=['o','v','s','^','D'];years=sorted(yr.year.unique())
    for ax,method in zip(axs,PRIMARY):
        for h,col,style,mk in zip(FOCUS,colors,styles,markers):
            g=yr[yr.history.eq(h)&yr.method.eq(method)].set_index('year').loc[years];ax.plot(np.arange(6),g.accuracy,color=col,ls=style,marker=mk,lw=2,label=HL[h])
        ax.set_title(ML[method],fontsize=13);ax.set_xticks(np.arange(6),['2021','2022','2023','2024','2025','2026*'],rotation=30);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('方向准确率');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=11)
    fig.suptitle('固定年度变换：季度系数更新幅度 25% / 50% / 100%',fontsize=18,y=.97);fig.text(.5,.025,'同一批已查看历史；三种子概率等权。25% 为预定主候选。*2026 年截至 8 月，共 30 周。',ha='center',fontsize=10)
    for ext in ['png','svg']:p=OUT/f'fixed_transform_steps_yearly.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);(OUT/'report_manifest.json').write_text(json.dumps(dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2,ensure_ascii=False),encoding='utf-8');print('Chinese report and figure generated; visual review pending.',flush=True)
if __name__=='__main__':main()

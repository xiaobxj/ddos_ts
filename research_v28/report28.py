"""Read-only reporting with the existing Python314 plotting installation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
HL={'full':'全历史／年度','rolling5_matched':'5 年／年度／原重复量','rolling5_annual20':'5 年／年度／自然 20 遍','rolling5_quarterly20':'5 年／季度／自然 20 遍','rolling3_matched':'3 年／年度／原重复量','rolling3_annual20':'3 年／年度／自然 20 遍','rolling3_quarterly20':'3 年／季度／自然 20 遍'}
HISTORIES=list(HL)
LABELS={'learned_market':'加性 R18','learned_vol_interaction':'波动交互 R19','learned_order_extension':'联合顺序项 R23','learned_order_offset':'固定原模型＋顺序项 R25','native_mse':'原 MSE','training_frequency':'训练上涨频率'}
WL={'extension_2021_2023':'2021–2023','recent_2024_2026':'2024–2026 年 8 月','pooled_2021_2026':'合并描述'}
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
    for phase in ['training','scoring','evaluation']:
        for name,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/name)==d,name
    assert read(OUT/'verification.json')['status']=='PASS';return prep
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');m=csv('ensemble_metrics');yr=csv('yearly_metrics');pairs=read(OUT/'primary_comparisons.json');fit=read(OUT/'training_manifest.json');v=read(OUT/'verification.json');b=cfg['budget'];sig=[r for r in pairs if r['holm_adjusted_p']<.05]
    parts=['# 第二十八轮：固定窗口与季度 walk-forward','',
        '本轮完成最近 3 年、5 年固定窗口的自然 20 遍训练，并在同一批 272 个周信号上比较年度与季度重训。所有历史数据已在前轮查看，属于历史研究复核。','',
        '**原流程已经是年度 walk-forward。** 本轮检验两个具体变化：减少短窗口内的重复训练，以及缩短模型使用到下一次重训的时间。不能仅凭最近年份准确率下降就认定必须滚动，也不能把训练样本内损失和未来方向准确率混为一谈。','',
        '## 预先固定的实验','',
        '第一组：自然 20 遍的年度模型，对照第 27 轮相同窗口、相同截止日的重复训练模型。第二组：自然 20 遍的季度模型，对照同窗口自然 20 遍年度模型。全历史年度模型和第 27 轮两个短窗口对照原样复用。','',
        '每个截止日使用之前 3 或 5 个日历年的信号，并要求执行收益及辅助标签全部成熟。起始边界为“截止日减 K 年再加一天”；输入 125 根历史柱可早于训练信号窗口。年度模型在上一年末重训；季度模型在上一季末重训。遇到季末当天的信号，使用该信号之前的上一季模型。','',
        '每个新模型都从固定随机种子重新初始化，20 次完整随机排列，每条保留样本每遍恰好使用一次。网络结构、损失、学习率、正则、4 类市场描述、波动与顺序交互、分类头和三种子等权概率融合保持原配方。每次重训同时重新拟合目标尺度、表征、截尾和标准化、交互投影及分类系数。没有早停、窗口择优、种子择优、校准器或新增市场分类器。','',
        '季度更新改变了模型年龄及新旧数据的组成，同时增加全年计算量；因此比较的是整套更新政策。共同的年末模型只训练一次，推理按该模型所需日期并集计算一次，再把同一预测分配到两种日历。输入变换不在测试批次上拟合。','',
        table(['项目','数量'],[['独立截止日',23],['新网络',b['new_neural_fits']],['其中与年度方案共享的网络',36],['新分类解',b['new_head_fits']],['交互投影',b['new_projections']],['自然训练遍数总计',b['new_epochs']],['优化器更新',b['new_optimizer_steps']],['样本呈现总数',b['new_sample_presentations']],['新模型推理去重后的记录',b['unique_model_prediction_rows']],['路由后全部模型记录',30464],['三种子融合及频率记录',11424],['保留旧证据文件',4002]]),'',
        f"神经训练及分类拟合耗时 {fit['elapsed_seconds']/60:.2f} 分钟。138 个网络和 552 个分类解全部完成后，才开始本轮新周信号评分。训练窗口、逐周模型选择和实际预算分别见 membership.csv、routing.csv、budgets.csv。",'']
    for w in cfg['windows']:
        g=m[m.window.eq(w['name'])].set_index(['history','method']);parts += [f"## {WL[w['name']]}（{w['n']} 周）",'',table(['训练及更新','方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[HL[h],LABELS[method],f"{int(g.loc[(h,method),'correct_directions'])}/{w['n']}",f"{g.loc[(h,method),'accuracy']:.2%}",f"{g.loc[(h,method),'balanced_accuracy']:.2%}",num(g.loc[(h,method),'auroc']),num(g.loc[(h,method),'brier']),num(g.loc[(h,method),'log_loss'])] for h in HISTORIES for method in LABELS]),'']
    parts+=['## 预设配对比较','',f"64 项比较统一 Holm 校正，p<0.05 的改善 {sum(r['difference']<0 for r in sig)} 项、恶化 {sum(r['difference']>0 for r in sig)} 项。负损失差表示候选更好。每段采用固定种子的 10,000 次循环区块重采样，每块 8 个有效周观察；区间为边际 95% 区间，p 为中心化双侧值。此校正仅覆盖本轮，不能消除多轮查看历史的影响。",'',table(['时期','干预','候选','参照','方法/参照方法','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WL[r['window']],r['intervention'],HL[r['history']],HL[r['reference_history']],LABELS[r['candidate']]+'/'+LABELS[r['reference']],r['metric'],num(r['difference']),f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'']
    for method in ['learned_vol_interaction','learned_order_extension','learned_order_offset']:
        parts += [f'## 各年度：{LABELS[method]}','',table(['信号年','周数']+[HL[h] for h in HISTORIES],[[year,int(g.n.iloc[0])]+[f"{g.set_index('history').loc[h,'accuracy']:.2%}" for h in HISTORIES] for year,g in yr[yr.method.eq(method)].groupby('year')]),'']
    diag=csv('seed_fusion_diagnostics');parts+=['## 种子稳定性与复核','',table(['时期','更新政策','方法','三个种子准确率范围','融合准确率','方向分歧周数'],[[WL[r.window],HL[r.history],LABELS[r.method],f'{r.seed_accuracy_min:.2%}–{r.seed_accuracy_max:.2%}',f'{r.ensemble_accuracy:.2%}',r.disagreeing_weeks] for r in diag[diag.window.ne('pooled_2021_2026')&diag.method.eq('learned_vol_interaction')].itertuples()]),'',
        '全部方法和种子的结果见 seed_metrics.csv、seed_yearly_metrics.csv。逐周概率分歧及平方损失的方差恒等式已复核，仅作诊断，没有改成多数票或选择最好种子。', '',
        f"复核通过：{v['checkpoint_replays']} 个检查点及其训练/测试特征复现，{v['independent_head_solutions']} 个独立凸优化解比较；特征代数最大差 {v['maximum_feature_algebra_gap']:.3g}，概率最大差 {v['maximum_forecast_gap']:.3g}。训练样本、成熟边界、每遍排列、逐周模型选择、所有指标和比较均已核对。",'',
        '输入原始 OHLCV 与标签沿用第 26 轮逐项重构的冻结文件，包括 0.011 点舍入容差；本轮核验输入哈希。检查通过说明结果可复现，不等于预测有效或可以交易。', '',
        '第 19 轮 2018–2020 的局部改善和第 27 轮最近 5 年窗口的近期改善均保留。所有旧文件未改动；本轮不自动替换研究基线。2026 年仅到 8 月，最后完整标签信号为 8 月 21 日。当前比较不能证明市场状态的因果解释，也不是未见年份的独立确认。','',
        '图中的 R19 用于观察训练量与更新频率；完整 R18/R19/R23/R25 及两个对照以上表为准。','', '![R19 更新政策对比](fixed_window_wfo.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    report=OUT/'第二十八轮测试报告.md';report.write_text('\n'.join(parts),encoding='utf-8')
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained');colors=['#737b87','#87aecb','#3b82b4','#155679','#d4ae87','#c37a40','#8c461f']
    for row,w in enumerate(cfg['windows']):
        g=m[m.window.eq(w['name'])&m.method.eq('learned_vol_interaction')].set_index('history').loc[HISTORIES]
        for col,(field,title) in enumerate([('accuracy','方向准确率（越高越好）'),('brier','Brier（越低越好）')]):
            ax=axs[row,col];vals=g[field].to_numpy();ax.barh(np.arange(7),vals,color=colors);ax.set_yticks(np.arange(7),[HL[h] for h in HISTORIES],fontsize=10);ax.invert_yaxis();ax.set_title(WL[w['name']]+' · '+title,fontsize=12);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
            if field=='accuracy':
                ax.set_xlim(0,.77);ax.axvline(.5,color='#888888',ls='--',lw=.8);ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            else:ax.set_xlim(0,max(.40,vals.max()*1.2))
            for j,x in enumerate(vals):ax.text(x+.006,j,f'{x:.1%}' if field=='accuracy' else f'{x:.4f}',va='center',fontsize=10)
    fig.suptitle('固定 3/5 年窗口 × 自然 20 遍 × 年度/季度更新：波动交互 R19',fontsize=17)
    fig.supxlabel('同一批历史周信号；三种子平均。虚线为 50%，不代表合适的频率基准。',fontsize=10)
    files=[report]
    for ext in ['png','svg']:
        path=OUT/f'fixed_window_wfo.{ext}';fig.savefig(path,dpi=150);files.append(path)
    plt.close(fig);save(OUT/'report_manifest.json',dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('Report and PNG/SVG generated; visual review pending.',flush=True)
if __name__=='__main__':main()

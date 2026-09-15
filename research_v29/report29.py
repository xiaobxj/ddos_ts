"""Standalone reporting; no model or trigger-policy fitting."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results'
HL={'rolling5_matched':'5 年／年度／上一轮重复量','rolling5_annual20':'5 年／年度／自然 20 遍','rolling5_quarterly20':'5 年／季度／自然 20 遍','state90':'5 年／状态触发＋年度／自然 20 遍'}
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
    for phase in ['scoring','evaluation']:
        for name,d in read(OUT/f'{phase}_manifest.json')['artifacts'].items():assert sha(ROOT/name)==d,name
    assert read(OUT/'verification.json')['status']=='PASS';return prep
def main():
    prep=check();assert not (OUT/'report_manifest.json').exists();started=now();cfg=read(ROOT/'protocol.json');m=csv('ensemble_metrics');yr=csv('yearly_metrics');ev=csv('events');optional=ev[~ev.mandatory];b=csv('budgets');pairs=read(OUT/'primary_comparisons.json');sig=[r for r in pairs if r['holm_adjusted_p']<.05];v=read(OUT/'verification.json')
    parts=['# 第二十九轮：市场状态触发模型更新','',
        f"固定一条只依赖历史市场描述的触发规则，在 17 次非年末季度检查中触发 {int(optional.refit.sum())} 次额外更新；另有 6 次固定年末更新。全部计算复核通过。",'',
        '**本轮没有重新训练神经网络或分类头。** 第 28 轮已有所需各季度的 5 年窗口、自然 20 遍模型，本轮复用其冻结权重，模拟哪些日期启用新模型。共同预测保持原样；只补算被保留的季末模型在原评分季度之后所需的预测。所有 272 个历史周已经被查看，结果属于历史研究。','',
        '## 预先固定的触发规则','',
        '年末固定更新。其余季末，计算最近 60 个可用日信号的四维平均状态：趋势、波动、振幅、成交量变化。四维描述沿用 R18；截尾和标准化参数只使用当前模型截止日的 5 年成熟训练样本拟合。当前状态和参考状态均使用当时已观察到的价格，状态观察不要求未来标签成熟。', '',
        '参考状态为当前使用模型的截止日、最后 60 个可用日信号的平均状态。变化幅度 D 为四个标准化坐标均值差的平方再取平均。历史参照集只来自该模型截止日之前 5 年：取相隔同样观察天数的两段 60 日状态，计算相同 D。只有当前 D 严格超过历史 D 的第 90 百分位才更新。更新后重置参考状态，保留年末例行更新。', '',
        '间隔匹配用于区分距离上次更新约一季、两季或三季的情况。历史比较对重叠，因此第 90 百分位仅是预设的经验阈值，不代表 10% 假警报率或显著性检验。该规则识别描述均值变化，未建立经过验证的市场类别，也没有根据预测错误决定更新。', '',
        '神经训练窗口、20 遍、模型结构、市场与交互特征、分类参数和三种子等权融合均沿用 R28。触发日历对全部方法和种子相同；没有用收益方向、准确率、预测概率或损失选择更新日期，没有搜索其他阈值。季末当天的信号仍使用该检查日之前的模型，新决定仅影响之后的信号。','',
        '## 每次检查与实际选择','',table(['检查日','检查前模型','是否例行','D／阈值','是否更新','检查后模型'],[[r.cutoff,r.previous_model_cutoff if isinstance(r.previous_model_cutoff,str) else '—','是' if r.mandatory else '否','—' if r.mandatory else f'{r.ratio:.4f}','是' if r.refit else '否',r.selected_model_cutoff] for r in ev.itertuples()]),'',
        '阈值分布、历史窗口两端行号、最近与参考状态、模型训练行号保存在 cache/state_*.npz；每周选用的模型见 routing.csv。','',
        '## 若按历史逐期运行的训练预算','',table(['更新政策','更新日期数','三种子网络训练数','优化器更新次数'],[[HL[r.history],r.refit_dates,r.neural_fits_if_run_online,r.optimizer_steps_if_run_online] for r in b.itertuples()]),'',
        f"上述是逐期运行该政策时的预算。此次实际新增神经训练为 0、分类拟合为 0；对 {prep['extra_inference_weeks']} 个缺失模型周、{v['extension_replays']} 个已有网络补算推理。需要补算的日期在新评分前已冻结。",'']
    for w in cfg['windows']:
        g=m[m.window.eq(w['name'])].set_index(['history','method']);parts += [f"## {WL[w['name']]}（{w['n']} 周）",'',table(['政策','方法','正确/总数','准确率','平衡准确率','AUROC','Brier↓','对数损失↓'],[[HL[h],LABELS[method],f"{int(g.loc[(h,method),'correct_directions'])}/{w['n']}",f"{g.loc[(h,method),'accuracy']:.2%}",f"{g.loc[(h,method),'balanced_accuracy']:.2%}",num(g.loc[(h,method),'auroc']),num(g.loc[(h,method),'brier']),num(g.loc[(h,method),'log_loss'])] for h in HL for method in LABELS]),'']
    parts += ['## 预设配对比较','',f"28 项比较统一 Holm 校正，p<0.05 的改善 {sum(r['difference']<0 for r in sig)} 项、恶化 {sum(r['difference']>0 for r in sig)} 项。负损失差为状态政策更好。",'',table(['时期','候选方法','参照政策/方法','损失','差值','边际 95% 区间','原始 p','Holm p'],[[WL[r['window']],LABELS[r['candidate']],HL[r['reference_history']]+'/'+LABELS[r['reference']],r['metric'],num(r['difference']),f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]",num(r['p']),num(r['holm_adjusted_p'])] for r in pairs]),'',
        '两段分别使用 10,000 次循环移动区块重采样，每块 8 个有效周观察、固定种子 20260910；p 为中心化双侧值。校正只覆盖本轮比较，不能消除多轮查看历史的影响。合并 272 周仅作描述。','']
    for method in ['learned_vol_interaction','learned_order_extension','learned_order_offset']:
        parts += [f'## 各年度：{LABELS[method]}','',table(['信号年','周数']+[HL[h] for h in HL],[[year,int(g.n.iloc[0])]+[f"{g.set_index('history').loc[h,'accuracy']:.2%}" for h in HL] for year,g in yr[yr.method.eq(method)].groupby('year')]),'']
    diag=csv('seed_fusion_diagnostics');parts+=['## 种子稳定性','',table(['时期','政策','方法','种子准确率范围','融合准确率'],[[WL[r.window],HL[r.history],LABELS[r.method],f'{r.seed_accuracy_min:.2%}–{r.seed_accuracy_max:.2%}',f'{r.ensemble_accuracy:.2%}'] for r in diag[diag.window.ne('pooled_2021_2026')&diag.method.eq('learned_order_offset')].itertuples()]),'',
        '## 复核与边界','',
        f"17 个触发检查以独立逐窗求和、分位数插值和逐坐标变换复核，并逐次扰动未来特征检查决定不变。独立状态代数最大差 {v['independent_state']['maximum_independent_state_gap']:.3g}；新推理特征代数最大差 {v['maximum_feature_algebra_gap']:.3g}，概率最大差 {v['maximum_probability_gap']:.3g}。",'',
        '既有训练、凸分类解和旧预测继承 R28 的冻结验证；本轮核验选用检查点及旧文件哈希，独立复现补充推理和全部指标。4,522 个旧证据文件保持原样，上一轮 5 年窗口的局部改善继续保留。复核通过说明实现可复现，不能证明变化触发与未来可预测性的因果关系。', '',
        '当前规则只有季末检查，60 日平均可能忽略短暂突变；市场描述变化也可能与预测关系的失效无关。若本轮结果不理想，应保留失败结果，不根据已知好坏年份反向改变阈值。没有自动替换研究基线。','',
        '![状态触发与逐年对照](state_trigger.png)','',f"协议 SHA256：`{prep['protocol_sha256']}`",'']
    path=OUT/'第二十九轮测试报告.md';path.write_text('\n'.join(parts),encoding='utf-8');files=[path]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,1,figsize=(15,10),layout='constrained');ax=axs[0];x=np.arange(len(optional));colors=['#c44e52' if r else '#4c78a8' for r in optional.refit];ax.bar(x,optional.ratio,color=colors);ax.axhline(1,color='#555555',ls='--',lw=1);ax.set_xticks(x,optional.cutoff,rotation=45,ha='right',fontsize=9);ax.set_ylabel('状态变化幅度／历史阈值');ax.set_title('17 次非年末检查：红色为触发更新，蓝色为继续使用原模型');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axs[1];years=sorted(yr.year.unique());width=.20;colors=['#aaaaaa','#4c78a8','#72b7b2','#c44e52'];labels=['年度／旧重复量','年度／自然 20 遍','季度／自然 20 遍','状态触发＋年度']
    for j,h in enumerate(HL):
        g=yr[yr.history.eq(h)&yr.method.eq('learned_order_offset')].set_index('year').loc[years];pos=np.arange(6)+(j-1.5)*width;ax.bar(pos,g.accuracy,width,color=colors[j],label=labels[j])
        for xx,val in zip(pos,g.accuracy):ax.text(xx,val+.008,f'{val:.0%}',ha='center',fontsize=8)
    ax.set_xticks(np.arange(6),[str(y) if y<2026 else '2026 至 8 月' for y in years]);ax.set_ylim(0,.86);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.set_title('固定原模型＋顺序项 R25：同一批周信号的逐年准确率');ax.legend(ncol=4,loc='upper center',frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('5 年窗口、自然 20 遍：市场状态变化能否帮助选择更新时机？',fontsize=17);fig.supxlabel('历史研究；三种子等权融合。触发依据只使用当时已观察到的市场描述。',fontsize=10)
    for ext in ['png','svg']:
        p=OUT/f'state_trigger.{ext}';fig.savefig(p,dpi=150);files.append(p)
    plt.close(fig);save(OUT/'report_manifest.json',dict(phase='report',started_utc=started,finished_utc=now(),protocol_sha256=prep['protocol_sha256'],artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}));print('Report and figures generated; visual review pending.',flush=True)
if __name__=='__main__':main()

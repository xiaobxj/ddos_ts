"""Explicit historical replay of the previous Friday; never append a forecast."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT/'research_v51'))
from common51 import *
from common50 import pd,encoded,exclusive
from build50 import select_package
from data49 import load_snapshot,label_for,prefix
from quarter50 import Engine

SIGNAL='2026-09-11'
OUTPUT=PROJECT/'weekly_reviews/2026-09-11_reviewed_2026-09-15'
HISTORY_NAMES={
    'rolling5_annual20':'固定5年／年度更新',
    'weekly_state_validated':'季度市场状态修正',
    'annual_head_timeweight2y':'近期样本时间加权',
    'annual_head_weighted_intercept':'加权截距混合',
    'annual_head_weighted_slopes':'加权斜率混合',
}
METHOD_NAMES={
    'learned_market':'市场特征',
    'learned_vol_interaction':'波动率交互',
    'learned_order_extension':'涨跌顺序扩展',
    'learned_order_offset':'涨跌顺序修正',
}


def main():
    check_freeze(True)
    journal=Journal();events=journal.events()
    before={relative(p):sha(p) for p in journal.root.rglob('*') if p.is_file()}
    guard(not any(e['kind']=='prediction' and e['key']==SIGNAL for e in events),'Use the original committed prediction instead of replaying it')
    sources=[e for e in events if e['kind']=='snapshot' and e['payload']['end']>=SIGNAL]
    guard(sources,'No archived snapshot covering the requested Friday')
    source=sources[-1];full=load_snapshot(journal,source)
    frame=prefix(full[full.date.le(SIGNAL)].reset_index(drop=True),SIGNAL)
    entry,package=select_package(journal,SIGNAL)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    exclusive(OUTPUT/'observed_prefix.csv',frame.to_csv(index=False,lineterminator='\n').encode('utf-8'))
    save(OUTPUT/'request.json',dict(scope='historical_replay_not_prospective',requested_signal=SIGNAL,
        computed_utc=iso(utc()),source_snapshot_sequence=source['sequence'],
        source_snapshot_recorded_utc=source['recorded_utc'],source_snapshot_end=source['payload']['end'],
        source_snapshot_sha256=sha(journal.root/'events'/f'{source["sequence"]:06d}.json'),
        package_file=entry['payload']['package_file'],package_sha256=entry['payload']['package_sha256'],
        package_assembled_utc=package['assembled_utc'],annual_cutoff=package['annual_cutoff'],
        quarter_cutoff=package['quarter_cutoff'],implementation_freeze_sha256=sha(PROJECT/'research_v51/results/freeze.json'),
        input_prefix_sha256=sha(OUTPUT/'observed_prefix.csv'),script_sha256=sha(Path(__file__)),
        interpretation='Computed now from a later retrieved snapshot and later assembled model package. Input ends at signal, but this is not a contemporaneous or independent prospective forecast.'))
    # The existing engineering mode retains annual/quarter validity and file checks.
    forecast=Engine(package).predict(frame,SIGNAL,production=False)
    label=label_for(full,SIGNAL)
    guard(label['status']=='PENDING','Review the report wording if a mature result is now available')
    save(OUTPUT/'forecast.json',dict(scope='historical_replay_not_prospective',generated_utc=iso(utc()),forecast=forecast))
    save(OUTPUT/'label_status.json',dict(scope='historical_replay_only',snapshot_end=source['payload']['end'],label=label))
    rows=forecast['ensemble_predictions']
    state_names={'negative_low':'60日趋势为负、20日波动较低','negative_high':'60日趋势为负、20日波动较高',
                 'nonnegative_low':'60日趋势非负、20日波动较低','nonnegative_high':'60日趋势非负、20日波动较高'}
    signal_row=full[full.date.eq(SIGNAL)].iloc[0]
    prior=full[full.date.eq('2026-09-04')].iloc[0]
    week_return=float(signal_row.close/prior.close-1)
    table=['| 训练／更新方案 | '+' | '.join(METHOD_NAMES.values())+' |','|---|'+'---:|'*4]
    for history,name in HISTORY_NAMES.items():
        cells=[]
        for method in METHOD_NAMES:
            r=next(r for r in rows if r['history']==history and r['method']==method)
            cells.append(f"{r['probability']:.1%}{' ↑' if r['direction_up'] else ' ↓'}")
        table.append('| '+name+' | '+' | '.join(cells)+' |')
    report=f'''已按2026年9月11日（上周五）的行情前缀完成历史回放。

当日没有正式前瞻预测记录。以下概率于{utc().astimezone(TZ).strftime('%Y-%m-%d %H:%M')}北京时间计算，使用9月14日取得的行情快照及9月14日组装的现有参数包，输入仅保留到9月11日。参数的训练年度截止是2025-12-31、季度截止是2026-06-30；本次没有重训或调整任何方法。由于这些方案及文件是在事后取得和组装，结果不能当作9月11日已经发布的预测，也不计入正式前瞻成绩。

表中数值是模型输出的上涨概率，为三个原种子的平均值；↑表示概率高于50%，↓表示不高于50%。五套更新方案、四类方法分别保留，不根据本次结果选优。

{chr(10).join(table)}

行情状态识别：{state_names[forecast['state']]}。波动高低使用原年度冻结边界；“涨跌顺序”来自历史收盘价涨跌符号的排列关系。

该信号的目标尚未成熟。现有存档行情仅到2026-09-14，已经观察到9月11日之后的首根日线，但还没有观察到下一个有日线的周五及其之后的退出开盘价，因此无法判断9月11日信号最终预测是否正确。不能拿9月7日至11日已经发生的涨跌，来评价9月11日收盘后才形成的信号。

作为时间口径说明，存档行情中9月4日收盘为{float(prior.close):.2f}，9月11日收盘为{float(signal_row.close):.2f}，两次周五收盘之间变化为{week_return:+.2%}；这是已发生的行情变化，不是9月11日信号的目标收益或策略收益。

回放保存在本目录，未向prospective_r49追加预测或标签。原网页里的正式52周记录仍从2026-09-18开始。

[回放请求与来源](request.json) · [完整种子及组合输出](forecast.json) · [标签状态](label_status.json)
'''
    exclusive(OUTPUT/'上周五历史回放.md',report.encode('utf-8'))
    check_freeze(True)
    after={relative(p):sha(p) for p in journal.root.rglob('*') if p.is_file()}
    guard(before==after,'Real journal changed during the historical replay')
    guard(len(forecast['seed_predictions'])==60 and len(rows)==20,'Incomplete matrix')
    guard(frame.date.iloc[-1]==SIGNAL and (frame.date<=SIGNAL).all(),'Input includes future data')
    save(OUTPUT/'verification.json',dict(status='PASS',scope='historical_replay_not_prospective',
        input_rows=len(frame),input_end=SIGNAL,source_end=source['payload']['end'],seed_predictions=60,ensemble_predictions=20,
        actual_label_status=label['status'],new_training_runs=0,new_prospective_predictions=0,
        real_journal_files_unchanged=before,
        files={p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()}))
    print(encoded(dict(status='PASS',date=SIGNAL,state=forecast['state'],rows=rows,label=label,
        friday_close=float(signal_row.close),prior_friday_close=float(prior.close),observed_week_change=week_return,
        report=relative(OUTPUT/'上周五历史回放.md'))).decode('utf-8'))


if __name__=='__main__':main()

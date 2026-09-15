from pathlib import Path
import sys,re,json
ROOT=Path('D:/ddos_v3/research_v49');sys.path.insert(0,str(ROOT))
from common49 import *
from run49 import status,settle
from evaluate49 import summarize
from data49 import load_snapshot,unchanged_prefix

check_freeze();p=cfg();v=read(OUT/'verification.json');assert v['status']=='PASS'
assert len(v['cases'])==25 and v['real_prospective_predictions']==0 and v['new_neural_fits']==v['new_head_fits']==0
assert v['learned_seed_predictions_checked']==480 and v['ensemble_predictions_checked']==160 and v['label_parity_rows']==272
j=Journal();events=j.events();assert len(events)==1 and events[0]['kind']=='snapshot'
assert events[0]['payload']['end']=='2026-09-14' and events[0]['payload']['n']==4056
assert events[0]['payload']['source']['overlap_rows']==403 and events[0]['payload']['source']['overlap_exact']
frame=load_snapshot(j,events[0]);base=pd.read_csv(PROJECT/'research/data/1_000300.csv',float_precision='round_trip');unchanged_prefix(base,frame,base.date.iloc[-1])
assert len(frame)-len(base)==10 and not len(set(events[0]['payload'])&{'actual_up','exec_return','probability'})
assert settle(j)==[] and j.events()==events
state=status(j);summary=summarize(events,utc());assert state['predictions']==state['labels']==0 and state['scheduler_installed'] is False
assert state['next_scheduled_friday']=='2026-09-18' and state['latest_archived_snapshot']=='2026-09-14'
assert summary['status']=='DESCRIPTIVE_ONLY' and summary['metrics']==summary['comparisons']==[]
save(OUT/'startup_status.json',state);save(OUT/'initial_cohort_status.json',summary)
save(OUT/'snapshot_validation.json',dict(status='PASS',completed_utc=iso(utc()),snapshot_sequence=1,source=events[0]['payload']['source'],frozen_history_rows=len(base),merged_rows=len(frame),new_daily_bars=10,latest_bar='2026-09-14',original_frozen_prefix_exact=True,prospective_predictions=0,prospective_labels=0,settlement_without_predictions_is_noop=True,raw_and_csv_hashes_verified=True))
runtime_files={str(path.relative_to(PROJECT)):sha(path) for path in RUNTIME.rglob('*') if path.is_file()}
assert len(runtime_files)==4
save(OUT/'runtime_initial_manifest.json',dict(status='PASS',captured_utc=iso(utc()),scope='Initial runtime evidence only; future append-only events and blobs may be added outside immutable research_v49 delivery.',files=runtime_files,journal_tip=state['journal_tip_sha256']))
readme='''第49轮已建立前瞻预测记录流程，并完成工程验证。已冻结五套方案、三个原种子和52个周五观察时点；已采集截至2026年9月14日的新行情快照。目前正式前瞻预测和成熟标签均为0条，首次候选信号日为2026年9月18日。

使用原研究Python，在D:/ddos_v3目录执行：

```powershell
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v49\\run49.py status
```

周五北京时间18:00至23:00之间，先采集完整日线，再提交当周预测：

```powershell
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v49\\run49.py fetch
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v49\\run49.py record
```

只有当天的完整周五数据可以提交；节假日没有周五日线时，不用周四代替。窗口之外、数据缺失、重复预测或参数到期都会拒绝写入。不支持补录过去日期或通过命令行指定模拟时间。若依次手动执行两条命令，应确认fetch成功后再执行record；record本身也检查快照日期与参数。

有新的完整日线后，可读取成熟标签并查看累计结果：

```powershell
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v49\\run49.py fetch
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v49\\run49.py settle
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v49\\run49.py evaluate
```

使用同一来源的完整本地CSV时，可以把fetch替换为ingest --csv <路径>。所需列为date/open/high/low/close/volume，日期升序、唯一；必须完整保留原2010年至2026年8月31日冻结前缀。采集时间由程序现场记录，CSV文件中的时间说明不会用作前瞻时间证明。

当前程序是手动CLI，没有安装定时任务或后台服务，不会在关机或无人调用时自行预测。首个参数包使用2025年末年度模型及2026年6月末季度修正，仅适用到2026年9月30日。10月及之后需要按冻结的原R28/R39/R47配方生成并复核新参数包，再接入后续运行器。本轮没有实现未来季度／年度重训任务，也没有把旧季度参数延长一年。到期拦截意味着当前交付不是已完成全年自动滚动的服务。

不可变研究材料保存在research_v49；后续运行状态单独保存在同级prospective_r49。后者包含genesis.json、events和blobs；事件按序号与前一事件哈希关联。原始接口响应、规范化CSV、预测参数和标签来源都有哈希。成功提交的记录不会覆盖；异常中断残留writer.lock时需先核实是否仍有进程运行。本地哈希链不是第三方时间戳公证，不应宣称它能防止有权限的人重写整条证据链。

52个时点覆盖2026年9月18日至2027年9月10日。第13／26／39周可作描述性汇总，不做提前显著性筛选；主要评估最早2027年10月1日，需至少40个完整成熟的共同样本且无尚未成熟的已提交预测。40仅为操作门槛，不代表已经达到统计功效要求。缺失时点不补足或替换。42项固定比较对方向误差与Brier使用8周区块、10,000次重采样和统一Holm校正。缺失日历周保留在重采样轴上。

[结果与运行状态](results/结果解读与下一步.md) · [冻结协议](protocol.json) · [工程验证](results/verification.json) · [当前参数包](results/bootstrap_package.json) · [首次数据采集验证](results/snapshot_validation.json)
'''
save_text=lambda path,text:exclusive(path,text.encode('utf-8'))
save_text(ROOT/'README.md',readme)
report=f'''第49轮完成了前瞻记录流程的搭建与验证，并成功采集截至2026年9月14日的行情。当前正式预测0条、成熟标签0条，没有新增前瞻准确率。本轮没有重新挑选方法、种子或阈值，也没有新增神经训练或分类头拟合。

五套方案固定为原年度等权、原四状态季度、完整时间加权、只替换截距、只替换斜率。每套保留R19／R23／R25三个主要方法，共15个主要候选；R18作为诊断另存。所有候选统一使用原三个种子平均概率，严格p>0.5判断上涨。每个有效信号一次提交60条学习方法种子预测和20条集成预测；native MSE与原等权训练上涨频率仍单独保存为诊断对照。此前市场／订单交互成果，以及第48轮R19局部方向收益和斜率组合局部概率收益，继续保留，没有将其升级为确定有效的方案。

| 项目 | 固定安排 |
|---|---|
| 首个候选信号 | 2026-09-18；必须当天有完整周五日线 |
| 预测写入窗口 | 当天北京时间18:00（含）至23:00（不含） |
| 观察范围 | 52个日历周五，2026-09-18至2027-09-10 |
| 中途检查 | 第13、26、39周只作描述性汇总 |
| 主要评估 | 最早2027-10-01；至少40个完整成熟共同样本，无待成熟的已提交预测 |
| 主要比较 | 42项：季度对年度，以及三个加权／拆分方案分别对年度和季度；三种方法、两种损失 |
| 当前参数有效期 | 年度2025-12-31、季度2026-06-30，适用至2026-09-30 |
| 运行方式 | 手动CLI；没有定时任务或后台服务 |

预测时只接收以当日信号为末行的历史前缀，使用125根已观察日线重建原输入，不需要未来标签。输入转换、神经特征、原监督射线、市场和订单交互的坐标保持。两种混合方案继续直接交换固定的截距或斜率；R25订单gamma属于斜率。

结果标签在单独步骤中生成。仍按信号后的下一根日线开盘作为入场价，下一次实际出现的同星期几日线之后的开盘作为退出价，同时等待五根辅助OHLCV目标及原联合有效性条件满足。没有用固定七天代替原节假日处理。预测需在信号当天写入，严格早于后续入场和标签成熟；这比仅要求“标签成熟前写入”更严格。

已实测数据采集：原冻结文件为4,046根日线，新快照为4,056根，新增10根，末日为2026-09-14。接口返回与旧冻结行情重叠的403个交易日，OHLCV逐值一致；合并后的整个原冻结前缀也一致。腾讯原始JSON响应及完整CSV已按内容哈希归档，并记录实际UTC采集时间。新增行情只用于建立快照，没有回补8月28日、9月4日或9月11日的所谓前瞻成绩。无预测时执行标签结算返回空结果，不会制造标签记录。

25项工程检查通过。对2026年7—8月8个原历史周，重新从原始日线转换输入，并执行3个既有神经模型的逐条推理；125根输入和25×5打包与旧缓存逐元素一致。480条学习方法种子预测和160条集成预测方向全部与旧结果一致。最大概率差为{v['maximum_probability_gap']:.12g}，最大特征差为{v['maximum_feature_gap']:.12g}，最大native收益预测差为{v['maximum_native_return_gap']:.12g}，处于事先固定的工程回放容差内。这里包含24个历史种子窗口的新推理，不是新的训练，也不是新增盲测；不能把8周回放当成前瞻表现或宣称全历史输出逐位相同。

另外，272个原周信号的收益标签与联合成熟日期复现通过，包含节假日导致的非5交易日持有期。未来行混入、无效125根输入、缺失种子、非有限概率、0.5边界、季度／年度参数到期、迟报、重复写入、并发写入、孤立标签、数据前缀修订与遗漏、快照或哈希链改动均有拦截检查。内存中的合成数据用于验证评估时点、最少样本、待成熟标签、42项校正和缺失日历周重采样；合成数据未写入真实预测账本。

每个候选共享同一批有效成熟周。52个预定时点全部列示：尚未到期、无及时预测、等待标签、无效标签、完成。没有日线只说明没有可用信号数据，程序不会擅自判断是交易所休市还是数据缺口。至少40周是预先设定的操作门槛，没有做出足够统计功效的承诺。若最终不足门槛或仍有待成熟标签，只能报告证据不足，不能延长或替换缺失周来追求显著。

区块重采样保持52个日历位置，缺失位置不压缩。固定8周区块、10,000次抽样、原种子20260910、中心化双侧p，42项比较统一Holm校正。主要评估前不计算p值，也不自动晋升模型。流程仍不能消除数据缺失造成的选择性，单次未来观察也不保证结果有效。

当前接通的是首个参数包的预测、归档、成熟标签与评估流程。2026年9月30日后的季度参数，以及下一年年度模型，需要在各自截止数据可用后，按本轮固定的原配方生成并复核，再接入后续运行器。本轮尚未实现这些未来重训作业，当前运行器到期会拒绝使用旧包；不会把它默认为全年可用。下一项独立工程工作应是季度／年度参数更新与归档的衔接，而不是继续用已反复查看的结果挑选赢家。

实际下一次记录节点是9月18日完整日线可用之后。在此之前可以检查运行状态；本轮没有安装无人值守任务，程序不会自行在未来日期启动。完整手动命令见[操作说明](../README.md)。

6,206个旧文件保持不变；本轮7个源文件、协议、参数来源及依赖均冻结。运行账本放在prospective_r49，与不可变研究交付分开；本次归档了其起始清单。哈希与本机UTC时间是本地可审计记录，不是独立时间戳公证。没有涉及交易账户、实盘部署或收益宣称。

[冻结协议](../protocol.json) · [工程验证](verification.json) · [逐条工程回放](engineering_replay.csv) · [初始运行状态](startup_status.json) · [52个时点状态](initial_cohort_status.json) · [数据采集验证](snapshot_validation.json) · [初始运行证据清单](runtime_initial_manifest.json)
'''
save_text(OUT/'结果解读与下一步.md',report)
reports=[ROOT/'README.md',OUT/'结果解读与下一步.md']
for path in reports:
    text=path.read_text(encoding='utf-8');assert '\ufffd' not in text and '????' not in text
    width=None
    for line in text.splitlines():
        if line.startswith('|'):
            n=len(line.split('|'));width=width or n;assert n==width
        else:width=None
    for target in re.findall(r'\]\(([^)]+)\)',text):assert (path.parent/target).exists(),target
assert old_evidence()==read(OUT/'freeze.json')['old_evidence'];check_freeze()
save(OUT/'interpretation_checks.json',dict(status='PASS',completed_utc=iso(utc()),checks=['first_and_last52slots_and_delayed_primary_review','42comparisons_and_minimum40_gate','25engineering_checks_and_replay_counts','single_sample_roundoff_reported_exactly','successful_provider_snapshot_403_overlap_10new_bars','snapshot_archived_without_retrospective_forecast','zero_prospective_predictions_labels_and_metrics','manual_runtime_no_scheduler','bootstrap_expiry_and_future_refit_not_implemented_disclosed','6206oldfiles_and7sources_preserved','Chinese_encoding_tables_and_links'],artifacts={str(path.relative_to(ROOT)):sha(path) for path in reports+[OUT/'verification.json',OUT/'startup_status.json',OUT/'initial_cohort_status.json',OUT/'snapshot_validation.json',OUT/'runtime_initial_manifest.json']}))
files={str(path.relative_to(ROOT)):sha(path) for path in ROOT.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
save(OUT/'delivery_manifest.json',dict(status='PASS',completed_utc=iso(utc()),protocol_sha256=sha(ROOT/'protocol.json'),freeze_sha256=sha(OUT/'freeze.json'),files=files,old_files_preserved=6206,runtime_initial_manifest='results/runtime_initial_manifest.json',runtime_is_separate_append_only=True))
print(json.dumps(dict(status='PASS',delivery_files=len(files),old_files=6206,latest_snapshot='2026-09-14',prospective_predictions=0,first_signal='2026-09-18'),indent=2))

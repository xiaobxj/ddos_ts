from pathlib import Path
import sys,re
ROOT=Path('D:/ddos_v3/research_v50');sys.path.insert(0,str(ROOT))
from common50 import *
from build50 import activate,select_package
from run50 import status
check_freeze(True);v=read(OUT/'verification.json');l=read(OUT/'lifecycle_verification.json')
assert v['status']==l['status']=='PASS' and l['verification_sha256']==sha(OUT/'verification.json')
assert [v[k] for k in ['neural_replays','epochs','optimizer_steps','head_fits','quarter_cutoffs','quarter_offsets','quarter_gate_cells','future_cutoff_fits']]==[3,60,600,24,23,1104,368,0]
assert v['calibration_added_dates']==['2026-08-28','2026-09-04'] and v['calibration_added_seed_rows']==24
assert v['prospective_predictions']==l['actual_prospective_predictions']==0
assert old_evidence()==read(OUT/'freeze.json')['old_evidence']
j=Journal();before=j.events();assert len(before)==1 and before[0]['kind']=='snapshot'
bootstrap=PROJECT/'research_v49/results/bootstrap_package.json';receipt=OUT/'bootstrap_verification.json'
save(receipt,dict(status='PASS',completed_utc=iso(utc()),package_sha256=sha(bootstrap),package_source='Previously verified and frozen R49 bootstrap, unchanged; engineering replay models are not activated.',engineering_verification_sha256=sha(OUT/'verification.json'),lifecycle_verification_sha256=sha(OUT/'lifecycle_verification.json'),freeze_sha256=sha(OUT/'freeze.json')))
activation=activate(j,bootstrap,receipt);assert activation['sequence']==2 and activation['kind']=='parameter_package'
event,package=select_package(j,'2026-09-18');assert event==activation and package==read(bootstrap)
state=status(j);assert state['current_package']=='annual2025-12-31_quarter2026-06-30' and state['next_parameter_cutoff']=='2026-09-30'
assert state['predictions']==state['labels']==0 and state['scheduler_installed'] is False
save(OUT/'startup_status.json',state);save(OUT/'bootstrap_activation.json',activation)
runtime=base.RUNTIME;runtime_files={relative(p):sha(p) for p in runtime.rglob('*') if p.is_file()};assert len(runtime_files)==5
save(OUT/'runtime_handoff_manifest.json',dict(status='PASS',captured_utc=iso(utc()),files=runtime_files,journal_events=2,snapshots=1,registered_parameter_packages=1,prospective_predictions=0,prospective_labels=0,original_genesis_and_snapshot_preserved=True))
write=lambda p,t:exclusive(p,t.encode('utf-8'))
readme='''第50轮已接通季度／年度参数更新与前瞻预测器。继续使用第49轮冻结的五套方案、原种子、52个周五时点和42项主要比较，原预测账本也继续使用。新的入口为run50.py；run49.py及其交付保持原样。

在D:/ddos_v3目录查看当前状态：

```powershell
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py status
```

每个有完整周五日线的信号日，在北京时间18:00至23:00之间采集并记录：

```powershell
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py fetch
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py record
```

请先确认fetch成功。record会再次检查当天快照、参数的年度／季度截止、验证凭据和写入时点。没有当天周五日线、窗口已过、参数缺失或已提交过预测时会拒绝；不支持历史时间覆盖。

季度截止数据可用后，更新下一季度参数。第一次是2026-09-30，北京时间18:00之后，且采集快照至少覆盖该截止日期：

```powershell
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py fetch
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py refresh --cutoff 2026-09-30
```

2026年末使用同一入口；Dec31会触发原三个种子的年度20遍重训、等权／时间加权分类头拟合、状态边界更新，并按原规则将第一季度状态修正归零：

```powershell
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py refresh --cutoff 2026-12-31
```

同样必须先采集覆盖截止日期的快照。上述日期在当前尚未到达，提前运行会拒绝，不能把旧快照当作年末或季末数据。2027-03-31、2027-06-30继续使用同样的季度命令。截止日若没有交易数据，需要在后续快照覆盖该日后再更新；训练接口始终裁切到截止日。若漏做一季，应按截止日期顺序补建参数供后续更新使用；过期期间的预测仍不能补记。

每次refresh会创建一个独立构建目录，保存截止前缀、训练成员、模型及分类头、校准样本、验证结果。所有检查通过后才向原账本追加参数注册事件。失败构建保留failure.json且不注册；既有包不覆盖。同一季度不重复注册，已经出现预测的季度不允许事后替换参数。

读取成熟标签和累计结果：

```powershell
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py fetch
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py settle
& .\\research_v4\\.venv_gpu\\Scripts\\python.exe -B -u research_v50\\run50.py evaluate
```

fetch、完整CSV的ingest --csv、标签规则及评估口径沿用第49轮；先前成功的主要评估结果固定保存，不反复择时计算。参数构建写入prospective_r50/builds，预测／快照／标签／参数注册事件写入原prospective_r49。研究工程回放文件保存在research_v50/results，不能当作真实前瞻记录。

目前已注册的是原第49轮参数包，供9月18日及9月25日符合条件的信号使用。9月30日和12月31日的新包尚未生成；对应生成、验证和注册流程已实现并通过历史回放及边界测试。没有安装自动定时任务或后台服务，仍需手动运行。

[结果与验证](results/结果解读与下一步.md) · [冻结实现协议](protocol.json) · [工程回放](results/verification.json) · [参数注册测试](results/lifecycle_verification.json) · [当前状态](results/startup_status.json)
'''
write(ROOT/'README.md',readme)
report=f'''第50轮完成了季度／年度参数更新与前瞻预测器的衔接。更新流程现已实现，当前已将原第49轮参数包注册到同一前瞻账本，预测器会按年度与季度截止选择经过验证的包。9月30日和12月31日尚未到来，本轮没有提前拟合这些未来参数；正式前瞻预测仍为0条。

原五套方案、R19／R23／R25及R18诊断、三个原种子、市场／订单交互、52个周五和42项评估比较全部沿用第49轮。历史局部改善继续保留，不根据本轮工程回放重新选取候选。此次重点是让固定研究配方在以后取得截止数据时能够更新，而不是报告新的预测优势。

| 更新环节 | 实现内容 |
|---|---|
| 季度更新 | 使用截至季末已经联合成熟的周样本，52周训练、13周验证，并清除标签跨越验证起点的训练样本 |
| 状态修正 | 原四状态、训练至少10周、验证至少5周、原正则与幅度上限；Brier改善且方向正确数不降低才启用，否则精确回退 |
| 年度更新 | 固定5年、原三种子、每种子自然20遍；重新训练年度网络及原等权分类头，再在同坐标拟合固定时间加权分类头 |
| 混合组合 | 原等权斜率配加权截距、加权斜率配原等权截距；R25订单gamma仍随斜率处理 |
| 启用条件 | 文件冻结、独立数值核验通过、注册凭据一致，且年度／季度截止与信号匹配 |
| 失败处理 | 保留失败构建与错误原因，不注册，不覆盖已用参数，不回补错过的预测 |

完整年度回放选用已查看过的2025-12-31截止，仅用于工程验证。3个种子各20遍，共60遍、600次优化器更新；全部最终模型、优化器和随机数状态的哈希与原R28训练完全一致。1,206个训练成员、原125根输入及25×5打包、标准化标签与目标缩放参数也逐元素复现。

12个等权头和12个时间加权头均重新计算并核对，最大系数差为{v['maximum_coefficient_gap']:.12g}。额外执行了3个年度检查点的训练特征重放与24个头的独立梯度检查，最大梯度为{v['annual']['maximum_gradient']:.12g}。这是一次固定配方的复现，没有搜索训练轮数、窗口、半衰期或正则系数。

全部23个原季度截止点完成回放：1,104个状态修正系数、368个验证决策与旧结果一致，最大旧系数差为{v['maximum_archived_quarter_gap']:.12g}；独立的区间求根核验和成熟度清除也通过。年末归零、样本不足、Brier持平与零修正回退均得到检查。

重训得到的年度模型也已接到第49轮原始前缀预测接口。2026年7—8月8个历史周的480条学习方法种子方向预测全部与旧结果一致，最大概率差为{v['maximum_probability_gap']:.12g}，处于冻结的工程容差内。回放没有进入真实前瞻账本，不能据此计算新的前瞻准确率。

校准样本更新也完成了检查。原历史校准银行保留；截至现有9月14日快照，8月28日和9月4日新增24条种子级校准记录，明确标为“仅校准回放”。这些日期早于前瞻开始或没有及时预测，不会补录成前瞻成绩。另用隔离的内存样本验证：若已存在当时提交的原年度预测，校准会优先使用那条真实记录的概率、logit与状态，而不会重新拟合或重算来替换它。新校准标签均核对截止前缀中的原联合成熟定义。

实现时发现并保留了一次训练前检查失败。旧R5收益标签来自R4观察表写CSV后、按pandas默认精度读回的数值；直接从行情重算，收益缩放参数会相差约10⁻¹⁷。初次检查没有启动训练或参数注册。复现原CSV读回步骤后，标签与缩放参数精确一致；辅助目标不变。初版9个源码、协议、冻结清单及诊断共13个文件保存在research_v50_attempt1，最终流程保留了原数值口径，没有靠放宽匹配标准使训练通过。

工程主体有{len(v['cases'])}项检查，注册与运行衔接另有{len(l['cases'])}项隔离测试。提前要求9月30日／12月31日更新、使用过旧快照、非季度截止、缺少状态参数、参数有效期错误、旧前缀修订、重复注册、在已预测季度更换参数、失败验证凭据、工程回放包误注册与已注册凭据被改动，均有对应拦截。真实未来更新请求在写入构建目录前即被拒绝，真实账本未被测试样本改动。

完成验证后，只新增了一个实际参数注册事件：原第49轮的年度2025-12-31／季度2026-06-30包，字节不变，有效至2026-09-30。工程重训得到的模型没有替换这个包。当前实际账本为1个行情快照、1个参数注册事件、0条预测、0条标签；原起始记录与快照均保持。

接下来在9月18日完整周五行情到齐后，用run50.py的fetch、record记录首条符合条件的预测；9月30日数据可用后执行refresh --cutoff 2026-09-30。年末同一命令使用2026-12-31会触发年度重训并产生次年第一季度参数。仍为手动流程，没有安装自动调度。[完整操作命令](../README.md)已保存。

这些检查证明了既定更新配方和参数切换的实现可复现旧结果，并没有证明它们在未来一定改善预测。第49轮的前瞻统计安排保持：中途只作描述性汇总，主要评估最早2027-10-01，至少40个完整成熟共同样本且没有待成熟的已提交预测。没有改变交易账户、实盘代码或收益口径。

6,230个旧研究／起始运行文件保持，初次检查的13个归档文件也纳入最终哈希核验；本轮9个最终源文件与实现协议冻结。新的参数包采用独立目录，原账本追加注册事件。哈希链与本机时间提供本地可审计证据，仍不等于第三方时间戳公证。

[冻结实现协议](../protocol.json) · [年度与季度回放验证](verification.json) · [23季度核验表](quarter_replay_checks.csv) · [仅校准回放明细](calibration_extension_check.csv) · [注册与运行测试](lifecycle_verification.json) · [原包启用凭据](bootstrap_verification.json) · [实际注册事件](bootstrap_activation.json) · [当前运行状态](startup_status.json) · [运行交接清单](runtime_handoff_manifest.json)
'''
write(OUT/'结果解读与下一步.md',report)
paths=[ROOT/'README.md',OUT/'结果解读与下一步.md']
for path in paths:
    text=path.read_text(encoding='utf-8');assert '\ufffd' not in text and '????' not in text
    width=None
    for line in text.splitlines():
        if line.startswith('|'):
            n=len(line.split('|'));width=width or n;assert n==width
        else:width=None
    for target in re.findall(r'\]\(([^)]+)\)',text):assert (path.parent/target).exists(),target
assert old_evidence()==read(OUT/'freeze.json')['old_evidence'];check_freeze(True)
save(OUT/'interpretation_checks.json',dict(status='PASS',completed_utc=iso(utc()),checks=['three_checkpoint_digest_triplets_exact','60epochs600steps24heads','23quarters1104offsets368gates','480engineering_directions_with_roundoff_disclosed','24calibration_rows_not_prospective','first_preflight_failure_preserved','bootstrap_unchanged_registered_only_after_validation','no_future_cutoff_fit_or_prospective_prediction','49cohort_rules_unchanged_and_manual_scheduler','6230oldfiles_and13archivefiles_preserved','Chinese_encoding_tables_and_local_links'],artifacts={relative(path):sha(path) for path in paths+[OUT/'verification.json',OUT/'lifecycle_verification.json',OUT/'bootstrap_verification.json',OUT/'bootstrap_activation.json',OUT/'startup_status.json',OUT/'runtime_handoff_manifest.json']}))
files={str(path.relative_to(ROOT)):sha(path) for path in ROOT.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
save(OUT/'delivery_manifest.json',dict(status='PASS',completed_utc=iso(utc()),files=files,protocol_sha256=sha(ROOT/'protocol.json'),freeze_sha256=sha(OUT/'freeze.json'),old_files_preserved=6230,initial_attempt_files=13,runtime_handoff_manifest='results/runtime_handoff_manifest.json'))
print(encoded(dict(status='PASS',delivered_files=len(files),old_files=6230,registered_packages=1,prospective_predictions=0)).decode())

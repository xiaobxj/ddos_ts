"""Accept the actual manual CLI and write the R51 delivery documents once."""
from pathlib import Path
import json
import re
import subprocess
import sys

PROJECT=Path('D:/ddos_v3')
sys.path.insert(0,str(PROJECT/'research_v51'))
from common51 import *
from cycle51 import Backend,describe

check_freeze(True)
freeze=read(OUT/'freeze.json'); verification=read(OUT/'verification.json')
assert old_evidence()==freeze['old_evidence']
actual=Journal(); events=actual.events()
assert len(events)==2 and [e['kind'] for e in events]==['snapshot','parameter_package']
before={relative(p):sha(p) for p in previous.base.RUNTIME.rglob('*') if p.is_file()}
assert before==verification['real_runtime_files'] and len(before)==5
assert utc().astimezone(TZ).strftime('%Y-%m-%d')=='2026-09-15' and utc().astimezone(TZ).hour<18

def command(*args):
    return subprocess.run([sys.executable,'-B','-u',str(ROOT/'run51.py'),*args],cwd=PROJECT/'_work',capture_output=True,text=True,encoding='utf-8')

result=command('plan'); assert result.returncode==0,result.stderr
plan=json.loads(result.stdout)
assert plan['status']=='PRE_CLOSE_WAIT' and plan['next_unrecorded_signal']=='2026-09-18' and plan['next_parameter_cutoff']=='2026-09-30'
save(OUT/'startup_plan.json',plan)
assert before=={relative(p):sha(p) for p in previous.base.RUNTIME.rglob('*') if p.is_file()}
result=command('cycle'); assert result.returncode==0,result.stderr
cycle_result=json.loads(result.stdout)
assert cycle_result['status']=='PRE_CLOSE_WAIT' and cycle_result['steps']==[]
assert cycle_result['after']['recorded_predictions']==cycle_result['after']['labels']==0
save(OUT/'actual_cycle.json',cycle_result)
for args in [('cycle','--at','2026-09-18T18:30:00+08:00'),('cycle','--journal','test')]:
    result=command(*args)
    assert result.returncode==2 and 'unrecognized arguments' in result.stderr,result
assert before=={relative(p):sha(p) for p in previous.base.RUNTIME.rglob('*') if p.is_file()}
assert not (PROJECT/'prospective_r50').exists()
operations={relative(p):sha(p) for p in RUNTIME.rglob('*') if p.is_file()}
assert len(operations)==2 and not (RUNTIME/'cycle.lock').exists()
save(OUT/'cli_acceptance.json',dict(status='PASS',completed_utc=iso(utc()),
     checks=['plan_is_read_only_and_works_outside_project_cwd','actual_preclose_cycle_only_writes_operational_receipts',
             'timestamp_override_rejected','custom_journal_override_rejected','real_scientific_journal_unchanged',
             'no_future_parameter_build_directory','cycle_lock_released'],
     real_runtime_files=before,operational_receipt_files=operations))

replay=read(OUT/'synthetic_rehearsal.json')
assert replay['synthetic_test_only'] is True and replay['future_parameter_fits']==0
assert len(verification['cases'])==22 and replay['seed_predictions']==60 and replay['ensemble_predictions']==20
assert replay['statistical_comparisons']==0 and replay['label']['entry_date']=='2026-09-21' and replay['label']['exit_date']=='2026-09-28'
report='''第51轮已完成手动运行入口和完整链路演练。现在一条 cycle 命令会按顺序采集行情、检查并更新到期参数、记录当周预测、结算成熟标签、汇总固定样本。之前确认的固定5年滚动年度训练、季度状态识别与修正、市场及订单交互继续保留，本轮没有改变训练配方或增加候选搜索。

工程验证共22项，其中15项检查执行顺序与边界，7项检查真实模型在隔离模拟夹具中的整条链路。另有7项正式命令验收。所有检查通过。由于未来真实行情还没有发生，本轮没有开展新的未来参数训练，也没有新的前瞻准确率。

| 模拟时点（北京时间） | 验证结果 |
|---|---|
| 9月18日17:59:59 | 仅等待，不采集、不提交预测 |
| 9月18日18:30 | 存档模拟行情，原3个种子产生60条方法／种子输出，汇总为20条方法／方案输出，只记录1个周预测事件 |
| 同周五再次运行 | 复用当天快照和已提交预测，不增加科学账本事件 |
| 9月25日23:05 | 已过预测窗口，不补记本周；上周标签仍因退出价格尚未观察到而等待 |
| 9月28日18:30 | 上周标签成熟，入场日期为9月21日、退出及联合成熟日期为9月28日；收益标签与手算开盘价比值完全一致 |
| 成熟后再次运行 | 不重复追加标签 |
| 修改已观察的后续行情前缀 | 后续流程停止，保留错误记录 |

上述日期后的新增行情全部为人工构造，日期也是隔离测试时钟；不是对这些未来交易日是否开市的认定。原模型与原参数包字节保持，预测、标签、评价均调用既有函数。模拟结果只证明这条实现链能衔接，不能证明模型未来有效，60条种子输出和20条组合输出也不是60或20个独立周样本。模拟评价只有1个成熟周，继续只作描述性汇总，没有任何主要统计比较。

边界检查还覆盖了采集失败后停止、周五日线缺失不替换成周四、采集或参数更新跨过23点后不再预测、重复运行、周末不索取当天日线、来源验证失败、完整流程并发锁、漏做的季度按序调度、季度截止数据不足时等待、截止日18点后准备下一季度、禁止提前年度更新、更新失败保留已取得快照，以及主要评估只保存一次。季度和年度调度使用模拟后端检查调用，没有在模拟未来日期执行真实拟合；真正的年度重训和季度数值复现已在第50轮验证。

正式入口使用实际系统时钟；日期覆盖和自选账本参数均被拒绝。本轮在2026-09-15上午执行了正式 plan 与 cycle，结果都是 PRE_CLOSE_WAIT。真实科学账本仍只有1个行情快照和1个参数注册事件，共5个文件，0条预测、0条标签；文件哈希完全保持。只新增了独立的运行开始／完成凭据。未来参数目录尚未创建，也没有安装自动调度。

下一次有意义的动作是在2026-09-18完整周五日线可用后，于北京时间18点至23点之前运行 cycle，记录首条真实前瞻预测。后续工作日18点之后可用同一命令采集并检查成熟标签；9月30日覆盖截止数据后，同一入口会调用既定季度更新，12月31日则调用原年度滚动重训。若某个时点缺数据或已错过窗口，保留缺失记录，不回补成绩。

6,279个已有研究／交接文件通过哈希保持检查，本轮5个Python源码、PowerShell入口和实现协议共7个文件冻结。运行日志使用独立目录，既有研究文件和真实账本记录均未改写。模拟证据另行存放并明确标记。原52个候选周五、42项比较、至少40个成熟共同样本及最早2027-10-01的主要评估安排保持；中途不据此选优或宣布改进有效。

[操作命令](../README.md) · [工程验证](verification.json) · [模拟链路](synthetic_rehearsal.json) · [正式命令验收](cli_acceptance.json) · [当前计划](startup_plan.json) · [本次实际运行](actual_cycle.json)
'''
exclusive(OUT/'结果与运行安排.md',report.encode('utf-8'))
for path in [ROOT/'README.md',OUT/'结果与运行安排.md']:
    content=path.read_text(encoding='utf-8');assert '\ufffd' not in content and '????' not in content
    for target in re.findall(r'\]\(([^)]+)\)',content):
        assert (path.parent/target).is_file(),target
    width=None
    for line in content.splitlines():
        if line.startswith('|'):
            actual_width=len(line.split('|'));width=width or actual_width;assert width==actual_width
        else:width=None
check_freeze(True);assert old_evidence()==freeze['old_evidence']
save(OUT/'interpretation_checks.json',dict(status='PASS',completed_utc=iso(utc()),checks=[
     '22_engineering_and7_cli_checks','synthetic_chain_not_real_market_or_independent_samples',
     'unchanged_methods_and_training_recipe','no_new_forecast_accuracy_or_future_training',
     'real2_events5_files_unchanged','first_real_slot_and_next_cutoff_documented',
     'manual_operation_and_failure_recovery_limits_documented','original_evaluation_protocol_retained',
     '6279_prior_files_preserved','UTF8_links_and_table_valid']))
files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
save(OUT/'delivery_manifest.json',dict(status='PASS',completed_utc=iso(utc()),files=files,
     old_files_preserved=6279,freeze_sha256=sha(OUT/'freeze.json'),
     real_runtime_files=before,operational_receipt_files=operations,
     synthetic_fixture_directory=verification['synthetic_fixture_directory']))
print(encoded(dict(status='PASS',delivered_files=len(files)+1,old_files_preserved=6279,
     real_predictions=0,real_labels=0,actual_cycle_status=cycle_result['status'])).decode('utf-8'))

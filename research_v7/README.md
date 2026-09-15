# 第七轮：固定 20 轮候选的多种子时间块检验

本轮补齐第六轮留下的稳定性检验。候选固定为组合模型第 20 轮；主要对照是此前选中的原模型第 10 轮和训练均值。原模型第 20 轮用于同预算比较，组合模型第 10 轮用于训练前缀和近常数输出诊断，均不重新参与候选筛选。

完整历史直接复用第六轮两种架构、两个训练轮数、三个年度和三个种子的 36 个检查点。三种历史删除方式各跑三个种子、两种架构，共新增 54 次拟合，每次保存第 10、20 轮。共复验 108 个新检查点与 36 个既有检查点。

所有模型仍用第五轮相同原始输入和联合标签；删除掩码和保留样本的重复展示规则与第六轮相同。验证仍是已查看过的 2018—2020 年 141 周，不是独立确认。

[冻结协议](protocol.json) · [最终报告](results/第七轮测试报告.md)

在项目根目录使用既有 GPU 环境。准备和训练命令拒绝覆盖已有运行。需要从头复现时请另建研究目录，保留本轮证据。

```powershell
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v7\prepare7.py
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v7\test_contracts7.py
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v7\train7.py --arm baseline
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v7\train7.py --arm combined
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v7\evaluate7.py
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v7\verify7.py
python .\research_v7\figures7.py
python .\research_v7\report7.py
python .\research_v7\check_delivery7.py
```

必须等待准备完成后再运行行为测试，测试通过后才能启动训练。本轮准备过程中首次行为测试启动过早，遇到尚未生成的复用预测文件；保留了原始错误和原因，等待准备完成后使用相同测试重跑通过，没有改变训练协议。见 `results/initial_contract_scheduling_note.json`。

本次两个架构用独立进程同时训练，各有独立模型、优化器、随机状态、CUDA 上下文和输出。两个训练清单的运行时间相互重叠，不应相加为总耗时；顺序运行上述命令也可复现。

本轮共冻结此前 1,632 个证据文件；最终交付清单与只读检查脚本用于核对报告、图表、指标和文件哈希。只复验已有结果可从 `evaluate7.py` 开始，再依次更新报告及清单。

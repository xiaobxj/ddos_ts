# 第六轮：模型大小、正则化与训练历史稳定性

本目录延续第五轮数据，固定原始 OHLCV 输入和联合预测目标，测试减小宽度、增加 dropout、增加权重衰减和组合修改。实验范围及预算在训练前写入 [protocol.json](protocol.json)。此前各轮结果保持原样。

主实验：五组模型 × 三个年度折 × 三个种子，每次训练 20 轮，保留第 2、5、10、20 轮。

历史敏感性：分别删除最早、中间、最近约 20% 的训练标签样本块，五组模型均使用同一固定种子训练 10 轮。重复保留样本以匹配完整历史的更新次数。这是样本组成压力测试，不是清除全部相关底层行情的实验。

最终报告：[第六轮测试报告](results/第六轮测试报告.md)。所有日期均为已查看过的历史验证，不构成未使用过的新样本确认。

在项目根目录使用原有独立环境运行。`prepare6.py` 和 `train6.py` 会拒绝覆盖已有实验，请保留本轮证据；如需重跑，应另建目录。

```powershell
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v6\prepare6.py
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v6\test_contracts6.py
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v6\train6.py --phase main
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v6\train6.py --phase blocks
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v6\evaluate6.py
.\research_v4\.venv_gpu\Scripts\python.exe .\research_v6\verify6.py
python .\research_v6\figures6.py
python .\research_v6\report6.py
python .\research_v6\check_delivery6.py
```

只复验既有结果时，从 `evaluate6.py` 开始即可。模型与训练依赖原有 GPU 环境；图表和报告可由默认 Python 的 pandas、numpy、matplotlib 生成。无需安装新依赖。

`common6.py` 明确引用第五轮输入文件；本地 `cache` 仅保存三折样本掩码。`results/preparation_manifest.json` 冻结协议、来源代码、输入和之前 1353 个证据文件的哈希。完整检查点、逐种子预测、误差和样本删除定义均保存在 `results`。

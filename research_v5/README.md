# 第五轮：训练能力、连续表示与辅助预测目标

结果和图表见 [第五轮测试报告](results/第五轮测试报告.md)，模型回放及来源核验见 [verification.json](results/verification.json)。

本轮按 [冻结协议](protocol.json) 完成两层检查：

1. 固定 32 个 2017 年底前已完成标签的真实训练窗口，比较原始 OHLCV、Kronos 量化坐标、同维度量化前连续坐标的拟合能力。每种表示比较原收益读出、非零初始化、OHLCV 辅助目标，并以 MLP 拟合真实／打乱收益标签作容量对照。
2. 仅使用原始 OHLCV，在 2018–2020 年按时间先后做三折年度验证，比较三种训练设置的 2、5、10、20 轮结果。所有配置都完整运行，不根据小样本测试淘汰模型或扩大预算。

预训练编码器包含更晚的训练数据，因而早年样本上的编码结果只用于“能否记住训练样本”的诊断，不能用于早年预测验证或选参。连续表示和量化表示同为 20 维，每根 K 线向量范数为 1；前者保留取符号前的幅度信息。

在 D:/ddos_v3 工作目录，使用现有 research_v4/.venv_gpu/Scripts/python.exe 顺序运行 prepare.py、test_contracts.py、capacity.py、validate.py、evaluate.py、verify.py（路径均加 research_v5/ 前缀）。report.py 使用带 matplotlib 的默认 Python；它读取结果完成后撰写、仅用于解释的 results/findings.json。check_delivery.py 用默认 Python 进行最终文件校验。

输入准备、模型和结果都保存在本轮目录。复验已保存模型使用 verify.py；不要在原目录直接重复准备和训练，重新实验应新建版本目录。前四轮数据、模型、报告及虚拟环境保持原样。

results/preparation_first_pass_manifest.json 是接口检查完成前的初版准备记录；训练及最终复验使用 results/preparation_manifest.json 对应的最终缓存与源码清单。辅助头随机数作用域和显式模块导入的修正均在模型训练前完成。

本轮只报告拟合诊断和历史验证，不重新计算 2024–2026 年择时收益。验证结果属于已观察历史上的候选选择证据，不是新留出集的独立确认。辅助任务是本轮明确制定的未来五日 OHLCV 目标，并非声称精确重现原文的 OHLCAV 训练。

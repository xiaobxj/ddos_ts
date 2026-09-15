# 第四轮：Crossformer／分段／Kronos 编码消融

本目录运行冻结的七组比较：原始 OHLCV 或官方 Kronos tokenizer 编码，各比较固定、自适应、打乱分段，并增加同架构随机 tokenizer 对照。下游全部使用加入补零掩码、分段几何特征和收益读出层的官方 Crossformer；属于明确接口后的方法组合实验。

已完成的数值、判断和图表见 [第四轮测试报告](results/第四轮测试报告.md)，最终复验见 [verification.json](results/verification.json)。

实验协议见 [protocol.json](protocol.json)。共同窗口为 2024-07-05 至 2026-08-21 的 103 个周五信号，最后在 2026-08-31 开盘结算。训练轮数只由原始数值固定分段模型在 2018–2020 年验证期选择，然后固定给全部模型和种子。已看过的历史仅用于探索性开发，不是全新留出集。

## 运行

在 D:/ddos_v3 工作目录，使用本轮隔离的 research_v4/.venv_gpu/Scripts/python.exe 依次运行：

1. research_v4/prepare.py：核验冻结来源，逐窗口编码、分段和缓存。
2. research_v4/train.py：先选训练轮数，再完成全部滚动训练和预测。
3. research_v4/evaluate.py：统一收益、交易成本、分块 bootstrap 和配对比较。
4. research_v4/verify.py：运行九项测试、窗口重编码、回放 63 个开发期检查点、核对标签及净值。
5. research_v4/diagnostics.py：用默认 Python 汇总冻结预测在折内／折间的变化及训练损失。
6. research_v4/report.py：用带 matplotlib 的默认 Python 生成中文报告、PNG／SVG 图表和交付校验清单。
7. research_v4/check_delivery.py：用默认 Python 核对交付哈希、报告链接、文本和关键结论的数值来源。

第六步还读取结果完成后撰写的 results/findings.json。它仅保存结论文字，不参与训练或计算指标。不要在原目录直接重复步骤 1–2，以免覆盖冻结输出；复验模型使用步骤 4，重新实验应另建版本目录。

## 运行环境与来源

- 本轮独立 Python 3.13 虚拟环境，torch 2.6.0+cu124；沿用电脑现有 NVIDIA 驱动和 RTX 4060 Laptop GPU。
- 虚拟环境通过 --system-site-packages 读取已有 Python 3.13 数值计算依赖；新增包安装在本轮虚拟环境。精确 Python、numpy、pandas、torch、设备和源码哈希保存在训练清单。
- Crossformer 官方源码固定于 c10c8eadb153d1dd9798250967747ca3ebb81383，见 vendor/Crossformer；其许可证保留在仓库中。
- 复用第三轮 Kronos 官方源码和 LFS 核验过的 tokenizer 权重；不下载新数据，不改动前三轮证据。
- 冻结 tokenizer 每个历史窗口单独编码；预训练与随机编码对照都固定参数，只有下游 Crossformer 和新增读出层训练。

本文中的 10 bp 是每次单边仓位变化的假设总摩擦，不是已核实的交易所费率或券商报价。标的为沪深 300 价格指数，收益只是下一开盘价的多头／空仓代理回测，不是 ETF 可实现净收益。

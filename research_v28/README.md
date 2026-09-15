# 固定窗口与季度 walk-forward

自然 20 遍的 3/5 年窗口，年度和季度从头重训；保留 R26/R27 冻结对照。

GPU Python：`research_v4/.venv_gpu/Scripts/python.exe -B -u research_v28/run28.py prepare contract train score evaluate verify report`。

报告独立使用已有 Python 3.14 绘图库。全部源代码、协议、输入和 4,002 个旧文件在新训练前冻结。共享年末模型只拟合一次；本轮 138 个神经网络和 552 个分类解全部完成后才评分。

结果见 `results/第二十八轮测试报告.md`，逐周模型分配见 `results/routing.csv`。所有结果仍为已查看历史上的研究比较。

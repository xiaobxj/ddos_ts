# 市场状态触发模型更新

固定一条 60 日平均市场状态变化、第 90 百分位阈值、季末检查的规则，保留年末固定更新。使用上一轮 5 年窗口、自然 20 遍的冻结模型；本轮不新增网络或分类训练。

运行：`research_v4/.venv_gpu/Scripts/python.exe -B -u research_v29/run29.py prepare contract score evaluate verify report`。报告使用已有 Python 3.14 绘图库。视觉复核后单独运行 `delivery29.py`，避免对尚在写入的日志做交付哈希。

协议、源代码和旧证据先冻结，完整触发与逐周模型选择先生成，随后才补算缺失预测和评分。所有历史结果已经被查看，不能视为新盲测。

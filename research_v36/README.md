# 第三十六轮：新增成员的市场状态与覆盖诊断

四个粗状态：60日趋势符号×年度20日波动中位数。仅年度成熟训练样本定边界；不重新训练预测模型，不调整权重。保持 R35 全部四组概率与操作分解。

运行 `research_v4/.venv_gpu/Scripts/python.exe -B -u research_v36/run36.py prepare contract calibrate diagnose verify report`。图像审查后单独运行 `delivery36.py`。

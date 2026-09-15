# 第 26 轮：后续年份历史扩展

用户于 2026-09-12 授权执行已整理的后续年份方案。本轮只比较学习分支：R19、R23、R25，以及 R18 加性、原 MSE、训练上涨频率对照；未启动新的市场状态模型或实盘操作。

使用现有 `research_v4/.venv_gpu/Scripts/python.exe`，按 prepare → contract → train → score → evaluate → verify → report 执行。视觉检查通过后单独运行 delivery。全部训练结束后才生成下一年度评分，不根据中间年度结果修改协议。

每折上游重新训练，固定 18 次神经网络训练、72 次分类拟合及 272 周评分。后续年份在第 1–3 轮已有查看记录，因此结果标记为历史扩展检验。既有 3,672 份研究证据保留。

查看 `results/第二十六轮测试报告.md` 获取结果，`protocol.json` 获取冻结口径，`results/verification.json` 和 `results/delivery_manifest.json` 获取复核与交付哈希。

绘图环境说明：GPU Python 缺少 matplotlib。最终报告通过已有默认 Python 执行 `python -B research_v26/reporting/report26_portable.py`，适配器执行原冻结报告正文并记录自身哈希。原报告错误日志保留；模型、评分和统计代码保持冻结。结果生成后应先查看图表，再运行交付冻结。

首次预检查的 24 份文件及保全记录存于 `research_v26_attempt1`。总计保留 3,697 份既有文件。`results/结果解读与下一步.md` 是结果查看后的研究判断，尚未执行新的训练窗口或融合实验。

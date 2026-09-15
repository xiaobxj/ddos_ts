# 第 26 轮：后续年份历史扩展

用户于 2026-09-12 授权执行已整理的后续年份方案。本轮只比较学习分支：R19、R23、R25，以及 R18 加性、原 MSE、训练上涨频率对照；未启动新的市场状态模型或实盘操作。

使用现有 `research_v4/.venv_gpu/Scripts/python.exe`，按 prepare → contract → train → score → evaluate → verify → report 执行。视觉检查通过后单独运行 delivery。全部训练结束后才生成下一年度评分，不根据中间年度结果修改协议。

每折上游重新训练，固定 18 次神经网络训练、72 次分类拟合及 272 周评分。后续年份在第 1–3 轮已有查看记录，因此结果标记为历史扩展检验。既有 3,672 份研究证据保留。

查看 `results/第二十六轮测试报告.md` 获取结果，`protocol.json` 获取冻结口径，`results/verification.json` 和 `results/delivery_manifest.json` 获取复核与交付哈希。

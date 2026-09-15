本目录是 603986 兆易创新的 H1／H5 日频预测研究。请先看「快慢模型测试报告.md」和「历史问题核对.md」。

H1：下一交易日收盘涨跌；H5：未来第 5 个交易日收盘涨跌。两者每天使用最新 125 根日线生成输入；年度滚动训练与每天更新输入是两件事。这里已经算完 2023 年至 2026 年 9 月 15 日的固定历史回放。

研究数据和结果保存在本目录，原指数程序、前瞻账本与个股 v1 保持原样。打开报告不会自动下载数据。本版本没有安装自动任务，也没有接入原「打开每周预测.cmd」。

使用 GPU Python 3.13 解释器：
`D:\ddos_v3\research_v4\.venv_gpu\Scripts\python.exe -B -X utf8`

已完成的流水线为 prepare.py → train.py --h 1／--h 5 → score.py → verify.py models／outputs → audit.py → deliver.py。两个训练进程按周期隔离。已完成的拟合必须先通过哈希核对才可复用；评分和报告不覆盖旧结果。若要检验下一组新规则，应建立新研究目录并先冻结方案，不直接编辑本目录参数后覆盖成绩。

文件说明：
- protocol.json、results/freeze.json：训练前规则与源码／输入哈希。
- results/evaluation_implementation_freeze.json：评分前实现与数值容差。
- results/fits：84 个网络的成员、曲线、168 个检查点、头系数和原始概率缓存。
- results/predictions_h1.csv／predictions_h5.csv：全部逐日方案概率，matured=False 不计分。
- results/latest_probabilities.json：9 月 15 日输入的所有模型输出；按实际计算时刻标记。
- results/metrics.csv、year_metrics.csv、bootstrap.csv：统一样本评价。
- results/prior_problem_coverage.json：前 51 轮问题的逐项覆盖和未重测边界。
- results/delivery_manifest.json：最终交付文件哈希。

主模型预先固定为 mse_5y_e20.vol.U。其余候选是对照，不是可在结果最好时自动切换的投资策略。原生收益回归分数独立保存在 native_returns 中，不直接伪装成概率。

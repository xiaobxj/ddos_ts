# 沪深300反向DTW分段研究

已运行的研究代码、冻结数据和报告均位于本目录。

先读 [方法测试报告](results/方法测试报告.md)。本项目测试核心分段假设；未宣称复现原文的Kronos + Crossformer完整组合。

依赖：Python 3.13、numpy、pandas、scipy、numba、requests、matplotlib。

```powershell
python research/download_tencent.py
python research/run_experiment.py
python research/evaluate.py
python research/test_invariants.py
python research/verify_results.py
python research/make_report.py
```

`download_tencent.py`优先复用原始JSON快照，不自动刷新已有数据。`download_data.py`是未能完成的东方财富六列入口；本轮正式协议与结果均使用腾讯OHLCV。`protocol_six_channel_draft.json`只保留前期数据方案，不是当前执行配置。冻结协议为`protocol.json`。

如改变算法、数据范围、参数或测试规则，请建立新结果目录及新协议，避免覆盖本轮证据。11组逐周预测、全部参数选择和各年P0都在`results/`。

# 第二轮方法测试

完整结果见 [第二轮测试报告](results/第二轮测试报告.md)。

范围：实际持仓收益标签、经济与多尺度表示、3年／5年／扩展历史和年度／季度更新；另含概率与固定阈值诊断。12个建模方案、19组最终对照；2021—2026区间为已查看历史的开发验证。

复用`../research/data/`及原有年度P0，不改写上一轮文件。需要Python 3.13及numpy、pandas、scipy、numba；绘图需要matplotlib。

从项目根目录运行：

```powershell
py -3.13 research_v2/test_core.py
py -3.13 research_v2/run.py
py -3.13 research_v2/evaluate.py
py -3.13 research_v2/verify.py
python research_v2/report.py
```

本机Python 3.13提供数值依赖，默认Python 3.14提供matplotlib。单一完整环境也可全部用`python`执行。`run.py`会覆盖本轮`results/`；改变协议时应建立新目录，不要覆盖当前证据。

`protocol.json`先于运行创建；选参记录在`selection.json`。本轮并未实现完整Kronos＋Crossformer，也未使用交易账户。

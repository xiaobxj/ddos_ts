# 第三轮：模型能力与跨指数信息

报告：[第三轮测试报告](results/第三轮测试报告.md)。

冻结范围：Ridge／小型 HistGradientBoostingRegressor、同样本跨指数信息对照、官方 Kronos-small 冻结推理。传统模型完整区间 272 周；Kronos 按论文训练截止声明，比较 2024 年 7 月后共同的 103 周。全部属于已经查看历史的开发研究。

数据复用前两轮快照。运行需要 numpy、pandas、scipy、scikit-learn、threadpoolctl；Kronos 另需 torch、huggingface_hub、einops、safetensors、tqdm、pyyaml。独立绘图需要 matplotlib。

本机执行顺序（项目根目录）：

```powershell
python research_v3/download_models.py
py -3.13 research_v3/parity_reference.py
& 'D:\ddos\.venv\Scripts\python.exe' research_v3/test_contract.py
& 'D:\ddos\.venv\Scripts\python.exe' research_v3/run_local.py
& 'D:\ddos\.venv\Scripts\python.exe' research_v3/run_kronos.py
& 'D:\ddos\.venv\Scripts\python.exe' research_v3/evaluate.py
& 'D:\ddos\.venv\Scripts\python.exe' research_v3/verify.py
python research_v3/report.py
```

`parity_reference.py` 使用原有 Python 3.13／numba 环境导出第二轮特征与 Ridge 的独立参照；主实验使用既有 Python 3.14 CPU torch 环境。本轮补充包仅安装到 `vendor_py/`，由脚本前置搜索路径载入；精确版本见运行清单。官方仓库保存在 `vendor/Kronos/`，源码和权重 commit 固定于协议中，模型文件 SHA256 与首次上传版本核对。

必要运行包版本另列于 `requirements-runtime.txt`；本机实际 `torch.__version__` 为 `2.14.0+cpu`。绘图与导出第二轮独立参照使用上述单独环境，未混入主推理环境。

运行脚本会覆盖本轮同名产物。改变方案应创建下一轮目录，保留当前证据。`run_kronos.py` 每个日期保存进度，但重新启动时从头按同样种子确定性重算；没有自动跳过或根据回报挑选路径。

`common.py` 移植第二轮标签／特征／Ridge 函数以解除推理环境对 numba 的依赖；检查覆盖独立数值一致性、未来扰动不变性、时间边界、缺失数据、模型权重和成本核算。

Kronos 预测路径 CSV 为官方 float32 输出的十进制往返表示。独立复算时先按 float32 读取，再按原始 float32 求路径均值，最后将两个预测 open 转为 float64 计算收益；`verify.py` 对全部路径均值做逐位一致性检查。主预测与收益指标 CSV 为 float64。若把路径直接读成 float64，会产生约 1e-7 以内的收益复算差异。

这些程序不连接任何交易账户，不包含部署或下单功能。Crossformer 及神经自适应分段不在本轮范围内。

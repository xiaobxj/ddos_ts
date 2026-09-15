# 603986 独立个股研究版

从零训练个股参数，沿用原指数研究的网络和固定滚动方法。主结果见[建模结果](建模结果.md)。这不是向个股搬用指数的上涨概率，也不是已接入实盘的选股程序。

原始行情与权益调整、缺失日掩码、训练成员、模型、逐周预测及验证回执均单独保存。没有修改原有研究或指数前瞻记录。第一次下载曾发现接口对长区间静默截断，最终改为逐年请求并核对重叠数据。两个训练前准备问题及修复记录保存在 `results/*attempt*.json`。

代码流程（在项目根目录运行，使用已有 GPU 环境）：

```powershell
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/download.py
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/prepare.py
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/contract.py
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/train.py
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/verify.py
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/score.py
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/verify.py scoring
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/deliver.py
python -B -X utf8 stock_model_603986_v1/plot_results.py
& research_v4/.venv_gpu/Scripts/python.exe -B -X utf8 stock_model_603986_v1/finalize.py
```

以上为本次构建顺序，训练和评分输出使用排他创建，故已完成目录不能原地重跑覆盖。绘图使用装有matplotlib的桌面Python。新增数据、股票、预测周期或实验设定应新建版本并重新冻结协议。数据截点固定为2026年9月14日；本版没有接入每周自动下载或四周预测。

训练依赖同一项目内的既有数值模块，冻结清单记录其路径与 SHA256；不能只复制本目录后假定完全独立运行。所有历史检验结果均不是过去实际发布的预测记录。

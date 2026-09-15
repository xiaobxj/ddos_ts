# 第三十八轮：新增日样本的模型偏差能否迁移到后续周信号

固定 R37 模型与预测、R36 年度状态边界，比较新增全部日样本、新增周五子集和随后季度周信号的校准残差、样本组成和排序能力。检查执行标签区间重叠，并对原 R37 概率作 Brier 精确分解。不新增训练、特征推理、预测方案或显著性检验，不据此调整启用门槛。

运行：`research_v4/.venv_gpu/Scripts/python.exe -B -u research_v38/run38.py prepare contract diagnose verify report`。图像和结果解读复核后单独运行 `delivery38.py`。协议与八个源文件在诊断前冻结；5,428 个旧文件保留。

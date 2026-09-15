# 年度神经网络固定，季度只重拟合分类端

本轮固定 18 个年度神经网络，季度只重估五年窗口内的分类端缩放、监督方向、交互投影和系数。新增 51 组训练、204 个分类头、102 个残差投影；年末分类端直接复用，第一季度必须与年度方案相同。

已有训练和测试特征保留原值，仅补算缺失日样本的特征。每次拟合仅接收当时五年窗口和成熟标签内的输入。所有历史结果已被查看，本轮不是新盲测。

运行：`research_v4/.venv_gpu/Scripts/python.exe -B -u research_v32/run32.py prepare contract extract fit score evaluate verify report`。

完成图表视觉检查后，单独运行 `delivery32.py` 冻结交付清单，避免哈希仍在写入的日志。

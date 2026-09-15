# 第三十五轮：滚动训练成员加入与移出的受控对照

年度 A；只加入 A∪Q；只移出 A∩Q；同时操作 Q（复用 R34 固定年度变换、100% 系数更新）。各年重新定义 A。只改分类系数拟合成员；旧信息仍存在于固定年度神经网络和变换中。所有历史已查看。

先冻结后运行：`research_v4/.venv_gpu/Scripts/python.exe -B -u research_v35/run35.py prepare contract fit score evaluate diagnose verify report`；图像审查后单独运行 `delivery35.py`。

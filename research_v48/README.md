第48轮固定拆分原等权与第47轮时间加权系数：原斜率＋加权截距、加权斜率＋原截距。保留同一年度、种子和特征坐标，144条组合参数逐元素复用；无新增拟合、神经推理或校准。

对照原年度、完整时间加权、原四状态季度，固定72项探索性比较。另报告四组合的条件影响与对称代数分摊，区分logit可加性和概率／损失非线性；不作经济因果归因。原28条历史与6,143个旧文件保留，全部历史已查看。

运行 `research_v4/.venv_gpu/Scripts/python.exe -B -u research_v48/run48.py prepare contract assemble score evaluate verify report`。图表及解读核对后单独运行delivery48.py。全部10个源文件需在准备前就绪。

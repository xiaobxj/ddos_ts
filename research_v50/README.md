第50轮已接通季度／年度参数更新与前瞻预测器。继续使用第49轮冻结的五套方案、原种子、52个周五时点和42项主要比较，原预测账本也继续使用。新的入口为run50.py；run49.py及其交付保持原样。

在D:/ddos_v3目录查看当前状态：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py status
```

每个有完整周五日线的信号日，在北京时间18:00至23:00之间采集并记录：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py fetch
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py record
```

请先确认fetch成功。record会再次检查当天快照、参数的年度／季度截止、验证凭据和写入时点。没有当天周五日线、窗口已过、参数缺失或已提交过预测时会拒绝；不支持历史时间覆盖。

季度截止数据可用后，更新下一季度参数。第一次是2026-09-30，北京时间18:00之后，且采集快照至少覆盖该截止日期：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py fetch
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py refresh --cutoff 2026-09-30
```

2026年末使用同一入口；Dec31会触发原三个种子的年度20遍重训、等权／时间加权分类头拟合、状态边界更新，并按原规则将第一季度状态修正归零：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py refresh --cutoff 2026-12-31
```

同样必须先采集覆盖截止日期的快照。上述日期在当前尚未到达，提前运行会拒绝，不能把旧快照当作年末或季末数据。2027-03-31、2027-06-30继续使用同样的季度命令。截止日若没有交易数据，需要在后续快照覆盖该日后再更新；训练接口始终裁切到截止日。若漏做一季，应按截止日期顺序补建参数供后续更新使用；过期期间的预测仍不能补记。

每次refresh会创建一个独立构建目录，保存截止前缀、训练成员、模型及分类头、校准样本、验证结果。所有检查通过后才向原账本追加参数注册事件。失败构建保留failure.json且不注册；既有包不覆盖。同一季度不重复注册，已经出现预测的季度不允许事后替换参数。

读取成熟标签和累计结果：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py fetch
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py settle
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v50\run50.py evaluate
```

fetch、完整CSV的ingest --csv、标签规则及评估口径沿用第49轮；先前成功的主要评估结果固定保存，不反复择时计算。参数构建写入prospective_r50/builds，预测／快照／标签／参数注册事件写入原prospective_r49。研究工程回放文件保存在research_v50/results，不能当作真实前瞻记录。

目前已注册的是原第49轮参数包，供9月18日及9月25日符合条件的信号使用。9月30日和12月31日的新包尚未生成；对应生成、验证和注册流程已实现并通过历史回放及边界测试。没有安装自动定时任务或后台服务，仍需手动运行。

[结果与验证](results/结果解读与下一步.md) · [冻结实现协议](protocol.json) · [工程回放](results/verification.json) · [参数注册测试](results/lifecycle_verification.json) · [当前状态](results/startup_status.json)

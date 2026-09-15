第49轮已建立前瞻预测记录流程，并完成工程验证。已冻结五套方案、三个原种子和52个周五观察时点；已采集截至2026年9月14日的新行情快照。目前正式前瞻预测和成熟标签均为0条，首次候选信号日为2026年9月18日。

使用原研究Python，在D:/ddos_v3目录执行：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v49\run49.py status
```

周五北京时间18:00至23:00之间，先采集完整日线，再提交当周预测：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v49\run49.py fetch
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v49\run49.py record
```

只有当天的完整周五数据可以提交；节假日没有周五日线时，不用周四代替。窗口之外、数据缺失、重复预测或参数到期都会拒绝写入。不支持补录过去日期或通过命令行指定模拟时间。若依次手动执行两条命令，应确认fetch成功后再执行record；record本身也检查快照日期与参数。

有新的完整日线后，可读取成熟标签并查看累计结果：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v49\run49.py fetch
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v49\run49.py settle
& .\research_v4\.venv_gpu\Scripts\python.exe -B -u research_v49\run49.py evaluate
```

使用同一来源的完整本地CSV时，可以把fetch替换为ingest --csv <路径>。所需列为date/open/high/low/close/volume，日期升序、唯一；必须完整保留原2010年至2026年8月31日冻结前缀。采集时间由程序现场记录，CSV文件中的时间说明不会用作前瞻时间证明。

当前程序是手动CLI，没有安装定时任务或后台服务，不会在关机或无人调用时自行预测。首个参数包使用2025年末年度模型及2026年6月末季度修正，仅适用到2026年9月30日。10月及之后需要按冻结的原R28/R39/R47配方生成并复核新参数包，再接入后续运行器。本轮没有实现未来季度／年度重训任务，也没有把旧季度参数延长一年。到期拦截意味着当前交付不是已完成全年自动滚动的服务。

不可变研究材料保存在research_v49；后续运行状态单独保存在同级prospective_r49。后者包含genesis.json、events和blobs；事件按序号与前一事件哈希关联。原始接口响应、规范化CSV、预测参数和标签来源都有哈希。成功提交的记录不会覆盖；异常中断残留writer.lock时需先核实是否仍有进程运行。本地哈希链不是第三方时间戳公证，不应宣称它能防止有权限的人重写整条证据链。

52个时点覆盖2026年9月18日至2027年9月10日。第13／26／39周可作描述性汇总，不做提前显著性筛选；主要评估最早2027年10月1日，需至少40个完整成熟的共同样本且无尚未成熟的已提交预测。40仅为操作门槛，不代表已经达到统计功效要求。缺失时点不补足或替换。42项固定比较对方向误差与Brier使用8周区块、10,000次重采样和统一Holm校正。缺失日历周保留在重采样轴上。

[结果与运行状态](results/结果解读与下一步.md) · [冻结协议](protocol.json) · [工程验证](results/verification.json) · [当前参数包](results/bootstrap_package.json) · [首次数据采集验证](results/snapshot_validation.json)

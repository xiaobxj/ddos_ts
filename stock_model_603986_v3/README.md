# 603986 快慢周期第三轮

用户入口：双击项目根目录的 **打开个股每日预测.cmd**。需联网，每个交易日北京时间 18:00 后运行。

- 自动下载日线；快为下一交易日，慢为第 5 个交易日；使用已验证参数，不是每日重训。
- 预测和到期结果保存在 `../stock_daily_603986`，不要删除或编辑该目录。重复运行不重复计数，漏跑不补录。
- 本参数包有效至 2026-09-30；此后可结算旧预测，新预测需下一季度验证包。
- [本轮报告](REPORT.md) / [结果页](../stock_daily_603986/latest.html)。数值仍为研究输出，未证明稳定概率优势。

运行环境：项目已有 `research_v4/.venv_gpu/Scripts/python.exe`，Python 3.13、PyTorch 2.6 CUDA 环境及当前 NVIDIA GPU。不要用系统 Python 启动每日预测。

命令行（不打开结果页）：

```powershell
& .\research_v4\.venv_gpu\Scripts\python.exe -B -X utf8 stock_model_603986_v3/daily3.py --no-open
```

研究源代码和模型已冻结；训练、评分脚本采用排他写入，不能在原结果目录覆盖重跑。新研究应另建版本。`results/delivery_manifest.json` 是交付清单，`results/daily_package.json` 绑定每日预测依赖。日志采用 PowerShell 重定向，部分为 UTF-16，读取时注意编码。

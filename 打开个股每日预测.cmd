@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 603986 每日快慢预测：自动下载日线、结算到期结果、保存今天预测。
echo 请在交易日北京时间 18:00 后运行。重复运行不会覆盖已有预测。
"%~dp0research_v4\.venv_gpu\Scripts\python.exe" -B -X utf8 "%~dp0stock_model_603986_v3\daily3.py"
echo.
pause

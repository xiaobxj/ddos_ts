@echo off
if not exist "%~dp0research_v4\.venv_gpu\Scripts\pythonw.exe" (
  echo Python environment not found. Please keep this launcher in the project folder.
  pause
  exit /b 1
)
start "" "%~dp0research_v4\.venv_gpu\Scripts\pythonw.exe" -B "%~dp0weekly_app\app.py"

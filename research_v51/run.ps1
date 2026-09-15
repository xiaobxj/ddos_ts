param([ValidateSet('plan', 'cycle')][string]$Command = 'plan')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectRoot 'research_v4\.venv_gpu\Scripts\python.exe'
& $PythonExecutable -B -u (Join-Path $PSScriptRoot 'run51.py') $Command
exit $LASTEXITCODE

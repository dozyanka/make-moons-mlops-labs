@echo off
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv was not found.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "defense_demo\run_all.py"
echo.
pause

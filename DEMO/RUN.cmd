@echo off
cd /d "%~dp0\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0runner.ps1"
if errorlevel 1 (
    echo.
    echo DEMO finished with an error.
)
echo.
pause
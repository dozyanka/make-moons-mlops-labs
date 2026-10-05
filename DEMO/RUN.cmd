@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0runner.ps1" -AutoSetup
if errorlevel 1 (
    echo.
    echo DEMO finished with an error.
)
echo.
pause

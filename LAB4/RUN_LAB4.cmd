@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_LAB4.ps1"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" echo LAB4 FAILED: exit code %EXITCODE%
echo.
pause
exit /b %EXITCODE%

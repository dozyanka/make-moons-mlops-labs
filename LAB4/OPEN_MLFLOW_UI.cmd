@echo off
setlocal
cd /d "%~dp0.."
set "MLFLOW_ALLOW_FILE_STORE=true"
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found. Run LAB4\RUN_LAB4.cmd first.
  pause
  exit /b 1
)
echo MLflow UI: http://127.0.0.1:5000
echo Open the address in your browser. Press Ctrl+C to stop the service.
".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri "%CD%\mlruns" --port 5000
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo MLflow UI failed with exit code %RC%.
  pause
)
exit /b %RC%

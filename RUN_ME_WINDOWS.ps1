$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[all]"
& .\.venv\Scripts\python.exe scripts\run_all.py
Write-Host "Done. To inspect MLflow: .\.venv\Scripts\mlflow.exe ui --backend-store-uri .\mlruns --port 5000"

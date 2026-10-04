param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(1,8)]
    [int]$Lab
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[all]"
& .\.venv\Scripts\python.exe scripts\prepare_data.py

$Prereqs = @{
    1 = @(1)
    2 = @(2)
    3 = @(2,3)
    4 = @(2,4)
    5 = @(5)
    6 = @(2,6)
    7 = @(2,6,7)
    8 = @(1,2,3,4,5,6,7,8)
}

foreach ($N in $Prereqs[$Lab]) {
    Write-Host "=== Running LAB$N ==="
    & .\.venv\Scripts\python.exe "scripts\run_lab$N.py"
}

Write-Host "=== Pytest ==="
& .\.venv\Scripts\python.exe -m pytest --cov=moons_lab --cov-report=term-missing
Write-Host "LAB$Lab completed. Open reports\LAB$Lab and SUBMISSION_MAP.md."

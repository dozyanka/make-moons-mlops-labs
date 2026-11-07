# LAB4 standalone Windows runner. No admin rights required. No lab1-3 retraining.
param([switch]$SkipInstall, [switch]$SkipTests)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
# The specification requires mlruns/; MLflow 3.16+ blocks this legacy
# file backend unless explicitly enabled. The setting is scoped to this process.
$env:MLFLOW_ALLOW_FILE_STORE = "true"
Write-Host "=== Make Moons / LAB4 ==="
Write-Host "MLflow tracking: local mlruns/ (MLFLOW_ALLOW_FILE_STORE=true)"
Write-Host "Repository: $RepoRoot"
$Venv = Join-Path $RepoRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $created = $false
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($version in @("-3.12", "-3.13", "-3.11")) {
            try {
                & py $version -m venv $Venv 2>$null
                if ($LASTEXITCODE -eq 0 -and (Test-Path $Python)) { $created = $true; break }
            } catch {
                # This version is absent; try the next candidate.
                continue
            }
        }
    }
    if (-not $created) {
        $fallback = Get-Command python -ErrorAction SilentlyContinue
        if (-not $fallback) { throw "Python 3.11+ not found. Install Python from python.org." }
        & python -m venv $Venv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python)) { throw "Could not create .venv" }
    }
}
if (-not $SkipInstall) {
    Write-Host "Installing Lab4 dependencies (NumPy, scikit-learn, MLflow, pytest)..."
    & $Python -m pip install -e ".[tracking,test]"
    if ($LASTEXITCODE -ne 0) { throw "Python dependencies installation failed" }
}
Write-Host "Validating Python and MLflow..."
& $Python -c "import sys, mlflow, numpy, sklearn; assert sys.version_info >= (3,11); print('Python:',sys.version); print('MLflow:',mlflow.__version__); print('NumPy:',numpy.__version__); print('sklearn:',sklearn.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Python / MLflow import failed" }

Write-Host "Running LAB4 (only LAB4; existing LAB1-3 data/models stay unchanged)..."
& $Python scripts\run_lab4.py
if ($LASTEXITCODE -ne 0) { throw "LAB4 experiment failed; see console message" }

# Explicit isolation avoids inherited ACLs on
# %TEMP%\pytest-of-<user> and project .pytest_cache (WinError 5).
# It also makes successive elevated / non-elevated runs independent.
function New-Lab4TestWorkspace {
    $parents = @((Join-Path $RepoRoot "reports\LAB4\.pytest_runtime"))
    if ($env:LOCALAPPDATA) {
        $parents += (Join-Path $env:LOCALAPPDATA "make-moons-mlops-labs\lab4-tests")
    }
    foreach ($parent in $parents) {
        $candidate = Join-Path $parent ("run-" + [guid]::NewGuid().ToString("N"))
        try {
            New-Item -ItemType Directory -Path $candidate -Force -ErrorAction Stop | Out-Null
            New-Item -ItemType Directory -Path (Join-Path $candidate "temp") -ErrorAction Stop | Out-Null
            # A writable workspace is necessary for both pytest and coverage.
            $probe = Join-Path $candidate "write_probe.tmp"
            [System.IO.File]::WriteAllText($probe, "ok")
            Remove-Item -LiteralPath $probe -ErrorAction Stop
            return $candidate
        } catch {
            Write-Warning ("Test workspace is not writable: " + $candidate + "; trying fallback")
        }
    }
    throw "Could not create a writable LAB4 test workspace. Check write access to reports/LAB4 and LOCALAPPDATA."
}

if (-not $SkipTests) {
    Write-Host "Running LAB4 tests + coverage..."
    $CoverageLog = "reports\LAB4\pytest_coverage.log"
    $TestWorkspace = New-Lab4TestWorkspace
    $PytestBaseTemp = Join-Path $TestWorkspace "pytest-tmp"
    $PytestCache = Join-Path $TestWorkspace "pytest-cache"
    $OldTemp = $env:TEMP
    $OldTmp = $env:TMP
    $OldTmpDir = $env:TMPDIR
    $OldCoverageFile = $env:COVERAGE_FILE
    $TestSucceeded = $false
    try {
        $env:TEMP = Join-Path $TestWorkspace "temp"
        $env:TMP = $env:TEMP
        $env:TMPDIR = $env:TEMP
        $env:COVERAGE_FILE = Join-Path $TestWorkspace ".coverage"
        Write-Host "Isolated pytest workspace: $TestWorkspace"
        # --basetemp bypasses pytest-of-shata, -o bypasses .pytest_cache.
        # Instrument full LAB4 path: CLI, demo, shared config/data helpers and validator.
        & $Python -m pytest -q tests\test_validator.py tests\test_lab4_workflow.py tests\test_lab4_mlflow_tracking.py tests\test_lab4_runner.py tests\test_lab4_publication.py `
            "--basetemp=$PytestBaseTemp" -o "cache_dir=$PytestCache" `
            --cov=moons_lab.validator --cov=moons_lab.config --cov=moons_lab.data --cov=moons_lab.utils `
            --cov=scripts --cov=defense_demo --cov-report= 2>&1 | Tee-Object -FilePath $CoverageLog
        if ($LASTEXITCODE -ne 0) { throw "LAB4 tests failed; see $CoverageLog" }
        $Patterns = "*/src/moons_lab/validator.py,*/src/moons_lab/config.py,*/src/moons_lab/data.py,*/src/moons_lab/utils.py,*/scripts/run_lab4.py,*/defense_demo/lab4_demo.py"
        Write-Host "Coverage of all Lab4 components (including shared modules):"
        & $Python -m coverage report -m --include=$Patterns 2>&1 | Tee-Object -FilePath $CoverageLog -Append
        if ($LASTEXITCODE -ne 0) { throw "LAB4 coverage report failed" }
        Write-Host "Lab4-specific model, orchestration and demo coverage:"
        & $Python -m coverage report -m --include="*/src/moons_lab/validator.py,*/scripts/run_lab4.py,*/defense_demo/lab4_demo.py" 2>&1 |
            Tee-Object -FilePath $CoverageLog -Append
        if ($LASTEXITCODE -ne 0) { throw "LAB4 core coverage report failed" }
        & $Python -m coverage json -o reports\LAB4\coverage_details.json --include=$Patterns
        if ($LASTEXITCODE -ne 0) { throw "LAB4 JSON coverage report failed" }
        $TestSucceeded = $true
    } finally {
        # Windows PowerShell 5.1 Tee-Object defaults to UTF-16LE; GitHub
        # should receive a normal UTF-8 log even if pytest failed.
        try {
            & (Join-Path $PSScriptRoot "NORMALIZE_COVERAGE_LOG.ps1")
        } catch {
            Write-Warning ("Could not normalize LAB4 pytest log to UTF-8: " + $_.Exception.Message)
        }
        # Never leave a temporary environment variable change in the caller.
        $env:TEMP = $OldTemp
        $env:TMP = $OldTmp
        $env:TMPDIR = $OldTmpDir
        $env:COVERAGE_FILE = $OldCoverageFile
        if ($TestSucceeded) {
            Remove-Item -LiteralPath $TestWorkspace -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Write-Warning "Test failed; temporary diagnostic files were kept at: $TestWorkspace"
        }
    }
}
Write-Host "=== LAB4 COMPLETE ===" -ForegroundColor Green
Write-Host "Report: reports\LAB4\lab4_report.md"
Write-Host "Metrics: reports\LAB4\validator_metrics.csv"
Write-Host "Independent repeat: reports\LAB4\repeat_verification.json"
Write-Host "Coverage: reports\LAB4\pytest_coverage.log"
Write-Host "Quick demo: .venv\Scripts\python.exe defense_demo\lab4_demo.py"
Write-Host "MLflow UI: run LAB4\OPEN_MLFLOW_UI.cmd (includes file-backend opt-in)"

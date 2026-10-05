param(
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location $Root

Write-Host ""
Write-Host "========================================================================"
Write-Host "PREPARING PYTHON ENVIRONMENT"
Write-Host "========================================================================"

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$VenvDir = Join-Path $Root ".venv"

$venvWorks = $false
if (Test-Path $VenvPython) {
    try {
        & $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
        $venvWorks = ($LASTEXITCODE -eq 0)
    }
    catch {
        $venvWorks = $false
    }
}

if (-not $venvWorks) {
    if (Test-Path $VenvDir) {
        Write-Host "Removing an invalid or non-portable .venv..."
        Remove-Item $VenvDir -Recurse -Force
    }

    $BaseExe = $null
    $BaseArgs = @()

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.13 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $BaseExe = "py"
            $BaseArgs = @("-3.13")
        }
        else {
            & py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
            if ($LASTEXITCODE -eq 0) {
                $BaseExe = "py"
                $BaseArgs = @("-3")
            }
        }
    }

    if (-not $BaseExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $BaseExe = "python"
            $BaseArgs = @()
        }
    }

    if (-not $BaseExe) {
        $KnownPython = @(
            "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
            "C:\Program Files\Python313\python.exe",
            "C:\Python313\python.exe"
        )

        foreach ($Candidate in $KnownPython) {
            if (Test-Path $Candidate) {
                & $Candidate -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
                if ($LASTEXITCODE -eq 0) {
                    $BaseExe = $Candidate
                    $BaseArgs = @()
                    break
                }
            }
        }
    }

    if (-not $BaseExe) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw "Python 3.11+ was not found. Install Python 3.13 and run RUN_DEMO.cmd again."
        }

        Write-Host "Python was not found. Installing Python 3.13 with winget..."
        & winget install --id Python.Python.3.13 -e --source winget --accept-package-agreements --accept-source-agreements

        if ($LASTEXITCODE -ne 0) {
            throw "winget could not install Python 3.13."
        }

        $Candidate = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
        if (Test-Path $Candidate) {
            $BaseExe = $Candidate
            $BaseArgs = @()
        }
        else {
            throw "Python was installed but is not visible yet. Close this window and run RUN_DEMO.cmd again."
        }
    }

    Write-Host "Creating .venv..."
    & $BaseExe @BaseArgs -m venv ".venv"

    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv."
    }
}

if (-not (Test-Path $VenvPython)) {
    throw ".venv exists but python.exe was not found."
}

function Get-Sha256Hex([string]$Path) {
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    $stream = $null

    try {
        $stream = [System.IO.File]::OpenRead($Path)
        $hashBytes = $sha256.ComputeHash($stream)
        return ([System.BitConverter]::ToString($hashBytes)).Replace("-", "")
    }
    finally {
        if ($null -ne $stream) {
            $stream.Dispose()
        }
        $sha256.Dispose()
    }
}

$signatureParts = @()
foreach ($file in @("pyproject.toml", "requirements-lock.txt")) {
    $path = Join-Path $Root $file
    if (Test-Path $path) {
        $signatureParts += (Get-Sha256Hex $path)
    }
}

$dependencySignature = ($signatureParts -join ":")
$signatureFile = Join-Path $VenvDir ".instructor_demo_signature"
$installedSignature = ""

if (Test-Path $signatureFile) {
    $installedSignature = ([System.IO.File]::ReadAllText($signatureFile)).Trim()
}

if ($installedSignature -ne $dependencySignature) {
    Write-Host "Installing/updating project dependencies. The first run can take a few minutes..."

    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Could not update pip/setuptools/wheel."
    }

    & $VenvPython -m pip install -e ".[test,tracking,dl,report]"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install project dependencies."
    }

    [System.IO.File]::WriteAllText($signatureFile, $dependencySignature)
}
else {
    Write-Host "[OK] Dependencies are already up to date."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Write-Host ""
Write-Host "Checking libraries..."
& $VenvPython ".\DEMO\verify_environment.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependency verification failed. Repairing the environment..."

    & $VenvPython -m pip install -e ".[test,tracking,dl,report]"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not repair project dependencies."
    }

    & $VenvPython ".\DEMO\verify_environment.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Environment verification failed again."
    }
}

if ($SmokeTest) {
    & $VenvPython ".\DEMO\menu.py" --smoke-test
    if ($LASTEXITCODE -ne 0) {
        throw "DEMO smoke test failed."
    }
    exit 0
}

& $VenvPython ".\DEMO\menu.py"
if ($LASTEXITCODE -ne 0) {
    throw "DEMO menu exited with an error."
}
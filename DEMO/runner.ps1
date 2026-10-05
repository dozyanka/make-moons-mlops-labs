param(
    [switch]$AutoSetup,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location $Root

Write-Host ""
Write-Host "========================================================================"
Write-Host "ПОДГОТОВКА ОКРУЖЕНИЯ"
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
        Write-Host "Удаляется непереносимое или повреждённое .venv..."
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
            throw "Python 3.11+ не найден. Установите Python 3.13 и снова запустите RUN_DEMO.cmd."
        }

        Write-Host "Python не найден. Установка Python 3.13 через winget..."
        & winget install --id Python.Python.3.13 -e --source winget --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "winget не смог установить Python 3.13."
        }

        $Candidate = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
        if (Test-Path $Candidate) {
            $BaseExe = $Candidate
            $BaseArgs = @()
        }
        else {
            throw "Python установлен, но ещё не виден текущему процессу. Закройте окно и снова запустите RUN_DEMO.cmd."
        }
    }

    Write-Host "Создаётся .venv..."
    & $BaseExe @BaseArgs -m venv ".venv"
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось создать .venv."
    }
}

if (-not (Test-Path $VenvPython)) {
    throw ".venv создано, но python.exe не найден."
}

$ReadyFile = Join-Path $VenvDir ".instructor_demo_ready"

if (-not (Test-Path $ReadyFile)) {
    Write-Host "Устанавливаются зависимости проекта. На первом запуске это может занять несколько минут..."

    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось обновить pip/setuptools/wheel."
    }

    & $VenvPython -m pip install -e ".[test,tracking,dl,report]"
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось установить зависимости проекта."
    }

    Set-Content $ReadyFile "ready" -Encoding ascii
}
else {
    Write-Host "[OK] Локальное окружение уже подготовлено."
}

Write-Host ""
Write-Host "Проверка библиотек..."
& $VenvPython ".\DEMO\verify_environment.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Проверка выявила отсутствующие зависимости. Выполняется восстановление..."
    & $VenvPython -m pip install -e ".[test,tracking,dl,report]"
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось восстановить зависимости."
    }

    & $VenvPython ".\DEMO\verify_environment.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Окружение не прошло повторную проверку."
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
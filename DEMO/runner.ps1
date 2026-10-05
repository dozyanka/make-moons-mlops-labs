param(
    [switch]$AutoSetup
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$DemoDir = $PSScriptRoot
$Root = [System.IO.Path]::GetFullPath((Join-Path $DemoDir ".."))
Set-Location $Root

$script:VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$script:CurrentLog = $null

function Header([string]$Text) {
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host $Text
    Write-Host "========================================================================"
}

function Fail([string]$Text) {
    Write-Host ""
    Write-Host "[ОШИБКА] $Text"
    throw $Text
}

function Test-PythonCandidate([string]$Exe, [string[]]$PrefixArgs = @()) {
    try {
        & $Exe @PrefixArgs -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Find-BasePython {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        if (Test-PythonCandidate "py" @("-3.13")) {
            return @{ Exe = "py"; Args = @("-3.13") }
        }
        if (Test-PythonCandidate "py" @("-3")) {
            return @{ Exe = "py"; Args = @("-3") }
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        if (Test-PythonCandidate "python") {
            return @{ Exe = "python"; Args = @() }
        }
    }

    $known = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "C:\Program Files\Python313\python.exe",
        "C:\Python313\python.exe"
    )

    foreach ($candidate in $known) {
        if ((Test-Path $candidate) -and (Test-PythonCandidate $candidate)) {
            return @{ Exe = $candidate; Args = @() }
        }
    }

    return $null
}

function Install-PythonIfMissing {
    $base = Find-BasePython
    if ($base) {
        return $base
    }

    Header "PYTHON НЕ НАЙДЕН - АВТОУСТАНОВКА"

    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Fail "Python 3.11+ не найден, а winget недоступен. Установите Python 3.13 и повторите RUN.cmd."
    }

    Write-Host "Python 3.13 будет установлен через winget..."
    & winget install --id Python.Python.3.13 -e --source winget `
        --accept-package-agreements `
        --accept-source-agreements

    if ($LASTEXITCODE -ne 0) {
        Fail "winget не смог установить Python 3.13."
    }

    $base = Find-BasePython
    if (-not $base) {
        Fail "Python установлен, но текущий процесс его не видит. Закройте окно и снова запустите DEMO\RUN.cmd."
    }

    return $base
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
function Get-DependencySignature {
    $parts = @()

    foreach ($file in @("pyproject.toml", "requirements-lock.txt")) {
        $path = Join-Path $Root $file
        if (Test-Path $path) {
            $parts += (Get-Sha256Hex $path)
        }
    }

    return ($parts -join ":")
}

function Ensure-Environment([switch]$Force) {
    Header "ПОДГОТОВКА ОКРУЖЕНИЯ"

    $base = Install-PythonIfMissing

    $validVenv = $false
    if (Test-Path $script:VenvPython) {
        try {
            & $script:VenvPython -c "import sys; print(sys.version)" *> $null
            $validVenv = ($LASTEXITCODE -eq 0)
        }
        catch {
            $validVenv = $false
        }
    }

    if (-not $validVenv) {
        if (Test-Path (Join-Path $Root ".venv")) {
            Write-Host "Удаляется старое/перенесённое .venv..."
            Remove-Item (Join-Path $Root ".venv") -Recurse -Force
        }

        Write-Host "Создаётся новое .venv..."
        $baseExe = $base.Exe
        $baseArgs = @($base.Args)
        & $baseExe @baseArgs -m venv ".venv"
        if ($LASTEXITCODE -ne 0) {
            Fail "Не удалось создать .venv."
        }
    }

    $signature = Get-DependencySignature
    $signatureFile = Join-Path $Root ".venv\.instructor_demo_signature"
    $installedSignature = ""
    if (Test-Path $signatureFile) {
        $installedSignature = (Get-Content $signatureFile -Raw).Trim()
    }

    if ($Force -or ($installedSignature -ne $signature)) {
        Write-Host "Устанавливаются/обновляются зависимости проекта..."
        Write-Host "На первом запуске это может занять несколько минут."

        & $script:VenvPython -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) {
            Fail "Не удалось обновить pip."
        }

        & $script:VenvPython -m pip install -e ".[test,tracking,dl,report]"
        if ($LASTEXITCODE -ne 0) {
            Fail "Не удалось установить зависимости проекта."
        }

        Set-Content $signatureFile $signature -Encoding ascii
    }
    else {
        Write-Host "[OK] Зависимости уже установлены и pyproject.toml не менялся."
    }

    & $script:VenvPython ".\DEMO\verify_environment.py"
    if ($LASTEXITCODE -ne 0) {
        Fail "Проверка окружения не пройдена."
    }
}

function Start-Log([string]$Name) {
    $logDir = Join-Path $Root "demo_output\logs"
    New-Item -ItemType Directory -Force $logDir | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $script:CurrentLog = Join-Path $logDir "${Name}_${stamp}.log"
    "Run: $Name`r`nTime: $(Get-Date -Format o)`r`n" | Set-Content $script:CurrentLog -Encoding utf8
    Write-Host "Лог: $script:CurrentLog"
}

function Invoke-Python([string[]]$Arguments) {
    Write-Host ""
    Write-Host "> python $($Arguments -join ' ')"
    & $script:VenvPython @Arguments 2>&1 | Tee-Object -FilePath $script:CurrentLog -Append
    if ($LASTEXITCODE -ne 0) {
        Fail "Команда Python завершилась с ошибкой: $($Arguments -join ' ')"
    }
}

function Invoke-ExeLogged([string]$Exe, [string[]]$Arguments) {
    Write-Host ""
    Write-Host "> $Exe $($Arguments -join ' ')"
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $script:CurrentLog -Append
    if ($LASTEXITCODE -ne 0) {
        Fail "Команда завершилась с ошибкой: $Exe"
    }
}

function Save-ReproducedReport([string]$LabName, [scriptblock]$Body) {
    $official = Join-Path $Root "reports\$LabName"
    $destination = Join-Path $Root "demo_output\$LabName"
    $backupRoot = Join-Path $env:TEMP ("make-moons-" + $LabName + "-" + [Guid]::NewGuid().ToString("N"))
    $backup = Join-Path $backupRoot $LabName

    $hadOfficial = Test-Path $official

    # Important: keep the original report directory in place while the lab runs.
    # Some project scripts expect existing subfolders such as reports/LAB3/figures.
    if ($hadOfficial) {
        New-Item -ItemType Directory -Force $backupRoot | Out-Null
        Copy-Item $official $backup -Recurse -Force
    }
    else {
        New-Item -ItemType Directory -Force $official | Out-Null
    }

    # Lab 3 explicitly writes a plot into this folder.
    if ($LabName -eq "LAB3") {
        New-Item -ItemType Directory -Force (Join-Path $official "figures") | Out-Null
    }

    try {
        & $Body

        if (-not (Test-Path $official)) {
            Fail "$LabName не создала папку reports\$LabName."
        }

        if (Test-Path $destination) {
            Remove-Item $destination -Recurse -Force
        }

        New-Item -ItemType Directory -Force (Split-Path $destination -Parent) | Out-Null
        Copy-Item $official $destination -Recurse -Force

        Write-Host ""
        Write-Host "[OK] Новый результат сохранён отдельно:"
        Write-Host "     $destination"
    }
    finally {
        # Restore the exact official submitted reports after the reproduction run.
        if (Test-Path $official) {
            Remove-Item $official -Recurse -Force
        }

        if ($hadOfficial -and (Test-Path $backup)) {
            Copy-Item $backup $official -Recurse -Force
        }

        if (Test-Path $backupRoot) {
            Remove-Item $backupRoot -Recurse -Force
        }
    }
}

function Prepare-Data {
    Invoke-Python @(".\scripts\prepare_data.py")
}

function Run-Lab1 {
    Ensure-Environment
    Header "ЛР1 - ВОСПРОИЗВЕДЕНИЕ"
    Start-Log "LAB1"

    Save-ReproducedReport "LAB1" {
        Prepare-Data
        Invoke-Python @(".\scripts\run_lab1.py")
        Invoke-Python @(
            "-m", "pytest",
            ".\tests\test_data.py",
            ".\tests\test_eda.py",
            "-q"
        )

        if ((Test-Path ".git") -and (Test-Path ".\.venv\Scripts\pre-commit.exe")) {
            Invoke-ExeLogged ".\.venv\Scripts\pre-commit.exe" @("run", "--all-files")
        }
        else {
            Invoke-Python @(".\scripts\precommit_checks.py", "--all")
        }
    }

    Write-Host ""
    Write-Host "Оригинальные результаты: reports\LAB1"
    Write-Host "Новый прогон:             demo_output\LAB1"
}

function Run-Lab2 {
    Ensure-Environment
    Header "ЛР2 - ВОСПРОИЗВЕДЕНИЕ"
    Start-Log "LAB2"

    Save-ReproducedReport "LAB2" {
        Prepare-Data
        Invoke-Python @(".\scripts\run_lab2.py")
        Invoke-Python @(
            "-m", "pytest",
            ".\tests\test_data.py",
            ".\tests\test_metrics.py",
            ".\tests\test_ml.py",
            ".\tests\test_tracking_utils.py",
            "-q"
        )
    }

    Write-Host ""
    Write-Host "Оригинальные результаты: reports\LAB2"
    Write-Host "Новый прогон:             demo_output\LAB2"
}

function Run-Lab3 {
    Ensure-Environment
    Header "ЛР3 - ВОСПРОИЗВЕДЕНИЕ"
    Start-Log "LAB3"

    Save-ReproducedReport "LAB3" {
        Prepare-Data
        Invoke-Python @(".\scripts\run_lab3.py")
        Invoke-Python @(
            "-m", "pytest",
            ".\tests\test_data.py",
            ".\tests\test_metrics.py",
            ".\tests\test_dl.py",
            "-q"
        )
    }

    Write-Host ""
    Write-Host "Оригинальные результаты: reports\LAB3"
    Write-Host "Новый прогон:             demo_output\LAB3"
    Write-Host ""
    Write-Host "Если на ПК нет CUDA, основной прогон ЛР3 всё равно корректно работает на CPU."
    Write-Host "Сохранённая проверка CPU/GPU parity находится в reports\LAB3\device_parity.csv."
}

function Run-AllLabs {
    Header "ПОЛНОЕ ВОСПРОИЗВЕДЕНИЕ ЛР1-ЛР3"
    Run-Lab1
    Run-Lab2
    Run-Lab3

    Header "ИТОГ"
    Write-Host "ЛР1: PASS"
    Write-Host "ЛР2: PASS"
    Write-Host "ЛР3: PASS"
    Write-Host ""
    Write-Host "Официальные результаты сохранены без изменений в reports\."
    Write-Host "Новые результаты находятся в demo_output\."
}

function Run-QuickDemo {
    Ensure-Environment
    Header "БЫСТРАЯ ДЕМОНСТРАЦИЯ ГОТОВЫХ МОДЕЛЕЙ"
    Start-Log "QUICK_DEMO"
    Prepare-Data
    Invoke-Python @(".\defense_demo\run_all.py")
}

function Run-AllTests {
    Ensure-Environment
    Header "ТЕСТЫ ЛР1-ЛР3"
    Start-Log "TESTS"

    Invoke-Python @(
        "-m", "pytest",
        ".\tests\test_data.py",
        ".\tests\test_eda.py",
        ".\tests\test_metrics.py",
        ".\tests\test_ml.py",
        ".\tests\test_tracking_utils.py",
        ".\tests\test_dl.py",
        "-q"
    )
}

function Start-MLflowUi {
    Ensure-Environment
    Header "MLFLOW UI"

    $mlflow = Join-Path $Root ".venv\Scripts\mlflow.exe"
    if (-not (Test-Path $mlflow)) {
        Fail "mlflow.exe не найден."
    }

    if (-not (Test-Path (Join-Path $Root "mlflow.db"))) {
        Write-Host "mlflow.db пока отсутствует."
        Write-Host "Сначала запустите ЛР2/ЛР3, если хотите увидеть новые локальные runs."
        Write-Host "UI всё равно будет запущен."
    }

    Start-Process -FilePath $mlflow `
        -ArgumentList @("ui", "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5000") `
        -WorkingDirectory $Root

    Start-Sleep -Seconds 3
    Start-Process "http://127.0.0.1:5000"

    Write-Host "MLflow UI: http://127.0.0.1:5000"
}

function Open-Results {
    New-Item -ItemType Directory -Force (Join-Path $Root "demo_output") | Out-Null
    Start-Process explorer.exe (Join-Path $Root "reports")
    Start-Process explorer.exe (Join-Path $Root "demo_output")
}

function Show-Menu {
    Clear-Host
    Write-Host "========================================================================"
    Write-Host " MAKE-MOONS MLOPS LABS - ЗАПУСК ДЛЯ ПРОВЕРКИ"
    Write-Host "========================================================================"
    Write-Host ""
    Write-Host " 1. Проверить / восстановить окружение"
    Write-Host " 2. Запустить ЛР1"
    Write-Host " 3. Запустить ЛР2"
    Write-Host " 4. Запустить ЛР3"
    Write-Host " 5. Запустить ЛР1-ЛР3 с нуля"
    Write-Host " 6. Быстрая демонстрация сохранённых моделей"
    Write-Host " 7. Запустить тесты ЛР1-ЛР3"
    Write-Host " 8. Запустить MLflow UI"
    Write-Host " 9. Открыть оригинальные и новые результаты"
    Write-Host ""
    Write-Host " 0. Выход"
    Write-Host ""
}

if ($AutoSetup) {
    Ensure-Environment
}

while ($true) {
    Show-Menu
    $choice = Read-Host "Выберите пункт"

    try {
        switch ($choice) {
            "1" { Ensure-Environment -Force }
            "2" { Run-Lab1 }
            "3" { Run-Lab2 }
            "4" { Run-Lab3 }
            "5" { Run-AllLabs }
            "6" { Run-QuickDemo }
            "7" { Run-AllTests }
            "8" { Start-MLflowUi }
            "9" { Open-Results }
            "0" { return }
            default {
                Write-Host "Неизвестный пункт."
            }
        }
    }
    catch {
        Write-Host ""
        Write-Host "ОШИБКА: $($_.Exception.Message)"
    }

    if ($choice -ne "0") {
        Write-Host ""
        Read-Host "Нажмите Enter, чтобы вернуться в меню"
    }
}

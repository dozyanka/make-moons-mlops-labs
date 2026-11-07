# Normalize existing pytest log without retraining the model.
# Windows PowerShell 5.1 Tee-Object writes UTF-16LE by default.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$path = Join-Path $root 'reports\LAB4\pytest_coverage.log'
if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    Write-Host 'LAB4 pytest log has not been created yet.'
    return
}
$bytes = [System.IO.File]::ReadAllBytes($path)
if ($bytes.Length -eq 0) {
    Write-Host 'LAB4 pytest log is empty; no conversion needed.'
    return
}
$isUtf16 = $bytes.Length -ge 2 -and (
    ($bytes[0] -eq 255 -and $bytes[1] -eq 254) -or
    ($bytes[0] -eq 254 -and $bytes[1] -eq 255)
)
if ($isUtf16) {
    # ReadAllText detects the byte-order mark, including UTF-16BE/LE.
    $content = [System.IO.File]::ReadAllText($path)
    $utf8 = [System.Text.UTF8Encoding]::new($false, $true)
    $temporary = $path + '.' + [guid]::NewGuid().ToString('N') + '.log.tmp'
    try {
        [System.IO.File]::WriteAllText($temporary, $content, $utf8)
        $verified = [System.IO.File]::ReadAllText($temporary, $utf8)
        if ($verified -cne $content) { throw 'UTF-8 conversion verification failed' }
        Move-Item -LiteralPath $temporary -Destination $path -Force -ErrorAction Stop
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host 'LAB4 pytest_coverage.log: converted UTF-16 to UTF-8 (no BOM).'
} else {
    # Fail on invalid UTF-8 rather than silently replacing bytes.
    $utf8 = [System.Text.UTF8Encoding]::new($false, $true)
    $null = $utf8.GetString($bytes)
    Write-Host 'LAB4 pytest_coverage.log: already UTF-8; no changes made.'
}

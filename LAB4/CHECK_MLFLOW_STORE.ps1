# Read-only audit. Never modifies or deletes experiment history.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$store = Join-Path $root 'mlruns'
if (-not (Test-Path -LiteralPath $store -PathType Container)) {
    Write-Host 'No local mlruns/ store exists. Start RUN_LAB4.cmd first.'
    return
}
$invalid = @()
foreach ($dir in (Get-ChildItem -LiteralPath $store -Directory -Force)) {
    if ($dir.Name -notmatch '^\d+$') { continue }
    $meta = Join-Path $dir.FullName 'meta.yaml'
    if (Test-Path -LiteralPath $meta -PathType Leaf) { continue }
    $files = @(Get-ChildItem -LiteralPath $dir.FullName -File -Recurse -Force -ErrorAction Stop)
    $bytes = 0L
    foreach ($f in $files) { $bytes += $f.Length }
    $invalid += [pscustomobject]@{
        ExperimentId = $dir.Name
        Files = $files.Count
        Bytes = $bytes
        Directory = $dir.FullName
        State = 'MISSING meta.yaml'
    }
}
if ($invalid.Count -eq 0) {
    Write-Host 'MLflow filesystem structure: no experiments missing meta.yaml.' -ForegroundColor Green
} else {
    Write-Warning 'MLflow experiment directories with missing meta.yaml were detected:'
    $invalid | Format-Table -AutoSize | Out-Host
    Write-Host 'No files were modified. Do NOT delete these directories without reviewing their contents and making a backup.'
    Write-Host 'Successful LAB4 runs may coexist with these warnings.'
}

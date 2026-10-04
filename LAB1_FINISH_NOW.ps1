$ErrorActionPreference = "Stop"

$ROOT = "$env:USERPROFILE\Desktop\make-moons-mlops-labs"
Set-Location $ROOT

Write-Host "=== FINISH LAB 1 ==="

# Remove temporary helper scripts that contain dummy negative-control strings
foreach ($f in @(
    ".\FINALIZE_LAB1.ps1",
    ".\FINALIZE_LAB1_RESUME.ps1"
)) {
    if (Test-Path $f) {
        Remove-Item $f -Force
    }
}

# Remove interrupted temporary directory if it still exists
if (Test-Path ".\.lab1_negative_commit") {
    Remove-Item ".\.lab1_negative_commit" -Recurse -Force
}

Write-Host "=== PRE-COMMIT ==="
pre-commit run --all-files
if ($LASTEXITCODE -ne 0) {
    throw "pre-commit failed"
}

Write-Host "=== TESTS ==="
python -m pytest .\tests\test_data.py .\tests\test_eda.py -q
if ($LASTEXITCODE -ne 0) {
    throw "tests failed"
}

Write-Host "=== REPORTS ==="
python .\scripts\finalize_lab1.py
if ($LASTEXITCODE -ne 0) {
    throw "report generation failed"
}

Write-Host "=== GIT ==="
git add -A
$pending = git status --porcelain
if ($pending) {
    git commit -m "docs: finalize lab 1 defense package"
    git push
} else {
    Write-Host "Nothing new to commit."
}

Write-Host "=== ACTIONS ==="
Start-Sleep -Seconds 10

$ciRunId = gh run list `
    --workflow "CI" `
    --limit 1 `
    --json databaseId `
    --jq '.[0].databaseId'

if ($ciRunId) {
    gh run watch $ciRunId --exit-status
    if ($LASTEXITCODE -ne 0) {
        throw "CI failed"
    }
}

$lab1RunId = gh run list `
    --workflow "Lab 1 CI" `
    --limit 1 `
    --json databaseId `
    --jq '.[0].databaseId'

if ($lab1RunId) {
    gh run watch $lab1RunId --exit-status
    if ($LASTEXITCODE -ne 0) {
        throw "Lab 1 CI failed"
    }
}

Write-Host "=== ZIP ==="
$zip = "$env:USERPROFILE\Desktop\make-moons-mlops-labs-LAB1.zip"
if (Test-Path $zip) {
    Remove-Item $zip -Force
}
git archive --format=zip --output="$zip" HEAD

Write-Host ""
Write-Host "=========================="
Write-Host "LAB 1 DONE"
Write-Host "=========================="
Write-Host "DOCX: reports\LAB1\lab1_report.docx"
Write-Host "MD: reports\LAB1\lab1_report.md"
Write-Host "DEFENSE: reports\LAB1\DEFENSE.md"
Write-Host "SCREENCAST: reports\LAB1\SCREENCAST.md"
Write-Host "ZIP: $zip"
Write-Host ""

git status
gh run list --limit 5

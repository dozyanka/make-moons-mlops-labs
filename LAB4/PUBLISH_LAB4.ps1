# LAB4 publication: guarded, two-commit Git workflow. Run from Downloads or repo.
# No administrator rights required, but it can be started from elevated PowerShell.
[CmdletBinding()]
param(
    [string]$RepoPath = 'C:\Users\shata\Desktop\make-moons-mlops-labs',
    [switch]$NoPush
)
$ErrorActionPreference = 'Stop'

function Assert-Exit([string]$Stage) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage FAILED (exit $LASTEXITCODE). No force push or cleanup was attempted."
    }
}
function Invoke-Git([string[]]$Arguments) {
    & git @Arguments
    Assert-Exit ("git " + ($Arguments -join ' '))
}
function Git-Output([string[]]$Arguments) {
    $result = & git @Arguments
    Assert-Exit ("git " + ($Arguments -join ' '))
    return ($result | Out-String).Trim()
}
function Assert-File([string]$Relative) {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoPath $Relative) -PathType Leaf)) {
        throw "Missing required file: $Relative. Apply LAB4 patch 06 and 07 completely."
    }
}

try {
    $RepoPath = (Resolve-Path -LiteralPath $RepoPath).Path
    Set-Location -LiteralPath $RepoPath
    Write-Host '=== LAB4: safe GitHub publication (patch 07) ===' -ForegroundColor Cyan
    Write-Host "Repository: $RepoPath"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is not installed or not on PATH.' }
    if (-not (Test-Path -LiteralPath (Join-Path $RepoPath '.git'))) {
        throw 'This folder has no .git. Use the ORIGINAL Git clone, not an extracted GitHub ZIP.'
    }
    $projectRoot = Git-Output @('rev-parse', '--show-toplevel')
    if ([IO.Path]::GetFullPath($projectRoot).TrimEnd('\') -ne [IO.Path]::GetFullPath($RepoPath).TrimEnd('\')) {
        throw "Unexpected Git root: $projectRoot"
    }
    $url = Git-Output @('remote', 'get-url', 'origin')
    if ($url -notmatch '(?i)github\.com[:/]dozyanka/make-moons-mlops-labs(?:\.git)?/?$') {
        throw "Unexpected origin URL: $url. Refusing to publish to another repository."
    }
    $branch = Git-Output @('branch', '--show-current')
    if ($branch -ne 'main') {
        throw "Expected branch 'main', got '$branch'. Script only creates a new lab4 branch from main. No checkout was performed."
    }
    $staged = Git-Output @('diff', '--cached', '--name-only')
    if ($staged) { throw "Existing staged files detected. Unstage/commit them first. Refusing to mix commits: $staged" }
    $existingLocal = & git show-ref --verify --quiet 'refs/heads/lab4'
    if ($LASTEXITCODE -eq 0) { throw "A local 'lab4' branch already exists. No branch was overwritten." }
    if ($LASTEXITCODE -ne 1) { throw 'Cannot inspect the local lab4 branch.' }

    # Preflight must not alter any of LAB1-3's data, models, artifacts, or MLflow history.
    $python = Join-Path $RepoPath '.venv\Scripts\python.exe'
    Assert-File 'LAB4\data_integrity_baseline.json'
    Assert-File 'scripts\verify_lab4_data.py'
    Assert-File 'LAB4\RUN_LAB4.ps1'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw 'Missing .venv Python. Run LAB4\RUN_LAB4.cmd successfully before publishing.'
    }
    Write-Host 'Checking original CSV and LAB2 model SHA-256...'
    & $python scripts\verify_lab4_data.py
    Assert-Exit 'LAB4 input integrity'
    & $python scripts\precommit_checks.py --all
    Assert-Exit 'Project precommit checks'
    & git var GIT_AUTHOR_IDENT | Out-Null
    Assert-Exit 'Git author configuration'

    Write-Host 'Fetching origin/main and verifying that main is current...'
    Invoke-Git @('fetch', 'origin', 'main', '--quiet')
    $localMain = Git-Output @('rev-parse', 'HEAD')
    $remoteMain = Git-Output @('rev-parse', 'refs/remotes/origin/main')
    # A feature branch does NOT require local main to equal origin/main.
    # It is safe to start from the older local main when it is proven to be an
    # ancestor of origin/main. Leave both main refs unchanged. In the unusual
    # cases of local-only commits or divergent histories, stop for manual review.
    if ($localMain -ne $remoteMain) {
        & git merge-base --is-ancestor $localMain $remoteMain
        $ancestorStatus = $LASTEXITCODE
        if ($ancestorStatus -eq 0) {
            $aheadCount = Git-Output @('rev-list', '--count', "$localMain..$remoteMain")
            Write-Host "Local main is BEHIND origin/main by $aheadCount commit(s)." -ForegroundColor Yellow
            Write-Host "Local main: $localMain"
            Write-Host "Remote main: $remoteMain"
            Write-Host 'Safe: local main is an ancestor of origin/main.' -ForegroundColor Green
            Write-Host 'Creating lab4 from local main WITHOUT updating main or changing working files.'
            Write-Warning 'lab4 starts from the older main. GitHub Pull Request may need reconciliation with newer main changes before merging.'
            Write-Host 'New commits on origin/main:'
            Invoke-Git @('log', '--oneline', '--max-count=12', "$localMain..$remoteMain")
        } elseif ($ancestorStatus -eq 1) {
            & git merge-base --is-ancestor $remoteMain $localMain
            $reverseStatus = $LASTEXITCODE
            if ($reverseStatus -eq 0) {
                throw "Local main is AHEAD of origin/main. Cannot publish unreviewed local-only commits. Local=$localMain Remote=$remoteMain. No files were changed."
            }
            if ($reverseStatus -eq 1) {
                throw "Local main and origin/main have DIVERGED. Automatic merge would risk LAB1-3 changes. Local=$localMain Remote=$remoteMain. No files were changed."
            }
            throw "Git merge-base check (remote ancestor) failed with exit $reverseStatus."
        } else {
            throw "Git merge-base check (local ancestor) failed with exit $ancestorStatus."
        }
    } else {
        Write-Host 'Local main and origin/main match.' -ForegroundColor Green
    }
    $remoteLab4 = & git ls-remote --heads origin lab4
    Assert-Exit 'Remote lab4 branch check'
    if ($remoteLab4) { throw "A remote 'lab4' branch already exists; refusing to replace its history." }

    Write-Host 'Creating lab4 branch (main remains unchanged)...'
    Invoke-Git @('switch', '-c', 'lab4')
    $codePaths = @(
        '.gitattributes', '.gitignore', 'README.md',
        'pyproject.toml', 'LAB4', 'configs/lab4.json',
        '.github/workflows/lab4-ci.yml',
        'src/moons_lab/config.py', 'src/moons_lab/data.py',
        'src/moons_lab/utils.py', 'src/moons_lab/validator.py',
        'scripts/run_lab4.py', 'scripts/verify_lab4_data.py',
        'defense_demo/lab4_demo.py',
        'tests/test_validator.py', 'tests/test_lab4_workflow.py',
        'tests/test_lab4_mlflow_tracking.py', 'tests/test_lab4_runner.py',
        'tests/test_lab4_publication.py'
    )
    $baseline = Get-Content -Raw -Encoding UTF8 'LAB4/data_integrity_baseline.json' | ConvertFrom-Json
    foreach ($entry in $baseline.files.PSObject.Properties) { $codePaths += $entry.Name }
    Write-Host 'Staging only LAB4 code, metadata, and original data...'
    Invoke-Git (@('add', '--') + $codePaths)
    Invoke-Git @('diff', '--cached', '--check')
    & $python scripts\verify_lab4_data.py --check-git-index
    Assert-Exit 'Git byte-exact data staging'
    $staged = Git-Output @('diff', '--cached', '--name-only')
    if (-not $staged) { throw 'No code changes were staged; refusing an empty code commit.' }
    Write-Host 'The following files will enter the code commit:'
    Write-Host $staged
    Invoke-Git @('commit', '-m', 'feat(lab4): reproducible validator, data integrity and CI')
    $codeCommit = Git-Output @('rev-parse', 'HEAD')
    Write-Host "Code commit: $codeCommit" -ForegroundColor Green

    # These commands regenerate artifacts only for LAB4 and run the Windows test suite.
    # A nonzero exit means no second commit and no Git push.
    Write-Host 'Regenerating LAB4 report from the committed source code and running tests...'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoPath 'LAB4\RUN_LAB4.ps1') -SkipInstall
    Assert-Exit 'LAB4 full execution and pytest'
    & $python scripts\verify_lab4_data.py --check-git-index --check-published --expected-code-commit $codeCommit
    Assert-Exit 'LAB4 report provenance and reproduction'

    $reportPaths = @(
        'reports/LAB4/lab4_report.md',
        'reports/LAB4/validator_metrics.csv',
        'reports/LAB4/validator_params.npz',
        'reports/LAB4/validator_params_chunk.npz',
        'reports/LAB4/validator_negative.log',
        'reports/LAB4/comparison.json',
        'reports/LAB4/reproducibility.json',
        'reports/LAB4/repeat_verification.json',
        'reports/LAB4/negative_summary.json',
        'reports/LAB4/config_negative_test.log',
        'reports/LAB4/source_manifest.json',
        'reports/LAB4/pytest_coverage.log',
        'reports/LAB4/coverage_details.json'
    )
    Write-Host 'Staging generated LAB4 reports (never mlruns/)...'
    Invoke-Git (@('add', '--') + $reportPaths)
    Invoke-Git @('diff', '--cached', '--check')
    $stagedReports = Git-Output @('diff', '--cached', '--name-only')
    if (-not $stagedReports) { throw 'No report changes were staged; refusing empty report commit.' }
    Invoke-Git @('commit', '-m', 'docs(lab4): verified artifacts and report tied to source commit')
    $reportCommit = Git-Output @('rev-parse', 'HEAD')
    Write-Host "Report commit: $reportCommit" -ForegroundColor Green

    # Recheck that the report references the *preceding* code commit.
    & $python scripts\verify_lab4_data.py --check-git-index --check-published --expected-code-commit $codeCommit
    Assert-Exit 'Final published content verification'
    if ($NoPush) {
        Write-Host 'NoPush requested. Two local commits created; NOTHING was sent to GitHub.' -ForegroundColor Yellow
    } else {
        Write-Host 'Pushing branch lab4 without force, leaving main untouched...'
        Invoke-Git @('push', '-u', 'origin', 'lab4')
        Write-Host 'Branch pushed. GitHub Actions must still finish successfully.' -ForegroundColor Green
    }
    Write-Host "GitHub branch: https://github.com/dozyanka/make-moons-mlops-labs/tree/lab4"
    Write-Host "GitHub Actions: https://github.com/dozyanka/make-moons-mlops-labs/actions"
    Write-Host "Pull request page: https://github.com/dozyanka/make-moons-mlops-labs/compare/main...lab4"
    Write-Host 'Do not merge to main until CI is green and you approve the LAB4 submission.'
    Write-Host '=== LAB4 PUBLICATION FINISHED ===' -ForegroundColor Green
} catch {
    Write-Host "LAB4 PUBLISH STOPPED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'No files or MLflow directories were deleted; no force push was attempted.'
    Write-Host 'If a code commit was already created, keep branch lab4 and resolve the error before retrying.'
    exit 1
}
